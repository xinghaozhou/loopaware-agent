"""Probe Ouro batch memory on the longest contexts in the fixed BFCL sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.program_scheduling.bfcl_workload import (  # noqa: E402
    build_contexts,
    select_base_programs,
)
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"


def model_source() -> tuple[str | Path, dict[str, str]]:
    local = PROJECT_ROOT / ".cache/models/Ouro-2.6B-Thinking"
    shared = (
        Path("/workspace/.cache/huggingface/hub")
        / "models--ByteDance--Ouro-2.6B-Thinking/snapshots"
        / MODEL_REVISION
    )
    for candidate in (local, shared):
        weights = candidate / "model.safetensors"
        if weights.is_file() and weights.stat().st_size > 0:
            return candidate, {}
    return MODEL_ID, {"revision": MODEL_REVISION}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--programs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source, load_kwargs = model_source()
    tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    contexts = [
        context
        for row in select_base_programs(args.programs, args.seed)
        for context in build_contexts(row)
    ]
    measured = []
    for context in contexts:
        ids = tokenizer.apply_chat_template(
            context["messages"], tokenize=True, add_generation_prompt=True
        )
        measured.append((len(ids), context))
    measured.sort(key=lambda item: item[0], reverse=True)

    config = OuroConfig.from_pretrained(source, **load_kwargs)
    config.total_ut_steps = 4
    model = OuroForCausalLM.from_pretrained(
        source,
        **load_kwargs,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    model.early_exit_step = None
    model.early_exit_threshold = None
    runtime_layers = int(model.config.num_hidden_layers)

    results = []
    for batch_size in args.batch_sizes:
        selected = measured[:batch_size]
        prompts = [item[1]["messages"] for item in selected]
        encoded = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = {
            "batch_size": batch_size,
            "unpadded_input_tokens": [item[0] for item in selected],
            "padded_input_tokens": int(encoded.input_ids.shape[-1]),
            "max_new_tokens": args.max_new_tokens,
        }
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            result.update(
                {
                    "status": "fit",
                    "elapsed_seconds": time.perf_counter() - started,
                    "generated_shape": list(generated.shape),
                }
            )
            del generated
        except torch.cuda.OutOfMemoryError as error:
            result.update(
                {
                    "status": "oom",
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": str(error),
                }
            )
        finally:
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
            del encoded
            torch.cuda.empty_cache()

    payload = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "physical_ut": 4,
        "selected_ut_threshold_for_experiment": 0.2,
        "runtime_num_hidden_layers": runtime_layers,
        "programs": args.programs,
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
