"""Generate resumable full-compute Ouro traces for BFCL V3 programs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.program_scheduling.bfcl_workload import (  # noqa: E402
    build_contexts,
    select_base_programs,
)
from loopaware_agent.ut_trace import OuroUTTracer  # noqa: E402
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"
BFCL_REVISION = "cd9429ccf3d4d04156affe883c495b3b047e6b64"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--programs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=12000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def completed_turn_path(output: Path, program_id: str, turn_id: int) -> Path:
    return output.parent / "turns" / program_id / f"turn-{turn_id:04d}.json"


def main() -> None:
    args = arguments()
    source_rows = select_base_programs(args.programs, args.seed)
    contexts = {row["id"]: build_contexts(row) for row in source_rows}
    local_checkpoint = PROJECT_ROOT / ".cache/models/Ouro-2.6B-Thinking"
    shared_snapshot = (
        Path("/workspace/.cache/huggingface/hub/")
        / "models--ByteDance--Ouro-2.6B-Thinking/snapshots"
        / MODEL_REVISION
    )
    local_weights = local_checkpoint / "model.safetensors"
    shared_weights = shared_snapshot / "model.safetensors"
    if local_weights.is_file() and local_weights.stat().st_size > 0:
        model_source: str | Path = local_checkpoint
    elif shared_weights.is_file() and shared_weights.stat().st_size > 0:
        model_source = shared_snapshot
    else:
        model_source = MODEL_ID
    load_kwargs = {} if isinstance(model_source, Path) else {"revision": MODEL_REVISION}
    tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
    config = OuroConfig.from_pretrained(model_source, **load_kwargs)
    config.total_ut_steps = 4
    model = OuroForCausalLM.from_pretrained(
        model_source,
        **load_kwargs,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    model.early_exit_step = None
    model.early_exit_threshold = None
    runtime_layers = int(model.config.num_hidden_layers)
    if runtime_layers <= 0:
        raise ValueError(f"invalid runtime num_hidden_layers={runtime_layers}")
    print(
        f"full_compute_ut=4 selected_ut_threshold=0.2 "
        f"runtime_num_hidden_layers={runtime_layers}",
        flush=True,
    )

    # Untimed warm-up uses the same physical depth but is not part of any program trace.
    warmup = tokenizer("warm up", return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        model.generate(warmup, max_new_tokens=2, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    del warmup

    programs = []
    for program_index, row in enumerate(source_rows, start=1):
        turns = []
        for context in contexts[row["id"]]:
            path = completed_turn_path(args.output, row["id"], context["turn_id"])
            if args.resume and path.exists():
                turns.append(json.loads(path.read_text()))
                continue
            input_ids = tokenizer.apply_chat_template(
                context["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_tokens = int(input_ids.shape[-1])
            if input_tokens > args.max_input_tokens:
                raise ValueError(
                    f"{row['id']} turn {context['turn_id']} has {input_tokens} input "
                    f"tokens, above limit {args.max_input_tokens}"
                )
            input_ids = input_ids.to(model.device)
            attention_mask = torch.ones_like(input_ids)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            gpu_start.record()
            with torch.inference_mode(), OuroUTTracer(
                model, exit_threshold=0.2
            ) as tracer:
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gpu_end.record()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            gpu_ms = gpu_start.elapsed_time(gpu_end)
            generated = output_ids[:, input_ids.shape[-1] :]
            events = tracer.attach_output_tokens(generated, tokenizer)
            selected = [event.selected_ut_step for event in events]
            if not events or any(
                event.executed_ut_steps != 4 or event.decode_ut_step != 4
                for event in events
            ):
                raise RuntimeError("trace violated fixed full-compute UT=4 execution")
            turn = {
                "program_id": row["id"],
                "turn_id": context["turn_id"],
                "input_token_count": input_tokens,
                "output_token_count": len(events),
                "selected_ut_per_token": selected,
                "sum_selected_ut": sum(selected),
                "mean_selected_ut": sum(selected) / len(selected),
                "selected_ut_histogram": dict(sorted(Counter(selected).items())),
                "full_compute_gpu_time_ms": gpu_ms,
                "full_compute_wall_time_ms": wall_ms,
                "tool_wait_ms": 0.0,
                "executed_ut_per_token": 4,
                "selected_ut_threshold": 0.2,
                "runtime_num_hidden_layers": runtime_layers,
                "executed_decoder_layer_calls": len(events) * 4 * runtime_layers,
                "selected_decoder_layer_calls_counterfactual": sum(selected) * runtime_layers,
                "prompt_sha256": hashlib.sha256(
                    input_ids.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "generated_token_ids": generated[0].detach().cpu().tolist(),
                "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
                "ground_truth_calls": context["ground_truth_calls"],
            }
            atomic_json(path, turn)
            turns.append(turn)
            print(
                f"DONE program={program_index}/{args.programs} id={row['id']} "
                f"turn={context['turn_id'] + 1}/{len(contexts[row['id']])} "
                f"input={input_tokens} output={len(events)} gpu_ms={gpu_ms:.1f}",
                flush=True,
            )
            del input_ids, attention_mask, output_ids, generated, events
        programs.append({"program_id": row["id"], "arrival_time_ms": 0.0, "turns": turns})
        atomic_json(
            args.output,
            {
                "configuration": {
                    "dataset": "BFCL_v3_multi_turn_base",
                    "dataset_revision": BFCL_REVISION,
                    "model": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "program_count": args.programs,
                    "seed": args.seed,
                    "max_new_tokens": args.max_new_tokens,
                    "physical_ut": 4,
                    "selected_ut_threshold": 0.2,
                    "runtime_num_hidden_layers": runtime_layers,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "teacher_forcing": "recorded BFCL ground-truth tool calls and official execution results",
                    "tool_wait_policy": "zero because BFCL V3 has no tool timestamps",
                },
                "programs": programs,
            },
        )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
