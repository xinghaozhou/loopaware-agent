"""Small real-checkpoint validation of Ouro adaptive decode semantics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loopaware_agent.ut_trace import OuroUTTracer  # noqa: E402
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B-Thinking")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--total-ut-steps", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def generate_once(model, tokenizer, model_inputs, *, physical: bool, args):
    model.config.physical_early_exit = physical
    model.model.config.physical_early_exit = physical
    with torch.inference_mode(), OuroUTTracer(
        model, exit_threshold=args.threshold
    ) as tracer:
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[:, model_inputs["input_ids"].shape[1] :]
    events = tracer.attach_output_tokens(generated, tokenizer)
    return generated.detach().cpu(), events


def main() -> None:
    args = parse_args()
    model_source = args.model_path or args.model
    tokenizer_source = args.tokenizer_path or args.model
    source_revision = None if args.model_path is not None else args.revision
    tokenizer_revision = None if args.tokenizer_path is not None else args.revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=tokenizer_revision, trust_remote_code=True
    )
    model_inputs = tokenizer(
        "Explain why a cache needs an explicit missing-state policy.",
        return_tensors="pt",
    ).to("cuda")
    results = []

    for policy in ("last_available", "final_exit_shared"):
        config = OuroConfig.from_pretrained(model_source, revision=source_revision)
        config.total_ut_steps = args.total_ut_steps
        config.early_exit_threshold = args.threshold
        config.kv_cache_policy = policy
        config.physical_early_exit = False
        model = OuroForCausalLM.from_pretrained(
            model_source,
            revision=source_revision,
            config=config,
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        runtime_layers = model.config.num_hidden_layers
        if not isinstance(runtime_layers, int) or runtime_layers <= 0:
            raise ValueError(f"Invalid runtime num_hidden_layers={runtime_layers!r}")

        reference_ids, reference_events = generate_once(
            model, tokenizer, model_inputs, physical=False, args=args
        )
        physical_ids, physical_events = generate_once(
            model, tokenizer, model_inputs, physical=True, args=args
        )
        if not torch.equal(reference_ids, physical_ids):
            raise AssertionError(f"Generated token mismatch for {policy}")
        reference_selected = [event.selected_ut_step for event in reference_events]
        physical_selected = [event.selected_ut_step for event in physical_events]
        if reference_selected != physical_selected:
            raise AssertionError(f"Selected-depth mismatch for {policy}")

        result = {
            "policy": policy,
            "num_hidden_layers": runtime_layers,
            "generated_token_ids": physical_ids[0].tolist(),
            "selected_ut_steps": physical_selected,
            "reference_executed_ut_steps": [
                event.executed_ut_steps for event in reference_events
            ],
            "physical_executed_ut_steps": [
                event.executed_ut_steps for event in physical_events
            ],
            "physical_trace": [asdict(event) for event in physical_events],
        }
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "physical_trace"}))

        del model
        torch.cuda.empty_cache()

    payload = {
        "model": args.model,
        "revision": args.revision,
        "total_ut_steps": args.total_ut_steps,
        "threshold": args.threshold,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
