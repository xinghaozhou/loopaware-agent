"""Check that last-available physical decode at threshold 1 matches Ouro."""

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


MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"
PROMPTS = (
    "Explain why a cache needs an explicit missing-state policy.",
    "If a recurrent model always exits at its maximum depth, what cache states are available?",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--total-ut-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def set_mode(model, *, policy: str, physical: bool) -> None:
    model.config.kv_cache_policy = policy
    model.model.config.kv_cache_policy = policy
    model.config.physical_early_exit = physical
    model.model.config.physical_early_exit = physical
    model.config.early_exit_threshold = 1.0
    model.model.config.early_exit_threshold = 1.0
    model.early_exit_threshold = 1.0


def generate(
    model, tokenizer, inputs, *, policy: str, physical: bool, max_new_tokens: int
):
    set_mode(model, policy=policy, physical=physical)
    logits = []

    def capture(_module, _inputs, output):
        logits.append(output.logits[:, -1, :].detach().float().cpu())

    handle = model.register_forward_hook(capture)
    try:
        with torch.inference_mode(), OuroUTTracer(model, exit_threshold=1.0) as tracer:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    events = tracer.attach_output_tokens(generated, tokenizer)
    return generated.detach().cpu(), logits, events


def main() -> None:
    args = parse_args()
    torch.manual_seed(20261010)
    torch.cuda.manual_seed_all(20261010)
    model_source = args.model_path or args.model
    tokenizer_source = args.tokenizer_path or args.model
    revision = None if args.model_path else args.revision
    tokenizer_revision = None if args.tokenizer_path else args.revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=tokenizer_revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(list(PROMPTS), padding=True, return_tensors="pt").to("cuda")

    config = OuroConfig.from_pretrained(model_source, revision=revision)
    config.total_ut_steps = args.total_ut_steps
    config.early_exit_threshold = 1.0
    config.kv_cache_policy = "full_per_depth"
    config.physical_early_exit = False
    model = OuroForCausalLM.from_pretrained(
        model_source,
        revision=revision,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    runtime_layers = model.config.num_hidden_layers

    original_ids, original_logits, original_events = generate(
        model,
        tokenizer,
        inputs,
        policy="full_per_depth",
        physical=False,
        max_new_tokens=args.max_new_tokens,
    )
    adaptive_ids, adaptive_logits, adaptive_events = generate(
        model,
        tokenizer,
        inputs,
        policy="last_available",
        physical=True,
        max_new_tokens=args.max_new_tokens,
    )
    if len(original_logits) != len(adaptive_logits):
        raise AssertionError("Forward counts differ")
    step_max_abs = [
        float((left - right).abs().max())
        for left, right in zip(original_logits, adaptive_logits, strict=True)
    ]
    original_selected = [event.selected_ut_step for event in original_events]
    adaptive_selected = [event.selected_ut_step for event in adaptive_events]
    adaptive_executed = [event.executed_ut_steps for event in adaptive_events]
    expected_events = len(adaptive_events)
    result = {
        "model": args.model,
        "revision": args.revision,
        "runtime_num_hidden_layers": runtime_layers,
        "total_ut_steps": args.total_ut_steps,
        "threshold": 1.0,
        "batch_size": len(PROMPTS),
        "max_new_tokens": args.max_new_tokens,
        "comparison": "full_per_depth run-all vs last_available physical",
        "generated_token_parity": torch.equal(original_ids, adaptive_ids),
        "selected_ut_parity": original_selected == adaptive_selected,
        "all_selected_at_u_max": original_selected
        == [args.total_ut_steps] * expected_events,
        "all_physical_execution_at_u_max": adaptive_executed
        == [args.total_ut_steps] * expected_events,
        "maximum_absolute_logit_difference": max(step_max_abs, default=0.0),
        "maximum_absolute_logit_difference_by_forward": step_max_abs,
        "generated_ids": adaptive_ids.tolist(),
        "original_trace": [asdict(event) for event in original_events],
        "adaptive_trace": [asdict(event) for event in adaptive_events],
    }
    required = (
        result["generated_token_parity"],
        result["selected_ut_parity"],
        result["all_selected_at_u_max"],
        result["all_physical_execution_at_u_max"],
        result["maximum_absolute_logit_difference"] == 0.0,
    )
    result["passed"] = all(required)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if "trace" not in key and key != "generated_ids"}, indent=2))
    if not result["passed"]:
        raise AssertionError("Threshold-one equivalence sanity check failed")


if __name__ == "__main__":
    main()
