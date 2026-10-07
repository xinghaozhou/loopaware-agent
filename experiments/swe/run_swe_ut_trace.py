"""Generate SWE-bench trajectories and trace Ouro UT depth per output token."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from loopaware_agent.ut_trace import (  # noqa: E402
    OuroUTTracer,
    summarize_ut_trace,
    write_trajectory_jsonl,
)
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


DEFAULT_MODEL = "ByteDance/Ouro-2.6B-Thinking"
DEFAULT_DATASET = "SWE-bench/SWE-bench_Verified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--num-trajectories", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-ut-steps", type=int, default=None)
    parser.add_argument(
        "--trace-exit-threshold",
        "--exit-threshold",
        dest="trace_exit_threshold",
        type=float,
        default=0.8,
        help=(
            "Counterfactual cumulative exit-probability threshold used only to "
            "trace the selected UT step; decoding always uses the final UT step"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "traces" / "swe_ut_trajectories.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "traces" / "swe_ut_summary.json",
    )
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def build_prompt(instance: dict[str, Any]) -> str:
    return (
        "You are a software engineer debugging a real GitHub issue.\n\n"
        f"Repository:\n{instance['repo']}\n\n"
        f"Issue:\n{instance['problem_statement']}\n\n"
        "Analyze the issue carefully and explain how you would fix it.\n"
    )


def main() -> None:
    args = parse_args()
    if args.num_instances < 1 or args.num_trajectories < 1:
        raise ValueError("num-instances and num-trajectories must be positive")
    if args.temperature < 0:
        raise ValueError("temperature cannot be negative")

    dataset = load_dataset(args.dataset, split=args.split).select(
        range(args.num_instances)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = OuroConfig.from_pretrained(args.model)
    if args.total_ut_steps is not None:
        config.total_ut_steps = args.total_ut_steps
    model = OuroForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    # Keep decoding fixed at the final recurrent step. The tracer independently
    # computes the step that adaptive decoding would select at its threshold.
    model.early_exit_step = None
    model.early_exit_threshold = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.append:
        args.output.write_text("", encoding="utf-8")

    all_expected_depths: list[float] = []
    all_selected_depths: list[int] = []
    exit_probability_sums: list[float] = []
    trajectory_summaries: list[dict[str, Any]] = []
    for instance_index, instance in enumerate(dataset):
        prompt = build_prompt(instance)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs["input_ids"].shape[1]

        for trajectory_index in range(args.num_trajectories):
            trajectory_seed = args.seed + (
                instance_index * args.num_trajectories + trajectory_index
            )
            torch.manual_seed(trajectory_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trajectory_seed)

            with torch.inference_mode(), OuroUTTracer(
                model, exit_threshold=args.trace_exit_threshold
            ) as tracer:
                generation_options: dict[str, Any] = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.temperature > 0,
                    "use_cache": True,
                }
                if args.temperature > 0:
                    generation_options.update(
                        temperature=args.temperature, top_p=args.top_p
                    )
                output_ids = model.generate(**inputs, **generation_options)

            generated_ids = output_ids[:, prompt_length:]
            events = tracer.attach_output_tokens(generated_ids, tokenizer)
            output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            write_trajectory_jsonl(
                args.output,
                trajectory={
                    "instance_id": instance["instance_id"],
                    "repo": instance["repo"],
                    "trajectory_index": trajectory_index,
                    "seed": trajectory_seed,
                    "prompt_token_count": prompt_length,
                    "output_text": output_text,
                },
                events=events,
            )

            selected = [event.selected_ut_step for event in events]
            expected = [event.expected_ut_steps for event in events]
            all_selected_depths.extend(selected)
            all_expected_depths.extend(expected)
            if events and not exit_probability_sums:
                exit_probability_sums = [0.0] * events[0].executed_ut_steps
            for event in events:
                for step, probability in enumerate(event.exit_probabilities):
                    exit_probability_sums[step] += probability
            trajectory_summaries.append({
                "instance_id": instance["instance_id"],
                "trajectory_index": trajectory_index,
                "seed": trajectory_seed,
                **summarize_ut_trace(events),
            })
            print(
                f"[{instance_index + 1}/{len(dataset)}] "
                f"{instance['instance_id']} trajectory={trajectory_index} "
                f"tokens={len(events)}"
            )

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "num_instances": len(dataset),
        "num_trajectories_per_instance": args.num_trajectories,
        "decode_ut_policy": "fixed_final_step",
        "trace_exit_threshold": args.trace_exit_threshold,
        "total_ut_steps": model.model.total_ut_steps,
        "num_output_tokens": len(all_selected_depths),
        "global_mean_expected_ut_steps": (
            sum(all_expected_depths) / len(all_expected_depths)
            if all_expected_depths
            else None
        ),
        "global_selected_ut_step_histogram": dict(
            sorted(Counter(all_selected_depths).items())
        ),
        "global_mean_exit_probabilities": (
            [
                probability_sum / len(all_selected_depths)
                for probability_sum in exit_probability_sums
            ]
            if all_selected_depths
            else []
        ),
        "trajectories": trajectory_summaries,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
