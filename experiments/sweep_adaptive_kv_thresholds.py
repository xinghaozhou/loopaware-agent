"""Bounded real-checkpoint threshold sweep for adaptive Ouro KV policies."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loopaware_agent.ut_trace import OuroUTTracer  # noqa: E402
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


DEFAULT_PROMPTS = (
    "Explain why an adaptive recurrent model needs a KV fallback policy.",
    "Write a concise Python function that returns the first duplicate in a list.",
    "A test passes locally but fails in CI. Give a systematic debugging plan.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B-Thinking")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--total-ut-steps", type=int, default=6)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--batched-prompts",
        action="store_true",
        help="Run prompts together; default is independent semantic validation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_generation(model, tokenizer, model_inputs, *, physical: bool, threshold: float, max_new_tokens: int):
    model.config.physical_early_exit = physical
    model.model.config.physical_early_exit = physical
    model.config.early_exit_threshold = threshold
    model.model.config.early_exit_threshold = threshold
    model.early_exit_threshold = threshold

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    logits_by_step = []

    def capture_logits(module, inputs, output):
        del module, inputs
        logits_by_step.append(output.logits[:, -1, :].detach().float().cpu())

    logits_handle = model.register_forward_hook(capture_logits)
    with torch.inference_mode(), OuroUTTracer(model, exit_threshold=threshold) as tracer:
        try:
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        finally:
            logits_handle.remove()
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    generated = output_ids[:, model_inputs["input_ids"].shape[1] :]
    events = tracer.attach_output_tokens(generated, tokenizer)
    return {
        "generated_ids": generated.detach().cpu(),
        "events": events,
        "logits_by_step": logits_by_step,
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def summarize_condition(reference, physical, num_layers: int, total_ut_steps: int):
    if not torch.equal(reference["generated_ids"], physical["generated_ids"]):
        mismatch = (
            reference["generated_ids"] != physical["generated_ids"]
        ).nonzero(as_tuple=False)[0]
        batch_index, token_index = (int(value) for value in mismatch.tolist())
        reference_logits = reference["logits_by_step"][token_index][batch_index]
        physical_logits = physical["logits_by_step"][token_index][batch_index]
        reference_top = torch.topk(reference_logits, 2)
        physical_top = torch.topk(physical_logits, 2)
        raise AssertionError(
            "Physical generation differs from its run-all reference: "
            + json.dumps(
                {
                    "first_mismatch_batch": batch_index,
                    "first_mismatch_token": token_index,
                    "reference_token": int(
                        reference["generated_ids"][batch_index, token_index]
                    ),
                    "physical_token": int(
                        physical["generated_ids"][batch_index, token_index]
                    ),
                    "max_abs_logit_difference": float(
                        (reference_logits - physical_logits).abs().max()
                    ),
                    "reference_top2_ids": reference_top.indices.tolist(),
                    "reference_top2_logits": reference_top.values.tolist(),
                    "reference_top2_margin": float(
                        reference_top.values[0] - reference_top.values[1]
                    ),
                    "physical_top2_ids": physical_top.indices.tolist(),
                    "physical_top2_logits": physical_top.values.tolist(),
                    "physical_top2_margin": float(
                        physical_top.values[0] - physical_top.values[1]
                    ),
                    "reference_ids": reference["generated_ids"].tolist(),
                    "physical_ids": physical["generated_ids"].tolist(),
                    "reference_selected": [
                        event.selected_ut_step for event in reference["events"]
                    ],
                    "physical_selected": [
                        event.selected_ut_step for event in physical["events"]
                    ],
                }
            )
        )
    reference_selected = [event.selected_ut_step for event in reference["events"]]
    physical_selected = [event.selected_ut_step for event in physical["events"]]
    if reference_selected != physical_selected:
        raise AssertionError("Physical selected depths differ from run-all reference")

    # output_index=0 is predicted by full-depth prefill. Later events are
    # one-token decode calls where physical exit is enabled.
    decode_events = [
        event for event in physical["events"] if event.output_index > 0
    ]
    selected = [event.selected_ut_step for event in decode_events]
    physical_executed = sum(event.executed_ut_steps for event in physical["events"])
    reference_executed = sum(event.executed_ut_steps for event in reference["events"])
    generated_bytes = physical["generated_ids"].numpy().tobytes()
    return {
        "num_sequences": int(physical["generated_ids"].shape[0]),
        "num_generated_tokens": int(physical["generated_ids"].numel()),
        "num_decode_events": len(decode_events),
        "selected_ut_histogram_decode": {
            str(depth): count for depth, count in sorted(Counter(selected).items())
        },
        "mean_selected_ut_decode": sum(selected) / len(selected),
        "min_selected_ut_decode": min(selected),
        "max_selected_ut_decode": max(selected),
        "reference_executed_ut_total": reference_executed,
        "physical_executed_ut_total": physical_executed,
        "reference_decoder_layer_calls": reference_executed * num_layers,
        "physical_decoder_layer_calls": physical_executed * num_layers,
        "physical_recurrence_saving": 1.0 - physical_executed / reference_executed,
        "run_all_elapsed_seconds": reference["elapsed_seconds"],
        "physical_elapsed_seconds": physical["elapsed_seconds"],
        "wall_time_speedup": reference["elapsed_seconds"] / physical["elapsed_seconds"],
        "run_all_peak_allocated_gib": reference["peak_allocated_gib"],
        "physical_peak_allocated_gib": physical["peak_allocated_gib"],
        "run_all_peak_reserved_gib": reference["peak_reserved_gib"],
        "physical_peak_reserved_gib": physical["peak_reserved_gib"],
        "generated_token_sha256": hashlib.sha256(generated_bytes).hexdigest(),
        "token_parity": True,
        "selected_depth_parity": True,
        "total_ut_steps": total_ut_steps,
        "physical_trace": [asdict(event) for event in physical["events"]],
    }


def aggregate_prompt_summaries(summaries):
    histogram = Counter()
    traces = []
    generated_hashes = []
    for prompt_index, summary in enumerate(summaries):
        histogram.update(
            {
                int(depth): count
                for depth, count in summary["selected_ut_histogram_decode"].items()
            }
        )
        generated_hashes.append(summary["generated_token_sha256"])
        for trace in summary["physical_trace"]:
            traces.append({"prompt_index": prompt_index, **trace})

    decode_events = sum(summary["num_decode_events"] for summary in summaries)
    selected_total = sum(
        depth * count for depth, count in histogram.items()
    )
    reference_executed = sum(
        summary["reference_executed_ut_total"] for summary in summaries
    )
    physical_executed = sum(
        summary["physical_executed_ut_total"] for summary in summaries
    )
    run_all_seconds = sum(
        summary["run_all_elapsed_seconds"] for summary in summaries
    )
    physical_seconds = sum(
        summary["physical_elapsed_seconds"] for summary in summaries
    )
    return {
        "num_sequences": sum(summary["num_sequences"] for summary in summaries),
        "num_generated_tokens": sum(
            summary["num_generated_tokens"] for summary in summaries
        ),
        "num_decode_events": decode_events,
        "selected_ut_histogram_decode": {
            str(depth): count for depth, count in sorted(histogram.items())
        },
        "mean_selected_ut_decode": selected_total / decode_events,
        "min_selected_ut_decode": min(histogram),
        "max_selected_ut_decode": max(histogram),
        "reference_executed_ut_total": reference_executed,
        "physical_executed_ut_total": physical_executed,
        "reference_decoder_layer_calls": sum(
            summary["reference_decoder_layer_calls"] for summary in summaries
        ),
        "physical_decoder_layer_calls": sum(
            summary["physical_decoder_layer_calls"] for summary in summaries
        ),
        "physical_recurrence_saving": 1.0 - physical_executed / reference_executed,
        "run_all_elapsed_seconds": run_all_seconds,
        "physical_elapsed_seconds": physical_seconds,
        "wall_time_speedup": run_all_seconds / physical_seconds,
        "run_all_peak_allocated_gib": max(
            summary["run_all_peak_allocated_gib"] for summary in summaries
        ),
        "physical_peak_allocated_gib": max(
            summary["physical_peak_allocated_gib"] for summary in summaries
        ),
        "run_all_peak_reserved_gib": max(
            summary["run_all_peak_reserved_gib"] for summary in summaries
        ),
        "physical_peak_reserved_gib": max(
            summary["physical_peak_reserved_gib"] for summary in summaries
        ),
        "generated_token_sha256_by_prompt": generated_hashes,
        "token_parity": all(summary["token_parity"] for summary in summaries),
        "selected_depth_parity": all(
            summary["selected_depth_parity"] for summary in summaries
        ),
        "total_ut_steps": summaries[0]["total_ut_steps"],
        "physical_trace": traces,
    }


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.thresholds))
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Every threshold must be within [0, 1]")

    model_source = args.model_path or args.model
    tokenizer_source = args.tokenizer_path or args.model
    source_revision = None if args.model_path is not None else args.revision
    tokenizer_revision = None if args.tokenizer_path is not None else args.revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=tokenizer_revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if args.batched_prompts:
        model_input_groups = [
            tokenizer(list(DEFAULT_PROMPTS), padding=True, return_tensors="pt").to(
                "cuda"
            )
        ]
    else:
        model_input_groups = [
            tokenizer(prompt, return_tensors="pt").to("cuda")
            for prompt in DEFAULT_PROMPTS
        ]

    payload = {
        "model": args.model,
        "revision": args.revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_ut_steps": args.total_ut_steps,
        "thresholds": thresholds,
        "max_new_tokens": args.max_new_tokens,
        "prompts": list(DEFAULT_PROMPTS),
        "prompt_execution": (
            "batched" if args.batched_prompts else "independent"
        ),
        "conditions": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for policy in ("last_available", "final_exit_shared"):
        config = OuroConfig.from_pretrained(model_source, revision=source_revision)
        config.total_ut_steps = args.total_ut_steps
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

        for threshold in thresholds:
            prompt_summaries = []
            for model_inputs in model_input_groups:
                reference = run_generation(
                    model,
                    tokenizer,
                    model_inputs,
                    physical=False,
                    threshold=threshold,
                    max_new_tokens=args.max_new_tokens,
                )
                physical = run_generation(
                    model,
                    tokenizer,
                    model_inputs,
                    physical=True,
                    threshold=threshold,
                    max_new_tokens=args.max_new_tokens,
                )
                prompt_summaries.append(
                    summarize_condition(
                        reference, physical, runtime_layers, args.total_ut_steps
                    )
                )
            summary = aggregate_prompt_summaries(prompt_summaries)
            summary.update(
                {
                    "policy": policy,
                    "threshold": threshold,
                    "num_hidden_layers": runtime_layers,
                }
            )
            payload["conditions"].append(summary)
            args.output.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"PASS policy={policy} threshold={threshold:.1f} "
                f"mean_selected={summary['mean_selected_ut_decode']:.3f} "
                f"saving={summary['physical_recurrence_saving']:.3f} "
                f"speedup={summary['wall_time_speedup']:.3f}",
                flush=True,
            )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
