"""Minimal BF16 ordinary-batching numerical-stability check for Ouro."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loopaware_agent.ut_trace import OuroUTTracer, TokenUTTrace  # noqa: E402
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"
GSM8K_ID = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH500_ID = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


@dataclass(frozen=True)
class Request:
    request_id: str
    dataset: str
    prompt: str


@dataclass
class Trajectory:
    request: Request
    token_ids: list[int]
    logits: list[torch.Tensor]
    events: list[TokenUTTrace]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--total-ut-steps", type=int, default=4)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.4])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--regression-tokens", type=int, default=16)
    parser.add_argument("--examples-per-dataset", type=int, default=4)
    parser.add_argument("--determinism-prompts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def render_prompt(tokenizer: Any, question: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {question}\nAssistant:"


def load_requests(tokenizer: Any, count: int) -> list[Request]:
    gsm = load_dataset(
        GSM8K_ID, "main", split="test", revision=GSM8K_REVISION
    )
    math500 = load_dataset(
        MATH500_ID, split="test", revision=MATH500_REVISION
    )
    requests = []
    for index in range(count):
        requests.extend(
            (
                Request(
                    f"gsm8k:{index}",
                    "gsm8k",
                    render_prompt(tokenizer, gsm[index]["question"]),
                ),
                Request(
                    f"math500:{index}",
                    "math500",
                    render_prompt(tokenizer, math500[index]["problem"]),
                ),
            )
        )
    return requests


def position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(attention_mask == 0, 1)


def set_mode(
    model: Any, *, policy: str, physical: bool, threshold: float | None
) -> None:
    for config in (model.config, model.model.config):
        config.kv_cache_policy = policy
        config.physical_early_exit = physical
        config.early_exit_threshold = threshold
    model.early_exit_threshold = threshold
    model.early_exit_step = None


def selected_ut(exit_probabilities: list[float], threshold: float) -> int:
    cumulative = 0.0
    for index, probability in enumerate(exit_probabilities):
        cumulative += probability
        if cumulative >= threshold:
            return index + 1
    return len(exit_probabilities)


def cdf_diagnostics(
    exit_probabilities: list[float], threshold: float
) -> dict[str, float | int]:
    cdf = []
    running = 0.0
    for probability in exit_probabilities:
        running += probability
        cdf.append(running)
    selected = selected_ut(exit_probabilities, threshold)
    return {
        "selected_ut": selected,
        "selected_cdf": cdf[selected - 1],
        "selected_cdf_minus_threshold": cdf[selected - 1] - threshold,
        "minimum_abs_cdf_threshold_distance": min(
            abs(value - threshold) for value in cdf
        ),
    }


def model_forward(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: Any = None,
    cache_position: torch.Tensor,
) -> Any:
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids(attention_mask)[:, -input_ids.shape[1] :],
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        logits_to_keep=1,
    )


def generate_one(
    model: Any,
    tokenizer: Any,
    request: Request,
    *,
    max_new_tokens: int,
    policy: str,
    physical: bool,
    threshold: float | None,
) -> Trajectory:
    set_mode(model, policy=policy, physical=physical, threshold=threshold)
    encoded = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    mask = encoded["attention_mask"]
    prompt_width = encoded["input_ids"].shape[1]
    token_ids: list[int] = []
    logits: list[torch.Tensor] = []
    with torch.inference_mode(), OuroUTTracer(
        model, exit_threshold=threshold if threshold is not None else 0.2
    ) as tracer:
        output = model_forward(
            model,
            input_ids=encoded["input_ids"],
            attention_mask=mask,
            cache_position=torch.arange(prompt_width, device="cuda"),
        )
        cache = output.past_key_values
        for output_index in range(max_new_tokens):
            step_logits = output.logits[:, -1, :]
            logits.append(step_logits[0].detach().float().cpu())
            next_token = int(torch.argmax(step_logits, dim=-1).item())
            token_ids.append(next_token)
            if (
                next_token == tokenizer.eos_token_id
                or output_index + 1 >= max_new_tokens
            ):
                break
            mask = torch.cat(
                (mask, torch.ones(1, 1, dtype=mask.dtype, device="cuda")), dim=1
            )
            output = model_forward(
                model,
                input_ids=torch.tensor([[next_token]], device="cuda"),
                attention_mask=mask,
                cache=cache,
                cache_position=torch.tensor(
                    [prompt_width + output_index], device="cuda"
                ),
            )
            cache = output.past_key_values
    if len(tracer.events) != len(token_ids):
        raise RuntimeError("Generation trace is not aligned with output tokens")
    return Trajectory(request, token_ids, logits, tracer.events)


def replay_batch(
    model: Any,
    tokenizer: Any,
    references: list[Trajectory],
) -> tuple[list[list[torch.Tensor]], list[list[TokenUTTrace]]]:
    set_mode(model, policy="full_per_depth", physical=False, threshold=None)
    encoded = tokenizer(
        [reference.request.prompt for reference in references],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    mask = encoded["attention_mask"]
    prompt_width = encoded["input_ids"].shape[1]
    max_length = max(len(reference.token_ids) for reference in references)
    logits_by_row: list[list[torch.Tensor]] = [[] for _ in references]
    with torch.inference_mode(), OuroUTTracer(model, exit_threshold=0.2) as tracer:
        output = model_forward(
            model,
            input_ids=encoded["input_ids"],
            attention_mask=mask,
            cache_position=torch.arange(prompt_width, device="cuda"),
        )
        cache = output.past_key_values
        for output_index in range(max_length):
            step_logits = output.logits[:, -1, :].detach().float().cpu()
            for row, reference in enumerate(references):
                if output_index < len(reference.token_ids):
                    logits_by_row[row].append(step_logits[row])
            if output_index + 1 >= max_length:
                break
            previous_tokens = [
                (
                    reference.token_ids[output_index]
                    if output_index < len(reference.token_ids)
                    else tokenizer.pad_token_id
                )
                for reference in references
            ]
            mask = torch.cat(
                (
                    mask,
                    torch.ones(
                        len(references), 1, dtype=mask.dtype, device="cuda"
                    ),
                ),
                dim=1,
            )
            output = model_forward(
                model,
                input_ids=torch.tensor(previous_tokens, device="cuda")[:, None],
                attention_mask=mask,
                cache=cache,
                cache_position=torch.tensor(
                    [prompt_width + output_index], device="cuda"
                ),
            )
            cache = output.past_key_values
    events_by_row: list[list[TokenUTTrace]] = [[] for _ in references]
    for event in tracer.events:
        if event.output_index < len(references[event.batch_index].token_ids):
            events_by_row[event.batch_index].append(event)
    for row, reference in enumerate(references):
        if len(events_by_row[row]) != len(reference.token_ids):
            raise RuntimeError("Replay trace is not aligned with reference tokens")
    return logits_by_row, events_by_row


def compare_position(
    reference_logits: torch.Tensor,
    replay_logits: torch.Tensor,
    reference_event: TokenUTTrace,
    replay_event: TokenUTTrace,
    thresholds: list[float],
) -> dict[str, Any]:
    difference = (reference_logits - replay_logits).abs()
    reference_top = torch.topk(reference_logits, 2)
    replay_top = torch.topk(replay_logits, 2)
    replay_top1 = int(replay_top.indices[0])
    selected_matches = {
        str(threshold): selected_ut(reference_event.exit_probabilities, threshold)
        == selected_ut(replay_event.exit_probabilities, threshold)
        for threshold in thresholds
    }
    return {
        "max_abs_logit_difference": float(difference.max()),
        "mean_abs_logit_difference": float(difference.mean()),
        "top1_agreement": int(reference_top.indices[0]) == replay_top1,
        "reference_top1": int(reference_top.indices[0]),
        "replay_top1": replay_top1,
        "reference_top1_top2_margin": float(
            reference_top.values[0] - reference_top.values[1]
        ),
        "replay_top1_top2_margin": float(
            replay_top.values[0] - replay_top.values[1]
        ),
        "selected_ut_agreement": selected_matches,
    }


def compare_trajectories(
    reference: Trajectory, replay: Trajectory, thresholds: list[float]
) -> dict[str, Any]:
    count = min(len(reference.logits), len(replay.logits))
    positions = [
        compare_position(
            reference.logits[index],
            replay.logits[index],
            reference.events[index],
            replay.events[index],
            thresholds,
        )
        for index in range(count)
    ]
    return {
        "token_ids_equal": reference.token_ids == replay.token_ids,
        "compared_positions": count,
        "maximum_abs_logit_difference": max(
            (row["max_abs_logit_difference"] for row in positions), default=0.0
        ),
        "mean_abs_logit_difference": (
            sum(row["mean_abs_logit_difference"] for row in positions) / count
            if count
            else 0.0
        ),
        "top1_agreement_fraction": (
            sum(row["top1_agreement"] for row in positions) / count if count else 1.0
        ),
        "selected_ut_agreement": {
            str(threshold): (
                sum(row["selected_ut_agreement"][str(threshold)] for row in positions)
                / count
                if count
                else 1.0
            )
            for threshold in thresholds
        },
    }


def compare_batch(
    batch_size: int,
    references: list[Trajectory],
    replay_logits: list[list[torch.Tensor]],
    replay_events: list[list[TokenUTTrace]],
    thresholds: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    mismatches = []
    for row_index, reference in enumerate(references):
        for output_index in range(len(reference.token_ids)):
            comparison = compare_position(
                reference.logits[output_index],
                replay_logits[row_index][output_index],
                reference.events[output_index],
                replay_events[row_index][output_index],
                thresholds,
            )
            row = {
                "batch_size": batch_size,
                "request_id": reference.request.request_id,
                "output_index": output_index,
                **comparison,
            }
            rows.append(row)
            if not comparison["top1_agreement"] or not all(
                comparison["selected_ut_agreement"].values()
            ):
                row["cdf_diagnostics"] = {
                    str(threshold): {
                        "reference": cdf_diagnostics(
                            reference.events[output_index].exit_probabilities,
                            threshold,
                        ),
                        "replay": cdf_diagnostics(
                            replay_events[row_index][output_index].exit_probabilities,
                            threshold,
                        ),
                    }
                    for threshold in thresholds
                }
                mismatches.append(row)
    return rows, mismatches


def aggregate(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    count = len(rows)
    return {
        "positions": count,
        "maximum_abs_logit_difference": max(
            (row["max_abs_logit_difference"] for row in rows), default=0.0
        ),
        "mean_abs_logit_difference": (
            sum(row["mean_abs_logit_difference"] for row in rows) / count
            if count
            else 0.0
        ),
        "top1_agreement_fraction": (
            sum(row["top1_agreement"] for row in rows) / count if count else 1.0
        ),
        "selected_ut_agreement": {
            str(threshold): (
                sum(row["selected_ut_agreement"][str(threshold)] for row in rows)
                / count
                if count
                else 1.0
            )
            for threshold in thresholds
        },
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Ordinary BF16 Batching Numerical-Stability Sanity Check",
        "",
        f"- Model: `{result['model']}` at `{result['model_revision']}`",
        f"- Runtime decoder layers: {result['runtime_num_hidden_layers']}",
        f"- UT steps: {result['total_ut_steps']}",
        f"- Exit thresholds evaluated post hoc: `{result['thresholds']}`",
        f"- Reference: B=1 greedy trajectories, generated once",
        f"- Replay: original full-compute `full_per_depth`; physical exit disabled",
        "",
        "## Threshold-one regression",
        "",
        f"- Passed: **{result['threshold_one_regression']['passed']}**",
        f"- Maximum absolute logit difference: "
        f"{result['threshold_one_regression']['maximum_absolute_logit_difference']:.6f}",
        "",
        "## B=1 repeat determinism",
        "",
        f"- Deterministic: **{result['b1_determinism']['deterministic']}**",
        f"- Maximum absolute logit difference: "
        f"{result['b1_determinism']['maximum_absolute_logit_difference']:.6f}",
        "",
        "## Ordinary batching replay",
        "",
        "| Batch | Positions | Maximum absolute logit difference | Mean absolute logit difference | Top-1 agreement | Selected-UT @ 0.2 | Selected-UT @ 0.4 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch_size in (1, 4, 8):
        row = result["ordinary_batching"][str(batch_size)]
        lines.append(
            f"| {batch_size} | {row['positions']} | "
            f"{row['maximum_abs_logit_difference']:.6f} | "
            f"{row['mean_abs_logit_difference']:.6e} | "
            f"{100 * row['top1_agreement_fraction']:.3f}% | "
            f"{100 * row['selected_ut_agreement']['0.2']:.3f}% | "
            f"{100 * row['selected_ut_agreement']['0.4']:.3f}% |"
        )
    lines.extend(
        [
            "",
            f"Mismatch positions with additional diagnostics: {len(result['mismatches'])}",
            "",
            "The ~0.125 value is a comparison scale from the earlier observation, not a pass/fail tolerance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.examples_per_dataset != 4:
        raise ValueError("This reduced protocol requires exactly 4 examples per dataset")
    if args.total_ut_steps != 4:
        raise ValueError("This reduced protocol is fixed at UT=4")
    if args.max_new_tokens > 32:
        raise ValueError("This reduced protocol permits at most 32 output positions")
    thresholds = [float(value) for value in args.thresholds]
    set_determinism(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_source = args.model_path or args.model
    tokenizer_source = args.tokenizer_path or args.model
    revision = None if args.model_path else args.model_revision
    tokenizer_revision = None if args.tokenizer_path else args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=tokenizer_revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    requests = load_requests(tokenizer, args.examples_per_dataset)

    config = OuroConfig.from_pretrained(model_source, revision=revision)
    config.total_ut_steps = args.total_ut_steps
    config.kv_cache_policy = "full_per_depth"
    config.physical_early_exit = False
    config.early_exit_threshold = None
    model = OuroForCausalLM.from_pretrained(
        model_source,
        revision=revision,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    runtime_layers = model.config.num_hidden_layers
    print(f"runtime_num_hidden_layers={runtime_layers}", flush=True)

    references = []
    for index, request in enumerate(requests):
        reference = generate_one(
            model,
            tokenizer,
            request,
            max_new_tokens=args.max_new_tokens,
            policy="full_per_depth",
            physical=False,
            threshold=None,
        )
        references.append(reference)
        print(f"REFERENCE {index + 1}/{len(requests)} {request.request_id}", flush=True)

    repeat_rows = []
    for reference in references[: args.determinism_prompts]:
        repeated = generate_one(
            model,
            tokenizer,
            reference.request,
            max_new_tokens=args.max_new_tokens,
            policy="full_per_depth",
            physical=False,
            threshold=None,
        )
        repeat_rows.append(compare_trajectories(reference, repeated, thresholds))
    determinism = {
        "prompts": len(repeat_rows),
        "deterministic": all(
            row["token_ids_equal"]
            and row["maximum_abs_logit_difference"] == 0.0
            for row in repeat_rows
        ),
        "maximum_absolute_logit_difference": max(
            (row["maximum_abs_logit_difference"] for row in repeat_rows),
            default=0.0,
        ),
        "per_prompt": repeat_rows,
    }
    print(f"B1_DETERMINISTIC={determinism['deterministic']}", flush=True)

    regression_rows = []
    for reference in references[:4]:
        physical = generate_one(
            model,
            tokenizer,
            reference.request,
            max_new_tokens=min(args.regression_tokens, len(reference.token_ids)),
            policy="last_available",
            physical=True,
            threshold=1.0,
        )
        truncated_reference = Trajectory(
            reference.request,
            reference.token_ids[: len(physical.token_ids)],
            reference.logits[: len(physical.logits)],
            reference.events[: len(physical.events)],
        )
        regression_rows.append(
            compare_trajectories(truncated_reference, physical, [1.0])
        )
    threshold_one = {
        "prompts": len(regression_rows),
        "passed": all(
            row["token_ids_equal"]
            and row["maximum_abs_logit_difference"] == 0.0
            and row["selected_ut_agreement"]["1.0"] == 1.0
            for row in regression_rows
        ),
        "maximum_absolute_logit_difference": max(
            (row["maximum_abs_logit_difference"] for row in regression_rows),
            default=0.0,
        ),
        "per_prompt": regression_rows,
    }
    print(f"THRESHOLD_ONE_PASSED={threshold_one['passed']}", flush=True)

    all_rows = []
    mismatches = []
    by_batch = {
        "1": {
            "positions": sum(len(reference.token_ids) for reference in references),
            "maximum_abs_logit_difference": 0.0,
            "mean_abs_logit_difference": 0.0,
            "top1_agreement_fraction": 1.0,
            "selected_ut_agreement": {str(value): 1.0 for value in thresholds},
        }
    }
    for batch_size in (4, 8):
        batch_rows = []
        for start in range(0, len(references), batch_size):
            batch = references[start : start + batch_size]
            replay_logits, replay_events = replay_batch(model, tokenizer, batch)
            rows, batch_mismatches = compare_batch(
                batch_size,
                batch,
                replay_logits,
                replay_events,
                thresholds,
            )
            batch_rows.extend(rows)
            mismatches.extend(batch_mismatches)
        by_batch[str(batch_size)] = aggregate(batch_rows, thresholds)
        all_rows.extend(batch_rows)
        print(
            f"BATCH={batch_size} max_abs="
            f"{by_batch[str(batch_size)]['maximum_abs_logit_difference']:.6f}",
            flush=True,
        )

    result = {
        "experiment": "ordinary_bf16_batching_numerical_stability",
        "created_at": utc_now(),
        "model": args.model,
        "model_revision": args.model_revision,
        "runtime_num_hidden_layers": runtime_layers,
        "total_ut_steps": args.total_ut_steps,
        "thresholds": thresholds,
        "max_new_tokens": args.max_new_tokens,
        "requests": [asdict(request) for request in requests],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": str(next(model.parameters()).dtype),
        },
        "threshold_one_regression": threshold_one,
        "b1_determinism": determinism,
        "ordinary_batching": by_batch,
        "mismatches": mismatches,
        "comparison_to_previous_0_125": {
            str(batch_size): by_batch[str(batch_size)][
                "maximum_abs_logit_difference"
            ]
            / 0.125
            for batch_size in (4, 8)
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RESULTS.md").write_text(
        markdown(result), encoding="utf-8"
    )
    print(markdown(result), flush=True)


if __name__ == "__main__":
    main()
