"""Characterize BF16 numerical stability of physically batched Ouro decode.

This is a numerical-execution experiment, not a threshold-quality evaluation.
It compares the last-available KV policy's run-all semantic reference with
physical adaptive recurrence under deterministic batching, then replays the
reference token stream to measure logit drift without autoregressive feedback.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Iterable

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"
GSM8K_ID = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH500_ID = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
POLICY = "last_available"
MARGIN_BOUNDS = (0.01, 0.05, 0.10, 0.25)


@dataclass(frozen=True)
class Request:
    request_id: str
    dataset: str
    dataset_index: int
    source_id: str | None
    question: str
    prompt: str
    prompt_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.4])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--groupings", nargs="+", default=["contiguous", "reversed", "strided"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--total-ut-steps", type=int, default=6)
    parser.add_argument("--examples-per-dataset", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_prompt(tokenizer: Any, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"User: {question}\nAssistant:"


def load_requests(tokenizer: Any, count: int) -> list[Request]:
    gsm = load_dataset(
        GSM8K_ID,
        "main",
        split="test",
        revision=GSM8K_REVISION,
    )
    math500 = load_dataset(
        MATH500_ID,
        split="test",
        revision=MATH500_REVISION,
    )
    if count > len(gsm) or count > len(math500):
        raise ValueError("examples-per-dataset exceeds a requested dataset")

    requests: list[Request] = []
    # Interleave domains so contiguous batches are not accidentally domain-pure.
    for index in range(count):
        for dataset, question, source_id in (
            ("gsm8k", gsm[index]["question"], None),
            ("math500", math500[index]["problem"], math500[index]["unique_id"]),
        ):
            prompt = render_prompt(tokenizer, question)
            requests.append(
                Request(
                    request_id=f"{dataset}:{index}",
                    dataset=dataset,
                    dataset_index=index,
                    source_id=source_id,
                    question=question,
                    prompt=prompt,
                    prompt_sha256=sha256_text(prompt),
                )
            )
    return requests


def permutation(size: int, grouping: str) -> list[int]:
    if grouping == "contiguous":
        return list(range(size))
    if grouping == "reversed":
        return list(reversed(range(size)))
    if grouping == "strided":
        stride = 8 if size % 8 == 0 else 2
        return [index for offset in range(stride) for index in range(offset, size, stride)]
    raise ValueError(f"Unknown grouping {grouping!r}")


def batches_for(
    requests: list[Request], batch_size: int, grouping: str
) -> Iterable[tuple[int, list[Request]]]:
    indices = permutation(len(requests), grouping)
    for batch_index, start in enumerate(range(0, len(indices), batch_size)):
        yield batch_index, [requests[index] for index in indices[start : start + batch_size]]


def set_mode(model: Any, *, physical: bool, threshold: float) -> None:
    model.config.physical_early_exit = physical
    model.model.config.physical_early_exit = physical
    model.config.early_exit_threshold = threshold
    model.model.config.early_exit_threshold = threshold
    model.early_exit_threshold = threshold


def position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(attention_mask == 0, 1)


def prefill(model: Any, encoded: dict[str, torch.Tensor], *, physical: bool, threshold: float):
    set_mode(model, physical=physical, threshold=threshold)
    width = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            position_ids=position_ids(encoded["attention_mask"]),
            cache_position=torch.arange(width, device=encoded["input_ids"].device),
            use_cache=True,
            logits_to_keep=1,
        )
    return output


def decode_one(
    model: Any,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: Any,
    cache_position: int,
    *,
    physical: bool,
    threshold: float,
):
    set_mode(model, physical=physical, threshold=threshold)
    with torch.inference_mode():
        return model(
            input_ids=token_ids[:, None],
            attention_mask=attention_mask,
            position_ids=position_ids(attention_mask)[:, -1:],
            past_key_values=cache,
            cache_position=torch.tensor([cache_position], device=token_ids.device),
            use_cache=True,
            logits_to_keep=1,
        )


def margin_summary(logits: torch.Tensor) -> tuple[list[int], list[float]]:
    top = torch.topk(logits.float(), 2, dim=-1)
    return top.indices[:, 0].cpu().tolist(), (top.values[:, 0] - top.values[:, 1]).cpu().tolist()


def run_closed_loop(
    model: Any,
    tokenizer: Any,
    batch: list[Request],
    *,
    physical: bool,
    threshold: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    encoded = tokenizer(
        [request.prompt for request in batch],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    prompt_width = encoded["input_ids"].shape[1]
    started = time.perf_counter()
    first = prefill(model, encoded, physical=physical, threshold=threshold)
    logits = first.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1)
    generated = [next_token.detach().cpu()]
    top1, margins = margin_summary(logits)
    top1_by_step = [top1]
    margins_by_step = [margins]
    selected_by_step: list[list[int]] = []
    executed_by_step: list[list[int]] = []
    finished = next_token.eq(tokenizer.eos_token_id)
    cache = first.past_key_values
    mask = encoded["attention_mask"]

    for output_index in range(1, max_new_tokens):
        mask = torch.cat(
            (mask, torch.ones(mask.shape[0], 1, dtype=mask.dtype, device=mask.device)),
            dim=1,
        )
        output = decode_one(
            model,
            next_token,
            mask,
            cache,
            prompt_width + output_index - 1,
            physical=physical,
            threshold=threshold,
        )
        if output.selected_ut_steps is None or output.executed_ut_steps is None:
            raise RuntimeError("Adaptive decode did not expose execution metadata")
        selected_by_step.append(output.selected_ut_steps.detach().cpu().tolist())
        executed_by_step.append(output.executed_ut_steps.detach().cpu().tolist())
        logits = output.logits[:, -1, :]
        candidate = torch.argmax(logits, dim=-1)
        fill = torch.full_like(candidate, tokenizer.pad_token_id)
        next_token = torch.where(finished, fill, candidate)
        generated.append(next_token.detach().cpu())
        top1, margins = margin_summary(logits)
        top1_by_step.append(top1)
        margins_by_step.append(margins)
        finished |= candidate.eq(tokenizer.eos_token_id)
        cache = output.past_key_values
        if bool(finished.all()):
            break

    torch.cuda.synchronize()
    token_matrix = torch.stack(generated, dim=1)
    valid_lengths = []
    for row in token_matrix.tolist():
        try:
            valid_lengths.append(row.index(tokenizer.eos_token_id) + 1)
        except ValueError:
            valid_lengths.append(len(row))
    return {
        "generated_ids": token_matrix.tolist(),
        "valid_lengths": valid_lengths,
        "selected_ut_by_decode_step": selected_by_step,
        "executed_ut_by_decode_step": executed_by_step,
        "top1_by_output_step": top1_by_step,
        "top2_margin_by_output_step": margins_by_step,
        "elapsed_seconds": time.perf_counter() - started,
    }


def first_difference(left: list[Any], right: list[Any], limit: int) -> int | None:
    for index in range(min(limit, len(left), len(right))):
        if left[index] != right[index]:
            return index
    if min(len(left), limit) != min(len(right), limit):
        return min(len(left), len(right), limit)
    return None


def compare_closed_loop(
    reference: dict[str, Any], physical: dict[str, Any], request_ids: list[str]
) -> dict[str, Any]:
    rows = []
    for row, request_id in enumerate(request_ids):
        limit = min(reference["valid_lengths"][row], physical["valid_lengths"][row])
        token_diff = first_difference(
            reference["generated_ids"][row], physical["generated_ids"][row], limit
        )
        reference_depths = [step[row] for step in reference["selected_ut_by_decode_step"]]
        physical_depths = [step[row] for step in physical["selected_ut_by_decode_step"]]
        depth_diff = first_difference(reference_depths, physical_depths, max(0, limit - 1))
        rows.append(
            {
                "request_id": request_id,
                "compared_tokens": limit,
                "first_token_mismatch": token_diff,
                "first_selected_ut_mismatch_decode_index": depth_diff,
                "reference_length": reference["valid_lengths"][row],
                "physical_length": physical["valid_lengths"][row],
            }
        )
    return {
        "per_request": rows,
        "token_exact_match_fraction": sum(row["first_token_mismatch"] is None for row in rows) / len(rows),
        "selected_ut_exact_match_fraction": sum(
            row["first_selected_ut_mismatch_decode_index"] is None for row in rows
        ) / len(rows),
    }


def logit_metrics(reference: torch.Tensor, physical: torch.Tensor) -> dict[str, list[Any]]:
    left = reference.float()
    right = physical.float()
    delta = (left - right).abs()
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    top_left = torch.topk(left, 5, dim=-1)
    top_right = torch.topk(right, 5, dim=-1)
    top5_overlap = (
        top_left.indices[:, :, None] == top_right.indices[:, None, :]
    ).any(dim=2).sum(dim=1).float() / 5.0
    log_p = torch.log_softmax(left, dim=-1)
    log_q = torch.log_softmax(right, dim=-1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    p = log_p.exp()
    q = log_q.exp()
    js = 0.5 * (
        (p * (log_p - log_m)).sum(dim=-1)
        + (q * (log_q - log_m)).sum(dim=-1)
    )
    return {
        "max_abs_logit_difference": delta.amax(dim=-1).cpu().tolist(),
        "mean_abs_logit_difference": delta.mean(dim=-1).cpu().tolist(),
        "relative_l2_logit_difference": (
            torch.linalg.vector_norm(left - right, dim=-1) / left_norm.clamp_min(1e-12)
        ).cpu().tolist(),
        "js_divergence": js.cpu().tolist(),
        "reference_top1": top_left.indices[:, 0].cpu().tolist(),
        "physical_top1": top_right.indices[:, 0].cpu().tolist(),
        "top1_agreement": (top_left.indices[:, 0] == top_right.indices[:, 0]).cpu().tolist(),
        "top5_overlap": top5_overlap.cpu().tolist(),
        "reference_top2_margin": (
            top_left.values[:, 0] - top_left.values[:, 1]
        ).cpu().tolist(),
    }


def margin_bucket(value: float) -> str:
    lower = 0.0
    for upper in MARGIN_BOUNDS:
        if value <= upper:
            return f"({lower:.2f},{upper:.2f}]"
        lower = upper
    return f">{MARGIN_BOUNDS[-1]:.2f}"


def run_teacher_forced(
    model: Any,
    tokenizer: Any,
    batch: list[Request],
    reference_tokens: list[list[int]],
    valid_lengths: list[int],
    *,
    threshold: float,
) -> dict[str, Any]:
    encoded = tokenizer(
        [request.prompt for request in batch], padding=True, return_tensors="pt"
    ).to("cuda")
    token_matrix = torch.tensor(reference_tokens, device="cuda", dtype=torch.long)
    prompt_width = encoded["input_ids"].shape[1]
    reference_prefill = prefill(model, encoded, physical=False, threshold=threshold)
    physical_prefill = prefill(model, encoded, physical=True, threshold=threshold)
    prefill_metrics = logit_metrics(
        reference_prefill.logits[:, -1, :], physical_prefill.logits[:, -1, :]
    )
    reference_cache = reference_prefill.past_key_values
    physical_cache = physical_prefill.past_key_values
    reference_mask = encoded["attention_mask"]
    physical_mask = encoded["attention_mask"].clone()
    positions: list[dict[str, Any]] = []

    for input_index in range(token_matrix.shape[1] - 1):
        ones = torch.ones(
            token_matrix.shape[0], 1, dtype=reference_mask.dtype, device="cuda"
        )
        reference_mask = torch.cat((reference_mask, ones), dim=1)
        physical_mask = torch.cat((physical_mask, ones), dim=1)
        inputs = token_matrix[:, input_index]
        reference_output = decode_one(
            model,
            inputs,
            reference_mask,
            reference_cache,
            prompt_width + input_index,
            physical=False,
            threshold=threshold,
        )
        physical_output = decode_one(
            model,
            inputs,
            physical_mask,
            physical_cache,
            prompt_width + input_index,
            physical=True,
            threshold=threshold,
        )
        metrics = logit_metrics(
            reference_output.logits[:, -1, :], physical_output.logits[:, -1, :]
        )
        reference_depths = reference_output.selected_ut_steps.detach().cpu().tolist()
        physical_depths = physical_output.selected_ut_steps.detach().cpu().tolist()
        active_counts = [
            sum(depth >= recurrence for depth in physical_depths)
            for recurrence in range(1, model.model.total_ut_steps + 1)
        ]
        output_index = input_index + 1
        for row, request in enumerate(batch):
            if output_index >= valid_lengths[row]:
                continue
            position = {
                "request_id": request.request_id,
                "output_index": output_index,
                "reference_selected_ut": reference_depths[row],
                "physical_selected_ut": physical_depths[row],
                "selected_ut_match": reference_depths[row] == physical_depths[row],
                "active_rows_by_recurrence": active_counts,
            }
            for name, values in metrics.items():
                position[name] = values[row]
            position["margin_bucket"] = margin_bucket(
                position["reference_top2_margin"]
            )
            positions.append(position)
        reference_cache = reference_output.past_key_values
        physical_cache = physical_output.past_key_values

    return {
        "prefill_logit_metrics": prefill_metrics,
        "positions": positions,
    }


def summarize_teacher(positions: list[dict[str, Any]]) -> dict[str, Any]:
    if not positions:
        return {"num_positions": 0}
    by_margin: dict[str, list[dict[str, Any]]] = {}
    for row in positions:
        by_margin.setdefault(row["margin_bucket"], []).append(row)

    def mean(name: str, rows: list[dict[str, Any]]) -> float:
        return sum(float(row[name]) for row in rows) / len(rows)

    return {
        "num_positions": len(positions),
        "top1_agreement_fraction": mean("top1_agreement", positions),
        "selected_ut_agreement_fraction": mean("selected_ut_match", positions),
        "mean_max_abs_logit_difference": mean("max_abs_logit_difference", positions),
        "maximum_abs_logit_difference": max(
            float(row["max_abs_logit_difference"]) for row in positions
        ),
        "mean_js_divergence": mean("js_divergence", positions),
        "mean_top5_overlap": mean("top5_overlap", positions),
        "by_margin": {
            bucket: {
                "count": len(rows),
                "top1_agreement_fraction": mean("top1_agreement", rows),
                "mean_max_abs_logit_difference": mean(
                    "max_abs_logit_difference", rows
                ),
            }
            for bucket, rows in sorted(by_margin.items())
        },
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                keys.add(json.loads(line)["condition_key"])
    return keys


def environment_metadata(model: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "attention_implementation": model.config._attn_implementation,
        "dtype": str(next(model.parameters()).dtype),
        "num_hidden_layers": model.config.num_hidden_layers,
    }


def main() -> None:
    args = parse_args()
    if any(value <= 0 for value in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.thresholds):
        raise ValueError("thresholds must lie in [0, 1]")
    set_determinism(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "conditions.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"{records_path} exists; pass --resume to continue")

    model_source = args.model_path or args.model
    tokenizer_source = args.tokenizer_path or args.model
    model_revision = None if args.model_path else args.model_revision
    tokenizer_revision = None if args.tokenizer_path else args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=tokenizer_revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    requests = load_requests(tokenizer, args.examples_per_dataset)

    config = OuroConfig.from_pretrained(model_source, revision=model_revision)
    config.total_ut_steps = args.total_ut_steps
    config.kv_cache_policy = POLICY
    config.physical_early_exit = False
    model = OuroForCausalLM.from_pretrained(
        model_source,
        revision=model_revision,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    runtime_layers = model.config.num_hidden_layers
    if not isinstance(runtime_layers, int) or runtime_layers <= 0:
        raise ValueError(f"Invalid runtime num_hidden_layers={runtime_layers!r}")

    manifest = {
        "experiment": "physically_batched_adaptive_recurrence_numerics",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "numerical stability characterization; not threshold quality",
        "model": args.model,
        "model_revision": args.model_revision,
        "kv_policy": POLICY,
        "prefill": "full_depth",
        "decode": "one_token_adaptive",
        "total_ut_steps": args.total_ut_steps,
        "thresholds_are_numerical_probes": args.thresholds,
        "batch_sizes": args.batch_sizes,
        "groupings": args.groupings,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "datasets": {
            "gsm8k": {"id": GSM8K_ID, "revision": GSM8K_REVISION, "split": "test"},
            "math500": {"id": MATH500_ID, "revision": MATH500_REVISION, "split": "test"},
        },
        "environment": environment_metadata(model),
        "requests": [asdict(request) for request in requests],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    done = completed_keys(records_path)

    for threshold in args.thresholds:
        for batch_size in args.batch_sizes:
            groupings = [args.groupings[0]] if batch_size == 1 else args.groupings
            for grouping in groupings:
                for batch_index, batch in batches_for(requests, batch_size, grouping):
                    key = f"t={threshold}:b={batch_size}:g={grouping}:i={batch_index}"
                    if key in done:
                        continue
                    request_ids = [request.request_id for request in batch]
                    closed_runs = []
                    first_reference = None
                    for repeat in range(args.repeats):
                        reference = run_closed_loop(
                            model,
                            tokenizer,
                            batch,
                            physical=False,
                            threshold=threshold,
                            max_new_tokens=args.max_new_tokens,
                        )
                        physical = run_closed_loop(
                            model,
                            tokenizer,
                            batch,
                            physical=True,
                            threshold=threshold,
                            max_new_tokens=args.max_new_tokens,
                        )
                        if first_reference is None:
                            first_reference = reference
                        closed_runs.append(
                            {
                                "repeat": repeat,
                                "comparison": compare_closed_loop(
                                    reference, physical, request_ids
                                ),
                                "reference_elapsed_seconds": reference["elapsed_seconds"],
                                "physical_elapsed_seconds": physical["elapsed_seconds"],
                                "reference_generated_ids": reference["generated_ids"],
                                "physical_generated_ids": physical["generated_ids"],
                                "reference_selected_ut_by_decode_step": reference[
                                    "selected_ut_by_decode_step"
                                ],
                                "physical_selected_ut_by_decode_step": physical[
                                    "selected_ut_by_decode_step"
                                ],
                            }
                        )

                    if first_reference is None:
                        raise RuntimeError("No closed-loop reference was produced")
                    teacher = run_teacher_forced(
                        model,
                        tokenizer,
                        batch,
                        first_reference["generated_ids"],
                        first_reference["valid_lengths"],
                        threshold=threshold,
                    )
                    record = {
                        "condition_key": key,
                        "threshold": threshold,
                        "batch_size": len(batch),
                        "requested_batch_size": batch_size,
                        "grouping": grouping,
                        "batch_index": batch_index,
                        "request_ids": request_ids,
                        "closed_loop_runs": closed_runs,
                        "teacher_forced_summary": summarize_teacher(teacher["positions"]),
                        "teacher_forced": teacher,
                    }
                    append_jsonl(records_path, record)
                    done.add(key)
                    summary = record["teacher_forced_summary"]
                    closed_match = closed_runs[0]["comparison"]["token_exact_match_fraction"]
                    print(
                        f"PASS {key} closed_exact={closed_match:.3f} "
                        f"teacher_top1={summary.get('top1_agreement_fraction', float('nan')):.6f} "
                        f"positions={summary.get('num_positions', 0)}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
