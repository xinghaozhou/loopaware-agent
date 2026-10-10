"""Aggregate the checkpointed physically batched Ouro numerical experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}") from error
    return records


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_teacher(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_positions": 0}
    maximum_differences = [float(row["max_abs_logit_difference"]) for row in rows]
    mean_differences = [float(row["mean_abs_logit_difference"]) for row in rows]
    js_values = [float(row["js_divergence"]) for row in rows]
    return {
        "num_positions": len(rows),
        "top1_agreement_fraction": mean(bool(row["top1_agreement"]) for row in rows),
        "selected_ut_agreement_fraction": mean(
            bool(row["selected_ut_match"]) for row in rows
        ),
        "mean_max_abs_logit_difference": mean(maximum_differences),
        "median_max_abs_logit_difference": median(maximum_differences),
        "p95_max_abs_logit_difference": quantile(maximum_differences, 0.95),
        "p99_max_abs_logit_difference": quantile(maximum_differences, 0.99),
        "maximum_abs_logit_difference": max(maximum_differences),
        "mean_abs_logit_difference": mean(mean_differences),
        "mean_relative_l2_logit_difference": mean(
            float(row["relative_l2_logit_difference"]) for row in rows
        ),
        "mean_js_divergence": mean(js_values),
        "maximum_js_divergence": max(js_values),
        "mean_top5_overlap": mean(float(row["top5_overlap"]) for row in rows),
    }


def closed_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for run in record["closed_loop_runs"]:
            for request in run["comparison"]["per_request"]:
                rows.append(
                    {
                        "threshold": record["threshold"],
                        "batch_size": record["requested_batch_size"],
                        "grouping": record["grouping"],
                        "repeat": run["repeat"],
                        **request,
                    }
                )
    return rows


def teacher_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for position in record["teacher_forced"]["positions"]:
            rows.append(
                {
                    "threshold": record["threshold"],
                    "batch_size": record["requested_batch_size"],
                    "grouping": record["grouping"],
                    **position,
                }
            )
    return rows


def group_by(rows: list[dict[str, Any]], names: tuple[str, ...]):
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in names)].append(row)
    return grouped


def aggregate_closed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_request_runs": len(rows),
        "token_exact_match_fraction": mean(
            row["first_token_mismatch"] is None for row in rows
        ),
        "selected_ut_exact_match_fraction": mean(
            row["first_selected_ut_mismatch_decode_index"] is None for row in rows
        ),
        "token_divergence_count": sum(
            row["first_token_mismatch"] is not None for row in rows
        ),
        "selected_ut_divergence_count": sum(
            row["first_selected_ut_mismatch_decode_index"] is not None for row in rows
        ),
        "median_first_token_mismatch": (
            median(
                row["first_token_mismatch"]
                for row in rows
                if row["first_token_mismatch"] is not None
            )
            if any(row["first_token_mismatch"] is not None for row in rows)
            else None
        ),
    }


def repeat_stability(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference_equal = []
    physical_equal = []
    reference_depth_equal = []
    physical_depth_equal = []
    for record in records:
        runs = record["closed_loop_runs"]
        baseline = runs[0]
        for run in runs[1:]:
            reference_equal.append(
                run["reference_generated_ids"] == baseline["reference_generated_ids"]
            )
            physical_equal.append(
                run["physical_generated_ids"] == baseline["physical_generated_ids"]
            )
            reference_depth_equal.append(
                run["reference_selected_ut_by_decode_step"]
                == baseline["reference_selected_ut_by_decode_step"]
            )
            physical_depth_equal.append(
                run["physical_selected_ut_by_decode_step"]
                == baseline["physical_selected_ut_by_decode_step"]
            )
    return {
        "num_repeat_comparisons": len(reference_equal),
        "reference_token_repeatability": mean(reference_equal) if reference_equal else None,
        "physical_token_repeatability": mean(physical_equal) if physical_equal else None,
        "reference_depth_repeatability": mean(reference_depth_equal) if reference_depth_equal else None,
        "physical_depth_repeatability": mean(physical_depth_equal) if physical_depth_equal else None,
    }


def expected_conditions(manifest: dict[str, Any]) -> int:
    request_count = len(manifest["requests"])
    grouping_count = len(manifest["groupings"])
    per_threshold = 0
    for batch_size in manifest["batch_sizes"]:
        batches = math.ceil(request_count / batch_size)
        per_threshold += batches * (1 if batch_size == 1 else grouping_count)
    return len(manifest["thresholds_are_numerical_probes"]) * per_threshold


def summarize(records: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    closed = closed_rows(records)
    teacher = teacher_rows(records)
    closed_by_batch = {
        f"threshold={threshold},batch={batch}": aggregate_closed(rows)
        for (threshold, batch), rows in sorted(
            group_by(closed, ("threshold", "batch_size")).items()
        )
    }
    teacher_by_batch = {
        f"threshold={threshold},batch={batch}": aggregate_teacher(rows)
        for (threshold, batch), rows in sorted(
            group_by(teacher, ("threshold", "batch_size")).items()
        )
    }
    teacher_by_margin = {
        f"threshold={threshold},margin={margin}": aggregate_teacher(rows)
        for (threshold, margin), rows in sorted(
            group_by(teacher, ("threshold", "margin_bucket")).items()
        )
    }
    teacher_by_active_rows = {
        f"threshold={threshold},batch={batch},active={active}": aggregate_teacher(rows)
        for (threshold, batch, active), rows in sorted(
            group_by(
                [
                    {**row, "minimum_active_rows": min(row["active_rows_by_recurrence"])}
                    for row in teacher
                ],
                ("threshold", "batch_size", "minimum_active_rows"),
            ).items()
        )
    }
    return {
        "complete_conditions": len(records),
        "expected_conditions": expected_conditions(manifest),
        "repeat_stability": repeat_stability(records),
        "closed_loop_overall": aggregate_closed(closed) if closed else {},
        "teacher_forced_overall": aggregate_teacher(teacher),
        "closed_loop_by_threshold_and_batch": closed_by_batch,
        "teacher_forced_by_threshold_and_batch": teacher_by_batch,
        "teacher_forced_by_threshold_and_margin": teacher_by_margin,
        "teacher_forced_by_threshold_batch_and_minimum_active_rows": teacher_by_active_rows,
        "runtime_num_hidden_layers": manifest["environment"]["num_hidden_layers"],
    }


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.3f}%"


def scientific(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3e}"


def markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    dataset_counts: dict[str, int] = defaultdict(int)
    for request in manifest["requests"]:
        dataset_counts[request["dataset"]] += 1
    lines = [
        "# Physically Batched Adaptive Recurrence: Numerical Stability",
        "",
        "> This is a numerical-execution characterization, not a threshold-quality experiment.",
        "",
        "## Configuration",
        "",
        f"- Model: `{manifest['model']}` at `{manifest['model_revision']}`",
        f"- Runtime decoder layers: {summary['runtime_num_hidden_layers']}",
        f"- Policy: `{manifest['kv_policy']}`; BF16; greedy decoding; UT={manifest['total_ut_steps']}",
        f"- Threshold probes: `{manifest['thresholds_are_numerical_probes']}`",
        f"- Batch sizes: `{manifest['batch_sizes']}`",
        f"- Completed conditions: {summary['complete_conditions']} / {summary['expected_conditions']}",
        f"- Dataset: {dataset_counts['gsm8k']} pinned GSM8K test requests and "
        f"{dataset_counts['math500']} pinned MATH-500 test requests",
        "- Prefill is full-depth; physical adaptive exit applies only to one-token decode.",
        "",
        "## Closed-loop generation",
        "",
        "| Threshold | Batch | Request-runs | Token exact | Selected-UT exact | Divergences | Median first mismatch |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary["closed_loop_by_threshold_and_batch"].items():
        pieces = dict(part.split("=") for part in key.split(","))
        lines.append(
            f"| {pieces['threshold']} | {pieces['batch']} | {row['num_request_runs']} | "
            f"{percent(row['token_exact_match_fraction'])} | "
            f"{percent(row['selected_ut_exact_match_fraction'])} | "
            f"{row['token_divergence_count']} | {row['median_first_token_mismatch']} |"
        )
    lines.extend(
        [
            "",
            "## Teacher-forced fixed-trajectory replay",
            "",
            "| Threshold | Batch | Positions | Top-1 agree | Selected-UT agree | Mean max | P99 max | Maximum | Mean JS | Top-5 overlap |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, row in summary["teacher_forced_by_threshold_and_batch"].items():
        pieces = dict(part.split("=") for part in key.split(","))
        lines.append(
            f"| {pieces['threshold']} | {pieces['batch']} | {row['num_positions']} | "
            f"{percent(row['top1_agreement_fraction'])} | "
            f"{percent(row['selected_ut_agreement_fraction'])} | "
            f"{row['mean_max_abs_logit_difference']:.6f} | "
            f"{row['p99_max_abs_logit_difference']:.6f} | "
            f"{row['maximum_abs_logit_difference']:.6f} | "
            f"{scientific(row['mean_js_divergence'])} | "
            f"{percent(row['mean_top5_overlap'])} |"
        )
    repeat = summary["repeat_stability"]
    lines.extend(
        [
            "",
            "## Exact-repeat stability",
            "",
            f"- Reference token repeatability: {percent(repeat['reference_token_repeatability'])}",
            f"- Physical token repeatability: {percent(repeat['physical_token_repeatability'])}",
            f"- Reference selected-UT repeatability: {percent(repeat['reference_depth_repeatability'])}",
            f"- Physical selected-UT repeatability: {percent(repeat['physical_depth_repeatability'])}",
            "",
            "## Interpretation",
            "",
            "Interpretation is intentionally left for the completed run. Threshold probes must not be treated as quality or deployment recommendations.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    records = load_records(args.input_dir / "conditions.jsonl")
    expected = expected_conditions(manifest)
    if len(records) != expected and not args.allow_partial:
        raise RuntimeError(
            f"Expected {expected} completed conditions, found {len(records)}; "
            "pass --allow-partial to summarize an in-progress run"
        )
    summary = summarize(records, manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(summary, manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
