"""Aggregate a completed fixed-trajectory replay and write its result report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / denominator if denominator else None


def token_rows(run_dir: Path, condition: str, task_id: str, turn_idx: int) -> list[dict[str, Any]]:
    return read_jsonl(
        run_dir / "raw" / "requests" / condition / task_id / f"turn-{turn_idx:04d}.jsonl"
    )


def batch_metrics(profiles: list[list[int]]) -> dict[str, float]:
    horizon = len(profiles[0])
    variances, ranges = [], []
    useful = 0
    padded = 0
    for token_idx in range(horizon):
        depths = [profile[token_idx] for profile in profiles]
        variances.append(statistics.pvariance(depths))
        ranges.append(max(depths) - min(depths))
        useful += sum(depths)
        padded += len(depths) * max(depths)
    return {
        "mean_within_batch_variance": statistics.fmean(variances),
        "mean_within_batch_range": statistics.fmean(ranges),
        "recurrence_utilization": useful / padded if padded else 1.0,
    }


def mean_metric(rows: list[dict[str, float]], name: str) -> float | None:
    return statistics.fmean(row[name] for row in rows) if rows else None


def simulate_batches(
    run_dir: Path,
    requests: list[dict[str, Any]],
    task_order: list[str],
) -> list[dict[str, Any]]:
    request_map = {(row["task_id"], row["turn_idx"]): row for row in requests}
    maximum_turn = max((row["turn_idx"] for row in requests), default=-1)
    results = []
    for batch_size in (4, 8, 16):
        for horizon in (4, 8, 16, 32):
            fcfs_rows, oracle_rows = [], []
            for turn_idx in range(maximum_turn + 1):
                candidates = []
                for task_id in task_order:
                    summary = request_map.get((task_id, turn_idx))
                    if summary is None or summary["output_token_count"] < horizon:
                        continue
                    rows = token_rows(run_dir, summary["condition"], task_id, turn_idx)
                    candidates.append((task_id, [row["selected_ut_step"] for row in rows[:horizon]]))
                if len(candidates) < batch_size:
                    continue
                fcfs_profiles = [profile for _, profile in candidates[:batch_size]]
                fcfs_rows.append(batch_metrics(fcfs_profiles))
                choices = combinations(candidates, batch_size)
                if len(candidates) > 12:
                    choices = [tuple(candidates[:batch_size])]
                oracle_profiles = min(
                    ([profile for _, profile in choice] for choice in choices),
                    key=lambda profiles: batch_metrics(profiles)["mean_within_batch_variance"],
                )
                oracle_rows.append(batch_metrics(oracle_profiles))
            result = {"batch_size": batch_size, "horizon": horizon, "num_synchronized_batches": len(fcfs_rows)}
            for label, rows in (("fcfs", fcfs_rows), ("oracle", oracle_rows)):
                for metric in ("mean_within_batch_variance", "mean_within_batch_range", "recurrence_utilization"):
                    result[f"{label}_{metric}"] = mean_metric(rows, metric)
            results.append(result)
    return results


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metadata = read_json(run_dir / "experiment_config.json")
    completion = read_json(run_dir / "completion.json")
    requests = read_jsonl(run_dir / "request_summaries.jsonl")
    config = metadata["source_config"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        grouped[request["condition"]].append(request)

    aggregates = []
    task_rows = []
    all_batch_results = []
    for condition_config in config["conditions"]:
        condition = condition_config["name"]
        values = grouped.get(condition, [])
        histogram: Counter[int] = Counter()
        for value in values:
            histogram.update({int(key): count for key, count in value["selected_ut_histogram"].items()})
        token_count = sum(histogram.values())
        selected_total = sum(step * count for step, count in histogram.items())
        executed_total = sum(value["executed_ut_total"] for value in values)
        max_turn_by_task: dict[str, int] = defaultdict(int)
        for value in values:
            max_turn_by_task[value["task_id"]] = max(
                max_turn_by_task[value["task_id"]], value["turn_idx"]
            )
        normalized_progress = [
            value["turn_idx"] / max_turn_by_task[value["task_id"]]
            if max_turn_by_task[value["task_id"]]
            else 0.0
            for value in values
        ]
        aggregate = {
            "condition": condition,
            "total_ut_steps": condition_config["total_ut_steps"],
            "num_tasks": len({value["task_id"] for value in values}),
            "num_requests": len(values),
            "num_output_tokens": token_count,
            "runtime_num_hidden_layers": sorted({value["num_decoder_layers"] for value in values}),
            "selected_ut_histogram": dict(sorted(histogram.items())),
            "selected_ut_percent": {str(step): count / token_count * 100 for step, count in sorted(histogram.items())} if token_count else {},
            "mean_selected_ut": selected_total / token_count if token_count else None,
            "theoretical_recurrence_saving": 1 - selected_total / executed_total if executed_total else None,
            "elapsed_seconds": sum(value["elapsed_seconds"] for value in values),
            "peak_allocated_gib": max((value["peak_allocated_gib"] for value in values), default=None),
            "peak_reserved_gib": max((value["peak_reserved_gib"] for value in values), default=None),
            "correlation_input_tokens_mean_selected_ut": pearson(
                [value["input_token_count"] for value in values],
                [value["mean_selected_ut"] for value in values],
            ),
            "correlation_turn_index_mean_selected_ut": pearson(
                [value["turn_idx"] for value in values],
                [value["mean_selected_ut"] for value in values],
            ),
            "correlation_normalized_progress_mean_selected_ut": pearson(
                normalized_progress,
                [value["mean_selected_ut"] for value in values],
            ),
        }
        aggregates.append(aggregate)
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in values:
            by_task[value["task_id"]].append(value)
        for task_id, rows in sorted(by_task.items()):
            rows.sort(key=lambda row: row["turn_idx"])
            tokens = sum(row["output_token_count"] for row in rows)
            selected = sum(row["selected_ut_total"] for row in rows)
            executed = sum(row["executed_ut_total"] for row in rows)
            task_rows.append({
                "condition": condition,
                "task_id": task_id,
                "requests": len(rows),
                "tokens": tokens,
                "mean_selected_ut": selected / tokens if tokens else None,
                "theoretical_recurrence_saving": 1 - selected / executed if executed else None,
                "input_tokens_first": rows[0]["input_token_count"],
                "input_tokens_last": rows[-1]["input_token_count"],
                "mean_selected_ut_first": rows[0]["mean_selected_ut"],
                "mean_selected_ut_last": rows[-1]["mean_selected_ut"],
                "correlation_input_tokens_mean_selected_ut": pearson(
                    [row["input_token_count"] for row in rows],
                    [row["mean_selected_ut"] for row in rows],
                ),
                "correlation_turn_index_mean_selected_ut": pearson(
                    [row["turn_idx"] for row in rows],
                    [row["mean_selected_ut"] for row in rows],
                ),
            })
        if condition == "ut6_primary":
            batch_rows = simulate_batches(run_dir, values, condition_config["trajectory_ids"])
            for row in batch_rows:
                row["condition"] = condition
            all_batch_results.extend(batch_rows)

    results = {
        "completion": completion,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "dataset_repo": config["dataset_repo"],
        "dataset_revision": config["dataset_revision"],
        "trace_exit_threshold": config["trace_exit_threshold"],
        "condition_aggregates": aggregates,
        "task_aggregates": task_rows,
        "batch_simulation": all_batch_results,
    }
    atomic_json(run_dir / "aggregate_results.json", results)
    atomic_json(run_dir / "batch_heterogeneity.json", all_batch_results)

    lines = [
        "# Fixed-Trajectory Selected-UT Results",
        "",
        f"Run directory: `{run_dir}`  ",
        f"Model: `{config['model_id']}` at `{config['model_revision']}`  ",
        f"Dataset: `{config['dataset_repo']}` / `{config['dataset_model_subset']}` at `{config['dataset_revision']}`  ",
        f"Trace threshold: `{config['trace_exit_threshold']}`  ",
        f"Completion: `{completion['completed_requests']}/{completion['planned_requests']}` requests.",
        "",
        "The model always decoded at the condition's fixed final UT. `selected UT` is the counterfactual earliest UT whose cumulative exit probability crossed the threshold.",
        "",
        "## Overall distribution",
        "",
        "| Condition | Tasks | Requests | Tokens | Runtime layers | Mean selected UT | Theoretical recurrence saving | GPU time (h) | Peak allocated GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        layers = ",".join(str(value) for value in row["runtime_num_hidden_layers"])
        lines.append(
            f"| {row['condition']} | {row['num_tasks']} | {row['num_requests']} | {row['num_output_tokens']} | {layers} | {fmt(row['mean_selected_ut'])} | {fmt(row['theoretical_recurrence_saving'] * 100 if row['theoretical_recurrence_saving'] is not None else None, 1)}% | {fmt(row['elapsed_seconds'] / 3600, 2)} | {fmt(row['peak_allocated_gib'], 1)} |"
        )
        lines.extend(["", f"### {row['condition']} selected-UT histogram", "", "| Selected UT | Tokens | Percent |", "|---:|---:|---:|"])
        for step, count in row["selected_ut_histogram"].items():
            lines.append(f"| {step} | {count} | {row['selected_ut_percent'][str(step)]:.2f}% |")
        lines.extend([
            "",
            f"Pearson correlation of request mean selected UT with input length: `{fmt(row['correlation_input_tokens_mean_selected_ut'])}`; with raw turn index: `{fmt(row['correlation_turn_index_mean_selected_ut'])}`; with normalized trajectory progress: `{fmt(row['correlation_normalized_progress_mean_selected_ut'])}`.",
        ])

    lines.extend(["", "## Per-trajectory context-growth results", "", "Each row covers one complete recorded trajectory. The two correlations test whether selected UT changes as its own context and turn index grow.", "", "| Condition | Task | Turns | Input first→last | Mean UT first→last | Mean UT | Corr(input, UT) | Corr(turn, UT) |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in task_rows:
        lines.append(f"| {row['condition']} | {row['task_id']} | {row['requests']} | {row['input_tokens_first']}→{row['input_tokens_last']} | {fmt(row['mean_selected_ut_first'])}→{fmt(row['mean_selected_ut_last'])} | {fmt(row['mean_selected_ut'])} | {fmt(row['correlation_input_tokens_mean_selected_ut'])} | {fmt(row['correlation_turn_index_mean_selected_ut'])} |")

    lines.extend(["", "## Full turn-by-turn distributions", "", "This is the direct view of selected-UT variation as each trajectory's recorded context grows.", "", "| Condition | Task | Turn | Progress | Input tokens | Output tokens | Mean UT | P50 | P90 | Selected-UT histogram |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for condition_config in config["conditions"]:
        condition = condition_config["name"]
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for request in grouped.get(condition, []):
            by_task[request["task_id"]].append(request)
        for task_id in condition_config["trajectory_ids"]:
            rows = sorted(by_task.get(task_id, []), key=lambda row: row["turn_idx"])
            final_turn = rows[-1]["turn_idx"] if rows else 0
            for row in rows:
                progress = row["turn_idx"] / final_turn if final_turn else 0.0
                histogram_text = ", ".join(
                    f"UT{step}:{count}"
                    for step, count in sorted(
                        ((int(step), count) for step, count in row["selected_ut_histogram"].items())
                    )
                )
                lines.append(
                    f"| {condition} | {task_id} | {row['turn_idx']} | {progress:.2f} | {row['input_token_count']} | {row['output_token_count']} | {fmt(row['mean_selected_ut'])} | {fmt(row['p50_selected_ut'], 1)} | {fmt(row['p90_selected_ut'], 1)} | {histogram_text} |"
                )

    lines.extend(["", "## Synchronized batch heterogeneity", "", "The pilot has six primary trajectories, so only B=4 forms cross-task batches; B=8 and B=16 are reported as unavailable rather than reusing the same task concurrently.", "", "| B | H | Batches | FCFS variance | Oracle variance | FCFS utilization | Oracle utilization |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in all_batch_results:
        lines.append(f"| {row['batch_size']} | {row['horizon']} | {row['num_synchronized_batches']} | {fmt(row['fcfs_mean_within_batch_variance'])} | {fmt(row['oracle_mean_within_batch_variance'])} | {fmt(row['fcfs_recurrence_utilization'])} | {fmt(row['oracle_recurrence_utilization'])} |")
    lines.extend(["", "## Scope and limitations", "", "This is teacher-forced workload replay over resolved public trajectories; Ouro's generated text does not alter later recorded contexts. Long (26+ turn) complete trajectories were excluded because the smallest reached 15,881 input tokens and UT=6 exhausted the unchanged A40 at 16K. No context was silently truncated.", ""])
    (run_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(run_dir / "RESULTS.md")


if __name__ == "__main__":
    main()
