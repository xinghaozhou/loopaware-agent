"""Analyze token-level selected-UT heterogeneity for math and web trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any


THRESHOLDS = (0.6, 0.7, 0.8, 0.9)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / denominator if denominator else None


def selected_step(row: dict[str, Any], threshold: float) -> int:
    cumulative = row["cumulative_exit_probability_by_ut"]
    return next(
        (index + 1 for index, value in enumerate(cumulative) if value >= threshold),
        len(cumulative),
    )


def trace_path(run_dir: Path, summary: dict[str, Any]) -> Path:
    return (
        run_dir
        / "raw"
        / "requests"
        / summary["condition"]
        / summary["task_id"]
        / f"turn-{summary['turn_idx']:04d}.jsonl"
    )


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metadata = read_json(run_dir / "experiment_config.json")
    completion = read_json(run_dir / "completion.json")
    config = metadata["source_config"]
    summaries = read_jsonl(run_dir / "request_summaries.jsonl")
    summary_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    traces: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        summary_by_condition[summary["condition"]].append(summary)
        traces[summary["request_key"]] = read_jsonl(trace_path(run_dir, summary))

    condition_results = []
    trajectory_results = []
    for condition_config in config["conditions"]:
        condition = condition_config["name"]
        domain = condition_config["dataset"]
        requests = summary_by_condition[condition]
        threshold_results = []
        for threshold in THRESHOLDS:
            histogram: Counter[int] = Counter()
            request_means = []
            unique_depths = []
            adjacent_changes = 0
            adjacent_pairs = 0
            expected_values = []
            for request in requests:
                rows = traces[request["request_key"]]
                selected = [selected_step(row, threshold) for row in rows]
                histogram.update(selected)
                request_means.append(statistics.fmean(selected))
                unique_depths.append(len(set(selected)))
                adjacent_changes += sum(a != b for a, b in zip(selected, selected[1:]))
                adjacent_pairs += max(0, len(selected) - 1)
                expected_values.extend(row["expected_ut_steps"] for row in rows)
            total = sum(histogram.values())
            by_task: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for request, mean in zip(requests, request_means, strict=True):
                by_task[request["task_id"]].append((request["turn_idx"], mean))
            first = [sorted(values)[0][1] for values in by_task.values()]
            last = [sorted(values)[-1][1] for values in by_task.values()]
            threshold_results.append(
                {
                    "threshold": threshold,
                    "num_tokens": total,
                    "histogram": dict(sorted(histogram.items())),
                    "percent": {
                        str(step): count / total * 100
                        for step, count in sorted(histogram.items())
                    },
                    "mean_selected_ut": sum(step * count for step, count in histogram.items()) / total,
                    "saving_vs_fixed_ut": 1.0 - sum(step * count for step, count in histogram.items()) / total / condition_config["total_ut_steps"],
                    "correlation_input_tokens_mean_selected_ut": pearson(
                        [request["input_token_count"] for request in requests], request_means
                    ),
                    "correlation_turn_mean_selected_ut": pearson(
                        [request["turn_idx"] for request in requests], request_means
                    ),
                    "mean_first_turn": statistics.fmean(first),
                    "mean_final_turn": statistics.fmean(last),
                    "multi_depth_request_fraction": sum(value > 1 for value in unique_depths) / len(unique_depths),
                    "adjacent_token_transition_rate": adjacent_changes / adjacent_pairs,
                    "mean_probabilistic_expected_ut": statistics.fmean(expected_values),
                }
            )

        by_task_summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for request in requests:
            by_task_summaries[request["task_id"]].append(request)
        for task_id, task_requests in sorted(by_task_summaries.items()):
            task_requests.sort(key=lambda row: row["turn_idx"])
            trajectory_results.append(
                {
                    "domain": domain,
                    "condition": condition,
                    "task_id": task_id,
                    "turns": len(task_requests),
                    "input_first": task_requests[0]["input_token_count"],
                    "input_final": task_requests[-1]["input_token_count"],
                    "mean_ut_first_at_0.7": task_requests[0]["mean_selected_ut"],
                    "mean_ut_final_at_0.7": task_requests[-1]["mean_selected_ut"],
                    "correlation_input_mean_ut_at_0.7": pearson(
                        [row["input_token_count"] for row in task_requests],
                        [row["mean_selected_ut"] for row in task_requests],
                    ),
                }
            )
        condition_results.append(
            {
                "condition": condition,
                "domain": domain,
                "num_trajectories": len(by_task_summaries),
                "num_requests": len(requests),
                "runtime_num_hidden_layers": sorted(
                    {request["num_decoder_layers"] for request in requests}
                ),
                "elapsed_seconds": sum(request["elapsed_seconds"] for request in requests),
                "peak_allocated_gib": max(request["peak_allocated_gib"] for request in requests),
                "thresholds": threshold_results,
            }
        )

    results = {
        "completion": completion,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "datasets": config["datasets"],
        "conditions": condition_results,
        "trajectories": trajectory_results,
    }
    (run_dir / "aggregate_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Cross-Domain Token-Level Selected-UT Results",
        "",
        f"Completion: `{completion['completed_requests']}/{completion['planned_requests']}` requests with `{completion['errors_this_process']}` errors.  ",
        f"Model: `{config['model_id']}` at `{config['model_revision']}`.  ",
        "All decoding used fixed UT=6; threshold-selected UT is counterfactual.",
        "",
        "## Threshold sweep",
        "",
        "| Domain | Trajectories | Requests | Tokens | Threshold | Mean selected UT | Saving | Corr(input, UT) | First→final UT | Multi-depth turns | Adjacent transition rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in condition_results:
        for threshold in condition["thresholds"]:
            lines.append(
                f"| {condition['domain']} | {condition['num_trajectories']} | {condition['num_requests']} | {threshold['num_tokens']} | {threshold['threshold']:.1f} | {threshold['mean_selected_ut']:.3f} | {threshold['saving_vs_fixed_ut'] * 100:.1f}% | {fmt(threshold['correlation_input_tokens_mean_selected_ut'])} | {threshold['mean_first_turn']:.3f}→{threshold['mean_final_turn']:.3f} | {threshold['multi_depth_request_fraction'] * 100:.1f}% | {threshold['adjacent_token_transition_rate'] * 100:.1f}% |"
            )
            distribution = ", ".join(
                f"UT{step} {threshold['percent'][str(step)]:.2f}%"
                for step in threshold["histogram"]
            )
            lines.append(f"|  |  |  |  |  | Distribution | {distribution} |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Complete trajectories at threshold 0.7",
            "",
            "| Domain | Trajectory | Turns | Input first→final | Mean UT first→final | Corr(input, UT) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in trajectory_results:
        lines.append(
            f"| {row['domain']} | {row['task_id']} | {row['turns']} | {row['input_first']}→{row['input_final']} | {row['mean_ut_first_at_0.7']:.3f}→{row['mean_ut_final_at_0.7']:.3f} | {fmt(row['correlation_input_mean_ut_at_0.7'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This experiment characterizes selected-UT heterogeneity only. It does not evaluate batching or realize adaptive-compute speedups. Context correlations are descriptive because trajectory progress, observations, and response content change together.",
            "",
        ]
    )
    (run_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(run_dir / "RESULTS.md")


if __name__ == "__main__":
    main()
