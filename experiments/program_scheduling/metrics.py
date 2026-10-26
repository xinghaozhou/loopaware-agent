"""Summary and characterization metrics for scheduling traces."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Iterable


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_simulation(result: dict) -> dict:
    rows = result["programs"]
    completion = [row["completion_time_ms"] for row in rows]
    queue = [row["queue_time_ms"] for row in rows]
    makespan_ms = max(completion)
    return {
        "policy": result["policy"],
        "mean_ttlt_ms": mean(completion),
        "median_ttlt_ms": median(completion),
        "p95_ttlt_ms": percentile(completion, 0.95),
        "mean_queue_delay_ms": mean(queue),
        "p95_queue_delay_ms": percentile(queue, 0.95),
        "throughput_programs_per_s": len(rows) / (makespan_ms / 1000.0),
        "mean_slowdown": mean(row["slowdown"] for row in rows),
        "mean_execution_ms": mean(row["execution_time_ms"] for row in rows),
        "mean_tool_wait_ms": mean(row["tool_wait_time_ms"] for row in rows),
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def _ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in ordered[index:end]:
            result[position] = rank
        index = end
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(_ranks(xs), _ranks(ys))


def pairwise_order_accuracy(proxy: list[float], runtime: list[float]) -> float:
    score = comparable = 0.0
    for left in range(len(proxy)):
        for right in range(left + 1, len(proxy)):
            if runtime[left] == runtime[right]:
                continue
            comparable += 1
            if proxy[left] == proxy[right]:
                score += 0.5
            else:
                score += (proxy[left] < proxy[right]) == (runtime[left] < runtime[right])
    return score / comparable if comparable else float("nan")


def ready_pairwise_order_accuracy(simulation: dict, proxy_field: str) -> float:
    """Score all pairs that are simultaneously READY at each decision point."""

    score = comparable = 0.0
    for decision in simulation["schedule"]:
        rows = decision["ready_candidates"]
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                runtime_left = rows[left]["full_compute_gpu_time_ms"]
                runtime_right = rows[right]["full_compute_gpu_time_ms"]
                if runtime_left == runtime_right:
                    continue
                comparable += 1
                proxy_left = rows[left][proxy_field]
                proxy_right = rows[right][proxy_field]
                if proxy_left == proxy_right:
                    score += 0.5
                else:
                    score += (proxy_left < proxy_right) == (runtime_left < runtime_right)
    return score / comparable if comparable else float("nan")
