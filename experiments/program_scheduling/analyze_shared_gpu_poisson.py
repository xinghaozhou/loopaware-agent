"""Aggregate the physical shared-GPU Poisson scheduling matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


LOADS = ("low", "medium", "high")
POLICIES = (
    "FCFS", "AUTELLIX_PLAS", "UT_LAS", "ORACLE_TOKEN_SUF",
    "ORACLE_UT_SUF", "ORACLE_RUNTIME_SJF",
)
METRICS = (
    "mean_ttlt_ms", "p95_ttlt_ms", "mean_queueing_time_ms", "makespan_ms",
    "throughput_programs_per_s", "physical_gpu_utilization",
    "time_weighted_batch_fill", "mean_ready_queue_length",
)


def mean_std(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot aggregate an empty sample")
    return {
        "mean": fmean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=2)
    args = parser.parse_args()
    runs = []
    for path in sorted(args.experiment_root.glob("*/*/*/results.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        verification = value["shared_gpu_verification"]
        if not verification["verified"] or verification["maximum_overlapping_gpu_batches"] != 1:
            raise RuntimeError(f"shared-GPU verification failed: {path}")
        runs.append(
            {
                "load": path.parts[-4].upper(),
                "seed": value["configuration"]["arrival_seed"],
                "policy": value["configuration"]["policy"],
                "summary": value["summary"],
                "path": str(path),
            }
        )
    discovered_loads = sorted({run["load"] for run in runs})
    expected_runs = len(discovered_loads) * len(POLICIES) * args.expected_seeds
    if len(runs) != expected_runs:
        raise RuntimeError(f"expected {expected_runs} completed runs, found {len(runs)}")
    aggregates = []
    for load in discovered_loads:
        for policy in POLICIES:
            selected = [
                run["summary"] for run in runs
                if run["load"] == load and run["policy"] == policy
            ]
            if len(selected) != args.expected_seeds:
                raise RuntimeError(
                    f"expected {args.expected_seeds} runs for {load}/{policy}"
                )
            aggregates.append(
                {"load": load, "policy": policy, **{
                    metric: mean_std([float(row[metric]) for row in selected])
                    for metric in METRICS
                }}
            )
    comparisons = []
    for load in discovered_loads:
        per_seed = []
        for seed in sorted({run["seed"] for run in runs if run["load"] == load}):
            ttlts = {
                run["policy"]: float(run["summary"]["mean_ttlt_ms"])
                for run in runs if run["load"] == load and run["seed"] == seed
            }
            fcfs, plas = ttlts["FCFS"], ttlts["AUTELLIX_PLAS"]
            ut_las = ttlts["UT_LAS"]
            token = ttlts["ORACLE_TOKEN_SUF"]
            ut, runtime = ttlts["ORACLE_UT_SUF"], ttlts["ORACLE_RUNTIME_SJF"]
            per_seed.append(
                {
                    "autellix_improvement_over_fcfs": (fcfs - plas) / fcfs,
                    "ut_las_improvement_over_autellix": (plas - ut_las) / plas,
                    "token_improvement_over_fcfs": (fcfs - token) / fcfs,
                    "ut_improvement_over_token": (token - ut) / token,
                    "ut_runtime_headroom_recovery_from_fcfs": (
                        (fcfs - ut) / (fcfs - runtime) if fcfs > runtime else None
                    ),
                    "ut_runtime_headroom_recovery_from_autellix": (
                        (plas - ut) / (plas - runtime) if plas > runtime else None
                    ),
                }
            )
        comparisons.append(
            {"load": load, **{
                key: (
                    mean_std(values)
                    if (values := [float(row[key]) for row in per_seed if row[key] is not None])
                    else None
                )
                for key in per_seed[0]
            }}
        )
    payload: dict[str, Any] = {
        "configuration": {
            "programs": 64, "batch_size": 4,
            "arrival_seeds": args.expected_seeds,
            "execution_model": "one_shared_static_gpu_batch",
            "teacher_forced": True, "physical_ut": 4,
        },
        "aggregates": aggregates,
        "comparisons": comparisons,
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Physical Shared-GPU BFCL Poisson Scheduling", "",
        f"All values are mean +/- sample standard deviation across {args.expected_seeds} arrival seeds.", "",
        "| Load | Policy | Mean TTLT (s) | P95 TTLT (s) | Mean queue (s) | Makespan (s) | Programs/s | GPU util | Batch fill | Mean ready queue |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        def fmt(name: str, scale: float = 1.0) -> str:
            value = row[name]
            return f"{value['mean'] * scale:.3f} +/- {value['std'] * scale:.3f}"
        lines.append(
            f"| {row['load']} | {row['policy']} | {fmt('mean_ttlt_ms', .001)} | "
            f"{fmt('p95_ttlt_ms', .001)} | {fmt('mean_queueing_time_ms', .001)} | "
            f"{fmt('makespan_ms', .001)} | {fmt('throughput_programs_per_s')} | "
            f"{fmt('physical_gpu_utilization', 100)}% | {fmt('time_weighted_batch_fill', 100)}% | "
            f"{fmt('mean_ready_queue_length')} |"
        )
    lines.extend(["", "The JSON artifact contains paired-seed policy comparisons and every raw run path.", ""])
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
