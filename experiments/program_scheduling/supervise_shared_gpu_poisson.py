"""Continue the physical shared-GPU Poisson matrix after capacity calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


POLICIES = (
    "FCFS",
    "AUTELLIX_PLAS",
    "UT_LAS",
    "ORACLE_TOKEN_SUF",
    "ORACLE_UT_SUF",
    "ORACLE_RUNTIME_SJF",
)
LOADS = ("medium",)
SEEDS = (2026102701, 2026102702)


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results" / "program_scheduling"
    calibration = results / "shared_gpu_burst_calibration" / "results.json"
    while not calibration.exists():
        print("WAIT calibration", flush=True)
        time.sleep(30)
    calibrated = json.loads(calibration.read_text(encoding="utf-8"))
    capacity = float(calibrated["summary"]["throughput_programs_per_s"])
    print(f"CALIBRATED capacity_programs_per_second={capacity}", flush=True)

    arrival_dir = results / "shared_gpu_arrivals" / "poisson"
    run(
        [
            sys.executable,
            "-m",
            "experiments.program_scheduling.prepare_shared_gpu_arrivals",
            "--source-trace",
            str(results / "bfcl64_seed20261026" / "trace.json"),
            "--output-dir",
            str(arrival_dir),
            "--capacity-programs-per-second",
            str(capacity),
            "--loads",
            "MEDIUM",
            "--seeds",
            *(str(seed) for seed in SEEDS),
        ]
    )
    experiment_root = results / "shared_gpu_poisson_b4"
    measurement_cache = results / "shared_gpu_measurements"
    for load in LOADS:
        for seed in SEEDS:
            arrival = arrival_dir / f"{load}_seed{seed}.json"
            for policy in POLICIES:
                output = experiment_root / load / f"seed{seed}" / policy.lower()
                if (output / "results.json").exists():
                    print(f"SKIP complete {load} {seed} {policy}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "experiments.program_scheduling.run_shared_gpu_poisson",
                    "--source-trace",
                    str(results / "bfcl64_seed20261026" / "trace.json"),
                    "--arrival-trace",
                    str(arrival),
                    "--output-dir",
                    str(output),
                    "--measurement-cache-dir",
                    str(measurement_cache),
                    "--batch-size",
                    "4",
                    "--policy",
                    policy,
                    "--programs",
                    "64",
                ]
                if (output / "batches").exists():
                    command.append("--resume")
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        run(command)
                        break
                    except subprocess.CalledProcessError:
                        if attempts >= 3:
                            raise
                        print(
                            f"RETRY {load} {seed} {policy} attempt={attempts + 1}",
                            flush=True,
                        )
                        if "--resume" not in command:
                            command.append("--resume")
                        time.sleep(10)
    run(
        [
            sys.executable,
            "-m",
            "experiments.program_scheduling.analyze_shared_gpu_poisson",
            "--experiment-root",
            str(experiment_root),
            "--output-json",
            str(experiment_root / "summary.json"),
            "--output-md",
            str(root / "experiments" / "program_scheduling" / "RESULTS_BFCL_SHARED_GPU_POISSON.md"),
        ]
    )
    print("MATRIX COMPLETE", flush=True)


if __name__ == "__main__":
    main()
