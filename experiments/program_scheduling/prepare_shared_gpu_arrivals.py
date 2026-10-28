"""Create burst or Poisson program-arrival traces for shared-GPU replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity-programs-per-second", type=float)
    parser.add_argument("--burst-output", type=Path)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[2026102701, 2026102702, 2026102703, 2026102704, 2026102705],
    )
    parser.add_argument(
        "--loads", nargs="+", choices=["LOW", "MEDIUM", "HIGH"],
        default=["LOW", "MEDIUM", "HIGH"],
    )
    args = parser.parse_args()
    programs = json.loads(args.source_trace.read_text(encoding="utf-8"))["programs"][:64]
    ids = [program["program_id"] for program in programs]
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError("source trace must provide 64 unique programs")
    if args.burst_output is not None:
        atomic_json(
            args.burst_output,
            {
                "kind": "burst_calibration",
                "seed": 0,
                "lambda_programs_per_second": None,
                "programs": [
                    {"program_id": program_id, "inter_arrival_ms": 0.0,
                     "arrival_time_ms": 0.0}
                    for program_id in ids
                ],
            },
        )
    if args.capacity_programs_per_second is None:
        return
    if args.capacity_programs_per_second <= 0:
        raise ValueError("capacity must be positive")
    loads = {"LOW": 0.25, "MEDIUM": 0.65, "HIGH": 0.95}
    for load in args.loads:
        offered_load = loads[load]
        arrival_rate = args.capacity_programs_per_second * offered_load
        for seed in args.seeds:
            rng = random.Random(seed)
            arrival = 0.0
            rows = []
            for index, program_id in enumerate(ids):
                gap = 0.0 if index == 0 else rng.expovariate(arrival_rate) * 1000.0
                arrival += gap
                rows.append(
                    {"program_id": program_id, "inter_arrival_ms": gap,
                     "arrival_time_ms": arrival}
                )
            atomic_json(
                args.output_dir / f"{load.lower()}_seed{seed}.json",
                {
                    "kind": "poisson_initial_program_arrivals",
                    "load": load,
                    "offered_load": offered_load,
                    "seed": seed,
                    "rng": "python_random.Random.expovariate",
                    "capacity_programs_per_second": args.capacity_programs_per_second,
                    "lambda_programs_per_second": arrival_rate,
                    "programs": rows,
                },
            )


if __name__ == "__main__":
    main()
