"""Replay fixed BFCL traces through all program-level scheduling policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import (
    pairwise_order_accuracy,
    pearson,
    ready_pairwise_order_accuracy,
    spearman,
    summarize_simulation,
)
from .policies import POLICIES
from .simulator import simulate
from .trace_schema import validate_programs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--program-limit",
        type=int,
        default=None,
        help="Evaluate a nested prefix of the fixed trace.",
    )
    args = parser.parse_args()
    bundle = json.loads(args.trace.read_text())
    programs = bundle["programs"][: args.program_limit]
    if args.program_limit is not None and len(programs) != args.program_limit:
        raise ValueError(
            f"trace contains {len(bundle['programs'])} programs, cannot select "
            f"{args.program_limit}"
        )
    validate_programs(programs)
    simulations = [simulate(programs, policy) for policy in POLICIES]
    summaries = [summarize_simulation(result) for result in simulations]
    by_policy = {row["policy"]: row for row in summaries}
    plas = by_policy["GPU_PLAS_PROXY"]["mean_ttlt_ms"]
    ut_las = by_policy["UT_LAS"]["mean_ttlt_ms"]
    ut_oracle = by_policy["ORACLE_UT_SUF"]["mean_ttlt_ms"]
    sjf = by_policy["ORACLE_RUNTIME_SJF"]["mean_ttlt_ms"]
    denominator = plas - sjf
    turns = [turn for program in programs for turn in program["turns"]]
    uts = [float(turn["sum_selected_ut"]) for turn in turns]
    tokens = [float(turn["output_token_count"]) for turn in turns]
    runtimes = [float(turn["full_compute_gpu_time_ms"]) for turn in turns]
    result = {
        "configuration": {
            **bundle["configuration"],
            "evaluated_program_count": len(programs),
        },
        "program_ids": [program["program_id"] for program in programs],
        "summaries": summaries,
        "headroom": {
            "oracle_sjf_over_plas_fraction": (plas - sjf) / plas,
            "oracle_ut_improvement_over_plas_fraction": (plas - ut_oracle) / plas,
            "sjf_headroom_recovered_by_ut_fraction": (
                (plas - ut_oracle) / denominator if denominator > 0 else None
            ),
            "ut_las_improvement_over_gpu_plas_fraction": (plas - ut_las) / plas,
        },
        "characterization": {
            "turns": len(turns),
            "pearson_selected_ut_runtime": pearson(uts, runtimes),
            "spearman_selected_ut_runtime": spearman(uts, runtimes),
            "pearson_output_tokens_runtime": pearson(tokens, runtimes),
            "spearman_output_tokens_runtime": spearman(tokens, runtimes),
            "pairwise_selected_ut_runtime_accuracy": pairwise_order_accuracy(uts, runtimes),
            "pairwise_output_tokens_runtime_accuracy": pairwise_order_accuracy(tokens, runtimes),
            "ready_pairwise_selected_ut_runtime_accuracy_under_gpu_plas": (
                ready_pairwise_order_accuracy(
                    next(result for result in simulations if result["policy"] == "GPU_PLAS_PROXY"),
                    "sum_selected_ut",
                )
            ),
            "ready_pairwise_output_tokens_runtime_accuracy_under_gpu_plas": (
                ready_pairwise_order_accuracy(
                    next(result for result in simulations if result["policy"] == "GPU_PLAS_PROXY"),
                    "output_token_count",
                )
            ),
        },
        "simulations": simulations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {key: value for key, value in result.items() if key != "simulations"}
    args.output.with_name("summary.json").write_text(
        json.dumps(compact, indent=2) + "\n"
    )
    print(json.dumps({"summaries": summaries, **result["headroom"], **result["characterization"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
