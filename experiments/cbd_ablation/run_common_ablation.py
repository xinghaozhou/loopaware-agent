"""Run the 16-request scheduler-only B=4 ablation on one CDB substrate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from looped_cdb.benchmarks.runner import EngineArgs, RunConfig, ServingSession, measure_once
from looped_cdb.benchmarks.workload import Workload


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pred90_groups(path: Path, positions: int) -> tuple[list[list[list[int]]], float]:
    checkpoint = json.loads(path.read_text())
    calibration = checkpoint["calibration"]
    predictions = calibration["predictions"]
    if len(predictions) != positions or any(len(row) != 16 for row in predictions):
        raise ValueError(f"expected predictions shaped [{positions}, 16]")
    groups: list[list[list[int]]] = []
    for row in predictions:
        ordered = sorted(range(16), key=lambda index: int(row[index]))
        formed = [ordered[start : start + 4] for start in range(0, 16, 4)]
        formed.sort(key=lambda group: max(int(row[index]) for index in group))
        groups.append(formed)
    return groups, float(calibration["realized_accuracy"])


def engine_args(*, refill: bool, frontier_policy: str | None, groups: list[list[list[int]]] | None) -> EngineArgs:
    return EngineArgs(
        backend="cdb",
        block_size=256,
        num_blocks=64,
        max_num_batched_tokens=8192,
        max_model_len=2048,
        max_recurrent_depth=4,
        max_num_seqs=16,
        decode_batch_size=4,
        min_free_slots=None,
        min_recurrent_steps=1,
        no_delay_gate_consumption=True,
        kv_policy="single",
        kv_slots_per_layer=None,
        exit_threshold=0.2,
        sync=True,
        refill=refill,
        min_coda_batch_size=1,
        safety_margin=0.0,
        kv_pressure_mode="none",
        cpu_offload_space=None,
        replay_eos_finishes=False,
        mem_fraction_static=None,
        frontier_policy=frontier_policy,
        frontier_groups=groups,
    )


def main() -> None:
    cli = args()
    workload = Workload.load(cli.workload)
    if workload.num_requests != 16 or set(workload.output_lens.tolist()) != {128}:
        raise ValueError("common ablation requires 16 requests with 128 output tokens each")
    groups, predictor_accuracy = pred90_groups(cli.predictions, positions=127)
    run_config = RunConfig(
        model=str(cli.model),
        attn_implementation="paged|flash_attention_2",
        seed=20261012,
        min_exit_step=1,
        exit_delay_steps=0,
        cuda_graph_mode="none",
    )
    policies = [
        ("fixed_fcfs", False, "fcfs", None),
        ("pred90_suf", False, "pred90_suf", groups),
        ("cdb_refill", True, None, None),
    ]
    rows = []
    session = ServingSession(run_config, max_recurrent_depth=4, kv_policy="single")
    runtime_layers = int(session.model.config.num_hidden_layers)
    print(f"runtime config.num_hidden_layers={runtime_layers}", flush=True)
    for name, refill, frontier_policy, policy_groups in policies:
        print(f"running policy={name}", flush=True)
        eargs = engine_args(refill=refill, frontier_policy=frontier_policy, groups=policy_groups)
        with session.prepared(workload, eargs, run_config) as prepared:
            measured = measure_once(prepared)
            stats = prepared.engine.last_stats
            output_lengths = [len(output.generated_tokens) for output in measured.outputs]
            row = {
                    "policy": name,
                    "decode_wall_time_s": stats.decode_wall_time_s,
                    "end_to_end_wall_time_s": measured.wall_time_s,
                    "requests": len(output_lengths),
                    "generated_tokens": sum(output_lengths),
                    "output_lengths": output_lengths,
                    "recurrent_steps": stats.recurrent_steps,
                    "recurrent_batches": stats.recurrent_batches,
                    "coda_tokens": stats.coda_tokens,
                    "coda_batches": stats.coda_batches,
                    "mean_recurrent_batch_size": stats.mean_recurrent_batch_size,
                    "mean_coda_batch_size": stats.mean_coda_batch_size,
                    "scheduler_refills": stats.scheduler_refills,
                    "max_depth_queue_size": stats.max_depth_queue_size,
                    "runtime_num_hidden_layers": runtime_layers,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    baseline = float(rows[0]["decode_wall_time_s"])
    for row in rows:
        row["speedup_vs_fixed_fcfs"] = baseline / float(row["decode_wall_time_s"])
    result = {
        "experiment": "common_substrate_scheduler_ablation_b4",
        "model": str(cli.model),
        "runtime_num_hidden_layers": rows[0]["runtime_num_hidden_layers"],
        "workload": str(cli.workload),
        "selected_ut_threshold": 0.2,
        "predictor_realized_accuracy": predictor_accuracy,
        "requests": 16,
        "output_tokens_per_request": 128,
        "timed_decode_tokens_per_request": 127,
        "resident_requests": 16,
        "decode_launch_width": 4,
        "cuda_graphs": False,
        "async_batching": False,
        "kv_policy": "single",
        "attention": "paged|flash_attention_2",
        "timing_boundary": "first decode prelude through final CUDA completion; excludes prompt prefill and output materialization",
        "rows": rows,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {cli.output}", flush=True)


if __name__ == "__main__":
    main()
