"""Profile Original Ouro versus the current Adaptive Pred90 KV runtime."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.ordinary_batching_numerics import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    set_determinism,
)
from experiments.runtime_adaptive_ut_scheduling import (  # noqa: E402
    BATCH_SIZE,
    PREDICTOR_ACCURACY,
    REQUEST_COUNT,
    THRESHOLD,
    fcfs_groups,
    load_workload,
    predictor_tables,
    prepare_inputs,
    run_once,
    scheduled_groups,
    set_profile_ranges as set_benchmark_profile_ranges,
)
from vendor.ouro_hf.cache_ouro import (  # noqa: E402
    set_profile_ranges as set_cache_profile_ranges,
)
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


KV_RANGES = ("kv_prepare",)
CUSTOM_RANGES = (
    "recurrent_core",
    *KV_RANGES,
    "batch_regroup",
    "physical_compaction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timing-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261012)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def event_device_us(event: Any, *, self_time: bool = False) -> float:
    attribute = "self_device_time_total" if self_time else "device_time_total"
    return float(getattr(event, attribute, 0.0) or 0.0)


def event_cpu_us(event: Any, *, self_time: bool = False) -> float:
    attribute = "self_cpu_time_total" if self_time else "cpu_time_total"
    return float(getattr(event, attribute, 0.0) or 0.0)


def union_duration_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def profiler_summary(profiler: Any, wall_seconds: float) -> dict[str, Any]:
    averages = list(profiler.key_averages())
    by_key = {event.key: event for event in averages}
    ranges = {}
    for name in CUSTOM_RANGES:
        event = by_key.get(name)
        ranges[name] = {
            "count": int(event.count) if event else 0,
            "cpu_total_seconds": event_cpu_us(event) / 1e6 if event else 0.0,
            "gpu_total_seconds": event_device_us(event) / 1e6 if event else 0.0,
        }

    cuda_api = []
    for event in averages:
        key_lower = event.key.lower()
        if "cuda" in key_lower and event_cpu_us(event, self_time=True) > 0:
            cuda_api.append(
                {
                    "name": event.key,
                    "count": int(event.count),
                    "self_cpu_seconds": event_cpu_us(event, self_time=True) / 1e6,
                }
            )
    cuda_api.sort(key=lambda row: row["self_cpu_seconds"], reverse=True)

    operators = []
    for event in averages:
        if event.key in CUSTOM_RANGES or event.key == "ProfilerStep*":
            continue
        device_us = event_device_us(event, self_time=True)
        if device_us <= 0:
            continue
        operators.append(
            {
                "name": event.key,
                "count": int(event.count),
                "self_gpu_seconds": device_us / 1e6,
                "self_cpu_seconds": event_cpu_us(event, self_time=True) / 1e6,
            }
        )
    operators.sort(key=lambda row: row["self_gpu_seconds"], reverse=True)
    gpu_active_seconds = sum(
        event_device_us(event, self_time=True) for event in averages
    ) / 1e6
    launch_count = next(
        (int(event.count) for event in averages if event.key == "cudaLaunchKernel"),
        0,
    )

    kv_gpu = sum(ranges[name]["gpu_total_seconds"] for name in KV_RANGES)
    kv_cpu = sum(ranges[name]["cpu_total_seconds"] for name in KV_RANGES)
    recurrent_gpu = ranges["recurrent_core"]["gpu_total_seconds"]
    compaction_gpu = ranges["physical_compaction"]["gpu_total_seconds"]
    return {
        "ranges": ranges,
        "gpu_active_seconds": gpu_active_seconds,
        "gpu_busy_ratio": gpu_active_seconds / wall_seconds,
        "visible_gpu_idle_seconds": max(0.0, wall_seconds - gpu_active_seconds),
        "cuda_activity_event_count": launch_count,
        "kernel_launch_count": launch_count,
        "kv_gpu_seconds": kv_gpu,
        "kv_cpu_range_seconds": kv_cpu,
        "kv_overhead_ratio": kv_gpu / wall_seconds,
        "recurrent_core_gpu_seconds_inclusive": recurrent_gpu,
        "recurrent_compute_gpu_seconds_estimate": max(
            0.0, recurrent_gpu - kv_gpu - compaction_gpu
        ),
        "recurrent_compute_ratio_estimate": max(
            0.0, recurrent_gpu - kv_gpu - compaction_gpu
        )
        / wall_seconds,
        "top_gpu_kernels": [
            {
                "name": row["name"],
                "gpu_seconds": row["self_gpu_seconds"],
                "count": row["count"],
            }
            for row in operators[:10]
        ],
        "top_gpu_operators": operators[:15],
        "top_cuda_api_calls": cuda_api[:10],
    }


class ChunkCollector:
    """Retain aggregate tables only; never export a Kineto/Chrome trace."""

    def __init__(self, output_dir: Path, system: str):
        self.output_dir = output_dir
        self.system = system
        self.summaries: list[dict[str, Any]] = []

    def __call__(self, profiler: Any) -> None:
        self.summaries.append(profiler_summary(profiler, 1.0))


def aggregate_profile_chunks(
    chunks: list[dict[str, Any]], reference_wall_seconds: float, scale: float
) -> dict[str, Any]:
    if not chunks:
        raise RuntimeError("Profiler produced no completed chunks")
    ranges = {
        name: {
            "count": round(
                scale * sum(chunk["ranges"][name]["count"] for chunk in chunks)
            ),
            "cpu_total_seconds": scale * sum(
                chunk["ranges"][name]["cpu_total_seconds"] for chunk in chunks
            ),
            "gpu_total_seconds": scale * sum(
                chunk["ranges"][name]["gpu_total_seconds"] for chunk in chunks
            ),
        }
        for name in CUSTOM_RANGES
    }

    def aggregate_rows(key: str, time_key: str) -> list[dict[str, Any]]:
        totals: dict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0, time_key: 0.0}
        )
        for chunk in chunks:
            for row in chunk[key]:
                totals[row["name"]]["count"] += row["count"]
                totals[row["name"]][time_key] += row[time_key]
        rows = [
            {
                "name": name,
                "count": round(scale * values["count"]),
                time_key: scale * values[time_key],
            }
            for name, values in totals.items()
        ]
        rows.sort(key=lambda row: row[time_key], reverse=True)
        return rows

    gpu_active = scale * sum(chunk["gpu_active_seconds"] for chunk in chunks)
    kv_gpu = sum(ranges[name]["gpu_total_seconds"] for name in KV_RANGES)
    kv_cpu = sum(ranges[name]["cpu_total_seconds"] for name in KV_RANGES)
    compaction_gpu = ranges["physical_compaction"]["gpu_total_seconds"]
    recurrent_gpu = ranges["recurrent_core"]["gpu_total_seconds"]
    recurrent_compute = max(0.0, recurrent_gpu - kv_gpu - compaction_gpu)
    recurrent_cpu = max(
        0.0,
        ranges["recurrent_core"]["cpu_total_seconds"]
        - kv_cpu
        - ranges["physical_compaction"]["cpu_total_seconds"],
    )
    coarse_regions = {
        "recurrent_core": {
            "cpu_seconds": recurrent_cpu,
            "cuda_seconds": recurrent_compute,
        },
        "kv_prepare": {
            "cpu_seconds": kv_cpu,
            "cuda_seconds": kv_gpu,
        },
        "batch_regroup": {
            "cpu_seconds": ranges["batch_regroup"]["cpu_total_seconds"],
            "cuda_seconds": ranges["batch_regroup"]["gpu_total_seconds"],
        },
        "physical_compaction": {
            "cpu_seconds": ranges["physical_compaction"]["cpu_total_seconds"],
            "cuda_seconds": ranges["physical_compaction"]["gpu_total_seconds"],
        },
    }
    profiled_cuda_total = sum(
        region["cuda_seconds"] for region in coarse_regions.values()
    )
    for region in coarse_regions.values():
        region["profiled_cuda_seconds"] = region["cuda_seconds"]
        cuda_share = (
            region["profiled_cuda_seconds"] / profiled_cuda_total
            if profiled_cuda_total
            else 0.0
        )
        # CUDA activity profiling materially slows this eager workload. Preserve
        # raw profiler seconds above, but project the measured stage share onto
        # the separate clean wall-clock replay for an interpretable breakdown.
        region["cuda_seconds"] = reference_wall_seconds * cuda_share
        region["cpu_percent_of_wall"] = (
            100.0 * region["cpu_seconds"] / reference_wall_seconds
        )
        region["cuda_percent_of_wall"] = 100.0 * cuda_share
    return {
        "profile_chunks": len(chunks),
        "ranges": ranges,
        "gpu_active_seconds": gpu_active,
        "gpu_busy_ratio": gpu_active / reference_wall_seconds,
        "visible_gpu_idle_seconds": max(0.0, reference_wall_seconds - gpu_active),
        "cuda_activity_event_count": round(scale * sum(
            chunk["cuda_activity_event_count"] for chunk in chunks
        )),
        "kernel_launch_count": round(
            scale * sum(chunk["kernel_launch_count"] for chunk in chunks)
        ),
        "kv_gpu_seconds": kv_gpu,
        "kv_cpu_range_seconds": kv_cpu,
        "kv_overhead_ratio": kv_gpu / reference_wall_seconds,
        "recurrent_core_gpu_seconds_inclusive": recurrent_gpu,
        "recurrent_compute_gpu_seconds_estimate": recurrent_compute,
        "recurrent_compute_ratio_estimate": recurrent_compute
        / reference_wall_seconds,
        "coarse_regions": coarse_regions,
        "profiled_coarse_cuda_seconds": profiled_cuda_total,
        "sampling_scale": scale,
        "top_gpu_kernels": aggregate_rows("top_gpu_kernels", "gpu_seconds")[:10],
        "top_gpu_operators": aggregate_rows(
            "top_gpu_operators", "self_gpu_seconds"
        )[:15],
        "top_cuda_api_calls": aggregate_rows(
            "top_cuda_api_calls", "self_cpu_seconds"
        )[:10],
    }


def scheduling_metrics(
    selected_by_request: list[list[int]], predictions: list[list[int]], u_max: int
) -> dict[str, Any]:
    del u_max
    positions = len(selected_by_request[0])
    previous: tuple[tuple[int, ...], ...] | None = None
    membership_changes = 0
    active_by_depth: dict[int, list[int]] = defaultdict(list)
    batch_depth_executions = 0
    for offset in range(positions):
        groups = scheduled_groups(
            [selected_by_request[row][offset] for row in range(REQUEST_COUNT)],
            predictions[offset],
        )
        signature = tuple(tuple(group) for group in groups)
        if previous is not None and signature != previous:
            membership_changes += 1
        previous = signature
        for group in groups:
            depths = [selected_by_request[row][offset] for row in group]
            for depth in range(1, max(depths) + 1):
                active_by_depth[depth].append(sum(value >= depth for value in depths))
                batch_depth_executions += 1
    return {
        "regroup_decisions": positions,
        "actual_grouping_changes": membership_changes,
        "microbatch_executions": positions * (REQUEST_COUNT // BATCH_SIZE),
        "batch_depth_executions": batch_depth_executions,
        "average_active_batch_size_by_depth": {
            str(depth): statistics.fmean(values)
            for depth, values in sorted(active_by_depth.items())
        },
    }


def format_seconds(value: float) -> str:
    return f"{value:.3f}"


def markdown(result: dict[str, Any]) -> str:
    original = result["systems"]["original"]
    adaptive = result["systems"]["pred90_suf"]
    rows = (
        ("Wall-clock decode time", original["timing"]["mean"], adaptive["timing"]["mean"]),
        ("Executed row-recurrence steps", original["executed_recurrence_steps"], adaptive["executed_recurrence_steps"]),
        ("Recurrent compute wall-equivalent CUDA time", original["profile"]["coarse_regions"]["recurrent_core"]["cuda_seconds"], adaptive["profile"]["coarse_regions"]["recurrent_core"]["cuda_seconds"]),
        ("KV management wall-equivalent CUDA time", original["profile"]["coarse_regions"]["kv_prepare"]["cuda_seconds"], adaptive["profile"]["coarse_regions"]["kv_prepare"]["cuda_seconds"]),
        ("Physical compaction wall-equivalent CUDA time", original["profile"]["coarse_regions"]["physical_compaction"]["cuda_seconds"], adaptive["profile"]["coarse_regions"]["physical_compaction"]["cuda_seconds"]),
        ("Batch regroup CPU time", original["profile"]["ranges"]["batch_regroup"]["cpu_total_seconds"], adaptive["profile"]["ranges"]["batch_regroup"]["cpu_total_seconds"]),
    )
    lines = [
        "# Original Ouro vs Adaptive Pred90: KV Runtime Profile",
        "",
        "## Configuration",
        "",
        f"- Model: `{result['model']}` at `{result['model_revision']}`",
        f"- GPU: {result['gpu']}; dtype: {result['dtype']}",
        f"- Runtime decoder layers: {result['runtime_num_hidden_layers']}",
        f"- Requests: {REQUEST_COUNT}; B={BATCH_SIZE}; positions={result['positions']} ({result['decode_positions']} timed)",
        f"- Adaptive: physical exit, Last-Available KV, q={THRESHOLD}, controlled Pred90, homogeneous batching, SUF",
        "- Wall-clock timing is unprofiled. Coarse attribution is extrapolated from eight evenly spaced profiled decode positions over a separate complete replay.",
        "- The profiler records CPU and CUDA activity only: no trace export, stacks, shapes, or memory profile.",
        "- Recurrent compute is the inclusive `recurrent_core` range minus its nested `kv_prepare` and `physical_compaction` ranges.",
        "- Because CUDA activity profiling slows this eager workload, raw profiler CUDA seconds are retained in JSON and measured CUDA shares are projected onto the clean wall time below.",
        "",
        "## Primary comparison",
        "",
        "| Component | Original Ouro | Adaptive Pred90 |",
        "|---|---:|---:|",
    ]
    ratio_names: set[str] = set()
    integer_names = {"Executed row-recurrence steps"}
    for name, left, right in rows:
        if name in ratio_names:
            left_text, right_text = f"{100 * left:.2f}%", f"{100 * right:.2f}%"
        elif name in integer_names:
            left_text, right_text = f"{left:.0f}", f"{right:.0f}"
        else:
            left_text, right_text = format_seconds(left), format_seconds(right)
        lines.append(f"| {name} | {left_text} | {right_text} |")

    for system_key, label in (
        ("original", "Original Ouro coarse regions"),
        ("pred90_suf", "Adaptive Pred90 coarse regions"),
    ):
        system = result["systems"][system_key]
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"Unprofiled wall-clock decode time: **{system['timing']['mean']:.3f} s**.",
                "",
                "| Region | CPU time (s) | CPU % wall | CUDA time (s) | CUDA % wall |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for region_name in CUSTOM_RANGES:
            region = system["profile"]["coarse_regions"][region_name]
            lines.append(
                f"| `{region_name}` | {region['cpu_seconds']:.3f} | "
                f"{region['cpu_percent_of_wall']:.2f}% | "
                f"{region['cuda_seconds']:.3f} | "
                f"{region['cuda_percent_of_wall']:.2f}% |"
            )
        lines.extend(
            [
                "",
                "Top CUDA operations/kernels by cumulative self CUDA time:",
                "",
                "| Operation/kernel | Calls | CUDA time (s) |",
                "|---|---:|---:|",
            ]
        )
        for operation in system["profile"]["top_gpu_operators"][:10]:
            lines.append(
                f"| `{operation['name']}` | {operation['count']} | "
                f"{operation['self_gpu_seconds']:.3f} |"
            )

    profile = adaptive["profile"]
    scheduling = adaptive["scheduling"]
    lines.extend(
        [
            "",
            "## Adaptive KV attribution",
            "",
            f"- KV CUDA share / clean wall time: **{profile['coarse_regions']['kv_prepare']['cuda_percent_of_wall']:.2f}%**.",
            f"- Recurrent-compute CUDA share / clean wall time: **{profile['coarse_regions']['recurrent_core']['cuda_percent_of_wall']:.2f}%**.",
            f"- Regroup decisions: {scheduling['regroup_decisions']}; actual grouping changes: {scheduling['actual_grouping_changes']}.",
            f"- Microbatch executions: {scheduling['microbatch_executions']}; batch-depth executions: {scheduling['batch_depth_executions']}.",
            f"- Average active B by depth: `{json.dumps(scheduling['average_active_batch_size_by_depth'], sort_keys=True)}`.",
            "",
            "### KV ranges",
            "",
            "| Range | Calls | CPU inclusive (s) | GPU inclusive (s) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in KV_RANGES:
        row = profile["ranges"][name]
        lines.append(
            f"| `{name}` | {row['count']} | {row['cpu_total_seconds']:.3f} | {row['gpu_total_seconds']:.3f} |"
        )
    lines.extend(["", "### Top GPU kernels", "", "| Kernel | Calls | GPU time (s) |", "|---|---:|---:|"])
    for row in profile["top_gpu_kernels"][:5]:
        lines.append(f"| `{row['name']}` | {row['count']} | {row['gpu_seconds']:.3f} |")
    lines.extend(["", "### Top CUDA API calls", "", "| API | Calls | Self CPU time (s) |", "|---|---:|---:|"])
    for row in profile["top_cuda_api_calls"][:5]:
        lines.append(f"| `{row['name']}` | {row['count']} | {row['self_cpu_seconds']:.3f} |")

    kv_percent = profile["coarse_regions"]["kv_prepare"]["cuda_percent_of_wall"]
    kv_substantial = kv_percent >= 10.0
    scheduler_ratio = profile["ranges"]["batch_regroup"]["cpu_total_seconds"] / adaptive["timing"]["mean"]
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"1. KV/cache reconstruction is {'a substantial' if kv_substantial else 'not a dominant'} measured fraction of Adaptive Pred90 runtime ({kv_percent:.2f}% by projected CUDA stage share).",
            f"2. Batch-regroup Python work accounts for approximately {100 * scheduler_ratio:.2f}% of unprofiled wall time.",
            "3. Remaining latency is attributable to model compute, active-row execution shape, launch overhead, and visible GPU gaps as itemized above.",
            "4. The next optimization should target the largest measured adaptive KV range; no redesign was performed in this experiment.",
            f"5. A KV redesign is {'justified for the next optimization study' if kv_substantial else 'not yet justified solely by this profile'} under the 10% materiality criterion used here.",
            "",
            "Profiler totals are stratified estimates. `recurrent_core` contains KV and compaction; the reported recurrent-compute estimate subtracts those nested ranges to avoid double counting.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.warmups < 1 or args.timing_runs < 1:
        raise ValueError("warmups and timing-runs must both be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(args.seed)
    model_source = args.model_path or args.model
    tokenizer_source = args.tokenizer_path or args.model
    revision = None if args.model_path else args.model_revision
    tokenizer_revision = None if args.tokenizer_path else args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=tokenizer_revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    workload = load_workload(args.trace_dir.resolve(), tokenizer, args.positions)
    inputs = prepare_inputs(tokenizer, workload)

    config = OuroConfig.from_pretrained(model_source, revision=revision)
    config.total_ut_steps = workload.manifest["total_ut_steps"]
    config.kv_cache_policy = "full_per_depth"
    config.physical_early_exit = False
    config.early_exit_threshold = None
    model = OuroForCausalLM.from_pretrained(
        model_source,
        revision=revision,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    runtime_layers = model.config.num_hidden_layers
    if not isinstance(runtime_layers, int) or runtime_layers <= 0:
        raise ValueError(f"Invalid runtime num_hidden_layers={runtime_layers!r}")
    print(f"runtime_num_hidden_layers={runtime_layers}", flush=True)

    # Match the previous Pred90 experiment exactly: calibrate labels with the
    # physical adaptive path, because full-compute counterfactual labels can
    # change the homogeneous batch composition.
    calibration = run_once(
        model,
        workload,
        inputs,
        system="adaptive_fcfs",
        predictions=None,
        dynamic_cache=False,
    )
    physical_true_uts = calibration["selected_ut_by_request"]
    predictions, _, realized_accuracy = predictor_tables(
        physical_true_uts, u_max=workload.manifest["total_ut_steps"]
    )
    systems = {
        "original": {"predictions": None, "dynamic_cache": False},
        "pred90_suf": {"predictions": predictions, "dynamic_cache": True},
    }

    result_systems = {}
    decode_positions = args.positions - 1
    sample_positions = sorted(
        {round(index * (decode_positions - 1) / 7) for index in range(8)}
    )

    def sample_schedule(step: int) -> torch.profiler.ProfilerAction:
        if step in sample_positions:
            return torch.profiler.ProfilerAction.RECORD_AND_SAVE
        return torch.profiler.ProfilerAction.NONE

    # Measure both systems before activating CUPTI. Profiling one system must
    # not perturb the clean wall-clock timing of the system that follows it.
    timing_by_system = {}
    for name, policy in systems.items():
        set_cache_profile_ranges(False)
        set_benchmark_profile_ranges(False)
        for warmup in range(args.warmups):
            warm = run_once(
                model,
                workload,
                inputs,
                system=name,
                predictions=policy["predictions"],
                dynamic_cache=policy["dynamic_cache"],
                ground_truth_uts=physical_true_uts if name != "original" else None,
            )
            print(f"WARMUP {name} {warmup + 1}/{args.warmups} {warm['decode_makespan_seconds']:.4f}s", flush=True)
        timed = []
        last_timing = None
        for repetition in range(args.timing_runs):
            last_timing = run_once(
                model,
                workload,
                inputs,
                system=name,
                predictions=policy["predictions"],
                dynamic_cache=policy["dynamic_cache"],
                ground_truth_uts=physical_true_uts if name != "original" else None,
            )
            timed.append(last_timing["decode_makespan_seconds"])
            print(f"TIMING {name} {repetition + 1}/{args.timing_runs} {timed[-1]:.4f}s", flush=True)
        timing_by_system[name] = {
            "summary": mean_std(timed),
            "last": last_timing,
        }

    for name, policy in systems.items():
        timed_summary = timing_by_system[name]["summary"]
        set_cache_profile_ranges(True)
        set_benchmark_profile_ranges(True)
        collector = ChunkCollector(args.output_dir, name)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            schedule=sample_schedule,
            on_trace_ready=collector,
        )
        profiled = run_once(
            model,
            workload,
            inputs,
            system=name,
            predictions=policy["predictions"],
            dynamic_cache=policy["dynamic_cache"],
            ground_truth_uts=physical_true_uts if name != "original" else None,
            profile_context=profiler,
            profile_step=True,
        )
        profile = aggregate_profile_chunks(
            collector.summaries,
            timed_summary["mean"],
            decode_positions / len(sample_positions),
        )
        result_systems[name] = {
            "timing": timed_summary,
            "profile_wall_seconds": profiled["decode_makespan_seconds"],
            "executed_recurrence_steps": profiled["executed_recurrence_steps"],
            "profile": profile,
            "profile_sample_positions": sample_positions,
        }
        if name == "pred90_suf":
            result_systems[name]["scheduling"] = scheduling_metrics(
                profiled["selected_ut_by_request"],
                predictions,
                workload.manifest["total_ut_steps"],
            )
        print(
            f"PROFILE {name} wall={profiled['decode_makespan_seconds']:.4f}s "
            f"gpu_active={profile['gpu_active_seconds']:.4f}s "
            f"kv_gpu={profile['kv_gpu_seconds']:.4f}s",
            flush=True,
        )
        del profiler, collector
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "experiment": "profile_original_vs_adaptive_pred90_kv",
        "model": args.model,
        "model_revision": args.model_revision,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(next(model.parameters()).dtype),
        "runtime_num_hidden_layers": runtime_layers,
        "request_count": REQUEST_COUNT,
        "batch_size": BATCH_SIZE,
        "positions": args.positions,
        "decode_positions": args.positions - 1,
        "threshold": THRESHOLD,
        "predictor_target_accuracy": PREDICTOR_ACCURACY,
        "predictor_realized_accuracy": realized_accuracy,
        "warmups": args.warmups,
        "timing_runs": args.timing_runs,
        "profiler": "torch.profiler (Nsight Systems unavailable)",
        "systems": result_systems,
        "trace_manifest": workload.manifest,
    }
    result_path = args.output_dir / "results.json"
    report_path = args.output_dir / "report.md"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(markdown(result), encoding="utf-8")
    print(markdown(result), flush=True)


if __name__ == "__main__":
    main()
