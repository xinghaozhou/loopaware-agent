"""Measure fixed-trajectory runtime for adaptive exit and UT-aware scheduling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.analyze_predictor_headroom import (  # noqa: E402
    controlled_predictions,
    read_jsonl,
    selected_ut,
)
from experiments.ordinary_batching_numerics import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    load_requests,
    model_forward,
    set_determinism,
    set_mode,
)
from loopaware_agent.ut_scheduling import (  # noqa: E402
    UTSample,
    form_homogeneous_batches,
    schedule_shortest_ut_first,
)
from vendor.ouro_hf.cache_ouro import (  # noqa: E402
    AdaptiveDecodeCache,
    merge_adaptive_prefill_caches,
)
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


BATCH_SIZE = 4
REQUEST_COUNT = 16
THRESHOLD = 0.2
PREDICTOR_ACCURACY = 0.9
PREDICTOR_SEED = 20261012
MODELED_B4_HEADROOM = 0.2088


@dataclass(frozen=True)
class Workload:
    request_ids: list[str]
    prompts: list[str]
    prompt_lengths: list[int]
    token_ids: list[list[int]]
    true_uts: list[list[int]]
    manifest: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261012)
    return parser.parse_args()


def load_workload(
    trace_dir: Path, tokenizer: Any, positions: int
) -> Workload:
    manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
    request_ids = manifest["request_order"]
    if len(request_ids) != REQUEST_COUNT:
        raise ValueError(f"Expected {REQUEST_COUNT} requests, found {len(request_ids)}")
    if positions < 2 or positions > manifest["max_new_tokens"]:
        raise ValueError("positions must be within [2, trace max_new_tokens]")
    requests = load_requests(tokenizer, count=REQUEST_COUNT // 2)
    requests_by_id = {request.request_id: request for request in requests}
    prompts = []
    token_ids = []
    true_uts = []
    for entry in manifest["requests"]:
        request_id = entry["request_id"]
        prompt = requests_by_id[request_id].prompt
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if digest != entry["prompt_sha256"]:
            raise RuntimeError(f"Prompt hash mismatch for {request_id}")
    for request_id in request_ids:
        prompt = requests_by_id[request_id].prompt
        records = read_jsonl(
            trace_dir / "requests" / f"{request_id.replace(':', '-')}.jsonl"
        )[:positions]
        if len(records) != positions:
            raise ValueError(f"Trace for {request_id} has only {len(records)} positions")
        prompts.append(prompt)
        token_ids.append([int(record["token_id"]) for record in records])
        true_uts.append(
            [
                selected_ut(record["cumulative_exit_probability_by_ut"], THRESHOLD)
                for record in records
            ]
        )
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in prompts
    ]
    return Workload(
        request_ids=request_ids,
        prompts=prompts,
        prompt_lengths=prompt_lengths,
        token_ids=token_ids,
        true_uts=true_uts,
        manifest=manifest,
    )


def predictor_tables(
    physical_true_uts: list[list[int]], *, u_max: int
) -> tuple[list[list[int]], list[list[int]], float]:
    flat_true = [
        physical_true_uts[request][position]
        for position in range(len(physical_true_uts[0]))
        for request in range(REQUEST_COUNT)
    ]
    flat_predicted, realized = controlled_predictions(
        flat_true,
        accuracy=PREDICTOR_ACCURACY,
        u_max=u_max,
        seed=PREDICTOR_SEED,
    )
    width = REQUEST_COUNT
    pred90 = [
        flat_predicted[start : start + width]
        for start in range(0, len(flat_predicted), width)
    ]
    oracle = [
        flat_true[start : start + width]
        for start in range(0, len(flat_true), width)
    ]
    return pred90, oracle, realized


def fcfs_groups() -> list[list[int]]:
    return [
        list(range(start, start + BATCH_SIZE))
        for start in range(0, REQUEST_COUNT, BATCH_SIZE)
    ]


def scheduled_groups(
    true_uts: list[int], predicted_uts: list[int]
) -> list[list[int]]:
    samples = [
        UTSample(str(index), true_uts[index], predicted_uts[index])
        for index in range(REQUEST_COUNT)
    ]
    batches = schedule_shortest_ut_first(
        form_homogeneous_batches(samples, batch_size=BATCH_SIZE, u_max=4)
    )
    return [[int(sample.sample_id) for sample in batch.samples] for batch in batches]


def prepare_inputs(tokenizer: Any, workload: Workload) -> dict[str, torch.Tensor]:
    encoded = tokenizer(workload.prompts, padding=True, return_tensors="pt").to("cuda")
    positions = len(workload.token_ids[0])
    base_mask = encoded["attention_mask"]
    masks = [base_mask]
    for _ in range(1, positions):
        masks.append(
            torch.cat(
                (
                    masks[-1],
                    torch.ones(
                        REQUEST_COUNT,
                        1,
                        dtype=base_mask.dtype,
                        device=base_mask.device,
                    ),
                ),
                dim=1,
            )
        )
    tokens = torch.tensor(workload.token_ids, dtype=torch.long, device="cuda")
    return {
        "input_ids": encoded["input_ids"],
        "base_mask": base_mask,
        "tokens": tokens,
        "masks": masks,
    }


def prefill(
    model: Any,
    inputs: dict[str, Any],
    *,
    merge_for_dynamic_scheduling: bool,
) -> tuple[list[Any] | AdaptiveDecodeCache, float]:
    groups = fcfs_groups()
    torch.cuda.synchronize()
    started = time.perf_counter()
    caches = []
    with torch.inference_mode():
        for group in groups:
            indices = torch.tensor(group, dtype=torch.long, device="cuda")
            output = model_forward(
                model,
                input_ids=inputs["input_ids"].index_select(0, indices),
                attention_mask=inputs["base_mask"].index_select(0, indices),
                cache_position=torch.arange(
                    inputs["input_ids"].shape[1], device="cuda"
                ),
            )
            caches.append(output.past_key_values)
        result = (
            merge_adaptive_prefill_caches(caches)
            if merge_for_dynamic_scheduling
            else caches
        )
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def percentile_nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def run_once(
    model: Any,
    workload: Workload,
    inputs: dict[str, Any],
    *,
    system: str,
    predictions: list[list[int]] | None,
    dynamic_cache: bool = False,
    ground_truth_uts: list[list[int]] | None = None,
) -> dict[str, Any]:
    adaptive = system != "original"
    set_mode(
        model,
        policy="last_available" if adaptive else "full_per_depth",
        physical=adaptive,
        threshold=THRESHOLD if adaptive else None,
    )
    caches, prefill_seconds = prefill(
        model, inputs, merge_for_dynamic_scheduling=dynamic_cache
    )
    prompt_width = inputs["input_ids"].shape[1]
    decode_positions = len(workload.token_ids[0]) - 1
    metadata: list[tuple[int, list[int], torch.Tensor, torch.Tensor]] = []
    completion_events: list[torch.cuda.Event | None] = [None] * REQUEST_COUNT

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    with torch.inference_mode():
        for offset, output_position in enumerate(
            range(1, len(workload.token_ids[0]))
        ):
            if predictions is None:
                groups = fcfs_groups()
            else:
                groups = scheduled_groups(
                    [row[offset] for row in ground_truth_uts],
                    predictions[offset],
                )
            for batch_number, group in enumerate(groups):
                indices = torch.tensor(group, dtype=torch.long, device="cuda")
                if dynamic_cache:
                    if not isinstance(caches, AdaptiveDecodeCache):
                        raise TypeError("Adaptive execution requires a shared cache")
                    caches.set_decode_batch_indices(indices, validate=False)
                    cache = caches
                else:
                    if not isinstance(caches, list):
                        raise TypeError("Fixed grouping requires per-batch caches")
                    cache = caches[batch_number]
                output = model_forward(
                    model,
                    input_ids=inputs["tokens"].index_select(0, indices)[
                        :, output_position - 1 : output_position
                    ],
                    attention_mask=inputs["masks"][output_position].index_select(
                        0, indices
                    ),
                    cache=cache,
                    cache_position=torch.tensor(
                        [prompt_width + output_position - 1], device="cuda"
                    ),
                )
                if adaptive:
                    metadata.append(
                        (
                            offset,
                            group,
                            output.executed_ut_steps.detach(),
                            output.selected_ut_steps.detach(),
                        )
                    )
                if output_position + 1 == len(workload.token_ids[0]):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    for request_index in group:
                        completion_events[request_index] = event
    torch.cuda.synchronize()
    makespan = time.perf_counter() - started
    completion = [
        start_event.elapsed_time(event) / 1000.0
        for event in completion_events
        if event is not None
    ]
    if len(completion) != REQUEST_COUNT:
        raise RuntimeError("Missing request completion event")
    selected_ut_by_request = [
        [0] * decode_positions for _ in range(REQUEST_COUNT)
    ]
    recurrence_steps = 0
    selected_matches = 0
    selected_count = 0
    prediction_matches = 0
    prediction_count = 0
    for offset, group, executed_tensor, selected_tensor in metadata:
        executed = executed_tensor.cpu().numpy()
        selected = selected_tensor.cpu().numpy()
        recurrence_steps += int(executed.sum())
        for local_index, request_index in enumerate(group):
            actual = int(selected[local_index])
            selected_ut_by_request[request_index][offset] = actual
            if ground_truth_uts is not None:
                selected_matches += int(
                    actual == ground_truth_uts[request_index][offset]
                )
                selected_count += 1
            if predictions is not None:
                prediction_matches += int(
                    actual == predictions[offset][request_index]
                )
                prediction_count += 1
    if not adaptive:
        recurrence_steps = (
            REQUEST_COUNT
            * decode_positions
            * workload.manifest["total_ut_steps"]
        )
    decode_tokens = REQUEST_COUNT * decode_positions
    return {
        "prefill_seconds": prefill_seconds,
        "decode_makespan_seconds": makespan,
        "end_to_end_seconds": prefill_seconds + makespan,
        "completion_latency_seconds": completion,
        "mean_completion_latency_seconds": statistics.fmean(completion),
        "median_completion_latency_seconds": statistics.median(completion),
        "p95_completion_latency_seconds": percentile_nearest_rank(completion, 0.95),
        "max_completion_latency_seconds": max(completion),
        "decode_tokens": decode_tokens,
        "tokens_per_second": decode_tokens / makespan,
        "executed_recurrence_steps": recurrence_steps,
        "selected_ut_by_request": selected_ut_by_request if adaptive else None,
        "selected_ut_agreement_with_ground_truth": (
            selected_matches / selected_count if selected_count else None
        ),
        "predictor_accuracy_vs_executed_ut": (
            prediction_matches / prediction_count if prediction_count else None
        ),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "prefill_seconds",
        "decode_makespan_seconds",
        "end_to_end_seconds",
        "mean_completion_latency_seconds",
        "median_completion_latency_seconds",
        "p95_completion_latency_seconds",
        "max_completion_latency_seconds",
        "tokens_per_second",
        "executed_recurrence_steps",
    )
    result: dict[str, Any] = {"runs": runs}
    for key in scalar_keys:
        values = [float(run[key]) for run in runs]
        result[f"mean_{key}"] = statistics.fmean(values)
        result[f"std_{key}"] = statistics.stdev(values) if len(values) > 1 else 0.0
    agreements = [
        run["selected_ut_agreement_with_ground_truth"]
        for run in runs
        if run["selected_ut_agreement_with_ground_truth"] is not None
    ]
    result["mean_selected_ut_agreement_with_ground_truth"] = (
        statistics.fmean(agreements) if agreements else None
    )
    predictor_accuracies = [
        run["predictor_accuracy_vs_executed_ut"]
        for run in runs
        if run["predictor_accuracy_vs_executed_ut"] is not None
    ]
    result["mean_predictor_accuracy_vs_executed_ut"] = (
        statistics.fmean(predictor_accuracies) if predictor_accuracies else None
    )
    return result


def reduction(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


def build_comparisons(systems: dict[str, dict[str, Any]]) -> dict[str, float]:
    original = systems["original"]["mean_decode_makespan_seconds"]
    adaptive = systems["adaptive_fcfs"]["mean_decode_makespan_seconds"]
    dynamic_fcfs = systems["adaptive_dynamic_fcfs"]["mean_decode_makespan_seconds"]
    pred90 = systems["pred90_suf"]["mean_decode_makespan_seconds"]
    oracle = systems["oracle_suf"]["mean_decode_makespan_seconds"]
    oracle_delta = adaptive - oracle
    return {
        "adaptive_exit_speedup": original / adaptive,
        "adaptive_exit_reduction": reduction(original, adaptive),
        "scheduler_speedup_pred90": adaptive / pred90,
        "scheduler_reduction_pred90": reduction(adaptive, pred90),
        "dynamic_cache_reduction_vs_fixed_fcfs": reduction(adaptive, dynamic_fcfs),
        "homogeneous_suf_reduction_vs_dynamic_fcfs": reduction(
            dynamic_fcfs, pred90
        ),
        "oracle_scheduler_reduction": reduction(adaptive, oracle),
        "recovered_oracle_headroom": (
            (adaptive - pred90) / oracle_delta if oracle_delta else float("nan")
        ),
        "realized_fraction_of_modeled_b4_headroom": (
            reduction(adaptive, pred90) / MODELED_B4_HEADROOM
        ),
    }


def percent(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{100 * value:.2f}%"


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Runtime Latency Evaluation of Adaptive Exit and UT-Aware Scheduling",
        "",
        "Primary timing is CUDA-synchronized decode wall-clock time. Prefill is measured separately and excluded from the primary makespan. Tokenizer, dataset, and model loading are excluded.",
        "",
        "## Configuration",
        "",
        f"- Runtime decoder layers: {result['runtime_num_hidden_layers']}",
        f"- Requests: {result['request_count']} (8 GSM8K + 8 MATH-500)",
        f"- Output positions per request: {result['positions']}",
        f"- Timed decode positions per request: {result['decode_positions']}",
        f"- B={BATCH_SIZE}, UT={result['total_ut_steps']}, q={THRESHOLD}, BF16",
        f"- Warmups: {result['warmups']}; measured runs: {result['measured_runs']}",
        f"- Controlled predictor realized calibration-label accuracy: {percent(result['predictor_realized_accuracy'])}",
        "- Runtime UT ground truth: untimed physical Adaptive-FCFS calibration",
        "- Adaptive KV policy: Last-Available",
        "- Original Ouro and required Adaptive FCFS keep four fixed B=4 caches and groups",
        "- An additional shared-cache FCFS control isolates cache architecture; Pred90 and Oracle dynamically regroup before every decode position",
        "- No GPU-to-CPU metadata transfer or synchronization occurs inside the timed decode loop",
        "",
        "## Aggregate results",
        "",
        "| System | Makespan (s) | Mean completion (s) | Median (s) | P95 / max (s) | Tokens/s | Executed recurrence steps | Runtime-UT agreement | Actual predictor accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "original": "Original Ouro",
        "adaptive_fcfs": "Adaptive + FCFS",
        "adaptive_dynamic_fcfs": "Adaptive + dynamic cache + FCFS control",
        "pred90_suf": "Adaptive + 90% UT + homogeneous + SUF",
        "oracle_suf": "Adaptive + oracle UT + homogeneous + SUF",
    }
    for key, label in labels.items():
        row = result["systems"][key]
        agreement = row["mean_selected_ut_agreement_with_ground_truth"]
        predictor_accuracy = row["mean_predictor_accuracy_vs_executed_ut"]
        lines.append(
            f"| {label} | {row['mean_decode_makespan_seconds']:.4f} +/- {row['std_decode_makespan_seconds']:.4f} | "
            f"{row['mean_mean_completion_latency_seconds']:.4f} | "
            f"{row['mean_median_completion_latency_seconds']:.4f} | "
            f"{row['mean_p95_completion_latency_seconds']:.4f} / {row['mean_max_completion_latency_seconds']:.4f} | "
            f"{row['mean_tokens_per_second']:.2f} | {row['mean_executed_recurrence_steps']:.0f} | {percent(agreement)} | {percent(predictor_accuracy)} |"
        )
    comparisons = result["comparisons"]
    lines.extend(
        [
            "",
            "## Decomposition",
            "",
            f"- Physical adaptive exit reduction over Original Ouro: **{percent(comparisons['adaptive_exit_reduction'])}** ({comparisons['adaptive_exit_speedup']:.3f}x).",
            f"- Additional Pred90 scheduler reduction over Adaptive FCFS: **{percent(comparisons['scheduler_reduction_pred90'])}** ({comparisons['scheduler_speedup_pred90']:.3f}x).",
            f"- Dynamic-cache FCFS reduction versus fixed-cache Adaptive FCFS: **{percent(comparisons['dynamic_cache_reduction_vs_fixed_fcfs'])}**.",
            f"- Pred90 homogeneous + SUF reduction versus dynamic-cache FCFS: **{percent(comparisons['homogeneous_suf_reduction_vs_dynamic_fcfs'])}**.",
            f"- Oracle scheduler reduction over Adaptive FCFS: **{percent(comparisons['oracle_scheduler_reduction'])}**.",
            f"- Pred90 recovered Oracle wall-clock opportunity: **{percent(comparisons['recovered_oracle_headroom'])}**.",
            f"- Realized scheduler reduction / prior 20.88% modeled B=4 headroom: **{percent(comparisons['realized_fraction_of_modeled_b4_headroom'])}**.",
            "",
            "## Raw runs",
            "",
        ]
    )
    for key, label in labels.items():
        values = [
            f"{run['decode_makespan_seconds']:.6f}"
            for run in result["systems"][key]["runs"]
        ]
        lines.append(f"- {label}: {', '.join(values)} seconds")
    lines.extend(
        [
            "",
            "## Workload",
            "",
            "| Request | Dataset | Sample ID | Prompt tokens | Output positions |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for request in result["requests"]:
        dataset, sample_id = request["request_id"].split(":", 1)
        lines.append(
            f"| {request['request_id']} | {dataset} | {sample_id} | {request['prompt_tokens']} | {request['output_positions']} |"
        )
    lines.extend(
        [
            "",
            "The fixed reference token IDs and ground-truth UT traces are preserved in the raw JSON result and source trace. Predicted UT affects only batch formation and execution order; physical model exits remain threshold-selected.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise ValueError("runs must be positive and warmups non-negative")
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

    calibration = run_once(
        model,
        workload,
        inputs,
        system="adaptive_fcfs",
        predictions=None,
        dynamic_cache=False,
    )
    physical_true_uts = calibration["selected_ut_by_request"]
    pred90, oracle, predictor_accuracy = predictor_tables(
        physical_true_uts, u_max=workload.manifest["total_ut_steps"]
    )
    print(
        "CALIBRATION adaptive_fcfs "
        f"decode={calibration['decode_makespan_seconds']:.4f}s "
        f"steps={calibration['executed_recurrence_steps']}",
        flush=True,
    )

    policies = {
        "original": (None, False),
        "adaptive_fcfs": (None, False),
        "adaptive_dynamic_fcfs": (None, True),
        "pred90_suf": (pred90, True),
        "oracle_suf": (oracle, True),
    }
    measured: dict[str, list[dict[str, Any]]] = {key: [] for key in policies}
    for phase, repetitions in (("warmup", args.warmups), ("measured", args.runs)):
        for repetition in range(repetitions):
            for system, (predictions, dynamic_cache) in policies.items():
                row = run_once(
                    model,
                    workload,
                    inputs,
                    system=system,
                    predictions=predictions,
                    dynamic_cache=dynamic_cache,
                    ground_truth_uts=physical_true_uts if system != "original" else None,
                )
                if phase == "measured":
                    measured[system].append(row)
                print(
                    f"{phase.upper()} {repetition + 1}/{repetitions} {system} "
                    f"decode={row['decode_makespan_seconds']:.4f}s "
                    f"steps={row['executed_recurrence_steps']}",
                    flush=True,
                )
                torch.cuda.empty_cache()

    systems = {key: aggregate(rows) for key, rows in measured.items()}
    result = {
        "experiment": "runtime_adaptive_ut_scheduling",
        "model": args.model,
        "model_revision": args.model_revision,
        "runtime_num_hidden_layers": runtime_layers,
        "dtype": str(next(model.parameters()).dtype),
        "gpu": torch.cuda.get_device_name(0),
        "request_count": REQUEST_COUNT,
        "batch_size": BATCH_SIZE,
        "positions": args.positions,
        "decode_positions": args.positions - 1,
        "threshold": THRESHOLD,
        "total_ut_steps": workload.manifest["total_ut_steps"],
        "warmups": args.warmups,
        "measured_runs": args.runs,
        "predictor_seed": PREDICTOR_SEED,
        "predictor_target_accuracy": PREDICTOR_ACCURACY,
        "predictor_realized_accuracy": predictor_accuracy,
        "runtime_ground_truth": "untimed physical adaptive-exit + Last-Available + FCFS calibration",
        "calibration": calibration,
        "prefill_in_primary_timing": False,
        "systems": systems,
        "comparisons": build_comparisons(systems),
        "requests": [
            {
                "request_id": workload.request_ids[index],
                "prompt_tokens": workload.prompt_lengths[index],
                "output_positions": len(workload.token_ids[index]),
                "reference_token_ids": workload.token_ids[index],
                "ground_truth_ut": workload.true_uts[index],
                "physical_fcfs_ground_truth_ut": physical_true_uts[index],
            }
            for index in range(REQUEST_COUNT)
        ],
        "trace_manifest": workload.manifest,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(result), encoding="utf-8")
    print(markdown(result), flush=True)


if __name__ == "__main__":
    main()
