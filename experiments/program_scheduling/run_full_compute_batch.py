"""Run resumable, real full-compute batched BFCL program replay."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.program_scheduling.bfcl_workload import (  # noqa: E402
    build_contexts,
    select_base_programs,
)
from experiments.program_scheduling.metrics import summarize_simulation  # noqa: E402
from experiments.program_scheduling.policies import priority  # noqa: E402
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--policy", default="FCFS")
    parser.add_argument("--programs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--reuse-measurements-from",
        type=Path,
        nargs="*",
        default=[],
        help="Reuse checkpoints only when the ordered batch members match exactly.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def generated_lengths(
    generated: torch.Tensor, prompt_width: int, eos_token_ids: set[int]
) -> list[int]:
    lengths = []
    for row in generated[:, prompt_width:].detach().cpu().tolist():
        length = len(row)
        for index, token_id in enumerate(row):
            if token_id in eos_token_ids:
                length = index + 1
                break
        lengths.append(length)
    return lengths


def load_model_source(args: argparse.Namespace) -> tuple[str | Path, dict[str, str]]:
    if args.model_path is not None:
        return args.model_path, {}
    return MODEL_ID, {"revision": MODEL_REVISION}


def initial_states(programs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **program,
            "next_turn": 0,
            "ready_time_ms": float(program.get("arrival_time_ms", 0.0)),
            "attained_gpu_ms": 0.0,
            "attained_ut": 0,
            "queue_time_ms": 0.0,
            "execution_time_ms": 0.0,
            "completion_time_ms": None,
        }
        for program in programs
    ]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    bundle = json.loads(args.source_trace.read_text(encoding="utf-8"))
    programs = bundle["programs"][: args.programs]
    if len(programs) != args.programs:
        raise ValueError("source trace does not contain the requested programs")

    rows = select_base_programs(args.programs, args.seed)
    expected_ids = [program["program_id"] for program in programs]
    if [row["id"] for row in rows] != expected_ids:
        raise RuntimeError("BFCL selection does not match the source trace")
    contexts = {
        (row["id"], context["turn_id"]): context
        for row in rows
        for context in build_contexts(row)
    }

    source, load_kwargs = load_model_source(args)
    tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = OuroConfig.from_pretrained(source, **load_kwargs)
    config.total_ut_steps = 4
    model = OuroForCausalLM.from_pretrained(
        source,
        **load_kwargs,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    model.early_exit_step = None
    model.early_exit_threshold = None
    runtime_layers = int(model.config.num_hidden_layers)
    if runtime_layers <= 0:
        raise RuntimeError("invalid runtime decoder-layer count")

    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_token_ids = {eos_ids}
    else:
        eos_token_ids = {int(value) for value in eos_ids}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "batches"
    existing = sorted(checkpoint_dir.glob("batch-*.json"))
    if existing and not args.resume:
        raise RuntimeError("batch checkpoints exist; pass --resume to continue")

    reusable: dict[str, tuple[dict[str, Any], str]] = {}
    for reuse_dir in args.reuse_measurements_from:
        for path in sorted((reuse_dir / "batches").glob("batch-*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("batch_size") != args.batch_size:
                continue
            signature = json.dumps(value["members"], sort_keys=True)
            reusable.setdefault(signature, (value, str(path)))

    warmup = tokenizer("warm up", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model.generate(**warmup, max_new_tokens=2, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    del warmup

    states = initial_states(programs)
    clock = min(state["ready_time_ms"] for state in states)
    schedule = []
    batch_index = 0
    while any(state["completion_time_ms"] is None for state in states):
        ready = [
            state
            for state in states
            if state["completion_time_ms"] is None
            and state["ready_time_ms"] <= clock
        ]
        if not ready:
            clock = min(
                state["ready_time_ms"]
                for state in states
                if state["completion_time_ms"] is None
            )
            continue
        chosen = sorted(
            ready, key=lambda candidate: priority(args.policy, candidate)
        )[: args.batch_size]
        members = [
            {"program_id": state["program_id"], "turn_id": state["next_turn"]}
            for state in chosen
        ]
        checkpoint = checkpoint_dir / f"batch-{batch_index:04d}.json"
        if checkpoint.exists():
            if not args.resume:
                raise RuntimeError(f"checkpoint unexpectedly exists: {checkpoint}")
            measurement = json.loads(checkpoint.read_text(encoding="utf-8"))
            if measurement["members"] != members:
                raise RuntimeError(f"resume schedule mismatch at batch {batch_index}")
        else:
            signature = json.dumps(members, sort_keys=True)
            cached = reusable.get(signature)
            if cached is not None:
                measurement = {
                    **cached[0],
                    "batch_index": batch_index,
                    "reused_measurement_from": cached[1],
                }
                atomic_json(checkpoint, measurement)
            else:
                prompts = [
                contexts[(member["program_id"], member["turn_id"])]["messages"]
                for member in members
                ]
                unpadded = [
                    len(
                        tokenizer.apply_chat_template(
                            prompt, tokenize=True, add_generation_prompt=True
                        )
                    )
                    for prompt in prompts
                ]
                encoded = tokenizer.apply_chat_template(
                    prompts,
                    tokenize=True,
                    add_generation_prompt=True,
                    padding=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(model.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                wall_started = time.perf_counter()
                gpu_start.record()
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gpu_end.record()
                torch.cuda.synchronize()
                gpu_ms = float(gpu_start.elapsed_time(gpu_end))
                measurement = {
                    "batch_index": batch_index,
                    "members": members,
                    "batch_size": len(members),
                    "unpadded_input_tokens": unpadded,
                    "padded_input_tokens": int(encoded.input_ids.shape[-1]),
                    "generated_tokens": generated_lengths(
                        generated, int(encoded.input_ids.shape[-1]), eos_token_ids
                    ),
                    "gpu_time_ms": gpu_ms,
                    "wall_time_ms": (time.perf_counter() - wall_started) * 1000.0,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
                atomic_json(checkpoint, measurement)
                del encoded, generated
                torch.cuda.empty_cache()

        start = clock
        finish = start + float(measurement["gpu_time_ms"])
        ready_snapshot = [
            {
                "program_id": state["program_id"],
                "turn_id": state["next_turn"],
            }
            for state in ready
        ]
        for state in chosen:
            turn = state["turns"][state["next_turn"]]
            state["queue_time_ms"] += start - state["ready_time_ms"]
            state["execution_time_ms"] += float(measurement["gpu_time_ms"])
            state["attained_gpu_ms"] += float(measurement["gpu_time_ms"])
            state["attained_ut"] += int(turn["sum_selected_ut"])
            state["next_turn"] += 1
            if state["next_turn"] == len(state["turns"]):
                state["completion_time_ms"] = finish
            else:
                state["ready_time_ms"] = finish + float(turn["tool_wait_ms"])
        schedule.append(
            {
                **measurement,
                "ready_candidates": ready_snapshot,
                "start_time_ms": start,
                "finish_time_ms": finish,
            }
        )
        clock = finish
        print(
            f"DONE batch={batch_index + 1} members={len(chosen)} "
            f"gpu_ms={measurement['gpu_time_ms']:.1f} "
            f"padded_input={measurement['padded_input_tokens']} "
            f"outputs={measurement['generated_tokens']}",
            flush=True,
        )
        batch_index += 1

    program_results = []
    for state in states:
        completion = state["completion_time_ms"] - float(
            state.get("arrival_time_ms", 0.0)
        )
        program_results.append(
            {
                "program_id": state["program_id"],
                "completion_time_ms": completion,
                "queue_time_ms": state["queue_time_ms"],
                "execution_time_ms": state["execution_time_ms"],
                "tool_wait_time_ms": 0.0,
                "slowdown": completion / state["execution_time_ms"],
            }
        )
    simulation = {
        "policy": f"{args.policy}_B{args.batch_size}",
        "programs": program_results,
        "schedule": schedule,
    }
    summary = summarize_simulation(simulation)
    batch_times = [float(item["gpu_time_ms"]) for item in schedule]
    token_count = sum(sum(item["generated_tokens"]) for item in schedule)
    result = {
        "configuration": {
            "dataset": "BFCL_v3_multi_turn_base",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "program_count": args.programs,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "policy": args.policy,
            "max_new_tokens": args.max_new_tokens,
            "physical_ut": 4,
            "selected_ut_threshold": 0.2,
            "runtime_num_hidden_layers": runtime_layers,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "teacher_forcing": "recorded BFCL calls and official tool results",
        },
        "summary": summary,
        "latency": {
            "batch_count": len(schedule),
            "total_gpu_time_ms": sum(batch_times),
            "mean_batch_gpu_time_ms": statistics.fmean(batch_times),
            "p50_batch_gpu_time_ms": statistics.median(batch_times),
            "p95_batch_gpu_time_ms": percentile(batch_times, 0.95),
            "generated_token_count_including_eos": token_count,
            "throughput_tokens_per_second": token_count / (sum(batch_times) / 1000.0),
            "max_peak_allocated_gib": max(
                item["peak_allocated_gib"] for item in schedule
            ),
            "max_peak_reserved_gib": max(
                item["peak_reserved_gib"] for item in schedule
            ),
            "batch_size_histogram": dict(
                sorted(Counter(item["batch_size"] for item in schedule).items())
            ),
        },
        "simulation": simulation,
    }
    atomic_json(args.output_dir / "results.json", result)
    compact = {key: value for key, value in result.items() if key != "simulation"}
    atomic_json(args.output_dir / "summary.json", compact)
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
