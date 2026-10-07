"""Resumable fixed-trajectory Ouro UT tracing for mini-SWE-agent records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from loopaware_agent.ut_trace import OuroUTTracer  # noqa: E402
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip requests already checkpointed in request_summaries.jsonl",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    """Atomically checkpoint one request's token records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def request_trace_path(
    run_dir: Path, condition_name: str, instance_id: str, turn_idx: int
) -> Path:
    return (
        run_dir
        / "raw"
        / "requests"
        / condition_name
        / instance_id
        / f"turn-{turn_idx:04d}.jsonl"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                values.append(json.loads(line))
    return values


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def trajectory_filename(subset: str, instance_id: str) -> str:
    return (
        f"swebench_verified_raw/{subset}/{instance_id}/"
        f"{instance_id}.traj.json"
    )


def load_trajectory(config: dict[str, Any], instance_id: str) -> dict[str, Any]:
    path = hf_hub_download(
        repo_id=config["dataset_repo"],
        repo_type="dataset",
        revision=config["dataset_revision"],
        filename=trajectory_filename(config["dataset_model_subset"], instance_id),
    )
    with Path(path).open(encoding="utf-8") as stream:
        trajectory = json.load(stream)
    if trajectory["instance_id"] != instance_id:
        raise ValueError(f"Trajectory ID mismatch for {instance_id}")
    if not trajectory["info"].get("resolved"):
        raise ValueError(f"Trajectory {instance_id} is not marked resolved")
    return trajectory


def normalized_message(message: dict[str, Any]) -> dict[str, str]:
    """Preserve recorded content and make native tool calls visible to Ouro."""

    content = message.get("content") or ""
    tool_calls = message.get("tool_calls")
    if tool_calls:
        content += (
            "\n<tool_calls>\n"
            + json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
            + "\n</tool_calls>"
        )
    return {"role": message["role"], "content": content}


def replay_requests(
    tokenizer: Any, trajectory: dict[str, Any]
) -> list[dict[str, Any]]:
    prefix: list[dict[str, str]] = []
    requests = []
    turn_idx = 0
    for message in trajectory["messages"]:
        if message["role"] == "assistant":
            input_ids = tokenizer.apply_chat_template(
                prefix,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            requests.append(
                {
                    "turn_idx": turn_idx,
                    "input_ids": input_ids,
                    "input_token_count": int(input_ids.shape[-1]),
                    "recorded_response_id": message.get("id"),
                }
            )
            turn_idx += 1
        prefix.append(normalized_message(message))
    return requests


def percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def request_summary(
    *,
    key: str,
    condition: dict[str, Any],
    instance_id: str,
    request: dict[str, Any],
    events: list[Any],
    num_decoder_layers: int,
    elapsed_seconds: float,
    peak_allocated_gib: float,
    peak_reserved_gib: float,
) -> dict[str, Any]:
    selected = [event.selected_ut_step for event in events]
    executed = [event.executed_ut_steps for event in events]
    selected_total = sum(selected)
    executed_total = sum(executed)
    mean = selected_total / len(selected) if selected else None
    variance = (
        sum((value - mean) ** 2 for value in selected) / len(selected)
        if selected
        else None
    )
    return {
        "request_key": key,
        "condition": condition["name"],
        "total_ut_steps": condition["total_ut_steps"],
        "task_id": instance_id,
        "request_id": f"{instance_id}:turn-{request['turn_idx']}",
        "turn_idx": request["turn_idx"],
        "recorded_response_id": request["recorded_response_id"],
        "input_token_count": request["input_token_count"],
        "output_token_count": len(events),
        "num_decoder_layers": num_decoder_layers,
        "selected_ut_total": selected_total,
        "executed_ut_total": executed_total,
        "mean_selected_ut": mean,
        "std_selected_ut": math.sqrt(variance) if variance is not None else None,
        "p50_selected_ut": percentile(selected, 0.50),
        "p90_selected_ut": percentile(selected, 0.90),
        "p95_selected_ut": percentile(selected, 0.95),
        "min_selected_ut": min(selected) if selected else None,
        "max_selected_ut": max(selected) if selected else None,
        "selected_ut_histogram": dict(sorted(Counter(selected).items())),
        "selected_decoder_layer_calls_counterfactual_total": (
            selected_total * num_decoder_layers
        ),
        "executed_decoder_layer_calls_total": executed_total * num_decoder_layers,
        "theoretical_recurrence_saving": (
            1.0 - selected_total / executed_total if executed_total else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "completed_at": utc_now(),
    }


def token_records(
    *,
    condition: dict[str, Any],
    instance_id: str,
    request: dict[str, Any],
    events: Iterable[Any],
    num_decoder_layers: int,
) -> Iterable[dict[str, Any]]:
    events = list(events)
    output_count = len(events)
    for event in events:
        cumulative = []
        running = 0.0
        for probability in event.exit_probabilities:
            running += probability
            cumulative.append(running)
        yield {
            "condition": condition["name"],
            "total_ut_steps": condition["total_ut_steps"],
            "task_id": instance_id,
            "request_id": f"{instance_id}:turn-{request['turn_idx']}",
            "turn_idx": request["turn_idx"],
            "token_idx": event.output_index,
            "token_id": event.token_id,
            "token": event.token,
            "input_token_count": request["input_token_count"],
            "output_token_count_final": output_count,
            "num_decoder_layers": num_decoder_layers,
            "executed_ut_steps": event.executed_ut_steps,
            "selected_ut_step": event.selected_ut_step,
            "decode_ut_step": event.decode_ut_step,
            "expected_ut_steps": event.expected_ut_steps,
            "executed_decoder_layer_calls": (
                event.executed_ut_steps * num_decoder_layers
            ),
            "selected_decoder_layer_calls_counterfactual": (
                event.selected_ut_step * num_decoder_layers
            ),
            "gate_probability_by_ut": event.gate_probabilities,
            "exit_probability_by_ut": event.exit_probabilities,
            "cumulative_exit_probability_by_ut": cumulative,
            "trace_exit_threshold": event.exit_threshold,
        }


def summarize_tasks(run_dir: Path) -> None:
    requests = read_jsonl(run_dir / "request_summaries.jsonl")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        grouped[(request["condition"], request["task_id"])].append(request)
    summaries = []
    for (condition, task_id), values in sorted(grouped.items()):
        values.sort(key=lambda value: value["turn_idx"])
        selected_total = sum(value["selected_ut_total"] for value in values)
        executed_total = sum(value["executed_ut_total"] for value in values)
        summaries.append(
            {
                "condition": condition,
                "task_id": task_id,
                "num_turns": len(values),
                "input_token_count_by_turn": [
                    value["input_token_count"] for value in values
                ],
                "mean_selected_ut_by_turn": [
                    value["mean_selected_ut"] for value in values
                ],
                "selected_ut_total_by_turn": [
                    value["selected_ut_total"] for value in values
                ],
                "theoretical_saving_by_turn": [
                    value["theoretical_recurrence_saving"] for value in values
                ],
                "selected_ut_total": selected_total,
                "executed_ut_total": executed_total,
                "theoretical_recurrence_saving": (
                    1.0 - selected_total / executed_total if executed_total else None
                ),
                "elapsed_seconds": sum(value["elapsed_seconds"] for value in values),
            }
        )
    atomic_json(run_dir / "task_summaries.json", summaries)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_output = run_dir / "experiment_config.json"
    if config_output.exists():
        existing = json.loads(config_output.read_text(encoding="utf-8"))
        if existing["source_config"] != config:
            raise RuntimeError("Run directory contains a different experiment config")
    else:
        atomic_json(
            config_output,
            {
                "source_config": config,
                "source_config_path": str(args.config.resolve()),
                "git_commit": git_commit(),
                "started_at": utc_now(),
                "torch_version": torch.__version__,
            },
        )

    request_path = run_dir / "request_summaries.jsonl"
    completed = {
        value["request_key"] for value in read_jsonl(request_path)
    } if args.resume else set()
    if not args.resume and request_path.exists():
        raise RuntimeError("Run output exists; pass --resume or choose a new run directory")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        trust_remote_code=True,
    )
    all_ids = sorted(
        {
            instance_id
            for condition in config["conditions"]
            for instance_id in condition["trajectory_ids"]
        }
    )
    trajectories = {
        instance_id: load_trajectory(config, instance_id) for instance_id in all_ids
    }
    requests_by_id = {
        instance_id: replay_requests(tokenizer, trajectory)
        for instance_id, trajectory in trajectories.items()
    }
    planned_requests = sum(
        len(requests_by_id[instance_id])
        for condition in config["conditions"]
        for instance_id in condition["trajectory_ids"]
    )
    print(
        f"run_dir={run_dir} planned_requests={planned_requests} "
        f"already_completed={len(completed)}",
        flush=True,
    )

    completed_this_process = 0
    error_count = 0
    for condition in config["conditions"]:
        ouro_config = OuroConfig.from_pretrained(
            config["model_id"], revision=config["model_revision"]
        )
        ouro_config.total_ut_steps = condition["total_ut_steps"]
        model = OuroForCausalLM.from_pretrained(
            config["model_id"],
            revision=config["model_revision"],
            config=ouro_config,
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        model.early_exit_step = None
        model.early_exit_threshold = None
        num_decoder_layers = model.config.num_hidden_layers
        if not isinstance(num_decoder_layers, int) or num_decoder_layers <= 0:
            raise ValueError(
                f"Invalid runtime num_hidden_layers={num_decoder_layers!r}"
            )
        print(
            f"condition={condition['name']} total_ut_steps="
            f"{condition['total_ut_steps']} runtime_num_hidden_layers="
            f"{num_decoder_layers}",
            flush=True,
        )

        for instance_id in condition["trajectory_ids"]:
            for request in requests_by_id[instance_id]:
                key = (
                    f"{condition['name']}:{instance_id}:"
                    f"turn-{request['turn_idx']}"
                )
                if key in completed:
                    continue
                if request["input_token_count"] > config["max_input_tokens"]:
                    append_jsonl(
                        run_dir / "errors.jsonl",
                        {
                            "request_key": key,
                            "error_type": "InputTooLong",
                            "input_token_count": request["input_token_count"],
                            "max_input_tokens": config["max_input_tokens"],
                            "recorded_at": utc_now(),
                        },
                    )
                    error_count += 1
                    print(f"SKIP {key} input too long", flush=True)
                    continue

                input_ids = request["input_ids"].to(model.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                try:
                    with torch.inference_mode(), OuroUTTracer(
                        model, exit_threshold=config["trace_exit_threshold"]
                    ) as tracer:
                        output_ids = model.generate(
                            input_ids,
                            max_new_tokens=config["max_new_tokens"],
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    generated_ids = output_ids[:, input_ids.shape[1] :]
                    events = tracer.attach_output_tokens(generated_ids, tokenizer)
                    atomic_jsonl(
                        request_trace_path(
                            run_dir,
                            condition["name"],
                            instance_id,
                            request["turn_idx"],
                        ),
                        token_records(
                            condition=condition,
                            instance_id=instance_id,
                            request=request,
                            events=events,
                            num_decoder_layers=num_decoder_layers,
                        ),
                    )
                    summary = request_summary(
                        key=key,
                        condition=condition,
                        instance_id=instance_id,
                        request=request,
                        events=events,
                        num_decoder_layers=num_decoder_layers,
                        elapsed_seconds=elapsed,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    )
                    append_jsonl(request_path, summary)
                    completed.add(key)
                    completed_this_process += 1
                    atomic_json(
                        run_dir / "progress.json",
                        {
                            "planned_requests": planned_requests,
                            "completed_requests": len(completed),
                            "errors_this_process": error_count,
                            "last_completed_request": key,
                            "updated_at": utc_now(),
                        },
                    )
                    print(
                        f"DONE {len(completed)}/{planned_requests} {key} "
                        f"input={request['input_token_count']} output={len(events)} "
                        f"seconds={elapsed:.1f}",
                        flush=True,
                    )
                    del output_ids, generated_ids, events
                except Exception as error:
                    elapsed = time.perf_counter() - start
                    append_jsonl(
                        run_dir / "errors.jsonl",
                        {
                            "request_key": key,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "input_token_count": request["input_token_count"],
                            "elapsed_seconds": elapsed,
                            "recorded_at": utc_now(),
                        },
                    )
                    error_count += 1
                    print(
                        f"ERROR {key} {type(error).__name__}: {error}", flush=True
                    )
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                finally:
                    del input_ids

        del model
        torch.cuda.empty_cache()

    summarize_tasks(run_dir)
    atomic_json(
        run_dir / "completion.json",
        {
            "planned_requests": planned_requests,
            "completed_requests": len(completed),
            "completed_this_process": completed_this_process,
            "errors_this_process": error_count,
            "finished_at": utc_now(),
        },
    )
    print(
        f"FINISHED completed={len(completed)}/{planned_requests} errors={error_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
