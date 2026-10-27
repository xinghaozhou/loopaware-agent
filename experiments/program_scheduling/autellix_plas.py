"""Trace-driven approximation of Autellix's single-threaded PLAS scheduler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import inf
from statistics import mean, median
from typing import Any


DEFAULT_BOUNDARIES_MS = (0.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0, 32_000.0, 64_000.0, inf)
DEFAULT_QUANTA_MS = (1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0, 32_000.0, 64_000.0)


@dataclass(frozen=True)
class PLASConfig:
    boundaries_ms: tuple[float, ...] = DEFAULT_BOUNDARIES_MS
    quanta_ms: tuple[float, ...] = DEFAULT_QUANTA_MS
    anti_starvation_beta: float | None = 8.0

    def __post_init__(self) -> None:
        if len(self.boundaries_ms) != len(self.quanta_ms) + 1:
            raise ValueError("boundaries must have one more element than quanta")
        if self.boundaries_ms[0] != 0 or any(
            left >= right for left, right in zip(self.boundaries_ms, self.boundaries_ms[1:])
        ):
            raise ValueError("boundaries must increase from zero")
        if any(value <= 0 for value in self.quanta_ms):
            raise ValueError("quanta must be positive")

    def queue_for_service(self, service_ms: float) -> int:
        for index, (low, high) in enumerate(zip(self.boundaries_ms, self.boundaries_ms[1:])):
            if low <= service_ms < high:
                return index
        return len(self.quanta_ms) - 1


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def simulate_batched(
    programs: list[dict[str, Any]],
    policy: str,
    *,
    batch_size: int,
    config: PLASConfig = PLASConfig(),
) -> dict[str, Any]:
    """Replay fixed call service with preemptive PLAS or non-preemptive baselines."""

    if policy not in {"FCFS", "GPU_PLAS_PROXY", "AUTELLIX_PLAS"}:
        raise ValueError(f"unsupported policy: {policy}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    states = {
        program["program_id"]: {
            "program": program,
            "next_turn": 0,
            "ready_time": float(program.get("arrival_time_ms", 0.0)),
            "service": 0.0,
            "waiting": 0.0,
            "execution": 0.0,
            "completion": None,
            "active": None,
        }
        for program in programs
    }
    queues = [deque() for _ in config.quanta_ms]
    waiting: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    order = 0
    clock = min(state["ready_time"] for state in states.values())
    preemptions = 0
    timeline: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    def enqueue_ready() -> None:
        nonlocal order
        for program_id in sorted(states):
            state = states[program_id]
            if state["completion"] is not None or state["active"] is not None:
                continue
            if state["ready_time"] > clock:
                continue
            turn_idx = state["next_turn"]
            turn = state["program"]["turns"][turn_idx]
            order += 1
            queue_idx = config.queue_for_service(state["service"])
            call = {
                "program_id": program_id,
                "turn_id": turn_idx,
                "remaining": float(turn["full_compute_gpu_time_ms"]),
                "model_time": 0.0,
                "wait_time": 0.0,
                "run_window": 0.0,
                "wait_window": 0.0,
                "inherited_service": state["service"],
                "queue_index": queue_idx,
                "quantum_remaining": config.quanta_ms[queue_idx],
                "enqueue_order": order,
                "preemptions": 0,
            }
            state["active"] = call
            if policy == "AUTELLIX_PLAS":
                queues[queue_idx].append(call)
            else:
                waiting.append(call)

    def pop_next() -> dict[str, Any] | None:
        if policy == "AUTELLIX_PLAS":
            for queue in queues:
                if queue:
                    return queue.popleft()
            return None
        if not waiting:
            return None
        if policy == "FCFS":
            chosen = min(waiting, key=lambda call: call["enqueue_order"])
        else:
            chosen = min(
                waiting,
                key=lambda call: (call["inherited_service"], call["enqueue_order"]),
            )
        waiting.remove(chosen)
        return chosen

    def queued_calls() -> list[dict[str, Any]]:
        return [call for queue in queues for call in queue] if policy == "AUTELLIX_PLAS" else waiting

    while any(state["completion"] is None for state in states.values()):
        enqueue_ready()
        dispatched = []
        while len(running) < batch_size:
            call = pop_next()
            if call is None:
                break
            running.append(call)
            dispatched.append(
                {
                    "program_id": call["program_id"],
                    "turn_id": call["turn_id"],
                    "queue_index": call["queue_index"],
                    "remaining_ms": call["remaining"],
                    "inherited_service_ms": call["inherited_service"],
                }
            )
        if dispatched:
            timeline.append({"time_ms": clock, "event": "dispatch", "calls": dispatched})
        if not running:
            clock = min(
                state["ready_time"]
                for state in states.values()
                if state["completion"] is None and state["active"] is None
            )
            continue

        delta = min(call["remaining"] for call in running)
        if policy == "AUTELLIX_PLAS":
            delta = min(delta, *(call["quantum_remaining"] for call in running))
        future = [
            state["ready_time"] - clock
            for state in states.values()
            if state["completion"] is None
            and state["active"] is None
            and state["ready_time"] > clock
        ]
        if future:
            delta = min(delta, min(future))
        for call in queued_calls():
            call["wait_time"] += delta
            call["wait_window"] += delta
        for call in running:
            call["remaining"] -= delta
            call["model_time"] += delta
            call["run_window"] += delta
            if policy == "AUTELLIX_PLAS":
                call["quantum_remaining"] -= delta
        clock += delta

        for call in list(running):
            state = states[call["program_id"]]
            turn = state["program"]["turns"][call["turn_id"]]
            if call["remaining"] <= 1e-9:
                running.remove(call)
                state["service"] += call["model_time"]
                state["waiting"] += call["wait_time"]
                state["execution"] += call["model_time"]
                state["next_turn"] += 1
                state["active"] = None
                calls.append({**call, "finish_time_ms": clock})
                timeline.append(
                    {"time_ms": clock, "event": "finish", "program_id": call["program_id"], "turn_id": call["turn_id"]}
                )
                if state["next_turn"] == len(state["program"]["turns"]):
                    state["completion"] = clock
                else:
                    state["ready_time"] = clock + float(turn.get("tool_wait_ms", 0.0))
            elif policy == "AUTELLIX_PLAS" and call["quantum_remaining"] <= 1e-9:
                running.remove(call)
                call["queue_index"] = min(call["queue_index"] + 1, len(queues) - 1)
                call["quantum_remaining"] = config.quanta_ms[call["queue_index"]]
                order += 1
                call["enqueue_order"] = order
                call["preemptions"] += 1
                preemptions += 1
                queues[call["queue_index"]].append(call)
                timeline.append(
                    {"time_ms": clock, "event": "preempt", "program_id": call["program_id"], "turn_id": call["turn_id"], "queue_index": call["queue_index"]}
                )

        if policy == "AUTELLIX_PLAS" and config.anti_starvation_beta is not None:
            for queue_index in range(1, len(queues)):
                for call in list(queues[queue_index]):
                    state = states[call["program_id"]]
                    denominator = max(1.0, state["service"] + call["run_window"])
                    if (state["waiting"] + call["wait_window"]) / denominator >= config.anti_starvation_beta:
                        queues[queue_index].remove(call)
                        call["queue_index"] = 0
                        call["quantum_remaining"] = config.quanta_ms[0]
                        call["wait_window"] = 0.0
                        call["run_window"] = 0.0
                        order += 1
                        call["enqueue_order"] = order
                        queues[0].append(call)

    completion = [float(state["completion"]) - float(state["program"].get("arrival_time_ms", 0.0)) for state in states.values()]
    queueing = [state["waiting"] for state in states.values()]
    makespan = max(float(state["completion"]) for state in states.values())
    return {
        "policy": policy,
        "batch_size": batch_size,
        "summary": {
            "mean_program_ttlt_ms": mean(completion),
            "median_program_ttlt_ms": median(completion),
            "p95_program_ttlt_ms": _percentile(completion, 0.95),
            "mean_queueing_time_ms": mean(queueing),
            "throughput_programs_per_s": len(programs) / (makespan / 1000.0),
            "makespan_ms": makespan,
            "preemptions": preemptions,
        },
        "programs": [
            {
                "program_id": program_id,
                "completion_time_ms": float(state["completion"]),
                "model_service_ms": state["service"],
                "queueing_time_ms": state["waiting"],
            }
            for program_id, state in states.items()
        ],
        "calls": calls,
        "timeline": timeline,
        "queue_config": {
            "boundaries_ms": list(config.boundaries_ms),
            "quanta_ms": list(config.quanta_ms),
            "anti_starvation_beta": config.anti_starvation_beta,
        },
    }
