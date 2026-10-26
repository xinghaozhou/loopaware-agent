"""Single-worker event-driven replay of fixed multi-turn program traces."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .policies import priority
from .trace_schema import validate_programs


def simulate(programs: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    validate_programs(programs)
    states = []
    for program in deepcopy(programs):
        states.append(
            {
                **program,
                "next_turn": 0,
                "ready_time_ms": float(program.get("arrival_time_ms", 0.0)),
                "attained_gpu_ms": 0.0,
                "attained_ut": 0,
                "queue_time_ms": 0.0,
                "execution_time_ms": 0.0,
                "tool_wait_time_ms": 0.0,
                "completion_time_ms": None,
            }
        )
    clock = min(state["ready_time_ms"] for state in states)
    schedule = []
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
        state = min(ready, key=lambda candidate: priority(policy, candidate))
        turn = state["turns"][state["next_turn"]]
        ready_candidates = []
        for candidate in ready:
            candidate_turn = candidate["turns"][candidate["next_turn"]]
            ready_candidates.append(
                {
                    "program_id": candidate["program_id"],
                    "turn_id": candidate["next_turn"],
                    "output_token_count": candidate_turn["output_token_count"],
                    "sum_selected_ut": candidate_turn["sum_selected_ut"],
                    "full_compute_gpu_time_ms": candidate_turn["full_compute_gpu_time_ms"],
                }
            )
        start = clock
        queue = start - state["ready_time_ms"]
        runtime = float(turn["full_compute_gpu_time_ms"])
        finish = start + runtime
        state["queue_time_ms"] += queue
        state["execution_time_ms"] += runtime
        state["attained_gpu_ms"] += runtime
        state["attained_ut"] += int(turn["sum_selected_ut"])
        schedule.append(
            {
                "program_id": state["program_id"],
                "turn_id": state["next_turn"],
                "ready_time_ms": state["ready_time_ms"],
                "start_time_ms": start,
                "finish_time_ms": finish,
                "queue_time_ms": queue,
                "ready_candidates": ready_candidates,
            }
        )
        clock = finish
        state["next_turn"] += 1
        if state["next_turn"] == len(state["turns"]):
            state["completion_time_ms"] = finish
        else:
            wait = float(turn["tool_wait_ms"])
            state["tool_wait_time_ms"] += wait
            state["ready_time_ms"] = finish + wait

    rows = []
    for state in states:
        arrival = float(state.get("arrival_time_ms", 0.0))
        isolated = state["execution_time_ms"] + state["tool_wait_time_ms"]
        completion = state["completion_time_ms"] - arrival
        rows.append(
            {
                "program_id": state["program_id"],
                "completion_time_ms": completion,
                "queue_time_ms": state["queue_time_ms"],
                "execution_time_ms": state["execution_time_ms"],
                "tool_wait_time_ms": state["tool_wait_time_ms"],
                "slowdown": completion / isolated,
            }
        )
    return {"policy": policy, "programs": rows, "schedule": schedule}
