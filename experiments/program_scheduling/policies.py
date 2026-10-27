"""Priority functions for non-preemptive program-level scheduling."""

from __future__ import annotations

from typing import Any


POLICIES = (
    "FCFS",
    "GPU_PLAS_PROXY",
    "UT_LAS",
    "ORACLE_TOKEN_SUF",
    "ORACLE_UT_SUF",
    "ORACLE_RUNTIME_SJF",
)


def priority(policy: str, state: dict[str, Any]) -> tuple[Any, ...]:
    turn = state["turns"][state["next_turn"]]
    stable = (state["ready_time_ms"], state["program_id"])
    if policy == "FCFS":
        return stable
    if policy in {"GPU_PLAS", "GPU_PLAS_PROXY"}:
        return (state["attained_gpu_ms"], *stable)
    if policy == "UT_LAS":
        return (state["attained_ut"], *stable)
    if policy == "ORACLE_TOKEN_SUF":
        return (turn["output_token_count"], *stable)
    if policy == "ORACLE_UT_SUF":
        return (turn["sum_selected_ut"], *stable)
    if policy == "ORACLE_RUNTIME_SJF":
        return (turn["full_compute_gpu_time_ms"], *stable)
    raise ValueError(f"unknown policy: {policy}")
