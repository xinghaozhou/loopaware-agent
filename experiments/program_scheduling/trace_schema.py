"""Validated JSON schema helpers for fixed BFCL program traces."""

from __future__ import annotations

from typing import Any


REQUIRED_TURN_FIELDS = {
    "program_id",
    "turn_id",
    "output_token_count",
    "selected_ut_per_token",
    "sum_selected_ut",
    "mean_selected_ut",
    "full_compute_gpu_time_ms",
    "full_compute_wall_time_ms",
    "tool_wait_ms",
}


def validate_turn(turn: dict[str, Any], *, total_ut_steps: int = 4) -> None:
    missing = REQUIRED_TURN_FIELDS - turn.keys()
    if missing:
        raise ValueError(f"turn is missing fields: {sorted(missing)}")
    selected = turn["selected_ut_per_token"]
    if len(selected) != turn["output_token_count"]:
        raise ValueError("selected-UT count does not match output-token count")
    if selected and (min(selected) < 1 or max(selected) > total_ut_steps):
        raise ValueError("selected UT is outside the configured recurrent depth")
    if sum(selected) != turn["sum_selected_ut"]:
        raise ValueError("sum_selected_ut is inconsistent")
    if turn["full_compute_gpu_time_ms"] <= 0:
        raise ValueError("GPU runtime must be positive")


def validate_programs(programs: list[dict[str, Any]], *, total_ut_steps: int = 4) -> None:
    ids = [program["program_id"] for program in programs]
    if len(ids) != len(set(ids)):
        raise ValueError("program IDs must be unique")
    for program in programs:
        if not program["turns"]:
            raise ValueError(f"{program['program_id']} has no turns")
        for index, turn in enumerate(program["turns"]):
            if turn["program_id"] != program["program_id"] or turn["turn_id"] != index:
                raise ValueError("program/turn identity is inconsistent")
            validate_turn(turn, total_ut_steps=total_ut_steps)
