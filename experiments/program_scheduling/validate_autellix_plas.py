"""Print and assert a small, manually inspectable PLAS schedule."""

from __future__ import annotations

import json

from .autellix_plas import PLASConfig, simulate_batched


def turn(service: float, tool_wait: float = 0.0) -> dict:
    return {
        "full_compute_gpu_time_ms": service,
        "tool_wait_ms": tool_wait,
        "output_token_count": 1,
        "sum_selected_ut": 1,
    }


def workload() -> list[dict]:
    return [
        {"program_id": "A", "arrival_time_ms": 0, "turns": [turn(6, 3), turn(1)]},
        {"program_id": "B", "arrival_time_ms": 0, "turns": [turn(1)]},
        {"program_id": "C", "arrival_time_ms": 0, "turns": [turn(5)]},
        {"program_id": "D", "arrival_time_ms": 7, "turns": [turn(1)]},
    ]


def main() -> None:
    config = PLASConfig(
        boundaries_ms=(0, 2, 4, 8, float("inf")),
        quanta_ms=(2, 2, 4, 8),
        anti_starvation_beta=None,
    )
    result = simulate_batched(workload(), "AUTELLIX_PLAS", batch_size=2, config=config)
    calls = {(call["program_id"], call["turn_id"]): call for call in result["calls"]}
    assert calls[("A", 0)]["preemptions"] >= 1
    assert calls[("C", 0)]["preemptions"] >= 1
    assert calls[("A", 1)]["inherited_service"] == 6
    assert calls[("A", 1)]["queue_index"] >= 2
    assert calls[("A", 1)]["inherited_service"] == calls[("A", 0)]["model_time"]
    assert result["summary"]["preemptions"] >= 2
    for event in result["timeline"]:
        if event["event"] == "dispatch":
            assert len(event["calls"]) <= 2
    print("time_ms  event     details")
    for event in result["timeline"]:
        details = {key: value for key, value in event.items() if key not in {"time_ms", "event"}}
        print(f"{event['time_ms']:7.1f}  {event['event']:<9} {json.dumps(details, sort_keys=True)}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
