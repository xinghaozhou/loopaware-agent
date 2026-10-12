"""Smoke-test homogeneous UT batching and shortest-UT-first scheduling."""

from __future__ import annotations

from dataclasses import asdict
import json

from loopaware_agent.ut_scheduling import (
    UTSample,
    evaluate_lockstep_adaptive_exit,
    form_fcfs_batches,
    form_homogeneous_batches,
    schedule_shortest_ut_first,
)


def main() -> None:
    true_uts = [1, 4, 1, 4] * 4
    samples = [
        UTSample(f"sample-{index:02d}", true_ut=value, predicted_ut=value)
        for index, value in enumerate(true_uts)
    ]
    fcfs = evaluate_lockstep_adaptive_exit(
        form_fcfs_batches(samples, batch_size=4, u_max=4),
        batch_size=4,
        u_max=4,
    )
    aware_batches = form_homogeneous_batches(samples, batch_size=4, u_max=4)
    aware = evaluate_lockstep_adaptive_exit(
        schedule_shortest_ut_first(aware_batches),
        batch_size=4,
        u_max=4,
    )
    assert fcfs.total_execution_cost == 64
    assert aware.total_execution_cost == 40
    assert aware.total_useful_work == fcfs.total_useful_work == 40
    assert aware.total_wasted_slots == 0
    assert [item.batch.predicted_depth for item in aware.batches] == [1, 1, 4, 4]
    print(
        json.dumps(
            {
                "fcfs": asdict(fcfs),
                "ut_aware": asdict(aware),
                "cost_reduction": 1
                - aware.total_execution_cost / fcfs.total_execution_cost,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
