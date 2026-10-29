from __future__ import annotations

import unittest

from loopaware_agent.ut_scheduling import (
    UTSample,
    evaluate_lockstep_adaptive_exit,
    form_fcfs_batches,
    form_homogeneous_batches,
    schedule_shortest_ut_first,
)


class UTSchedulingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            UTSample(f"s{index}", true_ut=value, predicted_ut=value)
            for index, value in enumerate([1, 4, 1, 4] * 4)
        ]

    def test_homogeneous_batching_uses_predicted_ut(self) -> None:
        samples = [
            UTSample("a", true_ut=4, predicted_ut=1),
            UTSample("b", true_ut=1, predicted_ut=4),
            UTSample("c", true_ut=3, predicted_ut=1),
            UTSample("d", true_ut=2, predicted_ut=4),
        ]
        batches = form_homogeneous_batches(samples, batch_size=2, u_max=4)
        self.assertEqual(
            [[sample.sample_id for sample in batch.samples] for batch in batches],
            [["a", "c"], ["b", "d"]],
        )

    def test_shortest_ut_first_orders_by_predicted_batch_depth(self) -> None:
        batches = form_homogeneous_batches(self.samples, batch_size=4, u_max=4)
        scheduled = schedule_shortest_ut_first(list(reversed(batches)))
        self.assertEqual([batch.predicted_depth for batch in scheduled], [1, 1, 4, 4])

    def test_cost_uses_true_ut_not_prediction(self) -> None:
        samples = [
            UTSample(f"s{index}", true_ut=4, predicted_ut=1)
            for index in range(4)
        ]
        result = evaluate_lockstep_adaptive_exit(
            form_homogeneous_batches(samples, batch_size=4, u_max=4),
            batch_size=4,
            u_max=4,
        )
        self.assertEqual(result.total_execution_cost, 16)
        self.assertEqual(result.total_useful_work, 16)
        self.assertEqual(result.batches[0].active_rows_by_ut, (4, 4, 4, 4))

    def test_homogeneous_oracle_reduces_lockstep_waste(self) -> None:
        fcfs = evaluate_lockstep_adaptive_exit(
            form_fcfs_batches(self.samples, batch_size=4, u_max=4),
            batch_size=4,
            u_max=4,
        )
        aware = evaluate_lockstep_adaptive_exit(
            schedule_shortest_ut_first(
                form_homogeneous_batches(self.samples, batch_size=4, u_max=4)
            ),
            batch_size=4,
            u_max=4,
        )
        self.assertEqual(fcfs.total_execution_cost, 64)
        self.assertEqual(aware.total_execution_cost, 40)
        self.assertEqual(fcfs.total_useful_work, aware.total_useful_work)
        self.assertEqual(aware.mean_true_ut_range, 0.0)
        self.assertLess(
            aware.mean_batch_completion_time, fcfs.mean_batch_completion_time
        )

    def test_rejects_partial_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            form_homogeneous_batches(self.samples[:3], batch_size=4, u_max=4)


if __name__ == "__main__":
    unittest.main()
