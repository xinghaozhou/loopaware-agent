from __future__ import annotations

import unittest

from experiments.analyze_predictor_headroom import (
    controlled_predictions,
    evaluate_groups,
)


class ControlledPredictorTest(unittest.TestCase):
    def test_oracle_is_exact(self) -> None:
        true = [1, 2, 3, 4] * 4
        predicted, accuracy = controlled_predictions(
            true, accuracy=1.0, u_max=4, seed=7
        )
        self.assertEqual(predicted, true)
        self.assertEqual(accuracy, 1.0)

    def test_wrong_predictions_are_adjacent_and_exact_count(self) -> None:
        true = [1, 2, 3, 4] * 10
        predicted, accuracy = controlled_predictions(
            true, accuracy=0.8, u_max=4, seed=11
        )
        differences = [abs(left - right) for left, right in zip(true, predicted)]
        self.assertEqual(differences.count(1), 8)
        self.assertEqual(differences.count(0), 32)
        self.assertEqual(accuracy, 0.8)
        self.assertTrue(all(1 <= value <= 4 for value in predicted))

    def test_fixed_seed_is_reproducible(self) -> None:
        true = [1, 2, 3, 4] * 10
        first = controlled_predictions(true, accuracy=0.7, u_max=4, seed=13)
        second = controlled_predictions(true, accuracy=0.7, u_max=4, seed=13)
        self.assertEqual(first, second)

    def test_single_batch_has_zero_reordering_headroom(self) -> None:
        group = [
            [
                (f"sample-{index}", value)
                for index, value in enumerate([4, 1, 3, 2] * 4)
            ]
        ]
        predictions = [value for _, value in group[0]]
        fcfs = evaluate_groups(group, None, batch_size=16, u_max=4)
        oracle = evaluate_groups(group, predictions, batch_size=16, u_max=4)
        self.assertEqual(fcfs["mean_total_cost"], oracle["mean_total_cost"])
        self.assertEqual(fcfs["mean_total_cost"], 64)


if __name__ == "__main__":
    unittest.main()
