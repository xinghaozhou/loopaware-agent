from __future__ import annotations

import unittest

from experiments.ordinary_batching_numerics import (
    cdf_diagnostics,
    selected_ut,
)


class OrdinaryBatchingNumericsTest(unittest.TestCase):
    def test_selected_ut_uses_first_cdf_crossing(self) -> None:
        probabilities = [0.1, 0.2, 0.3, 0.4]
        self.assertEqual(selected_ut(probabilities, 0.2), 2)
        self.assertEqual(selected_ut(probabilities, 0.4), 3)
        self.assertEqual(selected_ut(probabilities, 1.0), 4)

    def test_cdf_diagnostics_report_threshold_distance(self) -> None:
        diagnostics = cdf_diagnostics([0.1, 0.2, 0.3, 0.4], 0.4)
        self.assertEqual(diagnostics["selected_ut"], 3)
        self.assertAlmostEqual(diagnostics["selected_cdf"], 0.6)
        self.assertAlmostEqual(
            diagnostics["minimum_abs_cdf_threshold_distance"], 0.1
        )


if __name__ == "__main__":
    unittest.main()
