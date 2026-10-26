from __future__ import annotations

import unittest

from experiments.program_scheduling.policies import POLICIES
from experiments.program_scheduling.simulator import simulate
from experiments.program_scheduling.metrics import pairwise_order_accuracy


def turn(program: str, index: int, runtime: float, selected: int) -> dict:
    return {
        "program_id": program,
        "turn_id": index,
        "output_token_count": 1,
        "selected_ut_per_token": [selected],
        "sum_selected_ut": selected,
        "mean_selected_ut": float(selected),
        "full_compute_gpu_time_ms": runtime,
        "full_compute_wall_time_ms": runtime,
        "tool_wait_ms": 0.0,
    }


class SimulatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.programs = [
            {"program_id": "A", "turns": [turn("A", 0, 8, 4)]},
            {"program_id": "B", "turns": [turn("B", 0, 3, 1)]},
            {"program_id": "C", "turns": [turn("C", 0, 6, 2)]},
        ]

    def test_fcfs_order(self) -> None:
        result = simulate(self.programs, "FCFS")
        self.assertEqual([row["program_id"] for row in result["schedule"]], ["A", "B", "C"])

    def test_ut_and_runtime_oracles_match_manual_order(self) -> None:
        for policy in ("ORACLE_UT_SUF", "ORACLE_RUNTIME_SJF"):
            result = simulate(self.programs, policy)
            self.assertEqual([row["program_id"] for row in result["schedule"]], ["B", "C", "A"])

    def test_every_policy_completes_every_program(self) -> None:
        for policy in POLICIES:
            result = simulate(self.programs, policy)
            self.assertEqual(len(result["programs"]), 3)
            self.assertEqual(len(result["schedule"]), 3)

    def test_attained_service_policies_match_manual_multiturn_order(self) -> None:
        programs = [
            {"program_id": "A", "turns": [turn("A", 0, 8, 4), turn("A", 1, 1, 1)]},
            {"program_id": "B", "turns": [turn("B", 0, 3, 1), turn("B", 1, 1, 1)]},
            {"program_id": "C", "turns": [turn("C", 0, 6, 2), turn("C", 1, 1, 1)]},
        ]
        expected = [("A", 0), ("B", 0), ("C", 0), ("B", 1), ("C", 1), ("A", 1)]
        for policy in ("GPU_PLAS", "UT_LAS"):
            result = simulate(programs, policy)
            actual = [(row["program_id"], row["turn_id"]) for row in result["schedule"]]
            self.assertEqual(actual, expected)

    def test_pairwise_proxy_tie_receives_half_credit(self) -> None:
        self.assertEqual(pairwise_order_accuracy([1, 1], [1, 2]), 0.5)


if __name__ == "__main__":
    unittest.main()
