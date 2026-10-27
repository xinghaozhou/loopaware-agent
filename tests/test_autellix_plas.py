from __future__ import annotations

import unittest

from experiments.program_scheduling.autellix_plas import PLASConfig, simulate_batched
from experiments.program_scheduling.validate_autellix_plas import turn, workload


class AutellixPLASTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PLASConfig(
            boundaries_ms=(0, 2, 4, 8, float("inf")),
            quanta_ms=(2, 2, 4, 8),
            anti_starvation_beta=None,
        )

    def test_program_service_inheritance_demotion_and_tool_wait(self) -> None:
        result = simulate_batched(workload(), "AUTELLIX_PLAS", batch_size=2, config=self.config)
        calls = {(call["program_id"], call["turn_id"]): call for call in result["calls"]}
        self.assertGreater(calls[("A", 0)]["preemptions"], 0)
        self.assertEqual(calls[("A", 1)]["inherited_service"], 6)
        self.assertGreaterEqual(calls[("A", 1)]["queue_index"], 2)
        self.assertEqual(calls[("A", 0)]["model_time"], 6)

    def test_work_conserving_batch_fill(self) -> None:
        result = simulate_batched(
            [
                {"program_id": str(index), "arrival_time_ms": 0, "turns": [turn(4)]}
                for index in range(5)
            ],
            "AUTELLIX_PLAS",
            batch_size=4,
            config=self.config,
        )
        first = next(event for event in result["timeline"] if event["event"] == "dispatch")
        self.assertEqual(len(first["calls"]), 4)

    def test_proxy_and_autellix_are_distinct_policies(self) -> None:
        proxy = simulate_batched(workload(), "GPU_PLAS_PROXY", batch_size=2, config=self.config)
        plas = simulate_batched(workload(), "AUTELLIX_PLAS", batch_size=2, config=self.config)
        self.assertEqual(proxy["summary"]["preemptions"], 0)
        self.assertGreater(plas["summary"]["preemptions"], 0)


if __name__ == "__main__":
    unittest.main()
