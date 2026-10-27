from __future__ import annotations

import unittest

import torch

from experiments.program_scheduling.run_full_compute_batch import generated_lengths


class FullComputeBatchTest(unittest.TestCase):
    def test_generated_lengths_stop_at_first_eos(self) -> None:
        generated = torch.tensor(
            [
                [9, 9, 4, 5, 2, 2],
                [9, 9, 8, 7, 6, 5],
            ]
        )
        self.assertEqual(generated_lengths(generated, 2, {2}), [3, 4])


if __name__ == "__main__":
    unittest.main()
