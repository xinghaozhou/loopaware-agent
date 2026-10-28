from __future__ import annotations

import unittest

from experiments.program_scheduling.shared_gpu import verify_single_shared_gpu


class SharedGpuPoissonTest(unittest.TestCase):
    def test_shared_gpu_batches_do_not_overlap(self) -> None:
        result = verify_single_shared_gpu(
            [
                {"batch_size": 4, "start_time_ms": 0, "finish_time_ms": 10},
                {"batch_size": 2, "start_time_ms": 10, "finish_time_ms": 17},
            ],
            maximum_batch_size=4,
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["maximum_overlapping_gpu_batches"], 1)
        self.assertEqual(result["observed_maximum_batch_size"], 4)

    def test_independent_worker_overlap_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            verify_single_shared_gpu(
                [
                    {"batch_size": 1, "start_time_ms": 0, "finish_time_ms": 10},
                    {"batch_size": 1, "start_time_ms": 5, "finish_time_ms": 12},
                ],
                maximum_batch_size=4,
            )


if __name__ == "__main__":
    unittest.main()
