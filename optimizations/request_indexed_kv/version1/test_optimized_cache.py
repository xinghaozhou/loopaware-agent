from __future__ import annotations

import unittest

import torch

from optimizations.request_indexed_kv.version1.optimized_cache import (
    InPlaceFullPerDepthCache,
    RequestIndexedLastAvailableCache,
)


class OptimizedCacheTest(unittest.TestCase):
    def test_original_appends_in_place_without_changing_values(self) -> None:
        cache = InPlaceFullPerDepthCache(1)
        first = torch.arange(12, dtype=torch.float32).reshape(2, 1, 3, 2)
        second = torch.full((2, 1, 1, 2), 99.0)
        key, _ = cache.update(first, first, 0)
        self.assertTrue(torch.equal(key, first))
        key, _ = cache.update(second, second, 0)
        self.assertTrue(torch.equal(key[:, :, :3], first))
        self.assertTrue(torch.equal(key[:, :, 3:], second))

    def test_request_indexed_membership_does_not_reorder_persistent_rows(self) -> None:
        cache = RequestIndexedLastAvailableCache(1, 2)
        prompt = torch.zeros(3, 1, 2, 1)
        for slot in range(2):
            cache.update(prompt + slot, prompt + slot, slot)
        cache.set_decode_batch_indices(torch.tensor([2, 0]), validate=False)
        cache.begin_decode(2)
        cache.update(
            torch.tensor([[[[20.0]]], [[[0.0]]]]),
            torch.zeros(2, 1, 1, 1),
            0,
            {"batch_indices": torch.tensor([0, 1])},
        )
        cache.update(
            torch.tensor([[[[21.0]]], [[[1.0]]]]),
            torch.zeros(2, 1, 1, 1),
            1,
            {"batch_indices": torch.tensor([0, 1])},
        )
        cache.commit(torch.tensor([2, 2]))
        cache.set_decode_batch_indices(torch.tensor([1]), validate=False)
        cache.begin_decode(1)
        cache.update(
            torch.tensor([[[[10.0]]]]),
            torch.zeros(1, 1, 1, 1),
            0,
            {"batch_indices": torch.tensor([0])},
        )
        cache.update(
            torch.tensor([[[[11.0]]]]),
            torch.zeros(1, 1, 1, 1),
            1,
            {"batch_indices": torch.tensor([0])},
        )
        cache.commit(torch.tensor([2]))
        cache.set_decode_batch_indices(torch.tensor([1]), validate=False)
        cache.begin_decode(1)
        key, _ = cache.update(
            torch.tensor([[[[12.0]]]]),
            torch.zeros(1, 1, 1, 1),
            0,
            {"batch_indices": torch.tensor([0])},
        )
        self.assertEqual(key.shape[2], 4)
        self.assertEqual(float(cache._key_buffers[0][2, 0, 2, 0]), 20.0)
        self.assertEqual(float(cache._key_buffers[0][0, 0, 2, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
