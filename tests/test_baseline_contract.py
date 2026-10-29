"""Independent regression checks for the supported baseline contract."""
from __future__ import annotations

from contextlib import contextmanager
import unittest

import torch

from test_adaptive_kv import scalar_kv, tiny_model
from vendor.ouro_hf.cache_ouro import LastAvailableCache


@contextmanager
def scripted_exits(model, depths, physical):
    """Drive heterogeneous exits independently of randomly initialized gates."""
    step = 0

    def gate(_module, _args, output):
        nonlocal step
        step += 1
        active = [depth for depth in depths if not physical or depth >= step]
        return torch.tensor(
            [20.0 if depth == step else -20.0 for depth in active],
            dtype=output.dtype, device=output.device,
        ).reshape(-1, 1, 1)

    handle = model.model.early_exit_gate.register_forward_hook(gate)
    try:
        yield
    finally:
        handle.remove()


class BaselineContractTest(unittest.TestCase):
    def prefill_cache(self):
        cache = LastAvailableCache(1, 3)
        for depth in range(3):
            values = scalar_kv([10.0 * (depth + 1), 10.0 * (depth + 1) + 1])
            cache.update(values, -values, depth)
        return cache

    def commit_row(self, cache, row):
        cache.set_decode_batch_indices(torch.tensor([row]))
        cache.begin_decode(1)
        cache.update(scalar_kv([100.0 + row]), scalar_kv([-100.0 - row]), 0)
        cache.commit(torch.tensor([1]))

    def test_duplicate_frontier_request_is_rejected_without_losing_first_row(self):
        cache = self.prefill_cache()
        self.commit_row(cache, 0)
        with self.assertRaisesRegex(ValueError, "already committed"):
            cache.set_decode_batch_indices(torch.tensor([0]))
        self.commit_row(cache, 1)
        cache.begin_decode(2)
        key, _ = cache.update(scalar_kv([900.0, 901.0]), scalar_kv([0.0, 0.0]), 2)
        self.assertEqual(key[:, 0, :-1, 0].tolist(), [[30.0, 100.0], [31.0, 101.0]])
        cache.abort_decode()

    def test_beam_expansion_updates_request_count(self):
        cache = self.prefill_cache()
        cache.reorder_cache(torch.tensor([1, 0, 1]))
        cache.begin_decode(3)
        cache.update(scalar_kv([101.0, 100.0, 102.0]), scalar_kv([0.0] * 3), 0)
        cache.commit(torch.tensor([1, 1, 1]))
        self.assertEqual(cache.get_seq_length(), 2)
        cache.reorder_cache(torch.tensor([2, 0]))
        cache.begin_decode(2)
        key, _ = cache.update(scalar_kv([900.0, 901.0]), scalar_kv([0.0, 0.0]), 2)
        self.assertEqual(key[:, 0, :-1, 0].tolist(), [[31.0, 102.0], [31.0, 101.0]])
        cache.abort_decode()

    def test_partial_frontier_cannot_be_reordered_or_decoded_as_full_batch(self):
        cache = self.prefill_cache()
        self.commit_row(cache, 0)
        with self.assertRaisesRegex(RuntimeError, "partially committed"):
            cache.reorder_cache(torch.tensor([1, 0]))
        with self.assertRaisesRegex(ValueError, "Partial frontier"):
            cache.begin_decode(2)
        self.assertFalse(cache.in_decode)
        self.commit_row(cache, 1)
        self.assertEqual(cache.get_seq_length(), 2)

    def test_invalid_committed_depth_can_be_retried(self):
        cache = self.prefill_cache()
        cache.begin_decode(2)
        cache.update(scalar_kv([100.0, 101.0]), scalar_kv([0.0, 0.0]), 0)
        for depths in ([0, 1], [1, 4]):
            with self.assertRaisesRegex(ValueError, "UT range"):
                cache.commit(torch.tensor(depths))
        cache.commit(torch.tensor([1, 1]))
        self.assertEqual(cache.get_seq_length(), 2)

    def test_invalid_exit_does_not_leave_active_transaction(self):
        model = tiny_model("last_available", True)
        with torch.inference_mode():
            cache = model(torch.tensor([[1, 4, 5]]), use_cache=True).past_key_values
            with self.assertRaises(ValueError):
                model(torch.tensor([[6]]), past_key_values=cache, exit_at_step=3)
            self.assertFalse(cache.in_decode)
            model(torch.tensor([[6]]), past_key_values=cache, exit_at_step=0)
        self.assertEqual(cache.get_seq_length(), 4)

    def test_heterogeneous_padded_batch_matches_run_all_across_tokens(self):
        torch.manual_seed(31)
        reference = tiny_model("last_available", False)
        physical = tiny_model("last_available", True, reference.state_dict())
        prompt = torch.tensor([[0, 1, 4], [1, 5, 6], [0, 0, 7]])
        mask = (prompt != 0).long()
        with torch.inference_mode():
            ref = reference(prompt, attention_mask=mask, use_cache=True)
            phy = physical(prompt, attention_mask=mask, use_cache=True)
            for depths in ([1, 2, 3], [3, 1, 2], [2, 3, 1]):
                mask = torch.cat((mask, torch.ones(3, 1, dtype=torch.long)), dim=1)
                tokens = torch.tensor([[8], [9], [10]])
                with scripted_exits(reference, depths, False):
                    ref = reference(tokens, attention_mask=mask,
                                    past_key_values=ref.past_key_values, exit_threshold=0.5)
                with scripted_exits(physical, depths, True):
                    phy = physical(tokens, attention_mask=mask,
                                   past_key_values=phy.past_key_values, exit_threshold=0.5)
                self.assertEqual(ref.selected_ut_steps.tolist(), depths)
                self.assertEqual(phy.executed_ut_steps.tolist(), depths)
                torch.testing.assert_close(ref.logits, phy.logits, rtol=1e-5, atol=1e-6)

    def test_regrouped_model_matches_fixed_batch_over_multiple_frontiers(self):
        torch.manual_seed(37)
        fixed = tiny_model("last_available", True)
        grouped = tiny_model("last_available", True, fixed.state_dict())
        prompt = torch.tensor([[1, 4, 5], [1, 6, 7], [1, 8, 9]])
        with torch.inference_mode():
            full = fixed(prompt, use_cache=True)
            cache = grouped(prompt, use_cache=True).past_key_values
            for position, groups in enumerate(([[2, 0], [1]], [[1, 2], [0]])):
                tokens = torch.tensor([[10 + position], [12 + position], [14 + position]])
                full = fixed(tokens, past_key_values=full.past_key_values, exit_at_step=position)
                collected = torch.empty_like(full.logits)
                for rows in groups:
                    indices = torch.tensor(rows)
                    cache.set_decode_batch_indices(indices)
                    part = grouped(tokens.index_select(0, indices), past_key_values=cache,
                                   exit_at_step=position)
                    collected.index_copy_(0, indices, part.logits)
                torch.testing.assert_close(full.logits, collected, rtol=1e-5, atol=1e-6)
                self.assertEqual(cache.get_seq_length(), 4 + position)


if __name__ == "__main__":
    unittest.main()
