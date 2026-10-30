"""Independent oracles and failure-recovery checks for the core baseline."""
from contextlib import contextmanager
import unittest

import torch

from test_adaptive_kv import scalar_kv, tiny_model
from vendor.ouro_hf.cache_ouro import LastAvailableCache


@contextmanager
def scripted_exits(model, depths, physical):
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


class CoreCacheContractTest(unittest.TestCase):
    def cache(self):
        cache = LastAvailableCache(1, 3)
        for depth in range(3):
            values = scalar_kv([10.0 * (depth + 1)])
            cache.update(values, -values, depth)
        return cache

    def test_staged_current_kv_attends_but_is_not_committed_history(self):
        cache = self.cache()
        cache.begin_decode(1)
        key, value = cache.update(scalar_kv([100.0]), scalar_kv([-100.0]), 0)
        self.assertEqual(key.flatten().tolist(), [10.0, 100.0])
        self.assertEqual(value.flatten().tolist(), [-10.0, -100.0])
        self.assertEqual(cache.get_seq_length(), 1)
        cache.abort_decode()
        self.assertEqual(cache.get_seq_length(), 1)
        cache.begin_decode(1)
        key, _ = cache.update(scalar_kv([200.0]), scalar_kv([-200.0]), 0)
        self.assertEqual(key.flatten().tolist(), [10.0, 200.0])
        cache.commit(torch.tensor([1]))
        self.assertEqual(cache.get_seq_length(), 2)

    def test_last_available_multilayer_multitoken_oracle(self):
        """Expected KV is built from the policy equation, independently of cache code."""
        layers, steps, batch = 2, 4, 2
        cache = LastAvailableCache(layers, steps)
        prompts = {}
        for depth in range(steps):
            for layer in range(layers):
                values = [[1000 * depth + 100 * layer + 10 * row + p
                           for p in range(2)] for row in range(batch)]
                prompts[depth, layer] = values
                key = torch.tensor(values, dtype=torch.float32).reshape(batch, 1, 2, 1)
                cache.update(key, -key, depth * layers + layer)
        history = []
        for token, exits in enumerate(([1, 4], [3, 2], [2, 1], [4, 3], [1, 2])):
            cache.begin_decode(batch)
            for depth in range(steps):
                for layer in range(layers):
                    current = [10000 + 1000 * token + 100 * depth + 10 * layer + row
                               for row in range(batch)]
                    key, value = cache.update(scalar_kv(current), scalar_kv([-x for x in current]),
                                              depth * layers + layer)
                    expected = []
                    for row in range(batch):
                        past = [10000 + 1000 * old_token + 100 * min(depth, old_exits[row] - 1)
                                + 10 * layer + row for old_token, old_exits in enumerate(history)]
                        expected.append(prompts[depth, layer][row] + past + [current[row]])
                    self.assertEqual(key[:, 0, :, 0].tolist(), expected)
                    self.assertEqual(value[:, 0, :, 0].tolist(), [[-x for x in row] for row in expected])
            cache.commit(torch.tensor(exits))
            history.append(exits)
            self.assertEqual(cache.get_seq_length(), 2 + len(history))
            retained = sum(slot.batch_indices.numel() for slot in cache._committed[-1].slots
                           if slot is not None)
            self.assertEqual(retained, layers * sum(exits))

    def test_missing_selected_depth_cannot_be_committed(self):
        cache = self.cache()
        cache.begin_decode(1)
        cache.update(scalar_kv([100.0]), scalar_kv([-100.0]), 0)
        with self.assertRaisesRegex(RuntimeError, "Missing staged KV"):
            cache.commit(torch.tensor([2]))
        self.assertEqual(cache.get_seq_length(), 1)
        cache.commit(torch.tensor([1]))
        self.assertEqual(cache.get_seq_length(), 2)

    def test_invalid_committed_depth_can_be_retried(self):
        cache = self.cache()
        cache.begin_decode(1)
        cache.update(scalar_kv([100.0]), scalar_kv([-100.0]), 0)
        for depth in (0, 4):
            with self.assertRaisesRegex(ValueError, "UT range"):
                cache.commit(torch.tensor([depth]))
        cache.commit(torch.tensor([1]))

    def test_commit_owns_exit_depth_metadata(self):
        cache = self.cache()
        cache.begin_decode(1)
        for depth in range(2):
            cache.update(scalar_kv([100.0 + depth]), scalar_kv([-100.0 - depth]), depth)
        selected = torch.tensor([2])
        cache.commit(selected)
        selected.fill_(3)
        cache.begin_decode(1)
        key, _ = cache.update(scalar_kv([200.0]), scalar_kv([-200.0]), 2)
        self.assertEqual(key.flatten().tolist(), [30.0, 101.0, 200.0])
        cache.abort_decode()

    def test_incomplete_prefill_is_rejected_before_transaction(self):
        cache = LastAvailableCache(1, 3)
        cache.update(scalar_kv([10.0]), scalar_kv([-10.0]), 0)
        with self.assertRaisesRegex(RuntimeError, "prefill"):
            cache.begin_decode(1)
        self.assertFalse(cache.in_decode)

    def test_request_count_cannot_change_without_reorder(self):
        cache = self.cache()
        with self.assertRaisesRegex(ValueError, "batch size"):
            cache.begin_decode(2)
        self.assertFalse(cache.in_decode)

    def test_clear_and_reuse_has_no_historical_state(self):
        cache = self.cache()
        cache.begin_decode(1)
        cache.update(scalar_kv([100.0]), scalar_kv([-100.0]), 0)
        cache.commit(torch.tensor([1]))
        cache.clear()
        self.assertEqual(cache.get_seq_length(), 0)
        self.assertFalse(cache.in_decode)
        for depth in range(3):
            cache.update(scalar_kv([20.0 + depth]), scalar_kv([-20.0 - depth]), depth)
        cache.begin_decode(1)
        key, _ = cache.update(scalar_kv([200.0]), scalar_kv([-200.0]), 2)
        self.assertEqual(key.flatten().tolist(), [22.0, 200.0])
        cache.abort_decode()


class CoreModelContractTest(unittest.TestCase):
    def test_threshold_boundaries_and_actual_decoder_work(self):
        for physical in (False, True):
            model = tiny_model("last_available", physical)
            with torch.inference_mode():
                model.model.early_exit_gate.weight.zero_()
                model.model.early_exit_gate.bias.zero_()
                for threshold, selected in ((0.0, 1), (0.5, 1), (0.75, 2), (1.0, 3)):
                    prefill = model(torch.tensor([[1, 4, 5]]), use_cache=True)
                    calls = []

                    def record(_module, _args, _output):
                        calls.append(1)

                    handles = [layer.register_forward_hook(record) for layer in model.model.layers]
                    try:
                        out = model(torch.tensor([[6]]), past_key_values=prefill.past_key_values,
                                    use_cache=True, exit_threshold=threshold)
                    finally:
                        for handle in handles:
                            handle.remove()
                    executed = selected if physical else 3
                    self.assertEqual(out.selected_ut_steps.tolist(), [selected])
                    self.assertEqual(out.executed_ut_steps.tolist(), [executed])
                    self.assertEqual(len(calls), executed * model.config.num_hidden_layers)

    def test_closed_loop_greedy_matches_run_all(self):
        torch.manual_seed(47)
        reference = tiny_model("last_available", False)
        physical = tiny_model("last_available", True, reference.state_dict())
        with torch.inference_mode():
            for threshold in (0.0, 0.7, 1.0):
                kwargs = dict(max_new_tokens=5, do_sample=False, eos_token_id=None,
                              pad_token_id=0, exit_threshold=threshold)
                expected = reference.generate(torch.tensor([[1, 4, 5]]), **kwargs)
                actual = physical.generate(torch.tensor([[1, 4, 5]]), **kwargs)
                self.assertTrue(torch.equal(actual, expected))

    def test_invalid_exit_does_not_leave_active_transaction(self):
        model = tiny_model("last_available", True)
        with torch.inference_mode():
            cache = model(torch.tensor([[1, 4, 5]]), use_cache=True).past_key_values
            for kwargs in ({"exit_at_step": 3}, {"exit_threshold": -0.1}):
                with self.assertRaises(ValueError):
                    model(torch.tensor([[6]]), past_key_values=cache, **kwargs)
                self.assertFalse(cache.in_decode)
            model(torch.tensor([[6]]), past_key_values=cache, exit_at_step=0)
        self.assertEqual(cache.get_seq_length(), 4)

    def test_adaptive_policy_requires_cache(self):
        model = tiny_model("last_available", True)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "use_cache"):
            model(torch.tensor([[1, 4, 5]]), use_cache=False)

    def test_threshold_one_executes_final_depth_even_if_gate_saturates(self):
        model = tiny_model("last_available", True)
        with torch.inference_mode():
            model.model.early_exit_gate.weight.zero_()
            model.model.early_exit_gate.bias.fill_(100.0)
            out = model(torch.tensor([[1, 4, 5]]), use_cache=True, exit_threshold=1.0)
            forced = model(torch.tensor([[1, 4, 5]]), use_cache=True, exit_at_step=2)
            torch.testing.assert_close(out.logits, forced.logits, rtol=0, atol=0)
            out = model(torch.tensor([[6]]), past_key_values=out.past_key_values,
                        use_cache=True, exit_threshold=1.0)
            self.assertEqual(out.selected_ut_steps.tolist(), [3])
            self.assertEqual(out.executed_ut_steps.tolist(), [3])

    def test_cached_threshold_one_matches_full_prefix_recomputation(self):
        torch.manual_seed(41)
        original = tiny_model("full_per_depth", False)
        physical = tiny_model("last_available", True, original.state_dict())
        prefix = torch.tensor([[1, 4, 5]])
        with torch.inference_mode():
            out = physical(prefix, use_cache=True, exit_threshold=1.0)
            for token in (6, 7, 8):
                prefix = torch.cat((prefix, torch.tensor([[token]])), dim=1)
                out = physical(torch.tensor([[token]]), past_key_values=out.past_key_values,
                               use_cache=True, exit_threshold=1.0)
                expected = original(prefix, use_cache=False, exit_threshold=1.0)
                torch.testing.assert_close(out.logits[:, -1], expected.logits[:, -1],
                                           rtol=1e-5, atol=1e-6)

    def test_exception_aborts_current_token_and_allows_exact_retry(self):
        torch.manual_seed(43)
        model = tiny_model("last_available", True)
        reference = tiny_model("last_available", True, model.state_dict())
        with torch.inference_mode():
            cache = model(torch.tensor([[1, 4, 5]]), use_cache=True).past_key_values
            ref = reference(torch.tensor([[1, 4, 5]]), use_cache=True)

            def fail(_module, _args, _output):
                raise RuntimeError("injected gate failure")

            handle = model.model.early_exit_gate.register_forward_hook(fail)
            try:
                with self.assertRaisesRegex(RuntimeError, "injected gate failure"):
                    model(torch.tensor([[6]]), past_key_values=cache, exit_at_step=1)
            finally:
                handle.remove()
            self.assertFalse(cache.in_decode)
            self.assertEqual(cache.get_seq_length(), 3)
            actual = model(torch.tensor([[6]]), past_key_values=cache, exit_at_step=1)
            expected = reference(torch.tensor([[6]]), past_key_values=ref.past_key_values,
                                 exit_at_step=1)
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)

    def test_fixed_rows_with_different_exits_match_run_all(self):
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


if __name__ == "__main__":
    unittest.main()
