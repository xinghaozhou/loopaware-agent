from __future__ import annotations

import unittest

import torch

from vendor.ouro_hf.cache_ouro import (
    FinalExitSharedCache,
    LastAvailableCache,
)
from vendor.ouro_hf.configuration_ouro import OuroConfig
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM


def scalar_kv(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(len(values), 1, 1, 1)


class AdaptiveCachePolicyTest(unittest.TestCase):
    def _prefilled_cache(self, cache_type):
        cache = cache_type(num_hidden_layers=1, total_ut_steps=3)
        for depth, value in enumerate((10.0, 20.0, 30.0)):
            cache.update(scalar_kv([value]), scalar_kv([-value]), depth)
        return cache

    def _commit_depth_two(self, cache) -> None:
        cache.begin_decode(batch_size=1)
        cache.update(scalar_kv([101.0]), scalar_kv([-101.0]), 0)
        cache.update(scalar_kv([102.0]), scalar_kv([-102.0]), 1)
        cache.commit(torch.tensor([2]))

    def test_last_available_policy_oracle(self) -> None:
        cache = self._prefilled_cache(LastAvailableCache)
        self._commit_depth_two(cache)
        expected = ([10.0, 101.0], [20.0, 102.0], [30.0, 102.0])
        cache.begin_decode(batch_size=1)
        for depth, history in enumerate(expected):
            key, value = cache.update(
                scalar_kv([900.0 + depth]), scalar_kv([-900.0 - depth]), depth
            )
            self.assertEqual(key.flatten().tolist()[:-1], history)
            self.assertEqual(value.flatten().tolist()[:-1], [-x for x in history])
        cache.abort_decode()

    def test_final_exit_shared_policy_oracle(self) -> None:
        cache = self._prefilled_cache(FinalExitSharedCache)
        self._commit_depth_two(cache)
        cache.begin_decode(batch_size=1)
        for depth in range(3):
            key, value = cache.update(
                scalar_kv([900.0 + depth]), scalar_kv([-900.0 - depth]), depth
            )
            self.assertEqual(key.flatten().tolist()[:-1], [30.0, 102.0])
            self.assertEqual(value.flatten().tolist()[:-1], [-30.0, -102.0])
        cache.abort_decode()

        committed = cache._committed[0]
        self.assertEqual(
            [index for index, slot in enumerate(committed.slots) if slot is not None],
            [1],
        )

    def test_policies_support_different_exit_depths_in_one_batch(self) -> None:
        for cache_type in (LastAvailableCache, FinalExitSharedCache):
            with self.subTest(policy=cache_type.policy):
                cache = cache_type(num_hidden_layers=1, total_ut_steps=3)
                for depth, values in enumerate(
                    ([10.0, 11.0], [20.0, 21.0], [30.0, 31.0])
                ):
                    cache.update(scalar_kv(values), scalar_kv(values), depth)
                cache.begin_decode(batch_size=2)
                cache.update(
                    scalar_kv([100.0, 101.0]),
                    scalar_kv([100.0, 101.0]),
                    0,
                    {"batch_indices": torch.tensor([0, 1])},
                )
                cache.update(
                    scalar_kv([202.0]),
                    scalar_kv([202.0]),
                    1,
                    {"batch_indices": torch.tensor([1])},
                )
                cache.commit(torch.tensor([1, 2]))

                cache.begin_decode(batch_size=2)
                key, _ = cache.update(
                    scalar_kv([900.0, 901.0]),
                    scalar_kv([900.0, 901.0]),
                    0,
                    {"batch_indices": torch.tensor([0, 1])},
                )
                if cache_type is LastAvailableCache:
                    self.assertEqual(key[:, 0, :-1, 0].tolist(), [[10.0, 100.0], [11.0, 101.0]])
                else:
                    self.assertEqual(key[:, 0, :-1, 0].tolist(), [[30.0, 100.0], [31.0, 202.0]])
                cache.abort_decode()


def tiny_model(policy: str, physical: bool, state_dict=None) -> OuroForCausalLM:
    config = OuroConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        layer_types=["full_attention", "full_attention"],
        total_ut_steps=3,
        kv_cache_policy=policy,
        physical_early_exit=physical,
        early_exit_threshold=0.7,
    )
    model = OuroForCausalLM(config).eval()
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


class AdaptiveModelParityTest(unittest.TestCase):
    def test_full_per_depth_baseline_still_generates(self) -> None:
        torch.manual_seed(5)
        config = OuroConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            layer_types=["full_attention", "full_attention"],
            total_ut_steps=3,
            kv_cache_policy="full_per_depth",
            physical_early_exit=False,
        )
        model = OuroForCausalLM(config).eval()
        with torch.inference_mode():
            generated = model.generate(
                torch.tensor([[1, 4, 5]]),
                max_new_tokens=3,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )
        self.assertEqual(generated.shape, (1, 6))

    def test_physical_exit_matches_run_all_policy_reference(self) -> None:
        torch.manual_seed(11)
        for policy in ("last_available", "final_exit_shared"):
            with self.subTest(policy=policy), torch.inference_mode():
                reference = tiny_model(policy, physical=False)
                physical = tiny_model(policy, physical=True, state_dict=reference.state_dict())

                reference_output = reference(
                    torch.tensor([[1, 4, 5]]), use_cache=True, exit_at_step=2
                )
                physical_output = physical(
                    torch.tensor([[1, 4, 5]]), use_cache=True, exit_at_step=2
                )
                torch.testing.assert_close(
                    reference_output.logits, physical_output.logits, rtol=0, atol=0
                )

                reference_cache = reference_output.past_key_values
                physical_cache = physical_output.past_key_values
                for token, forced_step in ((6, 1), (7, 2), (8, 0), (9, 2)):
                    reference_output = reference(
                        torch.tensor([[token]]),
                        past_key_values=reference_cache,
                        use_cache=True,
                        exit_at_step=forced_step,
                    )
                    physical_output = physical(
                        torch.tensor([[token]]),
                        past_key_values=physical_cache,
                        use_cache=True,
                        exit_at_step=forced_step,
                    )
                    torch.testing.assert_close(
                        reference_output.logits,
                        physical_output.logits,
                        rtol=1e-5,
                        atol=1e-6,
                    )
                    expected_depth = forced_step + 1
                    self.assertEqual(
                        reference_output.selected_ut_steps.tolist(), [expected_depth]
                    )
                    self.assertEqual(
                        physical_output.selected_ut_steps.tolist(), [expected_depth]
                    )
                    self.assertEqual(
                        reference_output.executed_ut_steps.tolist(), [3]
                    )
                    self.assertEqual(
                        physical_output.executed_ut_steps.tolist(), [expected_depth]
                    )
                    self.assertEqual(
                        reference_cache.get_seq_length(), physical_cache.get_seq_length()
                    )

    def test_threshold_exit_matches_run_all_policy_reference(self) -> None:
        torch.manual_seed(13)
        for policy in ("last_available", "final_exit_shared"):
            with self.subTest(policy=policy), torch.inference_mode():
                reference = tiny_model(policy, physical=False)
                physical = tiny_model(policy, physical=True, state_dict=reference.state_dict())
                for model in (reference, physical):
                    model.model.early_exit_gate.weight.zero_()
                    model.model.early_exit_gate.bias.zero_()

                ref = reference(torch.tensor([[1, 4, 5]]), use_cache=True)
                phy = physical(torch.tensor([[1, 4, 5]]), use_cache=True)
                ref = reference(
                    torch.tensor([[6]]),
                    past_key_values=ref.past_key_values,
                    use_cache=True,
                )
                phy = physical(
                    torch.tensor([[6]]),
                    past_key_values=phy.past_key_values,
                    use_cache=True,
                )
                self.assertEqual(ref.selected_ut_steps.tolist(), [2])
                self.assertEqual(phy.selected_ut_steps.tolist(), [2])
                self.assertEqual(ref.executed_ut_steps.tolist(), [3])
                self.assertEqual(phy.executed_ut_steps.tolist(), [2])
                torch.testing.assert_close(ref.logits, phy.logits, rtol=1e-5, atol=1e-6)

    def test_generate_uses_adaptive_cache(self) -> None:
        torch.manual_seed(17)
        for policy in ("last_available", "final_exit_shared"):
            with self.subTest(policy=policy), torch.inference_mode():
                model = tiny_model(policy, physical=True)
                model.model.early_exit_gate.weight.zero_()
                model.model.early_exit_gate.bias.zero_()
                generated = model.generate(
                    torch.tensor([[1, 4, 5]]),
                    max_new_tokens=3,
                    do_sample=False,
                    eos_token_id=None,
                    pad_token_id=0,
                )
                self.assertEqual(generated.shape, (1, 6))

    def test_batched_threshold_exit_matches_run_all_reference(self) -> None:
        torch.manual_seed(19)
        for policy in ("last_available", "final_exit_shared"):
            with self.subTest(policy=policy), torch.inference_mode():
                reference = tiny_model(policy, physical=False)
                physical = tiny_model(policy, physical=True, state_dict=reference.state_dict())
                prompt = torch.tensor([[1, 4, 5], [1, 7, 8]])
                ref = reference(prompt, use_cache=True)
                phy = physical(prompt, use_cache=True)
                for tokens in (torch.tensor([[6], [9]]), torch.tensor([[10], [11]])):
                    ref = reference(
                        tokens, past_key_values=ref.past_key_values, use_cache=True
                    )
                    phy = physical(
                        tokens, past_key_values=phy.past_key_values, use_cache=True
                    )
                    self.assertEqual(
                        ref.selected_ut_steps.tolist(), phy.selected_ut_steps.tolist()
                    )
                    torch.testing.assert_close(
                        ref.logits, phy.logits, rtol=1e-5, atol=1e-6
                    )


if __name__ == "__main__":
    unittest.main()
