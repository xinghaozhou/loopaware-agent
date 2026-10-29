"""Fast CPU smoke test for Ouro adaptive-exit cache integration."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vendor.ouro_hf.cache_ouro import AdaptiveDecodeCache  # noqa: E402
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


def make_model(policy: str) -> OuroForCausalLM:
    torch.manual_seed(7)
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
        physical_early_exit=True,
        early_exit_threshold=0.7,
    )
    return OuroForCausalLM(config).eval()


def smoke_policy(policy: str) -> None:
    model = make_model(policy)
    with torch.inference_mode():
        prefill = model(torch.tensor([[1, 4, 5]]), use_cache=True, exit_at_step=2)
        cache = prefill.past_key_values
        assert isinstance(cache, AdaptiveDecodeCache)
        assert cache.get_seq_length() == 3

        depth_two = model(
            torch.tensor([[6]]),
            past_key_values=cache,
            use_cache=True,
            exit_at_step=1,
        )
        assert depth_two.selected_ut_steps.tolist() == [2]
        assert depth_two.executed_ut_steps.tolist() == [2]
        assert cache.get_seq_length() == 4

        depth_three = model(
            torch.tensor([[7]]),
            past_key_values=cache,
            use_cache=True,
            exit_at_step=2,
        )
        assert depth_three.selected_ut_steps.tolist() == [3]
        assert depth_three.executed_ut_steps.tolist() == [3]
        assert cache.get_seq_length() == 5
        assert torch.isfinite(depth_three.logits).all()


def main() -> None:
    for policy in ("last_available", "final_exit_shared"):
        smoke_policy(policy)
        print(f"PASS {policy}")


if __name__ == "__main__":
    main()
