"""KV-cache policies for Ouro's recurrent decoder.

The adaptive policies intentionally stage the KV for the token currently being
decoded.  That token attends with the KV from its current recurrent step, while
only already-committed historical tokens are resolved through the configured
policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers.cache_utils import Cache


FULL_PER_DEPTH = "full_per_depth"
LAST_AVAILABLE = "last_available"
FINAL_EXIT_SHARED = "final_exit_shared"
KV_POLICIES = (FULL_PER_DEPTH, LAST_AVAILABLE, FINAL_EXIT_SHARED)


class _CacheUtilities(Cache):
    """Small compatibility surface required by Hugging Face generation."""

    layers: list[Any]

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int = 0
    ) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        del new_seq_length
        return self.get_seq_length(layer_idx)

    @property
    def is_compileable(self) -> bool:
        return False


class FullPerDepthCache(_CacheUtilities):
    """Original Ouro cache: one dense history for every (UT depth, layer)."""

    policy = FULL_PER_DEPTH

    def __init__(self, max_cache_size: Optional[int] = None):
        self.key_cache: list[Optional[torch.Tensor]] = []
        self.value_cache: list[Optional[torch.Tensor]] = []
        self.layers = []
        self._seen_tokens = 0
        self.max_cache_size = max_cache_size

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        if self.max_cache_size is not None and layer_idx >= self.max_cache_size:
            raise IndexError(
                f"Cache index {layer_idx} exceeds max_cache_size="
                f"{self.max_cache_size}"
            )
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        cached_key = self.key_cache[layer_idx]
        cached_value = self.value_cache[layer_idx]
        if cached_key is None:
            result_key = key_states
            result_value = value_states
        else:
            if cached_value is None:
                raise RuntimeError("Key cache exists without a value cache")
            _validate_kv_shapes(cached_key, key_states)
            result_key = torch.cat((cached_key, key_states), dim=2)
            result_value = torch.cat((cached_value, value_states), dim=2)
        self.key_cache[layer_idx] = result_key
        self.value_cache[layer_idx] = result_value
        self._seen_tokens = result_key.shape[2]
        return result_key, result_value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        layer_idx = 0 if layer_idx is None else layer_idx
        if layer_idx < 0 or layer_idx >= len(self.key_cache):
            return 0
        entry = self.key_cache[layer_idx]
        return 0 if entry is None else entry.shape[2]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for index, key in enumerate(self.key_cache):
            if key is None:
                continue
            value = self.value_cache[index]
            if value is None:
                raise RuntimeError("Key cache exists without a value cache")
            indices = beam_idx.to(key.device)
            self.key_cache[index] = key.index_select(0, indices)
            self.value_cache[index] = value.index_select(0, indices)

    def clear(self) -> None:
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0


@dataclass
class _SparseKV:
    batch_indices: torch.LongTensor
    key: torch.Tensor
    value: torch.Tensor

    def row(self, batch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        matches = (self.batch_indices == batch_index).nonzero(as_tuple=False)
        if matches.numel() != 1:
            raise RuntimeError(
                f"Missing or duplicate staged KV for batch index {batch_index}"
            )
        row = int(matches.item())
        return self.key[row : row + 1], self.value[row : row + 1]


@dataclass
class _CommittedToken:
    selected_depth: torch.LongTensor
    slots: list[Optional[_SparseKV]]


class AdaptiveDecodeCache(_CacheUtilities, ABC):
    """Base storage for decode-only adaptive KV policies.

    Prefill is always executed and stored at every depth. During one-token
    decode, calls to :meth:`update` stage the current token. :meth:`commit`
    makes exactly the selected-depth histories visible to later tokens.
    """

    def __init__(self, num_hidden_layers: int, total_ut_steps: int):
        if num_hidden_layers <= 0 or total_ut_steps <= 0:
            raise ValueError("Layer and UT counts must be positive")
        self.num_hidden_layers = num_hidden_layers
        self.total_ut_steps = total_ut_steps
        self.max_cache_size = num_hidden_layers * total_ut_steps
        self.layers: list[Any] = []
        self.prompt_key_cache: list[Optional[torch.Tensor]] = [
            None
        ] * self.max_cache_size
        self.prompt_value_cache: list[Optional[torch.Tensor]] = [
            None
        ] * self.max_cache_size
        self._prompt_length = 0
        self._committed: list[_CommittedToken] = []
        self._staged: list[Optional[_SparseKV]] = [None] * self.max_cache_size
        self._decode_batch_size: Optional[int] = None

    def begin_decode(self, batch_size: int) -> None:
        if self._decode_batch_size is not None:
            raise RuntimeError("A decode transaction is already active")
        if any(slot is not None for slot in self._staged):
            raise RuntimeError("Uncommitted KV exists before decode")
        if self._prompt_length == 0:
            raise RuntimeError("Adaptive decode requires a completed prefill")
        self._prepare_prompt_for_decode()
        self._decode_batch_size = batch_size

    @property
    def in_decode(self) -> bool:
        return self._decode_batch_size is not None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_slot(layer_idx)
        _validate_kv_shapes(key_states, value_states)
        if not self.in_decode:
            return self._update_prefill(key_states, value_states, layer_idx)
        if key_states.shape[2] != 1:
            raise ValueError("Adaptive decode accepts exactly one current token")

        cache_kwargs = cache_kwargs or {}
        batch_indices = cache_kwargs.get("batch_indices")
        if batch_indices is None:
            batch_indices = torch.arange(key_states.shape[0], device=key_states.device)
        batch_indices = batch_indices.to(device=key_states.device, dtype=torch.long)
        if batch_indices.numel() != key_states.shape[0]:
            raise ValueError("batch_indices must identify every incoming KV row")
        self._staged[layer_idx] = _SparseKV(
            batch_indices=batch_indices.detach().clone(),
            key=key_states,
            value=value_states,
        )

        depth, physical_layer = divmod(layer_idx, self.num_hidden_layers)
        historical_key, historical_value = self._resolve_history(
            depth, physical_layer, batch_indices
        )
        return (
            torch.cat((historical_key, key_states), dim=2),
            torch.cat((historical_value, value_states), dim=2),
        )

    def _update_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._committed:
            raise RuntimeError("Multi-token continuation after decode is unsupported")
        old_key = self.prompt_key_cache[layer_idx]
        old_value = self.prompt_value_cache[layer_idx]
        if old_key is None:
            new_key, new_value = key_states, value_states
        else:
            if old_value is None:
                raise RuntimeError("Prompt key cache exists without values")
            _validate_kv_shapes(old_key, key_states)
            new_key = torch.cat((old_key, key_states), dim=2)
            new_value = torch.cat((old_value, value_states), dim=2)
        self.prompt_key_cache[layer_idx] = new_key
        self.prompt_value_cache[layer_idx] = new_value
        self._prompt_length = new_key.shape[2]
        return new_key, new_value

    def commit(self, selected_depth: torch.LongTensor) -> None:
        if self._decode_batch_size is None:
            raise RuntimeError("No adaptive decode transaction to commit")
        selected_depth = selected_depth.detach().to(dtype=torch.long)
        if selected_depth.ndim != 1 or selected_depth.numel() != self._decode_batch_size:
            raise ValueError("selected_depth must contain one value per batch row")
        if bool(((selected_depth < 1) | (selected_depth > self.total_ut_steps)).any()):
            raise ValueError("selected_depth is outside the configured UT range")

        for batch_index, depth in enumerate(selected_depth.tolist()):
            for current_depth in range(depth):
                for layer in range(self.num_hidden_layers):
                    slot = self._staged[current_depth * self.num_hidden_layers + layer]
                    if slot is None:
                        raise RuntimeError(
                            f"Missing staged KV at depth={current_depth + 1}, "
                            f"layer={layer}"
                        )
                    slot.row(batch_index)
        retained_slots: list[Optional[_SparseKV]] = []
        for flat_index, slot in enumerate(self._staged):
            if slot is None:
                retained_slots.append(None)
                continue
            depth = flat_index // self.num_hidden_layers
            keep_rows = [
                row
                for row, batch_index in enumerate(slot.batch_indices.tolist())
                if self._retain_token_depth(
                    depth, int(selected_depth[batch_index]) - 1
                )
            ]
            if not keep_rows:
                retained_slots.append(None)
                continue
            row_indices = torch.tensor(keep_rows, device=slot.key.device)
            retained_slots.append(
                _SparseKV(
                    batch_indices=slot.batch_indices.index_select(
                        0, row_indices.to(slot.batch_indices.device)
                    ),
                    key=slot.key.index_select(0, row_indices),
                    value=slot.value.index_select(0, row_indices),
                )
            )
        self._committed.append(
            _CommittedToken(selected_depth=selected_depth, slots=retained_slots)
        )
        self._staged = [None] * self.max_cache_size
        self._decode_batch_size = None

    def abort_decode(self) -> None:
        self._staged = [None] * self.max_cache_size
        self._decode_batch_size = None

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        del layer_idx
        return self._prompt_length + len(self._committed)

    def _resolve_history(
        self,
        current_depth: int,
        physical_layer: int,
        batch_indices: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_depth = self._prompt_source_depth(current_depth)
        prompt_slot = prompt_depth * self.num_hidden_layers + physical_layer
        prompt_key = self.prompt_key_cache[prompt_slot]
        prompt_value = self.prompt_value_cache[prompt_slot]
        if prompt_key is None or prompt_value is None:
            raise RuntimeError(
                f"Missing prompt KV at depth={prompt_depth + 1}, "
                f"layer={physical_layer}"
            )
        prompt_rows = batch_indices.to(prompt_key.device)
        key_parts = [prompt_key.index_select(0, prompt_rows)]
        value_parts = [prompt_value.index_select(0, prompt_rows)]

        for token in self._committed:
            token_keys = []
            token_values = []
            for batch_index in batch_indices.tolist():
                source_depth = self._token_source_depth(
                    current_depth, int(token.selected_depth[batch_index]) - 1
                )
                slot = token.slots[
                    source_depth * self.num_hidden_layers + physical_layer
                ]
                if slot is None:
                    raise RuntimeError("Policy selected an unavailable historical KV")
                key_row, value_row = slot.row(batch_index)
                token_keys.append(key_row)
                token_values.append(value_row)
            key_parts.append(torch.cat(token_keys, dim=0))
            value_parts.append(torch.cat(token_values, dim=0))
        return torch.cat(key_parts, dim=2), torch.cat(value_parts, dim=2)

    @abstractmethod
    def _prompt_source_depth(self, current_depth: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def _token_source_depth(self, current_depth: int, exit_depth: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def _retain_token_depth(self, stored_depth: int, exit_depth: int) -> bool:
        raise NotImplementedError

    def _prepare_prompt_for_decode(self) -> None:
        """Policy hook invoked before each one-token decode transaction."""

    def _validate_slot(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.max_cache_size:
            raise IndexError(
                f"Cache index {layer_idx} is outside [0, {self.max_cache_size})"
            )

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.in_decode:
            raise RuntimeError("Cannot reorder an active decode transaction")
        for index, key in enumerate(self.prompt_key_cache):
            if key is None:
                continue
            value = self.prompt_value_cache[index]
            if value is None:
                raise RuntimeError("Prompt key cache exists without values")
            indices = beam_idx.to(key.device)
            self.prompt_key_cache[index] = key.index_select(0, indices)
            self.prompt_value_cache[index] = value.index_select(0, indices)
        for token in self._committed:
            old_depth = token.selected_depth
            token.selected_depth = old_depth.index_select(0, beam_idx.to(old_depth.device))
            for index, slot in enumerate(token.slots):
                if slot is None:
                    continue
                keys, values, new_indices = [], [], []
                for new_index, old_index in enumerate(beam_idx.tolist()):
                    matches = (slot.batch_indices == old_index).nonzero(as_tuple=False)
                    if matches.numel() == 1:
                        row = int(matches.item())
                        keys.append(slot.key[row : row + 1])
                        values.append(slot.value[row : row + 1])
                        new_indices.append(new_index)
                token.slots[index] = (
                    None
                    if not keys
                    else _SparseKV(
                        batch_indices=torch.tensor(
                            new_indices, device=slot.batch_indices.device
                        ),
                        key=torch.cat(keys, dim=0),
                        value=torch.cat(values, dim=0),
                    )
                )

    def clear(self) -> None:
        self.prompt_key_cache = [None] * self.max_cache_size
        self.prompt_value_cache = [None] * self.max_cache_size
        self._prompt_length = 0
        self._committed = []
        self.abort_decode()


class LastAvailableCache(AdaptiveDecodeCache):
    """Use historical KV from ``min(current_depth, token_exit_depth)``."""

    policy = LAST_AVAILABLE

    def _prompt_source_depth(self, current_depth: int) -> int:
        return current_depth

    def _token_source_depth(self, current_depth: int, exit_depth: int) -> int:
        return min(current_depth, exit_depth)

    def _retain_token_depth(self, stored_depth: int, exit_depth: int) -> bool:
        return stored_depth <= exit_depth


class FinalExitSharedCache(AdaptiveDecodeCache):
    """Use every historical token's final/exit-depth KV at all future depths."""

    policy = FINAL_EXIT_SHARED

    def _prompt_source_depth(self, current_depth: int) -> int:
        del current_depth
        return self.total_ut_steps - 1

    def _token_source_depth(self, current_depth: int, exit_depth: int) -> int:
        del current_depth
        return exit_depth

    def _retain_token_depth(self, stored_depth: int, exit_depth: int) -> bool:
        return stored_depth == exit_depth

    def _prepare_prompt_for_decode(self) -> None:
        # Prompt tokens are defined to exit at U_max, so only that depth can be
        # observed by future decode recurrences under this policy.
        for depth in range(self.total_ut_steps - 1):
            for layer in range(self.num_hidden_layers):
                index = depth * self.num_hidden_layers + layer
                self.prompt_key_cache[index] = None
                self.prompt_value_cache[index] = None


def make_ut_cache(
    policy: str, num_hidden_layers: int, total_ut_steps: int
) -> _CacheUtilities:
    if policy == FULL_PER_DEPTH:
        return FullPerDepthCache(num_hidden_layers * total_ut_steps)
    if policy == LAST_AVAILABLE:
        return LastAvailableCache(num_hidden_layers, total_ut_steps)
    if policy == FINAL_EXIT_SHARED:
        return FinalExitSharedCache(num_hidden_layers, total_ut_steps)
    raise ValueError(f"Unknown Ouro KV policy {policy!r}; expected one of {KV_POLICIES}")


def _validate_kv_shapes(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.ndim != 4 or right.ndim != 4:
        raise ValueError("KV tensors must have [batch, heads, sequence, head_dim]")
    for dimension in (0, 1, 3):
        if left.shape[dimension] != right.shape[dimension]:
            raise ValueError("KV tensors must match on batch, heads, and head_dim")


# Backward-compatible name used by existing callers.
UniversalTransformerCache = FullPerDepthCache


__all__ = [
    "AdaptiveDecodeCache",
    "FinalExitSharedCache",
    "FullPerDepthCache",
    "LastAvailableCache",
    "UniversalTransformerCache",
    "KV_POLICIES",
    "make_ut_cache",
]
