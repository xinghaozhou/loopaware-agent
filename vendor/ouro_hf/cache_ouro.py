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
        self._pending: Optional[_CommittedToken] = None
        self._pending_request_count = 0
        self._history_key_cache: list[Optional[torch.Tensor]] = [
            None
        ] * self.max_cache_size
        self._history_value_cache: list[Optional[torch.Tensor]] = [
            None
        ] * self.max_cache_size
        self._staged: list[Optional[_SparseKV]] = [None] * self.max_cache_size
        self._decode_batch_size: Optional[int] = None
        self._prompt_batch_size: Optional[int] = None
        self._next_decode_batch_indices: Optional[torch.LongTensor] = None
        self._decode_batch_indices: Optional[torch.LongTensor] = None

    def set_decode_batch_indices(
        self, batch_indices: torch.LongTensor, *, validate: bool = True
    ) -> None:
        """Identify global request rows for the next decode microbatch.

        Multiple disjoint microbatches may contribute one logical token before
        it becomes visible in history. This enables token-level regrouping
        without copying the accumulated KV cache between physical batches.
        """

        if self.in_decode or self._next_decode_batch_indices is not None:
            raise RuntimeError("Decode batch indices are already configured")
        if self._prompt_batch_size is None:
            raise RuntimeError("Scheduled decode requires a completed prefill")
        indices = batch_indices.detach().to(dtype=torch.long)
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("batch_indices must be a non-empty vector")
        if validate:
            if bool(((indices < 0) | (indices >= self._prompt_batch_size)).any()):
                raise ValueError("batch_indices are outside the prefill batch")
            if indices.unique().numel() != indices.numel():
                raise ValueError("batch_indices must be unique")
        self._next_decode_batch_indices = indices

    def begin_decode(self, batch_size: int) -> None:
        if self._decode_batch_size is not None:
            raise RuntimeError("A decode transaction is already active")
        if any(slot is not None for slot in self._staged):
            raise RuntimeError("Uncommitted KV exists before decode")
        if self._prompt_length == 0:
            raise RuntimeError("Adaptive decode requires a completed prefill")
        self._prepare_prompt_for_decode()
        self._decode_batch_size = batch_size
        if self._next_decode_batch_indices is not None:
            if self._next_decode_batch_indices.numel() != batch_size:
                raise ValueError("Configured batch indices do not match decode batch size")
            self._decode_batch_indices = self._next_decode_batch_indices
            self._next_decode_batch_indices = None

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
        if self._decode_batch_indices is not None:
            batch_indices = self._decode_batch_indices.to(key_states.device).index_select(
                0, batch_indices
            )
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
        if self._prompt_batch_size is None:
            self._prompt_batch_size = new_key.shape[0]
        elif self._prompt_batch_size != new_key.shape[0]:
            raise ValueError("Prefill batch size changed across cache layers")
        return new_key, new_value

    def commit(self, selected_depth: torch.LongTensor) -> None:
        if self._decode_batch_size is None:
            raise RuntimeError("No adaptive decode transaction to commit")
        selected_depth = selected_depth.detach().to(dtype=torch.long)
        if selected_depth.ndim != 1 or selected_depth.numel() != self._decode_batch_size:
            raise ValueError("selected_depth must contain one value per batch row")
        if self._prompt_batch_size is None:
            raise RuntimeError("Decode transaction has no prefill batch")
        global_indices = (
            torch.arange(selected_depth.shape[0], device=selected_depth.device)
            if self._decode_batch_indices is None
            else self._decode_batch_indices.to(selected_depth.device)
        )
        selected_global = torch.zeros(
            self._prompt_batch_size,
            dtype=torch.long,
            device=selected_depth.device,
        )
        selected_global.index_copy_(0, global_indices, selected_depth)

        retained_slots: list[Optional[_SparseKV]] = []
        for flat_index, slot in enumerate(self._staged):
            if slot is None:
                retained_slots.append(None)
                continue
            depth = flat_index // self.num_hidden_layers
            exit_depth = selected_global.index_select(
                0, slot.batch_indices.to(selected_depth.device)
            ) - 1
            keep = depth <= exit_depth if self.policy == LAST_AVAILABLE else depth == exit_depth
            retained_slots.append(
                _SparseKV(
                    batch_indices=slot.batch_indices[keep],
                    key=slot.key[keep],
                    value=slot.value[keep],
                )
            )
        if self._pending is None:
            self._pending = _CommittedToken(
                selected_depth=torch.zeros(
                    self._prompt_batch_size,
                    dtype=torch.long,
                    device=selected_depth.device,
                ),
                slots=[None] * self.max_cache_size,
            )
        self._pending.selected_depth.index_copy_(0, global_indices, selected_depth)
        self._pending_request_count += selected_depth.shape[0]
        for index, slot in enumerate(retained_slots):
            if slot is None:
                continue
            pending_slot = self._pending.slots[index]
            if pending_slot is None:
                self._pending.slots[index] = slot
            else:
                self._pending.slots[index] = _SparseKV(
                    batch_indices=torch.cat(
                        (pending_slot.batch_indices, slot.batch_indices)
                    ),
                    key=torch.cat((pending_slot.key, slot.key), dim=0),
                    value=torch.cat((pending_slot.value, slot.value), dim=0),
                )
        if self._pending_request_count == self._prompt_batch_size:
            self._append_resolved_token(self._pending)
            self._committed.append(self._pending)
            self._pending = None
            self._pending_request_count = 0
        elif self._pending_request_count > self._prompt_batch_size:
            raise RuntimeError("Too many request rows committed at one token position")
        self._staged = [None] * self.max_cache_size
        self._decode_batch_size = None
        self._decode_batch_indices = None

    def abort_decode(self) -> None:
        self._staged = [None] * self.max_cache_size
        self._decode_batch_size = None
        self._decode_batch_indices = None
        self._next_decode_batch_indices = None
        self._pending = None
        self._pending_request_count = 0

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
        prompt_key = prompt_key.index_select(0, prompt_rows)
        prompt_value = prompt_value.index_select(0, prompt_rows)
        flat_index = current_depth * self.num_hidden_layers + physical_layer
        history_key = self._history_key_cache[flat_index]
        history_value = self._history_value_cache[flat_index]
        if history_key is None or history_value is None:
            return prompt_key, prompt_value
        return (
            torch.cat((prompt_key, history_key.index_select(0, prompt_rows)), dim=2),
            torch.cat((prompt_value, history_value.index_select(0, prompt_rows)), dim=2),
        )

    def _append_resolved_token(self, token: _CommittedToken) -> None:
        if self._prompt_batch_size is None:
            raise RuntimeError("Resolved token requires a prefill batch")
        for current_depth in range(self.total_ut_steps):
            for layer in range(self.num_hidden_layers):
                flat_index = current_depth * self.num_hidden_layers + layer
                base_slot = token.slots[layer]
                if base_slot is None:
                    raise RuntimeError("Every request must execute the first UT depth")
                key = torch.zeros(
                    (self._prompt_batch_size, *base_slot.key.shape[1:]),
                    dtype=base_slot.key.dtype,
                    device=base_slot.key.device,
                )
                value = torch.zeros_like(key)
                if self.policy == LAST_AVAILABLE:
                    source_depths = range(current_depth + 1)
                else:
                    source_depths = range(self.total_ut_steps)
                for source_depth in source_depths:
                    slot = token.slots[source_depth * self.num_hidden_layers + layer]
                    if slot is None:
                        continue
                    rows = slot.batch_indices.to(key.device)
                    key.index_copy_(0, rows, slot.key)
                    value.index_copy_(0, rows, slot.value)
                old_key = self._history_key_cache[flat_index]
                old_value = self._history_value_cache[flat_index]
                self._history_key_cache[flat_index] = (
                    key if old_key is None else torch.cat((old_key, key), dim=2)
                )
                self._history_value_cache[flat_index] = (
                    value if old_value is None else torch.cat((old_value, value), dim=2)
                )

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
        for index, key in enumerate(self._history_key_cache):
            if key is None:
                continue
            value = self._history_value_cache[index]
            indices = beam_idx.to(key.device)
            self._history_key_cache[index] = key.index_select(0, indices)
            self._history_value_cache[index] = value.index_select(0, indices)

    def clear(self) -> None:
        self.prompt_key_cache = [None] * self.max_cache_size
        self.prompt_value_cache = [None] * self.max_cache_size
        self._prompt_length = 0
        self._prompt_batch_size = None
        self._committed = []
        self._pending = None
        self._pending_request_count = 0
        self._history_key_cache = [None] * self.max_cache_size
        self._history_value_cache = [None] * self.max_cache_size
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


def merge_adaptive_prefill_caches(
    caches: list[AdaptiveDecodeCache],
) -> AdaptiveDecodeCache:
    """Combine equal-width prefill caches for dynamically scheduled decode."""

    if not caches:
        raise ValueError("At least one prefill cache is required")
    first = caches[0]
    if first.get_seq_length() == 0:
        raise ValueError("Cannot merge an empty prefill cache")
    combined = type(first)(first.num_hidden_layers, first.total_ut_steps)
    for cache in caches:
        if type(cache) is not type(first):
            raise TypeError("All prefill caches must use the same policy")
        if (
            cache.num_hidden_layers != first.num_hidden_layers
            or cache.total_ut_steps != first.total_ut_steps
            or cache.get_seq_length() != first.get_seq_length()
        ):
            raise ValueError("Prefill cache configurations do not match")
        if cache._committed or cache._pending or cache.in_decode:
            raise ValueError("Only untouched prefill caches can be merged")
    for index in range(first.max_cache_size):
        keys = [cache.prompt_key_cache[index] for cache in caches]
        values = [cache.prompt_value_cache[index] for cache in caches]
        if any(value is None for value in keys + values):
            raise ValueError("Every prefill cache layer must be populated")
        combined.prompt_key_cache[index] = torch.cat(keys, dim=0)
        combined.prompt_value_cache[index] = torch.cat(values, dim=0)
    combined._prompt_length = first.get_seq_length()
    combined._prompt_batch_size = sum(
        int(cache._prompt_batch_size or 0) for cache in caches
    )
    return combined


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
    "merge_adaptive_prefill_caches",
    "UniversalTransformerCache",
    "KV_POLICIES",
    "make_ut_cache",
]
