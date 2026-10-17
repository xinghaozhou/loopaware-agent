"""Version 1 in-place KV caches for Original Ouro and adaptive Pred90.

This module deliberately subclasses the unmodified reference cache types so the
existing Ouro model can exercise the optimized storage without changes outside
this experiment directory.
"""

from __future__ import annotations

from typing import Optional

import torch

from vendor.ouro_hf.cache_ouro import (
    AdaptiveDecodeCache,
    FULL_PER_DEPTH,
    LAST_AVAILABLE,
    FullPerDepthCache,
    LastAvailableCache,
    _validate_kv_shapes,
    profile_range,
)


def _capacity(required: int, minimum: int = 256) -> int:
    value = max(required, minimum)
    return 1 << (value - 1).bit_length()


class InPlaceFullPerDepthCache(FullPerDepthCache):
    """Original Ouro semantics with geometrically allocated sequence storage."""

    def __init__(self, max_cache_size: Optional[int] = None):
        super().__init__(max_cache_size)
        self._key_buffers: list[Optional[torch.Tensor]] = []
        self._value_buffers: list[Optional[torch.Tensor]] = []
        self._lengths: list[int] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        _validate_kv_shapes(key_states, value_states)
        if layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
        if self.max_cache_size is not None and layer_idx >= self.max_cache_size:
            raise IndexError("cache layer exceeds max_cache_size")
        while len(self._key_buffers) <= layer_idx:
            self._key_buffers.append(None)
            self._value_buffers.append(None)
            self._lengths.append(0)
            self.key_cache.append(None)
            self.value_cache.append(None)

        old_length = self._lengths[layer_idx]
        required = old_length + key_states.shape[2]
        with profile_range("kv_prepare"):
            key_buffer = self._key_buffers[layer_idx]
            value_buffer = self._value_buffers[layer_idx]
            if key_buffer is None:
                capacity = _capacity(required)
                shape = list(key_states.shape)
                shape[2] = capacity
                key_buffer = torch.empty(
                    shape, dtype=key_states.dtype, device=key_states.device
                )
                value_buffer = torch.empty_like(key_buffer)
            elif required > key_buffer.shape[2]:
                capacity = _capacity(required, key_buffer.shape[2] * 2)
                shape = list(key_buffer.shape)
                shape[2] = capacity
                grown_key = torch.empty(
                    shape, dtype=key_buffer.dtype, device=key_buffer.device
                )
                grown_value = torch.empty_like(grown_key)
                grown_key[:, :, :old_length].copy_(key_buffer[:, :, :old_length])
                grown_value[:, :, :old_length].copy_(value_buffer[:, :, :old_length])
                key_buffer, value_buffer = grown_key, grown_value
            key_buffer[:, :, old_length:required].copy_(key_states)
            value_buffer[:, :, old_length:required].copy_(value_states)

        self._key_buffers[layer_idx] = key_buffer
        self._value_buffers[layer_idx] = value_buffer
        self._lengths[layer_idx] = required
        self.key_cache[layer_idx] = key_buffer[:, :, :required]
        self.value_cache[layer_idx] = value_buffer[:, :, :required]
        self._seen_tokens = required
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def clear(self) -> None:
        super().clear()
        self._key_buffers = []
        self._value_buffers = []
        self._lengths = []


class RequestIndexedLastAvailableCache(LastAvailableCache):
    """Last-Available semantics decoupled from transient microbatch layout."""

    def __init__(self, num_hidden_layers: int, total_ut_steps: int):
        super().__init__(num_hidden_layers, total_ut_steps)
        self._key_buffers: list[Optional[torch.Tensor]] = [None] * self.max_cache_size
        self._value_buffers: list[Optional[torch.Tensor]] = [None] * self.max_cache_size
        self._capacity = 0
        self._decoded_tokens = 0

    def _ensure_buffers(self) -> None:
        if self._key_buffers[0] is not None:
            return
        if self._prompt_batch_size is None or self._prompt_length == 0:
            raise RuntimeError("request-indexed storage requires completed prefill")
        self._capacity = _capacity(self._prompt_length + 1)
        for index, prompt_key in enumerate(self.prompt_key_cache):
            prompt_value = self.prompt_value_cache[index]
            if prompt_key is None or prompt_value is None:
                raise RuntimeError("every prompt KV slot must be populated")
            shape = list(prompt_key.shape)
            shape[2] = self._capacity
            key_buffer = torch.empty(
                shape, dtype=prompt_key.dtype, device=prompt_key.device
            )
            value_buffer = torch.empty_like(key_buffer)
            key_buffer[:, :, : self._prompt_length].copy_(prompt_key)
            value_buffer[:, :, : self._prompt_length].copy_(prompt_value)
            self._key_buffers[index] = key_buffer
            self._value_buffers[index] = value_buffer

    def _grow(self, required: int) -> None:
        if required <= self._capacity:
            return
        new_capacity = _capacity(required, self._capacity * 2)
        used = self._prompt_length + self._decoded_tokens
        for index, key_buffer in enumerate(self._key_buffers):
            value_buffer = self._value_buffers[index]
            if key_buffer is None or value_buffer is None:
                raise RuntimeError("cannot grow an uninitialized KV slot")
            shape = list(key_buffer.shape)
            shape[2] = new_capacity
            grown_key = torch.empty(
                shape, dtype=key_buffer.dtype, device=key_buffer.device
            )
            grown_value = torch.empty_like(grown_key)
            grown_key[:, :, :used].copy_(key_buffer[:, :, :used])
            grown_value[:, :, :used].copy_(value_buffer[:, :, :used])
            self._key_buffers[index] = grown_key
            self._value_buffers[index] = grown_value
        self._capacity = new_capacity

    def begin_decode(self, batch_size: int) -> None:
        self._ensure_buffers()
        self._grow(self._prompt_length + self._decoded_tokens + 1)
        super().begin_decode(batch_size)

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
            raise ValueError("adaptive decode accepts one token")
        cache_kwargs = cache_kwargs or {}
        local_rows = cache_kwargs.get("batch_indices")
        if local_rows is None:
            local_rows = torch.arange(key_states.shape[0], device=key_states.device)
        local_rows = local_rows.to(device=key_states.device, dtype=torch.long)
        global_rows = (
            local_rows
            if self._decode_batch_indices is None
            else self._decode_batch_indices.to(key_states.device).index_select(
                0, local_rows
            )
        )
        key_buffer = self._key_buffers[layer_idx]
        value_buffer = self._value_buffers[layer_idx]
        if key_buffer is None or value_buffer is None:
            raise RuntimeError("request-indexed buffers are not initialized")
        position = self._prompt_length + self._decoded_tokens
        with profile_range("kv_prepare"):
            key_buffer[global_rows, :, position : position + 1, :] = key_states
            value_buffer[global_rows, :, position : position + 1, :] = value_states
            return (
                key_buffer.index_select(0, global_rows)[:, :, : position + 1, :],
                value_buffer.index_select(0, global_rows)[:, :, : position + 1, :],
            )

    def commit(self, selected_depth: torch.LongTensor) -> None:
        if self._decode_batch_size is None or self._prompt_batch_size is None:
            raise RuntimeError("no adaptive decode transaction to commit")
        selected_depth = selected_depth.detach().to(dtype=torch.long)
        global_rows = (
            torch.arange(selected_depth.shape[0], device=selected_depth.device)
            if self._decode_batch_indices is None
            else self._decode_batch_indices.to(selected_depth.device)
        )
        if self._pending is None:
            from vendor.ouro_hf.cache_ouro import _CommittedToken

            self._pending = _CommittedToken(
                selected_depth=torch.zeros(
                    self._prompt_batch_size,
                    dtype=torch.long,
                    device=selected_depth.device,
                ),
                slots=[None] * self.max_cache_size,
            )
        self._pending.selected_depth.index_copy_(0, global_rows, selected_depth)
        self._pending_request_count += selected_depth.shape[0]
        if self._pending_request_count == self._prompt_batch_size:
            position = self._prompt_length + self._decoded_tokens
            exits = self._pending.selected_depth - 1
            request_rows = torch.arange(self._prompt_batch_size, device=exits.device)
            with profile_range("kv_prepare"):
                for layer in range(self.num_hidden_layers):
                    depth_keys = torch.stack(
                        [
                            self._key_buffers[depth * self.num_hidden_layers + layer][
                                :, :, position, :
                            ]
                            for depth in range(self.total_ut_steps)
                        ],
                        dim=1,
                    )
                    depth_values = torch.stack(
                        [
                            self._value_buffers[depth * self.num_hidden_layers + layer][
                                :, :, position, :
                            ]
                            for depth in range(self.total_ut_steps)
                        ],
                        dim=1,
                    )
                    for target_depth in range(self.total_ut_steps):
                        sources = torch.minimum(
                            exits,
                            torch.full_like(exits, target_depth),
                        )
                        resolved_key = depth_keys[request_rows, sources]
                        resolved_value = depth_values[request_rows, sources]
                        slot = target_depth * self.num_hidden_layers + layer
                        self._key_buffers[slot][:, :, position, :].copy_(resolved_key)
                        self._value_buffers[slot][:, :, position, :].copy_(resolved_value)
            self._decoded_tokens += 1
            self._pending = None
            self._pending_request_count = 0
        elif self._pending_request_count > self._prompt_batch_size:
            raise RuntimeError("too many request rows committed")
        self._decode_batch_size = None
        self._decode_batch_indices = None
        self._staged = [None] * self.max_cache_size

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        del layer_idx
        return self._prompt_length + self._decoded_tokens

    def clear(self) -> None:
        super().clear()
        self._key_buffers = [None] * self.max_cache_size
        self._value_buffers = [None] * self.max_cache_size
        self._capacity = 0
        self._decoded_tokens = 0


def make_optimized_ut_cache(
    policy: str, num_hidden_layers: int, total_ut_steps: int
) -> FullPerDepthCache | AdaptiveDecodeCache:
    if policy == FULL_PER_DEPTH:
        return InPlaceFullPerDepthCache(num_hidden_layers * total_ut_steps)
    if policy == LAST_AVAILABLE:
        return RequestIndexedLastAvailableCache(num_hidden_layers, total_ut_steps)
    raise ValueError(f"optimized experiment does not support policy {policy!r}")
