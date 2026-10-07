"""Per-output-token tracing for Ouro's Universal Transformer (UT) steps.

Ouro evaluates every configured recurrent step and exposes a learned halting gate.
This module can counterfactually select a step from those gates while decoding
continues to use the fixed final step. ``selected_ut_step`` is therefore the
adaptive-depth decision being mimicked, while ``executed_ut_steps`` describes the
compute actually performed by the reference HF model.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class TokenUTTrace:
    """Actual and counterfactual UT measurements for one output token."""

    output_index: int
    batch_index: int
    input_position: int | None
    token_id: int | None
    token: str | None
    gate_probabilities: list[float]
    exit_probabilities: list[float]
    expected_ut_steps: float
    selected_ut_step: int
    decode_ut_step: int
    executed_ut_steps: int
    exit_threshold: float


def _exit_distribution(gate_logits: torch.Tensor) -> torch.Tensor:
    """Convert Ouro's conditional halt logits to an exit-step probability mass."""

    gate_probabilities = torch.sigmoid(gate_logits.float())
    remaining = torch.ones_like(gate_probabilities[..., 0])
    probabilities = []
    for step in range(gate_probabilities.shape[-1]):
        if step == gate_probabilities.shape[-1] - 1:
            probabilities.append(remaining)
        else:
            probability = gate_probabilities[..., step] * remaining
            probabilities.append(probability)
            remaining = remaining * (1.0 - gate_probabilities[..., step])
    return torch.stack(probabilities, dim=-1)


class OuroUTTracer(AbstractContextManager["OuroUTTracer"]):
    """Collect counterfactual Ouro UT decisions during ``model.generate``.

    The tracer installs a forward hook on ``OuroForCausalLM.model``.  One event is
    captured per batch element and generation forward pass.  Only the final input
    position is recorded because that position supplies the next-token logits in
    autoregressive generation. Installing the tracer never changes which hidden
    state the model uses for decoding.
    """

    def __init__(self, model: Any, *, exit_threshold: float | None = None):
        base_model = getattr(model, "model", None)
        if base_model is None or not hasattr(base_model, "early_exit_gate"):
            raise TypeError("Expected an OuroForCausalLM-compatible model")

        configured_threshold = getattr(model, "early_exit_threshold", None)
        if configured_threshold is None:
            configured_threshold = getattr(model.config, "early_exit_threshold", 1.0)
        self.exit_threshold = float(
            configured_threshold if exit_threshold is None else exit_threshold
        )
        if not 0.0 <= self.exit_threshold <= 1.0:
            raise ValueError("exit_threshold must be between 0 and 1")

        self.model = model
        self.events: list[TokenUTTrace] = []
        self._handle: Any = None
        self._next_output_index: dict[int, int] = {}

    def __enter__(self) -> "OuroUTTracer":
        if self._handle is not None:
            raise RuntimeError("This tracer is already active")
        self._handle = self.model.model.register_forward_hook(
            self._capture_forward, with_kwargs=True
        )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _capture_forward(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        del module, args
        if not isinstance(output, tuple) or len(output) < 3:
            raise RuntimeError("Ouro base model did not return its UT gate tensors")
        gate_list = output[2]
        if not gate_list:
            raise RuntimeError("Ouro base model returned no UT gate tensors")

        # [batch, sequence, steps], reduced to the position predicting the next token.
        gate_logits = torch.stack(
            [gate.detach().squeeze(-1) for gate in gate_list], dim=-1
        )[:, -1, :]
        gate_probabilities = torch.sigmoid(gate_logits.float())
        exit_probabilities = _exit_distribution(gate_logits)
        cumulative = exit_probabilities.cumsum(dim=-1)
        crossed = cumulative >= self.exit_threshold
        selected = crossed.to(torch.int64).argmax(dim=-1)
        selected = torch.where(
            crossed.any(dim=-1),
            selected,
            torch.full_like(selected, exit_probabilities.shape[-1] - 1),
        )
        step_numbers = torch.arange(
            1,
            exit_probabilities.shape[-1] + 1,
            device=exit_probabilities.device,
            dtype=exit_probabilities.dtype,
        )
        expected = (exit_probabilities * step_numbers).sum(dim=-1)

        positions = _last_cache_positions(
            kwargs.get("cache_position"), gate_logits.shape[0]
        )
        for batch_index in range(gate_logits.shape[0]):
            output_index = self._next_output_index.get(batch_index, 0)
            self._next_output_index[batch_index] = output_index + 1
            self.events.append(
                TokenUTTrace(
                    output_index=output_index,
                    batch_index=batch_index,
                    input_position=positions[batch_index],
                    token_id=None,
                    token=None,
                    gate_probabilities=gate_probabilities[batch_index]
                    .cpu()
                    .tolist(),
                    exit_probabilities=exit_probabilities[batch_index]
                    .cpu()
                    .tolist(),
                    expected_ut_steps=float(expected[batch_index].cpu()),
                    selected_ut_step=int(selected[batch_index].cpu()) + 1,
                    decode_ut_step=exit_probabilities.shape[-1],
                    executed_ut_steps=exit_probabilities.shape[-1],
                    exit_threshold=self.exit_threshold,
                )
            )

    def attach_output_tokens(
        self,
        generated_token_ids: Sequence[Sequence[int]] | torch.Tensor,
        tokenizer: Any | None = None,
    ) -> list[TokenUTTrace]:
        """Attach sampled token IDs to their corresponding prediction events."""

        if isinstance(generated_token_ids, torch.Tensor):
            token_rows = generated_token_ids.detach().cpu().tolist()
        else:
            token_rows = [list(row) for row in generated_token_ids]

        events_by_batch: dict[int, list[TokenUTTrace]] = {}
        for event in self.events:
            events_by_batch.setdefault(event.batch_index, []).append(event)
        if set(events_by_batch) != set(range(len(token_rows))):
            raise ValueError("Trace batch size does not match generated token batches")

        attached: list[TokenUTTrace] = []
        for batch_index, token_ids in enumerate(token_rows):
            batch_events = events_by_batch[batch_index]
            if len(batch_events) != len(token_ids):
                raise ValueError(
                    "Expected one Ouro forward pass per generated token, but got "
                    f"{len(batch_events)} trace events and {len(token_ids)} tokens "
                    f"for batch item {batch_index}. Beam search and assisted decoding "
                    "are not supported."
                )
            token_strings = (
                tokenizer.convert_ids_to_tokens(token_ids)
                if tokenizer is not None
                else [None] * len(token_ids)
            )
            for event, token_id, token_string in zip(
                batch_events, token_ids, token_strings, strict=True
            ):
                values = asdict(event)
                values["token_id"] = int(token_id)
                values["token"] = token_string
                attached.append(TokenUTTrace(**values))
        self.events = sorted(
            attached, key=lambda event: (event.batch_index, event.output_index)
        )
        return self.events


def _last_cache_positions(
    cache_position: torch.Tensor | None, batch_size: int
) -> list[int | None]:
    if cache_position is None:
        return [None] * batch_size
    values = cache_position.detach().cpu()
    if values.ndim == 1:
        return [int(values[-1])] * batch_size
    if values.ndim == 2 and values.shape[0] == batch_size:
        return [int(values[index, -1]) for index in range(batch_size)]
    return [None] * batch_size


def summarize_ut_trace(events: Sequence[TokenUTTrace]) -> dict[str, Any]:
    """Build JSON-serializable selected/expected UT distributions."""

    if not events:
        return {
            "num_output_tokens": 0,
            "mean_expected_ut_steps": None,
            "mean_selected_ut_steps": None,
            "mean_exit_probabilities": [],
            "selected_ut_step_histogram": {},
        }
    expected = [event.expected_ut_steps for event in events]
    selected = [event.selected_ut_step for event in events]
    mean_exit_probabilities = [
        sum(event.exit_probabilities[step] for event in events) / len(events)
        for step in range(events[0].executed_ut_steps)
    ]
    histogram = {
        str(step): selected.count(step) for step in sorted(set(selected))
    }
    return {
        "num_output_tokens": len(events),
        "mean_expected_ut_steps": sum(expected) / len(expected),
        "mean_selected_ut_steps": sum(selected) / len(selected),
        "mean_exit_probabilities": mean_exit_probabilities,
        "selected_ut_step_histogram": histogram,
    }


def write_trajectory_jsonl(
    path: str | Path,
    *,
    trajectory: dict[str, Any],
    events: Sequence[TokenUTTrace],
) -> None:
    """Append one self-contained trajectory and its token-level UT trace."""

    record = {
        **trajectory,
        "ut_summary": summarize_ut_trace(events),
        "token_traces": [asdict(event) for event in events],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False)
        stream.write("\n")
