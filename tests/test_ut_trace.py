from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from loopaware_agent.ut_trace import OuroUTTracer, summarize_ut_trace


class FakeOuroBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.early_exit_gate = nn.Linear(1, 1)

    def forward(self, values: torch.Tensor, cache_position=None):
        batch, sequence = values.shape
        gates = [torch.zeros(batch, sequence, 1) for _ in range(3)]
        return SimpleNamespace(), [], gates


class FakePhysicalOuroBase(FakeOuroBase):
    total_ut_steps = 3

    def forward(self, values: torch.Tensor, cache_position=None):
        batch, sequence = values.shape
        gates = [
            torch.zeros(batch, sequence, 1),
            torch.zeros(batch, sequence, 1),
        ]
        metadata = SimpleNamespace(
            physical_early_exit=True,
            selected_ut_steps=torch.tensor([2]),
            executed_ut_steps=torch.tensor([2]),
        )
        return SimpleNamespace(), [], gates, metadata


class FakeOuroForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeOuroBase()
        self.config = SimpleNamespace(early_exit_threshold=0.7)
        self.early_exit_threshold = 0.7


class FakePhysicalOuroForCausalLM(FakeOuroForCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakePhysicalOuroBase()


class OuroUTTracerTest(unittest.TestCase):
    def test_traces_and_attaches_one_token_per_forward(self) -> None:
        model = FakeOuroForCausalLM()
        with OuroUTTracer(model) as tracer:
            model.model(torch.zeros(1, 4), cache_position=torch.arange(4))
            model.model(torch.zeros(1, 1), cache_position=torch.tensor([4]))

        events = tracer.attach_output_tokens([[11, 12]])
        self.assertEqual([event.token_id for event in events], [11, 12])
        self.assertEqual([event.input_position for event in events], [3, 4])
        self.assertEqual(events[0].gate_probabilities, [0.5, 0.5, 0.5])
        self.assertEqual(events[0].exit_probabilities, [0.5, 0.25, 0.25])
        self.assertEqual(events[0].selected_ut_step, 2)
        self.assertEqual(events[0].executed_ut_steps, 3)
        self.assertAlmostEqual(events[0].expected_ut_steps, 1.75)
        self.assertEqual(events[0].decode_ut_step, 3)

        summary = summarize_ut_trace(events)
        self.assertEqual(summary["mean_exit_probabilities"], [0.5, 0.25, 0.25])
        self.assertEqual(summary["selected_ut_step_histogram"], {"2": 2})

    def test_rejects_token_and_forward_count_mismatch(self) -> None:
        model = FakeOuroForCausalLM()
        with OuroUTTracer(model) as tracer:
            model.model(torch.zeros(1, 1))
        with self.assertRaisesRegex(ValueError, "one Ouro forward pass"):
            tracer.attach_output_tokens([[1, 2]])

    def test_uses_authoritative_physical_exit_metadata(self) -> None:
        model = FakePhysicalOuroForCausalLM()
        with OuroUTTracer(model) as tracer:
            model.model(torch.zeros(1, 1), cache_position=torch.tensor([4]))
        event = tracer.attach_output_tokens([[11]])[0]
        self.assertEqual(event.selected_ut_step, 2)
        self.assertEqual(event.executed_ut_steps, 2)
        self.assertEqual(event.decode_ut_step, 2)
        self.assertEqual(event.gate_probabilities, [0.5, 0.5])
        self.assertEqual(event.exit_probabilities, [0.5, 0.25])
        self.assertIsNone(event.expected_ut_steps)


if __name__ == "__main__":
    unittest.main()
