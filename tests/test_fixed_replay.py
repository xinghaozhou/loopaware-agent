from __future__ import annotations

import unittest
from pathlib import Path

from experiments.trajectory.run_fixed_replay import (
    normalized_message,
    percentile,
    request_trace_path,
    normalize_web_messages,
)


class FixedReplayTest(unittest.TestCase):
    def test_normalized_message_preserves_native_tool_call(self) -> None:
        message = {
            "role": "assistant",
            "content": "THOUGHT: inspect",
            "tool_calls": [
                {
                    "function": "bash",
                    "arguments": {"command": "ls"},
                }
            ],
        }
        normalized = normalized_message(message)
        self.assertEqual(normalized["role"], "assistant")
        self.assertIn("THOUGHT: inspect", normalized["content"])
        self.assertIn('"function": "bash"', normalized["content"])
        self.assertIn('"command": "ls"', normalized["content"])

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(percentile([1, 2, 3, 4], 0.95), 3.8499999999999996)
        self.assertIsNone(percentile([], 0.5))

    def test_request_trace_path_is_unique_per_request(self) -> None:
        path = request_trace_path(Path("run"), "ut6", "owner__repo-1", 12)
        self.assertEqual(
            path.as_posix(),
            "run/raw/requests/ut6/owner__repo-1/turn-0012.jsonl",
        )

    def test_web_thought_and_tool_call_form_one_assistant_turn(self) -> None:
        messages = normalize_web_messages(
            [
                {"role": "system", "content": "system"},
                {"role": "assistant", "content": "thought"},
                {"role": "tool_call", "content": '{"name":"search"}'},
                {"role": "tool", "content": "observation"},
            ]
        )
        self.assertEqual([message["role"] for message in messages], ["system", "assistant", "tool"])
        self.assertIn("thought", messages[1]["content"])
        self.assertIn('"name":"search"', messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
