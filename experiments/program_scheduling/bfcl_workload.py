"""Build deterministic teacher-forced BFCL V3 multi-turn model contexts."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BFCL_ROOT = PROJECT_ROOT / "vendor/bfcl/berkeley-function-call-leaderboard"
DATA_ROOT = BFCL_ROOT / "bfcl_eval/data"
sys.path.insert(0, str(BFCL_ROOT))

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: E402
    execute_multi_turn_func_call,
)


CLASS_DOC = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_base_programs(count: int, seed: int) -> list[dict[str, Any]]:
    rows = read_jsonl(DATA_ROOT / "BFCL_v3_multi_turn_base.json")
    answers = {
        row["id"]: row["ground_truth"]
        for row in read_jsonl(DATA_ROOT / "possible_answer/BFCL_v3_multi_turn_base.json")
    }
    chosen = random.Random(seed).sample(rows, count)
    for row in chosen:
        row["ground_truth"] = answers[row["id"]]
    return chosen


def _tool_documents(entry: dict[str, Any]) -> list[dict[str, Any]]:
    used_names = {item.rsplit(".", 1)[-1] for item in entry["path"]}
    documents: list[dict[str, Any]] = []
    for class_name in entry["involved_classes"]:
        path = DATA_ROOT / "multi_turn_func_doc" / CLASS_DOC[class_name]
        documents.extend(
            document for document in read_jsonl(path) if document["name"] in used_names
        )
    return documents


def build_contexts(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one model query per BFCL user turn and execute recorded tools."""

    tools = _tool_documents(entry)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a tool-using assistant. Reason briefly, then choose the "
                "appropriate tools. Available tool specifications:\n"
                + json.dumps(tools, ensure_ascii=False, sort_keys=True)
            ),
        }
    ]
    contexts = []
    for turn_id, (user_messages, calls) in enumerate(
        zip(entry["question"], entry["ground_truth"], strict=True)
    ):
        messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in user_messages
        )
        contexts.append(
            {
                "program_id": entry["id"],
                "turn_id": turn_id,
                "messages": [dict(message) for message in messages],
                "ground_truth_calls": calls,
            }
        )
        results, _ = execute_multi_turn_func_call(
            calls,
            entry["initial_config"],
            entry["involved_classes"],
            "ouro_fixed_trace",
            entry["id"],
            long_context=False,
            is_evaL_run=False,
        )
        messages.append(
            {
                "role": "assistant",
                "content": "<tool_calls>\n" + json.dumps(calls) + "\n</tool_calls>",
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(results, ensure_ascii=False),
            }
        )
    return contexts
