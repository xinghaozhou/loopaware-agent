"""Inspect exact Ouro context lengths for candidate math and web trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def normalized_message(message: dict[str, Any]) -> dict[str, str]:
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls")
    if tool_calls:
        content += "\n<tool_calls>\n" + json.dumps(tool_calls, sort_keys=True) + "\n</tool_calls>"
    return {"role": message["role"], "content": content}


def normalize_web(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Merge EntiWeave's thought + tool_call records into one assistant turn."""

    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if role == "tool_call":
            if not normalized or normalized[-1]["role"] != "assistant":
                raise ValueError("tool_call does not follow an assistant thought")
            normalized[-1]["content"] += f"\n<tool_call>\n{content}\n</tool_call>"
        else:
            normalized.append({"role": role, "content": content})
    return normalized


def request_lengths(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    prefix: list[dict[str, str]] = []
    lengths = []
    for message in messages:
        if message["role"] == "assistant":
            token_ids = tokenizer.apply_chat_template(
                prefix, tokenize=True, add_generation_prompt=True
            )
            lengths.append(len(token_ids))
        prefix.append(message)
    return lengths


def iter_math(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            messages = [normalized_message(message) for message in row["conversations"]]
            yield {
                "id": row["sample_id"],
                "domain": "math",
                "category": row["category"],
                "messages": messages,
            }


def iter_web(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        table = pq.read_table(path, columns=["id", "trajectory", "version"])
        for row in table.to_pylist():
            yield {
                "id": row["id"],
                "domain": "web_search",
                "category": f"version-{row['version']}",
                "messages": normalize_web(json.loads(row["trajectory"])),
            }


def inspect(
    tokenizer: Any,
    records: Iterable[dict[str, Any]],
    minimum_turns: int,
    maximum_tokens: int,
) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        assistant_turns = sum(
            message["role"] == "assistant" for message in record["messages"]
        )
        if assistant_turns < minimum_turns:
            continue
        # This is a shortlist optimization, not the final safety check. English
        # trajectories far above this bound cannot plausibly fit the target
        # token window; retained records are still tokenized exactly below.
        total_characters = sum(len(message["content"]) for message in record["messages"])
        if total_characters > maximum_tokens * 6:
            continue
        lengths = request_lengths(tokenizer, record["messages"])
        if lengths and max(lengths) <= maximum_tokens:
            candidates.append(
                {
                    "id": record["id"],
                    "domain": record["domain"],
                    "category": record["category"],
                    "assistant_turns": assistant_turns,
                    "input_tokens_by_turn": lengths,
                    "max_input_tokens": max(lengths),
                }
            )
    return sorted(candidates, key=lambda row: (row["max_input_tokens"], -row["assistant_turns"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--math-file", required=True, type=Path)
    parser.add_argument("--web-files", required=True, type=Path, nargs="+")
    parser.add_argument("--max-input-tokens", default=8200, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    results = {
        "math": inspect(
            tokenizer, iter_math(args.math_file), 4, args.max_input_tokens
        ),
        "web_search": inspect(
            tokenizer, iter_web(args.web_files), 5, args.max_input_tokens
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for domain, rows in results.items():
        print(f"{domain}: {len(rows)} memory-safe candidates")
        for row in rows[:20]:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
