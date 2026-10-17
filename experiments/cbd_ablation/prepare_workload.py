"""Convert the loopaware 16-request Ouro trace into a CBD workload bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from looped_cdb.benchmarks.workload import Workload, depths_from_pdf
from looped_cdb.models.ouro.configuration_ouro import OuroConfig

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    return parser.parse_args()


def load_prepared_arrow(cache_name: str, config_name: str, revision: str) -> Dataset | None:
    hf_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    pattern = f"{cache_name}/{config_name}/*/{revision}/*-test.arrow"
    candidates = sorted((hf_root / "datasets").glob(pattern))
    return Dataset.from_file(str(candidates[-1])) if candidates else None


def render_prompt(tokenizer: object, question: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(  # type: ignore[attr-defined, no-any-return]
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {question}\nAssistant:"


def load_prompts(tokenizer: object) -> dict[str, str]:
    gsm = load_prepared_arrow("openai___gsm8k", "main", GSM8K_REVISION) or load_dataset(
        "openai/gsm8k", "main", split="test", revision=GSM8K_REVISION
    )
    math500 = load_prepared_arrow("HuggingFaceH4___math-500", "default", MATH500_REVISION) or load_dataset(
        "HuggingFaceH4/MATH-500", split="test", revision=MATH500_REVISION
    )
    prompts: dict[str, str] = {}
    for index in range(8):
        prompts[f"gsm8k:{index}"] = render_prompt(tokenizer, gsm[index]["question"])
        prompts[f"math500:{index}"] = render_prompt(tokenizer, math500[index]["problem"])
    return prompts


def trace_path(trace_dir: Path, request_id: str) -> Path:
    return trace_dir / "requests" / f"{request_id.replace(':', '-')}.jsonl"


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.trace_dir / "manifest.json").read_text())

    # This is deliberately loaded from the checkpoint, not copied from the trace manifest.
    config = OuroConfig.from_pretrained(args.checkpoint)
    runtime_num_hidden_layers = int(config.num_hidden_layers)
    recorded_num_hidden_layers = int(manifest["runtime_num_hidden_layers"])
    if runtime_num_hidden_layers != recorded_num_hidden_layers:
        raise ValueError(
            "checkpoint/trace layer mismatch: "
            f"config.num_hidden_layers={runtime_num_hidden_layers}, trace={recorded_num_hidden_layers}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    prompts = load_prompts(tokenizer)
    manifest_requests = {row["request_id"]: row for row in manifest["requests"]}

    ids: list[str] = []
    input_lens: list[int] = []
    output_lens: list[int] = []
    pdf_rows: list[list[float]] = []
    for request_id in manifest["request_order"]:
        prompt = prompts[request_id]
        actual_hash = hashlib.sha256(prompt.encode()).hexdigest()
        expected_hash = manifest_requests[request_id]["prompt_sha256"]
        if actual_hash != expected_hash:
            raise ValueError(f"prompt hash mismatch for {request_id}: {actual_hash} != {expected_hash}")

        rows = [json.loads(line) for line in trace_path(args.trace_dir, request_id).read_text().splitlines()]
        if [row["token_index"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"non-contiguous token indices for {request_id}")
        for row in rows:
            if int(row["num_hidden_layers"]) != runtime_num_hidden_layers:
                raise ValueError(f"runtime layer mismatch in {request_id} token {row['token_index']}")
            pdf_rows.append(row["exit_probability_by_ut"])

        ids.append(request_id)
        input_lens.append(len(tokenizer(prompt, add_special_tokens=True)["input_ids"]))
        output_lens.append(len(rows))

    pdf = np.asarray(pdf_rows, dtype=np.float32)
    total_ut_steps = int(manifest["total_ut_steps"])
    if pdf.shape[1] != total_ut_steps:
        raise ValueError(f"trace has {pdf.shape[1]} UT columns, expected {total_ut_steps}")
    exit_depths = depths_from_pdf(pdf, threshold=args.threshold).astype(np.int32)
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    np.cumsum(np.asarray(output_lens, dtype=np.int64), out=offsets[1:])

    workload = Workload(
        ids=ids,
        input_lens=np.asarray(input_lens, dtype=np.int32),
        output_lens=np.asarray(output_lens, dtype=np.int32),
        offsets=offsets,
        exit_depths=exit_depths,
        max_depth=total_ut_steps,
        meta={
            "dataset": "loopaware_16_selected_ut_q0p2",
            "source_trace": str(args.trace_dir.resolve()),
            "source_model": manifest["model"],
            "source_model_revision": manifest["model_revision"],
            "threshold": args.threshold,
            "num_requests": len(ids),
            "recur_steps": total_ut_steps,
            "runtime_num_hidden_layers": runtime_num_hidden_layers,
            "depth_defaults": {"min_exit_step": 1, "exit_delay_steps": 0},
            "note": "CBD replays prompt lengths with seeded token IDs; original prompt content is hash-verified but not replayed.",
        },
    )
    json_path, npz_path = workload.save(args.output)

    counts = np.bincount(exit_depths, minlength=total_ut_steps + 1)[1:]
    effective = exit_depths.copy()
    effective[offsets[:-1]] = total_ut_steps  # First output token is produced by prefill at full depth.
    effective_counts = np.bincount(effective, minlength=total_ut_steps + 1)[1:]
    print(f"config.num_hidden_layers={runtime_num_hidden_layers}")
    print(f"requests={len(ids)} output_tokens={len(exit_depths)}")
    print(f"input_lens={input_lens}")
    print(f"selected_ut_counts={counts.tolist()}")
    print(f"effective_ut_counts_first_token_full_depth={effective_counts.tolist()}")
    print(f"mean_selected_ut={float(exit_depths.mean()):.6f}")
    print(f"mean_effective_ut={float(effective.mean()):.6f}")
    print(f"workload_json={json_path}")
    print(f"workload_schedule={npz_path}")


if __name__ == "__main__":
    main()
