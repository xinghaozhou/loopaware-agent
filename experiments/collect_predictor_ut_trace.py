"""Collect a small full-compute UT=4 trace for predictor headroom analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ordinary_batching_numerics import (  # noqa: E402
    GSM8K_ID,
    GSM8K_REVISION,
    MATH500_ID,
    MATH500_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    generate_one,
    load_requests,
    set_determinism,
)
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--examples-per-dataset", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--total-ut-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def trace_path(output_dir: Path, request_id: str) -> Path:
    return output_dir / "requests" / f"{request_id.replace(':', '-')}.jsonl"


def main() -> None:
    args = parse_args()
    if args.examples_per_dataset != 8:
        raise ValueError("The controlled pilot requires 8 examples per dataset")
    if args.total_ut_steps != 4:
        raise ValueError("The controlled pilot is fixed at UT=4")
    if args.max_new_tokens not in (64, 128):
        raise ValueError("The controlled pilot supports 64 or 128 output positions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(args.seed)

    model_source = args.model_path or args.model
    tokenizer_source = args.tokenizer_path or args.model
    revision = None if args.model_path else args.model_revision
    tokenizer_revision = None if args.tokenizer_path else args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=tokenizer_revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    requests = load_requests(tokenizer, args.examples_per_dataset)

    config = OuroConfig.from_pretrained(model_source, revision=revision)
    config.total_ut_steps = args.total_ut_steps
    config.kv_cache_policy = "full_per_depth"
    config.physical_early_exit = False
    config.early_exit_threshold = None
    model = OuroForCausalLM.from_pretrained(
        model_source,
        revision=revision,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    runtime_layers = model.config.num_hidden_layers
    if not isinstance(runtime_layers, int) or runtime_layers <= 0:
        raise ValueError(f"Invalid runtime num_hidden_layers={runtime_layers!r}")

    manifest = {
        "experiment": "controlled_accuracy_ut_predictor_trace",
        "created_at": utc_now(),
        "model": args.model,
        "model_revision": args.model_revision,
        "runtime_num_hidden_layers": runtime_layers,
        "total_ut_steps": args.total_ut_steps,
        "execution": "full-compute fixed-final-UT",
        "kv_policy": "full_per_depth",
        "physical_adaptive_exit": False,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "datasets": {
            "gsm8k": {
                "id": GSM8K_ID,
                "revision": GSM8K_REVISION,
                "split": "test",
                "indices": list(range(args.examples_per_dataset)),
            },
            "math500": {
                "id": MATH500_ID,
                "revision": MATH500_REVISION,
                "split": "test",
                "indices": list(range(args.examples_per_dataset)),
            },
        },
        "request_order": [request.request_id for request in requests],
        "requests": [
            {
                "request_id": request.request_id,
                "dataset": request.dataset,
                "prompt_sha256": hashlib.sha256(
                    request.prompt.encode("utf-8")
                ).hexdigest(),
            }
            for request in requests
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": str(next(model.parameters()).dtype),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and args.resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key != "created_at"}
        requested = {key: value for key, value in manifest.items() if key != "created_at"}
        if comparable != requested:
            raise RuntimeError("Existing trace manifest does not match requested run")
    else:
        atomic_json(manifest_path, manifest)

    counts = {}
    for index, request in enumerate(requests):
        output_path = trace_path(args.output_dir, request.request_id)
        if output_path.exists():
            if not args.resume:
                raise FileExistsError(f"{output_path} exists; pass --resume")
            counts[request.request_id] = sum(
                1 for line in output_path.read_text(encoding="utf-8").splitlines() if line
            )
            continue
        trajectory = generate_one(
            model,
            tokenizer,
            request,
            max_new_tokens=args.max_new_tokens,
            policy="full_per_depth",
            physical=False,
            threshold=None,
        )
        records = []
        for token_index, (token_id, event) in enumerate(
            zip(trajectory.token_ids, trajectory.events, strict=True)
        ):
            cumulative = []
            running = 0.0
            for probability in event.exit_probabilities:
                running += probability
                cumulative.append(running)
            records.append(
                {
                    "request_id": request.request_id,
                    "dataset": request.dataset,
                    "token_index": token_index,
                    "token_id": token_id,
                    "token": tokenizer.decode([token_id]),
                    "num_hidden_layers": runtime_layers,
                    "executed_ut_steps": args.total_ut_steps,
                    "exit_probability_by_ut": event.exit_probabilities,
                    "cumulative_exit_probability_by_ut": cumulative,
                }
            )
        atomic_jsonl(output_path, records)
        counts[request.request_id] = len(records)
        del trajectory
        print(
            f"TRACE {index + 1}/{len(requests)} {request.request_id} "
            f"tokens={counts[request.request_id]}",
            flush=True,
        )

    common_positions = min(counts.values())
    completion = {
        "completed_at": utc_now(),
        "request_count": len(requests),
        "token_count": sum(counts.values()),
        "tokens_by_request": counts,
        "common_frontier_positions": common_positions,
        "scheduling_instances_of_16": common_positions,
    }
    atomic_json(args.output_dir / "completion.json", completion)
    print(json.dumps(completion, indent=2), flush=True)
    if common_positions < 50:
        raise RuntimeError(
            f"Only {common_positions} complete 16-request frontiers; need at least 50"
        )


if __name__ == "__main__":
    main()
