"""Generate pinned GSM8K examples under Ouro adaptive KV policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loopaware_agent.ut_trace import OuroUTTracer  # noqa: E402
from vendor.ouro_hf.configuration_ouro import OuroConfig  # noqa: E402
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM  # noqa: E402


MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--total-ut-steps", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def extract_gsm8k_answer(text: str) -> str | None:
    values = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not values:
        return None
    value = values[-1].replace(",", "")
    return value[:-2] if value.endswith(".0") else value


def ground_truth(example: dict) -> str:
    return example["answer"].rsplit("####", 1)[-1].strip().replace(",", "")


def set_mode(model, *, policy: str, physical: bool, threshold: float, force_final: bool):
    model.config.kv_cache_policy = policy
    model.model.config.kv_cache_policy = policy
    model.config.physical_early_exit = physical
    model.model.config.physical_early_exit = physical
    model.config.early_exit_threshold = threshold
    model.model.config.early_exit_threshold = threshold
    model.early_exit_threshold = threshold
    model.early_exit_step = model.model.total_ut_steps - 1 if force_final else None


def generate(model, tokenizer, prompt: str, *, name: str, policy: str, physical: bool, threshold: float, force_final: bool, max_new_tokens: int):
    set_mode(
        model,
        policy=policy,
        physical=physical,
        threshold=threshold,
        force_final=force_final,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode(), OuroUTTracer(model, exit_threshold=threshold) as tracer:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    events = tracer.attach_output_tokens(generated, tokenizer)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return {
        "condition": name,
        "kv_policy": policy,
        "physical_early_exit": physical,
        "force_final_ut": force_final,
        "text": text,
        "extracted_answer": extract_gsm8k_answer(text),
        "generated_token_count": generated.shape[1],
        "selected_ut_steps": [event.selected_ut_step for event in events],
        "executed_ut_steps": [event.executed_ut_steps for event in events],
        "trace": [asdict(event) for event in events],
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(20261010)
    torch.cuda.manual_seed_all(20261010)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        revision=GSM8K_REVISION,
    )
    config = OuroConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    config.total_ut_steps = args.total_ut_steps
    config.kv_cache_policy = "full_per_depth"
    config.physical_early_exit = False
    model = OuroForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    results = []
    for index in args.indices:
        example = dataset[index]
        truth = ground_truth(example)
        conditions = []
        for name, policy, physical, force_final in (
            ("original_full_ut", "full_per_depth", False, True),
            ("last_available", "last_available", True, False),
            ("final_exit_shared", "final_exit_shared", True, False),
        ):
            result = generate(
                model,
                tokenizer,
                example["question"],
                name=name,
                policy=policy,
                physical=physical,
                threshold=args.threshold,
                force_final=force_final,
                max_new_tokens=args.max_new_tokens,
            )
            result["correct"] = result["extracted_answer"] == truth
            conditions.append(result)
            print(
                f"index={index} condition={name} answer={result['extracted_answer']!r} "
                f"truth={truth!r} correct={result['correct']}",
                flush=True,
            )
        results.append(
            {
                "dataset_index": index,
                "question": example["question"],
                "ground_truth_reasoning": example["answer"],
                "ground_truth_answer": truth,
                "conditions": conditions,
            }
        )
    payload = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": "openai/gsm8k",
        "dataset_revision": GSM8K_REVISION,
        "threshold": args.threshold,
        "total_ut_steps": args.total_ut_steps,
        "runtime_num_hidden_layers": model.config.num_hidden_layers,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
