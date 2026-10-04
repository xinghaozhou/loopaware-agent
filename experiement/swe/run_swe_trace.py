import json
from pathlib import Path
import sys

import torch
import statistics
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vendor.ouro_hf.configuration_ouro import OuroConfig
from vendor.ouro_hf.modeling_ouro import OuroForCausalLM

TRACE_DIR = PROJECT_ROOT / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

TRACE_FILE = TRACE_DIR / "current_recurrence.jsonl"
OUTPUT_FILE = TRACE_DIR / "swe_recurrence_trace.jsonl"

NUM_INSTANCES = 10
MAX_NEW_TOKENS = 128


# ============================================================
# Load SWE-bench
# ============================================================

dataset = load_dataset(
    "princeton-nlp/SWE-bench_Verified",
    split="test",
).select(range(NUM_INSTANCES))


# ============================================================
# Load Ouro
# ============================================================

MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

config = OuroConfig.from_pretrained(
    MODEL_ID,
)

config.total_ut_steps = 8

model = OuroForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",

)


model.eval()


# ============================================================
# Helper
# ============================================================

def load_recurrence_trace():
    if not TRACE_FILE.exists():
        return []

    traces = []

    with TRACE_FILE.open("r") as f:
        for line in f:
            line = line.strip()

            if line:
                traces.append(json.loads(line))

    return traces


# ============================================================
# Clear previous experiment output
# ============================================================

OUTPUT_FILE.write_text("")


# ============================================================
# Run
# ============================================================

for idx, instance in enumerate(dataset):

    instance_id = instance["instance_id"]
    repo = instance["repo"]
    problem_statement = instance["problem_statement"]

    print("=" * 80)
    print(f"[{idx + 1}/{NUM_INSTANCES}] {instance_id}")
    print("=" * 80)

    # --------------------------------------------------------
    # Reset recurrence trace for this SWE-bench instance
    # --------------------------------------------------------

    TRACE_FILE.write_text("")

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = f"""You are a software engineer debugging a real GitHub issue.

Repository:
{repo}

Issue:
{problem_statement}

Analyze the issue carefully and explain how you would fix it.
"""

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    prompt_length = inputs["input_ids"].shape[1]

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    model.early_exit_threshold = 0.8
    model.config.early_exit_threshold = 0.8

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Only generated tokens
    generated_ids = output_ids[0, prompt_length:]

    generated_token_ids = generated_ids.tolist()

    output_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    generated_tokens = tokenizer.convert_ids_to_tokens(
        generated_token_ids
    )

    # --------------------------------------------------------
    # Load recurrence traces produced by modeling_ouro.py
    # --------------------------------------------------------

    recurrence_trace = load_recurrence_trace()

print(
    "Generated tokens:",
    len(generated_token_ids),
)

print(
    "Recurrence records:",
    len(recurrence_trace),
)

# --------------------------------------------------------
# Extract recurrence depths
# --------------------------------------------------------

selected_depths = [
    record["selected_recurrence"]
    for record in recurrence_trace
    if "selected_recurrence" in record
]

# --------------------------------------------------------
# Aggregate recurrence statistics
# --------------------------------------------------------

if selected_depths:
    mean_recurrence = statistics.mean(selected_depths)

    variance_recurrence = statistics.pvariance(
        selected_depths
    )

    std_recurrence = statistics.pstdev(
        selected_depths
    )

    median_recurrence = statistics.median(
        selected_depths
    )

    min_recurrence = min(selected_depths)
    max_recurrence = max(selected_depths)

    cv_recurrence = (
        std_recurrence / mean_recurrence
        if mean_recurrence != 0
        else None
    )

    histogram = dict(
        sorted(
            Counter(selected_depths).items()
        )
    )

else:
    mean_recurrence = None
    variance_recurrence = None
    std_recurrence = None
    median_recurrence = None
    min_recurrence = None
    max_recurrence = None
    cv_recurrence = None
    histogram = {}


# --------------------------------------------------------
# Final record for this SWE instance
# --------------------------------------------------------

record = {
    "instance_id": instance_id,

    "num_generated_tokens": len(
        generated_token_ids
    ),

    "num_trace_records": len(
        recurrence_trace
    ),

    "recurrence_summary": {
        "mean": mean_recurrence,
        "variance": variance_recurrence,
        "std": std_recurrence,
        "median": median_recurrence,
        "min": min_recurrence,
        "max": max_recurrence,
        "cv": cv_recurrence,
        "histogram": histogram,
    },
}


# --------------------------------------------------------
# Append to final JSONL
# --------------------------------------------------------

with OUTPUT_FILE.open("a") as f:
    f.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )


# --------------------------------------------------------
# Console summary
# --------------------------------------------------------

print(
    f"Mean: {mean_recurrence:.3f}"
    if mean_recurrence is not None
    else "Mean: None"
)

print(
    f"Median: {median_recurrence}"
)

print(
    f"Variance: {variance_recurrence:.3f}"
    if variance_recurrence is not None
    else "Variance: None"
)

print(
    f"Std: {std_recurrence:.3f}"
    if std_recurrence is not None
    else "Std: None"
)

print(
    f"CV: {cv_recurrence:.3f}"
    if cv_recurrence is not None
    else "CV: None"
)

print(
    "Histogram:",
    histogram,
)


print()
print("Finished.")
print("Results written to:")
print(OUTPUT_FILE)