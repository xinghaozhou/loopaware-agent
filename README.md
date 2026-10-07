# Setup

This branch traces Ouro's Universal Transformer (UT) decision for every output
token generated across multiple SWE-bench trajectories. The Hugging Face
reference model always computes every configured UT step. This experiment keeps
decoding fixed at the final UT step and uses the learned gates only to trace which
UT step adaptive decoding would have selected. The trace records both values so
the mimicked selected depth is not mistaken for actual saved compute.

Clone with submodules, install
[uv](https://docs.astral.sh/uv/getting-started/installation/), and synchronize the
locked environment:

```bash
git clone --recurse-submodules <repository-url> loopaware-agent
cd loopaware-agent
uv sync
```

If uv's cache is on a filesystem that does not support reflinks, use:

```bash
UV_CACHE_DIR=/tmp/loopaware-uv-cache uv sync
```

## Per-output-token UT experiment

The vendored repositories are registered as submodules. Model weights and the
dataset parquet are Git LFS objects; a normal submodule checkout retrieves their
pointers, while the experiment downloads the requested artifacts through the
Hugging Face libraries.

Run a small sampled experiment:

```bash
uv run python experiments/swe/run_swe_ut_trace.py \
  --num-instances 1 \
  --num-trajectories 4 \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --trace-exit-threshold 0.8
```

The command writes:

- `traces/swe_ut_trajectories.jsonl`: one record per trajectory, including a
  token-level list of gate probabilities, exit probabilities, expected UT depth,
  counterfactually selected UT step, fixed decoding UT step, and actually
  executed UT steps.
- `traces/swe_ut_summary.json`: per-trajectory summaries and a global selected
  UT-step histogram, plus the mean learned exit probability for every UT step.

Greedy decoding is available with `--temperature 0`. The token-to-forward
alignment currently supports standard generation only, not beam search or
assisted/speculative decoding.

`--trace-exit-threshold` affects only the reported selected UT step. It does not
change generation: output-token logits always come from the final configured UT
step.

Run the lightweight tests with:

```bash
uv run python -m unittest discover -s tests -v
```
