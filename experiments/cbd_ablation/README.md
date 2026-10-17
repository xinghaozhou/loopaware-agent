# Common-substrate B=4 scheduler ablation

This directory is the complete text-only CBD change set kept separate from the
pre-CBD `xinghao-codex` branch. It targets upstream repository
`https://github.com/kschwethelm/continuous-depth-batching` at commit
`abdfae0adbdd12be8c6123ccb34eee98a38afb39`.

This experiment compares only scheduling policy. All rows use Ouro-2.6B-Thinking,
the same 16-request selected-UT trace (threshold 0.2), 128 generated tokens per
request (127 timed decode tokens after the full-depth prefill token), 16 resident
states, recurrent/coda launch width B=4, eager synchronous execution, the same
single-slot paged KV cache, FlashAttention-2 kernels, and the same decode timer.

Policies:

- `fixed_fcfs`: global token frontier, stable request-order groups of four.
- `pred90_suf`: global token frontier, homogeneous Pred90 groups scheduled by
  shortest predicted UT first. True selected UT controls execution.
- `cdb_refill`: continuous depth batching refills freed depth slots from its
  ready queue.

The timer begins immediately before the first decode prelude and ends after the
final CUDA completion. Prompt prefill and output materialization are excluded.
Earlier native-CDB results are intentionally not included because their batch
size, CUDA graphs, timing scope, and runtime configuration differ.

## Reproduce

```bash
git clone https://github.com/kschwethelm/continuous-depth-batching.git /tmp/continuous-depth-batching
git -C /tmp/continuous-depth-batching checkout abdfae0adbdd12be8c6123ccb34eee98a38afb39
git -C /tmp/continuous-depth-batching apply /root/loopaware-agent/experiments/cbd_ablation/continuous_depth_batching.patch
cd /tmp/continuous-depth-batching
uv sync --frozen --extra fa2
```

Regenerate the exact workload bundle from the existing selected-UT trace:

```bash
uv run python /root/loopaware-agent/experiments/cbd_ablation/prepare_workload.py \
  --trace-dir /root/loopaware-agent/results/predictor_trace_ut4_128_20261017 \
  --checkpoint /tmp/loopaware-hf-cache/hub/models--ByteDance--Ouro-2.6B-Thinking/snapshots/f1edd81e7ac41355db670500ceaf204e0f73af68 \
  --output /tmp/continuous-depth-batching/loopaware_16_q0p2.json \
  --threshold 0.2
```

Run the controlled comparison:

```bash
uv run python /root/loopaware-agent/experiments/cbd_ablation/run_common_ablation.py \
  --model /tmp/loopaware-hf-cache/hub/models--ByteDance--Ouro-2.6B-Thinking/snapshots/f1edd81e7ac41355db670500ceaf204e0f73af68 \
  --workload /tmp/continuous-depth-batching/loopaware_16_q0p2.json \
  --predictions /root/loopaware-agent/results/request_indexed_kv_profile_20261017/checkpoint.json \
  --output /tmp/continuous-depth-batching/common_b4_results.json
```

`continuous_depth_batching.patch` contains only the upstream compatibility fix,
the independent decode launch-width control, the two frontier policies, and the
common decode timing hook. Generated logs, model/cache files, the cloned upstream
repository, and Python bytecode are deliberately excluded from this branch.
