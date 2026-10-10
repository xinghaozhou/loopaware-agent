# Adaptive KV Threshold Sweep: 0.1–0.7

## Setup

- Model: `ByteDance/Ouro-2.6B-Thinking`
- Revision: `f1edd81e7ac41355db670500ceaf204e0f73af68`
- Runtime decoder layers: 48
- Maximum recurrence: UT=6
- Thresholds: `0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7`
- Policies: `last_available`, `final_exit_shared`
- Decode: greedy, BF16, maximum 16 new tokens
- Workload: three fixed prompts, run independently at batch size 1
- Validation: every physical run was paired with a run-all reference using the
  same KV policy and threshold

Running prompts independently preserves an identical kernel batch shape before
the selected exit. All 14 conditions matched their policy-specific run-all
reference exactly in generated tokens and selected UT depths.

`mean selected UT` below excludes the first output token because that token is
predicted by full-depth prefill. Recurrence saving includes prefill and is:

```text
1 - physical_executed_ut_total / run_all_executed_ut_total
```

## Results

| Policy | Threshold | Decode tokens | Mean selected UT | Selected-UT histogram | Recurrence saving | Wall-time speedup |
|---|---:|---:|---:|---|---:|---:|
| `last_available` | 0.1 | 45 | 1.800 | `{1:22, 2:12, 3:9, 4:2}` | 65.6% | 3.096× |
| `last_available` | 0.2 | 45 | 2.222 | `{1:17, 2:7, 3:15, 4:6}` | 59.0% | 2.373× |
| `last_available` | 0.3 | 45 | 2.622 | `{2:22, 3:18, 4:5}` | 52.8% | 2.098× |
| `last_available` | 0.4 | 45 | 2.756 | `{2:19, 3:18, 4:8}` | 50.7% | 1.974× |
| `last_available` | 0.5 | 45 | 3.089 | `{2:8, 3:25, 4:12}` | 45.5% | 1.796× |
| `last_available` | 0.6 | 45 | 3.644 | `{3:22, 4:17, 5:6}` | 36.8% | 1.546× |
| `last_available` | 0.7 | 45 | 3.956 | `{3:15, 4:18, 5:11, 6:1}` | 31.9% | 1.418× |
| `final_exit_shared` | 0.1 | 45 | 1.844 | `{1:11, 2:30, 3:4}` | 64.9% | 2.981× |
| `final_exit_shared` | 0.2 | 34 | 1.824 | `{1:11, 2:18, 3:5}` | 64.0% | 2.846× |
| `final_exit_shared` | 0.3 | 34 | 1.824 | `{1:11, 2:18, 3:5}` | 64.0% | 2.681× |
| `final_exit_shared` | 0.4 | 22 | 1.773 | `{1:11, 2:5, 3:6}` | 62.0% | 2.737× |
| `final_exit_shared` | 0.5 | 20 | 2.450 | `{2:12, 3:7, 4:1}` | 51.4% | 2.260× |
| `final_exit_shared` | 0.6 | 20 | 2.450 | `{2:12, 3:7, 4:1}` | 51.4% | 2.117× |
| `final_exit_shared` | 0.7 | 45 | 2.800 | `{2:16, 3:22, 4:7}` | 50.0% | 1.856× |

The number of decode tokens varies for `final_exit_shared` because some
thresholds generated EOS before the 16-token cap. Cross-threshold selected-depth
means are therefore not expected to be strictly monotonic: changing the
threshold changes the generated trajectory and sometimes its length. Within
each condition, physical and run-all outputs remained identical.

Peak allocated memory stayed near 5.0 GiB in every condition. These contexts
are short and model weights dominate the measurement, so this pilot does not
establish a decode-cache memory reduction.

## Batched numerical-stability finding

An initial three-prompt batched run passed at threshold 0.1. At threshold 0.2,
`last_available` produced its first physical/run-all token mismatch at output
index 14 for batch row 0, despite identical selected-depth traces.

At that position:

| Measurement | Run-all | Physical |
|---|---:|---:|
| Top token IDs | `[975, 288]` | `[975, 288]` |
| Top logits | `[17.750, 17.625]` | `[17.625, 17.625]` |
| Top-two margin | `0.125` | `0.000` |

The maximum absolute logit difference was 0.125, one BF16-scale quantization
step at this magnitude. Active-row compaction changed the kernel batch shape,
turning a narrow margin into an exact tie; greedy tie-breaking then selected a
different token. This is consistent with numerical sensitivity rather than a
different selected-depth or KV-policy decision, but it means strict token
identity is not guaranteed when batch rows compact at different depths.

For semantic validation and fixed-trajectory replay, use batch size 1 or a
teacher-forced common token stream. Treat production batched trajectory
stability as a separate engineering problem requiring a defined numerical
tolerance and possibly stable-shape masked execution.

## Interpretation

- `last_available` shows the expected compute/threshold tradeoff. Threshold
  0.4 is a useful middle point in this pilot: 50.7% recurrence saving and a
  measured 1.97× speedup.
- `final_exit_shared` selects shallower depths on these prompts, but it defines
  a different generation process and often changes termination behavior.
- The measured wall-time gains demonstrate physical compute reduction; they
  are not merely counterfactual selected-UT statistics.
- This is a bounded prompt-level sweep, not the full SWE-bench trajectory
  replay.

Machine-readable output is stored at
`results/adaptive_kv_threshold_sweep_0.1_0.7.json`.
