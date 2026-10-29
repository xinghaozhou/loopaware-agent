# Ordinary BF16 Batching Numerical-Stability Sanity Check

## Purpose

This is a small robustness check, not a task-quality evaluation. It isolates
ordinary BF16 batching from physical adaptive exit and asks whether changing
only the batch shape can perturb Ouro logits and threshold-derived selected UT.

## Configuration

- Model: `ByteDance/Ouro-2.6B-Thinking`
- Revision: `f1edd81e7ac41355db670500ceaf204e0f73af68`
- Runtime decoder layers: 48, read from the loaded checkpoint
- Total UT steps: 4
- Precision: BF16
- Requests: 4 pinned GSM8K test prompts and 4 pinned MATH-500 test prompts
- Reference: one greedy B=1 trajectory per request
- Replay: exact B=1 tokens, teacher-forced at B=4 and B=8
- Output positions: 16 per request, 128 total
- KV policy: `full_per_depth`
- Physical adaptive exit: disabled
- Exit thresholds evaluated post hoc: 0.2 and 0.4

No batched closed-loop generation, task-level evaluation, KL/top-5 metric,
B=2/B=16 condition, or per-condition repetition was performed.

## Controls

Two B=1 prompts were executed twice. Their generated tokens and logits were
bitwise identical, with maximum absolute logit difference 0.0.

For four prompts, threshold-one Original Ouro `full_per_depth` execution was
compared with physical `last_available` execution. Generated tokens,
selected UT, and logits matched exactly; maximum absolute logit difference was
0.0.

## Ordinary batching result

| Batch | Positions | Maximum absolute logit difference | Mean absolute logit difference | Top-1 agreement | Selected-UT @ 0.2 | Selected-UT @ 0.4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.000000 | 0.000000 | 100.000% | 100.000% | 100.000% |
| 4 | 128 | 0.601562 | 0.035245 | 100.000% | 99.219% | 99.219% |
| 8 | 128 | 1.000000 | 0.037813 | 100.000% | 100.000% | 99.219% |

There were three batch/position threshold mismatches, representing two unique
token positions:

- `gsm8k:2`, output position 0, B=4, threshold 0.2: the B=1 CDF crossed the
  threshold by only 0.000415 at UT=2, while the batched CDF remained only
  0.000102 from the boundary and selected UT=3.
- `math500:3`, output position 9, threshold 0.4: the B=1 UT=2 CDF was 0.000999
  below the threshold and selected UT=3. B=4 crossed it by 0.000730 and B=8
  crossed it by 0.001900, so both selected UT=2.

All top-1 tokens remained unchanged. The B=1 top-1/top-2 margins at these two
positions were 3.125 and 0.25, respectively.

## Conclusion

The controls isolate the observed drift to ordinary BF16 batching rather than
repeat nondeterminism or the physical adaptive-exit code path. Ordinary batching
produced mean absolute logit differences around 0.035--0.038 and maximum
differences of 0.60--1.00. Thus the previously observed approximately 0.125
perturbation is plausible from batching alone; this small sample actually
contains larger maxima.

The selected-UT changes occurred only where the exit CDF was within roughly
0.002 of a threshold. This supports treating boundary-adjacent selected-UT
differences as expected BF16 batching sensitivity rather than immediate evidence
of an adaptive-cache correctness bug.
