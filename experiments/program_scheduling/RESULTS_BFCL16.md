# BFCL V3 16-Program Quick Validation

## Configuration

- Dataset: BFCL V3 `multi_turn_base`, 16 programs sampled with seed `20261026`
- Model: `ByteDance/Ouro-2.6B-Thinking`, revision `f1edd81e...`
- Physical execution: fixed full compute at UT=4
- Counterfactual selected-UT threshold: 0.2
- Runtime decoder layers: 48, read from `model.config.num_hidden_layers`
- Generation: deterministic greedy decoding, at most 128 tokens per reasoning turn
- Simulator: one GPU worker, non-preemptive turns, burst arrival at time zero
- Trajectory progression: BFCL ground-truth calls and official tool results are teacher-forced
- Tool waits: zero because BFCL V3 does not provide timestamps

The 16 programs produced 66 reasoning turns and 5,200 output tokens. Program
lengths ranged from 2 to 6 turns. Thirty turns reached the 128-token cap. Input
contexts ranged from 841 to 2,619 tokens (median 1,588).

Every generated token physically executed all four recurrent iterations. The
counterfactual selected-UT histogram was:

| Selected UT | Tokens |
|---:|---:|
| 1 | 1,874 |
| 2 | 1,499 |
| 3 | 1,787 |
| 4 | 40 |

Mean selected UT was 1.999. Total measured GPU service was 843.5 seconds.

## Scheduling results

| Scheduler | Mean TTLT (s) | P95 TTLT (s) | Mean queue (s) | Throughput (program/s) | vs GPU-PLAS |
|---|---:|---:|---:|---:|---:|
| FCFS | 751.96 | 835.23 | 699.24 | 0.01897 | -8.16% |
| GPU-PLAS | 695.24 | 834.57 | 642.52 | 0.01897 | baseline |
| UT-LAS | 697.79 | 834.57 | 645.07 | 0.01897 | -0.37% |
| Oracle Token-SUF | 564.77 | 827.98 | 512.05 | 0.01897 | +18.77% |
| Oracle UT-SUF | 492.77 | 810.34 | 440.05 | 0.01897 | +29.12% |
| Oracle Runtime SJF | 457.51 | 781.50 | 404.79 | 0.01897 | +34.19% |

Explicit headroom results:

```text
Oracle SJF headroom over PLAS: 34.19%
Oracle UT-SUF improvement over PLAS: 29.12%
Fraction of SJF headroom recovered by UT oracle: 85.17%
UT-LAS improvement over GPU-PLAS: -0.37%
```

Throughput is identical because all programs arrive together, tool waits are
zero, there is one work-conserving worker, and every policy replays identical
full-compute service.

## Signal characterization

| Relationship | Pearson | Spearman |
|---|---:|---:|
| Sum selected UT vs runtime | 0.9911 | 0.9144 |
| Output tokens vs runtime | 0.9988 | 0.9487 |

Tie-aware pairwise runtime-order accuracy over all turns was 87.74% for selected
UT and 88.53% for output tokens. Restricting comparisons to turns simultaneously
READY under the GPU-PLAS schedule gave 73.31% for selected UT and 78.91% for
output tokens.

Despite token count having higher correlation and pairwise accuracy, Oracle
UT-SUF achieved lower mean TTLT than Oracle Token-SUF in this sample. A likely
reason is that 30 turns tie at the 128-token cap; selected recurrent depth adds
ranking resolution within that large tied group. This must be tested without a
short output cap and over multiple seeds before attributing the difference to a
general UT-specific scheduling signal.

## Preliminary decision

This run provides a strong **oracle headroom signal** but no positive signal for
historical UT-LAS. Future selected-UT workload appears worth further validation:
Oracle UT recovered 85.2% of turn-level SJF headroom. However, the practical
attained-service policy did not improve GPU-PLAS in this seed.

This is not yet sufficient to justify training a predictor. Required next checks
are five or more 16-program seeds, a 32-program run, a higher generation cap,
and runtime-repeat sensitivity. The current run uses one runtime measurement per
turn, zero tool waits, one sampled workload, and teacher-forced BFCL progression.

## Plots

- [Normalized mean TTLT](../../results/program_scheduling/bfcl16_seed20261026/plots/normalized_mean_ttlt.svg)
- [P95 TTLT](../../results/program_scheduling/bfcl16_seed20261026/plots/p95_ttlt.svg)
- [Completion-time CDF](../../results/program_scheduling/bfcl16_seed20261026/plots/completion_cdf.svg)
- [Selected UT vs runtime](../../results/program_scheduling/bfcl16_seed20261026/plots/selected_ut_vs_runtime.svg)
- [Output tokens vs runtime](../../results/program_scheduling/bfcl16_seed20261026/plots/tokens_vs_runtime.svg)
- [Pairwise ordering accuracy](../../results/program_scheduling/bfcl16_seed20261026/plots/pairwise_accuracy.svg)

The fixed raw trace and full simulator output are stored with the plots under
`results/program_scheduling/bfcl16_seed20261026/`.
