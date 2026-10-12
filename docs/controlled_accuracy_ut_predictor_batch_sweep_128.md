# Controlled-Accuracy UT Predictor: 128-Token Batch Sweep

## Setup

- Model: `ByteDance/Ouro-2.6B-Thinking` at revision
  `f1edd81e7ac41355db670500ceaf204e0f73af68`
- Runtime `config.num_hidden_layers`: **48** (read from the loaded checkpoint)
- Full-compute trace: UT=4, BF16, `full_per_depth`, physical adaptive exit disabled
- Inputs: GSM8K test indices 0--7 and MATH-500 test indices 0--7 at the
  pinned revisions recorded in the trace manifest
- Trace size: 16 requests x 128 output tokens = 2,048 labels and 128 aligned
  scheduling instances
- Scheduling: N=16, B in {4, 8, 16}, controlled accuracy in
  {70%, 80%, 90%}, plus the oracle required to define headroom
- Incorrect predictions are local errors (`true UT +/- 1`) and results average
  ten fixed seeds. All execution costs use true threshold-derived UT.

The model trace ran from `2026-10-12T22:44:26Z` to
`2026-10-12T22:52:44Z` (8m18s including setup). The analytical sweep took
under ten seconds.

## Threshold 0.2

| B | Accuracy | Mean cost | Reduction vs FCFS | Recovered oracle headroom | Mean UT range |
|---:|---:|---:|---:|---:|---:|
| 4 | FCFS | 37.125 | 0.00% | 0.00% | 1.205 |
| 4 | 70% | 33.378 | 10.09% | 48.35% | 0.827 |
| 4 | 80% | 32.244 | 13.15% | 62.98% | 0.701 |
| 4 | 90% | 31.009 | 16.47% | 78.91% | 0.559 |
| 4 | Oracle | 29.375 | **20.88%** | 100.00% | 0.361 |
| 8 | FCFS | 41.375 | 0.00% | 0.00% | 1.566 |
| 8 | 70% | 37.862 | 8.49% | 52.04% | 1.268 |
| 8 | 80% | 37.062 | 10.42% | 63.89% | 1.166 |
| 8 | 90% | 36.188 | 12.54% | 76.85% | 1.046 |
| 8 | Oracle | 34.625 | **16.31%** | 100.00% | 0.820 |
| 16 | FCFS / all accuracies / oracle | 44.625 | **0.00%** | N/A | 1.789 |

## Threshold 0.4

| B | Accuracy | Mean cost | Reduction vs FCFS | Recovered oracle headroom | Mean UT range |
|---:|---:|---:|---:|---:|---:|
| 4 | FCFS | 46.375 | 0.00% | 0.00% | 0.855 |
| 4 | 70% | 43.531 | 6.13% | 50.56% | 0.532 |
| 4 | 80% | 42.784 | 7.74% | 63.83% | 0.448 |
| 4 | 90% | 41.866 | 9.72% | 80.17% | 0.332 |
| 4 | Oracle | 40.750 | **12.13%** | 100.00% | 0.193 |
| 8 | FCFS | 47.625 | 0.00% | 0.00% | 0.973 |
| 8 | 70% | 45.812 | 3.81% | 34.94% | 0.793 |
| 8 | 80% | 45.112 | 5.28% | 48.43% | 0.726 |
| 8 | 90% | 44.100 | 7.40% | 67.95% | 0.611 |
| 8 | Oracle | 42.438 | **10.89%** | 100.00% | 0.426 |
| 16 | FCFS / all accuracies / oracle | 48.500 | **0.00%** | N/A | 1.031 |

## Interpretation

The useful headroom is strongest at B=4. At 90% controlled accuracy it
recovers 78.91%--80.17% of the oracle opportunity and reduces analytical
recurrence-slot cost by 9.72%--16.47%. B=8 still has meaningful oracle
headroom, but 90% prediction recovers less of it, especially at threshold 0.4.

B=16 is an intentional negative control. Because N=16 produces exactly one
batch, sorting changes neither membership nor lockstep cost. It therefore has
zero oracle headroom; this is not evidence against prediction at B=16 when a
larger active pool permits multiple batches.

These are analytical adaptive-exit slot savings, not measured GPU speedups.
They omit predictor overhead and runtime scheduling costs.

Raw artifacts are in
`results/predictor_headroom_trace_ut4_128_20261012/` (gitignored), including
`headroom_batch_sweep.json` and `BATCH_SWEEP.md`.
