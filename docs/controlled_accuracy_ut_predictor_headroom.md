# Controlled-Accuracy UT Predictor Headroom Study

This is an analytical adaptive-exit scheduling simulation, not measured GPU
speedup. Predicted UT is used only for grouping and ordering; every execution
cost uses threshold-derived true UT.

## Exact trace source

- GSM8K: `openai/gsm8k`, revision
  `740312add88f781978c0658806c59bc2815b9866`, test indices 0--7
- MATH-500: `HuggingFaceH4/MATH-500`, revision
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, test indices 0--7
- Model: `ByteDance/Ouro-2.6B-Thinking`, revision
  `f1edd81e7ac41355db670500ceaf204e0f73af68`
- Runtime decoder layers: 48, read from the checkpoint
- Execution: BF16, UT=4, B=1, `full_per_depth`, physical exit disabled
- Trajectory: one fixed-final-UT greedy trajectory per request, 64 tokens
- Trace size: 16 requests and 1,024 token-level exit CDFs

Each of the 64 scheduling instances uses the same output position from all 16
requests. Instances are therefore non-overlapping active-request frontiers, not
arbitrary adjacent rows from a trace file.

## Controlled predictor and scheduling

For each target accuracy, an aggregate exact-match count is selected across all
1,024 labels. Incorrect predictions are true UT plus or minus one, clipped to
UT 1--4. Results for 70%, 80%, and 90% average ten fixed random seeds; 100% is
the deterministic oracle.

FCFS retains original request order and partitions consecutive groups of four.
UT-aware batching stably sorts by predicted UT, forms four adjacent batches,
and executes them from shortest predicted depth to longest. Adaptive exit is
simulated with true UT. Batch cost is always:

```text
4 * max(true UT in the batch)
```

## Threshold 0.2

Oracle headroom: **23.42%**.

| Strategy | Predictor accuracy | Realized accuracy | Mean total cost | Cost reduction vs FCFS | Recovered oracle headroom | Mean UT range | Mean batch completion time |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | N/A | N/A | 35.500 | 0.00% | 0.00% | 1.164 | 22.000 |
| UT-aware | 70% | 70.02% | 31.331 | 11.74% | 50.15% | 0.777 | 17.586 |
| UT-aware | 80% | 79.98% | 30.306 | 14.63% | 62.48% | 0.675 | 16.692 |
| UT-aware | 90% | 90.04% | 28.725 | 19.08% | 81.50% | 0.509 | 15.414 |
| UT-aware Oracle | 100% | 100.00% | 27.188 | 23.42% | 100.00% | 0.324 | 14.219 |

True-UT histogram: UT1 572, UT2 343, UT3 109.

## Threshold 0.4

Oracle headroom: **12.72%**.

| Strategy | Predictor accuracy | Realized accuracy | Mean total cost | Cost reduction vs FCFS | Recovered oracle headroom | Mean UT range | Mean batch completion time |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | N/A | N/A | 45.188 | 0.00% | 0.00% | 0.809 | 28.250 |
| UT-aware | 70% | 70.02% | 42.069 | 6.90% | 54.24% | 0.507 | 24.958 |
| UT-aware | 80% | 79.98% | 41.487 | 8.19% | 64.35% | 0.438 | 24.512 |
| UT-aware | 90% | 90.04% | 40.456 | 10.47% | 82.28% | 0.321 | 23.642 |
| UT-aware Oracle | 100% | 100.00% | 39.438 | 12.72% | 100.00% | 0.195 | 22.812 |

True-UT histogram: UT2 663, UT3 355, UT4 6.

## Answers

- RQ1: perfect prediction provides meaningful headroom: 23.42% at threshold
  0.2 and 12.72% at threshold 0.4.
- RQ2: 70%, 80%, and 90% prediction recover progressively more oracle benefit.
- RQ3: 90% accuracy recovers 81.50% and 82.28% of oracle headroom, respectively.
- RQ4: UT-aware grouping substantially reduces within-batch UT range at every
  tested accuracy.
- RQ5: at 90% accuracy, the mean batch-completion proxy falls by 29.94% at
  threshold 0.2 and 16.31% at threshold 0.4.

## Conclusion

Next-token UT prediction is worth pursuing for a real predictor pilot under
this local-error model. The oracle opportunity is material, and 90% prediction
recovers more than 80% of it at both thresholds.

This does not establish measured GPU speedup. It excludes predictor inference
overhead, queueing delay, fairness/starvation controls, and production kernel
behavior.
