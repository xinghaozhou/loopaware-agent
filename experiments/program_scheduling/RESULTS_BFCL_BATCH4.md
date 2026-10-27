# BFCL V3 Full-Compute Batch-4 Latency

> Results for all six scheduling policies are reported in
> [`RESULTS_BFCL_BATCH4_SCHEDULERS.md`](RESULTS_BFCL_BATCH4_SCHEDULERS.md).

## Result

Real batch-4 execution reduced the 64-program FCFS makespan from **3,034.86 s**
to **1,130.89 s**, a **2.684x throughput speedup** over the prior batch-1 run.
Mean program time-to-last-token (TTLT) fell from **2,523.35 s** to **889.47 s**
(`-64.75%`).

This is a physical batched execution measurement, not a simulator-only scaling
factor. Ouro executed all four configured recurrent steps for every generated
token. The runtime checkpoint reported 48 decoder layers.

## Fixed configuration

- Dataset: BFCL V3 `multi_turn_base`
- Programs: the same nested 64-program seed-`20261026` sample
- Model: `ByteDance/Ouro-2.6B-Thinking`
- Model revision: `f1edd81e7ac41355db670500ceaf204e0f73af68`
- Decode: greedy, at most 128 new tokens
- Physical recurrence: fixed UT=4
- Selected-UT threshold: 0.2 in the source trace; it does not alter decoding
- Scheduling: FCFS
- Tool waits: zero
- Device: one NVIDIA A40
- Batch completion semantics: all members complete when their ordinary padded
  `generate` call completes; no streaming completion credit is assigned

Recorded BFCL calls and official tool observations remain teacher-forced into
later turns. Generated model output is measured but does not control subsequent
trajectory history.

## Latency comparison

| Metric | Batch 1 | Batch 4 | Change |
|---|---:|---:|---:|
| Total GPU makespan | 3,034.86 s | 1,130.89 s | 2.684x faster |
| Mean program TTLT | 2,523.35 s | 889.47 s | -64.75% |
| Median program TTLT | 2,693.40 s | 957.10 s | -64.46% |
| P95 program TTLT | 3,007.41 s | 1,128.69 s | -62.47% |
| Mean queue delay | 2,475.93 s | 818.79 s | -66.93% |
| Program throughput | 0.02109/s | 0.05659/s | 2.684x |
| Mean accumulated execution time/program | 47.42 s | 70.68 s | +49.05% |
| Generated-token throughput | 6.401/s | 16.717/s | 2.612x |

The increased per-program execution time is expected: every member is charged
the wall-clock service time of each lockstep batch containing it. Despite that
increase, parallel service substantially reduces queue delay and end-to-end
program TTLT.

## Batch distribution and memory

- 240 turns were executed in exactly 60 batches of four.
- Mean batch GPU time: 18.85 s.
- Median batch GPU time: 21.86 s.
- P95 batch GPU time: 24.01 s.
- Observed batch-time range: 8.13--27.04 s.
- Maximum peak allocated memory: 22.00 GiB.
- Maximum peak reserved memory: 31.57 GiB.
- No OOM or execution errors occurred.

Ordinary lockstep decoding makes batch latency depend mainly on the longest
generation in the group. A group whose four outputs ended within 45 tokens took
about 9.7 s, whereas groups containing a 128-token generation generally took
about 22--27 s. This leaves clear headroom for length-aware batch formation.

## Output-length caveat

Batch size can change greedy-decoding numerics. Of the 240 requests, 228 had the
same output length as the prior batch-1 trace; 12 differed. The batch-4 run
generated 18,905 tokens versus 19,425 at batch 1 (`-2.68%`). Therefore the raw
makespan ratio is slightly helped by less generated work. Token-normalized
throughput still improved by **2.612x**, so the batching gain remains large.

This runner saved output lengths but not batch-4 token IDs, so this experiment
does not claim token-for-token output equivalence.

## Artifacts

- `results/program_scheduling/bfcl64_seed20261026/fcfs_b4_full/results.json`
- `results/program_scheduling/bfcl64_seed20261026/fcfs_b4_full/summary.json`
- 60 atomic per-batch checkpoints under `fcfs_b4_full/batches/`
- `results/program_scheduling/bfcl64_seed20261026/fcfs_b4_full/run.log`

The result directory is a local, gitignored artifact. This Markdown file is the
tracked canonical result summary.
