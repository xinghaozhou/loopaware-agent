# BFCL V3 Full-Compute Batch-4 Scheduler Latency

## Result

With real fixed-UT=4 batch execution, future-work oracles still reduce **mean**
program TTLT, but they trade away tail latency and makespan. Oracle Runtime-SJF
has the lowest mean TTLT at **736.81 s**. Oracle selected-UT is close at
**754.75 s**, recovering **89.23%** of the available mean-TTLT improvement over
GPU-PLAS.

Cumulative selected-UT service (`UT_LAS`) remains worse than measured GPU
service (`GPU_PLAS`): mean TTLT increases from **903.48 s** to **918.07 s**.

## Fixed configuration

- BFCL V3 `multi_turn_base`, the same 64-program seed-`20261026` sample
- `ByteDance/Ouro-2.6B-Thinking`, revision
  `f1edd81e7ac41355db670500ceaf204e0f73af68`
- physical full-compute UT=4; runtime decoder-layer count 48
- trace-only selected-UT threshold 0.2
- greedy decoding, maximum 128 new tokens, physical batch size 4
- recorded tool calls and official observations teacher-forced into later turns
- zero tool wait and burst program arrivals
- one NVIDIA A40

Each scheduler formed batches from programs READY at that decision point. The
GPU time of the real padded `generate` call advanced the scheduling clock, and
all members were credited completion at the end of the batch.

## Scheduler comparison

| Scheduler | Mean TTLT | P95 TTLT | Makespan | Mean batch | Programs/s | Mean TTLT vs GPU-PLAS |
|---|---:|---:|---:|---:|---:|---:|
| FCFS | 889.47 s | **1,128.69 s** | **1,130.89 s** | 18.85 s | **0.05659** | +1.55% |
| GPU-PLAS | 903.48 s | 1,162.52 s | 1,189.81 s | 19.51 s | 0.05379 | baseline |
| UT-LAS | 918.07 s | 1,230.64 s | 1,233.61 s | 20.56 s | 0.05188 | -1.61% |
| Oracle Token-SUF | 850.88 s | 1,334.61 s | 1,338.50 s | 22.31 s | 0.04781 | +5.82% |
| Oracle UT-SUF | 754.75 s | 1,298.07 s | 1,317.32 s | 21.60 s | 0.04858 | +16.46% |
| Oracle Runtime-SJF | **736.81 s** | 1,315.63 s | 1,411.34 s | 22.76 s | 0.04535 | **+18.45%** |

A positive value in the last column means lower mean TTLT than GPU-PLAS. The
oracle-runtime mean-TTLT headroom is 18.45%; Oracle UT-SUF captures 16.46
percentage points, or 89.23% of that headroom.

## Interpretation

There are two different latency objectives:

1. **Mean program completion.** Shortest-future-work scheduling completes many
   short programs early. Oracle Runtime-SJF improves mean TTLT by 18.45% versus
   GPU-PLAS, and Oracle UT-SUF improves it by 16.46%.
2. **Tail latency and makespan.** The same policies defer long programs. FCFS has
   the best P95 and makespan in this burst-arrival, zero-tool-wait workload.
   Runtime-SJF lowers mean TTLT most but increases makespan by 24.80% versus
   FCFS.

Selected UT therefore remains useful as a future-size signal under physical
batching, but raw cumulative selected UT remains a poor attained-service unit.
The sign of the UT-LAS result agrees with the earlier batch-1 experiment.

Batch formation also changes physical service. Schedulers that create mixed
short/128-token groups pay for the longest member under ordinary lockstep
generation. This explains why the oracle policies can improve mean completion
while reducing total program throughput.

## Integrity and limitations

Every policy covers all 240 unique `(program, turn)` requests exactly once.
FCFS, GPU-PLAS, Oracle Token-SUF, Oracle UT-SUF, and Oracle Runtime-SJF used
their own physical batch measurements. UT-LAS physically executed 47 distinct
batches and reused 13 previously measured batches whose four ordered members
matched exactly; no unmatched measurement was reused.

The policies produce different batch groupings, and batched greedy numerics can
change generated output lengths. Accordingly, makespan and token throughput are
properties of each realized batched run, not a fixed-token simulation. These
are single sequential measurements without repeated-run confidence intervals.

## Artifacts

Each local, gitignored result directory contains `results.json`, `summary.json`,
the run log, and atomic per-batch checkpoints:

- `results/program_scheduling/bfcl64_seed20261026/fcfs_b4_full/`
- `results/program_scheduling/bfcl64_seed20261026/gpu_plas_b4_full/`
- `results/program_scheduling/bfcl64_seed20261026/ut_las_b4_full/`
- `results/program_scheduling/bfcl64_seed20261026/oracle_token_suf_b4_full/`
- `results/program_scheduling/bfcl64_seed20261026/oracle_ut_suf_b4_full/`
- `results/program_scheduling/bfcl64_seed20261026/oracle_runtime_sjf_b4_full/`
