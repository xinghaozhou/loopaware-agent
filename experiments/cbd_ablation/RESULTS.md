# Common-substrate scheduler ablation (B=4)

## Outcome

On this single quick run, continuous depth batching (CBD refill) had the lowest
decode latency. It was 1.417× faster than fixed-FCFS and 1.092× faster than
Pred90/SUF. Pred90/SUF was 1.298× faster than fixed-FCFS.

| Scheduling policy | Decode latency (s) | Speedup vs fixed-FCFS | Recurrent launches | Mean recurrent batch | Coda launches | Mean coda batch |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-FCFS frontier | 55.152 | 1.000× | 1,178 | 2.869 | 508 | 4.000 |
| Pred90/SUF frontier | 42.502 | 1.298× | 997 | 3.390 | 508 | 4.000 |
| CBD refill | 38.907 | 1.417× | 872 | 3.876 | 838 | 2.425 |

CBD reduced decode latency by 29.5% relative to fixed-FCFS and by 8.5%
relative to Pred90/SUF. Pred90/SUF reduced latency by 22.9% relative to
fixed-FCFS.

All rows executed exactly 3,380 recurrent token-steps and produced 2,048
tokens (128 for each of 16 requests). Thus the observed difference is from
scheduling/batch formation, not a different selected-UT workload.

## Fixed configuration

- Model: `ByteDance/Ouro-2.6B-Thinking` at revision
  `f1edd81e7ac41355db670500ceaf204e0f73af68`.
- Runtime `config.num_hidden_layers`: **48** (read from the loaded checkpoint,
  not hard-coded).
- Workload: the same 16-request trace at selected-UT threshold 0.2.
- Decode length: 128 output tokens per request; the first token is produced by
  full-depth prefill and 127 tokens per request are inside the decode window.
- Selected-UT work: 3,380 recurrent token-steps for every policy.
- Resident requests: 16; recurrent/coda launch width: **B=4**.
- Runtime substrate: identical Ouro model code, eager synchronous execution,
  `paged|flash_attention_2`, 256-token pages, 64 pages, and single-slot KV.
- CUDA graphs: disabled for every row.
- Timing boundary: immediately before the first decode prelude through final
  CUDA completion. Prompt prefill and output materialization are excluded.
- Pred90 realized accuracy: 0.9000984252. Predicted UT controls only grouping
  and SUF order; recorded selected UT controls actual execution.

## Interpretation

Fixed-FCFS leaves more recurrent slots idle as rows exit at different depths.
Pred90/SUF improves homogeneity, reducing recurrent launches by 15.4%. CBD
refill dynamically replaces exited depth slots, reaching a mean recurrent batch
size of 3.876/4 and reducing recurrent launches by 26.0% versus fixed-FCFS.
CBD issues more, smaller coda batches; its recurrent utilization gain is large
enough to outweigh that overhead in this run.

This is a one-measurement quick ablation, so it establishes the direction and
approximate magnitude, not a confidence interval. The previous native-CBD
numbers are intentionally omitted because they used a different batch size,
CUDA graphs, KV/runtime setup, and timing scope.
