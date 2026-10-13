# Runtime Latency Evaluation of Adaptive Exit and UT-Aware Scheduling

## Corrected outcome

This report supersedes the earlier result from the same experiment. The earlier
harness dynamically remapped all adaptive configurations and converted GPU
metadata to Python inside the timed decode loop. That did not implement the
intended fixed-group Adaptive-FCFS baseline and introduced synchronization
overhead.

The corrected harness keeps Original Ouro and Adaptive FCFS in four persistent
B=4 request groups. Only the UT-aware policies dynamically regroup requests at
each token position. A shared-cache dynamic-FCFS control is included to separate
the shared-cache runtime effect from homogeneous batching and SUF scheduling.
There are no tensor-to-host conversions or intermediate synchronizations in the
timed decode loop. CUDA synchronization occurs only immediately before and after
the complete decode interval, as required for wall-clock timing.

At `q=0.2`, B=4, and 127 timed decode positions, fixed-group physical Adaptive
FCFS was **53.09% slower** than Original Ouro despite executing roughly half as
many recurrence steps. Pred90 homogeneous batching + SUF reduced makespan by
**35.34%** versus fixed Adaptive FCFS and by **18.28%** versus the clean
shared-cache dynamic-FCFS control. It recovered **88.75%** of the Oracle
scheduler opportunity measured against fixed Adaptive FCFS.

## Fixed workload and runtime configuration

- Model: `ByteDance/Ouro-2.6B-Thinking`, revision
  `f1edd81e7ac41355db670500ceaf204e0f73af68`
- Runtime `config.num_hidden_layers`: **48**, read from the loaded checkpoint
- GPU: NVIDIA A40
- Dtype: BF16
- UT maximum: 4
- Exit threshold: 0.2
- Physical batch size: 4
- Requests: eight GSM8K and eight MATH-500 samples
- Reference trajectory: 128 fixed token IDs per request
- Primary workload: positions 1--127, or 2,032 timed decode tokens; position 0
  is emitted by prefill
- Adaptive KV policy: Last-Available
- One full warmup and two measured runs
- Tokenizer, dataset, model loading, and prefill are excluded from primary
  decode timing

An untimed fixed-group Adaptive-FCFS replay collected the physical selected-UT
sequence. The controlled 90% and Oracle labels were derived from that trace.
Predicted UT affects only dynamic batch membership and batch execution order;
the model always exits according to its runtime threshold decision.

## Scheduling and cache behavior

| System | Request grouping | Cache organization |
|---|---|---|
| Original Ouro | Fixed FCFS groups | Four persistent B=4 full-compute caches |
| Adaptive + FCFS | Fixed FCFS groups | Four persistent B=4 Last-Available caches |
| Adaptive + dynamic FCFS control | Fixed FCFS membership/order at every step | One request-indexed Last-Available cache |
| Adaptive + Pred90 homogeneous + SUF | Regrouped every token position | One request-indexed Last-Available cache |
| Adaptive + Oracle homogeneous + SUF | Regrouped every token position | One request-indexed Last-Available cache |

The dynamic-FCFS control is diagnostic. It is not a replacement for the
required fixed Adaptive-FCFS system; it lets us compare Pred90 against the same
shared-cache machinery and attribute the remaining difference to homogeneous
batching plus SUF rather than cache organization.

## Aggregate decode results

Times are seconds, reported as mean +/- sample standard deviation across two
measured runs.

| System | Makespan | Mean completion | Median | P95 / max | Tokens/s | Recurrence steps |
|---|---:|---:|---:|---:|---:|---:|
| Original Ouro | 116.9106 +/- 3.3103 | 116.5503 | 116.5503 | 116.9103 / 116.9103 | 17.39 | 8,128 |
| Adaptive + fixed FCFS | 178.9736 +/- 5.9978 | 178.4810 | 178.4305 | 178.9732 / 178.9732 | 11.36 | 3,987 |
| Adaptive + dynamic-cache FCFS control | 141.5984 +/- 6.7365 | 141.1622 | 141.0984 | 141.5981 / 141.5981 | 14.37 | 3,987 |
| Adaptive + Pred90 homogeneous + SUF | 115.7188 +/- 1.3834 | 115.1940 | 115.1206 | 115.7185 / 115.7185 | 17.56 | 3,990 |
| Adaptive + Oracle homogeneous + SUF | 107.6983 +/- 5.4596 | 107.2111 | 107.1428 | 107.6980 / 107.6980 | 18.89 | 3,988 |

Runtime selected-UT agreement with the calibration trace was 100% for both
FCFS controls, 99.56% for Pred90, and 99.66% for Oracle. Realized predictor
accuracy was 89.67% for Pred90 and 99.66% for Oracle; small batching-dependent
BF16 changes near the threshold explain the departures from the labels.

## Raw measured makespans

| System | Run 1 | Run 2 |
|---|---:|---:|
| Original Ouro | 119.2514 | 114.5699 |
| Adaptive + fixed FCFS | 174.7326 | 183.2147 |
| Adaptive + dynamic-cache FCFS control | 146.3619 | 136.8350 |
| Pred90 + homogeneous + SUF | 114.7406 | 116.6971 |
| Oracle + homogeneous + SUF | 111.5588 | 103.8377 |

## Decomposition

- Physical adaptive exit, fixed FCFS versus Original Ouro:
  - speedup: 0.653x
  - latency reduction: **-53.09%** (a slowdown)
- Shared dynamic cache, FCFS versus fixed-cache Adaptive FCFS:
  - latency reduction: **20.88%**
- Pred90 homogeneous + SUF versus dynamic-cache FCFS:
  - additional latency reduction on the same runtime infrastructure: **18.28%**
- Pred90 homogeneous + SUF versus required fixed Adaptive FCFS:
  - speedup: 1.547x
  - additional latency reduction: **35.34%**
- Oracle homogeneous + SUF versus fixed Adaptive FCFS:
  - latency reduction: **39.82%**
- Pred90 recovered Oracle opportunity versus fixed Adaptive FCFS: **88.75%**

The scheduler results are meaningful, but the 35.34% number combines the
shared-cache architecture improvement and scheduling. The cleaner isolated
scheduler comparison is the **18.28%** reduction from dynamic FCFS to Pred90,
because those systems use identical adaptive execution and shared-cache paths.

## Request IDs and lengths

| Request | Dataset index | Prompt tokens | Reference output positions |
|---|---:|---:|---:|
| `gsm8k:0` | 0 | 86 | 128 |
| `math500:0` | 0 | 69 | 128 |
| `gsm8k:1` | 1 | 46 | 128 |
| `math500:1` | 1 | 132 | 128 |
| `gsm8k:2` | 2 | 77 | 128 |
| `math500:2` | 2 | 68 | 128 |
| `gsm8k:3` | 3 | 55 | 128 |
| `math500:3` | 3 | 35 | 128 |
| `gsm8k:4` | 4 | 130 | 128 |
| `math500:4` | 4 | 421 | 128 |
| `gsm8k:5` | 5 | 76 | 128 |
| `math500:5` | 5 | 61 | 128 |
| `gsm8k:6` | 6 | 64 | 128 |
| `math500:6` | 6 | 39 | 128 |
| `gsm8k:7` | 7 | 91 | 128 |
| `math500:7` | 7 | 96 | 128 |

Exact token IDs, calibration UT traces, per-request completion events, and raw
runs are stored in
`results/runtime_adaptive_ut_scheduling_q02_b4_128_nosync_v2_20261013/results.json`.
The results directory is gitignored.

## Interpretation

The corrected result explains why Original Ouro appeared fastest in the first
comparison: reducing recurrence work is not sufficient when the current
Last-Available physical-exit implementation pays substantial cache
materialization, indexing, and small-kernel overhead. The effect persists after
removing timed-loop host synchronization. Four independent fixed caches add
further overhead, as shown by the dynamic-FCFS control.

UT-aware homogeneous batching is nevertheless useful. On identical dynamic
adaptive infrastructure, Pred90 reduces makespan by 18.28%, and Oracle by
23.94%, relative to dynamic FCFS. Pred90 therefore captures most of the
scheduling opportunity. In absolute terms Pred90 is approximately level with
Original Ouro (1.02% faster), while Oracle is 7.88% faster, but the current
physical adaptive path still needs optimization before adaptive exit alone is a
latency win.

This experiment does not isolate SUF from homogeneous grouping. With equal
128-position trajectories and a synchronization frontier after every token,
most makespan benefit comes from homogeneous batch composition. A
homogeneous-only ablation would be needed to attribute a separate SUF effect on
mean completion latency.
