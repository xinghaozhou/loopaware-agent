# Runtime Latency Evaluation of Adaptive Exit and UT-Aware Scheduling

## Outcome

At `q=0.2`, B=4, and 127 timed decode positions per request, controlled
90%-accurate UT scheduling reduced mean decode makespan by **17.92%** relative
to the same physical adaptive-exit + Last-Available implementation under FCFS.
It recovered **82.58%** of the Oracle scheduling opportunity.

Physical adaptive exit itself did not outperform Original Ouro in this
implementation: Adaptive FCFS was **28.27% slower** even though it reduced
executed token-recurrence steps from 8,128 to 3,987. The current adaptive cache
and dynamic-compaction overhead outweigh the saved transformer work at this
workload and context length.

## Fixed workload and runtime configuration

- Model: `ByteDance/Ouro-2.6B-Thinking`, revision
  `f1edd81e7ac41355db670500ceaf204e0f73af68`
- Runtime `config.num_hidden_layers`: **48**, read from the loaded checkpoint
- GPU: NVIDIA A40
- Dtype: BF16
- UT maximum: 4
- Exit threshold: 0.2
- Physical batch size: 4
- Requests: the same eight GSM8K and eight MATH-500 samples used by the
  controlled-headroom experiment
- Reference trajectory: 128 fixed token IDs per request
- Primary timed workload: positions 1--127, or 2,032 decode tokens total;
  position 0 is emitted by prefill
- Adaptive KV policy: Last-Available
- Tokenizer, dataset loading, and model loading are excluded
- Prefill is separately timed and excluded from primary decode makespan
- One full warmup and two measured runs

The initial full-compute B=1 trace agreed with physical Last-Available selected
UT on only 73.82% of decode positions. It was therefore not used as the runtime
Oracle. An untimed Adaptive-FCFS fixed-token replay first collected the physical
selected-UT sequence. The 90% labels and Oracle labels were constructed from
that calibration trace. During measured execution, Pred90 retained 89.67%
accuracy against actually executed UT; Oracle retained 99.66%. The remaining
difference is caused by small batching-dependent numerical shifts around exit
thresholds.

Dynamic regrouping occurs before every decode position. Four batches are formed
from the 16 ready requests, executed in predicted-UT order, and atomically
committed into a shared request-indexed KV history. Predicted UT affects only
batch membership and execution order. The model always exits from its own
runtime threshold decision.

## Aggregate decode results

Times are seconds, reported as mean +/- sample standard deviation across two
measured runs.

| System | Makespan | Mean completion | Median | P95 / max | Tokens/s | Recurrence steps |
|---|---:|---:|---:|---:|---:|---:|
| Original Ouro | 114.4009 +/- 5.2422 | 114.0598 | 114.0600 | 114.4009 / 114.4009 | 17.78 | 8,128 |
| Adaptive + FCFS | 146.7381 +/- 8.1756 | 146.2747 | 146.2057 | 146.7380 / 146.7380 | 13.87 | 3,987 |
| Adaptive + 90% UT + homogeneous + SUF | 120.4372 +/- 2.8168 | 119.9235 | 119.8501 | 120.4372 / 120.4372 | 16.88 | 3,990 |
| Adaptive + Oracle UT + homogeneous + SUF | 114.8908 +/- 7.7739 | 114.3751 | 114.3040 | 114.8907 / 114.8907 | 17.73 | 3,988 |

Mean prefill times were 2.8670s, 2.9331s, 2.9128s, and 2.9172s respectively.
Including prefill does not change the conclusions because prefill differs by
less than 0.07s across systems.

## Raw measured makespans

| System | Run 1 | Run 2 |
|---|---:|---:|
| Original Ouro | 110.6941 | 118.1077 |
| Adaptive + FCFS | 140.9571 | 152.5191 |
| Pred90 + homogeneous + SUF | 122.4290 | 118.4455 |
| Oracle + homogeneous + SUF | 120.3877 | 109.3938 |

Paired scheduler reductions were 13.14% and 22.34% for Pred90, and 14.59%
and 28.28% for Oracle. Only two measured runs were used because each full
matrix is expensive; the means are directionally useful, but their uncertainty
must not be hidden.

## Decomposition

- Physical adaptive exit versus Original Ouro:
  - speedup: 0.780x
  - latency reduction: **-28.27%** (a slowdown)
- Pred90 scheduler versus Adaptive FCFS:
  - speedup: 1.218x
  - additional makespan reduction: **17.92%**
  - mean-completion reduction: **18.01%**
- Oracle scheduler versus Adaptive FCFS:
  - makespan reduction: **21.70%**
- Pred90 recovered Oracle wall-clock opportunity: **82.58%**
- Pred90 runtime reduction divided by the earlier 20.88% analytical B=4
  headroom: **85.84%**

The last ratio is contextual rather than an identity: the analytical study used
full-compute B=1 threshold labels and a `B * max(UT)` cost model, whereas this
runtime experiment uses physical Last-Available calibration labels and actual
GPU execution.

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

Exact token IDs, the original full-compute UT traces, calibrated physical UT
traces, per-request completion times, and per-run measurements are stored in
`results/runtime_adaptive_ut_scheduling_q02_b4_128_20261013/results.json`.
That results directory is gitignored.

## Interpretation

The answer to the scheduler question is provisionally **yes**: on the same
physical adaptive model, 90%-accurate homogeneous batching + SUF produced a
meaningful real makespan reduction and recovered most of the Oracle opportunity.
Nearly identical recurrence counts show that this gain comes from more
GPU-efficient batch shapes rather than additional early exits.

The answer for the current physical adaptive implementation as a whole is
**no**: its cache and compaction overhead make Adaptive FCFS slower than
Original Ouro. Scheduler gains almost recover that loss—Oracle is approximately
level with Original Ouro—but do not yet establish an end-to-end adaptive-exit
speedup.

Mean request completion latency falls with scheduled makespan, but this run does
not isolate SUF from homogeneous grouping. Because all requests have the same
128-position replay length and synchronize at every token frontier, SUF affects
completion order primarily during the last frontier. A homogeneous-only
ablation would be required to attribute a separate SUF latency benefit.
