# BFCL V3 Nested 16/32/64-Program Validation

## Result

The larger nested workloads reinforce the first-run result: cumulative past
selected-UT service (`UT_LAS`) does not improve on cumulative measured GPU
service (`GPU_PLAS`). In contrast, perfect knowledge of the upcoming turn's
selected-UT workload (`ORACLE_UT_SUF`) consistently recovers most of the
turn-level runtime-SJF headroom.

All sizes use prefixes of the same seed-`20261026` BFCL V3 `multi_turn_base`
sample. The 16-program trace is an exact prefix of 32, which is an exact prefix
of 64. Execution remains fixed full-compute Ouro at UT=4; threshold-0.2 selected
UT is scheduling metadata only.

| Programs | GPU-PLAS mean TTLT | UT-LAS mean TTLT | UT-LAS vs PLAS | Oracle UT mean TTLT | Oracle UT vs PLAS | Oracle SJF mean TTLT | SJF headroom recovered |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 695.24 s | 697.79 s | -0.37% | 492.77 s | +29.12% | 457.51 s | 85.17% |
| 32 | 1,279.43 s | 1,308.16 s | -2.25% | 929.03 s | +27.39% | 827.09 s | 77.46% |
| 64 | 2,405.78 s | 2,435.64 s | -1.24% | 1,699.81 s | +29.34% | 1,527.42 s | 80.37% |

The sign is consistent at all three sizes:

```text
UT-LAS improvement over GPU-PLAS:       negative at 16, 32, and 64
Oracle UT-SUF improvement over PLAS:    approximately 27% to 29%
SJF scheduling headroom recovered:      approximately 77% to 85%
```

## Full policy tables

### 32 programs

| Scheduler | Mean TTLT (s) | P95 TTLT (s) | Mean queue (s) | vs GPU-PLAS |
|---|---:|---:|---:|---:|
| FCFS | 1,359.25 | 1,601.04 | 1,308.79 | -6.24% |
| GPU-PLAS | 1,279.43 | 1,597.36 | 1,228.97 | baseline |
| UT-LAS | 1,308.16 | 1,597.36 | 1,257.70 | -2.25% |
| Oracle Token-SUF | 1,038.74 | 1,572.69 | 988.28 | +18.81% |
| Oracle UT-SUF | 929.03 | 1,570.66 | 878.57 | +27.39% |
| Oracle Runtime SJF | 827.09 | 1,495.14 | 776.63 | +35.35% |

### 64 programs

| Scheduler | Mean TTLT (s) | P95 TTLT (s) | Mean queue (s) | vs GPU-PLAS |
|---|---:|---:|---:|---:|
| FCFS | 2,523.35 | 3,007.41 | 2,475.93 | -4.89% |
| GPU-PLAS | 2,405.78 | 3,005.11 | 2,358.37 | baseline |
| UT-LAS | 2,435.64 | 3,006.04 | 2,388.22 | -1.24% |
| Oracle Token-SUF | 1,885.23 | 2,952.56 | 1,837.81 | +21.64% |
| Oracle UT-SUF | 1,699.81 | 2,964.15 | 1,652.39 | +29.34% |
| Oracle Runtime SJF | 1,527.42 | 2,806.49 | 1,480.00 | +36.51% |

Throughput is unchanged within each workload because the simulator has one
work-conserving worker, zero tool waits, burst arrivals, and identical physical
service under every policy.

## Workload and signal characterization

| Programs | Turns | Output tokens | 128-token capped turns | Mean selected UT | Maximum input tokens |
|---:|---:|---:|---:|---:|---:|
| 16 | 66 | 5,200 | 30 | 1.999 | 2,619 |
| 32 | 120 | 10,277 | 63 | 2.038 | 2,619 |
| 64 | 240 | 19,425 | 115 | 2.019 | 2,770 |

At 64 programs, selected-UT/runtime Pearson correlation is 0.9859 and Spearman
correlation is 0.8779. Tie-aware READY-pair ordering accuracy under the GPU-PLAS
schedule is 74.20% for selected UT and 76.99% for output tokens.

Output tokens remain the stronger simple correlational proxy. Nevertheless,
Oracle UT-SUF improves mean TTLT more than Oracle Token-SUF at all three sizes,
primarily because selected depth provides ordering information among the many
turns tied at the 128-token cap. The UT advantage over the token oracle therefore
requires confirmation at a higher output limit.

## Interpretation

The evidence now supports two separate conclusions for this fixed-compute setup:

1. **Raw cumulative selected UT is not a better attained-service unit.** It is
   counterfactual metadata rather than the GPU service physically consumed, so
   replacing GPU time with it slightly worsens mean completion time.
2. **Future selected-UT workload remains a promising size signal.** Perfect
   upcoming-turn UT knowledge consistently recovers roughly four-fifths of the
   available runtime-SJF headroom.

This is a stable size-scaling trend for one nested seed, not statistical proof.
Before training a predictor, the next validation should use multiple independent
seeds and a higher generation cap to remove the dominant token-length ties.

## Artifacts

- `results/program_scheduling/bfcl64_seed20261026/trace.json`
- `results/program_scheduling/bfcl64_seed20261026/programs32/summary.json`
- `results/program_scheduling/bfcl64_seed20261026/programs64/summary.json`
- Per-size plots are stored alongside each summary.
