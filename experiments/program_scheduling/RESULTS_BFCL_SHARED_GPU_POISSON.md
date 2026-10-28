# Physical Shared-GPU BFCL Poisson Scheduling

All values are mean +/- sample standard deviation across 2 arrival seeds.

| Load | Policy | Mean TTLT (s) | P95 TTLT (s) | Mean queue (s) | Makespan (s) | Programs/s | GPU util | Batch fill | Mean ready queue |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MEDIUM | FCFS | 104.330 +/- 22.126 | 181.607 +/- 52.864 | 27.257 +/- 17.871 | 1972.068 +/- 393.716 | 0.033 +/- 0.007 | 90.007 +/- 9.235% | 70.723 +/- 10.810% | 0.962 +/- 0.772 |
| MEDIUM | AUTELLIX_PLAS | 102.853 +/- 23.171 | 223.450 +/- 110.574 | 28.101 +/- 19.232 | 1961.791 +/- 404.908 | 0.033 +/- 0.007 | 89.354 +/- 9.397% | 69.529 +/- 10.751% | 1.003 +/- 0.834 |
| MEDIUM | UT_LAS | 101.774 +/- 22.810 | 215.022 +/- 108.505 | 26.988 +/- 17.902 | 1962.801 +/- 403.480 | 0.033 +/- 0.007 | 89.452 +/- 9.828% | 69.474 +/- 11.248% | 0.960 +/- 0.781 |
| MEDIUM | ORACLE_TOKEN_SUF | 106.867 +/- 26.114 | 174.077 +/- 58.445 | 28.346 +/- 19.728 | 1983.299 +/- 391.875 | 0.033 +/- 0.007 | 90.706 +/- 9.790% | 71.141 +/- 12.188% | 0.997 +/- 0.834 |
| MEDIUM | ORACLE_UT_SUF | 107.643 +/- 27.397 | 227.369 +/- 141.889 | 29.968 +/- 21.153 | 1978.921 +/- 398.503 | 0.033 +/- 0.007 | 90.815 +/- 9.131% | 70.512 +/- 12.798% | 1.060 +/- 0.897 |
| MEDIUM | ORACLE_RUNTIME_SJF | 113.510 +/- 35.731 | 208.999 +/- 98.953 | 32.430 +/- 25.941 | 1988.573 +/- 383.722 | 0.033 +/- 0.006 | 92.040 +/- 10.490% | 72.243 +/- 14.423% | 1.146 +/- 1.056 |

The JSON artifact contains paired-seed policy comparisons and every raw run path.

## Pilot finding

UT-LAS had the lowest mean program TTLT in both seeds. Its paired mean TTLT
improvement was 2.58% over FCFS and 1.04% over AUTELLIX-PLAS. With only two
seeds, this is a pilot result rather than a statistically conclusive finding.

The machine verifier passed for every run: the configured maximum batch size
was four, the maximum number of overlapping physical batches was one, and all
refills occurred only after the active static batch completed.

Reproducibility data are under `shared_gpu_poisson_artifacts/`: the aggregate
summary, exact `(lambda, seed)` arrival traces, and raw results for all 12 runs.
