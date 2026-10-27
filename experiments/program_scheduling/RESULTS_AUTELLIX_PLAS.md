# Autellix-Style PLAS on BFCL V3

## Scope

This experiment implements the core single-threaded PLAS behavior from the
[Autellix paper](https://arxiv.org/abs/2502.13965): program-level inherited
attained model service, discrete priority queues, FCFS within each queue,
quantum exhaustion and demotion, program-level anti-starvation, and
work-conserving batch filling. It is not a reproduction of the complete
Autellix serving system or its paper results.

The old baseline is retained and renamed `GPU_PLAS_PROXY`. It non-preemptively
selects the READY program with the exact smallest cumulative GPU service. The
new policy is `AUTELLIX_PLAS`.

## Configuration

The paper describes the queue algorithm but does not publish exact boundaries,
quanta, beta, or scheduling interval. Defaults are therefore configurable. This
run uses geometric values consistent with the available
[independent implementation](https://github.com/Crystalclear9/Autellix), scaled
to the BFCL trace's millisecond service unit:

```text
boundaries: 0, 2s, 4s, 8s, 16s, 32s, 64s, infinity
quanta:        1s, 2s, 4s,  8s, 16s, 32s, 64s
anti-starvation beta: 8
```

The workload is the unchanged nested seed-`20261026` BFCL trace. Model service
is `full_compute_gpu_time_ms` from the existing Ouro fixed-UT=4 trace. Tool wait
never contributes to attained service. A preempted call resumes with its
remaining service. The simulator assumes perfect resume and does not add KV
swap or scheduler overhead.

## Results

### 32 programs

| Batch | Policy | Mean TTLT | Median TTLT | P95 TTLT | Mean queue | Programs/s | Makespan | Preemptions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | FCFS | 341.35 s | 372.01 s | 404.00 s | 290.89 s | 0.07866 | 406.82 s | 0 |
| 4 | GPU-PLAS-PROXY | **322.93 s** | 328.37 s | 404.91 s | **272.47 s** | **0.07822** | **409.08 s** | 0 |
| 4 | AUTELLIX-PLAS | 331.29 s | **325.17 s** | **403.84 s** | 280.83 s | 0.07768 | 411.97 s | 483 |
| 8 | FCFS | 172.02 s | 185.20 s | **204.53 s** | 121.56 s | **0.15236** | **210.03 s** | 0 |
| 8 | GPU-PLAS-PROXY | **164.11 s** | **167.22 s** | 210.19 s | **113.65 s** | 0.14820 | 215.92 s | 0 |
| 8 | AUTELLIX-PLAS | 170.01 s | 171.57 s | 206.22 s | 119.55 s | 0.15218 | 210.27 s | 159 |

### 64 programs

| Batch | Policy | Mean TTLT | Median TTLT | P95 TTLT | Mean queue | Programs/s | Makespan | Preemptions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | FCFS | 633.09 s | 676.66 s | 756.47 s | 585.67 s | 0.08398 | 762.05 s | 0 |
| 4 | GPU-PLAS-PROXY | **604.39 s** | **625.49 s** | **755.26 s** | **556.97 s** | 0.08343 | 767.11 s | 0 |
| 4 | AUTELLIX-PLAS | 626.57 s | 646.91 s | 756.17 s | 579.15 s | **0.08405** | **761.47 s** | 2,910 |
| 8 | FCFS | 318.23 s | 338.47 s | 379.88 s | 270.81 s | **0.16656** | **384.24 s** | 0 |
| 8 | GPU-PLAS-PROXY | **304.31 s** | 313.48 s | 382.46 s | **256.89 s** | 0.16580 | 386.01 s | 0 |
| 8 | AUTELLIX-PLAS | 308.97 s | **308.59 s** | **379.44 s** | 261.55 s | 0.16596 | 385.63 s | 954 |

## Answer

Yes, faithful PLAS behaves differently from the exact global-sort proxy.

- It preempts and demotes long calls, while the proxy is non-preemptive.
- Queue discretization intentionally gives similarly attained programs equal
  priority and FCFS ordering rather than continuously re-sorting them.
- Across these four cases, AUTELLIX-PLAS mean TTLT is 1.53%--3.67% worse than
  GPU-PLAS-PROXY with the chosen defaults.
- AUTELLIX-PLAS sometimes improves median or P95 latency and usually brings
  makespan closer to FCFS, so its behavior is not merely a noisy proxy copy.

The exact-service proxy has an informational advantage: it continuously sorts
by precise measured service, whereas real PLAS deliberately bins priorities to
limit churn. This trace simulator also assigns zero cost to preemption, cache
swapping, and scheduling. The results should therefore be used to validate
scheduler behavior and establish a baseline, not to claim paper-equivalent
performance.

## Synthetic validation

The four-program validation confirms that new programs enter the highest queue,
long calls demote on quantum exhaustion, later calls inherit completed program
service, tool wait does not count as service, and all available batch slots are
filled. Run it with:

```bash
uv run python -m experiments.program_scheduling.validate_autellix_plas
```

The raw BFCL output is local and gitignored at:

```text
results/program_scheduling/autellix_plas_bfcl_seed20261026/results.json
```
