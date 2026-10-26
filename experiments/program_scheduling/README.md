# Full-Compute Ouro Program Scheduling

This experiment isolates program-level scheduling. Ouro physically executes four
recurrent iterations for every generated token. The adaptive-exit gate is traced
at threshold `0.2`, but its selected UT never changes decoding or model state.

The quick validation samples 16 BFCL V3 `multi_turn_base` programs. BFCL's
ground-truth calls are executed with the official tool simulator, and their
results are teacher-forced into later turns. Tool waits are zero because BFCL V3
does not contain timestamps. All policies replay the same fixed trace.

```bash
uv sync
uv run python -m experiments.program_scheduling.generate_traces \
  --output results/program_scheduling/bfcl16/trace.json --resume
uv run python -m experiments.program_scheduling.run_experiment \
  --trace results/program_scheduling/bfcl16/trace.json \
  --output results/program_scheduling/bfcl16/results.json
uv run python -m unittest tests.test_program_scheduling
```

The six policies are FCFS, GPU_PLAS, UT_LAS, ORACLE_TOKEN_SUF,
ORACLE_UT_SUF, and ORACLE_RUNTIME_SJF. Turns are non-preemptive and the
simulator has one GPU worker.
