# Adaptive-Exit KV Semantics for Ouro

## Scope

This implementation adds correctness-first physical early exit during
one-token autoregressive decode. Prompt prefill always executes all configured
UT steps. No threshold sweep was run as part of this work.

The runtime reads `num_hidden_layers` and `total_ut_steps` from the loaded
configuration. The real-checkpoint validation reported 48 decoder layers for
`ByteDance/Ouro-2.6B-Thinking` at revision
`f1edd81e7ac41355db670500ceaf204e0f73af68`.

## Separated policies

The implementations live in `vendor/ouro_hf/cache_ouro.py` behind a small
shared cache interface:

- `FullPerDepthCache`: original Ouro behavior, with one dense history for each
  `(UT depth, decoder layer)` pair. It remains the default and always executes
  all UT steps.
- `LastAvailableCache`: for historical token `j`, recurrence `u` reads the KV
  from `min(u, U_j)`. Prompt tokens have `U_j = U_max` and retain depth-aligned
  KV.
- `FinalExitSharedCache`: every future recurrence reads historical token `j`
  from its exit-depth KV, `KV_j^(U_j)`. Prompt tokens have `U_j = U_max`, so
  all decode recurrences read their final-depth prompt KV.

The current token is transactional. Its KV from the current recurrent step is
used for its own attention, but it is not made historical until its exit depth
is known. Commit then retains every available depth needed by
`last_available`, or only the exit depth for `final_exit_shared`.

## Execution behavior

Configuration fields:

```python
config.kv_cache_policy = "full_per_depth"  # or last_available/final_exit_shared
config.physical_early_exit = False
config.early_exit_threshold = 0.7
```

For an adaptive policy, `physical_early_exit=False` is its run-all semantic
reference: all recurrences execute, logits come from the threshold-selected
state, and only policy-observable KV is committed. With
`physical_early_exit=True`, rows that reach their exit are removed from the
active batch before the next recurrence. Both modes therefore share identical
historical KV semantics.

The first generated token is predicted by full-depth prompt prefill. Physical
exit begins with subsequent one-token decode calls.

Model outputs and the tracer report authoritative `selected_ut_steps` and
`executed_ut_steps`. A physically exited event cannot report a full-distribution
expected UT because gates after its exit were deliberately not evaluated, so
`expected_ut_steps` is `null` for those events.

## Validation

Validation order followed the requested smoke-first sequence:

1. Tiny CPU smoke test: full prefill, depth-2 physical exit, then a depth-3
   token that must resolve missing history. Passed for both adaptive policies.
2. Hand-built KV policy oracles. Passed for prompt and decoded-token lookup.
3. Heterogeneous two-row batch cache test. Passed with selected depths 1 and 2.
4. Tiny-model run-all versus physical parity across forced depths and learned
   threshold decisions. Logits and selected depths matched.
5. Hugging Face `generate` integration and the `full_per_depth` regression.
6. Complete project suite: 15 tests passed.
7. Real Ouro-2.6B checkpoint validation at UT=6 and threshold 0.7.

Real-checkpoint results:

| Policy | Selected UT | Run-all executed UT | Physical executed UT | Token parity |
|---|---|---|---|---|
| `last_available` | `[4, 4, 4, 4]` | `[6, 6, 6, 6]` | `[6, 4, 4, 4]` | exact |
| `final_exit_shared` | `[4, 3, 4, 2]` | `[6, 6, 6, 6]` | `[6, 3, 4, 2]` | exact |

The leading physical value of 6 is expected because it corresponds to the
prefill-produced first output token. The two policies need not generate the
same continuation because their historical-KV semantics intentionally differ.
Each physical run matched its own run-all reference exactly.

The machine-readable result is in
`results/adaptive_kv_checkpoint_validation.json`.

## Current limitations

- Adaptive per-token prefill is intentionally not implemented.
- Adaptive continuation accepts exactly one new token per forward call.
- The policy cache favors transparent semantics over optimized kernel-level KV
  gathering. It physically avoids exited decoder work and unnecessary
  committed KV states, but its Python history assembly is not a production
  serving optimization.
- Greedy generation and heterogeneous batched forward execution are tested.
  Beam cache reordering is implemented, but real-checkpoint beam generation has
  not been validated.
- Adaptive policies are not silently applied to an already-populated generic
  Hugging Face cache; that situation raises an error.

These constraints isolate decode-time cache correctness so adaptive prefill and
serving-oriented storage can be evaluated separately later.
