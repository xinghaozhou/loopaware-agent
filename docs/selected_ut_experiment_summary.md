# Selected-UT Trajectory Experiments: Precise Summary

## Research question

The current goal is to test whether `ByteDance/Ouro-2.6B-Thinking` assigns
heterogeneous recurrent depth to individual output tokens during complete,
multi-turn agent trajectories. Batching and scheduling are intentionally out of
scope for this conclusion.

The experiments answer the following narrow question:

> When Ouro always executes a fixed number of Universal Transformer (UT) steps,
> what UT step would its learned halting gate have selected for each generated
> token at a specified cumulative-exit threshold?

This is a counterfactual measurement. It does not reduce computation or measure
an adaptive-decoding speedup.

## Model and tracer definition

- Model: `ByteDance/Ouro-2.6B-Thinking`
- Model revision: `f1edd81e7ac41355db670500ceaf204e0f73af68`
- Precision: BF16
- Decoding: greedy, batch size 1, at most 512 new tokens per request
- Primary executed recurrence: fixed UT=6
- Reference recurrence: fixed UT=4 on two SWE-bench trajectories
- Device: one NVIDIA A40 with approximately 46 GiB VRAM

The checkpoint is loaded with the vendored Ouro-HF implementation. The runner
reads `model.config.num_hidden_layers` after loading; it reported **48** in every
completed condition. No analysis hard-codes 48. For each token:

```text
executed_decoder_layer_calls = executed_ut_steps * runtime_num_hidden_layers
selected_decoder_layer_calls = selected_ut_step * runtime_num_hidden_layers
```

If the learned conditional halt probability at recurrence step `k` is `h_k`,
the tracer converts the gates to an exit distribution:

```text
p_1 = h_1
p_k = h_k * product_{j<k}(1 - h_j), for k before the final configured step
p_K = product_{j<K}(1 - h_j)
```

The last configured step therefore receives all remaining probability mass. At
threshold `q`, selected UT is the first `k` for which
`sum_{j<=k} p_j >= q`. Expected UT is `sum_k k*p_k`. The model still executes
all `K` steps and decodes from step `K`; selection never changes the generated
token. One trace event is captured from the final sequence position of each
autoregressive forward pass and then paired with that pass's generated token.

## Exact replay semantics

These are **fixed-history, free-generation replays**:

1. Take the recorded message prefix immediately before an assistant turn.
2. Serialize that prefix with the Ouro chat template and a generation prompt.
3. Let Ouro generate up to 512 tokens while tracing every generated token.
4. Save the generated output's UT trace, but do not add that output to history.
5. Add the dataset's original assistant action and following observation.
6. Repeat for the next recorded assistant turn.

Consequently, later requests use the correct recorded trajectory history, not
Ouro's earlier generations. The experiment preserves realistic context growth
and explicit turn boundaries while avoiding on-policy divergence. It does not
teacher-force or score the recorded assistant response token by token. The
source messages are preserved, but the request is reserialized for Ouro; it is
not a byte-identical copy of the original model provider's HTTP request.

## Trajectory sources and selection

### SWE-bench Verified

- Dataset: `tarsur385/swebench-verified-trajectories`
- Revision: `773748a7c1222e8a642a7059821498e14293562a`
- Subset: `gpt-5-mini`
- Eligibility: `info.resolved == true`
- Coverage: all 500 SWE-bench Verified task IDs are present; 281 trajectories
  are harness-resolved.
- Native schema: `instance_id`, ordered `messages`, `info.resolved`, agent/model
  configuration, and statistics. Assistant and tool messages define turns.

Six complete, resolved trajectories were selected: four short and two medium,
with 6--14 assistant turns and final contexts of 4,627--8,180 tokens. The
smallest resolved trajectory with at least 26 turns reaches 15,881 input tokens.
An actual UT=6 probe at 16K exhausted the A40, so no long trajectory was silently
truncated or partially replayed.

### Math reasoning

- Dataset: `mai-ll/ipw-sft-trajectories`
- Revision: `fba4d8a229d244de458d36edf26d93c68c9681e7`
- File: `all_correct_trajectories_deduped.jsonl`
- Eligibility: configured records must have `success == true`; the source file
  is explicitly the all-correct split.
- Native schema used: `sample_id`, `success`, `category`, and ordered
  `conversations`.

Five complete trajectories were selected, containing 6--7 assistant turns and
final contexts of 1,931--8,114 tokens.

### Web search

- Dataset: `jxg25/EntiWeave`
- Revision: `20f4c223b8ddec46afc92a764275acc77c528fdb`
- Files: both `data/train_sft-*.parquet` shards
- Native schema used: `id`, JSON-encoded `trajectory`, and `answer`.
- Adapter: an assistant `thought` followed by a `tool_call` is merged into one
  model-facing assistant message; tool observations remain subsequent messages.

Five complete trajectories were selected, containing 7--12 assistant turns and
final contexts of 3,193--8,025 tokens. These records are complete and contain an
answer, but this pilot did not run an independent web-task success verifier.
They must not be described as benchmark-verified in the same sense as the SWE
sample.

## Completed runs and integrity

| Run | Trajectories | Requests | Generated tokens | Errors |
|---|---:|---:|---:|---:|
| SWE UT=6 | 6 | 61 | 23,829 | 0 |
| SWE UT=4 reference | 2 | 22 | 7,817 | 0 |
| Math UT=6 | 5 | 31 | 14,071 | 0 |
| Web-search UT=6 | 5 | 48 | 18,474 | 0 |
| **All conditions** | **18 condition-trajectories** | **162** | **64,191** | **0** |
| **UT=6 only** | **16 trajectories** | **140** | **56,374** | **0** |

Every request has one unique summary and one raw token-trace file, and saved raw
token counts match the request summaries. The SWE UT=6 run used 1.41 sequential
GPU-hours and peaked at 23.8 GiB allocated. Math used 0.812 GPU-hours and peaked
at 24.0 GiB; web used 1.074 GPU-hours and peaked at 23.15 GiB.

## Results

### Cross-domain threshold sensitivity

All threshold rows for a domain are offline re-evaluations of the **same UT=6
generated tokens**. Raising the threshold does not rerun or alter decoding.

| Domain | Threshold | Mean selected UT | Selected-UT distribution | Saving vs. UT=6 |
|---|---:|---:|---|---:|
| SWE | 0.6 | 3.326 | UT3 68.74%, UT4 29.94%, UT5 1.32%, UT6 <0.01% | 44.6% |
| SWE | 0.7 | 3.528 | UT3 55.11%, UT4 37.07%, UT5 7.71%, UT6 0.11% | 41.2% |
| SWE | 0.8 | 4.328 | UT4 68.69%, UT5 29.83%, UT6 1.48% | 27.9% |
| SWE | 0.9 | 5.313 | UT5 68.74%, UT6 31.26% | 11.5% |
| Math | 0.6 | 3.183 | UT3 81.79%, UT4 18.17%, UT5 0.05% | 47.0% |
| Math | 0.7 | 3.478 | UT3 52.97%, UT4 46.23%, UT5 0.80% | 42.0% |
| Math | 0.8 | 4.180 | UT4 82.05%, UT5 17.89%, UT6 0.06% | 30.3% |
| Math | 0.9 | 5.175 | UT4 0.01%, UT5 82.47%, UT6 17.53% | 13.7% |
| Web | 0.6 | 3.431 | UT3 57.86%, UT4 41.18%, UT5 0.96% | 42.8% |
| Web | 0.7 | 3.696 | UT3 37.59%, UT4 55.27%, UT5 7.07%, UT6 0.06% | 38.4% |
| Web | 0.8 | 4.434 | UT3 0.01%, UT4 57.72%, UT5 41.12%, UT6 1.16% | 26.1% |
| Web | 0.9 | 5.419 | UT4 0.01%, UT5 58.04%, UT6 41.95% | 9.7% |

The expected monotonic threshold effect is present in all three domains: a
higher cumulative-exit threshold produces a higher mean selected UT. At the
working threshold 0.7, output tokens use multiple selected depths in every math
and web request. Adjacent generated tokens change selected depth 34.7% of the
time in math and 39.1% in web. SWE also has substantial aggregate heterogeneity,
with 55.11%/37.07%/7.71%/0.11% of tokens at UT3/4/5/6.

At threshold 0.9, the UT6 bucket contains 17.53% of math, 31.26% of SWE, and
41.95% of web tokens. Because UT6 is the forced final residual bucket, these
values are right-censored: an UT=8 diagnostic could split that tail, but such a
split alone would not demonstrate that depth tracks token difficulty.

### Change as recorded context grows

| Domain | Threshold | Corr(input tokens, request mean UT) | Mean first turn | Mean final turn |
|---|---:|---:|---:|---:|
| SWE | 0.7 | -0.617 | 3.926 | 3.221 |
| Math | 0.7 | -0.070 | 3.501 | 3.395 |
| Web | 0.7 | -0.322 | 3.810 | 3.526 |

All six SWE and all ten cross-domain trajectories individually have negative
input-length/mean-UT correlations, although the math relationship is weak. The
turn-level curves are not monotonic; some intermediate turns return to greater
depth. The negative pooled tendency persists across the tested thresholds, but
it is descriptive, not causal: context length, observation content, response
content, and trajectory progress all change together.

The fixed-history design also supplies the successful recorded actions and tool
observations to later turns. Later contexts may therefore make the next action
more constrained or easier. This is a plausible explanation for declining UT,
not a fact established by the experiment.

### Native UT=4 reference

Ouro is natively described/trained with four recurrent steps (R4); UT=6 is an
inference-time extrapolation supported by the implementation. On two SWE
trajectories, the UT=4 reference generated 7,817 tokens:

- UT3: 4,522 tokens (57.85%)
- UT4: 3,295 tokens (42.15%)
- mean selected UT: 3.422
- theoretical saving relative to four executed steps: 14.5%

The corresponding UT=6 mean on those contexts was 3.442. This is not an
identical-token comparison because UT=4 and UT=6 can generate different text
and output lengths.

## What the evidence establishes

The current experiments establish that:

1. The learned gate produces clear token-level selected-depth heterogeneity in
   complete multi-turn SWE, math, and web-search trajectories.
2. The aggregate distribution depends on domain; at threshold 0.7, web has the
   largest mean selected UT (3.696), followed by SWE (3.528) and math (3.478).
3. Stricter thresholds monotonically increase selected UT as required by the
   stopping rule.
4. Larger recorded context does **not** produce increasing mean selected UT in
   these samples; the observed association is negative in all three domains.

The experiments do **not** yet establish that:

1. selected UT tracks independently measured token difficulty;
2. hard output tokens receive more recurrence than easy output tokens;
3. context length causes selected UT to decrease;
4. the learned gate is an optimal adaptive-exit policy;
5. the counterfactual saving becomes a real latency or FLOP saving; or
6. increasing configured recurrence beyond the native R4 regime improves the
   quality of the depth decision.

The key missing variable is an independent, token-level difficulty label.
Context length and turn index are not substitutes for token difficulty. The
present result is therefore evidence of **heterogeneity**, not yet evidence of
**difficulty-aware heterogeneity**.

## Recommended next experiment

Before expanding to UT=8 or batching, run a native UT=4, teacher-forced token
difficulty diagnostic over recorded target responses. For every target token
and each UT step, compute:

- target-token negative log-likelihood and probability;
- predictive entropy and top-1 margin;
- earliest step whose argmax agrees with the final-step prediction;
- KL divergence from intermediate to final-step distributions;
- target-probability improvement from UT1 to UT4; and
- token class (punctuation/common word/number/operator/identifier/tool argument).

Then test whether the gate-selected UT rises with shallow-step surprisal and
with prediction-stabilization depth. Three outcomes are distinguishable:

- both correlate: evidence supports difficulty-aware learned halting;
- trajectory stabilization correlates but the gate does not: recurrent states
  contain useful difficulty information, but the learned gate is a weak readout;
- neither correlates: this checkpoint/evaluation does not reproduce the claimed
  token-difficulty behavior.

Only after this test should UT=8 be used to inspect the high-threshold censored
tail. To keep that comparison interpretable, execute eight recurrent steps but
hold the decoded output to a fixed reference step where feasible.

## Reproduction

From the repository root:

```bash
uv sync

uv run python experiments/trajectory/supervise_fixed_replay.py \
  --config experiments/trajectory/configs/pilot_6.json \
  --run-dir results/trajectory_pilot6_20261007 \
  --model-path <LOCAL_MODEL_PATH>

uv run python experiments/trajectory/analyze_fixed_replay.py \
  --run-dir results/trajectory_pilot6_20261007

uv run python experiments/trajectory/supervise_fixed_replay.py \
  --config experiments/trajectory/configs/crossdomain_pilot10.json \
  --run-dir results/crossdomain_pilot10_20261007 \
  --model-path <LOCAL_MODEL_PATH>

uv run python experiments/trajectory/analyze_crossdomain_replay.py \
  --run-dir results/crossdomain_pilot10_20261007
```

Detailed SWE turn-by-turn tables remain in
`results/trajectory_pilot6_20261007/RESULTS.md`; detailed math/web tables remain
in `results/crossdomain_pilot10_20261007/RESULTS.md`. Result directories are
local artifacts and are gitignored; this document is the tracked canonical
summary.
