# Loop-Aware Agent Trajectory Experiment Overview

> The canonical cross-domain summary, exact interpretation boundaries, and
> recommended token-difficulty follow-up are in
> [`selected_ut_experiment_summary.md`](selected_ut_experiment_summary.md).

## Objective

This study characterizes counterfactual recurrent-depth demand during realistic,
multi-turn software-engineering agent workloads. Its purpose is to determine
whether naturally occurring selected-UT heterogeneity is large and structured
enough to justify later work on recurrence-aware batching.

This phase does **not** implement a recurrence predictor, scheduler, vLLM
integration, global queue, or on-policy agent.

The analysis hierarchy is:

```text
task -> turn/request -> generated token -> selected UT step
```

## Approved trajectory source

Use the public Hugging Face dataset:

```text
tarsur385/swebench-verified-trajectories
subset: gpt-5-mini
filter: info.resolved == true
```

The inspected subset contains:

- 500 trajectories matching all 500 SWE-bench Verified task IDs;
- 281 harness-resolved trajectories;
- 4,994 assistant turns across the resolved trajectories;
- 6–51 turns per resolved trajectory, with a mean of 17.8 and median of 16;
- 25 resolved trajectories with 5–10 turns;
- 217 resolved trajectories with 11–25 turns;
- 39 resolved trajectories with more than 25 turns.

Each native `mini-swe-agent-1.1` record contains an `instance_id`, ordered
`messages`, a `resolved` outcome, the mini-SWE-agent version, model statistics,
and the full agent/model/environment configuration. Assistant and tool messages
provide explicit turn boundaries and tool-call/result pairing.

## Replay semantics

The experiment is fixed-trajectory replay, also described as teacher-forced
workload characterization.

For every assistant turn:

1. Form the request from the recorded message prefix before that assistant turn.
2. Serialize the preserved roles and content with the target Ouro chat format.
3. Run Ouro autoregressive generation and collect token-level UT traces.
4. Discard Ouro's generated response for trajectory progression.
5. Advance with the original recorded assistant message and tool observation.
6. Repeat for the next recorded turn.

Ouro is therefore not controlling the agent environment. The task trajectory,
actions, observations, and subsequent contexts remain those of the recorded
successful run.

The dataset preserves the ordered model-facing message content, agent templates,
observation template, and tool calls. It does not preserve a byte-for-byte copy
of the original provider HTTP request or a standalone formal JSON schema for the
bash tool. The replay is exact with respect to recorded message content and
history, followed by target-model serialization for Ouro.

## Target model and recurrence conditions

Target model:

```text
ByteDance/Ouro-2.6B-Thinking
```

Primary condition:

- `total_ut_steps = 6`
- counterfactual trace threshold `= 0.7`
- greedy decoding
- batch size `= 1`
- maximum new tokens `= 512`
- fixed-final-UT decoding

Reference condition for approximately 10 trajectories:

- `total_ut_steps = 4`
- all other settings identical

The threshold affects only counterfactual `selected_ut_step`. Ouro-HF continues
to execute all configured recurrent iterations, and output logits come from the
fixed final UT step.

The decoder-layer count must never be hard-coded. After loading the checkpoint,
the runner will read and validate:

```python
num_decoder_layers = model.config.num_hidden_layers
```

The checkpoint inspected on the current machine reports 48 layers, but the
runtime value remains authoritative. Derived values use:

```text
executed_decoder_layer_calls = executed_ut_steps * num_decoder_layers
selected_decoder_layer_calls_counterfactual = selected_ut_step * num_decoder_layers
```

## Token-level trace

Every generated token will record at least:

```json
{
  "task_id": "...",
  "request_id": "...",
  "turn_idx": 0,
  "token_idx": 0,
  "token_id": 0,
  "input_token_count": 0,
  "output_token_count_final": 0,
  "num_decoder_layers": 0,
  "executed_ut_steps": 6,
  "selected_ut_step": 4,
  "decode_ut_step": 6,
  "executed_decoder_layer_calls": 0,
  "selected_decoder_layer_calls_counterfactual": 0,
  "gate_probability_by_ut": [],
  "exit_probability_by_ut": [],
  "cumulative_exit_probability_by_ut": []
}
```

## Aggregation and analysis

Per request/turn, compute token counts, input length, selected/executed UT totals,
mean, standard deviation, P50/P90/P95, range, decoder-layer-call totals, and:

```text
theoretical_recurrence_saving = 1 - selected_ut_total / executed_ut_total
```

Per task, analyze input length, mean/total selected UT, and theoretical saving by
both raw turn index and normalized task progress. Measure relationships with
context length, turn index, and normalized progress.

After tracing, simulate candidate concurrent batches at `B = 4, 8, 16` and token
horizons `H = 4, 8, 16, 32`. Compare random/FCFS grouping with oracle grouping
using known future selected-UT profiles. Do not train a predictor in this phase.

## Workload size and context distribution

The 281 resolved trajectories contain 4,994 replay requests. Approximate context
lengths, obtained by preserving recorded roles/content/tool calls and tokenizing
with the Ouro tokenizer, are:

- mean: 10,226 tokens;
- median: 9,199 tokens;
- P90: 18,460 tokens;
- P95: 22,480 tokens;
- maximum: 49,273 tokens.

Approximate request counts by context range:

| Context range | Requests |
|---|---:|
| `<4K` | 908 |
| `4K–8K` | 1,234 |
| `8K–16K` | 2,089 |
| `16K+` | 763 |

Final counts may change slightly after exact Ouro chat serialization is defined.

## Current-device timing and memory

Measurements were taken on one NVIDIA A40 with approximately 46 GiB VRAM, using
Ouro-2.6B, UT=6, BF16, fixed-final-UT decoding, and batch size 1.

| Input context | 8-token elapsed time | Peak allocated | Peak reserved |
|---:|---:|---:|---:|
| 1,024 | 2.29 s | 7.3 GiB | 8.3 GiB |
| 4,096 | 4.03 s | 14.2 GiB | 19.9 GiB |
| 8,192 | 6.68 s | 23.5 GiB | 34.6 GiB |

Steady decoding is approximately 4–4.5 output tokens per second. Tracing overhead
was within timing noise in a 32-token comparison. A 512-token request is expected
to take approximately 2.0–2.2 minutes when its context fits in memory.

Estimated sequential primary-condition runtime:

| Trajectories | Approximate requests | UT=6 runtime |
|---:|---:|---:|
| 30 | 533 | 18–20 hours |
| 40 | 711 | 24–27 hours |
| 50 | 889 | 30–33 hours |

The approximately 10-trajectory UT=4 reference adds about 4–5 hours. Including
validation, memory tests, and analysis, a 40-trajectory study is expected to take
approximately 28–31 GPU-hours.

These are conservative estimates that assume most generations reach the
512-token limit. Early EOS reduces runtime.

## Memory constraint and initial pilot

UT=6 retains KV-cache state for every recurrent pass. On the current A40, an 8K
input already reserves approximately 34.6 GiB. A 16K request is borderline, and
the observed 49K maximum cannot fit with the current cache implementation.

Before selecting the full 30–50 trajectories:

1. Validate exact replay serialization and token alignment on one request.
2. Test memory progressively near 8K, 10K, 12K, and then 16K only if safe.
3. Establish and record the safe context cutoff.
4. Report excluded requests rather than silently truncating them.

The approved first run on the unchanged A40 is a **six-trajectory pilot**:

- four short trajectories and two medium trajectories;
- complete trajectories only, with exact Ouro-serialized maximum contexts from
  4,627 through 8,180 tokens;
- UT=6 for all six (61 requests total);
- UT=4 reference runs for two of the six (22 requests total).

The progressive probe succeeded through 12K input tokens, but an actual 16K
UT=6 request exhausted the A40. The smallest complete resolved long trajectory
(26 or more turns) reaches 15,881 tokens, so no long trajectory can honestly be
included without truncation on this device. This limitation is recorded rather
than silently changing the workload.

The 83-request pilot should take about **2.5–3 GPU-hours** under the conservative
assumption that most generations reach the 512-token cap. It is large enough
to validate replay correctness, output schemas, uninterrupted artifact writing,
context growth, and summary/oracle-analysis code before committing to a day-long
run.

## Reproducibility and output policy

Every run will save, without overwriting earlier results:

- raw token traces;
- per-request summaries;
- per-task summaries;
- batch-heterogeneity results;
- oracle-grouping results;
- the exact experiment configuration;
- target model identifier and revision;
- trajectory dataset identifier and revision;
- runtime `num_hidden_layers`;
- Git commit hash;
- peak GPU memory and elapsed time.

The full 30–50 trajectory experiment will not start until the replay adapter,
stratified sample, example replay, exact commands, memory cutoff, and revised
runtime estimate have been reviewed.

## Completed pilot results

The six-trajectory pilot completed on 2026-10-07 with all 83 planned requests
and no errors. The run produced 61 UT=6 requests across six complete
trajectories and 22 UT=4 reference requests across two of those trajectories.
The raw traces contain 31,646 generated tokens in total. There is one raw trace
file for every request, all request keys are unique, and every raw token count
matches its request summary.

At runtime, `model.config.num_hidden_layers` reported **48** for both
conditions. The primary condition executed exactly six UT iterations for every
token, and the reference condition executed exactly four. The layer count and
all executed/selected decoder-layer-call fields were derived from those runtime
values rather than a hard-coded constant.

### Primary UT=6 selected-depth distribution

The UT=6 condition generated 23,829 output tokens. At the 0.7 cumulative-exit
threshold, its counterfactual selected-UT distribution was:

| Selected UT | Tokens | Percentage |
|---:|---:|---:|
| 3 | 13,132 | 55.11% |
| 4 | 8,833 | 37.07% |
| 5 | 1,837 | 7.71% |
| 6 | 27 | 0.11% |

The mean selected depth was **3.528 UT**, compared with six executed UT steps.
This corresponds to a **41.2% counterfactual recurrence saving**. With 48
runtime decoder layers, the mean selected work is approximately 169.3 decoder
layer calls per token, versus 288 calls actually executed by fixed-UT decoding.

### Exit-threshold sensitivity

Because every token trace stores the complete exit distribution, thresholds can
be evaluated offline on the identical 23,829 UT=6 output tokens. The mean below
is the threshold-selected depth, not a new decoding run:

| Threshold | Mean selected UT | UT distribution | Saving vs. UT=6 | Corr(input, UT) | First-turn mean | Final-turn mean |
|---:|---:|---|---:|---:|---:|---:|
| 0.6 | 3.326 | UT3 68.74%, UT4 29.94%, UT5 1.32%, UT6 <0.01% | 44.6% | -0.615 | 3.639 | 3.077 |
| 0.7 | 3.528 | UT3 55.11%, UT4 37.07%, UT5 7.71%, UT6 0.11% | 41.2% | -0.617 | 3.926 | 3.221 |
| 0.8 | 4.328 | UT4 68.69%, UT5 29.83%, UT6 1.48% | 27.9% | -0.615 | 4.645 | 4.071 |
| 0.9 | 5.313 | UT5 68.74%, UT6 31.26% | 11.5% | -0.618 | 5.627 | 5.071 |

As expected, a stricter threshold monotonically shifts selection toward later
UT steps. More importantly, the correlation with growing input context remains
stable at approximately -0.62 for every threshold, and the final-turn mean is
lower than the first-turn mean in every condition. The observed context-growth
tendency is therefore not specific to the originally chosen 0.7 threshold.

Separately, the mean probabilistic expectation of the learned exit distribution
is **3.154 UT** (median 2.816). This expectation is threshold-independent;
threshold selection converts that distribution into a discrete stopping rule.

### Variation over growing multi-turn contexts

Selected depth generally decreased as the recorded trajectory context grew:

- correlation of request mean selected UT with input-token count: **-0.617**;
- correlation with raw turn index: **-0.659**;
- correlation with normalized trajectory progress: **-0.714**;
- unweighted mean across the six first turns: **3.926 UT**;
- unweighted mean across the six final turns: **3.221 UT**.

All six complete trajectories independently showed a negative relationship:

| SWE-bench task | Turns | Input tokens, first→last | Mean UT, first→last | Corr(input, UT) |
|---|---:|---:|---:|---:|
| `django__django-11451` | 6 | 1,684→6,841 | 3.859→3.164 | -0.908 |
| `pytest-dev__pytest-6202` | 10 | 2,035→8,163 | 4.066→3.344 | -0.558 |
| `scikit-learn__scikit-learn-14141` | 14 | 1,195→7,648 | 3.910→3.268 | -0.703 |
| `sphinx-doc__sphinx-8475` | 8 | 1,365→4,627 | 3.850→3.239 | -0.686 |
| `sphinx-doc__sphinx-9258` | 13 | 1,300→8,180 | 3.986→3.163 | -0.821 |
| `sympy__sympy-16886` | 10 | 1,212→5,971 | 3.887→3.148 | -0.781 |

The turn-level series are not monotonic: several trajectories temporarily
return to higher selected depth at intermediate turns. Context length and task
progress therefore carry signal, but do not fully determine token-level depth.
The relationship is descriptive rather than causal because turn content,
response length, and context size change together in this replay design.

### UT=4 reference

The UT=4 reference generated 7,817 tokens and selected UT=3 for 4,522 tokens
(57.85%) and UT=4 for 3,295 tokens (42.15%). Its mean selected depth was 3.422
and its counterfactual recurrence saving was 14.5%. On the same two trajectory
contexts, the UT=6 condition's mean selected depth was 3.442.

The UT=4 and UT=6 decoders can generate different output text and lengths, so
this is a paired-context condition comparison, not an identical-output-token
comparison.

### Candidate batch heterogeneity

With only six primary trajectories, honest cross-task synchronized batches are
available for `B=4` but not `B=8` or `B=16`. Across ten synchronized B=4
batches, recurrence utilization declined as the lookahead horizon grew:

| Horizon | FCFS variance | Oracle variance | FCFS utilization | Oracle utilization |
|---:|---:|---:|---:|---:|
| 4 | 0.019 | 0.014 | 0.979 | 0.985 |
| 8 | 0.099 | 0.079 | 0.944 | 0.954 |
| 16 | 0.123 | 0.111 | 0.934 | 0.941 |
| 32 | 0.218 | 0.196 | 0.888 | 0.900 |

The pilot shows measurable cross-request recurrence heterogeneity, although the
oracle grouping advantage is modest at this scale. A larger trajectory sample
is required to evaluate B=8/B=16 and to estimate scheduling gains reliably.

### Runtime and scope

The UT=6 requests consumed 1.41 sequential GPU-hours and peaked at 23.8 GiB
allocated memory. The UT=4 reference consumed 0.31 GPU-hours and peaked at
17.0 GiB. The reported saving is counterfactual: Ouro-HF still executed fixed
final-UT decoding during measurement.

These results cover complete resolved trajectories whose contexts fit the
unchanged A40. They do not include the long-trajectory bin, because its smallest
candidate reached 15,881 input tokens and the UT=6 16K memory probe failed. No
trajectory context was truncated. Replay is teacher-forced: each next request
uses the original successful assistant action and tool observation, not Ouro's
newly generated output.
