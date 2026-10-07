# Loop-Aware Agent Trajectory Experiment Overview

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
