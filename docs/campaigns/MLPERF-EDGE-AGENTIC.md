---
status: current
owns: Proposed hipEngine MLPerf Inference v6.1 Edge Agentic evaluation, fixed diagnostic subsets, comparison boundaries, and runtime budgeting.
---
# MLPerf Edge Agentic evaluation campaign

## 1. Objective and scope

Evaluate hipEngine through the **MLCommons Edge Agentic harness**, starting on
one Ryzen AI Max+ 395 / gfx1151 Linux host. Measure complete single-user coding
turns and pass the separate Berkeley Function Calling Leaderboard (BFCL) v4
accuracy gate. Start with small, fixed subsets to find integration failures and
estimate full-run cost before committing a device for hours.

The **140-turn paired autoregressive screen** is recorded in
[`benchmarks/mlperf-edge/README.md`](../../benchmarks/mlperf-edge/README.md).
Both engines complete all turns; mean latency is 8.925 s for hipEngine and
9.157 s for llama.cpp on the same gfx1151 host and Qwen3.6-27B Q4_K_M artifact.
Inline command-overlap scores are 0.5586 and 0.6420 respectively. This is one
pass, not a stable speedup or quality-parity finding. Harness TPOT has a
chunk-dependent tokenization defect and cannot support a decode-rate comparison.
The earlier repaired 27-turn integration diagnostic and original failure are
preserved. A compatibility-patched gufo run with the unchanged Qwen3.6 target,
a user-selected Qwen3.8 DFlash2 Q4_K_M draft and FP16 attention KV completes
140/140 turns in 1,041.49 s (mean 7.439 s, inline IoU 0.6387). Actual draft
acceptance is 15.79%; prefix token reuse is 96.55%. It buffers tool output on
132/140 turns. The speculative hipEngine/llama.cpp screens also complete all
140 turns, in 793.15/827.10 s (mean 5.665/5.908 s), with acceptance 86.20%/92.50%
and prefix reuse 95.31%/95.95%. Their separate GGUF container preserves all
original target tensor bytes and adds Qwen3.6 NextN weights; the original file
contains no MTP block. BF16 target KV differs from gufo's FP16 target KV.
The separate fixed BFCL diagnostic completes 96/96 cases in each engine:
hipEngine initially 52 correct, llama.cpp 76, gufo 73. Fixing hipEngine's
OpenAI-incompatible parallel-call opt-in default raises its unchanged-diagnostic
result to 75/96: all 23 newly passing cases are parallel-call cases, with no
pass-to-fail regressions. Llama.cpp and gufo outputs are unchanged. Gufo has three additional
failures versus llama.cpp. These diagnostic scores do not establish accuracy
parity or certify the full gate. A later clean build of halo-box
`strix-llama.cpp@03895887abe6` on Vulkan/RADV completes the same replay in
643.08 s (mean 4.593 s) and passes 77/96 diagnostic cases, using BF16 target KV
and the same target-plus-NextN GGUF. This is a separate engine/backend arm;
existing measurements are unchanged. Full performance and BFCL gates
have not run. Writing this plan
does not start or resume another optimization
loop. Kernel tuning, an NVFP4 port,
Windows bring-up, and an official submission are outside the initial scope.

Primary model: **Qwen3.6-27B Q4_K_M GGUF**, matching the reference model and
quantization. Qwen3.8-27B is a separate optional extension, never a replacement
for the v6.1 model in a comparison row. Missing prior model evidence does not
block loading: attempt the supported path and record any concrete failure.

Success means a reproducible full performance pass, the full prescribed BFCL
sample, both accuracy verdicts, and a report explaining latency and configuration
differences. A slow result or a failed accuracy gate is still a useful campaign
outcome; neither should be hidden or turned into an admission restriction.

## 2. Sources and frozen workload

Sources inspected on 2026-09-24:

- [MLCommons introduction](https://mlcommons.org/2026/07/mlperf-inference-v61-edge-agentic/):
  benchmark rationale, reference model, single-stream workload, and two accuracy
  thresholds. Its `10_Edge_Agentic_Example` path has moved.
- [Pinned example README](https://github.com/mlcommons/endpoints/blob/e71b928f8a72fd0c9d850dc5ddd8fd7760356354/examples/11_Edge_Agentic_Example/README.md),
  [full config](https://github.com/mlcommons/endpoints/blob/e71b928f8a72fd0c9d850dc5ddd8fd7760356354/examples/11_Edge_Agentic_Example/online_edge_full_run.yaml),
  and [configuration schema](https://github.com/mlcommons/endpoints/blob/e71b928f8a72fd0c9d850dc5ddd8fd7760356354/src/inference_endpoint/config/schema.py).
  The discovery pin is `mlcommons/endpoints@e71b928f8a72fd0c9d850dc5ddd8fd7760356354`.
  Before claiming equivalence to an Atlas submission, reconcile this pin with
  that submission's actual harness revision and resolved configuration.
- [StorageReview coverage](https://www.storagereview.com/news/mlperf-inference-v6-1-5-7x-per-accelerator-gains-a-512-gpu-run-and-vera-rubins-first-peer-reviewed-numbers):
  reports Atlas Qwen3.6-27B NVFP4 at 20.1 tok/s on Spark and 19.63 tok/s on Strix,
  with Spark completing the performance replay in under 64 minutes. These are
  attributed external results, not local baselines. Raw submission artifacts and
  the exact token-rate aggregation have not been independently verified here.
- [Atlas README](https://github.com/Atlas-Inf/atlas): Qwen3.8 results use a different
  model and include short-prompt and BFCL-subset measurements. Do not treat those,
  the quoted Windows result, or an HA-20 score as the v6.1 accuracy gate.

### Reference contract

| Axis | Initial campaign setting |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`; reference GGUF from `unsloth/Qwen3.6-27B-GGUF` |
| Served window | 32,768 tokens, not the old Atlas comparison's 256K window |
| Sampling | Temperature 0, seed 42, maximum 1,024 new tokens; reasoning off |
| Load | One worker, one connection, concurrency 1; one pass over 20 trajectories |
| Performance | Recorded SWE-bench-style replay; 1,007 generated turns, not live issue solving |
| Accuracy | BFCL v4 single-turn; `non_live: 62%`, `live: 10%`, `hallucination: 10%`, `subset_floor: 25`, loader seed 42; approximately 995 samples |
| Time limits | 600 seconds per turn; four-hour performance issue cap; six-hour whole-run timeout |

The pinned `agentic_coding_2.5h.jsonl` was inspected directly: **2,014 JSONL rows**
comprise 20 user rows, 1,007 assistant rows, and 987 tool rows. Its SHA-256 is
`b7da8c4ffbe1cabd79c4d8169e5201541e006364b4bfc016d01d893e20ee6f70`.
Rows are not requests; filtering the first 100 lines is not a 100-turn replay.

The introductory article describes recorded inter-turn pauses. In the pinned
code, `AgenticInferenceConfig.inject_tool_delay` defaults to `false`, and the
reference YAML does not override it. The dataset contains 636.392608 seconds of
recorded delays. Preserve the resolved reference setting and report it; do not
silently add or remove delays to improve wall time. Any delay-on experiment is a
separate diagnostic. Likewise, record the resolved salt, ordering, cache, and
streaming settings rather than assuming them from prose.

### Accuracy and metric interpretation

Both BFCL metrics must meet **0.97 times their respective references**:

- Overall reference 86.23%; exact product 83.6431% (displayed upstream as 83.64%).
- Normalized reference 87.96%; exact product 85.3212% (displayed as 85.32%).

Use the pinned scorer and ruleset precision, not rounded display thresholds.
Record category counts, category scores, both aggregate scores, missing samples,
and scorer/dependency revisions. Do not reconstruct a different score from
rounded example tables. A partial BFCL sample cannot certify either gate.

Headline: **mean end-to-end latency per completed turn**, with full-pass wall
time. Retain the harness's time-to-first-token (TTFT), time-per-output-token
(TPOT), turn-latency and input/output token-length distributions, including
p50/p90/p99/max. Preserve original result fields and units. Report decode tok/s
only with its exact denominator and aggregation: inverse mean TPOT, mean inverse
TPOT, and total tokens divided by total wall time are different quantities.
Never count streamed chunks as tokens, especially with speculative decoding or
structured tool calls. Check that the canonical tokenizer populates token
metrics and that structured tool-call output is counted correctly.

The replay's inline executable-call multiset intersection-over-union score
(`scores.json`) is supplementary, not a substitute for BFCL and not a SWE-bench
issue-resolution score. Preserve it and require zero missing turns for a full
valid replay. Do not invent an inline pass threshold absent from the pinned
rules. Empty text with valid tool calls is not empty output; no text **and** no
valid tool output is a failure to investigate, not a speed win.

## 3. Baselines and prior local work

Use three clearly separated comparisons:

1. **Primary engine comparison:** hipEngine versus llama.cpp on the same physical
   gfx1151 host, identical Qwen3.6-27B GGUF bytes and tokenizer, same harness and
   request settings. Record each server's cache, KV precision, execution profile,
   and speculation settings. Include a true autoregressive (AR) arm if making a
   multi-token prediction (MTP) speedup claim.
2. **Optional local Atlas comparison:** same workload and host, Atlas's executable
   NVFP4 recipe versus hipEngine Q4_K_M. Label it cross-quantization, not an
   isolated engine improvement. Recheck Atlas's actual served route before timing.
3. **Published reference context:** Atlas Strix/Spark and upstream Thor results
   stay attributed external rows. Same GPU architecture is not same-host evidence.

Prior [Atlas comparison notes](../../benchmarks/atlas-agent/README.md) and the
[Atlas failure entry](../../worklog/entries/20260920T092257.143222Z-lhl-atlas-mtp-unusable-9785f4.md)
record a local Qwen3.8 MTP buffer-allocation failure while AR served successfully.
That observation does not establish a failure on Qwen3.6, a different build, or
another host. The suggested ROCm cause was a hypothesis, not a diagnosis. A
broken Atlas arm does not block hipEngine or the same-GGUF llama.cpp baseline.

The [cross-engine replay entry](../../worklog/entries/20260921T085132.835613Z-lhl-agentic-session-cross-engine-harness-4b8e21.md)
and `scripts/agentic_session_compare.py` supply useful HTTP fidelity diagnostics,
but use a different transcript and are **not** the official workload. The old
`benchmarks/atlas-agent/http_1to1_bench.py` synthetic multi-turn arm also is not a
replacement for the MLCommons runner.

## 4. Execution stages and deliverables

Setup, the repaired 27-turn diagnostic and the paired 140-turn screen are recorded;
full performance and accuracy stages are pending.
The linked run record preserves the original failures and the passing repeat.
Do not run competing servers or another GPU
campaign on the selected device. Record physical host, device, memory allocation,
power configuration, OS, ROCm/compiler, model revision and hash, tokenizer and
chat-template hashes, server commits/dirtiness, harness pin, installed scoring
dependencies, exact commands, and resolved configurations. Follow
[OPTIMIZATION.md](../OPTIMIZATION.md) for measured claims and any later kernel work.

### Stage 0 — Install and verify the integration

1. Use a separate Python 3.12+ environment for Endpoints and its `[bfcl]` extra;
   do not add evaluator dependencies to hipEngine's torch-free runtime. Read
   dependency/build steps before running them. Keep external peer checkouts
   read-only; use a dedicated campaign checkout for config copies and outputs.
2. Pin the harness to the discovery revision above. Obtain the model and tokenizer
   at immutable revisions, and verify the replay hash. Download time is separate
   from benchmark wall time. For a formal historical comparison, resolve the
   submitted v6.1 revision/configuration first rather than silently using HEAD.
3. Attempt loading Qwen3.6 Q4_K_M through `hipengine serve` on gfx1151, production
   profile, BF16 KV and a 32K window. Use the allocation-first capacity protocol
   in [HARNESSES.md](../../benchmarks/HARNESSES.md) if memory fit is uncertain.
4. Probe `/v1/models`, `/v1/hipengine/capabilities`, and a real streamed chat request
   carrying tools. Check automatic tool selection, assistant `tool_calls`, tool
   IDs/results, terminal usage, stop reason, and the reasoning-off template.
   Capture a replay request at the HTTP boundary and check the same properties.
5. Start with shipped cache/speculation behavior and record the selected route,
   candidate budget, and any fallback reason. Startup configuration is not proof
   that MTP ran. An explicitly requested unsupported route must fail loudly;
   do not silently modify the harness workload to make it run.

The existing `benchmarks/atlas-agent/serve-hipengine.sh` is a launch reference,
not a ready-made campaign command: it defaults to Qwen3.8, 256K context and
machine-local paths. Create a dedicated launch recipe with explicit overrides.

For reasoning off, the inspected Endpoints schema supports
`model_params.chat_template_kwargs`, and hipEngine's
`_thinking_control_from_request` accepts `enable_thinking`. In the campaign copy
of the upstream config set:

```yaml
model_params:
  name: Qwen3.6-27B-Q4_K_M
  tokenizer_name: Qwen/Qwen3.6-27B
  temperature: 0
  seed: 42
  max_new_tokens: 1024
  chat_template_kwargs:
    enable_thinking: false
```

This replaces only that block; preserve the rest of the reference config. Set
`endpoint_config.endpoints` and a unique `report_dir`. Verify forwarding and
rendered behavior live; this mapping was inspected in source, not exercised here.
Record the config diff, including this explicit equivalent of llama.cpp's
`--reasoning off`.

The upstream CLI syntax is verified from its pinned README; installation and
these invocations have **not** been run in this documentation task. From the
campaign Endpoints checkout root, with the prepared config at `$EDGE_CONFIG`:

```bash
# Complete performance plus accuracy against one unchanged server.
inference-endpoint benchmark from-config --config "$EDGE_CONFIG"

# Full prescribed accuracy sample alone, when diagnosing quality.
inference-endpoint benchmark from-config --config "$EDGE_CONFIG" --accuracy-only
```

### Stage 1 — One complete trajectory: 27 turns

Select `pallets__flask-5014`, retaining every original row for that conversation
in original order, with tools, system message, recorded calls/results and delays
unchanged. This is the shortest trajectory by generated-turn count, chosen for
cheap integration screening, not representativeness.

Make a diagnostic config copy containing only the performance dataset, pointing
to that filtered JSONL, with `num_trajectories_to_issue: 1`. Keep the sampler,
window, inline checker, timeout and concurrency unchanged. Use the same
`benchmark from-config` invocation, not a new timing client.

Acceptance: all 27 turns accounted for, no request/parse errors or missing
outputs, nonempty token metrics, recorded inline score, verified reasoning mode
and actual execution route. A valid result permits the larger diagnostic; it
does not certify performance competitiveness or BFCL accuracy.

### Stage 2 — Fixed whole-trajectory screen: 140 turns

Use these three conversations, chosen before any hipEngine timing:

| Conversation | Generated turns | Selection role |
| --- | ---: | --- |
| `pallets__flask-5014` | 27 | Shortest trajectory |
| `pydata__xarray-6721` | 52 | Near-median trajectory, different project |
| `pytest-dev__pytest-6202` | 61 | Longest trajectory |

Retain all source rows and their relative order; set
`num_trajectories_to_issue: 3`. Run the subset afresh rather than splicing the
Stage 1 timings into it. This spans trajectory lengths, **not necessarily prompt
lengths**. Report actual input/output token distributions and coverage against
the full dataset; specifically inspect late turns near the reference's reported
23.5K peak input. If long-input coverage is missing, add a separately named
long-context diagnostic instead of changing the frozen subset after timing.

Run this same 140-turn workload on hipEngine and the same-GGUF llama.cpp baseline.
Compare per-trajectory and pooled mean turn latency, TTFT and TPOT, token counts,
inline scores, and failures. A second fresh-server subset pass checks whether a
large gap is reproducible without paying for another full gate. Do not infer a
stable p99 from 140 correlated turns.

For a separate early quality diagnostic, plan a fixed approximately 100-case
selection from the prescribed BFCL draw, spanning all three categories and
multiple function-call subsets. Freeze IDs before observing answers, keep the
upstream extractor/scorer, and report raw counts. Implement and test selection
without changing the full gate; no small-sample CLI switch is assumed here.
A small sample can expose broken tool formatting, not prove a 3% accuracy band.
The full approximately 995-case draw is itself a sample of BFCL v4; do not replace
it with arbitrary percentages or the entire BFCL corpus.

### Stage 3 — Full reference-shaped evaluation

Run the combined upstream config with the full 20 trajectories and the exact
BFCL draw against an unchanged hipEngine server. Keep config and output paths
separate from diagnostics. Check:

- Exactly 1,007 completed generated turns across 20 conversations, zero missing
  turns, no hidden timeout truncation, and a preserved inline score.
- Full BFCL sample manifest, predictions, category counts and both gate verdicts.
- Token accounting, TTFT/TPOT and latency distributions, wall time, selected
  serving routes, cache telemetry and resource state.
- No process/config/model change between the timed phase and quality phase.

For a completed primary comparison, run the same combined evaluation on
llama.cpp using identical GGUF bytes on the same host. If either arm has failures,
retain the failed evidence and diagnose it; do not average only successful
requests into a headline win. A watchdog stopping a partial run makes it partial,
not a completed slow result. Preserve artifacts before changing time limits;
any extended-budget diagnostic must be labelled as such.

### Stage 4 — Report and close

Write compact artifacts under `benchmarks/results/`, retain raw upstream reports
outside Git, and add a dated immutable worklog entry. Preserve
`performance/result_summary.json`, `accuracy/accuracy_results.json`, `scores.json`,
resolved `config.yaml`, per-turn records, commands and logs by stable artifact
reference/hash. Apply the normal benchmark rollup only once there is a measured
row; this plan adds no scoreboard result.

Report local runs as **MLPerf Edge Agentic harness evaluations**, not official,
submitted or peer-reviewed MLPerf results. An official submission additionally
requires the applicable ruleset/checker, complete system/measurement metadata,
and MLCommons submission/review process. Resolve any discovery-pin versus v6.1
submission differences before making that claim.

## 5. Runtime budget and ballpark decision

These are **planning estimates**, not measured hipEngine rates or promises.
They assume installed dependencies, cached model weights, one idle gfx1151 host,
reasoning off, concurrency 1, and no integration defect.

| Work | Initial device-time allowance per engine | Basis and limits |
| --- | --- | --- |
| 27-turn smoke | 2–10 minutes | Linear scaling from the external 64-minute to 157-minute performance anchors gives approximately 1.7–4.2 minutes, widened for trajectory skew; excludes model load/JIT |
| 140-turn screen | 10–30 minutes | Same scaling gives approximately 9–22 minutes; prefix/output-length mix can move it outside this range |
| Approximately 100-case BFCL diagnostic | 15–30 minutes | Rough scaling of upstream's approximately three-hour, 995-case gate; category/output-length mix matters |
| Full performance replay | 1–3 hours | StorageReview's under-64-minute Atlas Spark report and upstream's 2 h 37 m Thor reference are external anchors, not a hardware-normalized prediction |
| Full BFCL gate | 2–4 hours | Upstream documents approximately three hours; actual generated lengths and scoring overhead need measurement |
| Combined full run | Roughly 3–7 hours | Sum of planning ranges; the reference config times out at six hours, so the upper end will not complete under its unchanged deadline |

Reserve **a half-day device slot for one full combined arm**. The September 25
screen measures about 21 minutes per arm; allow 25–30 minutes including startup
and drain for a repeat. Keep the full performance allowance at 1–3 hours rather
than claiming a stratified prediction from only three trajectories. BFCL runtime
has not been measured. A complete same-host two-engine comparison needs
roughly twice the device time, plus restarts; a short paired screen is the
recommended first investment. One-time environment/model acquisition and API
integration may take hours independently of device timing; network speed and
any compatibility repair are not yet estimated.

After Stage 2, replace linear scaling with:

```text
predicted performance wall = startup outside measurement
                           + sum over full workload strata
                             (turn count × measured mean turn latency)
                           + enabled recorded delays + client/scorer overhead
```

Keep measured-phase wall and total job wall separate. Stratify by input length,
output length and early/late trajectory position, using actual generated lengths
where available; recorded outputs are only predictors. Do not add TTFT again if
using end-to-end turn latency. Add the independently estimated BFCL phase and a
stated scheduling margin. If delay injection is off, its delay term is zero.

The ballpark decision is descriptive, not an automated keep/kill threshold:
report whether the paired subset has similar latency, a reproducible material
gap, or inconclusive variance/coverage. Compare quality and generated lengths
alongside speed. The external roughly 20 tok/s headline is context only until
its metric definition and workload/configuration match. A slower screen directs
profiling toward prefill, decode, cache reuse or client overhead; it does not
justify shrinking the official workload or disabling an implemented feature.

## 6. Initial handoff checklist

- [x] Prepare pinned Endpoints/BFCL environment and exact Qwen3.6 model/tokenizer.
- [x] Confirm actual reference settings and preserve the upstream config diff.
- [x] Create and test deterministic whole-conversation smoke/config generation.
- [x] Verify a real hipEngine HTTP tool/reasoning/token-accounting probe.
- [x] Clear the output/timing failures from the completed 27-turn attempt.
- [x] Run the fixed 140-turn paired screen; revise the ETA.
- [ ] Repair harness TPOT's post-first-tool-chunk tokenization before using it
  for decode-rate comparisons; preserve original metrics.
- [x] Compare gufo on the frozen Qwen3.6 subset, pinning its executable and actual
  decoding settings; do not substitute its published Qwen3.8 configuration.
- [x] Run the small BFCL diagnostic if needed, without claiming a gate pass.
- [ ] Run full combined evaluation and same-GGUF baseline; retain both verdicts.
- [ ] Publish qualified-by-scope local artifacts and close with findings.

Qwen3.8, another physical host, local Atlas NVFP4, and Windows are optional
follow-ups with separate configurations and result rows. None is needed to
answer the initial Qwen3.6 reference-workload question.
