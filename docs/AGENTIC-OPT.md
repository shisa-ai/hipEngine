# Agentic Serving Optimization Board

Last updated: 2026-08-26

The A0-A6 gfx1100 board below and its quality-only ZBook
[`AGENTIC-QUALITY2`](AGENTIC-QUALITY2.md) follow-up are both closed. The ZBook
campaign freezes disjoint development/heldout external-oracle tasks, publishes
Qwen3.6/Qwen3.8/Ornith product-quality rows, and retains `no_implementation`
after finding no runtime-owned candidate trigger. It carries no performance
claim.

`AGENTIC-OPT.md` is the status, measurement, and optimization board for
using hipEngine as a local coding-agent runtime. The functional server contract
and feature roadmap remain in [`AGENTIC.md`](AGENTIC.md); continuous-batching
correctness and throughput evidence remain in
[`CONCURRENCY.md`](CONCURRENCY.md); sampling behavior remains in
[`SAMPLING.md`](SAMPLING.md). This document answers four narrower questions:

1. What is proven today on gfx1100/W7900?
2. What still limits coding-agent and harness use?
3. Which improvements should be attempted first?
4. Which benchmark decides whether an improvement is retained?

All performance rows remain subject to [`BENCHMARK.md`](BENCHMARK.md). A number
in this document is not a new claim: it is a compact pointer to the retained
artifact named beside it.

## Executive status

hipEngine already has a useful single-model local-agent server contract and a
strong greedy continuous-batching backend on gfx1100. The next gfx1100 program
should therefore optimize the **complete coding-agent turn**, not another
isolated speculative leaf.

The key workload distinction is:

- one active agent is dominated by turn-to-turn TTFT, repeated-prefix prefill,
  sampled/controlled token selection, and tool-call-ready latency;
- several active agents can also exploit the retained physical c2/c4/c8 GGUF
  model steps and continuous membership;
- both cases benefit from preserving exact resident state/KV ownership and
  avoiding full-vocabulary host readback.

The repeated W7900 A1 packet shows flat active-SSE goodput at 4K from C1 through
C8, a modest long-context C4 gain, linearly worsening buffered tool-ready
latency, and one full-vocabulary D2H row per generated token. A2 is closed: the
scoped radix route is exact and lifecycle-safe but decisively regresses every C1
family, so it remains explicit-only and cache-off remains the default. A3 is
also closed before timing: native-eligible auto-tool sampling fails the frozen
turn-1 strict-envelope oracle, while valid specific/required tool forcing remains
explicit host fallback. The GGUF native sampler therefore stays default-off.
A4 routing is also closed without promotion: every candidate fails a balanced
mixed-arrival SLO or exact-ID gate. A5 now closes pressure/soak on unchanged
package defaults: all nine workloads pass exactness, SLO, bounded-resource, and
final-ownership gates across 122 requests. A6 closes the first broad
external-oracle quality packet: 10/48 complete tool turns pass across four
families, with no performance claim. The queued A0-A6 measurement program and
the ZBook [`AGENTIC-QUALITY2`](AGENTIC-QUALITY2.md) AQ0-AQ13 follow-on are
complete; the latter retains no runtime mechanism. The independent A4
performance blocker remains historical gfx1100 work and is not part of that
quality denominator.

## Current status report

### Server and harness contract

Current main provides:

- OpenAI-compatible `/v1/completions` and `/v1/chat/completions`;
- blocking and SSE responses with exact generated-token accounting where the
  backend supplies IDs;
- Qwen reasoning extraction, no-think controls, bounded thinking hints, and
  host-sampler soft/hard close behavior;
- OpenAI function tools, forced/specific/parallel tool calls, transcript
  validation, streamed argument fragments, and fail-closed malformed output;
- result validation for strict tool schemas and JSON/schema/regex/choice/
  unified-diff outputs;
- request deadlines, cooperative disconnect cancellation, bounded per-request
  stream queues, overload errors, drain/close ownership, and Prometheus metrics;
- app-local transcript sessions with commit policy, fork, rollback, snapshot,
  restore, context fitting, and deterministic buffered continuation handles;
- `/v1/hipengine/capabilities`, tokenizer diagnostics, error taxonomy, and
  opt-in redacted replay artifacts;
- checked-in local-agent and pi configurations.

The current-main deterministic contract gate is **130/130 tests passed** across
`tests/test_agentic_server_conformance.py`,
`tests/test_agentic_harness_traces.py`, and `tests/test_local_agent_config.py`.
The golden-trace fixture contains **56 traces** covering tool loops, reasoning,
strict failures, structured results, sessions/continuations, finish phases,
sampling metadata, and HTTP/SSE errors. These tests prove API behavior and
fail-safe envelopes; they do not prove broad live-model tool-use quality.

### Retained gfx1100 serving evidence

| Scope | Current result | Evidence |
| --- | --- | --- |
| GGUF direct native c8 graph decode | **246.872 aggregate tok/s** | `benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-scaling-closure.json` |
| Real OpenAI GGUF c1 | **25.583 aggregate tok/s** at p512/d128 SSE cycle wall | `benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-f1-server-scaling-closure.json` |
| Real OpenAI GGUF physical c8 | **136.122 aggregate tok/s** | same F1 artifact |
| Real OpenAI logical C13 | **111.380 aggregate tok/s**, physical c8 plus sparse c8 | same F1 artifact |
| Same-loop serial C13 control | **31.708 aggregate tok/s** | same F1 artifact |
| Coding-agent A1 active SSE | Small **16.239/15.995/16.020**, growing **15.100/15.231/15.036**, medium **4.127/4.629/4.339 tok/s** at C1/C4/C8 | `benchmarks/results/2026-07-21-w7900-agentic-a1-repeated-baseline.json` |
| Coding-agent A4 routing/SLO | **Blocked, no performance row**: 8 candidates x 3 balanced mixed-arrival passes; control misses one TTFT SLO and alternatives produce 9 late p512/d48 ID mismatches | `benchmarks/results/2026-07-22-w7900-agentic-a4-routing-decision.json` |
| Coding-agent A5 pressure/soak | **Passed correctness/SLO; no comparative performance claim**: 122 requests, 108 completions, 12 exact retryable rejects, one disconnect, one deadline; 80 s soak is 40/40 exact and final ownership is zero | `benchmarks/results/2026-07-22-w7900-agentic-a5-pressure-soak-closure.json` |
| Coding-agent A6 broad automatic-tool quality | **10/48 successful turns; no performance claim**: valid call/correct tool 18/48, exact arguments/external-oracle pass 16/48, safe patch success 0/6, external test success 8/8; 24/24 repeat pairs exact and ownership zero | `benchmarks/results/2026-07-22-w7900-agentic-a6-broad-quality.json` |
| PARO direct selected-batch c2 | **121.923 aggregate tok/s**, +5.09% vs c1 and +20.81% vs serial c2 | `benchmarks/results/2026-07-18-gfx1100-paro-g2-selected-batch-c2-retained.json` |
| PARO MTP N4 after parallel router | **66.303/66.259 tok/s**, about **0.592x true AR** | `benchmarks/results/2026-07-20-w7900-paro-mtp-n4plus-parallel-router-topk.json` |

The GGUF F1 packet also retains exact prompt/output accounting, arbitrary-C
physical-group manifests, continuous admission, cancellation/reclaim, bounded
streaming, and final ownership. C13 is not a native width-13 claim.

### Useful evidence from gfx1151 that is not yet a gfx1100 claim

The gfx1151 GGUF production program independently retains:

- occupancy-adaptive physical c1/c2/c4/c8 OpenAI execution;
- deterministic sampled blocking/SSE c4 and `n=3`, exact logprob accounting,
  stop/EOS, strict tool forcing, and fail-closed structured results;
- active-current p256+s1 prefix reuse with continuation TTFT
  **249.269 -> 21.188 ms (11.765x)** and one fewer live page;
- completed-source snapshot restore with continuation TTFT
  **249.446 -> 22.013 ms (11.332x)** at **72,089,600 bytes** of cache residency.

Those results justify a gfx1100 transfer experiment. They do not establish
W7900 performance, broader prefix history, sampled prefix reuse, or general
resident-session semantics.

## Current limitations

### Runtime and performance

1. **The retained coding-agent baseline exposes no useful 4K concurrency
   scaling.** Active-SSE goodput is flat from C1 through C8 while buffered
   tool-ready p50 grows about 4x/8x. Medium C4 is only **1.122x** C1 and C8 falls
   to **0.937x** C4. Prefix reuse and native sampling now have measured targets.
2. **gfx1100 routing-policy promotion is correctness-blocked.** The balanced A4
   screen completes all eight predeclared policy/chunk/burst/window candidates,
   but no candidate passes all three delayed mixed-arrival repetitions. The
   package control misses TTFT p95 once; alternatives produce nine late
   `fixed-0011` p512/d48 trajectory mismatches after 20-24 exact IDs. A4 must
   localize that state/KV or physical-width transition before any C1/C2/C4/C8
   promotion timing. Long-context pressure is covered separately by the A5 closure.
3. **GGUF has no promotable native sampled tool route.** The explicit candidate
   removes full-vocabulary D2H for supported c1 and dense compatible c>N rows,
   but specific/required tool forcing, close queues, and other dynamic processors
   intentionally remain host-backed. Native-eligible `tool_choice=auto` fails the
   frozen sampled-agent strict-envelope preflight, so the route remains
   default-off and the host path still copies one FP32 vocabulary row per token
   for valid strict tool turns.
4. **gfx1100 PARO serving remains width-1.** The direct selected-batch c2 model
   step is retained, but physical c4/c8 and attachment to the shared resident
   OpenAI owner are not.
5. **PARO N4 MTP is not an AR replacement.** The exact full-suite route remains
   about 0.592x true AR. More isolated N4 leaf tuning is lower priority than
   proposal-ownership redesign or dense-model speculative work.
6. **Cancellation is cooperative.** The server observes cancellation around
   tokenization, prefill, decode, graph, and stream boundaries; it does not
   preempt a running kernel or graph.
7. **Fairness is bounded but not preemptive.** The current policy is
   `fifo_compatible_sampling_key`; advanced per-tenant/fair-share scheduling is
   not implemented.

### Agent and API behavior

1. **Sessions and continuations re-prefill.** App-local sessions retain visible
   transcript data, not resident KV/decode state. Capabilities correctly report
   `resident_state_reuse=false`.
2. **Prefix reuse is narrow, default-off, and performance-rejected for the
   frozen agentic suite.** The scoped GGUF radix opt-in has exact active/current
   and completed-source boundaries plus bounded LRU/COW/cancellation economics.
   Resident device-state fork/rollback is explicitly unsupported. Broader
   history and sampled reuse remain open only behind a future model-general
   redesign; the measured latest-boundary policy is not a promotion candidate.
3. **Tools and structured outputs are not fully constrained decoding.** Marker
   forcing, narrow JSON close repair, parsing, and result validation fail
   closed, but broad token-level JSON/tool/patch grammars are absent. A failed
   generation can therefore cost a full turn before rejection.
4. **Streaming tool arguments are validated before publication.** The public
   fragments are safe and concatenable, but they are not yet canonical
   lower-loop live grammar events that permit speculative early tool execution.
5. **Routing is single-model exact-match.** Multiple resident models,
   capability-aware fallback, model-family substitution, multi-worker routing,
   and tensor parallel runtime are not implemented.
6. **No OpenAI Responses API.** Chat Completions supports pi and the checked-in
   local-agent adapter. `/v1/responses` should be added only when a target
   harness requires it and supplies conformance traces.
7. **Live-model quality remains weak and synthetic.** The broad W7900 A6 packet
   now covers 24 repeated repository/general/English/Japanese tasks with external
   result, patch, and test oracles, but only **10/48** complete turns pass and no
   patch turn succeeds. This is materially broader than one strict schema, yet
   it is not BFCL-style public evaluation, generated-patch execution, autonomous
   repository success, or a cross-model quality leaderboard.

### Maintainability

`hipengine/server/api.py` is nearly 15,000 lines. The contract is well tested,
but batching, sessions/continuations, constraints, streaming envelopes, and
routing should eventually become focused modules. This is refactor debt rather
than the first performance priority: split only in behavior-preserving units
with the full server contract gate green after each unit.

## Improvement priority order

### P0 — Establish the W7900 coding-agent baseline — complete

The retained real-Uvicorn A1 packet covers all three frozen workloads at
C1/C4/C8 with one discarded warmup plus three measured runs each. It preserves
17,316 response-owned IDs across 702 strict tool turns, passes all independent
blocking/SSE and ownership gates, and explicitly separates active SSE waves from
the oracle-inclusive first-to-last harness wall. See the 2026-07-21 closure
below.

### P1 — Transfer and broaden gfx1100 GGUF prefix reuse

Why first: coding agents repeatedly submit a stable system prompt, tool schemas,
repository context, and prior visible turns. Avoiding repeated prefill should
improve every tool round even at occupancy one.

Required gates:

- current-active and completed-source boundaries on gfx1100;
- at least 2K and 8K stable prefixes plus a mixed incremental-turn shape;
- exact deterministic output IDs and byte-exact survivor state/KV;
- sampled same-seed non-perturbation before sampled reuse is enabled;
- bounded cache residency, refcount/COW correctness, eviction, cancellation,
  fork/rollback rejection or support, and zero final ownership;
- paired off/on TTFT, tool-call-ready, total-turn wall, and VRAM economics.

A2.0 is correctness-ready on gfx1100: active-current and completed-source
p256+s1 both preserve exact output, every Conv/GDN/live-KV byte, and four
teacher-forced steps (`KL=0`, top-1 100%) while refcounts/COW and final drain
pass. Host gates preserve the latest aligned snapshot across an unaligned tail,
bound snapshot/page/byte ownership, and explicitly fall back for sampled and
exact-full-prompt boundaries. The live collector now pins radix outputs to the
retained A1 cache-off IDs and records hit source/boundary, avoided/executed
prefill tokens, reused pages/bytes, state-clone bytes, prefill time, and bounded
final cache ownership. The 2K/8K deterministic correctness gates are now
closed as described below; this A2.0 packet itself carries no performance
result.

The first A2.1 `small_repo` C1 pair then exposed a narrower production boundary:
strict forced-tool rows resolve deterministic `processed_argmax`, not
`greedy_fast`, so all four radix turns correctly reported
`sampling_unsupported` with zero lookups/hits/reused tokens. The one observed
pair remains diagnostic-only and the rest of that timing matrix stopped.

A2.1 correctness now admits only deterministic `processed_argmax` beside the
existing greedy route; stochastic host/GPU sampling still fails closed with
`sampling_unsupported`. A private miss bulk-prefills the final aligned boundary
once and consumes only an unaligned tail through exact c1, leaving one bounded
snapshot. A hit restores active or completed state/pages, executes only the
unmatched suffix through c1, requests full-vocabulary logits on the final
suffix token, and applies the unchanged processor state. W7900 p2048+s1 and
p8192+s1 active-current/completed-source gates preserve the exact five response
IDs (including a two-token forced sequence), all Conv/GDN/live-KV bytes, and
teacher-forced logits (`KL=0`, top-1 100%). They reuse 8/32 pages, clone
66,846,720 state bytes, preserve expected zero-COW page-aligned behavior, and
drain final refs to zero; completed cache residency is bounded at 108,789,760 /
234,618,880 bytes before explicit eviction. Radix observability is iterative at
8K depth, so telemetry no longer depends on Python's recursion limit.

The complete A2.1 C1 matrix rejects this route for agentic promotion. Three
balanced off/radix pairs per family pass exact response/tool, bounded ownership,
and GPU0 exclusivity gates, but the retained latest-boundary snapshot rarely
matches the next forced-tool prompt: small hits **0/12**, growing **3/24**, and
medium **3/18** measured turns. Radix versus paired off regresses active-SSE
median goodput **64.19%/65.63%/26.64%** and worsens buffered tool-ready p50
**181.90%/196.09%/38.81%**. Radix medians **4.727/4.216/2.838 tok/s** and
**5.247/6.097/6.632 s** all fail the A1 C1 guards, and growing/medium variance
exceeds 5%. Exact miss/tail handling costs more than the sparse hits save.
Therefore keep radix default-off and stop the prerequisite-gated C4/C8
promotion matrix; do not optimize snapshot placement to these fixed prompts.
Lifecycle/pressure correctness is also closed independently of speed. Four real
W7900 p2048/p8192 active-current/completed-source gates preserve exact response
IDs and all Conv/GDN/live-KV state (`KL=0`, top-1 100%), bound current/high-water
pool bytes, retain completed cache residency only within declared limits, and
drain every ref after eviction. The real agentic packet bounds final cache
residency at **124,518,400/129,761,280/255,590,400 bytes** for
small/growing/medium, with zero non-cache final owners; useful hits cost
**42,240/28,160 bytes per reused token** for growing/medium. Host ownership
checks close LRU replacement, COW/refcount/pin/unpin, admission rollback, and
cache-owned eviction. Server control checks close cancellation, disconnect,
slow-consumer backpressure, deadline reuse, and survivor isolation. Resident
state fork/rollback remains explicitly unsupported: app-local transcripts are
deep-copied without forking or rolling back device state. This correctness
closure cannot reverse the C1 performance rejection.
`benchmarks/results/2026-07-21-w7900-agentic-a2-prefix-lifecycle-closure.json`.

The final A2.4 decision therefore keeps `HIPENGINE_PREFIX_CACHE=off`. Every C1
family regresses both primary metrics, all A1 guards fail, growing/medium radix
variance exceeds 5%, and the failed C1 prerequisite intentionally prevents
C4/C8 timing or a medium-C4 promotion claim. Radix remains an explicit
diagnostic only; no prompt-conditioned snapshot retargeting is permitted. This
selected the now-closed A3 host/native screen from the cache-off control.
`benchmarks/results/2026-07-21-w7900-agentic-a2-prefix-decision.json`.

### P2 — Transfer low-occupancy/SLO routing to gfx1100 — blocked

Protect a lone coding agent from paying a masked wide-batch or artificial batch
window while still filling physical c2/c4/c8 under concurrent arrivals.

Required gates:

- occupancy 1/2/4/8 plus delayed admission and mixed generation lengths;
- first-turn and continuation TTFT, p50/p95/p99 ITL, queue wait, and aggregate
  goodput;
- exact generated IDs and unchanged stable request/state/KV ownership while
  execution width changes;
- cancellation/backpressure and overload behavior;
- no material c1 regression for a retained c>N gain.

The predeclared A4 stage-1 screen runs **8 candidates x 3 balanced repetitions**
over 12 delayed mixed-shape requests per candidate: **288 requests / 8,640
response-owned IDs** total. Native c1/c2/c4 routes and final ownership pass, but
there is no complete correctness+SLO candidate. The zero-window
`protect_decode:256/burst-1` control is exact but misses the 10 s TTFT p95 SLO
once at **10.983 s**. Every apparently faster alternative changes the late
`fixed-0011` p512/d48 trajectory in at least one pass after 20-24 correct token
`9710` IDs; there are **9 mismatched rows** in total. The apparent median
SLO-goodput gains up to **+63.81%** are diagnostic-only and invalid for
promotion.

The funnel therefore stops before occupancy C1/C2/C4/C8, strict-tool, and
cancellation/backpressure/overload promotion rows. No timing is inferred, no
runtime default changes, and gfx1100 remains on
`protect_decode:256/burst-1` with the zero-ms package window. A4 can rerun only
after a model-general exactness gate localizes and repairs the late state/KV or
physical-width transition. A5 pressure/soak proceeded independently on the
unchanged default and is closed below. Artifacts:
`benchmarks/results/2026-07-22-w7900-agentic-a4-routing-screen-blocked.json`
and
`benchmarks/results/2026-07-22-w7900-agentic-a4-routing-decision.json`.

The independent A5 pressure/soak packet is now complete on those unchanged
defaults. One clean W7900 process serves all nine real-Uvicorn workloads over
**122 requests**: **108 complete**, **12** overload rows receive exact retryable
`429 engine_busy`, one row disconnects after two exact IDs with **44.5 ms**
reclaim acknowledgement, and one deadline returns the distinct timeout path.
All **2,482** observed generated IDs are exact, including the disconnected pair;
completed rows own **2,480** IDs. The bounded 80-second soak completes **40/40**
requests at **11.151 exact SLO-goodput tok/s**, while the overload wave completes
**20** and rejects **12** at **21.717 tok/s**.

Queue depth reaches its declared **16** cap, the slow consumer's stream queue
peaks at **1/16**, resident active/pending rows peak at **4/3**, and the dynamic
KV pool grows **3 -> 12 pages** then records **15 grow / 15 shrink** events with
zero failures. The packet records **28 graph captures / 998 replays / 28
invalidations**, releases **7,245,205,456 workspace bytes**, returns tracked
memory below baseline, and closes with zero request, queue, KV ref/pin,
graph-entry, workspace, or model owners. Forty-one KFD samples see only the
target process on GPU0 and zero GPU1 activity. Cache remains off by the A2
performance decision; cache-pressure/eviction coverage is cryptographically
linked to the exact p2048/p8192 A2 lifecycle closure, whose explicit eviction
ends with zero refs. This is valid bounded reliability and absolute SLO evidence,
not a tuning comparison or multi-day soak claim. Artifact:
`benchmarks/results/2026-07-22-w7900-agentic-a5-pressure-soak-closure.json`.

### P3 — Integrate native GGUF GPU sampling

Reuse the registered native sampler primitives where the GGUF logits/state ABI
can satisfy them. The target is one device-resident row/batch selection path,
not one host sampler loop per physical row.

Required gates:

- fixed-seed repeatability and CPU-reference distribution sanity;
- processors, stop/EOS, tool marker forcing, thinking close queues, and bounded
  logprobs for every advertised native combination;
- explicit fallback/reject metadata for unsupported combinations;
- full-vocabulary D2H bytes reduced to zero on the retained native route;
- c1 and c>N latency/goodput non-regression with exact API accounting.

The correctness prerequisite is complete. The explicit GGUF candidate admits
exactly `supports_native_gpu_sampling()` rows, keeps forced/repair/JSON/thinking
and unsupported stochastic shapes on host logits, and uses one sampler launch
for dense compatible packed rows. The real W7900 p256/c4 gate repeats four
fixed-seed rows exactly, matches same-shape forced-host Conv/GDN/logical live-KV
bytes, passes stop/EOS/bounded-logprob and API telemetry, records six batch
sampler launches, and drains all refs with zero COW. Supported rows report
`full_vocab_logits_d2h=false` / `logits_d2h_bytes=0`. Artifact:
`benchmarks/results/2026-07-21-w7900-gguf-native-sampler-correctness.json`.

The A3 real-Uvicorn preflight then fails closed before measured SSE. At
`temperature=0.85`, `top_k=8`, `top_p=0.82`, `min_p=0.08`, fixed seed 17, and
the realistic penalty/logprob set, host and native auto-tool C1 both repeat the
valid first `small_repo` turn but reach 64 tokens on turn 1 with
`invalid_tool_call`. Native telemetry is correctly `gpu_sample` with zero logits
D2H; host telemetry is `host_logits_sample` with **63,569,920 bytes** of
full-vocabulary D2H on that failed turn. Conversely, two repeats of all four
specific strict-tool turns are exact and valid under the native-enabled server,
but every row reports `host_logits_sample` /
`native_gpu_unsupported_request`, totaling **198,656,000 bytes** of D2H across
200 generated tokens. All three servers drain request/session/KV/graph/workspace
ownership to zero.

No request route is therefore both native-eligible and valid across the frozen
strict-tool workload. The C1/C4/C8 timing matrix did not start, no active-SSE or
tool-ready number is retained or inferred, and GGUF native sampling remains
explicit/default-off. Artifact:
`benchmarks/results/2026-07-22-w7900-agentic-a3-native-sampler-blocked.json`.

### P4 — Complete gfx1100 PARO c4/c8 resident serving

Generalize one physical selected-batch algorithm rather than stacking c2 groups,
then attach retained widths to the backend-neutral owner. Do this if PARO remains
a deployed coding-agent endpoint after the GGUF packet; otherwise keep GGUF as
the primary serving path.

### P5 — Improve strict decoding only from observed failures

Add tokenizer-aware tool/JSON/patch grammar primitives when the benchmark or a
real harness shows material invalid-call/retry cost. Preserve current
fail-closed behavior as the fallback. Do not build a broad grammar engine before
recording the actual failure distribution.

### Deferred

- More 35B-A3B PARO N4 leaf tuning without a proposal-ownership hypothesis.
- Multi-model routing or TP before a real deployment needs it.
- `/v1/responses` without a target-client trace suite.
- The block-aligned W4/dp4a rewrite as a quick task; it is a separate substantial
  c1 kernel/repack project.

## Coding-agent benchmark plan

### Benchmark objective

Measure the server behavior that determines local coding-agent usability:

- time until the first assistant token;
- time until a complete validated tool call can be executed;
- time from tool-result submission to the next action/final answer;
- exact generated-token and tool-envelope correctness;
- per-turn prefill/decode/cache ownership;
- single-agent latency and multi-agent goodput;
- cancellation, backpressure, cache pressure, and final resource ownership.

This first benchmark is an engine/server benchmark, not an autonomous coding
quality score. A later quality lane can add BFCL-, HumanEval-, or repository-task
oracles without changing the timing contract.

### Frozen workload families

Use several committed fixtures rather than one prompt:

| Family | Stable prefix | Turns | Purpose |
| --- | ---: | ---: | --- |
| `small_repo` | about 2K tokens | 4 | Normal local tool loop and low TTFT |
| `medium_repo` | about 8K tokens | 6 | Prefix/cache economics and repeated prefill |
| `growing_history` | starts about 2K, adds bounded tool results each turn | 8 | Incremental history and cache-boundary changes |
| `mixed_agents` | mix of the above | 4-8 | Continuous arrival, fairness, and occupancy changes |

Fixtures contain a stable system policy, deterministic tool definitions, a
synthetic repository summary/file map, user requests, and bounded tool results.
They must not contain benchmark-specific expected token IDs or wording selected
to improve model acceptance/performance.

### Request lanes

1. **Deterministic contract lane**
   - `temperature=0`, fixed tool choice where required, SSE on;
   - exact output IDs must match the independent/cache-off oracle;
   - used for prefix, routing, lifecycle, and regression decisions.
2. **Sampled agent lane**
   - fixed seed, bounded `top_k<=64`, top-p, and one realistic penalty set;
   - strict tool schema plus stop/EOS/logprob metadata;
   - repeated same-seed behavior and cache/non-cache non-perturbation required;
   - used to compare host and future native GGUF sampling.
3. **Auto-tool quality lane**
   - tool choice `auto`, natural task prompts, external correctness oracle;
   - records valid-call, correct-tool, valid-arguments, patch/test success, and
     repair count;
   - correctness/quality only until the task set is broad enough for a retained
     performance claim.

### Concurrency and scheduling matrix

The initial retained matrix is:

- client concurrency: **1, 4, 8**;
- batch window: package default plus one explicit zero-window control;
- prefix cache: **off** and explicit **radix**;
- prompt/prefill policy: package default, with alternatives selected only by a
  same-workload A/B;
- stream mode: real localhost Uvicorn SSE;
- warmup: one complete untimed workload per configuration;
- measurements: at least three fresh or balanced repeated runs, with order
  rotated for A/B candidates;
- hardware: W7900/gfx1100, both HIP and ROCR visibility pinned.

Pressure/soak adds delayed arrivals, one slow consumer, one disconnected row,
one deadline, queue/KV rejection, cache eviction, and at least 100 completed
requests.

### Required per-request and per-turn records

- workload/fixture name and SHA-256;
- agent/session/turn/request ids;
- requested and served model identity;
- exact prompt token count/hash and generated token IDs/hash;
- declared tool schema hash, selected tool name, argument JSON validity, and
  transcript linkage;
- submit, admitted, prefill-start/end, first-token, tool-call-ready, final-token,
  response-done, and tool-result-submit timestamps where available;
- queue wait, TTFT, ITL samples, tool-call-ready latency, generation wall,
  complete turn wall, and end-to-end workload wall;
- sampler mode, processor blockers/fallback, logits D2H bytes, physical width,
  graph route, and serial/resident fallback counters;
- prefix lookup/hit/source/boundary/reused-token fields, cache action, cache
  bytes/pages/refcounts/pins, and per-request KV usage when exposed;
- finish/error details, cancellation/deadline observation, and retry count.

### Rollups

Report, without mixing timing scopes:

- TTFT, tool-call-ready latency, ITL, and complete-turn wall p50/p95/p99;
- exact generated tok/s and complete validated tool calls/s;
- workload completion wall and SLO-goodput;
- prefix hit rate, reused tokens, avoided prefill tokens/time, cache bytes, and
  bytes per reusable prefix token;
- host/native sampler rows and full-vocabulary D2H bytes;
- physical c1/c2/c4/c8 steps, occupancy, queue wait, fallback counts, and
  cancellation latency;
- final request/session/KV/graph/workspace ownership.

### Correctness and acceptance gates

Every retained comparison requires:

1. every HTTP/SSE response is schema-valid and ends correctly;
2. no raw `<think>` or `<tool_call>` markup leaks into public assistant fields;
3. every tool call names a declared tool, has valid JSON arguments, passes the
   declared strict schema, and links to exactly one result;
4. deterministic cache/routing candidates match cache-off independent generated
   IDs and preserve state/KV/reference-count oracles;
5. sampled candidates are fixed-seed repeatable and do not change solely because
   prefix reuse is enabled;
6. exact prompt/generated-token denominators and timing owners are present;
7. rejected, cancelled, disconnected, and slow-consumer requests do not corrupt
   survivors;
8. final active/pending requests, sessions, KV refs/pins, graph/workspace owners,
   and stream producers are zero or equal to declared bounded cache residency;
9. a performance candidate improves its primary predeclared metric, passes all
   correctness gates, and does not materially regress the paired c1/SLO guard;
10. all retained rows update the benchmark artifact, README rollup, changelog,
    and `WORKLOG.md` under the normal evidence policy.

### Implementation phases

- **A0 — deterministic benchmark core:** fixture/schema loader, fake-server
  contract tests, timestamp/rollup math, and artifact validation.
- **A1 — live W7900 baseline:** current package defaults, cache off, deterministic
  real-Uvicorn C1/C4/C8 packet.
- **A2 — prefix A/B (closed, rejected):** exact/lifecycle-safe scoped radix,
  but materially regressive at C1; cache-off remains default and C4/C8 was
  skipped by prerequisite.
- **A3 — sampled path (closed, blocked before timing):** native-eligible
  auto-tool rows fail the frozen turn-1 strict-envelope oracle, while all valid
  specific/required tool rows use explicit host fallback; no C1/C4/C8 timing or
  promotion claim exists.
- **A4 — routing/SLO (closed, blocked at stage 1):** all eight candidates finish
  three balanced delayed mixed-arrival passes, but the control misses one TTFT
  SLO and every faster alternative has at least one late p512/d48 ID mismatch;
  no occupancy/promotion timing or default change exists.
- **A5 — pressure/soak (complete):** all nine workloads pass cancellation,
  slow-consumer, queue/KV/cache pressure, eviction-link, SLO, GPU-exclusivity,
  and final-ownership gates across 122 requests; no tuning comparison.
- **A6 — quality lane (complete):** 2 repeats of 24 externally scored turns
  complete on clean W7900 source across repository, general-English, Japanese,
  and mixed Japanese/English families; 10/48 pass, with no performance claim.

### Current implementation status

A0 is implemented by:

- `benchmarks/prompts/agentic-coding-v1.json`: three synthetic repository
  workloads with 2K/8K/growing-history targets and 4/6/8 strict tool turns;
- `benchmarks/schemas/agentic-coding-workloads.schema.json`;
- `benchmarks/schemas/agentic-coding-records.schema.json`;
- `benchmarks/schemas/agentic-coding-benchmark.schema.json`;
- `hipengine/benchmark/agentic.py`: fixture identity, independent strict
  argument checks, exact-token/timestamp/batch-owner/ownership validation, and
  latency/goodput/cache/backend rollups;
- `scripts/agentic_coding_bench.py`: model-free normalized-record to artifact
  CLI;
- `tests/test_agentic_coding_benchmark.py`: RED/GREEN contract and failure
  matrix.

The normalized A0 command is:

```bash
python3 scripts/agentic_coding_bench.py \
  --workloads benchmarks/prompts/agentic-coding-v1.json \
  --records /tmp/agentic-coding-records.json \
  --json /tmp/agentic-coding-a0.json
```

A0 intentionally accepts only successful deterministic assistant-tool-result
turns. Cancellation, deadline, slow-consumer, and sampled records remain A3/A5
extensions; automatic-tool quality now uses the separate non-performance A6
artifact below rather than weakening the initial exact tool denominator.

A1 collection is implemented by `hipengine/benchmark/agentic_live.py` and
`scripts/agentic_coding_live.py`. It:

- expands/detokenizes each synthetic repository prefix to the exact target under
  the served tokenizer and rejects a non-exact roundtrip;
- builds stable prior assistant-tool-result transcripts with deterministic call
  ids, independent of random server response ids;
- obtains an independent non-streaming c1 exact-token/tool oracle outside each
  measured SSE interval;
- releases C concurrent real HTTP SSE requests together, reconstructs strict
  streamed tool arguments, and requires oracle/fixture equality;
- records batch/choice timing ownership, physical group rows, sampler/D2H/
  fallback metadata, prompt/output hashes, and public TTFT/tool-ready wall;
- polls readiness, sessions/continuations, and KV refs/pins to close final
  ownership before A0 accepts the packet.

Validated tool SSE is often a safely buffered public projection rather than one
event per generated model token. Such turns are explicitly labeled
`token_timing_mode=buffered_public`, and ITL is withheld instead of assigning
fabricated per-token timestamps. A1 now consumes cumulative response-owned IDs
from the final SSE done choice, requires exact equality to the independent
non-streaming oracle, and records `generated_token_ids_source=response` with
`sse_exact_ids_observed=true`. If a backend omits the final IDs, the old
`matched_nonstreaming_oracle`/false row remains diagnostic and cannot support a
retained exact-token performance claim. Exact response IDs make the denominator
valid; a retained ITL row additionally requires `live_exact` lower-loop events.
The first diagnostic command is:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python3 scripts/agentic_coding_live.py \
  --base-url http://127.0.0.1:8100/v1 \
  --model Qwen3.6-35B-A3B --backend hip_gfx1100 \
  --workload small_repo --concurrency 1 --runs 1 \
  --cache-mode off --max-tokens 128 \
  --records-json /tmp/agentic-small-c1-records.json \
  --json /tmp/agentic-small-c1-a1.json
```

The complete A1 baseline is retained across cache-off C1/C4/C8 for all three
workload families with one warmup and three measurements per configuration.
Earlier single-family rows remain diagnostics; only active SSE wave time from
the repeated packet is a performance denominator.

#### First W7900 A1 smoke: blocked before timing

The first dirty-tree diagnostic used W7900/gfx1100,
`/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, cache off, and the small 2K
family. Launch-capacity checks behaved correctly:

- the default 256-position resident capacity rejected the 2,619-position rendered
  first turn before generation;
- 12,288/c8 and 4,096/c8 startup scratch probes failed with HIP OOM and left
  readiness false;
- 4,096/c1 passed eager load, the full 4,095-token scratch probe, warmup chat,
  and readiness, with 23.45 GiB used and 21.54 GiB free after startup.

The viable diagnostic server was:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
HIPENGINE_GENERATION_BATCH_WINDOW_MS=0 \
HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1 \
HIPENGINE_GGUF_GDN_PREFILL_MODE=exact \
HIPENGINE_GGUF_AR_STREAM_DECODE=0 \
HIPENGINE_GGUF_AR_PACKED_DECODE=1 \
HIPENGINE_MAX_PREFILL_CHUNK_TOKENS=256 \
python3 -m hipengine.server \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --backend hip_gfx1100 --served-model-name Qwen3.6-35B-A3B \
  --max-context-tokens 4096 --max-active-requests 1 \
  --prefix-cache off --metrics prometheus --eager-load \
  --host 127.0.0.1 --port 8100
```

The original fixture had an ambiguous `read` mode; the model validly selected
`raw` while the fixture expected `summary`. Every deterministic user turn now
states its exact arguments. The clarified first oracle then selected the exact
`read(path="pyproject.toml", mode="summary")` call with 32 generated IDs and
`finish_reason=tool_calls`, but also returned
`message.content="<|im_end|><|im_end|>"`. A1 rejects non-empty content beside
these explicitly tool-only calls, so no measured SSE interval, artifact, or
performance number was retained.

The terminal-residue follow-up is now GREEN for blocking and SSE. The parser
removes repeated `<|im_end|>` only from the outer edges of text surrounding a
successfully parsed tool block; an interior literal and non-tool structured
output remain unchanged. On the same W7900 rerun, turn 0 then passed both its
oracle and measured SSE. Turn 1 exposed the next boundary: after the valid prior
assistant/read/result transcript, Qwen emitted an empty `<tool_call>`, role/
template tokens, and then reasoning instead of the forced `grep` arguments. At
128 tokens the server correctly returned `finish_reason=length` with
`finish_details.reason=invalid_tool_call`, rather than publishing malformed
output. No complete-workload artifact or timing row was retained.

Next actions are split by purpose. The A6 quality lane below keeps natural
multi-turn failures and reports success rates without a performance denominator.
The performance lane still needs a deterministic, non-prompt-conditioned way to
obtain valid multi-turn envelopes; do not hardcode fixture argument tokens to
make it pass. Response-owned SSE IDs are now implemented and live-verified, so
that provenance issue is closed. C4/C8 for the 2K family still needs a capacity
plan that passes the startup guard rather than bypassing it; the 8K family needs
a separate lower-concurrency context configuration.

## 2026-07-20 — Separate natural auto-tool quality from performance (A6)

A6 now has a distinct fail-closed quality contract and live blocking collector:

- `hipengine/benchmark/agentic_quality.py` normalizes natural `tool_choice=auto`
  responses, requires blocking response-owned generated IDs, and preserves
  model failures as explicit outcomes instead of aborting the workload;
- `scripts/agentic_coding_quality.py` evaluates every turn against an independent
  canonical valid prior transcript, so one failed attempt is measured without
  corrupting or cascading into later turns;
- `agentic-coding-quality-{records,benchmark}.schema.json` pin a separate
  artifact kind whose `performance_claim` is always false;
- rollups report valid-call, correct-tool, exact-argument, success, repair, and
  outcome counts/rates, with no TTFT, tok/s, or goodput fields;
- the deterministic A0/A1 validator remains unchanged and all-success: natural
  quality rows cannot enter its exact latency/performance denominator.

The fake-transport gate retains a turn with
`finish_details.reason=invalid_tool_call`, completes all four turns, and reports
three successes rather than rejecting the packet. The first clean W7900 run is
now published as a bounded quality diagnostic below. The exact live command is:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python3 scripts/agentic_coding_quality.py \
  --base-url http://127.0.0.1:8100/v1 \
  --model Qwen3.6-35B-A3B --backend hip_gfx1100 \
  --workload small_repo --concurrency 1 --runs 1 \
  --cache-mode off --max-tokens 128 \
  --records-json /tmp/agentic-small-c1-quality-records.json \
  --json /tmp/agentic-small-c1-quality.json
```

RED/GREEN validation:

```bash
python3 -m pytest -q tests/test_agentic_coding_quality.py
# RED: ModuleNotFoundError: hipengine.benchmark.agentic_quality
# GREEN: 5 passed
python3 -m pytest -q tests/test_agentic_coding_quality.py \
  tests/test_agentic_coding_live.py tests/test_agentic_coding_benchmark.py
# 17 passed
ruff check hipengine/benchmark/agentic_quality.py \
  hipengine/benchmark/__init__.py scripts/agentic_coding_live.py \
  scripts/agentic_coding_quality.py tests/test_agentic_coding_quality.py
# All checks passed
```

The clean `4d01f897` W7900/gfx1100 run completed all four independent
`small_repo` attempts and retained **0/4 successes** over **274 exact
response-owned generated IDs**. Two turns selected the correct tool and exact
arguments but returned Qwen role/EOT residue in public content; two produced no
valid call and ended as `invalid_tool_call`. The quality artifact contains no
latency, tok/s, or goodput fields, sets `performance_claim=false`, and records
zero final request/session/KV/graph/workspace ownership:
[`2026-07-20-w7900-agentic-a6-small-c1-quality.json`](../benchmarks/results/2026-07-20-w7900-agentic-a6-small-c1-quality.json).
This one-run/four-turn result is a failure-distribution diagnostic, not a broad
model-quality score.

## 2026-07-20 — Expose response-owned exact IDs on tool SSE

`GenerationStreamChunk` now carries an optional cumulative
`generated_token_ids` tuple. Final GGUF direct, resident, and scheduler-chunk
stream events populate and preserve it without copying the growing ID history on
every intermediate token; generic resident scheduling, stream
coercion, buffered/detail projection, phase rewriting, and final output
conversion no longer drop it. With `stream_options.include_hipengine=true`, the
final completion/chat SSE done choice publishes `generated_token_ids` and
`generated_tokens` when the final backend chunk supplies exact IDs. Capabilities
advertise this as
`choice_generated_token_ids=done_event_when_backend_supplies_exact_ids`.

A1 reads only response IDs actually present in SSE, checks their count and exact
sequence against the independent blocking oracle, and rejects drift before
normalizing a record. Successful exact-ID tool streams use `source=response`
and `sse_exact_ids_observed=true`. Buffered tool arguments remain
`token_timing_mode=buffered_public`: exact IDs repair the throughput denominator,
not ITL resolution. Backends that omit IDs retain the prior explicit diagnostic
oracle source rather than receiving fabricated response provenance.

RED/GREEN and validation:

```bash
python3 -m pytest -q \
  tests/test_generation_registry.py::test_generation_stream_chunk_preserves_token_logprobs \
  tests/test_server_api.py::test_streaming_chat_tool_done_exposes_response_owned_generated_ids \
  tests/test_agentic_coding_live.py::test_sse_normalizer_uses_exact_response_ids_and_rejects_oracle_drift
# RED: 3 failed
# GREEN: 3 passed
python3 -m pytest -q tests/test_server_api.py
# 503 passed
python3 -m pytest -q tests/test_generation_registry.py \
  tests/test_agentic_coding_live.py tests/test_agentic_coding_benchmark.py \
  tests/test_generation_qwen35_gguf_sampling.py
# 99 passed
```

The clean W7900 rerun verifies the transport but still rejects the baseline.
The 4K/c1 server passed its guarded 4,095-token startup probe in **79.425 s** at
**23.447 GiB used / 21.537 GiB free**. A1 turn 0 then matched its independent
oracle with **32 response-owned IDs**, exact
`read(path="pyproject.toml", mode="summary")`, and
`token_timing_mode=buffered_public`. Turn 1's independent blocking oracle did
not finish with a valid forced `grep` tool call, so the collector correctly
stopped before opening that measured SSE interval and emitted no partial A1
records or performance artifact. The blocked packet is
[`2026-07-20-w7900-agentic-a1-small-c1-blocked.json`](../benchmarks/results/2026-07-20-w7900-agentic-a1-small-c1-blocked.json).

The next correctness unit is model-general strict-tool JSON constraint/repair,
not fixture-conditioned argument forcing. A complete small-repo c1 packet must
pass before prefix A/B or any latency/goodput claim. C4/C8 remains a separate
startup-capacity blocker.

## 2026-07-20 — Make selected-tool JSON prefix forcing atomic

The blocked turn-1 IDs exposed a tokenizer-boundary bug rather than an argument
oracle gap. Qwen tokenizes `<tool_call>` with a final `>` token, but tokenizes
`<tool_call>{"name":"grep","arguments":` with a merged `>{` token. The prior
route forced the short marker and attempted to complete the longer prefix with
a token-sequence DFA; because those tokenizations are non-composable, only the
marker was forced and the model could emit template text immediately after it.

Specific function choices, and `required` with exactly one declared function,
now tokenize and queue the entire selected
`<tool_call>{"name":"...","arguments":` prefix atomically from token zero. A
multi-tool `required` request still forces only the opening marker so tool
selection remains model-owned. Thinking-budget requests hold the same complete
prefix until answer phase, and the existing close-marker completion remains.
No prompt text, benchmark token ID, expected argument key, or expected argument
value is inspected or forced; arguments remain model-generated and strict-schema
validated. Capabilities report the scope as
`atomic_tool_call_name_and_arguments_key`.

The host-only RED/GREEN and broad server/generation gates pass. No GPU result or
performance claim changed. The next step is the clean W7900 A6/A1 rerun; only a
complete all-success A1 packet can open prefix/routing performance work.

## 2026-07-20 — Post-tool-fix W7900 A6/A1 remains blocked

The clean `f25362d0` W7900 4K/c1 cache-off rerun improves the natural A6 lane
from **0/4 to 2/4 successful turns** over the same **274 exact response-owned
IDs**. The two previously valid `read` calls now have empty public content and
pass exact-argument scoring. Natural `grep` and `run` remain
`invalid_tool_call`; this is quality-only evidence and contains no timing.

The deterministic A1 lane still fails closed before opening its first measured
SSE interval. Atomic prefix forcing produces
`<tool_call>{"name":"read","arguments":`, but the model starts an invalid
argument object, emits Qwen controls, and later emits a second complete valid
`read(path="pyproject.toml", mode="summary")` envelope. The parser extracts
that later call but leaves the unfinished first envelope and controls in public
content. A same-canonical-request diagnostic contains **63 response-owned IDs**
and the exact parsed call, but it is not an A1 record or performance row.

Final request/session/KV/graph/workspace ownership is zero. The next correctness
unit is an envelope-scoped incomplete-duplicate-prefix repair after a valid
call parses, with ordinary-content and interior-literal guards. C4/C8 and any
prefix/routing performance work remain closed.

## 2026-07-20 — Strip incomplete duplicate forced-prefix residue

The exact 63-ID A1 turn-0 response is now a blocking and SSE regression. After a
later valid tool block parses, public cleanup discards an earlier incomplete
canonical `<tool_call>{"name":"...","arguments":` prefix only when its tool
name exactly matches the first parsed call and the remainder contains optional
whitespace, at most one unmatched `{`, and otherwise only recognized Qwen
controls/role labels. The check runs on the original remainder before generic
edge cleanup.

This does not repair arguments or reinterpret malformed-only output. Ordinary
text before the prefix, a mismatched tool name, interior literal markers, and an
incomplete prefix without a later valid call are all preserved or rejected by
their previous fail-closed routes. Blocking and SSE share the parser, and the
capability manifest advertises
`incomplete_duplicate_tool_prefix_control_residue`.

The exact blocking/SSE tests fail before the implementation and pass afterward;
ordinary, mismatched-name, literal-marker, and malformed-only guards pass. The
final **510-test server** and **142-test agentic/config** gates are green. No GPU
result or performance claim changed. Next: rerun the clean W7900 deterministic
A1 packet before opening any performance lane.

## 2026-07-20 — Deterministic A1 advances to the turn-1 JSON boundary

Clean `5d6a2883` on the W7900 passes the 4,095-token startup probe and advances
through turn 0's independent blocking oracle plus measured exact-SSE equality
gate. The fail-closed collector then reaches turn 1, proving the duplicate-prefix
cleanup repaired the previous boundary without retaining any partial timing.

Turn 1 still fails before measured SSE. The response begins with the atomic
`<tool_call>{"name":"grep","arguments":` prefix, emits `{`, Qwen template and
tool-response controls, then produces free-form reasoning until the 128-token
cap. It finishes `length` / `invalid_tool_call` with **128 response-owned IDs**,
no parsed call, and empty public content. The collector emits no records or
complete artifact; final ownership is zero.

This is now an in-envelope generation problem, not a parser-residue problem.
The next model-general unit must constrain JSON from the declared function
schema after the selected-tool prefix while leaving argument keys/values
model-selected—no fixture IDs, expected arguments, or prompt-conditioned repair.
A complete A1 packet remains required before any performance lane opens.

## 2026-07-20 — Anchor selected strict-tool JSON and stop at its envelope

A full vocabulary JSON-schema grammar is not required for the blocked Qwen
trajectory. For a selected function whose declared schema is strict, is an
object with `additionalProperties: false`, and has a first required string
property, hipEngine now atomically extends the existing forced tool prefix
through that schema-derived property key and its opening value quote. The model
still selects the string value and every remaining property; unsupported schema
shapes retain the shorter name-and-arguments prefix and all results remain
subject to independent strict postvalidation.

The same narrow route enables tokenizer-derived `}}</tool_call>` completion in
addition to the existing close-marker completion. A candidate is omitted if its
complete token sequence appears in the forced opening prefix. Once either safe
close candidate begins, its remainder is forced and the completed sequence is
also a stop boundary, preventing post-envelope Qwen transcript continuation.
The structural candidate is never enabled for multi-tool `required`, non-strict
schemas, unsupported first-property types, or failed schema-prefix tokenization.
Capabilities disclose both the schema-anchor scope and close/stop scope. This is
not a general JSON grammar and does not inspect prompts, fixtures, expected
arguments, or benchmark token IDs.

A deliberately dirty-tree W7900 diagnostic justified the bounded design before
commit: the complete four-turn small-repo c1 collector passed all four
independent blocking oracles, all four response-owned SSE equality gates, and
final zero-ownership checks over 100 generated IDs. That pre-commit diagnostic
has `performance_claim=false`; none of its timing fields are retained.

Clean committed `f7a38fd1` repeats the result: **4/4 blocking oracles + 4/4
exact-SSE gates** pass for exact `read`/`grep`/`read`/`run` arguments over **100
response-owned IDs**, with zero final ownership. The published
[completed A1 diagnostic](../benchmarks/results/2026-07-20-w7900-agentic-a1-small-c1-post-schema-anchor.json)
reports TTFT p50 **1536.618 ms** and **9.077 exact generated tok/s**, but remains
`performance_claim=false`: this is one small-repo c1 run without the required
warmups/repeats, C4/C8, or medium/growing-history coverage. Host validation is
green across **529 generation tests**, **513 server tests**, and **142
agentic/config tests**, plus Ruff. Next: solve capacity for the remaining A1
matrix before opening A2 prefix A/B.

## 2026-07-20 — Small-repo c4 is physically viable; sampled reuse repaired

The missing guarded capacity point is now measured: clean `fee9ee85` with a
4,096-token context and `--max-active-requests 4` passes the unchanged exact
4,095-token startup probe, including packed width-2/4 warmup, at **37.43 GiB
used / 7.56 GiB free**. Startup completed in **81.427 s**. This is capacity and
correctness evidence, not a performance result.

The first live c4 request then exposed two independent runtime defects. Sampled
resident prefill addressed the raw KV cache base even when the scheduler had
assigned a shifted dynamic allocation; after rebasing it through the same
block-table-aware packed route used by greedy prefill, that route still rejected
`return_logits=True`. The packed prefill path now returns one finite FP32 logits
row per active slot for the existing host sampler, and long shifted-contiguous
prompts remain on their per-session slot-local AOTriton cache view. Genuinely
non-contiguous paged scatter remains conservatively limited below 1,024 context
tokens. No requested sampling semantics are weakened or converted to top-1.

The startup probe also now respects the registry-selected plain-AR physical
width. An admission setting of C8 therefore probes the retained physical c4
route instead of allocating an unsupported width-8 plain-AR workspace. An
actually enabled MTP route retains its own admission-width warmup; the exact
capacity guard remains mandatory in both cases.

A dirty-tree correctness gate completed **8/8 c2 turns** and **16/16 c4 turns**
for `small_repo`, with every independent blocking oracle, response-owned SSE ID
equality check, strict tool argument check, and final zero-ownership check
passing. Both collector outputs set `performance_claim=false`; their timings
were discarded because the tree was dirty and host tests ran concurrently.
Clean committed C8 admission/physical-c4 coverage and the 8K/growing-history
context-capacity split remain required before A1 can become a retained baseline.

## 2026-07-20 — Separate logical C8 admission from physical c4 residency

The first clean `84fd737a` C8 launch still failed the startup guard before any
collector request. Although the probe itself was intended to clamp, the LLM
resident loop had already allocated eight 4K session slots: resident preparation
used **36.84 GiB**, leaving **8.14 GiB**, and scratch warmup failed with HIP OOM.
This run produced no A1 artifact and readiness remained false.

The physical-width contract now crosses the entire registry boundary. The
registered gfx1100 GGUF generator advertises plain-AR width four; `LLM` caps its
resident loop to the generator-advertised width while preserving the caller's
logical request/admission setting. The outer HTTP queue can therefore accept C8
without allocating eight resident slots. The same mechanism preserves gfx1151's
separately registered c8 width and avoids backend branches in `LLM` or server
dispatch. Public choice-scoped telemetry in the C8 diagnostics reports width
one, so this capacity result does not claim physical-c4 model-step execution.

A dirty W7900 validation then reached readiness with logical
`queue.max_active_requests=8`, scratch `max_batch_size=4`, packed AR warmups
`[2,4]`, and the same **37.43 GiB used / 7.56 GiB free** c4-residency footprint.
The full `small_repo` C8 collector passed **32/32 turns**, artifact validation,
and final zero ownership. Its artifact sets `performance_claim=false` and its
timing is discarded.

## 2026-07-20 — Clean C8 capacity closes all frozen A1 families

Clean pushed `56c91f87` repeats C8 correctness across every frozen family:

- `small_repo`: exact prompts **2,504-2,851 tokens**, **32/32 turns** and 800
  response-owned IDs pass at 4K context;
- `growing_history`: exact prompts **2,498-3,281 tokens**, **64/64 turns** and
  1,592 IDs pass at 4K context;
- `medium_repo`: exact prompts **8,644-9,229 tokens**, **48/48 turns** and 1,160
  IDs pass at 10,240 context.

Including the 128-token output budget, their exact minimum contexts are
**2,979 / 3,409 / 9,357** tokens. Both guarded servers retained logical C8 while
probing resident width four: 4K used/free **37.43/7.56 GiB** and 10K used/free
**38.26/6.72 GiB**. Every independent blocking oracle, response-owned SSE ID,
strict tool argument, collector validation, and final request/session/KV/graph/
workspace ownership check passed. The first medium attempt is excluded because
its external background server expired at 600 seconds; the 30-minute retry
completed cleanly.

The [capacity matrix](../benchmarks/results/2026-07-20-w7900-agentic-a1-c8-capacity-matrix.json)
and three linked collector artifacts all set `performance_claim=false`. Their
single-run, no-warmup timing is non-promotable, and width-one public telemetry
cannot support a physical-c4 claim. The now-frozen repeated baseline protocol is:
4K for small/growing, 10,240 for medium; logical C1/C4/C8; one discarded complete
workload warmup and three measured cache-off runs per configuration. All nine
configurations must pass before A2 prefix A/B opens.

## 2026-07-21 — Repeated W7900 A1 baseline retained

Clean pushed source `44c76674` completes the frozen real-Uvicorn, cache-off
matrix: three workload families x logical C1/C4/C8, each with one complete
discarded warmup and three target-GPU-exclusive measured runs. Across **702
strict tool turns** and **17,316 response-owned IDs**, every independent blocking
oracle, SSE exact-ID comparison, strict argument/schema gate, and final
request/session/KV/graph/workspace ownership check passes. All active-SSE rate
rows have less than **0.91% stdev/median**.

| Workload | C1 | C4 | C8 | C4/C1 | C8/C1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small, 4K | **16.239** | **15.995** | **16.020** | 0.985x | 0.987x |
| Growing history, 4K | **15.100** | **15.231** | **15.036** | 1.009x | 0.996x |
| Medium, 10,240 | **4.127** | **4.629** | **4.339** | 1.122x | 1.052x |

Rates are exact generated tok/s over the sum of measured SSE wave walls. The
original first-submit-to-last-tool-result rollup included independent blocking
oracles between turns and produced misleading **9.134/7.912/2.527 tok/s** C1
figures; it is retained only as an explicitly named diagnostic. The artifact and
A0 builder now expose both scopes. Public tool SSE remains `buffered_public`, so
its **1.526/1.655/4.396 s** C1 p50 is validated tool-ready latency, not true
lower-loop TTFT, and no ITL claim is made.

Short/growing C4/C8 provide no aggregate benefit, while medium C4 gains 12.17%
and C8 gives back 6.25% versus C4. Every output token still records one FP32
full-vocabulary D2H row (up to **1.473 GiB/run** at growing C8). A2 prefix reuse
was evaluated next and rejected at C1; native GPU sampling is now the next
measurement while preserving C1 and medium-C4 guards. ROCm GPU0 is the W7900
target. Concurrent work explicitly pinned to the separate ROCm GPU1/XTX is not
target contention and is recorded but allowed.

[Retained artifact](../benchmarks/results/2026-07-21-w7900-agentic-a1-repeated-baseline.json).

## 2026-07-22 — Freeze broad A6 external-oracle quality packet

The A6 lane now has a committed broad protocol instead of extrapolating the
four-turn `small_repo` diagnostic. `benchmarks/prompts/agentic-quality-v2.json`
contains **6 workloads / 24 turns**: 8 repository turns, 8 general-English
turns, 4 Japanese turns, and 4 mixed Japanese/English turns. Prompts cover file
inspection/search, operations lookup, bounded arithmetic, three safe patch
selections, and four focused/full test selections. The quality system message
asks the model to choose an appropriate declared tool; unlike the deterministic
A1 policy, it does not say that the user specifically requested a tool name.

`benchmarks/oracles/agentic-quality-v2.json` is a separately hashed external
oracle over committed synthetic files, summaries, knowledge entries, exact
rational arithmetic, one-region patch definitions, and post-patch file-hash
test suites. It executes selected arguments rather than deriving success from
fixture equality: for example, `19 * 37` passes the same result oracle as the
canonical `37 * 19` while remaining a non-exact argument row. Patch output is
never arbitrary model code; models select one of three committed safe patch IDs,
and the oracle applies it to an in-memory file before checking scheduler, cache,
release, or full expected hashes.

The artifact contract still reports valid-call, correct-tool, schema-valid and
exact arguments, repair counts, and outcomes. V2 additionally reports external
result-oracle, patch, and test pass rates plus per-family distributions. Every
turn continues to require response-owned generated IDs; final ownership must be
zero. The multi-workload live collector uses independent canonical valid
histories, supports repeated `--workload` or `--all-workloads`, and can attach
clean canonical source/model/hardware provenance. Quality artifacts remain
`performance_claim=false` and contain no TTFT, latency, tok/s, or goodput rollup.

The predeclared first measurement is cache-off/native-sampler-off W7900 c1,
all six workloads, **2 complete runs / 48 turns**, temperature zero, 128-token
cap, and real localhost blocking OpenAI responses. It reports failures rather
than aborting on invalid model envelopes; it makes no model-quality leaderboard
claim beyond this committed synthetic packet. RED failed on the missing external
oracle module. GREEN passes **27/27** broad/legacy quality, live-normalization,
and deterministic artifact tests; all **24/24** committed oracle cases execute
successfully with targeted Ruff, JSON parsing, Python compilation, and diff
checks. No GPU result is claimed by this protocol unit.

The clean live packet then ran from pushed `878d07a9` with the W7900 pinned as
GPU0, cache and the native sampler off, exact GDN prefill, packed AR, a 4,096
context cap, and one active request. Both complete runs are response-exact for
all **24/24** task pairs after excluding random call IDs. Prompt lengths are
**1,420-1,751 tokens**, all **4,538** generated IDs come directly from blocking
responses, no raw tool markup leaks, and final request/session/KV/graph/workspace
ownership is zero.

The broad result is intentionally sobering:

| Family | Attempts | Valid call / correct tool | Exact arguments / oracle pass | Complete success |
| --- | ---: | ---: | ---: | ---: |
| Repository | 16 | 6 | 6 | 2 |
| General English | 16 | 6 | 6 | 4 |
| Japanese | 8 | 2 | 0 | 0 |
| Mixed Japanese/English | 8 | 4 | 4 | 4 |
| **Total** | **48** | **18** | **16** | **10** |

Outcomes are **10 passed / 20 invalid-tool-call / 10 no-tool-call / 6
content-alongside-tool-call / 2 wrong-arguments**. The independent oracles pass
**16/48** selected results. Safe patch selection is **0/6**; independent test
selection is **8/8**, although two exact test calls still fail the complete-turn
gate because public assistant content accompanies the call. The protocol makes
no repair request, so repair attempts are exactly zero rather than inferred.
Japanese is the weakest family at 0/8 complete turns; the two valid `lookup`
calls choose the wrong key.

Fifty KFD samples at five-second intervals from startup through collection see
only the target server PID on GPU0; GPU1 remains at 0% throughout. Canonical
provenance is clean, model/source/hardware fingerprints are bound, and the
artifact contains no latency, TTFT, wall, tok/s, or goodput fields. This remains a committed synthetic
quality diagnostic, not a public benchmark or performance row. Artifact:
`benchmarks/results/2026-07-22-w7900-agentic-a6-broad-quality.json`.
