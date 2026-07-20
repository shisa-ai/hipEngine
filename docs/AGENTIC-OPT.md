# Agentic Serving Optimization Board

Last updated: 2026-07-20

`AGENTIC-OPT.md` is the active status, measurement, and optimization board for
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

The recommended next unit is a W7900 GGUF coding-agent benchmark followed by
three measured candidates: gfx1100 prefix reuse, low-occupancy routing, and GGUF
native GPU sampling. The benchmark, not intuition, selects their order after the
baseline packet exists.

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

1. **No retained W7900 coding-agent workload packet.** Existing server rows use
   throughput-oriented prompt/decode shapes. They do not measure repeated
   assistant-tool-result turns, tool-call-ready latency, prefix hits, or
   per-turn queue/prefill/decode ownership.
2. **gfx1100 production-policy transfer is incomplete.** GGUF continuous
   membership and physical c8 are retained, but the broader occupancy-adaptive
   low-load, sampled API, prefix-economics, long-context pressure, and SLO packet
   has not been independently transferred from gfx1151.
3. **GGUF sampled decoding uses host logits.** Functional host sampling is
   correct and explicit, but may copy one full FP32 vocabulary row per generated
   token. Native GPU sampler integration and true batched c>N selection remain
   open.
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
2. **Prefix reuse is narrow and default-off.** The retained implementation is a
   scoped GGUF radix opt-in with limited exact boundaries. Broader history,
   sampled reuse, LRU pressure, fork/rollback semantics, and gfx1100 economics
   remain open.
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
7. **Live-model quality coverage is narrow.** The gfx1151 sampled gate proves
   one bounded strict tool schema and one structured failure. It does not claim
   broad coding-agent quality, BFCL-style tool selection, patch correctness, or
   autonomous repository success.

### Maintainability

`hipengine/server/api.py` is nearly 15,000 lines. The contract is well tested,
but batching, sessions/continuations, constraints, streaming envelopes, and
routing should eventually become focused modules. This is refactor debt rather
than the first performance priority: split only in behavior-preserving units
with the full server contract gate green after each unit.

## Improvement priority order

### P0 — Establish the W7900 coding-agent baseline

Build and retain a real-Uvicorn benchmark that exercises several complete
assistant -> tool call -> tool result turns. It must measure concurrency one as
well as concurrent agent swarms, preserve exact prompt/tool identities, and
finish with zero request/session/KV ownership. No optimization is promoted
before this baseline exists.

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
- paired off/on TTFT, tool-call-ready, total-turn wall, and HBM economics.

Do not make radix caching default merely because p256+s1 is fast. Default
promotion requires the multi-turn workload and bounded LRU/pressure gates.

### P2 — Transfer low-occupancy/SLO routing to gfx1100

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
- **A2 — prefix A/B:** off/radix active and completed-source boundaries with
  2K/8K/growing-history fixtures.
- **A3 — sampled path:** host-logits baseline, then native GGUF candidate when
  implemented.
- **A4 — routing/SLO:** batch-window and prefill-policy A/B under delayed mixed
  arrivals.
- **A5 — pressure/soak:** cancellation, slow consumer, queue/KV/cache pressure,
  eviction, and final ownership.
- **A6 — quality lane:** automatic tool selection and repository-task oracles,
  reported separately from deterministic engine performance.

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

Current validated tool SSE is often a safely buffered public projection rather
than one event per generated model token. Such turns are explicitly labeled
`token_timing_mode=buffered_public`; the candidate IDs are labeled
`generated_token_ids_source=matched_nonstreaming_oracle` and
`sse_exact_ids_observed=false`, and ITL is withheld instead of assigning
fabricated per-token timestamps. Oracle/tool equality is useful diagnostic
coverage, but it does **not** satisfy the retained exact-SSE-ID gate. A retained
performance row requires the measured response to expose its own exact IDs, and
a retained ITL row additionally requires `live_exact` lower-loop events. The
first diagnostic command is:

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

The complete A1 baseline still requires clean cache-off C1/C4/C8 runs over all
three workload families with warmup and repeated measurements. The initial
single-family smoke is diagnostic, not a retained performance claim.

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

Next actions are now split by purpose. The A6 quality lane below keeps natural
multi-turn failures and reports success rates without a performance denominator.
The performance lane still needs a deterministic, non-prompt-conditioned way to
obtain valid multi-turn envelopes; do not hardcode fixture argument tokens to
make it pass. Independently, expose measured SSE response-owned generated IDs.
C4/C8 for the 2K family also needs a capacity plan that passes the startup guard
rather than bypassing it; the 8K family needs a separate lower-concurrency
context configuration.

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
three successes rather than rejecting the packet. No W7900 quality result has
been run or published yet. The exact live command is:

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

Next: expose exact response-owned generated IDs on the measured tool SSE done
event, consume them in A1 with strict oracle equality, then run the A6 c1 quality
packet and retry the deterministic c1 performance lane without conflating their
acceptance rules.
