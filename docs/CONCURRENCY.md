# Concurrency and Continuous Batching

Last updated: 2026-07-16.

This document is the source-of-truth roadmap and punchlist for making `c=N` a
first-class model pipeline in hipEngine. The destination is fully native
single-GPU continuous batching for:

- GGUF and PARO model paths;
- gfx1100 and gfx1151 backends;
- prompt prefill, decode, row retirement, cancellation, streaming, and reclaim;
- greedy and normal per-row sampling.

This is a planning document, not an experiment diary. Results appear here only
as one-line status summaries with pointers to `WORKLOG.md` or retained artifacts.
Detailed commands, measurements, negative probes, and bisection history belong
in `WORKLOG.md` and `benchmarks/results/`.

Related source-of-truth documents:

- [`PLAN.md`](PLAN.md) — architecture invariants and the batch-shaped runtime
  design.
- [`BENCHMARK.md`](BENCHMARK.md) — c=N correctness and performance acceptance
  protocols.
- [`ROADMAP.md`](ROADMAP.md) — project-level phase ordering.
- [`KVCACHE.md`](KVCACHE.md) — KV policy and future DMS work.
- [`PREFILL.md`](PREFILL.md) — prefill implementations and boundaries.
- [`SAMPLING.md`](SAMPLING.md) — normal sampling support.
- [`ENVS.md`](ENVS.md) — supported runtime knobs.

## Goal

hipEngine must treat an active group of requests as one canonical runtime shape,
not as N independent c=1 sessions hidden behind a server coalescer.

When this roadmap is complete:

1. One model-owning engine loop outlives individual API calls.
2. Requests may be admitted while other requests are decoding.
3. Prefill and decode are scheduled as peer work classes.
4. One native model step advances every active decode row once.
5. Finished, cancelled, and disconnected rows are reclaimed at commit points
   without stopping live neighbors.
6. Physical slots may be sparse, compacted, and reused without changing stable
   request identity.
7. Per-row sampling parameters and stop conditions coexist in one active group.
8. KV allocation, prefix reuse, graph buckets, and metrics are owned by the same
   loop.
9. The same contracts work for GGUF/PARO and gfx1100/gfx1151 without backend or
   quant branches in engine/model code.

The first retained milestone is not “the server accepts concurrent requests.”
It is a correctness- and profiler-proven native c=N model step. The final
milestone is live admission/reclaim around that step.

## Terminology and claim levels

| Term | Meaning |
| --- | --- |
| HTTP concurrency | Multiple client requests are in flight. Backend execution may still be serial. |
| Static prompt-list batch | One API call contains multiple prompts and runs one fixed batch from start to finish. |
| Serial bridge | Batch-shaped host metadata, but complete c=1 model work runs row by row. Correctness diagnostic only. |
| Exact hybrid | Some operations are batch-shaped while explicitly identified c1-exact row operations remain. May be a production-safe fallback, but is not a fully native throughput claim. |
| Native c=N model step | One model-step contract consumes `[C, ...]` state and advances all active rows without complete c=1 session/layer replay or host-side per-row model loops. |
| Native group width | Largest number of rows advanced by one native model step. Two serial c4 groups are not native c8. |
| Continuous batching | Requests are admitted, prefilled, decoded, finished, and reclaimed while other requests remain live under one model-owning loop. |
| First-class c=N pipeline | Native c=N is the canonical internal path. c=1 is the `C=1` instance, not a separate architecture. |
| Retained c=N row | Correctness, lifecycle, profiler, scaling, memory, and observability gates all pass under `docs/BENCHMARK.md`. |

### Fully native does not require one giant kernel

A native model step may launch multiple kernels and may preserve independent-row
math inside a batch-shaped kernel. It must not:

- create one resident c=1 session per active row;
- execute a Python loop over complete layers or complete model steps per row;
- call the c=1 layer runner once per row and label the result native;
- claim c8 when execution is two serial c4 groups;
- hide per-row/split/serial fallbacks from artifact and profiler metadata.

Exact row-local arithmetic is allowed when launched through an honest c-aware
batch interface and reported as such. Performance promotion still requires the
profiler to show that the route scales better than the serial bridge.

## Current truth

**hipEngine does not yet have production server continuous batching.** The
gfx1100 GGUF non-streaming model loop is now `continuous_eq_ok`: one long-lived
runner and fixed reusable resident-session pool admit controlled submissions
while a neighbor decodes, commit bounded prompt chunks, cancel or retire rows,
and reuse the exact session while survivor token/Conv/GDN/live-KV state remains
c1-exact. Public blocking `LLM.generate()` calls drive the same loop. D3 and D4
remain open: admission is not connected to the real device KV pool, and the
OpenAI streaming/backpressure/drain path does not yet consume the controlled
per-request events.

### Model/backend coverage

| Model path | Backend | Current c>N status | Production behavior | First missing gate |
| --- | --- | --- | --- | --- |
| GGUF Q4_K_M / BF16 KV | gfx1151 | Packed exact hybrid in groups of at most c4; short natural c10 runs as c4+c4+c2 | Packed server AR route is available; not a retained c>N throughput row | Per-layer hidden capture, standard all-row 512/128 gate, live admission/cancel, profiler/scaling |
| GGUF Q4_K_M / BF16 KV | gfx1100 | `retained` direct native-c4 model step plus `continuous_eq_ok` D2 loop: graph/eager p512/d128 and sparse c4→c1 gates are exact; same-session c4 is 184.993 aggregate tok/s (2.179x c1, 2.199x serial-c4); clean p512 live admission/cancel/retire/reuse preserves independent-c1 tokens and Conv/GDN/live-KV through native c2 membership changes | Public non-streaming calls and controlled submit/poll share one model-owning loop and reusable c4 session pool; production server streaming and device-KV-pool admission remain open | D3 real device KV pool, then D4 OpenAI streaming/backpressure/drain |
| GGUF Q5_K/Q6_K/Q8_0 / BF16 KV | gfx1100/gfx1151 | Not executed end to end under c>N | c1 | Q4_K_M c4 closure first |
| PARO W4 / BF16 KV | gfx1151 | Exact greedy c2 hybrid below 1024 total context; not fully native or retained | Unsupported groups fail closed to true width-1 sessions | Lifecycle/hidden/profiler/repetition gates, then remove row-local hybrid boundaries |
| PARO W4 / BF16 KV | gfx1100 | Historical primitive/token diagnostics only; no current retained native route | Width-1 sessions | Re-establish the current-HEAD c2 correctness baseline on W7900 |
| PARO W4 / INT8 KV | gfx1100/gfx1151 | Not started | Width-1 | BF16 native path first |

### Implemented scaffolding — not production-loop evidence

The following components exist and have focused CPU/host tests:

- `ResidentBatchScheduler` request ids, physical slots, pending/admitted state,
  active masks, finish reasons, and reclaim callbacks;
- `ResidentEngineLoop.submit/poll/cancel` around an abstract runner;
- prompt-list and `n>1` API lowering plus per-request output queues;
- `ChunkedKVPool` host/fake-runtime growth, shrink, refcounts, and stable block
  identity checks;
- `RadixCache` host-side prefix indexing and copy-on-write metadata;
- per-row sampling parameter blocks and EOS/reclaim metadata;
- graph-bucket, request, and pool observability schemas;
- Prometheus endpoint plumbing.

The gfx1100 GGUF D2 runner now owns real device state through live
prefill/decode/reclaim transitions and proves requests can enter, leave, cancel,
and reuse sessions while neighbors remain exact. The remaining scaffolding
becomes production server continuous batching only after D3 connects admission
to real device KV allocation and D4 routes OpenAI streaming, backpressure, and
drain through these events; PARO still lacks an equivalent model-owning runner.

### Result pointers

- Host scaffolding and its limitations: `WORKLOG.md`, **2026-07-13 — Re-baseline
  PARO and GGUF concurrency**.
- gfx1151 GGUF natural-suite token equality: the same 2026-07-13 WORKLOG entry
  and `benchmarks/results/2026-07-13-gfx1151-gguf-natural10-cn-token-equality.json`.
- gfx1151 GGUF exact packed state/KV lifecycle: `WORKLOG.md`, **2026-07-13 —
  Make GGUF packed AR state/KV exact through c4 lifecycle**, and
  `benchmarks/results/2026-07-13-gfx1151-gguf-packed-ar-exact-lifecycle.json`.
- gfx1151 PARO exact c2 hybrid: `WORKLOG.md`, **2026-07-13 — Re-baseline PARO
  and GGUF concurrency**.
- gfx1100 Phase-A c1/serial-c2/c4 controls and package-c4 route inventory:
  `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF Phase-A controls**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-phase-a-controls.json`.
- gfx1100 B1 c2/c4 primitive gates, package-policy inventory, and cached-only
  packed-c2 route trace: `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF B1
  primitive and route preflight**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-b1-preflight.json`.
- gfx1100 B2 strict-exact c2/c4, sparse, ragged, and all-layer lifecycle:
  `WORKLOG.md`, **2026-07-16 — Retain clean gfx1100 GGUF B2 direct lifecycle**,
  and `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-b2-direct-lifecycle.json`.
- gfx1100 B3/B4 all-row 512/128 and prompt-diversity closure: `WORKLOG.md`,
  **2026-07-16 — Close gfx1100 GGUF B3 standard lifecycle** and **Close
  gfx1100 GGUF B4 prompt diversity and Phase B**, plus
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-b3-standard-lifecycle.json`
  and `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-b4-category-lifecycle.json`.
- gfx1100 C1 runtime/profiler hybrid census: `WORKLOG.md`, **2026-07-16 — Close
  gfx1100 GGUF C1 hybrid-boundary census**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c1-hybrid-census.json`.
- gfx1100 C2 indexed recurrent closure and repeated Phase-B equality:
  `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF C2 recurrent linear
  attention**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c2-recurrent-closure.json`.
- gfx1100 C3 c-aware family census and copy-free steady metadata closure:
  `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF C3 model boundaries**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c3-model-boundaries-closure.json`.
- gfx1100 C4 state-bound graph replay and retained direct native-c4 scaling:
  `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF C4 native graph scaling**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json`.
- gfx1100 D1 long-lived GGUF model/session ownership and clean cross-call c2
  equality: `WORKLOG.md`, **2026-07-16 — Attach D1 to persistent GGUF
  model/session state**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d1-resident-model-runner-closure.json`.
- gfx1100 D2 bounded live admission, packed-group cancellation, max-token
  retirement, exact session reuse, and survivor state/KV equality: `WORKLOG.md`,
  **2026-07-16 — Close gfx1100 GGUF D2 live scheduling**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d2-live-lifecycle-closure.json`.
- Historical PARO c1-c8 catalog and lifecycle: `docs/BENCHMARK.md` §PARO c1-c8
  exact concurrency matrix.
- Historical c>N graph replay and output-tiled GEMV: `WORKLOG.md`, **2026-06-08
  — C3.0b-5 (piece D): c>1 decode graph capture + replay (correctness GREEN)**
  and **2026-06-08 — concurrency C3.0c output-tiled GEMV decode throughput: NULL
  (dispatch-bound)**; neither is a retained production c>N row.

## Architectural contract

### One owner

The production engine loop is the only owner of:

- admission and pending queues;
- request-id ↔ physical-slot maps;
- active masks and batch shape;
- prompt cursors and decode positions;
- canonical KV allocation and mutation;
- Conv/GDN recurrent state and model scratch;
- sampling, completion, streaming events, and reclaim;
- graph-bucket capture/invalidation;
- dynamic-pool grow/shrink decisions.

Server adapters may validate and enqueue. They may not own a second scheduler or
hold a request-lifetime model lock.

### Canonical batch state

The canonical internal shape is batch-aware even at C=1:

```text
request_ids[C]                 stable logical identity
slots[C]                       physical resident slots
active_mask[C]
token_ids[C]
positions[C]
context_lengths[C]
finish_flags[C]
hidden[C, hidden_size]
logits[C, vocab_size]          or row-tiled equivalent
conv_state[num_linear_layers, slots, ...]
gdn_state[num_linear_layers, slots, ...]
kv_spans[layer, C]             KVLiveSpans
sampler_params[C]
```

Request identity and slot identity must never be conflated. Kernels receive row
maps or state indices when routed lanes do not map directly to physical slots.

### Runner contract

A production model runner must expose equivalent operations to:

```python
prefill_batch(batch_state, prompt_slab, *, commit) -> PrefillResult
decode_batch(batch_state, *, commit) -> DecodeResult
compact_batch(batch_state, new_row_map) -> None
reclaim_slots(slot_ids) -> None
```

The exact Python names may differ. Required semantics do not:

- one call owns one batch-shaped model transition;
- canonical state mutates only at commit points;
- every fallback is explicit in execution metadata;
- GGUF and PARO implement the same scheduler-facing contract;
- backend-specific launch geometry comes from package capabilities/registry
  variants, never engine branches.

### Work classes and commit points

Each loop tick performs one work class:

| Work class | Action | Commit |
| --- | --- | --- |
| `RECLAIM` | Release finished/cancelled row state and KV | Free slots/pages; emit completion |
| `ADMIT` | Move pending requests into free slots | Publish slot/request map |
| `PREFILL_CHUNK` | Advance one or more prompt chunks | Prompt cursor, model state, KV |
| `DECODE_STEP` | Advance every decode-ready active row once | Token, position, state, KV |
| `VERIFY_STEP` | Future SpecDec target verification | Accepted rows only |
| `PACK_STEP` | Future DMS compaction | Compact KV only |

Default ordering is `RECLAIM → ADMIT → choose(PREFILL_CHUNK, DECODE_STEP)`.
`protect_decode` remains the default policy until latency evidence justifies
another default.

Cancellation, disconnect, EOS, max-tokens, timeout, and errors all converge on
`RECLAIM`. Mid-kernel cancellation never mutates canonical state.

### KV contract

- `KVLiveSpans` is the only attention/KV-write ABI.
- The scheduler allocates KV; kernels and model code do not.
- A logical block id remains stable for its lifetime.
- While a block is resident and referenced by a captured graph, its HBM backing
  pointer is pinned.
- Pool shrink may free only unreferenced, graph-unpinned tail chunks.
- Future KV tier movement must update allocator indirection and invalidate or
  rebind affected graph buckets before execution. “Stable block id” does not
  mean a captured stale pointer remains valid after a tier move.
- Prefix sharing remains default-off until a real model loop passes shared-prefix
  lifecycle and savings gates.

This resolves the apparent conflict between current append-only HBM allocation
and future KVTC pointer movement.

### Graph contract

Graph buckets include every axis that changes captured work:

```text
(model plan, backend, C bucket, context/page bucket, active mask,
 KV dtype/layout, prefill/decode mode, top-k/experts, replay length)
```

A cache lookup hit is not a replay claim. Retained evidence requires a replayed
kernel path in `rocprofv3`, positive replay counts, and matching bucket metadata.
Bucket invalidation is required after pointer-changing pool/tier events.

### Sampling contract

The active group may contain incompatible row parameters:

- temperature;
- top-k and top-p;
- repetition penalty;
- seed/RNG state;
- stop tokens;
- max-new-token budget.

Greedy exactness is the bring-up oracle. Normal sampling becomes production only
when deterministic seeded row-local tests and distribution checks pass. The
sampler may use multiple kernels, but not one complete LM-head/sampler host loop
per row in a native throughput claim.

### Backpressure and fairness

The loop enforces:

- maximum active requests;
- maximum pending requests;
- current KV admission capacity;
- maximum prefill tokens per tick;
- bounded streaming queues;
- explicit admission blocker reasons.

A slow streaming client pauses only its row at a commit point. It must not block
other rows or hold a model/session lock.

## Correctness and evidence gates

`docs/BENCHMARK.md` is authoritative. This section summarizes the concurrency
requirements so roadmap items have local exit criteria.

### Gate 1 — primitive kernels

For every backend, row count, KV dtype, and kernel family used by the runner:

- KV append key/value mismatch = 0;
- batch attention vs independent c1 satisfies the protocol tolerance;
- CPU/NumPy oracle tolerance passes;
- repeated A/A determinism passes where required;
- `rocprofv3 --kernel-trace` confirms the intended kernel, workgroup, resources,
  and positive duration.

### Gate 2 — direct model equality

For greedy sampling and SpecDec disabled:

- every generated token equals N independent c1 sessions;
- logits are finite;
- per-layer hidden capture passes the declared exact/tolerance contract;
- all persistent Conv/GDN state matches the c1 oracle;
- all live K/V prefixes match the c1 oracle;
- no serial bridge or undeclared fallback is present.

GGUF persistent state should be byte-exact when the c1 and c>N route claim the
same arithmetic. PARO uses its fixture-specific hidden/state contract, but token
equality remains mandatory.

### Gate 3 — lifecycle equality

At minimum:

- steady c2 and c4;
- ragged prompt lengths;
- front, middle, and tail row retirement;
- c4→c3→c2→c1 sparse-slot survival;
- cancellation of one row while neighbors continue;
- one newly admitted row while existing rows continue;
- prefill/decode interleaving;
- standard all-row prompt 512 / decode 128.

A short-horizon token-only suite is supportive evidence, not lifecycle closure.

### Gate 4 — native execution

Artifacts and profiles must prove:

- one native group has the claimed width;
- no complete c1 session/layer replay per row;
- no host per-row metadata allocation/copy in steady decode;
- c-aware projection, attention, state, MoE, and sampler paths are named;
- fallbacks are empty or explicitly disqualify the row;
- graph replay evidence is real when replay is claimed.

### Gate 5 — retained performance

Each retained c=N row records:

- exact model fingerprint, quant, KV dtype, backend, hardware, command, and clean
  source revision;
- c1 and serial-bridge baselines from the same protocol;
- aggregate tok/s and per-request tok/s;
- aggregate/c1, per-request/c1, and aggregate/serial ratios;
- TTFT and inter-token-latency p50/p95;
- occupancy over time and row-count transitions;
- tracked allocator/KV/workspace memory;
- profiler family durations and launch counts;
- admission/completion timestamps and finish reasons.

A native row must beat the serial bridge and c1 aggregate to become a scaling
claim. There is no arbitrary fixed speedup target; exact, non-regressive wins are
kept under the repository performance policy.

### Prompt coverage

Do not tune concurrency to one repeated-token prompt. Before promotion, run:

- deterministic raw-token 512/128 fixtures for exact comparison;
- all categories in `benchmarks/prompts/mtpbench-code-general-ja.jsonl`;
- ragged and sparse lifecycle fixtures;
- category-heldout prompts when sampling or route selection changes.

## Roadmap

The ordering is deliberate: transfer the strongest existing GGUF correctness
anchor to W7900, make and measure one c4 group as fully native, attach the live
scheduler on W7900, then generalize across backends and widths. PARO follows the
same pipeline once the GGUF contracts are proven.

### Phase A — refresh the roadmap and baselines

**Objective:** establish current-HEAD W7900 controls without changing kernels.

Punchlist:

- [x] Replace the historical concurrency notebook with this roadmap.
- [x] Record clean gfx1100 c1 GGUF 512/128 package-default baseline.
- [x] Record gfx1100 c2/c4 serial-bridge controls with exact accounting.
- [x] Capture a current packed c4 route/dispatch inventory before modifying it.
- [x] Confirm ROCm/compiler/model fingerprint and cached-build discipline.
- [x] Add a compact branch kickoff entry to `WORKLOG.md`.

Exit: one compact baseline artifact names every current fallback and supplies the
control for Phase B. No c>N performance claim.

### Phase B — gfx1100 GGUF c4 exact lifecycle

**Objective:** transfer the gfx1151 packed Q4_K_M/BF16 correctness anchor to
W7900 before optimizing or integrating the live loop.

B1. Primitive and route preflight:

- [x] Run c2/c4 KV append and full-attention primitive gates on gfx1100.
- [x] Confirm the production gfx1100 package defaults used by packed prefill and
      decode, including router, device metadata, GDN, quant, and graph policy.
- [x] Trace one packed step and verify every intended registry route.

B2. Direct lifecycle oracle:

- [x] Run `scripts/gguf_packed_ar_state_oracle.py` on gfx1100 for steady c2/c4.
- [x] Run c4→c3→c2→c1 middle-hole retirement.
- [x] Run ragged `[512,64,64,64]` and short equal-length prompts.
- [x] Add/enable per-layer hidden capture for the packed route.
- [x] Compare every generated token, 30 Conv/GDN families, 10 live-KV families,
      and per-layer hidden outputs against independent c1.

B3. Standard protocol capacity:

- [x] Make all-row c4 prompt-512/decode-128 fit through a correct chunked packed
      prefill path. The old 768-row hidden-slab ceiling is not an acceptable
      substitute for the standard gate.
- [x] Prove no row is silently serialized because the packed slab is full.
- [x] Pass the complete 512/128 lifecycle gate with exact accounting.

B4. Prompt diversity:

- [x] Run the full 10-prompt category suite for at least three repeats.
- [x] Add heldouts if route selection or sampling behavior changes.

Exit: gfx1100 GGUF c4 qualifies as `exact_hybrid` for
tokens/hidden/state/KV/lifecycle. It must remain `performance_claim=false` until
C4 replay and scaling pass.

### Phase C — fully native GGUF c4

**Objective:** remove exact-hybrid row replay from one c4 model step while
preserving Phase B equality.

C1. Make the fallback boundary observable:

- [x] Emit one execution manifest per layer family: projection, Conv/GDN,
      full-attention, MoE/FFN, LM head, and sampler.
- [x] Count host row loops, per-row launches, metadata copies, synchronizations,
      and scalar fallbacks.
- [x] Add profiler buckets for packed native work versus exact row-local work.

Clean `88f10724` attributes one p512 steady c4 transition's 1,386 dispatches to
**840 exact-row-local** launches (480 projection, 240 Conv/GDN, 120 RMSNorm) and
**546 packed-native** launches. The exact-row-local bucket accounts for 41.10%
of instrumented GPU duration. The runtime manifest reports 30 host row-loop
sites/120 row iterations, nine H2D copies, one four-value D2H token read, two
synchronizations, and zero steady state import/scatter or scalar fallback. The
single profiled transition is a route census, not a throughput measurement.

C2. Close recurrent linear attention:

- [x] Add a RED fixture that captures the first packed-vs-c1 state/hidden drift.
- [x] Implement/register a c-aware exact Conv/GDN decode route for gfx1100.
- [x] Preserve c1 arithmetic order or document and pass the numerical contract.
- [x] Mutate packed state in place with slot/segment metadata; do not scatter a
      complete state slab after every step.
- [x] Keep the unfused primitive chain as the required fallback.

Clean `f6e8363e` profiles one p512 steady c4 transition as **756 packed-native**
dispatches and **zero exact-row-local** dispatches, down from C1's 1,386 total
(**-45.45%**) and 840 row-local launches. The manifest reports zero host
model-row loops/iterations, zero steady state import/scatter, and the same nine
small H2D copies plus one vector D2H read. Indexed Conv and segmented FP32-output
GDN each execute 30 times with zero scratch. Clean lifecycle/category gates at
`f6e8363e`/`799d29b9` pass **20,640/20,640** p512/d128 and **560/560** sparse
shrink hidden comparisons plus **1,350/1,350** category tokens and
**54,000/54,000** category hidden comparisons. The route remains
`exact_hybrid`; profiler durations are diagnostic and no throughput/native-c4
claim is made. See the compact [C2 closure artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c2-recurrent-closure.json).

C3. Close remaining per-row model work:

- [x] Make full-attention append/context/O/post consume sparse c4 spans without
      complete row replay.
- [x] Confirm MoE routing, selected experts, expert dispatch, and reduction are
      truly c-aware at c4.
- [x] Make LM head and greedy sampler return one token per live row without
      full-vocab host readback.
- [x] Move steady-step metadata preparation to persistent device buffers.

Clean `db1ce640` profiles one p512 steady c4 transition as **749 packed-native**
dispatches and **zero exact-row-local** dispatches. Full attention launches 10
context plus 10 KV-write kernels at row extent four; all 40 MoE layers launch
32 selected lanes for gate-up/down plus one c4 combine; one c4 Q6 rowtile feeds
two row-wise argmax stages and a single four-i32 D2H read. The new registry-
resolved metadata producer replaces eight H2D uploads with one 64-thread,
16-VGPR, zero-scratch/LDS kernel, leaving exactly the token H2D and token-vector
D2H copy dispatches. Clean p512/c4/d128 and sparse c4→c1 gates pass
**20,640/20,640** and **560/560** layer-hidden comparisons with exact tokens,
Conv/GDN state, and live KV. The route remains correctness-only `exact_hybrid`;
C4 owns graph replay, repeated replay equality, scaling, and any native-c4 or
performance claim. See the compact [C3 closure artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c3-model-boundaries-closure.json).

C4. Replay, measure, and prove:

- [x] Capture/replay at least c1/c2/c4 decode buckets after eager equality passes.
- [x] Pass Phase B again with replay enabled and disabled.
- [x] Trace the full c4 step and prove zero undeclared c1 layer/session fallback.
- [x] Run same-protocol c1/c2/c4 plus explicit chunked-c8 controls.
- [x] Report aggregate/per-request throughput, TTFT/ITL, memory, occupancy, and
      profiler family/launch accounting.
- [x] Compare native c4 against c1 and the serial bridge in one clean session.

Clean graph/eager p512/c4/d128 each pass **20,640/20,640** hidden comparisons
plus exact tokens, Conv/GDN state, and live KV. Clean sparse graph replay is
exact through c4→c3→c2→c1 (**560/560** hidden rows). The marker-sliced c4 graph
launch contains **747 packed-native / 0 exact-row-local / 0 copy** dispatches,
with one device-position metadata launch and one token-feedback commit. In one
clean shared-runner 1+3 packet, c1/c2/c4/chunked-c8/serial-c4 aggregate decode
is **84.907/126.909/184.993/183.900/84.140 tok/s**. Native c4 is **2.179x** c1
and **2.199x** serial-c4; its per-request rate is **46.248 tok/s**. The tradeoff
is explicit: c4 model-step ITL p50/p95 is **21.641/21.888 ms**, TTFT is
**2.027/2.031 s**, and four-session tracked/HIP-used peaks are
**23.396/23.823 GiB**. Chunked c8 remains two serialized c4 groups and is not a
native-c8 claim. See the compact [C4 closure artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json).

Exit: satisfied for the direct gfx1100 GGUF c4 decode model step. Phase D still
owns production model-loop admission, fairness, cancellation, reclaim, and
streaming delivery.

### Phase D — gfx1100 production model-owning continuous loop

**Objective:** connect the proven and measured gfx1100 c4 GGUF model step to one
loop that survives API calls and changes membership while generation is active.

D1. Runner ownership:

- [x] Add one long-lived GGUF runner implementing the scheduler-facing prefill,
      decode, compact, and reclaim contracts.
- [x] Move model/session creation outside `generate()` calls.
- [x] Remove the request-lifetime generation lock from the model execution path.
- [x] Make `SubmitPollTextGenerator` adapt to the shared loop rather than create
      a loop per call.

Clean `285c20b4` W7900 evidence compares the old direct resident control, the
first shared-loop call, and a second call for two independent 16-token/d4 rows.
All three trajectories are exact (`[9707,9707,9707,9707]` and
`[9708,9709,9708,9709]`); all four preallocated session identities remain
stable and all four return idle after each call. The final route records three
native c2 model steps with no serial fallback. Focused host validation is **78
passed, 4 skipped** and includes context-pool rebuild, greedy packed/c1,
sampled and zero-token compatibility, commit, compact, and reclaim contracts. D1 does
not claim live admission: each synchronous caller still drives loop ticks, and
real prompt execution occurs at the final prefill commit rather than being
interleaved with live decode. See the compact [D1 closure artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d1-resident-model-runner-closure.json).

D2. Real scheduling:

- [x] Admit a request while another row is decoding.
- [x] Interleave bounded prefill chunks with decode according to the selected
      policy.
- [x] Retire EOS/max-token rows and reuse slots without stopping live neighbors.
- [x] Cancel/disconnect one row without cancelling or corrupting its group.
- [x] Preserve deterministic token/state/KV equality across membership changes.

Clean `b51bd688` W7900 evidence uses four controlled p512 requests, 256-token
prefill chunks, the `protect_ttft` policy, and one c4 resident pool. A emits two
tokens before B is admitted; B executes `256+256`, joins A, runs one native c2
step, and is cancelled while packed state is dirty. C reuses B's exact session,
executes `256+256`, joins A, and retires at max-tokens after another native c2
step. D reuses that session again and is cancelled after its first 256-token
chunk. A remains live throughout and matches its independent c1 trajectory at
all eight tokens. A after B cancel, B before reset, A after C retire, C before
reset, A across D cancel, and final A each have zero Conv/GDN or live-KV hash
mismatches against the corresponding c1/stability oracle. All four sessions
return idle; focused host validation is **380 passed, 4 skipped** and includes
EOS, max-token, cancel, disconnect, and timeout reclaim paths. This advances only
the non-streaming gfx1100 GGUF model loop to `continuous_eq_ok`; D3 device KV
allocation and D4 server streaming remain open. See the compact [D2 closure
artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d2-live-lifecycle-closure.json).

D3. Device KV pool:

- [ ] Connect scheduler admission to real paged KV allocation.
- [ ] Grow only at admission barriers; shrink only idle, free, unpinned chunks.
- [ ] Enforce high-water rejection without partial request mutation.
- [ ] Audit allocator-visible and tracked current/peak memory.
- [ ] Pass burst→steady→idle→burst with pointer/graph validity checks.

D4. API and streaming:

- [ ] Route both `LLM.generate()` and the OpenAI server through the same loop.
- [ ] Stream token events per request without holding model/session locks.
- [ ] Bound slow-client queues and isolate backpressure by row.
- [ ] Support request cancellation, timeout, disconnect, and shutdown drain.
- [ ] Keep non-streaming prompt-list behavior as a compatibility adapter, not a
      second execution architecture.

D5. Observability:

- [ ] Export pending/admitted/active counts and physical bucket occupancy.
- [ ] Export prefill/decode/reclaim work counts and scheduler policy.
- [ ] Export request queue, TTFT, inter-token, service, and completion latency.
- [ ] Export KV current/high-water bytes, pages, refcounts, grow/shrink/failures.
- [ ] Export graph hit/capture/replay/invalidation counts by bucket.
- [ ] Include route/fallback manifests in benchmark artifacts.

Exit: live requests enter and leave one real gfx1100 GGUF model loop, with exact
lifecycle evidence and no per-call inner-loop ownership.

### Phase E — cross-backend GGUF and native c8

**Objective:** make GGUF concurrency a backend-capability path rather than a
one-device special case, then raise the native group width.

E1. gfx1151 symmetry:

- [ ] Run the Phase B/C gates unchanged on gfx1151.
- [ ] Register gfx1151-specific launch variants only where measurement requires
      them; keep runner and scheduler code backend-neutral.
- [ ] Replace the current gfx1151 exact-hybrid boundaries with the same native
      model-step contract.
- [ ] Run the Phase D live-loop lifecycle gate unchanged on gfx1151.
- [ ] Keep separate gfx1100 and gfx1151 artifacts and route catalogs.

E2. Native widths:

- [ ] Support physical buckets c1/c2/c4/c8 with active masks.
- [ ] Run one true native c8 model step; c4+c4 may remain an explicit fallback
      but cannot qualify as c8.
- [ ] Pass c8 steady, ragged, sparse, cancellation, and 512/128 equality.
- [ ] Validate non-edge survivors through c8→c1 retirement without compaction.
- [ ] Validate optional compaction separately with state/KV hashes at every move.

E3. Arbitrary request counts:

- [ ] Lower arbitrary live C into one or more declared physical buckets.
- [ ] Prove tail buckets and masked lanes cannot mutate inactive state or KV.
- [ ] Record logical C, physical bucket width, number of groups, and active mask
      in every artifact.
- [ ] Establish policy for C>8 from measurement: wider native buckets, multiple
      groups, or both. Never report multiple groups as one native width.

Exit: GGUF Q4_K_M/BF16 has exact native c1/c2/c4/c8 model steps and live
admission on gfx1100 and gfx1151, with honest arbitrary-C lowering.

### Phase F — retain and tune GGUF concurrency

**Objective:** promote the exact live loop to the default and optimize measured
walls without benchmark gaming.

F1. Retention packet:

- [ ] Run c1/c2/c4/c8 prompt-512/decode-128 on gfx1100 and gfx1151.
- [ ] Include same-protocol c1, serial bridge, exact hybrid, and native rows.
- [ ] Run burst/live-admission traces with per-request latency percentiles.
- [ ] Run the complete prompt-category suite plus heldouts.
- [ ] Update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, compact artifacts,
      and `WORKLOG.md` for every retained promotion.

F2. Profile-directed tuning order:

1. remove host synchronization and metadata transfers;
2. amortize launch/graph dispatch cost;
3. improve small-C projection and LM-head geometry;
4. improve c-aware Conv/GDN and full attention;
5. improve MoE grouping/routing/selected-expert utilization;
6. overlap pool/admission work only after exact ownership is stable.

- [ ] Attribute launch count and GPU time by family for c2/c4/c8.
- [ ] Keep exact non-regressive wins and promote them to package defaults.
- [ ] Record rejected and neutral probes in `WORKLOG.md`, not this roadmap.
- [ ] Remove obsolete experiment flags through `docs/REFACTOR.md` after defaults
      settle.

Exit: native live GGUF c=N is the production default and has retained scaling
rows on gfx1100 and gfx1151.

### Phase G — PARO native concurrency

**Objective:** implement the same first-class pipeline for PARO after GGUF has
proven the runner, scheduler, KV, lifecycle, and evidence contracts.

PARO work may begin earlier when it does not destabilize the active GGUF closure
set, but it must reuse rather than fork the production loop.

G1. Re-establish c2 controls:

- [ ] Run current-HEAD gfx1100 PARO c1 and c2 exact/serial controls.
- [ ] Re-run the gfx1151 exact c2 hybrid with lifecycle, hidden/state/KV, and
      repetition gates.
- [ ] Separate graph/eager policy per backend using registered capabilities.
- [ ] Preserve true width-1 fail-closed behavior for unsupported groups.

G2. Fully native c2:

- [ ] Bisect the first hidden divergence with a reusable layer/stage comparator.
- [ ] Replace row-local full-attention and selected-c1 hybrid boundaries with
      exact c-aware routes.
- [ ] Close batch-GEMV QKV/Z/O/FFN output projections.
- [ ] Close Conv/GDN segmented state mutation and selected-expert MoE.
- [ ] Pass 512/128 direct and shrinking-lifecycle equality on both backends.
- [ ] Trace one true c2 step with no rowchunk/serial model fallback.

G3. Native c4/c8:

- [ ] Generalize the c2 algorithms rather than stacking c2 groups and calling
      them c4/c8.
- [ ] Pass the complete c1-c8 exact matrix from `docs/BENCHMARK.md`.
- [ ] Pass sparse c8→c1 lifecycle, cancellation, and compaction gates.
- [ ] Capture/replay validated width/context buckets where replay wins.
- [ ] Retain backend-specific c1/c2/c4/c8 rows with honest native widths.

G4. Attach PARO to the production loop:

- [ ] Implement the shared runner contract without a PARO-only scheduler.
- [ ] Exercise live admission, chunked prefill, decode, cancellation, reclaim,
      streaming, and device KV pool on both backends.
- [ ] Route `LLM.generate()` and server requests through the same model-owning
      loop.
- [ ] Promote only after the same Gate 1–5 packet used by GGUF passes.

Exit: PARO W4/BF16 has retained native continuous batching on gfx1100 and
gfx1151.

### Phase H — coverage and advanced composition

These items follow the Q4_K_M/PARO-W4 BF16 core. They must not broaden the active
closure set prematurely.

- [ ] GGUF Q5_K, Q6_K, and Q8_0 c=N route/equality coverage.
- [ ] PARO INT8 KV c=N coverage.
- [ ] Seeded temperature/top-k/top-p/repetition-penalty sampling.
- [ ] Prefix reuse through `RadixCache` with device-pool refcounts and COW.
- [ ] Long-context 4K/32K/128K admission and memory-pressure policies.
- [ ] Graph-pool invalidation under real grow/shrink events.
- [ ] DMS/KVTC tier movement under the stable-id/rebind contract.
- [ ] MTP/DFlash verify/commit/scatter as a new scheduler work class.
- [ ] Tensor-parallel compatibility under `docs/TENSOR_PARALLEL.md`.

MTP/DFlash throughput must use a true no-MTP AR baseline and the complete
multi-prompt acceptance suite. Speculative work never weakens the AR c=N gates.

## Active execution queue

Work this list in order unless a measured blocker is recorded in `WORKLOG.md`.
The active lane is deliberately narrow.

1. **Completed — C4:** prove one fully native replayable c4 model step.
2. **Completed — C4:** publish the direct c1/c2/c4 and chunked-c8 performance packet.
3. **Completed — D1:** attach that c4 step to one long-lived gfx1100 model runner.
4. **Completed — D2:** close live admission, retirement, and cancellation on W7900.
5. **Active — E1:** run the same model-step and loop gates unchanged on gfx1151.
6. **E2:** generalize from native c4 to one true native c8 group.

Do not start broad c8 tuning, PARO c4/c8, prefix caching, DMS, or speculative
integration before item 6 unless the current blocker explicitly depends on it.

## Coverage ledger

Use only these status values:

- `not_started` — no current-HEAD evidence on that backend;
- `primitive_ok` — kernel primitives pass, no model equality;
- `token_diag` — generated-token equality only; no hidden/state/KV lifecycle proof;
- `direct_eq_ok` — direct steady/sparse/ragged token/hidden/state/KV equality passes,
  but the complete standard lifecycle/prompt gate is still open;
- `exact_hybrid` — the complete direct lifecycle gate passes with declared row-local work;
- `native_eq_ok` — native direct/lifecycle equality passes;
- `continuous_eq_ok` — live admission/reclaim equality passes;
- `retained` — Gate 1–5 artifact is promoted.

| Path | gfx1100 | gfx1151 | Target |
| --- | --- | --- | --- |
| GGUF Q4_K_M / BF16, c2 | `direct_eq_ok` | `exact_hybrid` | `retained` |
| GGUF Q4_K_M / BF16, c4 | `retained` | `exact_hybrid` | `retained` |
| GGUF Q4_K_M / BF16, c8 native group | `not_started` | `not_started` | `retained` |
| GGUF Q4_K_M / BF16, live admission | `continuous_eq_ok` | `not_started` | `retained` |
| PARO W4 / BF16, c2 | `token_diag` | `exact_hybrid` | `retained` |
| PARO W4 / BF16, c4 | `not_started` | `not_started` | `retained` |
| PARO W4 / BF16, c8 | `not_started` | `not_started` | `retained` |
| PARO W4 / BF16, live admission | `not_started` | `not_started` | `retained` |

Update this table only when the named gate changes. A new timing alone does not
advance status.

## Implementation guardrails

### Registry boundaries

- No `if backend == ...` or `if quant == ...` in engine, scheduler, or model
  dispatch code.
- Backend/quant capability selection belongs in the four-axis plugin registry or
  package metadata.
- gfx1100 and gfx1151 may register different launch variants behind the same
  model-step contract.
- Every fused route retains a numerically equivalent unfused chain.

### Correctness before tuning

- Add a RED fixture before changing model math whenever practical.
- Preserve independent c1 sessions as the external oracle until c=N becomes the
  canonical route and a frozen fixture exists.
- Never keep a speed win that regresses required token/state/KV equality.
- Fail closed to true c1 for unsupported shapes; never silently enter an
  unvalidated native route.

### Honest execution labels

Every route emits:

```text
logical_concurrency
physical_bucket_width
native_group_width
native_group_count
active_mask
prefill_route
decode_route
state_route
attention_route
moe_route
sampler_route
serial_or_row_fallbacks
graph_bucket_key
graph_replay_count
```

`native_caware_decode=true` is an execution label, not a correctness or
throughput claim. The coverage status and artifact gates determine the claim.

### Memory accounting

Report both:

1. tracked allocator/pool bytes; and
2. process/GPU-visible current and peak memory.

The difference is unattributed runtime/driver/workspace overhead and remains
visible. Prefix sharing claims report bytes avoided, extra metadata, refcounts,
and copy-on-write cost.

### No benchmark gaming

- Do not specialize to fixed prompts, token IDs, candidate IDs, or measured
  completion patterns.
- Do not tune only the first layer or one synthetic row shape and generalize the
  claim.
- Do not compare c=N aggregate throughput against c1 without per-request and
  serial-bridge ratios.
- Do not promote diagnostics with stale hardware, a different model fingerprint,
  or undeclared fallback paths.

## Failure handling

| Failure | Action |
| --- | --- |
| Primitive mismatch | Stop model work; add/fix the primitive RED fixture. |
| First hidden drift | Capture the earliest layer/stage; do not tune downstream kernels. |
| Token equality but state/KV mismatch | Reject lifecycle gate; token coincidence is insufficient. |
| Exact route slower than serial | Keep as correctness anchor, profile family/launch walls, and do not promote. |
| Graph/eager mismatch | Disable replay for that backend/bucket and keep eager canonical. |
| Pool pointer changes under replay | Invalidate/rebind graph before reuse; treat stale execution as correctness failure. |
| Unsupported C/context/KV dtype | Fail closed to declared groups or c1 and record the fallback. |
| Cancellation harms neighbors | Reject production loop; cancellation must be row-scoped at commit. |
| OOM/high-water rejection | Reject admission atomically; do not partially mutate request or KV state. |

## Documentation and result discipline

For each meaningful iteration:

1. Put commands, measurements, diagnosis, and next action in `WORKLOG.md`.
2. Put compact machine-readable evidence in `benchmarks/results/` when the
   benchmark protocol applies.
3. Add only a one-line result pointer here if it changes current truth,
   coverage status, phase exit, or queue order.
4. Update `benchmarks/README.md` and `benchmarks/CHANGELOG.md` only for retained
   benchmark rows under the repository evidence policy.
5. Update `docs/REFACTOR.md` when an experiment flag, duplicate path, or fallback
   should be removed after promotion.

Do not append experiment narratives, profiler tables, or per-iteration queues to
this file. Git history and `WORKLOG.md` preserve them.

## Definition of done

This roadmap is complete only when all of the following are true for both GGUF
and PARO on both gfx1100 and gfx1151:

- [ ] c1/c2/c4/c8 native model steps pass token/hidden/state/KV equality.
- [ ] Sparse retirement, cancellation, compaction, and new admission pass while
      neighbors remain live.
- [ ] One model-owning loop serves `LLM.generate()` and the OpenAI server.
- [ ] Prefill/decode scheduling, real device KV allocation, reclaim, streaming,
      and backpressure are integrated.
- [ ] Greedy and normal per-row sampling are supported under their gates.
- [ ] Profiler evidence proves the claimed native widths and replay paths.
- [ ] Retained c=N rows beat c1 aggregate and the serial bridge without
      regressing per-request latency or memory outside documented tradeoffs.
- [ ] The first-class path is package-default; obsolete bridges/flags are removed
      or have concrete blockers in `docs/REFACTOR.md`.

Until then, the honest project claim is: **host scaffolding and exact hybrids
exist; fully native production continuous batching is in progress.**
