# Concurrency and KV Architecture, Generation 2

Last updated: 2026-08-20.

_Status: Generation-2 implementation spans C2-0 through C2-8; dense gfx1100
short/long serving and the canonical W7900 production load are retained, while
cross-backend/external and DMS product closure remain open. The executable audit
reports 31 passed, 3 blocked, and 1 unavailable rows. This document remains the
source of truth for the server scheduler, request
lifecycle, and shared KV-pool architecture. [`CONCURRENCY.md`](CONCURRENCY.md)
remains the historical c=N kernel/resident-runner record._

Related source-of-truth documents:

- [`PLAN.md`](PLAN.md) — project architecture and plugin invariants.
- [`KVCACHE.md`](KVCACHE.md) — storage formats, `KVLiveSpans`, dense INT8, and
  FastDMS-derived compact DMS.
- [`BENCHMARK.md`](BENCHMARK.md) — correctness and performance evidence.
- [`TESTING.md`](TESTING.md) — RED/GREEN workflow and test tiers.
- [`QWEN38-INT8-KV-CONTINUOUS.md`](QWEN38-INT8-KV-CONTINUOUS.md) — the
  no-mirror INT8 storage/kernel qualification campaign; it plugs into this
  scheduler and must not create a second concurrency implementation.
- [`NATIVE_SPEC_CYCLE.md`](NATIVE_SPEC_CYCLE.md) — speculative transaction and
  verification work classes.

## Current implementation and product status

| Phase | Implemented / retained | Product status / remaining gate |
| --- | --- | --- |
| C2-0 contracts/simulator | Complete; current affected bundle 68/68 plus fixture/smoke checks. | Closed. |
| C2-1 sole engine service | Independent blocking/SSE/library children, terminal reclaim, per-child rejection/cancellation. | Closed. |
| C2-2 resource admission | Format-neutral atomic ledger, fit-aware bounded lookahead, named pressure telemetry. | Closed. |
| C2-3 global pool/dense | Stable `GlobalKVPoolSet`; gfx1100 Qwen GGUF BF16 uses `global-arbitrary-pages:g1`. | Dense short route retained; legacy chunk path remains for unported packages. |
| C2-4 prefix cache | Generation-checked immutable snapshots, COW, quotas, LRU/TTL, pressure eviction. | Dense host conformance closed; DMS prefix remains deliberately off. |
| C2-5 token budget/c1-c32 | Logical c1-c32, byte-exact physical-c4/c8 decomposition (registered `(1,2,4,8)`), same-round prefill/decode fairness. | Closed for retained gfx1100 short route; c4/c8 promoted byte-exact (steady/shrink-sparse/graph/p512) with c1-c32 end-to-end exact. |
| C2-6 production | Exact c1-c32, live refill, actual c2 1K/4K/16K/32K/64K, mixed context, pressure, changed-page graphs, and the canonical W7900 production packet. | W7900 load/default scope closed; gfx1151 and matched external serving comparisons remain unavailable. |
| C2-7 compact DMS | Strict retrofit metadata, compact extents, no-shadow host pack/decode oracle, c1-c32 lifecycle, fixture-qualified INT8 composition. | Exact Qwen artifact has no DMS retrofit; HIP correctness/rocprof/device soak remain open. |
| C2-8 optional tiering | Fingerprinted KVTC-style host/NVMe objects, quotas/LRU, atomic offload/restore/rollback/drain. | Default-off; realistic model restore-vs-recompute TTFT remains a product gate. |

Executable source-to-evidence audit:
[`2026-08-18-concurrency2-completion-audit.json`](../benchmarks/results/2026-08-18-concurrency2-completion-audit.json).

## Performance snapshot and old-design comparison

### Retained Generation-2 short-request packet

W7900, exact-file Qwen3.6-35B-A3B `UD-Q4_K_M`, BF16 KV, p128/d8,
token-budget scheduling, zero generation batch window, same-loaded-server c1
oracles. After the c4/c8 promotion, logical cN lowers to registered physical
c4/c8 buckets (plus a c1 edge):

| Logical c | 1 | 2 | 4 | 8 | 17 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggregate HTTP wall tok/s (c4/c8) | 27.443 | 34.394 | 43.337 | 46.158 | 45.797 | 44.320 |
| Aggregate HTTP wall tok/s (prior c2 cap) | 28.355 | 37.466 | 36.594 | 35.892 | 35.102 | 35.119 |
| Exact rows | 1/1 | 2/2 | 4/4 | 8/8 | 17/17 | 32/32 |

The c4/c8 promotion recovers the c2-cap cost: c4/c8/c17/c32 are **+16.6 / +28.2 /
+29.7 / +28.6%** over the prior c2-only rows, all byte-exact. The binding
same-protocol c8 comparison (one warmup, three measurements) is now **44.031
tok/s native physical-c8 versus 27.634 tok/s exact serial-c1 (+59.27%)**; it was
35.773 versus 27.586 (+29.68%) at the c2 cap. Live refill is 17/17 exact at
c17. This is the valid retained performance claim for Generation 2.
Evidence: [`c4/c8 promotion packet`](../benchmarks/results/2026-08-17-concurrency2-c2-8-w7900-shared-slot-c4-c8-promotion.json).

### Retained old implementation (not an apples-to-apples A/B)

The August 8 Generation-1 server packet used p512/d128, one warmup/three
measurements, SSE, 20 ms generation batching, a different source commit/ROCm
generation, and physical c8 kernels:

| Route | c1 | c8 | c9 | c13 |
| --- | ---: | ---: | ---: | ---: |
| Aggregate server tok/s | 72.169 | 158.542 | 137.001 | 129.507 |
| Scale vs old c1 | 1.000x | 2.197x | 1.898x | 1.794x |

These absolute rows **must not be divided against the p128/d8 Generation-2
numbers** on protocol grounds: prompt/decode lengths, HTTP protocol, batching
window, source/model fingerprint, and runtime differ. To close that gap the
current engine was re-run under the **exact old protocol** (p512/d128, SSE
streaming-primary, 20 ms generation batch window, 256 prefill chunk, ctx 1024,
1 warmup + 3 measured, independent c1 oracles, W7900 device 0).

### Measured apples-to-apples old-protocol comparison (2026-08-17)

| c | Current G2 (SSE median) | Old G1 (SSE) | Ratio |
| ---: | ---: | ---: | ---: |
| 1 | 76.371 tok/s | 72.169 tok/s | **1.058x** |
| 8 (c2 cap, pre-flip) | 47.239 tok/s | 158.542 tok/s | 0.298x |
| 8 (c4/c8, post-flip) | ~154.322 tok/s | 158.542 tok/s | **0.973x** |

- **c1**: Generation-2 is ~5.8% faster on the identical protocol — the cleanest
  apples-to-apples point, reflecting graph/workspace improvements.
- **c8 (pre-flip)**: ~70% slower **entirely because of the physical-c2
  shared-slot cap**: logical c8 lowered to c2 groups, while the old design ran
  one physical c8 — a physical-width-composition effect, not a kernel regression.
- **c8 (post-flip)**: after the shared-slot c4/c8 promotion, c8 reaches
  **~154-156 tok/s ≈ 0.973x the old design**, closing the c2-cap gap. All c8
  burst/streaming rows are byte-exact.

The separate c8 live-admission sub-gate (the new harness's join-after-N-decode
protocol, which differs from the old live test) hit a server-side request
cancellation (HTTP 499) pre-flip and a GPU page fault post-flip. The canonical
fault mechanism is now fixed and the full production packet passes, but this
specific p512/d128 join-after-N protocol still needs a focused post-fix rerun.

Evidence:
[`Generation-2 global/native`](../benchmarks/results/2026-08-16-concurrency2-c2-6-w7900-global-native-accepted.json),
[`old context-scoped c8 server`](../benchmarks/results/2026-08-08-gfx1100-context-scoped-c8-server-refresh.json),
[`measured apples-to-apples old-protocol diagnostic`](../benchmarks/results/2026-08-17-concurrency2-oldproto-p512d128-c1-c8-apples-to-apples-diagnostic.json),
and
[`c4/c8 promotion packet`](../benchmarks/results/2026-08-17-concurrency2-c2-8-w7900-shared-slot-c4-c8-promotion.json).

## Executive decision

hipEngine will have **one long-lived engine service per loaded model replica**.
That service, not an HTTP batcher and not an individual `generate()` caller,
will be the sole owner of:

- request admission and scheduling;
- resident model state and physical execution-row maps;
- one resolved KV-cache-backend pool set shared by all compatible requests;
- prefix-cache ownership and eviction;
- graph/workspace selection;
- token commit, completion, cancellation, and reclaim.

Every prompt/choice becomes an independent child request with its own stable
request ID and output collector. Blocking and streaming endpoints submit the
same request type to the same engine. A blocking endpoint buffers its own
output; it does **not** submit a static backend batch and wait for every sibling
before resolving. A multi-prompt or `n>1` API call may aggregate its child
results at the HTTP boundary, but each child frees backend resources as soon as
that child terminates.

The first backend will use one preplanned global dense page-pool set with stable
block IDs. A pool set may contain several format-owned planes (for example K/V
payload, scales/zeros, protected BF16 windows, or codec metadata); the scheduler
never assumes that one logical token is one BF16 page. Full immutable prefix
pages may be shared by refcount; writable tails are private or copy-on-write;
zero-active-reference cache pages are LRU-evictable. The scheduler admits by a
backend-produced resource-claim vector and current pool state, not by a
hardcoded physical c4/c8 route width or dtype-specific byte formula. Physical
c1/c2/c4/c8 kernels are execution buckets: 23 ready rows may be lowered to
several certified groups in one fairness round without limiting resident
concurrency to eight.

FastDMS contributes the compact allocator, per-layer/per-head metadata, and
streaming no-dense-shadow shape. It does **not** supply the production request
frontend or scheduler design. Initial DMS serving will use the same engine
service and a global compact pool set, but cross-request DMS prefix sharing stays
disabled until immutable-snapshot or per-request eviction-overlay semantics pass
independent correctness and lifecycle gates. BF16, FP8, INT8, INT4/HIGGS-like,
AQUA residual, mixed BF16/INT2, and future hot formats resolve to different
KV-cache backend descriptors and kernel bundles over that same lifecycle; none
gets a format-specific scheduler, admission queue, output path, or cancellation
implementation.

The first production qualification target is smooth correctness, bounded
memory, and useful throughput from concurrency 1 through 32. Offered load may be
higher, but “unlimited concurrency” means a bounded, observable queue feeding a
capacity-managed resident set; no finite GPU or host can promise an unbounded
number of resident requests.

## Goals and non-goals

### Goals

1. Remove non-streaming head-of-line response and admission blocking.
2. Make blocking, SSE, library, cancellation, timeout, and shutdown paths use one
   request lifecycle.
3. Separate logical concurrency, resident capacity, scheduled rows, and native
   physical kernel width.
4. Share one real KV capacity pool across all compatible requests for a loaded
   model replica.
5. Support safe dense prefix reuse, copy-on-write, cache eviction, and immediate
   per-request reclaim.
6. Admit from complete memory/resource accounting before a HIP allocation or
   graph launch can fail.
7. Schedule multiple bounded prefills and all due decode rows fairly, without a
   long prompt or a slow output consumer stalling unrelated work.
8. Keep c=1 on its fastest exact route while scaling through c=32 and beyond by
   composing measured physical buckets.
9. Make DMS a KV retention topology over the same scheduler rather than a
   second serving architecture.
10. Preserve speculative KV transactions, plugin dispatch, and `KVLiveSpans`.
11. Replace BF16, INT8, or any later KV format by resolving another registered
    KV-cache backend, without editing request lifecycle or scheduling policy.

### Non-goals

- Linear 32x speedup from c1 to c32. Weight bandwidth and physical kernel
  plateaus remain real; the requirement is no correctness, memory, admission,
  or latency cliff at bucket boundaries.
- Infinite resident GPU requests. Excess work queues or receives explicit
  overload rejection.
- Copying vLLM/SGLang multiprocessing, Torch, CUDA, or ZMQ architecture.
- Enabling DMS on checkpoints without validated DMS metadata/training.
- Sharing mutable DMS state as though it were an immutable dense prefix.
- Using prefix hits, fixed prompts, or request reordering to game benchmark
  scores.

## Concurrency dimensions

The old design often used one `capacity` or `max_active_requests` value for
several different concepts. Generation 2 uses distinct counters:

| Name | Meaning |
| --- | --- |
| Offered concurrency | Requests currently presented by clients, including work not accepted into the bounded ingress queue. |
| Queued concurrency | Valid child requests waiting for resource admission. |
| Resident concurrency | Requests holding model-state and/or KV leases. |
| Ready-decode concurrency | Resident requests eligible to advance a target token now. |
| Scheduled rows | Request or verification rows selected for one scheduling round. |
| Physical group width | Rows advanced by one certified model execution, such as c1/c2/c4/c8. |
| Output concurrency | Terminal or streaming results waiting for frontend consumption. |

Use `C_offered`, `C_queued`, `C_resident`, `C_ready`, `C_step`, and
`C_physical` in telemetry and benchmark artifacts. A physical width is never an
admission limit. An operator may configure a resident cap for isolation, but it
is a separate backstop after resource feasibility, not a hidden model route
clamp.

## Current implementation audit

The current tree is valuable scaffolding, not a throwaway prototype. Preserve
what is already exact and replace the ownership boundaries that cause scaling
failures.

### Keep and generalize

| Existing component | Decision |
| --- | --- |
| Stable `RequestState`, `ActiveBatch`, `WorkItem`, and request-to-slot maps | Keep. Split resident slots from ephemeral execution rows and remove fixed physical-width assumptions. |
| `ResidentEngineLoop` admission/prefill/decode/reclaim contract | Keep the lifecycle semantics. Replace caller-driven polling and one-work-item ticks with one service-owned scheduling loop and multi-item rounds. |
| Native GGUF/PARO c1/c2/c4/c8 runners | Keep as certified execution buckets behind model/backend capability registration. |
| Row-scoped cancellation and reclaim callbacks | Keep. Move command handling into the sole engine service. |
| `KVPolicy`, `KVLiveSpans`, and speculative `begin/commit/rollback` | Keep the liveness and transaction invariants. Evolve the fixed-page `KVPolicy` shim into the `KVCacheBackend` contract below so storage codecs do not leak into the scheduler. |
| `DeviceChunkedKVPool` refcounts, COW tests, and pointer-stability checks | Reuse their invariants and fixtures. Replace one-backing/contiguous-per-request constraints with the production global pool substrate. |
| `RadixCache` complete-page matching scaffolding | Reuse token-prefix semantics. Integrate cache-owned eviction, generation checks, quotas, and the real device arena. |
| Per-row sampler and stop state | Keep. Attach it to child request records rather than submission-wide completion. |
| Exact lifecycle/profiler/evidence gates from `CONCURRENCY.md` | Keep as migration gates. |

### Replace or repair

1. **Blocking HTTP coalescing is a completion barrier.**
   `_GenerationBatcher._run()` awaits one `_run_group(group)` before selecting
   the next blocking group. `_run_group()` submits a prompt list, while
   `SubmitPollTextGenerator.generate_detailed()` waits until all request IDs in
   that submission have outputs. A short row may retire internally, but its HTTP
   future, active-request count, and the next queued group remain blocked by the
   longest row.
2. **Streaming and blocking use different ownership shapes.** Controlled SSE
   launches independent producer tasks, while blocking requests use static
   groups. Correctness converges in the resident runner, but admission,
   backpressure, response resolution, and metrics do not.
3. **Callers drive the model loop.** Multiple iterators call `poll()` behind a
   lock. The lock is tick-scoped, which is correct, but it still makes frontend
   tasks responsible for engine progress and complicates shutdown, priority,
   and output fairness.
4. **One scheduler tick executes one work item.** `next_prefill_work()` emits one
   row, and the loop chooses one prefill or one decode item. `protect_decode`
   can serialize admissions; `protect_ttft` can damage ITL; even `fair` is a
   policy around a one-item limitation rather than a token-budget plan.
5. **Capacity conflates resident slots and physical width.** Registry route caps
   protect measured configurations, but they also prevent spare KV capacity
   from admitting more logical rows that could be executed as multiple physical
   groups.
6. **The device pool is not yet fully fungible.** A request's pages must be a
   contiguous run in one backing chunk because the current runner binds one base
   pointer plus an int32 block table. Prefix sharing also requires the suffix to
   fit in that same backing. Runtime growth can therefore fragment, duplicate
   large backing layouts, and create graph pin/rebind cliffs.
7. **Admission is page-oriented and head-of-queue.** The first pending request
   owns admission order even when it temporarily cannot fit, and page count does
   not by itself budget scale planes, mirrors, prefill scratch, graphs, model
   state, or workspaces.
8. **Prefix cache and pool eviction are not one allocator state machine.** The
   current radix scaffold can retain entries, but production pressure ordering,
   stale-node generations, quotas, and general completed-prefix eviction are not
   complete.
9. **Completion storage remains submission-owned.** Scheduler completions are
   independently keyed, but the public result path consumes and releases a
   complete submission together.
10. **The current KV protocol still exposes fixed-format assumptions.**
    `KVPolicy.admission_cap()` is scalar, `KVReservation` is one block table,
    and `KVLiveSpans` special-cases one INT8 scale structure. Those are useful
    compatibility seams, but they are not sufficient for multi-plane INT8,
    HIGGS/TurboQuant codebooks, AQUA cross-layer residuals, OSCAR-style
    BF16+INT2 tiers, or cold KVTC storage. Leaving them as the C2 contract would
    recreate the v1 mistake: each format would grow its own admission and
    continuous-batching path.

## Reference review

The review used local read-only source as present on 2026-08-17:

| Reference | Revision | Files reviewed |
| --- | --- | --- |
| FastDMS | `/home/lhl/FastDMS` at `c602b0e` | `fastdms/engine/{llm_engine,scheduler,block_manager,compact_kv,sequence}.py`, model runner, compact attention, README |
| SGLang | `/home/lhl/ai/sglang/sglang` at `00ce7e31` (`v0.4.3.post2-109`; local tree dirty) | `srt/managers/{scheduler,schedule_batch,tokenizer_manager}.py`, `srt/mem_cache/{memory_pool,radix_cache}.py` |
| vLLM | `/home/lhl/ai/vllm/vllm` at `42d9a2c4c` (`v0.8.4-378`) | V1 scheduler, KV cache manager, block pool/free queue, async LLM, output processor |
| KV quantization research | `/home/lhl/kvcache-quantization-research` at `31979ce` | `README.md`, `docs/{PLAN-COMPOSE,PLAN-PROD,PLAN-KVTC,OSCAR}.md`, `DMS-to-vLLM.md`, packed/AQUA cache prototypes, FastDMS compact manager/scheduler |

These are design references, not claims about every newer upstream release.

### vLLM lessons

- One scheduler reasons in **tokens**, not separate permanent prefill/decode
  phases: `num_computed_tokens` catches up to request tokens under one batched
  token budget. That naturally covers chunked prefill, normal decode, prefix
  hits, and speculative rows.
- One global `KVCacheManager` maps request IDs to blocks from a refcounted
  `BlockPool`. Full hash-addressed blocks can remain as zero-reference eviction
  candidates; allocation evicts from the free/LRU queue as needed.
- Running work is considered before new waiting work; if allocation fails, a
  lower-priority request may be preempted and later recomputed.
- `EngineCoreOutput` is request-ID keyed. `OutputProcessor` pushes each request's
  result into its own `RequestOutputCollector`; one background output handler
  drives all async generators. This is the key response-lifecycle pattern to
  adopt.
- The reviewed scheduler notes that iterating all running requests can become a
  bottleneck at 1K+ rows. hipEngine should maintain ready indexes and touch only
  scheduled/changed records per round rather than copy that scan.

### SGLang lessons

- One `running_batch` survives across decode iterations. Finished rows are
  filtered, newly prefetched rows are merged, and mixed chunked prefill may run
  with decode rows.
- `ReqToTokenPool` separates request metadata rows from a global
  `TokenToKVPool`. The radix cache maps token prefixes to KV indices rather than
  owning separate per-request KV arrays.
- Radix nodes have lock references. Active prefixes are protected; unlocked
  leaves are LRU-evictable. Allocation can evict prefix leaves before retracting
  live decode requests.
- Request IDs map to independent async events in `TokenizerManager`; batched
  scheduler output is split and signalled per request.
- Decode retraction under pressure and prefix-aware scheduling are useful, but
  they must be bounded by starvation controls and complete resource accounting.

### FastDMS lessons

- The reusable design is the global compact storage plus per-sequence,
  per-layer, per-KV-head `base_offsets`, `range_capacity`, `live_counts`,
  `token_positions`, and `evict_mask`.
- Streaming pack writes surviving prompt K/V directly into compact storage and
  avoids retaining dense pages. Decode expires/compacts live rows and scans only
  actual `live_counts`.
- Compact admission must be based on actual/projected compact capacity rather
  than logical dense sequence pages. This is the hard scheduler boundary.
- The current FastDMS `LLMEngine.generate()` is synchronous and static: it adds
  all prompts, steps until all finish, then returns ordered outputs. Its
  scheduler is phase-separated and the DMS allocation gate partly lives in the
  model runner. That frontend/ownership shape is **not** the design to port.
- FastDMS streaming-DMS mode bypasses dense block/prefix allocation. It does not
  solve safe cross-request prefix reuse after per-head, per-sequence eviction
  diverges.

### KV quantization research lessons

- Hot storage is not one `dtype` field. HIGGS/TurboQuant-style caches own packed
  indices plus scales/codebooks; AQUA owns heterogeneous full-KV and predicted
  residual layers plus a calibration artifact; OSCAR owns BF16 sink/recent
  regions, an INT2 historical region, scale/zero planes, and a demotion step.
- Retention topology, hot codec/layout, and cold tiering are separate concerns.
  Dense paging or DMS can in principle compose with several hot codecs; KVTC is
  a turn/offload codec that restores into a hot format rather than an attention
  dtype.
- Composition order is backend behavior, not scheduler behavior. In a proposed
  DMS+OSCAR path, a token lives in a BF16 grace ring and is either evicted or
  demoted to INT2 when it ages out. The scheduler needs honest resource deltas
  and maintenance work, not knowledge of that algorithm.
- Mixed-layout concurrency requires stronger ownership gates than c1 quality.
  The reviewed OSCAR evidence includes a reported concurrent long-context
  corruption after mixed-KV slot-accounting changes. Abort, retract, demotion,
  page reuse, and reclaim must therefore be backend-conformance tests, not
  optional format-specific cleanup.
- FastDMS demonstrates why allocation failure discovered in model-runner
  preparation is too late. Every format must expose provisional reserve,
  commit/release deltas, fragmentation, and all transient workspace before a
  work item is scheduled.

### Synthesis

Adopt vLLM's central token-budget owner and per-request output collectors,
SGLang's global token pool plus lock-aware radix lifetime, and FastDMS's compact
metadata/no-shadow storage. Add a format-neutral resource-claim/pool-set
contract so the research formats are replaceable rather than scheduler forks.
Keep hipEngine's smaller torch-free host, backend-neutral registry,
`KVLiveSpans`, exact c-aware kernels, and explicit KV transactions.

## Width-adaptive GEMM selection: measured model and dispatch policy

The engine must serve an **arbitrary, time-varying** number of concurrent
requests. That means it cannot ship a hand-picked kernel per width: it has to
*determine from measurement*, on each target backend, which kernel and which
group decomposition are best for the widths it actually sees, and then use
them. This section states the model, the current W7900 measurements, and the
decision procedure the packed-decode path must implement.

Three decisions have to be made, and they are separable:

- **D1 primitive and complete-group choice** — which registered operation
  variants and physical-group route run for a qualified artifact/profile/shape.
- **D2 group decomposition** — for `W` ready rows, which sequence of certified
  `(active rows, physical rows, mask)` groups executes.
- **D3 coverage** — every relevant quant/layout/operation and every logical width
  has bounded measured regret versus the best certified composition.

Numbers below are W7900 / gfx1100 observations on one model. **The procedure
is the normative part; the constants are not portable.** Every quantity marked
`MEASURE` is re-measured per backend and per layer shape.

### Where the packed-decode cost actually is

**Current clean direct curve (after the Q5T16 and planar-qmicro Q6T16 rows 5-8
promotions).** W7900 / gfx1100, exact-file Qwen3.8-27B-Q4_K_M, BF16 KV, one
shared load, graph replay, p128/d8, one warmup plus two measured runs, commit
`3ea17c73`. Every direct c1-c8 trajectory matches the repeating independent
c4/c1 fixture and both controls are exact.

| direct physical c | aggregate tok/s | scale vs c1 | (was, pre-Q5/Q6) |
| ---: | ---: | ---: | ---: |
| 1 | 30.303 | 1.000× | 30.220 |
| 2 | 53.788 | 1.775× | 53.672 |
| 3 | 75.474 | 2.491× | 75.493 |
| 4 | 93.490 | 3.085× | 93.603 |
| 5 | 105.673 | 3.487× | 67.173 |
| 6 | 115.295 | 3.805× | 74.000 |
| 7 | 122.364 | 4.038× | 63.483 |
| 8 | 127.323 | 4.202× | 69.747 |

Both the Q5T16 (48 calls / 4.30 ms at c8) and planar-qmicro Q6T16 (64 calls /
10.76 ms at c8) now keep their true rowtiles through rows 8, closing the
c5/c7 cliffs. **Native c8 is now 127.32 tok/s and beats honest two-c4 chunked
c8 (91.13 tok/s) by 39.7%**; `native_c8_scaling_gate_passed = True`. c1-c4 are
unchanged (within noise). These are model-step results, not production-server
rows: the continuous owner still advertises only physical `(1,2,4,8)` and
rounds c3 to masked c4 and c5-c7 to masked c8. Direct c3/c5/c6/c7 are not yet
product-reachable.

**Kernel threshold traces (one profiler-instrumented decode transition;
durations are diagnostic).** The pre-promotion controls localized the cliffs;
the clean combined post-promotion c8 census proves their removal:

| state / width | total kernel sum | dense projection | Q5T16 route (48 calls) | planar-Q6T16 BF16 route (64 calls) |
| --- | ---: | ---: | --- | --- |
| baseline c4 | 39.1 ms | 28.1 ms | true col4 rowtile, 3.34 ms | true col8 rowtile, 6.79 ms |
| baseline c5 | 69.0 ms | 60.9 ms | padded WMMA, 21.31 ms | direct per-row GEMV, 17.35 ms |
| baseline c7 | 104.6 ms | 95.0 ms | padded WMMA, 19.84 ms | padded WMMA, 51.39 ms |
| **post-promotion c8** | **54.1 ms** | **43.3 ms** | **true col4 rowtile, 4.30 ms** | **true col8 rowtile, 10.76 ms** |

The post-promotion c8 trace has zero Q5/Q6 WMMA, Q5 row8 at VGPR72/LDS512/
scratch0, and planar-Q6 row8 at VGPR136/LDS1024/scratch0. Both true rowtiles
now cover rows2-8. The earlier numerical similarities c5≈c4+c1 and c7≈c6+c1
did **not** imply decomposition—the baseline manifests/traces showed one
physical c5/c7 group.

**Exact decode-path tensor census.** The active model has 64 decode layers, not
all 65 GGUF blocks:

| role | active decode quant/count |
| --- | --- |
| `ffn_gate`, `ffn_up` | Q4_K 64 each |
| `ffn_down` | Q4_K 32 + Q6_K 32 |
| `ssm_out` | Q5_K 48 |
| `attn_qkv` | Q4_K 24 + Q6_K 24 |
| `attn_v` | Q4_K 8 + Q6_K 8 |
| `attn_output` | Q4_K 16 |

Therefore the c7/c8 fallback count is **exactly 112**: 48 Q5 calls plus 64 Q6
calls (`32 ffn_down + 24 attn_qkv + 8 attn_v`). The earlier expectation of 114
counted block 64's Q6 `ffn_down` and `attn_v`; that block is NextN and does not
execute in normal AR decode. The bucket spans FFN, GDN, and attention because it
is a quant/layout cut, not a stage cut.

### Roofline: three roofs and a parallelism floor

For a quantized GEMM with `M` rows, `K` inputs, `N` outputs, `bpw` bytes per
weight, arithmetic intensity is `AI = 2·M·K·N / (K·N·bpw) = 2M/bpw` FLOP/byte —
**linear in M, independent of the layer shape**. But there is no single
compute roof to ride against, and RDNA3 makes the distinction matter:

| roof | W7900 value | source |
|---|---|---|
| VRAM bandwidth | 864 GB/s theoretical; 650–735 GB/s sustained for large streams (75–85%) | `ROOFLINE.md` §1.4 |
| L3 / Infinity Cache | 96 MB @ 2.30 TB/s | `ROOFLINE.md` §1.2 |
| matrix (BF16 WMMA) | 123 spec / **84.8 measured** TFLOP/s | `ROOFLINE.md` §1.3 |
| vector (FP32 FMA) | 30.7 TFLOP/s, 61.3 with VOPD dual-issue | `ROOFLINE.md` §2 |

- **Matrix is only 1.4–2.8× vector on this architecture** (84.8 measured vs
  30.7/61.3). Unlike CDNA or NVIDIA parts, "move to matrix cores" is not
  automatically a large win, and a well-dual-issued vector kernel is a
  legitimate competitor at moderate M.
- **Dequant shares the SIMD issue port.** A Q4 kernel spends ~4 VALU ops per
  weight building operands; those cycles are not available to WMMA. For the
  current dense WMMA kernel the B-fragment build is roughly 64 VALU ops per
  32-cycle WMMA per row-tile, so the achievable compute roof is ~28 TFLOP/s at
  one row-tile per block and ~56 at four — **not 84.8**. Ridge points computed
  against 84.8 are upper bounds only.
- **Parallelism is a fourth limiter that shape controls.** The grid must fill
  192 SIMDs. The down projection (`K=17408, N=5120`) generates 3.4× fewer
  output tiles than gate/up for identical weight bytes, and that alone —
  not bandwidth — explains its behaviour under the WMMA kernel (below).

Consequently there are **two ridges, and each kernel must be judged against
its own**: `M_ridge = AI_ridge · bpw / 2`.

| ridge | AI_ridge | M_ridge (bpw = 0.578) | applies to |
|---|---:|---:|---|
| matrix, spec/theoretical | 142 F/B | 41 | upper bound only |
| matrix, measured/sustained | 115 F/B | 33 | WMMA-class kernels |
| matrix, dequant-derated | ~38–76 F/B | 11–22 | this Q4 WMMA family |
| vector, no VOPD | 35.5 F/B | 10 | rowtile-class kernels |

The single "`M_ridge ≈ 27`" figure previously in this section mixed a measured
numerator with a theoretical denominator and applied a matrix ridge to a
vector kernel. Quote the range and the kernel class, not the point estimate.

### Measured kernel behaviour (corrected)

`bpw` for the Q4 T16 tile layout is **exactly 0.578125** — 2368 B per 16 cols ×
256 k (`gguf_t16_selected_gemv.hip:27,40-45`: 32 + 32 + 128 + 128 + 2048), so a
5120×17408 tile is 51.53 MB. The probe's 0.5 fallback undercounts by ×1.156.

`dense_rowtile` (vector class, gate/up 5120×17408, per call):

| M | ms | marginal ms/row | note |
|---:|---:|---:|---|
| 2 | 0.081 | — | |
| 3 | 0.083 | 0.002 | last width with meaningful weight amortization |
| 4 | 0.093 | 0.010 | |
| 7 | 0.124 | 0.010 | |
| 8 | ~0.134 | 0.010 | extrapolated; production uses the fused dual kernel |

The marginal cost is **0.0103 ms/row = 8.65e12 FMA/s = 56% of the non-VOPD
vector roof** — i.e. above M≈3 this kernel is vector-issue bound, and each extra
row costs full price. It is not a bandwidth kernel at M=8, and no amount of
tuning makes it one. Two consequences: (a) "reach rowtile-grade bandwidth
efficiency at M=16/32" is not a coherent target for a rowtile-shaped kernel;
(b) `acc[ROW_TILE][8]` is 64 VGPRs at ROW_TILE=8 and 128 at 16, so the register
tile caps this family near M≈8–12 regardless. The unexploited lever here is
VOPD dual-issue (56% → potentially ~112% of the scalar roof).

**Closed Q4 `ROW_TILE=16/32` prototype.** The predeclared prediction was
falsified quantitatively on the down shape: direct row8/16/32 measured
0.1530/0.2744/5.3206 ms in the same single-tile synchronized diagnostic.
Row16 is only **1.115×** faster than 2×row8 (0.3060 ms), not the predicted
1.40×; row32 is **8.69× slower** than 4×row8 (0.6120 ms), confirming catastrophic
register/scratch pressure. The hot tile fits in L3 and timing includes per-call
synchronization, so these values are not width-map or bandwidth evidence, but
the A/B is sufficient to reject row32 and deprioritize row16. No production
physical-16 route consumes it. The dirty launcher/wrapper changes and all six
obsolete `tmp_*` probes were discarded; row8 remains the supported Q4 owner.

`t16_wmma_prefill` (matrix class, gate/up, per call): 0.338 / 0.402 / 0.397 /
0.411 / 0.418 / 0.465 ms at M = 1 / 2 / 4 / 8 / 16 / 32.

That curve is nearly flat because **the kernel executes the same MAC work at
every M in 1..64**: `ROW_TILES_PER_BLOCK = 4` and `grid.y = ceil(rows/64)`
(`gguf_k_t16_selected_prefill.hip:533-537, 1686-1687`), and the `valid_row`
guard suppresses only the *load*, substituting row 0, while the WMMA issues
unconditionally. At M=16 it therefore computes 64 rows to deliver 16. Its
executed rate is 2×64×89.13e6 / 0.418 ms = **27.3 TFLOP/s ≈ 32% of the measured
matrix roof** — an unremarkable number for a kernel that also dequantizes in
registers. It is *additionally* occupancy-starved: 32-thread blocks, 363 blocks
for gate/up (1.9 waves/SIMD) and **107 blocks for the down shape** (~1 wave per
CU), the latter costing ≈1.0 ms/call for the same weight bytes.

So the earlier diagnosis — "latency/occupancy-bound at 11% of peak bandwidth,
and closing 11% → 64% is the biggest lever" — named the wrong mechanism and the
wrong metric. The kernel is **row-padding bound first, parallelism bound
second**, and a bandwidth percentage is not a meaningful score for a kernel that
over-computes 4×. The corresponding fixes are a template parameter and a grid
change, not a new kernel.

**Measurement caveat that bounds all of the above.** The probe hot-loops a
single 51.53 MB tile, which fits in the 96 MB Infinity Cache. After the first
iteration the weights are L3-resident, so the "% of VRAM peak" column measured
nothing about VRAM streaming — and against the 2.30 TB/s L3 roof both kernels
are far from any bandwidth limit, which independently supports the compute/issue
diagnosis. Production reads ~10 GB of FFN weights per step with zero reuse. Any
retained bandwidth claim must come from a tile-rotating probe (below).

### Corrections to the previous version of this section

| retracted claim | status |
|---|---|
| "The FFN linear layers are dominant" / "non-FFN is 76%" | Both came from the invalid `step − one Q4 probe ×64` partition. Do not retain either stage percentage. The measured bottleneck is the Q5/Q6 projection routes across FFN, GDN, and attention. |
| "Rowtile multi-group beats the wmma prefill at every M>8 (2.9× at 16, 1.45× at 32)" | Priced an 8-row group at the M=2 efficiency. Measured: 2×0.134 = 0.268 vs 0.418 → **1.56×** at M=16; 4×0.134 = 0.536 vs 0.465 → **wmma wins 1.15×** at M=32. Crossover is M≈24–32. |
| "wmma prefill is latency/occupancy-bound at 11% of peak BW" | Wrong mechanism and wrong metric; it is row-padding bound (64 rows of MAC work at every M ≤ 64), measured under L3 residency. |
| "Rowtile is a good bandwidth kernel (64% of peak)" | True only at M≈2–3, and only against a VRAM denominator the probe could not exercise. Vector-issue bound from M≈4. |
| "M_ridge ≈ 27" | Mixed measured/theoretical roofs and applied a matrix ridge to a vector kernel. Use the ridge table above. |
| "Aggregate stays flat at ≈c8 (~68 tok/s)" | Pre-promotion measurement exposed c5/c7 cliffs; the retained post-promotion curve is 30.30/53.79/75.47/93.49/105.67/115.30/122.36/127.32 tok/s. |
| "No multi-group scheduler exists" / "D2 is unimplemented" | Grouping exists and is exact, but uses largest-width chunking plus ceiling padding. Before Q5/Q6, two-c4 beat native c8; post-promotion native c8 wins 127.32 versus 91.13 tok/s. The missing piece is artifact-backed optimization for c9-c14 remainders and other SLOs. |
| "Close the 11% → 64% BW gap is the single biggest lever" | Wrong metric. The largest current kernel lever is true Q5/planar-Q6 rowtile coverage through c8; the engine lever is cost-aware group selection. |
| "The non-FFN cliff is GDN/full-attention" | False. Baseline traces localized Q5/Q6 quant-layout projection cliffs spanning all stages; the rows2-8 Q5/Q6 promotions close them and cut c8 kernel sum to 54.05 ms. |
| "c5 runs as c4+c1; c4→c5 and c6→c7 are D2" | False. The direct benchmark manifest and traces show one physical c5/c7 group. Similar wall times were coincidence; the two jumps are the Q5 and Q6 thresholds above. |
| "Mixed-quant effective width must be clamped to the minimum family cap" | Too strong. One group can mix registered variants. Price each operation and the complete group; use D2 when the mixed route loses. |
| "Q4 direct ROW_TILE=16 should win ~1.40× and row32 might spill" | Measured down-leaf result: row16 wins only 1.115× versus 2×row8; row32 is 8.69× slower than 4×row8. Prototype discarded. |

### Dispatch policy the engine must implement

**D1 — resolve two measured maps, not one heuristic.** The primitive map chooses
a registered implementation for one operation. Its key includes physical host
and hardware identity, backend, model/artifact and layout fingerprint,
execution profile, registry quant, operation boundary (for example
`linear_pair_silu` versus `linear`), layer role, `K×N`, active rows, physical
rows/mask class, and graph/eager mode when that changes cost. Its value is a
four-axis registry key, strict fallback, correctness fixture/manifest hash,
resource data, and measured time. Width is **not** a fifth registry axis: the
cold model/package plan resolves the measured record to an immutable set of
ordinary `(backend, layer, quant, variant)` keys.

The model-step map prices a complete certified physical group after those
primitive choices, including attention, GDN/state, norms, LM head, graph replay,
and gather/scatter. The scheduler consumes this second map; it must not estimate
a serving step by multiplying one probed Q4 tensor across a mixed-quant model.
Absent/uncertified records fail closed to a registered strict route. Compact
artifacts live under `benchmarks/results/`; bulky raw autotune/profiler logs do
not.

**D2 — decomposition exists; cost-aware decomposition does not.**
`plan_physical_batch_groups(..., compact_active_rows=True)` already lowers
arbitrary logical concurrency. Today it chunks by the largest registered width
and rounds each remainder **up** to the next bucket: c3→masked-c4,
c5/c6/c7→masked-c8, c13→c8+masked-c8. This is a correct ceiling-bucket planner,
not the measured optimizer described here.

The target planner enumerates certified candidates `(active_rows,
physical_rows, mask_class, variant_manifest)` and uses dynamic programming to
minimize complete model-step wall subject to per-request ITL/fairness and
workspace constraints. Serial groups use the sum of **measured full-group**
costs; overlapped/pipelined groups require a separately measured model. Before
Q5/Q6 promotion, direct c8 was 114.72 ms versus 88.74 ms for two c4 groups,
exposing the old heuristic error. The clean post-promotion packet reverses that
decision: native c8 is **63.53 ms** versus **88.55 ms** for two c4 groups, so
the current c8 choice is now correct. Cost-aware D2 remains needed for arbitrary
remainders, masked widths, c>8 compositions, SLO objectives, and future backend
maps; its acceptance control must use the post-promotion artifact rather than
force the superseded two-c4 choice.

**D3 — coverage is bounded regret across every relevant dimension.** Every
logical width must resolve to either a certified native/masked group or a D2
composition. Coverage is per `(artifact, hardware, profile, quant/layout,
operation, active rows, physical rows/mask)`, not merely `(layer role, width)`.
A generic fallback is not automatically wrong and a rowtile is not automatically
right; the binding test is measured regret versus the best certified complete
composition under the same SLO. The earlier proposed fixed 1.25× neighbouring-
width bound is not retained as a universal constant—it must be derived per
backend/SLO and checked on the composed route.

Sparse ladders remain forbidden. A wrapper cannot advertise `2..32` while its
C switch accepts only `{2..8,16,32}`; unsupported interior widths must reject
before HIP and D2 must cover them explicitly. Coverage tests walk every
quant/layout/operation/width record, assert selected and strict-fallback registry
keys exist, compare the declared route with the measured artifact, and then walk
logical c1-c32 through the actual scheduler.

**The quant/layout axis explains the current cliffs but does not globally clamp
physical width.** On this exact Qwen3.8 artifact, standard Q4T16 has true
rowtiles through 8; Q5T16 has a true rowtile through 8 (promoted 2026-08-20);
planar-qmicro Q6T16 now also has a true col8 rowtile through 8 (promoted
2026-08-20), replacing its disguised direct-per-row rows5/6 fallback and its
padded-WMMA rows7/8. The full model may mix those variants in one physical
group, so there is no correct rule saying "clamp the engine to the minimum
family width." D1 prices each operation and D2 prices the complete mixed route.

**Promotion gate.** A primitive or group record becomes default only when its
declared strict/production contract passes, its route and resource provenance
are repeatable, and the actual scheduler's composed objective improves. Native
c8 is the cautionary and recovery case: Q4-only leaf wins did not beat two c4
groups, while the complete Q5+Q6 promotion now does.

### Measurement protocol (what the autotune probe must do)

Any probe whose numbers enter the width map must:

1. **Rotate weight tiles across layers** so the working set exceeds L3 (96 MB
   on W7900). A single hot tile measures cache, not the serving path. A
   single-tile probe may still A/B two kernels at one shape — per-row ms stays
   comparable — but its absolute bandwidth figures must never be quoted or
   entered in the width map.
2. **Read `bpw` from the allocation size** and fail loudly if unavailable; no
   silent 0.5 fallback.
3. **Time inside a captured graph**, not per-call `perf_counter` +
   `device_synchronize` (~5–10 µs/launch, ≈10% at 0.08 ms, and it biases the
   comparison toward the slower kernel).
4. **Measure both FFN shapes** — `K=5120,N=17408` and `K=17408,N=5120` — and the
   attention projections. Per-shape parallelism, not just M, selects the winner.
5. **Sweep M across 1..W_max**, and report executed as well as useful work, so
   padding is visible instead of appearing as low "bandwidth".
6. **Include every registered candidate**, notably
   `gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel` — it already stages
   decoded B in LDS and reuses it across row tiles, which is the structure the
   large-M path needs; it has never been benchmarked and is currently tiled at
   256 rows/block.
7. **Check register and scratch usage statically** before spending a device
   run on a wider register tile: build with `-Rpass-analysis=kernel-resource-usage`
   (or read the generated `.s`) and reject any variant with non-zero
   `ScratchSize`, or whose VGPR count drops occupancy below the level its
   latency hiding needs. The Q4 row32 prototype is already rejected; apply this
   rule to every future candidate.
8. **Measure complete physical groups and compositions.** Primitive wins enter
   the model-step map only after graph/eager full-model A/B. Measure exact native,
   masked, and serial-composed candidates with all sessions resident; never infer
   scheduler cost from a sum of one-layer microbenchmarks alone.
9. **Emit the compact artifact and rollup rows** per the evidence policy.

### Next steps, in priority order

1. ~~**Establish a complete direct c1-c8 baseline and threshold traces.**~~
   **DONE (2026-08-20).** Clean same-load graph packet at `d63b694b4`: c1-c8 is
   **30.22/53.67/75.49/93.60/67.17/74.00/63.48/69.75 tok/s**, every direct row is
   exact against the repeating independent fixture, and honest two-c4 c8 is
   **91.06 tok/s / 88.74 ms** versus native c8 **69.75 / 114.72 ms**. Clean c5/c7
   traces bind the two cliffs to Q5 and planar-qmicro Q6 projection routes.
2. **Remove the measured quant/layout cliffs (highest current kernel leverage).**
   ~~Extend Q5T16's true rowtile from 4→8~~ and ~~planar-qmicro Q6T16's true
   col8 rowtile from 4→8~~ **DONE (2026-08-20)**: both primitives now cover
   rows 2-8 (strict bit-parity vs their per-row producers) and the dispatch
   routes rows 5-8 to the true rowtiles
   (`GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT[gguf_q5_k_t16_v1]=8` and
   `[gguf_q6_k_t16_qmicro_planar_v1]=8`). The planar Q6 export no longer routes
   rows 5+ to the per-row fallback. Result c1-c8
   **30.30/53.79/75.47/93.49/105.67/115.30/122.36/127.32 tok/s** (c5 +57%,
   c6 +56%, c7 +93%, c8 +82%), all exact; native c8 kernel census:
   Q5 true rowtile 48/4.30 ms + planar-Q6 true rowtile 64/10.76 ms, zero
   Q6 WMMA. Native c8 now beats two-c4 chunked c8 by 39.7%.
3. **Implement the artifact-backed D2 resolver (host-first).** Preserve the
   current ceiling planner as strict fallback, add dynamic programming over
   certified `(active,physical,mask)` group records, and test every logical
   c1-c32. Populate it from the clean post-promotion full-step map, not hand-
   entered microbench constants. Its current-map controls must choose native c8
   over two c4 groups and evaluate c9/c13/c17 remainder alternatives honestly.
4. **Certify product reachability, then promote.** Direct c3/c5/c6/c7 are
   benchmark-only today; the continuous owner still advertises `(1,2,4,8)` and
   maps c5-c7 to masked c8. Run the dynamic serving matrix (ragged neighbours,
   retirement/refill, cancellation, graph rebuild, slot permutation, state/KV
   hashes) for every proposed physical width. Only then expand package
   capabilities and compare the actual server's chosen D2 plan with the artifact.
5. **Re-profile the post-fix ledger before wider kernels.** Attack the new top
   complete-model family. The dirty Q4 `ROW_TILE=16/32` prototype is closed:
   direct row16 down was only 1.115× faster than 2×row8, row32 was 8.69× slower
   than 4×row8, it had no product route, and all prototype code/probes were
   discarded. A physical-16 kernel is reconsidered only if the promoted D2 map
   shows repeated weight reads dominate after Q5/Q6 coverage; row32 is rejected.
6. **Port protocol.** On a new backend (gfx1151 first), regenerate both primitive
   and complete-step maps and rerun the lifecycle matrix. Copy the decision
   procedure and schemas, never W7900 constants.

## Target architecture

```text
HTTP / library callers
        |
        | submit child request / cancel / consume output
        v
+--------------------------------------------------------------+
| Frontend adapters                                            |
| ParentRequest aggregation only; no model batching            |
+---------------------------+----------------------------------+
                            | bounded command channel
                            v
+--------------------------------------------------------------+
| EngineService (one per loaded model replica)                 |
|                                                              |
|  RequestTable + ready queues + deadline/fairness policy      |
|  OutputRouter (one collector/mailbox per child request)      |
|  ResourceLedger + AdmissionController                        |
|  PrefixIndex                                                 |
|  KVCacheBackend (resolved plugin)                            |
|    KVPoolSet -> payload/scale/tier/metadata pools            |
|    topology -> dense/sliding/DMS/...                         |
|    codec    -> BF16/INT8/FP8/HIGGS/AQUA/OSCAR/...            |
|  ExecutionPlanner -> c1/c2/c4/c8/verify/prefill groups       |
|  ModelRunner + graph/workspace caches                        |
+---------------------------+----------------------------------+
                            |
                            | EngineOutput(request_id, ...)
                            v
+--------------------------------------------------------------+
| Independent collectors                                      |
| blocking buffer | SSE bounded queue | parent aggregation     |
+--------------------------------------------------------------+
```

The `EngineService` may run in a dedicated thread initially and a process later.
The ownership contract is the same. Frontend event-loop tasks enqueue commands;
they never call model `poll()` themselves and never hold the scheduler lock.

## Request and output lifecycle

### Child request is the scheduling unit

One child request represents one generated sequence. It owns:

```text
request_id                 stable for the complete lifecycle
parent_id / choice_index   optional HTTP aggregation metadata
phase                      queued | prefill | decode | verify | terminal
prompt and output tokens   canonical token history
sampling/stop state        independent per row
priority/deadline          scheduling metadata
resident_slot              stable model-state slot, if admitted
kv_lease                   backend-owned typed resources, if admitted
output_collector           blocking buffer or streaming mailbox
resource_estimate          admission and growth accounting
```

A multi-prompt call or `n>1` creates a `ParentRequest` plus child records. The
parent can preserve public ordering and wait for all choices, but it owns no KV,
model slot, or physical batch. Child completion is never delayed for parent
completion.

### One lifecycle for blocking and streaming

1. Parse/tokenize and validate the child request.
2. Put a `SUBMIT` command on the bounded engine channel.
3. The engine records it in the request table and admission queue.
4. The scheduler admits it when all required resources can be reserved
   atomically.
5. Prefill/decode/verify work emits request-ID-keyed token events.
6. The output router updates that child's collector.
7. At terminal commit, copy final host-visible result metadata, release model/KV
   ownership immediately, and publish the terminal event.
8. Blocking code resolves that child's future; SSE emits its terminal chunk.
   Parent aggregation happens outside engine ownership.

Non-streaming is therefore “stream internally, buffer locally,” not “form a
static prompt-list backend call.”

### Independent completion invariant

If requests A and B share an execution group and A finishes first:

- A's terminal result is publishable immediately;
- A's model slot and private KV are reclaimable at that commit barrier;
- a queued request C may be admitted into the freed capacity before B finishes;
- B's output, state, and KV are unchanged;
- a parent that contains both A and B may still wait to format its public
  response, but it cannot retain A's backend resources.

This invariant gets a dedicated RED test before implementation.

### Output isolation and slow consumers

The model loop must never await a client queue.

- A blocking collector appends tokens/text directly into its bounded final
  buffer; its bound derives from validated output limits.
- An SSE collector uses a bounded per-request mailbox. `put` is non-blocking.
- If one SSE consumer exceeds its mailbox budget, cancel only that request with
  `client_backpressure`; do not stop decode for neighbors.
- Disconnect and timeout enqueue O(1) cancellation commands by request ID.
- Detokenization/stop holdback state is per child, never per static submission.
- Completed collector records have a TTL/size bound so abandoned clients cannot
  become a host-memory leak.

## Engine ownership and scheduling rounds

### Sole driver

One engine driver repeatedly:

1. drains a bounded number of submit/cancel/control commands;
2. commits pending cancellations and terminal cleanup;
3. performs cache eviction/maintenance required for admission;
4. admits fitting requests atomically;
5. builds one `SchedulingRound` under token, row, workspace, and latency budgets;
6. executes its ordered work items;
7. commits each result, routes outputs, and reclaims terminal rows;
8. publishes metrics and sleeps on command/GPU readiness when no work exists.

No request-lifetime lock exists. No frontend iterator owns progress. One round
may contain several model executions; a “round” is a fairness/accounting unit,
not one giant kernel.

### Work classes

The scheduler keeps peer work classes:

- `PREFILL`: one or more bounded chunks, grouped by execution compatibility key;
- `DECODE`: ready resident requests, lowered into certified physical groups;
- `VERIFY_CHAIN` / `VERIFY_TREE`: speculative rows with scratch/journal KV;
- `MAINTENANCE`: cancellation, cache eviction, DMS compaction, graph rebind;
- `RECLAIM`: normally part of commit, never a separately delayed batch.

A work item always carries stable request IDs, resident slots, ephemeral
execution rows, token positions, the KV execution compatibility key, a
`KVBatchView` containing `KVLiveSpans`, and an honest route label.

### Token-budget planning

Replace one-prefill-or-one-decode ticks with a budgeted planner:

```text
max_batched_tokens_per_round
max_prefill_tokens_per_round
max_prefill_chunk_tokens
max_decode_rows_per_round
max_work_items_per_round
workspace_budget_bytes
round_wall_budget_us
```

Running decode rows receive due times derived from the ITL target. Queued
prefill chunks receive age/TTFT priority. The planner first protects overdue
terminal/cancellation work, then due decode, then spends bounded budget on
prefill, while guaranteeing aged prefills eventual service. Several short
prefills may share one round. Long prefills are chunked and cannot monopolize
the engine.

Mixed prefill+decode in one physical kernel is optional. Generation 2 first
needs the scheduler to plan both in one round; it may execute a prefill group and
one or more decode groups serially. A registered mixed kernel can replace those
items later if exact and faster.

### Fairness and fit-aware admission

Pure FCFS can let one temporarily non-fitting long request block many fitting
short requests. Pure best-fit can starve long requests. Use bounded bypass:

- reject requests that can never fit the configured model/backend-spec envelope;
- keep temporary no-fit requests queued with a named blocking resource;
- consider a bounded lookahead for fitting requests;
- increment an explicit bypass count/age;
- after the threshold, enter a reservation/drain mode or enforce tenant/priority
  policy so the aged request eventually fits;
- cancel/timeout is removable from every queue in O(1).

Prefix-aware priority may improve TTFT but cannot override the starvation bound
or benchmark fairness protocol.

## Stable identity and physical execution

Generation 2 has three row identities:

1. **Request ID** — stable public/scheduler identity.
2. **Resident slot** — stable row in model state and metadata while admitted;
   compactable only at a commit barrier through an explicit move plan.
3. **Execution row** — ephemeral dense row in one physical kernel group.

Execution gathers resident slots into dense c1/c2/c4/c8 rows and scatters
results back by request ID/slot map. KV pages do not move merely because
physical width changes.

For 23 ready decode rows, a backend may select `8+8+4+2+1`. Every selected row
advances once in the fairness round before a second normal decode step for the
same request. This allows `C_resident=32` with a largest certified physical
kernel of c8. Backend capabilities register supported widths, shape/context
limits, workspace estimates, and exact fallbacks. Engine code does not branch on
backend or quant.

`max_active_requests` becomes an optional operator limit on **resident child
requests**. It does not select a physical batch and is not silently clamped by a
route capability. If an operator value exceeds a hard metadata/state limit,
startup rejects it or reports the explicit effective limit and reason. The
normal primary gate is the resource ledger.

## Swappable KV-cache backend contract

### Resolve a composition, not a dtype branch

A loaded model replica resolves one immutable `KVBackendSpec` before its engine
starts. The specification identifies at least:

```text
topology_key            paged_dense | sliding_sink | dms_compact | ...
hot_codec_key           bf16 | fp8 | int8_per_token_head | higgs4 | aqua_higgs | ...
tier_key                device_only | host_offload | kvtc_cold | ...
layout_fingerprint      plane shapes, strides, grouping, protected regions
artifact_fingerprint    scales, codebooks, predictors, rotations, DMS metadata
prefix_mode             unsupported | immutable_pages | snapshot_overlay
transaction_mode        journal | scratch | snapshot | unsupported
kernel_bundle_key       prefill/store/decode/verify/maintenance capabilities
```

These dimensions describe composition; they do not assert that every Cartesian
combination is valid. A registry factory validates the complete combination and
returns one resolved `KVCacheBackend`. A DMS+BF16 backend and a DMS+INT8 backend
may share the DMS topology implementation while supplying different pool plans,
store/attention kernels, and quality artifacts. The engine sees the same
protocol in both cases.

`storage_dtype` remains useful kernel metadata, but it is not the policy object.
AQUA cross-layer dependencies, HIGGS codebooks, or an OSCAR BF16+INT2 layout
cannot be represented honestly by one dtype enum. Model and scheduler code must
not switch on any of these keys.

### Scheduler-facing protocol

The Generation-2 target replaces scalar `admission_cap()` as the scheduling
contract with operations shaped like:

```python
class KVCacheBackend(Protocol):
    spec: KVBackendSpec

    def plan_pools(self, load_plan: DeviceLoadPlan) -> KVPoolPlan: ...
    def estimate(self, request, prefix, stage) -> ResourceClaimSet: ...
    def reserve(self, claims: ResourceClaimSet) -> KVLease: ...
    def prepare(self, work_item) -> KVBatchView: ...
    def begin_transaction(self, rows, draft) -> KVTransaction: ...
    def commit(self, operation, result) -> ResourceDelta: ...
    def rollback(self, operation) -> ResourceDelta: ...
    def reclaim(self, lease) -> ResourceDelta: ...
    def prefix_lookup(self, tokens) -> PrefixMatch: ...
    def maintenance(self, budget) -> list[MaintenanceWork]: ...
```

The scheduler decides **when** a request or work item may run and owns every
commit barrier. The backend decides **what** storage resources that work needs,
how logical K/V maps to physical planes, and which registered kernels implement
it. Backend methods may return plans/deltas; they may not run an independent
queue, await frontend output, mutate another request, or make a hidden HIP
allocation after admission.

The existing `KVPolicy`/`FixedPagedKVPolicy` becomes a compatibility adapter to
this protocol while C2 lands. New codecs do not add methods to
`EngineService`; they implement the backend contract and kernel capabilities.

### Resource claims and pool sets

`ResourceClaimSet` is an atomic vector of named resources and lifetimes, not one
page count or one bytes-per-token estimate. Common claim classes include:

```text
pool_id + units/bytes          persistent payload or metadata capacity
resident metadata rows        request/head/layer descriptors
prefill staging               hidden/oracle/pack buffers
attention workspace           split-K or dequant/reconstruction partials
maintenance workspace         demotion, compaction, or tier transfer
transaction scratch/journal   speculative uncommitted state
graph/static slab delta       newly required execution bucket
whole-device reserve          unclassified runtime safety margin
lifetime                      load | lease | work_item | transaction | cache
confidence                    exact | bounded | unknown
```

A claim set may also carry a small, typed, backend-private metadata tuple (for
example private-page and next-growth-credit counts). The generic ledger
preserves this metadata opaquely; only the backend materializer interprets it,
so it cannot become a codec formula in scheduler code.

A `KVPoolPlan` may create one or many stable pools. Examples:

- BF16 dense: K and V page planes;
- per-token/head INT8: K/V payload plus K/V scale planes;
- HIGGS/TurboQuant: packed indices plus scale/codebook metadata;
- AQUA: full base layers plus residual planes and predictor artifacts;
- OSCAR-like: BF16 sink/recent pages, INT2 history, scale/zero planes, and
  demotion workspace;
- DMS plus a codec: per-layer/per-head compact payload planes plus live-span
  metadata and any codec planes;
- KVTC: a hot backend plus host/NVMe cold objects and transfer/decode workspace.

The central ledger only understands pool IDs, capacities, share/lifetime rules,
and deltas. It never derives format bytes from logical token count. Reserve is
all-or-nothing across every plane. Commit can release unused provisional units
or publish newly freed DMS/demotion capacity; rollback and reclaim must return
exactly the ownership they acquired. Every delta is tied to a lease/operation ID
so conservation is mechanically checkable.

### `KVLiveSpans` and storage views

`KVLiveSpans` remains the mandatory attention and K/V-write ABI for liveness:
`base_offsets`, `live_counts`, `token_positions`, and `evict_mask` retain their
meaning for every backend. It must be extended with, or reference, a registered
`KVStorageView` rather than accumulating a special field for each quantizer:

```text
KVStorageView
  layout_key / generation
  stable raw-pointer plane views: role, dtype, shape, strides
  device metadata descriptor pointer/size
  calibration/artifact fingerprint
```

A `KVBatchView` is `KVLiveSpans` plus this storage view and the selected
registered kernel bundle. Current `KVScaleMetadata` is the INT8 compatibility
adapter, not the universal format interface. Kernel wrappers resolve the exact
layout key and receive raw pointers; the scheduler does not inspect scale
strides, codebooks, predictor state, protected windows, or tier merge rules.
Unknown layout/kernel combinations fail before admission.

Execution groups require the same **execution compatibility key**: model and
hardware backend, topology/layout fingerprint, kernel bundle, work class,
context bucket, and physical width. This prevents accidental batching of
incompatible storage while allowing all requests using the same resolved
backend to share its global pools at arbitrary logical concurrency.

### Sharing domain and backend replacement

“All requests share the KV pool” means all requests compatible with one resolved
backend draw from its global pool set. It does not mean an INT8 page can be
reinterpreted as BF16, or that pages with different calibration artifacts may be
prefix-shared. Initial production uses one active KV backend per loaded model
replica; requests asking for another format route to another replica or fail
validation. This keeps every pool fungible within its compatibility domain.

Changing the configured backend is a replica lifecycle operation:

1. stop or redirect new admission;
2. drain active leases, or use an explicitly registered lossless transcode/
   snapshot operation;
3. invalidate backend-keyed graphs and prefix snapshots that cannot transfer;
4. destroy the old pool set and initialize the new load-time plan;
5. resume the unchanged engine service and scheduler with the new spec.

A live in-place reinterpretation is forbidden. Later multi-backend co-residency
may expose several pool sets under one ledger, but it uses the same protocol and
separate compatibility queues; it is not required for C2. The essential gate is
that adding such a pool set does not change request lifecycle, fairness,
completion, cancellation, or overload code.

## First backend: global dense paging

### One planned pool set per compatible backend

At model load, a resource planner measures or declares:

```text
device_limit
- weights and permanent model state
- stable execution/state metadata slabs
- graph and model workspace budget
- prefill scratch budget
- sampler/output device buffers
- backend/runtime safety reserve
= allocatable KV arena budget
```

The production dense path allocates the backend's complete pool set from that
budget, ideally at startup after model/workspace profiling. A simple dense
codec may plan:

```text
K_payload[layer][page_id][token_in_page][kv_head][packed_or_head_dim]
V_payload[layer][page_id][token_in_page][kv_head][packed_or_head_dim]
codec_planes[...]       # scales/zeros/codebooks only when the backend declares them
```

The pool-plane base pointers, block metadata arrays, resident-slot metadata, and
graph input slabs remain stable. Kernels consume changing block IDs and
`KVLiveSpans`/`KVStorageView`; graph capture must not embed request-private
allocation pointers. A page is protected only while referenced or in flight,
not forever because a graph once saw it.

If a backend cannot allocate a pool plane in one object, it must implement a
registered page-pointer/segment-table ABI with stable indirection. The current
“all pages for one request must fit contiguously in one backing chunk” rule is a
compatibility fallback, not the Generation-2 production design. Hot-path HIP
allocation and surprise pool growth are disabled in the promoted server mode.

### Page states and ownership

Each dense page has explicit state:

```text
FREE
ACTIVE_PRIVATE        one writable request owner
ACTIVE_SHARED         immutable full page, one or more request refs
RESERVED_CREDIT       physically reserved for the lease's next growth boundary
CACHED_EVICTABLE      zero active refs, indexed by prefix cache
PINNED_SESSION        explicit continuation/session lease with quota/TTL
IN_FLIGHT             transient execution epoch fence
```

Track active references separately from cache/session ownership. Allocation may
reuse `FREE`, then evict `CACHED_EVICTABLE`. It never evicts an active, session-
pinned, or in-flight page. All state changes are scheduler-thread operations at
commit barriers.

### KV leases and growth credits

A `KVLease` belongs to one child request and identifies shared prefix pages,
private full pages, writable tail, backend metadata, and reserved next-step
credits. Admission does not reserve the request's entire declared `max_tokens`;
that would waste most of the pool. It must reserve:

- all uncached prompt pages needed for the admitted prefill chunk/window;
- writable-tail/COW cost;
- enough growth credit for the next decode allocation quantum;
- backend metadata and model state;
- applicable workspace share and safety reserve.

Before each growth boundary the scheduler renews credits. When credits cannot
be renewed, it stops new admission, evicts cache pages, and only then considers
preemption. This guarantees that already scheduled work cannot fail halfway
through a commit while still allowing high utilization.

### Complete resource estimate

`KVCacheBackend.estimate(request, prefix_match, stage)` returns an atomic
`ResourceClaimSet`, not a single page count. The following are common claims,
not a format-hardcoded schema:

```text
payload pages/bytes
scale and auxiliary metadata bytes
BF16 mirror bytes, if any
resident model-state bytes
prefill scratch peak/share
attention/split-K workspace peak/share
graph bucket/pinned workspace delta
next-step growth credits
whole-device safety reserve
confidence: exact | bounded | unknown
```

Unknown estimates fail closed to a conservative registered backend or explicit
rejection. Admission reserve is atomic: either every component in every pool is
leased and the request becomes resident, or all provisional mutations roll
back.

## Prefix cache

### Index and allocator are separate

Use a token radix index for longest-prefix lookup, but share only complete,
immutable KV pages. The radix entry stores page handles plus a generation; the
backend pool set owns bytes, refs, and eviction state. A stale generation is a miss, never a
use-after-free.

The cache key includes every value that changes KV meaning:

```text
model artifact fingerprint and revision
adapter/LoRA identity
hardware/model/weight-quant identity
resolved KV backend topology/codec/tier/layout/artifact fingerprint
RoPE/scaling and position semantics
relevant multimodal/input hashes
prompt token IDs
```

A hit increments active refs and attaches pages to the new lease. Divergence in
a partial page allocates a private page and copies the valid prefix cells before
write. Full pages remain immutable.

### Completion and eviction

On normal completion:

- drop request refs immediately;
- keep eligible full pages as zero-active-reference `CACHED_EVICTABLE` entries;
- free private/incomplete tails unless an explicit session lease owns them;
- enforce global and per-tenant cache byte quotas plus TTL;
- evict LRU leaves before preempting live requests.

Active prefixes are protected through refs, like SGLang's lock semantics. Cache
entries do not depend on the lifetime of the source HTTP request. Session
continuation is a separate pin/lease class so normal prefix caching cannot grow
without bound.

### Prefix rollout

1. Deterministic BF16 dense complete-page reuse and COW.
2. Completed-prefix LRU ownership, pressure eviction, and stale-generation tests.
3. Sampled requests whose KV semantics are unchanged by sampling.
4. Broader historical/session boundaries with quotas and graph-safe metadata.
5. DMS prefix semantics only after the backend below is proven.

## DMS integration

### Same scheduler, different resolved backend

The DMS topology component plugs into a resolved `KVCacheBackend` and therefore
uses the same request table, admission controller, output router, and execution
planner. The resolved backend owns global per-layer compact pool planes and
per-resident-sequence metadata compatible with `KVLiveSpans`:

```text
base_offsets    [rows, layers, kv_heads] int32
range_capacity  [rows, layers, kv_heads] int32
live_counts     [rows, layers, kv_heads] int32
token_positions [rows, layers, kv_heads, capacity] int32
evict_mask      [rows, layers, kv_heads, capacity] bool
```

Port FastDMS count/rank/scatter, streaming prefill pack, append/expiry, and
compact grouped split-K attention into registered HIP kernels. Qualify the DMS
topology first with BF16 payload and no retained dense shadow, then compose the
same topology with each qualified codec through `KVCacheBackend` pool plans and
kernel bundles. That ordering is a correctness ladder, not a second concurrency
implementation. Compressed storage has an independent quality/capacity gate
under `KVCACHE.md`; switching DMS BF16 to DMS FP8/INT8/HIGGS-like storage must
not change scheduler code.

### Scheduler-owned compact admission

Do not copy the FastDMS boundary where model-runner preparation discovers
compact allocation failure. Before scheduling a DMS prefill chunk, the backend
atomically reserves bounded extents for every affected layer/head and codec
plane plus metadata and workspace. The kernel commits actual survivors and
releases unused provisional capacity. Decode append/expiry/demotion similarly
mutates canonical live counts and resource claims only at commit.

Admission uses actual and projected physical live rows, fragmentation, and
near-term growth—not logical context length and not dense pages. Export both
logical tokens and physical live cells/bytes.

### Fragmentation and compaction

Compact pool planes need extent/slab accounting in addition to free bytes.
Track largest free extent, per-layer utilization, internal fragmentation, and
compaction moves for every coupled plane. A compaction plan may relocate
backend-owned ranges only at a barrier, then atomically update stable metadata
before any graph replay. Graphs consume metadata indirection rather than
captured range addresses.

### DMS and shared prefixes

Initial rule: **one global compact pool set, private sequence/head ranges, no
cross-request DMS KV sharing.** This still shares physical capacity across all
compatible requests and obtains DMS capacity benefits without corrupting
per-sequence eviction decisions.

Future prefix reuse requires one of two independently gated designs:

1. immutable no-evict compact prefix snapshots plus private post-prefix DMS
   overlays; or
2. cloning a compact snapshot into private ranges when eviction state diverges.

A dense-prefix page cannot simply be labelled shared DMS after two sequences
make different per-head decisions. Until an overlay/snapshot design passes
state/KV/refcount/eviction gates, a DMS radix lookup returns a miss.

## Pressure, preemption, and overload

### Pressure order

When a request or growth quantum does not fit:

1. reclaim all terminal/cancelled ownership;
2. release unused provisional credits/workspace;
3. evict zero-active-reference prefix-cache pages by quota/LRU;
4. shrink or compact backend metadata/extents at a safe barrier;
5. stop admitting new resident requests;
6. optionally preempt/recompute the lowest-priority eligible live request;
7. reject new work explicitly if the bounded queue/resource SLO is exhausted.

Never partially admit and then expose a HIP OOM. Never evict active/shared-live,
session-pinned, transaction scratch needed for commit, or in-flight pages.

### Preemption

Dense phase-1 production should normally avoid live preemption through growth
credits. If enabled later, preemption is scheduler-visible recompute:

- record canonical tokens/sampling state;
- free model/KV leases at a barrier;
- return the request to waiting with a `preempted_recompute` reason;
- reuse any surviving immutable prefix on resume;
- expose count and lost-work tokens/seconds.

DMS preemption additionally needs a reproducible no-evict/full-prefix restore or
private compact snapshot; it is disabled until that gate exists.

### Bounded ingress

There are separate limits for tokenization work, queued child requests, queued
prompt tokens/bytes, resident metadata slots, and completed-result retention.
Queue full returns retryable overload (`429`/`Retry-After` at the OpenAI
boundary). A huge impossible request is rejected during validation rather than
blocking the queue. Readiness reports configured and effective limits.

## Graphs, workspaces, and state

- Graph keys describe execution shape, not request ownership:
  `(work class, physical width, context/page bucket, KV backend execution
  compatibility key, active mask class, sampler mode, draft/tree shape,
  expert/top-k shape, replay steps)`.
- Graph inputs point to stable execution/state/metadata slabs. Per-request page
  IDs and span counts are copied into those slabs before replay.
- KV page allocation or cache eviction does not invalidate a graph when arena
  and metadata pointers stay stable.
- A real arena/metadata resize increments a generation and invalidates/rebinds
  affected graphs before reuse.
- Workspaces are reserved by the resource ledger and reused by non-overlapping
  work items. Concurrent streams must declare overlapping workspace ownership.
- c=1 keeps an independently measured physical c1 graph/eager route; it is not a
  masked c8 launch.

## Complexity and scaling rules

To remain smooth beyond c32:

- queued and resident requests live in ID-indexed tables;
- cancellation/removal is O(1) plus queue-node unlink, not a full deque rebuild;
- ready queues are incremental; the planner does not scan all queued history;
- one output handler routes a batch by request ID into independent collectors;
- completion records and metrics rings are bounded;
- prefix lookup scales with matched prompt length/page boundaries, not number of
  active requests;
- allocator operations are O(pages allocated/evicted), with fragmentation
  indexes rather than full-pool scans;
- physical grouping is O(C_ready log W) or better for a small registered width
  set;
- host per-token work is measured at c1/c8/c32 before moving it to C++.

Do not add C++/Cython merely because other engines use it. First remove
submission barriers, global scans, duplicate allocation, and frontend-driven
polling; then profile. A C++ engine-step remains an extraction option if Python
planning/output overhead is material after those fixes.

## Observability contract

### Per request

- queue, admission, prefill-start, first-token, terminal, and frontend-response
  timestamps;
- queue time, TTFT, ITL samples, service time, and response-publication delay;
- parent/choice IDs, priority, deadline, finish/cancel reason;
- prefix matched/recomputed tokens and COW pages;
- resolved KV topology/codec/tier/artifact fingerprint, logical tokens,
  per-plane owned/shared/live bytes, peak bytes, and growth credits;
- preemption/bypass counts and admission-blocked resource;
- physical groups/routes/fallbacks used;
- output mailbox high-water and backpressure outcome.

### Engine/pool

- offered/queued/resident/ready/scheduled concurrency;
- per-round prefill tokens, decode rows, work-item count, and planning/commit
  wall;
- physical-width histogram and group count;
- dense pages by free/private/shared/cache/session/in-flight state;
- cache hit tokens, active refs, evictable bytes, evictions, COW copies, and stale
  generations;
- DMS live cells, allocated extents, target/actual compression, largest free
  extent, fragmentation, and compaction moves;
- complete per-pool/per-plane resource-ledger budget versus measured
  current/peak GPU, host, and cold-tier memory;
- graph hits/misses/invalidations and workspace usage;
- queue rejections, no-fit bypasses, preemptions, and recovery;
- backend-terminal-to-HTTP-response delay and slots reclaimed while a parent
  response is still pending.

The `/ready` and capability payloads derive values from the loaded engine. Do
not publish hardcoded `continuous_decode` or route-cap fields disconnected from
runtime ownership.

## Implementation roadmap

Work phases in order unless a durable blocker changes the dependency graph.
Each phase is one or more validated atomic commits, not one giant rewrite.

### C2-0 — contract and RED simulator

- [x] Add child/parent request and output-collector host types.
- [x] Freeze `KVBackendSpec`, `ResourceClaimSet`, `KVPoolPlan`, `KVLease`,
      `ResourceDelta`, `KVStorageView`, and `KVCacheBackend` host protocols before
      implementing the real scheduler.
- [x] Add a deterministic fake engine/resource ledger with queued, resident,
      scheduled, physical-width, and per-pool claim counters.
- [x] Provide fake backends with identical logical K/V but different ownership:
      dense BF16; dense INT8 payload+scales; mixed BF16+packed history with
      demotion; and DMS-like variable live spans.
- [x] RED: short A completes and request C is admitted while long sibling B from
      the old static group is still decoding.
- [x] RED: blocking and streaming child requests share scheduling order,
      cancellation, and reclaim semantics.
- [x] RED/property: random c1-c32 arrival/length/cancel/demotion/compaction
      sequences preserve unique IDs/slots, every pool's resource conservation,
      independent c1 outputs, and final drain for all fake backends.
- [x] RED/architecture: selecting another fake backend changes no
      engine/scheduler/frontend type or queue implementation.

Exit: the new lifecycle, concurrency dimensions, and backend-swap contract are
executable without GPU. The executable contracts live in
`hipengine/generation/concurrency2.py`, `hipengine/kvcache/backend.py`, and the
host-only conformance simulator/tests.

### C2-1 — independent outputs and sole engine driver

- [x] Introduce one `EngineService` command/output loop around the existing
      resident runner.
- [x] Submit every blocking/SSE/library child independently; remove static HTTP
      groups from model ownership.
- [x] Resolve/reclaim each child at terminal commit; keep only parent aggregation
      at the API boundary.
- [x] Move stop holdback, timeout, disconnect, and slow-consumer handling to
      request-owned collectors/commands.
- [x] Preserve a compatibility adapter for synchronous `LLM.generate()` that
      submits children then waits, without preventing other clients from using
      the engine.

Exit: the observed non-streaming head-of-line barrier is gone in host and real
server tests; one driver owns progress and shutdown. Native resident generators
are wrapped by `hipengine/generation/engine_service.py`; non-resident generators
retain the explicit serial compatibility adapter.

### C2-2 — generic resource ledger and concurrency separation

- [x] Implement atomic named-resource claim sets, per-pool capacities/lifetimes,
      provisional reserve, commit/release deltas, rollback, and conservation
      checks; the ledger contains no BF16/INT8/DMS formulas.
- [x] Add registered backend estimators for model state, all persistent codec
      planes, mirrors/protected regions, prefill/maintenance scratch, attention
      workspace, graphs, cold transfers, and reserve.
- [x] Split resident metadata capacity from supported physical widths.
- [x] Replace route-cap clamping with fit-aware admission plus an optional clear
      operator resident cap.
- [x] Add bounded lookahead/starvation control and impossible-request rejection.
- [x] Expose effective limits, per-plane estimates, and blocking resources.

Exit: overload is atomic/retryable and never first appears as HIP OOM for any
conforming backend; c9-c32 can remain resident while using certified <=c8
physical groups. `hipengine/kvcache/ledger.py` consumes only backend-produced
claim vectors, and its admission coordinator passes stable request IDs—not
format metadata—into the resident scheduler.

### C2-3 — production global pool substrate and first dense backends

- [x] Allocate one stable backend-declared pool set from the load-time plan.
- [x] Bind runner graphs/kernels to global pool planes, `KVStorageView`, and
      stable metadata slabs rather than request-private backing bases.
- [x] Implement typed page/plane leases, growth credits, complete state
      accounting, COW tails, and in-flight epochs.
- [x] Port current dynamic-pool lifecycle fixtures; add cross-plane
      fragmentation, partial-reserve rollback, and pressure recovery tests.
- [x] Run the same lifecycle/scheduler suite with dense BF16 and the
      artifact-qualified no-mirror INT8 backend; only pool plans, storage views,
      and registered kernels may differ.
- [x] Keep the old chunked backing path only as an explicit fallback until the
      new pool substrate passes both gfx11 gates; track its removal in
      `REFACTOR.md`.

Exit: all requests compatible with a resolved backend draw from one fungible
pool set, a request may use arbitrary free pages without same-chunk constraints,
and BF16-to-INT8 replacement does not fork continuous batching.
`GlobalKVPoolSet` owns one stable arbitrary-page table per plane;
`DenseKVResidentRunnerAdapter` refuses runners that do not consume a
backend-produced `KVBatchView`. C2-6 now wires the gfx1100 Qwen3.6 GGUF BF16
package through a model-specific `GlobalDeviceKVPool` bridge: one load-time
arena, stable per-layer/per-plane device pointer tables, generation-tagged
`KVStorageView`, and arbitrary free-page leases. The legacy chunk path remains
a separately tracked numerical/peer-package fallback until the remaining
long-context and second-gfx11 gates pass.

### C2-4 — integrated radix cache and eviction

- [x] Make the radix index reference generation-checked backend snapshot handles,
      initially dense immutable pages.
- [x] Include the complete backend/artifact fingerprint in every key and treat
      incompatible format/calibration snapshots as misses, never casts.
- [x] Retain completed immutable pages as evictable cache ownership independent
      of source-request lifetime.
- [x] Add LRU/TTL/quota eviction, COW partial tails, stale-generation rejection,
      and cache-first pressure handling.
- [x] Keep the default unchanged and gate promotion on active/completed p256+s1
      plus agentic 2K/8K correctness/economics on both gfx11 targets; the host
      rows are covered now and the hardware/default gate remains in C2-6.
- [x] Keep sampled reuse fail-closed until exact state/KV gates explicitly
      register an eligibility policy.

Exit: shared prefixes save real device pages and never pin capacity without
quota/eviction. `BackendRadixCache` transfers each unique cached page from its
source lease to one cache ledger owner, reference-counts overlapping snapshots,
and transfers or frees that ownership on eviction. Dense admission advances
only the generic prefill cursor after a compatible complete-page hit. No prefix
default or hardware performance claim changes in C2-4.

### C2-5 — token-budget scheduling and c1-c32

- [x] Plan multiple compatible prefill chunks and all due decode groups per
      fairness round.
- [x] Register per-backend physical width/context/workspace capabilities,
      group only identical execution compatibility keys, and emit honest
      fallback labels.
- [x] Prove every logical width 1..32 through bucket boundaries, mixed lengths,
      sparse retirement, cancellation, and refill.
- [x] Add TTFT/ITL-derived token budgets as the Generation-2 policy; keep generic
      `protect_decode`/`protect_ttft`/`fair` only as tracked Generation-1
      compatibility choices until C2-6 promotes measured backend defaults.
- [x] Instrument host planner/output overhead and cover c1/c8/c32 planning; the
      model/hardware production profile remains part of C2-6 default promotion.

Exit: no admission/response/memory cliff at c2/c4/c8/c9/c16/c17/c32 and c1
retains its direct route. A token-budget round rotates prefill by stable slot,
bounds prompt tokens, and then advances every due decode row exactly once.
`ExecutionCompatibilityKey` includes backend, layout, kernel bundle, work class,
context bucket, workspace, and physical widths. Lowering reports
`registered_masked_or_exact`, `registered_dense_compaction`, or
`serial_c1_fallback`; it never labels fallback work as a native batch.
Generation-2 dense adapters explicitly qualify multi-prefill and same-round
prefill/decode transitions. Legacy runners without those capabilities execute
one prefill transition per maintenance barrier and defer decode, preventing a
new round planner from silently violating an old model-state lifecycle.

### C2-6 — graphs, long context, and production load

- [x] Prove graph replay over changing page IDs, prefix eviction, and slot reuse.
- [x] Qualify 4K/16K/32K and model-supported long-context mixed membership under
      real resource accounting.
- [x] Run fixed, ragged, burst, Poisson, overload/recovery, disconnect, and
      sustained c1-c32 soaks.
- [ ] Compare matched same-model/quant/hardware serving against prior hipEngine,
      llama.cpp where applicable, vLLM, and SGLang; qualify unsupported backends
      honestly.
- [x] Promote defaults only after correctness, SLO, memory, and throughput gates.

Exit: one production configuration handles offered load above 32 with bounded
queueing and smooth resident c1-c32 operation.

The W7900 load/default scope is now **closed**; matched external serving and
second-device qualification remain unavailable. The generation-checked
graph/page/slot gates, resource-accounted long/mixed contexts, canonical load,
and c1-c32 planner/conservation suites pass for exact-file Qwen3.6-35B-A3B
`UD-Q4_K_M` BF16 KV:

- one batch-shaped target scratch owns execution workspaces while lightweight
  views preserve slot-local recurrent/KV/cursor state;
- the live allocator is `GlobalKVPoolSet` through the model-specific
  `GlobalDeviceKVPool`, with stable per-layer/per-plane pointer tables,
  `global-arbitrary-pages:g1`, arbitrary free-page leases, and zero final
  active/refcounted/pinned ownership;
- the Q8_1 direct-top1 shortcut is c1-only; registered gfx1100 shared-slot
  physical widths are `(1, 2, 4, 8)`, and wider logical batches decompose into
  exact registered buckets plus an honest c1 edge;
- owner state is scattered whenever another physical group reuses the packed
  workspace.

Same-loaded-server p128/d8 is exact for c1/c2/c4/c8/c17/c32; c17 live refill
admits before the first completion. The current matched c8 packet measures
**44.031 tok/s** native physical-c8 versus **27.634 tok/s** exact serial
(**+59.27%**), while the clean canonical production run passes tuning and all
nine load modes. gfx1151 and matched vLLM/SGLang/llama.cpp serving remain
unavailable on this host.
Evidence: accepted canonical packet
[`2026-08-18-concurrency2-c2-6-w7900-canonical-production-accepted.json`](../benchmarks/results/2026-08-18-concurrency2-c2-6-w7900-canonical-production-accepted.json),
physical c4/c8 promotion
[`2026-08-17-concurrency2-c2-8-w7900-shared-slot-c4-c8-promotion.json`](../benchmarks/results/2026-08-17-concurrency2-c2-8-w7900-shared-slot-c4-c8-promotion.json),
and promoted global/native packet
[`2026-08-16-concurrency2-c2-6-w7900-global-native-accepted.json`](../benchmarks/results/2026-08-16-concurrency2-c2-6-w7900-global-native-accepted.json).

The subsequent actual long-context campaign passes exact c2 1K/4K/16K/32K/64K,
exact mixed 1K/4K/32K, and changed-page graph replay. Under a 134-page global
generation, an exact 32K survivor coexists with an isolated retryable 4K
rejection; regrow changes its table from pages `0..128` to `5..133`, records
**4 captures / 100 replays / 4 invalidations**, bounds the 1K blocker's max ITL
at **0.803 s**, and drains all 134 pages with zero refs/pins. Static production
c1/c8 also pass exactness, SLO, route, memory, and ownership gates.

The clean canonical packet is **accepted** at `ff440cd01`: all six tuning
candidates pass and token-budget/256 is selected; static c1/c8, ragged, fixed,
Poisson, cancellation/disconnect, bounded 40-request overload, idle recovery,
and a 60-second 120-request soak all pass. The packet records **210/210
correctness-accounted rows**, fixed **12/12 at 53.196 SLO-goodput tok/s**,
overload **16 completed / 24 bounded `engine_busy` rejections**, and soak
**120/120 at 43.652**. It drains 271 admissions/reclaims, all 24 pages are free,
refs/pins are zero, and tracked-memory delta is zero. The prior fault was a
replayable decode graph surviving a prefill overwrite of its shared private
state; prefill now invalidates binding graphs and flushes synchronized state
before reuse. gfx1151 and matched external comparisons remain unavailable.
Evidence:
[`2026-08-18-concurrency2-c2-6-w7900-canonical-production-accepted.json`](../benchmarks/results/2026-08-18-concurrency2-c2-6-w7900-canonical-production-accepted.json)
and superseded blocker
[`2026-08-17-concurrency2-c2-6-w7900-long-load-blocked.json`](../benchmarks/results/2026-08-17-concurrency2-c2-6-w7900-long-load-blocked.json).

### C2-7 — FastDMS topology and codec composition

- [x] Complete the metadata/checkpoint gate from `KVCACHE.md`.
- [x] Add global compact pool-set/extent accounting and scheduler-owned atomic
      admission through the existing backend protocol.
- [ ] Port streaming no-shadow prefill pack and compact decode in BF16 first.
- [x] Qualify c1, then c2/c4/c8/c16/c32 through the same engine service.
- [x] Add DMS pressure, fragmentation, cancellation, reclaim, and soak gates.
- [x] Replace the BF16 payload with at least one qualified compressed codec by
      changing topology/codec composition, pool plan, and kernel bundle only;
      rerun the same scheduler/lifecycle suite.
- [x] Keep prefix lookup off for DMS until snapshot/overlay semantics pass.

Exit: DMS provides allocator-visible capacity and attention-work savings without
forking the concurrency architecture, and changing its payload format does not
fork it either.

Host/backend status: the torch-free implementation now includes strict
checkpoint metadata, atomic per-layer/head extents, variable `KVLiveSpans`,
streaming no-shadow BF16 pack, transactional append/eviction, grouped-GQA CPU
attention, common token-budget c1-c32 lifecycle, pressure/fragmentation/drain,
and a fixture-qualified INT8 composition over the same topology. Prefix reuse
is hard-off. The exact Qwen3.6 artifact correctly fails the metadata gate
because it has no retrofit. The BF16 HIP kernel port landed at unit level on
gfx1100 (C2-7 U1-U4 worklog entries of 2026-08-18: extract-decision,
streaming pack, append/decode, and compact decode attention in the
`dms_compact` family, bit-exact data movement plus the KL/top-1 gated
attention, each with its registered cpu_reference strict fallback). The
kernels are registered but not defaulted or wired into the production DMS
path, where the host parent remains the production path; the device
payloads are available behind the opt-in `device_payloads` flag on
`DMSCompactBackend` (U6, 2026-08-18 worklog entry: bit-exact
pack/append/attention parity vs the host parent, fail-closed overflow,
no host payload shadow, determinism). The family's rocprof
kernel-identity rows are blocked on this W7900 host by a deterministic
`rocprofv3 --kernel-trace` dispatch hang (recorded in the U1 entry,
reproduced 2026-08-19, not faked); real checkpoint quality, device savings,
and hardware soak remain blocked by the missing trained
`dms_metadata.json`, so the BF16 kernel-port checkbox and product exit
remain open. Evidence:
[`2026-08-17-concurrency2-c2-7-dms-host-blocked.json`](../benchmarks/results/2026-08-17-concurrency2-c2-7-dms-host-blocked.json).

### C2-8 — optional hot/cold tiering

- [x] Add backend-declared offload/restore maintenance work and host/NVMe
      resource pools without blocking due decode work.
- [x] Treat KVTC-style storage as a cold codec that restores to the resolved hot
      backend; do not present it as an attention dtype.
- [x] Key cold objects by complete hot-backend/artifact identity and validate
      deterministic restore, cancellation, eviction, quotas, and final drain.
- [x] Measure transfer/decompression workspace and TTFT against prefix
      recomputation before promotion.

Exit: tiering adds a backend capability and maintenance work class, not another
scheduler or request lifecycle.

Implemented host status: `TieredKVCacheBackend` delegates all execution views to
the hot backend and adds typed offload/restore/evict work, separate host/NVMe
cache and transfer-workspace ledger pools, complete fingerprinted cold keys,
KVTC-style checksummed compression, tenant quotas, pin-aware LRU, atomic hot/cold
ownership transfer, restore rollback, and deterministic drain. A synthetic 1
MiB host packet measures median restore **1.203 ms** versus **8.967 ms**
recompute proxy, but its repeated payload is intentionally non-representative;
no model TTFT or default claim follows. Evidence:
[`2026-08-17-concurrency2-c2-8-tier-host-accepted.json`](../benchmarks/results/2026-08-17-concurrency2-c2-8-tier-host-accepted.json).

## Acceptance gates

### Functional and lifecycle

- Every child output matches its independent c1 oracle for deterministic gates.
- Fast children resolve and free backend resources before slow siblings.
- New work fills a reclaimed slot before unrelated long requests finish.
- Blocking and SSE produce equivalent IDs, finish reasons, accounting, timeout,
  cancellation, and final ownership.
- Sparse retirement, compaction, COW, prefix eviction, and slot reuse preserve
  survivor hidden/state/KV hashes.
- Queue, pool, model state, graph refs, collectors, and completion records drain
  to their documented idle baselines.

### Width matrix

At minimum test logical widths:

```text
1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 24, 32
```

For every width report physical decomposition, native group count, fallbacks,
prefill/decode work, aggregate and per-request throughput, TTFT/ITL p50/p95,
peak memory, pages/live cells, and exact output counts. C>8 is not native c>8
unless one physical group actually has that width.

### Load shapes

- simultaneous fixed prompt/decode lengths;
- ragged prompt and completion lengths;
- delayed and Poisson arrivals during active decode;
- one long prompt among short prompts;
- one slow SSE consumer among normal blocking/SSE clients;
- cancellation before admission, during prefill, and during decode;
- queue overload, no-fit large request, recovery, and aged-request fairness;
- shared system-prefix hit/miss/COW/eviction pressure;
- 60-second smoke and longer production soak with offered concurrency above 32.

### Performance

- Compare against the exact same model, weight quant, KV backend composition,
  context, output shape, hardware, and command.
- Keep c1 transition/complete-request performance within the retained regression
  budget from `BENCHMARK.md`; occupancy one must select physical c1.
- c=N promotion must beat the old server path and honest serial composition on
  aggregate goodput without violating the declared TTFT/ITL/memory SLO.
- Boundary widths 9 and 17 must not show unexplained collapse from route caps,
  reallocation, response barriers, or graph rebuilds.
- External-engine comparisons are secondary to same-engine old/new proof and
  must disclose feature/backend mismatches.

### KV and pressure

- Pool accounting equals request refs + cache/session ownership + free/in-flight
  states at every commit barrier.
- Resource estimates and measured high-water are reported together; unexplained
  memory remains visible.
- Admission failure is atomic and identifies the rejecting resource.
- Cache eviction never changes active-request outputs.
- DMS reports logical tokens, physical live cells, allocator bytes,
  fragmentation, and actual compression; masked dense storage is not a compact
  claim.
- Speculative reject/partial/full transactions leave canonical KV exact.

### KV-cache backend conformance

Every promoted backend or topology+codec composition runs one common suite:

- load-time pool planning exactly accounts all persistent planes and artifacts;
- estimate/reserve is atomic under injected failure at every claim boundary;
- prefill, decode, growth, terminal, cancellation, rollback, and reclaim conserve
  every pool independently;
- c1-c32 stagger/refill behavior and outputs match the backend's independent c1
  oracle without engine or scheduler special cases;
- prefix attach/COW/eviction either passes under the declared mode or fails
  closed as unsupported;
- graph replay sees only current storage-view generations after page reuse,
  demotion, compaction, or restore;
- mixed-format backends additionally pass concurrent demotion, abort, retract,
  slot reuse, pressure, and long-context soak guards;
- telemetry identifies topology, codec, tier, artifact, kernel bundle, every
  persistent plane, transient workspace, and any shadow/mirror bytes.

The minimum architecture matrix is dense BF16, dense no-mirror INT8, a synthetic
mixed-tier backend, and a synthetic per-head-variable backend at C2-0/C2-3;
FastDMS and real mixed/tiered codecs replace the synthetic rows as they qualify.

## Migration from `CONCURRENCY.md`

The old document remains useful evidence, but its active queue is no longer the
architecture order. Unresolved work is mapped as follows:

| Old work | Generation-2 disposition |
| --- | --- |
| A4 late physical-width exactness and frozen rerun | Preserve as a required gfx1100 correctness gate before promoting that affected route through C2-5; it does not block C2-0/C2-4 infrastructure. |
| IKV-C1 through IKV-C7 | Continue under `QWEN38-INT8-KV-CONTINUOUS.md`; implement the dense-INT8 `KVCacheBackend` pool plan, storage view, kernels, claims, and conformance tests rather than another scheduler. |
| Sampled/general historical prefix reuse | C2-4 after deterministic dense arena/cache ownership is proven. |
| Long-context pressure and graph invalidation | C2-3 and C2-6. |
| Route-cap follow-up / KV-budget admission | Replaced by C2-2 and C2-3; physical route widths cease to be primary admission caps. |
| Prefill co-admission | C2-5 token-budget rounds. |
| DMS/KV tier movement | C2-7/C2-8 plus `KVCACHE.md`; DMS is a topology/backend composition, not a new loop. |
| MTP/DFlash verify/commit/scatter | Keep as peer work classes in the C2 scheduler under `NATIVE_SPEC_CYCLE.md`. |
| gfx1100 PARO c4/c8 owner and broader GGUF/PARO/quant coverage | Migrate each runner after C2-1/C2-3, preserving its existing direct correctness/profiler gate. |
| Tensor parallel | Later replica/shard integration under `TENSOR_PARALLEL.md`; each replica/shard group still exposes one engine/KV owner. |

Completed C4-F5/GGUF/PARO evidence is not recopied here. It remains the migration
oracle in `CONCURRENCY.md` and retained benchmark artifacts.

## Guardrails

- No `if backend == ...`, `if quant == ...`, `if kv_dtype == ...`, or codec-key
  switch in engine, scheduler, or model dispatch. Register backend factories,
  capabilities, storage views, and kernel bundles.
- `KVLiveSpans` remains the mandatory attention/KV-write liveness ABI; new
  formats add registered `KVStorageView` planes, not scalar shortcuts or
  quantizer-specific scheduler fields.
- The resource ledger never computes codec bytes from dtype/token count; the
  resolved backend submits named claims and deltas for all planes.
- A KV backend may not own a request queue, output collector, scheduling loop,
  or hidden allocation path.
- One active backend per loaded replica is the initial production rule. Backend
  replacement drains/transcodes leases and invalidates incompatible snapshots;
  it never reinterprets live bytes.
- Fused kernels keep numerically equivalent unfused fallbacks.
- Canonical KV mutates only through scheduler-owned commit/rollback points.
- Never call a per-row complete model/session loop and label it native c=N.
- Never hide physical group decomposition, serial fallback, cache hit, or graph
  rebuild from telemetry.
- Do not tune admission, priority, prefix cache, or physical grouping to fixed
  benchmark prompts/token IDs.
- Do not retain dense shadow KV for a claimed DMS capacity path.
- Do not rely on HIP OOM as admission control.
- Do not automatically preempt live requests while evictable cache capacity
  remains.
- Keep unrelated old paths until their replacement passes the same backend,
  model, correctness, lifecycle, and performance gates; then record cleanup in
  `REFACTOR.md`.

## Failure handling

| Failure | Required action |
| --- | --- |
| Short response waits for long sibling | Reject C2-1; inspect parent/child collector and terminal publication ownership. |
| Freed resident slot is not refillable | Inspect delayed KV/model refs and admission credits; frontend response formatting cannot own backend resources. |
| Width 9/17 collapses | Inspect route-cap leakage, group planning, repeated prefill, graph rebuild, and workspace serialization. |
| Pool has free bytes but request cannot fit | Report fragmentation/largest extent; compact safely or use page indirection, never hide the failure. |
| Estimated fit reaches HIP OOM | Backend claim/pool plan or whole-device reserve bug; add the missing plane/workspace and fail admission earlier. |
| Prefix hit changes output/state | Invalidate cache entry, capture the first divergent page/layer, and add a COW/key-generation fixture. |
| DMS shared prefix diverges | Disable DMS prefix reuse; private ranges are canonical until overlay/snapshot semantics are proven. |
| New KV codec requires scheduler/HTTP branching | Reject the integration; extend the generic backend/claim/storage-view contract or declare the composition unsupported. |
| One codec plane leaks or aliases after cancellation/demotion | Reject backend conformance; quarantine that layout and add operation-ID conservation plus concurrent slot-reuse fixtures. |
| Backend replacement reuses incompatible prefix/graph state | Treat as a fingerprint/generation bug; invalidate and drain rather than transcode implicitly. |
| Slow client stalls GPU | Cancel only its collector/request; engine output routing must be non-blocking. |
| Cancellation harms neighbor | Reject lifecycle gate and locate mutation outside request-owned commit. |
| Graph sees stale page/range | Disable that bucket, increment arena generation, and require stable metadata indirection before replay. |
| c1 regresses | Keep old c1 default and profile transition/collector/planner overhead before further c>N tuning. |

## Definition of done

Generation 2 is complete for a model/hardware/KV-cache-backend combination only
when:

- [x] one engine service owns all blocking, SSE, and library child requests;
- [x] independent terminal publication/reclaim removes the head-of-line barrier;
- [x] one format-neutral resource ledger governs atomic admission and pressure;
- [x] all compatible requests share one backend-declared global pool set;
- [x] immutable complete prefixes are refcounted, COW-safe, quota-bounded, and
      evictable;
- [x] logical resident concurrency is independent of physical kernel width;
- [x] the width/load/overload/lifecycle matrices pass through c32;
- [x] c1 retains its direct route and c=N beats honest old/serial baselines under
      declared SLOs;
- [x] graph, pool, state, collectors, and completion ownership drain cleanly;
- [ ] the common conformance suite passes for dense BF16 and a format-distinct
      compact backend without scheduler/frontend forks;
- [x] replacing topology/codec/tier requires only a registered backend,
      artifacts, pool/storage plans, and kernels—not a new concurrency path;
- [x] documentation, artifacts, and telemetry disclose exact routes and memory.

DMS is complete only after the same list passes with compact allocator-visible
storage, DMS checkpoint quality gates, per-head live-span accounting, and no
dense shadow. Its first BF16 payload is a correctness rung; compressed DMS must
reuse the same engine and backend contract. Until then, dense global paging is
the canonical Generation-2 KV path and DMS prefix sharing remains off.

The executable completion audit now records **31 passed, 3 blocked, and 1
unavailable** rows with no missing evidence. The W7900 canonical C2-6
load/default scope is closed; the product goal remains open for unavailable
matched external serving plus DMS HIP/checkpoint conformance. See
[`2026-08-18-concurrency2-completion-audit.json`](../benchmarks/results/2026-08-18-concurrency2-completion-audit.json).
