# Concurrency and Continuous Batching

Last updated: see git log.

This document is the working guide for turning the current single-request
resident runtime into a vLLM-style continuous-batching serving path on a
single GPU. It covers what serving looks like when this work is done, what is
implemented today, the contracts the implementation must satisfy (engine loop,
elastic KV pool, prefix sharing, per-row sampler, streaming, observability),
and the benchmark gates a retained c>N row must pass.

Tensor parallelism (TP), expert parallelism (EP), compact DMS, and speculative
decoding (MTP / DFlash / EAGLE3) are **out of scope here** and live in their
own feature branches and docs. Concurrency-side decisions that must not paint
those follow-ons into a corner are called out in
§[Forward-compatibility guardrails](#forward-compatibility-guardrails).

Related source-of-truth docs:

- [`PLAN.md`](PLAN.md) — architecture invariants, long-form concurrent decode
  design, and §Multi-GPU Strategy for the TP/EP follow-on.
- [`BENCHMARK.md`](BENCHMARK.md) — evidence policy and c=N benchmark gates.
- [`KVCACHE.md`](KVCACHE.md) — dense INT8 KV capacity path and the compact DMS
  roadmap (the next-feature DMS plan).
- [`PREFILL.md`](PREFILL.md) — native/compact prefill details.
- [`ENVS.md`](ENVS.md) — knobs introduced by this doc.

## Definitions

| Term | Meaning |
| --- | --- |
| HTTP concurrency | Multiple client requests are in flight at the server at once. This can still be serialized internally. |
| Prompt-list batching | One API call carries multiple prompts, e.g. OpenAI completions `prompt=[...]`. Counts as true c>N only if the generator advances those prompts together. |
| c>N decode | `N` independent live requests each advance one target token in the same model step. |
| Continuous batching | The scheduler can admit, prefill, decode, finish, compact, and reclaim requests while other requests keep running, under a single long-lived engine loop. |
| Engine loop | One long-lived scheduler tick driving admission, prefill, decode, verify, reclaim, and pool resize across all active requests. |
| Elastic KV pool | Dense paged KV backed by an allocator that can grow and shrink between admission cycles up to a high-water cap. |
| Append-only block id | Allocator contract: a block id, once issued, keeps a fixed device pointer until freed; growth issues fresh ids past the current high water. |
| Prefix sharing | Multiple requests share refcounted KV pages for a common token prefix via a radix-tree index; the first divergent token forces a copy-on-write fork. |
| KVTC (KV tiered cache) | Future-direction multi-tier KV storage: hot HBM pages → pinned host RAM → optional NVMe spill, behind the same `KVLiveSpans` and block-id contracts so block ids stay stable across tier moves. KVTC is a follow-on feature branch, not in CONCURRENCY scope. |
| Per-row sampler | Sampling parameters (temperature, top-k, top-p, repetition penalty, seed, stop tokens) are independent per active row. |
| Packed/native prefill | Multiple prompt rows packed into one prefill slab and launched through row-shaped kernels. |
| Serial bridge | A correctness-first path with batch-shaped slots/KV metadata but active rows execute through the c=1 layer path. Diagnostics only; not a throughput claim. |

## Destination state

When this work is done, hipEngine on a single W7900 (or any single supported
GPU) runs as:

- One long-lived **engine loop** (one background driver thread under `hipengine
  serve` and `LLM.generate()`) admits new HTTP requests mid-stream up to
  *current* pool capacity, grows the **elastic KV pool** toward a high-water
  cap when load demands, and shrinks back toward a low-water floor when idle.
- The loop interleaves **chunked prefill** with **decode** under an explicit
  prefill-vs-decode policy; finished requests are reclaimed at the next commit
  point without waiting for the longest active request.
- Common token prefixes are shared via **refcounted KV pages**; the first
  divergent token forks a request onto fresh pages. `n>1` lowers to N
  scheduler requests with a shared prefix.
- The **per-row sampler** lets requests with different temperature, top-k,
  top-p, repetition penalty, and stop tokens decode together in one step.
- **Streaming** and non-streaming traffic share the same loop and the same
  reclaim path; cancellation, client disconnect, EOS, and max-tokens are one
  unified path.
- **Per-request and per-pool observability** is exported on `/metrics`
  (Prometheus) and recorded in retained benchmark artifacts.

Primary target workloads are agentic loops and long-context multi-turn
chat; both depend on heavy prefix sharing across requests and across turns.
`n>1` lowering is the third major prefix-sharing consumer.

Single-GPU, single-process, single-rank. Multi-GPU TP, DMS compact KV,
RadixCache eviction policies under variable-span KV, multi-tier KV storage
(KVTC), and speculative decoding are explicit follow-on feature branches.

## Current answer

**hipEngine does not yet support true vLLM-style c>N serving.**

The public Qwen/PARO generator has a first prompt-list c>N path: it admits all
prompt rows into `ResidentBatchScheduler`, uses native compact packed prefill
for BF16 KV prompt lists, and routes output by request id. Production decode
after the seed token still uses `step_batch_serial` — batch-shaped slots/KV
metadata but rows execute serially through the c=1 layer path.

The OpenAI server coalesces compatible non-streaming HTTP generations over a
short configurable batch window (`--generation-batch-window-ms`, default 5 ms)
before one prompt-list `LLM.generate()` call. Streaming remains
one-request-at-a-time and `n>1` is still rejected. The coalescer is a
submission-time join, not a continuous-batching admission.

An experimental native decode path (`step_batch_native`) is gated behind
`HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1`. With the default
`HIPENGINE_QWEN35_BATCH_SAMPLE_MODE=serial_lm_head` workaround, the earlier
batched-LM-head drift is fixed for reduced 512/32 diagnostics: L1/L3/L4 and
L40 c=2 512/32 pass generated-token equality vs independent c=1. The full
40-layer c=2 512/128 gate still fails on current tip
(`/tmp/hipengine-retained/guarded-L40-c2-512-128-current.json`, row 0 idx 87,
`batch=271` vs `c1=1165`) and `throughput_claim_eligible=false`. A separate
`eq-L8-selectedmoe.json` failure points at selected-MoE/native-row mapping.
The path is not a throughput claim.

The elastic KV pool, prefix sharing, per-row sampler, streaming routing,
cancellation reclaim path, and `/metrics` observability are **not** in code
yet. The phase ladder below sequences them.

## Readiness matrix

| Layer | Current status | Evidence / code | Blocks true c>N |
| --- | --- | --- | --- |
| OpenAI server | Compatible non-streaming HTTP generations are coalesced into one prompt-list `LLM.generate()` call behind a grouped safety lock; `n>1` rejected; streaming is one request at a time. | `hipengine/server/api.py:_GenerationBatcher`, `create_app`, `_validate_generation_request`. | Replace coalescer with engine-loop `submit/poll/cancel`; route streaming through the loop; remove the coarse lock. |
| Public `LLM.generate()` | Prompt lists with `len(prompts)>1` use `ResidentBatchScheduler`, BF16 packed native prefill, request-id output routing, and the serial slot bridge for decode. Streaming is one prompt only. | `hipengine/generation/qwen35_paro.py:Qwen35ParoOneTokenGenerator._generate_batch`. | Lower `LLM.generate()` to `submit+poll` over the engine loop; native c-aware decode. |
| Engine loop / scheduler | `ResidentBatchScheduler` owns pending/admitted queues, slots, active masks, compact prefill slabs, decode work, graph bucket keys, and completion routing within a single `_generate_batch` call. The loop does not persist beyond one call. | `hipengine/generation/batch_scheduler.py`; `_generate_batch`. | Promote to a long-lived background driver with submit/poll/cancel, work-class ticks, and commit-point semantics. |
| Prefill | Single-request native prefill and prompt-list BF16 packed native prefill are live. INT8 packed prefill is not wired. Chunked prefill is not interleaved with decode. | `Qwen35ParoResidentSession.prefill_native`, `prefill_native_packed`, `ResidentBatchScheduler.next_compact_prefill_slabs`. | INT8 packed prefill; chunked prefill interleaved with decode under a policy. |
| Decode runtime | Production c>N prompt-list decode uses `step_batch_serial`. Experimental `step_batch_native` is gated by `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1`; current default `serial_lm_head` passes L40 c=2 512/32 as a reduced diagnostic but fails full c=2 512/128 generated-token equality. | `step_batch_serial`, `step_batch_native`, `_sample_batch_from_hidden`, `batch_execution_metadata`; `/tmp/hipengine-retained/guarded-L40-{512-32,c2-512-128}-current.json`. | Layer-level hidden-state bisection; selected-MoE/native-row fix; c=2/4/8 512/128 equality; row-aware split-K full-attention; native sampler. |
| Sampler | Greedy `argmax_f32` per row. Sampling parameters apply globally to the call, not per row. The coalescer requires identical sampling keys per batch. Experimental native decode currently defaults to `serial_lm_head`; `batched_lm_head` is diagnostic only. | `_sample_from_hidden`, `_sample_batch_from_hidden`. | Per-row temperature/top-k/top-p/rep-penalty/seed/stop tokens; per-row EOS handling; replace the per-row LM-head loop after equality is proven. |
| Attention / KV primitives | BF16 batched paged KV append and batched full-attention context decode pass c=1/2/4/8 primitive correctness. Split-K reducer is c1-only. INT8 batched paths exist as wrappers but lack the end-to-end gate. | `scripts/qwen35_batch_correctness.py`; `hipengine/kernels/hip_gfx1100/attention/`. | Row-aware split-K reducer; INT8 batched end-to-end gate. |
| MoE / quant kernels | Many wrappers accept `rows` or routed-lane counts; end-to-end selected-MoE decode still follows c1 assumptions. `eq-L8-selectedmoe.json` diverges from c=1 reference at idx 13. | `hipengine/kernels/hip_gfx1100/quant/*`, `hipengine/runtime/qwen35_paro.py`. | Token-row to routed-lane mapping; grouped-by-expert execution for c=4/8; c-aware dispatch thresholds. |
| KV pool | Fixed-size pool sized at startup from `hipMemGetInfo()` after weights resident (v0.2.2). Pool does not grow or shrink during a session. Block ids reuse pointers across reallocation. | `hipengine/runtime/qwen35_paro_runner.py` startup path. | Append-only block id contract; chunked grow up to high water; idle shrink to low water; admission against current capacity. |
| Prefix / radix cache | Not implemented in code today. `grep RadixCache hipengine` returns nothing. | — | Refcounted pages; RadixCache trie; copy-on-write fork; `n>1` lowering. Flat prefix-LRU is intentionally not a peer implementation. |
| Observability | Server emits standard FastAPI logs. No request-level timings, no pool counters, no `/metrics`. | — | Per-request timings; per-pool counters; per-bucket histograms; `/metrics` endpoint. |

DMS / compact KV serving status lives in [`KVCACHE.md`](KVCACHE.md) and is not
mirrored in this matrix.

## Engine-loop contract

The engine loop is the single owner of admission, work scheduling, KV
allocation, sampling, completion, reclaim, and pool resize. The
`generation_lock` in `hipengine/server/api.py` exists today as a guard against
non-reentrant session mutation; by the end of C4 the lock should protect only
brief mutation regions (not whole generations) or be removed entirely.

### C1 lock-scope audit

The current lock is acceptable for C1 because C1 is only submission-time HTTP
coalescing plus static prompt-list batching, not continuous batching:

- Startup eager-load/warmup holds `generation_lock` around resident-session
  preparation and the one warmup `engine.generate(...)` call.
- Non-streaming requests call `generate(...)`, which holds the lock only for
  resident-context preparation, sampling construction, and context-budget
  validation, then enqueue into `_GenerationBatcher`.
- `_GenerationBatcher._run_group(...)` holds the lock around one grouped
  `engine.generate(tuple(prompts), sampling)` call. This is intentionally a
  whole-generation lock in C1 because the resident Qwen/PARO session mutates
  shared KV, linear-attention recurrent state, hidden buffers, scratch, and
  sampler state during `LLM.generate()`.
- Streaming chat still holds the lock while it drives `engine.stream(...)` or
  fallback `engine.generate(...)`. This is not C4-ready, but it matches the C1
  contract that streaming is one request at a time.

The exact C4 blocker is ownership: the resident session is not reentrant and
there is no long-lived engine loop that owns request admission, slot mapping,
KV mutation, token queues, cancellation, and reclaim at commit points. Once C4
adds that single-owner loop, server endpoints should call `submit/poll/cancel`
instead of holding `generation_lock` across generation. Any remaining lock
should then protect only process-level model/session initialization.

### Public interface (target)

```python
request_id = engine.submit(
    prompt_tokens: Sequence[int],
    sampling: SamplingParams,
    max_new_tokens: int,
    stream: bool = False,
) -> int

events = engine.poll(timeout: float | None = None) -> list[Event]
# Event(request_id, kind: 'token' | 'finish' | 'error', payload)

ok = engine.cancel(request_id: int) -> bool
```

`LLM.generate()` and the OpenAI server become thin adapters over this surface.
Both streaming and non-streaming traffic call the same `submit/poll/cancel`.

### Work classes

Each engine tick picks **one** of the following work classes for the next
kernel-launch sequence:

| Class | Action | Commit at end |
| --- | --- | --- |
| `ADMIT` | Move pending requests into active slots up to current pool capacity. Try one pool grow per cycle if grow-on-admission is enabled. | New slot table |
| `PREFILL_CHUNK` | Run one chunked prefill step over one or more admitted requests. | Per-request prompt cursor; KV append |
| `DECODE_STEP` | One token of decode for every active request whose prefill is done. | Per-request token; KV append |
| `RECLAIM` | Free KV pages, refcounts, scratch from finished/cancelled requests. | Free list; pool shrink eligibility |
| `VERIFY_STEP` *(SpecDec, later)* | One target-verify pass over draft rows. | Accept-list; transactional KV commit |
| `PACK_STEP` *(DMS, later)* | One streaming-pack sweep over a finished prefill layer/chunk. | Compact KV append; dense scratch release |

Default per-tick policy: `RECLAIM` → `ADMIT` → choose between `PREFILL_CHUNK`
and `DECODE_STEP` under the **prefill-vs-decode policy** (see below). Verify
and pack classes are inserted by SpecDec / DMS feature branches without
changing the loop contract.

### Commit points

KV mutation, generated-token delivery, streaming event emission, and
cancellation are honored **only at commit points**. A commit point is the
boundary between two work-class steps. Mid-step mutations are scratch.

This is what protects KV writes from being torn by mid-step cancellations,
gives SpecDec a clean accept/rollback gate, and lets DMS pack between active
requests' decode steps without races.

### Prefill-vs-decode policy

| Policy | Behavior | Default |
| --- | --- | --- |
| `protect_decode` | Decode always wins when any active request can decode. Prefill chunks fill remaining cycles up to a token budget. | yes |
| `protect_ttft` | Prefill wins for any newly admitted request until its first decode token. | — |
| `fair` | Round-robin between prefill and decode. Token-equivalent budgets are a later latency/metrics refinement. | — |

Knob: `HIPENGINE_PREFILL_DECODE_POLICY` / `--prefill-decode-policy`. Default
`protect_decode` (vLLM-equivalent default; minimizes inter-token-latency
regressions for active requests).

## Dynamic KV pool

Continuous batching's admission policy is a function of current KV capacity.
A fixed startup-sized pool either wastes VRAM that could hold extra slots or
caps `C` for no reason. The pool must size against actual load.

### Allocator contract

- **Block id is permanent.** Once a block id `b` is allocated, its backing
  device pointer never changes. `KVLiveSpans` and captured `hipGraph` buckets
  that reference `b` stay valid until `b` is freed.
- **Growth is append-only.** New block ids come from chunks allocated past the
  current high-water mark. Existing live blocks are never relocated.
- **Shrink frees from the free list only.** A block is freeable iff its
  refcount is zero *and* no captured graph bucket has recorded a pointer for
  it. Shrink trims tail chunks; the high-water mark is monotonic during steady
  state.
- **Allocation granularity is a chunk.** `hipMalloc` happens in chunks of
  `kv_pool_chunk_pages` (default 128 pages or ≥ 64 MiB equivalent, whichever
  is larger), then sub-allocated into block ids. Avoids `hipMalloc` storms
  under bursty admission.
- **All allocation goes through the scheduler.** No path in dispatch / model /
  kernel code allocates KV pages directly. Admission is the only producer.

### Sizing policy

| Knob | Default | Notes |
| --- | --- | --- |
| `kv_pool_initial_bytes` | auto = v0.2.2 startup estimate | First chunk allocation. |
| `kv_pool_low_water_bytes` | `kv_pool_initial_bytes` | Pool never shrinks below this. |
| `kv_pool_high_water_bytes` | `min(free_after_weights * 0.9, kv_pool_initial_bytes * 4)` | Pool never grows above this. |
| `kv_pool_chunk_pages` | 128 (or ≥ 64 MiB equivalent) | Grow granularity. |
| `kv_pool_idle_grace_seconds` | 60 | Time below `low_water + chunk` before shrinking. |
| `kv_pool_grow_on_admission` | true | If false, admission rejects when the pool is full instead of trying to grow. |

CLI: `--kv-pool-{initial,low-water,high-water,chunk-pages,idle-grace}-*`.
Env: `HIPENGINE_KV_POOL_*`. Document in `docs/ENVS.md`.

### Admission rule (every cycle)

1. If the request fits in free pages → admit.
2. Else if `kv_pool_grow_on_admission` and `current_bytes + chunk_bytes ≤
   high_water_bytes` and `hipMemGetInfo()` permits → grow one chunk; admit.
3. Else queue with an explicit `admission_blocked_reason`
   (`kv_capacity_high_water_reached` / `device_oom` / etc.).

### Shrink rule (background, between scheduler ticks)

1. If `free_bytes > low_water_bytes + chunk_bytes` continuously for
   `idle_grace_seconds` → free one tail chunk.
2. Never free a chunk containing a non-zero-refcount block, regardless of idle
   time (protects refcounted prefix pages).

### Admission accounting

- `KVPolicy.admission_cap()` returns **current** free-page equivalents, not
  startup capacity. Dense fixed-page policy returns `free_pages`; DMS (later)
  returns compact-live-token capacity over current free pages.
- The pending queue carries a `kv_pages_needed_estimate` per request, computed
  from `prompt_tokens + max_new_tokens` at submit time and revised as actual
  decode positions advance.
- Admission decisions run after the current step's `RECLAIM` so that finishing
  requests free pages before the next admit attempt.

### Acceptance for a dynamic-pool-enabled c>N row

In addition to the existing benchmark gates:

- Pool grew and shrank on a designed burst+idle workload (artifact records
  ≥1 `grow_event` and ≥1 `shrink_event`), or the run fit in the initial chunk
  and the artifact says so explicitly.
- `kv_pool_grow_events ≤ ceil((peak_bytes - initial_bytes) / chunk_bytes)`
  (no `hipMalloc` storms).
- Debug check: no block-id pointer changed during the run.
- Memory audit: tracked allocator peak ≤
  `kv_pool_high_water_bytes + non_kv_baseline_bytes`.

## KV sharing: RadixCache (+ KVTC forward-compat)

Refcounted block ids unlock prefix sharing across requests. Prefix sharing is
the first non-trivial reduction of KV bytes per active request and a
prerequisite for cheap `n>1` lowering. The structure is RadixCache; flat
block-LRU is explicitly not implemented as a peer (rationale below).
Multi-tier KV storage (KVTC) is a follow-on feature branch; CONCURRENCY work
must honor the KVTC ABI guardrails so that branch lands cleanly later.

### Refcount semantics

- Every block id carries a refcount; default 1 when first written by a
  request.
- A second request that walks the same prompt prefix into an existing
  refcounted block increments the refcount and reuses the block id.
- A request finishing (`RECLAIM`) decrements refcounts on its block ids.
- A block is freeable when refcount reaches zero.
- A captured graph bucket holding a pointer for a block keeps the *chunk*
  alive against shrink (but not against free).

### Copy-on-write fork

- Two requests share a block until one of them writes a token that diverges
  from the other's path.
- At divergence, the diverging request gets a fresh block id (allocated under
  the same admission rule as any new write), copies the shared prefix's last
  partial block if needed, and continues independently.
- The original shared block stays refcounted on the non-diverging path.

### Why radix and not flat block-hash LRU

The primary target workloads — agentic loops, multi-turn chat, `n>1`
sampling — are all tree-structured: branches off a common root, where flat
block-hash LRU only catches one path at a time. RadixCache catches sharing
at every branch point, including partial-block edges. The ~200 LoC delta
over a flat structure (per [`PLAN.md`](PLAN.md)) is well-spent for this
workload mix; carrying two prefix schemes also doubles the surface area
where prefix sharing × dynamic pool × cancellation can interact badly, so
flat prefix-LRU is not implemented as a peer.

### Prefix index

Knob: `HIPENGINE_PREFIX_CACHE` / `--prefix-cache` in `{off, radix}`.
Default `off` until correctness gates pass; then `radix`. Pick `off` to
disable prefix reuse entirely.

### Tiered storage (KVTC, future feature branch)

KVTC (KV tiered cache) is the planned multi-tier storage layer that sits
under prefix sharing: hot pages stay in HBM, cold but refcounted prefix
pages spill to pinned host RAM, and very cold session state spills to
NVMe / disk. KVTC is **not** in CONCURRENCY scope; it is called out here
so that the contracts in C2 / C4 / C5 do not preclude it. The reference
designs are vLLM v0.6+ CPU offload and SGLang hierarchical cache.

Rough tier roadmap (sketch, not committed in this doc):

| Tier | Storage | Latency | Use |
| --- | --- | --- | --- |
| T0 | Device HBM | ns | Active live KV; hot prefix nodes. |
| T1 | Pinned host RAM | µs (PCIe DMA) | Refcounted but cold prefix pages; admission-eligible without recompute. |
| T2 | NVMe / disk | ms | Session save/restore; very cold long prefixes. |

ABI requirements that CONCURRENCY work must already honor for KVTC to
land cleanly later are listed in
§[Forward-compatibility guardrails](#forward-compatibility-guardrails)
under "Don't break KVTC."

### `n>1` lowering

- The API layer accepts `n > 1` by submitting N scheduler requests with the
  same prompt tokens and a per-call seed offset.
- The prefix cache shares prompt KV across the N requests until the first
  divergent sampled token (immediate, for distinct seeds).
- Output is collected via N `request_id`s and returned to the client under the
  OpenAI `n` schema.
- This is the first user of prefix sharing in production and the natural
  staging ground for the contract.

### What's deliberately deferred

- **Eviction under variable-span KV (DMS).** Per-sequence eviction overlays
  for shared prefix blocks are an open design point; until then DMS disables
  prefix sharing (see [`KVCACHE.md`](KVCACHE.md) Phase K2).
- **Disk session save/restore.** Possible follow-on; ABI-compatible with the
  block-id-stable contract.

## Streaming, cancellation, and reclaim

### Per-request output queue

- Each active request owns a bounded token queue (default 64 tokens).
- The streaming adapter (SSE for `/v1/chat/completions` and
  `/v1/completions`) pulls from the queue and emits OpenAI-format events.
- When the queue is full (slow client), the request's slot is paused at the
  next commit point. It does not block other requests' decode steps.
- Knob: `HIPENGINE_STREAM_QUEUE_DEPTH` / `--stream-queue-depth`.

### Cancellation paths

| Trigger | Effect |
| --- | --- |
| `engine.cancel(request_id)` | Marked at next commit; slot is reclaimed. |
| Client disconnect (SSE) | Same as `cancel`. |
| EOS token sampled | Same as `cancel` with `finish_reason="stop"`. |
| `max_new_tokens` reached | Same as `cancel` with `finish_reason="length"`. |
| Per-request timeout (optional) | Same as `cancel` with `finish_reason="timeout"`. |

All five funnel through the same `RECLAIM` work class. There is one reclaim
implementation, not five.

### In-flight semantics

- Cancel during prefill: drop at the next chunk boundary.
- Cancel during decode: drop at the next step boundary.
- Cancel during verify (SpecDec, later): discard the verify journal; no
  canonical KV mutation.
- Cancel during pack (DMS, later): finish the in-flight pack; drop at its
  natural boundary.

Mid-step cancellation is never honored. This is what keeps KV mutation atomic
and what lets graph capture buckets stay valid across cancels.

## Per-row sampler and `n>1`

The coalescer's "compatible sampling key" requirement is a current-runtime
limitation, not a target architecture. Continuous batching needs the sampler
to accept per-row parameters in one kernel launch.

- Logits computed per row in one `w8a16_linear_bf16_f32_out` launch (current
  code path; already row-shaped).
- Sampling reads a **per-row params block** instead of scalar params:
  - `temperature[C]`
  - `top_k[C]` (or `0` = greedy)
  - `top_p[C]` (or `1.0` = no top-p)
  - `repetition_penalty[C]`
  - `seed[C]`
  - `stop_token_id[C][K_STOP_MAX]`
- Per-row EOS handling: the sampler emits a `done` flag per row when a stop
  token matches; the scheduler reclaims that row at the next commit.
- The submission-time coalescer (`_GenerationBatcher`) becomes redundant once
  the engine loop is live; remove it or keep it as a cold-path latency
  optimization for empty-pool startup bursts only.

`n>1` then lowers naturally: N submissions of the same prompt with distinct
seeds, shared prefix until the first divergent token.

## Observability contract

### Per-request fields (recorded in completion event and `/metrics`)

| Field | Meaning |
| --- | --- |
| `queue_seconds` | Time between `submit` and first `ADMIT`. |
| `prefill_seconds` | Wall time spent in `PREFILL_CHUNK` ticks owned by this request. |
| `decode_seconds` | Wall time spent in `DECODE_STEP` ticks where this row is active. |
| `tokens_generated` | Sampled tokens (excluding seed). |
| `kv_pages_owned` | Pages currently refcounted to this request at finish. |
| `kv_pages_peak` | Peak pages referenced by this request during its lifetime. |
| `kv_pool_bytes_at_admit` | Pool size when the request was admitted. |
| `bucket_key` | Decode graph bucket the request ran under most. |
| `admission_blocked_reason` | If queued; one of `kv_capacity_high_water_reached`, `pending_queue_full`, `device_oom`, …. |
| `finish_reason` | `stop`, `length`, `cancel`, `timeout`, `error`. |

### Per-pool counters

| Field | Meaning |
| --- | --- |
| `kv_pool_current_bytes` | Allocator-visible KV pool size right now. |
| `kv_pool_high_water_observed` | Largest size the pool has reached. |
| `kv_pool_grow_events` | Successful chunk allocations. |
| `kv_pool_grow_failures` | Allocations that hit `device_oom` or `high_water`. |
| `kv_pool_shrink_events` | Tail-chunk frees. |
| `kv_pool_free_pages` | Free pages right now. |
| `kv_pool_refcounted_pages` | Pages whose refcount > 1 (prefix sharing). |

### Per-bucket counters

| Field | Meaning |
| --- | --- |
| `graph_bucket_entries` | Distinct keys currently captured. |
| `graph_bucket_hits` | Replays since last reset. |
| `graph_bucket_misses` | Uncaptured fallbacks. |
| `graph_bucket_miss_reason` | `new_shape`, `chunk_added`, `mask_changed`, …. |
| `step_kernel_seconds` | Histogram of kernel-wall time per step. |

### `/metrics` endpoint

- Prometheus text format, when running `hipengine serve`.
- Knob: `HIPENGINE_METRICS` / `--metrics` in `{off, prometheus}`. Default `off`
  until C5.
- Per-request fields are exposed as histograms; per-pool / per-bucket as
  gauges and counters.

## Quant / model coverage matrix under c>N

The four-axis registry means every `(model, quant, KV dtype)` triple needs its
own c>N validation. This matrix tracks coverage; rows without a green retained
cell at c=2/4/8 are not c>N-eligible regardless of the engine-loop work.

| (model, quant, KV) | c=1 long | c=2 512/128 | c=4 512/128 | c=8 512/128 |
| --- | --- | --- | --- | --- |
| Qwen3.5/PARO × w4_paro × BF16 | retained | rejected_correctness *(experimental)* | not_started | not_started |
| Qwen3.5/PARO × w4_paro × INT8/per-token-head | retained (capacity) | not_started | not_started | not_started |
| GGUF × Q4_K × BF16 | retained | not_started | not_started | not_started |
| GGUF × Q5_K × BF16 | retained | not_started | not_started | not_started |
| GGUF × Q6_K × BF16 | retained | not_started | not_started | not_started |
| GGUF × Q8_0 × BF16 | retained | not_started | not_started | not_started |
| W8A16 dense × BF16 | partial | not_started | not_started | not_started |

Status legend: `not_started`, `primitive_ok` (kernel correctness only),
`eq_ok` (generated-token equality vs c=1, blocked on protocol shape),
`retained` (accepted retained row), `rejected_correctness` (equality failed).

GGUF c>N coverage is required for the repo's namesake quant path. It can
follow the Qwen3.5/PARO equality template once the engine loop and per-row
sampler are live.

## Benchmark eligibility gates

A c>N row is not eligible for `accepted` status until all of these pass:

1. `scripts/qwen35_batch_correctness.py --rows N` passes for the exact
   primitive families used by the runner: `append_key_mismatch=0`,
   `append_value_mismatch=0`, `attn_batch_vs_c1_max_abs <= 1e-6`.
2. The resident batch runner emits generated-token IDs equal to N independent
   c=1 resident runs for the same fixed prompts with greedy sampling and
   SpecDec disabled.
3. The artifact records scheduler occupancy, active mask shape, graph bucket
   key, KV policy, packed-prefill status, compaction events, and whether any
   serial bridge remains.
4. Continuous-batching rows include admission/completion timestamps and
   per-request p50/p95 latency in addition to aggregate tok/s.
5. Performance summaries show both aggregate tok/s and per-request tok/s.
   Never compare c=N aggregate to c=1 without also showing aggregate/c1 and
   per-request/c1 ratios.
6. **(dynamic pool)** Pool grew and shrank on a burst+idle workload, or the
   run fit in the initial chunk and the artifact says so. `grow_events ≤
   ceil((peak_bytes - initial_bytes) / chunk_bytes)`.
7. **(stable block id)** Debug check asserts no block-id pointer changed
   during the run.
8. **(prefix sharing, when enabled)** Shared-prefix workload artifact shows
   KV-byte savings *and* per-request TTFT drop vs the same workload with
   prefix sharing off.

## Bite-sized implementation queue

The phase ladder below is the source of truth for C0..C5. This queue expands
those phase items into implementation-sized packets for a multiloop or a human
coder. A good packet is one logical commit with a narrow test/bench gate and a
WORKLOG entry when it changes runtime behavior, correctness, or performance.
Do not check a packet merely because code exists; check it only when the
listed acceptance gate has passed and the parent phase item can cite it.

Recommended order: finish the C2 correctness packets before C3/C4/C5 feature
work, because continuous batching and KV sharing only matter once native c>N
emits the same tokens as independent c=1. For multiloop progress, count open
or partial checkboxes in this queue only; the phase ladder below stays as the
roll-up/status view.

### C0 packets — make diagnostics durable

- [x] **C0.1 c-sweep CLI.** Add `hipengine bench c-sweep` (or an equivalent
      `scripts/qwen35_batch_c_sweep.py`) that runs c=1/2/4/8 primitive,
      serial-bridge, and native-diagnostic commands from one config without
      copy/paste loops. Acceptance: JSON summary records every command,
      status, artifact path, and dirty git state. Evidence:
      `hipengine bench c-sweep --dry-run ...` plus
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C0.2 artifact schema guard.** Add a CPU test/helper that rejects c>N
      diagnostic JSON missing `workload.native_compact_prefill`,
      `execution.batch_execution.native_compact_prefill`,
      `native_caware_decode` as an execution flag, a correctness/status field,
      and `throughput_claim_eligible`. Acceptance: failing fixture proves the
      guard catches a missing field. Evidence: `scripts/qwen35_batch_artifact_schema.py`
      plus `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C0.3 promote current diagnostics.** Move or regenerate the current
      c=2 accepted/rejected diagnostic JSONs under `benchmarks/results/` with
      `status=blocked` or `rejected_correctness` as appropriate. Acceptance:
      `WORKLOG.md` links exact commands and no scoreboard row is added unless
      `status=accepted`. Evidence:
      `benchmarks/results/2026-05-27-hipengine-qwen35-paro-c2-native-l40-512-32-blocked.json`
      and `benchmarks/results/2026-05-27-hipengine-qwen35-paro-c2-native-l40-512-128-rejected-correctness.json`.

### C1 packets — keep current integration safe

- [x] **C1.1 lock scope audit.** Trace the server/generator mutation paths
      protected by `generation_lock`; document which session state is still
      non-reentrant. Acceptance: a focused test or review note proves the lock
      is narrow enough for C1 and names the exact blocker for C4 removal.
      Evidence: §C1 lock-scope audit plus `hipengine/server/api.py` code refs.
- [x] **C1.2 API rejection contract.** Keep `n>1` rejected until C5 and add
      regression coverage if missing for completions and chat. Acceptance:
      server tests prove `n>1` returns the intended 4xx while prompt-list
      batching still works. Evidence: `pytest -q tests/test_server_api.py -q`.

### C2 packets — native BF16 c>N correctness first

- [x] **C2.1 remove compatibility shim.** Remove the generator
      `batch_execution_metadata(...)` `TypeError` compatibility path once all
      call sites use the settled signature. Acceptance: targeted generator and
      resident-layout tests pass. Evidence: commit removing the shim plus
      `pytest -q tests/test_generation_qwen35_paro.py tests/test_qwen35_resident_batch_layout.py -q`.
- [ ] **C2.2 hidden-state bisection harness.** Add a HIP-guarded diagnostic
      that compares c=2 native vs independent c=1 hidden tensors after each
      layer and optionally after sub-stages (attention, selected MoE, shared
      expert, combine, LM head). Acceptance: the harness can reproduce the
      current L40 c=2 512/128 divergence earlier than generated-token idx 87.
- [ ] **C2.3 selected-MoE lane-map fix.** Root-cause
      `/tmp/hipengine-retained/eq-L8-selectedmoe.json`; fix token-row → routed
      lane mapping or grouped metadata so selected MoE is hidden-equality green
      at c=2. Acceptance: C2.2 reports selected-MoE hidden equality for the
      failing fixture and generated-token equality progresses past the old
      idx-13 failure.
- [ ] **C2.4 full c=2 BF16 512/128 equality.** Re-run the full 40-layer c=2
      512/128 retained protocol with `serial_lm_head` default and no serial
      decode bridge. Acceptance: generated-token equality vs two c=1 sessions
      passes; if timing is still not retained, artifact is `blocked` for a
      non-correctness reason.
- [ ] **C2.5 c=4/c=8 BF16 equality.** Extend the same gate to c=4 and c=8.
      Acceptance: generated-token equality passes for both shapes, with
      aggregate/per-request scaling fields recorded even if not yet optimized.
- [x] **C2.6 sparse-slot and long-context guards.** Add CPU structural tests
      for sparse/non-contiguous slot rejection and `max_context >= 1024`
      rejection until row-aware split-K is live. Acceptance: tests fail if the
      experimental path silently accepts unsupported shapes. Evidence:
      `pytest -q tests/test_qwen35_resident_batch_layout.py -q`.
- [ ] **C2.7 row-aware split-K full attention.** Make full-attention decode
      and reduction consume per-row spans for `max_context >= 1024` before any
      long-context c>N claim. Acceptance: primitive correctness plus a
      generated-token diagnostic at a long-context shape.
- [x] **C2.8 append-only block-id contract.** Prevent block ids from changing
      backing pointer during a live request; add a debug/memory-audit test.
      Acceptance: the test would fail on pointer mutation or id reuse.
      Evidence: `FixedPagedKVPolicy(...).register(block_pointer_map=...)` plus
      `pytest -q tests/test_kvcache_policy.py -q`.
- [x] **C2.9 live admission cap.** Make `KVPolicy.admission_cap()` return
      current free capacity rather than startup capacity. Acceptance: fake
      policy/scheduler tests show reclaim changes admission capacity before the
      next admit. Evidence: `pytest -q tests/test_kvcache_policy.py -q`.

### C3 packets — widen kernel/model coverage

- [ ] **C3.1 INT8 KV c>N parity.** Validate batched INT8 KV append/decode
      end-to-end with the same generated-token gates as BF16. Acceptance:
      c=2 512/128 INT8 artifact is equality-green or explicitly
      `rejected_correctness` with first mismatch.
- [ ] **C3.2 per-row `KVLiveSpans` everywhere.** Audit full-attention decode,
      KV append, and storage-dtype wrappers for scalar `(block_table,
      context_len)` shortcuts. Acceptance: tests cover BF16 and INT8 per-row
      spans.
- [ ] **C3.3 linear-attention `[C]` state.** Remove c1 aliases from
      conv/recurrent state update paths and use active masks + slot ids.
      Acceptance: c=2 state fixtures compare against two c=1 references.
- [ ] **C3.4 c-aware projection dispatch.** Keep c=1 on GEMV/Marlin-K while
      routing c=2/4/8 to MMQ/GEMM/WMMA candidates only when they beat row-GEMV.
      Acceptance: dispatch tests prove thresholds and benchmark artifacts show
      aggregate/per-request ratios.
- [ ] **C3.5 GGUF c>N template.** Port the Qwen/PARO equality template to
      GGUF Q4_K/Q5_K/Q6_K/Q8_0. Acceptance: at least one GGUF c=2 diagnostic
      reaches an unambiguous `eq_ok`, `blocked`, or `rejected_correctness`
      status with exact command.
- [ ] **C3.6 native LM-head/sampler launch.** Replace the per-row
      `serial_lm_head` loop with a native row-aware LM-head/argmax only after
      C2 equality is green. Acceptance: c=2/4/8 equality stays green with
      `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE=batched_lm_head` or successor.

### C4 packets — continuous scheduler and dynamic KV pool

- [x] **C4.1 engine-loop skeleton.** Introduce long-lived
      `submit/poll/cancel` driver around existing resident sessions, initially
      using fake/CPU tests and the serial bridge. Acceptance: requests can be
      admitted, decoded, finished, and reclaimed without a one-call lifetime.
      Evidence: `hipengine/generation/engine_loop.py` plus
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [ ] **C4.2 adapter migration.** Lower `LLM.generate()` and non-streaming
      server endpoints onto `submit+poll` while preserving current outputs.
      Acceptance: existing generator/server tests pass and prompt-list
      batching still routes by request id.
- [x] **C4.3 tick policy.** Implement `RECLAIM → ADMIT → choose(PREFILL_CHUNK,
      DECODE_STEP)` with `protect_decode` default. Acceptance: scheduler tests
      cover decode protection and TTFT/fair alternatives. Evidence:
      `ResidentEngineLoop(prefill_decode_policy=...)` plus
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [ ] **C4.4 chunked KV pool.** Add chunked allocation, grow-on-admission,
      idle shrink, and high/low-water knobs behind fake-runtime tests first.
      Acceptance: burst+idle fixture records at least one grow and shrink or
      explicitly records that the initial chunk sufficed.
- [ ] **C4.5 pool/env docs.** Add CLI/env knobs for `HIPENGINE_KV_POOL_*` and
      `HIPENGINE_PREFILL_DECODE_POLICY` and document them in `docs/ENVS.md`.
      Acceptance: CLI/env tests and docs agree on defaults.
- [ ] **C4.6 streaming through loop.** Route streaming completions through
      per-request token queues instead of bypassing the batcher. Acceptance:
      streaming and non-streaming share reclaim/cancel tests.
- [x] **C4.7 unified reclaim.** Make cancel, disconnect, EOS, max-tokens, and
      timeout converge on one `RECLAIM` path. Acceptance: each finish reason
      frees KV/scratch exactly once in tests. Evidence:
      `ResidentBatchScheduler.cancel/disconnect/timeout(...)`, generated-token
      `stop`/`length` reclaim, and `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [ ] **C4.8 non-compact-slot native decode.** Extend native decode beyond
      compact `0..C-1` slots after scheduler compaction/reclaim. Acceptance:
      generated-token equality passes with a deliberately sparse/compacted
      slot schedule.
- [ ] **C4.9 observability fields.** Record per-request and per-pool fields in
      completion/artifact metadata. Acceptance: tests assert queue/prefill/
      decode seconds, KV pages, bucket key, admission blocker, and finish
      reason are present.

### C5 packets — prefix sharing, per-row sampling, `n>1`, metrics

- [ ] **C5.1 block refcounts.** Add block-id refcounts and reuse accounting.
      Acceptance: shared-prefix admission increments/decrements refcounts and
      reclaim only frees zero-refcount blocks.
- [ ] **C5.2 RadixCache.** Implement the token-id trie with
      `HIPENGINE_PREFIX_CACHE` / `--prefix-cache` in `{off, radix}`. Acceptance:
      prefix-hit/miss tests cover partial-block edges and cancellation.
- [ ] **C5.3 copy-on-write fork.** Fork fresh pages at the first divergent
      token while preserving shared prefix pages. Acceptance: two diverging
      requests keep prefix bytes shared and produce independent suffix KV.
- [ ] **C5.4 `n>1` lowering.** Replace API rejection with N scheduler
      requests sharing a prompt prefix and distinct seeds. Acceptance:
      OpenAI-compatible responses preserve `n` semantics and request IDs.
- [ ] **C5.5 per-row sampler.** Land per-row temperature/top-k/top-p/
      repetition-penalty/seed/stop-token handling. Acceptance: incompatible
      sampling params decode together and deterministic seeds are stable.
- [ ] **C5.6 per-row EOS/reclaim.** Finish rows independently inside a batch.
      Acceptance: one row can finish while others keep decoding and its KV is
      reclaimed at the next commit point.
- [ ] **C5.7 metrics endpoint.** Add Prometheus `/metrics` behind
      `HIPENGINE_METRICS` / `--metrics`. Acceptance: metrics are additive and
      include request, pool, and graph-bucket counters.
- [ ] **C5.8 retained-row enforcement.** Make the bench harness enforce gates
      for timestamps, p50/p95 latency, dynamic pool, stable block id, and
      prefix-sharing savings before `status=accepted`. Acceptance: a fixture
      missing any required field cannot be accepted.

### Performance packets — run only after correctness is green

- [ ] **P1 baseline bundle.** Establish c=1, serial bridge c=2/4/8, first
      green uncaptured native c>N, and primitive microbench baselines.
      Acceptance: artifacts include exact commands, hardware, correctness,
      aggregate/per-request ratios, and dirty-state.
- [ ] **P2 graph replay buckets.** Add decode hipGraph capture/replay buckets
      by `(C, context bucket, active mask, KV dtype, layer plan, top-k/experts,
      replay length)`. Acceptance: bucket hit/miss stats and profiler evidence
      show replay for common shapes.
- [ ] **P3 remove residual serial loops.** Remove full-attention per-row
      fallback, per-row metadata allocation, per-row LM-head launches, and
      Python per-layer dispatch from steady-state native decode. Acceptance:
      profiler summaries show the removed bottleneck and equality remains
      green.
- [ ] **P4 MoE/projection scaling.** Group routed lanes by expert and switch
      c=2/4/8 projections/MoE to kernels that beat row-GEMV. Acceptance:
      c=8 aggregate decode improves vs both c=1 and the serial bridge, with
      per-request ratios reported.
- [ ] **P5 retained scoreboard update.** Only after accepted artifacts exist,
      update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and compact
      JSON artifacts under `benchmarks/results/`. Acceptance: every perf claim
      cites correctness gate, profiler status, exact command, and hardware.

## Phase ladder

The phase ladder is the ground truth for c>N progress. Each phase has a
"definition of done" and a checklist. Boxes that are checked have shipped
commits in `git log`; unchecked are open work.

### C0 — keep diagnostics honest

Definition of done: every c>N number on disk is unambiguously labeled
`serial_bridge`, `experimental`, or `retained`.

- [x] Generate c≤8 prompt fixtures for 512/128 and 4K/128 diagnostics.
- [x] Run c=1/2/4/8 primitive correctness on GPU0.
- [x] Run c=1/2/4/8 scheduler serial bridge diagnostics and record blocked
      status.
- [x] Promote `rejected_correctness` as a distinct status in the retained
      bench harness so failing-equality rows are not silently `blocked`.
- [x] Add a `hipengine bench c-sweep` subcommand that runs the full
      diagnostic sweep without copy/paste loops.
- [x] Ensure every diagnostic artifact distinguishes
      `workload.native_compact_prefill`,
      `execution.batch_execution.native_compact_prefill`,
      `native_caware_decode` (execution flag, not correctness),
      a correctness-pass/status field, and `throughput_claim_eligible`.

### C1 — server and generator integration

Definition of done: prompt-list and short-window HTTP coalescing reach the
batch generator; `n>1` rejected; streaming unchanged.

- [x] Batch-capable Qwen/PARO generator path for prompt lists with scheduler
      request ids, physical slots, packed prefill slabs, and output routing.
- [x] Coalesce compatible non-streaming server generations into one
      prompt-list `LLM.generate()` call.
- [x] Preserve a narrow safety lock only around non-reentrant
      model/session mutation until the session is proven concurrency-safe.
- [x] Keep `n>1` rejected at the API layer until C5 lowers it to N
      scheduler requests.

### C2 — native c>N prefill/decode green

Definition of done: full 40-layer Qwen/PARO BF16 c=2/4/8 512/128 emits
generated-token IDs equal to independent c=1 runs, with no serial decode
bridge and `throughput_claim_eligible=true`. Append-only block-id contract
in place even though pool growth lands in C4.

- [x] Native compact packed BF16 prefill via `next_compact_prefill_slabs(...)`
      + `prefill_native_packed(...)`.
- [x] Guard `step_batch_native` behind
      `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1` and default
      `_sample_batch_from_hidden` to `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE=
      serial_lm_head` until row-aware sampler lands.
- [x] Document `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE` and
      `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE` in `docs/ENVS.md`.
- [x] Remove stale compatibility glue once the guarded native API is
      settled (removed the `batch_execution_metadata(...)` `TypeError`
      shim in the generator).
- [ ] Add HIP-guarded reduced-shape equality diagnostics that do **not**
      require full 40 layers, so failures can be bisected in CI/dev
      environments with ROCm. Keep full 40-layer 512/128 as the retained
      benchmark gate.
- [x] Re-run guarded current-default c=2 equality after `serial_lm_head`:
      L40 512/32 passes as a reduced diagnostic, but full L40 512/128 is
      still `rejected_correctness` at row 0 idx 87
      (`/tmp/hipengine-retained/guarded-L40-c2-512-128-current.json`).
- [x] Promote current c=2 accepted/rejected diagnostic artifacts under
      `benchmarks/results/` before using them as retained evidence.
- [ ] Root-cause and fix the selected-MoE c>N divergence
      (`/tmp/hipengine-retained/eq-L8-selectedmoe.json`).
- [ ] Add row-aware split-K full-attention decode/reduce before any
      long-context c>N claim (`max_context ≥ 1024`).
- [x] Add CPU-runnable structural tests for the experimental env gate,
      INT8 KV rejection, default/invalid sample mode, and
      `throughput_claim_eligible=false` for guarded diagnostics.
- [x] Extend structural tests for sparse-slot and long-context rejection.
- [x] **Append-only block id contract.** Make the KV allocator's block id
      permanent for its lifetime. Remove any path that reuses a block id at
      a different pointer. Add a debug check that fails on pointer mutation.
- [x] **Live admission cap.** `KVPolicy.admission_cap()` returns *current*
      free capacity, not startup capacity.

### C3 — kernel coverage

Definition of done: every retained `(model, quant, KV)` row in the coverage
matrix has at least one green retained c>N cell on the 512/128 protocol.

- [ ] Validate batched INT8 KV append/decode paths with the same gates as
      BF16; require generated-token equality.
- [ ] Make full-attention decode consume per-row `KVLiveSpans` for all
      retained KV storage dtypes.
- [ ] Make linear-attention conv/recurrent state updates consume
      `[C, ...]` state, active masks, and slot ids; remove c1 aliases.
- [ ] Replace selected-MoE c1 lane assumptions with token-row → routed-lane
      mapping; validate grouped-by-expert metadata for c=2/4/8.
- [ ] Keep c=1 GEMV dispatch separate from c>N MMQ/GEMM/WMMA candidates.
- [ ] Validate GGUF Q4_K/Q5_K/Q6_K/Q8_0 c=2/4/8 with the same gates.
- [ ] Native row-aware LM-head + sampler: replace the per-row argmax loop
      and prepare for per-row sampling params (C5 finishes this).

### C4 — scheduler-owned engine loop + dynamic KV pool

Definition of done: one long-lived background driver runs
`submit/poll/cancel`, ticks the work classes, grows/shrinks the KV pool, and
routes both streaming and non-streaming through the same path. `LLM.generate()`
becomes a `submit+poll` adapter.

- [ ] Promote the resident runner from static prompt-list batches to a
      scheduler-owned engine loop that persists beyond one
      `LLM.generate()` call.
- [ ] Implement `submit(prompt_tokens, sampling, max_new_tokens, stream) →
      request_id`, `poll(timeout) → events`, `cancel(request_id) → bool`.
- [ ] Lower `LLM.generate()` and OpenAI server endpoints to
      `submit + poll + cancel`.
- [x] Implement the per-tick policy: `RECLAIM → ADMIT → choose(PREFILL_CHUNK,
      DECODE_STEP)`; default `protect_decode`.
- [ ] Add `kv_pool_chunk_pages` chunked underlying allocation with one chunk
      at startup.
- [ ] Add grow-on-admission up to `kv_pool_high_water_bytes`, one attempt per
      admit cycle; record `grow_events` / `grow_failures`.
- [ ] Add idle shrink down to `kv_pool_low_water_bytes` with
      `kv_pool_idle_grace_seconds`; never free a chunk holding a non-zero
      refcount.
- [ ] Add CLI/env knobs `--kv-pool-{initial,low-water,high-water,
      chunk-pages,idle-grace}-*`,
      `HIPENGINE_KV_POOL_*`,
      `HIPENGINE_PREFILL_DECODE_POLICY` / `--prefill-decode-policy`;
      document in `docs/ENVS.md`.
- [ ] Add a burst-then-idle acceptance fixture that exercises grow and
      shrink and records the events.
- [ ] Add a memory-audit test that fails if a block id's backing pointer
      changes mid-run.
- [ ] Narrow or remove the coarse `generation_lock`; any remaining lock
      protects only non-reentrant session mutation, not the lifetime of a
      generated batch.
- [ ] Route server streaming through the engine loop and the per-request
      token queue; the streaming path no longer bypasses the batcher.
- [x] Unify cancel / disconnect / EOS / max-tokens / timeout into one
      `RECLAIM` path.
- [ ] Per-request observability fields (queue/prefill/decode seconds,
      kv pages owned/peak, bucket key, admission_blocked_reason,
      finish_reason).
- [ ] Per-pool observability counters
      (current_bytes, high_water_observed, grow/shrink events, free pages,
      refcounted pages).
- [ ] Extend native decode correctness to non-compact slots after
      scheduler compaction/reclaim moves requests; today only compact
      `0..C-1` slots are supported.

### C5 — KV sharing, per-row sampler, `n>1`, `/metrics`

Definition of done: refcounted prefix sharing on by default; per-row sampler
in code; `n>1` lowered to N scheduler requests; Prometheus `/metrics`
endpoint live; retained c>N rows include all gates above.

- [ ] Add block-id refcounts; admission increments refcount when reusing
      an existing block on a matched prefix.
- [ ] Implement RadixCache trie index over token ids; expose
      `HIPENGINE_PREFIX_CACHE` / `--prefix-cache` in `{off, radix}` with
      default `off` until acceptance gates pass.
- [ ] Implement copy-on-write fork at first divergent token.
- [ ] **KVTC ABI guardrail.** Block ids returned by the allocator must be
      stable across hypothetical tier moves; refcount and eviction state
      must attach to the radix node rather than the block pointer. KVTC
      itself ships in a separate feature branch.
- [ ] Lower `n > 1` at the API layer to N `submit()` calls with the same
      prompt tokens and distinct seeds; collect output by `request_id`;
      remove the `n>1 → 400` rejection.
- [ ] Land the per-row sampler params block (temperature, top-k, top-p,
      repetition penalty, seed, stop tokens) in one launch.
- [ ] Per-row EOS handling drives `RECLAIM` per-row, not per-batch.
- [ ] Remove or demote the submission-time coalescer
      (`_GenerationBatcher`) to a cold-path optimization.
- [ ] Add Prometheus `/metrics` endpoint;
      knob `HIPENGINE_METRICS` / `--metrics` in `{off, prometheus}`;
      default `off` until coverage is broad.
- [ ] Per-bucket graph-cache observability
      (entries, hits, misses, miss reason, kernel-time histogram).
- [ ] Retained-row gates 4 (admission/completion timestamps + p50/p95) and
      6/7/8 (dynamic pool + stable block id + prefix sharing artifact)
      enforced by the bench harness.

## Performance gates and optimization work

The phase ladder above is organized by *what is enabled* (correctness,
engine loop, sharing). Performance-scaling work runs **inside** each phase
after the correctness gate for that phase is green. This section collects
the shared performance contract; cite a specific phase when scheduling a
performance item.

### Baseline artifacts

Establish these before optimizing anything:

- c=1 native prefill/decode for the retained shapes.
- c=2/4/8 serial bridge diagnostics.
- First green uncaptured native c>N rows (no graph replay).
- Primitive/kernel microbenchmarks for attention, KV append, MoE,
  projection, and LM-head sampling.

### Scaling reported on every retained c>N row

- `prefill_tok_s_aggregate / c1_prefill_tok_s`.
- `decode_tok_s_aggregate / c1_decode_tok_s`.
- `decode_tok_s_per_request / c1_decode_tok_s`.
- p50/p95 first-token latency and inter-token latency.
- Active-batch occupancy over time.

### Target throughput envelope

- Decode aggregate speedup vs c=1 and vs the serial bridge. Per
  [`PLAN.md`](PLAN.md), c=8 decode plausibly lands around 2-4× c=1
  aggregate when kernels reuse enough work; do not promise 8×.
- Prefill aggregate scaling vs c=1 by keeping prompt rows packed, avoiding
  per-request Python loops, and using AOTriton/WMMA paths where they beat
  row-GEMV.

### Optimization checklist (overlay onto C2-C5)

- [ ] Add hipGraph capture/replay buckets for decode by `(C, context
      bucket, active mask, KV dtype, layer plan, top-k/experts, replay
      length)`, with an uncaptured fallback for rare shapes.
- [ ] Add graph-bucket cache hit/miss and replay statistics to artifacts.
- [ ] Eliminate residual serial loops on the native path after correctness
      is green:
  - full-attention per-row fallback;
  - per-row host metadata allocation/free;
  - per-row LM-head/argmax launches where a batched launch is correct;
  - Python per-layer dispatch overhead inside steady-state decode.
- [ ] c-aware projection dispatch thresholds:
  - c=1 stays on tuned GEMV/Marlin-K paths;
  - c=2/4/8 use MMQ/GEMM/WMMA-style kernels when they beat row-GEMV;
  - c>16 prefers GEMM/WMMA and grouped MoE designs over widening c1
    GEMV wrappers.
- [ ] MoE routed-lane reuse:
  - group lanes by expert;
  - use compact/WMMA grouped kernels when routed lanes justify it;
  - measure router, group-scatter, gate/up, down, shared expert, and
    combine time separately.
- [ ] Memory traffic and workspace reuse:
  - preallocate per-bucket scratch instead of allocating per step;
  - avoid host-device copies for metadata that can be updated on device;
  - keep JIT builds out of profiler runs with `require_cached`;
  - track peak allocator/KV/workspace bytes in artifacts.
- [ ] Backpressure and fairness policies once the scheduler is continuous:
  - max active requests, max queued requests, max prefill chunk tokens;
  - prefill-vs-decode policy to protect decode latency (default
    `protect_decode`, see §Engine-loop contract);
  - sampling-parameter grouping without starving incompatible requests
    once the per-row sampler is live.
- [ ] Profiler summaries for accepted rows: expected kernel names,
      duration/share for attention, MoE, projection, sampling, graph
      replay, and any CPU-side bottleneck.
- [ ] Only update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and
      `benchmarks/results/` for retained rows with correctness green,
      protocol shape satisfied, and profiler evidence. Rejected/blocked
      diagnostics stay useful but are not scoreboard entries.

## Out of scope (C6 onward)

C6 onward is out of scope for this doc:

- **TP / EP** (separate feature branch). Design in [`PLAN.md`](PLAN.md)
  §Multi-GPU Strategy. CONCURRENCY contracts are designed to be TP-safe;
  see §Forward-compatibility guardrails.
- **DMS compact KV serving** (separate feature branch). Roadmap in
  [`KVCACHE.md`](KVCACHE.md) Phase K2. CONCURRENCY contracts are designed to
  be DMS-safe; see §Forward-compatibility guardrails.
- **KVTC tiered KV storage** (separate feature branch). HBM → pinned host
  RAM → NVMe / disk. CONCURRENCY contracts are designed to be KVTC-safe;
  see §Forward-compatibility guardrails.
- **Speculative decoding** (separate feature branches). MTP, DFlash,
  EAGLE3; designs in [`MTP.md`](MTP.md), [`DFLASH.md`](DFLASH.md),
  [`SPECULATIVE-DECODE.md`](SPECULATIVE-DECODE.md).

## Forward-compatibility guardrails

CONCURRENCY-side decisions that the TP, DMS, SpecDec, and KVTC feature
branches depend on. The work in C0..C5 must already satisfy these; they are
not new tasks.

### Don't break TP

- **Scheduler / admission / sampler are single-owner.** TP rank-0 will own
  the engine loop; workers tick in lockstep. Do not put admission, sampling,
  or pool-resize decisions inside per-rank code.
- **hipGraph bucket keys are rank-agnostic.** Same
  `(C, context bucket, active mask, KV dtype, layer plan, top-k/experts,
  replay length)`. Per-rank capture/replay is fine; key derivation isn't
  per-rank.
- **All-reduce points are loop-visible.** Reductions happen after `o_proj`,
  after `down_proj`, after the shared expert
  (per [`PLAN.md`](PLAN.md) §Multi-GPU Strategy). Don't fold reductions into
  kernel internals where the loop can't see them.
- **KV is replicated per rank first, sharded later.** `KVLiveSpans` is
  per-rank; the scheduler does not assume rank-shared KV. The dynamic pool
  is per-rank; admission uses `min(per_rank_admission_cap)`.
- **No `if backend == "hip_tp_*"` branches in dispatch.** TP variants
  register as `(backend, layer, quant, variant)` tuples.

### Don't break DMS

- **`KVLiveSpans` stays the only attention / KV-write ABI.** No
  `(block_table, context_len)` shortcuts anywhere in the c>N decode path.
- **`KVPolicy.admission_cap()` is the scheduler's capacity unit.** Today
  returns dense-page capacity; DMS will return compact-live-token capacity.
  Continuous batching must not assume page == 1-token equivalent.
- **KV mutation is transactional.** Canonical KV updates only at scheduler
  commit points; verify/spec rows write scratch/journal. DMS evictions need
  the same commit-point gate.
- **Eviction-aware prefix sharing is a separate decision.** When prefix
  sharing lands in C5, either disable it under DMS or design per-sequence
  eviction overlays. Don't blind-share under variable-span eviction.
- **The engine loop must allow a `PACK_STEP` work class to be inserted
  between active requests' decode steps.** Don't model the loop as a strict
  `prefill ; decode_until_done` macro-pattern.

### Don't break SpecDec

- **Verify rows commit only at scheduler commit points.** Canonical KV (dense
  or compact) is updated only on accept; rejects discard scratch.
- **`DraftBatch` metadata is the verify ABI.** Verification kernels consume
  `request_id`, candidate token(s), parent position, draft depth, optional
  tree parent, and active mask. No c=1 chain shortcuts.
- **`VERIFY_STEP` is a peer work class.** The loop's per-tick policy can
  schedule verify steps without changing the contract.

### Don't break KVTC

- **Block id stays stable across tier moves.** A block id `b`'s id and
  refcount are preserved when its backing pointer moves between HBM,
  pinned host RAM, or NVMe. Consumers ask the allocator for the current
  pointer rather than caching it. This is a strict extension of the
  append-only block-id contract in §Dynamic KV pool.
- **Refcount and eviction state live on the radix node, not the block
  pointer.** Tier moves do not change prefix-sharing topology.
- **Tier moves happen only at scheduler commit points.** No mid-kernel
  tier promotion or demotion; no torn `KVLiveSpans` reads.
- **`KVLiveSpans` is unchanged by tiering.** A tier move is a pointer swap
  inside the allocator, not a span rewrite.
- **The `/metrics` schema is extensible.** When KVTC lands it adds
  counters like `kv_tier_promotions_total{tier}` and
  `kv_tier_demotions_total{tier}`; CONCURRENCY's per-pool counter block
  must be additive, not restructured.

## GPU0 diagnostic evidence

Most historical scratch fixtures and artifacts live under `/tmp/hipengine-prebench/`
and `/tmp/hipengine-retained/`. The current c=2 native-decode review artifacts
are promoted under `benchmarks/results/` as blocked/rejected diagnostics:

- `benchmarks/results/2026-05-27-hipengine-qwen35-paro-c2-native-l40-512-32-blocked.json`
- `benchmarks/results/2026-05-27-hipengine-qwen35-paro-c2-native-l40-512-128-rejected-correctness.json`

These are not scoreboard rows because neither has `status=accepted`.

Primitive c=1/2/4/8 correctness (BF16 batched KV append + batched paged
full-attention decode):

| c | append key mismatch | append value mismatch | attn batch-vs-c1 max abs |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0.0 |
| 2 | 0 | 0 | 0.0 |
| 4 | 0 | 0 | 0.0 |
| 8 | 0 | 0 | 0.0 |

Scheduler serial bridge sweep (Qwen3.6/PARO-35B-A3B, W7900, 40 layers, INT8
KV, prompt 512 + 4K, decode 128):

| Shape | Decode aggregate tok/s | Decode per-request tok/s |
| --- | ---: | ---: |
| c=1 512/128 | 102.12 | 102.12 |
| c=2 512/128 | 102.32 | 51.16 |
| c=4 512/128 | 101.47 | 25.37 |
| c=8 512/128 | 100.30 | 12.54 |
| c=1 4K/128 | 99.98 | 99.98 |
| c=8 4K/128 | 98.88 | 12.36 |

Aggregate decode stays flat while per-request falls as `1/c` — the signature
of the serial bridge. Full per-row artifacts:
`/tmp/hipengine-prebench/scheduler/qwen36-paro-cC-{512,4k}-128-serial-bridge.json`.

Experimental native decode (commit `86e6fa2`) currently has two distinct
correctness signals:

- Pre-workaround batched LM-head L8 512/32 rejected at row 0 idx 13
  (`/tmp/hipengine-retained/guarded-L8-512-32.json`). Switching to
  `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE=serial_lm_head` fixes that reduced
  512/32 drift: `/tmp/hipengine-retained/eq-{L8,L40}-512-32-serialsample.json`
  passed, and the current-default rerun
  `/tmp/hipengine-retained/guarded-L40-512-32-current.json` also passed
  equality (`status=blocked` only because 32 decode tokens is reduced).
- Full 40-layer c=2 512/128 still rejects on current tip with the
  `serial_lm_head` default:
  `/tmp/hipengine-retained/guarded-L40-c2-512-128-current.json`, row 0 idx 87
  (`batch=271`, `c1=1165`), `throughput_claim_eligible=false`. The separate
  `/tmp/hipengine-retained/eq-L8-selectedmoe.json` failure points at
  selected-MoE/native-row mapping, so the next correctness step is layer-level
  hidden-state bisection rather than more token-only sweeps.

## What not to claim yet

Do not describe any current row as:

- true c=2/4/8 serving throughput;
- continuous batching;
- radix/prefix-cache reuse;
- compact/DMS KV serving;
- c-aware decode graph replay;
- dynamic KV pool growth/shrink;
- KVTC / tiered KV storage.

The correct phrasing for current diagnostics is:

> c>N scheduler serial bridge diagnostic: batch-shaped slots and KV metadata,
> but active rows execute serially through the c=1 layer path. Aggregate
> decode throughput remains roughly c=1, so the row is blocked/non-retained.
