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

**hipEngine now has most host-side continuous-batching scaffolding in code, but
it still must not claim true retained c>N throughput.** The remaining hard gate
is Qwen/PARO native c>N generated-token equality vs independent c=1, followed by
profiler/timing evidence and benchmark rollups.

What is in place:

- The server and `LLM.generate()` paths have prompt-list batching, `n>1`
  lowering, streaming through per-request queues, request ids, per-row seeds,
  and Prometheus metrics hooks.
- `SubmitPollTextGenerator` and `ResidentEngineLoop` provide a persistent
  `submit`/`poll`/`cancel` driver around `ResidentBatchScheduler` for tests and
  host integration, with `RECLAIM → ADMIT → PREFILL/DECODE` tick policy,
  per-request completion metadata, graph-bucket bookkeeping, and unified cancel,
  disconnect, EOS, max-token, and timeout reclaim.
- The KV/prefix scaffolding exists: `ChunkedKVPool` grows/shrinks in chunks,
  keeps append-only block ids, reports current admission capacity, supports
  shared-prefix refcounts and copy-on-write forks, and `RadixCache` indexes
  block-aligned token prefixes.
- Per-row sampling parameters and per-row EOS/reclaim are represented in the
  scheduler; artifact/schema gates prevent serial bridges, fallback execution,
  non-native sampler metadata, or incomplete timing/profiler payloads from being
  promoted as accepted retained c>N rows.

What is still not green:

- The retained Qwen/PARO native c>N decode path is experimental. BF16 primitive
  c=2/4/8 KV append/full-attention correctness passes, but generated-token
  equality is still missing for the full c=2 512/128 gate and therefore for
  c=4/c=8.
- Hidden-state bisection now separates generated-token equality from hidden drift:
  focused L4/L8 controls keep tokens green, but native full-attention remains a
  hidden-only failure at L8 even at `hidden_atol=0.004`
  (`/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-native-fullattn-atol4e-3-focus1269.json`).
  The matching selected-c1 MoE control preserves the failure, so grouped-compact
  MoE is not the source of that larger native-full drift. C2.3/C2.4/C2.5 remain
  the correctness priority.
- Long-context c>N still uses a per-row split-K fallback label; no long-context
  native c>N claim is allowed until the split-K reducer is row-aware.
- INT8 c>N parity, runtime projection dispatch evidence, native LM-head/sampler,
  graph replay buckets, residual-serial-loop removal, and retained scoreboard
  updates remain open performance/coverage work.

## Readiness matrix

| Layer | Current status | Evidence / code | Blocks retained c>N |
| --- | --- | --- | --- |
| OpenAI server | Non-streaming compatible requests still coalesce through `_GenerationBatcher`; streaming and non-streaming now share request accounting, `n>1` lowers to multiple choices with request ids, and `/metrics` is available behind `--metrics prometheus` / `HIPENGINE_METRICS=prometheus`. | `hipengine/server/api.py:_GenerationBatcher`, `_choice_request_id`, `_row_seeds_for_request`, `_render_prometheus_metrics`; `pytest -q tests/test_server_api.py -q`. | Coalescer can be demoted once native c>N equality/perf is green; no retained throughput claim comes from HTTP coalescing alone. |
| Public `LLM.generate()` / loop adapter | The public generator can be wrapped by `SubmitPollTextGenerator`, preserving outputs while exercising submit/poll semantics in tests. | `hipengine/generation/engine_loop.py:SubmitPollTextGenerator`; `pytest -q tests/test_generation_batch_scheduler.py -q`. | Native Qwen/PARO c>N decode equality and retained benchmark evidence. |
| Engine loop / scheduler | `ResidentEngineLoop` and `ResidentBatchScheduler` own pending/admitted queues, slots, active masks, compact prefill slabs, decode work, graph bucket keys, completion routing, and unified reclaim. | `hipengine/generation/engine_loop.py:ResidentEngineLoop`; `hipengine/generation/batch_scheduler.py`; scheduler tests. | Runtime equality/perf gates, not host-loop shape. |
| Prefill | BF16 compact/native prompt-list prefill is live; scheduler tests cover chunk/policy plumbing. INT8 retained c>N prefill remains blocked. | `prefill_native_packed`, `CompactPromptSlab`, `scripts/qwen35_batch_packed_prefill_correctness.py`; `tests/test_generation_batch_scheduler.py`. | INT8 c>N parity and retained end-to-end equality. |
| Decode runtime | Safe/diagnostic paths remain non-claiming: serial bridge rows and experimental native rows are blocked/rejected unless generated-token equality and native execution metadata pass. Focused L4/L8 native-full controls are hidden-only red with tokens green; full c=2 512/128 equality is still open. | `step_batch_serial`, `step_batch_native`, `_sample_batch_from_hidden`, `batch_execution_metadata`; retained/hidden-bisect artifacts cited in C2. | C2.3 native full-attention/post-attention hidden drift; C2.4 c=2 equality; C2.5 c=4/c=8 equality. |
| Sampler | `PerRowSamplingParams` and sampler blocks exist; native `batched_lm_head` dispatch is evidence-gated and falls back before C2 equality. | `hipengine/generation/batch_scheduler.py:PerRowSamplingParams`; `hipengine.dispatch.sampling`; sampler dispatch tests. | C3.6 native row-aware LM-head/sampler after C2 equality is green. |
| Attention / KV primitives | BF16 batched paged KV append and batched full-attention context decode pass c=1/2/4/8 primitive correctness. Split-K long-context decode is labeled per-row fallback. | `scripts/qwen35_batch_correctness.py`; `/tmp/hipengine-multiloop-c{2,4,8}-correctness.json`; attention dispatch tests. | Row-aware split-K reducer; INT8 end-to-end gate. |
| MoE / quant kernels | Grouped compact MoE scratch replaced selected-MoE c1 wrappers for `tokens>1`; the latest native-full selected-c1 MoE control still fails hidden-only at L8, ruling out grouped-compact MoE as the source of that larger drift. | `hipengine/runtime/qwen35_paro.py`; `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-native-fullattn-selected-c1-moe-atol4e-3-focus1269.json`. | c=2/4/8 equality; c-aware projection/MoE evidence after native full-attention/post-attention drift is resolved. |
| KV pool | Chunked grow/shrink, append-only block ids, current admission capacity, prefix refcounts, and copy-on-write forks are implemented in host tests. | `hipengine/kvcache/pool.py:ChunkedKVPool`, `admit_with_shared_prefix`, `fork_copy_on_write`; `pytest -q tests/test_kvcache_policy.py -q`. | Device/runtime retained equality and perf, not the host allocator contract. |
| Prefix / radix cache | `RadixCache` indexes block-aligned token prefixes; server exposes prefix-cache mode and `n>1` lowering uses distinct row seeds/request ids. | `hipengine/kvcache/radix.py:RadixCache`; `hipengine/server/api.py`; kvcache/server tests. | Broader retained coverage and future DMS/KVTC policy work; no flat prefix-LRU peer path. |
| Observability | Completion artifacts and `/metrics` include request/pool counters; graph-bucket stats exist for scheduler observability. | `CompletedRequest.to_json_dict`, `KVPoolStats.to_json_dict`, `GraphBucketCache`, `_render_prometheus_metrics`; server/scheduler tests. | Accepted retained rows still need captured profiler summaries and benchmark rollup updates. |

DMS / compact KV serving status lives in [`KVCACHE.md`](KVCACHE.md) and is not
mirrored in this matrix.

## Engine-loop contract

The engine loop is the single owner of admission, work scheduling, KV
allocation, sampling, completion, reclaim, and pool resize in the host-side
scheduler contract. The FastAPI adapter now keeps only a short `session_lock`
in `hipengine/server/api.py` for model/session preparation; request-lifetime
`engine.generate(...)` calls are serialized by the batcher worker rather than a
coarse server lock, which is an adapter safety rail and not evidence that the C4
loop scaffolding is absent.

### C1 lock-scope audit

Current server lock scope after the C4/C5 host work:

- Startup eager-load holds `session_lock` only around LLM construction,
  resident-session preparation, context-budget validation, and capacity logging;
  the warmup `engine.generate(...)` call runs after the lock is released.
- Non-streaming requests call `generate(...)`, which holds `session_lock` only
  for lazy LLM construction, resident-context preparation, sampling
  construction, optional `n>1` row-seed lowering, and context-budget validation,
  then enqueue into `_GenerationBatcher`.
- `_GenerationBatcher._run_group(...)` no longer accepts or holds a generation
  lock. It owns an event-loop queue/worker and calls
  `engine.generate(tuple(prompts), sampling)` outside `session_lock`, so no
  request lifetime is covered by a server lock.
- Streaming chat holds `session_lock` for preparation only, then routes through
  `_GenerationBatcher.stream(...)`. The batcher owns a per-request queue, so
  streaming no longer directly bypasses the batcher through `engine.stream(...)`.

The remaining native-throughput blocker is correctness/performance, not host API
shape: server endpoints can be thinned further only after the resident path has
native c>N generated-token equality, retained execution metadata, and accepted
benchmark evidence. Until then, the batcher worker serializes grouped calls to
avoid concurrent mutation of shared KV, linear-attention recurrent state, hidden
buffers, scratch, and sampler state without holding a request-lifetime server
lock.

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
   `append_value_mismatch=0`, `attn_batch_vs_c1_max_abs=0.0`, and
   `0.0 <= attn_batch_vs_numpy_max_abs <= 2e-5`.
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
      formerly protected by `generation_lock`; document which session state is
      still non-reentrant. Acceptance: a focused test or review note proves the
      lock is narrow enough for C1 and names the exact blocker for native
      request-level concurrency. Evidence: §C1 lock-scope audit plus
      `hipengine/server/api.py` code refs and `pytest -q tests/test_server_api.py -q`.
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
- [x] **C2.2 hidden-state bisection harness.** Add a HIP-guarded diagnostic
      that compares c=2 native vs independent c=1 hidden tensors after each
      layer and optionally after sub-stages (attention, selected MoE, shared
      expert, combine, LM head). Acceptance: the harness can reproduce the
      current L40 c=2 512/128 divergence earlier than generated-token idx 87.
      Evidence: `scripts/qwen35_batch_hidden_bisect.py` plus
      `python3 scripts/qwen35_batch_hidden_bisect.py --fixture /tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json --prompt-length 512 --batch-size 2 --decode-tokens 16 --max-layers 8 --layer-limits 8 --max-sequence-length 1024 --json /tmp/hipengine-hidden-bisect-L8-512-16.json`
      emitted `status=mismatch_found`, first hidden mismatch at generated
      index 1, and first token mismatch at row 0 index 13 (< 87). CPU guard
      coverage: `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [ ] **C2.3 selected-MoE lane-map fix.** Root-cause
      `/tmp/hipengine-retained/eq-L8-selectedmoe.json`; fix token-row → routed
      lane mapping or grouped metadata so selected MoE is hidden-equality green
      at c=2. Acceptance: C2.2 reports selected-MoE hidden equality for the
      failing fixture and generated-token equality progresses past the old
      idx-13 failure. Progress: decode batch rows now use grouped compact MoE
      scratch for `tokens>1` instead of selected-MoE c1 wrappers, and
      decode-execution metadata reports `moe_decode_path`/`moe_decode_rows`,
      grouped-compact and selected-c1 fallback layer counts, and per-layer
      decode traces so retained gates and diagnostics can reject stale
      selected-c1 MoE paths; CPU coverage now locks the
      token-major routed-lane → token-row, selected-expert → expert-group,
      sorted routing-weight, weighted selected-branch accumulation, and
      lane-to-sorted-row helper semantics used by grouped MoE combine metadata,
      but
      `/tmp/hipengine-hidden-bisect-L1-8-512-1-grouped.json` still reports the
      first hidden mismatch at layer-limit 6 (row 0, generated index 1), and the
      old row-0 token idx-13 mismatch remains. Latest traced diagnostic
      `/tmp/hipengine-hidden-bisect-L6-512-1-traced.json` still emits
      `status=mismatch_found` at layer-limit 6 (row 0, generated index 1,
      `max_abs=0.00146484375`, no token mismatch) and now copies the failing
      step's `batch_decode_execution.layer_executions` into
      `correctness.first_hidden_mismatch`; that trace shows native batch
      full-attention at layer 3 and grouped-compact MoE on all six decoded
      layers, keeping C2.3 focused on the linear-attention+MoE layer rather than
      sampler, selected-c1 fallback, or split-K paths. The follow-up artifact
      `/tmp/hipengine-hidden-bisect-L6-512-1-maxdim.json` uses the richer hidden
      comparison schema and localizes the top row-0 difference to hidden dim
      1269 (`batch=0.8564453125`, `c1=0.85498046875`, signed diff
      `+0.00146484375`), giving the next lane-map fix a stable coordinate to
      inspect across selected-MoE/grouped-MoE substage traces. The latest
      top-diff artifact `/tmp/hipengine-hidden-bisect-L6-512-1-topdiff.json`
      adds `elements_over_atol=1` and the top eight hidden-coordinate diffs to
      each row comparison; row 0's only over-tolerance element is still dim 1269
      while row 1 remains within tolerance despite bit-level drift. A paired
      L5/L6 run at `/tmp/hipengine-hidden-bisect-L5-L6-512-1-topdiff.json`
      confirms L5 hidden/token equality still passes (`max_abs≤0.00048828125`,
      `elements_over_atol=0` for both rows) and L6 is the first failing layer,
      with the same row-0 dim-1269 single over-tolerance element. The refreshed
      transition artifact `/tmp/hipengine-hidden-bisect-L5-L6-512-1-transition.json`
      records this as `correctness.first_failing_layer_transition` with
      `previous_green_layer_limit=5`, `failing_layer_limit=6`,
      `adjacent_layer_limits=true`, and the embedded first-hidden-mismatch plus
      native decode trace. The refreshed row-scoped artifact
      `/tmp/hipengine-hidden-bisect-L5-L6-512-1-transition-rows.json` also tags
      the transition as `failure_modes=["hidden"]`,
      `hidden_failure_rows=[0]`, and `token_failure_rows=[]`, keeping the
      current C2.3 target to a row-0 hidden-state divergence before the longer
      decode token mismatch. The execution-scoped refresh
      `/tmp/hipengine-hidden-bisect-L5-L6-512-1-transition-exec.json` lifts the
      failing and previous-green layer execution records into the transition;
      both are `linear_attention` layers with `moe_decode_path=grouped_compact`
      and `full_attention_decode_path=not_applicable`, isolating the first red
      boundary to the layer-5 linear-attention/grouped-MoE decode output. The
      focus refresh `/tmp/hipengine-hidden-bisect-L5-L6-512-1-focus.json` adds
      `first_hidden_mismatch_focus` for row 0 / dim 1269; that coordinate is
      the failing layer's top diff but is not present in the previous-green L5
      row-0 top-diff list, narrowing the jump to the L6 layer output. The
      row-focus refresh `/tmp/hipengine-hidden-bisect-L5-L6-512-1-rowfocus.json`
      adds per-row focus lists for that coordinate: at L6, row 0 is the only
      hidden-failing row (`abs_diff=0.00146484375`, over tolerance) while row 1
      shares dim 1269 as its top diff but remains within tolerance
      (`abs_diff=0.00048828125`); at L5, neither row has dim 1269 in its
      top-diff list. A selected-c1 MoE probe at
      `/tmp/hipengine-hidden-bisect-L5-L6-512-1-selected-c1-moe.json` forces
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE=1` and records
      `moe_decode_path=selected_c1_forced`/`moe_grouped_compact_layers=0`; the
      L6 row-0 hidden mismatch persists at dim 1269 (`max_abs=0.001953125`, no
      token mismatch), so the reduced failure is not cleared by bypassing the
      grouped-compact WMMA MoE path. The per-row-linear probe at
      `/tmp/hipengine-hidden-bisect-L5-L6-512-1-per-row-linear.json` forces
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR=1`; it records
      `linear_attention_decode_path=selected_c1_per_row_fallback`,
      `native_caware_decode=false`, and the same L6 row-0 dim-1269 hidden
      failure (`max_abs=0.00146484375`, no token mismatch), so the reduced
      failure is also not cleared by replacing the batch linear-attention
      segment path with per-row c=1 linear layers. The per-row full-attention
      probe at `/tmp/hipengine-hidden-bisect-L5-L6-512-1-per-row-full-attn.json`
      forces `HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE=0` through
      `--batch-decode-full-attn-path per_row`; it records
      `full_attention_decode_path=per_row_context_fallback`,
      `native_full_attention_layers=0`, and again preserves the L6 row-0
      dim-1269 hidden failure (`max_abs=0.00146484375`, no token mismatch), so
      the reduced failure is not cleared by replacing the native batch
      full-attention layer either. Two tolerance probes then bracket the scale
      of the drift without changing the retained correctness gate:
      `/tmp/hipengine-hidden-bisect-L5-L6-512-1-atol2e-3.json` passes L5/L6 at
      `hidden_atol=0.002` (`status=eq_ok`, no token mismatch, L6 row-0 dim
      1269 still the top diff at `0.00146484375`), while
      `/tmp/hipengine-hidden-bisect-L1-8-512-1-atol2e-3.json` first fails at
      layer-limit 8 on the same row/dim after the next full-attention layer
      (`max_abs=0.002197265625`, previous-green layer-limit 7 has
      `max_abs=0.001953125`, no token mismatch). That means the strict 1e-3 L6
      report is a small BF16-scale drift, but it monotonically grows enough by
      L8 to remain a real hidden-state blocker before full 40-layer equality.
      A selected-coordinate trace at
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-focus1269.json`
      adds `--focus-hidden-flat-index 1269` so every row/layer records that
      coordinate even when it is outside the top-diff list: row 0 is exact at
      L5, jumps to `0.001953125` at L6 and remains there through L7, then grows
      to `0.00244140625` at L8; row 1 stays at or below `0.0009765625` and no
      token mismatch appears. The all-per-row variant
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-all-per-row-focus1269.json`
      combines selected-c1 MoE, per-row linear attention, and per-row
      full-attention fallbacks (`moe_grouped_compact_layers=0`,
      `moe_selected_c1_fallback_layers=8`); it still fails first at L8 on row 0
      dim 1269 (`max_abs=0.002197265625`, no token mismatch). The prefill-aware
      refresh
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-all-per-row-prefill-focus1269.json`
      adds final-prefill hidden comparisons for the same run; compact prefill
      vs independent c=1 final hidden passes at every L5-L8 limit under
      `hidden_atol=0.002` (L8 row-0 dim 1269 is only `0.0009765625`), while
      the first decode step still reaches `0.002197265625`. The linear-state
      refresh
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-all-per-row-state-focus1269.json`
      compares compact-prefill slot linear states with independent c=1 states
      before decode; final hidden remains green, but `prefill_linear_state_passed=false`
      for every L5-L8 limit at `state_atol=1e-6` (for example L6 layer-5
      recurrent state `max_abs=0.0072229355573654175`, conv state
      `max_abs=0.0078125`). The row-scoped refresh
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-all-per-row-state-rows-focus1269.json`
      adds per-row state maxima: at L6/layer 5, row 0 is the larger recurrent
      offender (`0.0072229355573654175` vs row 1 `0.0025315284729003906`),
      while conv is large for both rows (`0.0078125`). A per-segment linear
      prefill diagnostic at
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-perseg-prefill-all-per-row-focus1269.json`
      forces `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR=1` and
      records `linear_attention_prefill_path=per_segment`; it still leaves
      `prefill_linear_state_passed=false` and the same L8 row-0 dim-1269 decode
      failure (`max_abs=0.002197265625`, no token mismatch). This keeps the
      live fix target on packed-prefill linear-state materialization / slot-state
      contents, especially row-0 recurrent state, not final prefill hidden,
      segment state-index ordering alone, or any single native batch decode
      subpath. A pre-linear-input trace at
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-all-per-row-inputs-focus1269.json`
      shows the state drift is input-driven before those later linear layers:
      `prefill_linear_input_passed=false` from L5 onward, with the first bad
      layer-4 input already after full-attention layer 3 (row 0
      `max_abs=0.0059814453125` at prompt token 10 dim 751; row 1
      `0.00305938720703125` at token 176 dim 1237). Layer-5/layer-6 inputs
      then grow (`0.00951385498046875` and `0.01171875` row-0 maxima), while
      final prefill hidden still passes and the L8 decode failure remains row 0
      dim 1269 (`0.002197265625`, no token mismatch). A follow-up full-attn
      prefill diagnostic at
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-perseg-fullprefill-all-per-row-inputs-focus1269.json`
      forces `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN=1`
      with local block tables, per-slot caches, and c=1-style AOTriton prefill;
      that run clears the L5-L8 hidden/token gate (`status=eq_ok`) and clears
      `prefill_linear_input_passed` for every L5-L8 limit. L5 still reports
      state diffs under the strict `1e-6` state probe, but L6-L8 state summaries
      also pass. The retained-path fix then switches packed-varlen full-attention
      prefill to the AOTriton compact-varlen attention kernel using contiguous
      scratch K/V plus per-segment max sequence lengths; the follow-up retained
      artifact
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-packed-aotriton-all-per-row-inputs-focus1269.json`
      records `full_attention_prefill_path=packed_varlen_aotriton` with no
      forced blockers, `status=eq_ok`, and green `prefill_linear_input`,
      `prefill_linear_state`, hidden, and token gates for every L5-L8 limit.
      The longer L8/16 refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-packed-aotriton-focus1269.json`
      confirms the old row-0 generated-token idx-13 mismatch is gone
      (`token_passed=true`) but still finds a multi-step hidden drift at decode
      step 6 / generated index 7 (`row 0`, dim 1269,
      `max_abs=0.02685546875`) with native c-aware decode. An all-per-row
      decode variant at
      `/tmp/hipengine-hidden-bisect-L8-512-16-packed-aotriton-all-per-row-focus1269.json`
      also keeps tokens green but shifts the first hidden mismatch to decode
      step 11 / generated index 12 (`row 0`, dim 1543,
      `max_abs=0.010440826416015625`), so C2.3 is not closed yet. A decode-state
      trace refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-packed-aotriton-decode-states-focus1269.json`
      adds per-step compact-vs-c1 conv/recurrent state summaries; retained
      native c-aware decode still fails hidden equality at step 11 (`row 0`,
      dim 1167, `max_abs=0.0157470703125`) and reports strict state drift from
      step 0. The matching all-per-row trace
      `/tmp/hipengine-hidden-bisect-L8-512-16-packed-aotriton-all-per-row-decode-states-focus1269.json`
      is green (`status=eq_ok`, `decode_linear_state_passed=true`). A one-native-
      subpath sweep then narrows the retained target further: native grouped MoE
      alone stays green at
      `/tmp/hipengine-hidden-bisect-L8-512-16-native-moe-only-decode-states-focus1269.json`,
      while native linear-attention decode alone fails at step 2 / generated
      index 3 (`row 1`, dim 1073, `max_abs=0.00799560546875`) with strict
      decode-state drift from step 0, and native full-attention decode alone
      fails at step 6 / generated index 7 (`row 0`, dim 1269,
      `max_abs=0.02734375`). A native-linear input-trace refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-native-linear-only-decode-inputs-focus1269.json`
      shows `decode_linear_input_passed=false` only later (first input mismatch:
      step 12 / generated index 13, layer 6, row 0 dim 585,
      `max_abs=0.00811767578125`); inputs are still green through the first
      hidden mismatch at step 2, while strict state drift starts at step 0. The
      matching all-per-row input trace
      `/tmp/hipengine-hidden-bisect-L8-512-16-all-per-row-decode-inputs-focus1269.json`
      is green. Decode execution metadata now records
      `linear_attention_segment_metadata` (`cu_seqlens` and `state_indices`),
      and the latest row-1/segment probe
      `/tmp/hipengine-hidden-bisect-L8-512-16-c1-batch-segments-decode-metadata-focus1269.json`
      still fails hidden equality with `rows=1`, `state_indices=[0]`, and
      `decode_linear_input_passed=true` (first hidden mismatch: step 11 / dim
      1543, `max_abs=0.010478973388671875`). The matching rows=1 forced-c1
      linear wrapper at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c1-forced-linear-per-row-decode-focus1269.json`
      is green (`status=eq_ok`, hidden/token/state/input gates all true),
      confirming that the divergence is inside the native segment linear-decode
      wrapper/kernel path rather than row setup, full-attention fallback, or
      later selected-c1 MoE. The retained singleton bridge now defaults rows=1
      batch decode through the specialized c1 linear kernel; the refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c1-single-row-c1-linear-focus1269.json`
      is also green with `linear_attention_decode_path=single_row_c1` and the
      same `state_indices=[0]`. A grouped-MoE + native-linear c=2 probe at
      `/tmp/hipengine-hidden-bisect-L8-512-16-native-linear-grouped-decode-metadata-focus1269.json`
      records `state_indices=[0,1]` and fails later at the old row-0 idx-13
      token boundary. A post-singleton c=2 control refresh shows the correctness
      bridge boundaries explicitly: c1-linear + per-row full attention at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-per-row-full-per-row-after-singleton-focus1269.json`
      is green (`status=eq_ok`, all decode input/state/hidden/token gates true),
      while c1-linear + native full attention at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-per-row-full-native-after-singleton-focus1269.json`
      still fails at decode step 6 / generated index 7 (`row 0`, dim 1269,
      `max_abs=0.027587890625`, tokens still green). This is not only a
      full-layer grouped-MoE artifact: the earlier selected-c1/full-only control
      `/tmp/hipengine-hidden-bisect-L8-512-16-native-full-only-decode-states-focus1269.json`
      also failed at step 6 / row 0 dim 1269 (`max_abs=0.02734375`) with
      `token_passed=true`. The reduced prefill drift is fixed, all-per-row
      fallback, grouped compact MoE, selected-c1 full-layer MoE, singleton row setup,
      and segment state-index mapping are not the blocker at this shape; the
      next C2.3 targets are c>1 native full-attention decode and c>1 native
      linear segment decode, with the per-row linear/full fallbacks serving only
      as non-retained correctness controls until native paths pass. The
      full-attention I/O trace refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-full-attn-io-trace-focus1269.json`
      adds per-step input/output summaries from
      `scripts/qwen35_batch_hidden_bisect.py` and shows the first native-full
      mismatch at decode step 6 / generated index 7, layer 3 `output` (`row 0`,
      `max_abs=0.008148193359375`, tokens still green), before the layer-limit
      hidden max reaches row 0 dim 1269 at `0.027587890625`. The substage
      trace refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-full-attn-substages-focus1269.json`
      extends that schema with `attn_input`, `gated_attn`, and `o_proj` stage
      gates plus a compact `first_mismatch`; its first substage failure is
      earlier, at decode step 0 / generated index 1, layer 7 `attn_input`
      (`row 0`, dim 1269, `max_abs=0.015625`). The matching layer-3-only
      control at
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-full-attn-substages-focus1269.json`
      is green across all full-attention substages. The compact linear-first
      refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-first-mismatch-focus1269.json`
      adds `decode_linear_inputs.first_mismatch` and
      `decode_linear_states.first_mismatch`; it shows the strict state trace
      first diverges earlier at decode step 0, layer 4 `conv` row 0
      (`max_abs=0.0078125`), while visible linear input drift first appears at
      decode step 6, layer 4 row 0 dim 1504 (`max_abs=0.008148193359375`). The
      worst-diff refresh at
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-worst-diff-focus1269.json`
      keeps the layer-3-only run green but exposes its largest native-full
      subthreshold drift at layer 3 `output` row 0 dim 1269
      (`max_abs=0.00048828125`). The matching L8 artifact
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-worst-diff-focus1269.json`
      shows worst drift later in layer 7 `attn_input` (`max_abs=0.3984375`) and
      layer-4 `conv` state (`max_abs=0.390625`). Zero-tolerance controls then
      bracket the bit-exactness issue: before the first full-attention layer,
      `/tmp/hipengine-hidden-bisect-L3-512-16-c2-strict-before-full-focus1269.json`
      is exact (`status=eq_ok` with `hidden_atol=0`), and the all-per-row L4
      control
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-strict-all-per-row-focus1269.json`
      is also exact, while native-full L4
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-strict-native-full-focus1269.json`
      fails immediately at generated index 1 (`token_passed=true`, row 0 dim
      1269, `max_abs=0.00048828125`). The strict transition artifact
      `/tmp/hipengine-hidden-bisect-L3-L4-512-16-c2-strict-transition-focus1269.json`
      now records `previous_green_layer_limit=3`, `adjacent_layer_limits=true`,
      and compact trace summaries. The gate/context transition refresh at
      `/tmp/hipengine-hidden-bisect-L3-L4-512-16-c2-gate-context-transition-focus1269.json`
      shows `input`, `attn_input`, and `gate` are exact, while `attn_context`
      is the first mismatching substage at layer 3 (`max_abs=2.1604321002960205`,
      all 4096 row-0 context elements over zero tolerance). The final layer
      output drift is still row 0 dim 1269 (`max_abs=0.00048828125`). The
      metadata refresh at
      `/tmp/hipengine-hidden-bisect-L3-L4-512-16-c2-metadata-transition-focus1269.json`
      records the failing runtime layer with `positions=[512,512]`,
      `decode_live_counts=[513,513]`, `block_table_rows=[[0,1,2,3],[0,1,2,3]]`,
      and `attn_context_trace_source=attention_scratch.query_raw`. Matching
      model-shape primitive controls now fill paged rows across block boundaries
      and cover the 16-Q/2-KV/head-dim-256 path:
      `/tmp/hipengine-multiloop-c2-modelshape-primitive-correctness.json`
      (`context_lens=513,512`, with dense-c1 comparison) and
      `/tmp/hipengine-multiloop-c2-modelshape-primitive-correctness-513x2.json`
      (`context_lens=513,513`) both pass with append mismatches zero,
      `attn_batch_vs_c1_max_abs=0.0`, and NumPy max abs
      `1.4901161193847656e-08`; the dense short-context c1 comparison is also
      tolerance-green (`attn_batch_vs_dense_c1_max_abs=1.862645149230957e-08`).
      The KV-tail trace refresh at
      `/tmp/hipengine-hidden-bisect-L3-L4-512-16-c2-kv-tail-transition-focus1269.json`
      added BF16 cache samples for `first`, `previous`, and `current` tokens;
      the multipoint refresh at
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-kv-multipoint-focus1269.json`
      now samples `first`, `page0_last`, `page1_first`, `previous`, and `current`
      positions (`[0,255,256,511,512]` for the failing step). The failing layer's
      `decode_full_kv_samples` still passes at zero tolerance (`bit_mismatch=0`,
      worst `max_abs=0.0`) while `attn_context` still fails.
      The query refresh at
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-query-focus1269.json` also showed
      the FP32 `query` launch input was exact (`max_abs=0.0`) while
      `attn_context` remained the first mismatching substage (`max_abs=2.1604321002960205`).
      The context-oracle refresh found the launch-path asymmetry: c1 slot spans
      advertised `max_live_count=max_sequence_length=1024`, which routed the
      513-token c1 reference through split-K while native c=2 used the live
      513-token batch context path. `hipengine/runtime/qwen35_paro_runner.py`
      now keeps host `position_arr`/`context_arr` current and `_slot_full_spans`
      uses those live counts. Evidence:
      `/tmp/hipengine-hidden-bisect-L4-512-1-c2-context-oracle-live-max-focus1269.json`
      has exact input/query and NumPy-oracle-green context (`batch_context_vs_numpy`
      `5.960464477539062e-07`, `c1_context_vs_numpy` `2.384185791015625e-06`,
      `batch_numpy_vs_c1_numpy=0.0`), and
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-context-oracle-live-max-atol1e-3-focus1269.json`
      is `status=eq_ok` with `token_passed=true`, `hidden_passed=true`, and
      `decode_full_context_oracle.passed=true`. L8 still remains open because
      the later linear-attention state drift reaches layer 7 context (`batch_numpy_vs_c1_numpy`
      `0.5023813247680664` at decode step 10 in
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-per-row-full-native-live-max-atol1e-3-focus1269.json`).
      The tolerance-transition refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-tolerance-transition-focus1269.json`
      makes this distinction explicit: top-level `first_hidden_bit_drift` is the
      L4 strict-only native-full drift (`passed_under_atol=true`, `max_abs=0.00048828125`),
      while `first_tolerance_hidden_mismatch` and
      `first_failing_layer_transition.hidden_mismatch_kind=over_atol` point to
      L8 decode step 6 / row 0 dim 1269 (`max_abs=0.027587890625`). The transition
      now includes `decode_full_context_oracle` in its trace summaries. The
      linear-state focus refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-linear-state-focus1269.json`
      adds `first_hidden_mismatch_linear_state_focus`; at the first over-tolerance
      hidden mismatch, layers 0-2 conv/recurrent state diffs are zero while layer 4
      has row-0 `conv max_abs=0.390625` and `recurrent max_abs=0.021587848663330078`.
      The first-state focus refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-first-linear-state-focus1269.json`
      adds `first_linear_state_mismatch_focus`; it shows the earliest state drift
      in the L8 failing run is decode step 0 / layer 0 `recurrent` row 0
      (`max_abs=0.001646714168600738`) while that step's hidden row and layer-0
      decode input are still tolerance/exact green. The state-focus refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus2e-3-focus1269.json`
      adds `first_linear_state_mismatch_over_focus_atol`; with `state_atol=0`
      and `state_focus_atol=0.002`, the first focus-threshold state drift is
      decode step 0 / layer 4 `conv` row 0 (`max_abs=0.0078125`) while that
      step's hidden row and layer-4 decode input remain under the hidden
      tolerance (`max_abs=0.00048828125`). The history refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus-history2e-3-focus1269.json`
      adds `first_linear_state_mismatch_over_focus_atol_history`; the layer-4
      conv row remains over the 0.002 focus threshold from step 0, and at the
      first over-tolerance hidden step (decode step 6) the same row jumps to
      `state max_abs=0.390625` while `decode_linear_input.max_abs=0.0081787109375`
      and hidden row dim 1269 reaches `0.027587890625`. The same-index refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus-same-index2e-3-focus1269.json`
      tracks the original layer-4 conv index `[64, 3]`; it is only `0.015625` at
      step 6 while the row max moved to `[4852, 3]`, so the amplification is not
      one persistent component. The delta refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus-delta2e-3-focus1269.json`
      adds previous-state and update-delta comparisons; at step 6 the previous
      state row is still tiny (`max_abs=0.0078125`) but the update delta is already
      large (`max_abs=0.390625`, top index `[4852, 3]`). The execution refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus-exec2e-3-focus1269.json`
      lifts the focused step/layer execution into the history: step 6 is layer 4
      `linear_attention` over rows `[0,1]` / slots `[0,1]` with
      `linear_attention_decode_path=selected_c1_per_row_fallback` and
      `native_caware_decode=false`. The row-map refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus-rowmap2e-3-focus1269.json`
      records `linear_attention_row_state_map=[{row:0,slot:0,state_index:0},{row:1,slot:1,state_index:1}]`
      and matching `state_indices=[0,1]`, so the step-6 drift is not a row/slot
      metadata swap. The producer refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-layer4-input-producer-focus1269.json`
      ties that layer-4 input drift to the preceding layer-3 full-attention block:
      stages `input`, `attn_input`, `gate`, `query`, `attn_context`, `gated_attn`,
      and `o_proj` are green, while final `output` is over tolerance
      (`max_abs=0.0081787109375`, dim 1504) under `native_batch` full-attention
      plus `grouped_compact` MoE. The FP16 output-delta refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-output-minus-oproj-fp16-focus1269.json`
      fixes the trace-value conversion and records `output_minus_o_proj` as the first
      bad delta too (`max_abs=0.0081787109375`, dim 1504), while `o_proj` itself
      remains green (`max_abs=6.103515625e-05`). The post-attention component refresh
      at `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-post-attn-components-focus1269.json`
      traces `residual` and `mlp_input`: layer-3 `residual` is green
      (`max_abs=0.000244140625`, no elements over tolerance), but `mlp_input` is
      the first bad stage (`max_abs=0.00390625`, dim 100). The RMSNorm-oracle refresh
      at `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-rmsnorm-oracle-focus1269.json`
      infers the post-attention RMSNorm transform from the c=1 residual/mlp pair;
      applying it to the c=2 residual leaves only two over-tolerance FP16-ulp
      differences (`max_abs=0.001953125`, dims 135/2012), so the residual's small
      green drift explains much of `mlp_input` but not the last one-ulp gap. The
      per-row full-attention control at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-perrow-fullattn-focus1269.json`
      disables both native linear segments and native full-attention; it still
      reports `status=mismatch_found`, but the first hidden over-atol case moves to
      layer-limit 4 / decode step 1 / row 1 with only one element over tolerance
      (`max_abs=0.00146484375` at focus dim 1269) under `per_row_context_fallback`.
      This means native full-attention/post-attention is not the only c>N equality
      source; small FP16 state drift from the per-row fallback can also cross the
      strict `hidden_atol=0.001` gate. The tolerance-sensitivity refresh separates
      those regimes: the all-per-row control passes generated-token and hidden
      equality at `hidden_atol=0.004` in
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-perrow-fullattn-atol4e-3-focus1269.json`,
      while native full-attention is still over tolerance at the same threshold in
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-native-fullattn-atol4e-3-focus1269.json`
      (`max_abs=0.027587890625`, 345 elements over). Token/hidden classification:
      the `atol=0.002` all-per-row control is hidden-only fail (`token_passed=true`,
      `first_token_mismatch=null`); the `atol=0.004` all-per-row control is
      token+hidden pass; and the `atol=0.004` native-full control is again
      hidden-only fail (`token_passed=true`, `first_token_mismatch=null`), now
      also emitted as top-level `correctness.failure_modes=["hidden"]` in the
      native-full artifact. The selected-c1 MoE control at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-native-fullattn-selected-c1-moe-atol4e-3-focus1269.json`
      keeps native full-attention but bypasses grouped-compact MoE; it remains
      hidden-only red at L8 (`max_abs=0.02734375`, 346 elements over, tokens green),
      so grouped-compact MoE is not the source of the large native-full drift.
      The row-count refresh of that same artifact adds per-layer and top-level
      row-failure summaries: L4 has `failure_modes=[]`, `hidden_failure_rows=[]`
      but strict bit drift on both rows, while L8 has `failure_modes=["hidden"]`,
      `hidden_failure_rows=[0,1]`, and `token_failure_rows=[]`; top-level
      `correctness.row_failure_summary` matches hidden rows `[0,1]`, strict rows
      `[0,1]`, and token rows `[]`. The diagnostic schema now also emits
      `decode_full_attention.stage_failure_summary` with per-stage failing rows
      and a compact `first_failure` record,
      `decode_full_context_oracle.comparison_failure_summary` with per-comparison
      row/failure rollups, and the top-level
      `correctness.decode_full_context_oracle_failure_summary` aggregate, so
      native-full artifacts can tell attention-context,
      `mlp_input`/post-attention, and final hidden/token failures apart; CPU
      coverage lives in `test_hidden_bisect_summary_embeds_batch_decode_execution_trace`.
      A new diagnostic switch,
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_POST_ATTN=1` (or hidden-bisect
      `--batch-decode-post-attn-path per_row`), routes only the c>N
      full-attention decode post-attention add/RMSNorm boundary through token-1
      row kernels, labels the decode as a diagnostic fallback, and is covered by
      `test_qwen35_resident_run_layers_batch_decode_can_force_per_row_post_attention_probe`.
      The first focused artifact,
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-perrow-postattn-atol4e-3-focus1269.json`,
      remains hidden-only red (`token_passed=true`, `failure_modes=["hidden"]`):
      L4 final hidden/token stays green but full-attention substage drift is visible,
      and L8 first fails at decode step 2 / row 1 / dim 1073 (`max_abs=0.008148193359375`).
      Therefore the batch post-attention add/RMSNorm boundary is not the sole
      native-full blocker. A stricter core-isolation control,
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-native-full-core-perrow-linear-postattn-selected-c1-atol4e-3-focus1269.json`,
      keeps native full-attention decode but forces per-row linear, selected-c1 MoE,
      and per-row post-attention; it still fails hidden-only at L8 decode step 6 / row 0
      / dim 1269 (`max_abs=0.02734375`) while L4 stays green. The refreshed
      top-level context-oracle rollup shows L8 only fails `batch_numpy_vs_c1_numpy`
      (`first_failure`: decode step 0 / row 0 / context dim 2812,
      `max_abs=0.00811624526977539`; worst at step 10, `max_abs=0.4993577003479004`),
      while `batch_context_vs_numpy` and `c1_context_vs_numpy` pass. The added
      top-level full-attention stage rollup now shows L8 first fails at `attn_input`
      (decode step 0 / row 0 / dim 1269, `max_abs=0.015625`); raw stage `input`
      first fails only later at decode step 6, and `attn_context` first fails as
      fp32 at step 0 / dim 2812 (`max_abs=0.008107900619506836`). That means
      the context kernel matches its own oracle; the next C2.3 target is the
      layer-7 attention-input RMSNorm/QKV preparation or state feeding, not raw
      hidden input copy or softmax context math. A diagnostic
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_INPUT=1` control
      (hidden-bisect `--batch-decode-attn-input-path per_row`) now forces just
      the full-attention input RMSNorm through token-1 row kernels. The refreshed
      probe,
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-native-full-core-perrow-attninput-linear-postattn-selected-c1-atol4e-3-focus1269.json`,
      remains hidden-only red: L8 still first fails final hidden at decode step 6
      / row 0 / dim 1269 (`max_abs=0.02734375`), and the post-layer `attn_input`
      trace still first fails at L8 decode step 0 / row 0 / dim 1269
      (`max_abs=0.015625`). The immediate `attn_input_pre_qkv` plus new
      `attn_input_after_rotate`, `attn_input_after_project`, and
      `attn_input_after_prepare` traces all pass at L4 and L8, so the input
      RMSNorm/rotate/project/prepare path is not overwriting `attn_input`; the
      existing post-layer `attn_input` trace is polluted after those stages and
      should not drive the fix. C1 input-scratch tracing is now symmetric with
      native c=2 and includes `attn_input_pre_qkv`, `attn_input_after_rotate`,
      `attn_input_after_project`, and `attn_input_after_prepare`. The refreshed
      probe shows L4 producer stages now pass exactly, so the earlier L4
      `q_proj_key_after_project` signal was missing-c1-trace noise. The first
      full-attention stage drift is now L8 decode step 0 / row 0 at
      `attn_input_pre_qkv` (`max_abs=0.015625`, dim 1269), propagating through
      Q/K preparation (`q_proj_key_after_project` `max_abs=0.0078125`,
      `query_after_prepare` `max_abs=0.005970478057861328`, `key_after_prepare`
      `max_abs=0.005676984786987305`). The hidden-bisect context oracle now also
      stores per-token CRC32 hashes for the full BF16 K/V prefix and emits
      `correctness.decode_full_context_kv_prefix_failure_summary`, covered by
      `test_hidden_bisect_summary_embeds_batch_decode_execution_trace`. It now
      additionally snapshots post-prefill full-KV prefix hashes before any decode
      write and emits `correctness.prefill_full_kv_prefix_failure_summary`, also
      covered by that CPU test. The refreshed prefill-aware probe at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-prefill-kv-prefix-native-full-core-atol4e-3-focus1269.json`
      is still hidden-only red, has green post-prefill K/V prefix hashes in that
      run, and localizes the decode-time prefix/sample failure to L8 step 0 /
      layer 7 / row 0 current token 512. An L4-only repeat
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-prefill-kv-prefix-repeat-atol4e-3-focus1269.json`
      caught a pre-decode prompt-tail hash failure at layer 3 / row 0 token 500
      that the decode prefix then inherited; a second L4 repeat was green, so the
      immediate target is to make the compact-prefill K/V hash probe deterministic
      and audit prompt-tail/current-token slot contents before changing paged-KV
      writer code. Do not re-open context softmax math, row setup, native linear
      segment metadata, output trace/copy semantics, or grouped MoE output yet.
- [ ] **C2.4 full c=2 BF16 512/128 equality.** Re-run the full 40-layer c=2
      512/128 retained protocol with `serial_lm_head` default and no serial
      decode bridge. Acceptance: generated-token equality vs two c=1 sessions
      passes; if timing is still not retained, artifact is `blocked` for a
      non-correctness reason.
- [ ] **C2.5 c=4/c=8 BF16 equality.** Extend the same gate to c=4 and c=8.
      Acceptance: generated-token equality passes for both shapes, with
      aggregate/per-request scaling fields recorded even if not yet optimized.
      Progress: primitive GPU correctness now has a c=4 artifact at
      `/tmp/hipengine-multiloop-c4-correctness.json` (`append_*_mismatch=0`,
      `attn_batch_vs_c1_max_abs=0.0`, passed). This does not close C2.5 because
      generated-token equality vs independent c=1 for c=4/c=8 is still missing.
- [x] **C2.6 slot-validation and long-context fallback guards.** Add CPU
      structural tests for invalid slot orders/duplicates/out-of-range ids,
      INT8 KV rejection, and the current `max_context >= 1024` per-row split-K
      fallback until row-aware split-K is live. Acceptance: tests fail if the
      experimental path silently accepts unsupported shapes or routes long
      contexts through a false native c>N reducer. Evidence:
      `test_qwen35_resident_step_batch_native_rejects_invalid_sparse_slots`,
      `test_qwen35_resident_step_batch_native_rejects_int8_kv_when_experimental`,
      `test_qwen35_resident_step_batch_native_accepts_long_context_for_splitk_fallback`,
      `test_qwen35_resident_run_layers_batch_decode_uses_per_row_splitk_fallback_for_long_context`,
      and `pytest -q tests/test_qwen35_resident_batch_layout.py -q`.
- [ ] **C2.7 row-aware split-K full attention.** Make full-attention decode
      and reduction consume per-row spans for `max_context >= 1024` before any
      long-context c>N claim. Acceptance: primitive correctness plus a
      generated-token diagnostic at a long-context shape. Progress: the host
      long-context rejection is removed and split-K contexts now route through
      the existing per-row split-K fallback, with `/tmp/hipengine-hidden-bisect-L4-1024-1-splitk.json`
      showing reduced L4 1024/1 generated-token and hidden equality vs
      independent c=1. Batch execution metadata now records
      `decode_execution.full_attention_decode_path=per_row_splitk_fallback` and
      forces `native_caware_decode=false` when that fallback is used; the retained
      bench payload mirrors that execution flag, and accepted artifact schema now
      requires `decode_execution.full_attention_decode_path=native_batch` plus
      `decode_execution.native_caware_decode=true`, so artifacts cannot overclaim
      long-context native decode. The item remains open until the split-K reducer
      itself is row-aware/native c>N.
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
      `rejected_correctness` with first mismatch. Progress:
      `scripts/qwen35_batch_int8_diagnostic.py` emits the schema-checked
      blocked template `/tmp/hipengine-int8-c2-diagnostic.json` with the future
      retained-bench command and explicit blockers (`compact c>N native prefill`
      and `step_batch_native` INT8 rejection). The c-sweep planner includes this
      template behind `--include-int8`, producing an `int8_native_diagnostic`
      command in `/tmp/hipengine-c-sweep-int8-plan/summary.json`; dry-run summary
      tests now assert `options.include_int8=true`, `command_count=7`, and an
      `int8_native_diagnostic` category count. The item remains open because
      blocked-before-execution is not an accepted C3.1 terminal status.
- [x] **C3.2 per-row `KVLiveSpans` everywhere.** Audit full-attention decode,
      KV append, and storage-dtype wrappers for scalar `(block_table,
      context_len)` shortcuts. Acceptance: tests cover BF16 and INT8 per-row
      spans. Evidence: BF16/INT8 c>N `FixedPagedKVPolicy.batch_spans(...)`
      plus paged-KV-write/full-attention dispatch route checks in
      `pytest -q tests/test_kvcache_policy.py -q`.
- [x] **C3.3 linear-attention `[C]` state.** Remove c1 aliases from
      conv/recurrent state update paths and use active masks + slot ids.
      Acceptance: c=2 state fixtures compare against two c=1 references.
      Evidence: `_run_layers_batch_decode(...)` passes whole `[C]` conv/
      recurrent state plus `state_indices` slot ids, and
      `pytest -q tests/test_qwen35_resident_batch_layout.py -q` covers a
      c=2 `(slots=(0, 2))` state-index fixture against `_slot_linear_state(...)`
      c=1 reference views.
- [ ] **C3.4 c-aware projection dispatch.** Keep c=1 on GEMV/Marlin-K while
      routing c=2/4/8 to MMQ/GEMM/WMMA candidates only when they beat row-GEMV.
      Acceptance: dispatch tests prove thresholds and benchmark artifacts show
      aggregate/per-request ratios. Progress: `hipengine.dispatch.projection`
      now exposes a tested c-aware projection policy: c=1 is pinned to row-GEMV,
      c>N candidates require accepted benchmark evidence with aggregate and
      per-request speedups over row-GEMV, missing/slow/rejected evidence falls
      back to row-GEMV with explicit blockers, and
      `ProjectionDispatchEvidence.from_json_dict(...)`,
      `ProjectionDispatchCandidate.from_json_dict(...)`,
      `projection_dispatch_candidates_from_json(...)`,
      `projection_dispatch_candidates_from_artifact(...)`, and
      `plan_projection_dispatch_from_artifact(...)` schema-check retained
      artifact candidate/evidence lists before the policy can consume them;
      projection speedup evidence must reference an artifact under
      `benchmarks/results/` whose resolved target stays inside the active
      results tree and, when accepted, must beat row-GEMV on both
      aggregate and per-request ratios; accepted c>N artifact schema rejects
      malformed optional `projection_dispatch_candidates` metadata; accepted c>N
      artifact schema now requires `execution.batch_execution.projection_dispatch`
      to name an evidence-backed non-row-GEMV c-aware path whose selected
      candidate is present in `projection_dispatch_candidates` and profiler
      expected/trace/duration kernel names; retained native batch metadata records a `projection_dispatch` row-GEMV fallback
      with an explicit blocker when no c-aware projection candidate is available; and retained bench now blocks promotion before schema validation unless projection dispatch names an evidence-backed non-row-GEMV c-aware candidate present in `projection_dispatch_candidates` with matching row bounds, selection, retained artifact path, an accepted same-row evidence artifact JSON carrying self-matching `artifact_path`/`source_artifact_path` plus matching >1 aggregate/per-request row-GEMV speedup ratios, evidence, and profiler expected/trace/duration kernel names. The
      item remains open until runtime projection call sites are wired to this
      policy and retained benchmark artifacts provide the required ratios.
- [x] **C3.5 GGUF c>N template.** Port the Qwen/PARO equality template to
      GGUF Q4_K/Q5_K/Q6_K/Q8_0. Acceptance: at least one GGUF c=2 diagnostic
      reaches an unambiguous `eq_ok`, `blocked`, or `rejected_correctness`
      status with exact command. Evidence: `scripts/qwen35_batch_gguf_diagnostic.py`
      emitted `/tmp/hipengine-gguf-c2-diagnostic.json` with `status=blocked`
      and exact command `python3 scripts/qwen35_batch_gguf_diagnostic.py --fixture tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json --rows 2 --backend hip_gfx1100 --quant gguf_q4_k_m --max-new-tokens 4`; covered by
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [ ] **C3.6 native LM-head/sampler launch.** Replace the per-row
      `serial_lm_head` loop with a native row-aware LM-head/argmax only after
      C2 equality is green. Acceptance: c=2/4/8 equality stays green with
      `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE=batched_lm_head` or successor.
      Progress: `hipengine.dispatch.sampling` now gates `batched_lm_head` behind
      explicit c>N generated-token equality evidence and a retained artifact path
      under `benchmarks/results/`; `_sample_batch_from_hidden(...)` records the
      sampler decision and falls back to `serial_lm_head` when evidence is
      missing, failed, wrong-row, points outside retained artifacts, or resolves
      through a symlink outside the active results tree, and accepted c>N artifact
      schema requires a native sampler decision with requested mode `batched_lm_head`, row count and equality row count matching `workload.concurrency`, green retained equality
      evidence plus no blockers, and dispatch/retained bench now block promotion before schema validation unless sampler metadata records an explicitly requested native row-aware batched LM-head decision with matching rows/equality rows, a retained equality artifact whose JSON reports non-blank self-matching `artifact_path`/`source_artifact_path` plus generated-token equality vs independent c=1 (`passed=true`, `skipped=false`, matching non-empty typed integer batch/c1 sequence lists, empty mismatches) at the same row count, matching profiler expected/trace/duration evidence for a native batch sampler/LM-head kernel, and no blockers, so setting the mode cannot silently create a
      native sampler claim before same-concurrency equality and profiler evidence are green.

### C4 packets — continuous scheduler and dynamic KV pool

- [x] **C4.1 engine-loop skeleton.** Introduce long-lived
      `submit/poll/cancel` driver around existing resident sessions, initially
      using fake/CPU tests and the serial bridge. Acceptance: requests can be
      admitted, decoded, finished, and reclaimed without a one-call lifetime.
      Evidence: `hipengine/generation/engine_loop.py` plus
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C4.2 adapter migration.** Lower `LLM.generate()` and non-streaming
      server endpoints onto `submit+poll` while preserving current outputs.
      Acceptance: existing generator/server tests pass and prompt-list
      batching still routes by request id. Evidence:
      `SubmitPollTextGenerator` wraps resolved model generators in
      `LLM._get_text_generator()`, prompt order/row-seed coverage in
      `pytest -q tests/test_generation_batch_scheduler.py tests/test_llm_generate.py -q`,
      and server prompt-list batching remains covered by
      `pytest -q tests/test_server_api.py -q`.
- [x] **C4.3 tick policy.** Implement `RECLAIM → ADMIT → choose(PREFILL_CHUNK,
      DECODE_STEP)` with `protect_decode` default. Acceptance: scheduler tests
      cover decode protection and TTFT/fair alternatives. Evidence:
      `ResidentEngineLoop(prefill_decode_policy=...)` plus
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C4.4 chunked KV pool.** Add chunked allocation, grow-on-admission,
      idle shrink, and high/low-water knobs behind fake-runtime tests first.
      Acceptance: burst+idle fixture records at least one grow and shrink or
      explicitly records that the initial chunk sufficed. Evidence:
      `hipengine/kvcache/pool.py` plus `pytest -q tests/test_kvcache_policy.py -q`.
- [x] **C4.5 pool/env docs.** Add CLI/env knobs for `HIPENGINE_KV_POOL_*` and
      `HIPENGINE_PREFILL_DECODE_POLICY` and document them in `docs/ENVS.md`.
      Acceptance: CLI/env tests and docs agree on defaults. Evidence:
      `add_engine_loop_config_args(...)`, `docs/ENVS.md`, and
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C4.6 streaming through loop.** Route streaming completions through
      per-request token queues instead of bypassing the batcher. Acceptance:
      streaming and non-streaming share reclaim/cancel tests. Evidence:
      `_GenerationBatcher.stream(...)` per-request queues, single-row chat
      streaming routed through that batcher instead of `engine.stream`, and
      `pytest -q tests/test_server_api.py -q`.
- [x] **C4.7 unified reclaim.** Make cancel, disconnect, EOS, max-tokens, and
      timeout converge on one `RECLAIM` path. Acceptance: each finish reason
      frees KV/scratch exactly once in tests. Evidence:
      `ResidentBatchScheduler.cancel/disconnect/timeout(...)`, generated-token
      `stop`/`length` reclaim, and `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C4.8 non-compact-slot native decode.** Extend native decode beyond
      compact `0..C-1` slots after scheduler compaction/reclaim. Acceptance:
      generated-token equality passes with a deliberately sparse/compacted
      slot schedule. Evidence: `step_batch_native(...)` now accepts sorted
      sparse physical slots, `_batch_full_spans(...)` maps slot ids into
      row-relative KV block tables, `pytest -q tests/test_qwen35_resident_batch_layout.py -q`,
      and `python3 scripts/qwen35_batch_sparse_slot_correctness.py --json /tmp/hipengine-sparse-slot-L1.json`
      shows generated-token equality vs independent c=1 for a cancel-middle
      active slot history `[[0, 2], [0, 2]]`.
- [x] **C4.9 observability fields.** Record per-request and per-pool fields in
      completion/artifact metadata. Acceptance: tests assert queue/prefill/
      decode seconds, KV pages, bucket key, admission blocker, and finish
      reason are present. Evidence: `CompletedRequest.to_json_dict()`,
      `KVPoolStats.to_json_dict()`, accepted-artifact schema checks, and
      `pytest -q tests/test_generation_batch_scheduler.py -q`.

### C5 packets — prefix sharing, per-row sampling, `n>1`, metrics

- [x] **C5.1 block refcounts.** Add block-id refcounts and reuse accounting.
      Acceptance: shared-prefix admission increments/decrements refcounts and
      reclaim only frees zero-refcount blocks. Evidence:
      `ChunkedKVPool.admit_with_shared_prefix(...)`, prefix reuse counters, and
      `pytest -q tests/test_kvcache_policy.py -q`.
- [x] **C5.2 RadixCache.** Implement the token-id trie with
      `HIPENGINE_PREFIX_CACHE` / `--prefix-cache` in `{off, radix}`. Acceptance:
      prefix-hit/miss tests cover partial-block edges and cancellation.
      Evidence: `hipengine/kvcache/radix.py`, `HIPENGINE_PREFIX_CACHE`,
      `hipengine serve --prefix-cache`, and
      `pytest -q tests/test_kvcache_policy.py tests/test_server_api.py -q`.
- [x] **C5.3 copy-on-write fork.** Fork fresh pages at the first divergent
      token while preserving shared prefix pages. Acceptance: two diverging
      requests keep prefix bytes shared and produce independent suffix KV.
      Evidence: `ChunkedKVPool.fork_copy_on_write(...)`, COW fork counters,
      and `pytest -q tests/test_kvcache_policy.py -q`.
- [x] **C5.4 `n>1` lowering.** Replace API rejection with N scheduler
      requests sharing a prompt prefix and distinct seeds. Acceptance:
      OpenAI-compatible responses preserve `n` semantics and request IDs.
      Evidence: server completion/chat `n` lowering, distinct `row_seeds`,
      per-choice `request_id`, and `pytest -q tests/test_server_api.py -q`.
- [x] **C5.5 per-row sampler.** Land per-row temperature/top-k/top-p/
      repetition-penalty/seed/stop-token handling. Acceptance: incompatible
      sampling params decode together and deterministic seeds are stable.
      Evidence: `PerRowSamplingParams`, `SamplerParamsBlock`, and
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C5.6 per-row EOS/reclaim.** Finish rows independently inside a batch.
      Acceptance: one row can finish while others keep decoding and its KV is
      reclaimed at the next commit point. Evidence:
      `ResidentBatchScheduler(reclaim_callback=...)` and
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C5.7 metrics endpoint.** Add Prometheus `/metrics` behind
      `HIPENGINE_METRICS` / `--metrics`. Acceptance: metrics are additive and
      include request, pool, and graph-bucket counters. Evidence:
      `ServerConfig(metrics="prometheus")`, `hipengine.server.__main__ --metrics`,
      `docs/ENVS.md`, and `pytest -q tests/test_server_api.py -q`.
- [x] **C5.8 retained-row enforcement.** Make the bench harness enforce gates
      for timestamps, p50/p95 latency, dynamic pool, stable block id, and
      prefix-sharing savings before `status=accepted`. Acceptance: a fixture
      missing any required field cannot be accepted. Evidence:
      `scripts/qwen35_batch_artifact_schema.py` accepted-row gates now also
      require non-skipped generated-token equality vs independent c=1 with exact
      `batch_sequences == c1_sequences` lists whose row count matches
      `workload.concurrency`, whose per-row token counts match seed +
      `workload.warmup_decode_tokens` + `workload.gen_tokens_per_request`, and
      whose seed prefixes and measured-decode suffixes match
      `execution.seed_tokens` / `execution.generated_tokens`, whose
      `execution.completed` rows cover every request, carry prompt-token counts
      matching `workload.prompt_lengths`, and match generated-token and
      finish-reason records, no mismatches, and a passing primitive c>N GPU
      correctness JSON whose self-reported `artifact_path` and `rows` values match
      the retained reference path and `workload.concurrency`, plus
      retained-bench allocator/memory evidence merge/blockers in
      `test_qwen35_retained_allocator_memory_evidence_from_stats`,
      `test_qwen35_retained_memory_payload_uses_bench_evidence`, and
      `test_qwen35_retained_memory_evidence_blockers_cover_required_fields`,
      and `pytest -q tests/test_generation_batch_scheduler.py -q`.

### Performance packets — run only after correctness is green

- [ ] **P1 baseline bundle.** Establish c=1, serial bridge c=2/4/8, first
      green uncaptured native c>N, and primitive microbench baselines.
      Acceptance: artifacts include exact commands, hardware, correctness,
      aggregate/per-request ratios, and dirty-state. Progress: retained native
      c>N artifacts now require explicit c=1 and serial-bridge baseline JSONs
      before `performance_claim=true`; `scripts/qwen35_batch_c_sweep.py` wires
      those paths into the planned retained command (`--c1-baseline-json`,
      `--serial-bridge-json`) and now also passes the matching
      `--primitive-correctness-json` path (`primitive-cN.json`) plus the planned
      `--profiler-json` path (`profiler-cN.json`) so a green generated-token run
      without primitive GPU correctness or captured profiler evidence remains
      blocked instead of becoming a throughput claim, and retained bench now blocks promotion before schema validation when profiler artifact/trace/command/profiler-json/output-json/warmup-inclusive workload-shape/model-fixture/cached-build/reference-artifact/KV-policy provenance, symlink-escaped retained artifact/reference/profiler paths, exact rocprof separator count, pre-separator rocprof executable/option binding/uniqueness, rocprof separator/profiled-command binding, post-separator retained-flag binding/uniqueness, self-contained artifact command labels, concrete profiler trace paths, all concrete profiler command-label validation, profiler-command generated-equality gating, trace kernel names, expected kernel names, explicit profiler capture-status and expected-kernel-present verdicts, unique native-batch profiler kernel-name lists, expected-kernel trace membership, positive kernel-duration evidence, total-duration arithmetic, per-kernel duration-share arithmetic, duration-category total/share arithmetic, or CPU-side bottleneck total/share arithmetic are missing or inconsistent. Real c-sweep runs now skip
      retained native diagnostics if the matching primitive, c=1 baseline,
      serial-bridge, or profiler-summary artifact is missing, failed, has a
      mismatched profiler artifact path, row count, or prompt/decode shape, or
      lacks required row/shape/kernel/CPU-side bottleneck labels; the c=1 PARO
      bench now emits a
      first-class `workload` object with `concurrency=1`, prompt/decode token
      counts, and KV policy, and retained scaling summaries carry c=1/serial
      baseline `status`/`reason`, `workload_concurrency`, and prompt/decode
      labels, and the retained precondition records include the resolved baseline
      and profiler status/reason/c-sweep-and-schema-checked matching command/output-format/trace-dir/kernel-trace CSV/kernel-name/duration/provenance fields (`profiler_trace_synthesized_fields` in c-sweep preconditions, mandatory post-run c-sweep retained-artifact cross-checks with validated persisted summary rollups and singular failed `postcondition` records, and schema- plus retained-bench-checked `profiler.synthesized_fields` in retained artifacts; c-sweep and retained-bench load kernel names and durations from trace CSVs), structured retained/c=1/serial/primitive/profiler reference artifact paths (including retained-bench, c-sweep, and accepted-artifact-schema scaling-reference `reference_artifact_path` self-binding for c=1/serial source JSONs, retained-bench/source-schema `source_artifact_path` self-binding for primitive correctness and profiler JSONs with retained-bench profiler provenance blockers, c-sweep `profiler_source_artifact_path` precondition/postcondition self-binding, plus a c-sweep primitive precondition `primitive_artifact_path`, each validator-checked against the retained command's matching gate path),
      structured cached-build flags, structured model/fixture/run-shape labels, aggregate/per-request rates, profiler native-batch-only kernel duration/share keys and schema-checked kernel-row-derived category totals/shares, and
      CPU-side bottleneck totals/shares, so c-sweep preconditions and artifact schema
      validation reject c>N rows compared against missing, failed/unusable,
      reason-bearing, ambiguous, or wrong-shape baselines; the sweep writes `command_count`,
      `completed_command_count`, an `options` block, per-retained-command
      `preconditions`, `status_counts`, `category_status_counts`,
      `retained_precondition_counts`, and `skipped_preconditions` summary rollups
      for planned/passed/skipped/failed rows, has persisted-summary coverage and summary-validator checks for
      typed validator/CLI root object/schema/version plus pre-run/persisted exact summary/option key sets, typed dry-run/run-option booleans/non-blank typed model-fixture/deterministic-seed/workload-shape labels, parseable timezone-aware timestamps, typed pre-run/persisted non-empty batch-size-list and exact dry-run planned/skipped/simple-executed-row key sets plus dry-run/skipped-row status/duration/output/condition/postcondition semantics, stop-on-failure terminal-row semantics, exact git provenance key set plus non-blank dirty-state/status provenance, non-empty command list, known command key set and command identity fields (including non-blank category/artifact-path/command/strict-python-executable/non-empty-primitive+retained-argv-flag-value/argv, retained-flag uniqueness, parent-traversal-free artifact-path/`--json` and artifact `output_dir`, non-symlink artifact-path/`output_dir`/parent containment plus category/batch filename identity, batch-size/argv, run-shape argv, category/script consistency, fully-passed summary artifact regular-file/non-symlink existence, finite execution-duration metadata, precondition/postcondition scope/path integrity (including retained gate argv presence/parent-traversal-free non-symlink filename bindings), and status/returncode/precondition/postcondition consistency (including non-skipped failed-gate and zero-return passed-postcondition rejection)), typed derived command counts/dry-run+non-stop completeness/order, known condition key set plus exact minimal-failed/primitive-correctness/scaling-reference/profiler-summary precondition and passed/failed retained-postcondition key sets plus failed retained source/synthesized-evidence-free malformed-source plus typed JSON output-dir-bound non-symlink/symlink-parent-free parent-traversal-free source-provenance/source-mismatch and synthesized-field evidence known/unique/precondition-binding/pairing/typing and condition-entry schema/non-blank failed-reason shape, primitive-command rows/seed plus primitive-correctness schema/seed/fixture-shape/typed-context-lens/NumPy-oracle precondition fields, profiler-shape/command-path/strict-command-executable/single-command-separator/bidirectional-rocprof-retained-option-placement/rocprof-option-value/rocprof-option-uniqueness/profiled-command/profiled-command-flags/profiled-command-flag-value/profiled-command-flag-uniqueness/command-kernel-trace-flag/command-output-format/artifact-ref/parent-traversal-free non-symlink serial+native matched compiler-version/cache-required build-cache option+argv/precondition-synth-field/trace-kernel/trace-kernel-uniqueness/expected-kernel-uniqueness/trace-path/trace-path-canonical-containment/parent-traversal-free trace-dir+trace-file/non-symlink trace-dir+trace-file parent-containment/trace-file-extension/trace-file-uniqueness/trace-duration/category-arithmetic/CPU-bottleneck-arithmetic, and CLI-reported non-blank typed pre-run output-dir/summary-json/compiler-version paths plus persisted summary output-dir/compiler-version paths plus model-fixture/typed deterministic-seed/workload-shape plus validate-summary-json path/symlink-parent and persisted scaling-label/rate/arithmetic checks, passed retained-row postcondition presence/reason-shape/synthesized-field precondition-binding checks, and retained native gate/postcondition-kind checks, typed status/category-status/precondition/postcondition rollups (including empty-command, unknown-count top/leaf-label, nonnegative-count, and bool-count tamper checks plus `qwen35_batch_c_sweep.py --validate-summary-json`), typed exact-key command-derived skipped-precondition/failed-postcondition rollups, and skipped retained rows retaining both the complete `preconditions` list, first-failed reason/bounded-output-tail evidence, and
      typed singular first-failed `precondition` / `postcondition` (rejecting stale/stray/type-drifted singular entries when no condition failed or no matching condition list exists, and binding failed-postcondition output tails to the postcondition reason), and has unit coverage confirming
      usable references allow the retained command to run. Accepted artifact
      schema also rejects baseline statuses known to be unusable for claims
      (`missing`, `invalid_json`, `failed`, `rejected`, and
      `rejected_correctness`) before a c>N row can be promoted.
- [ ] **P2 graph replay buckets.** Add decode hipGraph capture/replay buckets
      by `(C, context bucket, active mask, KV dtype, layer plan, top-k/experts,
      replay length)`. Acceptance: bucket hit/miss stats and profiler evidence
      show replay for common shapes. Progress: graph-bucket stats now serialize
      `entries`, `hits`, `misses`, `replay_hit_rate`, miss-reason counts, and typed-integer kernel-time
      histogram buckets from `GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS`; retained bench profiler summaries populate those buckets, invalid hit/miss/replay-rate stats plus missing or unknown-bucket histogram observations block promotion, and retained accepted-artifact schema requires non-empty known-bucket kernel-time
      histogram evidence plus those
      observability fields plus accepted-schema-validated replay shape-key axes (`context_bucket`,
      `top_k`, `experts_per_token`, `replay_steps`, `draft_depth`, and
      `tree_shape`, with the context bucket covering the workload prompt length), positive profiler `graph_replay` expected-kernel/duration/category/share evidence whenever replay hits are positive, and per-bucket histogram observation counts covering both replay hits and profiler kernel-duration evidence before a c>N row can be promoted; `/metrics` exposes a
      hit/miss-derived replay-hit-rate gauge plus labeled miss-reason and known kernel-time-bucket counters for live runs.
- [ ] **P3 remove residual serial loops.** Remove full-attention per-row
      fallback, per-row metadata allocation, per-row LM-head launches, and
      Python per-layer dispatch from steady-state native decode. Acceptance:
      profiler summaries show the removed bottleneck and equality remains
      green. Progress: accepted/performance-claim c>N artifact schema now rejects
      serial-bridge paths, non-scheduler-owned execution, non-full-native, wrong-path, wrong-layer-limit, or unsupported-layer-bearing prefill plans, non-empty batch/prefill/decode-execution blockers, row executions labeled `serial`/`fallback`, missing or wrong-shape
      decode-execution row/slot/context/layer-count plus grouped-compact MoE path/row/layer-count metadata, missing or stale per-layer decode traces for native full-attention/grouped-MoE layers, native-batch decode contexts at or beyond 1024 before row-aware split-K lands, non-`native_batch` full-attention decode paths,
      per-row full-attention decode fallbacks, non-native sampler metadata, sampler requested-mode mismatches, sampler row/equality-row mismatches, and failed or wrong-row sampler equality artifacts,
      runtime short-context native full-attention metadata now reports the retained-compatible `native_batch` path,
      and retained bench now blocks promotion before schema validation for the same serial/fallback batch/decode metadata,
      so residual serial loops cannot be promoted as retained rows while this
      item remains open.
- [ ] **P4 MoE/projection scaling.** Group routed lanes by expert and switch
      c=2/4/8 projections/MoE to kernels that beat row-GEMV. Acceptance:
      c=8 aggregate decode improves vs both c=1 and the serial bridge, with
      per-request ratios reported.
- [ ] **P5 retained scoreboard update.** Only after accepted artifacts exist,
      update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and compact
      JSON artifacts under `benchmarks/results/`. Acceptance: every perf claim
      cites correctness gate, profiler status, exact command, and hardware.
      Progress: accepted/performance-claim c>N artifacts now fail schema
      validation unless they include fully native scheduler-owned batch/prefill/decode-execution metadata with empty blockers, the known full-native prefill path, null unsupported-layer fields, positive native full-attention layer evidence, decode rows/slots plus grouped-compact MoE decode rows matching `workload.concurrency` with positive grouped-compact layer count and zero selected-c1 fallback layers, decode context covering `workload.prompt_tokens_per_request` while staying below the open row-aware split-K threshold, and prefill layer limits matching `workload.max_layers`,
      workload native prefill/decode flags set, workload scheduler labels matching the execution path, per-layer decode traces matching the global native full-attention and grouped-MoE layer counts, projection evidence artifact JSON reporting accepted same-row evidence with self-matching `artifact_path`/`source_artifact_path` and matching >1 aggregate/per-request row-GEMV speedup ratios, sampler requested mode `batched_lm_head` plus rows/equality rows matching `workload.concurrency` and sampler equality artifact JSON reporting self-matching `artifact_path`/`source_artifact_path` and the same rows with generated-token equality vs independent c=1 (`passed=true`, `skipped=false`, matching batch/c1 sequence lists, empty mismatches),
      generated-token equality sequence lists matching `workload.concurrency`
      rows and seed + warmup + measured decode token counts per row, with
      `execution.seed_tokens` / `execution.generated_tokens` matching the seed
      prefix and measured-decode equality suffixes, and `execution.completed`
      rows covering every request with prompt-token counts matching
      `workload.prompt_lengths` plus matching generated-token and finish-reason
      records,
      primitive GPU correctness reference typed script schema (`schema=1`), source `artifact_path` matching the retained reference path, typed row count matching
      `workload.concurrency` in both source and summary preconditions,
      reference-/c-sweep-gated typed deterministic `seed=1234` provenance,
      deterministic typed fixture-shape
      metadata (`block_size`, `max_context_len`, `num_q_heads`, `num_kv_heads`,
      `head_dim`) in summary preconditions, reference-/c-sweep-gated typed
      per-row `context_lens` fixture coverage,
      reference-/c-sweep-gated typed source zero append-mismatch counters,
      reference-/c-sweep-gated exact-zero batch-vs-c1 attention error, and
      reference-/c-sweep-gated finite nonnegative NumPy-oracle attention error ≤ 2e-5,
      full 40-layer workload labels with concrete model/quant/KV storage dtype
      plus matching KV policy metadata,
      aggregate token labels and per-row prompt lengths matching per-request
      shape times concurrency, full-row admission/completion/per-request
      observability with finite admission/completion timestamps, completion
      after admission, finite nonnegative per-row timing, matching row ids,
      and latency samples matching completion-minus-admission plus derived
      percentiles (`p50` median, `p95 >= p50`) for every row in
      `workload.concurrency`, memory batch/sequence/KV-policy metadata
      matching workload shape, finite nonnegative allocator peak bytes,
      dynamic-pool evidence plus finite nonnegative counters,
      stable block-id audit, and prefix-sharing savings,
      execution scheduler metadata with decode shape-key active mask
      length/count matching workload concurrency plus graph-bucket entry/hit/miss arithmetic, positive replay hits, matching replay-hit-rate, and positive profiler graph-replay expected-kernel/duration/share evidence, positive finite
      aggregate/per-request throughput whose c-sweep scaling precondition
      concurrency and run-shape labels are typed and match `workload`, and whose
      native scaling copy matches the primary measurements, all
      required positive scaling ratios that mathematically match usable same-shape
      c=1 and usable same-shape/same-concurrency serial bridge
      baselines, aggregate ratios vs both references that beat 1.0,
      accepted-artifact schema checks for positive finite throughput and
      decode-step timing samples,
      a retained-benchmark command starting with a Python invocation of
      `scripts/qwen35_batch_retained_bench.py` with a top-level artifact path under
      `benchmarks/results/` matched by the
      retained benchmark/profiler `--json` outputs, explicit `--model` /
      `--fixture` plus `--batch-size`, `--prompt-length`, `--decode-tokens`, and
      `--max-layers` matching workload shape fields and baseline/correctness
      reference paths matching the retained artifact payload,
      a correctness-reference command that names generated-token equality vs
      independent c=1, with an embedded Python invocation of
      `scripts/qwen35_batch_correctness.py` whose own argv carries only
      `qwen35_batch_correctness.py` flags with unique `--rows` / `--seed` / `--json`,
      `--rows` matching `workload.concurrency`, `--seed` matching
      `correctness.primitive_batch_correctness.seed`, and `--json` matching
      `correctness.primitive_batch_correctness.artifact_path`, and a concrete
      `rocprofv3 --kernel-trace` profiler command targeting
      `scripts/qwen35_batch_retained_bench.py` after the rocprof `--` separator,
      with unique rocprof-only flags (`--kernel-trace`, `--output-format csv`, `-d`) before
      that separator, and unique retained shape/artifact/reference/cached-build
      flags validated from the post-separator profiled command segment
      (`--model`, `--fixture`, `--batch-size`, `--prompt-length`,
      `--decode-tokens`, `--max-layers`, `--c1-baseline-json`, `--serial-bridge-json`,
      `--primitive-correctness-json`, `--profiler-json`,
      `--compiler-version-file`, `--require-cached-build`), typed profiler
      precondition workload/warmup/layer labels matching the retained command,
      benchmark/profiler `--json` outputs plus primitive/scaling/compiler-version
      artifact paths resolving under the current `benchmarks/results/` tree with
      no parent traversal spelling and explicit `.json` regular-file references
      with no symlink file or parent-directory components, and the retained bench can now attach a
      captured profiler summary via `--profiler-json` / `--profiler-command`,
      synthesize `profiler.total_kernel_duration_ns`,
      `profiler.kernel_duration_shares`, `profiler.kernel_duration_categories_ns`,
      `profiler.kernel_duration_category_shares`, and CPU-side bottleneck
      summaries from per-kernel durations and retained wall-clock timings when
      the summary omits them, and require `--profiler-json` to match
      `profiler.artifact_path`; accepted artifacts now schema-check retained-payload
      benchmark rollup declarations (`artifact_path`, matching `source_artifact_path`,
      `benchmarks/README.md`, and `benchmarks/CHANGELOG.md`) while the post-run `validate_cn_diagnostic_rollup_evidence` gate
      (or the CLI `python3 scripts/qwen35_batch_artifact_schema.py <artifact>
      --rollup-evidence --summary-json
      benchmarks/results/<artifact-stem>-rollup-check.json`) verifies live
      `benchmarks/README.md` and `benchmarks/CHANGELOG.md` both mention the
      retained artifact path, the README carries `Last updated: YYYY-MM-DD`,
      and the changelog carries a same-line dated `YYYY-MM-DD` artifact entry whose date matches README `Last updated` and includes numeric old→new metric plus percent-delta evidence before promotion, writes self-validating
      schema-versioned (`schema=1`) closed-key pass/fail regular `.json` summary file evidence (canonical relative or absolute current-repo paths) resolving under the current repo `benchmarks/results/` for both `--summary-json` writes and existing-file `--validation-summary` rechecks before filesystem write/read attempts (no parent traversal spellings/escapes, symlink targets/parents, external repo paths, non-file targets, or non-directory parents), binds all schema and rollup summary write/recheck relative paths to the retained/source artifact location with write- and recheck-specific diagnostics (including failed summaries with no retained `artifact_path`), keeps rollup metadata out of generic artifact-schema summaries (even while preserving `status`/`performance_claim` labels), requires passed rollup summaries to assert `status=accepted`, `performance_claim=true`, and closed-key canonical README/CHANGELOG rollup metadata, rejects malformed/extra-key rollup metadata even on failed summaries, requires every validation-summary source/retained artifact path (including rollup metadata copies) to be repo-relative under `benchmarks/results/` with no parent traversal spelling and passed/rollup-bearing summaries to carry a non-null retained `artifact_path` with no nested copied-prefix, and can
      recheck those summaries with `--validation-summary`,
      exact environment capture command entries for `rocminfo | grep -E 'Name:|gfx' | head -4`,
      `rocm-smi --showmeminfo vram --showuse --showtemp`, `hipcc --version`,
      `git rev-parse HEAD`, and `git diff --quiet`,
      concrete non-empty hardware `gpu`/`arch` fields with `gpu` identifying an
      AMD/Radeon/Instinct device and `arch` formatted as a
      `gfx*` architecture string plus successful
      `hardware.rocminfo`/`hardware.rocm_smi` capture objects whose commands
      include the retained capture fragments (`rocminfo | grep -E`, `Name:|gfx`,
      `head -4`, and the `rocm-smi --showmeminfo vram --showuse --showtemp` flags),
      whose `rocminfo` output includes a `Name:` marker plus the recorded arch,
      and whose `rocm_smi` output includes GPU and VRAM markers,
      clean full-commit software fields (`software.hipengine_dirty == false`)
      plus a non-empty `hipcc_version` string containing a hipcc/HIP/clang version marker,
      and captured profiler evidence with a
      `profiler.artifact_path` under `benchmarks/results/`, profiler trace
      files canonically contained under `profiler.trace_dir`, native batch
      expected kernel names and duration/share-map keys as non-empty strings (no
      serial/per-row/fallback labels) present with every duration-map entry,
      including extra trace-listed entries, carrying positive finite numeric evidence,
      `profiler.total_kernel_duration_ns`
      matching the duration-map sum, exact per-kernel duration-share keys/values
      matching `duration / total`, finite exact-key category duration/share buckets for
      attention/MoE/projection/sampling/graph/other, finite exact-key CPU-side bottleneck
      duration/share totals, plus an accepted non-row-GEMV
      `projection_dispatch` decision whose selected candidate is listed with
      matching retained speedup evidence and appears in profiler expected/trace/duration kernel names, plus native sampler/LM-head expected/trace/duration profiler evidence.
      The scoreboard item remains open until accepted
      artifacts exist and the benchmark rollups are updated.

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
- [x] Add HIP-guarded reduced-shape equality diagnostics that do **not**
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
      long-context c>N claim (`max_context ≥ 1024`). The current long-context
      diagnostic uses the per-row split-K fallback, not a row-aware batch reducer.
- [x] Add CPU-runnable structural tests for the experimental env gate,
      INT8 KV rejection, default/invalid sample mode, and
      `throughput_claim_eligible=false` for guarded diagnostics.
- [x] Extend structural tests for invalid-slot and long-context rejection;
      sorted sparse slots are now accepted and covered by C4.8.
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
- [x] Make full-attention decode consume per-row `KVLiveSpans` for all
      retained KV storage dtypes.
- [x] Make linear-attention conv/recurrent state updates consume
      `[C, ...]` state, active masks, and slot ids; remove c1 aliases.
- [ ] Replace selected-MoE c1 lane assumptions with token-row → routed-lane
      mapping; validate grouped-by-expert metadata for c=2/4/8.
- [x] Keep c=1 GEMV dispatch separate from c>N MMQ/GEMM/WMMA candidates.
      Evidence: `plan_projection_dispatch(...)` pins c=1 to `row_gemv_c1`
      even when a faster c-aware candidate is supplied, c>N candidates require
      accepted benchmark evidence with aggregate and per-request speedups over
      row-GEMV, missing/slow/rejected evidence falls back to row-GEMV, and
      accepted c>N artifacts must record an evidence-backed non-row-GEMV
      `execution.batch_execution.projection_dispatch` decision before any
      projection throughput claim; covered by
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [ ] Validate GGUF Q4_K/Q5_K/Q6_K/Q8_0 c=2/4/8 with the same gates.
- [ ] Native row-aware LM-head + sampler: replace the per-row argmax loop
      and prepare for per-row sampling params (C5 finishes this).

### C4 — scheduler-owned engine loop + dynamic KV pool

Definition of done: one long-lived background driver runs
`submit/poll/cancel`, ticks the work classes, grows/shrinks the KV pool, and
routes both streaming and non-streaming through the same path. `LLM.generate()`
becomes a `submit+poll` adapter.

- [x] Promote the resident runner from static prompt-list batches to a
      scheduler-owned engine loop that persists beyond one
      `LLM.generate()` call. Evidence: `ResidentEngineLoop` in
      `hipengine/generation/engine_loop.py` plus
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] Implement `submit(prompt_tokens, sampling, max_new_tokens, stream) →
      request_id`, `poll(timeout) → events`, `cancel(request_id) → bool`.
      Evidence: `ResidentEngineLoop.submit/poll/cancel` and scheduler reclaim
      coverage in `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] Lower `LLM.generate()` and OpenAI server endpoints to
      `submit + poll + cancel`. Evidence: `SubmitPollTextGenerator` wraps
      public text generation, server non-streaming calls use the shared
      batcher/generator path, and
      `pytest -q tests/test_generation_batch_scheduler.py tests/test_server_api.py -q`.
- [x] Implement the per-tick policy: `RECLAIM → ADMIT → choose(PREFILL_CHUNK,
      DECODE_STEP)`; default `protect_decode`.
- [x] Add `kv_pool_chunk_pages` chunked underlying allocation with one chunk
      at startup. Evidence: `ChunkedKVPool(..., chunk_pages=...)` in
      `hipengine/kvcache/pool.py` and
      `pytest -q tests/test_kvcache_policy.py -q`.
- [x] Add grow-on-admission up to `kv_pool_high_water_bytes`, one attempt per
      admit cycle; record `grow_events` / `grow_failures`. Evidence:
      `ChunkedKVPool.allocate(...)` grow counters and
      `pytest -q tests/test_kvcache_policy.py -q`.
- [x] Add idle shrink down to `kv_pool_low_water_bytes` with
      `kv_pool_idle_grace_seconds`; never free a chunk holding a non-zero
      refcount. Evidence: `ChunkedKVPool.shrink_idle(...)` plus refcounted-tail
      coverage in `pytest -q tests/test_kvcache_policy.py -q`.
- [x] Add CLI/env knobs `--kv-pool-{initial,low-water,high-water,
      chunk-pages,idle-grace}-*`,
      `HIPENGINE_KV_POOL_*`,
      `HIPENGINE_PREFILL_DECODE_POLICY` / `--prefill-decode-policy`;
      document in `docs/ENVS.md`.
- [x] Add a burst-then-idle acceptance fixture that exercises grow and
      shrink and records the events. Evidence:
      `test_chunked_kv_pool_grows_and_shrinks_on_burst_idle` in
      `tests/test_kvcache_policy.py`.
- [x] Add a memory-audit test that fails if a block id's backing pointer
      changes mid-run. Evidence:
      `test_fixed_paged_policy_audits_append_only_block_pointers` in
      `tests/test_kvcache_policy.py`.
- [x] Narrow or remove the coarse `generation_lock`; any remaining lock
      protects only non-reentrant session mutation, not the lifetime of a
      generated batch. Evidence: `hipengine/server/api.py` now uses
      `session_lock` only for LLM construction/preparation/context-budget mutation,
      `_GenerationBatcher` no longer accepts a lock, and
      `test_generation_batcher_default_zero_window_queues_without_lifetime_lock`
      plus `pytest -q tests/test_server_api.py -q` cover the batcher path.
- [x] Route server streaming through the engine loop and the per-request
      token queue; the streaming path no longer bypasses the batcher.
- [x] Unify cancel / disconnect / EOS / max-tokens / timeout into one
      `RECLAIM` path.
- [x] Per-request observability fields (queue/prefill/decode seconds,
      kv pages owned/peak, bucket key, admission_blocked_reason,
      finish_reason).
- [x] Per-pool observability counters
      (current_bytes, high_water_observed, grow/shrink events, free pages,
      refcounted pages).
- [x] Extend native decode correctness to non-compact slots after
      scheduler compaction/reclaim moves requests; sorted sparse slots are
      supported by explicit physical slot ids.

### C5 — KV sharing, per-row sampler, `n>1`, `/metrics`

Definition of done: refcounted prefix sharing on by default; per-row sampler
in code; `n>1` lowered to N scheduler requests; Prometheus `/metrics`
endpoint live; retained c>N rows include all gates above.

- [x] Add block-id refcounts; admission increments refcount when reusing
      an existing block on a matched prefix.
- [x] Implement RadixCache trie index over token ids; expose
      `HIPENGINE_PREFIX_CACHE` / `--prefix-cache` in `{off, radix}` with
      default `off` until acceptance gates pass.
- [x] Implement copy-on-write fork at first divergent token.
- [x] **KVTC ABI guardrail.** Block ids returned by the allocator must be
      stable across hypothetical tier moves; refcount and eviction state
      must attach to the radix node rather than the block pointer. KVTC
      itself ships in a separate feature branch. Evidence:
      `PrefixCacheEntryState` exposes pointer-independent radix-node metadata
      (`block_ids`, `owner_request_ids`, `refcount`, `eviction_state`),
      `RadixCache.mark_entry_eviction_state(...)` updates tier/eviction state
      without rewriting block ids, and
      `test_radix_cache_entry_state_is_pointer_independent_kvtc_guardrail`
      covers stable block ids across tier-state changes and cancellation.
- [x] Lower `n > 1` at the API layer to N `submit()` calls with the same
      prompt tokens and distinct seeds; collect output by `request_id`;
      remove the `n>1 → 400` rejection.
- [x] Land the per-row sampler params block (temperature, top-k, top-p,
      repetition penalty, seed, stop tokens) in one launch.
- [x] Per-row EOS handling drives `RECLAIM` per-row, not per-batch.
- [x] Remove or demote the submission-time coalescer
      (`_GenerationBatcher`) to a cold-path optimization. Evidence: default
      `ServerConfig.generation_batch_window_ms` and
      `HIPENGINE_GENERATION_BATCH_WINDOW_MS`/`--generation-batch-window-ms`
      remain `0`, `_GenerationBatcher` applies no intentional zero-window delay
      and no longer holds a request-lifetime generation lock, and
      `test_generation_batcher_default_zero_window_queues_without_lifetime_lock`
      plus `test_metrics_prefix_cache_and_generation_batch_cli_env_defaults`
      cover the default/opt-in path.
- [x] Add Prometheus `/metrics` endpoint;
      knob `HIPENGINE_METRICS` / `--metrics` in `{off, prometheus}`;
      default `off` until coverage is broad.
- [ ] Per-bucket graph-cache observability
      (entries, hits, misses, miss reason, kernel-time histogram). Progress:
      `GraphBucketCache.stats.to_json_dict()` now includes miss-reason counts
      and typed-integer kernel-time histogram buckets, retained/serial scripts emit that
      shape, retained bench validates decode shape-key axes (including context-bucket coverage for the workload prompt length) and merges integer profiler kernel durations into the histogram, blocking promotion when shape keys are invalid, hit/miss/replay-rate stats are invalid, no known-bucket observations remain, or unknown buckets appear, and accepted-artifact schema shares the runtime bucket taxonomy and requires context-bucket workload coverage plus per-bucket known-bucket histogram observations that cover profiler kernel-duration evidence
      for accepted rows, and `/metrics` exports labeled miss-reason and
      known kernel-time-bucket counters; the item remains open until real replay
      profiler evidence populates kernel-time buckets.
- [x] Retained-row gates 4 (admission/completion timestamps + p50/p95) and
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
- [x] Add graph-bucket cache hit/miss and replay statistics to artifacts.
      Evidence: `GraphBucketStats.to_json_dict()` serializes `entries`,
      `hits`, `misses`, `replay_hit_rate`, `miss_reasons`, and typed-integer
      `kernel_time_histogram_ns`; `scripts/qwen35_batch_retained_bench.py` and
      serial diagnostics emit `decode_shape_key` / `graph_bucket_stats`, and the retained bench merges profiler kernel durations into that histogram;
      accepted-artifact schema requires those fields and non-empty known-bucket histogram
      observations for accepted rows using the runtime bucket taxonomy; `/metrics` exports graph-bucket counters and filters kernel-time buckets to that taxonomy;
      covered by
      `test_graph_bucket_cache_clear_resets_entries_and_counters`,
      `test_qwen35_retained_records_decode_graph_bucket_metadata`,
      `test_qwen35_batch_diagnostic_artifact_schema_enforces_accepted_row_gates`,
      and `test_metrics_endpoint_is_opt_in_and_additive`.
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
- [x] Profiler summaries for accepted rows: expected kernel names,
      duration/share for attention, MoE, projection, sampling, graph
      replay, and any CPU-side bottleneck. Evidence: accepted-artifact schema
      requires native batch expected kernel names, per-kernel durations/shares
      matching `kernel_durations_ns / total_kernel_duration_ns`, category
      duration/share buckets for attention/MoE/projection/sampling/graph/other,
      and CPU-side bottleneck durations/shares whose totals match; retained-bench
      ingestion synthesizes totals, shares, category buckets, and CPU-side
      bottleneck summaries when profiler summaries provide durations; covered by
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
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
  selected-MoE/native-row mapping. `scripts/qwen35_batch_hidden_bisect.py`
  now reproduces that reduced L8 failure with a hidden mismatch at generated
  index 1 and token mismatch at index 13, so the next correctness step is the
  selected-MoE lane-map fix rather than more token-only sweeps.

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
