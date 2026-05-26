# Concurrency and Continuous Batching

Last updated: 2026-05-26

This document is the working guide for turning the current Qwen/PARO resident
runtime into a vLLM-style `c=1..8` concurrent serving path. It captures what is
actually implemented today, what the GPU0 diagnostic sweep showed, and the gates
that must pass before any c>N number is a retained benchmark claim.

Related source-of-truth docs:

- [`PLAN.md`](PLAN.md) — architecture invariants and the long-form concurrent
  decode design.
- [`BENCHMARK.md`](BENCHMARK.md) — evidence policy and c=N benchmark gates.
- [`KVCACHE.md`](KVCACHE.md) — dense INT8 KV and compact DMS roadmap.
- [`PREFILL.md`](PREFILL.md) — native/compact prefill details.

## Definitions

| Term | Meaning |
| --- | --- |
| HTTP concurrency | Multiple client requests are in flight at the server at once. This can still be serialized internally. |
| Prompt-list batching | One API call carries multiple prompts, e.g. OpenAI completions `prompt=[...]`. This only counts as true c>N if the generator advances those prompts together. |
| c>N decode | `N` independent live requests each advance one target token in the same model step. |
| Continuous batching | The scheduler can admit, prefill, decode, finish, compact, and reclaim requests while other requests keep running. |
| Packed/native prefill | Multiple prompt rows are packed into one prefill slab and launched through row-shaped kernels. |
| Serial bridge | A correctness-first path with batch-shaped slots/KV metadata but active rows execute through the c=1 layer path. Useful for diagnostics; not a throughput claim. |

## Current answer

**hipEngine does not yet support true vLLM-style c>N serving.**

The public Qwen/PARO generator now has a first prompt-list c>N path: it admits
all prompt rows into `ResidentBatchScheduler`, uses native compact packed prefill
for BF16 KV prompt lists, and routes output by request id. Decode after the
seed token still uses `step_batch_serial`. The OpenAI server now coalesces
compatible non-streaming HTTP generations over a short configurable batch window
(`--generation-batch-window-ms`, default 5 ms) before one prompt-list
`LLM.generate()` call, but streaming remains one request at a time
and production decode is still serial. An experimental native decode diagnostic
exists behind `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1`, but it is
currently rejected on generated-token equality and is not a retained throughput
path. This is not continuous batching yet.

## Readiness matrix

| Layer | Current status | Evidence / code | Blocks true c>N |
| --- | --- | --- | --- |
| OpenAI server | Compatible non-streaming HTTP generations are coalesced into one prompt-list `LLM.generate()` call behind a grouped safety lock; `n>1` is still rejected and streaming remains one request at a time. | `hipengine/server/api.py:_GenerationBatcher`, `create_app`, `_validate_generation_request`. | Add continuous admission/completion timestamps, streaming routing, and latency/occupancy accounting. |
| Public `LLM.generate()` | Prompt lists with `len(prompts)>1` now use `ResidentBatchScheduler`, BF16 packed native prefill, request-id output routing, and serial slot-bridge decode. Streaming is one prompt only. | `hipengine/generation/qwen35_paro.py:Qwen35ParoOneTokenGenerator._generate_batch`. | Replace serial slot-bridge decode with native c-aware decode; add generated-token equality gates. |
| Scheduler | `ResidentBatchScheduler` owns pending/admitted queues, slots, active masks, compact prefill slabs, decode work, graph bucket keys, and completion routing in the prompt-list generator. | `hipengine/generation/batch_scheduler.py`; `Qwen35ParoOneTokenGenerator._generate_batch`. | Wire independent HTTP request admission and latency/occupancy accounting. |
| Prefill | Single-request native prefill is active. Prompt-list BF16 c>N uses `next_compact_prefill_slabs(...)` plus `prefill_native_packed(...)`. INT8 packed prefill is still not wired. | `Qwen35ParoResidentSession.prefill_native`, `prefill_native_packed`, `ResidentBatchScheduler.next_compact_prefill_slabs`. | Validate generated-token equality and add INT8 packed prefill before retained c>N claims. |
| Decode runtime | Production c>N prompt-list decode uses `step_batch_serial`: batch-shaped slots/KV, but row execution is serial c=1. An experimental `step_batch_native` diagnostic is guarded by `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1` and is currently `rejected_correctness` on L8 512/32 generated-token equality. | `Qwen35ParoOneTokenGenerator._generate_batch`, `Qwen35ParoResidentSession.step_batch_serial`, `step_batch_native`, `batch_execution_metadata`, `/tmp/hipengine-retained/guarded-L8-512-32.json`. | Audit row-aware decode kernels until generated-token equality vs independent c=1 passes; then wire graph replay. |
| Attention/KV primitives | BF16 batched paged KV append and batched full-attention context decode pass c=1/2/4/8 primitive correctness. | `scripts/qwen35_batch_correctness.py`. | Extend/validate the exact kernel families used by the resident runner, including INT8 KV paths and graph replay. |
| MoE/quant kernels | Many wrappers accept `rows` or routed-lane counts, but end-to-end selected-MoE decode still follows c1 assumptions in the current bridge. | `hipengine/kernels/hip_gfx1100/quant/*`, `hipengine/runtime/qwen35_paro.py`. | Token-row to routed-lane mapping, grouped-by-expert execution, c-aware dispatch thresholds. |
| KV cache | Dense fixed paged KV with uniform `KVLiveSpans`; BF16 and INT8-per-token/head storage are supported. c>1 metadata must be packed by the caller. | `hipengine/kvcache/policy.py`, `hipengine/kvcache/spans.py`. | Scheduler-owned allocation/admission/reclaim for multiple live requests; transactional scratch/journal for verify rows. |
| Prefix/radix cache | Not implemented in code today. `PLAN.md` mentions RadixCache as a target, but there is no runtime `RadixCache`/prefix cache implementation. | `grep RadixCache hipengine` returns no implementation. | Add only after dense c>N correctness is green; disable initially for DMS/eviction work. |
| DMS/compact KV | Planned, not active. | `docs/KVCACHE.md` Phase K2. | DMS metadata/checkpoint gate, compact allocator, streaming pack, compact attention, scheduler admission by compact live rows. |

## GPU0 diagnostic evidence

Temporary fixtures generated for the sweep:

| Fixture | Shape | Path |
| --- | --- | --- |
| 8 x 512 token rows | c≤8 512/128 diagnostics | `/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json` |
| 8 x 4096 token rows | c≤8 4K/128 diagnostics | `/tmp/hipengine-prebench/fixtures/qwen36_paro_8x4096_prompt_ids.json` |

### Primitive pre-bench

Command shape:

```bash
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
for c in 1 2 4 8; do
  python3 scripts/qwen35_batch_correctness.py \
    --rows "$c" \
    --json "/tmp/hipengine-prebench/correctness/qwen35-batch-c${c}-correctness.json"
done
```

Result:

| c | append key mismatch | append value mismatch | attention batch-vs-c1 max abs | attention batch-vs-NumPy max abs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0.0 | 0.0 |
| 2 | 0 | 0 | 0.0 | 2.2351741790771484e-08 |
| 4 | 0 | 0 | 0.0 | 2.9802322387695312e-08 |
| 8 | 0 | 0 | 0.0 | 5.960464477539063e-08 |

Interpretation: the tested BF16 batched KV append and batched paged attention
primitive wrappers are correct for rows 1/2/4/8. This is necessary, not
sufficient, for end-to-end c>N serving.

### Scheduler serial bridge sweep

Command shape:

```bash
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
MODEL=/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16
python3 scripts/qwen35_batch_serial_bench.py \
  --model "$MODEL" \
  --fixture /tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json \
  --batch-size C \
  --prompt-length 512 \
  --decode-tokens 128 \
  --warmup-decode-tokens 8 \
  --max-layers 40 \
  --kv-storage int8_per_token_head \
  --compiler-version-file /tmp/hipengine-prebench/hipcc-version.txt \
  --json /tmp/hipengine-prebench/scheduler/qwen36-paro-cC-512-128-serial-bridge.json
```

All rows are `status=blocked`, `performance_claim=false`, and
`native_caware_decode=false`.

| Shape | Correctness | Prefill tok/s | Decode aggregate tok/s | Decode per-request tok/s |
| --- | --- | ---: | ---: | ---: |
| c=1 512/128 | passed | 111.91 | 102.12 | 102.12 |
| c=2 512/128 | passed | 114.24 | 102.32 | 51.16 |
| c=4 512/128 | passed | 114.11 | 101.47 | 25.37 |
| c=8 512/128 | passed | 114.01 | 100.30 | 12.54 |
| c=1 4K/128 | passed | 111.24 | 99.98 | 99.98 |
| c=2 4K/128 | passed | 111.66 | 98.82 | 49.41 |
| c=4 4K/128 | passed | 111.41 | 98.76 | 24.69 |
| c=8 4K/128 | passed | 111.18 | 98.88 | 12.36 |

Interpretation: aggregate decode tok/s stays roughly flat while per-request
tok/s falls roughly as `1/c`. That is the signature of the serial bridge, not
native batched serving.

## Benchmark eligibility gates

A c>N row is not eligible for `accepted` status until all of these pass:

1. `scripts/qwen35_batch_correctness.py --rows N` passes for the exact primitive
   families used by the runner: `append_key_mismatch=0`,
   `append_value_mismatch=0`, and `attn_batch_vs_c1_max_abs <= 1e-6`.
2. The resident batch runner emits generated-token IDs equal to N independent
   c=1 resident runs for the same fixed prompts with greedy sampling and SpecDec
   disabled.
3. The artifact records scheduler occupancy, active mask shape, graph bucket key,
   KV policy, packed-prefill status, compaction events, and whether any serial
   bridge remains.
4. Continuous-batching rows include admission/completion timestamps and
   per-request p50/p95 latency in addition to aggregate tok/s.
5. Performance summaries show both aggregate tok/s and per-request tok/s. Never
   compare c=N aggregate to c=1 without also showing aggregate/c1 and
   per-request/c1 ratios.

## Implementation checklist

### Phase C0 — keep diagnostics honest

- [x] Generate c≤8 prompt fixtures for 512/128 and 4K/128 diagnostics.
- [x] Run c=1/2/4/8 primitive correctness on GPU0.
- [x] Run c=1/2/4/8 scheduler serial bridge diagnostics and record blocked
      status.
- [ ] Add a small script or `hipengine bench` subcommand that runs the full
      diagnostic sweep without copy/paste loops.
- [ ] Ensure every diagnostic artifact clearly distinguishes:
  - `workload.native_compact_prefill`
  - `execution.batch_execution.native_compact_prefill`
  - `native_caware_decode`
  - `throughput_claim_eligible`

### Phase C1 — server and generator integration

- [x] Add a batch-capable Qwen/PARO generator path for prompt lists that owns
      scheduler request ids, physical slots, packed prefill slabs, and output
      routing. It still receives public prompt strings rather than full server
      request metadata.
- [x] Coalesce compatible non-streaming server generations into one prompt-list
      `LLM.generate()` call so the generator-owned scheduler can admit c>N rows;
      keep a grouped safety lock around the non-reentrant engine call.
- [ ] Preserve a narrow safety lock only around non-reentrant model/session
      mutation until the session is proven concurrency-safe.
- [ ] Add request IDs, enqueue/admit timestamps, finish timestamps, and output
      routing for `/v1/completions` and `/v1/chat/completions`.
- [ ] Keep `n>1` rejected until it is represented as multiple scheduler requests
      with independent outputs and accounting.

### Phase C2 — native c>N prefill/decode

- [x] Use `ResidentBatchScheduler.next_compact_prefill_slabs(...)` and
      `Qwen35ParoResidentSession.prefill_native_packed(...)` for BF16 prompt-list
      concurrent prefill in `Qwen35ParoOneTokenGenerator._generate_batch`.
- [ ] Add generated-token equality vs independent c=1 for c=2/4/8 512/128.
- [ ] Replace `step_batch_serial` in the benchmark path with a native c-aware
      decode step.
- [ ] Capture/replay decode graphs by active `C`, context bucket, active mask,
      top-k/experts, and replay length.
- [ ] Add graph-bucket cache hit/miss and replay statistics to artifacts.

### Phase C3 — kernel coverage

- [ ] Validate batched INT8 KV append/decode paths with the same gates as BF16.
- [ ] Make full-attention decode consume per-row `KVLiveSpans` for all retained
      KV storage dtypes.
- [ ] Make linear-attention recurrent/conv state updates consume `[C, ...]`
      state and active masks.
- [ ] Replace c1 selected-MoE lane assumptions with token-row to routed-lane
      mapping.
- [ ] Add grouped-by-expert execution where c=4/8 routed lanes justify it.
- [ ] Keep c=1 GEMV dispatch separate from c>N MMQ/GEMM/WMMA candidates.

### Phase C4 — KV policy and prefix cache

- [ ] Add scheduler-owned KV allocation/admission/reclaim for multiple live
      requests with dense fixed pages first.
- [ ] Keep `KVLiveSpans` as the only attention/KV-write ABI.
- [ ] Add transactional scratch/journal semantics before speculative verify rows
      can mutate canonical KV.
- [ ] Add prefix/radix caching only after dense c>N correctness is green.
- [ ] Disable prefix/radix cache initially for DMS or any policy with eviction;
      shared prefixes need per-sequence eviction overlays before they are safe.
- [ ] Port compact DMS after dense c>N is stable, following [`KVCACHE.md`](KVCACHE.md).

## Remaining punchlist to vLLM-style c>N continuous batching

This is the working punchlist for the target pipeline. The order matters:
first make the basic c>N pipeline correct and observable, then make the green
path fast enough to scale against c=1 prefill and decode.

### Definition of done

| Milestone | Done when |
| --- | --- |
| Basic native c>N correctness | Full 40-layer Qwen/PARO dense fixed-page BF16 c=2/4/8 512/128 emits generated token IDs equal to independent c=1 runs, with no serial decode bridge and `throughput_claim_eligible=true` only after that gate passes. |
| Basic continuous batching | One long-lived scheduler loop can admit new HTTP requests while other requests decode, interleave chunked prefill with decode, finish/reclaim/compact slots independently, and route both streaming and non-streaming outputs by request id. |
| Performant c>N | Accepted artifacts show aggregate prefill and decode scaling versus the c=1 baseline and the serial bridge, with aggregate/c1 and per-request/c1 ratios, latency p50/p95, occupancy, graph-bucket stats, and profiler evidence. |

### A. Correctness and basic implementation first

- [ ] Remove stale compatibility glue once the guarded native API is settled
      (for example the `batch_execution_metadata(... )` `TypeError` shim in the
      generator).
- [ ] Add CPU-runnable structural tests for the current guardrails:
  - `step_batch_native` raises unless
    `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1`.
  - sparse/non-contiguous slots are rejected.
  - INT8 KV and long-context split-K native decode are rejected until wired.
  - metadata reports `throughput_claim_eligible=false` for guarded diagnostics.
- [ ] Add HIP-guarded reduced-shape equality diagnostics that do **not** require
      full 40 layers, so failures can be bisected in CI/dev environments with
      ROCm. Keep full 40-layer 512/128 as the retained benchmark gate.
- [ ] Triage the current native decode equality failures before enabling the
      path by default:
  - compare c=2 native vs independent c=1 after every layer for L1/L3/L8/full
    model slices;
  - isolate linear-attention state update, full-attention KV append/decode,
    MoE routing/selected-lane mapping, shared expert, O projection, and LM-head
    sampling;
  - verify `_batch_full_spans` block-table layout against the exact
    `qwen35_paged_full_attn_decode_context_bf16_batch_spans` cache addressing;
  - audit scratch aliasing and row views so row 0 cannot overwrite row 1
    temporaries or vice versa.
- [ ] Make native BF16 compact-slot decode correct for the smallest retained
      scope: dense fixed-page KV, compact physical slots `0..C-1`, context
      `<1024`, greedy sampling, SpecDec disabled.
- [ ] Extend native decode correctness to non-compact slots after scheduler
      compaction/reclaim moves requests.
- [ ] Add row-aware split-K full-attention decode/reduce before any long-context
      native c>N claim (`max_context >= 1024`).
- [ ] Add INT8-per-token/head native c>N append/decode coverage after BF16 is
      green; require the same generated-token equality gate.
- [ ] Make linear-attention conv/recurrent state updates consume `[C, ...]`
      state, active masks, and slot ids without c1 aliases.
- [ ] Replace selected-MoE c1 lane assumptions with explicit token-row to
      routed-lane mapping (`tokens=C`, `lanes=C*top_k`) and validate grouped
      by-expert metadata for c=2/4/8.
- [ ] Make batched sampling deterministic and isolated per row:
  - one row's argmax/logit buffers cannot be overwritten by another row;
  - EOS/stop-token handling is per request;
  - `n>1` remains separate scheduler requests, not one shared output stream.
- [ ] Promote the resident runner from static prompt-list batches to a
      scheduler-owned engine loop:
  - pending queue, active table, and physical slots live beyond one
    `LLM.generate()` call;
  - `next_prefill_work` and `next_decode_work` are interleaved;
  - completed requests are reclaimed without waiting for the longest request;
  - active masks, context lengths, positions, and output queues are updated at
    every commit point.
- [ ] Add scheduler-owned dense KV allocation/admission/reclaim for multiple
      live requests. The scheduler should reject or queue work based on KV page
      capacity before device allocation fails.
- [ ] Add cancellation, timeout, EOS, max-token, and client disconnect handling
      to the same completion/reclaim path used by normal generation.
- [ ] Route server streaming through the same scheduler loop instead of the
      current one-request-at-a-time path.
- [ ] Narrow or remove the coarse `generation_lock`; any remaining lock should
      protect only non-reentrant session mutation, not the whole lifetime of a
      generated batch.
- [ ] Add request-level observability: enqueue/admit/start-prefill/start-decode/
      first-token/finish timestamps, queue time, prefill time, decode time,
      tokens generated, slot moves, and final status.
- [ ] Keep prefix/radix cache, DMS/compact KV, and SpecDec disabled on the c>N
      correctness path until dense fixed-page c>N is green.

### B. Then make the green path fast

- [ ] Establish baseline artifacts before optimizing:
  - c=1 native prefill/decode for the retained shapes;
  - c=2/4/8 serial bridge diagnostics;
  - first green uncaptured native c>N rows;
  - primitive/kernel microbenchmarks for attention, KV append, MoE, projection,
    and LM-head sampling.
- [ ] Report scaling explicitly for every retained row:
  - `prefill_tok_s_aggregate / c1_prefill_tok_s`;
  - `decode_tok_s_aggregate / c1_decode_tok_s`;
  - `decode_tok_s_per_request / c1_decode_tok_s`;
  - p50/p95 first-token latency and inter-token latency;
  - active-batch occupancy over time.
- [ ] Target decode aggregate speedup versus c=1 and versus the serial bridge.
      Per [`PLAN.md`](PLAN.md), c=8 decode plausibly lands around 2-4x c=1
      aggregate when kernels reuse enough work; do not promise 8x.
- [ ] Target prefill aggregate scaling versus c=1 by keeping prompt rows packed,
      avoiding per-request Python loops, and using AOTriton/WMMA paths where they
      beat row-GEMV.
- [ ] Add hipGraph capture/replay buckets for decode by `(C, context bucket,
      active mask, KV dtype, layer plan, top_k/experts, replay length)`, with an
      uncaptured fallback for rare shapes.
- [ ] Eliminate residual serial loops on the native path after correctness is
      green:
  - full-attention per-row fallback;
  - per-row host metadata allocation/free;
  - per-row LM-head/argmax launches where a batched launch is correct;
  - Python per-layer dispatch overhead inside steady-state decode.
- [ ] Add c-aware projection dispatch thresholds:
  - c=1 stays on tuned GEMV/Marlin-K paths;
  - c=2/4/8 can use MMQ/GEMM/WMMA-style kernels when they beat row-GEMV;
  - c>16 should prefer GEMM/WMMA and grouped MoE designs over widening c1
    GEMV wrappers.
- [ ] Optimize MoE for routed-lane reuse:
  - group lanes by expert;
  - use compact/WMMA grouped kernels when routed lanes justify it;
  - measure router, group-scatter, gate/up, down, shared expert, and combine
    time separately.
- [ ] Optimize memory traffic and workspace reuse:
  - preallocate per-bucket scratch instead of allocating per step;
  - avoid host-device copies for metadata that can be updated on device;
  - keep JIT builds out of profiler runs with `require_cached`;
  - track peak allocator/KV/workspace bytes in artifacts.
- [ ] Add backpressure and fairness policies once the scheduler is continuous:
  - max active requests, max queued requests, max prefill chunk tokens;
  - prefill-vs-decode scheduling policy to protect decode latency;
  - sampling-parameter grouping without starving incompatible requests.
- [ ] Capture profiler summaries for accepted rows: expected kernel names,
      duration/share for attention, MoE, projection, sampling, graph replay, and
      any CPU-side bottleneck.
- [ ] Only update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and
      `benchmarks/results/` for retained rows with correctness green, protocol
      shape satisfied, and profiler evidence. Rejected/blocked diagnostics stay
      useful but are not scoreboard entries.

### C. After dense c>N is correct and fast

- [ ] Add prefix/radix cache with per-request ownership and invalidation; keep it
      disabled for eviction policies until prefix overlays are designed.
- [ ] Wire SpecDec/MTP verification through the same batch runner with
      transactional KV scratch/journals and accepted-token commit semantics.
- [ ] Port compact DMS/variable-span KV after dense fixed-page continuous
      batching is stable, using `KVLiveSpans` and `KVPolicy.admission_cap()` as
      the scheduler/kernel boundary.
- [ ] Revisit multi-GPU admission, TP/PP/EP, and cross-GPU KV ownership only
      after single-GPU W7900 c>N serving has retained c=2/4/8 rows.

## What not to claim yet

Do not describe any current row as:

- true c=2/4/8 serving throughput;
- continuous batching;
- radix/prefix-cache reuse;
- compact/DMS KV serving;
- c-aware decode graph replay.

The correct phrasing for current diagnostics is:

> c>N scheduler serial bridge diagnostic: batch-shaped slots and KV metadata,
> but active rows execute serially through the c=1 layer path. Aggregate decode
> throughput remains roughly c=1, so the row is blocked/non-retained.
