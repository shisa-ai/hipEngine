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

The current server and public Qwen/PARO generator are safe for local use, but
they serialize generation. Several lower-level c>N primitives are already green,
and the scheduler has batch-shaped metadata, but the end-to-end path still uses
a serial bridge for concurrent diagnostics.

## Readiness matrix

| Layer | Current status | Evidence / code | Blocks true c>N |
| --- | --- | --- | --- |
| OpenAI server | HTTP generation is serialized behind `generation_lock`; `n>1` is rejected. | `hipengine/server/api.py:create_app`, `_validate_generation_request`. | Replace one-request-at-a-time generation calls with scheduler admission/output routing. |
| Public `LLM.generate()` | Accepts an iterable of prompts, but Qwen/PARO loops prompts in Python. Streaming is one prompt only. | `hipengine/generation/qwen35_paro.py:Qwen35ParoOneTokenGenerator.generate/stream`. | Add a batch-capable generator/session that owns an active request table. |
| Scheduler | `ResidentBatchScheduler` exists with pending/admitted queues, slots, active masks, compact prefill slabs, decode work, graph bucket keys, and completion routing. | `hipengine/generation/batch_scheduler.py`. | Wire it into the public generator and server; add latency/occupancy accounting. |
| Prefill | Single-request native prefill is active. Packed prefill helpers and slab metadata exist. | `Qwen35ParoResidentSession.prefill_native`, `prefill_native_packed`, `ResidentBatchScheduler.next_compact_prefill_slabs`. | Make the c>N generator actually use packed prefill and validate generated-token equality. |
| Decode runtime | c>N diagnostic uses `step_batch_serial`: batch-shaped slots/KV, but row execution is serial c=1. | `Qwen35ParoResidentSession.step_batch_serial`, `batch_execution_metadata`. | Native c-aware decode graph replay and per-layer c>N kernels. |
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

- [ ] Add a batch-capable Qwen/PARO generator entry point that accepts request
      metadata, not just a tuple of prompt strings.
- [ ] Wire server generation into `ResidentBatchScheduler` instead of holding
      `generation_lock` for the full request.
- [ ] Preserve a narrow safety lock only around non-reentrant model/session
      mutation until the session is proven concurrency-safe.
- [ ] Add request IDs, enqueue/admit timestamps, finish timestamps, and output
      routing for `/v1/completions` and `/v1/chat/completions`.
- [ ] Keep `n>1` rejected until it is represented as multiple scheduler requests
      with independent outputs and accounting.

### Phase C2 — native c>N prefill/decode

- [ ] Use `ResidentBatchScheduler.next_compact_prefill_slabs(...)` and
      `Qwen35ParoResidentSession.prefill_native_packed(...)` for concurrent
      prefill.
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
