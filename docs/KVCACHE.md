# KV Cache Roadmap — Dense INT8 First, Compact DMS Next

_Status: K1 dense INT8 KV landed as a diagnostic/capacity path; 256K now passes both sampled and tracked 24GiB-class capacity targets after prefill buffer lifetime reductions and AOTriton query-scratch reuse. Correctness-preserving removal of the transient BF16 INT8-prefill oracle is deferred as future work. K2 compact DMS remains planned. Last updated: 2026-05-18._

This document is the focused plan for extending hipEngine's KV-cache stack past
current dense BF16 paged KV. It turns the current 128K-under-24GiB milestone
into a two-step roadmap:

1. **Dense paged INT8 KV with no BF16 shadowing** — capacity feature first. The
   goal is to make 256K context fit in the same 24GiB-class budget that now fits
   128K, while preserving exact dense-KV semantics except for quantization error.
2. **FastDMS-derived compact DMS** — algorithmic live-token reduction. The goal
   is to reduce the number of KV rows stored and scanned, using the large body
   of implementation and optimization work in `~/FastDMS` as the reference.

HIGGS/AQUA remain research-only for now. HIGGS was tabled in FastDMS because the
best serving path only reached about 50% of BF16/FP8 speed on RTX PRO 6000; on
RDNA3/gfx1100, LUT-style non-uniform quantization is not the first capacity or
speed lever.

## Current baseline and memory math

Current single-request Qwen3.5/PARO long-context evidence on **AMD Radeon Pro
W7900 / gfx1100**, model `Qwen3.5-35B-A3B-PARO`, quant `w4_paro`:

- 128K/128 dense BF16 KV with long-prefill chunking is the current baseline:
  `1021.180` prefill tok/s, `63.299` decode tok/s, sampled HIP VRAM peak
  `22.410 GiB`, tracked allocator peak `23.288 GiB`, retained KV
  `2.690 GB`.
- 128K/128 dense INT8 KV is **not a speed win** in the retained diagnostic row:
  `1011.064` prefill tok/s (`-0.99%`) and `61.275` decode tok/s (`-3.20%`).
  It is a storage/capacity feature: sampled HIP VRAM peak drops to
  `21.170 GiB` and retained KV drops to `1.355 GB`, while tracked allocator
  peak rises to `24.545 GiB` because the current INT8 prefill path still uses a
  temporary BF16 oracle K/V workspace before releasing it for decode.
- 256K/128 dense INT8 KV **runs and passes correctness** and now passes both
  sampled and tracked 24GiB-class capacity targets after replacing persistent
  full-prompt prefill double-buffering, releasing decode/phase scratch before
  bulk prefill, reusing AOTriton BF16 query scratch, and retaining a 3072-row
  full-attention query chunk: `651.636` prefill tok/s, `40.827` decode tok/s,
  sampled HIP VRAM peak `22.013 GiB`, tracked allocator high-water
  `23.766 GiB`, retained KV `2.708 GB`. The previous persistent
  `prefill_hidden`/`prefill_next_hidden` blocker is resolved; the transient BF16
  INT8-prefill oracle workspace still exists, with correctness-preserving
  removal deferred as future work.

Artifacts:

- 128K BF16-vs-INT8 diagnostic:
  [`benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-128k-quality-perf-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-128k-quality-perf-diagnostic.json)
- 128K/256K INT8 AOTriton query-reuse + q3072 diagnostic:
  [`benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-aotriton-query-reuse-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-aotriton-query-reuse-diagnostic.json)
- Superseded scratch-release diagnostic:
  [`benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-scratch-release-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-scratch-release-diagnostic.json)
- Superseded 256K single-buffer capacity diagnostic:
  [`benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-256k-single-buffer-capacity-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-256k-single-buffer-capacity-diagnostic.json)
- Superseded 256K blocked attempt:
  [`benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-256k-capacity-blocked.json`](../benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-256k-capacity-blocked.json)

For Qwen3.5/PARO, only the 10 full-attention layers own a dense KV cache:

```text
per-token BF16 KV bytes =
  10 full-attn layers * 2 KV heads * 256 head_dim * 2(K,V) * 2 bytes
= 20,480 bytes/token ≈ 20 KiB/token
```

Approximate retained KV sizes before allocator padding; INT8 includes measured
FP16 per-token/per-head K/V scale metadata:

| Context | BF16 retained KV | INT8 retained KV | Delta |
| ---: | ---: | ---: | ---: |
| 128K | `2.690 GB` | `1.355 GB` | `1.334 GB` saved (`-49.6%`) |
| 256K | ~`5.37 GB` projected | `2.708 GB` measured | ~`2.66 GB` saved |

Therefore 256K INT8 KV has roughly the same retained KV footprint as 128K BF16
KV. That is why dense INT8 KV remains the direct path to a 256K capacity row.
The caveat is strict: this only holds for retained KV if the implementation does
**not** keep a persistent BF16 shadow/staging arena. The current K1 path meets
that no-shadow rule and no longer keeps full-prompt prefill I/O buffers live
through decode. Its tracked high-water is under the 24GiB-class target; true
removal of the transient BF16 INT8-prefill oracle is deferred because direct
retained-INT8 prefill streaming failed the full E2E gate.

## Non-negotiable design rules

- **No BF16 shadowing for INT8 KV.** Persistent KV storage is INT8 plus compact
  scale metadata. Short-lived chunk-local BF16 tensors during prefill are allowed
  only when they are not retained after the chunk/layer finishes.
- **`KVLiveSpans` remains the ABI.** Dense INT8 fills uniform spans; DMS fills
  per-head variable spans. Attention kernels do not receive scalar
  `(block_table, context_len)` shortcuts.
- **Storage dtype and eviction policy are independent axes.** `paged_int8`,
  `dms_int8`, and future `dms_fp8`/`dms_int4_shadow` are policy registrations,
  not engine branches.
- **Capacity claims need memory audits.** Every retained row must record tracked
  allocator peak, sampled VRAM, KV bytes/shape, and evidence that no BF16 KV
  shadow is allocated.
- **Quality gates come before speed claims.** New/ported KV paths must pass the
  repository KL/top-1 gate and generated-token fixtures before performance rows
  are promoted.

## Phase K1 — Dense paged INT8 KV, no shadow

### Goal

Make the paged-KV path support `storage_dtype=int8_per_token_head` so 256K can
fit in the 24GiB-class envelope. Treat speed as a bonus; parent notes already
found dense INT8 KV neutral/negative at 32K and only marginal at 128K.

K1 implementation status (2026-05-18): the storage policy, writer, grouped-GQA
INT8 decode path, E2E correctness gate, no-shadow audit, and 128K/256K benchmark
artifacts are landed. 256K passes sampled and tracked 24GiB-class capacity
targets after single-buffer prefill staging, decode/phase scratch release,
AOTriton BF16 query reuse, and q3072 full-attention prefill chunks. The temporary
BF16 INT8-prefill oracle workspace still exists; correctness-preserving removal
is deferred to future work rather than a K1 capacity blocker.

### Storage format

Initial format:

```text
K cache:      int8 [layers, blocks/pages, block_size, kv_heads, head_dim]
V cache:      int8 [layers, blocks/pages, block_size, kv_heads, head_dim]
K scale:      fp16 or fp32 [layers, tokens/pages, kv_heads]
V scale:      fp16 or fp32 [layers, tokens/pages, kv_heads]
spans:        KVLiveSpans(storage_dtype=int8_per_token_head)
```

Preferred first scale granularity: **per token, per KV head, separate K/V
scales**. For Qwen3.5/PARO this is small enough:

```text
256K * 10 layers * 2 KV heads * 2(K,V) * 2 bytes(fp16 scale)
≈ 20 MiB scale metadata
```

Per-channel scales are not a first target because they erase much of the memory
win. Per-page scales can be tested later if per-token scales cost too much in
the decode producer.

### Kernels and host surfaces

1. `paged_kv_write_int8_per_token_head` — **landed**
   - Input: post-RoPE BF16/FP16 K/V rows.
   - Compute max-abs per `(row, kv_head, K/V)`, write signed INT8 row and scale.
   - Update the same dense/uniform `KVLiveSpans` fields used by BF16.
   - Public wrappers: `qwen35_write_paged_kv_int8_per_token_head_spans(...)`,
     `qwen35_write_paged_kv_int8_per_token_head_{prompt,batch}_spans(...)`.
2. `paged_attn_decode_int8_gqa_splitk` — **landed**
   - Load INT8 K/V and FP16/FP32 scales directly.
   - Accumulate QK in FP32; apply softmax/reduce in the retained split-K/GQA
     shape; dequantize V inside the producer/reduce path.
   - Avoid a separate INT8→BF16 cast kernel and avoid a BF16 cache-sized decode
     workspace.
   - Public wrappers: `qwen35_paged_attn_decode_int8_gqa_splitk_spans(...)`,
     `qwen35_paged_attn_decode_int8_gqa_splitk_gate_{bf16,fp16}_spans(...)`.
3. `paged_attn_prefill_int8_oracle_path` — **landed as diagnostic bridge**
   - For correctness and AOTriton parity, prefill computes full-attention from a
     temporary BF16 oracle K/V workspace, then appends retained INT8 K/V plus
     FP16 per-token/head scales.
   - The workspace is reused across full-attention layers and released before
     decode. It is not a persistent BF16 KV shadow, but it does raise the
     allocator high-water mark. A fully streaming/native INT8 prefill path is
     the next memory optimization if 256K+ runs OOM during prefill.
4. Policy/registry plumbing — **landed**
   - `FixedPagedKVPolicy` accepts `storage_dtype="int8_per_token_head"` with
     `scale_dtype="fp16"` and `scale_granularity="per_token_head"`.
   - Kernel keys remain `(backend="hip_gfx1100", layer="paged_attn_decode",
     quant/storage="int8_per_token_head", variant="gqa_splitk")`.

### Acceptance gates

Minimum correctness:

- Unit fixture: quantize/dequantize K/V edge cases, scale zero handling, page
  boundary writes, and `KVLiveSpans` bounds.
- Attention fixture: BF16 dense vs INT8 dense at short and long contexts;
  require KL ≤ 0.05 and top-1 ≥ 90%.
- End-to-end fixed prompt: generated-token equality where deterministic equality
  is expected; otherwise repository KL/top-1 gate.

Minimum capacity/perf evidence:

- 128K/128 BF16 dense baseline and 128K/128 INT8 dense row.
- 256K/128 INT8 dense row under the 24GiB-class target, or a blocked artifact
  explaining the exact allocation that prevented it.
- `rocprofv3 --kernel-trace` evidence that the INT8 decode kernel ran.
- Memory audit showing no persistent BF16 K/V cache or full-cache BF16 staging
  tensor exists after prefill.

Promotion policy:

- Do **not** make INT8 KV the default for short contexts if 4K/32K decode
  regresses. Default can be shape/memory gated: BF16 below the long-context
  threshold, INT8 when admission would otherwise exceed the budget.
- Promote the 256K row even if speed is neutral, if quality passes and memory
  stays under target. Capacity is the primary deliverable.

### K1 measured protocol and results

Common benchmark context: model `Qwen3.5-35B-A3B-PARO` from
`/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd`,
quant `w4_paro`, backend `hip_gfx1100`, hardware **AMD Radeon Pro W7900 /
gfx1100**, HIP `7.2.53211-d40244d`, token id `9707`, `max_layers=40`,
AOTriton prefill threshold `512`, graph replay decode, and long-context chunks
`linear=1024`, `moe=1024`, `full_attn_query=4096`, `full_attn_post=1024`,
`full_attn_rope=1024`.

Exact benchmark commands:

```bash
# 128K/128 BF16 dense baseline
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 131072 \
  --decode-tokens 128 --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-task15-128k-int8-kv/hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
  --prefill-full-attn-query-chunk-size 4096 \
  --prefill-full-attn-post-chunk-size 1024 \
  --prefill-full-attn-rope-chunk-size 1024 --kv-storage bf16 \
  --json /tmp/hipengine-task15-128k-int8-kv/qwen35-paro-128k128-bf16-rerun.json

# 128K/128 INT8 dense KV
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 131072 \
  --decode-tokens 128 --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-task15-128k-int8-kv/hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
  --prefill-full-attn-query-chunk-size 4096 \
  --prefill-full-attn-post-chunk-size 1024 \
  --prefill-full-attn-rope-chunk-size 1024 \
  --kv-storage int8_per_token_head \
  --json /tmp/hipengine-task15-128k-int8-kv/qwen35-paro-128k128-int8-rerun.json

# 256K/128 INT8 dense KV AOTriton query-reuse + q3072 diagnostic
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 262144 \
  --decode-tokens 128 --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-task19-aotriton-query-reuse/hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
  --prefill-full-attn-query-chunk-size 3072 \
  --prefill-full-attn-post-chunk-size 1024 \
  --prefill-full-attn-rope-chunk-size 1024 \
  --kv-storage int8_per_token_head \
  --json /tmp/hipengine-task19-aotriton-query-reuse/qwen35-paro-256k128-int8-aotriton-query-reuse-q3072.json
```

| Row | Status | Prefill tok/s | Decode tok/s | Sampled VRAM peak | Tracked peak | Retained KV | Correctness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/128 BF16 KV | baseline diagnostic | `1021.180` | `63.299` | `22.410 GiB` | `23.288 GiB` | `2.690 GB` | benchmark preview matched INT8 seed/final token IDs |
| 128K/128 INT8 KV | latest memory diagnostic, not speed claim | `1035.606` | `60.992` | `19.851 GiB` | `20.941 GiB` | `1.355 GB` | E2E fixture `max_kl=0.015328`, top-1 `100%`, generated IDs match; no BF16 shadow |
| 256K/128 INT8 KV | retained capacity diagnostic; sampled+tracked 24GiB targets pass, oracle-removal follow-up remains | `651.636` | `40.827` | `22.013 GiB` | `23.766 GiB` | `2.708 GB` | same E2E fixture gate passes; no BF16 shadow |

Profiler evidence for the 128K INT8 row used `rocprofv3 --kernel-trace
--selected-regions true`:

- Prefill selected region: `qwen35_write_paged_kv_int8_per_token_head_kernel<_Float16>`
  ran `320` calls, average `54.848 us`, max `69.761 us`, `Scratch_Size=0`.
- Decode selected region sampled 16 measured graph replays from the 128-token
  workload: `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_kernel<_Float16,8,16,2>`
  ran `160` calls, average `621.500 us`, max `641.850 us`; reduce-gate ran
  `160` calls, average `159.271 us`; decode append INT8 writer ran `160` calls,
  average `4.363 us`. All reported `Scratch_Size=0`.

Immediate memory work after the AOTriton query-reuse + q3072 diagnostic:

1. [x] Replace the persistent full-prompt prefill hidden/next-hidden double
   buffer. The old `2 x [262277,4096] fp16` live-through-decode blocker is gone;
   sampled HIP VRAM peak dropped `24.330 -> 22.013 GiB`.
2. [x] Release token-1 decode scratch before bulk prefill and release
   linear-prefill scratch before entering full-attention prefill. This dropped
   128K tracked peak `24.545 -> 21.525 GiB` and 256K tracked peak
   `24.699 -> 24.351 GiB` versus the previous INT8 rows.
3. [x] Reuse caller-owned AOTriton BF16 query scratch and retain q3072
   full-attention query chunks. This drops tracked peak further to `20.941 GiB`
   at 128K and `23.766 GiB` at 256K.
4. [ ] Future project: remove or stream the temporary BF16 INT8-prefill oracle
   K/V without regressing the E2E gate. It is already reused and released before
   decode, and 256K total tracked high-water is below 24GiB, but the oracle
   workspace itself still exists.
5. Keep dense INT8 KV/scales as expected capacity cost: 256K retained payload is
   `2.687 GB` at `1.0 B/element` plus `20.992 MB` of FP16 scales.

## Phase K2 — FastDMS-derived compact DMS

### Goal

After dense INT8 KV lands, port compact DMS semantics from `~/FastDMS` so the
engine stores and scans fewer live tokens. DMS is the better long-context and
concurrency lever because it reduces `live_counts`, not just bytes per live row.

DMS is checkpoint-dependent. It is not a drop-in policy for arbitrary models;
Qwen3.5/PARO needs a DMS-retrofitted checkpoint or a validated borrowed-channel
metadata block before DMS rows can be quality claims.

### FastDMS reference map

Use `~/FastDMS` as the semantic and optimization reference, but port to
hipEngine's torch-free HIP/plugin design rather than copying Triton/PyTorch
host code directly.

| FastDMS file | What to reuse |
| --- | --- |
| `fastdms/engine/dms.py` | DMS metadata loading, borrowed-query-channel eviction extraction, alpha scale/offset semantics, and zeroing the decision lane after extraction. |
| `fastdms/engine/compact_kv.py` | Compact allocator, per-layer/per-head `base_offsets`, `range_capacity`, `live_counts`, `token_positions`, `evict_mask`, streaming prefill pack, live-count/rank/scatter structure. |
| `fastdms/layers/compact_attention.py` | Fused decode preprocessing, compact append/store, inline Q RoPE option, grouped split-K compact attention, split-block tuning knobs. |
| `fastdms/engine/scheduler.py` | Admission through compact capacity instead of dense pages; releasing dense blocks after pack in non-streaming modes; streaming-pack mode with no dense blocks. |
| `fastdms/models/qwen3.py` | Qwen DMS integration points: extraction from Q, per-layer eviction recorder, fused preprocess eligibility. |
| `~/FastDMS/training/` | Retrofit recipe: neuron zeroing, DMS distillation, target compression ratio, window size, and metadata packaging. |

FastDMS performance evidence to keep in mind:

- Compact DMS was faster than vLLM BF16/FP8 on Llama-3.2-1B and Qwen3-8B in
  the validated c=1/c=8 rows while using much less allocator-visible KV memory.
- The strongest research compression stack was DMS + AQUA + HIGGS at 25.6×
  theoretical KV compression, but HIGGS speed did not hold; FastDMS promoted
  compact DMS without HIGGS/AQUA for the serving path.
- Streaming pack was important because it eliminates a persistent dense KV
  scratch. hipEngine should start with the streaming/no-shadow shape, not a
  sidecar compact cache that still reserves dense pages.

### hipEngine DMS shape

DMS should register as a `KVPolicy` and compact attention kernel family:

```python
policy = KVPolicy.dms_int8(
    target_cr=4 or 8,
    window_size=256,
    storage_dtype="int8_per_token_head",
)
```

Core metadata is already aligned with `KVLiveSpans`:

```text
base_offsets    [rows, layers, kv_heads] int32
live_counts     [rows, layers, kv_heads] int32
range_capacity  [rows, layers, kv_heads] int32 (policy-owned)
token_positions [rows, layers, kv_heads, max_live] int32
evict_mask      [rows, layers, kv_heads, max_live] bool
storage_dtype   int8_per_token_head initially
span_role       prefill | decode | verify_chain | verify_tree
```

### Bring-up sequence

1. **DMS metadata and training checkpoint gate**
   - Add `DMSRetrofitConfig` loader for `dms_metadata.json` / training-log style
     metadata.
   - Require explicit opt-in if metadata is missing; no silent DMS on a
     non-retrofitted checkpoint.
   - For Qwen3.5/PARO, train or import an eviction-head retrofit before any
     quality claim.
2. **Compact policy and admission**
   - Add `DMSKVPolicy` with allocator-visible compact capacity.
   - `admission_cap()` returns compact live-token capacity, not logical context
     length.
   - Add no-evict and forced-stride diagnostic modes only for testing the
     compact allocator/kernels; they are not quality claims.
3. **Streaming prefill pack**
   - Port FastDMS' count/rank/scatter structure to HIP.
   - Pack surviving K/V directly into compact INT8 storage after each full
     attention layer/chunk.
   - Do not retain a dense BF16 KV arena after pack.
4. **Decode append/preprocess**
   - Port fused Q/K RoPE + DMS decision extraction + compact INT8 store.
   - Zero the borrowed query decision lane before attention, matching FastDMS.
   - Update `live_counts`, `token_positions`, and `evict_mask` transactionally.
5. **Compact grouped split-K attention**
   - Port compact decode over variable `live_counts`.
   - Reuse the grouped-GQA lesson: scan each KV stream once for all Q heads that
     share it when split geometry makes reuse worthwhile.
   - Tune block-N/split caps only after correctness fixtures pass.
6. **Scheduler and c=N integration**
   - Start c=1, then c=2/4/8 after dense batched spans are green.
   - Continuous batching must account by actual compact live rows. Prefix cache
     should be disabled initially or implemented as per-sequence eviction
     overlays; do not share evicted prefix pages blindly.
7. **Speculative decode compatibility**
   - DMS writes must obey existing KV transaction semantics. Verify rows write
     scratch/journal spans and commit only accepted rows.

### DMS acceptance gates

Correctness/quality:

- DMS-off/no-evict compact mode equals dense reference.
- DMS-on mode passes KL ≤ 0.05 and top-1 ≥ 90% against no-evict/full-KV on the
  fixture set.
- Add a longer PPL/logit-distillation smoke for the DMS-retrofitted checkpoint;
  record token-match/KLD over scored decode tokens like FastDMS did.
- Forced accept/reject speculative fixtures remain isolated from canonical KV.

Capacity:

- Report logical context length, average and max `live_counts`, target vs actual
  compression ratio, compact KV bytes, scale metadata bytes, and allocator peak.
- DMS rows must demonstrate allocator-visible savings, not only masked attention
  over a dense pool.

Performance:

- Compare against dense BF16 and dense INT8 at 128K and 256K.
- Record producer, split-reduce, store/pack, and scheduler/admission time shares.
- Do not promote if compact attention is slower without a compensating capacity
  objective clearly stated.

Soak/stability:

- Include a c=1 long-context soak and a c=8 serving-shaped soak once c=N support
  is available.
- Enable debug checks for early development: bounds, monotonic positions, live
  count ≤ capacity, no negative slot mappings, and no stale `evict_mask` entries.

## Later research: AQUA, HIGGS, TurboQuant-style int4

These are deliberately after dense INT8 and DMS:

| Technique | Current decision | Reason |
| --- | --- | --- |
| AQUA-KV | Research after DMS | FastDMS found it was not required for best FP8+DMS serving quality. It may help if we revisit 4-bit storage. |
| HIGGS 4-bit KV | Defer | Best FastDMS work reached about 50% BF16/FP8 speed on PRO 6000; RDNA3 LUT/Hadamard cost is unlikely to be better. |
| TurboQuant/int4 KV | Optional comparator | Useful if users need maximum capacity, but vLLM/FastDMS evidence showed 4-bit KV can be slower and worse quality than DMS FP8/INT8. |

## Immediate punchlist

1. [x] Add a dense INT8 KV storage policy and metadata structs.
2. [x] Add INT8 paged KV write with per-token/per-head scales.
3. [x] Add INT8 grouped-GQA split-K decode, no BF16 full-cache staging.
4. [x] Add memory-audit tests that fail if BF16 shadow KV is allocated.
5. [x] Run 128K/128 BF16-vs-INT8 quality/perf comparison.
6. [x] Run 256K/128 INT8 capacity row under the sampled/tracked 24GiB-class
   target: completed with correctness/no-shadow passing at sampled `22.013 GiB`;
   tracked allocator high-water is `23.766 GiB`.
7. [x] Reduce persistent full-prompt prefill I/O buffers; the previous
   `prefill_hidden`/`prefill_next_hidden` live-through-decode blocker is gone.
8. [x] Release decode/phase scratch around bulk prefill to reduce tracked
   high-water without changing retained KV format.
9. [x] Reuse AOTriton prefill query scratch and retain q3072 full-attention
   chunks to bring 256K tracked high-water below the 24GiB-class target.
10. [ ] Stream or remove the temporary BF16 INT8-prefill oracle workspace itself.
11. [ ] Port FastDMS DMS metadata loader and compact allocator semantics.
12. [ ] Train/import a Qwen3.5/PARO DMS retrofit before DMS quality claims.
13. [ ] Port streaming pack and compact decode kernels to HIP.
14. [ ] Combine `dms` + `int8_per_token_head` as the first promoted compact policy.
