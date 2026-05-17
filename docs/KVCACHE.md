# KV Cache Roadmap — Dense INT8 First, Compact DMS Next

_Status: planning document. Last updated: 2026-05-17._

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

Current single-request Qwen3.5/PARO long-context evidence:

- 128K/128 with long-prefill chunking fits below the 24GiB deployment envelope:
  **23.656 GiB** tracked peak in `docs/PREFILL.md`.
- Decode at 128K is already attention-dominated; `docs/ROOFLINE.md` records the
  grouped-GQA producer as the first long-context bucket.
- Parent dense BF16 128K/128 source-lineage row was **27.42 GiB**, so the
  current chunked path is already materially better than the parent on memory.

For Qwen3.5/PARO, only the 10 full-attention layers own a dense KV cache:

```text
per-token BF16 KV bytes =
  10 full-attn layers * 2 KV heads * 256 head_dim * 2(K,V) * 2 bytes
= 20,480 bytes/token ≈ 20 KiB/token
```

Approximate arena sizes before allocator padding and scale metadata:

| Context | BF16 KV | INT8 KV | Delta |
| ---: | ---: | ---: | ---: |
| 128K | ~2.50 GiB | ~1.25 GiB | ~1.25 GiB saved |
| 256K | ~5.00 GiB | ~2.50 GiB | ~2.50 GiB saved |

Therefore 256K INT8 KV should have roughly the same raw KV footprint as 128K
BF16 KV. That is why dense INT8 KV is the direct path to a 256K capacity row.
The caveat is strict: this only holds if the implementation does **not** keep a
persistent BF16 shadow/staging arena.

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

Make the default paged-KV path support `storage_dtype=int8_per_token_head` so
256K can fit in the 24GiB-class envelope. Treat speed as a bonus; parent notes
already found dense INT8 KV neutral/negative at 32K and only marginal at 128K.

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

1. `paged_kv_write_int8_per_token_head`
   - Input: post-RoPE BF16/FP16 K/V rows.
   - Compute max-abs per `(row, kv_head, K/V)`, write INT8 row and scale.
   - Update the same dense/uniform `KVLiveSpans` fields used by BF16.
2. `paged_attn_decode_int8_gqa_splitk`
   - Load INT8 K/V and scales directly.
   - Accumulate QK in FP32; apply softmax/reduce in the retained split-K/GQA
     shape; dequantize V inside the producer/reduce path.
   - Avoid a separate INT8→BF16 cast kernel and avoid a BF16 cache-sized
     workspace.
3. `paged_attn_prefill_int8_oracle_path`
   - For initial correctness, prefill can still compute attention from the
     chunk-local BF16 K/V before quantized append. The retained KV after prefill
     must be INT8 only.
   - A fully INT8 prefill-attention path is optional; the capacity issue is
     retained KV, not temporary chunk math.
4. Policy/registry plumbing
   - `KVPolicy.paged_int8(scales="per_token_head")` or equivalent registered
     policy.
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

1. Add a dense INT8 KV storage policy and metadata structs.
2. Add INT8 paged KV write with per-token/per-head scales.
3. Add INT8 grouped-GQA split-K decode, no BF16 full-cache staging.
4. Add memory-audit tests that fail if BF16 shadow KV is allocated.
5. Run 128K/128 BF16-vs-INT8 quality/perf comparison.
6. Run 256K/128 INT8 capacity row under the 24GiB-class target.
7. Port FastDMS DMS metadata loader and compact allocator semantics.
8. Train/import a Qwen3.5/PARO DMS retrofit before DMS quality claims.
9. Port streaming pack and compact decode kernels to HIP.
10. Combine `dms` + `int8_per_token_head` as the first promoted compact policy.
