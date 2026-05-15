# hipENGINE Batched Prefill Plan

> Status: implementation plan. This doc is the authoritative punchlist for
> moving hipENGINE prefill from "c=1 serial token replay" to "bulk native
> prefill" parity with the `~/amd-gpu-tuning` PARO native engine, and then to
> compact c>N prompt batching.

## TL;DR

**The gap is not kernels, it is orchestration.**

- We have most of the gfx1100 kernels needed (see `docs/KERNELS.md`).
- We do **not** have a torch-free bulk prefill orchestrator: every prompt
  token currently goes through the c=1 decode path, layer by layer, position
  by position.
- The parent does bulk prefill: one batched `[T, hidden]` tensor flows through
  every layer, with chunked native conv / GDN / RoPE / KV-append / causal
  attention / grouped MoE. That is why parent gets ~2500 tok/s prefill and we
  get ~100 tok/s on the same model.

Punchlist sections [A](#a-foundation--bulk-buffers-and-positions) through
[H](#h-graph-capture-and-step-replay) translate the parent reference into
hipENGINE-shaped tasks. **A through F are required for correctness/throughput
parity. G/H are finishing.**

## Evidence: where the gap is

Parent native engine retained rows (Qwen3.5-35B-A3B-PARO, W7900, BF16/FP16
activations, W4 PARO weights):

| Shape       | Prefill tok/s | Decode tok/s | Notes (`~/amd-gpu-tuning/docs/PARO.md`) |
| ---         |          ---: |         ---: | --- |
| 512 / 128   |       554.21  |       64.71  | `bench_paro_native_engine.py --prefill-mode bulk`, lm_head dense GEMV |
| 4096 / 128  |      2140.71  |       60.32  | bulk, lm_head dense GEMV, 24GB path |
| 4096 / 4096 |      2155.60  |       56.79  | bulk, lm_head dense GEMV, 24GB path |
| 512 / 32    |      2682.66  |      116.26  | parent fixture row recorded in `fixtures/qwen35_paro/parent_512_32_seed1234.json` |

hipENGINE current rows on the same fixture
(`benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`):

| Shape     | Prefill tok/s | Decode tok/s | Notes |
| ---       |          ---: |         ---: | --- |
| 512 / 32  |       117.24  |      101.68  | resident c=1 path; prompt runs as 512 sequential decode steps |
| c=8 8/1   |       115.08  |      108.89  | `scheduler_serial_slot_bridge` diagnostic, not native compact |

Conclusion: decode is roughly within ~10–15% of parent. Prefill is ~23x slower
because we do not run a bulk path at all.

## How parent does bulk prefill

Reference: `~/amd-gpu-tuning/scripts/bench_paro_native_engine.py` and
`~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant.py`.

`ParoNativeC1Model.prefill_bulk(prompt_ids)` is the entry point:

```python
hidden = embed_tokens(prompt_ids).view(-1, hidden_size)     # [T, H]
positions = self.position_ids[:T]                            # device tensor
for layer in self.layers:
    if isinstance(layer, ParoQuantLinearAttentionLayer):
        hidden = layer.prefill_native(hidden, linear_states[layer.id])
    else:
        hidden = layer.prefill_native(hidden, full_caches[layer.id], positions=positions)
return lm_head(final_norm(hidden[-1:]))                      # only last row
```

Per-layer responsibilities:

- **`ParoQuantLinearAttentionLayer.prefill_native`**
  - input RMSNorm (batched, T rows)
  - optional fused `paro_rotate2` over `[mixed_qkv, z]` (batched)
  - PARO projections over `[T, H]`
  - native conv prefill via `hip_qwen35_linear_attn_conv_prefill_out`
  - native GDN prefill recurrent + RMSNorm gate
  - output projection
  - post-attn RMSNorm + add
  - **grouped MoE** (sparse top-k) over all T rows
  - chunked over prompt if `tokens > NANOVLLM_PARO_PREFILL_LINEAR_CHUNK_SIZE`

- **`ParoQuantFullAttentionLayer.prefill_native`**
  - input RMSNorm (batched)
  - fused `paro_rotate3` over `[Q, K, V]` (batched)
  - PARO Q/K/V projections (batched)
  - Q/K head RMSNorm (per-head)
  - partial RoPE with per-token positions
  - KV-cache **append** (bulk, all prompt rows at their positions)
  - **causal multi-query attention** (SDPA-style with lower-right causal mask
    over prompt+cached prefix)
  - gate-mul output, o_proj
  - post-attn RMSNorm + add
  - grouped MoE
  - chunked over prompt if `tokens > NANOVLLM_PARO_PREFILL_FULL_ATTN_POST_CHUNK_SIZE`

The key environment knobs that govern bulk chunking:

| Env var | Default | Meaning |
| --- | ---: | --- |
| `NANOVLLM_PARO_PREFILL_LINEAR_CHUNK_SIZE` | 0 (off) | Split linear-attn prefill into chunks of T rows |
| `NANOVLLM_PARO_PREFILL_FULL_ATTN_QUERY_CHUNK_SIZE` | 0 | Q-row chunk for full-attn prefill |
| `NANOVLLM_PARO_PREFILL_FULL_ATTN_POST_CHUNK_SIZE` | 0 | Post-attention residual+MoE chunk |
| `NANOVLLM_PARO_PREFILL_FULL_ATTN_ROPE_CHUNK_SIZE` | 0 | RoPE/Q/K-norm inplace chunk |
| `NANOVLLM_PARO_MOE_STACKED_COMPACT` | 1 | Stacked compact MoE for prefill |
| `NANOVLLM_PARO_MOE_GROUPED_DEVICE_GATHER` | 1 | Device-side grouped scatter/gather for MoE prefill |
| `NANOVLLM_PARO_GEMV_V8` | 1 | Pack8 GEMV used by MoE/proj |
| `NANOVLLM_PARO_NATIVE_ROUTER` | 1 | Native router top-k |

These are the parent's bulk-prefill tuning knobs; hipENGINE plugin equivalents
are listed in section G.

## What we already have

Reference: `docs/KERNELS.md`.

Landed kernels (hipENGINE-landed BF16/FP16 raw-pointer wrappers, no torch on
hot path):

| Family | Kernels |
| --- | --- |
| Norm | `paro_rmsnorm_out`, `paro_add_rmsnorm_out`, `qwen35_rmsnorm`, `qwen35_add_rmsnorm`, `qwen35_head_rmsnorm` |
| Rotation | `paro_rotate1`, `paro_rotate2`, `paro_rotate3` |
| Rotary | `qwen35_partial_rotary`, `qwen35_head_rmsnorm_partial_rotary*` |
| Q/G split | `qwen35_split_qgate_{bf16,fp16}` |
| Projections | `paro_awq_selected_dual_pack8`, `paro_awq_pack8_*`, `w8a16_linear_*` |
| Rotate→proj | `selected_dual_pack8_strided_rotate_out_*` |
| MoE | `qwen35_router_logits`, `qwen35_router_select`, `qwen35_router_topk_shared_out_*`, `silu_mul_dual*`, `paro_combine` |
| Linear-attn prefill | `qwen35_linear_attn_conv_prefill_f32`, `qwen35_linear_attn_prefill_prepare_{f32_bf16,f32_fp16}`, `qwen35_gdn_prefill_recurrent_{f32,f32_k2}`, `qwen35_gdn_prefill_rmsnorm_gate_{bf16,fp16}` |
| Linear-attn decode | `qwen35_linear_attn_conv_decode_*`, `qwen35_gdn_recurrent_rmsnorm_gate_lowp_*` |
| Paged KV write | `qwen35_write_paged_kv_mixed_value_{bf16,fp16}_spans` and `batch_spans`, `f32_spans` |
| Paged attention decode | `qwen35_paged_full_attn_decode_*` (split-K, GQA, gated, span/batch_spans) |
| Full-attention decode (dense ctx) | `qwen35_full_attn_decode_context_bf16` |
| lm_head | `lm_head_fp16_argmax_bf16` |
| Conversion | `f32_to_bf16`, `bf16_to_f32`, `f32_to_fp16`, `fp16_to_f32`, `fp16_to_bf16` |

Critically present for prefill: linear-attn conv/GDN **prefill** kernels, KV
batch-spans writer, dense GEMV/lm_head, rotation/RoPE, RMSNorm.

Not landed in hipENGINE yet:

- **Causal multi-query full-attention prefill** kernel (FlashAttn-style).
  Parent uses `F.scaled_dot_product_attention` with a causal mask; we need
  either a HIP causal SDPA kernel or a labelled fallback through existing
  decode kernels (slow but correct).
- **Grouped/compact MoE prefill** (W8A16 grouped gate_up/down, group
  scatter/prefix/scatter, packed hidden gather, c=1 grouped accumulate
  variants). These are LINEAGE GREEN in `nano-vllm-amd/csrc/amd/qwen35_expert.hip`
  but not yet ported to `hipengine/kernels/hip_gfx1100/moe/`.
- **WMMA grouped GEMM** (`wmma/wmma_i8_gemm.hip`, 4 kernels) used by long
  prompt grouped MoE in parent.
- **Activation int8 quant** kernels for W8A8 grouped path. Not on the W4 PARO
  critical path but may be needed for parent-bit-equal grouped MoE.

So the gap is split:

1. Orchestration: hipENGINE never calls a bulk prefill code path. All
   prefill currently runs `Qwen35ParoResidentSession.step(token, position=p)`
   in a Python loop. **This is the single biggest cause of the 23x gap.**
2. Two missing kernel families (causal prefill attention, grouped MoE
   prefill) that are required to match parent at long-prompt shapes.

## Target architecture

Goal: torch-free, raw-pointer, plugin-registered native `prefill_native(...)`
on `Qwen35ParoResidentSession` that accepts `[T, hidden]` and runs every layer
bulk.

Hot-path invariants (`AGENTS.md` / `docs/PLAN.md`):

- No `import torch` on the prefill path. All bulk tensors are
  `hipengine.core.tensor.Tensor` with raw device pointers.
- No `if backend == ...` or `if quant == ...` branches in
  dispatch/engine/model code. Layer kinds are plugins; quant is a registry
  key.
- Every fused composite has an unfused fallback registered. Bulk prefill
  composites (`linear_attn_prefill`, `full_attn_prefill`, `moe_prefill`) must
  each have a fallback chain that runs the same math with the existing
  primitives. The unfused fallback is the correctness oracle.
- Kernel bodies take raw device pointers. Host-side wrappers do the
  `Tensor → ptr` conversion.

ABI sketch:

```python
def prefill_native(
    self,
    token_ids: Sequence[int],
    *,
    sample: bool = False,                 # if True, return the last-row argmax
    allow_rejected_correctness: bool = False,
) -> Qwen35ParoAutoregressiveStepResult | None:
    """Run one bulk prefill over [T] tokens.

    - Materializes [T, hidden] in self.prefill_hidden (already allocated).
    - Calls bulk linear_attn_prefill_layer / full_attn_prefill_layer per
      layer kind, both registered in the kernel registry.
    - Optionally samples lm_head on the last token only.
    - Leaves linear-attn recurrent state, conv state, and full-attn KV caches
      live for subsequent decode.
    """
```

This already exists today as `prefill_linear_tokens_native` but only covers a
linear-attention prefix; the full version below replaces it.

## Phased plan / Punchlist

Each phase emits at least one accepted artifact under `benchmarks/results/`
and updates `docs/KERNELS.md` if a kernel lands. Every phase has a
correctness gate vs `kernels/cpu_reference/` and the parent fixture
`fixtures/qwen35_paro/parent_512_32_seed1234.json` before any perf claim.

Note: phases A–C unlock real performance for c=1 single-request prefill.
Phase D is the speedup multiplier (true compact prompt batching).
Phases E/F are c>N batching. G/H are polish.

### A. Foundation — bulk buffers and positions

Goal: have a `[T, hidden]` device buffer, per-token position table, and
bulk-shaped scratch ready for layer-level work.

Status today: `self.prefill_hidden` exists and is sized
`max_sequence_length * hidden`. `_run_linear_prefill_layers(tokens=T)`
already exercises this buffer for linear-only prefixes.

Punchlist:

- [x] `prefill_hidden` / `prefill_next_hidden` allocated per session
  (`hipengine/runtime/qwen35_paro_runner.py`, ~lines 1431-1494).
- [x] `_restore_decode_scratch_after_prefill` keeps decode scratch consistent
  after prefill.
- [ ] Native `position_ids` table allocated at session build (length
  `max_sequence_length`, `int64`) — extend `runtime/state.hip` or reuse
  existing `set_i64_vector`.
- [ ] Native batched token-embedding lookup over T rows
  (already have `embedding_lookup_batch_*_i64`; verify it covers prefill
  shape without c=1 slot mapping).
- [ ] Add `Qwen35ParoResidentSession.prefill_native(token_ids, *, sample)`
  shell that runs linear-only today but with the right batched ABI and
  artifact labelling. Validate against `_run_linear_prefill_layers` row by row.

Gate: correctness equal to current `prefill_linear_tokens_native` on the
3-layer linear prefix; new artifact
`benchmarks/results/2026-05-XX-hipengine-qwen35-bulk-prefill-linear-prefix-accepted.json`.

### B. Bulk linear-attention prefill (per layer, native)

Goal: every `linear_attention` layer in the 40-layer model runs bulk
prefill natively (currently we stop after 3 layers).

Status today: kernels are landed (conv prefill, GDN prefill recurrent, GDN
prefill RMSNorm gate, prefill prepare). What's missing is the wired layer
orchestrator over all linear layers in the bulk hidden buffer.

Punchlist:

- [ ] Add `Qwen35ParoDecodeState.run_linear_attention_moe_c1_layer_prefill(
        hidden, state, *, tokens)` that:
  - applies `paro_rmsnorm_out_*` (already supports row-batched output);
  - calls fused `paro_rotate2` (input-rotation A/B) if available, else
    unfused `paro_rotate1` + projection (both landed);
  - calls PARO projections over `[T, qkv_width]` (already pack8-batched);
  - calls `qwen35_linear_attn_conv_prefill_f32` then `silu`;
  - calls `qwen35_gdn_prefill_recurrent` and `qwen35_gdn_prefill_rmsnorm_gate`;
  - calls `out_proj`;
  - calls bulk MoE over `[T, hidden]` (see C);
  - writes back into `[T, hidden]` in place.
- [ ] Chunked variant (`NANOVLLM_PARO_PREFILL_LINEAR_CHUNK_SIZE`-style env
  knob) for long prompts.
- [ ] Update `qwen35_paro_native_prefill_plan` to report
  `linear_attention_native` for the all-linear case and remove the
  3-layer-only artifact assumption.

Gate: hidden states match the linear-prefix golden artifact bit-by-bit
through every linear layer; CPU oracle for one randomized fixture.

### C. Bulk grouped MoE prefill (per layer)

Goal: the MoE block in every prefill layer runs as a **grouped** path over T
rows, not as T independent c=1 calls.

Status today: c=1 MoE runtime path is landed and validated; grouped MoE
prefill kernels are LINEAGE GREEN but not ported.

Required kernel ports (`docs/KERNELS.md` catalog → hipENGINE-landed):

- [ ] `qwen35_router_logits_*` / `qwen35_router_select` over `[T, hidden]`.
  Router top-k already lands for BF16 hidden; verify the wrapper accepts T
  rows.
- [ ] **Port** `moe/group_scatter.hip` (11 kernels: count, prefix, scatter,
  scatter_gather, c1_group_metadata variants, gather_packed_hidden,
  build_lane_to_sorted, combine).
- [ ] **Port** `quant/w8a16_moe.hip` shared-expert / grouped variants
  (`w8a16_shared_gate_up_bulk4`, `w8a16_shared_down_bulk_combine`,
  `w8a16_shared_down_bulk_combine_w8a8_c1_selected`). Only the variants
  actually used by Qwen3.5/PARO shared+selected; do not port all 17 unless
  needed.
- [ ] **Port** `moe/w8a8_grouped.hip` only if Qwen3.5-PARO grouped path needs
  it; the W4 PARO path may not.
- [ ] **Port** `wmma/wmma_i8_gemm.hip` for long-prompt grouped GEMM. This is
  optional for first parity; the pack8 grouped path is already
  parent-equivalent at 512-tile shapes.
- [ ] Wire `Qwen35ParoDecodeState.run_moe_c1` into a `run_moe_bulk(T)`
  variant that uses the grouped scatter/gather kernels instead of per-token
  selected GEMV.
- [ ] Register the bulk MoE composite via the four-axis registry:
  `(hip_gfx1100, moe_prefill, w4_paro, qwen35)` with a fallback that loops
  `run_moe_c1` over rows.

Gate: bulk MoE output equals row-by-row `run_moe_c1` output within atol
that matches the parent grouped-vs-ungrouped tolerance
(`~/amd-gpu-tuning/docs/PARO.md` 2026-05-09 grouped device-gather row).
Emit `benchmarks/results/2026-05-XX-hipengine-qwen35-bulk-moe-prefill-accepted.json`.

### D. Bulk full-attention prefill (per layer, native)

This is the single biggest correctness/perf step. Without this, the model
falls back to serial c=1 for 10 of 40 layers, and bulk-prefill tok/s does not
go up.

Reference: blocker artifact
`benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-attn-boundary-blocked.json`
already enumerates the 5 components. Each is a punchlist item below.

Punchlist:

- [ ] **`full_attention_prefill_orchestrator`** in
  `hipengine/runtime/qwen35_paro.py`:
  - replace `tokens != 1` guards in
    `Qwen35ParoDecodeState.run_full_attention_moe_c1_layer_fp16` with a
    bulk path. Keep the c=1 path as a labelled fallback when T==1.
- [ ] **`full_attention_qkv_projection_layout`**:
  - `project_full_attention_qkv_fp16(T)` returns contiguous Q `[T, num_q, hd*2]`,
    K `[T, num_kv, hd]`, V `[T, num_kv, hd]` views suitable for downstream
    RoPE/KV-write.
- [ ] **`full_attention_rope_prepare_positions`**:
  - `prepare_full_attention_qkv_fp16(T, positions)` runs head RMSNorm + partial
    rotary using the per-token `positions` tensor instead of a single decode
    position scalar. The fused `qwen35_head_rmsnorm_partial_rotary_position_f32_bf16`
    kernel already takes positions; verify it covers T rows.
- [ ] **`full_attention_prefill_kv_append`**:
  - call `qwen35_write_paged_kv_mixed_value_fp16_batch_spans` with
    `KVLiveSpans` whose `live_counts` carries per-row prompt positions and
    `base_offsets` carries the block table. The wrapper is landed (see
    `tests/test_qwen35_paged_kv_write.py` and the c>1 batched smoke).
- [ ] **`full_attention_causal_prefill_attention`** — pick one of:
  - **D1.** Port/write a HIP causal multi-query SDPA prefill kernel (true
    speedup path). Source candidates: parent's `F.scaled_dot_product_attention`
    is the math reference, but the HIP analogue should be a multi-query
    causal variant of `qwen35_paged_full_attn_decode_split_k_gqa_*`. Stage as
    a new family in `hipengine/kernels/hip_gfx1100/attention/`.
  - **D2.** Labelled serial-fallback in
    `Qwen35ParoDecodeState.decode_full_attention_context_gate_fp16`: run T
    queries one at a time but as a single Python loop bound inside the
    bulk orchestrator. This is **not** a perf path, but it unblocks all 40
    layers running bulk prefill so that A/B/C wins are measurable and so
    `prefill_native` becomes the default.
  - Recommended order: land D2 first (correctness/orchestration unblocker),
    then land D1 as a perf upgrade with its own artifact.
- [ ] Add a chunked variant matching parent's
  `NANOVLLM_PARO_PREFILL_FULL_ATTN_QUERY_CHUNK_SIZE` /
  `..._POST_CHUNK_SIZE` / `..._ROPE_CHUNK_SIZE` for long prompts.

Gates:

- Per-component CPU-reference oracle (Q/K/V projection layout, RoPE prep,
  KV append, causal SDPA output) before any layer-level perf claim. See
  `docs/TESTING.md`.
- Layer-3 hidden-state drift gate against the parent fixture
  (`scripts/qwen35_native_prefill_fullattn_stage_probe.py` already exists for
  this; extend it to cover the new bulk path).
- New accepted artifact:
  `benchmarks/results/2026-05-XX-hipengine-qwen35-bulk-prefill-full-attn-accepted.json`
  with full 40-layer prefill matching parent token IDs on the 512/32 fixture.

### E. Wire bulk prefill into `LLM.generate` and the scheduler

Once A–D land, the c=1 single-request flow should switch to bulk by default.

Punchlist:

- [ ] `hipengine/generation/qwen35_paro.py`: replace per-token
  `session.step(...)` prefill loop with one
  `session.prefill_native(prompt_ids)` call, falling back to the serial
  step loop only if `prefill_native` raises `NotImplementedError` for an
  unsupported config. Hot path stays torch-free.
- [ ] `hipengine/llm.py`: no API change. `LLM.generate(prompts,
  SamplingParams(...))` keeps the same surface.
- [ ] `hipengine/generation/batch_scheduler.py`: `next_prefill_work(chunk_size)`
  should still emit prompt chunks for the future c>N case (see F), but the
  single-request path bypasses chunking and calls `prefill_native` end-to-end.

Gate: `LLM.generate` on the parent fixture must produce identical token IDs
and identical logits to the current resident E2E gate
(`benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`).

### F. Compact c>N prompt batching (multi-request)

This is the path that beats the current `scheduler_serial_slot_bridge` and
removes the "8 sequential c=1 prefills" diagnostic.

Today `ResidentBatchScheduler.next_prefill_work(chunk_size=...)` emits prompt
chunks per request but the runtime executes them serially through `step(...)`.

Punchlist:

- [ ] Define a "compact prompt slab" ABI:
  `[T_total, hidden]` packed across N requests, plus
  `cu_seqlens_q[N+1]`, `cu_seqlens_k[N+1]`, `positions[T_total]`,
  `row_to_request[T_total]`, and a per-row block-table-slot map for KV
  append. This mirrors `nano-vllm-amd/nanovllm/utils/context.py`
  `prepare_prefill`.
- [ ] Extend `KVLiveSpans` (already the ABI) to carry per-row prompt-row
  positions and per-request `live_counts`. Public wrapper updates only;
  no kernel changes needed for `paged_kv_write_*_batch_spans`.
- [ ] Add `Qwen35ParoResidentSession.prefill_native_packed(slab)` that
  runs the same bulk per-layer logic over the packed slab. Linear-attention
  layers will read/write per-request state caches; the linear-state and
  conv-state buffers must be **scatter-aware** so each request's tail is
  preserved.
- [ ] Causal multi-query prefill attention needs to be **var-len** aware
  (multi-request causal block-diagonal mask). Parent uses
  `flash_attn_varlen_func`; until our HIP causal prefill kernel lands, we
  can fall back to per-request bulk prefill in a single Python loop bound
  inside the bulk orchestrator.
- [ ] Replace `scheduler_serial_slot_bridge` with
  `scheduler_compact_prefill` once correctness gates pass, and re-emit the
  c=8 benchmark artifact as
  `2026-05-XX-hipengine-qwen35-c8-compact-prefill-accepted.json`.

Gate: same generated tokens as serial c=8 path
(`2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json`); finite
logits; prefill tok/s improves over the serial bridge by at least 2x at
c=8/T=512 before retaining a perf claim.

### G. Performance knobs and tuning

After A–F land, replicate the parent tuning surface in a
hipENGINE-appropriate way (config object, not env vars on the hot path).

Punchlist:

- [ ] Add a `PrefillConfig` dataclass under `hipengine/runtime/` with:
  - `linear_chunk_size: int = 0` (parent: `_PARO_PREFILL_LINEAR_CHUNK_SIZE`)
  - `full_attn_query_chunk_size: int = 0`
  - `full_attn_post_chunk_size: int = 0`
  - `full_attn_rope_chunk_size: int = 0`
  - `moe_grouped_device_gather: bool = True`
  - `moe_stacked_compact: bool = True`
- [ ] Wire the config into `Qwen35ParoResidentSession`. Defaults must match
  the retained parent OPTIMAL flags so that "out of the box" performance is
  parent-equivalent; document under `docs/OPTIMAL.md` (new) or under
  `docs/PLAN.md` "Performance Knobs".
- [ ] Add a sweep script
  (`scripts/qwen35_prefill_chunk_sweep.py`) to find the right defaults on
  W7900 and produce a retained row.

Gate: 4096/128 and 4096/4096 retained rows on hipENGINE match the parent
PARO rows from `~/amd-gpu-tuning/docs/PARO.md` within ~10%; emit two new
benchmark artifacts and update `benchmarks/README.md`.

### H. Graph capture and step replay

Optional perf finishing. Parent uses CUDA graphs for prefill warm-up and
decode step replay (`bench_paro_native_engine.py --decode-use-step-graph-replay`).
HIP graph capture is gfx1100-supported but adds engineering load.

Punchlist:

- [ ] HIP graph capture wrapper in `hipengine/runtime/graph.py` (does not
  exist yet) keyed by bucketed shapes (see `BatchShapeKey`).
- [ ] Prefill warm-up that captures the bulk layer chain for one fixed T
  bucket. Replay for repeated prefill requests.
- [ ] Same for decode step replay over the current resident decode loop.
- [ ] Roofline check vs `docs/ROOFLINE.md` (the parent already lives at
  about 25–28% of the simple memory roof; we should be within 1.5x of
  parent before chasing graph capture).

This phase is performance-only; correctness gates are inherited from earlier
phases.

## Validation strategy

Every phase must close a measurable gap and pass three layers of validation
before retaining a perf claim.

1. **Per-kernel correctness** (`docs/TESTING.md`):
   bit-exact vs `kernels/cpu_reference/` on fixture inputs; KL ≤ 0.05 and
   top-1 ≥ 90% on the parent fixture for any new GPU kernel
   (`docs/PLAN.md` Evidence Policy).
2. **Per-layer hidden-state drift** vs parent on
   `fixtures/qwen35_paro/parent_512_32_seed1234.json`:
   `scripts/qwen35_native_prefill_fullattn_stage_probe.py` is the template;
   extend its layer scope as each phase lands.
3. **End-to-end token equality + prefill tok/s** on the same fixture, with
   `scripts/qwen35_e2e_correctness.py` (already wired) and a new
   `scripts/qwen35_prefill_bench.py` (to be added in phase G) that records
   exact command, hardware, peak allocated, and retains the JSON artifact.

Correctness is non-negotiable: a faster prefill that fails the parent fixture
is a regression, not a win.

## Risks and open questions

- **Causal multi-query prefill attention kernel.** Writing a Flash-like
  prefill kernel on gfx1100 is the largest single piece of new kernel work.
  Workarounds (D2 fallback, chunked attention through existing decode
  kernels) lose perf but keep correctness. The decision on D1 vs D2 first
  belongs in this doc.
- **Grouped MoE port surface.** `csrc/amd/qwen35_expert.hip` contains 95
  kernels; we only need the subset on the W4 PARO prefill critical path.
  See `docs/KERNELS.md` "Source-lineage kernel catalog to port" for the
  full inventory; the families called by `ParoQuantMoE._forward_grouped_device_gather`
  are the actual target set.
- **Compact c>N prefill state-cache semantics.** Linear-attention conv/GDN
  state must be **per-request** during a packed slab pass. The scatter-aware
  fallback (one request at a time inside the bulk orchestrator) is the
  safe correctness path; the compact path needs new kernels or careful
  per-request strided launches. This decision belongs in phase F.
- **Plugin-registry coverage.** Each new bulk composite
  (`linear_attn_prefill`, `full_attn_prefill`, `moe_prefill`) needs a
  registry key and a CPU-reference fallback. Skipping the fallback is an
  architectural violation per `AGENTS.md`.

## References

- `docs/PLAN.md` — architecture, phase roadmap, LoC budgets, extensibility
- `docs/KERNELS.md` — kernel catalog, source-lineage, port playbook
- `docs/BENCHMARK.md` — benchmark protocols, correctness gate, artifact rollup
- `docs/TESTING.md` — RED/GREEN, fixtures, correctness gates
- `docs/DFLASH.md` — related speculative path (separate concern, similar
  ABI shape: `KVLiveSpans`, `TargetVerifyBatch`, per-row positions)
- `~/amd-gpu-tuning/docs/PARO.md` — parent retained rows and config
- `~/amd-gpu-tuning/scripts/bench_paro_native_engine.py` —
  `prefill_bulk(...)` reference
- `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant.py` —
  `ParoQuantLinearAttentionLayer.prefill_native` and
  `ParoQuantFullAttentionLayer.prefill_native` reference
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-attn-boundary-blocked.json`
  — current blocker with component breakdown
- `benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`
  — current c=1 perf row + parent comparison
- `benchmarks/results/2026-05-15-hipengine-qwen35-c8-scheduler-serial-bench-blocked.json`
  — current c=8 serial-bridge diagnostic
