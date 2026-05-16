# hipENGINE Native Bulk Prefill Plan

> Status: final implementation spec, corrected 2026-05-15. This document is
> the authoritative prefill punchlist for Qwen3.5-35B-A3B-PARO. `docs/PLAN.md`
> remains the architecture source of truth; update both files if the architecture
> changes.

## TL;DR

We are **not** landing throwaway intermediate prefill paths. hipENGINE already
has correct reference implementations: the original `nano-vllm-amd` native bulk
engine and hipENGINE's validated serial resident path. Use those as oracles and
build the complete native path directly.

Final target:

- One `Qwen35ParoResidentSession.prefill_native(...)` call embeds the whole
  prompt (or a configured prompt chunk) into `[T, hidden]` and runs every layer
  in bulk.
- Linear-attention layers use native conv/GDN prefill and update their prompt
  tail state once per layer.
- Full-attention layers use native batched Q/K/V projection, batched head
  RMSNorm+RoPE, batched KV append, and native causal GQA prefill attention.
- MoE uses the grouped/compact parent route over prompt rows, not the existing
  c1 selected-row MoE path as the retained implementation.
- Generation and benchmark scripts call `prefill_native(...)`; serial/token-loop
  paths remain only for reproduction and correctness comparisons.
- Compact c>N prefill packs multiple requests into a prompt slab and executes
  native kernels over that slab. Per-request invocation is an oracle/fallback for
  debugging, not a retained c>N throughput path.

Explicitly skipped as retained implementations:

| Skipped path | Allowed use |
| --- | --- |
| `linear_prefix_token_major_suffix` | Existing artifact reproduction only. |
| Layer-major full-attention row loop through c=1 decode kernels | Stage oracle/probe only; do not wire into generation or retain perf rows. |
| c1-style selected-row MoE as the prefill path | Oracle for grouped MoE and bring-up probes only. |
| Per-request c>N packed fallback | Debug/equality oracle only; no c>N throughput claim. |

Implementation landing policy: native pieces may land independently in code
behind `require_full_native=False` or test-only/probe entrypoints, using the
oracle paths above to fill missing pieces during bring-up. The first retained
prefill performance artifact is captured only after all native pieces are
present, `PrefillConfig.require_full_native=True` is the default, and the c1
selected-row MoE path no longer appears in the production prefill code path.
In-progress measurements live in `WORKLOG.md`; `benchmarks/README.md` keeps the
current 117.24 tok/s c=1 row as the retained performance baseline until native
prefill beats it. The first full single-request native correctness artifact is
accepted, but it is diagnostic-only for throughput.

Scope note: this plan targets `z-lab/Qwen3.5-35B-A3B-PARO` MoE hybrid. Dense
`Qwen3.5-0.8B-PARO` needs tied-lm-head and dense PARO MLP support first; that is
a separate loader/runtime task.

## Terms and shape conventions

| Term | Meaning |
| --- | --- |
| `T` | Prompt rows for one request in one prefill call or internal chunk. |
| `T_total` | Rows in a compact prompt slab packed across multiple requests. |
| `C` | Active decode requests; not the same as prompt rows. |
| Bulk prefill | Layer input/output buffers are `[rows, hidden]`; kernels operate on prompt rows. |
| Grouped/compact MoE | Parent-style route that scatters routed rows by expert and runs grouped bulk kernels. |
| Append-then-attend | First native full-attn design: append all prompt K/V rows to paged cache, then causal attention reads prefix+prompt K/V entirely from cache. |

KV span convention for this repo:

- KV **append** spans use `live_counts[row] = absolute_position` (0-based write
  position), matching the preserved parent writer ABI.
- KV **attention/decode** spans use `live_counts[row] = context_length` (1-based
  visible length).
- For prompt row `r` with `start_position`, append position is
  `start_position + r`; attention context length is `start_position + r + 1`.

## Evidence: current gap and references

Parent native engine retained rows (Qwen3.5-35B-A3B-PARO, W7900, BF16/FP16
activations, W4 PARO weights):

| Shape | Prefill tok/s | Decode tok/s | Notes (`~/amd-gpu-tuning/docs/PARO.md`) |
| --- | ---: | ---: | --- |
| 512 / 128 | 554.21 | 64.71 | `bench_paro_native_engine.py --prefill-mode bulk`, lm_head dense GEMV |
| 4096 / 128 | 2140.71 | 60.32 | bulk, lm_head dense GEMV, 24GB path |
| 4096 / 4096 | 2155.60 | 56.79 | bulk, lm_head dense GEMV, 24GB path |
| 512 / 32 | 2682.66 | 116.26 | parent fixture row recorded in `fixtures/qwen35_paro/parent_512_32_seed1234.json` |

hipENGINE current rows on the same 35B fixture:

| Shape | Prefill tok/s | Decode tok/s | Artifact / notes |
| --- | ---: | ---: | --- |
| 512 / 32 | 117.24 | 101.68 | `benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`; prompt runs as sequential resident steps |
| 512 / 32 | 45.72 fixture / 46.96 repeated-token diagnostic | 101.61 | `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json`; `single_request_native_full`, correctness accepted (`max_kl=0.0168`, top-1 100%), no perf row promoted |
| c=8 8/1 | 115.08 | 108.89 | `scheduler_serial_slot_bridge` diagnostic, not native compact batching |

Correctness/blocker artifacts already retained:

- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-scratch-restore-sweep.json`
  — native linear prefix accepted through layers 0..2.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-serial-suffix-full40-accepted.json`
  — native linear prefix plus token-major serial suffix matches serial resident
  outputs; no throughput claim.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json`
  — final single-request native prefill correctness gate accepted on the 512/32
  parent fixture (`max_kl=0.0168`, top-1 100%, generated IDs match serial and
  parent); diagnostic timing remains slower than serial and parent baselines.
- Earlier blocked boundary artifacts (`native-prefill-full-attn-boundary-blocked`,
  `native-prefill-plan-blocked`) are superseded by the accepted full native
  orchestration artifact above.

Reference files:

- `~/amd-gpu-tuning/scripts/bench_paro_native_engine.py`
- `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant.py`
- `~/amd-gpu-tuning/docs/PARO.md`
- `~/amd-gpu-tuning/docs/OPTIMAL.md`

## Current hipENGINE inventory

`docs/KERNELS.md` is authoritative for exact landed kernels and gates. If this
section disagrees with `docs/KERNELS.md`, `docs/KERNELS.md` wins and the follow-up
change should fix both files.

Landed and usable now:

| Area | Current usable pieces |
| --- | --- |
| Runtime state | `embedding_lookup_batch_{bf16,fp16}_i64`, mapped variants, `set_i64_vector`, scalar/vector decode position helpers. |
| Linear-attn prefill | `qwen35_linear_attn_conv_prefill_f32`, segment-aware `qwen35_linear_attn_conv_prefill_segments_f32`, `qwen35_linear_attn_prefill_prepare_f32_fp16`, `qwen35_gdn_prefill_recurrent_k2_f32`, segment-aware `qwen35_gdn_prefill_recurrent_segments_k2_f32`, `qwen35_gdn_prefill_rmsnorm_gate_fp16`. |
| Linear layer orchestrator | `run_linear_attention_moe_c1_layer_fp16(tokens=T)` already selects prefill conv/GDN when `tokens > 1`; final path must replace its c1 MoE tail with grouped MoE. |
| Full-attention decode/prelude | Existing c=1 Q/K/V projection, vector-position RoPE prefill prelude, KV append, native append-then-attend causal GQA prefill kernel including varlen/block-diagonal `cu_seqlens` ABI, context/GQA decode, gate, output projection. Decode kernels remain useful as oracle only for prefill attention. |
| KV append | `qwen35_write_paged_kv_mixed_value_fp16_prompt_spans(...)` appends all prompt rows into one request cache; row-major `*_batch_spans(...)` remains for c>N-shaped caches. Both consume per-row append positions in `spans.live_counts`. |
| KV metadata | `KVLiveSpans` already carries `request_ids`, `row_positions`, and `span_role`; compact prefill needs wiring/population, not a span redesign. |
| Graph primitives | `hipengine.core.hip.HipRuntime` exposes HIP graph capture/instantiate/launch; decode graph capture exists. |

Missing for the final path:

| Area | Required final work |
| --- | --- |
| Public API wiring | **Landed:** `prefill_native(...)` is the default single-request path; compact c>N uses `prefill_native_packed(slab)` and generated-equality gates now pass for c=2/4/8 prompt8. |
| Full-attn retained orchestration | **Landed for c=1:** batched Q/K/V + vector RoPE + prompt KV append + native causal prefill attention are wired and fixture-gated. |
| Grouped/compact MoE | **Landed for c=1:** grouped scatter/gather and compact AWQ WMMA expert kernels are wired into native prefill. |
| Compact c>N slab | **Correctness landed:** `CompactPromptSlab`, `bucketize_by_block_count`, physical slot metadata, segment-aware linear-attn conv/GDN, varlen/block-diagonal full-attn via `cu_seqlens`, grouped compact MoE, and final-row commit are wired through `prefill_native_packed(slab)`; c-aware decode graph replay and retained throughput remain future work. |
| Prefill config/tuning | Add typed `PrefillConfig`; no hot-path env lookups. |

## Final API and config contract

Add this public session API:

```python
def prefill_native(
    self,
    token_ids: Sequence[int],
    *,
    sample: bool = True,
    require_full_native: bool | None = None,
) -> Qwen35ParoAutoregressiveStepResult | None:
    """Run full native prefill from position 0 through len(token_ids)-1.

    If sample=True, return next-token logits/argmax from the final prompt row.
    If require_full_native is None, use PrefillConfig.require_full_native; an
    explicit per-call value overrides the config default. The final default is
    full-native required: unsupported configs raise NotImplementedError rather
    than silently using token-loop fallbacks.
    """
```

Add a typed config object, e.g. `hipengine/runtime/prefill.py`:

```python
@dataclass(frozen=True)
class PrefillConfig:
    linear_chunk_size: int = 0
    full_attn_query_chunk_size: int = 0
    full_attn_post_chunk_size: int = 0
    full_attn_rope_chunk_size: int = 0
    moe_grouped_device_gather: bool = True
    moe_stacked_compact: bool = True
    require_full_native: bool = True
```

Semantics:

- Public `prefill_native(token_ids, ...)` starts at position 0 on a fresh
  session. Non-zero external `start_position` is not a public API.
- Final native prefill requires `T >= config.linear_conv_kernel_dim` (typically
  4 for Qwen3.5/PARO) because the linear-attention conv prefill kernels require
  enough rows. Shorter prompts raise `ValueError`; no production serial fallback
  is added for this corner unless a future dedicated short-prompt native kernel
  lands.
- Internal chunking may process the prompt as multiple contiguous chunks, but it
  must preserve exactly the same final conv/recurrent state and KV cache as a
  single full-prompt call.
- If `sample=False`, the method performs all state/KV updates and returns
  `None`.
- After prefill, copy the final row into `self.hidden`, restore decode scratch
  sizes, and set `position_buf = T - 1`, `context_buf = T` so the next decode
  step appends at position `T`.
- Keep `prefill_linear_tokens_native(...)` only as a compatibility alias for
  retained artifact reproduction; update `scripts/qwen35_paro_bench.py` and new
  call sites to use `prefill_native(...)`.
- `hipengine/generation/qwen35_paro.py` should call `session.prefill_native(...)`
  directly; no generation-time serial prompt loop except an explicitly requested
  diagnostic mode.
- Prefill work does not change decode policy: multi-token decode scheduling and
  any future `Qwen35ParoOneTokenGenerator` rename/behavior changes are out of
  scope here. This plan only replaces prompt setup.

Path labels for artifacts:

| Label | Meaning | Retained perf claim? |
| --- | --- | --- |
| `serial_step_loop` | Existing token-by-token resident prefill. | Baseline only. |
| `native_prefill_full_single_request` | Final single-request native prefill: native full-attn + grouped MoE. | Yes. |
| `native_prefill_compact_cN` | Final multi-request compact slab path. | Yes. |
| `oracle_row_loop_full_attention` | c=1 row loop used by probes/tests. | No. |
| `oracle_c1_selected_moe_rows` | c1 selected-row MoE used as grouped-MoE oracle. | No. |
| `oracle_per_request_packed_fallback` | c>N metadata debug path invoking one request at a time. | No. |

## Final single-request native prefill pipeline

For one request with prompt length `T`:

1. **Prepare prompt tensors**
   - Validate `token_ids` and capacity.
   - Fill/copy `prefill_token_ids[int64, T]` and `prefill_positions[int64, T] =
     arange(T)`.
   - Resolve the embedding op through the backend/model dispatch path. Qwen3.5
     PARO uses FP16 hidden buffers, so the concrete gfx1100 launch is
     `embedding_lookup_batch_fp16_i64(...)` into `prefill_hidden[T, hidden]`.

2. **Layer-major execution with no production row-loop fallbacks**
   - Maintain `hidden[T, H]` and `next_hidden[T, H]` double buffers.
   - Invariant: every row of `next_hidden[0:T]` is written before the layer-end
     `hidden, next_hidden = next_hidden, hidden` swap.
   - For each layer, route by model layer type through plugin/registry keys; do
     not add backend/quant branches in engine code.

3. **Linear-attention layer final path**
   - Input RMSNorm over `T` rows.
   - PARO rotations/projections over `[T, H]`.
   - Native conv prefill + GDN recurrent prefill over the prompt rows.
   - Update the layer's conv/recurrent state to the prompt tail exactly once.
   - Output projection over `T` rows.
   - Post-attention add/RMSNorm.
   - Run final grouped/compact MoE (below), not retained c1 selected-row MoE.

4. **Full-attention layer final path**
   - Input RMSNorm over `T` rows.
   - Batched PARO Q/K/V projections producing contiguous:
     - `query_proj: fp16[T, num_q_heads, 2 * head_dim]` (query + gate),
     - `key_raw_lowp: fp16/bf16[T, num_kv_heads, head_dim]`,
     - `value: fp16[T, num_kv_heads, head_dim]`.
   - Split/cast query/gate as needed and run batched Q/K head RMSNorm + RoPE with
     per-row positions via `qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16`.
     The existing scalar-position kernel is oracle-only for prefill.
   - Specific scalar bug to avoid: current `prepare_full_attention_qkv_fp16(...)`
     casts only one row of K (`kv_width` elements) from FP16 to FP32. The bulk
     path must cast `T * kv_width` elements.
   - Append all `T` K/V rows to the single request's paged cache with
     `qwen35_write_paged_kv_mixed_value_fp16_prompt_spans(...)`, append spans
     `live_counts = positions`, and `span_role="prefill"`.
   - Run native causal GQA prefill attention over the paged cache using context
     spans `live_counts = positions + 1`.
   - Run output projection over `T` rows.
   - Post-attention add/RMSNorm.
   - Run final grouped/compact MoE.

5. **Final norm/lm head**
   - Only the final prompt row is sampled when `sample=True`.
   - The final hidden row seeds subsequent decode; no extra prompt-token decode
     step is allowed.

### Full-attention prefill kernel contract

First native design is **append-then-attend from cache**:

- Append all prompt K/V rows to paged BF16 KV cache first.
- Attention reads prefix+prompt keys/values entirely from the paged cache.
- Future optimization may read prefix-from-cache plus prompt-from-scratch to
  avoid one HBM round-trip; do not combine that two-source design with the first
  native kernel.

Register a gfx1100 kernel such as:

```python
KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "qwen35_causal_gqa_gate_fp16")
```

`w4_paro` is the model/dispatch identity, as with existing rotary/attention
registrations; the attention kernel itself does not dequantize weights.

Mirror the existing GQA split-K gate-fused decode shape
(`qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans`). What differs from
decode: the new kernel processes `T` query rows instead of one, consumes
`positions[row]`/per-row context spans, and applies a causal mask
`cache_position <= positions[row]`. Scratch layout, gate fusion, split-K/reduce
shape, and softmax scale should otherwise match decode.

- Inputs:
  - query `fp32[T, num_q_heads, head_dim]`,
  - gate `fp16[T, num_q_heads, head_dim]`,
  - BF16 paged key/value cache,
  - `KVLiveSpans` with per-row context lengths,
  - output buffer `fp16[T, num_q_heads * head_dim]`.
- For row `r`, attend only to cache positions `<= positions[r]`.
- GQA mapping: `kv_head = q_head // (num_q_heads // num_kv_heads)`.
- Apply the same softmax scale and gate semantics as decode.
- Output is **post-gate FP16**, ready for `project_full_attention_o_fp16(tokens=T)`.

### Grouped/compact MoE final path

The existing `run_moe_c1_fp16(tokens=T)` path is a correctness oracle only. The
retained prefill path must use grouped/compact MoE over prompt rows.

Required ports/wiring:

- Confirm router wrappers cover `[T, hidden]` and keep native router top-k.
- `moe/group_scatter.hip` count, prefix, scatter/scatter_gather, gather packed
  hidden, and WMMA tile-map metadata kernels are landed; the current grouped
  prefill wire-up builds `lane_to_row` in the weighted-lane combine kernel.
- The grouped compact route is wired over packed/sorted lanes and registered as
  `moe_prefill/w4_paro/qwen35_grouped_compact`; compact WMMA gate/up and down
  expert kernels are now the default grouped expert path. A retained throughput
  claim still requires full single-request prefill orchestration and benchmark
  artifact closure.
- The selected-row c1 path is registered only as
  `moe_prefill/w4_paro/qwen35_selected_c1_rows` oracle/fallback coverage; native
  multi-token prefill layer orchestration routes to grouped compact instead.
- Port the Qwen3.5/PARO-used subset of `quant/w8a16_moe.hip` shared/bulk
  variants if/when the parent call graph requires variants beyond the existing
  W8A16 shared expert wrappers; do not port all 17 variants speculatively.
- Port `moe/w8a8_grouped.hip` only if the W4 PARO parent retained path actually
  uses it.
- Port `wmma/wmma_i8_gemm.hip` for long-prompt grouped GEMM once the pack8
  grouped path is correctness-accepted.
- Register:
  - `(hip_gfx1100, moe_prefill, w4_paro, qwen35_grouped_compact)` as retained,
  - `(hip_gfx1100, moe_prefill, w4_paro, qwen35_selected_c1_rows)` only as an
    oracle/fallback key for tests.

## Compact c>N prompt batching final path

Final c>N prefill packs multiple requests into one slab. Per-request invocation
is allowed for debugging/equality tests only.

```python
@dataclass(frozen=True)
class CompactPromptSlab:
    token_ids: Tensor        # int64[T_total]
    positions: Tensor        # int64[T_total], absolute positions per row
    cu_seqlens_q: Tensor     # int32[N + 1]
    cu_seqlens_k: Tensor     # int32[N + 1]
    row_to_request: Tensor   # int64[T_total]
    request_ids: Tensor      # int64[N]
    block_tables: Tensor     # int32[T_total, blocks_per_request] == KVLiveSpans.base_offsets reshaped for the current batch-writer ABI
    append_counts: Tensor    # int64[T_total], 0-based append positions
    context_counts: Tensor   # int64[T_total], 1-based visible lengths
```

Kernel ABI convention: `cu_seqlens_q`/`cu_seqlens_k` define the varlen
block-diagonal attention segments passed to the native causal prefill kernel.
`row_to_request` remains scheduler/debug metadata and is used for validation,
state routing, and output ownership; it is not the primary mask input to the
attention kernel.

Final compact requirements:

- `ResidentBatchScheduler.next_compact_prefill_slabs(chunk_size=...)` forms
  compact slab descriptors for requests with prefill work; legacy
  `next_prefill_work(...)` remains the serial diagnostic path.
- An explicit `bucketize_by_block_count` step in the scheduler runs before slab
  construction and emits one slab per uniform block-table length.
- `Qwen35ParoResidentSession.prefill_native_packed(slab)` is present and
  fail-closed until the remaining packed full-attn and final commit stages
  land; it must eventually run the same native layer logic over `T_total` rows.
- Current batch KV writer constraint: `_check_write_batch_shape(...)` computes
  one `block_table_len = base_offsets.numel // rows`, so every row in a writer
  call must expose the same block-table length. Final scheduler policy should
  bucket slabs by common `blocks_per_request`; cross-bucket requests launch as
  separate native slabs. A true varlen block-table writer is a future kernel
  port, not a reason to use a serial per-request fallback.
- Linear-attention conv/GDN is segment-aware: `f32_segments` conv consumes
  `cu_seqlens` + state slots and `f32_k2_segments` GDN commits each request's
  recurrent tail independently. Packed prefill orchestration must call these
  landed kernels rather than retaining per-request invocation.
- Native causal prefill attention is var-len/block-diagonal:
  `qwen35_varlen_causal_gqa_gate_fp16` consumes `cu_seqlens_q/k`, row-shaped
  block tables, context counts, and positions. Packed prefill orchestration must
  call this landed kernel so a query row attends only to its request segment and
  positions not greater than the query position.
- Native compact prefill is non-speculative and commits canonical KV inline for
  admitted prompt rows. `CompactPromptSlab.slot_ids` carries the physical slots;
  `_commit_packed_prefill_final_rows(...)` commits each segment tail hidden row
  plus position/context metadata after packed layer execution. `KVPolicy.begin_transaction/commit/rollback` hooks remain
  for speculative verify/draft paths, not this ordinary prefill path.
- `prefill_native_packed(slab)` now runs the native compact prefill path and
  returns one final-row sample per request. Decode after the seed still uses
  `step_batch_serial`; replace that serial decode bridge only after c-aware
  decode graph replay lands.

## CPU references and oracles

Before registering new retained gfx1100 layer keys, add or identify the matching
correctness oracle:

- `hipengine/kernels/cpu_reference/ops.py` includes a torch-free NumPy
  `full_attn_prefill` CPU reference for tiny causal-GQA fixtures using
  pre-appended K/V.
- Add CPU-reference or row-by-row c1 oracle coverage for grouped MoE stages.
- Use hipENGINE's serial resident path and the parent `nano-vllm-amd` native bulk
  path as external stage/e2e oracles.
- Row-loop full-attention and c1 selected-row MoE may be implemented as test-only
  helpers if useful, but they must not be wired into generation or retained
  performance artifacts.

## Graph capture and tuning

Do not chase graph capture before the final native kernels are in place and
roofline/profiler data says dispatch is material.

- Low-level HIP graph wrappers already live in `hipengine/core/hip.py`.
- `Qwen35ParoResidentSession.capture_decode_graph(...)` exists for decode.
- Add a prefill graph cache only after native single-request and compact c>N
  paths are correct.
- Graph keys should use a small prechosen/power-of-two T bucket set, not one
  graph per exact prompt length.
- `PrefillConfig` chunk sizes mirror parent knobs:
  - `linear_chunk_size`,
  - `full_attn_query_chunk_size`,
  - `full_attn_post_chunk_size`,
  - `full_attn_rope_chunk_size`.
- Defaults must match retained parent OPTIMAL flags on W7900 once measured.

## Validation and definition of done

No intermediate perf wins are retained. The doc is complete when these final
artifacts/gates exist.

### Single-request native prefill done

Required checks:

1. Unit/CPU-reference tests for `full_attn_prefill` and grouped MoE stages.
2. Stage probes vs serial resident and/or parent native bulk for:
   - Q/K/V projection layout,
   - batched RoPE,
   - KV append,
   - causal attention post-gate output,
   - grouped MoE output,
   - full layer hidden output.
3. Full 40-layer fixture gate on `fixtures/qwen35_paro/parent_512_32_seed1234.json`:
   greedy generated IDs match the serial resident path, with KL ≤ 0.05 and
   top-1 agreement ≥ 90% on logits at each sampled position.
4. Chunk-equivalence sweep: for non-zero
   `PrefillConfig.{linear_chunk_size, full_attn_query_chunk_size,
   full_attn_post_chunk_size, full_attn_rope_chunk_size}` values, final hidden
   row and KV cache contents match the single-chunk run within the stage-probe
   tolerance, and generated decode IDs/logits satisfy the same KL/top-1 gate.
5. `rocprofv3 --kernel-trace` proves native full-attn prefill and grouped MoE
   kernels ran, with expected names and plausible durations.
6. `LLM.generate`/`Qwen35ParoOneTokenGenerator` uses `prefill_native(...)` by
   default and satisfies the fixture ID/KL/top-1 gate above.

Retained artifact target:

```text
benchmarks/results/2026-05-XX-hipengine-qwen35-native-prefill-full-single-request-accepted.json
```

It must include model, quant, workload shape, W7900 hardware, exact command,
peak memory, correctness gate, kernel names, and comparison to the current
117.24 tok/s c=1 fixture and parent rows.

### Compact c>N prefill done

Required checks:

1. c=2/4/8 generated-token equality vs independent serial c=1 sessions.
2. Finite logits and per-request state/KV bounds checks.
3. Native compact kernels run; no per-request prompt loop in the retained prefill path.
4. At c=8/T=512, prefill tok/s improves over `scheduler_serial_slot_bridge` by
   at least 2× before retaining a throughput claim. This perf row is still
   pending; c=2/4/8 prompt8 correctness is accepted but not a throughput claim.

Accepted correctness artifacts:

```text
benchmarks/results/2026-05-15-hipengine-qwen35-c2-native-compact-prefill-correctness-accepted.json
benchmarks/results/2026-05-15-hipengine-qwen35-c4-native-compact-prefill-correctness-accepted.json
benchmarks/results/2026-05-15-hipengine-qwen35-c8-native-compact-prefill-correctness-accepted.json
```

Retained throughput artifact target:

```text
benchmarks/results/2026-05-XX-hipengine-qwen35-native-prefill-compact-c8-accepted.json
```

Any retained performance row also updates `benchmarks/README.md`,
`benchmarks/CHANGELOG.md`, and a compact JSON artifact under
`benchmarks/results/`.

Correctness is non-negotiable: a faster prefill that fails the parent fixture is
a regression, not a win.

## Optimization diagnosis (2026-05-16): the 4K gap is one kernel

This section captures the trace-driven diagnosis after the 49-iteration
`prefill-perf` multiloop plateaued at 2039 tok/s @ 512/128, plus the
standing-rule reasoning chain for the next optimization spike. It is
standalone evidence: a future agent should be able to read this section and
reproduce the decision without re-running the audit.

### Where we stand

Measured with the standard bench command on the parent 512/32 fixture and the
4K/128 repeated-token diagnostic; both runs use `require_full_native=True`.

| Shape          | hipENGINE | nano-vllm-amd (parent) | parent / hipENGINE |
| -------------- | --------: | ---------------------: | -----------------: |
| 512 prefill    | 2039 tok/s | 2589 tok/s              | +27 %              |
| 4K prefill     |  659 tok/s | 1681 tok/s              | +155 %             |

The 4K gap is the load-bearing one. At T=4K, hipENGINE spends 6.21 s in
prefill vs the parent's ≈ 2.44 s. Multiloop iters 1–49 optimized only the 512
metric and treated 4K as a no-regression guard; that left the long-context
path structurally unexamined.

### Trace comparison

From `rocprofv3 --kernel-trace` on `qwen35_paro_bench.py` with
`--prompt-length {512,4096} --decode-tokens 0 --max-layers 40` and matching
flags from `~/amd-gpu-tuning/scripts/run_moe2_baselines.py::COMMON_ENV` on the
parent side. Numbers are summed across the 40 layers.

Top kernel buckets, hipENGINE 512 prefill (total kernel time 229.77 ms):

| ms      | calls | avg us  | kernel                                                |
| ------: | ----: | ------: | ----------------------------------------------------- |
|   41.22 |    30 |  1373.9 | `qwen35_gdn_prefill_recurrent_k2_kernel`              |
|   33.96 |    40 |   849.0 | `gemm_awq_selected_dual_pack8_wmma_compact_kernel`    |
|   26.16 |    10 |  2615.8 | `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<true>` |
|   23.21 |    80 |   290.2 | `awq_fusedw4_prefill_fp16_kernel<32,32,true>`         |
|   19.48 |    40 |   487.0 | `gemm_awq_selected_pack8_wmma_compact_kernel`         |
|   16.05 |    40 |   401.2 | `w8a16_shared_down_combine_residual_fp16_kernel`      |
|   15.56 |    40 |   389.0 | `w8a16_shared_gate_up_silu_fp16_kernel`               |
|   14.81 |    50 |   296.2 | `awq_fusedw4_prefill_fp16_kernel<32,32,false>`        |
|    9.24 |    80 |   115.5 | `paro_rotate1_kernel<_Float16>`                       |
|    8.62 |    40 |   215.5 | `qwen35_router_logits_token_tile_kernel<_Float16,4>`  |

Top kernel buckets, hipENGINE 4K prefill (total kernel time 6171.07 ms):

| ms       | calls | avg us    | kernel                                                |
| -------: | ----: | --------: | ----------------------------------------------------- |
|  4572.38 |    10 |  457237.5 | `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<false>` |
|   391.64 |    30 |   13054.8 | `qwen35_gdn_prefill_recurrent_k2_kernel`              |
|   199.96 |    40 |    4999.0 | `gemm_awq_selected_dual_pack8_wmma_compact_kernel`    |
|   170.16 |    80 |    2127.0 | `awq_fusedw4_prefill_fp16_kernel<32,32,true>`         |
|   133.07 |    80 |    1663.4 | `paro_rotate1_kernel<_Float16>`                       |
|   124.89 |    40 |    3122.3 | `w8a16_shared_down_combine_residual_fp16_kernel`      |
|   117.35 |    40 |    2933.8 | `w8a16_shared_gate_up_silu_fp16_kernel`               |
|   116.15 |    40 |    2903.8 | `gemm_awq_selected_pack8_wmma_compact_kernel`         |
|    86.33 |    50 |    1726.7 | `awq_fusedw4_prefill_fp16_kernel<32,32,false>`        |
|    65.39 |    40 |    1634.7 | `qwen35_router_logits_token_tile_kernel<_Float16,4>`  |

Key observations:

1. **The full-attention prefill kernel template flips between 512 and 4K.** At
   T=512 the trace runs `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<true>`
   (split-K enabled, 26.16 ms / 10 layers); at T=4K it runs the same kernel as
   `<false>` (split-K disabled, **4572.38 ms / 10 layers, 457 ms per layer**).
   8× the tokens produces 175× the kernel time. That is super-quadratic; a
   correctly-tiled Flash-Attention-style kernel scales as T² in compute but
   stays HBM-bandwidth-bound and finishes ≈ 64× of the T=512 cost, not 175×.
2. **`<false>` is 74 % of all 4K kernel time** (4572 / 6171 ms). Closing this
   one bucket is worth more than every other optimization in the multiloop
   combined.
3. **`paro_rotate1` is also super-linear** (115 us → 1663 us, 14.4× growth for
   8× tokens). Nano-vllm-amd's analogous `paroquant_rotate_kernel` is
   near-linear (36 us → 240 us, 6.7×). This is a secondary but real RDNA3
   tiling/occupancy issue, not the headline.
4. **GDN recurrent prefill scales as ~9.5×** (41 → 392 ms) — roughly linear
   for 8× tokens with mild overhead, and within 7 % of nano-vllm-amd parity.
   Multiloop iters 36–37 were working against a real ceiling there; further
   grinding on that bucket is unlikely to pay.
5. **MoE compact-WMMA + W8A16 shared family scales ~6–8×** as expected for
   linear MoE work. Combined hipENGINE MoE+shared kernel time is ≈ 1.27× the
   nano-vllm-amd equivalent, because nano-vllm-amd silently opts OUT of compact
   WMMA at long T and dispatches `hipBLASLt` HGEMM with per-shape autotuned
   tiles (MT96×96×32, MT128×48×32, MT96×32×32 observed). Compact WMMA is a
   correct prefill path but is not the W7900-optimal one at T ≥ 1K.

### Why our kernel mis-scales

Direct read of
`hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip:1039–1193`
(`qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel`) shows four
structural problems, all visible in the source:

1. **LDS scratch scales with `max_context_len`.** Line 1083:
   `extern __shared__ float shared[]; float* scores = shared; float* partial =
   scores + max_context_len; float* q_shared = partial + blockDim.x;`. The
   `scores` buffer is `max_context_len * 4 B` per block: 2 KiB at T=512,
   16 KiB at T=4K, ~128 KiB at T=32K. RDNA3 ships ~64 KiB LDS per CU, so block
   residency drops from ≥8 blocks/CU at 512 to ≤3 blocks/CU at 4K to
   single-block-per-CU at 32K. Occupancy collapse compounds the T² cost.
2. **The V@scores epilogue is a fully serial T-deep inner loop, per output
   dim.** Line 1170:
   `for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
   float acc = 0.0f; for (int64_t token = 0; token < visible_len; ++token)
   { ... acc += scores[token] * value_cache[v_offset]; } }`. Each thread
   walks the full T axis sequentially, fetching every V row from HBM with no
   LDS staging. At T=4K that is 4096 serial multiply-accumulates per
   `(thread, output_dim)` pair, with one HBM load each.
3. **GQA KV sharing is missing.** Line 1084: `kv_head = q_head / kv_group`,
   computed independently per block. With 16 Q-heads and 2 KV-heads, each of
   the 8 Q-heads in a KV group has its own block that re-streams the same
   K/V cache through HBM. That is 8× redundant K/V bandwidth.
4. **The `<true>` / `<false>` template flip is a red herring.** It toggles
   `SHORT_BLOCK256` for short-context block-table inlining (line 1090–1097);
   it does not change the inner attention algorithm. Both branches share the
   serial V-loop above. T=512 looks acceptable only because all three issues
   are small at that length.

Observed 512→4K scaling is 178× for 8× length. A correctly-tiled Flash-Attention
implementation is O(T²) in compute but stays bandwidth-bound and runs in
≈64× the T=512 cost; the extra ≈3× is exactly issues (1) + (2) + (3)
compounding. The `<false>` branch is not a one-off bug; the entire kernel
family is pre-Flash-Attention.

**The kernel does carry one piece of fused logic we have to preserve.** Lines
1191, 1350, 1410:
`out[...] = static_cast<half_t>(acc * sigmoid_f32(gate_v))`. The attention
epilogue multiplies the per-`(row, q_head, dim)` output by
`sigmoid(gate[row, q_head, dim])`, where `gate` is a separate FP16 tensor
produced by the upstream QKV projection split. AOTriton's `attn_fwd*` API
has no gate input; any AOTriton-based replacement must add a trivial
elementwise post-pass kernel (`out *= sigmoid(gate)`) immediately after the
attention call to maintain model semantics. At T=4K, head_dim=128,
num_q_heads=16 the post-pass is one HBM-bandwidth-bound pass over
≈ 4096 × 16 × 128 = 8.4 M FP16 elements per layer — expected cost ≤ 0.2 ms
per layer, well inside noise.

A Flash-Attention-style fix to the existing kernel is not a tuning change; it
is an algorithmic rewrite: tile Q in registers, stream K/V chunks through LDS,
maintain online softmax running statistics across the K loop, share K/V
fetches across the GQA group, and apply causal masking inline. That is
several thousand lines of HIP plus several iterations of LDS bank-layout
tuning. We are not going to get there inside the existing multiloop budget by
turning knobs on the current kernel.

### Options for fast prefill attention without `torch` in the hot path

The four-axis registry and torch-free hot path invariants mean we cannot just
clone nano-vllm-amd's call to `F.scaled_dot_product_attention`. The viable
plugin keys all keep `import torch` out of the generation path.

| Option                                          | Source on disk / API                                                                                                 | gfx1100 support | Effort  | Expected 4K result vs current |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------- | ------- | ----------------------------- |
| AOTriton 0.8 standalone C++ ABI                 | `~/Downloads/aotriton/aotriton/{include,lib}/`; symbols `aotriton::v2::flash::attn_fwd{,_compact_varlen}` in `libaotriton_v2.so.0.8.0` | yes, 480 pretuned Navi3x variants | 2–3 days | ≈ 1700 tok/s (closes ~94 % of 4K gap) |
| Hand-rolled HIP FA-2 with WMMA                  | new code under `kernels/hip_gfx1100/attention/`; oracle = AOTriton output                                            | yes (we write it) | 3–6 weeks | 1300–1900 tok/s depending on tuning |
| Composable Kernel `ck_tile/01_fmha`             | `~/amd-gpu-tuning/reference/composable_kernel/example/ck_tile/01_fmha/`                                              | **no** — `known_fails_gfx{90a,942,950}.txt` only; CDNA-targeted | n/a | not applicable on W7900 |
| vLLM-vendored CK FA (CK fork inside vLLM)       | `~/vllm/flash-attention/csrc/composable_kernel/CMakeLists.txt` builds for `gfx1100;gfx1101;gfx1102`                  | yes (claimed) | 1–2 weeks (build + wrap) | uncertain, likely 1400–1700 tok/s |
| FlashAttention-2 (Dao-AILab upstream)           | wheels in `~/Downloads/`, CDNA-only HIP path                                                                          | no            | n/a     | not applicable on W7900 |
| Patch the existing `<false>` branch in place    | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.{hip,py}`                                                  | yes (ours)    | unclear | uncapped; the path needs an FA rewrite, not a patch |

AOTriton specifics worth recording so a future agent does not re-derive them:

- `AOTRITON_NS::v2::flash::attn_fwd_compact_varlen` takes `cu_seqlens_q`,
  `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`, `is_causal`, and a
  `hipStream_t`-equivalent stream. This matches our existing
  `CompactPromptSlab.cu_seqlens_q/k` ABI almost verbatim and the `KVLiveSpans`
  prefill role; no scheduler changes are required.
- The tensor type is `AOTRITON_NS::TensorView<N>` — a `(void* ptr, shape[N],
  stride[N], dtype)` descriptor. There is no `torch::Tensor` anywhere in the
  AOTriton public headers (`include/aotriton/{flash,runtime,util,dtypes,cpp_tune}.h`).
- Disk footprint on gfx110x, verified on this host
  (`du -sh ~/Downloads/aotriton/aotriton/lib/aotriton.images/amd-gfx110x/flash/*`):
    - `libaotriton_v2.so.0.8.0` = 28 MB.
    - `flash/attn_fwd/` = 49 MB across 480 forward variants.
    - `flash/bwd_kernel_dk_dv/` = 26 MB, `flash/bwd_kernel_dq/` = 24 MB,
      backward-preprocess + debug ≈ 0.4 MB. We are inference-only; drop all
      `bwd_*` and `debug_*` subdirs.
    - Inference-only ship (`.so` + `attn_fwd/` full): **76 MB**.
    - Aggressive prune to the variants we actually call
      (bf16/fp16 × head_dim 128 × causal=true × dropout=false × no-bias):
      32 `.aks2` binaries totalling ≈ 3 MB images + 28 MB `.so` = **31 MB**.
      `ls FONLY__^{bf16,fp16}@16,False,128,*.aks2 | wc -l` confirmed 32.
- The 480 pretuned variants do per-shape kernel selection at call time; this
  is the value we would lose by hand-rolling. The aggressive prune is safe
  because hipengine's attention shape set is fixed at model-load time and
  small (one head_dim, causal-only).

### Why "surely native HIP beats Triton" is not a fast path

Triton lowers to AMDGPU LLVM IR through MLIR and emits the same instruction
class — `v_wmma_*`, `ds_read_b128`, `v_dual_*` — that a hand-written HIP
kernel would. On gfx1100 the Triton tax over a perfectly-tuned hand kernel is
typically 5–15 %, often less. The existing `<false>` branch is not slow
because HIP cannot match Triton; it is slow because it does not implement the
Flash-Attention algorithm. Catching up to AOTriton requires implementing FA-2
correctly first; only then does the per-shape hand-tuning headroom open up.

A hand-written FA-2 spike that lands in less than 30 days will almost
certainly underperform AOTriton on gfx1100, because AOTriton already ships
shape-specialized binaries and our spike will run one tile schedule. Using
AOTriton as the perf oracle is what makes a later native port tractable.

### Recommended phased plan

**Phase 1 — AOTriton attention plugin** (next multiloop spike).

- Add a new kernel-build module under `hipengine/kernels/hip_gfx1100/attention/`
  that loads `libaotriton_v2.so` via `ctypes`/dlopen and exposes a thin C
  wrapper for `attn_fwd_compact_varlen` (the varlen path matches
  `CompactPromptSlab`).
- Register `KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro",
  "aotriton_attn_fwd")` alongside
  `(... , "qwen35_causal_gqa_gate_fp16")`. No `if backend=="..."`,
  no `if quant=="..."`; the model layer asks the registry for an attention
  prefill key and gets one. The existing kernel stays registered as the
  short-T variant.
- Threshold via `PrefillConfig.attn_aotriton_min_tokens` (default 1024,
  re-measured); decode and short prefill continue on the existing hand-rolled
  kernel where it is fine.
- Add a tiny **gate-fusion post-pass kernel** in the same module:
  `out[row, q_head, dim] *= sigmoid(gate[row, q_head, dim])` over the
  AOTriton output. The existing prefill kernel fuses this inside its
  epilogue (`paged_attn_decode.hip:1191`) and we must preserve the
  semantics. Single elementwise pass; ≤ 0.2 ms at T=4K, head_dim=128,
  num_q_heads=16. Reuse the existing decode-side gate kernel pattern at
  `paged_attn_decode.hip:316,329` for the math; only the launch shape
  changes.
- Do not vendor the AOTriton binary into the hipengine git tree. Use the
  fetch-on-install + pinned-manifest scheme described in "AOTriton distribution
  and pinning strategy" below; resolve `libaotriton_v2.so` and the kernel
  images via the documented lookup chain at module load.
- Correctness gate: re-run `scripts/qwen35_native_prefill_fixture_gate.py` on
  `fixtures/qwen35_paro/parent_512_32_seed1234.json` and the 4K repeated-token
  diagnostic; require `passed=true`, `max_kl <= 0.05`, top-1 ≥ 90 %, and
  generated IDs equal to the serial path with the AOTriton variant active.
- Perf gate: 512/128 median prefill_tok_s ≥ current best (2039), 4K/128 ≥ 1500
  before a row is retained; target band 4K/128 ≥ 1700 after one round of
  threshold tuning.
- Plugin-registry compliance is the load-bearing invariant here: AOTriton
  enters as a new variant key, never as a branch in dispatch code.

**Phase 2 — hipBLASLt for MoE projection at T ≥ 1K** (parallel, independent).

- Wrap `hipblasLtMatmul` from `/opt/rocm/lib/libhipblaslt.so.1.2` for the
  shared-expert W8A16 path and grouped-stacked MoE projection at T ≥ 1K, where
  nano-vllm-amd's trace shows it dispatches HGEMM tiles instead of compact
  WMMA. Register as `(hip_gfx1100, shared_expert | moe_prefill, w8a16 |
  w4_paro, hipblaslt_hgemm)` variants; compact WMMA stays as the short-T
  variant.
- Expected delta: 5–10 % at 512, 15–25 % at 4K, on top of Phase 1.

**Phase 3 — native HIP FA-2 port** (optional, only if AOTriton bundle is
unacceptable or per-shape headroom is measurable).

- Reference: vLLM-vendored CK FA on `gfx1100;gfx1101;gfx1102`, ck_tile/01_fmha
  algorithm pattern. AOTriton output is the correctness and perf oracle.
- 3–6 weeks. Expected per-shape gain over AOTriton: 0–15 % at our exact shape
  (head_dim 128, kv_heads 2, num_q_heads 16, causal, BF16 paged cache,
  post-gate FP16 output). The unique win is fusing the existing post-gate
  semantics into the FA epilogue, which AOTriton cannot do for us.

### AOTriton distribution and pinning strategy

AOTriton (`https://github.com/ROCm/aotriton`) is under active development:
ABI churn is real between minors, release artifacts are matrixed across ROCm
minors, and the version PyTorch bundles is mangled (the conda installs on this
host show `libaotriton_v2.so.torch` symlinks under `torch/lib/`). We need a
scheme that gives us a deterministic build without inheriting any of that
churn.

#### What not to do

- **Do not add AOTriton as a git submodule and build from source.** AOTriton's
  source build compiles 480+ Triton kernel variants per architecture; it
  requires a working AMDGPU Triton fork and several GiB of build output, and
  takes hours on first run. Our CI does not need that surface, and AGENTS.md's
  git rules forbid committing compiled `.so` / JIT caches anyway.
- **Do not vendor the release tarball or extracted binaries into the
  hipengine repo.** AGENTS.md git rules forbid committing compiled `.so` and
  prebuilt kernel images. Binary blobs in-tree also make pin bumps unreviewable
  (a routine version bump becomes a multi-MB binary diff that no one can read
  in PR), and they couple repo state to a specific ROCm-minor build target.
  Footprint itself is not the issue — 76 MB inference-only or 31 MB pruned is
  a rounding error next to model weights — but committing binaries breaks the
  review and provenance contract.
- **Do not depend on PyTorch's bundled AOTriton.** The PyTorch installs we
  found here ship under `torch/lib/libaotriton_v2.so{,.torch,.0.8.0}` with
  PyTorch-specific symlinks. Reading from a PyTorch install couples our
  runtime to a torch version we do not import. AGENTS.md says `import torch`
  is forbidden on the hot path; reading from `torch/lib/` is the spiritual
  cousin of that violation.
- **Do not rely on `/opt/rocm/lib/libaotriton_v2.so`.** It is not shipped
  there on this host (verified `ls /opt/rocm/lib/` 2026-05-16), and even when
  a future ROCm release does ship it, the bundled version will lag.

#### What to do: fetch-on-install with a pinned manifest

Land a manifest file under
`hipengine/kernels/hip_gfx1100/attention/aotriton_release.toml` recording the
pin. Concrete recommended starting pin (matches the 0.8 binary already on this
host and the symbols we inspected):

```toml
[aotriton]
version = "0.8.2b"
git_sha1 = "33fb6bd5290b2e9e9bc71dbcf91f92c6ba7689b1"  # from include/aotriton/config.h
so_name = "libaotriton_v2.so.0.8.0"
rocm_min = "6.2"
rocm_max = "7.x"   # 0.8 needs libamdhip64.so.6 via ROCm ABI-compat shim

[aotriton.archive]
url = "https://github.com/ROCm/aotriton/releases/download/0.8.2b/aotriton-0.8.2b-manylinux_2_28_x86_64-rocm6.3-shared.tar.gz"
sha256 = "<fill in on bump>"
size_bytes = 374748255

[aotriton.prune]
# Keep only the kernel variants hipengine actually calls on gfx1100.
architectures = ["amd-gfx110x"]
flash_subdirs = ["attn_fwd"]   # forward only; we do not train
dtypes = ["bf16", "fp16"]
head_dims = [128]
causal = [true]
# 30 MB after pruning; suitable for ~/.cache placement.
```

Resolve at module load (mirror the dlopen pattern in `hipengine/core/hip.py`)
with this lookup chain, in order:

1. `HIPENGINE_AOTRITON_LIB` env var → use directly (developer override).
2. `${HIPENGINE_AOTRITON_HOME:-~/.cache/hipengine/aotriton}/<version>/lib/libaotriton_v2.so`
   → the fetch-on-install destination. Version-check the file's SONAME against
   the manifest and the AOTriton `images/<arch>` directory layout.
3. (Optional, gated) `/opt/rocm/lib/libaotriton_v2.so` → only when its SONAME
   matches the manifest band; warn otherwise.
4. Nothing found → emit one clear error pointing at
   `scripts/fetch_aotriton.sh`, and fall the registry back to the existing
   `qwen35_causal_gqa_gate_fp16` kernel for that prefill call so generation
   still works (degraded long-T perf, but no crash).

Ship a one-shot fetch helper as `scripts/fetch_aotriton.sh` (and a Python
twin `hipengine.aotriton.ensure_installed()` for SDK use):

```bash
scripts/fetch_aotriton.sh
  --manifest hipengine/kernels/hip_gfx1100/attention/aotriton_release.toml
  --dest ~/.cache/hipengine/aotriton
  [--prune]            # default true; strip non-gfx110x and non-causal binaries
  [--no-verify-sha]    # opt-out for offline mirrors only
  [--force]            # re-extract even if dest exists
```

It downloads to the manifest-listed URL, verifies SHA256, extracts to
`<dest>/<version>/`, optionally prunes per the manifest, writes a
`MANIFEST.local.json` recording (sha256, fetched_at, prune_state), and exits.
This is *not* part of `pip install hipengine`; it is one explicit step in the
bring-up checklist, recorded in `WORKLOG.md` per AGENTS.md.

#### Why not just `pip install aotriton`

There is no published PyPI wheel for AOTriton that ships the gfx110x tile
database usefully on its own. The pip distribution channel is PyTorch's, which
is the path we are explicitly avoiding. The tarball under
`https://github.com/ROCm/aotriton/releases/` is the upstream-blessed standalone
distribution; that is what we pin against.

#### Stable-ABI shim, not raw dlopen

The C++ symbol we want is mangled
`_ZN8aotriton2v25flash23attn_fwd_compact_varlenENS_10TensorViewILi4EEES3_...`.
Linking Python `ctypes` against that mangled name is fragile across AOTriton
minors. The safer pattern, which the existing hipengine JIT pipeline already
supports:

1. Add `hipengine/kernels/hip_gfx1100/attention/aotriton_wrap.{hip,cc,py}`.
2. The `.cc` includes `<aotriton/flash.h>` from the resolved
   `${HIPENGINE_AOTRITON_HOME}/<version>/include/` and links against the
   matching `lib/libaotriton_v2.so`, exposing a small `extern "C"` surface:
   `hipengine_aotriton_attn_fwd_compact_varlen(...)`.
3. The `.py` does the dlopen of *our* wrapper .so (built by the same hipcc
   JIT path that builds all other gfx1100 kernels), not AOTriton directly.
4. Bumping AOTriton means: bump the manifest, re-run `fetch_aotriton.sh`,
   re-JIT the wrapper (seconds), re-run the fixture gate.

This isolates ABI churn to one ~50-line C++ file. If AOTriton 0.9 renames the
entrypoint, only the wrapper needs to change; the hipengine kernel-registry
key, the runtime call site, and the Python ABI stay unchanged.

#### Concrete version on this host

- Available locally now: `~/Downloads/aotriton/aotriton/` (extracted 0.8.0
  build, `AOTRITON_VERSION_{MAJOR,MINOR,PATCH}=0,8,0`, `AOTRITON_GIT_SHA1
  33fb6bd5290b2e9e9bc71dbcf91f92c6ba7689b1`), and the matching tarball
  `~/Downloads/aotriton-0.8.2b-manylinux_2_28_x86_64-rocm6.3-shared.tar.gz`
  (357 MB compressed, 374 MB uncompressed full matrix, ≈30 MB pruned to
  gfx110x + causal + head_dim 128).
- The shared library NEEDED list includes `libamdhip64.so.6`, `liblzma.so.5`,
  `libstdc++.so.6`. System ROCm here is 7.2.2; ROCm ships an ABI compat
  symlink for `libamdhip64.so.6 -> libamdhip64.so.7`, so 0.8 loads on 7.x in
  practice. The manifest pin should record the ROCm-minor build target so a
  future ROCm-9 install does not silently load an incompatible shim.
- The single C-API entry we need (`attn_fwd_compact_varlen`) was introduced in
  AOTriton 0.7.x and remains in 0.8.x. We are not chasing a moving target on
  the surface area we actually consume, only on the surrounding ecosystem.
- Side note: `pytorch/pytorch#166397` (Nov 2025) marked gfx1100 as
  "experimental" in PyTorch's SDPA backend matrix. That is a PyTorch QA
  policy decision about which backends ship as production-grade defaults,
  *not* a statement about AOTriton kernel correctness on gfx1100; AOTriton's
  own gfx110x images continue to be released and tested. hipengine calls
  AOTriton directly via its C++ ABI and is unaffected by the PyTorch
  dispatch policy.

#### Production decision (2026-05-16)

Production target is the fetch-on-install + pinned-manifest scheme above
("What to do"). The in-flight Phase 1 spike may use any pattern that lands
working AOTriton-backed attention quickly — including a temporary submodule
or system-library probe — but the cleanup pass must converge on:

- A pinned `aotriton_release.toml` manifest in-tree.
- `scripts/fetch_aotriton.sh` (+ `hipengine.aotriton.ensure_installed()`) as
  the install path.
- Lookup chain at module load with graceful fallback to the existing
  hand-rolled kernel when AOTriton is absent (so `pip install hipengine`
  alone produces a usable, correct, slower-at-long-T install).
- A stable-ABI C++ wrapper (`aotriton_wrap.cc`) that hipengine owns; no
  raw-`ctypes`-against-mangled-C++-symbols dlopen on the hot path.
- No git submodule retained, no AOTriton binary tracked in git.

If the spike commits a submodule or vendored binary, that lands behind
`?? .gitmodules` / `?? third_party/aotriton` etc. only as a transient state;
the follow-up cleanup PR removes them and replaces with the manifest +
fetcher.

### Explicit non-goals for the next spike

- Do not patch the `<false>` branch of the existing prefill attention kernel.
  The algorithmic gap is FA-vs-not-FA, not tile-tuning; patches there waste
  iterations.
- Do not write a from-scratch FA-2 before AOTriton is wrapped. Without an
  oracle ceiling we cannot tell a good hand-rolled kernel from a mediocre one;
  iters 1–49 demonstrate the cost of optimizing without one.
- Do not introduce `import torch` in `hipengine/runtime/`,
  `hipengine/generation/`, `hipengine/models/`, `hipengine/dispatch/`, or any
  kernel module reached by `LLM.generate()`. AOTriton is loaded via dlopen and
  called through `ctypes`; the existing dlopen pattern in
  `hipengine/core/hip.py` is the template.
- Do not branch on `backend == "..."` or `quant == "..."` in dispatch or model
  code to route to AOTriton. Use the kernel registry; that is what the
  four-axis design exists for.

### Reproduction commands

Trace comparison evidence above was produced with:

```bash
# hipENGINE 512/0 trace
rocprofv3 --kernel-trace -d /tmp/iter50-shared-down-tile8-trace -o trace -- \
  python3 scripts/qwen35_paro_bench.py --token-id 9707 \
    --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 \
    --max-layers 40 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build --json /tmp/iter50.json

# hipENGINE 4K/0 trace
rocprofv3 --kernel-trace -d /tmp/iter52-4k-profile-trace -o trace -- \
  python3 scripts/qwen35_paro_bench.py --token-id 9707 \
    --prompt-length 4096 --decode-tokens 0 --warmup-decode-tokens 0 \
    --max-layers 40 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build --json /tmp/iter52.json
```

Parent comparison numbers (2589 tok/s @ 512, 1681 tok/s @ 4K) are from a
peer-system audit at the same git tree under
`~/amd-gpu-tuning/scripts/bench_paro_native_engine.py --model-preset
qwen35-a3b-paro --prompt-len {512,4096} --decode-len 0 --prefill-mode bulk
--no-warmup` with the `COMMON_ENV` flags from
`~/amd-gpu-tuning/scripts/run_moe2_baselines.py`. Re-running the parent profile
on this host is blocked behind ongoing GPU contention; the numbers above are
recorded against the parent audit transcript and treated as the comparison
baseline until reproduced locally.

## References

- `docs/PLAN.md` — architecture, phase roadmap, extensibility, KV ABI.
- `docs/KERNELS.md` — live kernel catalog and port playbook.
- `docs/BENCHMARK.md` — benchmark protocol and artifact rollup rules.
- `docs/TESTING.md` — RED/GREEN workflow, fixtures, correctness gates.
- `docs/ROOFLINE.md` — W7900/RDNA3 performance model.
- `docs/DFLASH.md` — related speculative path using the same batch-shaped ABI.
- `~/amd-gpu-tuning/docs/PARO.md` — parent retained rows and config.
- `~/amd-gpu-tuning/docs/OPTIMAL.md` — parent optimal Qwen3.5/PARO route and flags.
- `~/amd-gpu-tuning/scripts/bench_paro_native_engine.py` — parent `prefill_bulk(...)` reference.
- `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant.py` — parent layer implementations.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-attn-boundary-blocked.json` — current full-attention boundary.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-scratch-restore-sweep.json` — accepted linear-prefix correctness.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-serial-suffix-full40-accepted.json` — accepted legacy suffix correctness.
- `benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json` — current c=1 perf/correctness baseline.
- `benchmarks/results/2026-05-15-hipengine-qwen35-c8-scheduler-serial-bench-blocked.json` — current c=8 serial bridge diagnostic.
