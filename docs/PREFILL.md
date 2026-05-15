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
| Linear-attn prefill | `qwen35_linear_attn_conv_prefill_f32`, `qwen35_linear_attn_prefill_prepare_f32_fp16`, `qwen35_gdn_prefill_recurrent_k2_f32`, `qwen35_gdn_prefill_rmsnorm_gate_fp16`. |
| Linear layer orchestrator | `run_linear_attention_moe_c1_layer_fp16(tokens=T)` already selects prefill conv/GDN when `tokens > 1`; final path must replace its c1 MoE tail with grouped MoE. |
| Full-attention decode/prelude | Existing c=1 Q/K/V projection, vector-position RoPE prefill prelude, KV append, native append-then-attend causal GQA prefill kernel, context/GQA decode, gate, output projection. Decode kernels remain useful as oracle only for prefill attention. |
| KV append | `qwen35_write_paged_kv_mixed_value_fp16_prompt_spans(...)` appends all prompt rows into one request cache; row-major `*_batch_spans(...)` remains for c>N-shaped caches. Both consume per-row append positions in `spans.live_counts`. |
| KV metadata | `KVLiveSpans` already carries `request_ids`, `row_positions`, and `span_role`; compact prefill needs wiring/population, not a span redesign. |
| Graph primitives | `hipengine.core.hip.HipRuntime` exposes HIP graph capture/instantiate/launch; decode graph capture exists. |

Missing for the final path:

| Area | Required final work |
| --- | --- |
| Public API wiring | `prefill_native(...)` API/config skeleton exists; wire the final native implementation into it and update generation call sites after native kernels land. |
| Full-attn retained orchestration | Wire batched Q/K/V + vector RoPE + KV append + native causal prefill attention into the retained full-attention layer path and prove contiguous per-row Q/K/V/gate layouts in stage probes. |
| Grouped/compact MoE | Port/wire parent grouped scatter/gather and grouped expert kernels. |
| Compact c>N slab | Build packed prompt metadata and segment-aware linear-attn/full-attn kernels. |
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

- `ResidentBatchScheduler.next_prefill_work(chunk_size=...)` forms compact slabs
  for requests with prefill work.
- Add an explicit `bucketize_by_block_count` step in the scheduler before slab
  construction.
- `Qwen35ParoResidentSession.prefill_native_packed(slab)` runs the same native
  layer logic over `T_total` rows.
- Current batch KV writer constraint: `_check_write_batch_shape(...)` computes
  one `block_table_len = base_offsets.numel // rows`, so every row in a writer
  call must expose the same block-table length. Final scheduler policy should
  bucket slabs by common `blocks_per_request`; cross-bucket requests launch as
  separate native slabs. A true varlen block-table writer is a future kernel
  port, not a reason to use a serial per-request fallback.
- Linear-attention conv/GDN must be segment-aware so each request's tail state is
  preserved independently. Check the parent for any
  `linear_attn_conv_prefill_segments_*` equivalent before writing a new kernel;
  if absent, implement/port the segment-aware kernel rather than retaining
  per-request invocation.
- Native causal prefill attention must be var-len/block-diagonal: a query row may
  attend only to rows/cache positions for the same request id and positions not
  greater than the query position.
- Native compact prefill is non-speculative and commits canonical KV inline for
  admitted prompt rows. `KVPolicy.begin_transaction/commit/rollback` hooks remain
  for speculative verify/draft paths, not this ordinary prefill path.
- Replace `scheduler_serial_slot_bridge` with `native_prefill_compact_cN` only
  after equality gates pass.

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
3. Native compact kernels run; no per-request prompt loop in the retained path.
4. At c=8/T=512, prefill tok/s improves over `scheduler_serial_slot_bridge` by
   at least 2× before retaining a throughput claim.

Retained artifact target:

```text
benchmarks/results/2026-05-XX-hipengine-qwen35-native-prefill-compact-c8-accepted.json
```

Any retained performance row also updates `benchmarks/README.md`,
`benchmarks/CHANGELOG.md`, and a compact JSON artifact under
`benchmarks/results/`.

Correctness is non-negotiable: a faster prefill that fails the parent fixture is
a regression, not a win.

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
