# hipENGINE Batched Prefill Plan

> Status: implementation spec, corrected 2026-05-15. This document is the
> authoritative punchlist for moving Qwen3.5-35B-A3B-PARO prefill from
> token-by-token resident decode replay to native bulk prefill. `docs/PLAN.md`
> remains the architecture source of truth; update both files if the architecture
> changes.

## TL;DR

**The immediate gap is orchestration, not the first kernel port.** hipENGINE now
has a correctness-accepted native linear-attention prefix, but the resident
prefill path is still token-major once it reaches the first full-attention
layer.

Current state:

- `prefill_linear_tokens_native(...)` embeds a full prompt into `[T, hidden]`,
  runs the first 3 linear-attention layers in bulk, then falls back to a serial
  token-major suffix. This is correctness-accepted but is not full native
  prefill.
- `Qwen35ParoDecodeState.run_linear_attention_moe_c1_layer_fp16(tokens=T)`
  already runs the linear-attention prefill conv/GDN path plus the existing
  selected-MoE row path over `T` rows.
- `Qwen35ParoDecodeState.run_full_attention_moe_c1_layer_fp16(...)` still raises
  when `tokens != 1`; full-attention prefill is the first hard boundary.
- The batch KV writer exists, and `KVLiveSpans` already has `request_ids` and
  `row_positions`; the resident prefill path does not yet populate/use the
  packed prefill metadata.
- Grouped/compact MoE and a true causal full-attention prefill kernel are still
  needed for parent-level throughput, but they are not required to land the next
  correctness-preserving implementation step.

**Implementation order:**

1. Add the `prefill_native(...)` public session API as the non-legacy name for
   the existing native-prefix helper.
2. Replace the token-major serial suffix with a **layer-major bulk orchestrator**:
   linear layers run over `[T, hidden]`; full-attention layers use an explicitly
   labelled row loop through the existing c=1 path until the native causal kernel
   lands.
3. Wire this path into `LLM.generate`/`Qwen35ParoOneTokenGenerator` once it
   matches the current serial resident fixture.
4. Add the native causal full-attention prefill kernel and grouped MoE prefill as
   performance upgrades, each with its own correctness gate and benchmark
   artifact.
5. Only after single-request bulk prefill is correct, implement compact c>N
   prompt slabs.

Scope note: this plan targets `z-lab/Qwen3.5-35B-A3B-PARO` MoE hybrid. Dense
`Qwen3.5-0.8B-PARO` needs tied-lm-head and dense PARO MLP support first; that is
a separate loader/runtime task, not part of this prefill plan.

## Terms and shapes

| Term | Meaning |
| --- | --- |
| `T` | Prompt rows for one request in a single prefill call. |
| `T_total` | Rows in a compact slab packed across multiple requests. |
| `C` | Active decode requests; not the same as prompt rows. |
| Bulk prefill | Layer input/output buffers are `[T, hidden]`; kernels operate on prompt rows. |
| c1-style row MoE | Existing selected expert GEMV path with `tokens=T` and `rows=T * top_k`; correct but not grouped by expert. |
| Grouped/compact MoE | Parent-style route that scatters rows by expert and runs grouped bulk kernels; needed for peak prefill throughput. |
| D2 fallback | Full-attention prefill row loop through existing c=1 full-attention decode kernels inside a layer-major bulk orchestrator. Correctness unblocker, not a perf endpoint. |
| D1 native attention | New causal multi-query/GQA prefill kernel over `[T, heads, head_dim]`; true throughput path. |

KV span convention for this repo:

- KV **append** spans use `live_counts[row] = absolute_position` (0-based write
  position), matching the preserved parent writer ABI.
- KV **attention/decode** spans use `live_counts[row] = context_length` (1-based
  visible length).
- For prefill row `r` starting at `start_position`, append position is
  `start_position + r`; attention context length is `start_position + r + 1`.

## Evidence: where the gap is

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
| c=8 8/1 | 115.08 | 108.89 | `scheduler_serial_slot_bridge` diagnostic, not native compact batching |

Correctness/blocker artifacts already retained:

- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-scratch-restore-sweep.json`
  — native linear prefix accepted through layers 0..2.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-serial-suffix-full40-accepted.json`
  — native linear prefix plus token-major serial suffix matches serial resident
  outputs; no throughput claim.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-attn-boundary-blocked.json`
  — first native prefill boundary is layer 3 full attention.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-plan-blocked.json`
  — 40-layer plan reports 30 linear-attention and 10 full-attention layers, with
  native coverage currently stopping after the first 3 linear layers.

Conclusion: decode is close enough to parent to proceed. Prefill is roughly one
to two orders of magnitude behind because the model-level prefill path still
executes most layers token by token.

## Parent bulk-prefill reference

Reference files:

- `~/amd-gpu-tuning/scripts/bench_paro_native_engine.py`
- `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant.py`

Parent entry point shape:

```python
hidden = embed_tokens(prompt_ids).view(-1, hidden_size)      # [T, H]
positions = self.position_ids[:T]                             # device tensor
for layer in self.layers:
    if isinstance(layer, ParoQuantLinearAttentionLayer):
        hidden = layer.prefill_native(hidden, linear_states[layer.id])
    else:
        hidden = layer.prefill_native(hidden, full_caches[layer.id], positions=positions)
return lm_head(final_norm(hidden[-1:]))                       # last row only
```

Per-layer responsibilities:

- **Linear-attention layer**
  - input RMSNorm over `T` rows;
  - PARO rotation/projections over `[T, H]`;
  - conv prefill and GDN recurrent prefill, updating per-layer conv/recurrent
    state for the prompt tail;
  - output projection;
  - post-attention add/RMSNorm;
  - MoE over all rows.

- **Full-attention layer**
  - input RMSNorm over `T` rows;
  - PARO Q/K/V projections;
  - Q/K head RMSNorm and RoPE using per-row positions;
  - KV append for all prompt rows;
  - causal multi-query attention over prior prefix plus prompt rows up to the
    current query row;
  - gate/output projection;
  - post-attention add/RMSNorm;
  - MoE over all rows.

Parent tuning knobs to mirror later through `PrefillConfig`, not hot-path env
lookups:

| Parent env var | Meaning |
| --- | --- |
| `NANOVLLM_PARO_PREFILL_LINEAR_CHUNK_SIZE` | Split linear-attn prefill into row chunks. |
| `NANOVLLM_PARO_PREFILL_FULL_ATTN_QUERY_CHUNK_SIZE` | Q-row chunk for full-attn prefill. |
| `NANOVLLM_PARO_PREFILL_FULL_ATTN_POST_CHUNK_SIZE` | Post-attention residual+MoE chunk. |
| `NANOVLLM_PARO_PREFILL_FULL_ATTN_ROPE_CHUNK_SIZE` | RoPE/Q/K-norm inplace chunk. |
| `NANOVLLM_PARO_MOE_STACKED_COMPACT` | Use stacked compact MoE route. |
| `NANOVLLM_PARO_MOE_GROUPED_DEVICE_GATHER` | Use device-side grouped scatter/gather for MoE. |
| `NANOVLLM_PARO_GEMV_V8` | Pack8 GEMV route. |
| `NANOVLLM_PARO_NATIVE_ROUTER` | Native router top-k. |

## hipENGINE inventory and corrected status

Reference: `docs/KERNELS.md` is authoritative for exact landed kernels and gates.

### Landed and usable now

| Area | Current usable pieces |
| --- | --- |
| Runtime state | `embedding_lookup_batch_{bf16,fp16}_i64`, mapped variants, `set_i64_vector`, scalar/vector decode position helpers. |
| Linear-attn prefill | `qwen35_linear_attn_conv_prefill_f32`, `qwen35_linear_attn_prefill_prepare_f32_fp16`, `qwen35_gdn_prefill_recurrent_k2_f32`, `qwen35_gdn_prefill_rmsnorm_gate_fp16`. |
| Linear layer orchestrator | `run_linear_attention_moe_c1_layer_fp16(tokens=T)` already chooses prefill conv/GDN when `tokens > 1`. |
| c1-style row MoE | Router, selected pack8 gate/up/down, shared W8A16, and batched combine wrappers accept `tokens=T`; this is correct but not grouped/compact. |
| Full-attention decode | Existing c=1 Q/K/V projection, scalar-position RoPE, KV append, context/GQA decode, gate, output projection. |
| KV append | `qwen35_write_paged_kv_mixed_value_fp16_batch_spans(...)` exists; it consumes per-row append positions in `spans.live_counts`. |
| KV metadata | `KVLiveSpans` already carries `request_ids`, `row_positions`, and `span_role`; no schema extension is needed for compact prefill metadata. |
| Graph primitives | `hipengine.core.hip.HipRuntime` exposes HIP graph capture/instantiate/launch; `Qwen35ParoResidentSession.capture_decode_graph` exists for decode. |

### Missing or not wired

| Area | Correction / required work |
| --- | --- |
| Public prefill API | `prefill_native(...)` does not exist yet; current public helper is legacy-named `prefill_linear_tokens_native(...)`. |
| Model-level bulk orchestrator | Current `_run_prefill_suffix_layers_serial(...)` is token-major. It must become layer-major so linear layers after full-attention boundaries can still run bulk. |
| Full-attention `tokens > 1` | `project_full_attention_qkv_fp16`, `prepare_full_attention_qkv_fp16`, and `run_full_attention_moe_c1_layer_fp16` reject `tokens != 1`. D2 avoids this by row looping; D1 removes it. |
| Batched RoPE positions | Existing `qwen35_head_rmsnorm_partial_rotary_position_f32_bf16` reads `position_ptr[0]`; it is scalar-position only. A true D1 path needs a vector-position variant or a row loop. |
| Causal full-attn prefill | No HIP kernel yet for multi-query/GQA causal prefill over `[T, heads, dim]`. |
| Grouped/compact MoE | Parent grouped scatter/gather, selected grouped W8A16/W8A8, and optional WMMA grouped GEMM are lineage-green but not ported. |
| Prefill graph cache | General shape-bucket prefill graph capture/replay is not implemented; only low-level HIP wrappers and decode capture exist. |

## Target session API and state contract

Phase A adds this API to `Qwen35ParoResidentSession`:

```python
def prefill_native(
    self,
    token_ids: Sequence[int],
    *,
    sample: bool = True,
    require_full_native: bool = False,
) -> Qwen35ParoAutoregressiveStepResult | None:
    """Prefill one request from position 0 through len(token_ids)-1.

    If sample=True, return next-token logits/argmax from the final prompt row.
    If require_full_native=True, raise NotImplementedError whenever the path
    would use a labelled serial fallback. Generation should use the default
    False until D1/D grouped-MoE are accepted.
    """
```

Semantics:

- Validate non-empty `token_ids`, vocab bounds, and `len(token_ids) <=
  max_sequence_length` before launching kernels.
- Allocate/copy one int64 token vector and one int64 position vector
  `[0, 1, ..., T-1]`. Reuse session-owned buffers once A lands; temporary
  buffers are acceptable for the first shell.
- Embed all tokens into `self.prefill_hidden` as `[T, hidden]` with
  `embedding_lookup_batch_fp16_i64`.
- Run `_run_prefill_layers_layer_major(tokens=T, start_position=0)`.
- Copy the last row into `self.hidden` for subsequent decode.
- Call `_restore_decode_scratch_after_prefill()` before returning to decode.
- Set `position_buf = T - 1` and `context_buf = T`, so the next decode step at
  position `T` can consume the sampled seed token and append at the correct
  location.
- If `sample=False`, perform all state/KV updates but return `None`.
- Keep `prefill_linear_tokens_native(...)` as a compatibility alias only; remove
  `allow_rejected_correctness` from new call sites.

Path labels used in artifacts/logs:

| Label | Meaning | Throughput claim eligible? |
| --- | --- | --- |
| `serial_step_loop` | Existing per-token `session.step(...)` loop. | Baseline only. |
| `linear_prefix_token_major_suffix` | Current helper: bulk first 3 linear layers, token-major suffix. | No. |
| `layer_major_serial_full_attention` | New B path: all linear layers bulk, full-attn rows c=1. | Yes, if correctness gates pass, but label as partial native. |
| `full_attention_prefill_native` | C path with causal prefill attention kernel. | Yes. |
| `grouped_moe_prefill_native` | D path with grouped/compact MoE. | Yes. |
| `compact_prefill_cN` | F path with multi-request prompt slab. | Yes, after c>N equality gate. |

## Phased implementation plan

Each phase emits either an accepted or blocked artifact under
`benchmarks/results/`. Any landed kernel also updates `docs/KERNELS.md`, runs
`python3 scripts/check_lineage.py --kind kernel --diff stat` before porting, and
passes the `docs/TESTING.md` gate.

### A. Foundation — public API, buffers, and position metadata

Goal: make the supported prefill entry point explicit without changing math.

Status today:

- `prefill_hidden` / `prefill_next_hidden` are allocated per session.
- `_restore_decode_scratch_after_prefill()` exists.
- Batched embedding lookup exists.
- `set_i64_vector` exists, but no session-owned `prefill_positions` table is
  wired.
- `prefill_linear_tokens_native(...)` is accepted for the 3-layer linear prefix
  and serial suffix correctness.

Punchlist:

- [ ] Add session-owned `prefill_token_ids: int64[max_sequence_length]` and
  `prefill_positions: int64[max_sequence_length]`, or document a temporary
  allocation as the initial implementation.
- [ ] Add `prefill_native(...)` with the contract above.
- [ ] Keep `prefill_linear_tokens_native(...)` as a thin alias to
  `prefill_native(...)` for existing scripts/tests.
- [ ] Update `qwen35_paro_native_prefill_plan(...)` labels so artifacts
  distinguish legacy token-major suffix from the new layer-major path.
- [ ] Add/adjust unit tests in `tests/test_qwen35_resident_batch_layout.py` for
  validation errors and alias behavior.

Gate:

```bash
python3 -m pytest tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_native_prefill_correctness.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --token-id 9707 --prompt-length 4 --sweep-layer-prefixes 3 \
  --json benchmarks/results/2026-05-XX-hipengine-qwen35-prefill-native-api-linear-prefix-accepted.json
```

### B. Layer-major bulk orchestrator with labelled serial full-attention fallback

Goal: make the whole 40-layer model prefill through one layer-major bulk path.
Linear-attention layers run over `[T, hidden]`; full-attention layers initially
loop rows through the existing c=1 full-attention path. This is the next
implementation step.

Why this matters: the current serial suffix is token-major. After layer 3, it
runs every later linear layer one token at a time, so the landed linear prefill
kernels only help the first 3 layers. A layer-major orchestrator lets all 30
linear-attention layers use the existing bulk path even before D1 exists.

Implementation sketch:

```python
def _run_prefill_layers_layer_major(self, *, tokens: int, start_position: int = 0, stream: int = 0) -> Tensor:
    hidden = view(self.prefill_hidden, (tokens, hidden_size))
    next_hidden = view(self.prefill_next_hidden, (tokens, hidden_size))

    for layer_id, state in enumerate(self.states):
        if layer_type[layer_id] == "linear_attention":
            out = state.run_linear_attention_moe_c1_layer_fp16(
                hidden,
                conv_state=self.linear_states[layer_id][0],
                recurrent_state=self.linear_states[layer_id][1],
                linear_scratch=reserve(tokens),
                moe_scratch=reserve(tokens),
                tokens=tokens,
                library=self.libraries,
                stream=stream,
            )
            copy out -> next_hidden[T, H]

        elif layer_type[layer_id] == "full_attention":
            # D2 fallback: correctness unblocker.
            key_cache, value_cache = self._slot_full_cache(layer_id, 0)
            for row in range(tokens):
                position = start_position + row
                self._set_position(position, stream=stream)
                position_tensor, append_spans, decode_spans = self._slot_spans(0)
                row_hidden = row_view(hidden, row)
                row_out = state.run_full_attention_moe_c1_layer_fp16(
                    row_hidden,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    append_spans=append_spans,  # live_counts == position
                    decode_spans=decode_spans,  # live_counts == position + 1
                    position=position_tensor,
                    tokens=1,
                    ...,
                )
                copy row_out -> next_hidden[row]

        hidden, next_hidden = next_hidden, hidden

    return row_view(hidden, tokens - 1)
```

Requirements:

- The row loop must live inside the bulk orchestrator, not in the scheduler or
  generator, so generation still performs one prefill call per prompt.
- D2 full-attention fallback must be explicitly labelled in metadata/artifacts;
  do not call it fully native.
- For each full-attention layer, K/V append must happen row-by-row in ascending
  position order so decode spans see the correct context.
- Linear-attention conv/recurrent state must be updated once per linear layer
  for the full prompt, not once per token.
- Scratch sizing must be restored to decode-size after prefill.
- The legacy `_run_prefill_suffix_layers_serial(...)` should be retained only
  for diagnostics or deleted after tests move to the layer-major path.

Gate:

```bash
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_native_prefill_correctness.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --token-id 9707 --prompt-length 4 --max-layers 40 \
  --json benchmarks/results/2026-05-XX-hipengine-qwen35-layer-major-serial-fullattn-accepted.json
```

A performance artifact may be retained after correctness passes, but it must use
path label `layer_major_serial_full_attention` and must not be compared as full
parent parity.

### C. Native full-attention prefill (D1 throughput path)

Goal: replace the B-phase row loop for full-attention layers with true bulk
causal multi-query/GQA prefill.

Subtasks:

1. **Batched Q/K/V projection layout**
   - Replace the `tokens != 1` guard in `project_full_attention_qkv_fp16`.
   - For `T > 1`, do not assume the current combined dual Q/K projection layout
     is already split correctly. Either run separate Q and K projections, or add
     a layout/split helper that produces:
     - `query_proj: fp16[T, num_q_heads, 2 * head_dim]` (query + gate),
     - `key_raw_lowp: fp16/bf16[T, num_kv_heads, head_dim]`,
     - `value: fp16[T, num_kv_heads, head_dim]`.
   - Add a stage probe that compares these tensors against row-by-row c=1 for
     layer 3.

2. **Batched Q/K head RMSNorm + RoPE**
   - Existing `qwen35_head_rmsnorm_partial_rotary_position_f32_bf16` is scalar
     because it reads `position_ptr[0]` and launches only over heads.
   - Add either:
     - `qwen35_head_rmsnorm_partial_rotary_position_batch_f32_bf16(...)`, with
       grid `(tokens, heads)` and `positions[token]`, or
     - a labelled row-loop fallback used only until the batch kernel lands.
   - Fix the current scalar prepare path for `T > 1` to cast/copy
     `tokens * kv_width`, not one `kv_width`.

3. **Batched KV append**
   - Build append spans with `live_counts = positions` and `span_role="prefill"`.
   - Call `qwen35_write_paged_kv_mixed_value_fp16_batch_spans(...)` with
     `rows=T` and a block table repeated per row for c=1 single-request prefill.
   - For chunked prefill, `positions = start_position + arange(T_chunk)`.

4. **Causal prefill attention kernel**
   - Add/register a new kernel family under
     `hipengine/kernels/hip_gfx1100/attention/`, for example:
     `KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "qwen35_causal_gqa_fp16")`.
   - Inputs should be raw pointers only:
     - query `fp32[T, num_q_heads, head_dim]`,
     - gate `fp16[T, num_q_heads, head_dim]` or a separate gate-mul output stage,
     - paged key/value cache,
     - `KVLiveSpans` with per-row context lengths,
     - output `fp16[T, num_q_heads * head_dim]`.
   - For row `r`, attend only to keys with absolute position `<= positions[r]`.
     For single-request prefill this is the lower-triangular causal mask; for a
     later compact slab this becomes block-diagonal by request.
   - GQA mapping is `kv_head = q_head // (num_q_heads // num_kv_heads)`.
   - Use the same softmax scale and gate semantics as the existing decode path.

5. **Output projection and post-attention MoE**
   - Reuse `project_full_attention_o_fp16(tokens=T)` once attention output is
     `[T, q_width]`.
   - Reuse c1-style row MoE for correctness; grouped MoE is phase D.

Gates:

- CPU-reference oracle for causal attention with fixed tiny shapes.
- Layer-3 stage probe vs row-by-row c=1 for Q/K/V, RoPE, KV append, attention
  output, and final hidden.
- `rocprofv3 --kernel-trace` showing the new full-attention prefill kernel name
  with plausible duration and `Scratch_Size=0` or justified scratch.
- Full 40-layer fixture equality:

```bash
python3 scripts/qwen35_native_prefill_fullattn_stage_probe.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --prompt-length 4 --json benchmarks/results/2026-05-XX-hipengine-qwen35-fullattn-prefill-stage-accepted.json
```

Retained perf artifact:
`benchmarks/results/2026-05-XX-hipengine-qwen35-full-attn-prefill-native-accepted.json`.

### D. Grouped/compact MoE prefill

Goal: replace the correct c1-style selected row MoE path with parent-style
grouped/compact MoE over prompt rows.

Current correction: this is a performance/parity step, not the first correctness
blocker. The existing `run_moe_c1_fp16(tokens=T)` route is valid for phase B/C
correctness, but it streams expert weights per selected row rather than grouping
by expert.

Required ports/wiring:

- [ ] Confirm `qwen35_router_logits_*` / `qwen35_router_select` wrappers cover
  `[T, hidden]`; router top-k already has FP16-hidden coverage.
- [ ] Port the needed subset of `moe/group_scatter.hip`: count, prefix,
  scatter/scatter_gather, c1 group metadata variants, gather packed hidden,
  build lane-to-sorted, combine.
- [ ] Port the Qwen3.5/PARO-used subset of `quant/w8a16_moe.hip` shared/bulk
  variants. Do not port all 17 variants unless the parent call graph requires
  them.
- [ ] Port `moe/w8a8_grouped.hip` only if the W4 PARO parent path actually uses
  it for the retained prefill configuration.
- [ ] Port `wmma/wmma_i8_gemm.hip` for long-prompt grouped GEMM only after the
  pack8 grouped path is correctness-accepted.
- [ ] Add `run_moe_prefill_grouped_fp16(tokens=T)` and register:
  - `(hip_gfx1100, moe_prefill, w4_paro, qwen35_selected_c1_rows)` for the
    existing fallback,
  - `(hip_gfx1100, moe_prefill, w4_paro, qwen35_grouped_compact)` for the new
    path.
- [ ] Add a CPU-reference or row-by-row c1 oracle for every grouped kernel stage.

Gate:

- Grouped MoE output equals `run_moe_c1_fp16(tokens=T)` within the tolerance
  recorded in the parent grouped-vs-ungrouped artifact.
- `rocprofv3 --kernel-trace` proves grouped scatter/gather and grouped expert
  kernels ran.
- Retained artifact:
  `benchmarks/results/2026-05-XX-hipengine-qwen35-grouped-moe-prefill-accepted.json`.

### E. Wire bulk prefill into generation

Goal: the public generation path uses one prefill call per prompt instead of a
Python loop over prompt tokens.

Punchlist:

- [ ] In `hipengine/generation/qwen35_paro.py`, replace:

  ```python
  for position, token_id in enumerate(prompt_ids):
      next_result = session.step(token_id, position=position, sample=(position == len(prompt_ids) - 1))
  ```

  with:

  ```python
  try:
      next_result = session.prefill_native(prompt_ids, sample=True, require_full_native=False)
  except NotImplementedError:
      next_result = _serial_prefill_fallback(session, prompt_ids)
  ```

- [ ] Keep the old serial loop as an explicit fallback and diagnostic, not the
  default path.
- [ ] No `hipengine.LLM.generate()` surface change.
- [ ] Preserve greedy-only and EOS behavior in `Qwen35ParoOneTokenGenerator`.
- [ ] Emit path metadata in benchmark artifacts so partial-native B results are
  not confused with C/D full-native results.

Gate:

- Existing parent fixture token IDs and logits match the current resident E2E
  gate:
  `benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`.
- Add/retain a new artifact labelled with the prefill path used:
  `benchmarks/results/2026-05-XX-hipengine-qwen35-generate-prefill-native-accepted.json`.

### F. Compact c>N prompt batching

Goal: replace `scheduler_serial_slot_bridge` with a real compact prompt slab for
multiple concurrent requests.

Current correction: `KVLiveSpans` already has `request_ids` and `row_positions`.
The work is to build/populate packed tensors and make kernels honor row maps,
not to redesign the span dataclass.

Compact slab ABI:

```python
@dataclass(frozen=True)
class CompactPromptSlab:
    token_ids: Tensor        # int64[T_total]
    positions: Tensor        # int64[T_total], absolute positions per row
    cu_seqlens_q: Tensor     # int32[N + 1]
    cu_seqlens_k: Tensor     # int32[N + 1]
    row_to_request: Tensor   # int64[T_total]
    request_ids: Tensor      # int64[N]
    block_tables: Tensor     # int32[T_total, blocks_per_request] for current batch-writer ABI
    append_counts: Tensor    # int64[T_total], 0-based append positions
    context_counts: Tensor   # int64[T_total], 1-based visible lengths
```

Punchlist:

- [ ] Extend `ResidentBatchScheduler.next_prefill_work(chunk_size=...)` to emit
  a compact slab when more than one request has prefill work.
- [ ] Add `Qwen35ParoResidentSession.prefill_native_packed(slab)`.
- [ ] For first correctness, allow a **per-request bulk fallback** inside
  `prefill_native_packed` (one B/C single-request prefill per request) but label
  it `packed_metadata_per_request_bulk_fallback` and do not retain a c>N
  throughput claim from it.
- [ ] For true compact mode, make linear-attention state and conv caches
  segment-aware. Each request's tail state must be preserved independently.
- [ ] Native causal prefill attention must become var-len/block-diagonal:
  queries may attend only to rows with the same request id and positions not
  greater than the query position.
- [ ] Replace the benchmark path label `scheduler_serial_slot_bridge` with
  `scheduler_compact_prefill` only after equality gates pass.

Gate:

- Same generated tokens as serial c=8 path
  (`2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json`).
- Finite logits.
- At c=8/T=512, prefill tok/s improves over the serial bridge by at least 2x
  before retaining a throughput claim.
- Retained artifact:
  `benchmarks/results/2026-05-XX-hipengine-qwen35-c8-compact-prefill-accepted.json`.

### G. PrefillConfig and tuning

Goal: mirror parent tuning knobs through a typed config object, not environment
branches on the hot path.

Punchlist:

- [ ] Add `hipengine/runtime/prefill.py` with:

  ```python
  @dataclass(frozen=True)
  class PrefillConfig:
      linear_chunk_size: int = 0
      full_attn_query_chunk_size: int = 0
      full_attn_post_chunk_size: int = 0
      full_attn_rope_chunk_size: int = 0
      moe_grouped_device_gather: bool = True
      moe_stacked_compact: bool = True
      require_full_native: bool = False
  ```

- [ ] Thread `PrefillConfig` into `Qwen35ParoResidentSession` construction.
- [ ] Defaults must match the retained parent OPTIMAL flags for W7900 once C/D
  are landed.
- [ ] Add `scripts/qwen35_prefill_chunk_sweep.py` to search chunk defaults on
  W7900.
- [ ] Document retained defaults in this file or `docs/PLAN.md` Performance
  Knobs; do not reference a non-existent hipENGINE `docs/OPTIMAL.md` unless it
  is created in the same change.

Gate:

- 4096/128 and 4096/4096 retained rows on W7900 are within ~10% of the parent
  PARO rows, with exact command/hardware/correctness gate in artifacts.
- Update `benchmarks/README.md` and `benchmarks/CHANGELOG.md` for any retained
  perf claim.

### H. Graph capture and replay

Goal: optional dispatch-overhead cleanup after bulk prefill is within striking
distance of parent.

Corrected status:

- Low-level HIP graph wrappers already live in `hipengine/core/hip.py`.
- `Qwen35ParoResidentSession.capture_decode_graph(...)` exists for decode.
- There is no general prefill shape-bucket graph cache yet.

Punchlist:

- [ ] Add a small graph-cache layer keyed by `(mode, T bucket, layer path,
  full_attn path, moe path, active mask density)`.
- [ ] Capture prefill only for fixed buckets where all device pointers and
  scratch sizes are stable.
- [ ] Continue to fall back to uncaptured launches for rare shapes and for D2
  row-loop fallback.
- [ ] Re-check `docs/ROOFLINE.md` before chasing graph capture; do not add graph
  complexity if kernels are still far below the memory roof because of missing
  grouped MoE or native full-attention prefill.

This phase is performance-only; correctness gates are inherited from earlier
phases.

## Validation strategy

Every retained performance claim must include model, quant, workload shape,
hardware, exact command, result, and correctness gate.

Validation layers:

1. **Unit/CPU reference**
   - Existing or new CPU-reference kernels for math changes.
   - New/ported GPU kernels: KL ≤ 0.05 and top-1 agreement ≥ 90% vs
     `kernels/cpu_reference/` on fixture inputs.

2. **Stage probes**
   - Compare layer/stage tensors against row-by-row resident c=1 for the parent
     fixture.
   - Required probes for C: Q/K/V projection layout, RoPE, KV append, causal
     attention output, layer hidden output.

3. **End-to-end equality**
   - `scripts/qwen35_native_prefill_correctness.py` for prefill vs serial.
   - `scripts/qwen35_e2e_correctness.py` for generation fixture equality.
   - c>N compact: compare against independent serial c=1 runs and the current
     c=8 equality artifact.

4. **Kernel execution evidence**
   - `rocprofv3 --kernel-trace` entry for any new/ported kernel, with expected
     kernel name and plausible duration.

5. **Benchmark rollup**
   - Accepted perf rows update `benchmarks/README.md`,
     `benchmarks/CHANGELOG.md`, and a compact JSON artifact under
     `benchmarks/results/`.

Correctness is non-negotiable: a faster prefill that fails the parent fixture is
a regression, not a win.

## Risks and decisions now closed

- **D2 first, D1 second.** The corrected plan lands the labelled serial
  full-attention row-loop first because it unlocks the layer-major orchestrator
  and makes all later linear layers bulk. The native causal attention kernel is
  the next perf upgrade, not the first correctness step.
- **Existing RoPE position kernel is scalar.** Do not assume it covers `T` rows;
  add a batch-position variant or keep a labelled row loop until it lands.
- **Grouped MoE is a perf upgrade.** The c1-style row MoE path with `tokens=T`
  is acceptable as the oracle/fallback for B/C. Grouped/compact MoE is required
  for parent parity and must be labelled separately.
- **Packed metadata exists, wiring does not.** `KVLiveSpans` already has the
  fields F needs; the missing work is scheduler/runtime tensor construction and
  segment-aware kernels.
- **Avoid unlabelled fallbacks.** Any serial row loop or per-request compact
  fallback must be explicit in path metadata and benchmark artifacts.

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
