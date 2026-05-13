# HIPENGINE Kernel Catalog and Port Playbook

This doc is both the live kernel catalog and the mechanics for landing a kernel in HIPENGINE — porting from `~/amd-gpu-tuning/nano-vllm-amd/`, the JIT build layer, gotchas specific to this repo, and the correctness gate a port must pass.

**Kernel R&D does not live here.** Micro-tuning iteration loops (`rocprofv3 --kernel-trace` ranking, VGPR/occupancy hunting, `__launch_bounds__` sweeps, fusion experiments, the device-code gotcha catalog) belong in `~/amd-gpu-tuning/`. HIPENGINE receives *stable* kernels via the port pipeline below. If you find yourself opening a profiler inside the HIPENGINE tree, stop and move the experiment to the parent workspace.

See also:
- `docs/PLAN.md` "Kernel Port Strategy" — authoritative source inventory, split plan, per-family targets.
- `docs/TESTING.md` — RED/GREEN, CPU-reference fixtures, and math-correctness gates.
- `~/amd-gpu-tuning/AGENTS.md` — audit-first-via-rocprofv3, time-share/occupancy/iters-per-thread/VGPR discipline.
- `~/amd-gpu-tuning/LESSONS-LEARNED.md` — device-code gotchas and kernel lineage results.
- `~/amd-gpu-tuning/docs/OPTIMAL.md` — current optimal Qwen3.5/PARO native engine route and flags.
- `~/amd-gpu-tuning/PLAN-PAROQUANT.md` and `~/amd-gpu-tuning/docs/PARO.md` — Qwen3.5/PARO design history and evidence rows.

## Status legend

| Status | Meaning |
| --- | --- |
| **HIPENGINE landed** | Source lives in this repo, is registered or runnable through HIPENGINE, and has this repo's tests/smokes. |
| **CPU reference landed** | Torch-free NumPy oracle lives in `hipengine/kernels/cpu_reference/`; it is correctness infrastructure, not a HIP port. |
| **Lineage green** | Implemented/validated in `~/amd-gpu-tuning/nano-vllm-amd/`; source for HIPENGINE's copy+partition+retype port, but not yet landed here. |
| **Lineage dirty / experimental** | Observed in the parent checkout's uncommitted worktree or R&D notes. Do not make a default HIPENGINE path from it until it is promoted in `~/amd-gpu-tuning/`. |
| **Planned** | Architecture path is decided, but no HIPENGINE implementation yet. |

## HIPENGINE-landed kernels and oracles

This is the authoritative list of kernels/oracles that exist in this repo today. Empty backend family packages under `hipengine/kernels/hip_gfx1100/*/` are placeholders, not implemented kernels.

### CPU-reference primitive oracles (**CPU reference landed**)

Registered by `hipengine.kernels.cpu_reference.register_cpu_reference_kernels()` under `KernelKey("cpu_reference", <layer>, "fp16")`:

- `embed`
- `rmsnorm`
- `linear`
- `qkv_proj`
- `rotate`
- `attention_decode`
- `o_proj`
- `lm_head`

Fixture coverage currently includes `rmsnorm`, `linear`, `rotate`, and masked `attention_decode`; run with `python3 scripts/check_fixtures.py`.

### gfx1100 HIP kernels (**HIPENGINE landed**)

| Layer key | Quant key | Source | Public wrapper | Current gate |
| --- | --- | --- | --- | --- |
| `smoke_add` | `fp16` registry key, FP32 buffers | `hipengine/kernels/hip_gfx1100/smoke/smoke_add.hip` | `smoke_add_f32(...)` | `python3 scripts/smoke.py --mode smoke-add-hip --n 1024` → `max_abs=0.0` on W7900 |

`smoke_add` is a build/runtime smoke, not a model-layer primitive. It proves `hipengine.core.build`, lazy `libamdhip64.so`, device allocation/copy, launch, synchronize, and copyback without torch.

## Source-lineage drift check

Before porting a family, check whether the parent source moved since the last HIPENGINE catalog/audit baseline:

```bash
python3 scripts/check_lineage.py --diff stat
```

Useful filters:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
python3 scripts/check_lineage.py --file '*paroquant*' --diff patch
python3 scripts/check_lineage.py --fail-on-drift
```

The script is read-only. It uses `docs/source_lineage.json` to compare tracked files in `~/amd-gpu-tuning/nano-vllm-amd/` against the recorded baseline ref, then reports:

- current child-repo branch/HEAD,
- per-file dirty status,
- commits since baseline,
- diffstat or patch for the file,
- matching lines in `~/amd-gpu-tuning/WORKLOG.md` and relevant parent docs.

If a file reports **DRIFT**, inspect the listed commits/diff and read the evidence hits before copying code. Update `docs/source_lineage.json`'s baseline only after the catalog/port source is intentionally refreshed and logged in `WORKLOG.md`.

## Source-lineage kernel catalog to port

The stable source-lineage port set at the current HIPENGINE catalog baseline is the committed `nano-vllm-amd` Qwen3.5/PARO kernel set: **95** kernels from `csrc/amd/qwen35_expert.hip` plus **25** PARO kernels from `nanovllm/native/qwen35/paroquant_kernels.py` = **120 Qwen/PARO kernels**, plus the separate `smoke_add` build smoke. HIPENGINE ports these by family; bodies are preserved byte-for-byte except for includes and raw-pointer host-wrapper retyping.

### Atomic / primitive-oriented kernel families (**lineage green, not yet HIPENGINE-landed**)

- `wmma/wmma_i8_gemm.hip` (4):
  - `qwen35_wmma_i8_tile_kernel`
  - `qwen35_wmma_i8_gemm_kernel`
  - `qwen35_wmma_i8_gemm_a_row_major_kernel`
  - `qwen35_wmma_i8_gemm_grouped_a_row_major_kernel`
- `quant/w8a8_activation.hip` (2):
  - `qwen35_quantize_activation_i8_per_row_kernel`
  - `qwen35_quantize_activation_f32_i8_per_row_kernel`
- `moe/w8a8_grouped.hip` (10):
  - `qwen35_dequantize_w8a8_projection_kernel`
  - `qwen35_dequantize_w8a8_grouped_projection_kernel`
  - `qwen35_dequantize_w8a8_grouped_accumulate_kernel`
  - `qwen35_dequantize_w8a8_grouped_accumulate_deterministic_kernel`
  - `qwen35_dequantize_w8a8_c1_grouped_accumulate_kernel`
  - `qwen35_moe_grouped_accumulate_kernel`
  - `qwen35_moe_grouped_gate_up_kernel`
  - `qwen35_moe_grouped_down_kernel`
  - `qwen35_moe_grouped_down_flat_kernel`
  - `qwen35_moe_grouped_down_flat_accumulate_kernel`
- `moe/swiglu.hip` (2):
  - `qwen35_swiglu_packed_gate_up_kernel`
  - `qwen35_dequantize_swiglu_quantize_grouped_kernel`
- `quant/w8a16_moe.hip` (17):
  - `w8a16_selected_experts_kernel`
  - `w8a16_gate_up_kernel`
  - `w8a16_down_kernel`
  - `w8a16_gate_up_shared_kernel`
  - `w8a16_gate_up_shared_t_kernel`
  - `w8a16_gate_up_shared_t_decode_v2_kernel`
  - `w8a16_down_shared_kernel`
  - `w8a16_down_shared_bulk_combine_kernel`
  - `w8a16_down_shared_t_kernel`
  - `w8a16_down_shared_t_decode_v2_kernel`
  - `w8a16_down_shared_bulk_combine_t_kernel`
  - `w8a16_single_gate_up_kernel`
  - `w8a16_single_down_combine_kernel`
  - `w8a16_shared_gate_up_bulk_kernel`
  - `w8a16_shared_gate_up_bulk4_kernel`
  - `w8a16_shared_down_bulk_combine_kernel`
  - `w8a16_shared_down_bulk_combine_w8a8_c1_selected_kernel`
- `moe/group_scatter.hip` (11):
  - `qwen35_moe_group_count_kernel`
  - `qwen35_moe_group_prefix_kernel`
  - `qwen35_moe_group_scatter_kernel`
  - `qwen35_moe_group_scatter_gather_kernel`
  - `qwen35_moe_c1_group_metadata_kernel`
  - `qwen35_moe_c1_group_metadata_gather_kernel`
  - `qwen35_moe_c1_group_metadata_quantize_kernel`
  - `qwen35_moe_gather_packed_hidden_kernel`
  - `qwen35_moe_gather_quantize_packed_hidden_kernel`
  - `qwen35_build_lane_to_sorted_kernel`
  - `qwen35_moe_combine_kernel`
- `moe/router.hip` (6):
  - `qwen35_router_logits_kernel`
  - `qwen35_router_select_kernel`
  - `qwen35_token_rank_count_partial_kernel`
  - `qwen35_token_rank_count_finalize_kernel`
  - `qwen35_token_top2_partial_kernel`
  - `qwen35_token_top2_finalize_kernel`
- `quant/w8a16_linear.hip` (5):
  - `w8a16_linear_kernel`
  - `w8a16_linear_lowp_out_kernel`
  - `w8a16_linear_f32_kernel`
  - `w8a16_linear_batched_kernel`
  - `w8a16_linear_batched_f32_kernel`
- `linear_attn/conv.hip` (4):
  - `qwen35_linear_attn_conv_decode_kernel`
  - `qwen35_linear_attn_conv_decode_lowp_kernel`
  - `qwen35_linear_attn_conv_prefill_kernel`
  - `qwen35_linear_attn_conv_prefill_state_kernel`
- `linear_attn/gdn.hip` (6):
  - `qwen35_gdn_recurrent_decode_kernel`
  - `qwen35_gdn_rmsnorm_gate_kernel`
  - `qwen35_gdn_prefill_recurrent_kernel`
  - `qwen35_gdn_prefill_recurrent_k2_kernel`
  - `qwen35_gdn_recurrent_rmsnorm_gate_kernel`
  - `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel`
- `norm/rmsnorm.hip` Qwen primitive subset (4):
  - `qwen35_rmsnorm_kernel`
  - `qwen35_add_rmsnorm_kernel`
  - `qwen35_add_rmsnorm_f32_kernel`
  - `qwen35_head_rmsnorm_kernel`
- `rotary/rotary.hip` Qwen primitive subset (1):
  - `qwen35_partial_rotary_kernel`
- `attention/full_attn_decode.hip` (2):
  - `qwen35_full_attn_decode_kernel`
  - `qwen35_full_attn_decode_context_tensor_kernel`
- `attention/paged_attn_decode.hip` (13):
  - `qwen35_paged_full_attn_decode_kernel`
  - `qwen35_paged_full_attn_decode_context_tensor_kernel`
  - `qwen35_paged_full_attn_decode_8k_context_tensor_kernel`
  - `qwen35_paged_full_attn_decode_4k_kernel`
  - `qwen35_paged_full_attn_decode_8k_dyn_kernel`
  - `qwen35_paged_full_attn_decode_split_k_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_warp_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel`
  - `qwen35_paged_full_attn_decode_split_k_int8_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_int8_kernel`
  - `qwen35_paged_full_attn_decode_split_k_reduce_kernel`
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel`
- `attention/paged_kv_write.hip` (6):
  - `qwen35_write_paged_kv_kernel`
  - `qwen35_write_paged_kv_position_tensor_kernel`
  - `qwen35_write_paged_kv_mixed_value_kernel`
  - `qwen35_write_paged_kv_mixed_value_position_tensor_kernel`
  - `qwen35_write_paged_kv_int8_kernel`
  - `qwen35_write_paged_kv_position_tensor_int8_kernel`
- `quant/paro_awq_gemv.hip` stable PARO GEMV/projection subset (7; projection-pair fused variants are called out again below):
  - `gemv_awq_v8_kernel`
  - `gemv_awq_pack8_kernel`
  - `gemv_awq_dual_pack8_kernel`
  - `gemv_awq_selected_dual_pack8_strided_kernel`
  - `gemv_awq_selected_pack8_kernel`
  - `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`
  - `dense_gemv_out_kernel`
- `quant/paro_awq_dequant.hip` (2):
  - `dequant_awq_pack8_kernel`
  - `dequant_awq_pack8_dual_kernel`
- `norm/rmsnorm.hip` PARO subset (2):
  - `paro_rmsnorm_out_kernel`
  - `paro_add_rmsnorm_out_kernel`
- `rotary/rotary.hip` PARO subset (2):
  - `paro_rotate2_kernel`
  - `paro_rotate3_kernel`

### Fused / composite kernel families (**lineage green, not yet HIPENGINE-landed**)

Each fused kernel still requires an unfused fallback chain registered under its primitive components.

- Norm + rotary:
  - `qwen35_head_rmsnorm_partial_rotary_kernel`: `head_rmsnorm -> partial_rotary`.
  - `qwen35_head_rmsnorm_partial_rotary_position_kernel`: `head_rmsnorm -> position-indexed partial_rotary`.
- PARO selected-expert activation / rotation:
  - `silu_mul_dual_out_kernel`: `silu(gate) * up` for dual selected-expert outputs.
  - `silu_mul_dual_rotate_out_kernel`: `silu(gate) * up -> PARO down-rotation`.
  - `silu_mul_pair_rotate_out_kernel`: paired `silu(gate) * up -> rotate` variant.
- Weighted routing reductions:
  - `weighted_index_add_out_kernel`: routed weighted add into output rows.
  - `weighted_index_add_atomic_float_out_kernel`: atomic-float routed weighted add variant.
  - `weighted_lanes_inverse_kernel`: lane/weight inverse helper.
  - `weighted_lanes_sum_out_kernel`: lane-group weighted sum.
  - `weighted_sum_out_kernel`: selected-expert weighted sum.
- Shared-expert + selected-expert combine:
  - `shared_gate_combine_out_kernel`: `selected_moe + sigmoid(shared_gate) * shared_expert`.
  - `shared_gate_combine_residual_out_kernel`: above plus residual add.
  - `weighted_sum_shared_gate_combine_residual_out_kernel`: selected weighted sum + shared gate combine + residual add in one c=1 decode kernel.
- Full-attention gate fusion:
  - `full_attn_gate_mul_out_kernel`: `sigmoid(attn_gate) * attention_out` plus output conversion.
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel`: paged split-K reduce fused with PARO full-attention gate for device-context decode.
- Projection-pair fusion routes used by the PARO path:
  - `gemv_awq_dual_pack8_kernel`: dual W4 pack8 GEMV for two projections over the same input.
  - `gemv_awq_selected_dual_pack8_strided_kernel`: selected-expert dual W4 pack8 GEMV over compact/repacked expert weights.
  - `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`: selected-expert dual W4 pack8 GEMV plus output rotation.

### Source catalog drift requiring refresh before PARO/WMMA ports

The last manual HIPENGINE catalog audit (`docs/source_lineage.json` baseline `22405a9`) counted the committed PARO embedded-HIP set at 25 kernels and observed six additional parent-worktree kernels beyond that committed set:

- `gemv_awq_mbatch_dual_pack8_kernel`
- `gemv_awq_mbatch_pack8_kernel`
- `gemv_awq_expert_seq_dual_pack8_kernel`
- `gemv_awq_expert_seq_pack8_kernel`
- `gemm_awq_selected_dual_pack8_wmma_kernel`
- `gemm_awq_selected_pack8_wmma_kernel`

`~/amd-gpu-tuning/docs/OPTIMAL.md` now promotes a compact-WMMA route, and `scripts/check_lineage.py` reports drift in `qwen35_expert.hip`, `extension.cpp`, `paroquant_kernels.py`, `paroquant.py`, and `expert.py` after `22405a9`. Therefore, treat the 120-kernel catalog above as the **baseline catalog**, not the final PARO/WMMA port inventory. Before porting PARO/WMMA, refresh the exact kernel list from current parent source, read the listed WORKLOG/OPTIMAL evidence, and update this catalog in the same commit as the port-source refresh.

## Qwen3.5 MoE / PARO path map

This section maps the current source-lineage inference path that HIPENGINE should preserve when porting `z-lab/Qwen3.5-35B-A3B-PARO` (`w4_paro`, W4A16) from `nano-vllm-amd`. It is **not** an HIPENGINE performance claim yet; it is the target graph/kernel route to reproduce after the port.

### Current optimal route

Canonical source: `~/amd-gpu-tuning/docs/OPTIMAL.md` (2026-05-13 snapshot). Supporting design/history remains in `~/amd-gpu-tuning/PLAN-PAROQUANT.md` and `~/amd-gpu-tuning/docs/PARO.md`.

The current optimal parent route is compact-WMMA prefill plus one-step graph-replay decode, with all listed parent quality gates passing. Latest retained parent sweep:

| Shape | PARO prefill tok/s | PARO decode tok/s | Peak VRAM | Validation |
| --- | ---: | ---: | ---: | --- |
| 512/128 | 2557 | 115.7 | 18.86 GiB | graph/step true |
| 1K/128 | 2876 | 112.9 | 19.34 GiB | graph/step true |
| 4K/128 | 2703 | 112.0 | 21.64 GiB | graph/step true |
| 32K/128 | 1880 | 98.8 | 21.37 GiB | graph/step true |
| 128K/128 | 914 | 62.6 | 27.42 GiB | graph/step true |

Post-sweep parent spot checks retained these defaults:

- Native weighted-lane grouped-stacked accumulation: `2642.1` vs `2561.5` prefill tok/s at 512/128, graph validation true.
- Grouped SiLU + down-rotation fusion: `2632.2` vs `2631.4` prefill tok/s at 512/128, graph validation true.
- WMMA extension load is not the Vulkan decode-gap source: graph decode was `115.56` vs `115.04` tok/s with WMMA disabled.

Correctness hierarchy for these rows: HF PARO oracle for model correctness; scalar eager pure-native as the native debug reference; tensorized eager as serving/graph ABI reference; graph replay must match tensorized eager. Long scalar-vs-tensorized greedy equality is a diagnostic, not the only promotion gate; use KL/NLL/top-k/top-1 and repetition/coherence/long-context quality gates.

### Base flags to preserve

`OPTIMAL.md` lists 23 base environment flags. HIPENGINE should preserve the same routing decisions as registry/plugin configuration rather than copying env-var checks into engine code:

- **MoE dispatch:** compact stacked layout, in-place selected-MoE repack replacement, GPU expert gather, grouped-stacked max tokens `4096`, native weighted lanes, grouped-stacked SiLU+rotate fusion, decode selected-MoE SiLU/down-rotate fusion, native router.
- **GEMV / WMMA:** PARO vec8 GEMV, pack8 qweight replacement, transposed pack8 disabled on W7900, WMMA GEMM enabled for prefill MoE, compact WMMA buffers, `WMMA_MIN_TOKENS=64` (crossover vs GEMV is ~48 tokens).
- **Attention:** full-attention gate fusion, full-attention Q/K pack8 fusion, grouped-GQA paged context attention, paged max splits `512`.
- **Linear/projections:** W8A16 `lm_head`, W8A16 shared expert dense branch, fused linear-attention A/B projection, pack8 fused linear-attention QKV+Z projection.
- **Routing threshold:** native router prefill path begins at `512` tokens.

### Prefill route

- Benchmark protocol: `OPTIMAL.md` uses the parent `scripts/run_moe2_baselines.py` sweep and graph-replay bench command; short/mid-context quick start targets `--prompt-len 4096 --decode-len 128 --decode-use-step-graph-replay`.
- Router/MoE:
  - Real router runs per MoE layer; no HF model execution in the pure-native path.
  - Compact WMMA prefill MoE is the current optimal grouped-stacked route for `>=64` tokens; GEMV-only only wins at ~32 tokens.
  - Grouped-stacked max tokens is now `4096`, not the older 1024 short-prefill cap.
  - Weighted-lane accumulation and grouped-stacked SiLU+down-rotation fusion are default-on parent optimizations.
- Long prefill:
  - For `>=32K`, add chunking overrides from `OPTIMAL.md`: linear chunk `1024`, MoE chunk `1024`, full-attention post/RoPE chunk `1024`, and full-attention query chunk `4096`.
  - Do **not** set long-prefill chunking overrides for `<=4K`; they change the MoE prefill path and reduce throughput.
- Projection/quant:
  - Non-expert W4 pack8 replacement uses `[out/8, in]` pack8 qweights and frees original eligible AWQ qweights.
  - `lm_head` uses the W8A16 replacement path in the optimal route.

### Decode route

The target c=1 decode path is static-buffer, graph-replay-friendly, and mostly device-resident:

1. **RMSNorm / residual:** PARO-native `rmsnorm` and `add+rmsnorm` kernels; avoid per-token framework glue.
2. **Router:** native combined router/shared-gate logits with hot BF16 cache and FP16/BF16 hidden input; reuse decode-only output buffers.
3. **Selected MoE:** compact stacked selected-expert layout plus repacked replacement qweights; selected gate/up via dual W4 pack8 GEMV; selected down uses small-K specialization where applicable.
4. **Selected activation/down rotation:** `silu(gate) * up` and PARO down-rotation fused on the stacked decode path.
5. **Shared expert:** dense shared expert c=1 branch uses W8A16 gate/up/down where enabled; prefill keeps the dense path.
6. **MoE combine:** selected-expert weighted sum, shared-expert sigmoid/gate combine, and residual add fuse into `weighted_sum_shared_gate_combine_residual_out_kernel` on c=1 decode.
7. **Linear attention:** native conv/GDN recurrence; lowp FP16/BF16 inputs feed kernels while recurrent state/math stay FP32. A/B projections are concatenated for c=1; QKV/Z uses dual pack8 W4 GEMV after rotation.
8. **Full attention projections:** q/k use dual pack8 W4 GEMV after batched input rotation; v stays on the existing pack8 path.
9. **KV append:** BF16 full-attention KV cache with native mixed-input paged-KV writer; no tiny per-token framework appends.
10. **Full attention decode:** contiguous path for short contexts; paged/split-K path defaults at context `>= 1024`, with warp-cooperative context tensor QK, physical-offset address hoist, grouped-GQA reuse, split cap 512 for 128K-class rows, and gated split-K reduce where applicable.
11. **Final head:** W8A16 `lm_head` replacement path.
12. **Graph replay:** one reusable decode-step graph replay is the promoted graph shape; keep `--decode-step-graph-capture-steps=1`. Multi-step capture was tested and not promoted.

Parent decode profiling note from `OPTIMAL.md`: fused `lm_head + argmax` is not a current lever; the next decode target is the AWQ/GEMV decode family, about 40% of selected-region kernel time in the 512/128 graph profile.

### Alternative paths and caveats

- **W8A8 comparison path:** stays quality-safe and useful as a comparator; do not regress it while porting PARO.
- **40GB+ diagnostic PARO path:** stacked selected-expert diagnostics proved speed hypotheses but are not promotion candidates because 24GB W4 usability is a hard gate.
- **24GB non-stacked baseline:** green but slow; retained as a deployable-memory fallback, not the speed target.
- **Long-context decode:** contiguous full-attention decode cannot launch at 32K because dynamic LDS scales with context; long decode must use paged/split-K over the dense cache viewed as pages.
- **Tensorized paged-attention drift:** current parent docs localize long-tail scalar-vs-tensorized drift to paged context-tensor full attention. Graph replay matching tensorized eager is necessary; scalar-eager greedy equality alone is not sufficient promotion evidence.
- **Rejected standalone kernel ideas:** PARO v8 unroll-threshold 600, isolated wave32/no-LDS W4 GEMV, naive AWQ W4xQ8 dp4a, caller-owned paged workspace, and non-split-K 4K attention were tested but not promoted. Do not import them into HIPENGINE defaults without a fresh audit and correctness/perf evidence.

## Port = copy + partition + retype

The initial port is mechanical, not creative. Kernel bodies are preserved byte-for-byte (modulo `#include` headers). The three things that change during port:

1. **File split by family.** The monolithic `nano-vllm-amd/csrc/amd/qwen35_expert.hip` (13,769 lines, 95 `__global__`s) and the 3,766-line embedded HIP string in `nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` partition into `kernels/<backend>/<family>/*.hip` per the table in `docs/PLAN.md`. The near-duplicate `qwen35_expert_hip.hip` is dropped.
2. **Launch wrappers retyped.** Host-side wrappers go from `torch::Tensor` to raw pointer + shape/stride/dtype signatures. Scripted, ~1 day.
3. **Embedded HIP extracted.** `paroquant_kernels.py`'s `r'''...'''` block becomes real `.hip` files compiled through `hipengine.core.build` instead of `torch.utils.cpp_extension.load_inline`.

Preserve all `__launch_bounds__`, template specializations, and compiler flags (`-mcumode` for decode, `-amdgpu-unroll-threshold-local=600` for both profiles). A port that rewrites kernel bodies is not a port.

## Port correctness gate (non-negotiable)

A kernel split / port may only land when all three of these pass on the stated fixture set:

1. **Registry resolution.** Every kernel name still resolves via the 4-axis registry (`resolve(KernelKey(...))` returns a callable for every key previously exported by the monolithic `.so`).
2. **Profiler parity.** `rocprofv3 --kernel-trace` on the target decode smoke (Qwen3.6-35B-A3B unless noted) reports the same kernel set with matching `DurationNs` distribution as the monolithic build. A new kernel name, a missing kernel name, or a >10% duration shift is a split bug.
3. **Numerical parity.** KL ≤ 0.05 AND top-1 agreement ≥ 90% vs the monolithic build on the correctness fixtures. (For a *net-new* kernel, the oracle is `kernels/cpu_reference/`, not the monolithic build.)

Never land a split that regresses any of these.

## Build layer (`hipengine.core.build`)

HIPENGINE uses its own build layer, not `torch.utils.cpp_extension`. It calls `hipcc` (or `nvcc` for CUDA backends) via `subprocess.run`, links with `ctypes.CDLL`, and caches `.so` files by a hash of `(source, flags, hipcc version)` under `~/.cache/hipengine/build/`. Edit → bench loop stays at ~5–10 s per kernel change.

### Three build profiles (from `nano-vllm-amd/nanovllm/native/amd/extension.py`)

| Profile | Flags | Wavefront | Used for |
| --- | --- | --- | --- |
| `decode` | `-mcumode`, `-amdgpu-unroll-threshold-local=600` | 64 | Decode-phase kernels (paged attention, W8A8 grouped MoE decode, paro GEMV) |
| `prefill` | `-amdgpu-unroll-threshold-local=600` (WGP mode) | 32 | Prefill-phase kernels (GEMM, W8A16 linear prefill) |
| `baseline` | (none) | 32 | Debug / fallback |

Write device code for the target profile's wavefront width. Use `warpSize` (built-in), not a hard-coded 32 or 64.

### JIT cache gotcha

Symptom: kernel calls hang with GPU at 0% utilization and no error. This is almost always a stale cached `.so` that doesn't match the current source. Nuke the matching cache dir before re-importing:

```bash
rm -rf ~/.cache/hipengine/build/<family>-<hash>*
```

If the family is unknown, clearing the whole cache is cheap (~5 s per kernel to rebuild):

```bash
rm -rf ~/.cache/hipengine/build/
```

The hash incorporates the source file content, the flag set, and the `hipcc --version` string. If you change `hipcc` underneath an existing cache, the hash will change and old entries will be ignored — not overwritten. Prune manually when the cache grows unbounded.

## rocprofv3 smoke (port parity + new kernel check)

Minimum smoke a port or a new kernel must produce:

```bash
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke -- \
  uv run python scripts/smoke.py <model> <workload>
```

Expected output: a CSV with `KernelName`, `Grid_Size`, `Workgroup_Size`, `DurationNs`, `VGPR_Count`, `Scratch_Size`, `LDS_Block_Size`. Check:

- The expected kernel name appears.
- `DurationNs` is plausible (same order of magnitude as the reference).
- `Scratch_Size > 0` on a hot-path kernel is a red flag — escalate to `~/amd-gpu-tuning/` for audit.
- `VGPR_Count ≥ 96` may be squeezing occupancy — same.

rocprofv3 dumps are **not committed**. Store under `/tmp/` or outside the repo.

## Registering a kernel

Kernels self-register on module import:

```python
# hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.py
from hipengine.kernels.registry import KernelKey, register
from hipengine.core.build import build_hip

_so = build_hip(
    sources=["paged_attn_decode.hip"],
    profile="decode",
    family="attention",
)

def paged_attn_decode_fp16(...): ...

register(
    KernelKey(backend="hip_gfx1100", layer="paged_attn_decode",
              quant="fp16", variant="split_k_warp"),
    paged_attn_decode_fp16,
)
```

The resolver does narrowest-to-broadest match: `variant` → no-variant → `quant="fp16"` fallback → `backend="cpu_reference"`. A new backend implementation or a new fused composite is a `register(...)` call, never an `if backend == "..."` branch in dispatch code.

## Per-family port checklist

When bringing up a family (`attention/`, `moe/`, `quant/`, …), follow in order:

1. Copy the relevant kernels from the monolithic source into `kernels/hip_gfx1100/<family>/*.hip`. Preserve bodies byte-for-byte.
2. Retype the host-side launch wrappers.
3. Move the `PYBIND11_MODULE` / `TORCH_LIBRARY` entries for this family from `csrc/amd/extension.cpp` into `kernels/hip_gfx1100/common/extension.cpp` (the aggregator).
4. Write `register(KernelKey(...), ...)` calls in the Python wrapper module so the kernels resolve.
5. Add a CPU-reference implementation for every new `layer` key in `kernels/cpu_reference/`.
6. Run the port correctness gate (all three checks above).
7. Commit the family as one logical unit with `port:` prefix and `nano-vllm-amd@<sha>` in the body.

Do not interleave families in one commit. A commit that touches `attention/` and `moe/` together is harder to bisect and harder to review.
