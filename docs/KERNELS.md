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
| `rmsnorm` | `bf16` | `hipengine/kernels/hip_gfx1100/norm/rmsnorm.hip` | `qwen35_rmsnorm_bf16(...)` | `python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16` → `max_abs=0.0`, `bit_mismatch=0`; `rocprofv3` shows `qwen35_rmsnorm_kernel`, computed `DurationNs=6560`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0` on W7900 |
| `add_rmsnorm`, `add_rmsnorm_f32`, `head_rmsnorm` | `bf16` | same | `qwen35_add_rmsnorm_bf16(...)`, `qwen35_add_rmsnorm_f32_bf16(...)`, `qwen35_head_rmsnorm_f32_bf16(...)` | Build/registration tests landed; launch wrappers are source-family peers of `qwen35_rmsnorm_kernel` and share the same `.so` |
| `rmsnorm`, `add_rmsnorm` variant `paro_out` | `bf16`, `w4_paro` | same | `paro_rmsnorm_out_bf16(...)`, `paro_add_rmsnorm_out_bf16(...)` | `python3 scripts/smoke.py --mode paro-rmsnorm-hip --rows 2 --hidden-size 16` → bit-exact norm/add/residual; `rocprofv3` shows `paro_rmsnorm_out_kernel` (`DurationNs=5760`, `VGPR_Count=40`, `Scratch_Size=0`, `LDS_Block_Size=1024`) and `paro_add_rmsnorm_out_kernel` (`DurationNs=5040`, `VGPR_Count=56`, `Scratch_Size=0`, `LDS_Block_Size=1024`) on W7900 |
| `router_logits`, `router_select`, `router_topk_shared` variant `out` | `bf16`, `fp32` select, `w4_paro` shared route | `hipengine/kernels/hip_gfx1100/moe/router.hip` | `qwen35_router_logits_bf16(...)`, `qwen35_router_select(...)`, `qwen35_router_topk_shared_out_bf16(...)` | `python3 scripts/smoke.py --mode qwen35-router-hip --rows 2 --hidden-size 16` → `logits_max_abs=0.0`, `routing_max_abs=1.49e-08`, `selected_match=True`; `rocprofv3` shows `qwen35_router_logits_kernel` (`DurationNs=3520`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`) and `qwen35_router_select_kernel` (`DurationNs=5920`, `VGPR_Count=40`, `Scratch_Size=0`, `LDS_Block_Size=512`) on W7900 |
| `selected_dual_pack8_gemv`, `selected_pack8_gemv` variants `strided`, `transposed` | `w4_paro` with BF16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_selected_dual_pack8_*_bf16(...)`, `gemv_awq_selected_pack8_*_bf16(...)` | `python3 scripts/smoke.py --mode paro-selected-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact dual/single, strided/transposed (`dual_mismatch=0/0`, `single_mismatch=0/0`); `rocprofv3` shows both selected GEMV kernels with `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64` on W7900 |
| `rotate+selected_dual_pack8_gemv` variant `strided` | `w4_paro` with BF16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_selected_dual_pack8_strided_rotate_out_bf16(...)` | `python3 scripts/smoke.py --mode paro-selected-gemv-rotate-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact (`mismatch=0`, `max_abs=0.0`); `rocprofv3` shows `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel` with `DurationNs=7361`, `VGPR_Count=96`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64` on W7900 |
| `pack8_gemv`, `dual_pack8_gemv` variants `strided`, `transposed` | `w4_paro` with BF16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_pack8_*_bf16(...)`, `gemv_awq_dual_pack8_*_bf16(...)` | `python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact generic single/dual (`single_mismatch=0/0`, `dual_mismatch=0/0`); `rocprofv3` shows generic single/dual pack8 kernels with `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64` on W7900 |
| `silu_mul_dual`, `silu_mul_dual_rotate`, `silu_mul_pair_rotate` variant `out` | `bf16`, `w4_paro` | `hipengine/kernels/hip_gfx1100/fused/paro_silu.hip` | `silu_mul_dual_out_bf16(...)`, `silu_mul_dual_rotate_out_bf16(...)`, `silu_mul_pair_rotate_out_bf16(...)` | `python3 scripts/smoke.py --mode paro-silu-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact dual SiLU and dual/pair rotate (`*_mismatch=0`); `rocprofv3` shows all three kernels with `Scratch_Size=0` on W7900 |
| `weighted_sum`, `weighted_sum+shared_gate+residual`, `shared_gate_combine`, `shared_gate_combine+residual` variant `out` | `bf16`, `w4_paro` with FP32 weights/gate logits | `hipengine/kernels/hip_gfx1100/fused/paro_combine.hip` | `weighted_sum_out_bf16_f32w(...)`, `weighted_sum_shared_gate_combine_residual_out_bf16_f32w(...)`, `shared_gate_combine*_bf16(...)` | `python3 scripts/smoke.py --mode paro-combine-hip --rows 4 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact weighted/shared/residual combine; `rocprofv3` shows all four kernels with `Scratch_Size=0` on W7900 |
| `dense_gemv` variant `out` | `bf16`, `w4_paro` | `hipengine/kernels/hip_gfx1100/linear/dense_gemv.hip` | `dense_gemv_out_bf16(...)` | `python3 scripts/smoke.py --mode dense-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact (`mismatch=0`, `max_abs=0.0`); `rocprofv3` shows `dense_gemv_out_kernel` with `Scratch_Size=0` on W7900 |
| `w8a16_linear` variants `bf16_f32_out`, `bf16_lowp_out`, `f32_f32_out` | `w8a16`, `w4_paro` | `hipengine/kernels/hip_gfx1100/quant/w8a16_linear.hip` | `w8a16_linear_bf16_f32_out(...)`, `w8a16_linear_bf16_lowp_out(...)`, `w8a16_linear_f32_f32_out(...)` | `python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → `bf16_f32_max_abs=0.0`, `f32_f32_max_abs=4.77e-07`, `lowp_mismatch=0`; `rocprofv3` shows all three W8A16 linear kernels with `Scratch_Size=0` on W7900 |
| `paro_rotate2`, `paro_rotate3` variant `bf16` | `w4_paro` | `hipengine/kernels/hip_gfx1100/rotary/paro_rotate.hip` | `paro_rotate2_bf16(...)`, `paro_rotate3_bf16(...)` | `python3 scripts/smoke.py --mode paro-rotate-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact (`mismatches=[0, 0, 0, 0, 0]`, `max_abs=0.0`); `rocprofv3` shows `paro_rotate2_kernel` (`DurationNs=3400`, `VGPR_Count=24`, `Scratch_Size=0`) and `paro_rotate3_kernel` (`DurationNs=3160`, `VGPR_Count=24`, `Scratch_Size=0`) on W7900 |
| `partial_rotary`, `head_rmsnorm+partial_rotary` variants `qwen35_f32`, `qwen35_f32_bf16`, `qwen35_position_f32_bf16` | `w4_paro` full-attention prelude | `hipengine/kernels/hip_gfx1100/rotary/qwen35_rotary.hip` | `qwen35_partial_rotary_f32(...)`, `qwen35_head_rmsnorm_partial_rotary*_f32_bf16(...)` | `python3 scripts/smoke.py --mode qwen35-rotary-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → `partial_max_abs=0`, `head_max_abs=2.38e-07`, `position_max_abs=2.38e-07`; `rocprofv3` shows all three parent kernels with `Scratch_Size=0` on W7900 |
| `linear_attn_conv_decode` variants `f32`, `bf16` | `w4_paro` linear-attention decode | `hipengine/kernels/hip_gfx1100/linear_attn/conv.hip` | `qwen35_linear_attn_conv_decode_f32(...)`, `qwen35_linear_attn_conv_decode_bf16(...)` | `python3 scripts/smoke.py --mode qwen35-linear-attn-conv-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → `f32_out_max_abs=7.45e-09`, `f32_state_max_abs=0`, `bf16_out_max_abs=7.45e-09`, `bf16_state_max_abs=0`; `rocprofv3` shows both parent conv decode kernels with `Scratch_Size=0` on W7900 |
| `gdn_recurrent_rmsnorm_gate` variant `bf16_lowp` | `w4_paro` linear-attention decode | `hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip` | `qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(...)` | `python3 scripts/smoke.py --mode qwen35-linear-attn-gdn-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → `out_max_abs=2.98e-08`, `state_max_abs=1.49e-08`; `rocprofv3` shows `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<uint16_t>` with `DurationNs=12360`, `VGPR_Count=56`, `Scratch_Size=0`, `Workgroup_Size_X=128` on W7900 |

`smoke_add` is a build/runtime smoke, not a model-layer primitive. It proves `hipengine.core.build`, lazy `libamdhip64.so`, device allocation/copy, launch, synchronize, and copyback without torch.

`qwen35_rmsnorm` is the first real model-layer HIP family port. It is BF16-bit (`uint16_t`) at the raw pointer ABI; Qwen weights store deltas and the kernel applies `1.0 + weight_delta`. PARO `paro_out` RMSNorm variants use direct norm weights and caller-owned output buffers, matching the parent native PARO serving path.

`paro_awq_gemv` ports the selected-expert and generic pack8 GEMV bodies used by the current OPTIMAL MoE c=1 route and non-MoE projections. The fused rotate→selected-dual GEMV path is landed for the parent strided layout; the wrappers also cover strided and transposed qweight layouts for BF16 activation/scale buffers.

`paro_silu` ports the selected-expert activation and down-rotation stage, including the fused `silu_mul_dual_rotate_out_kernel` path used by the parent default and the unfused/separate-gate fallback kernels.

`paro_combine` ports the c=1 selected-weighted/shared-gate/residual combine kernels. The current HIPENGINE wrappers cover the parent default FP32 router-weight/gate-logit path; scalar-weight variants can be added if a future route needs them.

`dense_gemv` ports the parent PARO BF16 dense GEMV used by auxiliary dense paths such as linear-attention AB projections when they remain dense rather than W4/W8 quantized.

`paro_rotate` ports the parent PARO pairwise rotation helpers used by multi-projection PARO paths (`paro_rotate2`, `paro_rotate3`). `qwen35_rotary` ports the parent full-attention prelude (`partial_rotary`, fused head RMSNorm + partial rotary, and table-positioned fused head RMSNorm + partial rotary).

`w8a16_linear` ports the parent W8A16 GEMV kernels used by the current shared-expert default (`hip_w8a16_linear_lowp_out`) and W8A16 lm-head/auxiliary dense route. `scripts/smoke.py --mode w8a16-shared-expert-hip` chains W8A16 gate/up → `silu_mul_dual_out` → W8A16 down and is bit-exact against the staged BF16 NumPy oracle. `scripts/smoke.py --mode paro-moe-c1-hip --hidden-size 8` is the synthetic c=1 decode vertical smoke: PARO RMSNorm → router/shared-gate → selected W4 gate/up → SiLU → selected W4 down → W8A16 shared branch → weighted/shared/residual combine.

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

### Atomic / primitive-oriented kernel families (**source-lineage status; HIPENGINE-landed where noted**)

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
- `moe/router.hip` top-k subset (2) — **HIPENGINE landed for BF16 hidden/weight raw-pointer wrappers**:
  - `qwen35_router_logits_kernel`
  - `qwen35_router_select_kernel`
- `moe/router.hip` token-rank/top2 subset (4):
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
- `norm/rmsnorm.hip` Qwen primitive subset (4) — **HIPENGINE landed for BF16 raw-pointer wrappers**:
  - `qwen35_rmsnorm_kernel`
  - `qwen35_add_rmsnorm_kernel`
  - `qwen35_add_rmsnorm_f32_kernel`
  - `qwen35_head_rmsnorm_kernel`
- `norm/rmsnorm.hip` PARO subset (2) — **HIPENGINE landed for BF16 raw-pointer wrappers**:
  - `paro_rmsnorm_out_kernel`
  - `paro_add_rmsnorm_out_kernel`
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

`~/amd-gpu-tuning/docs/OPTIMAL.md` now promotes a compact-WMMA route, and `scripts/check_lineage.py` reports drift in `qwen35_expert.hip`, `extension.cpp`, `paroquant_kernels.py`, `paroquant.py`, and `expert.py` after `22405a9`. Therefore, treat the 120-kernel catalog above as the **baseline catalog**, not the final PARO/WMMA port inventory.

Current OPTIMAL source refresh at `nano-vllm-amd@59195ed` adds **5 kernels** over the baseline catalog: `qwen35_moe_wmma_tile_map_kernel` in `qwen35_expert.hip`, plus `gemm_awq_selected_dual_pack8_wmma_kernel`, `gemm_awq_selected_pack8_wmma_kernel`, `gemm_awq_selected_dual_pack8_wmma_compact_kernel`, and `gemm_awq_selected_pack8_wmma_compact_kernel` in `paroquant_kernels.py`. The current full Qwen/PARO HIP inventory is **96** monolithic kernels + **29** PARO/WMMA kernels = **125** kernels, excluding `smoke_add`. Before porting PARO/WMMA, read the listed WORKLOG/OPTIMAL evidence and keep this checklist synchronized with the source commit used.

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

### Current OPTIMAL MoE port checklist (`nano-vllm-amd@59195ed`)

The checklist below is the active port map for reproducing the parent compact-WMMA + graph-replay route. Status values are HIPENGINE status, not parent status.

#### Source refresh deltas since baseline `22405a9`

| Source | Current status | Required action |
| --- | --- | --- |
| `csrc/amd/qwen35_expert.hip` | DRIFT; 96 kernels | Include new `qwen35_moe_wmma_tile_map_kernel` with grouped MoE / compact WMMA port. |
| `csrc/amd/extension.cpp` | DRIFT; + bindings for tile-map path | Retype affected launch wrapper(s), do not copy PyTorch/TORCH_LIBRARY plumbing. |
| `nanovllm/native/qwen35/paroquant_kernels.py` | DRIFT; 29 kernels, 35 `m.def` exports | Extract current V8 + WMMA embedded HIP, including four WMMA kernels and compact wrappers. |
| `nanovllm/native/qwen35/paroquant.py` | DRIFT; dispatch logic changed | Adapt routing decisions into model/quant/kernel-plan plugins, not env-var branches in engine code. |
| `nanovllm/native/qwen35/expert.py` | DRIFT; added `hip_qwen35_moe_wmma_tile_map` | Port tile-map raw-pointer wrapper with grouped MoE metadata family. |

#### MoE decode c=1 path

| Stage | Parent kernels / wrappers | HIPENGINE status | Notes / gate |
| --- | --- | --- | --- |
| RMSNorm / residual | `paro_rmsnorm_out_kernel`, `paro_add_rmsnorm_out_kernel`; Qwen BF16 `qwen35_*rmsnorm*` family | **Landed for BF16 raw-pointer wrappers** | PARO out-kernels multiply direct norm weights; Qwen kernels use `1.0 + weight_delta`. Full inference still needs router/MoE/shared/attention dependencies. |
| Router + shared gate | `qwen35_router_logits_kernel`, `qwen35_router_select_kernel`, `hip_qwen35_router_topk_shared_out` | **Partial:** BF16 hidden/weight raw-pointer shared-out route landed | Current wrapper writes logits/selected/routing buffers and shared-gate logits; add FP16 hidden specialization if the final HIPENGINE path keeps FP16 router inputs. |
| Selected gate/up GEMV | `gemv_awq_selected_dual_pack8_strided_kernel`, `gemv_awq_selected_dual_pack8_kernel`, optional rotate-out variant | **Partial:** BF16 raw-pointer strided/transposed dual pack8 wrappers landed | Decode path uses stacked/repacked selected-expert W4 pack8 qweights. Fused rotate-out variant remains missing. Preserve small-K safety fix from `59195ed`. |
| Activation + down rotation | `silu_mul_dual_rotate_out_kernel` (fallback `silu_mul_dual_out_kernel` + rotate) | **Landed for BF16 raw-pointer fused and fallback wrappers** | Default `NANOVLLM_PARO_MOE_SILU_DOWN_ROTATE_FUSED=1`; fused dual rotate plus dual/pair fallback kernels are registered. |
| Selected down GEMV | `gemv_awq_selected_pack8_kernel` / strided wrapper | **Landed for BF16 raw-pointer strided/transposed pack8 wrappers** | Used for selected down projection; small-K specialization applies where safe. |
| Shared expert | W8A16 shared gate/up/down (`w8a16_*shared*`, `w8a16_single_*`, `w8a16_linear*`) | **Landed for current parent lowp-linear route** | `w8a16-shared-expert-hip` validates W8A16 gate/up → `silu_mul_dual_out` → W8A16 down (`gate_up_mismatch=0`, `intermediate_mismatch=0`, `out_mismatch=0`); specialized `w8a16_*shared*` fusion remains optional/future. |
| Weighted combine + residual | `weighted_sum_shared_gate_combine_residual_out_kernel`; fallback `weighted_sum_out_kernel`, `shared_gate_combine*` | **Landed for BF16 values with FP32 weights/gate logits** | c=1 decode promoted path fuses selected sum, shared sigmoid/gate combine, and residual add; scalar-weight fallback remains unported. |
| Synthetic c=1 vertical smoke | RMSNorm → router → selected W4 gate/up/down → W8A16 shared → weighted/shared/residual combine | **Landed** | `paro-moe-c1-hip --hidden-size 8`: all staged BF16 oracle checks bit-exact (`final_mismatch=0`); full model path still needs loader/model/attention plumbing. |

#### MoE prefill compact-WMMA path

| Stage | Parent kernels / wrappers | HIPENGINE status | Notes / gate |
| --- | --- | --- | --- |
| Lane grouping | `qwen35_moe_group_count_kernel`, `qwen35_moe_group_prefix_kernel`, `qwen35_moe_group_scatter[_gather]_kernel`, `qwen35_moe_gather_packed_hidden_kernel` | Missing | Required before either GEMV fallback or WMMA path can run. |
| Compact WMMA tile map | `qwen35_moe_wmma_tile_map_kernel` | Missing | New current-OPTIMAL kernel; maps compact expert starts to WMMA tiles without pad-multiple=16 overhead. |
| Gate/up compact WMMA | `gemm_awq_selected_dual_pack8_wmma_compact_kernel` | Missing | Current prefill route for `tokens >= 64`; noncompact WMMA and GEMV-only remain fallback/comparison paths. |
| Activation + down rotation | `silu_mul_dual_rotate_out_kernel` | Missing | `NANOVLLM_PARO_MOE_GROUPED_STACKED_SILU_ROTATE_FUSED=1` default. |
| Down compact WMMA | `gemm_awq_selected_pack8_wmma_compact_kernel` | Missing | Paired with compact tile map and compact buffers. |
| Weighted lane accumulation | `weighted_lanes_sum_out_kernel` | Missing | Default-on grouped-stacked weighted-lane accumulation; parent spot check +3.1% prefill at 512/128. |
| GEMV fallback/comparison | `gemv_awq_selected_dual_pack8*`, `gemv_awq_selected_pack8*` | Missing | Needed for token counts below WMMA crossover and for regression comparisons. |

#### Full-inference dependencies outside MoE

| Area | Required for reproducing parent inference | HIPENGINE status |
| --- | --- | --- |
| PARO quant plugin / weight layout | `w4_paro` plugin, pack8 replacement layout, compact stacked MoE weights, W8A16 shared/lm-head replacements | Missing; only `bf16` plugin landed. |
| Model plugin / scheduler | Qwen3.5 hybrid full-attn + linear-attn/GDN + MoE layer sequence, static decode buffers, one-step graph replay | Missing; `LLM.generate()` is still scaffolded. |
| Linear projections | `gemv_awq_pack8`, `gemv_awq_dual_pack8`, `dense_gemv_out`, rotation helpers | Missing. |
| Linear attention / GDN | `qwen35_linear_attn_conv_*`, `qwen35_gdn_*` incl. lowp recurrent RMSNorm gate | Partial: decode convolution FP32/BF16 variants and lowp recurrent RMSNorm+gate landed; prefill conv/state and remaining GDN fallback kernels still missing. |
| Full attention / KV | `qwen35_head_rmsnorm_partial_rotary*`, `qwen35_write_paged_kv_mixed_value*`, paged/split-K full-attention decode family, `full_attn_gate_mul_out` | Partial: full-attention prelude (`partial_rotary`, fused head RMSNorm + rotary, position variant) landed; KV write/decode still missing and must be reconciled with HIPENGINE `KVLiveSpans` ABI rather than parent `(block_table, context_len)` shortcuts. |
| Final head | W8A16 `lm_head` replacement path | Missing. |
| Eval harness | Parent baseline JSON capture + HIPENGINE JSON schema-2 artifacts + KL/top-1/sample/graph validation gates | Not yet landed. |

#### Port order for the OPTIMAL exercise

1. **Measurement harness first:** run/record the parent `512/128` and `4K/128` OPTIMAL commands as source-lineage artifacts, then create a blocked HIPENGINE artifact until `LLM.generate()` exists.
2. **MoE c=1 decode vertical slice:** PARO RMSNorm out-kernels → router/shared-gate → selected pack8 GEMV → fused activation/down-rotation → W8A16 shared expert → weighted shared-gate residual combine.
3. **MoE prefill compact-WMMA slice:** lane grouping/gather → compact tile map → compact dual/single WMMA → weighted-lane accumulation → GEMV fallback.
4. **Full-inference closure:** weight loader/model plugin, non-MoE projections, linear attention/GDN, full attention/KV, final head, graph replay, then end-to-end correctness/perf comparison.

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

Preserve all `__launch_bounds__`, template specializations, and compiler flags (`-mllvm -amdgpu-unroll-threshold-local=600` for decode/prefill, plus `-mcumode` for decode). A port that rewrites kernel bodies is not a port.

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
| `decode` | `-mllvm`, `-amdgpu-unroll-threshold-local=600`, `-mcumode` | 64 | Decode-phase kernels (paged attention, W8A8 grouped MoE decode, paro GEMV) |
| `prefill` | `-mllvm`, `-amdgpu-unroll-threshold-local=600` (WGP mode) | 32 | Prefill-phase kernels (GEMM, W8A16 linear prefill) |
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

If the target path uses `hipengine.core.build` JIT from Python, prebuild outside the profiler and make the profiled process cache-only. `rocprofv3` launch mode preloads into child processes; letting a profiled Python process spawn `hipcc`/clang can hang or abort in LLVM initialization.

```bash
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_smoke_add(load=False, compiler_version=version).output_path)
PY
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke -- \
  python3 scripts/smoke.py --mode smoke-add-hip --n 1024 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Expected output: a CSV with a kernel-name column (`Kernel_Name` / `KernelName`), grid/workgroup columns, `VGPR_Count`, `Scratch_Size`, and `LDS_Block_Size`. Some ROCm 7.13 traces emit `Start_Timestamp` + `End_Timestamp` instead of `DurationNs`; compute `DurationNs = End_Timestamp - Start_Timestamp` for summaries. Check:

- The expected kernel name appears.
- Duration is plausible (same order of magnitude as the reference).
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
