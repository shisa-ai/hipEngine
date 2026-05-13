# HIPENGINE Implementation Punchlist

Active implementation checklist. Keep this file lightweight; durable architecture lives in `docs/PLAN.md`, benchmark procedure in `docs/BENCHMARK.md`, and kernel port procedure in `docs/KERNELS.md`.

## Phase 0 — Foundation scaffold

- [x] Create package scaffold (`pyproject.toml`, `hipengine/`, `tests/`, `scripts/`, `benchmarks/results/`).
- [x] Add torch-free public API placeholders (`hipengine.LLM`, `SamplingParams`).
- [x] Add core value objects (`DType`, `Device`, `Tensor` handle scaffold).
- [x] Add 4-axis kernel registry (`KernelKey`, `register`, `resolve`, fallback order, clean missing errors).
- [x] Add model and quant plugin registries with toy model + fp16 quant plugin.
- [x] Add fusion planner spike (longest registered `+` composite, primitive fallback, plan resolution).
- [x] Add first CPU-reference kernels and correctness fixture format.
- [x] Add `hipengine.core.build` JIT cache implementation.
- [x] Add lazy HIP runtime/memory skeleton (no HIP library load on import).
- [x] Add first HIP smoke kernel source and dry-run registry/build plan (`smoke_add`).
- [x] Run first HIP smoke kernel (`smoke_add`) on GPU after explicit clearance.
- [x] Add source-lineage drift checker for `~/amd-gpu-tuning/nano-vllm-amd` port inputs.
- [x] Resolve `rocprofv3` trace hang for Python/ctypes smoke before first real kernel port.
- [x] Port first real gfx1100 model-layer family: Qwen3.5 BF16 `rmsnorm` raw-pointer wrappers.
- [ ] Add minimal `scripts/smoke.py` path that exercises `LLM.generate()` once the engine loop exists.

## OPTIMAL MoE/PARO reproduction exercise

Use `docs/KERNELS.md` "Current OPTIMAL MoE port checklist" as the live dependency map.

- [x] Map current `~/amd-gpu-tuning/docs/OPTIMAL.md` route against current parent HEAD and HIPENGINE-landed status.
- [x] Add parent-baseline/HIPENGINE-blocked benchmark artifacts for OPTIMAL 512/128 and 4K/128.
- [x] Port PARO RMSNorm out-kernels (`paro_rmsnorm_out`, `paro_add_rmsnorm_out`).
- [ ] Port MoE c=1 decode vertical slice (router, selected pack8 GEMV, fused activation/down-rotation, W8A16 shared expert, weighted shared-gate residual combine).
  - [x] Router/shared-gate BF16 hidden/weight raw-pointer path (`qwen35_router_topk_shared_out`).
  - [x] Selected gate/up and down pack8 BF16 raw-pointer wrappers (`gemv_awq_selected_dual_pack8_*`, `gemv_awq_selected_pack8_*`) plus fused rotate→selected dual GEMV (`gemv_awq_selected_dual_pack8_strided_rotate_out`).
  - [x] Fused SiLU/down-rotation and fallback BF16 raw-pointer wrappers (`silu_mul_dual_rotate_out`, `silu_mul_dual_out`, `silu_mul_pair_rotate_out`).
  - [x] Weighted selected/shared-gate/residual combine BF16 raw-pointer wrappers (`weighted_sum_shared_gate_combine_residual_out`, `weighted_sum_out`, `shared_gate_combine*`).
  - [x] W8A16 linear BF16/F32 raw-pointer kernels used by shared expert and lm-head (`w8a16_linear*`) plus composite W8A16 shared-expert smoke.
  - [x] Synthetic c=1 MoE decode vertical smoke (`paro-moe-c1-hip`) chaining RMSNorm, router, selected W4 experts, W8A16 shared branch, and weighted/shared/residual combine.
- [ ] Port MoE prefill compact-WMMA slice (lane grouping/gather, compact tile map, compact WMMA, weighted lanes, GEMV fallback).
- [ ] Port full-inference dependencies outside MoE (w4_paro loader/layout, Qwen3.5 model plugin, non-MoE projections, linear attention/GDN, full attention/KV, W8A16 lm_head, graph replay).
  - [x] Register `w4_paro` quant plugin metadata for dispatch/planning.
  - [x] Register Qwen3.5/PARO MoE model plugin metadata and representative decode layer sequence.
  - [x] Add torch-free safetensors/config discovery and tensor metadata index for loader scaffolding.
  - [x] Add torch-free safetensors host-to-device materialization helpers with owned raw-pointer tensor handles.
  - [x] Add Qwen3.5/PARO MoE c=1 normalized device-weight materialization map.
  - [x] Add Qwen3.5/PARO full-attention+MoE c=1 normalized device-weight materialization map.
  - [x] Add torch-free named runtime workspace allocator for scratch/device tensors.
  - [x] Add minimal Qwen3.5/PARO one-token decode-state scratch scaffold.
  - [x] Add Qwen3.5/PARO MoE c=1 checkpoint layout validator over tensor metadata.
  - [x] Port PARO BF16 dense GEMV (`dense_gemv_out`) for auxiliary dense projection paths.
  - [x] Port generic PARO pack8 GEMV (`gemv_awq_pack8*`, `gemv_awq_dual_pack8*`) for non-MoE Q/K/QKV/Z projection paths.
  - [x] Port PARO pairwise rotation helpers (`paro_rotate2`, `paro_rotate3`) for multi-projection PARO paths.
  - [x] Port Qwen full-attention prelude kernels (`partial_rotary`, `head_rmsnorm+partial_rotary`, position variant).
  - [x] Port Qwen linear-attention decode convolution (`qwen35_linear_attn_conv_decode*`, FP32/BF16 input variants).
  - [x] Port Qwen linear-attention recurrent GDN lowp RMSNorm+gate (`qwen35_gdn_recurrent_rmsnorm_gate_lowp`).
  - [x] Add `KVLiveSpans` scaffold and span-shaped Qwen paged-KV append bridge (`qwen35_write_paged_kv*_position_tensor`).
  - [x] Port span-shaped Qwen paged full-attention context-tensor decode (`qwen35_paged_full_attn_decode_context_tensor`).
  - [x] Port span-shaped Qwen split-K paged full-attention decode/reduce (`qwen35_paged_full_attn_decode_split_k_ctx_tensor`, `*_reduce`).
  - [x] Port span-shaped Qwen split-K gated FP32 reduce (`qwen35_paged_full_attn_decode_split_k_reduce_gate<float>`).
  - [x] Port span-shaped Qwen split-K gated BF16 reduce (`qwen35_paged_full_attn_decode_split_k_reduce_gate<hip_bfloat16>`).
  - [x] Port Qwen3.5 GQA-specialized split-K context kernels (`qwen35_paged_full_attn_decode_split_k_ctx_tensor_warp`, `*_gqa<8,16,2>`).
- [ ] Reproduce parent correctness gates and performance rows with HIPENGINE artifacts/rollup updates.

## Notes

- Kernel R&D remains in `~/amd-gpu-tuning/`; this repo receives stable ports.
- Any unchecked item that changes architecture should update `docs/PLAN.md` when it lands.
