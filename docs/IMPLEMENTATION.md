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
  - [x] Selected gate/up and down pack8 BF16 raw-pointer wrappers (`gemv_awq_selected_dual_pack8_*`, `gemv_awq_selected_pack8_*`); fused rotate-out variant still pending.
  - [x] Fused SiLU/down-rotation and fallback BF16 raw-pointer wrappers (`silu_mul_dual_rotate_out`, `silu_mul_dual_out`, `silu_mul_pair_rotate_out`).
  - [x] Weighted selected/shared-gate/residual combine BF16 raw-pointer wrappers (`weighted_sum_shared_gate_combine_residual_out`, `weighted_sum_out`, `shared_gate_combine*`).
- [ ] Port MoE prefill compact-WMMA slice (lane grouping/gather, compact tile map, compact WMMA, weighted lanes, GEMV fallback).
- [ ] Port full-inference dependencies outside MoE (w4_paro loader/layout, Qwen3.5 model plugin, non-MoE projections, linear attention/GDN, full attention/KV, W8A16 lm_head, graph replay).
- [ ] Reproduce parent correctness gates and performance rows with HIPENGINE artifacts/rollup updates.

## Notes

- Kernel R&D remains in `~/amd-gpu-tuning/`; this repo receives stable ports.
- Any unchecked item that changes architecture should update `docs/PLAN.md` when it lands.
