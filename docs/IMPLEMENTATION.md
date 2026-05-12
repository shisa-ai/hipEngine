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
- [ ] Resolve `rocprofv3` trace hang for Python/ctypes smoke before first real kernel port.
- [ ] Add minimal `scripts/smoke.py` path that exercises `LLM.generate()` once the engine loop exists.

## Notes

- Kernel R&D remains in `~/amd-gpu-tuning/`; this repo receives stable ports.
- Any unchecked item that changes architecture should update `docs/PLAN.md` when it lands.
