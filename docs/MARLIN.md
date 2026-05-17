# Marlin-K / Vulkan-Style W4 Layout Port Analysis

Date: 2026-05-16  
Target repo: `~/hipEngine`
Parent evidence repo: `~/amd-gpu-tuning` / `nano-vllm-amd` branch `gfx1100-qwen3.5`

## Executive summary

The parent workspace now has a retained, measured Marlin-K path for Qwen3.5/PARO non-expert c=1 decode. This is **not** a broad new quantization mode; it is a narrow but useful layout/wrapper change:

- Use a K-contiguous Marlin/Vulkan-style W4 buffer for rows==1 non-expert GEMV.
- Keep existing pack8/fused paths alive through a zero-copy `qweight_pack8` view of the same large W4 buffer.
- Keep original AWQ `qzeros/scales` for prefill/fused pack8 and small transposed `qzeros_mk/scales_mk` for decode locality.
- Do **not** port the rejected punchlist experiments (permute unpack, half2 FMA, launch-bound sweeps, etc.) as defaults.

Measured retained impact in `~/amd-gpu-tuning`:

| Model / shape | Pack8 decode | Marlin-K qweight-neutral decode | Delta | Peak memory note |
| --- | ---: | ---: | ---: | --- |
| 0.8B 4096/128 | 217.18 tok/s | 222.88 tok/s | **+2.6%** | 2.3289 -> 2.3379 GiB |
| 35B 512/128 | 108.26 tok/s | 109.91 tok/s | **+1.5%** | 18.8401 -> 18.8634 GiB |
| 35B 4096/128 | 105.94 tok/s | 107.14 tok/s | **+1.1%** | 21.6870 -> 21.7103 GiB |

The important memory result is that the earlier duplicate-buffer overhead is gone: 35B 512/128 Marlin-K peak overhead moved from the diagnostic `+0.621 GiB` row to `+0.023 GiB`. The remaining overhead is zero/scale metadata, not duplicate W4 qweight residency.

This is exciting for hipEngine because it is a stable, evidence-backed parent path with a small but real decode gain and a cleaner memory story. It is also a good test case for hipEngine's raw-pointer port discipline: the kernel is self-contained, shape-specialized, and narrow enough to port without touching the whole runtime first.

## What "Marlin-K" means here

This document uses **Marlin-K** to mean the frozen parent layout version `paro_marlin_k_v0`, not upstream CUDA Marlin and not the whole llama.cpp Vulkan backend.

Parent source layout contract from `nano-vllm-amd/nanovllm/native/qwen35/paroquant.py::_repack_awq_to_marlin_k_v0`:

```text
Current PARO/AWQ layout:
  qweight int32 [K, N/8]
  qzeros  int32 [K/group_size, N/8]
  scales  fp16/bf16/fp32 [K/group_size, N]

Marlin-K v0 layout:
  qweight_mk int32 [N/8, K/128, 128]
  qzeros_mk  int32 [N/8, K/128]
  scales_mk  scales dtype [N/8, K/128, 8]
```

The repack is simply:

```python
qweight_mk = qweight.contiguous().view(groups, group_size, n8).permute(2, 0, 1).contiguous()
qzeros_mk = qzeros.contiguous().transpose(0, 1).contiguous()
scales_mk = scales.contiguous().view(groups, n8, 8).permute(1, 0, 2).contiguous()
```

Constraints that should become hipEngine validation checks:

- `bits == 4`
- `group_size == 128`
- `K % group_size == 0`
- `out_features == out_packed * 8`
- `qweight_mk.shape == (out_packed, groups, 128)`
- `qzeros_mk.shape == (out_packed, groups)`
- `scales_mk.shape == (out_packed, groups, 8)`

## Parent implementation source references

### Source files and commits

Current source-of-truth parent checkout observed by `python3 scripts/check_lineage.py --file '*paroquant*' --diff stat` from hipEngine:

- Repo: `/home/lhl/amd-gpu-tuning/nano-vllm-amd`
- Branch: `gfx1100-qwen3.5`
- HEAD: `1522293 perf(paroquant): make marlin-k qweight neutral`

Relevant parent commits:

| Commit | Purpose | Main source files |
| --- | --- | --- |
| `3decc96 chore(paroquant): add marlin-k env gate` | Adds opt-in Marlin-K replacement knobs. | `nanovllm/native/qwen35/paroquant.py` |
| `f6afc2a feat(paroquant): add marlin-k repack buffers` | Adds Marlin-K repack/layout buffers. | `paroquant.py` |
| `fc756e1 feat(paroquant): add marlin-k fma kernel` | Adds first rows==1 FMA kernel. | `paroquant_kernels.py` |
| `73b1d06 feat(paroquant): add marlin-k q8 staging` | Adds Q8 staging experiment; not promoted. | `paroquant_kernels.py` |
| `ecfe9b3 feat(paroquant): add marlin-k sudot4 path` | Adds SUDOT4 experiment; not promoted. | `paroquant_kernels.py` |
| `64a3094 feat(native): prebuild-then-profile build cache + require-cached mode` | Adds parent JIT prebuild/require-cached support used during profiling. | `nanovllm/native/amd/build_cache.py`, `scripts/prebuild_native.py`, loaders |
| `7718fff perf(paroquant): speed up marlin-k fma decode` | Retained vec8 FMA speedup; establishes tuned Marlin-K microbench baseline. | `paroquant_kernels.py` |
| `1522293 perf(paroquant): make marlin-k qweight neutral` | Retained memory-neutral/qweight-neutral dispatch and measured E2E gain. | `paroquant.py` |

### Parent code anchors

Use these as the direct port anchors:

- `/home/lhl/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant.py`
  - `_repack_awq_to_marlin_k_v0(...)`
  - `ParoQuantLinear.__init__(...)` Marlin-K buffer materialization
  - `_marlin_k_memory_neutral` branch
  - zero-copy `qweight_pack8 = qweight_mk.view(out_packed, in_features)` registration
  - `ParoQuantLinear.forward(...)` rows==1 `gemv_paro_marlin_k_fma(...)` dispatch
- `/home/lhl/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py`
  - `_MARLIN_K_FMA_SRC`
  - `gemv_paro_marlin_k_fma_kernel`
  - `gemv_paro_marlin_k_fma(...)` wrapper shape checks and thread selection
  - `gemv_paro_marlin_k_q8_fma(...)` and `gemv_paro_marlin_k_sudot4(...)` are useful negative references but should not be promoted in hipEngine yet.
- `/home/lhl/amd-gpu-tuning/tools/paro_marlin_k_repack_reference.py`
  - Repack/reference helper to use when writing hipEngine's torch-free NumPy repack tests.
- `/home/lhl/amd-gpu-tuning/scripts/check_paro_marlin_k_fma_correctness.py`
  - Parent micro correctness gate for FMA/Q8/SUDOT4 parity.
- `/home/lhl/amd-gpu-tuning/scripts/bench_paro_gemv.py`
  - Parent c=1 GEMV shape microbench used throughout the §12 punchlist.

## Parent docs, worklog, and artifact references

### Current optimal configuration

- `/home/lhl/amd-gpu-tuning/docs/OPTIMAL.md`
  - Section: **Latest Retained Implementation Update (2026-05-16)**.
  - Adds `NANOVLLM_PARO_MARLIN_K_REPLACE=1` to the optimal flag set.
  - States to leave `NANOVLLM_PARO_MARLIN_K_KEEP_FALLBACK` unset/`0`.
  - Records the qweight-neutral E2E table and memory result.

### Design/roadmap evidence

- `/home/lhl/amd-gpu-tuning/PLAN-PAROQUANT2.md`
  - Section **11.11 / qweight-neutral replacement evidence**: table with the retained 0.8B and 35B rows.
  - Section **12. Marlin-K Optimization Roadmap (post §11.11)**: live-to-closed punchlist for every follow-up candidate.
  - Lane A:
    - `A1` accepted: qweight-neutral replacement.
    - `A2` accepted: prefill/fused pack8 preservation through the zero-copy view.
    - `A4` parked: residual `qzeros_mk/scales_mk` overhead is only ~0.023 GiB on 35B and not worth a decode regression.
  - Lanes B-F: rejected/parked negative evidence for deeper kernel/codegen changes.

### Worklog entries

- `/home/lhl/amd-gpu-tuning/WORKLOG.md`
  - `2026-05-15 20:10 UTC — Marlin-K qweight-neutral replacement`
    - Code commit: `nano-vllm-amd@1522293`.
    - Correctness command: `scripts/check_paro_marlin_k_fma_correctness.py --rows 2 --include-q8 --include-sudot4 --json`.
    - E2E table with finite logits, graph validation, matching samples, and memory.
  - `2026-05-15 20:45 UTC — §12 Marlin-K roadmap reconciliation after qweight-neutral work`
    - Ties the roadmap to `nano-vllm-amd@7718fff` and `@1522293`.
  - `2026-05-16 05:15 UTC — OPTIMAL.md refreshed for qweight-neutral Marlin-K`
    - Documents the optimal flag update and retained result rows.
  - Multiloop entries on 2026-05-15/16 for A3, B1-B7, C1-C4, D1-D6, E1-E6, F1-F4.
    - These are the negative evidence base; do not resurrect those changes as hipEngine defaults without a new parent-side audit.

### Parent artifacts

Key artifact paths in `/home/lhl/amd-gpu-tuning`:

- `artifacts/paro_marlink_memory_neutral_20260515_iter1/summary.md`
  - Primary retained E2E evidence for qweight-neutral replacement.
- `artifacts/multiloop_paro_marlink_prefill_20260515_a3/summary_rows.md`
  - Multi-row gate rejected; keep rows==1 dispatch initially.
- `artifacts/multiloop_paro_marlink_prefill_20260515_b1/`
  - qweight vector-load rejected.
- `artifacts/multiloop_paro_marlink_prefill_20260515_b2/`
  - activation vector-load rejected.
- `artifacts/multiloop_paro_marlink_prefill_20260515_b3/`
  - scale vector-load rejected.
- `artifacts/multiloop_paro_marlink_prefill_20260516_b4/summary.md`
  - `__builtin_amdgcn_perm` nibble unpack rejected.
- `artifacts/multiloop_paro_marlink_prefill_20260516_b5/summary.md`
  - scalar half and half2 FMA rejected.
- `artifacts/multiloop_paro_marlink_prefill_20260516_b6/summary.md`
  - accumulator layout parked.
- `artifacts/multiloop_paro_marlink_prefill_20260516_b7/summary.md`
  - inner-loop weight-local hoist rejected.
- `artifacts/multiloop_paro_marlink_prefill_20260516_f1/summary.md`
  - Triton c=1 GEMV prototype rejected.
- `artifacts/multiloop_paro_marlink_prefill_20260516_f2/summary.md`
  - c=1 WMMA parked; useful only for future naturally-full M16/grouped work.

## Retained parent algorithm

The retained FMA kernel has the following structure:

- Grid: `(out_packed, rows)`.
- One workgroup computes one output pack of 8 output channels for one input row.
- Default thread selection in parent wrapper:
  - `threads = 64` by default.
  - `threads = 128` for `in_features >= 4096`, `(in_features == 2048 and out_features <= 2048)`, or `out_features <= 512`.
  - Optional parent env `NANOVLLM_PARO_MARLIN_K_FMA_THREADS` accepts `32/64/128` for diagnostics.
- Inner loop consumes eight contiguous K elements per thread iteration:
  - `vec_stride = blockDim.x * 8`.
  - For each 8-K chunk, load one zero tuple and eight scales.
  - For each `j in 0..7`, load one packed int32 qweight word and accumulate 8 output lanes with FP32 `fmaf`.
- Reduction:
  - wave32 `__shfl_down` reductions inside each half-wave.
  - shared-memory exchange for multiple 32-lane groups.
  - thread 0 writes the eight output channels.

Important: several attempts to outsmart this loop were tried and rejected. For the hipEngine first port, preserve the retained kernel, not the experiments.

## Why qweight-neutral matters

The initial Marlin-K prototype won small decode speed but retained duplicate large W4 buffers: original pack8/AWQ residency plus Marlin-K qweight. That was unacceptable for 35B memory headroom.

The retained parent solution makes `qweight_mk` the one large buffer for eligible non-expert linears and exposes existing fused pack8 paths through a view:

```text
qweight_mk shape:        [out_packed, groups, 128]
qweight_pack8 view:      [out_packed, in_features]
qweight_pack8 storage:   same allocation as qweight_mk
```

In hipEngine this is more than a Python convenience: `DeviceWeightMap.free()` currently frees every `DeviceTensorAllocation` it owns. A zero-copy view must therefore be represented as either:

1. a `Tensor` alias that is not separately owned/freed, or
2. a named view stored outside the owning allocation map, or
3. an enhanced allocation map that distinguishes owning allocations from aliases.

Do **not** create two owning `DeviceTensorAllocation` entries with the same pointer; that risks double-free. This is one of the main hipEngine-specific design points for the port.

## hipEngine port surface

### Preferred first implementation files

Add a separate Marlin-K family rather than extending `paro_awq_gemv` in-place:

```text
hipengine/kernels/hip_gfx1100/quant/paro_marlin_k.hip
hipengine/kernels/hip_gfx1100/quant/paro_marlin_k.py
tests/test_paro_marlin_k_plan.py
```

Then wire the package export:

```text
hipengine/kernels/hip_gfx1100/quant/__init__.py
```

Reasons to keep it separate:

- `paro_awq_gemv.hip` already contains the landed pack8/fusedw4 family and is a large, active source file.
- Marlin-K has different tensor shapes and dispatch rules.
- A separate `.so` makes profiling and require-cached behavior easier to reason about.

### Runtime/loading integration files after the standalone kernel passes

Once the standalone wrapper/test is green, integrate with Qwen3.5/PARO loading and decode:

```text
hipengine/loading/qwen35_paro.py
hipengine/runtime/qwen35_paro.py
hipengine/runtime/qwen35_paro_runner.py
tests/test_qwen35_paro_layout.py
tests/test_qwen35_decode_state.py
```

Likely loading work:

- Add a torch-free NumPy equivalent of `_repack_awq_to_marlin_k_v0`.
- Materialize `qweight_mk`, `qzeros_mk`, and `scales_mk` via `load_host_array_to_device(...)` / `load_host_array_to_device_as_dtype(...)`.
- Preserve original `qzeros/scales` for pack8/fusedw4 paths.
- Represent `qweight_pack8` as a non-owning `Tensor` alias of `qweight_mk` with shape `(out_packed, in_features)`.
- Gate Marlin-K to non-expert rows==1 linears initially.

Likely runtime work:

- Register/resolve a new kernel key for Marlin-K decode, for example:
  - `KernelKey("hip_gfx1100", "pack8_gemv", "w4_paro", "marlin_k_fma_fp16")`, or
  - a distinct layer key such as `marlin_k_gemv` if the current registry naming prefers family-specific layers.
- Do not add backend/quant conditionals in general dispatch; keep the four-axis plugin registry invariant from `AGENTS.md`.
- Use Marlin-K only for rows==1 non-expert linears at first.
- Keep existing pack8/fusedw4 paths for prefill and fused multi-projection decode.

### Build-system notes

The C ABI wrapper should follow existing hipEngine raw-pointer wrappers:

- Use `hipengine.core.build.plan_hip_build(...)` / `build_hip(...)`.
- Family name suggestion: `paro_marlin_k`.
- Output name suggestion: `paro_marlin_k.so`.
- Start with the `decode` profile for gfx1100.
- Export one wrapper first:

```text
hipengine_gemv_paro_marlin_k_fma_fp16
```

BF16 can be added if the current hipEngine resident path needs it, but the parent optimal path and hipEngine PARO path are increasingly FP16 for parent-parity activation streams.

### C ABI contract sketch

A minimal raw-pointer wrapper should take:

```cpp
extern "C" int hipengine_gemv_paro_marlin_k_fma_fp16(
    const void* x,
    const int32_t* qweight_mk,
    const int32_t* qzeros_mk,
    const void* scales_mk,
    const void* bias,        // optional/null for first port
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int threads,
    hipStream_t stream);
```

Wrapper checks should enforce the layout constraints listed above before launch. In Python, mirror the style in `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.py`: `ctypes` argtypes, `HIP_SUCCESS` check, optional `library`, optional `runtime`, and `stream=0` default.

## Suggested hipEngine validation plan

### RED/GREEN unit tests

1. **Repack reference test**
   - Use small deterministic int32 qweight/qzeros and FP16 scales.
   - Compare NumPy hipEngine repack against the formula above and, if convenient, against `/home/lhl/amd-gpu-tuning/tools/paro_marlin_k_repack_reference.py`.

2. **Kernel plan/build test**
   - `plan_paro_marlin_k_build(...)` returns a `BuildArtifact` with expected family/output.
   - `build_paro_marlin_k(..., dry_run=True)` does not compile.

3. **GPU smoke**
   - Tiny shape: rows 1-2, `K=128 or 256`, `out_features=16 or 32`.
   - Compare against a NumPy CPU W4/AWQ oracle using the same layout.
   - Require exact or tight lowp parity consistent with existing pack8 tests.

4. **Parent parity micro gate**
   - Port enough fixture generation to compare Marlin-K output against hipEngine's existing pack8 GEMV for the same qweight/qzeros/scales.
   - This mirrors parent `check_paro_marlin_k_fma_correctness.py`.

### Kernel smoke / profiler gate

After the GPU smoke is correct:

```bash
rocprofv3 --kernel-trace -d /tmp/hipengine-marlin-k-trace -o trace -- \
  python3 scripts/smoke.py --mode paro-marlin-k-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Record the kernel name, plausible duration, `VGPR_Count`, `Scratch_Size`, `LDS_Block_Size`, grid, and workgroup size in `WORKLOG.md` and then `docs/KERNELS.md` when the port lands.

### Performance gate before promotion

The first port should **not** claim the parent speedups until hipEngine measures them. Parent numbers are evidence for prioritizing the port, not a hipEngine benchmark row.

Promotion should require:

- Correctness vs CPU reference fixture gate from `docs/TESTING.md`.
- Same generated sample / KL / top-1 gate at the model level if wired into Qwen3.5 runtime.
- A comparison against the existing hipEngine pack8/fusedw4 path on the same W7900, model, quantization, prompt/decode shape, and command.
- Benchmark rollup updates per `AGENTS.md` if and only if the path is retained as faster:
  - `benchmarks/README.md`
  - `benchmarks/CHANGELOG.md`
  - compact JSON under `benchmarks/results/`

## What not to port as default

The parent §12 punchlist closed with zero additional promoted wins beyond qweight-neutral Marlin-K. Keep these out of the first hipEngine port:

| Item | Parent result | hipEngine action |
| --- | --- | --- |
| A3 multi-row Marlin-K gate | Rejected; rows>1 had shape-dependent regressions. | Keep rows==1 first. |
| B1 qweight int4 vector load | Correctness passed but only ~1.5% kernel avg and 0.8B regressions. | Do not port. |
| B2 activation vector load | 35B avg regressed, many regressions. | Do not port. |
| B3 scale vector load | 35B avg regressed; q_proj worst. | Do not port. |
| B4 `__builtin_amdgcn_perm` nibble unpack | Correct but no target win; K>=4096 regressed. | Do not port. |
| B5 scalar half / `__half2` FMA | Correct but scalar tiny/noisy, half2 regressed. | Keep FP32 `fmaf`. |
| B6 accumulator layout | Parked; prior sweeps do not justify structural rewrite. | Do not port. |
| B7 weight-local hoist | Correct but no-op/slight regression. | Do not port. |
| C1/C2 multi-row/WGSIZE templates | Rejected. | Do not port. |
| C3 dual Marlin-K projection | Superseded by zero-copy pack8 fused paths. | Keep existing fused pack8 paths. |
| C4 selected-MoE Marlin-K | Parked; selected-pack8 path inactive under current 35B optimal route. | Target active grouped-stacked MoE separately. |
| D1-D3 Q8/SUDOT4 | Parked; current in-kernel staging is dramatically slower. | Do not port as speed path. |
| D4-D6 fused prologues / selected-MoE activation | Wrong granularity or blocked by inactive surfaces. | Do not port. |
| E1-E6 compiler/codegen sweeps | Rejected/parked. | Use retained flags/profile only. |
| F1 Triton GEMV | 0 winning shapes, much slower. | No Triton dependency. |
| F2 c=1 WMMA | Padded c=1 loses; M16 useful only when all rows useful. | Reserve for future grouped/prefill work. |

## Relationship to current hipEngine state

hipEngine already has the pack8/fusedw4 side of the story:

- `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip`
- `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.py`
- `tests/test_paro_awq_gemv_plan.py`
- `hipengine/runtime/qwen35_paro.py` imports and calls pack8/fusedw4 wrappers.

The Marlin-K port should fit beside these, not replace them wholesale. The parent retained route is a **hybrid**:

```text
rows == 1, eligible non-expert single GEMV  -> Marlin-K FMA
prefill / fused pack8 / paired projections -> existing pack8/fusedw4 via qweight_pack8 view
selected/grouped MoE active surfaces       -> existing grouped WMMA/fused MoE/W8A16 paths
```

That hybrid is the design to reproduce in hipEngine.

## Open design choices for the port

1. **Alias representation for `qweight_pack8`.**
   - Needed to preserve pack8/fused paths without duplicate memory.
   - Must avoid double-free in `DeviceWeightMap`.

2. **Kernel key naming.**
   - Decide whether Marlin-K is a variant of `pack8_gemv` or a distinct `marlin_k_gemv` layer key.
   - Prefer whatever keeps dispatch branch-free and explicit in the registry.

3. **Host repack timing.**
   - Parent repacks with torch tensors during module init.
   - hipEngine should repack torch-free on host with NumPy during load unless a future HIP repack kernel is needed for load-time memory pressure.

4. **Scale dtype coverage.**
   - Existing hipEngine PARO path has both BF16 and FP16 wrapper coverage in many kernels.
   - Start with FP16 if the target resident path is FP16; add BF16 if a current smoke/runtime path consumes BF16 scales/activations for this surface.

5. **Benchmark scope.**
   - Parent speedups were measured in nano-vllm-amd, not hipEngine.
   - hipEngine promotion needs its own pack8-vs-Marlin measurement after runtime integration.

## Proposed first port checklist

1. Create `paro_marlin_k.hip` by extracting only the retained FMA kernel and helpers from parent `paroquant_kernels.py@1522293` / `7718fff` lineage.
2. Create `paro_marlin_k.py` with build/ctypes wrapper and dry-run plan tests.
3. Add a NumPy repack helper and test; do not use torch.
4. Add a tiny GPU smoke against a CPU oracle.
5. Run `rocprofv3 --kernel-trace` smoke and record kernel metadata.
6. Update `docs/KERNELS.md` and `docs/source_lineage.json` only when the port lands, not while this is analysis-only.
7. Integrate qweight-neutral aliasing into `loading/qwen35_paro.py` and runtime rows==1 dispatch.
8. Measure hipEngine pack8 vs hipEngine Marlin-K before claiming any hipEngine speedup.

## Bottom line

Porting Marlin-K to hipEngine is worth doing now. The parent path has the two things we wanted before importing it into this repo:

1. **A retained implementation with correctness and memory evidence**, not just a kernel microbench.
2. **A closed negative-evidence punchlist**, so the first hipEngine port can be narrow and conservative.

The first hipEngine milestone should be a standalone raw-pointer Marlin-K FMA kernel + repack/oracle tests. Runtime promotion should follow only after we reproduce the parent hybrid memory story: Marlin-K for rows==1 non-expert GEMV, zero-copy pack8 view for existing fused paths, and no duplicate large W4 qweight buffer.
