# HIPENGINE Kernel Playbook

This doc covers the mechanics of landing a kernel in HIPENGINE — porting from `~/amd-gpu-tuning/nano-vllm-amd/`, the JIT build layer, gotchas specific to this repo, and the correctness gate a port must pass.

**Kernel R&D does not live here.** Micro-tuning iteration loops (`rocprofv3 --kernel-trace` ranking, VGPR/occupancy hunting, `__launch_bounds__` sweeps, fusion experiments, the device-code gotcha catalog) belong in `~/amd-gpu-tuning/`. HIPENGINE receives *stable* kernels via the port pipeline below. If you find yourself opening a profiler inside the HIPENGINE tree, stop and move the experiment to the parent workspace.

See also:
- `docs/PLAN.md` "Kernel Port Strategy" — authoritative source inventory, split plan, per-family targets.
- `~/amd-gpu-tuning/AGENTS.md` — audit-first-via-rocprofv3, time-share/occupancy/iters-per-thread/VGPR discipline.
- `~/amd-gpu-tuning/LESSONS-LEARNED.md` — device-code gotchas and kernel lineage results.

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
