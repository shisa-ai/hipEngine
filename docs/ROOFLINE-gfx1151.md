# ROOFLINE-gfx1151.md — Strix Halo / Radeon 8060S LLM Roofline

_Last updated: 2026-05-16_

This is the gfx1151 companion to `docs/ROOFLINE.md`.  The goal is not a
full architecture manual; it is a practical performance model for deciding
whether a local LLM inference result on Strix Halo is limited by memory
bandwidth, compute throughput, kernel launch/stack maturity, or insufficient
parallelism.

Short version:

- **Device:** Radeon 8060S Graphics, RDNA3.5 / `gfx1151`, integrated APU GPU.
- **Local geometry:** `rocminfo` reports **40 CUs**, **20 WGPs** via PyTorch,
  **80 SIMD32s**, wave32 default, 64 KiB exposed LDS/group memory, 2 MiB L2,
  and 32 MiB L3.
- **Peak compute:** about **59.4 TFLOP/s FP16/BF16/INT8 WMMA**, **118.8 TOP/s
  INT4 WMMA**, and **29.7 TFLOP/s FP32 with ideal VOPD dual issue** at 2.9 GHz.
- **Peak memory:** **256 GB/s decimal** from LPDDR5X-8000 on a 256-bit bus.
  Local/reference GPU memory tests reach roughly **221-234 GB/s** for large
  streams.
- **Against W7900/gfx1100:** ~48% of W7900 FP16/BF16/INT8 matrix compute but
  only ~30% of W7900 theoretical memory bandwidth, with much smaller L2/L3.
  Capacity can be larger because it is unified memory, but bandwidth is much
  lower than a dGPU.
- **Inference implication:** c=1 decode for quantized LLMs is still primarily
  a weight/KV bandwidth and occupancy problem; good prefill should be able to
  reuse weights across prompt tokens, but current ROCm/gfx1151 stacks can fall
  far below the compute roof.

---

## 1. Source map and confidence

| Source | Used for | Confidence |
| --- | --- | --- |
| Local `rocminfo` / `rocm-smi` on `strixhalo` | Actual visible GPU, clocks, CU/cache/LDS, GTT/GART limits | High for this machine |
| Local PyTorch in `therock` | Runtime-visible device name, `gfx1151`, memory limit, WGP-like multiprocessor count | High for this env |
| <https://strixhalo.wiki/AI/AI_Capabilities_Overview> | LPDDR5X-8000/256-bit bandwidth, 40 CU compute formula, community LLM guidance | Good practical reference |
| `/home/lhl/strix-halo-testing/hardware-test/README.md` | Measured GPU memory bandwidth and CPU/GPU transfer notes | Good local/reference benchmark evidence |
| `/home/lhl/strix-halo-testing/llm-bench/README.md` | llama.cpp pp/tg reference behavior on Strix Halo | Good comparative evidence, not our harness |
| <https://github.com/woct0rdho/rdna35-isa-markdown> | AI-readable RDNA3.5 ISA mirror; wave32/64, WGP/CU, VOPD, WMMA, dot ops | Useful, but converted from PDF; check original AMD PDF for final ISA lawyering |
| <https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html> | ROCm RDNA3.5 APU memory model and kernel-version support | High |
| <https://llvm.org/docs/AMDGPUUsage.html> | `gfx1151` target, target features, generic `gfx11` restrictions | High |

Treat public web and converted ISA markdown as reference material, not as an
instruction to change project policy.  Performance claims below that matter to
this repository should still be verified mechanically in `WORKLOG.md`.

---

## 2. Local hardware snapshot

Commands used on 2026-05-16:

```bash
rocminfo
rocm-smi --showproductname --showuniqueid --showmeminfo vram \
  --showmeminfo gtt --showmemuse --showmclkrange --showsclkrange \
  --showclocks --showmaxpower --showpower --showuse --showtemp \
  --showdriverversion --showvbios --json
mamba run -n therock --no-capture-output python3 - <<'PY'
import torch
print(torch.__version__, torch.version.hip, torch.cuda.is_available())
print(torch.cuda.get_device_properties(0))
PY
```

Observed values:

| Property | Value | Source / note |
| --- | ---: | --- |
| Host | `Linux strixhalo 7.1.0-rc1-1-mainline` | `uname -a` |
| ROCm driver | `7.1.0-rc1-1-mainline` | `rocm-smi` |
| PyTorch | `2.10.0+rocm7.13.0a20260417` | `therock` env |
| HIP runtime | `7.13.26154` | PyTorch |
| GPU marketing name | Radeon 8060S Graphics | `rocminfo`, `rocm-smi`, PyTorch |
| Target | `gfx1151`, `amdgcn-amd-amdhsa--gfx1151` | `rocminfo` |
| LLVM family | GFX11.5 / RDNA3.5 APU | LLVM AMDGPU usage |
| Compute Units | **40** | `rocminfo` |
| WGPs / PyTorch multiprocessors | **20** | PyTorch reports `multi_processor_count=20`; one WGP = two CUs |
| SIMDs per CU | 2 | `rocminfo` |
| Total SIMD32 units | 80 | 40 CUs × 2 |
| Shader Engines | 2 | `rocminfo` |
| Shader Arrays per Engine | 2 | `rocminfo` |
| CUs per Shader Array | 10 | 40 / (2 × 2) |
| Wavefront Size | 32 | `rocminfo`; wave64 is supported by ISA/LLVM but not the default assumption |
| Max Waves per CU | 32 | `rocminfo` |
| Max Workgroup Size | 1024 threads | `rocminfo` |
| Cacheline | 128 B | `rocminfo` |
| L1 | 32 KiB | `rocminfo` cache info |
| L2 | 2048 KiB | `rocminfo` / PyTorch `L2_cache_size=2MB` |
| L3 | 32768 KiB | `rocminfo` cache info |
| LDS / group segment | 64 KiB | `rocminfo`; ISA describes CU/WGP mode sharing/splitting details |
| HSA max clock | 2900 MHz | `rocminfo`; `rocm-smi` valid sclk range 600-2900 MHz |
| Current sampled clocks | `sclk≈1007 MHz`, `mclk=1000 MHz` | idle/light-load sample from `rocm-smi`, not a benchmark-load clock |
| BIOS/GART-like VRAM aperture | 512 MiB total, ~397 MiB used | `rocm-smi --showmeminfo vram`; expected small reservation |
| GTT limit | 125,829,120,000 B ≈ 117.2 GiB | `rocm-smi --showmeminfo gtt` |
| TTM pages limit | 31,457,280 pages = 120 GiB | `/sys/module/ttm/parameters/pages_limit` |
| PyTorch visible memory | 64,041 MiB | `torch.cuda.get_device_properties`; allocator/runtime limit, not physical LPDDR size |
| HSA pool size | 125,829,120 KiB ≈ 120 GiB | `rocminfo` coarse/fine global pools |

### 2.1 CU vs WGP naming trap

`rocminfo` says **40 Compute Units**.  PyTorch says
`multi_processor_count=20`.  For RDNA, PyTorch's count is WGP-like: one WGP
contains two CUs, and one CU contains two SIMD32s.  For grid-sufficiency audits
in this project, use the real **40 CU** count unless the profiler/tool reports
in WGP units.

### 2.2 Unified memory naming trap

ROCm's RDNA3.5 system optimization doc explains that these APUs use GPUVM over
physically shared system memory rather than a discrete VRAM pool.  GART/GTT
are mapping limits and allocation policies, not separate fast/slow physical
memories like dGPU VRAM vs host RAM.  Keeping firmware-reserved VRAM/GART small
and using large GTT-backed allocations is generally preferred for AI workloads.

On this machine, that is exactly what `rocm-smi` shows: a 512 MiB
VRAM/GART-like aperture plus a ~120 GiB GTT limit.  Do **not** compare
`rocm-smi --showmeminfo vram` directly to W7900's 48 GiB VRAM; for PyTorch and
native ROCm inference the relevant large pool is GTT/HSA global memory.

---

## 3. Peak throughput model

### 3.1 Compute peaks at 40 CU and 2.9 GHz

The Strix Halo wiki gives the practical RDNA3.5 AI compute formula for the
40-CU Radeon 8060S:

```text
512 ops/clock/CU × 40 CU × 2.9e9 clock = 59.392e12 ops/s
```

Using that and the RDNA3.5 ISA/LLVM feature set:

| Class | Peak | Formula / note |
| --- | ---: | --- |
| FP32 FMA, no VOPD | **14.85 TFLOP/s** | 40 CU × 2 SIMD32/CU × 32 lanes × 2 flop/FMA × 2.9 GHz |
| FP32 FMA, ideal VOPD dual issue | **29.70 TFLOP/s** | VOPD can encode two independent VALU ops, but only for legal wave32 pairs |
| FP16/BF16 WMMA | **59.39 TFLOP/s** | 512 ops/clock/CU class |
| INT8 WMMA | **59.39 TOP/s** | RDNA3.5 has `V_WMMA_I32_16X16X16_IU8`; same peak class as FP16/BF16 |
| INT4 WMMA | **118.78 TOP/s** | `V_WMMA_I32_16X16X16_IU4`; 2× INT8 op density |
| NPU INT8 | 50 TOP/s | Separate Ryzen AI NPU, not used by our HIP kernels |

Important caveats:

- VOPD is a **wave32** dual-issue encoding; the converted RDNA3.5 ISA says it
  must not be used by wave64 kernels.  It is useful for scalar/vector decode
  kernels only when the compiler or hand-written ISA can pair independent ops.
- WMMA operates on 16×16×16 matrix tiles.  It is excellent for batched/prefill
  GEMMs and attention tiles, but c=1 decode GEMV can waste most of a tile if
  forced into WMMA without enough rows/tokens to amortize it.
- LLVM lists `gfx1151` under GFX11.5 APU targets with `cumode` and
  `wavefrontsize64` features available.  Availability is not a recommendation:
  compile native `gfx1151`, probe wave behavior, and keep reductions correct
  for the actual wave mode.

### 3.2 Memory bandwidth peak and measured ceiling

The platform memory roof from the Strix Halo wiki and local
`strix-halo-testing` notes is:

```text
LPDDR5X-8000 × 256-bit bus / 8 = 256 GB/s decimal
```

Local/reference measurements from `/home/lhl/strix-halo-testing/hardware-test`:

| Test / setting | Write | Read/check | Note |
| --- | ---: | ---: | --- |
| AMDVLK default `memtest_vulkan` | 214.6 GB/s | 203.7 GB/s | Out-of-box Vulkan large stream |
| AMDVLK + `amd_iommu=off` | 216.7 GB/s | 212.0 GB/s | IOMMU off improves read bandwidth |
| Mesa RADV | 227.8 GB/s | 214.4 GB/s | Comparable or slightly better on that system |
| Mesa RADV + `tuned` accelerator-performance | **234.4 GB/s** | **221.0 GB/s** | Best retained local/reference row |
| ROCm peer/copy GPU-side row | 212.6 GB/s | — | `rocm-bandwidth-test.out` row in hardware notes |

For roofline work in this repo:

- Use **256 GB/s** as the theoretical memory roof.
- Use **~221 GB/s read** as a realistic optimistic roof for read-heavy inference
  until a ROCm/rocprof counter run proves a higher sustained value for the
  exact kernel.
- Use **~200-215 GB/s** as a conservative sustained decode expectation if the
  stack is not tuned (`amd_iommu`, `tuned`, clocks, driver path, kernel shape).

### 3.3 Operational-intensity break-even points

Operational intensity (OI) is operations per byte loaded/stored.  A kernel is
memory-bound below `compute_peak / memory_bandwidth` and compute-bound above it.

Break-even OI using the 256 GB/s theoretical bus:

| Compute roof | Peak | OI needed to be compute-bound |
| --- | ---: | ---: |
| FP32 scalar FMA, no VOPD | 14.85 TFLOP/s | **58 ops/B** |
| FP32 ideal VOPD | 29.70 TFLOP/s | **116 ops/B** |
| FP16/BF16/INT8 WMMA | 59.39 TOP/s | **232 ops/B** |
| INT4 WMMA | 118.78 TOP/s | **464 ops/B** |

Using the measured ~221 GB/s read ceiling makes the break-even points even
higher: ~67, ~134, ~269, and ~537 ops/B respectively.

Typical c=1 decode weight GEMV intensities are far below this:

| Weight format | Idealized math per weight byte | Roofline implication |
| --- | ---: | --- |
| BF16/F16 weight | ~1 flop/B | Strongly memory-bound |
| INT8/W8A16 weight | ~2 ops/B before scale/dequant overhead | Strongly memory-bound |
| INT4/W4A16 weight | ~4 ops/B before scale/dequant overhead | Strongly memory-bound |

Therefore, quantized c=1 decode on gfx1151 cannot get close to the 59 TOP/s
matrix roof unless it increases reuse by batching/concurrency/speculation or
changes the algorithm.  The immediate levers are bytes/token, coalescing,
occupancy, fewer launches, and avoiding register/scratch pathologies.

---

## 4. LLM inference regimes on gfx1151

### 4.1 c=1 decode

Single-token decode has low reuse: each active weight is usually consumed once
for one output token.  The model is therefore bounded by a combination of:

1. effective LPDDR5X read bandwidth,
2. wave/grid occupancy and outstanding memory requests,
3. dequant/layout overhead,
4. launch overhead and graph replay quality,
5. long-context attention/KV traffic once context grows.

A simple bandwidth upper bound for active-weight streaming is:

```text
tokens/s <= effective_BW_bytes_per_s / active_weight_bytes_per_token
```

This is intentionally optimistic because it ignores scales, activations,
attention, router/top-k, writebacks, synchronization, launch gaps, and cacheline
waste.

Examples with a 221 GB/s optimistic read roof:

| Active bytes/token | Example intuition | Pure bandwidth upper bound |
| ---: | --- | ---: |
| 0.4 GB | small active MoE / compact W4 path | 552 tok/s |
| 0.75 GB | ~1.5B active params at W4 | 295 tok/s |
| 1.5 GB | ~3B active params at W4 | 147 tok/s |
| 3.0 GB | ~3B active params at W8 or ~6B W4 | 74 tok/s |

These are upper bounds, not promises.  A result well below the bound may still
be reasonable if the profile shows fragmented kernels, low occupancy, high VGPR
pressure, scratch, uncoalesced layout, or attention dominance.

### 4.2 Prefill / prompt processing

Prefill has much higher potential OI because a tile of weights can be reused
across many prompt tokens.  A crude weight-only estimate for a prompt length `T`
is:

```text
OI_weight_only ≈ (2 ops/MAC × T) / bytes_per_weight
```

For W4 weights at `T=512`, this can be thousands of ops/B, so a perfectly
tiled/fused prefill could become compute-bound.  In practice, Strix Halo prefill
is often much lower than the compute roof because of ROCm/gfx1151 stack maturity,
attention implementation, dequant/repack overhead, kernel-launch granularity,
GEMM shape choices, and memory-system details.

This matches the `strix-halo-testing` llama.cpp summary: prompt processing has
improved substantially, especially with rocWMMA/hipBLASLt paths, but there is
still "a lot of performance on the table" for pp compared with the theoretical
59 TFLOP/s class roof.

### 4.3 Long-context attention

At short context, decode is usually dominated by active-weight streaming.  As
context grows, KV reads and attention math become more important.  gfx1151 has
only 32 MiB L3 and 2 MiB L2, so large KV streams will not stay on chip.  Long
context can therefore be limited by a different mixture:

- KV cache bandwidth and cacheline efficiency,
- attention kernel tiling and vectorization,
- rocWMMA/SDPA/AOTriton availability and correctness,
- graph replay or dispatch overhead for per-layer attention pieces.

Do not use a short-context decode roofline to explain a 32K-context result
without profiling the attention bucket.

---

## 5. gfx1151 vs gfx1100/W7900 comparison

| Property | gfx1151 Radeon 8060S | gfx1100 W7900 | gfx1151 / W7900 |
| --- | ---: | ---: | ---: |
| Architecture | RDNA3.5 APU | RDNA3 dGPU | — |
| CUs | 40 | 96 | 41.7% |
| WGPs | 20 | 48 | 41.7% |
| Max clock used for peak math | 2.9 GHz | 2.499 GHz product boost | 116% |
| FP16/BF16/INT8 matrix peak | 59.4 | 123 | 48.3% |
| INT4 matrix peak | 118.8 | 245 | 48.5% |
| FP32 ideal VOPD peak | 29.7 | 61.3 | 48.4% |
| External memory roof | 256 GB/s LPDDR5X | 864 GB/s GDDR6 | 29.6% |
| Measured/read-practical memory | ~221 GB/s | workload-dependent; W7900 theoretical 864 GB/s | 25.6% of W7900 theoretical |
| L2 | 2 MiB | 6 MiB | 33.3% |
| L3 / MALL / Infinity Cache | 32 MiB | 96 MiB | 33.3% |
| Capacity model | unified GTT/HSA pool; ~120 GiB configured, PyTorch sees ~64 GiB | 48 GiB dedicated VRAM | gfx1151 can fit bigger models, slower |

Interpretation:

- If a kernel is **pure compute-bound** and uses WMMA well, gfx1151 should be
  roughly half of W7900.
- If a kernel is **pure memory-bandwidth-bound**, gfx1151 should be roughly a
  quarter to a third of W7900 depending on achieved W7900 bandwidth and Strix
  tuning.
- If a kernel is **launch/occupancy/stack-bound**, ratios can be outside either
  simple roof.  Small models and short kernels often fall here.

Recent same-harness PARO sanity row from `WORKLOG.md` (model
`z-lab/Qwen3.5-0.8B-PARO`, native `gfx1151`, graph replay enabled):

| Shape | gfx1151 prefill | gfx1151 decode | W7900 retained prefill | W7900 retained decode | gfx1151 / W7900 prefill | gfx1151 / W7900 decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 2451.98 tok/s | 145.60 tok/s | 11363.34 tok/s | 251.78 tok/s | 21.6% | 57.8% |
| 4096/128 | 2477.52 tok/s | 137.72 tok/s | 12402.06 tok/s | 238.40 tok/s | 20.0% | 57.8% |

The decode ratio is better than the memory-bandwidth ratio because this small
model does not saturate W7900's memory roof and has fixed overheads.  The
prefill ratio is worse than the compute ratio, which points at stack/kernel
underutilization rather than a hard 59 TFLOP/s limit.

---

## 6. Kernel-development implications

### 6.1 Audit occupancy before micro-tuning

For gfx1151, the grid-sufficiency target is **40 CUs**.  A kernel with fewer
than 40 resident workgroups/waves cannot fill the GPU no matter how clever its
inner loop is.  Because PyTorch reports 20 WGPs, be explicit about whether a
profiler counter is in CU or WGP units.

Before HIP kernel micro-tuning, run the same audit style used for W7900:

```bash
cat > /tmp/pmc.txt <<'EOF'
pmc: SQ_WAVES
EOF
mamba run -n therock rocprofv3 -i /tmp/pmc.txt -f csv -o probe -- \
  python3 <bench.py>
mamba run -n therock rocprofv3 --kernel-trace -f csv -o trace -- \
  python3 <bench.py>
```

Then rank total kernel duration, check grid coverage, VGPR count, scratch size,
LDS use, within-block serialization, and iters/thread.  Do not assume attention
or GEMV is the bottleneck without the kernel-time table.

### 6.2 Prefer native gfx1151 code objects

Use `--offload-arch=gfx1151` / `PYTORCH_ROCM_ARCH=gfx1151` /
project-specific `*_HIP_ARCH=gfx1151` for native kernels.  In the 2026-05-16
PARO run, forcing `HSA_OVERRIDE_GFX_VERSION=11.0.0` with gfx1100 JIT targets
failed before model load with `HIP error: device kernel image is invalid`.

`gfx11-generic` can be useful for compatibility, but LLVM warns that generic
code can carry pessimizations and target restrictions.  For retained benchmark
numbers, prefer native `gfx1151` unless the experiment is explicitly a generic
code-object comparison.

### 6.3 Wave32, wave64, VOPD, and reductions

The RDNA3.5 ISA supports wave32 and wave64.  However:

- `rocminfo` reports wavefront size 32 on this machine.
- VOPD dual-issue is legal only for wave32 according to the converted ISA text.
- LLVM exposes `wavefrontsize64` and `cumode` as target features, but enabling
  a feature is not proof that a CUDA-style 64-lane collective behaves the way a
  kernel expects.
- For reductions over more than 32 lanes, keep LDS exchange or a verified
  per-wave scheme.  Do not remove barriers just because wave64 exists.

### 6.4 WMMA is for tiles, not magic c=1 GEMV

RDNA3.5 has WMMA instructions for F16, BF16, INT8, and INT4 16×16×16 tiles.
That is the right roof for prefill GEMM, batched decode, attention tiles, and
other shapes with enough rows.  For M=1 decode, scalar/vector W4/W8 kernels can
beat a naive WMMA path because WMMA must create tile occupancy and layout work
that may be mostly empty.

### 6.5 Memory settings matter, but they are not a substitute for bandwidth

Recommended Strix Halo AI setup from the ROCm doc and local notes:

- Keep firmware-reserved VRAM/GART small, for example 512 MiB.
- Raise TTM/GTT limits for large models (`/sys/module/ttm/parameters/pages_limit`).
- Consider `amd_iommu=off` if VFIO/passthrough is not needed; local notes show
  ~6% faster raw GPU memory reads, with smaller but nonzero llama.cpp effects.
- Use `tuned` `accelerator-performance` where appropriate; local notes show
  memory-bandwidth and pp improvements.
- Disable mmap for large ROCm llama.cpp model loads when documented by that
  stack; mmap can be catastrophically slow on large unified-memory ROCm loads.

---

## 7. Central gfx1151 tuning notes

This section intentionally goes beyond strict roofline analysis.  Keep it here
so the current Strix Halo tuning state is discoverable from one gfx1151 file.

### 7.1 Current tuning layout

As of this snapshot, there is **no** `nano-vllm-amd/tunings/` directory.  The
active tuning surfaces are:

- native/JIT architecture selection in `nano-vllm-amd/nanovllm/native/amd/`,
- PARO/nano-vllm environment flags, summarized in root `docs/OPTIMAL.md`,
- benchmark wrappers such as `scripts/bench_paro_native_engine.py` and
  `scripts/run_moe2_baselines.py`,
- local TileLang evidence in `/home/lhl/github/lhl/amd-tilelang-tuning`, and
- retained benchmark rows in `WORKLOG.md`.

Therefore, a future `gfx1151` preset should initially be a thin, explicit
collection of environment defaults and benchmark sweep ranges, not a separate
kernel architecture.  RDNA3.5 is close enough to RDNA3 that the existing
wave32/WMMA/pack8 paths run, but W7900-derived thresholds should not be assumed
optimal on the 40-CU, 256-GB/s APU.

### 7.2 Mandatory native-arch plumbing

Every retained gfx1151 native/JIT run should set native arch flags:

```bash
export NANOVLLM_AMD_HIP_ARCH=gfx1151
export NANOVLLM_HIP_ARCH=gfx1151
export PAROQUANT_HIP_ARCH=gfx1151
export PYTORCH_ROCM_ARCH=gfx1151
```

Do not rely on gfx1100 defaults.  The 2026-05-16 PARO trial with
`HSA_OVERRIDE_GFX_VERSION=11.0.0` plus gfx1100 JIT targets failed before model
load with `HIP error: device kernel image is invalid`.

Local `mamba run -n therock` can also hide `/usr/bin/c++` and `/usr/bin/as`
from PyTorch JIT builds.  If extension compilation fails with missing host tools,
run through `/usr/bin/env` with an explicit path:

```bash
mamba run -n therock --no-capture-output /usr/bin/env \
  PATH=/home/lhl/mambaforge/envs/therock/bin:/usr/bin:/bin \
  CC=/usr/bin/gcc CXX=/usr/bin/c++ \
  PYTHONPATH=nano-vllm-amd:paroquant \
  python3 <bench.py>
```

When changing HIP sources or arch flags, clear the relevant JIT caches before
interpreting hangs or stale behavior:

```bash
rm -rf ~/.cache/torch_extensions/py*/nanovllm_amd_native_gfx1151_*
rm -rf ~/.cache/nanovllm_amd/torch_extensions/paroquant_kernels_v8*
```

### 7.3 Current PARO/nano-vllm gfx1151 starting preset

The following W7900-derived flag set was correctness-tested on gfx1151 for
`z-lab/Qwen3.5-0.8B-PARO` 512/128 and 4096/128 with graph replay enabled.  It is
a **starting preset**, not a proven optimum:

```bash
# Native gfx1151 code objects.
export NANOVLLM_AMD_HIP_ARCH=gfx1151
export NANOVLLM_HIP_ARCH=gfx1151
export PAROQUANT_HIP_ARCH=gfx1151
export PYTORCH_ROCM_ARCH=gfx1151

# Router / MoE path selection.
export NANOVLLM_AMD_ROUTER_PREFILL_MIN_TOKENS=512
export NANOVLLM_PARO_NATIVE_ROUTER=1
export NANOVLLM_PARO_MOE_STACKED_COMPACT=1
export NANOVLLM_PARO_MOE_STACKED_REPACK_REPLACE=1
export NANOVLLM_PARO_MOE_GROUPED_DEVICE_GATHER=1
export NANOVLLM_PARO_MOE_GROUPED_STACKED_MAX_TOKENS=4096
export NANOVLLM_PARO_MOE_GROUPED_STACKED_WEIGHTED_LANES=1
export NANOVLLM_PARO_MOE_GROUPED_STACKED_SILU_ROTATE_FUSED=1
export NANOVLLM_PARO_MOE_SILU_DOWN_ROTATE_FUSED=1

# W4/W8 dense and selected-expert linear paths.
export NANOVLLM_PARO_GEMV_PACK8_REPLACE=1
export NANOVLLM_PARO_GEMV_PACK8_TRANSPOSE_MIN_OUT_FEATURES=999999
export NANOVLLM_PARO_GEMV_V8=1
export NANOVLLM_PARO_LM_HEAD_W8A16=1
export NANOVLLM_PARO_SHARED_EXPERT_W8A16=1

# Prefill / multi-token WMMA paths.
export NANOVLLM_PARO_WMMA_GEMM=1
export NANOVLLM_PARO_WMMA_COMPACT=1
export NANOVLLM_PARO_WMMA_MIN_TOKENS=64

# Fused attention/linear helpers used by the retained PARO rows.
export NANOVLLM_PARO_FULL_ATTN_FUSED_GATE=1
export NANOVLLM_PARO_LINEAR_ATTN_AB_FUSED=1
export NANOVLLM_PARO_LINEAR_ATTN_QKV_Z_PACK8_FUSED=1
export NANOVLLM_PARO_FULL_ATTN_QK_PACK8_FUSED=1

# Paged attention.
export NANOVLLM_AMD_PAGED_ATTN_GQA_GROUPED_CTX=1
export NANOVLLM_AMD_PAGED_ATTN_MAX_SPLITS=512
```

Retained evidence with this preset:

| Model / shape | Prefill tok/s | Decode tok/s | Peak allocated | Correctness |
| --- | ---: | ---: | ---: | --- |
| `Qwen3.5-0.8B-PARO`, 512/128 | 2451.98 | 145.60 | 1.214 GiB | finite logits + graph replay match |
| `Qwen3.5-0.8B-PARO`, 4096/128 | 2477.52 | 137.72 | 2.329 GiB | finite logits + graph replay match |

### 7.4 nano-vllm-amd knobs that should be retuned on gfx1151

Prioritize retuning in this order.  Use the correctness gates in section 8 before
keeping a value.

| Knob | Current/start | Sweep on gfx1151 | Why it matters |
| --- | ---: | --- | --- |
| `NANOVLLM_PARO_WMMA_MIN_TOKENS` | 64 | 16, 32, 64, 128, 256 | Chooses GEMV/fused selected path vs WMMA for multi-token prefill.  gfx1151 prefill is much weaker than the raw GEMM roof, so the crossover is high priority. |
| `NANOVLLM_PARO_WMMA_MAX_TOKENS` | 1024 default in code | 512, 1024, 2048, `0`/unbounded | Long prefill may want to stay in WMMA longer, but unified-memory bandwidth and attention/KV traffic can change the optimum. |
| `NANOVLLM_PARO_GEMV_V8_THREADS` | 128 | 64, 128, 256 | c=1 decode is memory/occupancy-bound; 64 can reduce reduction overhead, 128/256 can increase outstanding memory work when grid is small. |
| `NANOVLLM_PARO_GEMV_PACK8_THREADS` | 128 | 64, 128 | Pack8 wrappers currently accept 64/128; lower CU count and bandwidth may prefer one wave for some shapes. |
| `NANOVLLM_PARO_GEMV_SELECTED_PACK8_SMALL_K_THREADS` | 64 | 64, 128 | Small-K selected paths have a separate thread override. |
| `NANOVLLM_PARO_GEMV_SELECTED_PACK8_SMALL_K_MAX_IN` | 512 | 0, 512, 1024, 2048 | Controls when the small-K override applies. |
| `NANOVLLM_AMD_ROUTER_THREADS` | 512 default | 64, 128, 256, 512 | Router/top-k can be serial or launch-dominated; do not inherit W7900 defaults blindly. |
| `NANOVLLM_AMD_ROUTER_PREFILL_MIN_TOKENS` | 512 in retained preset, 2048 code default | 256, 512, 1024, 2048 | Determines when prefill switches to smaller router thread counts. |
| `NANOVLLM_AMD_ROUTER_PREFILL_THREADS` | 64 default after threshold | 64, 128, 256 | Prefill router can under-fill or over-synchronize depending on token count. |
| `NANOVLLM_AMD_EXPERT_THREADS` | 64 default | 64, 128, 256 | Dense fallback expert kernels and some unfused paths may prefer different occupancy. |
| `NANOVLLM_PARO_MOE_GROUPED_STACKED_MAX_TOKENS` | 4096 in retained preset | 1024, 2048, 4096, `0`/unbounded | Controls grouped-stacked path retention as token count grows. |
| `NANOVLLM_PARO_MOE_GROUPED_STACKED_GEMV_MAX_TOKENS` | 512 code default | 256, 512, 1024, 2048 | Interaction point between GEMV-style grouped path and WMMA token window. |
| `NANOVLLM_AMD_PAGED_ATTN_MAX_SPLITS` | 512 | 64, 128, 256, 512 | W7900 has 96 CUs; gfx1151 has 40.  Less split-K may reduce overhead while still filling the GPU. |
| `NANOVLLM_AMD_PAGED_ATTN_GQA_GROUPED_CTX` | 1 | 0, 1 | Keep on as baseline, but validate against context length and generated tokens. |
| extension compile flags (`-mcumode`, `-amdgpu-unroll-threshold-local=600`) | W7900-derived | keep first; only change with rocprof evidence | These affect all kernels and can hide regressions.  Profile before changing. |

### 7.5 TileLang gfx1151 evidence to import into our mental model

The recent `/home/lhl/github/lhl/amd-tilelang-tuning` work is useful because it
isolates raw FP16 WMMA GEMM behavior on the same gfx1151 machine.

Relevant TileLang RDNA override from `/home/lhl/github/lhl/tilelang/tilelang/carver/arch/rdna.py`:

```python
"gfx1151": _RDNATuningConfig(
    preferred_warps_per_block=4,     # 128 threads
    pipeline_stage=2,
    reduction_step_by_dtype_bits=((16, 32),),
)
```

Key measured configs and results:

| Shape / path | Best-ish gfx1151 result | Note |
| --- | ---: | --- |
| TileLang `4096^3`, `128x128x64`, stage 2, 128 threads, `trans_b=false` | ~26.9 TFLOP/s | Preferred explicit TileLang candidate. |
| TileLang `4096^3`, `64x256x32`, stage 2, 128 threads, `trans_b=false` | ~26.4-28.7 TFLOP/s depending shape/run | Competitive FullCol candidate. |
| Torch/hipBLASLt `4096^3` | ~29-31 TFLOP/s | Often wins plain GEMM on application layouts. |
| Reference `rocm_wmma_gemm` favorable/prepacked layout | ~31-33 TFLOP/s | Useful ceiling, but not always layout-comparable. |
| Roller/default `A @ B.T` / `trans_b=true` after fixes | ~18-19 TFLOP/s | Candidate generation works, but B-layout path remains weak. |

The latest sampled comparison in that repo (`results/gfx1151-main-tuned-20260515-032046`)
shows `rocm_wmma_best` around 29.8-32.7 TFLOP/s on 4096-class shapes, Torch
around 24.4-29.9 TFLOP/s, and fixed explicit TileLang configs around 20-28.7
TFLOP/s.  This is only ~45-55% of the 59.4 TFLOP/s theoretical FP16/BF16 WMMA
roof, which is a realistic upper-bound sanity check for our current prefill
expectations.

TileLang tuning lessons that transfer to nano-vllm-amd:

- **128-thread / 4-wave blocks are a strong gfx1151 starting point** for tiled
  WMMA GEMM.  The old 256-thread, stage-1 TileLang baseline was slower.
- **Stage-2/double-buffered K loops matter** for raw GEMM.  In nano-vllm's
  selected-expert W4 WMMA kernels, this suggests looking for larger multi-WMMA
  blocks or K-loop staging opportunities before declaring prefill compute-bound.
- **Layout dominates.**  `rocm_wmma_best` uses favorable/prepacked layouts and
  can be much faster than row-major application paths.  Do not compare a
  prepacked microbenchmark directly to a model path that has selected experts,
  dequant, scales, router, or transposed pack8 constraints.
- **`trans_b=true` remains weak in TileLang.**  Keep both B layouts in probes;
  do not assume the Strix Halo winner generalizes to W7900 or vice versa.
- **Non-power-of-2 96x96 TileLang tiles were not worth pursuing.**  The repo's
  non-pow2 experiments found slower or buggy TileLang codegen; the hipBLASLt
  gap is more likely LDS padding, scheduling, and offline-tuned layouts than a
  simple tile-size knob.
- **The speed gap is not a different instruction.**  Torch/hipBLASLt/Tensile,
  TileLang, and rocWMMA all use RDNA WMMA instructions for FP16 GEMM.  Remaining
  gaps are scheduling/layout/codegen issues.

### 7.6 Direct implications for nano-vllm-amd/PARO

- **Start with correctness-proven gfx1151 arch flags and W7900 PARO flags, then
  sweep thresholds.**  Do not fork kernels until a profile says which bucket is
  hot on gfx1151.
- **Prefill is the biggest suspicious gap.**  The retained 0.8B PARO prefill row
  is only 20-22% of W7900, while raw gfx1151 FP16 GEMM can reach roughly
  30 TFLOP/s.  This points to path selection, dequant/layout, attention, launch,
  or occupancy issues before a hard hardware limit.
- **c=1 decode should be tuned like a bandwidth/occupancy problem.**  Thread
  count, grid sufficiency, bytes/token, pack8 layout, and launch replay are more
  important than WMMA peak throughput.
- **For plain dense prefill GEMM, use Torch/hipBLASLt as a baseline.**  If an
  unfused dequant+Torch path beats a custom fused path for a shape, keep the
  faster correct path and record the extra memory cost.
- **Selected-expert WMMA kernels are currently one-wave/16x16-tile oriented.**
  TileLang evidence says high-throughput raw GEMM prefers 128-thread, stage-2
  blocks on gfx1151, but selected experts and compact row grouping make this a
  real redesign.  Treat larger multi-WMMA selected kernels as future work after
  profiling, not as an immediate preset tweak.
- **Paged attention split-K should scale down from W7900 assumptions.**  A split
  count that helps 96 CUs can over-split on 40 CUs.

### 7.7 hipENGINE / raw HIP production-harness notes

`/home/lhl/github/shisa-ai/hipENGINE` is the production harness most likely to
receive stable gfx1151 work.  It differs from `nano-vllm-amd` in ways that
change how tuning should be landed:

- hot-path runtime is torch-free and uses raw HIP/C++ kernels via `ctypes`,
- kernels are keyed by `(backend, layer, quant, variant)` and should not be
  selected by ad-hoc `if backend == ...` branches,
- `PrefillConfig` is typed; retained defaults should not depend on hot-path env
  lookups,
- JIT build artifacts are cached under `~/.cache/hipengine/build`, with build
  keys formed from source bytes, flags, compiler, and compiler version, and
- kernel R&D / profiler iteration still belongs in this `amd-gpu-tuning`
  workspace; hipENGINE should ingest stable, tested raw HIP ports.

Current hipENGINE snapshot from inspection:

| Area | gfx1151 status / implication |
| --- | --- |
| Backend tree | `hipengine/kernels/hip_gfx1151/` exists but is empty except `__init__.py`. |
| Registration | Qwen3.5/PARO generator and model defaults register `hip_gfx1100`, not `hip_gfx1151`. |
| Runtime imports | Qwen3.5 runtime currently imports `hipengine.kernels.hip_gfx1100.*` directly. A gfx1151 backend needs registry/backend plumbing, not copied `if` branches. |
| Build profiles | `hipengine/core/build.py` has `decode`, `prefill`, and `baseline`; decode uses `-mllvm -amdgpu-unroll-threshold-local=600 -mcumode`, prefill uses the unroll flag. |
| Native arch | Build commands do not currently show an explicit `--offload-arch=gfx1151`. Add a target-arch flag/path before treating a row as native gfx1151; because flags are in the cache key, the arch must be part of the planned artifact. |
| Kernel defaults | Raw wrappers expose thread/default knobs directly: pack8 GEMV defaults mostly 128 threads, W8A16 defaults 64, dense GEMV 256, router 512, lm-head 256, combine/silu mixed 128/256. |
| Current perf state | No accepted hipENGINE E2E throughput row yet; current rows are correctness/non-throughput or diagnostic.  W7900 native prefill is now correctness-clean but still far below parent throughput. |

The user's shorthand is directionally right: gfx1151 has roughly **42%** of
W7900's CUs but only about **26-30%** of W7900's external-memory bandwidth,
depending on whether we use measured Strix read bandwidth or the theoretical
LPDDR5X bus:

| Ratio | Value |
| --- | ---: |
| CUs: `40 / 96` | 41.7% |
| FP16/BF16/INT8 matrix compute: `59.4 / 123` | 48.3% |
| Theoretical MBW: `256 / 864` | 29.6% |
| Practical Strix read vs W7900 theoretical: `221 / 864` | 25.6% |
| Theoretical bandwidth per CU: `(256/40) / (864/96)` | 71.1% |
| Practical-read bandwidth per CU vs W7900 theoretical: `(221/40) / (864/96)` | 61.4% |
| Compute-per-byte ratio, theoretical roofs | ~1.63x W7900 |
| Compute-per-byte ratio, practical Strix read vs W7900 theoretical | ~1.89x W7900 |

Implications for hipENGINE/raw HIP tuning:

- **Bytes are even more expensive on gfx1151.**  Compared with W7900, Strix Halo
  has more compute per byte of memory bandwidth.  Prefer spending extra ALU on
  dequant/fusion/recompute if it removes global reads/writes; be skeptical of
  any path that materializes large intermediates to save arithmetic.
- **Fusion has higher upside but a sharper occupancy cliff.**  Fusing away
  global traffic is valuable, but not if VGPR pressure or LDS/barriers reduce
  outstanding memory requests.  For hot decode kernels, `Scratch_Size > 0` is a
  failed hypothesis until proven otherwise, and VGPR counts near/above ~96 need
  profiler attention.
- **Use 40 CUs for grid-sufficiency audits.**  W7900 split counts and CTA grids
  were often designed to cover 96 CUs.  On gfx1151, a split-K or per-output grid
  that just fills 40 CUs may be enough; over-splitting can waste launch/reduce
  work and extra scratch bandwidth.
- **Thread defaults need backend presets.**  hipENGINE exposes raw wrapper
  `threads` parameters, so a gfx1151 preset should live in typed config/backend
  defaults: sweep 64/128 for pack8 GEMV, 64/128/256 for W8A16 and selected
  small-K paths, and 128/256/512 for router/lm-head only with correctness gates.
- **The 64-thread prefill-attention fix is important evidence.**  hipENGINE's
  recent W7900 native-prefill work found the old 256-thread prefill attention
  wrapper was nondeterministic; 64 threads plus correctly-sized shared scratch
  made repeat launches stable.  Keep that as the starting point on gfx1151.
- **Shared transient prefill scratch matters more on unified memory.**  The
  recent hipENGINE 4K prefill unblock reused a session-level prefill workspace
  across layers.  On gfx1151, capacity is large but bandwidth is scarce, so avoid
  per-layer scratch growth and avoid copying scratch-like buffers through global
  memory when a fixed-address workspace can be reused.
- **Graph replay still matters, but it is not a bandwidth fix.**  hipENGINE's
  raw HIP/ctypes path should keep shape-bucketed graph replay for dispatch
  overhead, while profiles should still rank GPU kernel time; memory-bound decode
  will not be solved by capture alone.

Suggested hipENGINE gfx1151 bring-up order:

1. Add explicit native arch selection to the build layer, e.g.
   `--offload-arch=gfx1151`, and make the arch part of the artifact/cache key.
2. Add `hip_gfx1151` backend registration as a peer of `hip_gfx1100`; initially
   reuse the same kernel bodies only as a measured baseline, not as a claim that
   gfx1100 tuning is optimal.
3. Run smoke/build gates first (`smoke_add`, then narrow kernel-family smokes)
   with CPU-reference correctness and profiler traces showing the expected
   kernel names.
4. Port/enable Qwen3.5/PARO generation under `backend="hip_gfx1151"` only after
   the registry path avoids direct `hip_gfx1100` imports in the runtime hot path.
5. Re-run the same PARO 512/128 and 4096/128 benchmark shapes used in this doc,
   then add hipENGINE-specific gfx1151 rows only if they satisfy the benchmark
   artifact/rollup rules in hipENGINE's `docs/BENCHMARK.md`.
6. Start tuning with wrapper/config values (threads, split counts, chunk sizes,
   graph buckets).  Only redesign kernels after a gfx1151 `rocprofv3` trace says
   which family dominates.

### 7.8 Suggested gfx1151 tuning sweep order

1. Re-run the retained PARO 512/128 and 4096/128 rows with only one knob changed
   at a time, checking finite logits and graph replay match.
2. Sweep `NANOVLLM_PARO_WMMA_MIN_TOKENS` and `NANOVLLM_PARO_WMMA_MAX_TOKENS` for
   prefill throughput and peak memory.
3. Sweep `NANOVLLM_PARO_GEMV_*THREADS` on decode-heavy rows and on the standalone
   pack8/GEMV microbenchmarks if available.
4. Sweep paged-attention split count at 4K, 16K, and 32K contexts; do not infer
   long-context behavior from 512-token rows.
5. Run `rocprofv3 --kernel-trace` on the best and worst configs, rank total
   kernel time, then inspect SQ waves/VGPR/scratch for the top kernels.
6. Only after the profile points at a specific WMMA or GEMV kernel, consider
   structural changes such as 128-thread WMMA blocks, stage-2 K staging, LDS
   padding, or layout repacking.

---

## 8. Benchmark protocol for gfx1151 roofline claims

A retained gfx1151 benchmark row should include:

1. exact GPU and ROCm/PyTorch versions,
2. `rocminfo` CU/cache/LDS snapshot or a reference to this doc if unchanged,
3. GTT/GART/TTM state (`rocm-smi --showmeminfo vram/gtt`, TTM pages limit),
4. model, quantization, prompt/context, generation length, batch/concurrency,
5. exact command and env flags, especially `*_HIP_ARCH=gfx1151`,
6. correctness gates (`finite_prefill_logits`, graph replay validation/match,
   generated sample/token equality for comparable configs),
7. `prefill_tok_s`, `decode_tok_s`, `wall_seconds`, allocated-after-load, and
   peak allocated memory,
8. if optimizing kernels: rocprof kernel-time ranking and occupancy/scratch/VGPR
   audit before claiming a root cause.

Minimal local hardware snapshot:

```bash
rocminfo | grep -n -A90 -B10 'Name:                    gfx1151'
rocm-smi --showproductname --showuniqueid --showmeminfo vram \
  --showmeminfo gtt --showclocks --showsclkrange --showuse \
  --showpower --showtemp --json
cat /sys/module/ttm/parameters/pages_limit
```

Example native PARO arch env skeleton:

```bash
export NANOVLLM_AMD_HIP_ARCH=gfx1151
export NANOVLLM_HIP_ARCH=gfx1151
export PAROQUANT_HIP_ARCH=gfx1151
export PYTORCH_ROCM_ARCH=gfx1151
PYTHONPATH=nano-vllm-amd:paroquant \
  mamba run -n therock --no-capture-output \
  python3 scripts/bench_paro_native_engine.py \
    --model-path <model> \
    --prompt-len 512 --decode-len 128 \
    --decode-use-step-graph-replay --output <artifact>.json --json
```

---

## 9. Open questions / measurements to add

- ROCm-native sustained memory bandwidth from a simple HIP/torch kernel on this
  exact machine, not only Vulkan `memtest_vulkan` and `rocm-bandwidth-test`.
- `rocprofv3` counters for the current PARO gfx1151 path: achieved memory bytes,
  wave occupancy, VGPRs, scratch, and hot-kernel time share.
- Direct gfx1151 vs gfx1100 comparison on the same model/commit for larger
  active-weight models where c=1 decode is closer to bandwidth saturation.
- Prefill GEMM microbenchmarks that report achieved BF16/INT8/INT4 WMMA TOP/s
  for Qwen-shaped tiles.
- Clock/power behavior during sustained 5+ minute decode: the sampled
  `rocm-smi` clocks above were idle/light-load, not a sustained-load trace.

---

## References

- `docs/ROOFLINE.md` — W7900/gfx1100 roofline model and project profiling rules.
- `WORKLOG.md` — retained benchmark rows, including the 2026-05-16
  `z-lab/Qwen3.5-0.8B-PARO` gfx1151 run.
- `/home/lhl/strix-halo-testing/README.md` — local Strix Halo testing overview.
- `/home/lhl/strix-halo-testing/hardware-test/README.md` — raw memory bandwidth
  and CPU/GPU transfer notes.
- `/home/lhl/strix-halo-testing/llm-bench/README.md` — llama.cpp Strix Halo
  pp/tg summaries and setup notes.
- <https://strixhalo.wiki/AI/AI_Capabilities_Overview>
- <https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html>
- <https://llvm.org/docs/AMDGPUUsage.html>
- <https://github.com/woct0rdho/rdna35-isa-markdown>
