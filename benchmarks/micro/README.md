# hipEngine Microbenchmarks

This directory is the home for controlled HIP vs Vulkan attribution
microbenchmarks. The goal is not to produce a single "Vulkan is faster" number.
The goal is to classify each delta into a cause that can drive engineering work:
compiler codegen, workgroup geometry, wave/subgroup mode, dispatch/runtime
overhead, layout/quantization, or fusion topology.

The benchmark plan lives in `docs/HIP-vs-VULKAN.md`. This tree provides the
artifacts and utilities used to execute that plan.

## Directory Layout

```text
benchmarks/micro/
  README.md
  collect_env.py
  runners/
    dot_path.py
    geometry_sweep.py
    isa_stats.py
    memory_waitcnt.py
    vopd_sweep.py
    hip_dot_path.hip
    hip_geometry_sweep.hip
    hip_memory_waitcnt.hip
    hip_vopd_sweep.hip
    hip_dispatch_floor.py
    vulkan_dot_path.cpp
    vulkan_geometry_sweep.cpp
    vulkan_memory_waitcnt.cpp
    vulkan_vopd_sweep.cpp
    vulkan_dispatch_floor.py
    vulkan_dispatch_floor.cpp
  kernels/
    vulkan/
      dot_path.comp
      geometry_sweep.comp
      memory_waitcnt.comp
      vopd_sweep.comp
  schemas/
    environment.schema.json
    result.schema.json
  results/
    .gitkeep
```

Future benchmark code should keep source and retained artifacts under this
directory unless it needs shared hipEngine runtime code.

Suggested future layout:

```text
benchmarks/micro/
  runners/
    compare_results.py
  kernels/
    vulkan/
      dispatch_floor.comp
  results/
    gfx1100/
      w7900/
      7900xtx/
    gfx1151/
      strix_halo/
```

## Result Rules

Each retained result must include:

- exact command and working directory;
- git commit, branch, and dirty status;
- hardware identity: GPU name, gfx arch, driver/runtime versions where
  available;
- OS/kernel and Python version;
- HIP/ROCm compiler/runtime versions for HIP rows;
- Vulkan loader, device, driver, Mesa/RADV/ACO information for Vulkan rows;
- benchmark shape: backend, algorithm, K/N/rows/workgroup/wave or subgroup,
  warmup and measured iterations;
- correctness evidence against CPU or cross-backend oracle;
- timing distribution, not just one value;
- ISA/stat evidence when available: VGPR, SGPR, scratch, LDS, wave/subgroup,
  `v_dot4_i32_iu8`, VOPD, and waitcnt counts.

Use `schemas/result.schema.json` for result artifacts and
`schemas/environment.schema.json` for environment snapshots.

## Environment Capture

Capture the environment before running a microbench:

```bash
python3 benchmarks/micro/collect_env.py \
  --out /tmp/hipengine-micro-env.json \
  --pretty
```

For tests or machines without ROCm/Vulkan tools:

```bash
python3 benchmarks/micro/collect_env.py --skip-device-probes --pretty
```

The collector is dependency-free and intentionally tolerant: missing commands
are recorded as unavailable instead of failing the run.

## HIP Dispatch/Grid Floor

The first runner wraps the existing `scripts/graph_node_microbench.py` HIP
diagnostic and normalizes it into `schemas/result.schema.json`:

```bash
HIPENGINE_HIP_ARCH=gfx1100 \
python3 benchmarks/micro/runners/hip_dispatch_floor.py \
  --counts 1,50,200,941 \
  --kernels tiny,wide \
  --grid-sweep 1,128,1024,8192 \
  --reps 50 \
  --warmup 10 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/dispatch-floor.json
```

For compact retained artifacts, capture the environment once and reference it:

```bash
python3 benchmarks/micro/collect_env.py \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/environment.json

HIPENGINE_HIP_ARCH=gfx1100 \
python3 benchmarks/micro/runners/hip_dispatch_floor.py \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment.json \
  --counts 1,50,200,941 \
  --kernels tiny,wide \
  --grid-sweep 1,128,1024,8192 \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/dispatch-floor.json
```

Use `--legacy-input <path>` to normalize an existing
`scripts/graph_node_microbench.py --json` artifact without rerunning HIP.

## Vulkan Dispatch/Grid Floor

The Vulkan runner builds a tiny storage-buffer compute shader and a standalone
`libvulkan` C++ harness, records command buffers outside the timed region, and
normalizes submit+fence timing into the same result schema:

```bash
python3 benchmarks/micro/runners/vulkan_dispatch_floor.py \
  --counts 1,50,200,941 \
  --grid-sweep 1,128,1024,8192 \
  --reps 50 \
  --warmup 10 \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/vulkan-dispatch-floor.json
```

For paired retained rows, use the same environment artifact as the HIP run:

```bash
python3 benchmarks/micro/runners/vulkan_dispatch_floor.py \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment.json \
  --counts 1,50,200,941 \
  --grid-sweep 1,128,1024,8192 \
  --gfx-arch gfx1100 \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/vulkan-dispatch-floor.json
```

This is a dispatch/runtime diagnostic only. It does not establish RADV/ACO
shader codegen quality; it tells us how much of a HIP vs Vulkan delta can be
explained before any real math kernel runs.

## Packed Dot Path

`runners/dot_path.py` runs matched packed-int dot diagnostics on HIP and Vulkan.
The retained variants cover q8 signed dot, q4 unsigned-byte by signed-q8 dot,
q6 zero-point correction (`dot_u - 32 * q8_sum`), and a scalar q4 dequant
baseline. HIP uses the same `__builtin_amdgcn_sudot4` idiom as the shipped GGUF
diagnostic kernels; Vulkan requires `VK_KHR_shader_integer_dot_product` and
uses `dotPacked4x8EXT`.

Example paired run:

```bash
python3 benchmarks/micro/collect_env.py \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/environment-dot-path.json

HIPENGINE_HIP_ARCH=gfx1100 \
python3 benchmarks/micro/runners/dot_path.py \
  --backend hip \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment-dot-path.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment-dot-path.json \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --variants q8_signed:16,q4_unsigned:16,q6_zero:16,scalar_dequant:16 \
  --n 32768 \
  --body-iters 128 \
  --reps 20 \
  --warmup 5 \
  --samples 7 \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/hip-dot-path.json

python3 benchmarks/micro/runners/dot_path.py \
  --backend vulkan \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment-dot-path.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment-dot-path.json \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --variants q8_signed:16,q4_unsigned:16,q6_zero:16,scalar_dequant:16 \
  --n 32768 \
  --body-iters 128 \
  --reps 20 \
  --warmup 5 \
  --samples 7 \
  --debug-n 1024 \
  --debug-body-iters 8 \
  --quiet-shader-dump \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/vulkan-dot-path.json

python3 benchmarks/micro/runners/dot_path.py \
  --compare benchmarks/micro/results/gfx1100/w7900/hip-dot-path.json \
            benchmarks/micro/results/gfx1100/w7900/vulkan-dot-path.json \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/dot-path-comparison.json
```

The runner records CPU-oracle correctness, timing, HIP code-object metadata,
RADV final shader disassembly stats, final dot4 counts, waitcnt/load counts,
wave/subgroup size, HIP scratch/spill evidence, and SPIR-V `OpSDot`/`OpSUDot`
counts for the Vulkan rows. Use `--hip-wavefront-size 64` or
`--hip-wavefront-size 32` on HIP rows to run wave-mode controls; the normalized
artifact records both the requested mode and the code-object wave size.

## F32 GEMV Geometry Sweep

`runners/geometry_sweep.py` runs a matched math diagnostic on HIP or Vulkan.
Each row uses one workgroup per output row, repeat-shifted strided f32 FMA
accumulation over K, and a shared-memory tree reduction. The repeat-shift keeps
the body loop from collapsing into one identical dot product. HIP uses the
runtime block size; Vulkan uses a `local_size_x` specialization constant. The
harness validates every row against a CPU reference that mirrors the same
per-workgroup reduction order.

Example paired run:

```bash
python3 benchmarks/micro/collect_env.py \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/environment.json

HIPENGINE_HIP_ARCH=gfx1100 \
python3 benchmarks/micro/runners/geometry_sweep.py \
  --backend hip \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment.json \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --k-list 512,2048,8192 \
  --rows-list 1,4,8 \
  --workgroups 32,64,128,256 \
  --body-repeats 128 \
  --reps 20 \
  --warmup 5 \
  --samples 11 \
  --raw-json benchmarks/micro/results/gfx1100/w7900/hip-geometry-sweep-raw.json \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/hip-geometry-sweep.json

python3 benchmarks/micro/runners/geometry_sweep.py \
  --backend vulkan \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment.json \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --k-list 512,2048,8192 \
  --rows-list 1,4,8 \
  --workgroups 32,64,128,256 \
  --body-repeats 128 \
  --reps 20 \
  --warmup 5 \
  --samples 11 \
  --raw-json benchmarks/micro/results/gfx1100/w7900/vulkan-geometry-sweep-raw.json \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/vulkan-geometry-sweep.json

python3 benchmarks/micro/runners/geometry_sweep.py \
  --compare benchmarks/micro/results/gfx1100/w7900/hip-geometry-sweep.json \
            benchmarks/micro/results/gfx1100/w7900/vulkan-geometry-sweep.json \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/geometry-sweep-comparison.json
```

This family can classify workgroup-shape effects. It is not sufficient
`compiler_aco` evidence until paired ISA/stat extraction shows register,
scratch, waitcnt, VOPD, or instruction-count differences at identical shape.

## F32 Geometry ISA/Stat Extraction

`runners/isa_stats.py` extracts compiler/output statistics for the geometry
kernel without making a new timing claim. HIP extraction uses `hipcc
--save-temps`, `llvm-readobj --notes`, and `llvm-objdump`. Vulkan extraction
runs the Vulkan geometry harness under `RADV_DEBUG=shaders` and parses RADV
final shader disassembly. RADV does not always expose official VGPR/SGPR
allocation counts; when it does not, the artifact records estimated physical
register spans from the final disassembly instead of allocation-count claims.

Example paired run:

```bash
python3 benchmarks/micro/collect_env.py \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/environment-isa-stats.json

HIPENGINE_HIP_ARCH=gfx1100 \
python3 benchmarks/micro/runners/isa_stats.py \
  --backend hip \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment-isa-stats.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment-isa-stats.json \
  --geometry-result benchmarks/micro/results/gfx1100/w7900/hip-geometry-sweep.json \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --k 2048 \
  --rows 1 \
  --workgroups 64,256 \
  --body-repeats 128 \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/hip-geometry-isa-stats.json

python3 benchmarks/micro/runners/isa_stats.py \
  --backend vulkan \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment-isa-stats.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment-isa-stats.json \
  --geometry-result benchmarks/micro/results/gfx1100/w7900/vulkan-geometry-sweep.json \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --k 2048 \
  --rows 1 \
  --workgroups 64,256 \
  --body-repeats 128 \
  --reps 1 \
  --warmup 0 \
  --samples 1 \
  --quiet-shader-dump \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/vulkan-geometry-isa-stats.json

python3 benchmarks/micro/runners/isa_stats.py \
  --compare benchmarks/micro/results/gfx1100/w7900/hip-geometry-isa-stats.json \
            benchmarks/micro/results/gfx1100/w7900/vulkan-geometry-isa-stats.json \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/geometry-isa-stats-comparison.json
```

## Memory / Waitcnt Sweep

`runners/memory_waitcnt.py` runs paired device-memory load diagnostics with
ISA/stat extraction. The retained variants should cover coalesced vector-width
loads, strided loads, gather-ID addressing, and load/compute interleave. Vulkan
uses device-local storage buffers with staging copies so the timed region is not
just a host-visible-buffer diagnostic. Use `--hip-wavefront-size 64` or
`--hip-wavefront-size 32` on HIP rows to test whether the memory-side Vulkan
lead survives wave-mode control.

Example paired run:

```bash
python3 benchmarks/micro/collect_env.py \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/environment-memory-waitcnt.json

HIPENGINE_HIP_ARCH=gfx1100 \
python3 benchmarks/micro/runners/memory_waitcnt.py \
  --backend hip \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment-memory-waitcnt.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment-memory-waitcnt.json \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --n 32768 \
  --body-iters 128 \
  --reps 20 \
  --warmup 5 \
  --samples 7 \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/hip-memory-waitcnt.json

python3 benchmarks/micro/runners/memory_waitcnt.py \
  --backend vulkan \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment-memory-waitcnt.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment-memory-waitcnt.json \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --n 32768 \
  --body-iters 128 \
  --reps 20 \
  --warmup 5 \
  --samples 7 \
  --debug-n 1024 \
  --debug-body-iters 8 \
  --quiet-shader-dump \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/vulkan-memory-waitcnt.json

python3 benchmarks/micro/runners/memory_waitcnt.py \
  --compare benchmarks/micro/results/gfx1100/w7900/hip-memory-waitcnt.json \
            benchmarks/micro/results/gfx1100/w7900/vulkan-memory-waitcnt.json \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/memory-waitcnt-comparison.json
```

## VOPD Scheduling Sweep

`runners/vopd_sweep.py` runs paired pure-VALU diagnostics for VOPD and VALU
scheduling. Each variant is compiled separately with mode/accumulator macros so
the ISA evidence corresponds to exactly one kernel body. The default variants
cover independent f32 FMA chains, dependent f32 FMA chains, mixed int+float,
and dequant-like shift/mask/cvt/FMA chains.

Example paired run:

```bash
python3 benchmarks/micro/collect_env.py \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/environment-vopd-sweep.json

HIPENGINE_HIP_ARCH=gfx1100 \
python3 benchmarks/micro/runners/vopd_sweep.py \
  --backend hip \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment-vopd-sweep.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment-vopd-sweep.json \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --n 65536 \
  --body-iters 2048 \
  --reps 20 \
  --warmup 5 \
  --samples 7 \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/hip-vopd-sweep.json

python3 benchmarks/micro/runners/vopd_sweep.py \
  --backend vulkan \
  --environment-json benchmarks/micro/results/gfx1100/w7900/environment-vopd-sweep.json \
  --environment-ref benchmarks/micro/results/gfx1100/w7900/environment-vopd-sweep.json \
  --gfx-arch gfx1100 \
  --hardware-gpu "AMD Radeon Pro W7900" \
  --n 65536 \
  --body-iters 2048 \
  --reps 20 \
  --warmup 5 \
  --samples 7 \
  --debug-n 1024 \
  --debug-body-iters 64 \
  --quiet-shader-dump \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/vulkan-vopd-sweep.json

python3 benchmarks/micro/runners/vopd_sweep.py \
  --compare benchmarks/micro/results/gfx1100/w7900/hip-vopd-sweep.json \
            benchmarks/micro/results/gfx1100/w7900/vulkan-vopd-sweep.json \
  --pretty \
  --out benchmarks/micro/results/gfx1100/w7900/vopd-sweep-comparison.json
```

## Retained Results

| Date | Hardware | Bench | Finding | Artifacts |
| --- | --- | --- | --- | --- |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | packed dot path | Packed q8 signed, q4 unsigned-byte by signed-q8, q6 zero-corrected, and scalar q4 rows all pass the exact sampled CPU oracle. HIP and RADV both emit final dot4 instructions in q8/q4/q6 rows, and HIP reports no scratch/spills. Vulkan remains `3.29x-3.43x` faster, including scalar dequant, so the retained gap is not a missed-HIP-dot4 story and remains `diagnostic_unclassified`. | `results/gfx1151/strix-halo/dot-path-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | memory/waitcnt sweep | Device-memory rows all pass the sampled CPU oracle. Vulkan is `1.30x-2.35x` faster on most coalesced, strided, and interleave rows, while gather is essentially tied (`1.02x`). HIP reports wave32 and no scratch/spills; RADV final shaders are wave64 with only estimated register spans. Classified `diagnostic_unclassified`; strong memory-side evidence, but not yet a clean `compiler_aco` proof until wave/specialization controls land. | `results/gfx1151/strix-halo/memory-waitcnt-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | VOPD scheduling sweep | Pure VALU rows all pass the sampled CPU oracle. HIP emits VOPD in every retained row while RADV final disassembly emits 0 VOPD in every row. Vulkan is modestly faster only on independent-8 (`1.05x`), mixed int+float (`1.08x`), and dequant-like (`1.04x`); HIP is faster on independent-2/4 and dependent-4. Classified `diagnostic_unclassified`; this is a negative result for the "ACO wins through VOPD" hypothesis. | `results/gfx1151/strix-halo/vopd-sweep-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | f32 geometry ISA/stat extraction | K=2048 rows=1 wg64/wg256 extraction passed correctness references. HIP reports 18 SGPR, 11 VGPR, no scratch/spills, wave32, and 2 VOPD instructions; RADV final disassembly has no VOPD, wave64, and no official VGPR/SGPR allocation counts exposed. The geometry gap is not a missed-HIP-VOPD or HIP-spill story; still classified `diagnostic_unclassified`. | `results/gfx1151/strix-halo/geometry-isa-stats-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | f32 GEMV geometry sweep | Repeat-shifted matched f32 GEMV/reduction rows all pass the CPU oracle. HIP and Vulkan both prefer wg256, so workgroup shape alone does not explain the gap; Vulkan remains `5.79x-14.03x` faster on best-native rows. Classified `diagnostic_unclassified`; paired ISA extraction rules out a simple missed-HIP-VOPD explanation but does not yet identify the primary cause. | `results/gfx1151/strix-halo/geometry-sweep-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | dispatch/grid floor | Vulkan command-buffer replay is much cheaper than HIP direct/graph for one-block launch-heavy bursts (`0.043621 us` vs HIP tiny direct `2.0087 us` and HIP graph `1.8069 us` at N=941), but the gap narrows to about `1.10x` at 8192 blocks. Classified `runtime_dispatch`, not `compiler_aco`. | `results/gfx1151/strix-halo/dispatch-floor-comparison.json` |

## Classification

Every retained benchmark should choose one primary classification:

| Classification | Meaning |
| --- | --- |
| `compiler_aco` | Same algorithm/layout/geometry, Vulkan faster with better ISA stats |
| `geometry` | HIP closes the gap after matching Vulkan's workgroup/subgroup shape |
| `wave_mode` | HIP wave64 or subgroup-size control materially changes the result |
| `runtime_dispatch` | No-op/grid/command rows explain the gap |
| `layout_quant` | Dot/layout/quantization dominates compiler choice |
| `fusion_topology` | Per-op kernels match, but fused Vulkan topology wins |
| `not_reproducible` | The old difference disappears under the controlled harness |
| `diagnostic_unclassified` | Gap remains but the retained evidence is insufficient for one primary cause |

If a row cannot be classified, keep it diagnostic and do not use it to justify
LLVM work, kernel rewrites, or a Vulkan backend.
