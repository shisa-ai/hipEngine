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
    q4_selected_dual_isa_stats.py
    memory_waitcnt.py
    q4_selected_dual_real_slice.py
    reduction_sweep.py
    vopd_sweep.py
    hip_dot_path.hip
    hip_geometry_sweep.hip
    hip_memory_waitcnt.hip
    hip_vopd_sweep.hip
    hip_dispatch_floor.py
    vulkan_dot_path.cpp
    vulkan_geometry_sweep.cpp
    vulkan_memory_waitcnt.cpp
    vulkan_q4_selected_dual.cpp
    vulkan_vopd_sweep.cpp
    vulkan_dispatch_floor.py
    vulkan_dispatch_floor.cpp
  kernels/
    vulkan/
      dot_path.comp
      geometry_sweep.comp
      memory_waitcnt.comp
      q4_selected_dual.comp
      q8_1_quantize.comp
      reduction_extra_barrier.comp
      reduction_subgroup.comp
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
artifact records both the requested mode and the code-object wave size. Use
`--hip-fixed-block-index` on HIP rows to compile a fixed-indexing control with
`__launch_bounds__(256)` and `kBlockSize`-based global ID calculation instead
of `blockDim.x`; this isolates the tiny runtime-shape overhead that remains in
the normal HIP source.

## F32 GEMV Geometry Sweep

`runners/geometry_sweep.py` runs a matched math diagnostic on HIP or Vulkan.
Each row uses one workgroup per output row, repeat-shifted strided f32 FMA
accumulation over K, and a shared-memory tree reduction. The repeat-shift keeps
the body loop from collapsing into one identical dot product. HIP uses the
runtime block size; Vulkan uses a `local_size_x` specialization constant. The
harness validates every row against a CPU reference that mirrors the same
per-workgroup reduction order. Use `--hip-workgroup-specialization fixed` on
HIP rows to compile one fixed-workgroup binary per requested workgroup size and
merge the rows into one artifact. This removes the runtime `blockDim.x`
reduction/indexing path and is the paired control for Vulkan's
`local_size_x` specialization.
Use `--hip-wavefront-size 64` with fixed specialization to produce the combined
fixed-wave64 control; the normalized artifact records the requested wave mode
and CPU-oracle status.

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
runs the Vulkan geometry harness under `RADV_DEBUG=shaders,shaderstats` and
parses RADV final shader disassembly plus shaderstats allocation counts.
Estimated physical register spans are retained as cross-checks, but current
RADV rows use official shaderstats VGPR/SGPR/spill/scratch fields when present.
Use `--hip-workgroup-specialization fixed --hip-wavefront-size 64` on HIP rows
when extracting ISA for the combined fixed-wave64 geometry control; this
compiles one code object per requested workgroup.

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
lead survives wave-mode control. Use `--hip-fixed-block-index` on HIP rows to
compile the fixed 256-thread indexing and launch-bounds control.

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

## Q4 Selected-Dual HIP/RADV ISA Comparison

`runners/q4_selected_dual_isa_stats.py` compares the production HIP
Q4_K selected-dual q8_1+dp4a kernel against the retained Vulkan/RADV shaderstats
artifact for the same real slice. It compiles the HIP production source with
`hipcc --save-temps`, parses code-object metadata and `llvm-objdump`
disassembly, then joins the rows with the RADV shaderstats artifact.

Retained gfx1151 command:

```bash
HIPENGINE_HIP_ARCH=gfx1151 \
python3 benchmarks/micro/runners/q4_selected_dual_isa_stats.py \
  --hip-result benchmarks/micro/results/gfx1151/strix-halo/hip-real-q4-selected-dual-q8_1-dp4a.json \
  --vulkan-isa-result benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a-isa-stats.json \
  --build-dir /tmp/hipengine-micro-q4-selected-dual-isa-stats \
  --gfx-arch gfx1151 \
  --hardware-gpu "Radeon 8060S Graphics" \
  --out benchmarks/micro/results/gfx1151/strix-halo/q4-selected-dual-real-slice-isa-comparison.json
```

This is an attribution artifact for the retained Q4_K selected-dual Vulkan win.
It does not create a new timing claim by itself.

## Q4 Selected-Dual Vulkan Setup / Amortization Probe

`runners/q4_selected_dual_real_slice.py` now records Vulkan setup phase timings
from the underlying `vulkan_q4_selected_dual.cpp` harness. The retained
instrumented row reuses the positive local_size=64 Q4_K selected-dual real
slice and separates one-shot backend setup from steady pre-recorded command
buffer replay.

Retained gfx1151 command:

```bash
python3 benchmarks/micro/runners/q4_selected_dual_real_slice.py \
  --build-dir /tmp/hipengine-micro-q4-integration-retained \
  --out benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a-integration.json \
  --local-size 64 \
  --reps 120 \
  --warmup 30 \
  --samples 11
```

This is a bounded setup/amortization probe. It is not a production
`vulkan_radv_gfx11` backend result because it does not exercise hipEngine
registry integration or full inference residency.

## LDS / Barrier / Subgroup Reduction Sweep

`runners/reduction_sweep.py` runs HIP and Vulkan reduction-topology controls
using the geometry harness interface. The retained variants are HIP LDS tree,
HIP LDS tree with one extra barrier per reduction stage, HIP wave-shuffle
reduction, Vulkan LDS tree, Vulkan extra-barrier LDS tree, and Vulkan subgroup
reduction via `GL_KHR_shader_subgroup_arithmetic`.

Retained gfx1151 command:

```bash
HIPENGINE_HIP_ARCH=gfx1151 \
python3 benchmarks/micro/runners/reduction_sweep.py \
  --backend both \
  --environment-json benchmarks/micro/results/gfx1151/strix-halo/environment-fixed-shape-controls.json \
  --environment-ref benchmarks/micro/results/gfx1151/strix-halo/environment-fixed-shape-controls.json \
  --gfx-arch gfx1151 \
  --hardware-gpu "Radeon 8060S Graphics" \
  --build-dir /tmp/hipengine-micro-reduction-sweep \
  --out benchmarks/micro/results/gfx1151/strix-halo/reduction-sweep.json \
  --k-list 512,2048,8192 \
  --rows-list 1 \
  --workgroups 64,256 \
  --body-repeats 128 \
  --reps 20 \
  --warmup 5 \
  --samples 11 \
  --pretty
```

This is a reduction-topology control for the f32 geometry gap. It does not
stand alone as LLVM/RADV compiler attribution.

## Retained Results

| Date | Hardware | Bench | Finding | Artifacts |
| --- | --- | --- | --- | --- |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | Q4_K selected-dual Vulkan setup/amortization probe | Instrumented local_size=64 rerun passes full CPU correctness and keeps steady replay positive at `0.292745 ms` prequantized dot and `0.293117 ms` quantize+dot versus retained HIP `0.34638 ms` and `0.34582 ms`. Standalone backend setup before steady replay is `47.8645 ms`, dominated by `25.3268 ms` synthetic host staging and `17.4106 ms` Vulkan instance/device setup; pipeline creation is only `0.1736 ms`, device upload `3.4389 ms`, descriptor setup `0.0171 ms`, and command recording `0.0639 ms`. If all setup is charged to the retained Q4 quantize+dot delta, breakeven is about `908` calls. Classified bounded setup probe: useful only with persistent pipelines/resident buffers, not a production-backend win by itself. | `results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a-integration.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | LDS/barrier/subgroup reduction sweep | One-row K=512/2048/8192 wg64/wg256 rows all pass CPU correctness. HIP extra-barrier is only `1.002x-1.028x` slower than HIP LDS, HIP wave-shuffle is flat versus HIP LDS (`0.991x-1.005x`), Vulkan extra-barrier is flat versus Vulkan LDS (`0.991x-1.005x`), and Vulkan subgroup is mostly flat to modestly slower than Vulkan LDS (`0.984x-1.132x`). Matched Vulkan LDS remains `8.19x-14.55x` faster than matched HIP LDS, so reduction topology is not the missing f32 geometry switch. Classified `diagnostic_unclassified`. | `results/gfx1151/strix-halo/reduction-sweep.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | Q4_K selected-dual HIP/RADV ISA comparison | Targeted ISA comparison for the one positive Vulkan real-slice row shows the Q4 win is not missing HIP dot4, HIP spills, or RADV VOPD pairing. HIP and RADV both emit 3 dot4 instructions and no scratch/spills for the dot shader. HIP emits wave32, 31 SGPR / 22 VGPR, 564 static instructions, 35 waitcnt-family instructions, and 4 VOPD; RADV emits wave64, official 108 SGPR / 48 VGPR, 526 static instructions, 26 waitcnt-family instructions, and 0 VOPD. Classified slice-specific `real_slice_probe`; remaining Q4 follow-up is narrower scheduling/source/reduction work only if it changes backend or HIP implementation priority. | `results/gfx1151/strix-halo/q4-selected-dual-real-slice-isa-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | Vulkan Q4_K selected-dual real slice | Matched production-shaped Vulkan q8_1+dp4a probe passes full CPU correctness and does transfer a real Vulkan win. Best Vulkan local_size=64 is `0.29607 ms` prequantized dot and `0.29238 ms` quantize+dot versus retained HIP `0.34638 ms` and `0.34582 ms`, so Vulkan is `1.17x` faster on dot and `1.18x` faster combined. RADV final dot shader has 3 dot4 instructions, subgroup 64, 0 VOPD, official 48 VGPR / 108 SGPR, no scratch/spills, 26 waitcnt-family instructions, and 22 buffer loads. Classified slice-specific `real_slice_probe`, not broad `compiler_aco`. | `results/gfx1151/strix-halo/q4-selected-dual-real-slice-hip-vulkan-comparison.json`, `results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a.json`, `results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a-ls128.json`, `results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a-ls256.json`, `results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a-isa-stats.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | Vulkan Q6_K X8 selected-down real slice | Matched production-shaped Vulkan q8_1+dp4a probe passes full CPU correctness but does not transfer the synthetic Vulkan dot-path win. Best Vulkan local_size=64 is `0.03076 ms` prequantized dot and `0.03217 ms` quantize+dot versus retained HIP `0.01665 ms` and `0.01925 ms`, so Vulkan is `1.85x` slower on dot and `1.67x` slower combined. RADV final dot shader has 9 dot4 instructions, subgroup 64, 0 VOPD, official 48 VGPR / 108 SGPR, no scratch/spills, 89 waitcnt-family instructions, and 82 buffer loads. Classified `real_slice_probe`; do not pursue this q6 selected-down Vulkan port as implemented. | `results/gfx1151/strix-halo/q6-x8-real-slice-hip-vulkan-comparison.json`, `results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a.json`, `results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a-ls128.json`, `results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a-ls256.json`, `results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a-isa-stats.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S | HIP q8_1 real-slice layout controls | HIP production-layout q8_1 controls are positive. Q4_K selected-dual gate/up q8_1 quantize+dp4a is `2.77x` faster than raw selected-dual with top-1 `1.0`; Q6_K selected-down X8 q8_1 quantize+dp4a is `1.68x` faster than production T16 float with top-1 `1.0`. q8_1 quantization is only `0.0025-0.0027 ms`. Classified `layout_quant`; this is HIP-only evidence, not a matched Vulkan real-slice result. | `results/gfx1151/strix-halo/hip-real-q4-selected-dual-q8_1-dp4a.json`, `results/gfx1151/strix-halo/hip-real-q6-selected-down-x8-q8_1-dp4a.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | HIP fixed-wave64 geometry controls | Forcing HIP wave64 while also specializing fixed workgroup sizes does not close the f32 geometry gap. HIP fixed-wave64 is `1.13x-1.23x` slower than fixed wave32 on best-native rows, and Vulkan remains `6.31x-16.18x` faster than HIP fixed-wave64. HIP fixed-wave64 ISA for K=2048 rows=1 wg64/wg256 reports wave64, 20 SGPR, 11 VGPR, no scratch/spills, 0 VOPD, and 20/24 waitcnt-family instructions. Classified `diagnostic_unclassified`; wave mode is not the missing f32 geometry switch. | `results/gfx1151/strix-halo/hip-geometry-sweep-fixed-workgroup-wave64.json`, `results/gfx1151/strix-halo/geometry-sweep-fixed-workgroup-wave64-comparison.json`, `results/gfx1151/strix-halo/geometry-sweep-fixed-wave64-delta.json`, `results/gfx1151/strix-halo/hip-geometry-isa-stats-fixed-wave64.json`, `results/gfx1151/strix-halo/geometry-isa-stats-fixed-wave64-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | HIP fixed-shape controls | Fixed-shape controls do not close the retained gaps. Dot fixed-block indexing is `0.993x-1.000x` versus same-commit runtime HIP and Vulkan remains `3.31x-3.43x` faster. Memory fixed-block indexing is mixed (`0.906x-1.290x` fixed/runtime) and Vulkan remains faster on every row (`1.04x-2.36x`). Fixed-workgroup geometry improves some HIP wg256 rows by up to `6.3%`, but Vulkan still leads best-native geometry by `5.56x-14.03x`. Classified `diagnostic_unclassified`; runtime `blockDim`/specialization is not the missing switch. | `results/gfx1151/strix-halo/dot-path-fixed-block-comparison.json`, `results/gfx1151/strix-halo/memory-waitcnt-fixed-block-comparison.json`, `results/gfx1151/strix-halo/geometry-sweep-fixed-workgroup-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | HIP wave64 controls | Forcing HIP wave64 does not close the retained dot or memory gaps. Dot-path wave64 is `1.007x-1.061x` slower than HIP wave32 and still trails same-commit Vulkan by `2.63x-3.55x`. Memory/waitcnt wave64 is mixed, leaves Vulkan faster on most rows, and regresses gather `6.349x` versus HIP wave32. Classified `diagnostic_unclassified`; do not promote broad HIP wave64 routing from this evidence. | `results/gfx1151/strix-halo/dot-path-wave64-comparison.json`, `results/gfx1151/strix-halo/memory-waitcnt-wave64-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | packed dot path | Packed q8 signed, q4 unsigned-byte by signed-q8, q6 zero-corrected, and scalar q4 rows all pass the exact sampled CPU oracle. HIP and RADV both emit final dot4 instructions in q8/q4/q6 rows, and HIP reports no scratch/spills. Vulkan remains `3.28x-3.42x` faster, including scalar dequant, so the retained gap is not a missed-HIP-dot4 story and remains `diagnostic_unclassified`. | `results/gfx1151/strix-halo/dot-path-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | memory/waitcnt sweep | Device-memory rows all pass the sampled CPU oracle. Vulkan is `1.02x-2.25x` faster across retained rows, while gather is essentially tied (`1.02x`). HIP reports no scratch/spills; RADV shaderstats reports official 12/24/48 VGPR buckets, 108 SGPR, and no scratch/spills. HIP wave64 and fixed-block controls do not close the gap, so this remains strong memory-side evidence but not a clean `compiler_aco` proof until a memory-bound real slice transfers. | `results/gfx1151/strix-halo/memory-waitcnt-comparison.json`, `results/gfx1151/strix-halo/memory-waitcnt-wave64-comparison.json`, `results/gfx1151/strix-halo/memory-waitcnt-fixed-block-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | VOPD scheduling sweep | Pure VALU rows all pass the sampled CPU oracle. HIP emits VOPD in every retained row while RADV final disassembly emits 0 VOPD in every row. Vulkan is modestly faster only on independent-8 (`1.05x`), mixed int+float (`1.08x`), and dequant-like (`1.04x`); HIP is faster on independent-2/4 and dependent-4. Classified `diagnostic_unclassified`; this is a negative result for the "ACO wins through VOPD" hypothesis. | `results/gfx1151/strix-halo/vopd-sweep-comparison.json` |
| 2026-07-08 | gfx1151 / Radeon 8060S / RADV Mesa 26.1.2 | f32 geometry ISA/stat extraction | K=2048 rows=1 wg64/wg256 extraction passed correctness references. HIP reports 18 SGPR, 11 VGPR, no scratch/spills, wave32, and 2 VOPD instructions; RADV shaderstats reports 108 SGPR, 12 VGPR, no scratch/spills, wave64, and 0 VOPD. The geometry gap is not a missed-HIP-VOPD, HIP-spill, or missing-RADV-allocation-data story; still classified `diagnostic_unclassified`. | `results/gfx1151/strix-halo/geometry-isa-stats-comparison.json` |
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
