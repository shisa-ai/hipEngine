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
    hip_dispatch_floor.py
    vulkan_dispatch_floor.py
    vulkan_dispatch_floor.cpp
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

## Retained Results

| Date | Hardware | Bench | Finding | Artifacts |
| --- | --- | --- | --- | --- |
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

If a row cannot be classified, keep it diagnostic and do not use it to justify
LLVM work, kernel rewrites, or a Vulkan backend.
