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
    hip_dispatch_floor.py
    vulkan_dispatch_floor.py
    compare_results.py
  kernels/
    hip/
    vulkan/
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
