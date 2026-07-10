# hipEngine HIP/Vulkan Microbenchmarks

This directory contains controlled HIP versus Vulkan attribution benchmarks.
The goal is to identify actionable causes such as dispatch overhead, workgroup
geometry, memory scheduling, quantization layout, or compiler code generation.
It is not a source for an unconditional "Vulkan is faster" or "HIP is faster"
claim.

## Timing-Contract v2 Reset

Current timing evidence must use `timing_contract.py` and
`schemas/result.schema.json` with `schema_version=2`. The retained timing table
and command examples that previously appeared in this file were produced before
this contract and have been removed from the current dashboard. No pre-v2 timing
ratio is current evidence.

The two timing modes answer different questions:

| Mode | Work contract | Intended question |
| --- | --- | --- |
| `serial_latency` | Logical operations share or chain state. HIP uses stream order; Vulkan inserts compute-to-compute dependencies between repetitions. | How long does required ordered work take? |
| `independent_throughput` | Logical operations write disjoint output slices and have no cross-operation dependency. A multi-stage operation still keeps its required internal stage dependency. | What throughput is available when independent work is submitted efficiently? |

Do not compare one backend's `serial_latency` row with the other backend's
`independent_throughput` row. A cross-backend ratio requires the same timing
mode, logical iteration count, dispatches per iteration, work dependency, output
partitioning, and shape.

Every timed row has two controls:

- `single`: one logical operation, used to expose the one-operation floor.
- `burst`: the configured number of logical operations, used for steady replay
  or queue/stream throughput. The artifact records both whole-sequence and
  per-iteration distributions.

Every control records two clocks:

- `gpu_elapsed`: HIP events or Vulkan device timestamps. This is the primary
  cross-backend kernel/sequence comparison domain. Multi-queue Vulkan spans use
  an enabled calibrated-timestamps extension so timestamps from different queues
  can be placed on one device timeline.
- `host_wall`: `steady_clock` around the recorded submit/completion contract.
  A host-wall ratio is emitted only when both rows have the same normalized
  submission class, whether command recording is timed, whether submission is
  timed, and whether completion is timed. HIP direct or multi-stream submission
  is therefore not ratioed against Vulkan pre-recorded command-buffer replay.

`independent_throughput` must validate every disjoint output from the timed
burst. `serial_latency` may validate the chained final state. A single-dispatch
oracle alone is not sufficient for either mode.

## Core Files

| Path | Purpose |
| --- | --- |
| `timing_contract.py` | Shared mode, control, metric, correctness, and comparison gates |
| `comparison_claim.py` | Shared source, device, raw-identity, and matrix claim gate for joint wrappers |
| `runners/micro_timing_hip.hpp` | HIP event and multi-stream timing helpers |
| `runners/micro_timing_vulkan.hpp` | Vulkan timestamp, barrier, calibrated multi-queue timing helpers |
| `collect_env.py` | Dependency-free environment and device provenance capture |
| `schemas/result.schema.json` | v1 legacy plus v2 result/comparison artifact schema |
| `schemas/environment.schema.json` | Environment artifact schema |
| `results/` | Retained compact artifacts, separated by architecture and device |

## Paired Runner Inventory

All timing runners below support both `serial_latency` and
`independent_throughput`. "Separate" means run HIP and Vulkan artifacts first,
then invoke the same Python wrapper with `--compare`. "Joint" means one wrapper
runs both backends and emits the comparison artifact.

| Area | Wrapper(s) | Interface | Coverage |
| --- | --- | --- | --- |
| Dispatch/grid floor | `hip_dispatch_floor.py`, `vulkan_dispatch_floor.py` | Separate; compare through `hip_dispatch_floor.py --compare` | Tiny launch count and grid-size controls |
| F32 geometry | `geometry_sweep.py` | Separate | Matched FMA/reduction shapes and workgroups |
| Reduction variants | `reduction_sweep.py` | Joint, `--backend both` | LDS tree, extra barrier, subgroup/wave, multi-accumulator variants |
| Memory/waitcnt | `memory_waitcnt.py` | Separate | Coalesced, strided, gather, and load/compute interleave |
| Packed dot | `dot_path.py` | Separate | Q8 signed, Q4 unsigned/signed, Q6 zero correction, scalar dequant |
| VOPD/VALU | `vopd_sweep.py` | Separate | Independent/dependent FMA, mixed int/float, dequant-like work |
| Sampler | `sampler_argmax.py` | Separate | Deterministic top-1 and top-k argmax |
| Two-stage reduction | `two_stage_reduction.py` | Joint, `--backend both` | Partial reduction plus final reduction with an internal dependency |
| Q4 selected-dual | `q4_selected_dual_real_slice.py` | Separate | Production-shaped Q4_K selected-dual q8_1 plus dp4a |
| Q6 X8 selected-down | `q6_x8_real_slice.py` | Separate | Production-shaped Q6_K X8 selected-down q8_1 plus dp4a |
| Dense Q8_0 | `q8_0_dense_real_slice.py` | Separate | Raw GGUF Q8_0 dense q8_1 plus dp4a, including row tiles |

Multi-stage independent-throughput paths use matched HIP stream lanes and
Vulkan compute-queue lanes where a single queue would introduce an unintended
cross-operation dependency. Single-stage Vulkan paths may use independent,
disjoint dispatches in one pre-recorded command buffer.

Joint wrappers derive backend device identity from the raw HIP and Vulkan
harness outputs. Their one combined source hash covers both backend sources;
any backend-specific raw hashes are retained separately when available.

Packed-dot, memory/waitcnt, and VOPD timed-sequence checks cover the first 64
outputs from every logical repetition. That is a repeated sampled oracle, not
a full-output equality claim. Geometry, sampler, and production-slice runners
record their own stronger per-row oracle coverage in the artifact.

## ISA-Only Tools

These tools report static compiler/output evidence. They do not create or
repair a timing claim and have no `--timing-mode`:

| Tool | Evidence |
| --- | --- |
| `isa_stats.py` | HIP and RADV geometry code-object/disassembly statistics |
| `q4_selected_dual_isa_stats.py` | Q4 selected-dual HIP/RADV static comparison |
| `q6_x8_isa_stats.py` | Q6 X8 selected-down HIP/RADV static comparison |

Static fields such as wave size, VGPR/SGPR allocation, scratch, spills,
instruction counts, dot4, VOPD, and waitcnt counts may support an attribution
only when tied to the exact timed source/build. They must not be presented as a
timing ratio.

## Blocked Comparison

`q6_lm_head_rowtile_probe.py` remains available for per-backend diagnostics,
but its HIP BF16 Q6_K T16 rowtile path and Vulkan Q6_K X8 q8_1+dp4a path are not
the same algorithm/layout contract. It is not a valid HIP/Vulkan latency or
throughput comparison. Keep its comparison list empty or otherwise explicitly
blocked until both backends execute matched math and data movement.

## Common gfx1151 Setup

The templates below write to `/tmp`; they do not create retained claims. Run
from the repository root on the gfx1151 system.

```bash
export HIPENGINE_HIP_ARCH=gfx1151
export GPU_NAME="Radeon 8060S Graphics"
export MICRO_ROOT=/tmp/hipengine-micro-v2-gfx1151
export MICRO_ENV="$MICRO_ROOT/environment.json"

mkdir -p "$MICRO_ROOT"
python3 benchmarks/micro/collect_env.py \
  --out "$MICRO_ENV" \
  --pretty
```

For a retained run, replace `/tmp` paths with the appropriate
`results/gfx1151/<device>/` paths only after correctness, schema, source, and
comparison gates pass.

## Current gfx1151 Command Templates

### Dispatch

The dispatch pair uses separate HIP and Vulkan wrappers. Vulkan has no stream
count argument; its independent work is encoded by the command-buffer harness.

```bash
for MODE in serial_latency independent_throughput; do
  OUT="$MICRO_ROOT/$MODE"
  mkdir -p "$OUT"

  python3 benchmarks/micro/runners/hip_dispatch_floor.py \
    --counts 1,50,200,941 \
    --kernels tiny \
    --grid-sweep 1,128,1024,8192 \
    --reps 50 \
    --warmup 10 \
    --timing-mode "$MODE" \
    --independent-streams 4 \
    --environment-json "$MICRO_ENV" \
    --environment-ref "$MICRO_ENV" \
    --gfx-arch gfx1151 \
    --hardware-gpu "$GPU_NAME" \
    --out "$OUT/hip-dispatch-floor.json" \
    --pretty

  python3 benchmarks/micro/runners/vulkan_dispatch_floor.py \
    --counts 1,50,200,941 \
    --grid-sweep 1,128,1024,8192 \
    --reps 50 \
    --warmup 10 \
    --timing-mode "$MODE" \
    --environment-json "$MICRO_ENV" \
    --environment-ref "$MICRO_ENV" \
    --gfx-arch gfx1151 \
    --hardware-gpu "$GPU_NAME" \
    --out "$OUT/vulkan-dispatch-floor.json" \
    --pretty

  python3 benchmarks/micro/runners/hip_dispatch_floor.py \
    --compare "$OUT/hip-dispatch-floor.json" \
              "$OUT/vulkan-dispatch-floor.json" \
    --out "$OUT/dispatch-floor-comparison.json" \
    --pretty
done
```

### Separate-Backend Families

This helper invokes each wrapper's `--backend` and `--compare` interfaces for
both timing modes. The arguments following the name are passed unchanged to
both backend runs.

```bash
run_paired_micro() {
  local RUNNER=$1
  local NAME=$2
  shift 2

  local MODE OUT BACKEND
  for MODE in serial_latency independent_throughput; do
    OUT="$MICRO_ROOT/$MODE"
    mkdir -p "$OUT"

    for BACKEND in hip vulkan; do
      python3 "$RUNNER" \
        --backend "$BACKEND" \
        --timing-mode "$MODE" \
        --independent-streams 4 \
        --reps 20 \
        --warmup 5 \
        --samples 7 \
        --environment-json "$MICRO_ENV" \
        --environment-ref "$MICRO_ENV" \
        --gfx-arch gfx1151 \
        --hardware-gpu "$GPU_NAME" \
        "$@" \
        --out "$OUT/$BACKEND-$NAME.json" \
        --pretty
    done

    python3 "$RUNNER" \
      --compare "$OUT/hip-$NAME.json" "$OUT/vulkan-$NAME.json" \
      --out "$OUT/$NAME-comparison.json" \
      --pretty
  done
}
```

Run the synthetic families with bounded gfx1151 matrices:

```bash
run_paired_micro benchmarks/micro/runners/geometry_sweep.py geometry \
  --k-list 512,2048 \
  --rows-list 1,4 \
  --workgroups 64,256 \
  --body-repeats 32

run_paired_micro benchmarks/micro/runners/memory_waitcnt.py memory-waitcnt \
  --variants coalesced:4,strided:4,gather:1,interleave:4 \
  --n 32768 \
  --body-iters 64 \
  --workgroups 64,256

run_paired_micro benchmarks/micro/runners/dot_path.py packed-dot \
  --variants q8_signed:16,q4_unsigned:16,q6_zero:16,scalar_dequant:16 \
  --n 32768 \
  --body-iters 64 \
  --workgroups 64,256

run_paired_micro benchmarks/micro/runners/vopd_sweep.py vopd \
  --variants independent_fma:4,dependent_fma:4,mixed_int_float:4,dequant_like:4 \
  --n 65536 \
  --body-iters 512 \
  --workgroups 64,256

run_paired_micro benchmarks/micro/runners/sampler_argmax.py sampler \
  --rows-list 1,4,8 \
  --workgroups 64,256 \
  --top-k-list 1,8 \
  --vocab 32768
```

Run the production-shaped slice families through the same helper:

```bash
run_paired_micro benchmarks/micro/runners/q4_selected_dual_real_slice.py q4-selected-dual \
  --x-rows 4 \
  --rows 32 \
  --experts 256 \
  --in-features 2048 \
  --out-features 512 \
  --workgroups 64,128

run_paired_micro benchmarks/micro/runners/q6_x8_real_slice.py q6-x8-selected-down \
  --rows 8 \
  --experts 256 \
  --in-features 512 \
  --out-features 2048 \
  --local-size 64

run_paired_micro benchmarks/micro/runners/q8_0_dense_real_slice.py dense-q8 \
  --shapes 768x2048,2048x2048 \
  --rows-list 1,4 \
  --local-sizes 32,64,128 \
  --row-tiles 1,4
```

For profiling, prebuild HIP kernels outside `rocprofv3`, provide
`--compiler-version-file`, and use `--require-cached-build` on runners that
expose those options.

### Joint Reduction Families

Both joint wrappers run HIP and Vulkan and emit one v2 comparison artifact.

```bash
for MODE in serial_latency independent_throughput; do
  OUT="$MICRO_ROOT/$MODE"
  mkdir -p "$OUT"

  python3 benchmarks/micro/runners/reduction_sweep.py \
    --backend both \
    --k-list 512,2048 \
    --rows-list 1 \
    --workgroups 64,256 \
    --body-repeats 32 \
    --reps 20 \
    --warmup 5 \
    --samples 7 \
    --timing-mode "$MODE" \
    --independent-streams 4 \
    --environment-json "$MICRO_ENV" \
    --environment-ref "$MICRO_ENV" \
    --gfx-arch gfx1151 \
    --hardware-gpu "$GPU_NAME" \
    --out "$OUT/reduction-sweep.json" \
    --pretty

  python3 benchmarks/micro/runners/two_stage_reduction.py \
    --backend both \
    --k-list 8192,32768 \
    --rows-list 1,4 \
    --workgroups 128,256 \
    --split-counts 2,4 \
    --body-repeats 16 \
    --reps 20 \
    --warmup 5 \
    --samples 7 \
    --timing-mode "$MODE" \
    --independent-streams 4 \
    --environment-json "$MICRO_ENV" \
    --environment-ref "$MICRO_ENV" \
    --gfx-arch gfx1151 \
    --hardware-gpu "$GPU_NAME" \
    --out "$OUT/two-stage-reduction.json" \
    --pretty
done
```

## Result and Retention Rules

Every retained timing artifact must include:

- exact wrapper/build/run commands and working directory;
- commit, branch, dirty state, and source hash;
- detected GPU name, gfx target, device index, driver/runtime/compiler versions;
- full shape, quant/layout, workgroup, stream/queue count, warmup, repetitions,
  samples, timing mode, and submission strategy;
- `single` and `burst` GPU/host distributions;
- timed-sequence correctness and synchronization validation;
- a schema-v2 result or comparison classification.

A comparison artifact must reject missing, duplicate, or shape-mismatched rows.
GPU ratios require matched dependency contracts. Host ratios additionally
require matched submission classes. A result that fails either gate stays
diagnostic and must not be summarized as a backend speedup. Dirty,
commit-mismatched, device-mismatched, or failed-correctness inputs must set
`performance_claim=false` even when diagnostic ratios remain visible.

## Legacy Artifact Quarantine

Existing files under `results/` remain useful for provenance and for locating
the source of earlier hypotheses. Timing fields and ratios from artifacts that
predate timing-contract v2, omit the v2 dependency/submission/correctness
contract, or timed Vulkan repetitions without the required dependencies are
historical only. Do not use them in current dashboards, optimization priority,
or HIP/Vulkan performance claims.

Static ISA data in an older artifact may still be referenced as static evidence
when its source/build identity is known. It does not make the artifact's timing
ratio current. New retained rows should be added only after rerunning the exact
matrix above (or a documented superset) under both timing modes.

## Classification

Every retained artifact chooses one primary classification from the result
schema:

| Classification | Meaning |
| --- | --- |
| `compiler_aco` | Matched algorithm/layout/geometry with a supported compiler/ISA attribution |
| `geometry` | Workgroup/subgroup geometry accounts for the measured delta |
| `wave_mode` | A matched wave/subgroup control accounts for the delta |
| `runtime_dispatch` | Dispatch, submission, or grid overhead accounts for the delta |
| `layout_quant` | Quantization or data layout accounts for the delta |
| `fusion_topology` | Matched primitive kernels but different fused topology accounts for the delta |
| `real_slice_probe` | Correct production-shaped slice evidence with bounded transfer scope |
| `not_reproducible` | The prior difference disappears under the v2 controls |
| `diagnostic_unclassified` | Correct gap remains without enough evidence for one cause |

The benchmark plan and interpretation dashboard live in
`docs/HIP-vs-VULKAN.md`.
