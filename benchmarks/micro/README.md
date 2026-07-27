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

## Retained gfx1100 Matrix

The clean W7900 run at
`c57f21b5d5d5fd5f389a7f3921062578c53eb744` retains all 11 paired
families in both modes. All 22 comparisons and 232 burst GPU rows pass
correctness, exact-matrix, same-device/source, clean-provenance, and GPU-clock
gates. The compact result is
[`2026-07-11-hip-vulkan-timing-v2-bounded.json`](results/gfx1100/w7900/2026-07-11-hip-vulkan-timing-v2-bounded.json).

These are burst GPU Vulkan/HIP ratios (`HIP time / Vulkan time`); above `1.0x`
favors Vulkan.

| Family | Serial | Independent |
| --- | ---: | ---: |
| Dispatch/grid | `2.437x-10.122x` | `1.980x-65.325x` |
| Geometry | `0.360x-0.790x` | `1.100x-3.925x` |
| Reduction | `0.304x-0.729x` | `1.110x-4.035x` |
| Memory/waitcnt | `0.517x-0.936x` | `0.544x-2.139x` |
| Packed dot | `1.052x-1.133x` | `1.872x-2.106x` |
| VOPD | `0.391x-0.561x` | `0.516x-0.616x` |
| Sampler | `0.259x-0.501x` | `0.782x-2.563x` |
| Two-stage reduction | `0.324x-0.925x` | `0.394x-0.813x` |

Matched combined production slices mostly favor HIP. Serialized/independent Q4
selected-dual is `0.501x-0.562x` / `0.432x-0.477x`; Q6 selected-down X8 is
`0.675x` / `0.673x`; dense Q8_0 is `0.393x-0.966x` / `0.388x-1.030x`.
Only three small `768x2048` dense independent combined rows barely favor
Vulkan. A focused Q6 50-repetition independent follow-up failed its Vulkan
timed-sequence oracle and is retained only as rejected diagnostic evidence in
the compact artifact.

The final process used TheRock ROCm `7.15.0a20260711` root/core/generic
multi-arch libraries and excluded the stale installed gfx110X-all 7.13 library
path. See [`docs/HIP-vs-VULKAN.md`](../../docs/HIP-vs-VULKAN.md) for the
cross-architecture interpretation and caveats.

## Retained gfx1151 Matrix

The clean matched-protocol refresh at
`0e566a4559b52a8bfc65ccdbda22556ae9112279` retains all 11 paired families in
both modes on TheRock `7.15.0a20260711`, kernel `7.1.3-2-cachyos`, and
Mesa/RADV `26.1.4`. All 22 comparisons and 232 burst GPU rows pass
timed-command correctness, exact requested matrix, same-device/source, clean
provenance, and corrected-gfx1151 environment gates. The compact result is
[`2026-07-11-gfx1151-hip-vulkan-matched-protocol.json`](../results/2026-07-11-gfx1151-hip-vulkan-matched-protocol.json).

These are burst GPU Vulkan/HIP ratios (`HIP time / Vulkan time`); above `1.0x`
favors Vulkan.

| Family | Serial | Independent |
| --- | ---: | ---: |
| Dispatch/grid | `1.128x-10.751x` | `1.115x-142.384x` |
| Geometry | `0.707x-0.992x` | `2.619x-20.832x` |
| Reduction | `0.659x-0.984x` | `2.525x-21.024x` |
| Memory/waitcnt | `0.891x-1.109x` | `1.006x-1.170x` |
| Packed dot | `3.054x-3.204x` | `3.833x-4.197x` |
| VOPD | `1.061x-1.181x` | `1.010x-1.103x` |
| Sampler | `0.517x-1.142x` | `1.526x-10.015x` |
| Two-stage reduction | `0.681x-0.958x` | `0.825x-1.826x` |

Matched combined production slices favor HIP in every serialized row: Q4
selected-dual is `0.916x-0.980x`, Q6 selected-down X8 is `0.553x`, and dense
Q8_0 is `0.552x-0.879x`. Independent combined rows are Q4
`0.854x-0.973x`, Q6 `0.480x`, and dense Q8_0 `0.448x-1.152x`; only the smallest
dense rows favor Vulkan. Q6 lm-head remains blocked because the two backends
do not execute the same math/layout. The exact retained matrix took 249.323
seconds (`4m09.323s`). Its sampling counts intentionally match the prior
2026-07-10 retained run: paired `10/3/5`, dispatch `20/5`.

The current command templates below are stricter: paired `20/5/7`, dispatch
`50/10`. That coverage produces 20/22 valid comparisons because additional
input slices make Vulkan independent Q4 fail KL and Q6 fail top-1. In
zero-based fixture indexing, Q4 first fails at slice `10`; Q6's KL first changes
at slice `14` but remains valid, and its top-1 gate first fails at slice `17`.
Mesa 26.1.2 reproduces those failures, one/two/four-queue outputs are identical,
and both families pass at the retained 10-repetition coverage. This diagnostic
is therefore fixture coverage, not a Mesa, synchronization, architecture, or
TheRock regression. Its compact artifact is
[`2026-07-11-gfx1151-hip-vulkan-matched-stack-diagnostic.json`](../results/2026-07-11-gfx1151-hip-vulkan-matched-stack-diagnostic.json).
It took 5m49s including failure handling and one confirmatory rerun; budget
approximately six minutes for the stricter matrix.

## Core Files

| Path | Purpose |
| --- | --- |
| `timing_contract.py` | Shared mode, control, metric, correctness, and comparison gates |
| `redline_matrix.py` | Optional pinned-source Redline runner, normalizer, and tri-comparator |
| `redline_dispatch.py` | Direct profiled-PM4 dispatch/grid floor control |
| `REDLINE-NOTICE` | Apache-2.0 attribution required by the derived Redline orchestration |
| `comparison_claim.py` | Shared source, device, raw-identity, and matrix claim gate for joint wrappers |
| `runners/micro_timing_hip.hpp` | HIP event and multi-stream timing helpers |
| `runners/micro_timing_vulkan.hpp` | Vulkan timestamp, barrier, calibrated multi-queue timing helpers |
| `collect_env.py` | Dependency-free environment/device capture with the shared `hipengine_artifact_provenance` v1 block |
| `schemas/result.schema.json` | v1 legacy plus v2 result/comparison artifact schema |
| `schemas/environment.schema.json` | Environment artifact schema |
| `results/` | Retained compact artifacts, separated by architecture and device |

## Experimental Redline retained-PM4 arm

`redline_matrix.py` adds a default-off third result backend without modifying the
native HIP or Vulkan runners. It requires a separate, clean Redline checkout at
commit `33683f3d4f302a6c56bcc7a4c33ab8be3262dd2e`; the external repository is not
vendored or imported by the hipEngine runtime.

The adapter captures each unchanged HIP launch closure once to recover exact
kernel topology and argument bytes. Capture is outside every returned sample.
The timed path must then succeed through profiled `redline-capi` retained PM4;
there is no native-HIP fallback in this microbenchmark arm. Every normalized
result records the Redline checkout, shared-library and adapter hashes,
Radiowave/code-object sidecars, `retained_pm4_ib` submission, and
`redline_pm4_timestamp` GPU clock.

On gfx1100, Redline's supported auto policy resolves at most two independent
public queues. Therefore the controlled matrix runs HIP and Redline with the
same two-lane cap instead of comparing Redline Q2 against HIP Q4. Vulkan's
single-command-buffer families retain their native submission contract; the
multi-stage Vulkan families receive the same requested lane count. Host-wall
ratios are emitted only when submission classes match.

A clean source/math/shape comparison is not by itself a transport attribution:
the Radiowave sidecar and the native HIP harness code object are not asserted
byte-identical. `transport_attribution` remains
`blocked_no_same_hsaco_control` until a separate same-HSACO HIP/Redline control
passes. This distinction does not block reporting descriptive per-backend GPU
times, wins, and regressions.

Example (TheRock paths are intentionally explicit):

```bash
python3 benchmarks/micro/redline_matrix.py \
  --redline-root /home/lhl/redline \
  --rocm-root "$THEROCK_ROOT" \
  --hipcc "$THEROCK_ROOT/bin/hipcc" \
  --llvm-bin "$THEROCK_ROOT/lib/llvm/bin" \
  --gfx-arch gfx1100 \
  --gpu-name "AMD Radeon Pro W7900" \
  --visible-device 0 --independent-lanes 2 \
  --reps 20 --warmup 5 --samples 7 \
  --build-redline \
  --out-dir /tmp/hipengine-redline-micro
```

The core runner covers the ten timer-substitutable families. Dispatch/grid is a
separate direct-PM4 floor control because its native HIP harness already owns an
inner hipGraph. `redline_dispatch.py` compiles hipEngine's current
`gmb_noop_kernel`, uses the inspected kernarg ABI, measures every profiled replay
sample directly, and validates every output element after both single and burst
replay. Example:

```bash
python3 benchmarks/micro/redline_dispatch.py \
  --redline-root /home/lhl/redline \
  --rocm-root "$THEROCK_ROOT" \
  --hipcc "$THEROCK_ROOT/bin/hipcc" \
  --bundler "$THEROCK_ROOT/lib/llvm/bin/clang-offload-bundler" \
  --llvm-readobj "$THEROCK_ROOT/lib/llvm/bin/llvm-readobj" \
  --environment-json /tmp/hipengine-redline-micro/environment.json \
  --environment-ref /tmp/hipengine-redline-micro/environment.json \
  --build-dir /tmp/hipengine-redline-micro/build/redline/dispatch \
  --out /tmp/hipengine-redline-micro/redline-dispatch.json
```

The matrix manifest states whether this control and the same-HSACO control were
run rather than silently treating either as covered. The external sanity
reference is [ROCm issue #6409 comment 5061942739](https://github.com/ROCm/ROCm/issues/6409#issuecomment-5061942739):
on an RX 7900 XTX with ROCm 7.14 and identical HIP/Redline HSACO, Redline Q2
beat HIP on `227/240` rows (median `2.34x`) and Vulkan on `192/240` (median
`1.43x`), with a `6.88x` median serialized dispatch-grid advantage over direct
HIP. Those are comparison anchors, not pass thresholds for this W7900/ROCm-7.15
run; hardware, stack, matrix, lane, and same-HSACO differences must be stated
before treating a deviation as inconsistent.

### W7900 result and decision

The 2026-07-28 same-HSACO control passes **240/240** correctness rows. Redline
is first on **208**, beats HIP on **239** at median **2.792x**, and beats Vulkan
on **208** at median **1.696x**. Under the formal <=10% sample-range gate, it
wins **212/213** stable HIP pairs at median **2.810x** and **115/134** stable
Vulkan pairs at **1.523x**. The separate direct hipEngine dispatch control
passes 16/16 and beats hipGraph on every row at median **3.059x serial / 7.881x
independent**, but it loses the Vulkan GPU-timestamp floor. Full provenance and
raw hashes are in
[`2026-07-28-gfx1100-redline-transport-spike.json`](../results/2026-07-28-gfx1100-redline-transport-spike.json).

This admits a same-HSACO **microbenchmark transport claim only**. A broad
long-lived process faulted at address zero on row 49 after 48 passing rows;
family-isolated processes completed the deduplicated matrix. Process isolation
is a benchmark workaround, not a runtime fix. Redline remains default-off and
unvendored until the lifecycle fault is minimized and fixed upstream and a
strict hipEngine decode-graph replay passes bit-exact and end-to-end gates.

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
