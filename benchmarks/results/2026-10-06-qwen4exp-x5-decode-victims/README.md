# Q4 gate/up and Q8 down after first scratch

The four tested victim/trigger pairs show no multi-fold persistent slowdown
on Radeon 8060S/gfx1151 host `55ea6c509d0b49eea8de7094a1023668`.
This completes the gate/up and Q8 decode coverage missing from the earlier
X-5 screens. It does not establish a causal null, universal throttle absence,
or a serving-speed improvement.

The victims use real `blk.4` expert weights from unsloth Qwen3.8-Flash-Next
`UD-Q4_K_XL`, with 512 experts, top-k10 and deterministic synthetic activations
and routes. Q4_K gate/up includes BF16→Q8_1 quantization and the one-plane
DP4A dual+SiLU owner, in2560/out640. Q8_0 down uses selected BF16 GEMV,
in640/out2560, emitting ten expert rows; combine is not included.

Each pair runs in four serial fresh processes, ordered
`sham / scratch / scratch / sham`. Each process performs three victim warmups,
31 before samples, one trigger or victim sham, then 31 after samples. HIP
start/stop events enclose asynchronous launches; host synchronization occurs
outside the interval. Events can include launch gaps, so these are operation
windows, not pure kernel duration. Trigger buffers and libraries are prepared
in both arms before baseline without executing the trigger. Trigger numerical
checks and before/after output comparisons occur after the timing windows.

| Trigger | Victim | Sham after/before | Scratch after/before |
| --- | --- | ---: | ---: |
| GDN, 36 B private/lane | Q4 gate/up | 0.998221 / 0.975183 | 0.982723 / 0.976003 |
| GDN, 36 B private/lane | Q8 down | 0.982778 / 0.980630 | 0.989649 / 1.023389 |
| Dense-wide, 92 B private/lane | Q4 gate/up | 0.975490 / 0.978012 | 0.979787 / 0.975572 |
| Dense-wide, 92 B private/lane | Q8 down | 0.980990 / 0.999992 | 1.000649 / 0.980410 |

The Q8/GDN scratch-arm median ratio divided by the sham-arm median ratio
is 1.02528. One scratch run has a 2.34% within-process increase while the other
has a 1.04% decrease. This small effect is unresolved; two scratch processes
and boundary clock snapshots cannot distinguish scratch causality from timing
variation. No few-percent non-regression claim follows. Other pairs' normalized
ratios are 0.99256, 1.00095 and 1.00004; these are diagnostics, not speedups.

All 16 victim outputs are finite and byte-identical before/after, with one
output hash per victim across the eight processes. GDN output/state matches
the strict parent exactly. Dense-wide matches the CPU F16-operand reference.
Boundary clock snapshots are identical across all four stages of all 16 runs
(SCLK 2900 MHz, MCLK 1000 MHz). Loaded library, weight, compiler, command and
source hashes are recorded in the [artifact](artifact.json).

## First-allocation verification

Four separate cached `rocprofv3 --kernel-trace --scratch-memory-trace` captures
use three samples per window. Each shows six zero-private-segment victim calls
(three warmups plus three baseline samples), the first queue scratch allocation,
one trigger, and three subsequent victim calls, all on queue1. All dispatches
before the trigger, including runtime copy kernels, report zero private scratch.

| Trigger | First allocation bytes | Trigger private bytes/lane | Trigger duration µs, Q4 / Q8 capture |
| --- | ---: | ---: | ---: |
| Original wave-norm GDN, 64 tokens | 1,638,400 | 36 | 369.533 / 368.932 |
| Original F32 dense-wide, rows257/K256/N256 | 3,932,160 | 92 | 29.976 / 30.337 |

Allocation completes before trigger dispatch. No GDN executes in the dense
captures. In GDN captures the strict-parent validation runs only after the
post-trigger victim window. Profiler timings establish execution and order;
they are not used for the unprofiled ratios. Trace file hashes and compact
allocation/dispatch records are in the artifact; raw dumps remain local.

## Limits and earlier evidence

Both decode victims have zero private segments. The earlier
[scratch-enabled GDN growth screen](../2026-09-28-qwen4exp-scratch-growth/README.md)
and [Q5_1 first-trigger screens](../2026-09-28-qwen4exp-decode-scratch/README.md)
are separate evidence and were not rerun. This packet tests the original
36-byte GDN and 92-byte F32 dense-wide diagnostic leaves, not the promoted
28-byte checked-index GDN or zero-private padded F16 tail. It covers one
weight bank, one activation/route fixture, one host/compiler/queue, no long
mixed-session workload and no serving wall. Runtime/defaults are unchanged.

The first local probe revision is INVALID for timing: it synchronized the
device between start and stop events. Its dense-wide branch also omitted the
launch. Those attempts are excluded. Three CPU protocol tests fail before the
repair and pass afterward; the corrected 16-process screen and four traces
supply the evidence here. This correction does not invalidate the separately
committed September 28 screens.

## Reproduction

```bash
E=/home/lhl/miniforge3/envs/therock10-staging-20260828
S=$E/lib/python3.12/site-packages
export PATH="$S/_rocm_sdk_devel/bin:$S/_rocm_sdk_core/bin:$E/bin:$PATH"
export LD_LIBRARY_PATH="$S/_rocm_sdk_devel/lib:$S/_rocm_sdk_core/lib:$S/_rocm_sdk_libraries/lib"
export HIPENGINE_HIP_ARCH=gfx1151 HIPENGINE_REQUIRE_CACHED_BUILD=1
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipcc-version-qwen4exp.txt
export OPENBLAS_NUM_THREADS=1
MODEL=/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
OUT=/tmp/qwen4exp-scratch-audit/victim-screen-v2
mkdir -p "$OUT"
for trigger in gdn dense-wide; do
  for victim in q4-gate-up q8-down; do
    i=0
    for arm in sham scratch scratch sham; do
      "$E/bin/python3.12" scripts/qwen4exp_decode_scratch_victim_probe.py \
        --model-root "$MODEL" --arm "$arm" --victim "$victim" \
        --trigger "$trigger" --repeats 31 \
        --output "$OUT/$trigger-$victim-$i-$arm.json"
      i=$((i+1))
    done
  done
done
for trigger in gdn dense-wide; do
  for victim in q4-gate-up q8-down; do
    rocprofv3 --kernel-trace --scratch-memory-trace --output-format csv \
      --output-directory "$OUT/prof-$trigger-$victim" -- \
      "$E/bin/python3.12" scripts/qwen4exp_decode_scratch_victim_probe.py \
      --model-root "$MODEL" --arm scratch --victim "$victim" \
      --trigger "$trigger" --repeats 3 \
      --output "$OUT/prof-$trigger-$victim.json"
  done
done
```

All four libraries were already built outside the profiler; cached-only
execution succeeds. The artifact records the compiler-version file's contents
and loaded library hashes. Focused CPU validation:

```bash
uv run pytest -q tests/test_unit_qwen4exp_decode_scratch_victim_probe.py
python3 -m py_compile scripts/qwen4exp_decode_scratch_victim_probe.py
```
