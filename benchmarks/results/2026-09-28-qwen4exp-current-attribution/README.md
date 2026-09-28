# Qwen4Exp current-owner attribution — 2026-09-28 UTC

These captures prioritize experiments; they do not establish an optimization
speedup, production qualification, or a new throughput baseline.

## Scope and checks

- Source: `2789c19d634c2d86f6e67d5140181bcf45c71e69`.
- Physical host: Framework `gfx1151`, Radeon 8060S / Strix Halo,
  machine `55ea6c509d0b49eea8de7094a1023668`.
- Model: unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`, BF16 KV,
  `code-p4096`, warm PLE, chunk1024, production profile.
- Manifest: `35e57360e648bcc8ce8f21d89bf5583a0b4217061b306ec96616b2607cc6e565`.
- Runtime: TheRock 10.0.0 / HIP 7.15.26333; GPU policy `high`, CPU governors
  `powersave`; tuned daemon inactive. No contemporaneous temperature/load series.
- Prefill logits and manifest match across unprofiled, ordinary-profiled and
  supplemental-copy-profiled runs. Decode repeats match token and state hashes
  within and across unprofiled/profiled runs. All lifecycle checks pass with
  zero measured allocation growth and zero live allocations after close.
- These are repeatability/engagement checks, not a strict-teacher numerical,
  category/heldout, isolation, BF16-relative or task-quality gate.

The compact [artifact](artifact.json) preserves child reports, commands, host
identity, aggregated roles, symbols, layer engagement and resources. Raw CSVs
remain under `/tmp/qwen4exp-ledger-20260928-current/` and are not committed.

## Results

The first prefill capture attributes all 10,708 launches: **15,741.963 ms**
kernel union in a **16,319.888 ms** marked request. The unprofiled request
is 16,334.447 ms; one sample is not a stable performance baseline. Symbol-family buckets are
4,553.007 ms dense, 3,375.634 gate/up, 2,230.120 down, 2,587.665 HC,
1,040.374 attention, 844.295 GDN, 763.833 router/reduce, 319.494 norms,
17.520 indexer, 9.834 PLE and 0.188 other. These are not additive savings.
The exact coltile symbol costs 3,367.698 ms; HC up 2,490.646 ms; staged F16
casts 131.001 ms. Sparse attention has **24** launches / 455.080 ms; full
attention has 36 / 438.971 ms; scorer has 24 / 105.942 ms.

The supplemental capture, with memory-copy and scratch tracing enabled,
attributes all launches again: 15,600.436 ms kernels, 2.711 ms observed device
copies (four H2D and one D2H), 15,603.147 ms kernel-plus-copy union, and
556.256 ms outside that union in a 16,159.403 ms window. HIP API intervals
include synchronization; their sums must not be added to device time. Reset
memset kernels are included in kernel attribution. The residual is not proof
of exclusively recoverable Python/launch overhead.

Decode restores the same fixed-live4097 transition before each repeat:

| Repeat | Marked wall ms | Kernel union ms | Outside kernels ms |
| --- | ---: | ---: | ---: |
| 0 | 239.223 | 54.777 | 184.445 |
| 1 | 84.212 | 54.088 | 30.124 |
| 2 | 59.192 | 53.701 | 5.491 |

Unprofiled wall is 138.701 / 141.105 / 123.641 ms. This instability prevents a
throughput or serving-wrapper overhead claim. Device owners in repeat 1 are
MoE 24.625 ms, linear 23.121, QSA 2.594, GDN 2.386, GR read 0.851,
boundary 0.478 and PLE 0.034. The MoE gate/up symbol alone is 14.055 ms;
Q5_1 down is 5.447 ms. All 48 MoE graphs replay; allocation counts are stable.
The old prefill-only symbol mapper misclassifies decode experts, so decode
uses semantic owner ranges and explicit kernel names instead.

## Scratch and remaining prerequisites

Prefill reports `Scratch_Size=36` on GDN `<true,true>` and `92` on the
`dense_wide<...,true>` variant. The supplemental trace records queue scratch
allocations of 1,638,400 and 3,932,160 bytes before the measured request, with
one intervening free; no allocation/free occurs inside the measured window.
Decode reports zero scratch on observed kernels. These facts do not establish
or rule out a persistent scratch-trigger penalty. X-5 still needs code-object
private-segment metadata, disassembly and ordered trigger/victim A/B.

X-11 receives role/layer/symbol engagement, not repair-oracle or task admission.
X-4 receives a current runner device split, not a stable serving-wrapper split.
X-7 and X-10 remain open. No candidate is rejected or promoted by this packet.

## Reproduction

Use a separate output directory for each trace and run GPU work serially.
The artifact records every child argument, including absolute model paths.
The executed collector environment was:

```bash
ENV_PREFIX=/home/lhl/miniforge3/envs/therock10-staging-20260828
SITE=$ENV_PREFIX/lib/python3.12/site-packages
export PATH="$SITE/_rocm_sdk_devel/bin:$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib"
export HIPENGINE_HIP_ARCH=gfx1151 HIPENGINE_KERNEL_CENSUS=1
OUT=/tmp/qwen4exp-ledger-20260928-current
mkdir -p "$OUT"
hipcc --version > "$OUT/hipcc-version.txt"
```

Run `captures.warm.command` and `captures.decode-warm.command` from
`artifact.json` with `$ENV_PREFIX/bin/python`, prefixing their script basename
with `scripts/`, outside the profiler. Then run each corresponding child
command under:

```bash
rocprofv3 --kernel-trace --marker-trace --hip-trace --output-format csv \
  -d "$OUT/role-trace" -- "$ENV_PREFIX/bin/python" scripts/qwen4exp_profile_gap.py ...
rocprofv3 --kernel-trace --marker-trace --hip-trace --output-format csv \
  -d "$OUT/decode-trace" -- "$ENV_PREFIX/bin/python" scripts/qwen4exp_context_decode_profile.py ...
```

Here `...` means the complete remaining argument array in the named child
report, not shell syntax to execute. For the supplemental capture use
`supplemental_memory_capture.child.command`, add `--memory-copy-trace
--scratch-memory-trace` to the collector, and use `$OUT/memory-trace`.
All profiled children require cached builds and the compiler-version file.

Analyze each directory with `scripts/qwen4exp_trace_audit.py --trace-dir
<directory>/gfx1151 --measure-prefix <prefix> --output <json>`; prefixes are
`qwen4exp_prefill_p4096_` and `qwen4exp_decode_live4097_`. The original analyzer
accepts one marker only; the wrapper analyzes each decode repeat separately.
No trace needs to be rerun to repair that multi-marker analysis error.

Validation: 16 focused CPU tests pass across trace audit, role analysis,
context-decode profiling and dense-route mapping. The initial new-tool test
failed import before implementation (RED); the implemented wrapper passes.
