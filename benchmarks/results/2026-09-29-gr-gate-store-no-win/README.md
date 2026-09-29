# HC-3: the GR gate store is dead, and removing it is a 2% regression

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), ROCm platform 10.0.0, hostname `gfx1151`,
machine-id `55ea6c509d0b49eea8de7094a1023668`, kernel `7.1.6-1-cachyos`,
`power_dpm_force_performance_level=high`.
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL, `hc_ffn_up.weight`
**Shape:** rows=1024, K=320, 4 branches x hidden 2560 = 10240 outputs
**Command:** `scripts/qwen4exp_gr_projection_screen.py --model-root
/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL --rows 1024
--repetitions 5 --output /tmp/gr-nogate.json`
**Artifact:** `screen.json` (this directory)

## Verdict

**HC-3 is closed: no win is available.** The gate write is genuinely dead, and
suppressing it makes the kernel **slower**.

## The two halves

### 1. The write is dead (consumer census)

`run_qwen4_exp_gr_read` returns `Qwen4ExpGRReadDeviceResult(normalized, gate,
mixed, inject_logits)`. Across all 12 call sites in `qwen4_exp_runner.py` plus
`qwen4_exp_mtp.py`, **no production consumer reads `.gate`**. Lines 5561/5623
pass `scratch.*.gate` through when constructing synthetic results — they do not
consume the field. The single reader in the tree is
`tests/test_unit_qwen4_exp_cpu_reference.py:64`, a unit test asserting the value
on a synthetic result.

### 2. Removing it is not a win (measured)

The suppressed-store variant is a **strict** change: the gate values are already
in registers and already feed `mixed`, so dropping the store leaves `mixed`
bit-identical. That was verified rather than assumed —
`tests/test_gpu_qwen4exp_gr_no_gate.py` asserts uint32-view equality of `mixed`
between the two variants at four shapes including the production shape
(rows=257, k=320, h=2560), and all 5 tests pass. Because the computation is
identical, the timing comparison is a pure store-cost comparison.

| arm | median ms | vs production | `mixed` vs production |
|---|---:|---:|---|
| `fused_exact` (production) | 6.5198 | 1.0000 | reference |
| `fused_no_gate` | 6.6551 | 0.9797 | **bit-identical** |
| `staged_f32` | 2.7291 | 2.3890 | relative L2 7.442e-08 |
| `wide_f16in` | 1.4181 | 4.5976 | relative L2 4.709e-05 |

The 2% difference is real, not noise — the two five-sample sets do not overlap
in the way noise would require:

```
fused_exact    6.4979  6.5153  6.5198  6.5457  6.5560   (max 6.5560)
fused_no_gate  6.5410  6.6011  6.6551  6.6570  6.6818   (median 6.6551)
```

`fused_exact`'s **maximum** is below `fused_no_gate`'s **median**. Only
`fused_no_gate`'s single fastest sample falls inside the other set.

## Why this is plausible

The kernel is weight-traffic-bound: 3.565 GB of weight reads per launch against
a 42 MB gate store at rows=1024, so the store is ~1.2% of traffic and was
expected to be worth ~1% if it cost anything at all. The sign is what is
surprising. The most likely explanation is scheduling, not bandwidth: the store
is a latency sink that keeps the memory pipeline occupied while the next
residency's weight loads are in flight, and removing it leaves a bubble. That is
an inference, not a measurement — it was not profiled.

**Caveat that bounds the conclusion.** This variant was implemented as a
`if (WRITE_GATE)` branch on a template parameter inside the existing epilogue.
It is possible that a separately written kernel body with no branch at all
schedules differently and does show the expected small win. What is measured and
retainable is the negative result for this implementation: as built, suppressing
the store is a 2% regression, so it must not be adopted on the theory that a
dead write is free money.

## What this changes

HC-3 moves from OPEN to CLOSED. The ledger's own framing — "No arithmetic risk →
Remove or fuse the write" — was correct about the risk and wrong about the
direction. A future reader should not re-attempt this expecting a win; the
measurement is the reason.

The strict no-gate variant is retained as the instrument that proves the store
is purely additive, which is what makes the timing comparison meaningful. Its
removal trigger is recorded in `docs/REFACTOR.md`.

## Reproduction

`hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip` carries the kernel as
`template <bool WAVE_SCALE, bool WRITE_GATE>`; the export is
`hipengine_gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_no_gate_f32`
with the Python wrapper
`gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_no_gate_f32`. The
screen arm is `fused_no_gate`. Build environment: TheRock staging
`therock10-staging-20260828`, `HIPENGINE_HIP_ARCH=gfx1151`.

## Provenance caveat

Weights are real (layer 0/9/22/34/46 `hc_ffn_up` Q8_0 from the named GGUF);
activations and the `normalized` epilogue input are synthetic
(`np.random.default_rng(20260926)`, sigma 0.2), matching the existing screen
protocol. The timing boundary is an operation-complete host call after two
warmups, with layer and arm order rotated per repetition. This is not an
end-to-end benchmark.
