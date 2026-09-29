# DE-1 candidate screen: exact vs Q8_1-DP4A at the real decode gate+up shape

**Status:** kernel-level screen, **performance claim**, **candidate measured 2.0x faster**.
**Not** an end-to-end decode speedup, and **not** a promotion - the production numerical
gate has not been run.

## Why this screen

`DE-1`'s decode owner is `gguf_q4_k_selected_dual_silu_logical128_t64` at **14.013 ms/step**,
888.83 MB/step, **63.43 GB/s = 28.7% of the documented roof** - the single largest decode
kernel family member and the least efficient of the three GEMVs in its own block (the router
GEMV reaches 184.66 GB/s and `lm_head` 226.67 GB/s in the same step).

A **Q8_1 DP4A variant of the same `logical128/t64` structure already exists in-tree**
(`gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out`). It was reverted
on 2026-09-14 by the arithmetic recovery, whose triggering candidate explicitly left the decode
linear unchanged - and it has its **own passing production acceptance** from 2026-08-29
(`accepted_explicit_production_profile`, mean KL **2.72e-4**, max KL **5.77e-3**).

Before spending a profile gate on it, it has to be shown to be worth a gate.

## Protocol

`scripts/qwen4exp_de1_decode_gemv_screen.py`, at the shape the decode path actually runs:
**one token, 10 selected experts of 512, 2560 -> 640**, i.e. the real
`ffn_gate_exps`/`ffn_up_exps` geometry (each `(512, 640, 2560)` per layer).

- 18,432,000 bytes per dispatch (exact synthetic payload, not a bits-per-weight estimate)
- 5 warmups, 30 reps, **min** statistic, `device_synchronize` on both sides
- The DP4A arm is charged for **both** `gguf_q4_k_quantize_bf16_q8_1` **and** the matvec,
  as `DE-1`'s own action requires ("fused quantize+matvec must retain the admitted planes and
  charge repeated per-block preparation")
- Host: **gfx1151**, machine_id `55ea6c509d0b49eea8de7094a1023668` - the same host as every
  artifact in this campaign

## Result

Two independent runs of the same protocol:

| arm | ms (run 1) | GB/s (run 1) | ms (run 2) | GB/s (run 2) | % roof |
|---|---:|---:|---:|---:|---:|
| unfused reference (`dual_gemv` + `silu_mul_separate`) | 0.3216 | 57.32 | - | - | 25.9% |
| **exact `logical128_t64`** (current default) | **0.2412** | **76.42** | 0.2380 | 77.43 | **~35%** |
| **Q8_1-DP4A `logical128_t64`** (quantize + matvec) | **0.1163** | **158.45** | 0.1200 | 153.56 | **~71%** |
| - of which Q8_1 quantize | 0.0140 | - | - | - | - |
| - of which DP4A matvec | 0.1024 | - | - | - | - |

**`dp4a_speedup_vs_exact` = 2.073x (run 1) and 1.983x (run 2)** - **~2.0x**, with ~5%
run-to-run variance. The Q8_1 quantization is **12.0%** of the DP4A arm's total, so the
speedup survives being charged for the preparation step.

**The DP4A arm reaches 71% of the documented roof**, versus 35% for the exact arm and 26% for
the unfused reference.

## Correctness, measured and not asserted

| check | value |
|---|---|
| `exact_bitexact_vs_unfused` | **True** |
| `dp4a_bitexact_vs_unfused` | **False** |
| `dp4a_rows_matching_unfused` | **0.0** |

The exact variant is **bit-exact** to the unfused reference. The DP4A variant is **not** - no
row is bit-identical.

**That is not disqualifying.** `AGENTS.md` is explicit that there is no exactness contract: the
binding contract is the production numerical envelope in `docs/EXECUTION-PROFILES.md` (calibrated
mean/p95/p99/max KL, top-1 by category, determinism, isolation, task quality), and a candidate
that is not bit-exact must be re-reviewed under those gates rather than rejected for its
arithmetic. The 2026-08-29 acceptance is exactly such a review, and it passed.

**Equally, this screen does not establish admissibility.** It establishes only that the candidate
is worth a gate run.

## What this does and does not establish

**Establishes:** at the real decode shape, on the same host, the Q8_1-DP4A `logical128/t64`
variant is **~2.0x faster** than the exact variant that production currently runs, after being
charged for its activation quantization, and it reaches 71% of roof versus 35%.

**Does NOT establish:** end-to-end decode improvement, production admissibility, or that the
speedup survives the real model's dynamically-quantized layers. **No promotion is claimed and no
default was changed.**

## If it holds end-to-end

`DE-1`'s owner is 14.013 ms/step of a 54.218 ms/step decode kernel sum. At 2.0x it would be
~7.0 ms/step, a **~7.0 ms/step saving = ~12.9% of the decode kernel sum**. That projection is
arithmetic on a kernel-level ratio, **not a measurement**, and the real figure depends on the
gate and on whether other layers differ.

## Reproduce

```bash
export LD_LIBRARY_PATH=$ROCm_SP/_rocm_sdk_core/lib:$ROCm_SP/_rocm_sdk_devel/lib:$ROCm_SP/_rocm_sdk_libraries/lib
export PATH=$ROCm_ENV/bin:$PATH
python3 scripts/qwen4exp_de1_decode_gemv_screen.py --output <artifact>.json
```

Artifact: `screen.json` (second run).

## Limits

- **Kernel-level screen only.** No profile gate, no end-to-end decode run, no end-to-end claim.
- **One shape**: one token, ten selected experts of 512, 2560 -> 640. This model is dynamically
  quantized, so other layers may differ.
- **Synthetic weights**, not the model's own tensors.
- **One host** (gfx1151).
- The screen's exact-arm figure (~76-77 GB/s) is a min-of-30 micro-benchmark; the trace-derived
  figure for the same kernel is 63.43 GB/s from a profiled capture. They agree in magnitude, and
  neither is a serving-path number.
