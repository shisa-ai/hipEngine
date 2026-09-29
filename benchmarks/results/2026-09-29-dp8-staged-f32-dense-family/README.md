# Staged-f32 across the dense family: 1.23–1.33× on all three tensors measured

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), ROCm platform 10.0.0, machine-id
`55ea6c509d0b49eea8de7094a1023668`, kernel `7.1.6-1-cachyos`,
`power_dpm_force_performance_level=high`.
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL, Q8_0, rows=1024, layers 0/9/22/34/46
**Artifacts:** `family.json`, `dp8-attn_gate.weight.json`, `dp8-ssm_out.weight.json`
**Companion:** `benchmarks/results/2026-09-29-dp8-staged-f32-attn-qkv/` (the
`attn_qkv` run, whose numbers are re-derived here against the corrected baseline)

## Result

| tensor | shape (out×in) | leaf ms | staged_f32 ms | speedup | time | max abs |
|---|---|---:|---:|---:|---:|---:|
| `attn_qkv` | 10240×2560 | 16.1753 | 13.1121 | **1.2336×** | −18.94% | 1.049e-05 |
| `attn_gate` | 6144×2560 | 9.8217 | 7.3819 | **1.3305×** | −24.84% | 1.335e-05 |
| `ssm_out` | 2560×6144 | 9.2511 | 7.2360 | **1.2785×** | −21.78% | 2.241e-05 |
| **total** | | **35.2481** | **27.7300** | **1.2711×** | **−21.33%** | |

All three are means over the five layers. The prototype is the **fastest variant
in the sweep at every one of the 15 (layer, tensor) cases**.

Note on units: `speedup` is the ratio ("× faster"); `time` is the reduction in
elapsed time. They differ — 1.2336× is 18.94% less time, not 22%. Earlier notes
in this campaign labelled the ratio's excess as a percentage; the two are
separated here.

`baseline_leaf` is `coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out`, the
leaf the engine actually selects. Every number in this packet is against it.

## The extrapolation that does not work

**Do not scale these per-layer-step figures by a layer count to get a prefill
share.** The attempt, and why it fails:

- The three measured tensors total 35.25 ms at rows=1024.
- × 16 exact-chain layers (0–15) = 564 ms.
- × 48 layers = 1692 ms.
- The attribution puts the **entire** `dense_projection` family — every tensor,
  every layer — at **4553.007 ms of a 15741.963 ms prefill**.

Any scaling consistent with that 4553 ms total implies a row count or a
per-step multiplicity that this measurement does not establish. The per-tensor
speedups are solid; **the prefill share is not derivable from this data**, and an
earlier note in this campaign that quoted ~6.8% of prefill was built on exactly
this scaling and should be treated as withdrawn.

## What is and is not established

Established, by measurement:

- The staged-f32 tile is faster than the promoted leaf on three tensors of
  different shapes — one square-ish, one wide-N, one tall-K — across five layers
  each, and is the fastest variant in the sweep at all 15 cases.
- The gain is not shape-specific: 1.23× at 10240×2560, 1.33× at 6144×2560,
  1.28× at 2560×6144.
- `max_abs_diff` is 1.0e-05 to 2.2e-05, `bit_identical = false`.

Not established:

- The family-wide or prefill-wide effect (see above).
- A relative-error figure. The absolute differences are consistent with f32
  summation-order drift and far too small for f16 operand rounding, but no
  relative bound was computed. The "summation-order, not operand precision"
  characterization remains an inference.
- Behaviour on the three unmeasured Q8_0 suffixes: `ffn_gate_shexp` (640×2560),
  `ffn_up_shexp` (640×2560), `ffn_down_shexp` (2560×640).
- Behaviour on natural activations. Activations here are per-layer
  deterministic standard normal × 0.5, which is the right control for comparing
  kernel variants on identical operands, but it is not a real activation range.

## Status

Nothing changed in the engine. The prototype remains unregistered and
unreachable. Promoting it is not a strict change — the output is not
bit-identical — so it needs the `docs/EXECUTION-PROFILES.md` production gate
plus a registered strict fallback. That gate has not been run.

## Reproduction

```
scripts/qwen4exp_dense_projection_ab.py --model-root <UD-Q4_K_XL dir> \
  --tensor-suffix attn_gate.weight --in-features 2560 --out-features 6144 \
  --rows 1024 --repetitions 5 --output /tmp/dp8-attn_gate.weight.json
```

and the same with `ssm_out.weight 6144 2560`. Build environment: TheRock staging
`therock10-staging-20260828`, `HIPENGINE_HIP_ARCH=gfx1151`. The timing boundary is
a hip-event window over `repetitions=5` launches after a warm launch, per
(layer, rows, variant) — a per-kernel microbenchmark, not an engine rate.
