# Staged-f32 across the dense family: all six tensors, best in 30/30 cases

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), ROCm platform 10.0.0, machine-id
`55ea6c509d0b49eea8de7094a1023668`, kernel `7.1.6-1-cachyos`,
`power_dpm_force_performance_level=high`.
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL, Q8_0, rows=1024, layers 0/9/22/34/46
**Artifacts:** `family.json`, `dp8-attn_gate.weight.json`, `dp8-ssm_out.weight.json`
**Companion:** `benchmarks/results/2026-09-29-dp8-staged-f32-attn-qkv/` (the
`attn_qkv` run, whose numbers are re-derived here against the corrected baseline)

## Result

| tensor | shape (out×in) | leaf ms | staged_f32 ms | speedup | time | rel L2 | max abs |
|---|---|---:|---:|---:|---:|---:|---:|
| `attn_qkv` | 10240×2560 | 16.1753 | 13.1121 | **1.2336×** | −18.94% | — | 1.049e-05 |
| `attn_gate` | 6144×2560 | 9.8217 | 7.3819 | **1.3305×** | −24.84% | — | 1.335e-05 |
| `ssm_out` | 2560×6144 | 9.2511 | 7.2360 | **1.2785×** | −21.78% | — | 2.241e-05 |
| `ffn_gate_shexp` | 640×2560 | 1.0028 | 0.7251 | **1.3830×** | −27.70% | 9.116e-07 | 7.629e-06 |
| `ffn_up_shexp` | 640×2560 | 1.0060 | 0.7217 | **1.3939×** | −28.26% | 9.134e-07 | 6.199e-06 |
| `ffn_down_shexp` | 2560×640 | 1.6486 | 0.7440 | **2.2158×** | −54.87% | 4.622e-07 | 1.252e-06 |
| **total** | | **38.9055** | **29.9208** | **1.3003×** | **−23.09%** | | |

All are means over the five layers. The prototype is the **fastest variant in the
sweep at every one of the 30 (layer, tensor) cases**. This is every Q8_0 dense
projection in a layer step — no suffix is left unmeasured.

`ffn_down_shexp` is the standout at **2.22×**: the leaf costs 1.65 ms there for
the same 3.36 GFLOP that costs it 1.00 ms on the K=2560/N=640 shapes, so the
K=640/N=2560 geometry is the leaf's worst case and the staged tile is not
sensitive to it.

### The relative-error figure the earlier packets said was missing

`relative_l2_vs_production` is now recorded: **4.622e-07 to 9.134e-07**, i.e.
4–8 f32 ulp (`eps = 1.19e-07`). That is f32 accumulation-order magnitude.

For comparison, the f16 WMMA sibling route measures relative L2 **4.709e-05** on
the GR shape — roughly **50–100× worse**. This replaces the earlier
"consistent with summation-order drift but no relative bound was computed"
caveat with a measured one.

The field is `null` for the first three tensors because those artifacts predate
it; re-running them would fill it, and the three measured values span only a
factor of two, so the missing three are unlikely to change the conclusion.

Note on units: `speedup` is the ratio ("× faster"); `time` is the reduction in
elapsed time. They differ — 1.2336× is 18.94% less time, not 22%. Earlier notes
in this campaign labelled the ratio's excess as a percentage; the two are
separated here.

`baseline_leaf` is `coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out`, the
leaf the engine actually selects. Every number in this packet is against it.

## The extrapolation that still does not work

**Do not scale the per-layer-step total by a layer count to get a prefill
share.** With all six tensors measured the attempt is at least now complete, and
it still fails:

- the six tensors total **38.9055 ms** per layer step at rows=1024 (29.9208 ms
  staged);
- the attribution puts the **entire** `dense_projection` family at **4553.007 ms
  of a 15741.963 ms prefill**;
- the anchor splits that family into 3262.5 ms at layers 0–15 (exact chain) and
  1404.2 ms at 16–47 (wide route);
- `4553.007 / 38.906 = 117` and `3262.5 / 38.906 = 84`.

Neither 117 nor 84 matches 16, 48, or 192 layer-steps. **The per-step
multiplicity or row count of the profiled forward pass is therefore not
established**, and the prefill share remains underivable from this data. The
earlier ~6.8%-of-prefill figure stays withdrawn.

## What is and is not established

Established, by measurement:

- The staged-f32 tile is faster than the promoted leaf on three tensors of
  different shapes — one square-ish, one wide-N, one tall-K — across five layers
  each, and is the fastest variant in the sweep at all 15 cases.
- The gain is not shape-specific: 1.23× at 10240×2560, 1.33× at 6144×2560,
  1.28× at 2560×6144, 1.38–1.39× at 640×2560, 2.22× at 2560×640.
- `max_abs_diff` is 1.3e-06 to 2.2e-05, `bit_identical = false`.
- Relative L2 is 4.6e-07 to 9.1e-07 where measured — f32 accumulation-order
  magnitude, 50–100× tighter than the f16 sibling route's 4.7e-05.

Not established:

- The family-wide or prefill-wide effect (see above).
- The relative-error figure for the first three tensors (field added later).
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
