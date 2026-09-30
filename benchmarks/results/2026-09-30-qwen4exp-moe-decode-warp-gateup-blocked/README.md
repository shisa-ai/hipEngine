# MoE decode warp256 gate/up: 2.06x faster, blocked on a measured tail regression

**Status:** blocked. **Date:** 2026-09-30. **Host:** gfx1151 (Radeon 8060S), ROCm therock10-staging-20260828.

`HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP=1` selects
`gguf_q4_k_selected_dual_q8_1_dp4a_silu_warp256` for the decode MoE gate/up
matvec, replacing the promoted `..._logical128_t64` dp4a. The candidate is
**2.06x faster** and would close about **2.79 ms/step**, but it is a measured
numerical regression on the max row KL versus strict, so it is **not promoted**.

## Speed: 2.06x, measured at production geometry

`scripts/qwen4exp_moe_gemv_decode_screen.py`, 12 pairs, geometry
hidden 2560 / ffn 640 / experts 512 / top_k 10:

| arm | median | traffic | bandwidth | ref sha256 | drift vs dp4a |
| --- | ---: | ---: | ---: | --- | --- |
| `gate_up_dp4a` (promoted) | 107.416 us | 18,432,000 B | 171.594 GB/s | `687619302c66533c` | reference |
| `gate_up_warp256` | **52.093 us** | 18,432,000 B | **353.831 GB/s** | `60ec690de7315d53` | max 3.0517578125e-05, p999 0.0 |

The drift is one bf16 ulp at the maximum and zero at p999. The dispatch places
the warp256 inside the same `use_dp4a` branch, so it covers the same 43 of 48
layers and shares the activation quantize with the arm it replaces.

Against DE-1's own end-to-end traffic model (888.83 MB/step; the dp4a part is
5.422 ms/step of the 47.359 ms/step decode kernel total), 5.422 / 2.062 =
2.630 ms/step, so **2.792 ms/step saved, about 5.9% of decode kernel time**.

### DE-1's comparator was not a roof

DE-1's `headroom_closure` recorded `promoted_pct_of_comparator: 89.1` against a
184.66 GB/s comparator and concluded 93.7% of headroom was consumed, leaving
0.591 ms/step. The warp256 reaches 353.83 GB/s, **1.92x above that comparator**,
so it was not an upper bound and the remaining headroom was understated roughly
five-fold.

## Why it is blocked: 4.34x worse max KL versus strict

`scripts/qwen4exp_moe_decode_warp_logits_probe.py --packet dual --teacher`, the
18-case admission fixture, 1494 rows, comparison
`strict teacher (production-versus-strict, 6.1)`:

| metric | candidate | incumbent | bar |
| --- | ---: | ---: | ---: |
| mean KL | 0.0011640 | 0.0010539 | <= 0.001 |
| p95 KL | 0.0035360 | 0.0037318 | <= 0.005 |
| p99 KL | 0.0195653 | 0.0242993 | <= 0.02 |
| max KL | **0.2611882** | 0.0784897 | <= 0.05 |
| top-1 | 0.99331 | 0.99465 | >= 0.99 |

The candidate is better on p95 and p99, but **1.10x worse on mean** and
**3.33x worse on the max row**. The whole tail is four `mixed_ja_en` cases; the
other fourteen are all below 0.0094 and the two arms track each other there.

| case | candidate max | incumbent max | ratio | cand top-1 | inc top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `heldout_mixed_review` | 0.26119 | 0.06017 | **4.34x** | 0.9880 | **1.0000** |
| `mixed_ja_en_review` | 0.19023 | 0.05364 | **3.55x** | 1.0000 | 1.0000 |
| `heldout_mixed_summary` | 0.04060 | 0.07849 | 0.52x | 0.9880 | 0.9880 |
| `mixed_ja_en_translate` | 0.03594 | 0.04494 | 0.80x | 1.0000 | 1.0000 |

On `heldout_mixed_review` the candidate spends a top-1 flip that the incumbent
does not make.

## Two secondary findings

**The 18-case suite is failed by production too.** The incumbent exceeds the
0.05 ceiling on three cases (0.07849, 0.06017, 0.05364). It is a stress suite,
not an admission basis, and its bars do not discriminate between the two arms.
The 2026-09-10 "rejected at admission suite" verdict was reached on a basis the
shipped default also fails.

**The 996-row qualification was narrower than its wording.** The packet cited by
the flag's docstring and by `docs/EXECUTION-PROFILES.md` 6.1 used the canonical
12 cases, which do not include `heldout_mixed_review`. "All bars pass with
2.6-4.4x margin" was true of that fixture and does not extend to the heldout
set.

## The down leg, measured in the same screen

`HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_DOWN=1` (plain warp256 Q5_1 selected
weighted-sum) is **provably bit-exact** but **slower**, so it is also not
promoted.

| arm | median | traffic | bandwidth | ref sha256 | drift |
| --- | ---: | ---: | ---: | --- | --- |
| `down_wsum` (incumbent) | 104.727 us | 12,288,000 B | 117.334 GB/s | `c5f8d1188aedc413` | reference |
| `down_warp256` | 121.313 us | 12,288,000 B | 101.292 GB/s | `c5f8d1188aedc413` | **max 0.0, p999 0.0** |

The 18-case `--packet down` probe confirms it end to end: `kl_mean`, `kl_p95`,
`kl_p99` and `kl_max` are all exactly `0.0`, `top1_overall` is `1.0`, and
`exact_equal` is `True` on all 18 cases. The source reading is confirmed: the
warp256 reproduces the incumbent's reduction tree (`S_t = (L_t + L_{t+128}) +
(L_{t+64} + L_{t+192})`, `T_t = S_t + S_{t+32}`, the same `shfl_down` 16..1
tree, the same per-row bf16 round and the same sequential `fmaf` fold), and
`dequant_q5_1`'s `d*q+m` contracts to the same `fmaf`.

It is 13.7% slower because `__launch_bounds__(384, 2)` reserves 12 warps for one
warp per selected row while the production top-k is 10, so 2 of 12 warps idle.
Widening is the wrong direction for this kernel.

## Reproduction

```bash
# speed, both legs (production geometry, real weights)
python3 scripts/qwen4exp_moe_gemv_decode_screen.py \
  --model-root <qwen4kxl-root> --compiler-version-file <hipcc-version.txt> \
  --pairs 12 --output /tmp/r11-moe-gemv-screen.json

# candidate vs incumbent, 18 cases
python3 scripts/qwen4exp_moe_decode_warp_logits_probe.py \
  --model-root <root> --compiler-version-file <f> \
  --fixture benchmarks/fixtures/qwen4exp_admission_suite_18prompt.json \
  --decode-steps 82 --packet dual --output /tmp/dual-inc.json

# strict teacher, then both arms versus strict
python3 scripts/qwen4exp_moe_decode_warp_logits_probe.py ... \
  --decode-steps 82 --emit-teacher /tmp/teacher.npz --output /tmp/teacher-emit.json
python3 scripts/qwen4exp_moe_decode_warp_logits_probe.py ... \
  --decode-steps 82 --packet dual --teacher /tmp/teacher.npz \
  --output /tmp/dual-vs-strict.json
```

The `down` packet used by the bit-exactness proof is the one addition this
session made to `PACKETS` in the probe.
