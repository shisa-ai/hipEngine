# The dense-wide frontier is exactly 13-47: it passes the gate, beats 14-47 numerically, and is faster

The dense-wide Q8 prefill route delivers **+8.52% prefill** at layers 8-47 but fails
the gate. The frontier was pinned by testing each scope rather than searching
monotonically, because top-1 and mean KL do not rank scopes consistently.

## The gate frontier, all measured with the same harness

`scripts/execution_profile_q8_wmma_prefill_layers_gate.py --route dense_wide
--layers N-47 --decode-steps 128 --repeat-runs 3 --execution-profile production
--prefill-chunk-size 1024 --require-cached-build`

| scope | `kl_mean` (1e-3) | `kl_p95` (5e-3) | `kl_max` (5e-2) | top-1 (0.99) | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 8-47 | 8.839e-4 | 4.647e-3 | 4.146e-2 | 0.99096 | FAIL `general_en` 1.18x |
| 10-47 | 8.370e-4 | 4.291e-3 | 3.554e-2 | 0.98708 | FAIL top-1 + `general_en` 1.08x |
| 11-47 | 7.819e-4 | 3.868e-3 | **6.957e-2** | 0.99225 | **FAIL `kl_max` 1.39x** |
| 12-47 | 6.406e-4 | 3.229e-3 | 3.249e-2 | 0.98966 | FAIL top-1 by one row |
| **13-47** | **6.229e-4** | **3.158e-3** | **2.606e-2** | **0.99354** | **PASS** |
| 14-47 | 5.190e-4 | — | 3.979e-2 | 0.99225 | PASS |
| 16-47 | 4.527e-4 | — | — | — | PASS (default) |

**The frontier is exactly 13-47.** 11-47 and 12-47 fail on different axes (`kl_max`
and top-1 respectively) and 10-47 is worse than 12-47 on top-1, so no monotone
search would have found this.

**13-47 dominates 14-47** on top-1 (0.99354 against 0.99225) and `kl_max` (2.606e-2
against 3.979e-2) despite a marginally higher mean, so the newly-passing scope is
also numerically stronger than the previously-passing one. All four categories
clear their 0.97 floors: code 0.99225, general_en 0.98708, general_ja 1.0,
mixed_ja_en 0.99483.

## Speed: three-arm A/B/C at `code-p4096`, chunk 1024

One process per arm, 1 warmup + 3 repetitions, baseline on both sides:

| arm | scope | prefill tok/s | decode tok/s | samples (pp) |
| --- | --- | ---: | ---: | --- |
| A | 16-47 | 268.30 | 17.955 | 268.37 / 268.21 / 268.33 |
| B | **13-47** | **277.70** | **18.088** | 277.74 / 277.65 / 277.72 |
| C | 16-47 | 272.54 | 17.891 | 272.35 / 272.54 / 272.72 |

Same-code drift A against C is **+1.58%**; the pooled effect is **+2.69% prefill**
and **+0.92% decode**, a signal-to-drift ratio of **1.7x**.

**Why the effect is nonetheless likely real:** arm B is temporally *between* A and
C and exceeds **both** (277.70 against 268.30 and 272.54). A monotone session trend
cannot produce that. So the effect is bounded below by **+1.89%** (against the
higher baseline) and above by **+3.50%** (against the lower), with +2.69% as the
pooled estimate. Within-arm spread is 0.06-0.08%.

**Not claimed as a headline number.** The scope ladder declined 14-47 at a 1.2x
signal-to-drift ratio, and 1.7x is only modestly better; the 8-47 A/B in the same
session measured 0.01% drift, so the 1.58% here is session state rather than
protocol. Tightening it needs more repetitions, not a different design.

## Method notes

- The scope gate **cannot** be run by editing
  `PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS` in place: it raises
  `profile qualification requires a clean committed tracked tree`. The correct
  harness takes the scope as an argument and needs no edit; the invocation is
  recorded in the scope ladder artifact's `provenance.command` field.
- The speed A/B/C **does** use the file constant, set per arm and restored
  afterwards; `git status` confirmed the file unmodified after each run.
- 12-47 was re-run as a harness sanity check and reproduced the ladder's recorded
  figures exactly (`kl_mean` 6.4056e-4 against 6.406e-4, `kl_max` 3.2486e-2
  against 3.249e-2, top-1 0.98966 against 1532/1548).

## Limits

One host (gfx1151 Strix Halo, Radeon 8060S, not the gfx1100 default hardware), one
model (`unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`, UD-Q4_K_XL), one profile, one case
(`code-p4096`), one chunk size (1024), 3 repetitions per arm, 3 candidate repeats
in the gate. The speed claim is one case; the gate is the 12-case canonical
protocol. Not promoted in this artifact.

## End-to-end confirmation after promotion

The speed claim above is one case. With 13-47 as the **committed default**, the
canonical bench at chunk 1024 across the three target shapes:

| case | prefill tok/s | vs 16-47 baseline | llama.cpp | ratio | decode tok/s | decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code-p512 | 272.10 | **+2.6%** | 385.97 | 0.705 | 20.937 | 0.958 |
| code-p1024 | 295.77 | **+3.2%** | 421.60 | 0.702 | 19.972 | 0.906 |
| code-p4096 | 280.59 | **+3.0%** | 421.87 | 0.665 | 18.047 | 0.824 |

Baselines are this session's 16-47 readings at the same chunk (265.33 / 286.64 /
272.38).

**This lands inside the predicted range and validates the both-baselines argument.**
The A/B/C bounded the effect at +1.89% to +3.50% and estimated +2.69%; three
independent shapes measure +2.6%, +3.2% and +3.0%. A monotone session trend could
not have produced a consistent gain across three separate processes.

Decode is essentially unchanged (18.047 against 17.911 at p4096), so this is a
prefill-only gain.

**Not a same-session A/B**, so these are cross-session deltas against this
session's baselines rather than an interleaved measurement. The interleaved A/B/C
in the section above is the controlled version and agrees with it.
