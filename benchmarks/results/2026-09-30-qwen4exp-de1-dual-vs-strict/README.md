# DE-1 dual packet vs the strict teacher: the tail belongs to the candidate, and the suite fails both arms

**Date:** 2026-09-30 UTC
**Host:** AMD Radeon 8060S, **gfx1151** (not the gfx1100/W7900 default hardware), ROCm platform 10.0.0, hostname `gfx1151`, machine id `55ea6c509d0b49eea8de7094a1023668`
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL (4 shards), `qwen4exp` route
**Profile:** `ExecutionProfile.PRODUCTION`
**Command:** `scripts/qwen4exp_moe_decode_warp_logits_probe.py --packet dual --teacher`
**Fixture:** 18 cases / 4 categories / 1494 rows, `decode_steps` 82
**HEAD at run:** `cceed7336f0aded9a6a88cb77a9632dcae16eaa0` (`tracked_clean: false` — untracked
benchmark artifacts only; no tracked file differed, and no kernel/default/flag changed)
**Artifact:** [`dual-vs-strict.json`](dual-vs-strict.json)

## Why this run exists

DE-1's `warp256` gate/up candidate is **2.06x faster** (107.416 us / 171.594 GB/s ->
52.093 us / 353.831 GB/s) and recomputes to **2.792 ms/step = ~5.9%** of the decode kernel
total, but stayed default-off because an earlier run measured it **4.34x worse on max KL**
than the incumbent on `heldout_mixed_review`. The open question was which arm owned that
tail. Every prior measurement compared the candidate *against the incumbent*; this run
compares **both arms against the strict teacher** in one process, which is the only basis
that can answer it.

## Result 1 — the tail belongs to the candidate

| case | cand vs inc | inc vs strict | cand top1 | inc top1 |
| --- | --- | --- | --- | --- |
| `heldout_mixed_review` | **0.26119** | 0.06017 | 0.9880 | 1.0000 |
| `mixed_ja_en_review` | **0.19023** | 0.05364 | 1.0000 | 1.0000 |
| `heldout_mixed_summary` | 0.04060 | **0.07849** | 0.9880 | 0.9880 |

The candidate's worst case is **4.34x worse** than the incumbent's own error against strict
on the same case, and **3.55x worse** on `mixed_ja_en_review`. The regression is real and
does not belong to the incumbent.

## Result 2 — the flip is not a near-tie

The single top-1 flip on `heldout_mixed_review` is at **step 63 of 83**, and it is a large
margin, not a coin flip:

```
step 63: margin 0.954472   candidate KL 0.261188   incumbent KL 0.039213
```

The candidate is **6.7x worse** at that step, with a **0.954** margin on the flipped token.
The minimum step margin anywhere in that case is **0.1076**, so no step in the case is a
near-tie. This **corrects an earlier reading** that described the flips as near-ties with
min top-5 overlap 3-4 of 5 and min margin 0.0335-0.1686; that came from a different run and
does not describe this one.

## Result 3 — the suite is a stress suite, and it fails BOTH arms

The 18-case envelope against strict, with the binding bars:

| metric | bar | candidate | incumbent |
| --- | --- | --- | --- |
| `kl_mean` | 0.001 | **0.001164 FAIL** | **0.001054 FAIL** |
| `kl_p95` | 0.005 | 0.003536 pass | 0.003732 pass |
| `kl_p99` | 0.02 | 0.019565 pass | **0.024299 FAIL** |
| `kl_max` | 0.05 | **0.261188 FAIL** | **0.078490 FAIL** |
| `top1_overall` | 0.99 | 0.993307 pass | 0.994645 pass |
| `top1_by_category` | 0.97 | 0.99197-0.99398 pass | pass |

The incumbent exceeds the 0.05 max-KL ceiling against strict on **3 of 18** cases
(`mixed_ja_en_review` 0.05364, `heldout_mixed_summary` 0.07849, `heldout_mixed_review`
0.06017) and fails `p99` as well. **Neither arm passes this suite**, so its bars cannot
discriminate between them and it is not an admission basis for anything. This is the same
conclusion an earlier run reached on the canonical 12-case fixture (where the incumbent
failed `mean` 0.0012761 and `max` 0.0525692 while the candidate was equal on mean 0.0012891
and better on max 0.0477176), now reproduced on the larger 18-case suite.

## Result 4 — the candidate's regression is localized to `max` only

Relative to the incumbent against the same teacher, the candidate is:

- **worse on `max`**: 0.261188 vs 0.078490 (**3.33x**)
- **worse on `mean`**: 0.001164 vs 0.001054 (**1.10x**, and both fail the 0.001 bar)
- **better on `p99`**: 0.019565 vs 0.024299 (0.81x)
- **better on `p95`**: 0.003536 vs 0.003732 (0.95x)
- **worse on `top1_overall`**: 0.993307 vs 0.994645 (both pass 0.99)

So the candidate is **not uniformly worse**. Its entire envelope failure is the single
step-63 outlier on one case, which is exactly the shape the row's remaining action targets:
*"localize which layer and expert transition flips and test a narrower layer scope like
`HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS`."*

## What this establishes and what it does not

**Established:** the tail is the candidate's, the flip is a wide-margin flip rather than a
near-tie, the suite fails both arms so its bars do not discriminate, and the candidate's
regression is confined to `max` on one step of one case.

**Not established:** that the candidate can be promoted by narrowing the layer scope — that
requires the layer localization this run sets up. **No default, flag or dispatch changed**
(`runtime_default_changed: false`).

## Limits

One model, one quant, one host, one execution profile, one fixture family. `decode_steps`
82-83 per case. The run was on a tree with untracked artifacts present (`tracked_clean:
false`); no tracked file differed from HEAD and nothing that affects dispatch was modified.
The candidate's speed figure (2.06x / 2.792 ms/step) is a **microbenchmark** projection onto
DE-1's traffic model, not an end-to-end measurement, and this run makes no speed claim.
