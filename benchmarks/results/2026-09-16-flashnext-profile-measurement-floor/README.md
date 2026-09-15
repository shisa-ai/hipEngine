# Flash-Next Prefill Profile: Run-to-Run Measurement Floor

Three captures of the **same** 4096-token `code-p4096` prefill at the production
default, on Framework `gfx1151` (machine `55ea6c509d0b49eea8de7094a1023668`,
AMD Radeon 8060S), 2026-09-16. Two use the default configuration; the third adds
`HIPENGINE_QWEN4_EXP_PLE_WARM=1`.

The point of the third run was to check whether warm PLE changes the measured
prefill. It does not, and the exercise produced something more useful: a
quantified noise floor for any per-component comparison built on these profiles.

## The three runs

| Run | Configuration | Attributed | Kernels | (role, kernel) rows |
| --- | --- | ---: | ---: | ---: |
| cold-A | default | 22367.5 ms | 9652 | 2007 |
| cold-B | default | 22052.9 ms | 9652 | 2007 |
| warm-PLE | `PLE_WARM=1` | 22560.5 ms | 9652 | 2007 |

All three attribute 100% of kernels with 0 ms unattributed, and all three
contain the identical set of 2007 (role, kernel) rows. The configuration
difference is therefore not visible in the kernel set — only in the times.

## Warm PLE does not change the prefill

`HIPENGINE_QWEN4_EXP_PLE_WARM=1` populates the PLE table into page cache so that
**decode-step** row gathers hit warm pages. Its own source comment records the
cost: one mmap-touch sweep of ~28.8 GB, **+15.4 s and ~6,015 major faults per
runner construction**, amortized over 577–15,400 generated tokens (median
~2,050). A prefill-only profile generates no tokens, so it pays the construction
cost and receives none of the benefit.

It also does not perturb the prefill. Comparing cold-A vs cold-B (both default)
against cold-B vs warm-PLE:

| Comparison | net | scatter | median | stdev | range |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold-A vs cold-B | −314.6 ms | 544.1 ms | −0.33% | 5.09% | −32.0% … +35.8% |
| cold-B vs warm-PLE | +507.6 ms | 752.0 ms | +1.03% | 5.15% | −25.6% … +39.9% |

The scatter is the same whether or not the flag is set, and the net delta
changes sign between comparisons of the same configuration. **The flag is not
the source of the variance and does not explain the totals.** The three runs span
22053–22561 ms with no ordering that tracks it.

## The noise floor

For rows at or above 20 ms — 391 of 2007 rows, carrying 78% of the attributed
time — the per-(role, kernel) difference between two runs of the same
configuration is:

- **median −0.3%**, **stdev 5.1%**, range **−32% to +36%**.

Family aggregates are far more stable: `linear` moved +1.7%, `moe` +2.1%,
`qsa_prefill` +0.5% between cold-A and cold-B, against a 1.4% move in the total.

**So a per-component comparison is interpretable at the family level and not at
the individual-(role, kernel) level.** Any "recoverable ms" figure quoted per
component from single runs is inside this scatter.

## The scatter is not averaging noise

Random measurement noise shrinks as a row gets bigger. Here it grows:

| Row size | rows | median | stdev |
| --- | ---: | ---: | ---: |
| 20–50 ms | 233 | −0.5% | **1.3%** |
| 50–100 ms | 154 | −0.2% | **7.5%** |
| 100–300 ms | 5 | −3.8% | 14.4% |

That is the opposite of averaging behaviour and indicates a systematic effect —
clock or memory-state drift across the run — rather than independent per-launch
noise. The effect is not identified here.

The 100–300 ms band has only five rows, all `linear:layers.N.attn_q` (the
`N=12288` QSA query projection, the largest single projection in the prefill),
and they disagree with each other:

| Row | cold-A | cold-B | warm-PLE | cold-A→B | cold-B→warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `layers.15.attn_q` | 139.1 | 94.6 | 95.3 | −32.0% | +0.7% |
| `layers.47.attn_q` | 125.1 | 120.3 | 95.4 | −3.8% | −20.7% |
| `layers.31.attn_q` | 121.8 | 112.2 | 124.2 | −7.9% | +10.7% |
| `layers.27.attn_q` | 118.0 | 114.3 | 114.2 | −3.2% | −0.1% |
| `layers.23.attn_q` | 113.7 | 128.0 | 95.3 | +12.6% | −25.5% |

A median across these five rows moves with whichever run happened to be the
outlier, which is why the warm-PLE comparison first appeared to show a 15%
speedup on large rows. It does not survive inspection: `layers.15` swings −32%
then +0.7%, and `layers.23` swings +12.6% then −25.5%. No warm-PLE effect is
established.

## What this means for gap attribution

1. Attribute at the **family** level. The family aggregates move by ~1–2%
   between runs; individual rows move by ~5% with tails to 40%.
2. Where a per-component number is needed, take **repetitions and use medians**.
   A single capture cannot support one.
3. Do not read the ±15% on the five largest rows as signal without more samples.
   At n=5 with one outlier the median is not a statistic.
4. The prefill total itself carries ~1.4% run-to-run spread across these three
   captures, so a claimed prefill improvement below that needs a repeated A/B,
   not two single runs.

## Reproducing

```bash
python3 scripts/qwen4exp_profile_repeatability.py \
    --analysis cold-a.json cold-b.json warm.json \
    --labels cold-A cold-B warm-PLE \
    --output benchmarks/results/2026-09-16-flashnext-profile-measurement-floor/repeatability-4k.json
```

The three role-analysis inputs come from
`scripts/qwen4exp_role_analyze.py` over role-marked `rocprofv3` captures of the
same fixture case.

Diagnostic only. No performance claim is retained and no hipEngine rate changed.
