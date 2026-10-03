# Flash-Next prefill baseline at the current default

**Measured 2026-10-04 on gfx1151 (AMD Radeon 8060S, Strix Halo APU, machine
`55ea6c509d0b49eea8de7094a1023668`), commit `89264f901` + the DPP leaf from
`e6a99f446`.** Model `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL` (UD-Q4_K_XL), BF16 KV,
production execution profile, the canonical twelve cases, 1 warmup and 3 measured
repetitions each.

This replaces the `185.2 / 190.6 / 183.6` row in `benchmarks/README.md` as the
current-default prefill reference. That row is at `ddfc2a746` (2026-09-15), and
`git show ddfc2a746:hipengine/generation/qwen4_exp_profiles.py` contains no
`PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS` - the 6.74x wide-row Q8 route was
certified on 2026-09-17, two days later, so the published row prices the exact
coltile chain on all 48 layers.

## Result

Prompt tokens per second, total prompt tokens over total prompt time.

| Length | code | general_en | general_ja | mixed_ja_en | mean | published `ddfc2a746` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 299.06 | 300.42 | 302.70 | 298.74 | **300.23** | 185.2 |
| 1024 | 294.30 | 299.85 | 300.95 | 294.50 | **297.40** | 190.6 |
| 4096 | 262.30 | 268.36 | 269.12 | 263.74 | **265.88** | 183.6 |

Within-case spread is under 1% across the three repetitions; the run reports
`all_cases_deterministic: true` over 36 samples.

Decode at 512 is 20.41 tok/s on every category (19.70 at 1024 and 4096).

## What this means for the gap

Against the fastest same-file engine in that table, halo-box PR #63 head
`c4aa30229` at BF16 (479.4 / 860.8 / 1051.6, weighted 915.4):

| Length | hipEngine now | halo-box PR #63 | gap |
| ---: | ---: | ---: | ---: |
| 512 | 300.2 | 479.4 | **1.60x** |
| 1024 | 297.4 | 860.8 | **2.90x** |
| 4096 | 265.9 | 1051.6 | **3.96x** |

**The gap grows with prompt length, and that is the useful part of this
measurement.** A fixed per-token deficit would give a constant ratio. Converting to
wall time and taking marginal cost between the measured points:

| Interval | hipEngine ms/token | halo-box PR #63 ms/token | ratio |
| --- | ---: | ---: | ---: |
| 512 -> 1024 | 3.43 | 0.24 | 14.5x |
| 1024 -> 4096 | 3.98 | 0.88 | 4.5x |

So this engine's marginal prefill cost is **3.4-4.0 ms/token** where the
comparator's is **0.24-0.88**, and the reason the ratio at 512 looks comparatively
flattering is that the comparator pays a large fixed cost there (1068 ms for 512
tokens) while this engine does not.

**Neither side is linear.** The comparator's own marginal cost rises 3.7x between
the two intervals, so it carries a quadratic term too; ours rises only 1.16x. The
difference is in the constant, not in the exponent. Fitting `a*n + b*n^2` to the
three points gives `a = 3.34 ms/token` and `b = 6.0e-5 ms/token^2`, so the
quadratic term is about 1000 ms of the 15702 ms 4K prefill, 6.4% - real, and not
what closes a 3.96x gap.

## Where the remaining time is

The 2026-09-16 per-role capture (`results/2026-09-16-flashnext-per-role-cost/`)
prices a 22046 ms `code-p4096` prefill with 10553 ms of `linear:` roles on the exact
chain at all 48 layers. Correcting only the dense rows for the now-live wide route
(32 of 48 layers at the measured 6.74x) predicts 4563 ms of `linear:` and a 15956 ms
prefill against a measured 15986 ms for this case in the probe run - a 0.4% closure,
which is what licenses using the old capture's shape with corrected dense rows:

| family | corrected ms | share |
| --- | ---: | ---: |
| `moe` (gate+up 2710, down 1588, **repair 1370**, routing/scatter 931) | 6499 | **40.6%** |
| `linear` (dense projections) | 4563 | 28.5% |
| `gr_read` (pure gather, 2.3x the matmul it feeds) | 2539 | 15.9% |
| `qsa_prefill` | 1421 | 8.9% |
| `gdn` | 824 | 5.2% |
| boundary + ple | 110 | 0.7% |

The dense-projection lever is not exhausted either: layers 0-15 remain on the exact
chain because the f16 route "fails the calibrated envelope at 0-47 and passes at
16-47", so what remains there is a numerics question, not a kernel question.

## Limits

One host, one GPU architecture (gfx1151, not the gfx1100 default hardware), one
model file, one quant, one execution profile, BF16 KV, warm PLE cache. The
comparator rows are from a different session on the same host and the same file;
they are quoted for ratio, and the two protocols are not identical. The per-role
attribution is a 2026-09-16 capture with only its dense rows corrected by
arithmetic on a different run - a model that closed to 0.4%, not a fresh capture.
No kernel, default or flag changed to produce this row.

## Reproduce

```bash
python3 scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --execution-profile production --warmups 1 --repetitions 3 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --output /tmp/canonical-now/hipengine-current.json
```
