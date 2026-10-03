# The exact-iu8 risk multiplier is a 2.11% prefill dial

**Measured 2026-10-04 on gfx1151** (AMD Radeon 8060S, Strix Halo APU, machine
`55ea6c509d0b49eea8de7094a1023668`), model
`unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL` (UD-Q4_K_XL), production profile,
`code-p4096`. Harness `scripts/qwen4exp_profile_gap.py --mode prefill
--risk-diagnostics`; raw artifacts `risk-mult-4.0.json`, `risk-mult-1.0.json`,
`risk-mult-0.5.json` in this directory.

## Question

Reattributing the prefill put the routed MoE block at 40.6% of it, and inside that
a **1370 ms repair pass, 8.6% of the whole prefill**. The pass serves a Kahan-bound
criterion whose multiplier, `HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT`, defaults to
**4.0** while its own docstring records a **screened floor of 0.5-1** and a
">=4x margin" as the reason for 4.0. How much the pass is over-inclusive had never
been measured.

## Result

1504 `q4_iu8_exact:expert_gate_up` calls and 9,856,614,400 outputs per point, three
wall repetitions each:

| `RISK_MULT` | risk rows | repair rate | max rate | over-capacity | wall (s, 3 reps) | tok/s | vs 4.0 |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| **4.0** (default) | 70,167,220 | **0.7119%** | 0.8057% | 0 | 17.0430 / 17.0034 / 16.9939 | 240.75 | - |
| 1.0 | 20,621,412 | **0.2092%** | 0.2389% | 0 | 16.6418 / 16.6523 / 16.6723 | 245.92 | **-2.11%** |
| 0.5 | 11,074,348 | **0.1124%** | 0.1281% | 0 | 16.5641 / 16.5645 / 16.5647 | 247.27 | **-2.64%** |

Within-point spread is 0.2-0.3% at 4.0 and 1.0 and 0.004% at 0.5, so the deltas are
about 7x and 100x their own noise. `over_capacity_calls` is **0 everywhere**, so the
repair buffer is never the constraint and the incidence is the whole story.

**The rate is linear in the multiplier and the wall is not.** 4.0 -> 1.0 cuts the
repair incidence **3.40x** but the wall only **2.11%**; 4.0 -> 0.5 cuts it **6.33x**
for 2.64%. On the recorded 1370 ms share, a proportional repair pass would have
saved 968 ms (5.7% of the prefill) at 1.0 against a measured 358 ms. **So the repair
kernel carries a large component that does not scale with the flagged-row count** -
it is launched over `risk_capacity = compact * 2 * ffn` regardless of `risk_count`.
That is a separate, still-open lever and it is why the multiplier alone cannot
recover the whole 1370 ms.

## Status

**Not promoted.** The win is cheap - one environment constant, no kernel change, no
owner change - but it decides which outputs the exact repair serves, so it is an
arithmetic change and needs the production gate, not the wall. The gate is
`scripts/qwen4exp_layer2_profile_gate.py --candidate q4_iu8_risk_mult_1`.

The prior evidence is genuinely two-sided and both halves matter:

* `repair_free_gate_up` (`2026-09-30`) removed the repair entirely and was
  **rejected** on per-scope numerics: `prefill_last` / `prefill_to_c1`, 18 rows,
  mean KL **2.159e-3 against a 1e-3 limit**, p95 7.776e-3 against 5e-3, p99 2.131e-2
  against 2e-2, every aggregate gate passing. At a 0.71% incidence, per-element
  bounded drift accumulates across 48 layers and lands on the prefill path.
* That experiment repaired **nothing**, which is below the screened floor. This
  candidate repairs **29% of what production repairs** and keeps the criterion,
  which the docstring's own screening places inside the range where BF16 flips are
  still caught. It is a different change from the rejected one.

## Limits

One host, one GPU architecture (gfx1151, not the gfx1100 default hardware), one
model file, one quant, one prompt case, production profile. **The wall here is this
harness's prefill wall (17.014 s at the default) and is not comparable to the
canonical-benchmark row in `../2026-10-04-qwen4exp-prefill-baseline-current/`**
(16.0 s, different harness and warmup protocol); only the three points within the
table above are mutually comparable. The 1370 ms share is from a 2026-09-16 capture
and is used only to show the wall saving is below a proportional model.

## Reproduce

```bash
for MULT in 4.0 1.0 0.5; do
  HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT=$MULT python3 scripts/qwen4exp_profile_gap.py \
    --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
    --mode prefill --case-id code-p4096 --execution-profile production \
    --risk-diagnostics --output /tmp/risk-mult-$MULT.json
done
```
