# DP-8's staged 13-15 scope is worth ~0.6% of prefill, measured

DP-8's composition (staged f32 on layers 13-15, the wide route on 16-47) passes
every hard gate, but its **value** had only ever been an estimate:

> `3 layers x 8.985 ms per layer-step x 4 chunks of 1024 rows = ~0.66% of prefill`

That derivation is the one thing the source artifact explicitly forbids. Its own
`extrapolation_warning` reads:

> Do NOT scale the per-layer-step total by a layer count to get a prefill share.
> ... 4553.007/38.906 = 117 and 3262.5/38.906 = 84, neither of which matches 16,
> 48 or 192 layer-steps. The per-step multiplicity or row count in the profiled
> forward pass is therefore NOT established, and the prefill share remains
> underivable from this data.

This packet measures the share instead. Two independent runs, one protocol.

## What was measured

One resident generator, one 4096-token prompt, chunk 1024, production profile.
Both arms run the shipping configuration - the profile binder writes
`HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS=""` and `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE=1`
over layers 16-47 itself - and the arms differ by exactly one selector,
`HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS`.

| field | value |
| --- | --- |
| physical host | `gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`, `Desktop (AMD Ryzen AI Max 300 Series)` |
| model / quant | `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL`, `ud_q4_k_xl` |
| backend / arch | `hip_gfx1151` / `gfx1151` |
| execution profile | `production` |
| prompt | 4096 tokens (id 9707), prefill chunk 1024 |
| arms | `baseline` = exact chain on 13-15; `staged_13_15` = `staged_f32_bm128_bn64_bk32` |
| protocol | 2 warmups + 9 repetitions per arm, alternating arm order |
| command | `bash scripts/run_qwen4exp_dp8_staged_scope_prefill_ab.sh` |

Run 1 was taken at `2cca23c6c`; run 2 at `84290f3dd`. Both artifacts record
`provenance.dirty: true`, and the reason is the same for both: the harness's
engagement instrument was repaired in this unit (run 1) and then given the host
and command fields the evidence policy requires (run 2). Run 2's only
uncommitted difference is `_provenance()`, which reads host files before any
prefill and never enters the measured region. Run 1 predates the field, so its
host is established from the machine-id above and its own
`target_arch`/`resolved_backend`; both runs are on the same host.

## Engagement: exact, and the arms differ by one variant swap

This packet family has produced a false pass before - bit-identical arms, KL
exactly zero, `measurement_valid: true` - so engagement is asserted, not
inferred from a KL. Both runs report the identical census:

| launched variant | baseline | staged_13_15 |
| --- | ---: | ---: |
| `coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out` | 4824 | **3924** |
| `staged_f32_bm128_bn64_bk32` | **0** | **900** |
| `f32_hidden_f32_out` | 8640 | 8640 |
| `pack8_gemv_f32_f32_out` | 27 | 27 |

Three things follow, and all three are checks rather than commentary:

- **The replacement is one-for-one.** `4824 - 3924 = 900`, exactly the staged
  launch count. The staged tile took over the exact-chain leaf and nothing else.
- **The scope is exact.** The staged launches are on layers 13, 14 and 15 only
  (288 / 288 / 324); no other layer resolves it, and layers 13-15 have no
  exact-chain launches left in the candidate arm.
- **Nothing else moved.** The two variants the selector cannot claim are
  identical in both arms, so the measured delta is the swap and not a
  re-dispatch.

`uncounted_cache_entries_by_arm` is 0 in both arms and both runs, so no launch
escaped the counter.

## Result

Per-repetition paired difference (candidate minus baseline; negative is a
saving). The host stepped its level mid-run in **both** runs - baseline
repetitions 0-3 median 15.13 s, repetitions 4-8 median 17.83 s, **+18%** - so
repetitions are grouped by the regime both arms shared.

| estimator | samples | value | share of the 15742 ms X-3 prefill anchor |
| --- | ---: | ---: | ---: |
| both-fast regime, mean | 7 | **-96.1 ms** (SE 13.5) | **0.61%** |
| both-fast regime, median | 7 | -92.7 ms | 0.59% |
| pooled median | 18 | -89.5 ms (95% CI [-171.1, -56.0]) | 0.57% |
| run 1 median | 9 | -114.0 ms | 0.72% |
| run 2 median | 9 | -72.5 ms | 0.46% |
| run 1 median-of-arms | 9 | -145.5 ms | 0.92% |
| run 2 median-of-arms | 9 | -62.8 ms | 0.40% |

**17 of 18 paired differences are negative; all 7 same-regime fast samples are
negative.** The retained figure is **~90 ms, 0.6% of prefill**, with the
estimator spread 63-146 ms.

Two samples straddle the level step and are reported rather than dropped:
run 1 repetition 6 (`-1106.8 ms`, a transient inside the slow regime) and run 2
repetition 3 (`-2317.9 ms`, where the step landed *between* the two arms -
candidate fast at 14.89 s, baseline slow at 17.20 s). They are the reason the
median is quoted instead of the mean.

**The estimate was right.** The ledger's 107.8 ms (0.685%) sits inside this
range: run 1's median is 6% above it, the pooled median 17% below. What the
measurement changes is not the number but its standing - the estimate rested on
the same 4-chunk multiplicity that the source artifact's `extrapolation_warning`
declares unestablished, and the direct measurement replaces it rather than
confirming its derivation. The warning's own inconsistency is untouched and
still open: 38.906 ms per layer-step x 48 layers x 4 passes = 7469.9 ms, 1.64x
the 4553.007 ms family anchor it would have to fit inside, and this measurement
does not say where that 1.64x lives.

## What this does not establish

- **Not a promotion.** The composition's gate passes every hard gate but leaves
  `eligible_for_automatic_admission: false` (2 rows over the 0.02 KL review
  boundary) and `qualification_blockers` still lists category-heldout,
  strict-teacher/BF16-relative, control/isolation, task-quality and
  repair-oracle. No default, flag or dispatch changed here.
- **Not a correctness claim.** This harness measures wall time. The staged
  tile's arithmetic is qualified by the composition gate, not by this packet,
  and the identical `token_id` in both arms is weak evidence because the prompt
  is one repeated token.
- **Not a decode or serving claim.** Prefill only, one shape, one prompt, one
  model, one quant, one profile, one host.
- **Not the route's whole value.** Layers 13-15 is the deepest passing scope;
  16-47 also passes numerically but is 5-10x slower than the wide route already
  promoted there, so it has no marginal value. The 0-15 region, which is where
  the exact chain still runs and the value would be, fails the envelope.

## Artifacts

- `artifact-staged-13-15-prefill-ab.json` - run 1, `head 2cca23c6c`, the
  wrapper's default output path. It predates the host and command fields, so its
  host is established from the machine-id above and its own
  `target_arch` / `resolved_backend`.
- `artifact-staged-13-15-prefill-ab-run2.json` - run 2, `head 84290f3dd`, with
  the host and command fields.
- Composition gate:
  `benchmarks/results/2026-09-29-q8-staged-f32-dense-prefill-layers-gate/`.
- Source of the retired estimate:
  `benchmarks/results/2026-09-29-dp8-staged-f32-dense-family/family.json`.
