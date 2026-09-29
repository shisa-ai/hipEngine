# The `wmma` route produced a false pass at HEAD

**Status: cause established and fixed.** This file has been revised twice; the
history is kept rather than edited away because the first revision published a
wrong cause in a worklog entry and a ledger row.

## Symptom, reproduced twice

`--route wmma --layers 16-47` (the certified configuration) reported:

```
strict_logits_sha256    = 83732d4542e1847fd8975ca6...
candidate_logits_sha256 = 83732d4542e1847fd8975ca6...
```

Identical. KL exactly `0.0` on every statistic, `top1 1.0`, and
**`status: complete`, `measurement_valid: true`, `requires_outlier_review: false`**.

The 2026-09-16 certification of the same route had a genuine exact-chain teacher
(`550bb9b8...` against candidate `65370710...`), mean KL 3.796e-4. So this was a
regression, and it meant **the 16-47 certification was not reproducible at HEAD**:
a re-run returned KL exactly 0.0 and a green status.

## Cause

The `wmma` route cleared **only** its own selector. It never cleared
`HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE`, which the certified plan binds to 16-47 -
the same window this route is gated at. `_q8_dense_wide_claims_parent` accepts
**both** the exact parent (`raw`) and the WMMA-rewritten parent (`wmma_<parent>`
under ABI `wmma_raw`), and the wide route runs later in the launch chain. So it
claimed the dispatch whichever way the WMMA selector was set, both arms resolved
to `dense_wide256_f32_f32_out`, and the gate compared one configuration against
itself.

The reciprocal clearing was already understood and documented - the
`dense_wide` route clears the WMMA selector in both arms, and the `Route`
docstring explains exactly why ("the two routes would contend for the window
under test"). It was simply never added in the other direction.

## Confirmed by elimination, not assumed

`83732d45...` is the wide route's output, and the elimination is what shows it:

| configuration | teacher | candidate |
|---|---|---|
| `wmma` route (wide bound, WMMA cleared vs 16-47) | `83732d45` | `83732d45` |
| composed route (wide **cleared**, WMMA 16-47 + staged 13-15) | `550bb9b8` | `87dbf445` |

With the wide route bound, the two arms differ only in the WMMA selector and
produce identical logits - so the selector is provably inert. With it cleared,
the WMMA arm produces `87dbf445...` against an exact-chain teacher of
`550bb9b8...`, so WMMA does change the arithmetic when it is the only route
bound. The artifact's own recorded env settles it directly: the composed run's
provenance carries `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE: '0'`, and the `wmma`
route's carries no wide key at all.

## A wrong cause was published first, and is withdrawn

The first revision named the **dispatch memoization key** as the verified cause.
That was wrong. The key genuinely lacked a WMMA layer-scope term - a real latent
defect, fixed in `a7ad30e5f` with a RED test, and that fix stands - but the false
pass reproduced **unchanged** with the term present
(`artifact-control-wmma16-47-vs-exact-fixed.json`). Establishing that a suspected
cause is *present* is not establishing that it is *the* cause.

## Fix

The `wmma` route now clears `DENSE_WIDE_ENV` and `DENSE_WIDE_LAYERS_ENV` in both
arms, matching what the composed and staged routes already did. RED/GREEN
verified: with the fix reverted, `test_wmma_route_clears_the_wide_route_in_both_arms`
fails. `tests/test_unit_qwen4exp_wmma_gate_route_arms.py` also pins that the two
arms differ **only** in the WMMA selector, so the measured drift stays
attributable to this route alone.

## What this does not affect

The staged-f32 results are unaffected, and the composed certificate is
**exonerated rather than damaged**: the composed run cleared the wide route in
both arms, so its WMMA@16-47 component really was WMMA and its teacher really was
the exact chain. Its hashes are in the table above.
