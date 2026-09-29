# The `wmma` route produces a false pass at HEAD

**Status: confirmed defect. Cause NOT established.** An earlier revision of this
file named the dispatch memoization key as the verified cause. That was wrong,
and the correction is recorded below rather than quietly edited, because the
wrong claim was published in a worklog entry and a ledger row.

## Symptom, reproduced twice

`--route wmma --layers 16-47` (the certified configuration) reports:

```
strict_logits_sha256    = 83732d4542e1847fd8975ca6...
candidate_logits_sha256 = 83732d4542e1847fd8975ca6...
```

Identical. KL exactly `0.0` on every statistic, `top1 1.0`, and
**`status: complete`, `measurement_valid: true`, `requires_outlier_review: false`**.

`83732d45...` is the WMMA@16-47 hash - byte-identical to the teacher hash of the
composed staged runs, whose teacher arm binds `WMMA_LAYERS_ENV` to 16-47. The
exact chain is `550bb9b8...`. The teacher arm, recorded as
`teacher_selector: "cleared"`, did not resolve to the exact chain.

- `artifact-control-wmma16-47-vs-exact.json` - first occurrence.
- `artifact-control-wmma16-47-vs-exact-fixed.json` - **same result with the
  dispatch-key fix in place.**

## It is a regression

The 2026-09-16 certification of this route had a genuine exact-chain teacher
(`550bb9b8...` against candidate `65370710...`), mean KL 3.796e-4.

**The WMMA 16-47 gate cannot be re-run at HEAD to reproduce its certification.**
A re-run returns KL exactly 0.0 and a green status.

## Corrected cause analysis

**Withdrawn:** that the dispatch memoization key was the cause. The key genuinely
lacked a WMMA layer-scope term and that was a real latent defect (fixed in
`a7ad30e5f`, with a RED test), but the false pass reproduces unchanged with the
term present. A latent defect is not the same as the cause, and stating it as
"verified" was an over-claim.

**Leading hypothesis, not confirmed:** `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE` is
bound to 16-47 by the certified plan, and `_q8_dense_wide_claims_parent` accepts
**both** the exact parent (`raw`) and the WMMA-rewritten parent
(`wmma_<parent>` under ABI `wmma_raw`). The wide route also runs later in the
launch chain. If it claims the parent in both arms, the final variant is
`dense_wide256_f32_f32_out` regardless of whether WMMA ran, so the WMMA layer
selector cannot change the arithmetic for 16-47 and both arms are the same
kernel. Its own docstring supports this: the production default binds the WMMA
route to 16-47, "which is the window this route is scoped to".

**Why it is not confirmed:** the probe used to test the cache key cannot settle
this. A synthetic weight resolves to `exact_prefill_tile16x4_bf16_bf16_out`,
which is not in `_Q8_DENSE_WIDE_EXACT_PARENTS`, so the wide route declines and
the real dispatch path is never reached. Confirming it needs a real resident
weight, or an instrumented run of the gate itself.

## What this does not affect

The staged-f32 results are unaffected: every staged run's teacher hash was
verified to be `550bb9b8...` (the exact chain) and every candidate hash verified
to differ. The composed end-to-end certificate has teacher `550bb9b8...`,
candidate `87dbf445...`.

It does mean the composed run's **WMMA@16-47 component is not independently
re-verified at HEAD**; it is inherited from the 2026-09-16 certification, and
that certification is currently not reproducible. If the leading hypothesis
holds, that component is really `dense_wide` at HEAD and the composition's
arithmetic differs from what its evidence describes.
