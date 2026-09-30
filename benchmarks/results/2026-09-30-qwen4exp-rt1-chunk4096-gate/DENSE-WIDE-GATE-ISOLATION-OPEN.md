# Does the dense_wide gate isolate the wide route at 16-47?

**Status: OPEN LEAD, not a conclusion.** Recorded because two independent
certificates disagree with the expectation that they measure different routes,
and because nothing in either artifact proves which kernel ran.

## The observation

Two gate certificates for **different routes**, produced 13 days apart at
different commits, record **bit-identical published logits**:

| artifact | route | created | commit | strict hash | candidate hash |
| --- | --- | --- | --- | --- | --- |
| `2026-09-17-q8-dense-wide-16-47-gate/artifact-layers16-47-4cat.json` | `dense_wide` | 2026-09-16 | `1f06cd5e3` | `550bb9b832bc7ba2…` | `65370710c0b9c40f…` |
| `2026-09-29-q8-staged-f32-dense-prefill-layers-gate/artifact-control-wmma16-47-vs-exact-dwcleared.json` | `wmma` | 2026-09-29 | `ccd9f7a84` | `550bb9b832bc7ba2…` | `65370710c0b9c40f…` |

The **teacher** hash matching is expected: both gates clear every selector for
the teacher arm, so both teachers are the same exact chain.

The **candidate** hash matching is not. The two gates bind different selectors
(`wmma` clears `dense_wide` in both arms; `dense_wide` clears `wmma` in both
arms), and they record different candidate chains -
`wmma_prefill_f32_f32_out` and `dense_wide256_f32_f32_out`. Either those two
chains genuinely produce bit-identical logits at these shapes, or one gate did
not run the route it recorded.

## Why the artifacts cannot settle it

Neither certificate carries a `route_engagement` block. Engagement recording was
added later, in `056e6fbd8` ("record WMMA prefill engagement at the dispatch, not
from a KL"). So **neither artifact proves its candidate route engaged**, and the
recorded `candidate_chain` field is a label rather than a measurement.

## What HEAD says

HEAD records engagement, so the same gate can now be asked directly. Reduced
protocol (`--case-id code-p512 --case-id general_en-p512 --decode-steps 24
--repeat-runs 3`, chunk 1024), artifact
`dense-wide-16-47-engagement-probe.json`:

```
route: dense_wide | candidate_chain: dense_wide256_f32_f32_out
candidate_route_engaged: False
teacher_route_engaged  : False
wmma_route_engaged_candidate_only: False
candidate.engaged_launches: 80598
teacher.engaged_launches  : 27346
  native_batch_decode_session: teacher 0.0 / candidate 0.0 / differs False
  q4k_rowtile:                 teacher 1.0 / candidate 1.0 / differs False
  wmma_prefill:                teacher 0.0 / candidate 0.0 / differs False
kl_mean: 9.399214e-04 | kl_max: 1.231981e-02 | top1: 0.96
```

**No selector state differs between the arms**, and `candidate_route_engaged` is
`False`. The same reduced protocol for `--route wmma` returns the **same**
`kl_mean` and `kl_max` to every printed digit, which is consistent with the two
routes being one configuration at these shapes.

Note the arms still differ (`kl_mean` is not zero), so `candidate_route_engaged:
False` is not "the arms are identical" - it is "this route's own selector state
does not distinguish them".

## What this does and does not establish

- **Does establish:** the two recorded certificates are bit-identical in both
  hashes, and at HEAD the two `--route` values produce bit-identical results.
  Neither certificate carries engagement evidence.
- **Does not establish:** that either gate is a false pass. `dense_wide` may
  legitimately be the parent the `wmma` selector resolves through, in which case
  the two routes are one arithmetic and the identical hashes are correct - but
  then the two gates are not independent certificates and should not be cited as
  such.
- **Not yet checked:** the full 12-case/128-step protocol for `--route
  dense_wide` at HEAD. The comparison above is the reduced screen only.

## Next

1. Run `--route dense_wide --layers 16-47` at HEAD on the full canonical
   protocol and compare its engagement block and hashes against the `wmma` run
   in this same directory.
2. Resolve whether `_q8_dense_wide_claims_parent` makes the `wmma` candidate
   resolve to `dense_wide256_f32_f32_out`; if it does, the `wmma` gate is not
   measuring the WMMA chain it records.
3. Treat this as the same class as the two previously found false passes
   (`WMMA-ROUTE-FALSE-PASS.md`, and the staged-f32 run whose `measurement_valid`
   was true over a replayed teacher). The distinguishing feature here is that no
   engagement evidence exists in the artifacts at all, so the failure mode could
   not have been detected from them.
