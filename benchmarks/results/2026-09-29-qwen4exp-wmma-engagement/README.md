# WMMA prefill engagement, measured at the dispatch

The recorder added in `056e6fbd8` answers the question every artifact in this
packet had been inferring from a KL: **which route actually ran?**

## Result

Both protocols, `--route wmma --layers 16-47`, one decode step each, recorder
active:

| protocol | case | candidate launches | engaged | engaged variant |
|---|---|---|---|---|
| canonical | `code-p512` (512 tok) | 1000 | **264** | `wmma_prefill_f32_f32_out` |
| heldout | `code_merge_intervals` (40 tok) | 1096 | **328** | `wmma_prefill_f32_f32_out` |

Both arms: teacher `engaged = 0`, `decline_reasons = {use_wmma_false: ...}`. So the
teacher really is the cleared arm and the candidate really does run the route.
The route engages at rows 512 and 40 respectively and declines at rows 1.

## The prediction was wrong, and the result is more interesting

The preceding entry predicted that the canonical prefill would arrive carrying
the exact coltile leaf, fail the `prefill_` prefix predicate, and never engage -
which would have explained why its post-prefill row is bit-identical. **That is
falsified.** The route engages 264 times in the canonical run.

And the post-prefill row is *still* bit-identical:

| protocol | engaged launches | post-prefill KL | decode KL |
|---|---|---|---|
| canonical | 264 | **5.8014e-12** | 1.0184e-04 |
| heldout | 328 | 2.6345e-05 | 6.0842e-06 |

So `wmma_prefill_f32_f32_out` produces arithmetically equivalent results to the
exact chain at the prefill position, at both 512 and 40 rows. **The route runs and
contributes no measurable numerical difference.**

That makes the misattribution sharper rather than weaker. The `wmma` gate's
recorded `candidate_chain` is `wmma_prefill_f32_f32_out`, that chain demonstrably
runs, and the drift it reports is nonetheless not its: the canonical prefill row
is flat to f32 precision while the decode row moves. The drift belongs to the
other selectors `use_wmma` gates.

## A flaw in my own check, corrected before it misled anyone

The summariser first reported `arms_isolated`, and this run is exactly the case
that shows the name is wrong. It is `True` here - the route engaged in the
candidate arm and not the teacher - and the arms are still **not** isolated,
because `use_wmma` also gates `f_rowtile`, `enabled=not use_wmma`,
`use_rowtile` and the rows-2-8 native batch-decode session, none of which the
recorder can observe.

The flag is renamed `wmma_route_engaged_candidate_only`, joined by an explicit
`covers_other_use_wmma_selectors: False`, and the verdict now says in words that
it does not mean the arms are isolated. A test asserts the old name is absent.

This is the fifth time in this packet that a green or true-valued field was
narrower than its name. It is the same failure the recorder exists to prevent,
so it would have been a poor place to leave one.

## What this establishes

1. The WMMA prefill route engages in both protocols, at the row counts the gate
   exercises, with the teacher cleared. Verified, not inferred.
2. Its arithmetic is equivalent to the exact chain at the prefill position
   (5.8e-12 at 512 rows, 2.6e-05 at 40 rows), so it is not the source of the
   drift the `wmma` gate reports.
3. Therefore the gate's reported drift belongs to `f_rowtile` /
   `use_rowtile` / the rows-2-8 batch-decode session - selectors reached through
   `use_wmma` but outside the route the gate is named for.
4. The canonical and heldout prefill rows differ (5.8e-12 vs 2.6e-05) despite
   identical engagement, so the remaining difference is a property of those
   other selectors or of the shapes, not of whether the route ran.

## What is still open

The exact attribution *within* those other selectors. The recorder covers the
WMMA route only; extending it to `f_rowtile` and the batch-decode session is what
would close the loop, and until then the gate's drift is "not this route" rather
than "that selector".
