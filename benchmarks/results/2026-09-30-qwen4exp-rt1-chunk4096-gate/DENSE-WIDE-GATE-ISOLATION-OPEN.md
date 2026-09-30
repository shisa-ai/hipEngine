# RESOLVED: the dense_wide and wmma 16-47 certificates are the same measurement

**Originally opened as a possible third false pass. It is not one — the
byte-identical hashes are a documented, intended property.** Recorded because
the check was worth running, the resolution sharpens the stale-certificate
finding in this directory, and an earlier draft of this file framed it as an
anomaly.

## What was observed

Three gate certificates, produced across 13 days at three different commits,
record **bit-identical published logits**:

| artifact | route | created | commit | strict | candidate |
| --- | --- | --- | --- | --- | --- |
| `2026-09-16-q8-wmma-dense-prefill-layers-gate/artifact-layers16-47-4cat.json` | `wmma` | 2026-09-16 | `f1d676684` | `550bb9b8…` | `65370710…` |
| `2026-09-17-q8-dense-wide-16-47-gate/artifact-layers16-47-4cat.json` | `dense_wide` | 2026-09-16 | `1f06cd5e3` | `550bb9b8…` | `65370710…` |
| `2026-09-29-…/artifact-control-wmma16-47-vs-exact-dwcleared.json` | `wmma` | 2026-09-29 | `ccd9f7a84` | `550bb9b8…` | `65370710…` |

All three also agree on `kl_mean` 3.79552e-4, `kl_max` 1.530440e-2, top-1
0.99354, and 1548 rows. At HEAD, the reduced protocol for `--route dense_wide`
returns the same `kl_mean`/`kl_max` as `--route wmma`.

## The resolution

This is documented in `hipengine/generation/qwen4_exp_profiles.py` and is the
**stated reason the WMMA route was retired**:

```
PRODUCTION_Q8_WMMA_PREFILL_LAYERS = tuple(range(16, 48))
# Retired as the production default at this scope on 2026-09-17: the wide-row
# route below holds the identical envelope (byte-identical strict and candidate
# logits digests) and is 5.4% faster at 1K/4K.

PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS = tuple(range(16, 48))
# ... with `strict_logits_sha256` and `candidate_logits_sha256` equal to that
# route's gate artifact byte for byte.
```

So the identity is an **acceptance criterion that was met and cited**, not an
unexplained coincidence. The wide route replaced the WMMA route at 16-47
*because* it holds the identical envelope while being 5.4% faster. The
mechanism is in `_q8_dense_wide_claims_parent`: the WMMA rewrite runs earlier in
the launch chain, both families are f16-operand WMMA, and the wide route takes
precedence over the family it replaces at that tile.

## What the sweep did find

Two things, both worth keeping:

1. **The two certificates are not independent evidence.** They are one
   measurement cited twice. Any accounting that treats "wmma 16-47 certified"
   and "dense_wide 16-47 certified" as two supporting artifacts is
   double-counting. The project already knew this — the docstring cites the
   identity — but the artifact set does not make it visible.

2. **The stale certificate is the one admitting the current production
   default.** `dense_wide`, not `wmma`, is what runs at 16-47 today
   (`PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS`), and its certificate carries the
   same teacher hash `550bb9b8` that `20260930T110625` showed no longer
   reproduces at HEAD. The earlier entry said "`wmma` 16-47 is the production
   default for the wide dense-prefill shapes and this certificate is what admits
   it" — that is **imprecise**. The route in production is `dense_wide`; the
   WMMA selector is retired at this scope and retained for opt-in and re-gating.
   The finding is unchanged and if anything sharper: **the certificate admitting
   the live default no longer reproduces**, and its teacher movement is
   established by identity rather than by a separate run, since both routes share
   the teacher.

## What remains genuinely open

- The full 12-case/128-step `--route dense_wide` run at HEAD is still
  unmeasured. Its teacher hash would be `294e6c08…` (the value the `wmma` gate
  returns at HEAD), because the teacher clears every selector in both gates.
- No certificate produced before `056e6fbd8` carries a `route_engagement` block,
  so none of them proves its candidate route engaged. Here that gap is harmless
  because the property is documented elsewhere; it is not harmless as a general
  property of the artifact set.
- **The corrected action item is a re-gate of `dense_wide` at 16-47**, since
  that is the live default, not `wmma`.

## Method note

This was opened as an anomaly on the strength of two matching hashes, before
checking whether the project had documented the identity. It had — in the
profile module, in the comment that explains why the route changed. The check
that resolved it (`grep` for the claim in-tree) took under a minute and should
have preceded writing the open-lead file. This is the third time in this
campaign that a plausibility-first hypothesis was refuted by a cheap check run
late; the pattern is recorded rather than the individual instances.
