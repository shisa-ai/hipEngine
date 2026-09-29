# X-7 category-heldout evidence: the WMMA route fails at prefill→c1

X-7 is the ledger's declared prerequisite ("Close before extending the admitted
scope"). Its first missing certificate was **category-heldout numerical
evidence**, and the gate could not express a heldout run at all: it hardcoded one
canonical fixture hash, so the blocker was unfalsifiable from the artifact.

The gate now recognizes `qwen4exp_admission_suite_18prompt.json` as a **separate**
heldout protocol - 18 natural-length cases across the same four categories, 24
declared transitions - with its own coverage record that can never set the
canonical `complete` flag.

## The evidence is complete, and it fails

`artifact-wmma16-47-heldout-floor2.json`, route `wmma`, layers 16-47:

```
strict_logits_sha256    = 7b4385b7dff48201cfcb...
candidate_logits_sha256 = b46e7be47a519aa00dc2...
```

Different, so the route **did engage** - checked first, before any KL, because an
identical pair is the failure mode this packet has already recorded twice.

All seven heldout coverage checks pass. And then:

| | value | limit |
|---|---|---|
| mean KL | 2.73717e-4 | 1e-3 |
| p95 | 1.06668e-3 | 5e-3 |
| p99 | 3.77855e-3 | 2e-2 |
| max | **2.70641e-2** | 5e-2 |
| top-1 | 0.99556 | 0.99 |

The aggregate passes. **`hard_gates_passed: False`** anyway, because two
per-scope gates fail:

| failing scope | top-1 | mean KL | limit |
|---|---|---|---|
| `shape=prefill_last` | 1.0 | **2.06179e-3** | 1e-3 |
| `transition=prefill_to_c1` | 1.0 | **2.06179e-3** | 1e-3 |

Both fail on **KL, not top-1** - top-1 is perfect at 1.0 - and both are the same
row population, so the drift is concentrated at the **prefill→c1 transition**:
the first decode step after prefill, where mean KL is **2× the limit**. One row
exceeds the 0.02 review boundary, and there are 2 top-1 mismatch rows.

`requires_outlier_review: True`, `eligible_for_automatic_admission: False`.

## Why this matters

**The canonical suite passed and the heldout suite did not.** That is exactly the
value of heldout evidence: the canonical fixture is 512/1024/4096-token prompts
over 128 transitions, and the heldout suite is 39-71-token prompts over 24. The
shorter prompts put proportionally more weight on the prefill→c1 transition,
which is where this route's drift concentrates. So the certificate is not merely
a box ticked - it found a real per-scope failure the canonical protocol missed.

## Two errors of my own, recorded rather than edited away

1. **I claimed the suite "cannot engage" the route**, on a 256-row floor. Wrong:
   I read the floor from `_q8_iu8_wmma_dispatch`, a different route. The f16
   WMMA route declines only at `rows <= 1`. The run settled it - the hashes
   differ. So my check was producing a false *refusal*, the mirror image of the
   false pass it was built to prevent. Floors are now per-route, read from each
   dispatch's own guard, and the check discriminates: the same suite qualifies
   `wmma` and `staged_f32` and does not qualify `dense_wide`.
2. **I emitted `category_heldout_qualification_claim: true` on this failing
   run.** Collected is not passed, and the field name promises the second. The
   claim now requires complete coverage **and** passing quality; evidence
   availability is reported separately as `category_heldout_evidence_complete`.

Both were caught by reading the artifact rather than the summary line.

## State

- **Evidence availability: closed.** A complete, engaging heldout certificate
  exists and is reproducible.
- **Qualification: open, with a named failure.** The route fails per-scope KL at
  the prefill→c1 transition on heldout data. X-7's remaining blockers are
  unchanged.
