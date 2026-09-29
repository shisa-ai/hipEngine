# The `wmma` route produces a false pass at HEAD

**Found while running a control for the staged-f32 outlier review. This is a
defect in the already-promoted WMMA route's gate, not in the staged-f32 work.**

## Symptom

`artifact-control-wmma16-47-vs-exact.json` (`--route wmma --layers 16-47`, the
already-certified configuration) reports:

```
strict_logits_sha256    = 83732d4542e1847fd8975ca6...
candidate_logits_sha256 = 83732d4542e1847fd8975ca6...
```

Identical. `mean_kl`, `p95_kl`, `p99_kl` and `max_kl` are all exactly `0.0`,
`top1_agreement` is `1.0` — and the run reports **`status: complete`,
`measurement_valid: true`, `requires_outlier_review: false`**.

`83732d45...` is the **WMMA@16-47** hash: it is byte-identical to the teacher
hash of the composed staged runs, whose teacher arm explicitly binds
`WMMA_LAYERS_ENV` to 16-47. The exact chain's hash is `550bb9b8...`.

So the teacher arm, whose `route.teacher_selector` is recorded as `"cleared"`
and whose `route.teacher_chain` is recorded as `"exact
coltile8_rowbatch4_f32_f32_out"`, did **not** resolve to the exact chain.

## This is a regression

The 2026-09-16 certification of the same route had a genuine exact-chain teacher:

```
strict_logits_sha256    = 550bb9b832bc7ba273f6b3b3dc24fddc...
candidate_logits_sha256 = 65370710c0b9c40fdfdfb557f954e745...
```

Different hashes, mean KL 3.796e-4, `status: complete`. Engagement was real then.

**So the WMMA 16-47 gate cannot be re-run at HEAD to reproduce its
certification.** A re-run returns KL exactly 0.0 and a green status.

## Mechanism

Verified, not inferred:

- The dispatch memoization key contains **no WMMA layer-scope term**. It carries
  `use_wmma` as a boolean, and the layer scope is applied separately through
  `wmma_prefill_weight_filter_session`, a context-local (`ContextVar`) weight
  filter.
- A probe recording the key the real `launch_gguf_linear` uses shows the key is
  **identical between the route's candidate and teacher arms**.

The chain is therefore: the layer scope does not reach the cache key, the two
arms are otherwise the same shapes on the same weight pointers, so the teacher
arm hits the candidate arm's cache entry and replays its WMMA@16-47 dispatch.
`launch_gguf_linear` only reaches the dispatch chain inside `if cached is None:`.

This is the **same defect class** as the staged-f32 false pass recorded in
`INVALID.md`. The staged routes are correct only because `_staged_f32_key` was
added to the key; the WMMA route has no equivalent term.

## Not fixed here

Deliberately. The fix touches a **promoted** route and the dispatch key that
several routes share, in a high-conflict file, and it needs its own RED test and
its own gate re-run to re-establish the 16-47 certification at HEAD. Folding it
into the staged-f32 unit would have hidden it in an unrelated commit.

## What this does not affect

The staged-f32 results are unaffected, and the reason is checkable rather than
assumed: every staged run's teacher hash was verified to be `550bb9b8...`, the
exact chain, and every staged run's candidate hash was verified to differ. The
composed end-to-end certificate (`artifact-composed-staged13-15-over-wmma-vs-exact.json`)
has teacher `550bb9b8...` and candidate `87dbf445...`.

It does mean the composed run's **WMMA@16-47 component is not independently
re-verified at HEAD** by this packet; it is inherited from the 2026-09-16
certification.
