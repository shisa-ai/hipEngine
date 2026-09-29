# INVALID — this gate run measured the exact chain against itself

**Do not read this run as a pass.** `mean_kl`, `p95_kl`, `p99_kl` and `max_kl`
are all exactly `0.0` and `top1_agreement` is `1.0`, which is not an excellent
result — it is the signature of **zero route engagement**.

Decisive evidence, from `artifact-layers0-47-4cat.json`:

```
strict_logits_sha256    = 550bb9b832bc7ba273f6b3b3dc24fddcdb8edff724f19a2c781c46a400e118f6
candidate_logits_sha256 = 550bb9b832bc7ba273f6b3b3dc24fddcdb8edff724f19a2c781c46a400e118f6
```

The candidate and teacher produced **bit-identical logits** across all 12 cases.
The staged tile was never selected.

The gate already said so itself: `blockers` includes *"candidate route
engagement and repair-oracle coverage"*. `qualification_coverage()` checks the
protocol (fixture identity, 12 cases once each, D128, three repeats) and does
**not** check engagement, so `measurement_valid: true` and
`hard_gates_passed: true` are both consistent with a route that never ran.

## Root cause

`_raw_k_staged_f32_dispatch` reads its selector live, but
`launch_gguf_linear` only reaches the dispatch chain inside `if cached is None:`.
The dispatch memoization key did not include
`HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS`, so the **teacher arm's exact-chain
dispatch was replayed for the candidate arm**.

The sibling `dense_wide` route documents this exact hazard above its own key
term, for a related reason: *"All three must be in the memoization key or the
first resolution for a shape is replayed for every layer and the scope becomes
an artifact of cache-fill order rather than a contract."*

For the staged route the consequence is worse than a wrong scope. A gate's two
arms are the same shapes on the same weight pointers and differ only in this
selector, so the omission makes the two arms *identical by construction* — and
the failure is silent, because identical logits read as a flawless pass.

## Status

Fixed: `_staged_f32_key` (scope plus owning layer index) now participates in the
key, and `tests/test_unit_qwen4exp_staged_f32_route.py` pins it RED/GREEN.

**This artifact is retained only as the record of the invalid run.** A real gate
result requires re-running the gate on the fixed code.

## Follow-up: engagement confirmed on the fixed code

`engagement-probe.json` (one `code-p512` case, 8 decode steps, 3 repeats) is the
record that the fix works:

```
strict_logits_sha256    = 11b2f8eb1f69e715ccbfbfa046a01c68e5a977cb36b4ebebc871493f272a5f2a
candidate_logits_sha256 = 499ea85fc1651d91935c8b9a8a3c81186b8ceda2f3c34a21060575aa96d04177
```

The hashes now **differ**, where the invalid 12-case run had them identical. The
staged tile executes. `mean_kl` moved from exactly `0.0` to `2.947e-3` over 9 rows.

That probe is a screen, not a verdict: one case at 8 decode steps is far outside
the canonical 12-case D128 protocol, and the gate correctly returned
`invalid_or_screen_only`. Its KL is **not** comparable to the canonical numbers
below it in this file.

One signal is worth carrying forward as a caution, not a conclusion: at 0-47 the
screen's mean KL of `2.947e-3` is already above the `1e-3` limit, while its
`max_kl` (`1.35e-2`) and `top1` (`1.0`) are inside theirs. That is the same shape
as the f16 WMMA route, which fails 0-47 on the mean and had to retreat to 16-47.
Full-scope promotion may not survive the canonical run.
