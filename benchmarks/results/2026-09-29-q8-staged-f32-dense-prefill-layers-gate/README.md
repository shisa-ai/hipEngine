# LDS-staged f32 dense Q8_0 prefill: layer-scope numerical gate

The staged-f32 tile (`staged_f32_bm128_bn64_bk32`) gated against the exact
coltile chain on the Qwen4Exp UD-Q4_K_XL canonical exact-token fixture.

Host: AMD Radeon 8060S (gfx1151), TheRock staging `therock10-staging-20260828`.
Protocol: 12 canonical cases (4 categories x 3 shapes), D128, 3 repeats,
chunk 1024, production profile. Limits: mean 1e-3, p95 5e-3, p99 2e-2,
max 5e-2, top-1 0.99.

| Scope | Rows | Mean KL | p95 KL | p99 KL | Max KL | Top-1 | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| layers 0-47 | 1548 | 1.269e-3 | 6.663e-3 | 1.620e-2 | 2.925e-2 | 0.98385 | **fail** |
| limit | | 1e-3 | 5e-3 | 2e-2 | 5e-2 | 0.99 | |

## Layers 0-47: fail

Fails on **mean KL (1.27x over), p95 KL (1.33x over) and top-1**. The tail
passes: p99 and max are inside their limits with room to spare.

That combination is the useful part of this result. The drift is **broad-based,
not tail-driven** — this is not a rare-outlier or a single bad transition. A
route that failed only on max would invite a guard on the offending layers; this
one is off across the distribution, so narrowing the scope is the lever, which
is what the ladder below tests.

Determinism is clean: three repeats produced identical trajectory hashes on all
12 prompts, and `measurement_valid: true` with canonical coverage complete.

## The first 0-47 run was INVALID and is retained as such

An earlier 0-47 run reported `mean_kl 0.0`, `max_kl 0.0`, `top1 1.0` and
`hard_gates_passed: true`. It measured **nothing**: candidate and teacher logits
were bit-identical, because the selector was absent from the dispatch
memoization key and the teacher arm's exact-chain dispatch was replayed for the
candidate arm. See `INVALID.md`. `engagement-probe.json` records the fix.

The lesson is in the artifact, not just the worklog: a KL of exactly zero is a
failure signature, not a perfect score, and `measurement_valid` does not check
engagement — `qualification_coverage()` verifies the protocol only.
