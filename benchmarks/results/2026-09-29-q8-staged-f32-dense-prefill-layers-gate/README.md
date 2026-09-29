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
| layers 0-15 | 1548 | 1.177e-3 | 6.121e-3 | 1.371e-2 | 4.905e-2 | 0.98643 | **fail** |
| layers 0-11 | 1548 | 1.274e-3 | 6.585e-3 | 1.686e-2 | 4.686e-2 | 0.98643 | **fail** |
| layers 0-7 | 1548 | 1.354e-3 | 6.613e-3 | 1.929e-2 | 6.988e-2 | 0.98902 | **fail** |
| layers 8-47 | 1548 | 8.818e-4 | 3.843e-3 | 1.302e-2 | 6.441e-2 | 0.98837 | **fail** |
| layers 12-47 | 1548 | 6.593e-4 | 3.469e-3 | 9.765e-3 | 3.482e-2 | 0.98837 | **fail** |
| layers 13-47 | 1548 | 5.167e-4 | 2.716e-3 | 8.329e-3 | 2.504e-2 | 0.99419 | **pass** |
| layers 14-47 | 1548 | 4.367e-4 | 2.109e-3 | 6.332e-3 | 1.835e-2 | 0.99225 | **pass** |
| layers 15-47 | 1548 | 4.453e-4 | 1.961e-3 | 6.538e-3 | 6.349e-2 | 0.99548 | **fail** |
| layers 16-47 | 1548 | 3.236e-4 | 1.629e-3 | 4.843e-3 | 1.610e-2 | 0.99483 | **pass** |
| limit | | 1e-3 | 5e-3 | 2e-2 | 5e-2 | 0.99 | |

**Deepest passing scope: layers 13-47**, three layers beyond the currently
certified 16-47, with roughly 2x headroom on every gate (mean 1.94x, p95 1.84x,
p99 2.40x, max 2.00x).

## 13-47 passing does not yet establish production value

Every scope above is measured with **WMMA cleared in both arms**, so what these
rows certify is *staged on the named scope with no WMMA anywhere*. Production is
not that: the f16 WMMA route owns layers 16-47, so a staged scope of 13-15 would
only ever replace the **exact chain** on 13-15, with WMMA still running from 16.

The marginal cost of that composition cannot be read off this table. Adding
WMMA's 16-47 drift to the staged 13-15 increment would be a summed-KL budget,
and this packet's own ladders show that is invalid - see the non-monotonicity
section below. It needs its own gate against a WMMA@16-47 teacher, which is what
`staged_f32_over_wmma` measures.

## Narrowing the exact-chain region makes it worse, not better


## The only passing scope has no marginal value

**Layers 16-47 passes, and it should not be promoted on that basis.**

16-47 is the scope the f16 WMMA route already covers and already runs. On the
same shape (`attn_qkv`, 2560 -> 10240, 1024 rows) the arithmetic rates are:

| route | TFLOP/s |
| --- | ---: |
| `dense_wide256`, f16 input | 40.54 |
| `dense_wide256`, f32 input | 20.32 |
| **staged f32 tile** | **4.10** |
| exact coltile leaf | 3.32 |

The staged tile is **~5x slower** than the wide route with f32 input and ~10x
slower with f16 input, on the identical shape. So at 16-47 it would replace a
route that is already promoted and much faster: a numerical pass there buys
nothing. The staged tile's only advantage is over the *exact FMA chain*, which
is what runs where the wide route does not.

That is the real question, and it is what the 0-15 / 0-11 / 0-7 ladder measures.
A pass at 16-47 is not a win and is recorded as such.

## Narrowing the exact-chain region makes it worse, not better

The three scopes that cover layers the wide route does **not** run - the only
scopes where the staged tile could beat anything - all fail, and the trend is
backwards:

| Scope | Layers | Mean KL |
| --- | ---: | ---: |
| 0-47 | 48 | 1.269e-3 |
| 0-15 | 16 | 1.177e-3 |
| 0-11 | 12 | 1.274e-3 |
| 0-7 | 8 | **1.354e-3** |

**0-7 has a higher mean KL than 0-47.** Eight layers produce more whole-model
divergence than forty-eight. The drift is therefore **not cumulative over
layers** - it is concentrated in the earliest layers, which are exactly the
layers with marginal value, since the wide route starts at 16.

This is an observation, not an explained mechanism. The same non-monotonicity
appears in the upper ladder, where 8-47's `max_kl` (6.441e-2) exceeds 0-47's
(2.925e-2) while its mean is lower. Two independent directions of non-monotonic
behaviour mean no scope's numbers can be extrapolated from a neighbour's, and it
is the strongest single argument against the summed-KL budgeting the ledger
already warns about.

## Failure shape

0-47 fails on mean, p95 and top-1 while **p99 and max pass** - broad-based drift,
not a rare-outlier problem. That rules out a per-layer guard on offending layers
as the fix, because no small set of layers carries it.

8-47 is worth reading carefully: its `max_kl` (6.441e-2) is **higher** than
0-47's (2.925e-2) while its mean is lower. Scopes are not a monotonic additive
budget, so no summed-KL extrapolation from these rows is valid.

## The first 0-47 run was INVALID and is retained as such

An earlier 0-47 run reported `mean_kl 0.0`, `max_kl 0.0`, `top1 1.0` and
`hard_gates_passed: true`. It measured **nothing**: candidate and teacher logits
were bit-identical, because the selector was absent from the dispatch
memoization key and the teacher arm's exact-chain dispatch was replayed for the
candidate arm. See `INVALID.md`; `engagement-probe.json` records the fix.

A KL of exactly zero is a failure signature, not a perfect score.
`measurement_valid` does not check engagement - `qualification_coverage()`
verifies the protocol only.
