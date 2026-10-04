# The dense-wide Q8 prefill scope above 16-47

`PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS` binds the wide-row f16 Q8_0 prefill
route to **layers 16-47**; layers 0-15 keep the exact coltile chain. Extending
that binding is the largest single prefill lever this session found, and the
production gate **refutes it** above 16-47.

**No metric row changed. No kernel, default, flag or dispatch changed and
nothing was promoted.**

## Speed: the extension is large and real

Three arms, one process per arm, order A-B-C with the baseline on both sides of
the candidate. `code-p4096`, production profile, chunk 1024, 1 warmup + 3
repetitions, `--require-cached-build`, all under `uv run`.

| arm | dense-wide scope | pp tok/s | tg tok/s | output sha256 |
| --- | --- | ---: | ---: | --- |
| A (baseline) | 16-47 | 271.719 | 17.437 | `c18d529b…` |
| B (candidate) | **0-47** | **323.613** | **18.437** | `c18d529b…` |
| C (baseline again) | 16-47 | 272.731 | 17.420 | `c18d529b…` |

Baseline pooled mean 272.225 pp / 17.429 tg. Candidate against it: **+18.876%
prefill, +5.786% decode**, against a same-code A/C drift of **+0.372% / -0.10%**
- a **51x** separation. All three arms emit the identical generated-token digest
`c18d529b86947504e12e0d06aaa26f2ce7942387b98b95ec3bacdd65e6278eb5`, and it
equals the digest from the same case at chunks 1024 and 4096 measured earlier in
this tree.

The binding is written unconditionally by the profile binder, so an env override
cannot reach it; the file constant is the only lever, and it was verified per arm
under `uv run` (`len 32 first 16 last 47` for A and C, `len 48 first 0 last 47`
for B).

## The gate refutes it

Canonical protocol: 12 cases, 128 decode steps, 3 repeats, chunk 1024,
production profile, HEAD `04d0f4161`, one process per run, `--require-cached-build`.

| scope | engaged rate | kl_mean | kl_p95 | kl_max | top-1 | hard gates | why |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| **16-47** | 0.337 | 4.527e-4 | 2.320e-3 | 2.341e-2 | 1538/1548 | **PASS** | - |
| **14-47** | 0.359 | 5.190e-4 | 2.636e-3 | 3.979e-2 | 1536/1548 | **PASS** | deepest passing scope |
| 12-47 | 0.379 | 6.406e-4 | 3.229e-3 | 3.249e-2 | **1532**/1548 | FAIL | aggregate top-1, one row short of 1533 |
| 8-47 | 0.421 | 8.839e-4 | 4.647e-3 | 4.146e-2 | 1534/1548 | FAIL | `general_en` mean KL 1.181e-3 |
| 0-47 | 0.508 | **1.468e-3** | **7.517e-3** | **5.140e-2** | 1530/1548 | FAIL | KL and top-1, 6 scopes |

Limits: mean 1e-3, p95 5e-3, p99 2e-2, max 5e-2, aggregate top-1 0.99,
per-scope top-1 0.97, review boundary 0.02. The **strict teacher hash is
identical in all five runs**
(`4377006f4f0827dd419e9d92da9b61b0dbf65074195ffec5852fb98cc7e65ed7`), so the
candidate hashes are the only difference between them, and engagement scales
monotonically with scope (0.337 / 0.359 / 0.379 / 0.421 / 0.508).

At 0-47 the mean is **1.47x over**, p95 **1.50x over**, max **1.03x over**, and
top-1 is below the 0.99 minimum; all four categories fail, plus shape `c1` and
transition `steady`. At 8-47 every aggregate gate passes and **one** category
(`general_en`, mean KL 1.1812e-3) is 1.18x over the per-category mean limit. At
12-47 every KL gate passes with 1.5x headroom and there are **zero scope
failures** - it fails only the aggregate top-1, by **one row** (1532 against the
1533 needed).

### The boundary is not monotone

`kl_mean` and `kl_p95` grow monotonically with the scope, but the **tail** and
**count** statistics do not:

| scope | kl_mean | kl_max | top-1 matches |
| --- | ---: | ---: | ---: |
| 16-47 | 4.527e-4 | **2.341e-2** | **1538** |
| 14-47 | 5.190e-4 | 3.979e-2 | 1536 |
| 12-47 | 6.406e-4 | **3.249e-2** | **1532** |
| 8-47 | 8.839e-4 | 4.146e-2 | 1534 |
| 0-47 | 1.468e-3 | 5.140e-2 | 1530 |

12-47's `kl_max` is **better** than 14-47's while its mean is worse, and 8-47's
top-1 (1534) is **better** than 12-47's (1532). This is the same
non-monotonicity this project already measured on the staged-f32 route (15-47
failing while 14-47 and 16-47 pass), now reproduced on a second, independent
route. **Two consequences:** the boundary cannot be located by monotone search,
and a passing scope does not imply that a shallower one passes - so 14-47's pass
is not evidence about 13-47 or 12-47.

### 14-47 passes the gate but its speed win is not resolvable

The deepest passing scope is **14-47**, two layers beyond the certified 16-47.
Its speed was measured with the same three-arm A/B/C design:

| arm | scope | pp tok/s | tg tok/s |
| --- | --- | ---: | ---: |
| A | 16-47 | 268.877 | 17.545 |
| B | **14-47** | **275.305** | 17.556 |
| C | 16-47 | 272.589 | 17.378 |

Same-code drift A against C is **+1.38%**; the pooled candidate effect against
`mean(A,C) = 270.733` is **+1.69%** prefill and **+0.54%** decode. A
signal-to-drift ratio of 1.2 is **not a measurement**, and this is a much noisier
session than the 0-47 A/B (0.37% drift) was. So 14-47 is **numerically
admissible and not promoted**: the arithmetic change is real, the benefit is
predicted by the per-layer cost model to be about 2.2%, and the measurement is
consistent with that but cannot separate it from drift. Changing the KV
rounding of two layers for a benefit that cannot be shown is the wrong trade.
All three arms emit the same generated-token digest `c18d529b…`.

### Attribution is clean

The gate's baseline had drifted before in this project, so a 0-47 failure is
uninterpretable without a control at the same HEAD. The control was run first and
reproduces the recorded 16-47 certificate essentially bit-for-bit:

| metric | recorded (`cac0f1da5`) | control at HEAD (`04d0f4161`) |
| --- | ---: | ---: |
| `kl_mean` | 4.38817e-4 | 4.527038e-4 |
| `kl_p95` | 2.360965e-3 | 2.320395e-3 |
| `kl_max` | 2.3411946e-2 | **2.341195e-2** |
| top-1 | 0.994186 | 0.993540 |
| rows over review | 1 | 1 |

`kl_max` is identical to the printed precision and the single over-review row is
the same row (`general_en-p512`, teacher step 82). The control passes every hard
gate with 2.2x headroom on the tightest. **The strict teacher hash is identical
in all three runs**
(`4377006f4f0827dd419e9d92da9b61b0dbf65074195ffec5852fb98cc7e65ed7`), so the
candidate hashes are the only difference between them, and engagement scales
monotonically with scope (0.337 / 0.421 / 0.508).

### The drift enters through decode, not the prefill

| scope | `prefill_last` kl_mean | `prefill_last` top-1 |
| --- | ---: | ---: |
| 16-47 | 4.32e-9 | 1.0 |
| 8-47 | 3.15e-8 | 1.0 |
| 0-47 | 4.66e-8 | 1.0 |

The last prompt position's logits barely move at any scope; `steady` (decode)
carries the whole drift. That is the expected shape for a route that changes the
*rounding* of the projections feeding the KV cache - the final position's logits
are dominated by the last layer's own computation, while all 4096 cached
positions carry slightly different values that decode attention accumulates.

Per-category mean KL grows monotonically with scope for **every** category, so
this is not one anomalous category:

| scope | code | general_en | general_ja | mixed_ja_en |
| --- | ---: | ---: | ---: | ---: |
| 16-47 | 4.66e-4 | 5.41e-4 | 4.09e-4 | 3.96e-4 |
| 8-47 | 6.85e-4 | **1.18e-3** | 8.99e-4 | 7.70e-4 |
| 0-47 | **1.01e-3** | **1.75e-3** | **1.96e-3** | **1.15e-3** |

### Near-ties explain the failures and do not excuse them

All 18 top-1 mismatches at 0-47 are `strict_top1_candidate_rank: 2` with
`topk_overlap` 0.8-1.0 and strict margins of 0.003-0.11, so the top-5 sets are
largely identical and only the top-two order moved. That is the character the
2026-09-30 baseline-drift finding had, and it is recorded here for the same
reason: **the mean and p95 gates are aggregate statistics, not tail
statistics**, and they are over by 1.47x and 1.50x. This is a broad-based
failure, so a per-layer or per-row guard would not rescue it.

## What this closes and what it leaves open

**Closed:** the 18.9% prefill / 5.8% decode headline is not admissible, and no
scope tested recovers it. The region is not merely tail-sensitive; 0-47 fails
the aggregate envelope for every category, and 12-47 and 8-47 fail single
distinct gates.

**Open:** 14-47 passes the gate and is two layers of a route that runs 8.9x
faster per layer than the chain it replaces, but the benefit is below this
session's drift floor. A higher-precision A/B (more repetitions, interleaved
within one process) would resolve it; the per-layer model predicts about 2.2%.
Whether 13-47 passes is unknown and cannot be inferred from 14-47, because the
boundary is non-monotone.

## Artifacts

| file | what |
| --- | --- |
| `ab-A-16-47.json` | canonical bench, dense-wide 16-47, arm A |
| `ab-B-0-47.json` | canonical bench, dense-wide 0-47, arm B |
| `ab-C-16-47.json` | canonical bench, dense-wide 16-47, arm C (drift control) |
| `artifact-16-47-control.json` | production gate, 16-47 at HEAD |
| `artifact-14-47.json` | production gate, 14-47 (PASS, deepest passing) |
| `artifact-12-47.json` | production gate, 12-47 (fails aggregate top-1 by one row) |
| `artifact-8-47.json` | production gate, 8-47 |
| `artifact-0-47.json` | production gate, 0-47 |
| `ab14-A.json`, `ab14-B.json`, `ab14-C.json` | canonical bench, 16-47 / 14-47 / 16-47 |

## Limits

- One host (`gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`, not the
  gfx1100/W7900 default hardware), one model file, one quant, one execution
  profile, one chunk size (1024), one route.
- The gate reports `status: invalid_or_screen_only` and
  `measurement_valid: false` for the **control** too, because
  `qualification_blockers` carries the standard uncollected-certificate list plus
  the single over-review row. `hard_gates_passed` is the field that separates
  these runs, and `measurement_valid` should not be read as "the run is
  untrustworthy".
- The A/B speed arms are one case (`code-p4096`); the gate is the twelve-case
  protocol. The two were not measured on the same case set.
- The 14-47 A/B's same-code drift (1.38%) is larger than the effect it measures
  (1.69%), so that row carries no speed claim. The earlier 0-47 A/B in the same
  session had 0.37% drift, so the difference is session noise rather than a
  protocol change; the two A/Bs are not directly comparable in precision.
- 12-47 fails the aggregate top-1 by one row (1532 against 1533). The 16-47
  control reproduced its recorded certificate with a **one-row** top-1
difference (1538 here against 1539 recorded), so this metric has at least
  one row of run-to-run variation and 12-47 is *at* the boundary rather than
  clearly beyond it. It still cannot be certified.
- No kernel-level trace of any extended scope, so the per-layer cost model is
  inferred from the 16-47 trace rather than measured on the extended one.
- The profile constant is outside the recorded `manifest_sha256`, so the gate
  artifact alone would not show which scope ran; the engagement rate is the
  evidence.
