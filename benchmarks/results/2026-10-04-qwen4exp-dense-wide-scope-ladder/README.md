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

| scope | engaged rate | kl_mean | kl_p95 | kl_max | top-1 | worst category | hard gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| **16-47 (control)** | 0.337 | 4.527e-4 | 2.320e-3 | 2.341e-2 | 0.99354 | - | **PASS** |
| 8-47 | 0.421 | 8.839e-4 | 4.647e-3 | 4.146e-2 | 0.99096 | general_en 1.181e-3 | FAIL, 1 scope |
| 0-47 | 0.508 | **1.468e-3** | **7.517e-3** | **5.140e-2** | **0.98837** | general_en 1.753e-3 | FAIL, 6 scopes |

Limits: mean 1e-3, p95 5e-3, p99 2e-2, max 5e-2, aggregate top-1 0.99,
per-scope top-1 0.97, review boundary 0.02.

At 0-47 the mean is **1.47x over**, p95 **1.50x over**, max **1.03x over**, and
top-1 is below the 0.99 minimum; all four categories fail, plus shape `c1` and
transition `steady`. At 8-47 every aggregate gate passes and **one** category
(`general_en`, mean KL 1.1812e-3) is 1.18x over the per-category mean limit.

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
scope tested so far recovers it. The region is not merely tail-sensitive; it
fails the aggregate envelope, and it fails it for every category.

**Open:** the boundary between 8-47 and 16-47 is not located. This project has
already measured a **non-monotonic** scope boundary on a sibling route - the
staged-f32 ladder found 15-47 failing while 14-47 and 16-47 pass - so the
boundary may not be a simple threshold and no scope may be inferred from a
neighbour's result. The value of a passing scope is roughly 0.152 s per added
layer off a ~14 s prefill: about **4.3%** at 12-47 and **2.2%** at 14-47.

## Artifacts

| file | what |
| --- | --- |
| `ab-A-16-47.json` | canonical bench, dense-wide 16-47, arm A |
| `ab-B-0-47.json` | canonical bench, dense-wide 0-47, arm B |
| `ab-C-16-47.json` | canonical bench, dense-wide 16-47, arm C (drift control) |
| `artifact-16-47-control.json` | production gate, 16-47 at HEAD |
| `artifact-8-47.json` | production gate, 8-47 |
| `artifact-0-47.json` | production gate, 0-47 |

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
- No kernel-level trace of any extended scope, so the per-layer cost model is
  inferred from the 16-47 trace rather than measured on the extended one.
- The profile constant is outside the recorded `manifest_sha256`, so the gate
  artifact alone would not show which scope ran; the engagement rate is the
  evidence.
