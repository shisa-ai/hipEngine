# The WMMA 16-47 certification is stale at HEAD

**Result: re-running the certified `wmma` 16-47 dense-prefill gate with the
identical certified configuration no longer reproduces its recorded numbers.
It still passes every hard threshold, but one row now crosses the 0.02 review
boundary, so `requires_outlier_review` is true, `quality_passed` is false and
`measurement_valid` is false.**

The staleness is **independent of chunk size**: the certified chunk-1024
configuration and the chunk-4096 configuration are bit-identical at HEAD, and
both differ from the recorded certificate. Chunk invariance is therefore
re-confirmed at HEAD as a secondary result.

This is a certificate-staleness finding, not a performance result. No speed
claim is made here.

## What was run

Three runs of the same gate, same host, same model, same fixture:

| | recorded | HEAD, chunk 4096 | HEAD, chunk 1024 |
| --- | --- | --- | --- |
| artifact | `2026-09-29-qwen4exp-x6-chunk4096-gate/wmma16-47-chunk4096.json` | `artifact-wmma16-47-chunk4096.json` | `artifact-wmma16-47-chunk1024-at-head.json` |
| commit | `035185fb2` | `f4b12e0c2` | `f4b12e0c2` |
| chunk | 4096 | 4096 | 1024 |
| status | `complete` | `invalid_or_screen_only` | `invalid_or_screen_only` |
| `measurement_valid` | `true` | `false` | `false` |

The two HEAD runs are **bit-identical to each other on every compared field** -
both logits hashes and all five KL metrics to the last digit. Chunk size is
pinned per arm and is not the variable.

Command (identical for both this-packet arms apart from `--prefill-chunk-size`):

```
python3 scripts/execution_profile_q8_wmma_prefill_layers_gate.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --route wmma --decode-steps 128 --repeat-runs 3 --layers 16-47 \
  --prefill-chunk-size <4096|1024> \
  --output <artifact>
```

Protocol `canonical_12_case_d128_three_repeat_numerical_v1`: 12 canonical cases
(code / general_en / general_ja / mixed_ja_en at p512 / p1024 / p4096), 128
decode steps, 3 candidate repeats, 1548 teacher rows.

## Every protocol and route field is identical

Checked field by field against the recorded artifact - `prefill_chunk_size`,
`decode_steps`, `case_count`, `candidate_repeat_runs`, `execution_profile`,
`fixture_sha256`, `profile_manifest_sha256`, `strict_manifest_sha256`,
`route.name`, `route.candidate_layers`, `route.teacher_selector`,
`route.teacher_chain`, `route.candidate_chain`. **All identical.** The two
chunk-4096 runs differ in nothing but the commit.

## What changed

| metric | recorded (`035185fb2`) | HEAD (`f4b12e0c2`), both chunks | |
| --- | --- | --- | --- |
| `strict_logits_sha256` | `550bb9b832bc7ba2…` | `294e6c08d5b2c315…` | changed |
| `candidate_logits_sha256` | `65370710c0b9c40f…` | `10fd0ae50b6eb9ce…` | changed |
| `kl_mean` | 3.795524e-4 | 4.388172e-4 | 1.16x |
| `kl_p95` | 1.761933e-3 | 2.360965e-3 | 1.34x |
| `kl_p99` | 5.424248e-3 | 5.843182e-3 | 1.08x |
| `kl_max` | 1.530440e-2 | 2.341195e-2 | 1.53x |
| `top1_agreement` | 0.9935401 | 0.9941860 | higher |
| rows over review boundary | 0 | 1 | crosses 0.02 |
| `hard_gates_passed` | true | true | - |

The single row over the boundary is `general_en-p512`, teacher step 82, at
KL 2.341e-2 against a `max_kl_max` of 5e-2 and a review boundary of 2e-2.

`quality_passed` is defined in the gate as
`hard_gates_passed and finite and not requires_outlier_review`, so the
boundary crossing - not any hard threshold - is what turns the verdict.

## The cause is the shared baseline, not the route

**The teacher arm's hash changed too.** The teacher arm is the exact
`coltile8_rowbatch4_f32_f32_out` chain with every route selector cleared, and
the gate pins the chunk size in both arms. A teacher whose published logits
change therefore means the *baseline* arithmetic moved, not the WMMA route.

### Dispatch is provably unchanged

Every engagement and selector-state field is **identical** between the recorded
run and the HEAD run:

| field | recorded | HEAD |
| --- | ---: | ---: |
| `candidate.engaged_launches` | 2488860 | 2488860 |
| `teacher.engaged_launches` | 833004 | 833004 |
| `wmma_prefill` candidate / teacher | 0.373585 / 0.0 | 0.373585 / 0.0 |
| `q4k_rowtile` candidate / teacher | 0.996195 / 1.0 | 0.996195 / 1.0 |
| `wmma_route_engaged_candidate_only` | true | true |

So the same kernels ran, in the same proportions, with the same selector
states. The differing logits come from **different arithmetic inside kernels
that both runs dispatch identically** - not from a routing, registration or
memoization change.

**This excludes the leading hypothesis I started with.** `a9a33dd88` added a
component to the dispatch memoization key (`_iu8_role_key`), and a key change is
the most direct way to make a teacher arm resolve differently - but the
engagement evidence above shows the teacher resolved identically, so it did not
move these numbers. The same argument applies to `b9f73e808` (registration
order) and `76dd95d7f` (route removed).

### Bisected: two changes, both decode-path promotions

Binary searches over `035185fb2..HEAD` (88 commits) with a **reduced but
validated-sensitive** protocol - `--case-id code-p512 --case-id general_en-p512
--decode-steps 24 --repeat-runs 3` - identify both boundaries. The reduced
protocol reproduces the recorded certificate's `kl_max` exactly at `035185fb2`
(`1.530440e-02`), which is what makes it usable as a bisect probe.

| | commit | subject | teacher hash |
| --- | --- | --- | --- |
| start | `035185fb2` | (the recorded certificate) | `724d1bde44e23a14` |
| change 1 | **`c66db0264`** | perf: promote the decode dp4a linear to the production default | `9d9f3d4a7a7b1f7f` |
| change 2 | **`6c3c46b04`** | perf: promote DE-3 - the decode-shaped pack8 GEMV runs a 256-wide block at narrow projections | `e91c8bb18bc17106` |
| HEAD | `642d90e05` | | `e91c8bb18bc17106` |

HEAD's teacher hash equals the value after change 2, so **exactly these two
changes** lie in the range and there is no third.

**Both are decode-path promotions, and neither is a kernel swap.**

- `c66db0264` is a **one-line profile change**: it moves `Q4_DP4A64` into
  `PRODUCTION_Q8_QSA_RESTORED_FLAGS` so the binder stops zeroing the route, and
  its own message records "0 dp4a dispatches in the baseline trace, 210 in the
  candidate".
- `6c3c46b04` changes the decode-shaped pack8 GEMV's block width at
  `out <= 640` ("32 threads per output rather than 16; max relative divergence
  2.6e-7") as a runtime dispatch parameter in
  `hipengine/runtime/gguf_linear.py`.

This is consistent with the engagement evidence above: that block counts the
**prefill** selectors (`wmma_prefill`, `q4k_rowtile`) and does not cover either
decode route, which is why a genuine arithmetic change produced no visible
difference there.

### The systemic point

This gate certifies a **prefill** route, but measures its drift over **128
teacher-forced decode steps**. Its baseline is therefore decode-path
arithmetic, and **any promoted decode-path arithmetic change invalidates a
prefill-route certificate** - without touching the route under test. Both
changes above did exactly that, and neither is a defect in the route.

The route under test was not changed by either commit. What moved is the
measurement's baseline, and with it the recorded drift. That is the sense in
which the certificate is stale, and it is why the correct remedy is a re-gate
rather than a revert.

### Corrections to hypotheses named earlier

Two earlier revisions of this file named causes that the bisect refutes, and
both are recorded rather than quietly dropped:

- `a9a33dd88` (the dispatch cache-key fix) was named first, on the reasoning
  that a memoization-key change is the most direct way to make a teacher arm
  resolve differently. The engagement comparison refutes it.
- `6c3c46b04` (DE-3) was then named as the leading candidate. **It is real, but
  it is the second change, not the first.**

Both were named on plausibility before the cheap decisive check was run: for the
first, a 30-second diff of the two artifacts' engagement fields; for the second,
the bisect itself.

One observation that fits a near-tie explanation rather than a quality one: the
HEAD run has **better** top-1 agreement (0.9942 vs 0.9935) while its max KL is
worse (2.341e-2 vs 1.530e-2). Max KL on this fixture is a tail statistic
driven by near-ties, and a small arithmetic perturbation can move which
near-ties flip.

## Why this matters

`wmma` 16-47 is the **production default** for the wide dense-prefill shapes,
and the recorded certification is what admits it. That certification was
obtained at `035185fb2`, before the decode-path promotions. `git log
035185fb2..HEAD` shows no newer gate artifact for this route, so the route has
been running in production with a certificate that HEAD no longer reproduces -
at **either** chunk size.

The route still clears every hard gate - mean 4.39e-4 against 1e-3, max 2.34e-2
against 5e-2, top-1 0.9942 against 0.99 - so this is a **review requirement, not
a demonstrated quality failure**. It is the same class as the DP-8 composition,
which passed every gate with two rows over the boundary and was held for review.

## Scope and limits

- One model, one quant, one host (gfx1151 / Radeon 8060S), one execution
  profile, one route and layer set.
- `profile_qualification_claim` is false in all three arms: category-heldout,
  strict-teacher/BF16-relative, control/isolation/graph/lifecycle, task-quality
  and engagement/repair-oracle certificates are untouched by this packet.
- This is not a timing measurement and makes no performance claim.
- The two HEAD arms differ only in chunk size and are bit-identical, which is
  what makes the finding about code state rather than about chunking. It does
  **not** by itself say which commit moved the numbers.
- Environment note for reproduction: `hipcc` must come from the therock env's
  wrapper (`.../envs/therock10-staging-20260828/bin/hipcc`). The raw
  `_rocm_sdk_devel/bin/hipcc` fails to link `-lamdhip64` because
  `_rocm_sdk_core/lib` ships only the versioned `libamdhip64.so.7`.
