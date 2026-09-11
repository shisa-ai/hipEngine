# Production Accuracy Policy Review — 2026-08-31

Status: **evidence review and recalibration recommendation; not a normative
threshold change**

Normative policy remains
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md). This review records what the
current policy enforces, how it was calibrated, what practical quality evidence
exists, and which measured performance candidates it has excluded. No candidate
is admitted by this document, and the current layer-2 grouped-WMMA route remains
default-off.

## Assessment

The current envelope is defensible as an **automatic-admission** policy. It has
admitted material 5–17% same-host wins while rejecting candidates with broad
or outlier drift. The evidence does not support raising the mean-KL limit from
`0.001` after observing the current 5% candidate.

The review does identify a policy-design weakness: the evaluator applies the
same point limits to the 450-row aggregate, 100–150-row categories, and an
18-row prefill transition. This makes small scopes both brittle for acceptance
and underpowered as estimates of real-world quality. The current layer-2 result
is therefore a valid rejection under frozen policy, but it is not evidence of a
material task-quality regression.

The recommended next step is a predeclared recalibration campaign that keeps
exact semantic and lifecycle requirements, retains the current automatic lane,
and evaluates a separate borderline-review lane with larger transition samples,
BF16-relative evidence, and real task outcomes. The campaign must freeze its
controls, metrics, and decision rules before re-evaluating candidate identities.

## Current production cutoffs

The defaults are implemented by `EvaluationThresholds` and
`Bf16NoninferiorityThresholds` in
[`hipengine/benchmark/execution_profiles.py`](../hipengine/benchmark/execution_profiles.py)
and documented in
[`EXECUTION-PROFILES.md` section 6](EXECUTION-PROFILES.md#6-production-numerical-gate).

### Same-quant strict-teacher envelope

The evaluator compares full-vocabulary logits at identical strict-teacher
contexts. The following limits apply globally and independently to every
declared category, shape, and transition scope:

| Metric | Binding automatic-admission limit |
| --- | ---: |
| Mean KL, production versus strict | ≤ `0.001` |
| p95 row KL | ≤ `0.005` |
| p99 row KL | ≤ `0.02` |
| Maximum row KL | ≤ `0.05` |
| Overall top-1 agreement | ≥ `99%` |
| Per-scope top-1 agreement | ≥ `97%` |
| Manual-review boundary | any row KL > `0.02` |

Every limit binds together. A row above `0.02` requires explicit top-k,
strict-margin, state, and task diagnosis and is not automatically admitted even
when the `0.05` ceiling passes. Categories and transition scopes cannot
compensate for one another.

The implementation applies the complete mean/p95/p99/max tuple to every scope,
not only top-1. See `_summary_passes()` and `compare_profile_logits()` in the
[evaluator](../hipengine/benchmark/execution_profiles.py).

### Exact and non-numerical requirements

A production candidate must also satisfy all applicable requirements:

- finite logits and recorded numerical state;
- at least three bit-identical repeats under the same seed, manifest, shape,
  and execution schedule;
- exact request, token, position, mask, KV, routing-owner, transaction, graph,
  and lifecycle controls;
- same-width isolation from neighbor and inactive requests;
- a registered strict fallback and immutable variant-manifest provenance;
- every declared task and heldout check passing its predeclared paired
  non-inferiority rule; and
- the applicable dynamic width, cancellation, compaction, graph/eager, and
  lifecycle scenarios.

T3 representation, algorithm, routing-policy, speculative-acceptance, or
sampling changes are not admitted by this same-quant implementation-drift gate.
Production generated-ID equality to strict is diagnostic rather than binding;
strict and `batch_invariant` have stronger identity requirements. These rules
are normative in
[`EXECUTION-PROFILES.md` sections 4–10](EXECUTION-PROFILES.md#4-what-is-exact-in-every-profile).

### BF16-relative gate

When aligned BF16/full-precision logits are available, production must not
consume an unreported additional quality budget. The evaluator defaults are:

| Metric | Limit |
| --- | ---: |
| Candidate additional mean KL versus BF16, relative to strict | ≤ `0.001` |
| Candidate top-1 drop versus BF16, relative to strict | ≤ `1` percentage point |
| Paired prompt bootstrap | 95% interval must remain within both limits |

The implementation applies these checks globally, by category, by prompt, and
to a 10,000-sample paired-prompt bootstrap. The calibration artifact states
that these defaults are binding where BF16 is available but were **not
re-estimated from the same-quant calibration controls**. That limitation is
recorded in the
[threshold calibration artifact](../benchmarks/results/2026-08-16-execution-profile-threshold-calibration.json).

## Fixture granularity changes the effective policy

The current 18-prompt packet contains one prefill-last row and 24 c1 rows per
prompt, for 450 rows total. Percentage cutoffs become discrete counts:

| Scope | Rows | Nominal top-1 rule | Effective minimum | Misses allowed |
| --- | ---: | ---: | ---: | ---: |
| Overall | 450 | ≥ 99% | 446/450 = 99.111% | 4 |
| Code category | 150 | ≥ 97% | 146/150 = 97.333% | 4 |
| Other categories | 100 | ≥ 97% | 97/100 | 3 |
| Prefill-last / prefill-to-c1 | 18 | ≥ 97% | **18/18 = 100%** | **0** |

The same sample-size issue affects p95 and p99. NumPy can compute stable values
for a fixed 18-row fixture, but one prompt can materially move a percentile or
mean. Conversely, 18/18 observed top-1 matches do not tightly estimate the
population rate. The gate is strict as a fixture acceptance rule while still
being weak evidence about deployed task quality.

This is not an argument for averaging away a failed category. It is an argument
for collecting more independent rows in small declared scopes or using a
predeclared confidence-and-task review rule instead of treating every scope as
if it had the 450-row denominator.

## Calibration evidence and limitations

The frozen policy is recorded in:

- the [calibration artifact](../benchmarks/results/2026-08-16-execution-profile-threshold-calibration.json);
- the [calibration worklog](../worklog/entries/20260816T132134.449910Z-lhl-execution-profile-threshold-calibration-0c5cb7.md); and
- [`PRODUCTION-NUMERICS-CAMPAIGN.md`](PRODUCTION-NUMERICS-CAMPAIGN.md#34-calibrated-result).

### Fresh controls

| Control | Expected label | Mean / p95 / p99 / max KL | Top-1 | Result |
| --- | --- | --- | ---: | --- |
| Qwen3.5-0.8B native cluster8 | positive | `0.000244 / 0.000926 / 0.001562 / 0.004529` | 449/450 | pass |
| Qwen3.6-35B K2 | negative | `0.002005 / 0.008400 / 0.039164 / 0.152579` | 444/450 | fail |
| Qwen3.6-35B wave32-tree | negative | `0.001226 / 0.006281 / 0.016881 / 0.059872` | 446/450 | fail |
| Historical peer-wave route transplanted to gfx1151 | portability control | `0.001319 / 0.005218 / 0.016345 / 0.073151` | 441/450 | fail |

All controls repeated deterministically three times. The envelope cleanly
separates this one fresh positive from the two fresh negatives. The
wave32-tree control also shows why 446/450 top-1 alone cannot replace
full-distribution metrics.

### What calibration did not establish

The calibration validated a proposed envelope; it did not statistically fit a
quality boundary from downstream outcomes:

- only one fresh positive control was available;
- the threshold values were frozen unchanged after the controls separated;
- the historical Laguna controls were mostly summary-only and could not create
  new tail or task evidence;
- no borderline positive/negative localized to an 18-row transition was
  included;
- no threshold was selected from a relationship between KL and task scores;
- BF16-relative defaults were not recalibrated; and
- the fresh calibration was gfx1151-local and cannot transfer performance or
  arithmetic conclusions to gfx1100 without a new lane.

The `0.05` maximum ceiling and `0.02` review boundary have the strongest
historical separation: retained historical maxima reached roughly
`0.03–0.044`, while known rejected routes began around `0.059` and extended
above `1.0`. Evidence for applying the `0.001` mean and `0.005` p95 unchanged
to every small scope is weaker.

## 2026-08-31 layer-2 candidate

Primary evidence:

- [complete rejection artifact](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p1-layer2-grouped-profile-rejected.json);
- [reopened performance and trace artifact](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p1-layer2-grouped-reopened.json);
- [rejection worklog](../worklog/entries/20260831T065604.250278Z-lhl-qwen4exp-layer2-profile-rejected-e7b66c.md); and
- [reopened qualification worklog](../worklog/entries/20260831T060658.393488Z-lhl-qwen4exp-p1-layer2-reopened-535dc2.md).

### Measured result

The T2 layer-2 Q5_K grouped-WMMA route passes the aggregate, every category,
repeat determinism, request-local state/control checks, and lifecycle. Its only
binding numerical failure is the 18-row prefill scope:

| Metric | Result | Current cutoff |
| --- | ---: | ---: |
| Overall mean KL | `0.000503` | ≤ `0.001` |
| Overall p95 / p99 / max KL | `0.002653 / 0.006693 / 0.012383` | `0.005 / 0.02 / 0.05` |
| Overall top-1 | 446/450 | ≥ 446/450 |
| Prefill-last mean KL | **`0.001179`** | ≤ `0.001` |
| Prefill-last p95 / max KL | `0.004026 / 0.007332` | `0.005 / 0.05` |
| Prefill-last top-1 | 18/18 | 18/18 effective |
| Rows over `0.02` | 0 | manual review if nonzero |

The route cuts the traced layer-2 role from `371.10` to `88.13 ms` and Q5_K
gate/up from `279.86` to `16.66 ms`. Same-process p508 improves
`90.25` to `95.06 tok/s` (`+5.34%`), and all 20 category-balanced p512 pairs
are faster. These are diagnostic performance results because the complete
profile gate rejects promotion.

### What the proxy metrics imply

If the measured rows were representative:

- overall KL `0.000503` corresponds to about a `0.050%` multiplicative
  cross-entropy/perplexity effect;
- prefill-last KL `0.001179` corresponds to about `0.118%`;
- overall teacher-token NLL changes by `+0.000594` nats, about `+0.059%`
  perplexity; and
- prefill-last teacher-token NLL changes by `+0.00438` nats, about `+0.44%`.

These conversions are descriptive, not task scores. Mean KL does not prevent a
near-tie argmax flip or guarantee long-horizon quality.

The four teacher-forced top-1 flips all keep the strict winner at candidate
rank 2 with full top-5 overlap. Strict margins are `0.025–0.092` logits. No
prefill-last top-1 flips occur.

### Free generation and real-world uncertainty

The candidate repeats exactly on all 18 prompts. Strict and candidate outputs
are exact on 15/18 short, 32-token generations. The three differing beginnings
are readable code/Japanese alternatives, but the artifact explicitly records
`review_performed=false` because the numerical failure already decided the
candidate.

Therefore:

- a user may observe different greedy wording on some prompts;
- no measured task failure is attributable to the route;
- no measured task non-inferiority result exists either; and
- the result cannot establish executable-code correctness, long-form quality,
  structured-output validity, sampled-generation quality, long-context
  accumulation, or BF16-relative quality.

The correct current statement is: **the route shows small, deterministic,
localized implementation drift and no demonstrated material task regression,
but the available packet is insufficient to establish real-world
non-inferiority.**

## Performance excluded by the current envelope

This inventory searches retained post-calibration artifacts and worklogs that
apply or explicitly reference the frozen production limits. Historical
pre-policy rows are included only as calibration context. The candidates
below overlap, use different stack versions, and must not be summed.

### Borderline or policy-sensitive boundaries

| Candidate | Measured opportunity | Binding failure | Evidence and disposition |
| --- | ---: | --- | --- |
| Layer-2 grouped Q5_K WMMA | `+5.34%` p508; 20/20 category-balanced pairs faster | 18-row prefill mean `0.001179` | Largest current directly measured borderline opportunity; [rejection](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p1-layer2-grouped-profile-rejected.json) |
| Q5_1 MMQ suffix 28–47 | about `0.88%` incremental in the suffix screen versus 32–47 | prefill mean `0.00116` | Suffix 32–47 retained; no closure-grade current-stack paired value for the rejected increment; [artifact](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-q5-1-mmq-suffix32-production.json), [worklog](../worklog/entries/20260829T085906.856386Z-lhl-qwen4exp-q5-1-mmq-suffix32-60f624.md) |
| Q4_K MMQ suffix 34/32 | less than about `1%` in the available suffix screen beyond the admitted boundary | prefill p95 `0.005355 / 0.005038` | Suffix 35–47 retained at `-5.16%/-5.01%` p508/p1012; [artifact](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-q4-k-mmq-suffix35-production.json), [worklog](../worklog/entries/20260829T102134.770519Z-lhl-qwen4exp-q4-k-mmq-suffix35-d72ebf.md) |
| All-layer Q4 DP4A decode | no retained all-layer paired rate | 445/450, while all KL gates pass | Safe43 recovers 43/48 layers and `+10.70%` decode; value of the five remaining layers is unmeasured; [suffix24 artifact](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-production-dp4a24-decode.json), [safe43 artifact](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-production-dp4a-safe43-decode.json) |
| QSA flash 27–47 | one additional QSA layer over admitted 31–47 | 445/450 instead of 446/450 | Scoped admission recovered most measured value; later key-parallel arithmetic is retained on 35–47; [artifact](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-qsa-flash31-production.json), [worklogs](../worklog/entries/20260829T203508.060313Z-lhl-qwen4exp-qsa-flash31-production-3f8dd4.md) and [key-parallel follow-up](../worklog/entries/20260829T205924.859818Z-lhl-qwen4exp-qsa-flash-keyparallel-dd567a.md) |
| PARO fast D64 verifier | `1.171x` on the cited heldout path | 67/68 top-1 where 99% requires 68/68; task-decision mismatch | Later broader numerics passed, but complete task/BF16/economics evidence remained open; [numerical artifact](../benchmarks/results/2026-08-24-w7900-paro-fast-verifier-d64-numerical-rejected.json), [review artifact](../benchmarks/results/2026-08-24-w7900-paro-fast-verifier-four-category-review.json), [worklog](../worklog/entries/20260824T063019.079917Z-lhl-paro-mtp-spike-3108e9.md) |

The first row is the largest measured current complete-stack opportunity whose
only observed blocker is a small-scope point threshold. The other borderline
boundaries mostly exclude sub-1% incremental screen value or have recovered
nearly all measured speed through layer-scoped admission.

### Larger rejected candidates that fail more than a borderline cutoff

| Candidate | Measured performance | Quality/control evidence | Why this does not support immediate relaxation |
| --- | ---: | --- | --- |
| Q5_1 wave64 decode | `+6.1%` | mean/p95 KL `0.00256/0.00720`; only 32 rows | Both central and tail limits fail by material margins; [artifact](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-decode-wave64-candidate.json) |
| Full-layer Q5_1 MMQ | about `-22%` p508 versus its strict-era d4x3 baseline | mixed-category and prefill failures, review rows, and a first-run state anomaly | Suffix 32–47 was retained; full-layer result is not cutoff-only; [gate artifact](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-q5-1-mmq-full-layer-gate-failed.json), [worklog](../worklog/entries/20260829T083201.339436Z-lhl-qwen4exp-q5-1-mmq-selected-4407b2.md) |
| All-GDN peer route | `-10.99%` p508 | prefill mean/p95 `0.00167/0.00745`, 445/450 | Later column-warp suffix 27–47 retained a larger `-17.1%/-15.7%` p508/p1012 win; [peer worklog](../worklog/entries/20260829T123702.612022Z-lhl-qwen4exp-gdn-peer35-a225ce.md), [column-warp worklog](../worklog/entries/20260829T185915.254091Z-lhl-qwen4exp-gdn-colwarps-production-0a7106.md) |
| Full-layer WMMA MoE | no closure-grade full-layer paired row | mean KL approximately `0.0059` | Admitted layers 27–47 retain `-4.34%/-5.26%`; full-layer arithmetic is far outside the mean envelope; [artifact](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-wmma-moe27-production.json), [worklog](../worklog/entries/20260829T153503.768809Z-lhl-qwen4exp-wmma-moe27-production-c6f598.md) |
| Full fast-allrows profile | `6.64x` | mean/p95/p99/max `0.0128/0.0555/0.1215/0.8224`, top-1 `94.47%` | Every distribution and top-1 gate fails; [artifact](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-fast-allrows-rejected.json) |

These larger numbers are important optimization leads, but they are not
credible evidence that the current boundary alone is suppressing task-valid
production performance. They require different arithmetic, narrower dispatch,
or complete task/control repair.

## Policy conclusions

### What is supported

- Exact semantic/control ownership, finiteness, deterministic repeats,
  isolation, lifecycle, provenance, and strict fallbacks should remain hard
  requirements.
- The `0.05` maximum and `0.02` review boundary remain useful outer safety and
  outlier controls.
- Global mean and tail metrics are necessary: a candidate can pass 446/450
  top-1 while still showing rejected distribution tails.
- Layer-scoped admission is effective. It has recovered most performance from
  several otherwise rejected broad candidates.
- The current layer-2 candidate must remain default-off under the policy frozen
  before it was measured.

### What is not supported

- The evidence does not show that `0.001` is a universal real-world quality
  cliff.
- The evidence does not justify applying identical point thresholds to every
  18-row transition indefinitely.
- Short greedy exactness cannot substitute for task scoring, and readable
  divergent prefixes cannot be labeled task-valid without review.
- Performance-positive candidates cannot be summed across superseded and
  overlapping stacks.
- The current review cannot authorize a candidate-specific threshold change.

## Recalibration proposal

Create three predeclared decision lanes.

### 1. Hard rejection

Continue immediate rejection for any semantic/control ownership mismatch,
non-finite output, nondeterminism, isolation or lifecycle failure, task
regression, missing strict fallback/provenance, T3 change presented as
same-quant drift, or broad/outlier failure beyond the absolute safety envelope.

### 2. Automatic admission

Keep the current envelope until a replacement policy is calibrated and frozen.
It remains a conservative and operationally useful default lane.

### 3. Borderline review

Before defining numeric review-band limits, freeze a candidate-independent rule
for cases that:

- pass the global envelope;
- pass every exact semantic, state-ownership, determinism, isolation, and
  lifecycle requirement;
- have no row beyond the manual-review boundary;
- fail only a small-scope mean/p95 or one-row top-1 boundary; and
- can supply expanded task and BF16-relative evidence.

This lane would mean “requires more evidence,” not “promoted despite failure.”
The current layer-2 candidate would remain rejected until a newly frozen policy
and complete evidence packet authorize a fresh decision.

### Required recalibration evidence

1. Freeze known positive, negative, and borderline controls before measuring a
   revised policy.
2. Expand every small transition scope beyond the current 18 independent
   prompts; choose sample sizes through a predeclared precision/power analysis.
3. Use prompt-clustered uncertainty for mean KL rather than treating correlated
   rows as independent evidence.
4. Compare strict and candidate against BF16/full precision where feasible.
5. Run task-grounded checks: executable code/tests, structured-output parsing,
   multilingual scoring, retrieval/long-context checks, and longer generation.
6. Include predeclared sampled-generation seeds in addition to greedy output.
7. Re-evaluate all borderline candidates under the same frozen rule rather than
   selecting only the largest speed win.
8. Publish the revised controls, raw hashes, task results, and keep/reject
   outcomes before changing `EvaluationThresholds` defaults.

A likely structural improvement is to keep strict global limits and category
non-compensation while replacing raw point decisions for tiny transition scopes
with larger samples or a confidence-plus-task review. The revised numbers must
come from that campaign, not from choosing a band that includes `0.001179`.

## Review cadence and triggers

Revisit this review at the earliest of:

- completion of the predeclared recalibration packet;
- five additional production T1/T2 candidates landing within 25% of any
  numerical cutoff;
- availability of aligned BF16 evidence for the active Qwen3.8 model;
- a validated task benchmark showing a current pass/reject decision is
  misclassified; or
- **2026-11-30**.

A later review must be a new dated document. Do not rewrite this review's
measurements or conclusions after commit; link a superseding review and state
which evidence changed.

## Reference index

### Normative policy and implementation

- [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md)
- [`TESTING.md`](TESTING.md)
- [`PRODUCTION-NUMERICS-CAMPAIGN.md`](PRODUCTION-NUMERICS-CAMPAIGN.md)
- [`hipengine/benchmark/execution_profiles.py`](../hipengine/benchmark/execution_profiles.py)
- [`scripts/execution_profile_gate.py`](../scripts/execution_profile_gate.py)

### Calibration

- [2026-08-16 threshold calibration artifact](../benchmarks/results/2026-08-16-execution-profile-threshold-calibration.json)
- [2026-08-16 threshold calibration worklog](../worklog/entries/20260816T132134.449910Z-lhl-execution-profile-threshold-calibration-0c5cb7.md)

### Layer-2 decision

- [2026-08-31 complete rejection artifact](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p1-layer2-grouped-profile-rejected.json)
- [2026-08-31 performance recheck artifact](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p1-layer2-grouped-reopened.json)
- [2026-08-31 rejection worklog](../worklog/entries/20260831T065604.250278Z-lhl-qwen4exp-layer2-profile-rejected-e7b66c.md)
- [2026-08-31 recheck worklog](../worklog/entries/20260831T060658.393488Z-lhl-qwen4exp-p1-layer2-reopened-535dc2.md)

### Cutoff-sensitive and rejected performance evidence

- [Q5_1 MMQ suffix 32–47](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-q5-1-mmq-suffix32-production.json)
- [Q5_1 MMQ full-layer rejection](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-q5-1-mmq-full-layer-gate-failed.json)
- [Q4_K MMQ suffix 35–47](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-q4-k-mmq-suffix35-production.json)
- [DP4A suffix24 and all-layer screen](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-production-dp4a24-decode.json)
- [DP4A safe43](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-production-dp4a-safe43-decode.json)
- [QSA flash and key-parallel follow-up](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-qsa-flash31-production.json)
- [Q5_1 wave64 decode rejection](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-decode-wave64-candidate.json)
- [WMMA MoE suffix 27–47](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-wmma-moe27-production.json)
- [Full fast-allrows rejection](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-fast-allrows-rejected.json)
- [PARO fast-verifier numerical rejection](../benchmarks/results/2026-08-24-w7900-paro-fast-verifier-d64-numerical-rejected.json)
- [PARO fast-verifier four-category review](../benchmarks/results/2026-08-24-w7900-paro-fast-verifier-four-category-review.json)
