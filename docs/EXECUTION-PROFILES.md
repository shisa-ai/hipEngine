# Execution Profiles and Numerical Contracts

Status: **approved architecture; evaluator, fail-closed runtime plumbing, and production threshold calibration implemented; model-plan certification pending**
Approved: 2026-08-16
Authority: [`PLAN.md`](PLAN.md) remains the project architecture source of
truth. This document is the normative policy for arithmetic drift,
determinism, and batch-composition guarantees.

## 1. Why profiles exist

Correct inference state and identical floating-point arithmetic are different
contracts. hipEngine must never trade away request identity, positions, masks,
KV ownership, or transaction semantics for speed. It may, however, use a
batch-width-specific reduction, WMMA schedule, fused expression, or online
softmax whose arithmetic is not bit-identical to the reference when the
resulting model drift is tightly bounded and task quality is non-inferior.

Three properties must therefore be named separately:

1. **Semantic/control correctness** — the right request owns the right token,
   state, KV pages, positions, masks, graph bucket, and sampler stream.
2. **Repeat determinism** — an identical request and execution schedule repeat
   the same result.
3. **Composition invariance** — the result also remains the same when physical
   slot, neighbors, width, admission order, or compaction changes.

The first is mandatory in every profile. The second is mandatory for retained
strict and production routes. The third is an explicit reproducibility
contract rather than a universal serving requirement.

## 2. Public profiles

| Profile | Intended use | Arithmetic contract | Determinism contract | Batch-composition contract |
| --- | --- | --- | --- | --- |
| `strict` | Oracle, debugging, regression localization, parent/reference parity | Uses the registered reference arithmetic for the selected model/quant/KV policy. Fused kernels must match their declared strict fallback contract. | Same request, seed, shape, and schedule are bit-stable on retained fixtures. | Not by itself a promise about sampler scheduling or every dynamic server composition; use `batch_invariant` for that public guarantee. |
| `production` | Normal deployment and performance work | Exact control/state ownership with tightly bounded same-quant implementation drift. Width- and shape-specific arithmetic is allowed. | Same request, seed, resolved variant manifest, and execution schedule are deterministic. | Cross-width generated-ID equality is diagnostic, not a promotion requirement. Same-width neighbor data must never contaminate a request. |
| `batch_invariant` | RL, evaluations, debugging, reproducible serving | May use any independently validated stable arithmetic, but must preserve the request result across supported batch compositions. The first implementation may alias strict routes. | Fixed-seed repeats are deterministic. | Same fixed-seed request result across physical slots, neighbor prompts, supported widths, admission order, cancellation of peers, and compaction. |

There is no public `relaxed_all`, `fast`, or `aggressive` fourth profile. More
aggressive weight/KV representation changes, approximate routing, speculative
acceptance changes, and sampling relaxations remain explicit experiments until
the first three contracts are implemented and measured.

### 2.1 Migration and default

The current tree predates this contract and contains a mixture of exact and
quality-gated backend defaults. Those routes are **not automatically
grandfathered** into `production`.

Profile plumbing must land without silently changing current public behavior.
During the bounded migration, an omitted profile may preserve the pre-profile
selection internally, but that behavior is not a fourth named profile and must
be tracked for removal in [`REFACTOR.md`](REFACTOR.md). The public default may
change to `production` only after:

- the evaluator and profile manifest are retained;
- current non-exact defaults have been re-certified or replaced by strict
  fallbacks;
- the task-quality and applicable dynamic-serving candidate gates pass; and
- the complete serving packet, where applicable, shows the candidate is
  non-regressive against the current default under the declared SLO protocol.

There is **no minimum percentage threshold** for changing a default. Every
measured, correctness-qualified, non-regressive improvement is retained and
promoted within its validated scope; small wins accumulate across kernels and
features. A default may remain blocked only by a concrete correctness,
ownership, determinism, resource, applicability, or candidate-caused SLO
regression—not because an individual win is deemed too small. A pre-existing
product/SLO failure shared by control and candidate is tracked separately and
does not erase a candidate improvement.

### 2.2 First ZBook c1/cN default decision

The 2026-08-16 Qwen3.6 GGUF package-level campaign retains the incumbent
implementation routes but **does not change the public profile default**. The
actual bundle (cooperative c1 router, direct Q8T16 c2, rowtile c4/c8) is exact
against strict over 1,050 static/dynamic/sparse full-logit rows and passes a
separate c8 lifecycle control for tokens, ownership, masks, cancellation,
re-admission, compaction preservation, graph invalidation, session reuse, and
clean drain. Seven paired graph runs retain small c4/c8 wins.

The complete production-server packet nevertheless fails soak completion:
87/120 requests complete exactly and 33 are rejected under sustained offered
load. That shared serving failure remains a product/scheduler blocker, but win
magnitude is not. Omitted-profile package behavior stays unchanged only because
no route is certified through a named runtime profile and the task/BF16/control
schema debt remains open.
The compact evidence and raw hashes are in
[`2026-08-16-zbook-qwen36-production-profile-cn-blocked.json`](../benchmarks/results/2026-08-16-zbook-qwen36-production-profile-cn-blocked.json).

### 2.3 Qwen3.8 FP16 recurrent-state default decision

The 2026-08-20 gfx1151 Qwen3.8 `Q4_K_S` FP16 recurrent-state route remains an
explicit opt-in and does **not** become a named/public `production` default.
Its complete packed numerical, determinism, isolation, and ownership hard gate
passes, and the engine packet measures about 3% c4/c8 decode improvement.
The predeclared serving screen fails static-c8 ITL-p99 in both FP32 and FP16
modes (`0.8532/0.8287 s > 0.5 s`), but FP16 improves exact c8 server throughput
by `+1.33%` and is non-regressive at c1. The shared absolute SLO failure remains
a serving-path blocker, not a reason to discard or withhold this scoped default
improvement. The SLO is not relaxed after observing the result.

No runtime profile manifest is registered for this candidate. Its measured
FP32 denominator is the same compact-peer production arithmetic with FP32
state storage, not a certified model-level `strict` plan; labeling that route
public `strict` would violate this contract. The prior magnitude-based default
rejection is superseded; the scoped legacy-default promotion is handled through
the backend capability/default path while named-profile migration remains open.
Evidence:
[`serving rejection`](../benchmarks/results/2026-08-20-gfx1151-qwen38-27b-fp16-state-serving-screen-rejected.json)
and [`retained opt-in packet`](../benchmarks/results/2026-08-20-gfx1151-qwen38-27b-r2-fp16-state-repaired-production.json).

## 3. Profile is orthogonal to model representation

An execution profile selects implementation arithmetic and reproducibility. It
does not silently choose a different model or storage policy.

The following remain explicit, independently reported axes:

- model artifact and revision;
- weight quantization and repacked layout;
- KV storage policy and scale format;
- sampling method and parameters;
- speculative provider, acceptance policy, and draft depth;
- execution profile.

For example, `production + Q4_K_M + BF16 KV` and
`production + Q4_K_M + INT8 KV` are different product configurations. A same-
quant implementation-drift result cannot authorize a Q4-to-IQ4, BF16-to-INT8
KV, approximate-router, or greedy-to-probabilistic change.

## 4. What is exact in every profile

The following are control-plane or ownership semantics. Any mismatch is a bug,
not acceptable numerical drift.

| Surface | Exact requirement |
| --- | --- |
| Request identity | `request_id <-> scheduler slot <-> physical execution row` maps and response routing are correct at every transition. |
| Token ownership | Prompt slices, current token, generated-token accounting, stop handling, and per-request output queues never cross requests. |
| Positioning | Token positions, context lengths, RoPE positions, causal visibility, and graph position publication are correct. |
| Masks | Active, causal, finish, eviction, rollback, sparse-retirement, and verifier-parent masks match the declared scenario. |
| KV metadata | `KVLiveSpans`, block/page ownership, append destination, live count, base offset, token position, eviction, commit, and rollback metadata are exact. |
| Stateful ownership | Conv/GDN/SSM and recurrent-state buffers are indexed by the correct stable request and layer; admission, cancellation, compaction, and width changes cannot exchange state. |
| Graph/dispatch metadata | Resolved profile, variant manifest, graph bucket, row maps, and fallback decision match the declared run. |
| Sampling accounting | Per-request RNG stream/counter, seed ownership, accepted-token count, and speculative transaction accounting are correct. |
| Lifecycle | Allocation ownership, teardown, reclaim, and stale-pointer protections remain exact and leak-free. |

### 4.1 Numerical values that may differ in production

Within the quality budget, `production` may differ from `strict` in:

- BF16/FP16/FP32 intermediate values;
- KV and recurrent **values** produced by the same declared storage policy;
- reduction and split-merge association;
- softmax/PV association;
- fused-expression contraction and compiler scheduling;
- logits and generated IDs at near ties;
- MoE expert choices caused by bounded upstream numerical differences.

The corresponding ownership, valid ranges, finiteness, and scatter/gather maps
must still be correct. Approximate routing that intentionally changes top-k or
prunes route mass is a separate representation/algorithm experiment and is not
a normal `production` implementation-drift optimization.

### 4.2 Isolation versus composition invariance

Production does not promise that c1 and c8 use identical arithmetic. It does
promise isolation:

- replacing a neighbor prompt at the same physical width cannot inject that
  neighbor's data into the observed request;
- permuting rows while preserving the same row-local inputs and width must map
  outputs back to the correct requests;
- inactive rows cannot affect active-row state or KV;
- a width transition may change future floating-point association, but cannot
  lose, duplicate, or transfer authoritative state.

`batch_invariant` adds equality across widths, slot placements, admission order,
and compaction.

## 5. Arithmetic-source classification

The source class documents *why* a candidate differs. It does not waive any
whole-model gate.

| Class | Description | Initial profile eligibility |
| --- | --- | --- |
| T0 | Strict/reference arithmetic, including exact fused kernels and layout-only changes that preserve declared output bytes | All profiles |
| T1 | Local implementation drift: contraction, approximate intrinsic, or lower-precision intermediate with unchanged algorithm and representation | `production` after full gate |
| T2 | Association/layout drift: reduction reorder, split-K/online merge, WMMA accumulation order, fused chain reassociation, width-specific arithmetic | `production` after full gate |
| T3 | Representation, algorithm, or decision-policy change: weight/KV quant change, approximate routing, changed speculative acceptance, changed sampling distribution | Explicit experiment/product configuration only; not admitted by the initial campaign |

A candidate declaration must name its class, affected model/layers/shapes,
stateful surfaces, expected performance mechanism, strict fallback, and whether
it can alter downstream discrete decisions.

## 6. Production numerical gate

The initial gate compares the same model artifact, quant, KV policy, prompts,
teacher tokens, and positions under `strict` and candidate `production`.
Generated free-running ID equality is recorded but is not the denominator.

### 6.1 Calibrated production envelope

The 2026-08-16 calibration freezes the initial envelope unchanged. These are
binding automatic-admission limits, not tuning targets:

| Metric over full-vocabulary teacher-forced rows | Requirement |
| --- | ---: |
| Mean KL, production versus strict | <= `1e-3` |
| p95 row KL | <= `5e-3` |
| p99 row KL | <= `2e-2` |
| Maximum row KL | <= `5e-2` absolute ceiling |
| Top-1 agreement | >= `99%` overall |
| Top-1 agreement | >= `97%` in every declared category/shape/transition scope |

All global and per-scope limits bind together. Rows with KL above `2e-2`
require explicit top-k overlap, strict logit-margin, finite-state, and
applicable task diagnosis even when the absolute `5e-2` ceiling passes; they
are never admitted automatically. Every applicable task/heldout check must pass
its predeclared paired non-inferiority margin. There is no universal task score
and categories cannot compensate for one another.

The calibration used the backend-registered strict GDN route and full logits for
18 prompts/450 teacher-forced rows, with three bit-identical repeats. Native
gfx1151 Qwen3.5 cluster8 passed at mean/p95/p99/max KL
`0.000244/0.000926/0.001562/0.004529` and `99.778%` top-1. Fresh Qwen3.6 K2 and
wave32-tree controls failed at mean/p95/max KL
`0.002005/0.008400/0.152579` and `0.001226/0.006281/0.059872`; wave32-tree still
had `99.111%` top-1, demonstrating why top-1 alone is insufficient. The
historically accepted gfx1100 peer-wave route also failed when transplanted to
current gfx1151 (`0.001319/0.005218/0.073151`, `98.0%` top-1), so its old label
was not grandfathered across backend/current arithmetic. See the compact
[`calibration artifact`](../benchmarks/results/2026-08-16-execution-profile-threshold-calibration.json).

Historical retained summaries still explain the `2e-2` review and `5e-2`
ceiling: accepted maxima reached about `0.03-0.044`, while known rejected routes
began around `0.059` and extended above `1.0`. Missing raw logits cannot create
new tail evidence or qualify those routes. Mean, tails, category localization,
repeatability, and task behavior remain binding together.

The broad project floor, KL <= `0.05` and top-1 >= `90%` versus a CPU/reference
oracle, remains a useful new-kernel smoke and an outer safety ceiling. It is not
sufficient by itself for the default production profile.

### 6.2 BF16-relative non-inferiority

Where a BF16/full-precision teacher is available, report both:

- strict selected-quant versus BF16; and
- production selected-quant versus BF16.

Production must not consume an unreported additional quality budget. Use paired
prompt/category deltas and confidence intervals where the fixture count permits
it; at minimum report mean/p95/max KL and top-1 deltas by category. This gate
assesses implementation drift, not whether the selected quant is globally
identical to BF16.

### 6.3 Stateful and dynamic scenarios

Every stateful or c>N route must include the applicable matrix:

- c1/c2/c4/c8 fixed batches;
- ragged prompt and decode lengths;
- arbitrary prompt composition and row permutations;
- sparse active masks and retirement;
- delayed arrivals during decode;
- cancellation and reclaim;
- c1<->cN grow/shrink transitions;
- optional compaction;
- page boundaries, ring wrap, eviction, commit, and rollback;
- graph/eager parity and repeated replay.

The strict trajectory supplies teacher tokens so all profiles are compared at
identical contexts. Free-running divergence is a diagnostic, not a substitute
for this comparison.

### 6.4 Determinism and task quality

A retained production manifest must:

- repeat identically for at least three fixed-seed runs with the same manifest
  and execution schedule;
- remain finite at every recorded layer/state/logit boundary;
- pass the complete multi-category prompt suite and applicable heldouts;
- pass task-specific checks such as code execution/tests, structured-output
  parsing, retrieval/long-context checks, multilingual scoring, or agent-tool
  schema validation when the product path claims those capabilities; and
- show no material task-level regression versus strict under a predeclared
  paired criterion.

A route cannot compensate for a failed category by averaging it with easier
categories.

## 7. Strict and batch-invariant gates

### 7.1 Strict

Strict remains the primary bug-localization oracle. New fused/ported strict
variants require their declared exact or parent-parity RED test, the CPU/
reference correctness floor, and the expected kernel trace. A strict fallback
must remain registered for every production composite.

Strict guarantees are scoped to the declared model/quant/KV/backend and fixture.
An external engine using a different arithmetic implementation is a comparison
oracle, not the definition of hipEngine strict bytes.

### 7.2 Batch invariant

The batch-invariant gate holds one request's prompt, sampler configuration, and
seed fixed while varying:

- physical slot;
- neighbor prompts and neighbor lengths;
- supported batch width;
- admission order and delay;
- cancellation/retirement of peers; and
- compaction timing.

The request's generated IDs and declared returned probabilities/logits must
match according to the public API contract. Metadata and ownership are exact.
Performance is reported as the reproducibility tax versus production; no
minimum speedup is required.

## 8. Registry and runtime architecture

Execution profile is a selector over the existing
`(backend, layer, quant, variant)` registry, **not a fifth registry axis**.

At model/session construction, the public profile resolves to an immutable
variant plan containing:

- profile name and schema version;
- backend/model/quant/KV identities;
- selected variant per layer/shape bucket;
- strict fallback per selected production variant;
- graph bucket policy;
- calibration/evidence artifact identifiers; and
- a stable manifest hash.

Dispatch and graph capture consume the resolved variants. Engine/model hot
paths must not grow `if profile == ...`, `if backend == ...`, or
`if quant == ...` branches. Experimental environment variables may select a
candidate while it is under test, but retained behavior must be available
through the public profile/variant plan and recorded in logs and artifacts.

Missing or uncertified production variants fail closed to the registered strict
fallback. Unsupported batch-invariant scenarios either use a certified strict
fallback or reject clearly; they do not silently run production arithmetic.

The public selectors are `LLM(..., execution_profile=...)`, server
`--execution-profile`, and `HIPENGINE_EXECUTION_PROFILE`. Resolution is a
cold-path plugin registry keyed by model/backend/quant/profile. Every plan names
real `(backend, layer, registry_quant, variant)` keys; resolution verifies the
selected and strict-fallback keys are registered before constructing the model
plugin's profile-specific factory or invoking its binder. Production and
batch-invariant plans may override only a subset of strict scopes; absent scopes
are written into the manifest as strict selections. Captures bind to the
resulting immutable manifest hash.

During migration, omitting the selector bypasses the profile-plan registry and
preserves the incumbent package behavior. An explicit selector never falls back
to that unclassified route: without a registered strict plan it errors, and
without a certified production/batch-invariant override it constructs the
registered strict plan while reporting `fell_back_to_strict`.

## 9. Evidence and promotion

Every profile-sensitive artifact records:

- execution profile and profile-schema version;
- variant-manifest hash and selected/fallback variants;
- backend, hardware, software stack, model hash, quant, and KV policy;
- workload shape and dynamic scenario schedule;
- prompt-suite and heldout hashes;
- teacher source;
- mean/p95/p99/max KL and top-1 by category/shape/transition;
- finiteness, metadata, isolation, determinism, graph/eager, and lifecycle
  verdicts;
- task-quality verdicts;
- exact command and performance metrics; and
- whether generated-ID equality is binding or diagnostic for that profile.

Promotion is shape- and backend-qualified. A candidate that fails one width or
context may be retained only if dispatch excludes that scope and the artifact
states the exclusion. No benchmark prompt, token ID, or heldout result may be
hardcoded into selection.

## 10. Automatic rejection

Reject or fall back to strict on any of the following:

1. request/slot/token/position/mask/KV/state-ownership mismatch;
2. state contamination from a neighbor or inactive row;
3. invalid commit, rollback, eviction, or sampler accounting;
4. NaN/Inf or nondeterminism under an identical schedule;
5. any binding category/shape/transition threshold failure;
6. task-level material regression;
7. missing strict fallback or unrecorded profile/variant provenance;
8. a performance claim without the same-suite quality packet; or
9. prompt-, token-, or candidate-specific benchmark gaming.

A failed threshold is not fixed by relabeling a bug as numerical relaxation.
Budgets move only through an explicit policy decision backed by calibration
and task evidence.

## 11. Related documents

- [`PRODUCTION-ACCURACY-POLICY-REVIEW-2026-08-31.md`](PRODUCTION-ACCURACY-POLICY-REVIEW-2026-08-31.md) — dated evidence review of the frozen cutoffs, calibration limits, practical impact, excluded performance, and recalibration triggers; it does not change this normative policy.
- [`PRODUCTION-NUMERICS-CAMPAIGN.md`](PRODUCTION-NUMERICS-CAMPAIGN.md) — active
  implementation, calibration, historical-recovery, and c1/cN campaign.
- [`RELAXED.md`](RELAXED.md) — historical relaxed-mode inventory and provenance;
  no longer the normative public-profile policy.
- [`TESTING.md`](TESTING.md) — concrete test tiers and fixtures.
- [`BENCHMARK.md`](BENCHMARK.md) — performance protocols and artifact rules.
- [`CONCURRENCY.md`](CONCURRENCY.md) — serving scenarios and ownership gates.
- [`KERNELS.md`](KERNELS.md) — kernel fallback, lineage, and trace requirements.
