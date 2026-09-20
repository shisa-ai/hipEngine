---
status: normative
owns: Strict/production/batch-invariant contracts, numerical gates, exact ownership and failure-containment semantics, and registry resolution policy.
---
# Execution Profiles and Numerical Contracts

Status: **approved architecture; evaluator, fail-closed runtime plumbing,
production threshold calibration, and pre-device execution-failure containment
implemented; wider containment classes and per-request group partitioning
pending; model-plan certification pending**
Approved: 2026-08-16
Authority: [`PLAN.md`](PLAN.md) remains the project architecture source of
truth. This document is the normative policy for control/ownership correctness,
execution-failure containment, arithmetic drift, determinism, and
batch-composition guarantees.

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

Failure belongs to the first property: recovery must preserve ownership and
valid state outside the affected scope. Section 4.3 states the containment
contract and section 4.4 separates per-request capability from group scheduling.

### 1.1 Missing evidence is not a runtime failure

An implemented path that satisfies its declared input, resource, and execution
contracts is presumed runnable. Test it, observe its behavior, and fix failures;
do not presume it invalid because its exact prompt length, output horizon,
width, or workload has not appeared in a benchmark. This is a development and
admission rule, not a claim that untested code is proven correct.

Keep three questions separate:

- **Can it execute?** Check implemented semantics, compatible storage/layout,
  allocated bounds, ownership, and available resources. Missing code, violated
  preconditions, and known failures are concrete reasons to reject or select
  another path; missing benchmark rows alone are not.
- **What does evaluation show?** Make implemented candidates reachable through
  the real runner and serving path, with selected variants and fallback reasons
  observable. Use representative workloads, longer and mixed requests, targeted
  fault/transition tests, and quality/performance controls to discover behavior.
  Do not require successful evaluation as a prerequisite for running that
  evaluation. Preserve focused reproducers for known failures.
- **What may we claim or promote?** Production arithmetic changes and published
  quality/performance claims still require the applicable gates in this document.
  An unmeasured result is not a measured failure or a certified success. Evidence
  must cover the algorithm and relevant regimes/transitions; it is not an
  exhaustive allowlist of individual requests or every integer context length.

A restriction must name its concrete cause and scope: an implementation bound,
unsupported semantics, resource pressure, a schedule transition, or an observed
failure. A guard that exists only because a case was not tested is an evaluation
and cleanup task, not a permanent safety boundary. During its removal, provide
an explicit runnable evaluation path rather than silently returning a fallback.
A graph-bucket miss may select eager execution for that cycle; it does not by
itself invalidate later cycles or other requests.

Normal execution of untested code is different from continuation after an actual
failure has left device or shared state uncertain. The latter requires the
recovery evidence in section 4.3. This correction changes policy, not runtime
selection: existing evidence-table and override restrictions are described in
section 2.9 and are not removed by editing this document.

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

**Default decision (2026-09-12, v0.5.0).** An omitted profile now resolves to
`production` for any `(model, backend, quant)` combination that has both a
registered strict plan and a certified production plan. A combination with no
registered plan keeps the migration path, which remains tracked for removal in
[`REFACTOR.md`](REFACTOR.md). The registered combinations are
`qwen3_5_gguf`/`hip_gfx1100`/`gguf_q4_k_m`,
`qwen3_5_gguf`/`hip_gfx1151`/`gguf_q4_k_m`,
`qwen3_5_moe_gguf`/`hip_gfx1100`/`gguf_q4_k_m`,
`qwen3_5_moe_gguf`/`hip_gfx1151`/`gguf_q4_k_m`,
`qwen3_5_moe_paro`/`hip_gfx1100`/`w4_paro`,
`qwen4_exp_gguf`/`hip_gfx1151`/`gguf_q4_k_m`, and
`qwen4_exp_gguf`/`hip_gfx1151`/`gguf_ud_q4_k_xl`. Each one is covered by a
decision section below and keeps a registered strict fallback, so
`execution_profile="strict"` still selects exact arithmetic, and
`batch_invariant` still falls back per scope wherever its composition gate has
not passed.

The §2.2 ZBook soak failure (87 completed and 33 rejected of 120) is present
in both the migration arm and the production arm, so under the shared-failure
rule above it stays tracked as a product/scheduler blocker and does not block
this arithmetic default.

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
magnitude is not. Omitted-profile package behavior stayed unchanged at the time
only because no route was certified through a named runtime profile and the
task/BF16/control schema debt remained open; the named plan is now registered
and §2.1's default decision applies, while the soak failure stays tracked as
the shared product/scheduler blocker it is.
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

### 2.4 Qwen3.8 Q4_K_M production C2/K3 decision

**September 12, 2026 correction:** the unbounded gfx1151 production binder
now allocates FP32 recurrent state. The actual omitted-profile C1 D128
FP16 composition fails max KL at 0.07664; packed D128 fails at 0.22994.
Neither short verifier evidence nor the different Q4_K_S artifact authorizes
unbounded FP16 allocation. The FP32 replacement manifest
`c4a4a342e2243c2dcc430174606dde682393a2bd2e30acc83129027fcf572acc`
passes the 18-prompt C1 D128 numerical gate (2,322 rows, max KL 0.003451,
99.914% top-1, three deterministic repeats). Q4 rowtile selections and
strict fallbacks stay registered. The historical FP16 verifier cells below
retain their own manifest identity; they are not certificates for this
replacement. The final public C2/C4/C8 D128 packet passes 8,716 rows at
KL0/top1 100%, and the declared greedy blocking/SSE/cancellation/refill gates
pass. Production remains AR-only for automatic and explicit MTP requests;
strict C1/K3 uses its original qualified natural25 scope. (The scope axes in
this paragraph are historical: since [2.9](#29-serving-admission-is-physical-not-shape-scoped),
context, horizon, session length, and the manifest hash no longer gate serving.)
See [serving closure](../benchmarks/results/2026-09-12-gfx1151-qwen38-serving-mtp-closure.json)
and [headline evidence](../benchmarks/results/2026-09-12-gfx1151-qwen38-final-headline-refresh.json).
Evidence: [FP16 default rejection](../benchmarks/results/2026-09-12-gfx1151-qwen38-named-fp16-default-d128-rejected.json),
[FP32 C1 gate](../benchmarks/results/2026-09-12-gfx1151-qwen38-fp32-default-c1-qualified.json).

The 2026-08-28 gfx1151 Qwen3.8 `Q4_K_M` production manifest qualifies one
bounded T1+T2 serving cell: FP16 recurrent state plus standard-Q4 singleton and
gate/up rowtiles for packed C2/K3 physical R8. Six actual Q4 shapes select the
rowtile association; narrow K and every strict/profile/shape miss retain
registered small-M/shared-B or dual-WMMA fallbacks.

The binding D24 strict-teacher packet covers 240 canonical and 192 category-
heldout full-logit rows with three deterministic repeats. Canonical
mean/p95/p99/max KL is `6.056e-5/3.389e-4/5.657e-4/0.001155` and top-1 is
`99.583%`; heldout maximum KL is `0.000727` and top-1 is `100%`. Control
positions/input tokens remain exact. A D120 diagnostic fails the absolute tail
ceiling at max KL `0.08574`; therefore no long-horizon authorization transfers
from this result.

The automatic context1-128/D24 C2/K3 cell measures **17.031 vs 14.887 tok/s
(1.1441x AR)** with every category non-regressive. Static eligibility carries
an evidence bound of max group2 before independent requests form the resident
group; the realized C2 row remains the sole K3 owner. C3-C8, group-row and
resident-capacity misses, non-greedy sampling, and all identity/backend/quant/KV
misses select K0. Evidence:
[`production C2 result`](../benchmarks/results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json).

### 2.5 Qwen3.8 Q4_K_M production C3 rowtile decision

The same model/profile independently qualifies shape-scoped verifier rowtile
association at the three observed physical-C3 cells: K1/R6, K2/R9, and K3/R12.
R9 and R12 decompose into R7+R2 and R8+R4; R6 launches directly. The production
manifest hash is `af20ee3b22921dc9a0c988dd1c3f5c471932f0ecda4e557ec2ba4bbc8ef5d95f`;
the registered strict rollback hash is
`393155123c5e09700ff017f949f338fb5f519579e2f05bea3ffef7a43a09a71b`.
Strict profile, C4+, R16+, peer backends, narrow K5120/N1024 Q4, and unlisted
shapes retain their prior owners.

Each cell passes 240 canonical plus 192 category-heldout D24 full-logit rows
with three deterministic repeats (**1,296 rows total, 100% top-1**). The maximum
KL across all six packets is `0.0008685`. Positions/input ownership and teardown
are exact. Same-width D12 isolation is full-logit bit-identical when replacing
both neighbors and moving the observed request from slot 0 to slot 1.

The retained ten-prompt C3/K3 route improves **19.070 -> 19.934 tok/s (+4.53%)**
and **0.9200x -> 0.9589x true AR**, with 10/10 exact task cells. A clean R9
trace reduces steady target wall **495.37 -> 195.16-196.37 ms** and names the
expected Q4/Q5/Q6 rowtiles. Aggregate and `mixed_ja_en` complete wall still
trail AR, so this is a retained production-profile implementation association,
not an automatic serving promotion: C3 remains K0. Authorization is D24-only.
Evidence: [`production C3 rowtiles`](../benchmarks/results/2026-08-28-gfx1151-qwen38-c3-production-rowtiles-retained.json).

### 2.6 Qwen3.8 gfx1100 production C7 periodic-strict fused R28 decision

The W7900 Qwen3.8 `Q4_K_M` manifest qualifies one C7/K3/R28 T2 gate/up
schedule: layers 0/8/…/64 retain exact grouped-rowtile projections plus
standalone SiLU, while the other 56 of 65 layers use the fused row32 WMMA
sibling. The fixed schedule is model/layer/shape policy, never prompt-, token-,
or candidate-conditioned. Strict, omitted-profile behavior, explicit rollback,
rows other than 28, and scope misses retain exact registered fallbacks.

The final strict-teacher packet covers 1,922 full-vocabulary rows over all ten
category+heldout prompts. Mean/p95/p99/max KL is
`0.0000557/0.0003267/0.0008558/0.0015884`, top-1 is 100% globally and in every
category/shape/transition scope, and three candidate runs are bit-exact. The
tracked-clean counterbalanced server gate improves **76.510 -> 81.641 tok/s
(+6.71%)**, wins all 20 prompt-order cells and every category/heldout slice,
and preserves all generated IDs, acceptance sequences, and lifecycle
accounting. The current production/strict manifest hashes are
`2adc137a32d65bc63619947577f5233548d5835a474713abe270d666122a1960` and
`52a3d5b8b02c4dc8230c8c9dc8e43b01135db7ae1b44b027fc8915d66bedcdbb`;
the production hash advanced when §2.7 added its independent C8 Q6 scope, not
because this C7 schedule changed. Evidence:
[`periodic-strict fused R28`](../benchmarks/results/2026-09-02-w7900-q4km-k3-c7-fused-r28-periodic-strict-retained.json)
and [`manifest continuity`](../benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json).

### 2.7 Qwen3.8 gfx1100 production C8 grouped-Q6 DP4A decision

The W7900 Qwen3.8 `Q4_K_M` production plan admits the grouped q8_1 DP4A
planar-Q6 owner only at physical C8/K3. This is T1 implementation arithmetic:
activations are quantized to q8_1 per call while the model representation,
algorithm, target ownership, and acceptance policy remain unchanged. The
strict profile and explicit zero retain the registered grouped BF16 chain.
Runtime admission additionally requires gfx1100, model metadata containing
`Qwen3.8`, file type `MOSTLY_Q4_K_M`, and request count 8; C1-C7 and scope
misses cannot enter the candidate context.

The 18-prompt, 432-row strict-teacher gate passes at mean/p95/p99/max KL
`0.000140/0.000688/0.001606/0.007267`, 99.769% top-1, and three deterministic
candidate repeats. A fresh final-stack two-order task/economics gate improves
**95.708 -> 97.674 tok/s (+2.05%)**, with both orders and every
category/heldout slice positive and all 160 request trajectories returning the
same final IDs. `general_en_plan` needs one extra rejected proposal cycle per
request (overall acceptance `0.788945 -> 0.785000`), but its throughput remains
positive in both orders (+0.36%/+1.05%) and no task output changes. The clean
post-promotion automatic route measures **98.643 vs 88.250 AR tok/s
(1.1178x)** with 10/10 exact, engaged, and budget-conformed cells.

Automatic serving is narrower than kernel admission: production/BF16
`Q4_K_M`, resident capacity 8, realized C8, K3, and greedy sampling. Every
physical miss remains K0. The production/strict manifest hashes recorded with
this promotion are `2adc137a32d65bc63619947577f5233548d5835a474713abe270d666122a1960`
and `52a3d5b8b02c4dc8230c8c9dc8e43b01135db7ae1b44b027fc8915d66bedcdbb`; since
[2.9](#29-serving-admission-is-physical-not-shape-scoped) they identify the
measured build rather than gate admission.
Evidence: [`C8 automatic promotion`](../benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json).

### 2.8 Qwen3.8 gfx1151 production planar-Q6 integer-MMQ decision

**September 12 correction:** integer-MMQ workspace admission is target-verifier
only. Ordinary scalar and packed AR prefill use the registered BF16 owners.
The public FP32-state long-C8/p512/D128 gate exposed max KL 0.10617 when AR
prefill inherited this verifier association. Disabling only MMQ restores
exact strict logits over all 1,032 long-C8 rows and three repeats; disabling
F16 staging does not repair the failure. F16 staging and the bounded
verifier kernel remain available. See the
[phase-scope diagnosis](../benchmarks/results/2026-09-12-gfx1151-qwen38-ar-mmq-scope-rejected.json).

The 2026-09-03 gfx1151 Qwen3.8 `Q4_K_M` production profile qualifies one
shape- and row-bounded T2 association: BF16→Q8_1 activation packing plus the
sole-resident planar-Q6 integer `mmq64x64` consumer at physical rows17-48 for
K17408/N5120 FFN-down and K5120/N1024 narrow-V. The composite aliases a bounded
resident-session staging allocation and adds no persistent weight bytes.
Strict, profile fallback, peer backends, standard Q6, Q4/Q5, and every row or
shape miss retain registered exact A owners.

The canonical+category-heldout teacher packet covers 216 full-vocabulary rows
with three bit-exact candidate repeats. Mean/p95/p99/max KL is
`8.58e-5/4.91e-4/9.68e-4/0.002231`; overall and every-scope top-1 agreement is
`100%`, no row crosses the `2e-2` review line, and C5 neighbor substitution plus
C8 row permutation are bit-exact. The D24 task gate preserves all 40 AR and 40
MTP ID cells and exact acceptance/accounting.

One-group C5-C8 complete MTP improves `36.519/40.271/43.728/49.979` to
`37.280/41.048/44.492/50.893 tok/s` (+2.08%/+1.93%/+1.75%/+1.83%), with every
category positive. C8 reaches `1.0057x` its same-arm AR rate. This changes an
implementation association inside production; it does not widen automatic MTP
admission. Evidence:
[`B5 retention packet`](../benchmarks/results/2026-09-03-gfx1151-qwen38-b5-planar-q6-integer-mmq-retained.json).

### 2.9 Serving admission is physical, not shape-scoped

The implemented speculative-MTP resolver selects one exact model-plugin
evidence row over physical and ownership identity. This describes the current
selection mechanism, not proof that every unmatched cell is invalid. Under
section 1.1, implementation capabilities and concrete failures determine
runnability; evidence records measured guarantees and promotion decisions.
Evidence-only restrictions must not prevent evaluation of implemented paths.
The resolver currently checks:

| Axis | Why it binds |
| --- | --- |
| artifact SHA-256 and size, `content_verified` | The row certifies one artifact's content. |
| backend, target architecture | Kernels are architecture-scoped. |
| weight quant, KV storage, KV layout | Both change the verified arithmetic. |
| realized group rows, resident capacity | Ownership and physical width are exact contracts. |
| candidate depth (requested <= qualified) | A shallower chain is less speculative work on the same verified path. |
| sampling mode | The verifier's acceptance path is greedy-specific. |
| memory fit | Admission must precede allocation failure. |

Prompt context, output horizon, session length, and the resolved
execution-profile manifest are **not** admission axes. They describe the
envelope a benchmark measured, they change with ordinary serving traffic, and
the manifest changes with any kernel or variant selection. Gating on them
silently disables an already-qualified path: the 2026-08-29 E0 review had to
refresh manifest hashes to stop real requests from selecting K0 with
`execution_profile_manifest_not_qualified`
([`E0 baseline`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json)).
The removed rejection reasons were `execution_profile_not_qualified`,
`execution_profile_manifest_not_qualified`, `max_sequence_length_not_qualified`,
`context_bucket_not_qualified`, and `output_horizon_not_qualified`.

Two rules replace the removed axes:

- When several rows match one physical cell, the cell takes the strongest
  retained authorization: an automatic-eligible row wins over an explicit-only
  row, and remaining ties keep declaration order.
- Profile safety stays with the provider, not the admission table. FP16
  recurrent-state spec-dec2 still requires a complete non-fallback production
  manifest before mutation.

A retained row still records the shape envelope its artifact measured, and that
envelope remains part of the benchmark evidence. What changed is that the
envelope no longer decides admission.

#### Implemented explicit screening override

`HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1` is the existing mechanism for
measuring a physical cell that no retained row qualifies. It does not require
that the experiment pass before it can run:

- It currently applies only to a request that explicitly asks for speculation
  (`speculative_mtp: true`). Automatic intent, and `auto`/`enabled` without an
  explicit request, retain the existing evidence-based selection. These are
  implementation restrictions, not a declaration that unmatched paths are bad.
- It covers qualification gaps: a missing evidence row or an evidence mismatch
  on artifact, backend, target architecture, quant, KV storage/layout, physical
  group, capacity, or candidate depth. The chosen implementation must still
  support the actual inputs and semantics. Content verification, available
  memory and supported sampling remain independently checked; a summary of one
  qualification miss must not conceal a failed structural check.
- Screening eligibility is non-automatic in this implementation, and the
  response reports `qualification: explicit_screening_unqualified_cell`,
  `unqualified: true`, and the original rejection reason in `speculative_mtp`.
  These labels describe evidence status, not a correctness verdict.
- Record diagnostic measurements, failures and successful checks. They can
  contribute to the ordinary numerical, determinism, isolation, task and
  performance evaluation; an incomplete screen must not be presented as a
  completed production gate.

The longer-term contract is capability-based execution with evidence-backed
claims and promotion, not an ever-growing benchmark allowlist. Replacing these
selection restrictions requires implementation and tests; this policy revision
does not assert that the replacement has landed.

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
| Failure containment | Contain a recoverable fault to the smallest affected ownership scope the runner can establish. Preserve requests outside that scope; if safe continuation cannot be established, mark the service unhealthy and stop. See section 4.3. |
| Per-request eligibility | Preserve each request's identity, state and implementation capabilities under grouping. Schedule compatible work under explicit resource/fairness/cost policy; report the actual reason for a fallback, not a neighbor's ineligibility. See section 4.4. |

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

### 4.3 Execution failure containment

Containment is a recovery contract, not an exception handler or a prerequisite
for running a previously untested workload. After an actual failure, a raised
`ValueError` does not prove that no device work or shared-state mutation preceded
it. A blanket catch-and-continue fails this contract.

A runner may retire the smallest affected ownership scope it can establish and
keep serving only when its containment claim establishes both:

- **Quiescence:** all outstanding work that can access the affected resources
  has completed, including prior queued work and cross-stream dependencies.
  Reporting a device error alone does not establish safe resource reuse.
- **Reclaimability:** the named requests can be retired without invalidating
  survivors. Account for device buffers and host-side leases, reference counts,
  scheduler records, provider claims and graph references. Cleanup must finish
  before affected resources are reused.

The affected scope may be one request or a packed/shared ownership group. A
single triggering request does not imply that all preceding mutations were
request-local. Naming fewer requests than the failed work item asserts that all
unnamed rows still hold valid canonical state. The current claim API refuses
IDs outside the failed work item; if the affected scope cannot be contained
within it, the service must stop rather than omit affected owners.

The claim records affected IDs, phase, work kind, cause and mutation status.
Mutation status describes what happened; recovery outcome is a separate
assertion, not part of the definition of `committed`:

| Mutation | Meaning | Claimable |
| --- | --- | --- |
| `none` | The step raised before its first device call; host bookkeeping may still need cleanup. | Yes, once quiescence and reclaimability are established. |
| `partial` | Device work started; no canonical commit was claimed. | With runner-specific evidence of safe rollback, rebuild or retirement. |
| `committed` | A canonical commit happened before the failure. | With runner-specific evidence that committed state and emitted output remain consistent, or that safe restoration/retirement is complete. |
| `unknown` | The runner cannot establish the mutation window. | No. Recovery evidence must first resolve the status; otherwise the failure stays fatal. |

The recovery path must distinguish a recoverable refusal from an unhealthy
runtime. A classified resource refusal before device mutation can be local;
an illegal device access, uncertain shared-state mutation, unavailable runtime
needed for recovery, failed quiescence check, or unprovable cleanup requires a
controlled stop. Exception type alone is neither a recovery proof nor a
universal fatal classification. Unimplemented recovery is a concrete limitation
after a failure, not a reason to reject ordinary execution in advance.

An unsupported-shape refusal from an optional packed route (`NotImplementedError`
from a packed prefill or verifier layout check) is a route decision, not a
runtime fault, and it must never become the terminal outcome of a request that
already published output. A recovery path that runs after a canonical commit
re-routes the affected rows through the registered strict per-session route with
the same tokens and cursors, and records the route change as a fallback reason.
Only rows whose KV layout the strict route cannot represent (a shared or shifted
allocation that requires the block-table-aware entry) may fail closed, and the
failure then follows the containment rules below rather than surfacing as a
request-parameter error.

Fatal is a controlled stop, not a silent one. The service reports `ok`,
`unhealthy`, or `closed`; an unhealthy report names the failing phase, the
deepest frame, and affected IDs when known. Readiness reports the same state
with `ready: false`. Active requests receive a terminal error naming the fatal
cause, and queued or later submissions explain the refusal instead of returning
generic memory advice. Restoring a healthy runtime is required before serving
resumes.

Containment preserves survivors' committed output and authoritative ownership,
positions, KV and recurrent state. At the recovery boundary, compare state
against a control with the same committed history and schedule. If subsequent
group width or scheduling changes, future arithmetic follows the declared
profile: production uses its numerical/task and same-schedule determinism
gates, not unconditional ID equality to a differently scheduled run.
`batch_invariant` retains its stronger composition guarantee.

Validate the recovery mechanism with:

- local prefill and decode failures, including failures before device launch;
- a failure inside a packed group and both narrow and group-scoped retirement;
- cleanup failure, a simulated fatal device state and refused containment;
- survivors' state/ownership checks and profile-appropriate output controls;
- a successful subsequent request after recoverable failure;
- terminal errors carrying phase, affected IDs, work kind, mutation class and
  cause, with no lost or duplicated committed output.

The unit coverage for these items is
[`tests/test_unit_generation_execution_failure_containment.py`](../tests/test_unit_generation_execution_failure_containment.py)
(contained and refused prefill, grouped-prefill, decode and speculative
failures, a committed cycle, cleanup failure, a simulated fatal device error and
the service-level unhealthy report) plus the adapter verdicts in
[`tests/test_unit_qwen35_gguf_mtp2_seam.py`](../tests/test_unit_qwen35_gguf_mtp2_seam.py).

This is fault-class and ownership coverage, not an allowlist of every exception
message or workload permitted to run. Add a reproducer when a new failure is
observed and repair or scope that failure.

**Implemented scope:** the resident GGUF runner's `contain_execution_failure`
claims `none` and `partial` for prefill and decode, and `partial` or
`committed` for a failed speculative cycle, always after synchronizing the
affected runtimes.

- `none` covers a step that never advanced a row's device state: it raised
  before any device call, or its decode phase completed with no packed work
  (every row's first token came from prefill). The claim names the row whose
  failure the runner attributed, and the work item's rows when it could not.
- `partial` covers a prefill or decode step that entered its device phase
  without claiming a canonical commit. The runner marks every row immediately
  before its first device call and every row of a grouped prefill call, so the
  claim names exactly the rows whose device state may have advanced, plus the
  row whose step raised so its failure is reported; rows the step never reached
  stay canonical and are left unnamed. The loop retires the named rows through
  the same request-owned release path a single-row claim uses.
- A speculative cycle asks its resolved adapter for the mutation class and the
  rows it cannot prove canonical. A cycle whose adapter recorded a canonical
  commit before the failure (the eager commit path publishes the row's visible
  tokens and cursor, leaving the outer scheduler behind it) is `committed` and
  cannot fall back to autoregressive decoding; a cycle whose device cursors
  moved without a recorded commit is `partial`. Rows the cycle left canonical
  are unnamed and continue.

It conservatively refuses every `HipError` (the device itself reported a fault,
so shared device state is unproven), a missing or failing runtime, a device
phase that raised before it could establish that no row reached device work, an
adapter that cannot narrow the mutation window (including a physical target
commit that may or may not have landed), and every phase that does not mark its
device window. That is the current implementation, not a permanent rule that
every HIP error is unrecoverable. `unknown` is never a successful containment
claim: a refused failure is reported as a fatal `GenerationExecutionFailed`
whose `state_mutation` is `unknown`, so `health()` and a later refused
submission name the phase, the affected rows and the deepest cause.

### 4.4 Per-request eligibility under group execution

Request identity, authoritative state and implementation capabilities belong to
the request. Physical width and the selected execution route are scheduling
choices subject to those capabilities. A group may legitimately own shared
workspaces, graphs and transactions; it must not erase per-request eligibility
or transfer state between requests.

Resolve each request's supported semantics, provider readiness, storage and
bounds before combining compatible work. Missing benchmark coverage alone is
not ineligibility (section 1.1). Then select a valid schedule under declared
compatibility, resource, fairness and cost policy:

- a neighbor's context miss, provider refusal or verifier mode must not be
  copied into another request's eligibility;
- consider supported partitioning rather than blindly downgrading the group;
  partitioning preserves stable slots and executes each selected row once,
  without duplicated work or starvation through the existing scheduler;
- a scheduler may choose another valid route, including packed AR instead of
  costly serial MTP groups, for a concrete compatibility, resource or economic
  reason. Report that scheduling reason separately from per-request refusal;
- preserve the original request-local reason when it falls back. A generic
  group resource miss must not conceal a different cause;
- a local graph/schedule transition must not permanently revoke eligibility.
  Re-evaluate subsequent work and use an appropriate fast path when available;
- survivors retain canonical state and profile-correct execution when a peer
  falls back or fails.

This is not a guarantee of every request's fastest standalone route, identical
latency regardless of neighbors, or MTP at any cost. It forbids accidental
ineligibility propagation and requires accountable scheduling. Validate it with
mixed contexts, ready/refused providers, cache hits/misses, modes and live
membership, measuring both correctness and actual route/latency behavior.

**Implementation gap:** a group-level resource-claim miss can still downgrade
every row and replace its reason with the group's. Per-request eligibility and
compatible grouping before those checks remain implementation work; this
contract does not claim they are already fixed.

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

**Teacher-forced probe row-count standard (September 10, 2026):** the
top-1 bars are rate measurements and need row counts that can resolve
them: a 1-flip-in-60 probe resolves the 99% bar as "zero flips allowed"
(granularity 1.67%) and cannot distinguish a true 0.5% flip rate from 2%.
Screening probes at ~60 rows (prefill + 4 decode steps x 12 cases) are
valid quick passes only when they show ZERO top-1 flips; any flip, or any
promotion evidence, requires a 500-1000-row probe with shared-chain
teacher forcing (the incumbent arm's token chain forced into every arm,
so a flip cannot cascade into incomparable contexts - the calibration
definition). Reference packet: the 996-row R11 probe
(benchmarks/results/2026-09-10-qwen4exp-moe-decode-warp-envelope-996.json),
where the 60-row reading (98.3%, fail) resolved to a true 0.50% flip
rate (pass, all bars with 2.6-4.4x margin).

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

For public profile selection, missing or uncertified production variants retain
the registered strict fallback. This protects the advertised profile guarantee;
it does not prohibit explicitly selecting an implemented candidate for the
evaluation in section 1.1. Record that selection and its evidence status rather
than presenting it as certified. Routine workload variation within an
implementation's declared domain does not require a new certificate per shape.
Unsupported batch-invariant scenarios either use a fallback that preserves the
requested composition guarantee or reject clearly; they do not silently run
production arithmetic.

The public selectors are `LLM(..., execution_profile=...)`, server
`--execution-profile`, and `HIPENGINE_EXECUTION_PROFILE`. Resolution is a
cold-path plugin registry keyed by model/backend/quant/profile. Every plan names
real `(backend, layer, registry_quant, variant)` keys; resolution verifies the
selected and strict-fallback keys are registered before constructing the model
plugin's profile-specific factory or invoking its binder. Production and
batch-invariant plans may override only a subset of strict scopes; absent scopes
are written into the manifest as strict selections. Captures bind to the
resulting immutable manifest hash.

Profile plans select registered variants; the generic factory API does not
validate GGUF artifacts or manage factory side effects. Model loading owns
storage and consumer compatibility checks. Numerical evidence remains specific
to the model, quantization, workload and backend on which it was obtained; a
matching registry key alone does not extend that evidence to another artifact.

Existing GGUF binders write process-wide environment settings. Use one profile
per model-owning process; switching between models/profiles in one process is
not an isolation guarantee. Initial UD bring-up uses the unnamed existing path
(`execution_profile=None`) in a fresh process without production-profile
overrides. Named production-profile qualification is a later numerical gate,
not a prerequisite for implementing the missing formats. See
[UD-QUANTS.md](campaigns/UD-QUANTS.md) for the bring-up scope.

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

Promotion evidence names the tested shapes, backend and relevant implementation
regimes/transitions; it must not be rewritten as an exhaustive request allowlist.
A concrete failure at one width or context requires a scoped repair or exclusion
that the artifact explains. Untested points are reported as untested, not failed.
A focused diagnostic may reproduce a known failure; normal serving must not
silently select the known-bad path. No benchmark prompt, token ID, or heldout
result may be hardcoded into selection.

## 10. Automatic rejection

The following disqualify a production promotion or claim. At runtime, an
observed failure must trigger the appropriate scoped rejection, repair or safe
fallback. A fallback after state mutation requires section 4.3 recovery first;
selecting strict arithmetic cannot repair corrupted ownership or device state.
Missing evaluation alone does not prohibit running the evaluation.

1. request/slot/token/position/mask/KV/state-ownership mismatch;
2. state contamination from a neighbor or inactive row;
3. invalid commit, rollback, eviction, or sampler accounting;
4. NaN/Inf or nondeterminism under an identical schedule;
5. any binding category/shape/transition threshold failure;
6. task-level material regression;
7. missing strict fallback or unrecorded profile/variant provenance;
8. a claimed performance improvement without the same-suite quality packet;
9. prompt-, token-, or candidate-specific benchmark gaming;
10. terminating requests outside the established affected ownership scope,
    closing the service despite an established safe recovery, or continuing on
    uncertain device/shared-owner state after a failure; or
11. propagating a neighbor's ineligibility into another request, concealing the
    actual request or scheduling reason, or rejecting an implemented compatible
    path solely because its exact workload lacks prior benchmark evidence.

A failed threshold is not fixed by relabeling a bug as numerical relaxation.
Budgets move only through an explicit policy decision backed by calibration
and task evidence.

## 11. Related documents

- [`PRODUCTION-ACCURACY-POLICY-REVIEW-2026-08-31.md`](reference/PRODUCTION-ACCURACY-POLICY-REVIEW-2026-08-31.md) — dated evidence review of the frozen cutoffs, calibration limits, practical impact, excluded performance, and recalibration triggers; it does not change this normative policy.
- [`PRODUCTION-NUMERICS-CAMPAIGN.md`](reference/PRODUCTION-NUMERICS-CAMPAIGN.md) — active
  implementation, calibration, historical-recovery, and c1/cN campaign.
- [`RELAXED.md`](archive/RELAXED.md) — historical relaxed-mode inventory and provenance;
  no longer the normative public-profile policy.
- [`TESTING.md`](TESTING.md) — concrete test tiers and fixtures.
- [`BENCHMARK.md`](BENCHMARK.md) — performance protocols and artifact rules.
- [`CONCURRENCY.md`](archive/CONCURRENCY.md) — serving scenarios and ownership gates.
- [`KERNELS.md`](KERNELS.md) — kernel fallback, lineage, and trace requirements.
