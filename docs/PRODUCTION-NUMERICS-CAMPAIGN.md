# Production Numerics Performance Campaign

Status: **approved; P0-P4 infrastructure complete; first ZBook c1/cN package decision complete with public default blocked on soak; the next ZBook tuning cycle has a frozen PLAN/PUNCHLIST; W7900/specialized recovery and named-profile certification remain open**
Approved: 2026-08-16
Primary lane: AMD Radeon Pro W7900 / `gfx1100`, Qwen3.6-35B-A3B PARO,
same-model GGUF heldout/control
Current auxiliary lane: physical host `zbook`, Radeon 8060S / `gfx1151`.
ZBook rates are independent of the other gfx1151 machine and may not be used as
an old→new denominator for that host; only same-host A/B ratios are binding.
Normative contract: [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md)

Active ZBook execution plan:
[`QWEN36-35B-ZBOOK-PRODUCTION-NUMERICS.md`](QWEN36-35B-ZBOOK-PRODUCTION-NUMERICS.md).
It freezes the host/model/prompt identities, named-profile/control prerequisite,
commands, candidate sequence, quality/SLO gates, and per-candidate punchlist for
new gfx1151 work.

## 1. Objective

Formalize strict, production, and batch-invariant execution; build one reusable
quality/ownership evaluator; re-assess prior changed-arithmetic work under that
gate; then run focused c1 and c>N optimization campaigns without requiring
production arithmetic to be bit-identical to the c1/reference kernel.

The campaign optimizes implementation arithmetic, scheduling, and fusion while
preserving exact request/control ownership. It does not silently change model
weights, KV representation, routing policy, speculative acceptance, or sampling
distribution.

## 2. Success criteria

The campaign closes only when:

1. all three profile contracts are public, logged, and resolved through an
   immutable variant manifest over the existing registry;
2. the evaluator covers fixed and dynamic batching, exact metadata/isolation,
   strict-teacher logits, repeat determinism, task quality, and SLO metrics;
3. production thresholds have been calibrated against known accepted and
   rejected controls;
4. existing non-exact defaults have been re-certified or replaced by strict
   fallbacks;
5. at least one c1 and one c>N production-qualified performance result have
   complete quality and performance evidence, or the campaign records a
   measured no-win conclusion; and
6. the public-default decision is made from dynamic serving evidence rather
   than a single fixed prompt or leaf benchmark.

A profile default switch initially requires at least 3% SLO-goodput benefit with
no more than 1% c1 regression. Smaller exact or production-qualified kernel and
cycle-wall improvements remain retainable under normal project policy.

## 3. Independent evidence audit

The campaign began with an independent repository audit rather than accepting a
prior review at face value.

### 3.1 Corrections to the initial hypothesis

- The old A4 `+63.81%` screen includes a real width-transition state bug. Commit
  `29a0c75d6` repaired authoritative-token seeding before c1 graph capture,
  position publication, and the GGUF paged c1 reduction; follow-up evidence
  passed 560/560 layer comparisons and 360/360 IDs. The old screen cannot be
  retroactively promoted. A fresh post-fix baseline is required.
- A hard maximum KL of `0.02` is not yet supported by repository evidence.
  Existing retained quality-gated routes report maximum KL around `0.0308` to
  `0.0439`. Calibration therefore uses mean and tail distributions plus task
  evidence, with `0.02` as a review boundary and `0.05` as the absolute
  ceiling. P4 calibration subsequently froze those values.
- Q4_K_S/IQ formats/ROCmFP4, INT8/FP8 KV, approximate routing, and changed
  speculative or sampling policies are representation/algorithm changes (T3),
  not same-quant implementation drift. They remain separate campaigns.
- MoE expert choices and generated IDs are downstream numerical decisions.
  Their strict equality is useful diagnostics and may be a candidate-local
  gate, but ownership/scatter correctness—not universal cross-profile equality—
  is the control-plane invariant.

### 3.2 Positive calibration controls

These routes demonstrate that hipEngine already retains bounded reassociation
under semantic gates:

| Control | Retained evidence | Why it is useful |
| --- | --- | --- |
| gfx1100 GGUF peer-wave32 GDN prefill | `benchmarks/results/2026-07-15-gfx1100-gguf-gdn-peer-wave32-semantic-accepted.json` | Reassociated recurrence with complete semantic gate; reported max KL about `0.0417`. |
| gfx1151 Laguna global qrow2 online attention | `benchmarks/results/2026-07-23-gfx1151-laguna-global-qrow2-online-retained.json` | 320 teacher rows, max KL `0.030836`, top-1 `99.0625%`, deterministic, category-positive. |
| gfx1151 Laguna SWA qrow2 online attention | `benchmarks/results/2026-07-23-gfx1151-laguna-swa-qrow2-online-retained.json` | 320 teacher rows, max KL `0.042924`, top-1 `98.75%`, deterministic, category-positive. |
| gfx1151 Laguna compensated F16 WMMA SWA | `benchmarks/results/2026-07-23-gfx1151-laguna-f16-wmma-comp-swa-retained.json` | Large prefill win with compensated changed association and semantic promotion. |
| gfx1151 Qwen3.5-0.8B cumulative semantic packet | `benchmarks/results/2026-08-15-gfx1151-qwen35-08b-cumulative-semantic.json` | 1,800 current transitions, `99.6667%` top-1, max current KL `0.005930`, repeat/state/graph gates. |
| gfx1151 Qwen3.8 rows1 Q4 residual-Q8_1x2 DP4A | `benchmarks/results/2026-08-16-gfx1151-qwen38-dense-pair-requalification.json` | Quality-positive but performance-demoted control: 450 strict-teacher rows reach max KL `0.0008333`, `99.778%` top-1, and exact three-run repeats; the current-route A/B later loses to the exact local32 owner. |

These controls are not automatically certified for the new production profile.
They seed threshold calibration and expose missing evaluator fields.

### 3.3 Negative calibration controls

| Control | Existing failure |
| --- | --- |
| Laguna Q8 DP4A prefill | max KL about `0.1716` |
| Laguna online qrow4 | max KL `0.3946` |
| Laguna F32 hipBLASLt attention | max KL about `0.4447` |
| Laguna source-Q5 SGEMM | max KL about `1.1436` |
| Source attention / source MMQ families | maxima from about `1.8` through `4.16` |
| GDN K2 / wave32 rejected variants | maxima about `0.059` / `0.068` |
| PARO c8 old teacher-forced packet | mean KL `0.00721`, but max KL `2.575` and row means up to `0.0330` |

A calibrated gate must separate broad/outlier failure from accepted sparse
near-tie behavior without averaging away category or transition failures.

### 3.4 Calibrated result

The 2026-08-16 ZBook-local same-backend full-logit run freezes mean/p95/p99/max KL at
`1e-3/5e-3/2e-2/5e-2`, overall/per-scope top-1 at `99%/97%`, and the manual
review boundary at `2e-2`. The task gate is all applicable checks passing their
predeclared paired non-inferiority margins; task types do not share a universal
numeric margin.

The native gfx1151 Qwen3.5 cluster8 positive passes every category, shape, and
transition scope at mean/p95/p99/max KL
`0.000244/0.000926/0.001562/0.004529` and `99.778%` top-1. Fresh gfx1151
Qwen3.6 K2 and wave32-tree controls fail mean, p95, and max; wave32-tree's
`99.111%` top-1 confirms that top-1 cannot stand in for distributions. The old
gfx1100 peer-wave positive does not transfer to the current ZBook gfx1151
control: it reaches max
KL `0.073151` and `98.0%` top-1 and is rejected in that new scope. All controls
are deterministic across three repeats. The compact
[`policy artifact`](../benchmarks/results/2026-08-16-execution-profile-threshold-calibration.json)
records commands, clean source/model/hardware provenance, capture hashes, and
historical context. It qualifies no runtime profile and grandfathers no route.

## 4. Historical candidate queue

### 4.1 Re-certify or re-gate first

| Priority | Candidate | Existing upside | Missing evidence / action |
| --- | --- | ---: | --- |
| P1 blocked (W7900) | Post-fix A4 production route | Old performance premise was large but invalidated by the state fix | Requires the frozen real-server gfx1100/W7900 lane; ZBook values cannot substitute or form its baseline. |
| P1 closed (ZBook-local) | Former gfx1151 Qwen3.8 c1 Q4 residual-Q8_1x2 DP4A default | Full requalification passes 450-row quality at max KL `0.0008333`, `99.778%` top-1, and exact repeats | Seven current same-session p512/d128 pairs measure `0.998071x` the exact local32 route with one win; restore exact local32 as the serial package owner, preserve the separately qualified native B1 owner, and retain split-weight only as a diagnostic. |
| P1 closed (ZBook-local) | gfx1151 GGUF Q8T16 c4/c8 rowtile | Fresh strict-teacher static/dynamic/sparse gate is bit-identical over 1,050 rows with exact repeats; current p512/d128 A/B is `1.00445x` at c4 (7/7 wins) and `1.00666x` at c8 (6/7) | Retain a physical-width floor of 4. The same policy loses at c2 (`0.98205x`, 0/7), so c2 and gfx1100 keep the direct owner. Free-running strict/candidate IDs differ deterministically and remain diagnostic for production; runtime-profile qualification is still blocked on actual control/task evidence. |
| P1 blocked (W7900) | W7900 PARO native c4 | About `153.3 tok/s`, roughly `1.145x` c1 in the historical packet | Re-run on the original/current W7900 lane against strict and calibrated tails; ZBook rates are independent. |
| P2 blocked (W7900) | PARO c2 1024-thread attention | About `+2.4-2.6%` | Full category/state/isolation gate and current same-W7900 A/B. |
| P2 blocked (model) | Laguna compact Q4 shared-down | About `+0.388%` E2E, one-ULP leaf drift | Laguna weights are not present on ZBook; full teacher-forced/category gate remains open. |
| P2 deferred (specialized) | D64 fast verifier | Potential to avoid exact long-horizon fallbacks | Separate D64/D128 strict-teacher, state-bound, MTP task/economics packet; do not fold into the core AR lane. |
| P2 deferred (specialized) | DFlash multirow down projection | Prior single-prompt speed opportunity | Separate full prompt-suite verifier state/KV/transaction and acceptance/economics campaign. |

### 4.2 Diagnose before optimization

- **PARO c8 (blocked on W7900 lane):** localize the max-KL `2.575` rows by prompt, token, layer, width,
  margin, state boundary, and graph/eager route. Do not optimize the old route
  until the outlier is explained or eliminated.
- **Same-quant engine disagreement:** the 2026-08-16 Qwen3.6 quality audit found
  ROCmFPX-HIP versus hipEngine mean/max KL `0.005550/0.136877` and top-1
  `96.667%`. This is not the internal strict definition, but it is a warning to
  retain engine-shape attribution and BF16-relative controls.

### 4.3 Closed unless the numerical mechanism changes

Do not reopen candidates that are performance-negative, superseded by a faster
exact route, or already fail model quality by a wide margin. In particular,
Laguna Q8 DP4A, qrow4 online, F32 hipBLASLt attention, source-Q5 SGEMM, source
attention/MMQ families, and the rejected GDN K2/wave32 variants need a materially
new numerical repair—not merely the new profile label.

### 4.4 Host-qualified recovery checkpoint

The locally available historical-recovery checkpoint is complete, not the
global queue. On physical host `zbook`, Qwen3.8 dense-pair and Q8T16 batch-route
have current full-logit, repeat, and same-host performance dispositions. No
other P1 candidate can be honestly completed there: post-fix A4 and PARO c4/c2
belong to W7900/gfx1100, and Laguna weights are absent. D64/DFlash remain
specialized campaigns with their own task/economics contracts rather than
blockers for starting core ZBook c1/c>N work.

Accordingly, the campaign may proceed on an explicitly independent ZBook lane.
This does not close W7900 historical recovery, unblock a W7900 default decision,
or permit comparisons with the other gfx1151 host.

## 5. Evaluator design

Implement reusable components rather than another model-specific monolith.

### 5.1 Scenario fixture

A scenario fixture declares:

- request IDs, prompts, prompt slices, and expected tokenizer/model hashes;
- admission times and prompt-chunk schedule;
- physical capacity and supported execution widths;
- cancellation, retirement, compaction, and reclaim events;
- context/page/ring/eviction boundaries;
- sampler seeds and per-request RNG identities;
- strict teacher tokens and recorded teacher positions; and
- category/heldout/task labels.

Fixtures must include deterministic generated synthetic scenarios plus real
prompt suites. Synthetic metadata cases catch ownership failures; real prompts
supply meaningful logit and task distributions.

### 5.2 Runner output

Each profile run emits:

- per-step full logits or a losslessly comparable full-logit artifact;
- selected token and optional top-k/margin diagnostics;
- request/slot/row maps, tokens, positions, lengths, and masks;
- `KVLiveSpans` and transaction metadata;
- state/KV ownership hashes and finite/value summaries;
- graph bucket, stream/route identifiers, variant manifest and fallbacks;
- lifecycle/allocation counters; and
- TTFT, ITL, per-request/aggregate throughput, active occupancy, and SLO data.

Raw model-scale captures stay outside Git; compact summaries and hashes are
committed.

### 5.3 Comparator

The comparator performs:

1. exact control-plane and lifecycle validation;
2. same-width neighbor substitution/permutation isolation checks;
3. repeat determinism checks;
4. strict-teacher KL/top-k/margin metrics by category, width, context, and
   transition;
5. optional BF16-relative paired non-inferiority;
6. graph/eager and fallback-path reconciliation;
7. batch-invariant metamorphic comparisons; and
8. task-quality and performance verdicts.

The existing `qwen35_batch_teacher_forced_kl.py`, Laguna category benches,
Qwen3.6 quant-quality tools, Qwen3.8 cumulative semantic harness, and
concurrency/server fixtures are components or precedents. None alone covers the
binding matrix.

## 6. Performance protocol

### 6.1 Baselines

Every candidate compares:

- `strict` — correctness/performance cost reference;
- incumbent `production` — keep/revert performance denominator once one exists;
- candidate `production` — changed variant manifest; and
- `batch_invariant` where supported — reproducibility-tax diagnostic.

The model, quant, KV policy, prompts, sampling, cache state, timing boundary,
and software stack remain identical.

### 6.2 c1 lane

Primary W7900/gfx1100 Qwen3.6-35B-A3B PARO workloads:

- 512/128 and 4K/128 resident true AR;
- natural multi-category AR, not only repeated-token throughput;
- long-context sentinel where the affected state/attention can accumulate;
- same-model GGUF control where the mechanism is portable.

Primary metric is repeated complete-request true-AR wall/tok/s, supplemented by
verified cycle/kernel sub-windows and launch/H2D/D2H counts. Profile current
ownership before selecting a candidate. Initial T1/T2 search families are
reduction/WMMA association, projection/output fusion, attention split/online
merge, and GDN recurrence/fusion; the profile decides priority.

### 6.3 c>N and dynamic lane

Run fixed c1/c2/c4/c8 plus dynamic A4 scenarios. Report:

- aggregate and per-request tok/s;
- TTFT and p50/p95/p99 ITL;
- SLO goodput and request completion rate;
- active occupancy and width-transition counts;
- graph bucket/variant residency and fallbacks;
- memory/allocator peaks and lifecycle; and
- the complete profile quality packet.

Generated-ID equality to independent c1 is binding only for strict or
batch-invariant evidence. It remains a diagnostic for production.

## 7. Phases and exit gates

### P0 — Policy and campaign

Deliver:

- canonical execution-profile contract;
- this independently audited campaign;
- architecture/agent ground-rule updates; and
- historical relaxed-policy migration notice.

Exit: documents validate and commit with an immutable worklog entry. No runtime
default changes and no performance claim.

### P1 — Procedure alignment

Update testing, benchmarking, concurrency, kernel, exploration, refactor, and
documentation-index guidance. Freeze artifact fields and scenario matrix.

Exit: active procedures agree on which properties are exact and which are
profile-specific.

### P2 — Evaluator

Implement RED-first fixture schema, metrics, metadata/isolation comparator,
profile provenance, and adapters for the primary Qwen3.6 PARO lane.

Implemented 2026-08-16: the torch-free core, five JSON schemas, generic gate
CLI, and Qwen3.6 teacher-cache adapter now cover strict-teacher tails,
category/shape/transition scopes, exact controls, dynamic scenarios, repeats,
isolation/composition, BF16-relative paired prompt bootstrap, task verdicts,
and manifest/capture hashes. The adapter intentionally refuses to infer exact
control ownership from a legacy logits cache.

Exit: CPU unit tests cover metric/metadata/metamorphic failure cases; one strict
GPU smoke reproduces existing evidence without changing kernels. The CPU exit
is complete. The GPU smoke remains open until P3 runtime plumbing emits exact
control telemetry and a resolved profile manifest; a logits-only smoke would
not satisfy the approved contract.

### P3 — Runtime profiles

Add public `strict`, `production`, and `batch_invariant` selection. Resolve once
to immutable variant manifests over the registry `variant` axis. Keep current
public default unchanged during migration.

Implemented 2026-08-16: Python API, server CLI, and environment selectors now
resolve through a model/backend/quant/profile plan registry before generator or
resident-runner construction. Resolution verifies exact selected/fallback
kernel keys, fills missing production/batch-invariant scopes from strict,
binds immutable manifest metadata through the shared engine-loop wrapper, and
reports profile/hash provenance from server discovery endpoints. Omission still
preserves legacy package behavior; explicit unsupported profiles fail closed.

Exit: profile-selection tests, strict fallback tests, manifest provenance, and
one no-change strict/production smoke pass. Synthetic CPU plan/factory smokes
are complete. Primary Qwen3.6 PARO/GGUF strict plans, actual control capture,
and the no-change GPU smoke remain open; registering unverified incumbent
variants as `strict` would violate the contract.

### P4 — Calibration and current-route certification

Threshold calibration completed 2026-08-16 on physical host `zbook`. The
retained policy artifact freezes that independent ZBook numerical envelope from
one native gfx1151 positive, two
fresh negatives, a failed cross-backend portability control, and historical
summary context. Every capture uses the package-registered strict GDN baseline
and three exact repeats. It intentionally makes no performance or runtime-
profile qualification claim.

Current-route certification remains open. Re-certify every non-exact default
that would enter production with a resolved profile manifest, exact controls,
isolation/dynamic scenarios, tasks, and BF16-relative evidence where available.
The first historical c>N recovery is complete at the ZBook package-policy
level; its rates are not comparable with the other gfx1151 host: gfx1151 Q8T16
all-projection rowtiling is retained only from physical c4 after a 1,050-row
bit-identical strict-teacher gate and positive current c4/c8 A/B; c2 is
explicitly excluded by a `1.795%` median regression.

The independent ZBook c1 opening result also retains the incumbent
cooperative/persistent F32 router against the strict separate projection/select
chain. Its complete 18-prompt/450-row full-logit gate, three candidate repeats,
and 18 live-state fingerprints are byte-exact. Natural pN/d128 eager AR improves
`30.438 -> 33.219 tok/s` (`+9.136%`, 18/18 wins) with matching free-running IDs;
a fixed p512/d128 five-repeat control independently improves `+8.748%`. These
are package-level current-route results, not public profile certification.
Candidate-versus-strict logits are identical, so this route consumes no
additional BF16-relative quality budget.

The combined ZBook c1/cN decision is now measured. The corrected bundle leaves
Q8T16 rowtiling to package `MIN_ROWS=4` (c2 direct, c4/c8 rowtile) and enables
the cooperative router. Its 18-prompt static/dynamic/sparse packet is exact over
1,050 rows with three repeats. A separate c8 lifecycle run binds token
ownership, active masks, cancel/re-admit, state/KV preservation across
compaction, resource identity, graph invalidation, session reuse, and drain;
c1-versus-cN BF16 KV value differences remain reported as arithmetic rather
than ownership errors and are bound to the distributional gate. Seven graph
pairs are neutral at c2, positive at c4 (`1.00475x`, 6/7), and positive at c8
(`1.01006x` paired median, 6/7).

This closes package-level c>N recovery but not named-profile certification. The
complete server packet passes static, ragged, fixed/Poisson arrival,
cancellation, overload, and idle-recovery workloads, then fails the 60-second
soak with 87 completed and 33 overloaded requests out of 120. A resolved
Qwen3.6 strict/production manifest, standardized complete control capture,
and fresh task/BF16-relative verdicts are also absent. Evidence:
[`bundle decision`](../benchmarks/results/2026-08-16-zbook-qwen36-production-profile-cn-blocked.json).

Exit: calibrated policy artifact and manifest; uncertified routes fall back to
strict. No route is grandfathered.

### P5 — Historical recovery

Evaluate P1 candidates in order, then P2 candidates as hardware/model
availability permits. Commit retained variants separately; publish rejected or
blocked artifacts for informative attempts.

Exit: candidate ledger contains current evidence and no stale pre-fix result is
presented as promotable.

### P6 — Focused c1 campaign

Profile the primary lane, select one mechanism at a time, run strict-teacher and
task gates before final performance retention, and keep exact/production wins
according to project policy.

The auxiliary ZBook lane has its first retained package result. Corrected HIP
stage attribution ranks linear QKV/gate at `6.022 ms/token` and selected-expert
MoE at `5.949 ms/token`; the independently re-gated cooperative/persistent
router improves complete natural eager AR `30.438 -> 33.219 tok/s` (`+9.136%`,
18/18 wins) versus the strict separate chain. The candidate is byte-exact over
450 teacher-forced rows, all repeated live state, and all 18 free-running d128
trajectories. Because it was already the omitted-profile incumbent, this is not
a new runtime default. It satisfies the auxiliary package-level c1 result but
not the public-profile exit: exact control capture and a real Qwen3.6
strict/production manifest remain required.

Exit: at least one retained result or a reconciled no-first-order-target
conclusion with a refreshed profile.

### P7 — Focused c>N/A4 campaign and default decision

Optimize fixed widths, then admission/compaction/width transitions. Use SLO
rather than only fixed-batch aggregate throughput for the public decision.

ZBook package-level decision recorded 2026-08-16: retain the incumbent direct
c2 and rowtile c4/c8 implementation routes after exact numerical, repeat,
lifecycle, and paired-performance gates. Do **not** switch the public profile
default. The canonical server packet fails soak completion (87 completed,
33 rejected of 120), and the c>N gains do not reach the 3% materiality target.
This is a concrete blocker, not a numerical rejection; all completed server
requests are exact and the route/ownership/memory gates pass. Named-profile
manifest, standardized controls, task, and BF16-relative debt remains open.

Exit: retain qualified wins; either switch the public default to production or
record the concrete blocker and keep the migration debt open. The auxiliary
ZBook package-level branch has taken the latter exit; global W7900 and public
named-profile work remain open.

## 8. Candidate workflow

For each candidate:

1. Declare class (T0/T1/T2/T3), scope, hypothesis, expected ceiling, stateful
   surfaces, strict fallback, and binding metrics.
2. Add/extend RED coverage for the changed arithmetic or dynamic scenario.
3. Run leaf/oracle checks and expected-kernel trace.
4. Run strict-teacher metadata/isolation/determinism quality gate.
5. Run task gate.
6. Only then run the final complete performance A/B.
7. Retain only when quality passes and the measured route is non-regressive in
   its declared scope. Restrict dispatch if a shape fails.
8. Write a compact artifact, benchmark rollup/changelog when a performance row
   is retained, and an immutable worklog entry.
9. Remove rejected runtime selectors/workspaces unless they remain useful
   explicit diagnostics with a `REFACTOR.md` removal condition.

No token-, prompt-, or candidate-specific branch may be used to raise a fixed
suite metric.

## 9. Artifact additions

The implemented profile-aware artifact contract is
`benchmarks/schemas/execution-profile-evaluation.schema.json`, with companion
variant-manifest, external-logit-capture, actual-control-capture, and separate
expected-control-fixture schemas in the same directory.
`scripts/execution_profile_gate.py` validates and writes the compact artifact;
full logits remain external and are represented by
capture hashes.

The top level binds `execution_profile`, `execution_profile_schema`, selected
and strict manifest hashes, `arithmetic_class`, and `teacher_source=strict`.
Separate sections record quality summaries/outlier rows, exact control
semantics, determinism, isolation, optional batch invariance, BF16-relative
non-inferiority, task quality, binding-vs-diagnostic generated-ID equality, and
the final automatic-admission decision. This replaces the pre-P2 illustrative
shape rather than preserving an incompatible legacy artifact layout.

## 10. Stop conditions

Stop and repair rather than tune when:

- control metadata, state ownership, KV transactions, or lifecycle fail;
- graph/eager or same-schedule repeats are nondeterministic;
- the first quality failure is unexplained;
- the measured kernel family differs from the hypothesis;
- a candidate is below a reconciled complete-wall ceiling and has no structural
  follow-up;
- a T3 representation change is being smuggled into the T1/T2 campaign; or
- the only observed benefit is on one fixed prompt or one candidate-token set.
