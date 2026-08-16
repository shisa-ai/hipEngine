# Production Numerics Performance Campaign

Status: **approved; P0-P1 complete, P2 evaluator next**
Approved: 2026-08-16
Primary lane: AMD Radeon Pro W7900 / `gfx1100`, Qwen3.6-35B-A3B PARO,
same-model GGUF heldout/control
Normative contract: [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md)

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
  evidence, with `0.02` as a review boundary and `0.05` as the provisional
  absolute ceiling.
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
| gfx1151 Qwen3.8 rows1 Q4 residual-Q8_1x2 DP4A | `benchmarks/results/2026-08-15-gfx1151-qwen38-27b-q4-q8x2-dp4a.json` | Current retained-development/default route: natural AR `+0.785%` versus its post-control, all 120 AR/B1-B3 trajectories exact, and leaf max KL `3.66e-11`; it still lacks the new strict-teacher tail/BF16/profile-manifest packet. |

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

## 4. Historical candidate queue

### 4.1 Re-certify or re-gate first

| Priority | Candidate | Existing upside | Missing evidence / action |
| --- | --- | ---: | --- |
| P1 | Post-fix A4 production route | Old performance premise was large but invalidated by the state fix | Fresh strict/current/production performance baseline; dynamic teacher-forced logits; exact ownership and transition gates. |
| P1 | Current gfx1151 Qwen3.8 c1 Q4 residual-Q8_1x2 DP4A default | Retained development evidence improves repeated AR `+0.271%..+0.534%` by shape and natural AR `+0.785%`; leaf max KL `3.66e-11` | Re-certify rather than rediscover: add strict-teacher mean/tail rows, BF16-relative evidence where available, and profile-manifest provenance. |
| P1 | gfx1151 GGUF Q8T16 c4/c8 rowtile | About `+1.05%/+2.61%` | Dynamic composition/transition teacher-forced gate and current-route rebase. |
| P1 | W7900 PARO native c4 | About `153.3 tok/s`, roughly `1.145x` c1 in the historical packet | Re-run against current strict and calibrated tails; old max KL `0.0731` is outside the provisional ceiling. |
| P2 | PARO c2 1024-thread attention | About `+2.4-2.6%` | Full category/state/isolation gate and current A/B. |
| P2 | Laguna compact Q4 shared-down | About `+0.388%` E2E, one-ULP leaf drift | Full teacher-forced/category gate; low priority because complete gain is small. |
| P2 | D64 fast verifier | Potential to avoid exact long-horizon fallbacks | D64/D128 strict-teacher logits, state bounds, MTP task/economics packet. |
| P2 | DFlash multirow down projection | Prior single-prompt speed opportunity | Full prompt suite, verifier state/KV/transaction and acceptance/economics gates. |

### 4.2 Diagnose before optimization

- **PARO c8:** localize the max-KL `2.575` rows by prompt, token, layer, width,
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

Exit: CPU unit tests cover metric/metadata/metamorphic failure cases; one strict
GPU smoke reproduces existing evidence without changing kernels.

### P3 — Runtime profiles

Add public `strict`, `production`, and `batch_invariant` selection. Resolve once
to immutable variant manifests over the registry `variant` axis. Keep current
public default unchanged during migration.

Exit: profile-selection tests, strict fallback tests, manifest provenance, and
one no-change strict/production smoke pass.

### P4 — Calibration and current-route certification

Run positive and negative controls, freeze thresholds, then re-certify every
current non-exact default that would enter production.

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

Exit: at least one retained result or a reconciled no-first-order-target
conclusion with a refreshed profile.

### P7 — Focused c>N/A4 campaign and default decision

Optimize fixed widths, then admission/compaction/width transitions. Use SLO
rather than only fixed-batch aggregate throughput for the public decision.

Exit: retain qualified wins; either switch the public default to production or
record the concrete blocker and keep the migration debt open.

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

Profile-aware benchmark artifacts add:

```json
{
  "execution_profile": "production",
  "execution_profile_schema": 1,
  "variant_manifest_sha256": "...",
  "strict_manifest_sha256": "...",
  "arithmetic_class": "T2",
  "teacher_source": "strict",
  "quality": {
    "kl_mean": 0.0,
    "kl_p95": 0.0,
    "kl_p99": 0.0,
    "kl_max": 0.0,
    "top1_overall": 1.0,
    "categories": {},
    "rows_over_review_boundary": []
  },
  "control_semantics": {},
  "determinism": {},
  "task_quality": {},
  "generated_id_equality": {
    "binding": false,
    "diagnostic": {}
  }
}
```

The final schema is implemented and tested in P2; this example freezes intent,
not a Python module layout.

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
