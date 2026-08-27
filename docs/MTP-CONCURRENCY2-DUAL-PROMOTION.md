# CONCURRENCY2 MTP Dual-Model Promotion — W7900

- Status: **active; 35B quality qualified, 35B performance/serving and 27B qualification open**
- Started: **2026-08-27**
- Binding host: **`epyc` / AMD Radeon Pro W7900 / `gfx1100` / GPU 0**
- Normative contracts: [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
  [`TESTING.md`](TESTING.md), [`BENCHMARK.md`](BENCHMARK.md),
  [`CONCURRENCY2.md`](CONCURRENCY2.md), and [`SPECDEC2.md`](SPECDEC2.md)
- Supersedes the execution queue, not the historical evidence, in
  [`MTP-CONCURRENCY2-RECOVERY.md`](MTP-CONCURRENCY2-RECOVERY.md)

## 1. Objective and definition of done

Promote a real Generation-2 / `EngineService` MTP scope independently for both:

1. Qwen3.6-35B-A3B MoE GGUF `UD-Q4_K_M`; and
2. Qwen3.6-27B Dense GGUF `Q4_K_M`.

“Promoted” means an immutable model/hash/quant/KV/profile/shape/depth policy key
selects MTP automatically before mutation, the complete binding correctness and
serving packet passes, and the same-suite complete-wall result beats true no-MTP
AR. It does **not** mean every context, width, sampler, or candidate depth
inherits that result. Every unqualified key remains K0.

The campaign is incomplete until **both** models have at least one promoted
automatic production key and the cross-model fail-closed serving audit passes.
A legacy prelaunch bypass, direct model-owned generation, singleton work
reported as physical C>N, or a verifier-derived `off` denominator cannot close
the objective.

## 2. Frozen model identities and first promotion keys

| Lane | Exact artifact | Registry model | First key |
| --- | --- | --- | --- |
| 35B MoE | `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`; 22,663,387,424 bytes; full SHA-256 `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` | `qwen3_5_moe_gguf` | `hip_gfx1100` / `gguf_q4_k_m` / BF16 KV / `production` / resident C1 + physical C1 / K2 / raw greedy / context 4-95 / natural D24 |
| 27B Dense | `/models/gguf/Qwen3.6-27B-Q4_K_M.gguf`; 17,106,773,120 bytes; full SHA-256 `a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f` | `qwen3_5_gguf` | `hip_gfx1100` / `gguf_q4_k_m` / BF16 KV / `production` / resident C1 + physical C1 / K3 / raw greedy / context 4-95 / natural D24 |

These are first promotion keys, not benchmark-tuned prompt identities. The
context and horizon bounds are mechanical request-shape fields and must cover
the complete committed ten-prompt category/heldout suite. C2/C4, longer
contexts, different horizons, processed sampling, non-BF16 KV, and different
model hashes remain K0 until independent packets pass.

## 3. Correctness contract — generated-ID equality is not the production gate

Both candidates are declared **T2 production arithmetic** because target
verification, width/row scheduling, and fused/reassociated target execution can
change near-tie logits and downstream MoE expert choices without changing the
model representation or speculative acceptance policy.

### 3.1 Exact in every profile

The following bind byte-for-byte or integer-for-integer and cannot be waived as
numerical drift:

- request ID, scheduler slot, physical row, response, and output-queue mapping;
- prompt/current/generated tokens, accepted-token accounting, stops, and usage;
- positions, context lengths, RoPE positions, active/causal/finish masks;
- `KVLiveSpans`, pages, live counts, append/commit/rollback destinations;
- target and provider cursor, Conv/GDN/SSM state ownership, and reseed source;
- graph bucket, profile/manifest/fallback identity, and sampler/RNG ownership;
- cancellation, failure rollback, reclaim, teardown, and zero final owners.

Any mismatch is a state/control bug. It is never repaired by relaxing a
numerical threshold.

### 3.2 Binding production numerical gate

Use strict teacher tokens so strict and production compare identical contexts.
Free-running generated-ID equality is recorded only as a diagnostic.

Every global and category/shape/transition scope must satisfy all of:

| Metric | Requirement |
| --- | ---: |
| Mean full-vocabulary KL, production vs strict | `<= 1e-3` |
| p95 row KL | `<= 5e-3` |
| p99 row KL | `<= 2e-2` |
| Maximum row KL | `<= 5e-2` |
| Overall top-1 agreement | `>= 99%` |
| Per-category/shape/transition top-1 | `>= 97%` |

Rows above KL `2e-2` require explicit top-k overlap, strict margin, finiteness,
and task diagnosis and cannot auto-admit. The broad KL `<=0.05` / top-1
`>=90%` CPU-reference rule is only the outer kernel smoke floor.

### 3.3 Other binding gates

- at least three identical fixed-seed repeats under the same schedule and
  manifest;
- same-width neighbor substitution/permutation isolation and inactive-row
  isolation;
- finite recorded logits, KV, and recurrent/provider state;
- graph/eager reconciliation and registered strict fallback;
- complete `code`, `general_en`, `general_ja`, and `mixed_ja_en` suite plus the
  four fixed category heldouts;
- every applicable task validator passes its predeclared paired
  non-inferiority criterion versus strict; categories cannot compensate;
- strict selected-quant and production selected-quant are both reported versus
  a BF16/full-precision teacher where available, with no unreported added
  quality budget;
- fixed/ragged schedules, delayed arrival, sparse retirement, cancellation,
  refill, pressure/failure recovery, blocking/SSE, and clean drain.

Cross-width composition equality is required only for a separately advertised
`batch_invariant` key. It is diagnostic for these production keys.

## 4. Performance and automatic-policy gate

Performance is measured only after the same-suite quality packet passes.

- Denominator: a separate true no-MTP Generation-2 AR path in the same process,
  model, profile, prompts, sampling, cache state, warmup state, and timing
  protocol.
- Suite: committed `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, all ten
  prompts, six train plus four fixed heldouts.
- Primary timing: complete request / barrier-to-last-completion wall with
  authoritative generated-token accounting.
- Report: full/train/heldout and every-category tok/s, acceptance, accepted per
  output, TTFT, ITL, E2E, SLO goodput, memory, active occupancy, physical
  decomposition, graph/fallback routes, and ownership drain.
- Automatic threshold: aggregate MTP `>=1.10x` true AR in the declared key,
  with no heldout/category speed regression and no candidate-caused SLO or
  task regression. The project target remains `>1.30x`.
- Policy identity includes backend, full model hash, quant, KV, execution
  profile and manifest hash, provider, K, resident/physical C, context bucket,
  horizon, and sampling mode.

No prompt text, token ID, candidate ID, or heldout result may participate in
policy selection.

## 5. Implementation sequence

### M0 — current-state audit and RED seams — audit complete

- [x] Prove 27B resolves the staged adapter and 35B is rejected solely by the
  MoE adapter/profile boundary, not by model inventory or target capability.
- [x] Freeze current same-source AR, direct MTP, and staged MTP route telemetry.
- [ ] Add a RED seam requiring a model-plugin-selected MoE provider while
  retaining dense behavior and K0 for unsupported models.

Current-source W7900 evidence gives the execution order. Dense C1/K3 is already
real Generation-2 (`engine_service_verify_chain`, no legacy fallback) and the
complete ten-prompt D24 packet reaches **32.177 vs 22.492 tok/s (1.4306x true
AR)**, with train/heldout **1.4846x/1.3570x** and every category **1.3217x-
1.5238x**. It therefore moves directly to production-profile quality/serving
qualification rather than another kernel optimization. The current 35B MoE
model has all 22 NextN tensors and reusable device-resident MoE proposal/target
components, but no strict/production profile and `_resolved_mtp2_adapter()`
returns `None`; its unmodified MTP request executes K0 AR with zero cycles.
Evidence: [`current-state audit`](../benchmarks/results/2026-08-27-w7900-dual-concurrency2-mtp-current-state-audit.json).

### M1 — 35B MoE Generation-2 C1 — quality qualified

- [x] Add a model-attached MoE NextN provider adapter around the retained
  device-resident MoE draft runner.
- [x] Let `EngineService` / `ResidentEngineLoop` own admission, one bounded cycle
  per tick, claims, transaction, accept/commit publication, and teardown. The
  adapter never calls whole-request legacy MTP.
- [x] Register strict fallback and a non-default production candidate manifest;
  automatic stays K0 during qualification.

The retained C1/K2 implementation uses adapter-lifetime target-hidden and draft-
KV slabs because both native proposal and target graphs capture those pointers.
Request-owned allocation caused stale graph inputs across prompt changes; the
stable owner closes that control/lifecycle bug without changing arithmetic.
A ten-prompt sequential packet completes with real graph cycles and zero
failures/drain residue. Dirty-worktree feedback (not a performance claim) is
**84.261 vs 67.507 tok/s (1.2482x true AR)**, train/heldout
**1.2541x/1.2396x**, and every category **1.1338x-1.3524x** at 78.03% draft
acceptance. Production generated IDs match AR in 8/10 cells and remain
explicitly diagnostic. Evidence:
[`35B MoE C1 owner`](../benchmarks/results/2026-08-27-w7900-35b-moe-generation2-mtp-c1-owner.json).

The first binding strict-vs-N2 packet rejected the incumbent all-bulk target
route despite passing global KL tails: top-1 was `98.905%` and `general_ja`
failed its binding scope. A prompt-independent shape repair now selects the
native N2 graph for K2 and registered serial-exact target fallback for K1,
including terminal zero acceptance. The repaired full packet passes 276 aligned
rows at mean/p95/p99/max KL
`0.000407/0.002124/0.006020/0.023325`, 100% top-1, every category/shape/
transition scope, exact 252-row K2 graph/eager reconciliation, three repeats,
reverse-order isolation, paired task non-inferiority, profiles/fallback, and
clean drain. Automatic remains K0 pending fresh performance and serving gates.
Evidence: [`35B production quality`](../benchmarks/results/2026-08-27-w7900-35b-moe-mtp2-production-quality.json).

### D1 — 27B Dense C1 closure

- Retain the existing staged provider/frontier/transaction path.
- Add the missing production profile/manifest and complete control/logit/task
  capture rather than requiring strict/free-running ID equality.
- Optimize activation only if current complete-wall economics fail; do not
  reopen already-rejected acceptance heuristics or physical C2 target work.

### Q1/Q2 — independent qualification and promotion

Run the full Section 3/4 packet independently for 35B and 27B. Promote only a
passing exact key. One model cannot borrow another model’s quality, speed,
manifest, context, or policy evidence.

### X1 — cross-model fail-closed serving

Verify discovery/capabilities, explicit/automatic blocking and SSE, unsupported
sampling/context/profile/hash K0 before mutation, cancellation/failure recovery,
and zero final owners for both adapters in one shared architecture.

## 6. Prompt-to-artifact completion checklist

| Objective requirement | Required evidence before completion |
| --- | --- |
| “CONCURRENCY2 MTP” | EngineService live snapshot and response telemetry show `engine_service_verify_chain`, staged `VERIFY_CHAIN` work, real cycles, and no `legacy_prelaunch_fallback` |
| “35B MoE” | Exact 35B hash, MoE provider capability/manifest, RED/GREEN seam, full numerical/task/dynamic/perf artifact, automatic policy artifact |
| “27B Dense” | Exact 27B hash, dense provider capability/manifest, full numerical/task/dynamic/perf artifact, automatic policy artifact |
| “until both are promoted” | Two content-verified automatic keys select MTP on clean public LLM/server requests; all listed out-of-scope probes select K0 pre-mutation |
| “actual correctness gates, not exact match” | Execution-profile artifacts contain strict-teacher mean/p95/p99/max KL, overall/per-scope top-1, determinism, isolation, task and BF16-relative verdicts; generated-ID equality is explicitly `diagnostic` |
| Strict fallback | Variant manifests name registered selected/fallback variants; negative tests prove missing/uncertified production falls back to strict/K0 |
| True speedup | Same-command true-AR and MTP full-suite rows with authoritative token counts, full/train/heldout/category ratios, and server/SLO metrics |
| Production serving | Blocking/SSE outputs and telemetry, mixed arrival, cancellation, overload/soak, failure recovery, lifecycle/memory, and clean drain |
| No benchmark gaming | Committed suite/hash, fixed split identity, pure greedy selection guard, no prompt/token/candidate-conditioned code or policy |
| Publication | Compact artifacts, benchmark README/changelog, architecture/profile docs, immutable worklogs, tests, kernel trace if new kernel executes, atomic commits, pushed clean `origin/main` |

## 7. Completion audit rule

Before declaring this campaign complete, inspect every row in Section 6 against
real files and command output. A green unit suite, a manifest, a route label, or
a speed ratio is only supporting evidence; none is a proxy for the complete
per-model promotion packet. Any absent, stale, cross-host, cross-model, or
weakly covered row keeps the campaign active.
