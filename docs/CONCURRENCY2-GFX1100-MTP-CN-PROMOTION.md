# gfx1100 Physical C>N MTP Promotion Campaign

- Status: **Qwen3.6 lanes complete; 27B Dense and 35B MoE physical C2 promoted
  automatically. Qwen3.8 P7-P12 complete; P13 recovery audit complete and cross-engine parity continuation open**
- Started: **2026-08-27**
- Branch: **`campaign/gfx1100-mtp-cn-promotion`**
- Base commit: **`5c2be8d157c587caf42591b07d7c02b3181adabc`**
- Binding host: **`epyc` / AMD Radeon Pro W7900 / `gfx1100` / GPU 0**
- Models: **Qwen3.6-27B Dense `Q4_K_M`** and **Qwen3.6-35B-A3B MoE `UD-Q4_K_M`**;
  extension lane: **Qwen3.8-27B Dense `Q4_K_M`** (§11)
- KV/profile: **BF16 KV; strict fallback plus independently qualified production profiles**
- Normative contracts: [`PLAN.md`](PLAN.md), [`CONCURRENCY2.md`](CONCURRENCY2.md),
  [`SPECDEC2.md`](SPECDEC2.md), [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
  [`TESTING.md`](TESTING.md), and [`BENCHMARK.md`](BENCHMARK.md)
- Starting evidence: [`MTP-CONCURRENCY2-DUAL-PROMOTION.md`](MTP-CONCURRENCY2-DUAL-PROMOTION.md),
  [`MTP-CONCURRENCY2-RECOVERY.md`](MTP-CONCURRENCY2-RECOVERY.md), and
  [`SPECDEC2-GFX1100.md`](SPECDEC2-GFX1100.md); extension evidence:
  `campaign/qwen38-q4-external-bench@95f9ee32c`, artifact
  `benchmarks/results/2026-08-28-w7900-qwen38-q4km-canonical-rebaseline.json`

This is the dedicated gfx1100 physical multi-request campaign. It does not
transfer rates, thresholds, manifests, or promotion decisions from gfx1151. It
may reuse backend-neutral scheduler contracts and tests, but every gfx1100
profile, kernel route, numerical packet, and performance result is independent.

## 1. Objective and definition of done

Promote at least one **real physical C2 automatic MTP key** independently for
both:

1. `/models/gguf/Qwen3.6-27B-Q4_K_M.gguf`; and
2. `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.

Here `C` is the number of independent live requests in one physical speculative
cycle, `K` is candidate depth per request, and `R = sum(1 + k_i)` is target
frontier rows. A candidate budget, verifier row count, resident server capacity,
or two singleton calls is **not** a physical C2 claim.

The campaign is complete only when both models satisfy all of the following:

- the resident Generation-2 owner chooses MTP or K0 from the **realized due
  group** before speculative mutation;
- two requests execute one physical proposal/target/accept/selected-commit
  transaction with row-owned state, KV, output, and accounting;
- C1 -> C2, C2 -> C1, K0 -> MTP, and MTP -> K0 transitions pass delayed arrival,
  retirement, cancellation, refill, failure, and clean-drain gates;
- exact state/control ownership and the binding production numerical/task gates
  pass; generated-ID equality is diagnostic outside strict scope;
- the complete same-suite physical C2 wall beats true no-MTP C2 AR by at least
  `1.10x`, with no category, heldout, task, or SLO regression;
- an immutable content/profile/shape policy key selects C2 MTP automatically;
- every unsupported or losing key selects K0 before mutation with a stable
  reason; and
- the cross-model serving and publication audit passes on a clean branch.

C4 is the next width after a model's C2 promotion. It is not allowed to delay a
passing C2 promotion, and it cannot inherit C2 evidence. The original Qwen3.6
closure above remains complete; the independent Qwen3.8 extension has its own
definition of done in §11.2.

## 2. Frozen model identities and starting keys

| Lane | Exact artifact | Current automatic C1 | Physical C2 starting point |
| --- | --- | --- | --- |
| 27B Dense | `/models/gguf/Qwen3.6-27B-Q4_K_M.gguf`; 17,106,773,120 bytes; SHA-256 `a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f` | production C1/K3/D24, context 4-95; `32.076 vs 22.302 tok/s = 1.4382x` true AR | explicit physical C2/K2/D24 mechanics retained; `22.393 tok/s = 0.7156x` AR at 74.28% draft acceptance; automatic/capability false |
| 35B MoE | `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`; 22,663,387,424 bytes; SHA-256 `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` | production C1/K2/D24, context 4-95; `77.358 vs 67.858 tok/s = 1.1400x` true AR | no independently qualified Generation-2 physical C2 packet; current capacity-2 automatic request selects K0 |

These are controls, not assumptions about the winning C2 key. K1/K2/K3,
context, horizon, resident capacity, and production arithmetic are requalified
per model. No prompt identity or generated token may participate in policy
selection.

## 3. Starting architectural facts

The campaign does **not** begin from zero:

- `ResidentEngineLoop._maybe_run_speculative_cycle()` already sees a due work
  group, resolves a provider capability, plans one speculative transaction, and
  falls back to ordinary decode when the plan has no speculative rows.
- `Qwen35GGUFMTP2Adapter` already contains staged C1/C2/C4 structure, request-
  major proposal, physical target, grouped GPU accept, selected commit, provider
  repair, rollback, and physical telemetry seams.
- Current gfx1100 backend capability intentionally exposes C1 but not C4, and
  public serving rows are keyed to `realized_group_rows=1` and
  `resident_capacity=1`.
- The 27B physical C2 acceptance/state repair is retained. Its blocker is target
  economics, not missing candidate quality.
- The newly promoted 35B MoE adapter/profile is C1-qualified only. It may reuse
  generic transaction ownership but must independently prove MoE provider and
  target physical execution.

Therefore the first implementation task is an admission/transition audit with
RED tests. We will not add a second scheduler or revive whole-request legacy
MTP.

## 4. Frozen transition and admission contract

### 4.1 Decision point

Speculative selection occurs at the resident scheduler boundary after the due
request group and physical width are known, but before provider or target
mutation. Frontend request intent may request `auto` or explicit MTP; it cannot
finalize a physical route before the group exists.

For every cycle the owner chooses exactly one of:

- physical C1 MTP;
- physical C2 MTP (later C4);
- mixed partition into independently admitted groups; or
- K0/AR.

A losing or unqualified physical group falls back as a whole unless a declared
partition plan proves independent row ownership. No silent singleton MTP loop
may be reported as physical C2.

### 4.2 Required transitions

| Transition | Required behavior |
| --- | --- |
| K0 -> MTP | Initialize or catch up provider state from the canonical target row without replaying visible output or moving request identity. |
| MTP -> K0 | Commit/restore canonical target state and park or release provider ownership before AR executes. |
| C1 -> C2 | Admit the arriving row, form one physical group at a transaction boundary, and preserve both stable scheduler/KV identities. |
| C2 -> C1 | Commit both rows, retire/cancel one, and continue the survivor from its own selected target/provider state. |
| Cn -> Cm | Preserve request/slot/row/output maps through reshape; never transfer one row's checkpoint, RNG, state, or KV to another. |
| failure -> K0 | Before-commit failures roll back; after-commit failures rebuild from declared canonical state or fail visibly. No partially committed hidden fallback. |

### 4.3 Policy identity

Automatic evidence is keyed by backend, full model SHA-256, model plugin,
weight quant, KV representation/layout, execution profile and manifest hash,
provider, `K`, resident capacity, realized physical C, context bucket, output
horizon, and sampling mode. A C1 row cannot authorize C2, and a C2 row cannot
authorize a different resident capacity.

## 5. Correctness contract: state/control exact, production arithmetic numerical

The user requirement is explicit: promotion follows the project's actual
correctness gates, **not blanket generated-ID exact match**.

### 5.1 Exact in every profile

These are state/control contracts and bind byte-for-byte or integer-for-integer:

- request ID, stable scheduler slot, physical row, response, stream, and output-
  queue ownership;
- prompt/current/committed token accounting, positions, lengths, masks, stops,
  finish reason, and usage;
- `KVLiveSpans`, page identity, live counts, append/commit/rollback destinations,
  and prefix/COW ownership;
- target/provider cursor, target hidden seed, Conv/GDN/SSM state destinations,
  selected commit, and K0 catch-up source;
- graph bucket, profile/manifest/fallback identity, sampler/RNG ownership, and
  transaction stage;
- cancellation, failure rollback, reclamation, teardown, and zero final owners.

Any mismatch above is a bug, not tolerated numerical drift.

### 5.2 Binding production numerical gate

Production and strict consume identical strict-teacher contexts. Every global
and category/shape/transition scope must satisfy:

| Metric | Requirement |
| --- | ---: |
| Mean full-vocabulary KL, production vs strict | `<= 1e-3` |
| p95 row KL | `<= 5e-3` |
| p99 row KL | `<= 2e-2` |
| Maximum row KL | `<= 5e-2` |
| Overall top-1 agreement | `>= 99%` |
| Per category/shape/transition top-1 | `>= 97%` |

Rows above KL `2e-2` require explicit finiteness, strict margin, top-k overlap,
and task diagnosis. The broad kernel floor (KL `<=0.05`, top-1 `>=90%`) is not
a production promotion gate.

Also required:

- three identical fixed-seed repeats under one schedule and manifest;
- neighbor substitution/permutation and inactive-row isolation at the same
  physical width;
- graph/eager reconciliation and a registered strict fallback;
- all `code`, `general_en`, `general_ja`, and `mixed_ja_en` prompts plus fixed
  category heldouts;
- paired task non-inferiority by category; categories cannot compensate;
- strict and production selected-quant reporting versus BF16/full precision
  where available; and
- blocking/SSE parity, delayed arrival, sparse retirement, cancellation, refill,
  failure recovery, overload/soak, and clean drain.

Generated-ID equality across production versus strict or across widths is
recorded as a diagnostic unless a key separately claims `strict` or
`batch_invariant` semantics.

## 6. Performance and promotion contract

Performance work begins only after the explicit physical owner and correctness
packet pass.

- Denominator: a separate true no-MTP Generation-2 AR route under the same
  process, model, profile, prompts, sampling, cache/warmup state, resident
  capacity, realized width, and timing protocol.
- Suite: all ten committed prompts in
  `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, including all four
  categories and fixed heldouts.
- Primary metric: barrier-to-last-completion complete request wall with
  authoritative generated-token accounting.
- Report: full/train/heldout/category throughput, acceptance and visible tokens
  per cycle, TTFT, ITL, E2E, SLO goodput, occupancy, memory, proposal/target/
  accept/commit/repair costs, actual physical shapes, graph/fallback routes, and
  final ownership.
- Promotion floor: aggregate MTP `>=1.10x` true AR, no category or heldout speed
  regression, and no task/SLO regression. The project target remains `>1.30x`.

The existing 27B C2 result needs roughly a further `1.40x` complete-wall
improvement merely to reach parity with its measured AR and roughly `1.54x` to
reach the `1.10x` floor. The first attribution target remains physical target
verification, especially the residual gate/up owner; acceptance tuning is not
the starting premise.

## 7. Phase punchlist

### P0 — Branch and freeze contract

- [x] Create `campaign/gfx1100-mtp-cn-promotion` from clean synchronized main.
- [x] Freeze the two model identities, binding host, C2 definition of done,
      correctness/performance gates, and merge boundaries in this document.
- [x] Preserve existing C1 automatic keys and all unsupported-key K0 behavior.
- [ ] Record a current-source, same-model C1/C2/K0 audit before implementation.

### P1 — Realized-group admission and transition RED/GREEN

- [x] Trace frontend intent through submission, due-group planning,
      `speculative_capability()`, `plan_speculative_requests()`, claims, and
      execution; identify every point that can mutate provider/target state.
- [x] RED: capacity-2 server with one due row can select an independently
      qualified resident-capacity-2/C1 key rather than inheriting capacity-1.
- [x] RED: two compatible due rows select one C2 plan; an unqualified C2 plan
      selects K0 before provider/target mutation.
- [x] RED: delayed C1 -> C2 and retirement C2 -> C1 preserve request, slot,
      plan, and output identity at the common owner; model state/KV/cancellation
      remain binding P2 gates.
- [x] RED: K0 -> MTP and MTP -> K0 enter through complete cycle boundaries and
      invoke provider K0 preparation before AR; model canonical-state parity
      remains a binding P2 gate.
- [ ] RED: model-specific telemetry rejects per-request target backbone loops,
      legacy prelaunch, mislabeled candidate depth, and mismatched realized width.
- [x] GREEN with model evidence supplied by plugins, never shared
      `if backend == ...` or `if quant == ...` branches.

The audit found one shared blocker: request-time C1 plans were copied into the
queue, and a later C2 queue group compared its width to that stale plan and
selected K0 without asking the model plugin for an independently qualified C2
row. The server now re-resolves the immutable model plan after physical grouping
and before backend mutation. The serving resolver also selects the exact row
among multiple same-artifact C1/C2/resident-capacity entries instead of making
the first tuple entry shadow every later width. Focused tests prove one C2 model
call, exact realized decision telemetry, unqualified-C2 K0 preparation, and
C1 -> C2 -> C1 plus MTP -> K0 -> MTP cycle boundaries.

This does not enable a gfx1100 physical adapter or add evidence. The backend
capability remains false and current automatic C2 remains K0. P2 must carry the
model-specific plan through controlled streaming, prove physical target
telemetry/state, and add only an explicit qualified 27B C2 row.

Exit: the common owner can switch safely even though automatic C2 policy remains
K0.

### P2 — 27B Dense explicit physical C2 ownership

- [x] Enable a non-default gfx1100 C2 capability/profile candidate only for the
      exact 27B identity; it is strict and explicit-only, never automatic.
- [x] Execute request-major proposal, one physical target frontier, one grouped
      device accept, independently selected state/KV commits, and provider
      repair with zero candidate D2H before target.
- [x] Prove reject/partial/full accept, asymmetric acceptance, wrong-branch
      neighbor isolation, and following-cycle continuity through the physical
      adapter oracle/tests plus full-suite/public SSE traces.
- [x] Prove delayed arrival/refill, survivor, cancellation, prefix fail-closed,
      claim failure, injected pre/post-commit failure, and clean drain through
      the common/adapter suites and real two-stream cancellation packet.
- [x] Run strict fallback plus the production numerical/task/determinism/
      isolation packet. Do not require blanket production generated-ID equality.
- [x] Publish an explicit functional artifact even if it is slow.

The strict checkpoint passes all ten direct-owner and blocking-server C2 cells,
accepts **257/346 (74.28%)** drafts, and records physical width-2 proposal plus
normally R6 target/accept/commit with zero candidate D2H/recovery. Real
concurrent SSE executes exact K2 for both requests, drains provider ownership,
and an asymmetric cancel leaves the survivor exact through 19 K0 catch-ups.
Production passes **281** actual packed strict-teacher rows at mean/p95/p99/max
KL **0.000159/0.000786/0.002098/0.003763**, **99.644%** top-1, every scope,
three physical full-logit repeats, exact mapped row-permutation logits, paired
tasks, registered fallback, and clean lifecycle. Free-running reverse-pair IDs
remain explicitly diagnostic rather than a production gate. It remains
**22.390 vs 31.281 tok/s (0.7158x AR)**. The real unflagged strict/production
public routes independently pass 10/10 at **22.036/22.002 vs 30.652/30.623 tok/s
(0.7189x/0.7185x)**. Realized C1 inside resident capacity 2 stays K0 because
singleton continuation in the physical provider owner is not qualified. Evidence: [`explicit C2 ownership`](../benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-c2-explicit-ownership.json).

Exit: explicit C2 is functionally qualified; automatic still selects K0.

### P3 — 27B Dense economics and promotion

- [x] Refresh current true physical C2 AR and exact C2/K1-K3 attribution.
- [x] Profile operation-complete target/proposal/accept/commit/repair windows with
      cached builds; retain every exact or profile-qualified sub-window win.
- [x] Start from the retained R6 rows6 hybrid. Do not retry the rejected
      unrestricted strict-failing rowtile or slower dual-WMMA+SiLU route without
      a materially new premise.
- [x] Screen multi-prompt economics, then run the complete suite only when the
      route projects to the promotion floor.
- [x] Run full serving/SLO/correctness packet and register the exact automatic
      C2 key if all gates pass.
- [x] Evaluate C4 only after C2 promotion; keep every losing cell K0.

P3 exit passes independently. Output-normalized physical prompt streaming plus
production-scoped Q4/Q5/Q6 rows6 target owners move clean merged-source C2/K2
three-run economics to **34.872 vs 31.040 tok/s (1.1235x AR)**; all categories
are non-regressive. The actual unflagged public route reaches **34.488 vs 30.774
tok/s (1.1207x)** with 10/10 engaged/budget-conformed cells. The complete
three-repeat gate covers **281** strict-teacher rows at mean/p95/p99/max KL
**0.000182/0.001043/0.003095/0.003763** and **99.644%** top-1, plus mapped
permutation isolation, tasks, automatic SSE dynamic admission, cancellation,
wrong-key K0, and final drain. Strict remains explicit-only. Evidence:
[`automatic C2 promotion`](../benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-c2-automatic-promotion.json).

### P4 — 35B MoE explicit physical C2 ownership — complete

- [x] Extend the model-plugin-selected MoE adapter/profile to a physical C2
      candidate without forking the scheduler.
- [x] Prove one physical MoE proposal/target group rather than two C1 loops,
      including router/expert, target hidden, recurrent state, KV, and selected
      commit ownership per request.
- [x] Run the complete transition/lifecycle packet from P1/P2.
- [x] Run strict fallback plus production numerical/task/determinism/isolation
      gates; production generated-ID equality remains diagnostic.
- [x] Publish an explicit functional artifact even if it is slow.

### P5 — 35B MoE economics and promotion — complete

- [x] Establish true same-protocol C2 AR and K1/K2/K3 controls.
- [x] Profile target/router/expert/proposal/commit operation-complete wall and
      optimize gfx1100-specific owners behind registry variants.
- [x] Pass complete category+heldout correctness, serving, and SLO economics.
- [x] Register the exact automatic C2 key only at `>=1.10x` AR with every binding
      gate; otherwise keep K0 and continue attribution.
- [x] Evaluate C4 only after C2 promotion; keep it K0 pending independent evidence.

P4/P5 exit passes independently. The final production target combines exact R6
RMSNorm/Q8/alpha-beta, two R3 chain-Conv segments, no-copy segmented GDN state
journals with FP32/BF16 outputs, exact FP32 `ssm_out`, row-batched MoE, and bulk
NextN prompt K/V priming. The full gate covers **281** strict-teacher rows at
mean/p95/p99/max KL **0.0000266/0/0.000229/0.003588** with **100%** top-1.
Three-run economics are **98.505 vs 86.650 tok/s = 1.1368x AR**; all categories
are non-regressive and heldout is **1.1132x**. Public automatic is **92.419 vs
82.347 tok/s = 1.1223x**, 10/10 engaged/self-repeat exact, and blocking/SSE,
cancel/survivor, typed K0 controls, and zero final ownership pass. Evidence:
[`MoE C2 automatic promotion`](../benchmarks/results/2026-08-28-w7900-35b-moe-mtp2-c2-automatic-promotion.json).

### P6 — Cross-model closure — complete

- [x] Independently start real servers for both exact models and prove automatic
      C1, automatic C2, explicit modes, and unsupported-key K0.
- [x] Run delayed arrival, C1/C2 switching, retirement/refill, cancellation,
      failure recovery, overload, alternating soak, blocking/SSE, and zero final
      owners for each model.
- [x] Audit shared engine/dispatch for backend/model/quant branches and legacy or
      singleton masquerading routes.
- [x] Publish compact artifacts, benchmark README/changelog updates, immutable
      worklogs, campaign completion audit, and branch merge handoff.

P6 merged `origin/main@c08cf1cce` before closure. On the resulting source,
27B Dense public automatic is **34.341 vs 30.736 tok/s = 1.1173x AR** and
35B MoE is **93.644 vs 80.973 tok/s = 1.1565x AR**; both engage 10/10 cells,
all categories are non-regressive, and each drains to zero allocations. Real
blocking/SSE uses physical resident C2 for both clients and drains all provider
and prompt owners. Shared generation/speculative dispatch contains no
backend/model/quant branch and no singleton masquerade. The milestone run passed
**10,440** tests outside 15 isolated frozen-hash/README/order failures; the
complete seven-file failure bundle passes after current-source fixture refresh.
Evidence: [`dual-model final audit`](../benchmarks/results/2026-08-28-w7900-dual-model-physical-c2-campaign-final.json).

## 8. Merge and coordination boundaries

This branch intentionally isolates gfx1100 work, but the working tree remains
shared. Before editing any high-conflict shared file, rebase/merge the latest
main and inspect concurrent changes.

Likely shared surfaces requiring small, contract-level commits:

- `hipengine/generation/engine_loop.py`;
- `hipengine/speculative/policy.py` and `hipengine/speculative/serving.py`;
- `hipengine/server/api.py`; and
- common scheduler/serving tests.

Prefer gfx1100/model-specific work in:

- `hipengine/generation/qwen35_gguf_mtp2.py`;
- `hipengine/generation/qwen35_gguf.py` and model plugin registration;
- gfx1100 execution-profile manifests and registered kernel variants;
- focused scripts/tests/artifacts for this campaign.

Rules:

1. Land shared architecture seams as small standalone commits before model/kernel
   tuning so gfx1151 can cherry-pick or merge them independently.
2. Keep backend/model capability decisions in registries/plugins; never branch on
   backend or quant in the common engine.
3. Do not edit the gfx1151 campaign document or reuse its policy evidence.
4. Re-run common contract tests after syncing main; resolve semantic conflicts,
   not just textual conflicts.

## 9. Required evidence per model

| Requirement | Evidence |
| --- | --- |
| Real physical C2 | one `VERIFY_CHAIN` operation with two request IDs, request-major candidate rows, one physical target batch, grouped accept/commit telemetry, and no singleton/legacy fallback |
| Dynamic switching | C1 -> C2 -> C1 plus K0 <-> MTP traces at transaction boundaries with exact identity/state/KV ownership |
| Actual correctness | strict-teacher KL mean/p95/p99/max, overall/per-scope top-1, tasks, repeatability, isolation, finiteness, strict fallback; generated IDs labeled diagnostic outside strict |
| Lifecycle | delayed arrival, refill, retirement, cancellation, prefix policy, pressure/failure, overload/soak, blocking/SSE, bounded memory, zero owners |
| True speedup | same-protocol true AR and MTP complete wall, full/train/heldout/category/SLO rows and authoritative token counts |
| Automatic promotion | exact model/profile/shape key selects C2 MTP before mutation; all negative keys return stable K0 reasons |
| No gaming | committed prompt suite/hash; no prompt text, token ID, candidate ID, or heldout result in policy or dispatch |
| Publication | compact result artifacts, benchmark rollups, immutable worklogs, focused/full validation evidence, atomic commits, clean merge handoff |

## 10. Stop and retain rules

- Stop performance work on any request/state/KV/rollback/ownership failure.
- Add a strict-teacher oracle before retaining arithmetic changes.
- Never relax production gates to generated-ID equality or relax exact
  state/control ownership into a numerical tolerance.
- Never label singleton loops, candidate depth, or verifier rows as physical C.
- Keep every measured, correctness-passing gfx1100 sub-window win and promote it
  to the scoped default unless a concrete blocker is recorded.
- Reject or keep explicit every cell below true AR after attribution; automatic
  policy stays K0 until the full `>=1.10x` gate passes.
- Promote every exact model identity independently. One model cannot borrow
  another's profile, correctness, lifecycle, rate, or policy evidence.

## 11. Campaign extension: Qwen3.8-27B effective C=N (P7-P12)

Opened **2026-08-28** after the canonical Qwen3.8 `Q4_K_M` rebaseline. The
Qwen3.6 lanes above stay closed; this section is a new independent lane on the
same binding host and contracts. Nothing transfers automatically: Qwen3.8 gets
its own attribution, kernel-shape qualification, numerical packet, economics,
and policy keys.

### 11.1 Extension premise and evidence limits

Frozen lane identity: `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` on `epyc` /
W7900 / `gfx1100` GPU 0; 17,106,773,984 bytes; SHA-256
`7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`. The
measured host also has an AMD Ryzen 9 5950X and ROCm 7.2.4. The starting
protocol is BF16 KV, raw greedy, D24, K3, no prompt cache, the ten-prompt suite
(including four heldouts), and a fresh process per hipEngine run or width. All
rows below come from `campaign/qwen38-q4-external-bench@95f9ee32c`, artifact
`benchmarks/results/2026-08-28-w7900-qwen38-q4km-canonical-rebaseline.json`,
measured on `origin/main@d199f2778` plus docs-only commits.

| C | Effective route | True AR tok/s | Requested arm tok/s | Ratio | Draft acceptance | Evidence class |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | explicit strict MTP K3 | 21.958 | 32.153 tok/s | **1.4643x** | 78.89% | retained three-run aggregate; 30/30 strict AR-ID-equal diagnostics |
| 2 | forced physical K3 | 30.712 | 15.777 tok/s | **0.5137x** | 54.80% | one explicit diagnostic run; 10/10 strict AR-ID-equal |
| 3 | forced physical K3 | 35.732 | 16.425 tok/s | **0.4597x** | 44.28% | one explicit diagnostic run; 10/10 strict AR-ID-equal |
| 4 | forced physical K3 | 39.303 | 18.826 tok/s | **0.4790x** | 46.59% | one explicit diagnostic run; 10/10 strict AR-ID-equal |

A separate p128/d24 direct-AR control is **29.871 tok/s median** with 0.101%
stdev, finite logits, and stable IDs. Absent new attribution, the extension
should not treat ordinary AR kernels as the C2 blocker.

These exact-ID rows do not substitute for the §5 state/control, numerical,
lifecycle, or serving packet. In particular, aggregate acceptance falls 24.10
points from C1 to C2 and further at C3/C4. The campaign must first determine
whether that difference comes from workload composition, provider/target state,
batch arithmetic, candidate policy, or another route change; it must not assume
C1's 78.89% acceptance at wider widths.

The compact artifact does not retain an authoritative physical group-cycle
count. Therefore do **not** derive C2-C4 cycle wall by multiplying their
throughput by C1's ~2.9 visible tokens/cycle. P9 must record actual physical
cycle counts, actual visible tokens per cycle, and operation-complete marker
walls before assigning a cost share or break-even budget.

The old C1-C8 harness always ran an `opt_in` server and sent
`speculative_mtp=true` on its requested arm. Those rows are **explicit opt-in**,
not automatic serving evidence. The old C5-C8 slowdown therefore measured an
explicit request partitioned by the speculative route cap, not an automatic K0
product regression. Commit `230232754` adds a separate automatic mode (auto
server, request field omitted). On current merged source, actual automatic C2
and C5 are pure K0 at **1.0040x / 1.0013x** their explicitly disabled AR arms,
20/20 AR-ID-equal, with stable `automatic_mtp_scope_not_promoted` reasons.

P7 resolves the opening mechanism hypotheses:

1. **Production transfer candidate engages, but is not qualified.** The merged
   production diagnostic activates prompt streaming and inherited rows6
   Q4/Q5/Q6 capabilities, improving C2/K3 from **0.5137x to 0.5715x AR
   (+11.26% relative)**. Acceptance is unchanged at **257/467 = 55.03%**. The
   gfx1100 profile manifest still names only a generic C1 GDN selection and
   cites Qwen3.6 evidence; the physical capabilities need independent Qwen3.8
   profile/numerical evidence before promotion.
2. **Physical proposal batching already exists.** The shared executor issues one
   active-row batch forward per draft depth; current C2 telemetry records
   proposal rows=2, with rows=1 only for ragged tails. Do not implement another
   generic C*K-to-K batching layer.
3. **Acceptance/state is the binding pre-profile blocker.** C2 remains 23.86
   acceptance points below explicit C1 and records three recoverable
   `physical accept identities do not align` precommit fallbacks. P8 must explain
   and repair this before performance attribution.
4. **Other cycle owners remain unranked.** Proposal-head/projection, target,
   accept/commit, host synchronization, graph selection, copies, and provider
   repair wait for the correctness-qualified P9 profile.

The independently qualified gfx1151 Qwen3.8 C2/K3 route is a design reference
only; its rates and evidence do not transfer. See
[`2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json).

### 11.2 Extension definition of done

The primary objective is at least one **real physical C2 automatic MTP key** for
the frozen Qwen3.8 identity that passes every §1 gate: exact state/control
ownership, the production numerical/task/determinism/isolation packet, complete
same-suite economics at `>=1.10x` true AR with no category/heldout/task/SLO
regression, automatic selection before mutation, and ordinary-AR K0 behavior
with stable reasons everywhere else.

A truthful logical-C=N policy may separately partition a due group into
independent C1 MTP transactions if it passes its own complete economics,
fairness, lifecycle, and SLO gates. It must report `partitioned_c1`, cannot be
called physical C2, and does not satisfy the primary objective. C3/C4 are
considered only after C2 promotion. If measured attribution shows a hard
structural blocker, publish a durable worklog/artifact with the profile evidence
and stop rather than weakening the gate.

### 11.3 Phase order

#### P7 — Source reconciliation, lane freeze, and route audit

- [x] Fetch, then merge the latest `origin/main` into this branch, resolve
      semantic conflicts, and record the exact clean merge commit before
      benchmarking. Confirm that both the Qwen3.8 support and this branch's
      Qwen3.6 economics owners remain present.
- [x] Freeze model/suite hashes, ROCm/compiler versions, GPU/queue/power state,
      BF16 KV, sampling, prompt-cache state, warmup/process policy, provider
      composition (ngram disabled, zero lookup calls), strict and candidate
      production manifests, and exact commands.
- [x] Audit K0 plus strict/production C1/C2 route resolution. Record manifest
      selections, backend capability owners, strict fallbacks, and missing
      Qwen3.8 evidence; actual kernel names/durations remain the cached P9 trace.
- [x] Run focused true-AR, retained explicit C1, explicit physical C2, and
      automatic typed-K0 smokes. Defer full C3/C4 reruns until after C2.
- [x] Separate explicit route-cap decomposition from actual automatic serving;
      prove automatic C2/C5 use ordinary AR without provider mutation or hidden
      throughput loss.

P7 exits on clean merge `a4c7da9fa` plus harness `230232754`. Explicit strict
C1/K3/D24 remains exact and engaged at **32.802 vs 21.823 tok/s = 1.5031x AR**.
Explicit production C2/K3/D24 was physical and exact-ID diagnostic but only
**17.527 vs 30.668 tok/s = 0.5715x AR**, with 55.03% acceptance and three
recoverable accept-identity failures. P8 supersedes those failures below.
Automatic C2/C5 are pure K0 at parity and all allocations drain. Evidence:
[`P7 route audit`](../benchmarks/results/2026-08-28-w7900-qwen38-q4km-cn-p7-route-audit.json).

#### P8 — Explicit C2 ownership, acceptance, and correctness baseline

- [x] Prove one physical C2 proposal/target/accept/selected-commit transaction:
      two request IDs, request-major candidate rows, one R=2(1+K) target
      frontier, grouped accept/commit telemetry, no request-serial target or
      mislabeled singleton route, and zero candidate D2H before target.
- [ ] Differentially compare C2 proposals, draft logits/state, target rows, and
      accept decisions against independent C1 at identical teacher contexts and
      controlled prompt composition. Report acceptance by K, proposal depth,
      category, heldout, transition, and request—not only one aggregate rate.
- [ ] Stop and repair any request/state/KV/RNG/cursor/hidden/position mismatch
      before profiling. Generated-ID equality alone is insufficient.
- [ ] Qualify strict C2 K1/K2/K3 controls and the candidate production arithmetic
      through §5 numerical, repeat, neighbor/permutation, task, lifecycle,
      failure, graph/eager, strict-fallback, memory, and clean-drain gates.
- [ ] Prove C1 -> C2 -> C1 and K0 <-> MTP with delayed arrival, asymmetric
      retirement/cancellation, refill, prefix policy, and ordinary-AR negative
      keys before retaining any new performance result.

P8 correctness checkpoint (2026-08-28): mixed positive-K/K0 due groups now fail
closed before mutation, and private resident-slot K/V ownership is exact for
both singleton ragged tails and packed batch import. The oracle localizes the
old defect to nonzero private slots: finite inputs/K/V produced 12 `INT_MAX`
proposal rows after slot-0 K/V was imported. The retained zero-copy local view
plus slot-offset copy map moves **12 -> 0 sentinel rows**, acceptance
**269/492 = 54.67% -> 279/470 = 59.36% (+4.69 points)**, and oracle ratio
**0.5565x -> 0.5617x AR**. Default no-oracle is **17.097 vs 30.471 tok/s =
0.5611x**, 10/10 exact-ID diagnostic, zero candidate D2H/failures, and full
drain. GPU accept equals CPU on every oracle cycle. The rejected checkpoint/full
replay candidate left 12 sentinels and regressed to 0.5479x, so it was removed.
C2 remains 19.53 acceptance points below C1 and lacks its independent full §5
profile/lifecycle packet; P8 stays open and automatic stays K0. Evidence:
[`provider KV ownership repair`](../benchmarks/results/2026-08-28-w7900-qwen38-q4km-c2-provider-kv-ownership-repaired.json).

The repaired one-run fixed-depth screen selects K2: K1/K2/K3 are
**0.5180x / 1.0450x / 0.5611x AR** at **91.67% / 80.29% / 59.36%** draft
acceptance. C1/K2 is 90.68%, leaving a 10.39-point physical acceptance gap.
K2 train/heldout are 1.0520x/1.0347x and general EN/general JA/mixed are
1.1575x/1.1708x/1.2464x, but code is only **0.8796x**. Thus promotion needs at
least **12.04% complete-wall reduction** to make code non-regressive; the
aggregate 1.10x floor alone needs 5.00%. These are screen rows, not profile or
performance claims. Evidence:
[`repaired K1-K3 screen`](../benchmarks/results/2026-08-28-w7900-qwen38-q4km-repaired-c2-k1-k3-screen.json).

Stage/tile/acceptance diagnosis of the same repaired tree localizes both
blockers. `_read_target_batch_accept` opens with a full
`runtime.device_synchronize()`, so the accept stage absorbs all pending GPU
work: per request-cycle it costs K1 165.6 ms, K2 44.4 ms, K3 178.8 ms (62.1% /
26.0% / 52.9% of wall) while the target forward itself is only 23-28 ms.
gfx1100 admits only `GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_ROWS = {6}`; K1's R4,
K3's R8, and K2's ragged 2-5-row cycles fall back to the shared-B 256-row
padded tile at ~5.1x cost (within-run fit: 30.7 + 126.7 x ragged-cycle
fraction ms). That is the whole K1 paradox: 91.67% acceptance on an off-tile
R4 verify. The C2 code acceptance drop is concurrency-caused, not
profile-caused: C1 strict and production acceptance are identical
(0.9365/0.8485/0.8788/0.9375), while C2/K2 code is 0.7292. Conditioning on
the previous cycle shows code position-1 acceptance collapsing only after a
rejected cycle: K2 0.4706 (n=17), K3 0.3043 (n=46), while K1 recovers at 1.000
via k0 catchups (23 observed) and C1/K2 full accepts hold 0.9615. The reject
path restores the provider root snapshot
(`restore_request_root_state`: conv/recurrent states + cursors) and re-seeds
from verify hidden rows, unlike the healthy target-hidden k0-catchup repair.
A follow-up device position probe (20260829 worklog, oracle traces + wrapped
capture/restore/advance) verified 175/175 captures with
`consumed_position == root_position` and restores rewinding cursors exactly,
refuting a positional one-token hole; stale draft KV rows are overwritten at
the same positions by the next root advance. Post-reject proposals are
plausible-but-wrong (oracle traces: code 7 correct / 9 unrelated / 1
branch-continuation; root continuity 21/21). Two hypotheses remain:
(a) immediate re-speculation at a by-definition-hard correction token with
code-clustered difficulty (K1 escapes because its k0 catchups advance one more
token before re-speculating), or (b) a value-level provider-state corruption.
Discriminating experiment: force a K0 catchup after every reject at C2/K2 —
recovery to ~0.90 implies policy, persistence implies corruption.
D24 walls are ~50% prefill, so these ratios understate steady-state decode
gains. Evidence:
[`stage/tile/acceptance diagnosis`](../benchmarks/results/2026-08-28-w7900-qwen38-q4km-c2-stage-tile-acceptance-diagnosis.json).

P8 current punchlist (20260829):

- [x] `pad_candidate_graph_rows` primitive landed (inactive tail rows owned
      by the last request; padded `accept_from_top1` bit-identical to
      unpadded; GPU accept already honors `active_mask`).
- [x] Executor integration landed (commits `a3722550c`, `b5a974a5c`):
      `GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS = (6,)` on hip_gfx1100,
      adapter pad scratch, padded row-slot accept validation, pad-token tail
      upload; strict profiles and other backends stay unpadded.
      **Result: explicit C2/K2 rises 1.0450x -> 1.1902x true AR with every
      category >= 1.1329x** (code 0.8796x -> 1.1329x), acceptance and
      committed tokens bit-identical, 10/10 exact/engaged/budget-conformed,
      zero failures, full drain. K1 rises 0.5180x -> 1.0866x (code 1.0944x).
      Verify rows ride the rows6 tile every cycle ({6: 228} / {6: 175});
      accept stage falls to ~30 ms/request-cycle. Evidence:
      [`rows6 group padding`](../benchmarks/results/2026-08-29-w7900-qwen38-q4km-c2-rows6-group-padding.json).
- [x] Forced K0-catchup-after-reject experiment run (commits `d01365bc8`,
      `6a835dd72`; worklog 20260829T052637). Verdict: the catchup repairs
      code post-reject acceptance (0.471 -> 1.000) and lifts code to 82.54% /
      1.1820x, but degrades en/other post-reject cycles (en 0.167) so the
      one-shot policy nets 1.1480x < 1.1902x and stays default-off. The
      catchup's positional semantics (`root_position = target.position` at
      prepare time) need audit before any policy retry.
- [x] Catchup audit complete (worklog 20260829T053618/074350): catchup
      positions/tokens/hidden provenance verified correct; the en degradation
      was the packed AR decode never refreshing
      `_last_target_hidden_ptr`, seeding post-K0 proposals two positions
      behind. Fixed in `6b580ad25`; cooldown rerun recovers en post-reject
      0.167 -> 0.667 and 1.1480x -> 1.1700x; default no-cooldown path is
      output-identical (acceptance equal to four decimals) and its retained
      economics stand. Cooldown policy stays off (1.1700x < 1.1890x at D24).
- [x] Complete the P8 production numerics/determinism/lifecycle packet for
      explicit C2/K2; automatic stays K0 pending P9-P12. Cross-commit
      determinism is byte-identical (`76d94b2ab` -> `6b580ad25`), cross-width
      differential is 20/20 C2 == C1 in production and strict, and clean
      current-source strict C2 K1/K2/K3 controls are 10/10 exact/engaged at
      0.4813x/0.6895x/0.5234x AR with acceptance
      0.9167/0.8029/0.6139.
- [x] Strict-teacher R6 verifier numerics gate passed and retained
      (KL mean 5.2e-05 / max 5.8e-04 vs 1e-03/0.05; top-1 240/240 across
      category/shape/transition; 3/3 deterministic repeats) - commit
      `6dfc33500`.
- [x] Batch-route quality gate passed bit-exact (current_package_direct,
      widths 3,5,6,7: KL 0.0, top-1 1.0) and satisfies the lifecycle gate's
      quality-artifact contract.
- [x] Lifecycle under the drift protocol passes after separating stable
      scheduler/KV slots from the manifest-declared dense ephemeral execution
      rows (`bf1b26b58`): C13 -> C11 -> C13 masks are exactly 8+5 -> 8+3 ->
      8+5; tokens, externally authorized state/KV, cancellation/refill session
      reuse, declared widths, no-serial-fallback, allocator recovery, and drain
      all pass.
- [x] Current-source comprehensive production gate passes all **252** physical
      full-logit rows bit-exact, three-repeat live/teacher determinism,
      neighbor replacement + row permutation, all ten paired task verdicts,
      strict fallback/profile manifests, candidate-D2H-zero, zero recovery,
      ownership, and drain. The selected graph policy is eager-only, so no
      graph/eager arithmetic pair exists to reconcile.
- [x] Acceptance is reported by K, draft depth, category, train/heldout,
      prior-cycle transition, and each prompt/lane request. The concentration
      after prior rejects (especially code K2/K3), the cooldown discriminator,
      and exact position/hidden/KV/full-logit/isolation gates identify a
      by-definition-hard correction-token policy effect rather than persistent
      provider-state corruption.
- [x] Long contexts are explicitly out of physical-target scope
      (`GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT=95`) and fail closed to K0.
      A ten-task 1K/4K diagnostic executes zero candidate cycles and is
      per-slot AR/MTP equal; 9/10 cells are same-slot exact. One 4,089-token
      Japanese heldout has a deterministic one-token slot-local production-AR
      near-tie and reconverges immediately. Retain that pre-existing K0/AR
      diagnostic as a limitation; it does not authorize or block the bounded
      physical-C2 target candidate.

P8 exit achieved: explicit-only, correctness-qualified physical C2 baseline
and explained acceptance curve. Automatic remains K0. Evidence:
[`P8 closure`](../benchmarks/results/2026-08-29-w7900-qwen38-q4km-p8-c2-correctness-closure.json).

W7900 after-economics across C1-C8 (20260829, worklog 20260829T091847, artifact
`2026-08-29-w7900-qwen38-q4km-after-c1c8-economics.json`): AR scales 21.88 ->
44.35 tok/s (C1-C8) with all 80 cells exact; MTP K2 explicit wins only at
C1 (1.4834x at capacity 1; 1.3381x in the cap-8 sweep cell) and C2 (1.1914x)
and collapses to 0.53-0.64x at C3-C8 with acceptance held at ~0.68 because the
R9-R24 verifies ride the shared-B 256-row fallback (gfx1100 admits only
rows6; gfx1151 admits {6,8,9,12}). Prefill W7900 lane: 870.8 / 909.9 /
888.2 tok/s at 512/1K/4K. Same-host llama.cpp complete-wall comparison
(2026-08-28 rebaseline): hipEngine leads MTP by 1.2642x vs current and
1.1962x vs Laurent, AR by 1.047/1.110x. Remaining-loss ranking: (1) accept
stage 30.4 ms full-device sync, (2) C3+ off-tile verify, (3) proposal
13.3 ms vs <1 ms weights floor, (4) verify 22.5 ms vs ~17.4 ms weights floor.
Production-extraction punchlist (each behind its own numerics+perf gate):
rows {8,9,12} are now covered by padding to rows6 multiples and splitting each
projection into qualified rows6 launches; remaining candidates are Q8_T16
decode rowtile/pair min-rows, Q4/Q5/Q6 selected pairreuse min-rows, Q5 selected
tile8, and the indexed singleton decode. The retained rows6-multiple result
passes clean C2/K3/R8 and C3/K3/R12 full-logit gates (240 rows each, KL max
0.001341, top-1 100%) plus cached-build rowtile tracing. Same-protocol K2 moves
C3 **21.549 -> 32.776 tok/s (+52.10%, 0.9079x AR)** and C4 **24.314 ->
36.141 tok/s (+48.64%, 0.9196x AR)** with 20/20 exact/engaged cells and no
recoverable failures. C2-C4 K3 is mechanically clean but reaches only
0.9695x/0.8536x/0.7854x AR. Retain explicit scope; automatic C3/C4 remains K0.
Evidence: `2026-08-29-w7900-qwen38-q4km-rows6-multiple-rowtiles-retained.json`.
W7900 survey port complete (20260830): the standardized common-suite C1-C8
matrix now includes hipEngine plus current/Laurent llama.cpp HIP on the same
standard Q4_K_M file and complete-wall boundary. hipEngine leads AR at C1 and
C3-C5; Laurent leads AR C6-C8 and K3 MTP C1/C3-C8; current leads AR/MTP C2 and
prefill C1-C7. All current rows and 78/80 Laurent AR/MTP cells are content-
exact; the two Laurent C8 differences are deterministic/non-repetitive. K3 is
an engine diagnostic distinct from the promoted hipEngine C2/K2 product key.
Evidence: `2026-08-30-w7900-qwen38-q4km-c1c8-cross-engine.json`. q38rocm strict
K4 C1 remains a custom-FP4 technique reference; Kyanite warm replay and
DFlash2-adaptive sequential results stay excluded as invalid/unsafe.

#### P9 — Decision controls, cycle budget, and measured attribution

- [x] Screen correctness-qualified C2 K1/K2/K3 on the complete prompt suite
      against same-protocol true AR. Select production C2/K2/R6: K1/K2/K3 are
      1.0866x/1.1902x/0.9695x AR with 114/92/92 actual physical cycles and
      4.2105/5.2174/5.2174 visible tokens per physical cycle.
- [x] Audit the logical-C2 `partitioned_c1` control. No same-due owner exists:
      cap-2 singleton service and explicit API attempts both resolve pure K0,
      while serial calls are not a due group. The prototype commits were fully
      reverted (`6279548f1`) rather than fabricating aggregate/per-request/
      fairness/TTFT/ITL/E2E metrics. Implementing this control is new policy
      work, is not physical C2 evidence, and remains N/A for P9.
- [x] Compute physical-cycle budgets from generated visible tokens / actual
      physical cycles (never acceptance): K1/K2/K3 1.10x budgets are
      124.66/155.70/154.75 ms and 1.30x budgets 105.48/131.74/130.95 ms.
      Normalized complete walls are 126.20/143.90/175.59 ms, giving K2
      +11.80 ms headroom to 1.10x and -12.15 ms to 1.30x.
- [x] Prebuild/freeze cached builds and profile focused current-model C1/C2
      children with marker, kernel, HIP-runtime, copy, and allocation traces.
- [x] Reconcile C2 cycle wall **71.46 ms**: proposal 13.46 ms and target+
      accept+commit+provider 57.86 ms. One request's group timer decomposes to
      target 44.57, accept boundary 9.41, provider repair 3.21, and commit
      1.00 ms/cycle. Pure accept kernel is only 0.012 ms/cycle; 15 sync calls
      over four cycles wait 11.22 ms/cycle but overlap GPU work. DMA trace is
      zero; 344 rocclr copyBuffer kernels/cycle cost 1.70 ms; one tiny allocation
      appears over four cycles. C1 -> C2 wall/kernel slopes are +22.31/+20.85 ms,
      with actual R3 -> R6 Q4/Q5/Q6 kernel names and plausible durations.
- [x] Rank measured owners: target Q4/Q5/Q6 projection rowtiles 37.56 ms/cycle
      (high-risk arithmetic), proposal 13.46 ms (medium-risk, ~12.46 ms over the
      prior weights floor), overlapping synchronization ceiling 11.22 ms,
      copyBuffer kernels 1.70 ms, commit/repair 1.11 ms. P10 should not treat
      the accept timer as pure accept compute.

P9 exit achieved with one selected C2 `(K2,R6,production,Qwen3.8 NextN dense
chain)` candidate, measured 1.10x/1.30x budgets, and ranked cost centers.
Automatic remains K0. Evidence:
[`P9 attribution`](../benchmarks/results/2026-08-29-w7900-qwen38-q4km-p9-cycle-attribution.json).

#### P10 — Evidence-ranked implementation tracks

Work these tracks in measured ROI order; none is mandatory merely because it is
listed. Each arithmetic or kernel change starts with a RED oracle, keeps a
registered strict fallback, runs the applicable profile gate before timing,
records temporary flags in `REFACTOR.md`, and lands as its own validated commit.

##### Track A — Reuse or extend target owners

- [ ] Reuse/requalify existing target variants when Qwen3.8 representation and
      shapes truly match; otherwise add only the measured missing R4/R6/R8
      Q4/Q5/Q6 owner(s) behind four-axis registry variants.
- [ ] Run `scripts/check_lineage.py`, the strict/production kernel RED gate,
      CPU-reference outer floor, and `rocprofv3` engagement before retention.
- [ ] Gate success on the local Qwen3.8 cycle budget and same-route sub-window
      improvement, not an absolute Qwen3.6 per-row time.

##### Track B — Optimize the already-batched proposal family

- [x] P7 confirms one active-row proposal batch per shared depth, request-major
      device candidates, rows=1 ragged tails, and no candidate D2H before target.
      Generic draft batching is closed as already implemented.
- [x] Preserve exact proposal tokens/logits, provider hidden/KV/cursors,
      masks/positions, RNG, checkpoints, ragged K, EOS/stop, cancellation, and
      refill while optimizing the P9-measured synchronization boundary. Commit
      `54ab91b9d` keeps the device-resident packed NextN model step enqueue-only
      on stream 0; every other caller retains `synchronize=True`. Focused
      proposal sync calls fall 7 -> 0 and proposal marker wall
      13.46 -> 6.17 ms/cycle (-54.1%); cycle wall improves 0.93% under the
      matched profiler despite attribution moving into the following composite.
- [x] Do not add a new head/projection owner: after removing the barriers,
      unchanged kernel names/counts plus the stable 252-row gate show the next
      proposal optimization requires a separately measured kernel premise.
      Retain the current batched executor and rows=1 strict fallback.

##### Track C — Recover visible yield only after correctness

- [x] P8 refutes persistent state/composition corruption: full-logit isolation,
      exact ownership, and the cooldown discriminator identify hard correction
      tokens. No acceptance policy or prompt-conditioned mechanism is added.
- [ ] If acceptance is correct but insufficient, evaluate K/context/horizon
      admission and then provider-declared tree/adaptive proposals only when
      expected visible tokens per verified row improve under the full category
      suite and category-heldouts. No prompt text, token ID, candidate-ID rerank,
      or heldout-conditioned branch may enter policy/dispatch.

##### Track D — Follow any other measured owner

- [x] Remove the measured redundant accept dependency (`417da8a26`): the first
      bounded blocking default-stream D2H payload copy already retires the accept
      producer, so the preceding whole-device sync is unnecessary. Composite
      sync calls fall 8 -> 4 and wait 2.12 -> 1.35 ms/cycle; kernel counts stay
      unchanged. Wall is variance-flat, so retain this as an exact sync/queue-
      ownership win, not a headline speed claim. Legacy diagnostic accept keeps
      its strict synchronized path.

P10 exit achieved. Stable integrated quality passes 252/252 logits bit-exact
plus repeat/permutation/tasks/profiles/lifecycle. Complete-suite explicit C2/K2
remains **36.149 vs 30.296 tok/s = 1.1932x AR**, above the 1.10x budget but below
1.30x; acceptance is unchanged at 0.80294. The same-suite proposal timer falls
13.28 -> 2.65 ms/request-cycle and named-stage sum 70.35 -> 69.87 ms; aggregate
MTP wall is noise-flat (-0.30%). Automatic remains K0 for P11/P12. Evidence:
[`P10 sync wins`](../benchmarks/results/2026-08-30-w7900-qwen38-q4km-p10-sync-wins.json).

#### P11 — Integrated explicit C2 qualification

- [x] Integrated production passes all **252** actual full-logit rows bit-exact
      (KL 0, top-1 100%), three-repeat live/teacher determinism, same-width
      neighbor/permutation isolation, all ten tasks, strict/production manifests,
      and registered strict fallback. Current strict C2 K1/K2/K3 each pass 10/10
      exact/engaged. Batch-composition generated-ID equality remains diagnostic
      under production; the selected physical policy is eager-only, so graph/
      eager reconciliation is N/A rather than transferred to a future graph owner.
- [x] Retain P8 reject/partial/full and pre/postcommit failure evidence, then
      re-run integrated serving/resources: three concurrent SSE pairs are
      deterministic by lane; cancel-after-5 leaves the survivor exact; capacity-2
      overload completes three explicit requests. Prefix/COW fails closed at
      `cache_off` with zero COW/reuse/admission fallback. All runs drain active
      requests/states/provider groups/sinks/queues, candidate D2H and recovery are
      zero, and tracked allocator returns to zero after 24.233/23.354 GB peaks.
- [x] Retained production C1/K3/cap1 is non-regressive at **1.5022x AR**.
      Public automatic C2/C4 executes 0/20 MTP cells at **0.9989x/1.0012x** AR,
      proving ordinary K0 before provider mutation for unqualified keys.
- [x] Final cached-build trace at the integrated manifest records four physical
      cycles, rows6 Q4/Q5/Q6 target owners, request-major proposal rows,
      accept/commit owners, 5,774 kernel calls, zero DMA copies/candidate D2H,
      and only eager execution routes.

P11 exit achieved: the explicit-only physical C2/K2/R6 candidate passes the
complete correctness, lifecycle, resource, serving, negative-key, and trace
packet. Automatic remains K0 until P12. Evidence:
[`P11 integrated`](../benchmarks/results/2026-08-30-w7900-qwen38-q4km-p11-integrated-explicit-c2.json).

#### P12 — Economics, promotion, and wider widths

- [x] Three clean counterbalanced complete-wall runs on all ten prompts/four
      heldouts reach **1.1986x/1.1955x/1.1970x AR**, median **36.726 vs 30.720
      tok/s = 1.1970x** with stable 0.80294 acceptance and every category
      >=1.1363x. The >1.30x project target remains unmet. Predeclared streaming
      thresholds TTFT-p95<=2.0 s, ITL-p99<=0.35 s, E2E-p95<=3.0 s pass for all
      60 measured AR/MTP requests; SLO goodput is 30.503/32.193 tok/s and all
      cross-arm IDs/repeats are exact.
- [x] Register only full model SHA/size, gfx1100 production manifest, BF16,
      capacity2/realized-C2/K2, greedy, max-seq1024, context4-95, and D24.
      Unflagged automatic is 10/10 exact/engaged at **1.1976x AR**. Resident due-
      group telemetry shows transitional no-provider K0 before the C2 group
      selects `[2,2]`; no prompt content enters policy. Automatic SSE/cancel
      drains exactly.
- [x] `partitioned_c1` remains structurally unavailable and is not part of the
      promotion. No serial/K0 result is relabeled.
- [x] Post-policy negative runtime rows—cap4 C1/C2/C4, C3 policy unit, K3, D25,
      context>95, and sampled—are status-200 ordinary K0 with zero cycles and no
      adapter/provider allocation. Current explicit C3/C4 remains exact but
      only **0.9124x/0.9237x AR** (category minima 0.8564/0.8801), so automatic
      stays K0.
- [x] Publish P8-P12 artifacts, README/changelog, immutable worklogs, focused
      tests/traces, refactor note, and clean pushed commits.

P12 exit achieved: exact Qwen3.8 physical C2/K2 is automatic on W7900 within
its bounded key; C3/C4 and every scope miss remain K0. Evidence:
[`P12 promotion`](../benchmarks/results/2026-08-30-w7900-qwen38-q4km-p12-c2-automatic-promotion.json).

## 12. P13 extension: audit recovery and cross-engine parity continuation

Opened **2026-08-31** after reviewing all 28 commits after the fresh-coder
handoff (`01dba507d..dbf5d263c`). P7-P12 remain complete: the bounded
capacity-2/C2/K2 automatic product key is retained. P13 is a separate recovery
and parity lane for the standardized capacity-8 C1-C8 matrix. It does not reopen
or weaken P12's product gate.

### 12.1 Operator and evidence provenance

Qwen 3.8 Flash Next NVFP4 was used as the coding assistant for the reviewed
segment. This is operator provenance only. It is not the Qwen3.8-27B GGUF model
under test, a benchmark arm, or evidence about NVFP4 inference. The review found
repeated source/provenance/arithmetic mistakes, so outputs from that assistant
are not trusted campaign evidence without independent source, artifact, and
device verification. It is removed from the trusted campaign-author role.

Immutable historical worklogs are not edited. The correction entry
`worklog/entries/20260831T074458.097679Z-lhl-qwen38-parity-audit-recovery-e72fe5.md`
supersedes the affected conclusions and names every mutable artifact/doc update.

### 12.2 Corrected current matrix and success criteria

Compare hipEngine consistently with the **strongest** current/Laurent peer in
each cell of the standardized W7900 matrix:

| Axis | Current result | Remaining strongest-peer deficits |
| --- | --- | --- |
| True AR / decode | wins 7/8 | C2 **-9.94%**; C1 is +1.28%, while C3-C8 lead +32.55% to +95.07% |
| Explicit K3 diagnostic | wins C3/C4 | C1 -3.90%, C2 -2.98%, C5 -6.01%, C6 -11.50%, C7 -19.88%, C8 -38.87% |
| Prefill | wins C2/C3/C4/C7 | C1 -21.48%, C5 -3.21%, C6 -5.08%, C8 -6.26% |

Material prefill work is now C1. Atomic ready-cohort admission moved C2/C3 past
the strongest peer; the C5/C6/C8 differences remain near parity and must be
judged against a same-protocol repeat band rather than used as a kernel tuning
target from one packet. The exact final per-cell matrix is in §12.6.

P13 is complete only when:

1. every current claim and retained command is provenance-valid and every
   cited artifact is covered by the checkers;
2. true AR/decode and prefill beat the strongest peer in every binding C1-C8
   cell, or a measured structural blocker is recorded without weakening the
   gate;
3. the explicit K3 engine-ranking diagnostic beats the strongest peer at every
   width under the same full-suite protocol, while remaining visibly distinct
   from automatic product routing;
4. the exact capacity-2/C2/K2 automatic product key stays qualified and every
   capacity-8/scope miss remains truthful K0 unless independently promoted;
5. all retained changes pass the applicable exact/profile, lifecycle, task,
   repeat, and same-host evidence contracts.

### 12.3 Audit corrections that bind P13

- **Stale K3 packet.** `...hipengine-refresh-post-promotions.json` was produced
  before grouped prefill. Its 31 tok/s plateau and width-dependent acceptance
  are historical pre-grouping evidence. They cannot drive current tasks.
  Pre-#30 grouped evidence reports draft acceptance **0.7889 at C1-C8**. The
  final one-group C8 packet records **0.7850** (1,256/1,600) versus the C4
  rollback's **0.7889** (the same 1,256 accepted from 1,592 proposals).
- **Serving path.** The serial loop in `_generate_greedy_batch` is a direct
  compatibility/control path. Current resident serving groups mixed-length
  full prompts in `_try_prefill_native_work_batch`, with grouped counters
  observed at C2-C8. Do not optimize the direct loop as a server-matrix fix.
- **W7900 geometry.** The device has **96 CUs**, not 512. The down projection's
  107 blocks × four wave32s are about **4.46 waves/CU** versus a maximum 32.
  Additional tiling may improve latency hiding, but split-K is a candidate to
  measure, not a proven 4.8x CU-underfill fix.
- **T16 timing protocol (resolved by #25).** The retained sweep executed one
  fixed arm order and two back-to-back passes; it was not counterbalanced
  despite saying so. The repaired harness now runs forward/reverse order pairs,
  reverses repeated pair order, takes no best-of selection, and returns owned
  per-pass output captures. A tracked-clean all-shape repeat confirms ULP-0,
  finite output, the two retained row-128 shapes, and every recorded loss.
- **Band edge (resolved by #25).** Historical `(5120,12288)` row 128 measured
  0.970x and 1.022x. A focused five-pair/50-rep repeat retains row 112 at
  **1.1017x** in both orders but measures rows 120/124/128 at
  **0.9943x/0.9949x/0.9992x**, each regressive in both orders. That shape now
  stops at row 112; shared-B owns rows 113+.
- **Command integrity.** Missing `--output` and prose such as “then the same
  tool” are not as-run commands. Unknown argv is recorded as null; executable
  reconstructions are labelled templates.
- **Teardown.** `host_unregister` failure must not prevent the adapter's
  remaining ngram/workspace/scratch cleanup.

### 12.4 P13 task order

#### P13-A — Evidence recovery (#24)

- [x] Publish the immutable correction entry and extend this document.
- [x] Correct mutable artifacts, benchmark caveats, commands, strongest-peer
      ratios, stale citations, and 96-CU geometry.
- [x] Pass JSON/worklog/README/provenance/command checks and commit atomically.

#### P13-B — Current prefill attribution and admission repair (#22/#29 complete)

- [x] On current HEAD, collect a C1-C3 same-protocol repeat pair with current
      single-wave ownership and observe whether each wave actually groups.
- [x] Attribute actual EngineLoop serving wall across render/tokenize,
      admission preparation, native grouped prefill, and residual wall; record
      group sizes, not only cumulative counters.
- [x] Reconcile the conflicting “~97% WMMA kernel” and “~0.4 s non-kernel”
      readings before choosing host versus kernel work.
- [x] Publish ready default-AR children to EngineService atomically without
      coupling their completion, cancellation, error, or reclaim lifecycle.
- [x] Retain only after a tracked-clean same-build rollback/default C1-C3
      packet proves full groups, exact IDs, clean memory, and no C1-specific loss.

The pair separates two regimes. C1 is **89.9% complete native-prefill call**
(268.3/298.4 ms), with under 1 ms of listed frontend work and 29.7 ms residual,
so C1 remains kernel-shaped. At C2/C3, explicit AR enters the server as
independent queue items (`request_count=1` in all 40 cells) and races resident
admission: only 5/20 C2 cells form a full size-2 native group; C3 forms 5/20
size-3 and 15/20 size-2 groups. The omitted-request automatic arm declines MTP,
executes exact K0/default, but queues full waves 20/20 and measures stable
**259.3/344.4 prompt tok/s**, **1.082x/1.329x** the strongest peer. Therefore
#29 coalesces ready default-AR submissions before any C2/C3 kernel conclusion;
#25 is complete; #11 subsequently retained the exact row64 down-projection
owner described below. Evidence:
[`current server attribution`](../benchmarks/results/2026-08-31-w7900-q4km-c1c3-current-server-prefill-attribution.json).

Task #29 now closes that race. Compatible default-route items already ready
inside the configured frontend window enter one EngineService admission command;
per-request handles preserve independent completion, cancellation, failure,
backpressure, and reclaim, and dynamic arrivals use later free capacity. In a
tracked-clean same-build rollback/default packet, explicit C2 full native groups
move **2/10→10/10** and C3 full groups **0/10→10/10**. Prompt throughput moves
**178.504→261.748 (+46.63%)** at C2 and **223.643→346.923 (+55.12%)** at C3,
positive in every category and **1.092x/1.339x** the strongest peer. C1 is
**158.868→157.774 (-0.69%)**, but the untouched grouped K0 control moves the same
-0.68%; candidate non-native residual decreases by 0.057 ms, so no candidate-
specific C1 loss is measured. All 120 cross-packet generated rows match, both
runs end with zero active allocations, and 58 lifecycle/admission contracts
pass. Evidence:
[`ready-cohort retention`](../benchmarks/results/2026-08-31-w7900-q4km-default-ar-ready-cohort-retained.json).

#### P13-C — T16 evidence and down-projection occupancy (#25, then #11)

- [x] Counterbalance the T16 harness, test its arm schedule/output capture, and
      correct exact command records.
- [x] Narrow the weak `(5120,12288)` edge or retain it only after repeats.
- [x] Measure ISA/runtime occupancy for `(17408,5120)` under the 96-CU model;
      try a finer row/column tile before a reduction, and retain split-K only if
      operation-complete timing repays accumulation.

The task-#25 correction is tracked-clean at harness revision `0c7dc9150`.
Across the all-shape sweep, `(5120,17408)` is **1.4053x..1.0753x** and
`(5120,10240)` is **1.2182x..1.1261x** over sampled rows 2..128, with every
retained point positive in both arm orders. The four omitted shapes lose at
all 40 sampled cells. The shape-specific `(5120,12288)` cap changes
**128 -> 112**; no published throughput row changes. Evidence:
[`counterbalanced T16 correction`](../benchmarks/results/2026-08-31-w7900-q4km-t16-single-wave-counterbalanced-band-correction.json).

Task #11 measured the incumbent at **428 wave32s / 107 blocks**
(**4.46 waves/CU**) with VGPR256/LDS24 KiB under rocprof. The retained row64
sibling preserves the four-wave shared-weight arithmetic but reduces ownership
from 256 to 64 rows, cuts runtime VGPR to 248, and supplies 856 waves above row
64 without any reduction. Five forward/reverse pairs are ULP-0 and positive at
every admitted point: **1.009x-1.855x** through row 192; row 193 crosses to
**0.897x**, so the strict parent owns row193+. On the same build, full-suite C1
prompt throughput improves **4.63%/4.51%** in both exact arms and every category;
the stable full-group K0 control is **+0.62% at C2** and flat-positive at C3.
Explicit C2/C3 AR remains excluded from the kernel verdict because group
composition races, as assigned to #29. Split-K was not attempted because the
reduction-free exact owner already wins. Evidence:
[`row64 down-projection retention`](../benchmarks/results/2026-08-31-w7900-q4km-t16-downproj-row64-retained.json).

#### P13-D — Current K3 attribution and physical-width remedy (#23/#12/#30 complete)

- [x] From a post-grouping packet, report decode-only rate, request-local and
      conditional positional acceptance, physical group sizes, adapter
      `max_requests`, and accept-window costs at C1/C3/C5/C8.
- [x] Decide whether C5-C8 is limited by grouping, accept/readback/commit,
      proposal, or verifier work before changing verify kernels.
- [x] Measure rowtile versus small-M only through the production sidecar payload;
      never reconstruct the layout that previously wedged the GPU.
- [x] Widen the qualified gfx1100 production physical transaction through C8,
      including server admission, frontier, cycle/accept owners, exact fallback,
      failure cleanup, and a full-stack C4 rollback.
- [x] Close C5/C8 with tracked-clean D1/D24 full-category+heldout packets,
      exact candidate/rollback trajectories, physical-shape evidence, and drain.

A clean current D1/D24 pair excludes acceptance: aggregate draft acceptance is
**0.788944724** and conditional P1/P2/P3 is
**0.929577/0.836066/0.833333** at every measured width; each prompt/request
candidate-and-accept sequence is identical at C1/C3/C5/C8. Decode-only MTP/AR
is **1.722x/1.146x/0.611x/0.603x**. The capacity-8 owner nevertheless resolves
adapter `max_requests=4`, workspace `[4,5120]`, and actual groups exactly
`[1]`, `[3]`, `[4,1]`, `[4,4]` in all ten waves per width. EngineLoop executes
those subgroups serially. A size-4 group costs the same at C5 and C8
(**146.097/146.340 ms** named stages), while C8 pays two: **207.488 ms** of its
**292.681 ms** named-stage sum is accept, closed by **48.832 ms** selected
commit plus **158.228 ms** blocking readback. That readback is a dependency wait
including queued device retirement, not pure copy compute. The primary blocker
is therefore the physical group-of-four ceiling plus one accept dependency
window per serial subgroup; direct verifier submit is secondary in host timing.
#12 confirms the leaf is not the remedy: on normal materializer-owned tiles,
small-M is strict-exact in all **35** standard-Q4 role/row cells but loses every
one, **2.52-14.84x** by HIP events and **2.51-13.53x** operation-complete. The
shipping rowtile/col4 owners remain default; no route changed, so #23's fresh
40/40 incumbent packet is the applicable MTP gate rather than a fictitious new
promotion. #30 owns a RED-first wider-physical-group or shared-readback design.
Evidence: [`current post-grouping K3 attribution`](../benchmarks/results/2026-08-31-w7900-q4km-current-post-grouping-k3-attribution.json) ·
[`production-sidecar small-M rejection`](../benchmarks/results/2026-08-31-w7900-q4km-t16-production-sidecar-smallm-rejected.json).

Task #30 removes the measured serial ceiling without changing arithmetic. The
package-qualified production owner and explicit server route now resolve through
C8; strict, automatic/default-AR, and the full-stack environment rollback stay
at C4. In the final clean `477ee2471` gate, C5 changes `[4,1]→[5]` and improves
**49.227→57.345 tok/s (+16.49%)**; C8 changes `[4,4]→[8]` and improves
**56.414→61.785 (+9.52%)**. Every prompt/category is positive, all 520 D1/D24
candidate/rollback generated rows match, D1 remains K0 without an MTP owner,
D24 has zero recoverable failures, and all four processes drain. The canonical
C5 row moves **47.960→57.345**; the final non-best-of C8 repeat replaces
**62.985→61.785** and remains +10.61% over the pre-task 55.860 cell. Automatic
C5-C8 remains K0. Evidence:
[`C5/C8 physical-group closure`](../benchmarks/results/2026-08-31-w7900-q4km-c5c8-physical-group-closure.json).

The final audit found that the canonical C6/C7 cells still named the old split
route even though the retained capability changes those widths too. A clean
same-commit completion packet closes that rollup hole: C6 `[4,2]→[6]` improves
**50.421→66.042 tok/s (+30.98%)** and C7 `[4,3]→[7]` improves
**55.983→62.719 (+12.03%)**, positive in every prompt/category with 260/260
candidate/rollback rows equal and clean drain. The published C6/C7 cells move
**49.020→66.042 (+34.73%)** and **55.225→62.719 (+13.57%)**. Evidence:
[`P13 final audit`](../benchmarks/results/2026-08-31-w7900-q4km-p13-final-audit.json).

#### P13-E — Lifecycle and tooling cleanup (#26, #27)

- [x] Make accept-staging release failure-safe through all subsequent cleanup.
- [x] Repair the packed-prefill runner probe to share the owner's runtime/runner
      and label it runner-level only; serving attribution comes from server
      route counters.

The repaired #27 harness loads one owner, identity-checks every peer against its
runtime/runner, compares mixed/equal packed arms only with their identical
serial prompts, and emits canonical clean-source provenance. A two-lane W7900
smoke passes both arm orders (direct-runner packed/serial **1.70x-1.88x**) but is
explicitly `serving_path_claim_eligible=false`; these smoke ratios do not alter
server attribution or any scoreboard row. Evidence:
[`runner-only grouping probe`](../benchmarks/results/2026-08-31-w7900-q4km-packed-prefill-runner-probe.json).

#### P13-F — Recovery audit (#28 complete)

- [x] Run all changed focused bundles plus worklog/README/provenance/command
      gates, refresh this punchlist, and publish a durable handoff.

Final gate at `34efd0614` plus this documentation tree:

- **273 focused tests pass**: all 241 tests in the nine files changed since the
  audit reset, plus 32 server batcher/EngineService/physical-route nodes.
- Worklog validates **1,289** immutable entries; README exports are synchronized.
- Artifact provenance parses **65/65** cited artifacts with zero violations and
  three pre-existing warnings; published-command drift has zero violations and
  four matched historical exceptions.
- `git diff --check` is clean for the final tree and for
  `dbf5d263c..34efd0614`. Kernel lineage reports 18 tracked references and four
  known external-parent drift entries; the later pinned entries for those
  parent files are clean, and no unreviewed parent code entered P13.
- The audit's missing C6/C7 current-route packet passes 20/20 exact/engaged/
  budget cells per arm, 260/260 candidate/rollback generated rows, every
  prompt/category performance check, physical C6/C7 ownership, zero recoverable
  failures, and zero final tracked allocations.
- The repository-wide concurrency completion audit has integrity `passed=true`,
  no missing evidence, and no false passes. Its expected nonzero status records
  three unrelated declared global blockers (external serving engines, full
  continuous-SpecDec product economics, and no trained DMS checkpoint), not a
  P13 regression.

P13 recovery is closed. Its parity success criteria are not weakened: the
cross-engine objective remains open at the exact deficits below.

#### P13-G — Post-recovery C1 prefill follow-up (#31/#32 complete)

- [x] Profile all ten C1 category+heldout prompts after P13 with cached JIT and
      operation-complete `rocprofv3` boundaries.
- [x] Retain one content-agnostic exact owner only after strict fallback,
      measured crossover, expected-symbol tracing, counterbalanced full-suite
      C1-C3 controls, exact trajectories, and clean drain.

The one-wave rollback profile averages **255.792 ms** operation span,
**242.517 ms** kernel sum, and 1,322.4 launches/prompt. Fused Q4 gate/up is the
largest family at 77.258 ms; planar-Q6 is the best isolated exact target at
**60.704 ms / 64 launches**. The retained gfx1100 package policy applies only
to planar-Q6 FFN-down `(K,N)=(17408,5120)`: row64 owns rows33-128 and the
existing shared256 sibling owns rows129-511; all misses and the non-WMMA
physical verifier keep their prior owners, and one-wave remains separately
registered.

Two tracked-clean counterbalanced full-suite pairs improve mean AR prompt
throughput **157.659→168.217 (+6.70%)** at C1,
**260.070→266.440 (+2.45%)** at C2, and **345.956→353.722 (+2.24%)** at C3.
Every category and heldout scope improves in both exact arms; all 240 paired
rows and both 120-row repeat sets match, all C2/C3 waves remain full native
groups, and all four packets drain. The expected row64 symbol executes 32 times
per public C1 operation. The canonical C1 deficit narrows from **-21.48% to
-16.29%**; fused Q4 gate/up is a separate future optimization unit. Evidence:
[`planar-Q6 prefill retention`](../benchmarks/results/2026-08-31-w7900-q4km-planar-q6-prefill-retained.json).

### 12.5 P13 stop rules

- Do not use pre-grouping rates or acceptance to prioritize current work.
- Do not promote a fixed-order microbenchmark result as counterbalanced.
- Do not tune to one prompt or one width; full category and heldout coverage
  remains binding.
- Do not claim product MTP from the explicit K3 diagnostic or from a capacity
  that misses the exact promoted automatic key.
- Stop on any ownership, state, KV, lifecycle, numerical, task, or repeat gate
  failure and localize it before further optimization.

### 12.6 Durable handoff and exact strongest-peer matrix

This is the canonical retained matrix after the final P13 rollup. Each cell is
`hipEngine / strongest peer (delta)` in total tok/s. “Current” and “Laurent” name
the stronger of the two llama.cpp rows; the underlying exact protocol and
artifacts remain linked from `benchmarks/README.md`.

| Axis | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True AR | 21.999 / 21.720 current (+1.28%) | 31.916 / 35.440 current (**-9.94%**) | 45.309 / 30.787 current (+47.17%) | 54.151 / 27.760 current (+95.07%) | 61.881 / 36.473 Laurent (+69.66%) | 71.226 / 45.826 Laurent (+55.43%) | 74.903 / 52.537 Laurent (+42.57%) | 78.667 / 59.348 Laurent (+32.55%) |
| Explicit K3 diagnostic | 31.455 / 32.733 Laurent (**-3.90%**) | 39.820 / 41.042 current (**-2.98%**) | 54.590 / 45.947 Laurent (+18.81%) | 55.780 / 51.054 Laurent (+9.26%) | 57.345 / 61.013 Laurent (**-6.01%**) | 66.042 / 74.628 Laurent (**-11.50%**) | 62.719 / 78.281 Laurent (**-19.88%**) | 61.785 / 101.072 Laurent (**-38.87%**) |
| Prefill | 168.217 / 200.946 current (**-16.29%**) | 266.440 / 239.658 current (+11.17%) | 353.722 / 259.036 current (+36.55%) | 318.412 / 281.828 current (+12.98%) | 312.682 / 323.043 current (**-3.21%**) | 347.625 / 366.213 current (**-5.08%**) | 426.692 / 374.207 current (+14.03%) | 397.655 / 424.202 Laurent (**-6.26%**) |

Completed recovery work has no hidden “remaining #22/#23/#11/#12” tail:

- #22/#29 measured the real server prefill path and fixed default-AR admission;
  C2/C3 beat the peer and C1 remained complete-prefill shaped. Post-recovery
  #31/#32 then traced the complete suite and retained the exact planar-Q6
  sibling, narrowing C1 from **-21.48% to -16.29%** while preserving C2/C3
  full-group gains.
- #25/#11 repaired T16 evidence and retained the exact Q4 row64 down-projection
  owner. #31/#32 independently measured and retained the planar-Q6 row64/
  shared256 composition; the remaining largest exact family is fused Q4
  gate/up, not another unprofiled down-projection tile.
- #23/#12 rejected acceptance and verifier small-M as the C5-C8 cause. #30 then
  removed the serial physical ceiling across C5-C8. The remaining K3 deficit is
  no longer attributable to split groups; C6/C7/C8 still trail by
  **11.50%/19.88%/38.87%** and require a fresh one-group proposal/target/accept/
  commit Amdahl trace before another kernel change.
- AR remains ahead at seven widths. C2's **-9.94%** is still open and has no
  P13 measurement that localizes it to admission, decode kernels, or host wall;
  do not infer its cause from the now-fixed one-token prefill race.
- Prefill C5/C6/C8 and K3 C1/C2/C5 are repeat-sensitive near-parity cells.
  Re-run the same-host peer and hipEngine protocols in a counterbalanced packet
  before treating them as optimization targets.

Product truth is unchanged: only the bounded capacity-2/C2/K2 automatic key is
promoted. Capacity-8 automatic requests and every scope miss remain ordinary
K0; every K3 number above is an explicit engine diagnostic.
