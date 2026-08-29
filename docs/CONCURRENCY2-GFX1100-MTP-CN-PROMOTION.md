# gfx1100 Physical C>N MTP Promotion Campaign

- Status: **Qwen3.6 lanes complete; 27B Dense and 35B MoE physical C2 promoted
  automatically. Extended 2026-08-28: Qwen3.8-27B effective C=N lane open (§11, P7-P12)**
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
D24 walls are ~50% prefill, so these ratios understate steady-state decode
gains. Evidence:
[`stage/tile/acceptance diagnosis`](../benchmarks/results/2026-08-28-w7900-qwen38-q4km-c2-stage-tile-acceptance-diagnosis.json).

Exit: an explicit-only, correctness-qualified physical C2 baseline and an
explained acceptance curve. Automatic remains K0.

#### P9 — Decision controls, cycle budget, and measured attribution

- [ ] Screen correctness-qualified C2 K1/K2/K3 on the complete prompt suite
      against same-protocol true AR; choose R4/R6/R8 from measured visible yield
      and complete wall rather than inheriting C1/K3. Counterbalance order and
      retain every sample.
- [ ] Run the logical-C2 `partitioned_c1` control under the same due schedule.
      Report aggregate/per-request throughput, fairness, TTFT/ITL/E2E, and
      telemetry honestly; it is a serving control, not physical C2 evidence.
- [ ] For each finalist compute break-even and promotion budgets from actual
      telemetry:
      `max_cycle_wall = visible_tokens_per_physical_cycle /
      (1.10 * true_AR_aggregate_tok_s)`. Also report the `1.30x` project-target
      budget. Never substitute acceptance rate for physical cycle count.
- [ ] Prebuild every `.so`, freeze the compiler-version file, require cached
      builds, and profile focused children—not the parent prompt-suite harness—
      with `rocprofv3` marker, kernel, HIP-runtime, copy, and allocation traces.
- [ ] Reconcile complete cycle wall with proposal, target, accept, commit,
      provider repair, H2D/D2H, synchronization, graph/fallback, allocation, and
      uncovered host shares. Compare C1/C2 marginal slopes and verify actual
      kernel names, shapes, launch counts, and plausible durations.
- [ ] Rank changes by recoverable complete-wall milliseconds and implementation
      risk. If the top cost is not verifier or draft work, change P10 ordering
      rather than forcing the original hypothesis.

Exit: one selected C2 `(K,R,profile,provider)` candidate, a measured cycle
budget, and ranked cost centers with enough recoverable wall to justify P10.

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
- [ ] Preserve exact proposal tokens/logits, provider hidden/KV/cursors,
      masks/positions, RNG, checkpoints, ragged K, EOS/stop, cancellation, and
      refill while optimizing only a P9-measured proposal leaf.
- [ ] If full-vocabulary head or another projection dominates, reuse/requalify a
      genuine rows=C weight-amortized owner behind the registry; retain the
      current batched executor and rows=1 strict fallback.

##### Track C — Recover visible yield only after correctness

- [ ] If P8 proves the acceptance drop is a state/composition defect, fix that
      defect before considering policy changes.
- [ ] If acceptance is correct but insufficient, evaluate K/context/horizon
      admission and then provider-declared tree/adaptive proposals only when
      expected visible tokens per verified row improve under the full category
      suite and category-heldouts. No prompt text, token ID, candidate-ID rerank,
      or heldout-conditioned branch may enter policy/dispatch.

##### Track D — Follow any other measured owner

- [ ] If accept/commit, host synchronization, graph selection, copies, repair,
      or another family owns more recoverable wall, optimize it first with the
      same RED/profile/fallback/registry discipline.

Exit: retained independent wins meet the selected candidate's promotion budget,
or a profile-backed blocker closes the implementation tracks.

#### P11 — Integrated explicit C2 qualification

- [ ] Re-run exact request/state/KV/control ownership, strict and production
      numerical/task gates, same-schedule determinism, same-width isolation,
      batch-composition diagnostics/declared guarantees, graph/eager
      reconciliation, and registered strict fallback on the integrated route.
- [ ] Re-run reject/partial/full accept, delayed arrival, refill, sparse
      retirement, cancellation, prefix/COW policy, pressure/failure recovery,
      overload/soak, blocking/SSE, bounded peak/workspace, and zero final owners.
- [ ] Prove retained C1 is non-regressive and every unsupported or losing key
      reaches ordinary AR before provider mutation without speculative-cap
      decomposition or hidden allocation.
- [ ] Capture a final cached-build `rocprofv3 --kernel-trace` showing the expected
      proposal/target/accept/commit owners under the exact candidate manifest.

Exit: one explicit-only physical C2 candidate passes the complete correctness,
lifecycle, resource, and trace packet. Automatic remains K0 until P12.

#### P12 — Economics, promotion, and wider widths

- [ ] Run counterbalanced three-run complete-wall true AR versus physical C2 on
      the exact committed suite and category-heldouts, with authoritative token
      counts, all categories non-regressive, `>=1.10x` aggregate, and complete
      TTFT/ITL/E2E/SLO-goodput reporting. Preserve the `>1.30x` project target.
- [ ] Register only the exact model-SHA/profile/manifest/K/resident-capacity/
      realized-C/context/horizon/sampling/provider key that passes. Prove real
      automatic due-group selection before mutation, stable K0 reasons and
      ordinary-AR performance for every negative key, and no prompt-conditioned
      policy.
- [ ] A passing `partitioned_c1` cell may be retained under its truthful name and
      independent policy evidence, but cannot satisfy the physical-C2 objective.
- [ ] Evaluate C3/C4 only after C2 promotion; independently repeat the
      applicable P8-P11 ownership/correctness/attribution gates plus P12
      economics, and keep every unsupported/losing wider cell K0.
- [ ] Publish compact artifacts, benchmark README/changelog updates, immutable
      worklogs, focused/milestone validation, refactor updates, and clean merge
      handoff.

Exit: Qwen3.8 physical C2 promotion, or a durable profile-backed structural
blocker record with all safe independent wins retained.
