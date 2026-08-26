# SPECDEC2-PERF-GFX1100 — W7900 Activation and Hot-Cycle Campaign

- Status: **closed; exact explicit C1 retained, zero product promotion, automatic K0**
- Approved: **2026-08-25**
- Functional predecessor: [`SPECDEC2-GFX1100.md`](SPECDEC2-GFX1100.md), G1/P1 foundations complete
- Mechanism reference: [`SPECDEC2-PERF.md`](SPECDEC2-PERF.md), with no gfx1151 evidence transfer
- Development/retention host: **`epyc`, AMD Radeon Pro W7900 / `hip_gfx1100`, GPU 0**
- W7900 identity: **SKU `D7070100`, unique ID `0xe282895b62c2b295`**
- Diagnostic peer only: **RX 7900 XTX, GPU 1**; its rates never form a W7900 old→new comparison
- Dense lane: **Qwen3.6-27B `Q4_K_M`, BF16 KV, strict FP32 recurrent state**
- Packed lane: **Qwen3.6-35B-A3B PARO W4A16 target + BF16 MTP sidecar, BF16 KV**
- Automatic policy at entry: **K0 for every cell**
- Explicit functional scope at entry: **GGUF strict C1 K1-K3; PARO strict/production C1 K1**
- Source base: **`ccd077a1f` or a clean descendant containing both C1 foundations**
- Normative dependencies: [`PLAN.md`](PLAN.md), [`SPECDEC2.md`](SPECDEC2.md),
  [`SPECDEC2-GFX1100.md`](SPECDEC2-GFX1100.md),
  [`SPECDEC2-PERF.md`](SPECDEC2-PERF.md),
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), [`TESTING.md`](TESTING.md),
  [`BENCHMARK.md`](BENCHMARK.md), and [`KERNELS.md`](KERNELS.md)

This is the independent gfx1100 execution of the SPECDEC2 activation and
hot-cycle campaign. It preserves the common Generation-2 scheduler/frontier/
transaction architecture while qualifying two separate provider/target lanes.
It does not copy gfx1151 rates, graph buckets, queue policy, row thresholds,
profile results, or K-policy constants.

The dense GGUF and packed PARO lanes share measurement records and lifecycle
invariants only. They do **not** share model arithmetic, provider capability,
physical target decomposition, manifests, policies, or promotion evidence.

## 1. Goal and definition of done

The preferred result is at least one W7900 staged product cell above `1.30x` the
same-host, same-model, same-profile true-AR path. Automatic speculation requires
more than `1.10x` true AR plus every binding correctness, category, task, SLO,
memory, lifecycle, and isolation gate. Because both lane-specific direct MTP
controls already beat true AR, replacing either direct owner additionally
requires same-source/common-boundary staged wall to be non-regressive versus
that direct control. Exact/profile-qualified sub-window and operation wins
remain retainable even when no staged automatic cell promotes.

The campaign closes when both lanes have durable verdicts for:

1. common timing boundaries for true AR, direct/native control, and staged
   Generation-2 execution;
2. activation and decode-only/steady-cycle walls;
3. streaming prompt priming, retained or rejected under the common protocol;
4. zero cycle-local allocation/free after warmup in every qualified cell;
5. proposal → target → accept → selected commit residency and bounded readback;
6. registered selected and strict-fallback manifests;
7. complete target-family, queue-gap, synchronization, copy, and allocation
   profiles for every admitted physical row bucket;
8. provider repair retain/no-go decisions after the target/device boundary is
   stable;
9. strict and applicable production-profile quality/task/state verdicts;
10. independently fingerprinted W7900 K-policy cells built from retained local
    evidence only;
11. exact K0-before-mutation reasons for losing or unqualified cells; and
12. current compact artifacts, benchmark rollup/changelog, worklogs, refactor
    debt, and closure handoff.

A zero-promotion closure is valid. It must name the measured blocker and retain
strict/direct fallbacks; it may not weaken true AR or silently restore a
whole-request owner.

## 2. Frozen evidence and gap map

### 2.1 Shared architecture

Already closed and not repeated unless owned code changes:

- backend-neutral SPECDEC2 contracts, transactions, provider SPI, policy,
  simulator, frontier, engine-loop, and engine-service: 61 tests;
- Generation-2 request/output ownership, complete claims, rollback, fairness,
  cancellation, and final conservation; and
- local W7900 C1 package exposure with C2/C4 still fail-closed.

### 2.2 Dense GGUF foundation

Retained evidence:
[`2026-08-25-w7900-specdec2-gguf-c1-foundation.json`](../benchmarks/results/2026-08-25-w7900-specdec2-gguf-c1-foundation.json).

- strict manifest `b27ee5d053ccbda64a47084430114948e25feb303d8e90976188e317f9a2012d`;
- C1 K1/K2/K3 staged/direct/AR d8 IDs exact;
- repeat provider fingerprints exact and owners drained;
- staged graph route engages; and
- K2/K3 beat warm AR on one short screen, which is not promotable evidence.

Missing:

- category/heldout common bridge with counterbalanced repeats;
- decode-only and activation attribution;
- target/provider state/KV/following-AR packet;
- cached marked profiler and allocation/synchronization accounting;
- streaming prompt priming (the staged adapter still retains
  `_prompt_hidden_rows` and performs post-prefill replay);
- stable provider/accept/result slabs (proposal still allocates/frees a
  `hidden_batch` inside the cycle);
- C1 candidate/device chain without host reconstruction; and
- physical C2/C4 capability, profiles, production manifest, and local policy.

### 2.3 Packed PARO direct control

The earlier direct-PARO checklist is **frozen evidence, not missing work**:

- three-run canonical D24: exact `720/720`, deterministic acceptance;
- production fast: `115.770` versus `110.830` true-AR tok/s (`1.0446x`);
- borrowed full-vocabulary W8A16 scorer lifecycle/memory: passed, saving
  `1,017,114,848` bytes with one bounded non-growing 8-byte runtime residue;
- strict manifest `3199678e604dca83d40f3d538deb3c05ceed3144b0100ef89b3ecfb94aed5723`;
- production manifest `920737601cca41bc64963437ce1daece0e4ebca1bc52ba5bdd22ae817aba9561`;
- production T2 `decode_batched` verifier qualified against strict `c1_loop`,
  which remains the registered fallback.

Primary evidence:
[`promotion audit`](../benchmarks/results/2026-08-24-w7900-paro-mtp-promotion-audit.json),
[`D24 production`](../benchmarks/results/2026-08-24-w7900-paro-fast-d24-3run-default.json),
and [`borrowed-pointer lifecycle`](../benchmarks/results/2026-08-24-w7900-paro-mtp-lifecycle-gate.json).

These controls may be rerun only when needed as a same-source/timing arm or when
owned code changes. Their qualification is not reopened by this campaign.

### 2.4 Packed PARO staged foundation

Retained evidence:
[`2026-08-25-w7900-specdec2-paro-c1-foundation.json`](../benchmarks/results/2026-08-25-w7900-specdec2-paro-c1-foundation.json).

- production and strict staged d8 IDs equal true AR;
- real C1/K1 staged eager cycles engage;
- final-normalized BF16 prompt rows stream into NextN with one carried 10 KiB
  hidden row and no prompt-sized replay;
- pooled proposer reuse engages; and
- warm staged production is about 9.1% slower than AR on the short screen.

Missing:

- common direct/AR/staged timing and complete D24 category packet;
- persistent claim-backed candidate/result/accept/commit/update slabs;
- device-resident candidate and target result through selected commit;
- staged borrowed-owner close/reuse, zero-hot-allocation, memory, failure, and
  cached-profiler proof (the direct evidence is a prerequisite, not a proxy);
- complete staged production numerical/task/state linkage to the selected
  manifest; and
- physical request-major C2/C4 proposal, R4/R8 target, per-request
  `KVLiveSpans`, parent-indexed Conv/GDN, isolation, and local policy.

### 2.5 Prior gfx1100 implementation reuse map

The implementation pass must start from these shipped or retained mechanisms,
not reconstruct them from the phase names:

| Existing gfx1100 work | Evidence/status | Campaign use |
| --- | --- | --- |
| Dense natural25 AR + direct MTP B1/B2/B3 suite | Current W7900 snapshot is exact across ten prompts; true AR `29.457`, B3 `60.929 tok/s` (`2.0684x`) | Reuse `scripts/qwen36_dense_gguf_suite.py` execution/metrics as the direct arm and oracle. The new bridge adds staged/common-boundary ownership; it does not replace a working suite with duplicate model logic. |
| Dense state-bound PM4 AR replay | Current p512/p1K/p4K AR snapshot uses the retained PM4 transport with stable state and memory | Preserve as the true-AR product control. Do not substitute eager or HIP graph solely to make staged MTP look faster. |
| N1R reusable B1/B2/B3 target graphs | Fixed-address scratch, live positions/context/`KVLiveSpans`, eager pre-mutation fallback | Reuse as target graph/oracle infrastructure. Never recreate the rejected position-bound per-cycle N1 capture. |
| N2 device accept + selected target commit | Retained exact reject/partial/full selected hidden/Conv/GDN/KV/cursor/rollback; one bounded accept/result payload | Wire the staged C1 path to this owner. Device accept/selected commit is not net-new dense work. |
| N3/N3P complete direct cycle and chained proposal→target retirement | One scheduler-facing direct call; cached proposal graph records an event, target waits/copies device IDs, and one final synchronization retires both graphs | Reuse descriptors, stable buffers, graph generations, and bounded results inside one Generation-2 cycle. Remaining work is staged ownership/claims and provider repair, not another whole-request owner. |
| OI-3 exact streaming prompt components | `TargetHiddenChunkSink`, `_StreamingNextNPromptSink`, and `enqueue_prompt_rows()` are retained shared source | Integrate them into dense staged activation and qualify on W7900; do not rewrite the sink. |
| Shared SPECDEC2 physical C2/C4 dense source | `propose_batch_device`, device `CandidateGraph.token_ids`, packed target lowering, one GPU accept/group, and selected group commit are already in the merged adapter from gfx1151 S4 | Treat as unqualified gfx1100 source, not missing architecture. Remove hot allocation and run independent W7900 capability/shape/state/profiler gates before exposing C2/C4. |
| Dense sole-T16 resident layout and compact scratch/arena work | Current target+NextN has no alternate Q4 payload; prefill/decode memory and exact teardown safeguards are retained | Add only bounded staged slabs. Do not reintroduce raw/pack8 weight sidecars or duplicate roots to solve a cycle-local workspace problem. |
| PARO direct B1 provider + strict/production manifests | Corrected final-norm/selected-reseed/borrowed-W8A16 provider; strict `c1_loop`; qualified production T2 `decode_batched` | Frozen direct oracle and product denominator. Staged work must consume these registered identities rather than env-only hybrid routes. |
| PARO N4 target/accept and selected-commit infrastructure | Explicit/default-off gfx1100 NativeSpecCycle target graph supports single-request B1/B2/B3/... and selected linear-state commit; older complete economics were not a win | Reuse ABI, fixed metadata/accept buffers, and selected-commit primitives only after the common current-provider profile. Do not promote the historical N4 wrapper or assume it matches the qualified fast verifier. |
| PARO gfx1100 direct selected-batch C2 AR owner | Retained direct c2 under the unified target contract; public/OpenAI remains width-1 and c4/c8 symmetry is open | Reuse target row/selected-batch mechanics as a C2 oracle. It does not provide request-major MTP proposal, frontier, per-request provisional state, or staged C2 capability. |
| Shared Generation-2 and resource-claim records | K0 planning, complete composed claims, one-cycle fairness, cancellation/recovery, publication, and conservation are closed | Extend only lane-owned physical resource vectors and telemetry; do not add another scheduler or output owner. |

### 2.6 Prior no-go / do-not-repeat map

- **No per-cycle target graph capture.** Initial N1 was exact but capture wall lost;
  use reusable N1R/N2 buckets and generation invalidation.
- **No candidate buffering as a standalone claim.** An older W7900 device-chain
  experiment moved `0.6876x -> 0.6795x`; the retained N3P result wins only because
  it removes intermediate synchronization/host work. A staged candidate must
  net-remove boundaries, not relocate them.
- **No launch-count-only KV batching.** The exact shared-cache writer reduced
  writers `448 -> 112` and kernel sum, but raised target host wall and complete
  marker wall; scalar writers remain the control until a new complete-window
  premise exists.
- **No all-row residual fusion sweep.** The prior all-row policy regressed B3;
  only the retained scope-qualified exact routes may be inherited.
- **No historical N4 wrapper promotion.** Old PARO N4 added one synchronization
  and roughly `0.216-0.447 ms/cycle`; reuse its primitives only if the current
  staged operation-complete profile proves a gain.
- **No private/capped PARO F16 head.** The borrowed full-vocabulary W8A16 owner is
  qualified and saves `1,017,114,848` bytes.
- **No confidence gate, content-adaptive K, max-B active cap, naive side-stream
  overlap, B4/B5, deeper PARO drafts, or full-vocabulary dense scoring pass**
  before fixed staged cells and a fresh complete profile provide a materially new
  premise. Prior direct campaigns rejected or parked these mechanisms.
- **No generic raw-Q4/duplicate-layout resurrection.** Current dense sole-T16
  ownership and W7900 memory safeguards bind; local cycle scratch is not a reason
  to duplicate multi-GiB weights.

### 2.7 Local hot-cycle planning bounds

For one non-tail greedy request:

```text
expected visible tokens / cycle = 1 + K * measured draft acceptance
maximum cycle wall for 1.10x AR
  = expected visible tokens * matched AR step wall / 1.10
```

Applying that formula to prior **direct** W7900 controls gives only a planning
screen, not staged evidence:

| Lane/cell | Prior direct cycle wall | Prior acceptance | 1.10x-AR cycle budget | Planning interpretation |
| --- | ---: | ---: | ---: | --- |
| dense C1/K1 | 41.570 ms | 91.27% | 59.029 ms | direct has 29.58% budget headroom |
| dense C1/K2 | 47.073 ms | 82.97% | 82.072 ms | direct has 42.64% budget headroom |
| dense C1/K3 | 51.829 ms | 77.17% | 102.309 ms | direct has 49.34% budget headroom |
| PARO C1/K1 production | 15.426 ms | 80.92% | 14.840 ms | about 3.80% cycle reduction is needed for 1.10x hot-cycle speed |

Dense values come from the current natural25 publication; PARO values come from
the qualified production D24 packet. The horizons, prompt rendering, timing
aggregation, and owner boundaries differ, and neither table row includes the new
staged owner transition. P1 replaces these values with common-protocol,
cycle-trajectory-weighted committed-token economics. No candidate is admitted
from this table alone.

## 3. Non-negotiable invariants

1. `EngineService` / `ResidentEngineLoop` remains the only request, fairness,
   visible-output, cancellation, and reclaim owner.
2. One bounded target+provider transaction owns proposal, target, accept,
   selected commit, output, repair, and rollback for each cycle.
3. Reject/partial/full acceptance leaves canonical provider KV/cursor/hidden and
   following proposal state under the declared lane contract.
4. Request ID, resident slot, physical row, frontier row, and provider row remain
   distinct through refill and compaction.
5. A promoted device chain does not reconstruct full candidate/target IDs on the
   host before accept/selected commit; only a bounded committed result returns.
6. Qualified warmed cells perform zero allocation/free, graph creation/destruction,
   or lazy build inside a cycle.
7. Resource claims cover every persistent and transient slab before mutation;
   generation/profile/model changes invalidate stale pointers safely.
8. Every fused/graph/production route has a registered strict eager fallback
   selected before mutation.
9. Request/control/KV/state/transaction/lifecycle correctness is exact in every
   profile.
10. PARO production T2 arithmetic keeps its calibrated mean/tail/max KL,
    top-1, repeat, isolation, task, state, and strict-fallback contract.
11. GGUF production remains fail-closed to strict until its own complete
    production profile qualifies.
12. True AR is a separate no-provider path under the same protocol. `off`/B0
    verifier telemetry is diagnostic only.
13. Policy may use C, remaining horizon, context/page bucket, profile, memory
    fit, and a predeclared bounded online acceptance statistic. It may not use
    prompt text/hash, token IDs, category, or heldout identity.
14. W7900 and XTX evidence are independent; no gfx1151 or XTX absolute rate is a
    W7900 old→new denominator.
15. Queue policy is measured locally. The gfx1151 `GPU_MAX_HW_QUEUES=2` setting
    does not transfer; the W7900 baseline records the current resolved setting.
16. Before every phase, fetch `origin/main` and inspect the parallel gfx1151
    lane. A designated merge owner serializes edits to shared dense adapter,
    NextN, runner, and SPECDEC2 source-of-truth files; both affected seam bundles
    rerun after synchronization.
17. Every timing/profiler packet holds the W7900 lease with no concurrent model,
    profiler, or power-changing process. Commands expected to exceed five minutes
    state purpose, duration, GPU identity, stop budget, and output path, then run
    through the background-task owner.

## 4. Common measurement contract

### 4.1 Arms

The committed bridge must expose realized route and physical shape for:

- `true_ar`: no provider and no speculative mutation;
- `direct`: current-source qualified direct/native control for that lane;
- `staged`: bounded Generation-2 SPECDEC2 path; and
- at most one scoped candidate/control pair after admission.

GGUF direct means the current exact dense NextN/native control. PARO direct means
registered production fast or explicit strict, never an unclassified manual
hybrid. Direct-PARO qualification remains frozen, but its current-source arm is
measured to establish the common timing boundary.

### 4.2 Required workloads

| Lane | Packet | C | K | Outputs | Purpose |
| --- | --- | ---: | --- | ---: | --- |
| GGUF | short activation | 1 | 1/2/3 | 8 | reproduce foundation and split cold/warm activation |
| GGUF | category bridge | 1 | 1/2/3 | 24, then retained horizon | full suite economics and attribution |
| GGUF | physical bridge | 2/4 | 1/2/3 | 24 | only after physical capability |
| PARO | short activation | 1 | 1 | 8 | reproduce staged foundation |
| PARO | category bridge | 1 | 1 | 24/64 | direct/AR/staged production and strict |
| PARO | physical bridge | 2/4 | 1 initially | 24 | only after request-major physical capability |

Promotion and keep/revert decisions use all ten committed prompts in
`benchmarks/prompts/mtpbench-code-general-ja.jsonl`, including all four
categories and the six-train/four-heldout split. A fixed greeting is mechanics
only.

Every retained performance row has at least three complete paired repetitions,
counterbalanced AR→MTP / MTP→AR without prompt inspection. Report samples,
median, spread/CV, order, warm/cold state, and available thermal/power data.

### 4.3 Timing and ownership payload

Report complete-request wall and decode-only wall plus:

```text
tokenize/admission/claims_reserve
target_prefill/provider_prompt_prime/provider_open
resident_owner_transition
cycle_total
  proposal/frontier_lower/target_verify
  candidate_or_target_readback/accept_device
  selected_target_commit/provider_update/bounded_result_readback
  output_publish
claims_release/terminal_reclaim
```

ROCTX/profile evidence records marker wall, kernel interval union, queue/API
synchronization, H2D/D2H/D2D, allocation/free, and residual separately.
Residual is not called recoverable without an A/B.

Each row also records exact response-owned generated IDs/counts, prompt IDs/hash,
candidate/accepted depth trajectory, target/provider positions and state/KV
fingerprints at focused gates, physical proposal/target decomposition,
request/slot/row maps, transaction/generation identities, route/fallback,
manifest hashes, allocation high-water, and final conservation.

## 5. Phase graph

```text
P0 independent handoff and gap freeze
  -> P1 lane-aware common bridge + current-main baseline
    -> P2 GGUF streaming activation / PARO streaming revalidation
      -> P3 stable claimed slabs + zero hot allocation
        -> P4 device-resident proposal -> target -> accept -> commit
          -> P5 complete W7900 physical target profiles
            -> P6 at most one admitted target candidate per lane
              -> P7 conditional provider repair
                -> P8 profile completion
                  -> P9 physical C2/C4 + policy/product qualification
                    -> P10 closure
```

Each phase is one committed logical unit with an immutable worklog. Downstream
work never begins from failing upstream gates.

## 6. P0 — independent handoff

- [x] Audit the complete gfx1151 mechanism plan without transferring evidence.
- [x] Select stable W7900 GPU0 as the only retention host.
- [x] Keep GGUF and PARO capabilities/evidence independent.
- [x] Freeze completed direct-PARO D24/lifecycle/manifest gates as controls.
- [x] Map the actual dense and staged-PARO gaps.
- [x] Define common arms, timing boundaries, phase ordering, stop rules, and
      automatic threshold.
- [x] Validate docs/worklog and publish this P0 commit to `origin/main`.

## 7. P1 — common bridge and current-main baseline

- [x] Add the committed `scripts/specdec2_perf_gfx1100_bridge.py` row contract,
      content-agnostic arm planner, strict validator/aggregator, and atomic
      checkpoint writer.
- [x] Reuse the shared `scripts/specdec2_perf_bridge.py` for dense GGUF after its
      backend-neutral gfx11 generalization; add `run-loaded-paro` so packed PARO
      emits AR/staged row-contract checkpoints from one loaded process.
- [x] Add unavoidable-reload PARO direct attachment and parent packet assembly
      through the qualified economics child, including raw exact IDs, selected/
      strict manifests, target/provider activation, decode, and clean source.
- [ ] Execute and attach current-source production/strict PARO direct packets;
      production is attached; strict was executed but attachment correctly fails
      exact compact-product-AR versus serial-direct parity on one heldout prompt.
      Do not add backend/model policy in engine hot paths.
- [x] Add schema, aggregation, counterbalance, exact-denominator, missing-owner,
      malformed-manifest, incomplete-suite, dirty-provenance, stage-reconciliation,
      and atomic-checkpoint tests.
- [x] Support lane-appropriate AR/direct/staged arms, profiles, entry C/K,
      horizons, prompt scopes, repeats, immediate progress, and atomic checkpoints.
- [ ] Emit complete/decode-only attribution, exact accounting, physical shapes,
      trajectories, fingerprints, allocation/synchronization counters, and drain.
- [x] Add named packed-PARO target-prompt, provider-prompt-prime, tokenize, and
      decode timing; loaded/direct rows preserve non-overlapping ownership and
      keep unresolved admission/transition/publication work as residual.
- [ ] Complete resident-owner transition instrumentation: packed-AR
      flush/scatter/discard, graph close/invalidation, root-hidden handoff, and
      provider attach counts must reconcile rather than disappear into residual.
- [x] Add cached single-child profiler mode; never profile a nested parent.
- [x] Run GGUF C1 K1-K3 d8 and ten-prompt natural25 retained-horizon packets.
- [x] Run PARO C1/K1 production and strict d8 and ten-prompt D24 packets.
- [x] Freeze activation, cycle, target, update, synchronization/allocation,
      residual, and exact reduction needed for each candidate cell; detailed
      resident-transition subowners remain an explicit instrumentation follow-up.
- [x] Publish independent compact checkpoint artifact and commit.

Exit: current-source common attribution, not historical mixed timing, chooses the
next implementation owner in each lane.

**Retained W7900 checkpoint:**
[`2026-08-25-w7900-specdec2-perf-p1-p3-checkpoint.json`](../benchmarks/results/2026-08-25-w7900-specdec2-perf-p1-p3-checkpoint.json),
[`dense P3 stable slabs`](../benchmarks/results/2026-08-25-w7900-specdec2-perf-p3-dense-stable-slabs.json),
[`packed P4 device candidate`](../benchmarks/results/2026-08-25-w7900-specdec2-perf-p4-paro-device-candidate.json),
[`corrected dense P4`](../benchmarks/results/2026-08-25-w7900-specdec2-perf-p4-dense-device-chain-retained.json),
[`P5 target profiles`](../benchmarks/results/2026-08-25-w7900-specdec2-perf-p5-target-profiles.json),
and [`campaign closure`](../benchmarks/results/2026-08-25-w7900-specdec2-perf-campaign-closure.json).
Corrected dense P4 is exact at `1.272x/1.407x/1.439x` AR for K1/K2/K3 but
remains behind direct; p128/p512 is exact but slow and p4K/p16K is pre-mutation K0.
Packed production device candidate is exact with 372/372 zero-allocation cycles
and improves to `0.979x` AR, still below product promotion. Dense P3 stable slabs
are exact/wall-neutral; request-local graph
first use remains P4. No product cell promotes; both automatic policies remain K0.

## 8. P2 — streaming prompt activation

### Dense GGUF

- [x] Retained shared `TargetHiddenChunkSink`, `_StreamingNextNPromptSink`, and
      `Qwen35GGUFNextNExecutor.enqueue_prompt_rows()` primitives exist.
- [ ] RED-test their staged one-shot/chunked shifted NextN prompt state at chunk
      sizes 1/2/7/8/9, ragged tails, offsets, page transitions, cancellation,
      prefix/COW, and pressure rejection.
- [x] Wire retained target hidden chunk ownership into staged provider priming.
- [x] Retain one carried BF16 hidden row only.
- [x] Remove staged `_prompt_hidden_rows` and post-prefill host replay for the
      qualified streaming capability; keep it only as the documented oracle fallback.
- [x] Prove O(hidden), not O(prompt×hidden), transient ownership: every retained
      request carries one 10,240-byte row.
- [ ] Run p128/p512/p4K/p16K activation and full category bridge exactness,
      state/KV/following generation, wall, and drain gates.  W7900 native target
      graph/arithmetic is now fail-closed above 95 live tokens after deterministic p128
      NaN/sentinel; streaming activation continues through eager/serial fallback
      while p4K/p16K stay pre-mutation K0 above the 1023 capability cap.

### Packed PARO

- [x] Streaming final-normalized BF16 priming exists with one carried 10 KiB row.
- [x] The staged C1 owner now consumes final-layer rows from the same compact
      packed target prefill as true AR, normalizes through the one-row BF16
      capture, and primes NextN without serial target replay.  The focused
      strict `general_ja_explain` D25 gate is exact after the earlier serial /
      packed activation paths diverged at generated-token index 11.
- [ ] Revalidate chunk/tail/cancel/prefix/pressure lifetime and activation wall
      under the common bridge; do not reimplement the sink without a failure.

## 9. P3 — stable slabs and zero hot allocation

- [x] Dense direct N1R/N2/N3P already provide fixed-address graph scratch,
      stable target accept/results, graph generations, and bounded readback;
      shared C2/C4 staged source already has persistent `_batch_accept_resources`.
- [x] Add common bridge per-cycle allocation/free byte/count telemetry for
      target/accept/commit; remove the temporary packed-only diagnostic flag
      after P4 qualification. Cached ROCTX/HIP remains the authoritative
      library-internal allocation and API view.
- [ ] RED-test pointer reuse and generation invalidation across close/reuse,
      shrink/refill, compaction, prefix restore, pressure, fallback, and failure.
- [x] Pre-reserve packed-PARO C1 R2 linear/MoE verifier scratch plus the
      production `decode_batched` full-attention/MoE scratch after target /
      provider prompt priming and before the first speculative plan/mutation,
      rather than lazily resizing 1,110 strict or 41 production workspace
      allocations in cycle 1.
- [x] Replace GGUF cycle-local proposal/repair `hidden_batch` allocation/free
      with one claimed provider-group workspace; W7900 K1/K2/K3 wall is neutral
      and remaining allocation is persistent request-local graph first use.
- [ ] Add persistent lane-specific candidate, target result, accept, selected
      commit, provider update, row-map, and bounded result slabs.
- [x] Bucket packed-PARO proposer token/KV/snapshot capacity by a content-agnostic
      power of two with a 256-token floor, so the warmed provider pool does not
      rebuild for every larger prompt in one service-capacity cell.
- [ ] Ensure PARO borrowed target pointers are never owned/freed by provider or
      staged workspace.
- [ ] Confirm no request observes peer scratch after slot reuse.
- [x] Cached strict traces and production/strict full packets prove zero
      allocation/free in warmed cycle markers (372/372 production cycles) and
      exact final conservation.

## 10. P4 — device-resident bounded cycle

### Dense GGUF

- [x] Direct N2 owns device accept/selected target commit and direct N3P owns
      cached proposal→target event retirement with bounded final results.
- [x] Shared physical C2/C4 staged source can carry device candidate IDs into
      packed target lowering; gfx1100 package capability remains false.
- [x] Adapt budget-specific C1 N3P proposal ownership to one bounded staged
      cycle. The first `abed6101d` adapter was rejected on following-AR safety;
      corrected shared pre-capture/commit-table source supersedes it and passes.
- [x] Keep C1 candidates on device into target/accept/selected commit/provider
      repair with zero pre-target candidate D2H; bounded tail fallback remains.
- [x] Preserve canonical checkpoint/repair fingerprints for reject/every
      partial/full acceptance across the complete exact category packet.

### Packed PARO

- [x] Existing explicit N4 supplies gfx1100 target/accept fixed buffers and an
      independently gated selected linear-state commit primitive; it is a
      reusable oracle/primitive, not a promoted current-provider route.
- [x] Replace bounded host-I32 candidate handoff with a stable borrowed-W8A16
      INT32 device descriptor; target consumes it before accept and bounded
      candidate materialization occurs only after target synchronization.
- [ ] Keep target top-1/selected-row/provider-update metadata in stable slabs.
- [ ] Run GPU accept directly and read back only bounded committed IDs/lengths/
      status after selected target/provider commit.
- [ ] Keep CPU accept and strict target as qualification/debug oracles outside
      the promoted cycle.

Both lanes must pass EOS/stop/tail, cancel/deadline, prefix/pressure/compaction,
failure/restart, graph/eager miss, selected state/KV, following-AR, profiler
engagement, full category exact/profile quality, and final ownership gates.

## 11. P5-P7 — profile, one candidate, conditional repair

For each lane independently:

- [ ] Profile every currently admitted complete target row bucket with ROCTX,
      kernel, HIP API, copies, allocations, queue gaps, resources, and exact
      physical decomposition. Packed production/strict R2 is complete; dense
      R2/R3/R4 and future physical buckets remain.
- [x] Reconcile packed R2 marker wall, kernel-family interval union, API/copies,
      and complete cycle. One post-commit sync/cycle is removed, but operation
      wall is neutral because accept readback absorbs queued work.
- [x] Admit at most one target candidate with a named operation-complete owner,
      RED oracle, strict fallback, exact C/K/R/context/profile scope, and either
      `>=1.10x` projected operation speed plus `>=1%` request saving or enough
      projected saving to cross one automatic cell. No dense/packed target
      kernel candidate meets this admission rule.
- [x] Retain/reject the candidate on operation-complete plus complete-category
      evidence; launch count alone is insufficient. Packed post-commit stream
      ordering is retained mechanically but wall-neutral; dense device chaining
      is rejected on state safety.
- [x] Reprofile provider repair only after target/device changes; admit it under
      the same materiality rule or publish no-go. Dense repair is 4.95-6.56
      ms/request and remains no-go without a state-safe device-chain premise.

GGUF required logical target rows are R2/R3/R4 at C1 and, after physical
admission, R4/R6/R8/R12/R16. PARO begins at R2 and adds R4/R8 only with genuine
request-major C2/C4 ownership.

## 12. P8 — profile completion

### Dense GGUF

- [x] Preserve strict FP32-state manifest and exact route.
- [x] Qualify a production profile only if an independently measured product
      arithmetic candidate exists; none exists, so production reports
      fail-closed strict fallback.
- [ ] Any T1/T2 candidate runs strict-teacher mean/p95/p99/max KL, top-1 by
      category/shape/transition, three same-schedule repeats, neighbor isolation,
      state/KV ownership, BF16-relative where applicable, tasks, lifecycle,
      profiler selected/fallback variants, and production-AR economics.

### Packed PARO

- [x] Direct B1 production T2 and strict manifests are qualified.
- [x] Prove staged execution resolves and reports those exact manifests and does
      not introduce an unclassified arithmetic/route combination.
- [x] Link staged production rows to the complete direct numerical/task/state
      packet; P2-P5 ownership changes are T0/control-only and exact full packets
      rerun every changed surface.

## 13. P9 — physical C2/C4 and product policy

### Dense GGUF

- [x] Merged shared source already implements physical proposal/target, device
      candidate handoff, group accept, and selected group commit for C2/C4; only
      gfx1151 evidence exists and gfx1100 exposure remains false.
- [ ] Qualify and enable C2/C4 only after W7900 proposal/target backbone counts,
      per-request accept/selected state/KV, refill, cancellation, compaction,
      prefix COW, pressure, recovery, profiler, and drain gates pass.
- [ ] Cover K1-K3 and R4/R6/R8/R12/R16 with local decomposition profiles.

### Packed PARO

- [x] Retained gfx1100 direct selected-batch C2 target execution is an AR/target
      oracle only; public c4/c8 target-owner symmetry and all staged MTP C>1
      ownership remain open.
- [ ] Implement request-major C2/C4 K1 proposal, R4/R8 target, per-request
      `KVLiveSpans`, parent-indexed Conv/GDN, one group accept, independent
      selected commit, and cross-request isolation.
- [ ] Do not expose C2/C4 from singleton loops or transfer dense row kernels.

For both lanes:

- [x] Re-run fixed C1 cells before adaptive policy; no adaptive policy is admitted.
- [x] Fingerprint the local table and prerequisite artifacts in the closure JSON.
- [x] Keep every losing/unqualified cell K0 before mutation with exact reason.
- [x] Run applicable train/heldout/category/context/lifecycle/memory gates. No
      automatic cell is proposed, so serving/SLO/load qualification is not
      triggered; existing Generation-2 service ownership evidence remains frozen.

## 14. P10 — closure

- [x] List retained units, promoted cells, explicit/default-off cells, and every
      K0/rejected/no-go cell with durable reason in the closure artifact.
- [x] Remove the temporary packed allocation flag; retain remaining replay/
      backup/oracle debt in `REFACTOR.md` until its exact removal gate passes.
- [x] Update plans/status, profile docs, compact artifacts, benchmark rollup/
      changelog, and immutable worklogs. No new kernel body/lineage changed.
- [x] Do not export the public root README because no product scope promotes.
- [x] Run milestone tests plus Worklog2, benchmark sync, registry, JSON, and diff
      checks; broad isolated failures use the repository focused-repair rule.
- [x] Commit, sync origin, push, and verify local/remote equality.

### Post-closure Generation-2 recovery checkpoint — 2026-08-26

A current p512/d128 audit finds no raw AR-rate regression versus the old fixed-C
server: exact C1 is `77.176` blocking / `76.925` SSE tok/s versus `72.169`, and
native C8 raw wall is `161.882` versus `158.542`. The C8 rate is not strict-
eligible: varied D128 IDs differ from C1, and hidden drift begins at decode step
1/layer 0 under both eager and graph submission. Treat AR profile qualification
separately from scheduler performance; do not attribute this to CONCURRENCY2
host overhead or publish the raw C8 rate as strict.

Physical gfx1100 C2 remains unexposed, but its acceptance blocker is closed.
Differential tracing found packed provider cursor metadata one token ahead and
physical selected commit missing the pre-output-norm BF16 target hidden row.
After repair, full-suite D24 K2 is exact at `260/338 = 76.92%` draft acceptance
versus the prior physical `18.43%`; zero candidate D2H/recovery remains. It is
still only `16.974 vs 31.230 tok/s = 0.544x` AR because R6 target/accept costs
207.8 ms/group. Break-even is approximately <=101 ms after proposal, so capability and
automatic policy remain false/K0.

C1 attribution also changes the tuning order: sampled K2 staged decode is
`416.6 ms` versus direct `433.5 ms`, while staged target prefill plus NextN
priming is `339.3 + 25.3 ms` versus direct prefill `305.9 ms`. Activation and
shared prompt/provider ownership rank ahead of more hot target-leaf tuning.
Evidence: [`recovery profile`](../benchmarks/results/2026-08-26-w7900-mtp-concurrency2-recovery-profile.json)
and [`C2 root cause`](../benchmarks/results/2026-08-26-w7900-specdec2-c2-acceptance-root-cause.json).

## 15. Stop and no-chase rules

- Stop on ownership, rollback, state/KV, isolation, determinism, or manifest
  failure before timing work.
- Stop and add a numerical oracle before changing arithmetic.
- Do not infer C2/C4 from request-serial singleton loops.
- Do not tune accept alone while target/device boundaries dominate.
- Do not add adaptive K before fixed cells have complete evidence.
- Do not begin C8, deeper PARO drafts, arbitrary trees, overlap, or multi-cycle
  device loops without a winning fixed-cell premise and current profile.
- Do not capture graphs before pointers are stable and trace evidence names a
  recoverable operation-complete gap.
- Do not tune attention or generic AR/MoE kernels without a current complete
  target profile naming that family.
- Do not condition policy on prompt, token, category, or heldout identity.
- Do not compare W7900 performance against gfx1151 or XTX absolute rates.
- Publish negative evidence and stop a mechanism unless a materially new
  representation/dataflow premise reopens it.

## 16. Expected code and artifact map

| Path | Role |
| --- | --- |
| `docs/SPECDEC2-PERF-GFX1100.md` | This independent ledger. |
| `docs/SPECDEC2-GFX1100.md` | S7 capability/status pointer. |
| `scripts/specdec2_perf_gfx1100_bridge.py` | Common lane-aware bridge and artifact writer. |
| `scripts/specdec2_perf_gfx1100_profile_child.py` | Final cached marked profiler child if not a bridge mode. |
| `hipengine/generation/qwen35_gguf_mtp2.py` | Dense streaming, slabs, device-chain, and repair ownership. |
| `hipengine/generation/qwen35_paro_mtp2.py` | Packed slabs, device-chain, staged profile ownership. |
| `hipengine/runtime/qwen35_gguf_nextn.py` | Dense exact proposal/device descriptors only where provider-owned. |
| `hipengine/runtime/qwen35_gguf_mtp.py` | Dense target/accept/selected-commit oracle and descriptors. |
| `hipengine/speculative/mtp_native.py` | PARO provider/device descriptors without borrowed-pointer ownership drift. |
| `hipengine/runtime/qwen35_paro_runner.py` | PARO physical target and selected-state ownership. |
| `hipengine/kernels/hip_gfx1100/` | Only P5-admitted in-tree primitives. |
| `tests/test_specdec2_perf_gfx1100_bridge.py` | Schema/counterbalance/provenance/timing RED tests. |
| `tests/test_qwen35_gguf_mtp2_seam.py` | Dense integration and lifecycle RED gates. |
| `tests/test_qwen35_paro_mtp2_seam.py` | Packed integration and lifecycle RED gates. |
| `benchmarks/results/` | Compact retained/rejected/no-go artifacts. |

Kernel work first runs `scripts/check_lineage.py`, updates `KERNELS.md` when
ownership changes, retains raw-pointer ABIs, and keeps strict registered
fallbacks. No normal generation path imports torch.
