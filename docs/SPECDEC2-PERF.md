# SPECDEC2-PERF — gfx1151 Activation and Hot-Cycle Campaign

- Status: **approved; implementation handoff ready**
- Approved: **2026-08-25**
- Functional predecessor: [`SPECDEC2.md`](SPECDEC2.md), S1-S6 closed
- Performance owner: **stable physical host `gfx1151` agent**
- Development/retention hardware: **AMD Radeon 8060S Graphics / `hip_gfx1151`**
- Primary artifact: **Qwen3.8-27B `Q4_K_S`, BF16 KV**
- Initial profile: **strict FP32 recurrent state**
- Product profile follow-up: **scoped FP16 recurrent state with FP32 fallback**
- Selected implementation base: **`9070d59ca` or its clean descendant containing the complete S6 merge and this plan**
- Automatic policy at entry: **K0 for C1-C32**
- Explicit functional scope at entry: **strict C1/C2/C4 K1-K3 through R16, default-off**
- Normative dependencies: [`PLAN.md`](PLAN.md), [`SPECDEC2.md`](SPECDEC2.md),
  [`CONCURRENCY2.md`](CONCURRENCY2.md),
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
  [`TESTING.md`](TESTING.md), [`BENCHMARK.md`](BENCHMARK.md), and
  [`KERNELS.md`](KERNELS.md)

This is a focused performance follow-up, not a scheduler rewrite. SPECDEC2 S1-S6
already established the correct target-frontier, provider, transaction,
Generation-2, lifecycle, API, pressure, recovery, and output ownership model.
This campaign activates already-retained exact prompt/proposal machinery inside
that model, removes hot ownership debt, makes the cycle device-resident, and
then profiles the physical target before admitting one kernel/graph candidate.

The ZBook may be used for CPU tests, static review, or correctness bring-up, but
it does **not** produce keep/revert wall evidence for this campaign. Every
performance decision is measured on the stable `gfx1151` host. ZBook and stable
`gfx1151` absolute rates are independent lanes and are never presented as an
old-to-new comparison.

## 1. Executive decision

Proceed in this order:

1. establish common timing boundaries for old native, staged SPECDEC2, and true
   AR;
2. integrate retained exact OI-3 streaming NextN prompt priming;
3. remove post-warmup allocation/free and stabilize all cycle pointers;
4. chain proposal → target top-1 → GPU accept → selected commit on device;
5. profile R6/R8/R12/R16 and admit at most one physical target candidate;
6. optimize provider repair only if it remains material;
7. qualify the product FP16-state profile against FP16 AR;
8. rebuild deterministic K policy cells; and
9. run product gates only for cells that already beat true AR.

Do **not** begin with C8, adaptive K, deeper drafts, multi-stream overlap, or
accept-kernel micro-tuning. Fixed C1/C2/C4 cells must first become economically
competitive under the complete protocol.

## 2. Campaign goal and definition of done

### 2.1 Goal

Make at least one honest gfx1151 SPECDEC2 product cell faster than the same-host,
same-model, same-profile true AR path while preserving all S6 correctness and
ownership contracts. The preferred project result is >1.30x true AR; automatic
promotion requires >1.10x plus every binding quality, SLO, memory, lifecycle,
and isolation gate.

### 2.2 Successful closure

The campaign closes successfully when all of the following have durable
verdicts, even if no automatic cell promotes:

1. activation and steady-cycle walls are independently measured;
2. true AR, old native exact MTP, and staged SPECDEC2 use common timing
   boundaries on current source;
3. streaming prompt priming is integrated or explicitly rejected under the
   common protocol;
4. qualified hot cells have zero cycle-local allocation/free after warmup;
5. candidate and target IDs remain device-resident through accept/selected
   commit in every promoted device-chain cell;
6. R6/R8/R12/R16 target ownership is profiled with complete family and queue-gap
   accounting;
7. every admitted target candidate is retained or rejected by operation-complete
   plus complete-request evidence;
8. provider repair receives a measured retain/no-go verdict;
9. production FP16-state SPECDEC2 receives a full profile verdict;
10. policy cells are rebuilt from retained evidence only;
11. every automatic cell exceeds 1.10x true AR and passes all binding gates;
12. every losing/unqualified cell selects K0 before mutation with an exact
    reason; and
13. artifacts, benchmark rollup/changelog, worklog, refactor debt, and handoff
    are current.

A zero-promotion closure is valid. It must identify the measured remaining
blocker rather than weakening the baseline or silently restoring the old
whole-request owner.

### 2.3 Out of scope until reopened by evidence

- scheduler/frontier/transaction redesign;
- gfx1100 transfer or policy reuse;
- PARO MTP2, DFlash, DFlash2, EAGLE, n-gram, or remote providers;
- arbitrary trees or sampled speculative decoding;
- C8/R24/R32 native ownership without a winning fixed-cell premise;
- multi-cycle device-resident generation loops;
- proposal/target multi-stream overlap before stable pointers and a trace-proven
  queue gap;
- prompt-, token-, category-, or heldout-conditioned policy;
- speculative quality trade-offs disguised as strict optimization; and
- unrelated AR, MoE, attention, KV-codec, or server-SLO campaigns.

## 3. Frozen starting evidence

### 3.1 Functional closure

[`2026-08-25-gfx1151-specdec2-s6-product-closure.json`](../benchmarks/results/2026-08-25-gfx1151-specdec2-s6-product-closure.json)
proves strict C1/C2/C4 K1-K3 API, category, lifecycle, pressure, recovery,
streaming, and soak correctness. Automatic C1-C32 selects K0. The full ten-prompt
counterbalanced K2 wall is 2.1211x/2.3117x true AR at C2/C4 with 49.15% draft
acceptance. No prior S1-S6 correctness task is repeated unless this campaign
changes its owned surface.

### 3.2 Short fixed-cell economics

The short fixed prompt is an implementation diagnostic only, never a
promotion suite. Current K2 rows are:

| C | True AR tok/s | K2 MTP tok/s | MTP/AR wall | Status |
| ---: | ---: | ---: | ---: | --- |
| 1 | 7.025 | 6.876 | 1.022x | approximately parity, not promoted |
| 2 | 11.574 | 9.500 | 1.218x | 21.8% slower |
| 4 | 21.585 | 16.053 | 1.345x | 34.5% slower |

Physical MTP scales 2.33x from C1 to C4, but AR scales 3.07x. The physical
frontier is real: R16 target wall is 479.945 ms versus 816.891 ms for two R8
sweeps, a 1.702x target improvement. The issue is absolute cycle and activation
cost, not fake batching or scheduler correctness.

Sources:

- [`S4 physical C2/C4`](../benchmarks/results/2026-08-25-gfx1151-specdec2-s4-physical-c2-c4.json)
- [`S5 cost/policy`](../benchmarks/results/2026-08-25-gfx1151-specdec2-s5-cost-policy.json)

### 3.3 C2/K2 attributed wall

| Component | Current measured wall | Interpretation |
| --- | ---: | --- |
| target verification | 392.8 ms | dominant explicit cycle owner |
| proposal | 32.3 ms | physical C2 proposal; allocation outside timer is possible |
| provider repair | 7.2 ms | canonical checkpoint/replay work |
| accept/oracle + selected commit + candidate readback | ~1.0 ms | small direct timer, but host boundary blocks stable chaining |
| complete warm request | 1053.5 ms | includes activation, owner transitions, allocations, synchronization, and output lifecycle |

The target is >90% of the explicitly attributed proposal+target+repair work.
The difference between their sum and complete wall is not automatically
"scheduler overhead": prompt activation, target/NextN prefill, allocation,
synchronization, HTTP/service ownership, and more than one cycle can contribute.
The common bridge must label them before optimization.

### 3.4 Required total-wall reductions

For automatic promotion, MTP wall must be at most `AR wall / 1.10`.

| Cell | Current MTP/AR wall | Minimum MTP-wall reduction to 1.10x speedup |
| --- | ---: | ---: |
| short C2/K2 | ~1.221x | **25.5%** |
| short C4/K2 | ~1.352x | **32.8%** |
| full-suite C2/K2 | 2.121x | **57.1%** |
| full-suite C4/K2 | 2.312x | **60.7%** |

The short→full gap proves kernel tuning alone cannot close the campaign.
Activation and repeated partial-accept cycle economics are binding.

### 3.5 Older native control is diagnostic, not an automatic baseline

The 2026-08-17 exact native Q4_K_S B3 packet reports 24.193 tok/s, 1.823x its
true AR, 64.92% draft acceptance, and roughly 111 ms target wall per cycle over
25 visible outputs. It is useful as an execution-efficiency control, but it is
not directly comparable to S6 complete HTTP C2/C4 max-8 wall:

- concurrency and output horizon differ;
- decode-only and complete-request timing differ;
- target/provider prompt activation differs;
- current staged execution has stronger provider checkpoint/repair ownership;
- current source and retained kernel routes differ; and
- legacy direct provider fingerprints are not canonical after every rejected
  suffix.

The bridge phase must rerun old native and staged routes on the same current
commit and timing boundary. Do not promise to recover a stale absolute rate.

Source:
[`exact native B3`](../benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json).

## 4. Non-negotiable invariants

All S6 invariants remain binding. Performance work adds these:

1. **Generation-2 remains the owner.** `EngineService`/`ResidentEngineLoop` own
   admission, fairness, visible output, cancellation, and reclaim. A fast bundle
   executes one bounded cycle and returns; it never restores a request-life
   model loop.
2. **One target+provider transaction per cycle.** Proposal, target, accept,
   selected target/provider commit, output, and rollback share one operation and
   generation identity.
3. **Canonical provider state.** Reject/partial/full acceptance leaves NextN KV,
   cursor, hidden seed, and following proposal equivalent to the declared exact
   provider contract. Legacy rejected-suffix drift is not a speed feature.
4. **Stable identity, movable slots.** Request ID, resident slot, physical row,
   frontier row, and provider row remain distinct through compaction and refill.
5. **Device residency is explicit.** A promoted device chain performs no full
   candidate or target-ID host reconstruction before accept/selected commit.
   Only a bounded final committed-token/status payload is read back.
6. **No hot allocation.** Qualified warmed cells perform zero `malloc/free`,
   graph create/destroy, or hidden lazy library build inside a cycle. Claims own
   all slabs and generation changes invalidate them safely.
7. **Graph keys describe execution.** Keys include model/artifact, backend,
   quant, KV, profile manifest, C/K/R, decomposition, context/page bucket,
   transaction modes, pointer generation, and variant manifest.
8. **Strict fallback is registered and tested.** Every graph/fused/production
   path has a qualified eager/unfused/FP32 fallback selected before mutation.
9. **Control correctness is exact in every profile.** Request/row/position,
   state/KV, acceptance, commit, rollback, output, and resource conservation
   never become numerical tolerances.
10. **Profile arithmetic is honest.** Strict stays exact. FP16 production uses
    declared FP32 accumulation/scratch and passes its calibrated distribution,
    determinism, isolation, BF16-relative where applicable, and task gates.
11. **No benchmark gaming.** Policy may use concurrency, remaining horizon,
    context/profile, memory fit, and a predeclared online acceptance statistic.
    It may not inspect prompt text, token IDs, category, or heldout identity.
12. **Performance uses true AR.** Verifier-derived B0/off telemetry is
    diagnostic only.
13. **Same-host evidence only.** Stable `gfx1151` retention rows compare both
    arms under one command/session protocol. No ZBook or older-host rate is an
    old/new denominator.
14. **Launch reduction is not presumed useful.** Graph/PM4 work requires an
    operation-complete marker premise; launch count alone is not recoverable
    wall.

## 5. Stable hardware and source protocol

### 5.1 Worktree and provenance

The gfx1151 owner begins from the committed handoff:

```bash
git fetch origin main
git worktree add /home/lhl/hipEngine-specdec2-perf \
  -b specdec2-perf origin/main
cd /home/lhl/hipEngine-specdec2-perf
git status -sb
```

If that branch/worktree already exists, inspect it; never force-delete another
owner's work. Record:

- exact `HEAD`, merge base, tracked-clean state, and explicit shared untracked
  exclusions;
- physical hostname, GPU, architecture, power mode, ROCm/driver, and
  `hipcc --version`;
- full or sampled model hash with method, file size, GGUF metadata, quant, KV,
  and NextN tensor identity;
- prompt fixture SHA-256;
- execution/strict fallback variant manifests; and
- exact environment including `GPU_MAX_HW_QUEUES=2`.

### 5.2 GPU preflight

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
rocminfo | grep -E 'Name:|gfx'
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Before profiling, prebuild outside `rocprofv3`, write one compiler-version file,
and run children with `HIPENGINE_REQUIRE_CACHED_BUILD=1`. Do not profile a
parent harness that launches nested children.

### 5.3 GPU lease

The gfx1151 campaign owner holds the stable GPU lease during every timing or
profiler packet. No other model load, benchmark, profiler, or power-changing
work runs concurrently. Before a >5-minute command, state command/purpose,
expected duration, lock identity, stop budget, and output path. Use a background
task and durable exit notification; do not poll in a foreground loop.

## 6. Common measurement contract

### 6.1 Arms

The committed bridge harness (planned path
`scripts/specdec2_perf_bridge.py`) must support:

- `true_ar`: no provider, no speculative mutation;
- `legacy_native`: current-source old native complete-cycle control;
- `specdec2`: staged Generation-2 path;
- optional candidate/control labels for one scoped optimization.

The harness must not silently substitute the legacy whole-request route for
staged execution. Every output reports realized route and physical shape.

### 6.2 Workloads

Required bridge cells:

| Packet | C | K | Outputs | Purpose |
| --- | --- | --- | ---: | --- |
| short activation | 1/2/4 | 2, plus K1/K3 diagnostic | 5/8 | reproduce S4/S6 and split activation from cycle cost |
| natural common bridge | 1 | 1/2/3 | 25 | compare old native/staged/AR under old topline horizon |
| physical bridge | 2/4 | 2/3 | 25 | post-activation fixed-cell economics |
| output horizon | retained candidate C/K | — | 8/25/64/128 | deterministic horizon policy premise |
| prompt length | retained candidate C/K | — | p128/p512/p4K/p16K | activation amortization and context route |
| product | promoted candidates only | policy | protocol-specific | SLO/load/soak gate |

Acceptance and speed keep/revert decisions always run the complete ten-prompt
`benchmarks/prompts/mtpbench-code-general-ja.jsonl` suite with the six-train,
four-heldout split and all four categories. Fixed greeting rows admit mechanics
only.

### 6.3 Timing windows

Each arm reports both:

1. **complete request wall** — tokenize/admission, target prompt activation,
   provider prompt activation, decode cycles, output publication, and terminal
   reclaim; and
2. **decode-only wall** — after target/provider activation and graph warmup,
   through the last committed decode transition, excluding teardown.

Additional non-overlapping or explicitly nested stages:

```text
tokenize
admission
claims_reserve
target_prefill
nextn_prompt_prime
provider_open
cycle_total
  proposal
  frontier_lower
  target_verify
  candidate_or_target_readback
  accept_device
  accept_cpu_oracle_debug
  selected_target_commit
  provider_repair
  bounded_result_readback
  output_publish
claims_release
terminal_reclaim
```

Host timers alone are insufficient when asynchronous kernels cross a boundary.
The profile child emits ROCTX stage markers and the artifact records marker wall,
kernel interval union, queue/API synchronization, H2D/D2H/D2D, allocation/free,
and residual separately. Residual is not labeled recoverable without an A/B.

### 6.4 Counterbalancing and repeats

- Alternate AR→MTP and MTP→AR by prompt index without consulting prompt
  content.
- For retained performance, use at least three complete paired repetitions or
  the campaign's predeclared stronger packet.
- Report samples, median, spread/CV, order, warm/cold state, and thermal/power
  telemetry available on the stable host.
- One-run kernel screens may reject obvious losses but cannot promote.

### 6.5 Correctness and ownership payload

Every bridge row records:

- exact generated IDs and counts from response-owned accounting;
- prompt IDs/hash/count, not decoded-text re-tokenization;
- candidate IDs/counts and accepted depth per cycle;
- GPU/CPU accept agreement in qualification;
- target/provider positions, visible KV hashes, selected hidden/state hashes,
  and following-AR continuity at focused gates;
- physical proposal widths and target R decomposition;
- graph/eager route, graph key/generation, fallback reason;
- request/slot/row maps and transaction IDs;
- allocation high-water and final zero ownership; and
- strict/selected/fallback manifest hashes.

## 7. Candidate admission and keep/revert policy

### 7.1 General admission

A runtime/kernel candidate begins only when the latest stable-host profile has:

- a named operation-complete owner;
- a correctness oracle;
- a registered strict fallback or a documented no-arithmetic control;
- a mechanism that is not already in the do-not-repeat ledger; and
- either >=1.10x operation-complete projected speedup with >=1% complete-request
  saving, or enough projected saving to cross one predeclared automatic-cell
  threshold.

Tiny measured cycle-wall, verified sub-window, sync, or transfer wins remain
first-class when exact and same-suite non-regressive, but they do not authorize
an automatic cell until complete economics pass.

### 7.2 Retention

Retain and default a qualified cell when:

- every binding correctness/profile/category/task gate passes;
- operation-complete and complete-wall measurements are non-regressive;
- every category is non-regressive;
- true AR speedup is >1.10x for automatic policy;
- TTFT/ITL/E2E, pressure, memory, cancellation, and neighbor SLOs pass; and
- fallback/rollback remains available.

### 7.3 Rejection

Reject/revert when:

- exact ownership, provider fingerprint, or profile gate fails;
- only launch count improves;
- a microbenchmark does not translate to operation-complete wall;
- train improves while heldout/category regresses;
- target speed improves but complete request regresses beyond noise;
- policy needs prompt/token/category knowledge;
- allocation, graph, or provider generations leak; or
- the result cannot plausibly advance a declared cell.

Publish negative evidence and stop that mechanism. Do not tweak a near-miss
without a materially new representation/dataflow premise.

## 8. Phase graph

```text
P0 approve/freeze handoff (this document)
  -> P1 common bridge + stable baseline
    -> P2 streaming prompt activation
      -> P3 stable slabs / zero hot allocation
        -> P4 device-resident cycle boundary
          -> P5 physical target profile
            -> P6 one admitted target candidate
              -> P7 conditional provider repair
                -> P8 production FP16-state capability
                  -> P9 policy + product qualification
                    -> P10 closure
```

A downstream phase never starts with failing upstream tests or an uncommitted
logical unit. Each phase gets a unique immutable worklog entry and immediate
commit after validation.

## 9. P0 — campaign handoff

Owner: current planning session. No GPU run.

- [x] Approve a separate SPECDEC2-PERF campaign rather than reopening S6.
- [x] Select stable physical `gfx1151` for all performance retention.
- [x] Preserve SPECDEC2 architecture; reject rewrite scope.
- [x] Freeze current evidence, reduction requirements, priorities, and no-chase
      list.
- [x] Create the detailed phase/punchlist and structured TaskList handoff.
- [x] Name OI-3 streaming prompt priming as the first implementation candidate.
- [x] Defer gfx1100 S7 while this user-approved gfx1151 performance follow-up is
      active.
- [x] Commit and push this plan/worklog/pointer unit.

Exit: committed handoff on `origin/main`, no tracked local changes, and tasks
P1-P10 owned by `gfx1151-agent`.

## 10. P1 — common bridge and current-main baseline

TaskList: #24-#25. Expected stable-GPU cost: 30-90 minutes after harness tests.

### P1.1 Harness

- [ ] Add `scripts/specdec2_perf_bridge.py` or an equivalently focused committed
      harness; do not grow the S6 closure harness into an unreviewable monolith.
- [ ] Add schema/aggregation/counterbalance unit tests.
- [ ] Support AR, legacy native, and staged arms in one loaded process where
      ownership permits; report unavoidable reloads explicitly.
- [ ] Support C1/C2/C4, K1-K3, output horizon, prompt limit, repeat count, and
      train/full scopes.
- [ ] Emit immediate progress and atomic checkpoints after each prompt/arm.
- [ ] Emit complete and decode-only timing, stage attribution, exact accounting,
      physical shapes, candidate/acceptance trajectories, provider fingerprints,
      allocation/synchronization counters, and teardown.
- [ ] Add a final single-child profiler mode with cached builds and ROCTX markers.
- [ ] RED-test malformed/missing timing owners, duplicated batch timing,
      incomplete prompt suite, invalid AR denominator, and dirty provenance.

### P1.2 Stable baseline

- [ ] Create a clean worktree/branch from the handoff commit.
- [ ] Record full platform/model/compiler/prompt/profile provenance.
- [ ] Run C1 K1-K3 natural25 counterbalanced AR/native/staged packet.
- [ ] Run C2/C4 K2 natural25 and short activation packet; K3 follows only as a
      fixed-depth diagnostic.
- [ ] Run one cached staged C2/K2 and C4/K2 marked/kernel trace.
- [ ] Verify old-native/staged outputs and acceptance separately; do not require
      legacy provider fingerprints to equal the canonical staged contract.
- [ ] Freeze current activation, steady-cycle, target, repair, synchronization,
      and residual ceilings.
- [ ] Recalculate the exact reduction needed for each candidate cell.
- [ ] Publish `benchmarks/results/<date>-gfx1151-specdec2-perf-p1-bridge.json`.
- [ ] Update campaign status/worklog and commit.

Suggested command shape after harness implementation:

```bash
env HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1151 \
  GPU_MAX_HW_QUEUES=2 HIPENGINE_REQUIRE_CACHED_BUILD=1 \
  HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
  .venv/bin/python scripts/specdec2_perf_bridge.py \
  --model /models/gguf/Qwen3.8-27B-Q4_K_S.gguf \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --concurrency 1,2,4 --budgets 1,2,3 \
  --max-tokens 25 --runs 3 --counterbalanced \
  --execution-profile strict --require-cached-build \
  --output /tmp/specdec2-perf-p1-bridge.json --fail-on-fail
```

Exact CLI is owned by P1 and must be recorded in its artifact.

Exit: common-protocol attribution identifies activation and steady-cycle costs;
P2 is admitted without relying on mixed historical timing.

## 11. P2 — streaming exact NextN prompt activation

TaskList: #26-#27. Source mechanism: retained commit `51e990ee8`.

### P2.1 RED contracts

- [ ] One-shot and chunked target prefill produce identical shifted NextN KV and
      cursor (`token[0]+zero`, `token[i>0]+target_hidden[i-1]`).
- [ ] Chunk sizes 1/2/7/8/9, ragged C2/C4 tails, warm offsets, and page/context
      transitions are exact.
- [ ] Cancel before/mid/after activation leaves target/provider owners terminal
      and reusable.
- [ ] Prefix reuse/COW and pressure rejection do not publish or leak provider
      state.
- [ ] Target hidden chunk source remains live until NextN consumption completes.
- [ ] No prompt-sized host hidden slab is allocated or retained.

### P2.2 Implementation

- [ ] Wire `TargetHiddenChunkSink` and `_StreamingNextNPromptSink` into the
      Generation-2 target prefill owner before provider catch-up is needed.
- [ ] Feed `Qwen35GGUFNextNExecutor.enqueue_prompt_rows()` on the target prefill
      stream as chunks complete.
- [ ] Retain one request-owned BF16 carried hidden row only.
- [ ] Remove SPECDEC2 `_prompt_hidden_rows` and post-prefill host F32→BF16/H2D
      replay when the streaming capability is selected.
- [ ] Decouple provider activation capacity from the short 1023-token reusable
      target-graph cap. Longer activation may qualify strict eager/K0 routing,
      but it does not auto-admit an unqualified long target graph.
- [ ] Preserve an explicit strict fallback for unsupported prefill ownership.
- [ ] Keep provider/resource claims complete before mutation; no hidden lazy
      allocation.
- [ ] Update `REFACTOR.md` for any temporary migration flag/path.

### P2.3 Gate

- [ ] Focused tests and CPU deterministic bundle pass.
- [ ] p128/p512/p4K/p16K activation first token, cursor, NextN KV, target state,
      and following generation are exact.
- [ ] Stable-host counterbalanced activation wall is non-regressive at every
      required length; report OI-3 historical 12.6-20.8% only as context.
- [ ] Full ten-prompt bridge is exact and every category non-regressive.
- [ ] Tracked prompt transient is O(hidden), not O(prompt×hidden).
- [ ] Final allocations/claims/provider states are zero.
- [ ] Publish artifact/rollup/changelog/worklog and commit retained or rejected
      unit.

Exit: prompt activation is no longer a post-prefill full-prompt replay on every
qualified SPECDEC2 request.

## 12. P3 — stable slabs and zero hot allocation

TaskList: #28-#29.

Current review found cycle-local `malloc/free` around proposal/repair hidden
batches and workspace creation. Allocation/copy/free outside current stage timers
must be measured; `hipFree` may synchronize the device.

### P3.1 RED and instrumentation

- [ ] Add per-cycle allocation/free bytes/counts and explicit synchronize/API
      counters without perturbing production timing when disabled.
- [ ] RED-test pointer reuse across warm cycles, pointer-generation invalidation,
      request close, group shrink/refill, compaction, prefix restore, pressure,
      graph miss, and injected failure.
- [ ] Assert zero hot allocation/free after warmup for qualified C1/C2/C4 cells.
- [ ] Assert one request cannot observe peer scratch after row/slot reuse.

### P3.2 Implementation

- [ ] Replace proposal/repair `hidden_batch = malloc(...); free(...)` with
      provider-group `RuntimeWorkspace` slabs.
- [ ] Persist acceptance/result/remaining-decode and row-map buffers under
      complete resource claims.
- [ ] Build stable pointer tables for candidate, target top-1, accept, selected
      commit, provider repair, and bounded result payloads.
- [ ] Allocate graph-compatible maximum qualified C/K/R shapes or declared
      buckets; do not hide reallocations.
- [ ] Close/invalidate all slabs on provider generation, model/profile change,
      pressure teardown, fatal cycle failure, and engine close.
- [ ] Keep strict eager path independently usable.

### P3.3 Gate

- [ ] Reject/every-partial/full provider fingerprints and following AR pass.
- [ ] Lifecycle/pressure/prefix/cancel/recovery/compaction tests pass.
- [ ] Cached trace confirms no `hipMalloc`/`hipFree` in warmed cycle markers.
- [ ] Complete bridge is exact and same-suite non-regressive.
- [ ] Zero final ownership and exact allocation/free conservation pass.
- [ ] Publish artifact/worklog and commit.

Exit: every later graph/device-chain candidate can rely on stable claimed
addresses.

## 13. P4 — device-resident proposal → target → accept → commit

TaskList: #30-#31.

### P4.1 C1 fast-cycle bridge

- [ ] Enable budget-specific proposal graph ownership instead of unconditional
      `allow_graph=False` where the exact capability admits it.
- [ ] Adapt the existing N2 target/accept/selected-commit graph into one bounded
      staged cycle result; do not call the old whole-request generator.
- [ ] Pass proposal tokens to target through a device descriptor without host
      candidate materialization.
- [ ] Preserve canonical provider checkpoint/repair semantics; compare provider
      fingerprint after reject/every-partial/full.
- [ ] Return to `ResidentEngineLoop` after one committed/rolled-back cycle.

### P4.2 Physical C2/C4 device result

- [ ] Target verifier writes compact device top-1/result rows into stable slabs.
- [ ] Candidate and target IDs feed GPU acceptance directly.
- [ ] GPU accept payload selects target hidden/Conv/GDN/KV rows and provider
      commit metadata per request.
- [ ] Host reads only bounded committed token IDs/lengths/status after selected
      commit.
- [ ] CPU acceptance remains a strict/debug oracle controlled outside the
      promoted cycle; it is not a permanent production synchronization.
- [ ] Device/result descriptors include request/slot/row/transaction generation.

### P4.3 Gate

- [ ] C1/C2/C4 K1-K3 reject/every-partial/full exact gates pass.
- [ ] Candidate/target/GPU accept match CPU oracle in qualification.
- [ ] Selected hidden/Conv/GDN/KV/provider state and following AR are exact.
- [ ] Output tails, EOS/stop, cancel/deadline, prefix/pressure/compaction,
      failure/restart, and graph/eager miss paths pass.
- [ ] Profile shows no pre-accept candidate/target-ID D2H or Python
      `TargetVerifyBatch` reconstruction in the promoted route.
- [ ] Proposal/target/accept/commit named kernels execute with plausible positive
      durations and zero unexpected scratch.
- [ ] Common bridge/full suite is exact and non-regressive by category.
- [ ] Publish artifact/rollup/changelog/worklog and commit.

Exit: synchronization is one bounded final-result boundary, not a sequence of
host materialize/reconstruct/re-upload steps.

## 14. P5 — physical target profile and candidate admission

TaskList: #32. Stable-GPU cost: approximately 15-45 minutes after cache warmup.

### P5.1 Required profiles

- [ ] Mark complete R6 (C2/K2), R8 (C2/K3 or C4/K1), R12 (C4/K2), and R16
      (C4/K3) target windows.
- [ ] Capture kernel, HIP API, memory-copy, and marker traces from final cached
      children.
- [ ] Attribute exact Q4_K_S projection weights/shapes/row routes rather than
      grouping all GEMV by symbol alone.
- [ ] Separate dense projection, Conv/GDN provisional state, attention/KV,
      LM-head/top-1, accept, selected commit, and other.
- [ ] Record call counts, interval union, queue gaps, workgroup/grid, VGPR/SGPR,
      LDS, scratch, and bytes where known.
- [ ] Compare target marker, kernel family sum, and complete cycle; do not infer
      savings from launch count or kernel sum alone.
- [ ] Reconcile R16 versus two R8 under the current post-P4 path.

### P5.2 Candidate ladder

Screen in evidence order, not all at once:

1. exact existing projection route/shape mismatch at R6/R12/R16;
2. stable target graph/PM4 bundle if marker gaps are recoverable;
3. provisional-state slab/layout traffic;
4. selected linear-state commit if operation-complete material; and
5. another representation/dataflow only with a named byte/work premise.

Full attention is not presumed hot. The S4 sampled trace shows packed full
attention tiny and packed GDN modest; the complete profile decides.

### P5.3 Admission

- [ ] Admit at most one candidate meeting the general admission gate.
- [ ] Record expected operation-complete and projected request saving.
- [ ] Name RED oracle, strict fallback, exact C/K/R/context/profile scope, and
      profiler kernel expected.
- [ ] If none qualifies, publish a no-go artifact and skip P6 runtime code.
- [ ] Update `KERNELS.md`/lineage only if dispatch/kernel ownership changes.
- [ ] Commit profile/admission decision before implementation.

## 15. P6 — one physical target optimization

TaskList: #36-#37.

- [ ] Write RED oracle/route test before device changes.
- [ ] Implement only the P5-admitted candidate.
- [ ] Keep C/K/R/context/profile scope explicit through registry/capability;
      engine/model code receives no backend/quant hot branch.
- [ ] Retain strict eager/parent route for every miss.
- [ ] Run primitive/operation exact or declared production numerical gate.
- [ ] Run CPU-reference outer floor and profiler engagement for new kernels.
- [ ] Compare operation-complete target against shipped route with balanced
      samples.
- [ ] Run complete bridge/full category/heldout gate.
- [ ] Retain only if every category is non-regressive and total wall advances a
      declared cell; otherwise revert runtime changes.
- [ ] Publish retained/rejected artifact, rollup/changelog/worklog, and commit.

## 16. P7 — conditional provider repair

TaskList: #38.

Provider repair measured ~7.2 ms in C2/K2 before earlier phases, far below the
392.8 ms target. It is revisited only after P2-P6 because relative ownership may
change.

- [ ] Reprofile repair marker/kernel/synchronization/allocation wall.
- [ ] Admit only at >=1.10x operation-complete repair and >=1% projected request
      saving, or enough to cross a fixed-cell gate.
- [ ] Prefer exact prefix-KV/live-cursor commit of already-produced rows over
      restore plus depth-by-depth replay when the Qwen3.8 NextN state contract
      permits it.
- [ ] Preserve correction/bonus catch-up and canonical provider fingerprints.
- [ ] Use persistent group workspace; no cycle malloc/free.
- [ ] Gate reject/every-partial/full, following proposal/AR, failure rollback,
      compaction/refill, memory, profile, and complete suite.
- [ ] Publish retain/reject/no-go evidence and commit.

## 17. P8 — production FP16 recurrent-state capability

TaskList: #33.

Automatic Q4_K_S currently selects K0 before mutation because strict chain base
state readers require FP32, while normal product AR uses scoped FP16 recurrent
state. A strict-only speed win cannot become the normal product default.

### P8.1 RED/profile contract

- [ ] Resolve a runtime production manifest and strict fallback manifest.
- [ ] Add FP16 resident-state-aware root/parent/candidate readers with declared
      FP32 accumulation/scratch.
- [ ] Add exact control ownership and numerical fixtures for root, every parent
      depth, selected commit, rollback, and following AR.
- [ ] Keep unsupported profile/shape/context K0 before mutation.

### P8.2 Binding gates

- [ ] strict-teacher mean/p95/p99/max KL and top-1 per category/shape/transition;
- [ ] same-schedule deterministic repeats and neighbor/permutation isolation;
- [ ] state/KV/provider/output/cursor ownership and finite values;
- [ ] applicable BF16-relative and external task gates, with explicit N/A only
      when normative docs allow it;
- [ ] cached profiler expected production+fallback variants and manifest hashes;
- [ ] memory high-water/recovery and lifecycle/pressure/prefix/cancel/soak; and
- [ ] complete wall against **production FP16 AR**, not slower FP32 AR.

- [ ] Retain compatibility only when every profile gate passes.
- [ ] Promote no cell solely because FP16 is now supported.
- [ ] Publish artifact/rollup/changelog/worklog and commit.

## 18. P9 — policy and product qualification

TaskList: #34.

### P9.1 Fixed cells first

- [ ] Re-run fixed C1/C2/C4 K1-K3 after all retained changes.
- [ ] Keep K1 deprioritized unless new measurements overturn its extra-cycle
      loss.
- [ ] Build policy only from cells with complete evidence.
- [ ] Allow features: realized C, remaining output horizon, context/page bucket,
      profile, memory fit, and a predeclared bounded acceptance statistic.
- [ ] Forbid prompt text/hash, token IDs, category, heldout identity, task
      result, and post-hoc oracle selection.
- [ ] Fingerprint the table and evidence links.

### P9.2 Wider work admission

C8, R24/R32, adaptive K, or proposal/target overlap begins only when:

- at least one fixed C1/C2/C4 cell already exceeds 1.10x true AR;
- a current profile shows a measured premise;
- physical ownership is not simulated by multiple hidden weight sweeps; and
- the new cell has a correctness oracle and strict fallback.

No such premise means K0 and no implementation.

### P9.3 Product packet

For each proposed automatic cell:

- [ ] full train/heldout/category quality and task gates;
- [ ] counterbalanced same-host true-AR complete economics;
- [ ] output-horizon/context boundary rows;
- [ ] fixed/ragged/delayed admission/refill/retirement;
- [ ] mixed AR/MTP neighbors and fairness;
- [ ] prefix reuse/COW and pressure/regrow;
- [ ] cancel/deadline/EOS/stop/failure/circuit breaker/restart;
- [ ] blocking completion/chat and SSE completion/chat;
- [ ] below/near/above Poisson load plus overload;
- [ ] TTFT/ITL median/p95/p99, queue, E2E, exact/SLO goodput;
- [ ] memory high-water/fragmentation/final return;
- [ ] focused soak, then final promotion soak only after all shorter gates pass;
      and
- [ ] exact route/reason/result reporting.

Automatic promotion requires >1.10x true AR, every binding gate, and
non-regressive AR-neighbor SLO. The project target remains >1.30x. Other cells
remain K0 with exact measured/unqualified reasons.

## 19. P10 — closure

TaskList: #35.

- [ ] List every retained implementation unit and default scope.
- [ ] List every promoted automatic `(model/backend/profile/C/K/context/horizon)`
      cell.
- [ ] List every explicit/default-off functional cell.
- [ ] List every K0/rejected/no-go cell and durable reason/artifact.
- [ ] Remove superseded SPECDEC2 prompt-hidden replay, allocation, CPU-oracle,
      and migration flags only when their removal gates pass.
- [ ] Update `REFACTOR.md` for retained debt and remove resolved entries.
- [ ] Update `KERNELS.md`, lineage, `PLAN.md`, `SPECDEC2.md`, and execution-profile
      docs where ownership changed.
- [ ] Update benchmark README/changelog/artifacts for every retained/rejected
      measurement.
- [ ] Export public root README only if a product scope promotes; keep internal
      implementation history out of it.
- [ ] Run milestone validation under the focused-repair rule.
- [ ] Run Worklog2, benchmark sync, fixtures, registry, JSON/link, and diff
      checks.
- [ ] Commit final closure, merge clean campaign branch, push, and verify
      local/remote equality.
- [ ] Hand gfx1100 S7 a strict functional source and the no-repeat/performance
      lessons; transfer no absolute rates or thresholds.

## 20. Do-not-chase ledger

Do not start with or repeat:

- candidate device handoff alone: measured about +0.14%/+0.20% wall, neutral;
- accept-kernel micro-optimization while target is ~393-480 ms;
- K1 C2/C4 under current costs: more cycles and the worst fixed path;
- cheaper/truncated draft vocabulary before proposal graphs/stable activation:
  historical Q4_K_S draft share capped the gain and carries acceptance risk;
- adaptive K before fixed cells win: the first content-agnostic EMA policy was
  already rejected on the adjacent Q4_K_M lane;
- C8/R32 without a qualified physical proposal/target owner;
- multi-stream overlap before stable pointers and trace-proven slack;
- graph capture as a launch-count exercise without marker-wall evidence;
- attention tuning without a current complete target profile;
- generic AR/MoE/LM-head tuning unrelated to the R target profile;
- prompt/category/token-conditioned routing; or
- a return to request-serial/whole-request speculative ownership.

A mechanism can reopen only after retained earlier phases materially change its
cost share or a new exact representation/dataflow gives a measured premise.

## 21. Expected code map

| Path | Expected campaign role |
| --- | --- |
| `docs/SPECDEC2-PERF.md` | This ledger and current handoff. |
| `scripts/specdec2_perf_bridge.py` | Common AR/native/staged bridge and artifact writer. |
| `scripts/specdec2_perf_profile_child.py` or focused child mode | Final cached ROCTX/rocprof child. |
| `hipengine/generation/qwen35_gguf_mtp2.py` | Streaming activation integration, stable staged slabs, device-result orchestration, provider repair. |
| `hipengine/generation/engine_loop.py` | Only bounded-cycle timing/result plumbing; no provider/backend policy. |
| `hipengine/runtime/qwen35_gguf_nextn.py` | Existing prompt streaming/proposal graph/device candidate primitives; extend only with provider-owned exact contracts. |
| `hipengine/runtime/qwen35_gguf_mtp.py` | Existing target/accept/selected-commit strict oracle and device descriptors. |
| `hipengine/runtime/gguf_native_spec_cycle.py` | Existing N2/complete-cycle components reused inside staged ownership, not as request owner. |
| `hipengine/runtime/qwen35_gguf_runner.py` | Target hidden chunk sink and physical target lowering where unavoidable. |
| `hipengine/kernels/hip_gfx1100/` | Shared source only for a P5-admitted in-tree kernel/graph primitive. |
| `hipengine/kernels/hip_gfx1151/` | Independent capability/variant registration. |
| `tests/test_specdec2_*.py` | Contracts, lifecycle, policy, and focused route tests. |
| `tests/test_qwen35_gguf_mtp2_seam.py` | Dense staged provider/target integration gates. |
| `benchmarks/results/` | Compact retained/rejected/no-go artifacts. |

No normal generation path imports torch. Kernel bodies keep raw pointer ABIs.
Capability/profile selection remains cold and registry-driven.

## 22. Validation tiers and commit discipline

For every phase:

1. `git status -sb`; preserve unrelated files.
2. Read the latest campaign/worklog handoff.
3. Mark the corresponding TaskList item `in_progress`.
4. Add/identify RED first.
5. Run narrow failing node.
6. Implement the minimum scoped change.
7. Run repaired node and affected narrow bundle.
8. Run required CPU/GPU/profile/common-bridge gates.
9. Publish artifact/docs/worklog/rollup as required.
10. `python3 scripts/worklog.py check`.
11. `python3 scripts/sync_benchmark_readme.py --check`.
12. `git diff --check`; explicitly stage only owned paths.
13. Inspect staged names and full staged diff.
14. Commit immediately before starting the next phase.
15. Mark task complete and inspect TaskList for the next unblocked item.

Do not automatically repeat a completed broad suite after an isolated scoped
failure; follow `AGENTS.md` focused-repair policy. Do not commit exploratory or
broken states. A rejected candidate commits its durable artifact/doc/worklog
with runtime changes reverted.

## 23. Artifact contract

Each phase artifact records at minimum:

- schema/kind/status/date and performance-claim eligibility;
- exact source commit/cleanliness/worktree and excluded unrelated files;
- host/device/arch/power/ROCm/compiler/queue policy;
- model path/hash method/hash/size/GGUF identity/quant/KV/state/profile;
- prompt fixture hash/train/heldout/categories;
- exact commands/environment/warmup/repeats/order/timing scopes;
- AR/native/staged/candidate samples and exact token accounting;
- activation and per-cycle stage attribution;
- C/K/logical R/physical decomposition/weight sweeps;
- acceptance/proposal/cycle/output counts and per-depth histogram;
- route/graph/eager/fallback/generation ownership;
- correctness/profile/ownership/quality/task verdicts;
- kernel/marker/API/copy/allocation trace hashes and summaries;
- memory high-water/final conservation;
- total-wall reduction required/achieved and category split;
- keep/revert/no-go decision, blocker, and next trigger; and
- links/hashes for prerequisite evidence.

Raw profiler dumps and terminal logs remain outside Git.

## 24. Current handoff to the gfx1151 agent

Start with TaskList #24 from the committed P0 handoff. Do not begin by editing
kernels. Build the common bridge, run the stable current-main baseline, and
confirm whether activation and device synchronization explain the complete wall.
Then integrate the already-retained OI-3 streaming sink in P2.

The first expected implementation touchpoint is the mismatch between:

- retained `TargetHiddenChunkSink`, `_StreamingNextNPromptSink`, and
  `Qwen35GGUFNextNExecutor.enqueue_prompt_rows()`; and
- SPECDEC2 `Qwen35GGUFMTP2Adapter._catch_up_provider{,_batch}()`, which currently
  consumes a prompt-sized host hidden slab after target prefill.

After P2, eliminate hot allocation before enabling graphs. After stable pointers,
make target→accept→commit genuinely device-resident. Only then profile the
physical target and select one candidate. This ordering is binding unless fresh
stable-host evidence changes it and the campaign ledger is updated before work.

The gfx1100 portability task is deferred by explicit user scope while this
stable gfx1151 performance campaign runs. When P10 closes, gfx1100 receives the
functional architecture, exact fallbacks, and measured do-not-repeat lessons,
not gfx1151 rates, thresholds, or assumed graph buckets.
