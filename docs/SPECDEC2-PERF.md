# SPECDEC2-PERF — gfx1151 Activation and Hot-Cycle Campaign

- Status: **P1-P10 closed; explicit compatibility retained, zero automatic cells, K0 default**
- Approved: **2026-08-25**
- Functional predecessor: [`SPECDEC2.md`](SPECDEC2.md), S1-S6 closed
- Performance owner: **stable physical host `gfx1151` agent**
- Development/retention hardware: **AMD Radeon 8060S Graphics / `hip_gfx1151`**
- Primary artifact: **Qwen3.8-27B `Q4_K_S`, BF16 KV**
- Initial profile: **strict FP32 recurrent state**
- Product profile follow-up: **scoped FP16 recurrent state with FP32 fallback**
- Selected implementation base: **`ccd077a1f` or its clean descendant containing the complete S6 merge, this plan, and the independent gfx1100 S7 C1 foundations**
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

The independent [`SPECDEC2-GFX1100`](SPECDEC2-GFX1100.md) S7 lane is active and
may proceed concurrently in a separate worktree/on separate hardware. It shares
backend-neutral contracts and some dense-GGUF source files, but no rates,
thresholds, policy cells, profile manifests, or physical bucket decisions.
Same-file work is serialized through a designated merge owner before either lane
continues.

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
- gfx1100 execution, qualification, or policy work inside this campaign (the independent S7 lane continues under [`SPECDEC2-GFX1100.md`](SPECDEC2-GFX1100.md));
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

### 3.5 Hot-cycle break-even planning bound

For a non-tail greedy cycle, the first-order value of one target pass is:

```text
expected visible tokens / request / cycle = 1 + expected accepted draft tokens
break-even speculative cycle wall
  <= expected visible tokens * matched physical-AR step wall
```

At the full-suite K2 draft acceptance of 49.15%, the planning expectation is
`1 + 2*0.4915 = 1.983` visible tokens per request/cycle. Combining that with the
S5 short physical-row ladder gives a deliberately rough bound:

| Cell | Current reported proposal+target+repair/other direct stages | Approximate AR-equivalent target budget | Reduction to hot-cycle parity | Reduction to 1.10x hot-cycle speedup |
| --- | ---: | ---: | ---: | ---: |
| C2/K2 | ~433.3 ms | `1.983 * R2 133.3 ms = 264.3 ms` | ~39.0% | ~44.5% |
| C4/K2 | >=506.6 ms | `1.983 * R4 198.9 ms = 394.4 ms` | >=22.2% | >=29.2% |

This is a **planning bound, not retained evidence**: it combines full-suite
acceptance with the short target ladder, excludes tails/variable active rows,
and does not resolve activation or asynchronous residual. P1 replaces it with
cycle-trajectory-weighted committed-token economics from one common protocol.
A target/kernel candidate is not admitted merely because it meets this estimate.

### 3.6 Older native control is diagnostic, not an automatic baseline

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

### 5.2 Cross-lane coordination

The gfx1151 performance owner fetches `origin/main` before each phase and checks
for changes from the active gfx1100 S7 lane. Shared-file edits are not developed
in parallel:

1. identify the designated merge owner for the shared file;
2. let that owner stage, validate, and commit first;
3. synchronize the other worktree without discarding local work;
4. rerun both lanes' affected CPU/seam tests; and
5. keep hardware evidence independent even when a backend-neutral source change
   is shared.

A common adapter improvement may be retained once under backend-neutral
contracts, but it cannot transfer a gfx1151 performance verdict to gfx1100 or a
gfx1100 correctness capability to gfx1151.

### 5.3 GPU preflight

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
rocminfo | grep -E 'Name:|gfx'
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Before profiling, prebuild outside `rocprofv3`, write one compiler-version file,
and run children with `HIPENGINE_REQUIRE_CACHED_BUILD=1`. Do not profile a
parent harness that launches nested children.

### 5.4 GPU lease

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
provider_k0_attach         # initial provider ownership after prefill
  nextn_prompt_prime       # nested initial/refill activation work
provider_open              # later speculative prepare/refill owner
resident_owner_transition  # packed-AR flush/scatter/discard and SPECDEC2 handoff
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
- [x] Create the detailed durable phase/punchlist; session-local task trackers
      may mirror it but are not handoff identifiers.
- [x] Name OI-3 streaming prompt priming as the first implementation candidate.
- [x] Coordinate the independently active gfx1100 S7 lane through separate
      evidence, worktrees/hardware, and serialized ownership of shared files.
- [x] Commit and push this plan/worklog/pointer unit.

Exit: committed handoff on `origin/main`, no tracked local changes, and phases
P1-P10 owned by `gfx1151-agent`.

## 10. P1 — common bridge and current-main baseline

**Closed 2026-08-25.** Durable evidence:
[`P1 bridge`](../benchmarks/results/2026-08-25-gfx1151-specdec2-perf-p1-bridge.json).
Strict C1 K1/K2/K3 reaches **1.138x/1.300x/1.273x true AR** on the full
natural25 suite; K2 is positive in every category but remains a strict premise,
not an automatic/product promotion. Physical C2/C4 K2 remains blocked at
**2.786x/3.142x true-AR wall** and requires **67.37%/71.07%** total-wall
reduction to reach 1.10x. Draft acceptance is **90.42% at C1/K2** but only
**18.43% at both physical widths**, making proposal quality/cycle count a
separate physical blocker. All retained cells are exact, cached, clean, and
zero-final-ownership.

Durable handoff: P1.1-P1.2.

### P1.1 Harness

- [x] Add `scripts/specdec2_perf_bridge.py` as the focused committed harness;
      S6 closure remains unchanged.
- [x] Add schema/aggregation/counterbalance unit tests.
- [x] Support AR, legacy native, and staged arms in one loaded process where
      ownership permits. Strict C1 requires capacity1; physical C2/C4 share
      capacity4, and mixed-capacity invocations fail before model load.
- [x] Support C1/C2/C4, K1-K3, output horizon, prompt limit, repeat count, and
      train/full scopes.
- [x] Emit immediate progress and atomic checkpoints after each prompt/arm.
- [x] Emit complete and decode-only timing, stage attribution, exact accounting,
      physical shapes, candidate/acceptance trajectories, allocation/API
      counters, and teardown. Provider fingerprints remain focused-gate evidence.
- [x] Attribute packed-AR owner transition, graph close, initial K0 provider
      attach, NextN priming, later provider-open, proposal, target-cycle, AR
      tail, and reclaim separately with declared nesting/residual.
- [x] Add final single-child profiler mode with cached builds and ROCTX markers.
- [x] RED-test malformed/missing timing owners, duplicated batch timing,
      incomplete prompt suite, invalid AR denominator, route substitution,
      mixed capability capacity, and dirty provenance.

### P1.2 Stable baseline

- [x] Run from clean synchronized `main`, serializing shared-file merges with the
      independent gfx1100 lane.
- [x] Record full platform/model/compiler/prompt/profile provenance.
- [x] Run C1 K1-K3 natural25 counterbalanced AR/native/staged packet, three
      repeats over all ten prompts.
- [x] Run C2/C4 K2 natural25 and fixed-greeting short activation packets.
- [x] Run cached staged C2/K2 R6 and C4/K2 R12 marker/kernel/HIP-API traces.
- [x] Verify old-native/staged outputs and acceptance separately; legacy direct
      control is C1-only because its C>1 owner is request-serial.
- [x] Freeze activation, initial K0 provider attach, steady-cycle, target,
      repair, synchronization/allocation, and residual ceilings.
- [x] Recalculate exact reduction needed for every measured fixed cell.
- [x] Publish
      `benchmarks/results/2026-08-25-gfx1151-specdec2-perf-p1-bridge.json`.
- [x] Update campaign status/worklog/rollup and commit.

Retained commands use two capability-honest loads (full commands and hashes are
in the artifact/worklog):

```bash
# C1/capacity1, including K1/K2/K3 diagnostics
.venv/bin/python scripts/specdec2_perf_bridge.py ... \
  --concurrency 1 --budgets 1,2,3 --max-tokens 25 --runs 3 \
  --require-cached-build --output /tmp/specdec2-perf-p1-c1-k123-full-fixed.json

# Physical C2/C4/capacity4, retained K2 packet
.venv/bin/python scripts/specdec2_perf_bridge.py ... \
  --concurrency 2,4 --budgets 2 --max-tokens 25 --runs 3 \
  --require-cached-build --output /tmp/specdec2-perf-p1-c2c4-k2-full-fixed.json
```

Do not combine C1 with C2/C4 in one invocation: capacity4 correctly declines
strict C1 and would produce a false K0 row.

Exit: common-protocol attribution identifies activation and steady-cycle costs;
P2 is admitted. P3 also has a concrete premise: cold profile children observe
**143 allocation/free API calls inside three C2 and C4 cycle windows**. P4/P5
must treat the C1→physical acceptance collapse as binding economics rather than
assuming target kernels alone close C2/C4.

## 11. P2 — streaming exact NextN prompt activation

**Closed 2026-08-25.** Durable evidence:
[`P2 activation`](../benchmarks/results/2026-08-25-gfx1151-specdec2-perf-p2-streaming-activation.json).
Strict C1/K2 streams every target prompt row into NextN, retains one 10 KiB
carried row, and improves full-suite staged throughput **14.294→16.237 tok/s
(+13.59%)**; every category is positive and staged reaches **1.486x true AR**.
The physical C2/C4 streaming candidate is rejected despite aggregate gains
because `general_en` regresses **11.86%/10.16%**; those widths retain exact
host-replay fallback. Exact p4K/p16K eager streaming also regresses versus the
direct control, so contexts above the qualified short bucket select K0 before
provider mutation. Automatic/product policy remains K0.

Durable handoff: P2.1-P2.3. Source mechanism: retained commit `51e990ee8`.

### P2.1 RED contracts

- [x] One-shot and chunked target prefill produce identical shifted NextN KV and
      cursor (`token[0]+zero`, `token[i>0]+target_hidden[i-1]`).
- [x] Chunk sizes 1/2/7/8/9, ragged packed offsets, warm offsets, and context
      transitions are exact in focused contracts.
- [x] Cancel/failure during activation drains target/provider/sink/claim owners
      and leaves later reuse healthy.
- [x] Prefix reuse and contexts beyond the qualified bucket select K0 before
      provider mutation; retained S6 prefix/pressure ownership remains binding.
- [x] Target hidden chunk source remains live until NextN consumption completes.
- [x] Selected C1 allocates no prompt-sized host hidden slab; telemetry reports
      one 10,240-byte carried row independent of prompt length.

### P2.2 Implementation

- [x] Wire `TargetHiddenChunkSink` and `_StreamingNextNPromptSink` into the
      Generation-2 target prefill owner before provider catch-up is needed.
- [x] Feed `Qwen35GGUFNextNExecutor.enqueue_prompt_rows()` on the target prefill
      stream as chunks complete and publish the exact batch-session cursor.
- [x] Retain one request-owned BF16 carried hidden row only.
- [x] Remove `_prompt_hidden_rows` and post-prefill F32→BF16/H2D replay from the
      selected C1 path; keep internal strict/physical fallback only.
- [x] Keep long target/session capacity independent, but select K0 above the
      currently qualified 1023-token speculative bucket after measured eager
      regressions; no long graph/eager scope is auto-admitted.
- [x] Preserve exact host-replay fallback for physical C2/C4 after its category
      gate rejects streaming selection.
- [x] Reserve exact prompt-row/carried-row/provider-slot work-item claims before
      provider mutation and release them on every terminal path.
- [x] Update `REFACTOR.md` with rollback/fallback removal triggers.

### P2.3 Gate

- [x] Focused sink/provider/adapter/prefill/bridge bundles pass.
- [x] p128/p512 streaming is exact; p4K/p16K diagnostics are exact and final
      policy proves pre-mutation `target_context_k0` with no provider streaming.
- [x] p128/p512 staged wall beats the current direct activation control by
      **50.4%/17.4%**; exact p4K/p16K streaming was rejected at
      **1.202x/1.231x direct wall** and replaced by K0.
- [x] Full ten-prompt C1 bridge is exact and every category improves
      **10.50%-18.21%**. Physical streaming fails this gate and is unselected.
- [x] Selected prompt transient is O(hidden): exactly 10,240 carried bytes from
      p128 through p512 and in long diagnostics.
- [x] Final allocations, prompt claims, provider states, and request owners are
      zero in selected/fallback/current-commit gates.
- [x] Publish artifact/rollup/changelog/worklog and commit the scoped retain/
      reject/K0 verdict.

Exit: qualified C1 prompt activation no longer performs post-prefill full-prompt
replay. Physical C2/C4 retains exact replay after rejection; p4K/p16K is K0.

## 12. P3 — stable slabs and zero hot allocation

**Closed 2026-08-26.** Durable evidence:
[`P3 stable slabs`](../benchmarks/results/2026-08-25-gfx1151-specdec2-perf-p3-stable-slabs.json).
Adapter-persistent max-width proposal/repair slabs eliminate the two steady
allocation/free pairs per cycle: all **510 C2 + 510 C4** full-suite cycle samples
allocate/free zero bytes, and **219/252 C1** cycles are zero after request-local
graph bucket first use. Cached C2/C4 profile windows 2–3 contain no
`hipMalloc`/`hipFree`; final ownership is zero. Complete C1/C2/C4 throughput is
statistically neutral (**-0.264%/-0.300%/-0.128%**, versus 4.98%-10.14% sample
CV), so this is a retained mechanical/stable-pointer prerequisite, not a speed
or policy claim.

Durable handoff: P3.1-P3.3.

P1 confirmed cycle-local `malloc/free` around proposal/repair hidden batches and
workspace creation: the cold cached C2 and C4 profiler children each record
**131 `hipMalloc` + 12 `hipFree` calls inside three cycle windows**. Allocation,
copy, and free outside substage timers remain explicit; `hipFree` may synchronize
the device.

### P3.1 RED and instrumentation

- [x] Add bridge-only per-cycle allocation/free byte/count deltas; production
      timing remains untouched, while HIP-API traces own sync/copy counts.
- [x] RED-test stable distinct pointer reuse, shape drift, close, group width,
      failure/lifecycle, and packed peer ownership.
- [x] Assert zero hot allocation/free for every full-suite C2/C4 cycle and every
      warmed C1 graph-shape cycle; first-use graph capture remains P4 ownership.
- [x] Assert proposal and repair use distinct complete max-width slabs, so stale
      peer rows are never read after group shrink/refill.

### P3.2 Implementation

- [x] Replace proposal/physical-repair `hidden_batch` allocation/free with one
      adapter `RuntimeWorkspace` containing distinct proposal/repair slabs.
- [x] Retain the already-persistent accept/result/remaining-decode and target
      row-map buffers under complete cycle claims; add exact cycle hidden claims.
- [x] Preserve stable candidate/target/accept/commit/provider/result pointers;
      expose proposal/repair pointer/shape contract for diagnosis.
- [x] Allocate maximum qualified physical C (1 or 4) × hidden-size slabs and
      fail closed on shape drift rather than silently reallocating.
- [x] Close slabs with adapter/model generation; failure retains valid blankable
      workspace and final engine close returns every byte.
- [x] Keep strict eager/host replay and selected C1 streaming paths independently
      usable.

### P3.3 Gate

- [x] Full-suite reject/partial/full acceptance trajectories remain exact; C1
      selected streaming and physical replay fallback preserve provider state.
- [x] Lifecycle/prefix/failure/soak/compaction and workspace bundles pass.
- [x] Cached C2/C4 traces confirm zero `hipMalloc`/`hipFree` in cycle windows
      2–3; mechanical samples prove every warmed physical cycle zero.
- [x] Complete full-suite bridge is exact and neutral within measured variance;
      no speed/default claim is made from small mixed category deltas.
- [x] Zero final ownership and exact total allocated/freed conservation pass.
- [x] Publish artifact/worklog/rollup and commit.

Exit: every later graph/device-chain candidate can rely on stable claimed
addresses. P4 owns request-local graph first-use and device-result chaining.

## 13. P4 — device-resident proposal → target → accept → commit

Durable handoff: P4.1-P4.3.

### P4.1 C1 fast-cycle bridge

- [x] Enable budget-specific proposal graph ownership instead of unconditional
      `allow_graph=False` where the exact capability admits it.
- [x] Adapt the existing N2 target/accept/selected-commit graph into one bounded
      staged cycle result; do not call the old whole-request generator.
- [x] Pass proposal tokens to target through a device descriptor without host
      candidate materialization.
- [x] Preserve canonical provider checkpoint/repair semantics; compare provider
      fingerprint after reject/every-partial/full.
- [x] Return to `ResidentEngineLoop` after one committed/rolled-back cycle.

### P4.2 Physical C2/C4 device result

- [x] Target verifier writes compact device top-1/result rows into stable slabs.
- [x] Candidate and target IDs feed GPU acceptance directly.
- [x] GPU accept payload selects target hidden/Conv/GDN/KV rows and provider
      commit metadata per request.
- [x] Host reads only bounded committed token IDs/lengths/status after selected
      commit.
- [x] CPU acceptance remains a strict/debug oracle controlled outside the
      promoted cycle; it is not a permanent production synchronization.
- [x] Device/result descriptors include request/slot/row/transaction generation.

### P4.3 Gate

- [x] C1/C2/C4 K1-K3 reject/every-partial/full exact gates pass.
- [x] Candidate/target/GPU accept match CPU oracle in qualification.
- [x] Selected hidden/Conv/GDN/KV/provider state and following AR are exact.
- [x] Output tails, EOS/stop, cancel/deadline, prefix/pressure/compaction,
      failure/restart, and graph/eager miss paths pass.
- [x] Profile shows no pre-accept candidate/target-ID D2H or Python
      `TargetVerifyBatch` reconstruction in the promoted route.
- [x] Proposal/target/accept/commit named kernels execute with plausible positive
      durations and zero unexpected scratch.
- [x] Common bridge/full suite is exact; C1 is positive by every category while
      physical C2/C4 remain performance-blocked by measured acceptance economics.
- [x] Publish artifact/rollup/changelog/worklog and commit.

Exit: synchronization is one bounded final-result boundary, not a sequence of
host materialize/reconstruct/re-upload steps.

## 14. P5 — physical target profile and candidate admission

Durable handoff: P5.1-P5.3. Stable-GPU cost: approximately 15-45 minutes after cache warmup.

### P5.1 Required profiles

- [x] Mark complete R6 (C2/K2), R8 (C2/K3 or C4/K1), R12 (C4/K2), and R16
      (C4/K3) target windows.
- [x] Capture kernel, HIP API, memory-copy, and marker traces from final cached
      children.
- [x] Attribute exact Q4_K_S projection weights/shapes/row routes rather than
      grouping all GEMV by symbol alone.
- [x] Separate dense projection, Conv/GDN provisional state, attention/KV,
      LM-head/top-1, accept, selected commit, and other.
- [x] Record call counts, interval union, queue gaps, workgroup/grid, VGPR/SGPR,
      LDS, scratch, and bytes where known.
- [x] Compare target marker, kernel family sum, and complete cycle; do not infer
      savings from launch count or kernel sum alone.
- [x] Reconcile R16 versus two R8 under the current post-P4 path.

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

- [x] Admit at most one candidate meeting the general admission gate.
- [x] Record expected operation-complete and projected request saving.
- [x] Name RED oracle, strict fallback, exact C/K/R/context/profile scope, and
      profiler kernel expected.
- [x] If none qualifies, publish a no-go artifact and skip P6 runtime code.
      Not applicable: one existing-route candidate qualifies.
- [x] Update `KERNELS.md`/lineage only if dispatch/kernel ownership changes.
      No kernel body/lineage changes in P5; P6 owns any scoped dispatch update.
- [x] Commit profile/admission decision before implementation.

## 15. P6 — one physical target optimization

Durable handoff: P6 checklist below.

- [x] Write RED oracle/route test before device changes.
- [x] Implement only the P5-admitted candidate.
- [x] Keep C/K/R/context/profile scope explicit through registry/capability;
      engine/model code receives no backend/quant hot branch.
- [x] Retain strict eager/parent route for every miss.
- [x] Run primitive/operation exact or declared production numerical gate.
- [x] Run CPU-reference outer floor and profiler engagement for new kernels.
      No new kernel body; the existing exact parent passes the outer gate and
      engages at every admitted width.
- [x] Compare operation-complete target against shipped route with balanced
      samples.
- [x] Run complete bridge/full category/heldout gate.
- [x] Retain only if every category is non-regressive and total wall advances a
      declared cell; otherwise revert runtime changes.
- [x] Publish retained/rejected artifact, rollup/changelog/worklog, and commit.

## 16. P7 — conditional provider repair

Durable handoff: P7 checklist below.

P7 retains persistent after-root provider snapshots plus conditional physical
repair. Reject restores the root snapshot, accepted `K-1` publishes the already-
current proposal state, full acceptance advances only the last candidate, and
other depths retain exact checkpoint replay. Cached mixed K2 repair improves
**127.743→30.518 ms (4.186x)** at C2 and **138.779→34.292 ms (4.047x)** at C4;
projected request saving is **1.173%/1.020%** and the matched one-prompt wall
improves **1.345%/1.351%**. Full-suite physical throughput improves
**5.718→5.796 (+1.37%) / 9.331→9.445 tok/s (+1.22%)**, every category is
positive, while C1 is neutral/out of scope. Physical remains below true AR and
automatic/product remains K0. Evidence:
[`P7 retained`](../benchmarks/results/2026-08-26-gfx1151-specdec2-perf-p7-provider-repair-retained.json).

- [x] Reprofile repair marker/kernel/synchronization/allocation wall.
- [x] Admit only at >=1.10x operation-complete repair and >=1% projected request
      saving, or enough to cross a fixed-cell gate.
- [x] Prefer exact prefix-KV/live-cursor commit of already-produced rows over
      restore plus depth-by-depth replay when the Qwen3.8 NextN state contract
      permits it.
- [x] Preserve correction/bonus catch-up and canonical provider fingerprints.
- [x] Use persistent group workspace; no cycle malloc/free.
- [x] Gate reject/every-partial/full, following proposal/AR, failure rollback,
      compaction/refill, memory, profile, and complete suite.
- [x] Publish retain/reject/no-go evidence and commit.

## 17. P8 — production FP16 recurrent-state capability

Durable handoff: P8.1-P8.2.

At P8 entry, automatic Q4_K_S selected K0 before mutation because strict chain
base-state readers required FP32 while normal product AR used scoped FP16
recurrent state. P8 qualifies the FP16 compatibility surface but deliberately
does not promote a cell before P9's complete fixed-cell policy gate.

### P8.1 RED/profile contract

P8.1 retains the existing typed FP16 chain-row writer under a non-fallback
production manifest with explicit FP32 strict fallback. GREEN bring-up corrected
the audit's initial snapshot premise: gfx1151 intentionally excludes
producer-folded snapshot/Q5-chain aliases, so production preserves P4's
consumer-owned dtype-sized D2D rollback and exact unfused cast. C1/C2/C4
production-FP16 smokes match production AR without recovery. P8.2 subsequently
proved native target graphs diverge after layer 51 under FP16 state, so all FP16
target verify/selected commit remains on the exact eager owner; automatic policy
remains K0 through P8 pending P9's complete fixed-cell rebuild. Durable details:
[`P8 audit`](../worklog/entries/20260825T215042.972658Z-gfx1151-agent-specdec2-perf-p8-fp16-audit-a613ab.md),
[`P8 capability`](../worklog/entries/20260825T221958.712228Z-gfx1151-agent-specdec2-perf-p8-fp16-capability-4f8ea6.md).

- [x] Resolve a runtime production manifest and strict fallback manifest.
- [x] Add FP16 resident-state-aware root/parent/candidate readers with declared
      FP32 accumulation/scratch.
- [x] Add exact control ownership and numerical fixtures for root, every parent
      depth, selected commit, rollback, and following AR.
- [x] Keep unsupported profile/shape/context K0 before mutation.

### P8.2 Binding gates

P8.2 retains production FP16 compatibility with target verification on the
exact eager owner. The fresh general gate passes 450 strict-teacher rows; the
SPECDEC2 K1-K3 operation gate passes 36/36 top-1, exact chain-vs-scalar logits,
reject/partial/full commit/following-logit controls, and post-commit rollback.
C2/C4 same-width repeats, neighbor substitution, and permutation pass. Cached
traces bind 288 `_Float16` selected-chain dispatches to production and 288
`float` dispatches to strict fallback. The rejected FP16 graph path remains
recorded, not hidden.

The final full 10-prompt/three-run D25 K2 packet is exact in all 90 C1/C2/C4
cells against **production FP16 AR**. C1 reaches **15.204 vs 10.807 tok/s
(1.407x)** at 90.42% acceptance. Physical C2/C4 remain performance-blocked at
**5.810 vs 15.213 (0.382x) / 9.469 vs 27.598 tok/s (0.343x)** and 18.43%
acceptance. No P8 cell promotes: automatic stays K0 until P9 re-runs all K1-K3
fixed cells.

Production lifecycle passes controlled reject/full, compaction/refill/cancel,
completed-prefix COW, pressure/memory recovery, proposal/target/postcommit
readback recovery, and a 25-wave/110-request soak with zero request-scoped pages
and final ownership. Postcommit accept-readback faults rebuild target state from
scheduler-owned canonical tokens before AR fallback; this is fault-only and adds
no normal-cycle work. Evidence:
[`P8 qualification`](../benchmarks/results/2026-08-26-gfx1151-specdec2-perf-p8-fp16-qualification.json),
[`P8 retained profile`](../benchmarks/results/2026-08-26-gfx1151-specdec2-perf-p8-fp16-retained.json).

- [x] strict-teacher mean/p95/p99/max KL and top-1 per category/shape/transition;
- [x] same-schedule deterministic repeats and neighbor/permutation isolation;
- [x] state/KV/provider/output/cursor ownership and finite values;
- [x] applicable BF16-relative and external task gates, with explicit normative
      N/A because model quant/BF16 KV and claimed task capabilities are unchanged;
- [x] cached profiler expected production+fallback variants and manifest hashes;
- [x] memory high-water/recovery and lifecycle/pressure/prefix/cancel/soak; and
- [x] complete wall against **production FP16 AR**, not slower FP32 AR.

- [x] Retain compatibility only when every profile gate passes.
- [x] Promote no cell solely because FP16 is now supported.
- [x] Publish profile qualification plus final economics/lifecycle
      artifact/rollup/changelog/worklog and commit.

Exit: production FP16 SPECDEC2 is a retained explicit compatibility surface with
strict FP32 fallback; automatic remains K0 and P9 owns fixed-cell policy.

## 18. P9 — policy and product qualification

Durable handoff: P9.1-P9.3.

### P9.1 Fixed cells first

The final post-reseed grid on clean `b8708c41c` is complete: 540/540
production/strict C1/C2/C4 K1-K3 cells are exact with zero candidate
D2H/recovery and complete tracked memory return. Production C1 reaches
**1.2310x/1.4087x/1.4037x AR** at K1/K2/K3. K2 is the sole product candidate:
it leads K3 by 0.34% aggregate and uses fewer target rows. Strict C1 reaches
**1.2472x/1.4392x/1.4960x**, but strict is fallback evidence, not the product
denominator.

The target-reseed repair raises production physical acceptance to
**95.0%/89.8%/77.7%** at K1/K2/K3. Target cost remains binding: best C2/C4 is
K3 at only **0.6975x/0.5843x AR**. The fingerprinted fixed table admitted C1/K2
to P9.3 only; the product gate below rejects it. Automatic remains K0 and every
physical/unqualified width remains K0. Evidence:
[`P9 fixed policy`](../benchmarks/results/2026-08-26-gfx1151-specdec2-perf-p9-fixed-policy.json).

- [x] Re-run fixed C1/C2/C4 K1-K3 after all retained changes.
- [x] Keep K1 deprioritized unless new measurements overturn its extra-cycle
      loss.
- [x] Build policy only from cells with complete evidence.
- [x] Allow features: realized C, remaining output horizon, context/page bucket,
      profile, memory fit, and a predeclared bounded acceptance statistic.
- [x] Forbid prompt text/hash, token IDs, category, heldout identity, task
      result, and post-hoc oracle selection.
- [x] Fingerprint the table and evidence links.

### P9.2 Wider work admission

C8, R24/R32, adaptive K, or proposal/target overlap begins only when:

- at least one fixed C1/C2/C4 cell already exceeds 1.10x true AR;
- a current profile shows a measured premise;
- physical ownership is not simulated by multiple hidden weight sweeps; and
- the new cell has a correctness oracle and strict fallback.

No such premise means K0 and no implementation. Although repaired C1/K2 passes
the fixed threshold, no current profile supplies an operation-complete premise
for C8, R24/R32, adaptive K, or overlap. P9.2 admits **no wider work**; only the
C1/K2 P9.3 product packet proceeds.

### P9.3 Product packet

The sole fixed candidate fails the first product precondition. Boundary gates
find a narrow economic rectangle: D8 is only **1.058x AR**, D25 **1.409x**,
D64 **1.582x**, and D128 diverges deterministically 3/3 at token 81; p128
D25/D64 is **1.291x/1.506x**, while p512 D25 is **1.008x** with a category
regression. Long p4K/p16K K0 remains exact/no-mutation.

Under real automatic HTTP serving, the normal capacity-4 owner reports selected
D25/p128/SSE routes but every selected row has **zero speculative cycles** and
`specdec2_mtp2_used=false`; rows carry
`physical_streaming_category_rejected` and complete through AR. Therefore the
candidate does not engage the measured capacity-1 C1 path and is rejected before
load/SLO/soak. Response labels without cycle engagement are not evidence.
Artifact: [`P9 product no-go`](../benchmarks/results/2026-08-26-gfx1151-specdec2-perf-p9-product-no-go.json).

For the proposed automatic cell:

- [x] full train/heldout/category quality and task gates (fixed/profile evidence);
- [x] counterbalanced same-host true-AR complete economics;
- [x] output-horizon/context boundary rows;
- [x] exact automatic route/reason/result reporting exposes non-engagement;
- [x] fixed/ragged/delayed admission/refill/retirement: not applicable after
      route-engagement precondition fails;
- [x] mixed AR/MTP neighbors and fairness: not applicable after precondition;
- [x] prefix reuse/COW and pressure/regrow: retained K0 evidence, no product cell;
- [x] cancel/deadline/EOS/stop/failure/circuit breaker/restart: not applicable;
- [x] blocking completion/chat and SSE completion/chat: rejected on route owner;
- [x] below/near/above Poisson load plus overload: not run for invalid route;
- [x] TTFT/ITL median/p95/p99, queue, E2E, exact/SLO goodput: not applicable;
- [x] memory high-water/fragmentation/final return: boundary runs return clean;
- [x] focused/final promotion soak: not run for invalid route.

Automatic promotion requires >1.10x true AR, every binding gate, and
non-regressive AR-neighbor SLO. No cell satisfies the complete contract.
Automatic remains K0; explicit production FP16 compatibility remains retained.

## 19. P10 — closure

Durable closure:
[`campaign artifact`](../benchmarks/results/2026-08-26-gfx1151-specdec2-perf-campaign-closure.json).

Retained runtime units are C1 short streaming activation, stable cycle slabs,
device proposal→target→GPU accept→selected commit, the exact physical Q4 small-
row route, conditional provider snapshots/repair, production FP16 compatibility
with eager target ownership and FP32 strict fallback, packed target reseeding,
and fault-only postcommit canonical rebuild. These are explicit/default-off
SPECDEC2 mechanics unless independently shared by AR/runtime paths.

**Promoted automatic cells: none.** Automatic C1-C32 remains K0. Explicit
strict/production C1/C2/C4 K1-K3 compatibility remains functional/default-off.
The closure artifact lists every context/horizon/physical/wider K0/no-go reason
and the sole reopen trigger. No public root README export is made.

- [x] List every retained implementation unit and default scope.
- [x] List every promoted automatic `(model/backend/profile/C/K/context/horizon)`
      cell (empty set).
- [x] List every explicit/default-off functional cell.
- [x] List every K0/rejected/no-go cell and durable reason/artifact.
- [x] Remove temporary P9 qualification policy/override; retain older
      prompt/oracle/fallback debt only where named removal gates remain.
- [x] Update `REFACTOR.md` for retained debt and remove resolved entries.
- [x] Update `KERNELS.md`, `PLAN.md`, `SPECDEC2.md`, and profile status where
      ownership changed; no new kernel lineage change belongs to P10.
- [x] Update benchmark README/changelog/artifacts for retained/rejected results.
- [x] Keep public root README unchanged because no product scope promotes.
- [x] Run milestone validation under the focused-repair rule.
- [x] Run Worklog2, benchmark sync, fixtures, registry, JSON/link, and diff
      checks.
- [x] Commit final closure, sync/merge current origin, push, and verify equality.
- [x] Publish mechanism-only no-repeat lessons to the gfx1100 ledger without
      transferring rates, thresholds, profile manifests, or bucket decisions.

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
3. Mark the corresponding durable phase checklist (and any session-local mirror) `in_progress`.
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
15. Mark the phase complete and inspect the durable dependency graph before starting the next item.

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

## 24. Closure handoff

P1-P10 are closed with zero automatic promotion. Retain the production FP16
explicit compatibility surface and strict FP32 fallback; automatic C1-C32 stays
K0. The fixed capacity-1 C1/K2 premise is **1.4087x AR**, but normal capacity-4
server ownership rejects singleton staged execution and runs AR, so it is not a
product cell. No adaptive K, C8, R24/R32, or overlap work is admitted from this
no-go.

Reopen C1 product work only after independently qualifying true singleton staged
execution on a normal capacity>1 server owner with nonzero cycle/kernel/commit
engagement. Reopen physical widths only for a new target representation or
dataflow that can close the measured operation-complete target cost. Every
reopened candidate repeats the full anti-gaming horizon/context/product packet
against same-profile true AR.

The independent gfx1100 lane remains separately governed. Shared-file edits—
especially `qwen35_gguf_mtp2.py`, `qwen35_gguf_nextn.py`,
`qwen35_gguf_runner.py`, and the SPECDEC2 source-of-truth docs—must be serialized
through one merge owner. Reusable architecture and no-repeat lessons transfer;
gfx1151 rates, thresholds, profile manifests, policy fingerprints, and graph
buckets never do.

## 25. Post-closure exact small-M target update

The merged gfx1100 R6 recovery reopened only the standard-Q4 physical projection
owner, not product policy. On gfx1151, the generic rowtile transfer was rejected
because it changed BF16 output bytes. The retained replacement is a strict
one-wave/one-16-row-tile WMMA sibling that preserves the prior K16 FP16-WMMA,
FP32-accumulation, and BF16-store schedule while removing invalid 16-row tiles.

Actual Qwen3.8 Q4_K_M weights are exact in **28/28** R6/R8/R12/R16 shape rows.
Six shapes win **1.755x-2.424x** and now use the small-M owner; narrow
`5120→1024` loses at **0.651x-0.656x** and retains shared-B. Clean strict
operation profiles reduce R6/R8/R12/R16 target wall by
**20.07%/19.25%/17.68%/16.30%**. The complete ten-prompt Q4_K_S gate is exact
in all **120/120** parent/candidate C2/C4 cells and improves staged throughput
**9.958→11.462 (+15.10%) / 15.718→17.555 tok/s (+11.69%)**, with every
category positive and zero candidate D2H/recovery/final tracked ownership.

This is a retained scoped kernel/default improvement, not a campaign or product
reopening. Physical strict SPECDEC2 remains **0.7510x/0.6218x true AR** at
C2/C4, so automatic K0 and all closure no-go decisions remain unchanged.
Evidence: [`2026-08-27-gfx1151-specdec2-smallm-q4-wmma-retained.json`](../benchmarks/results/2026-08-27-gfx1151-specdec2-smallm-q4-wmma-retained.json).
