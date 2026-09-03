# Qwen3.8 gfx1100 C8/K3 Optimization Campaign

Status: **active; C8-P2 retained both Q5 MMQ owners (raw D4S4 R24/R32 plus FP32-metadata source-layout R32) and the canonical C8 row is 90.139 tok/s, confirmed at 91.041 tok/s in a fresh single-arm run at `6d6b9c555`; the remaining peer-derived Q5 layout gap is next.**

This campaign owns the sole remaining cell in the standardized W7900
Qwen3.8-27B `Q4_K_M` cross-engine matrix where hipEngine trails the strongest
HIP peer: explicit Generation-2 C8/K3 decode. It is a profiler-led performance
campaign, not permission to weaken correctness, change the benchmark workload,
or promote capacity-8 MTP automatically.

Authoritative evidence:

- [`Current C8 profile`](../benchmarks/results/2026-09-02-w7900-q4km-k3-c8-current-profile.json)
- [`Peer/role census`](../benchmarks/results/2026-09-02-w7900-q4km-k3-c8-peer-role-census.json)
- [`Current C1-C8 scoreboard`](../benchmarks/results/2026-09-02-w7900-qwen38-q4km-c1c8-current-scoreboard.json)
- [`Exact C8 R32 retention`](../benchmarks/results/2026-09-02-w7900-q4km-k3-c8-fused-row32-retained.json)
- [`Grouped Q5 R8 retention`](../benchmarks/results/2026-09-02-w7900-q4km-k3-c8-q5-grouped-r8-retained.json)
- [`Raw-Q5 MMQ retention`](../benchmarks/results/2026-09-02-w7900-q4km-k3-c8-q5-raw-mmq-retained.json)
- [`Accepted-tail K/V-only retention`](../benchmarks/results/2026-09-02-w7900-q4km-k3-c5c8-nextn-accepted-tail-kv-only-retained.json)
- [`Prior one-group attribution`](../benchmarks/results/2026-08-31-w7900-q4km-one-group-k3-c6c8-attribution.json)
- [`Execution profiles`](EXECUTION-PROFILES.md), [`testing`](TESTING.md),
  [`benchmark policy`](BENCHMARK.md), [`kernel catalog`](KERNELS.md), and
  [`roofline`](ROOFLINE.md)

---

## 1. Objective and closure levels

### 1.1 Frozen metric

The binding metric is aggregate generated tokens per complete server wall
second for the exact C8/K3/D24 ten-prompt suite, including all four categories
and four heldouts. It uses one physical C8 group, resident capacity 8, a 20 ms
batch window, raw greedy sampling, BF16 K/V, and the production execution
profile. Throughput is measured from the blocking barrier to the last request
completion; prompt processing, target verification, acceptance, publication,
and reclaim remain inside the boundary.

Current canonical rows:

| Route | Aggregate tok/s | hipEngine gap |
| --- | ---: | ---: |
| hipEngine explicit C8/K3 | **90.139** | — |
| llama.cpp current HIP C8/K3 | 94.735 | **+5.0986% required** |
| llama.cpp Laurent HIP C8/K3 | 101.072 | **+12.1289% required** |
| hipEngine true AR C8 | 83.939 | current K3 is 1.0739x |

The 90.139 row is the retained controlled default-versus-rollback gate at
`8c36a17e` (source-layout MMQ promotion, 20/20 prompt cells, both orders). A
fresh single-arm confirmation at `6d6b9c555` — after the later T16-dense C1
decode kernel work, which the C8 route does not dispatch — measured 91.041
MTP / 88.033 AR aggregate tok/s with every cell exact; the canonical row is
unchanged (benchmarks/results/2026-09-04-w7900-q4km-k3-c8-head-confirmation.json).

Closure is staged:

1. **Exact-peer closure:** exceed current llama.cpp HIP's 94.735 tok/s while
   preserving hipEngine's exact AR trajectory, acceptance, state, and lifecycle
   contracts.
2. **Strongest-peer closure:** exceed Laurent HIP's 101.072 tok/s under the
   same complete protocol, or record a measured structural blocker. Laurent's
   two deterministic C8 AR/MTP output differences must remain visible; they are
   not permission for hipEngine to weaken its own contract.
3. **Campaign closure:** refresh the full current/Laurent/hipEngine C1-C8 matrix,
   prove no retained shared change regresses C1-C7, and leave the automatic
   product policy truthful.

Every measured exact or fully qualified production-profile improvement is
retained and promoted even if it does not close a peer threshold by itself.
Small device-wall and launch-count wins compound.

### 1.2 Explicit diagnostic versus product routing

This campaign optimizes the **explicit C8/K3 engine path**. Capacity-8 automatic
requests currently select K0/ordinary AR at every width. The independently
qualified automatic Qwen3.8 key is resident-capacity-2 C2/K2 only. No C8 result
changes automatic policy unless a separate full economics, quality, serving,
and negative-key promotion packet passes.

### 1.3 Non-goals

- Do not change model weights, quantization, K/V format, prompts, or timing
  boundary to make the cell appear faster.
- Do not use prompt, category, token ID, candidate ID, or heldout knowledge in
  routing, vocabulary selection, or arithmetic.
- Do not substitute K1/K2/K4 for the fixed K3 cross-engine objective.
- Do not report one-prompt profiler throughput as a benchmark result.
- Do not compare absolute W7900 rates with RX 7900 XTX, gfx1151, CUDA, or a
  different GGUF artifact as old-to-new evidence.
- Do not add backend or quant conditionals to engine/model dispatch; use the
  four-axis registry and backend capabilities.
- Do not remove strict fallbacks for fused or production-arithmetic variants.

---

## 2. Frozen lane identity

| Field | Value |
| --- | --- |
| Physical host | `epyc` |
| CPU | AMD Ryzen 9 5950X |
| GPU | AMD Radeon Pro W7900, GPU 0 |
| Architecture | `gfx1100`, 96 CUs, wave32 default |
| ROCm runtime | 7.2.4 (`HIP 7.2.53211-3d9ef42`) |
| Profiler | rocprofv3 1.3.2 package (reports ROCm 7.15.0; traced HIP runtime 7.2.0) |
| Model | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` |
| File size | 17,106,773,984 bytes |
| Full SHA-256 | `7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b` |
| Weight quant | GGUF `Q4_K_M` |
| K/V | BF16 |
| Recurrent state | FP32 |
| Profile | `production`, with registered strict fallback |
| Benchmark suite | `benchmarks/prompts/mtpbench-code-general-ja.jsonl` |
| Binding shape | C8, resident capacity 8, K3, D24, max sequence 1,024 |
| Sampling | raw greedy, temperature 0, top-p 1 |
| Batch window | 20 ms |

The C8-P0 profile was captured tracked-clean at `b920ead8ee` with production
manifest `a7e5ad33...` and strict manifest `52a3d5b8...`. Every later candidate
records its own commit and manifest hashes; these hashes are not assumed stable
across arithmetic or registry edits.

---

## 3. C8-P0 current-source attribution

C8-P0 warmed every JIT object outside `rocprofv3`, required cached builds in the
profiled child, and traced kernel, ROCTx marker, HIP runtime, memory-copy, and
allocation domains. The workload is one train prompt at D12 and is attribution
only.

### 3.1 Phase wall

| Phase | Host wall/call | Kernel sum/call | Device interval union/call | Launches/call |
| --- | ---: | ---: | ---: | ---: |
| Full speculative cycle | **190.307 ms** | 173.329 ms | **159.622 ms** | 1,816.7 |
| Proposal | 22.286 ms | 18.136 ms | 16.464 ms | 288.3 |
| Target + accept + commit + provider | **167.862 ms** | 154.974 ms | **142.938 ms** | 1,528.0 |

The target composite is **88.21% of cycle wall**. Its device interval union is
**85.15% of target wall**. The 24.924 ms wall-minus-device-union gap is an upper
bound, not automatically removable host time: submission and blocking API
calls overlap queued GPU work.

### 3.2 Target kernel families

| Family | ms/cycle | Calls/cycle | Share of target wall | Share of kernel sum |
| --- | ---: | ---: | ---: | ---: |
| Q4 T16 | **72.620** | 249.7 | **43.26%** | 46.86% |
| Q6 T16 | **46.519** | 68.7 | **27.71%** | 30.02% |
| Q5 T16 | **15.052** | 48.0 | **8.97%** | 9.71% |
| GDN / linear attention | 8.304 | 96.0 | 4.95% | 5.36% |
| Commit/repair | 4.662 | 24.0 | 2.78% | 3.01% |
| Norm/RoPE | 1.891 | 151.7 | 1.13% | 1.22% |
| Copy/fill kernels | 1.072 | 492.7 | 0.64% | 0.69% |
| Full attention | 0.575 | 34.3 | 0.34% | 0.37% |

Q4+Q6+Q5 consume **134.191 ms/cycle**, **79.94% of target wall** and
**86.59% of summed kernel duration**. Short-context BF16 attention is too small
to explain the peer gap.

### 3.3 Largest target symbols

| Kernel | ms/cycle | Calls/cycle | Resources |
| --- | ---: | ---: | --- |
| grouped planar-Q6 R8 | **30.353** | 64.0 | WG128, VGPR112, LDS 1 KiB |
| grouped Q4 R6 | **25.309** | 96.0 | WG64, VGPR192, no LDS |
| fused Q4 gate/up+SiLU row32 | **23.461** | 42.7 | WG128, VGPR248, LDS 32 KiB |
| grouped Q4 R8 | **22.928** | 106.7 | WG64, VGPR224, no LDS |
| grouped Q5 R8 | 10.310 | 32.0 | WG128, VGPR72, LDS 512 B |
| planar-Q6 F32 rowtile R8 | 9.924 | 3.7 | WG128, VGPR112, LDS 1 KiB |
| segmented GDN state rows | 7.341 | 48.0 | recurrent attention state update |
| direct planar-Q6 F32 | 6.242 | 1.0 | WG128, VGPR96, LDS 512 B |
| grouped Q5 R6 | 4.742 | 16.0 | WG128, VGPR64, LDS 512 B |
| chunked linear-state commit | 4.301 | 8.0 | already independently compacted |

The trace contains two full R32 target cycles and one accepted-tail R24 cycle:
`[32,32,24]`. Therefore the R8 and R6 symbols are not interchangeable noise;
a candidate must identify whether it changes the two common R32 cycles, the
R24 tail, or both.

### 3.4 Host/API interpretation

Inside each target composite, rocprof reports approximately 39.3 `hipMemcpy`
APIs taking 113.984 ms and 1,008.3 `hipLaunchKernel` APIs taking 6.062 ms. The
memory-copy domain records **zero DMA operations**. Prior C8 tracing proved that
the long blocking copies carry required default-stream producer dependencies;
packing or making the first token H2D asynchronous did not improve complete
wall. HIP API durations must not be added to kernel time or treated as a
113.984 ms host opportunity.

There are zero in-cycle allocation operations, zero candidate D2H after target,
one physical C8 group, no serial fallback, no recoverable failure, and zero
tracked allocations after close. Scheduling, grouping, DMA, and allocation are
not the current primary blockers.

### 3.5 Progress from the prior one-group profile

The comparable one-prompt C8 diagnostic at `a81e42440` predates the accumulated
exact row/launch/state work:

| Diagnostic | Prior | Current | Change |
| --- | ---: | ---: | ---: |
| Cycle wall | 301.393 ms | **190.307 ms** | **-36.86%** |
| Target wall | 267.762 ms | **167.862 ms** | **-37.31%** |
| Target launches | 3,620.7 | **1,528.0** | **-57.80%** |
| Q4 | 125.361 ms | **72.620 ms** | **-42.07%** |
| Q6 | 69.024 ms | **46.519 ms** | **-32.60%** |
| Q5 | 19.123 ms | **15.052 ms** | **-21.29%** |

This comparison validates ownership movement but is not a canonical throughput
claim. The D24 ten-prompt row remains the only closure metric.

---

## 4. What the profile means

1. **No single missing C8 batching switch remains.** One physical group and
   exact R32/R24 rows are active.
2. **The original source-only ranking is superseded by the peer census.**
   Deleting all full-attention work would still save less than 0.4% of target
   wall, but the direct current/Laurent comparison identifies a materially
   different Q5 matrix-tile mechanism with an estimated **11.392 ms/cycle**
   upside. C8-P2 therefore tests Q5 before resuming Q4/Q6 work.
3. **Q4 has three distinct owners.** Grouped R6 dominates the R24 tail, grouped
   R8 dominates exact R32 projections, and the 32 KiB/VGPR248 fused dual WMMA
   owner dominates gate/up+SiLU. They need separate hypotheses and fallbacks.
4. **Q6 has both projection and head-shaped work.** P1 maps the 30.353 ms
   grouped-R8 family to recurrent QKV, full-attention V, and FFN down. The F32
   rowtile/direct symbols are the root full-vocabulary and selected proposal
   heads. Each role needs its own traffic and top-1 contract.
5. **Launch-only upside is bounded.** Target launch API time is roughly
   6.1 ms/cycle and dynamic host graphs already failed. Persistent/device-driven
   submission is considered only after kernel work exposes launch gaps or a
   prototype avoids dynamic recapture.
6. **The peer deficit has two scopes.** Current llama.cpp's 94.735 row is the
   exact-output peer. Laurent's 101.072 row is the strongest speed result but
   differs from its AR output in two C8 cells. The retained score artifact did
   not expose acceptance; P1's direct response timings now do, and separate a
   matched-output Q5 mechanism from the two arithmetic-different cells.

---

## 5. Retained route inventory

These mechanisms are part of the C8 baseline and must not be accidentally
bypassed by a candidate:

| Mechanism | Retained result / contract |
| --- | --- |
| One physical C8 group | Removed `[4,4]` serial ownership; explicit K3 only |
| Exact C8 target rows | R32 replaces padded R36; rollback restores R36 |
| Fused Q4 row32 | Two active waves preserve the fused transition at exact R32 |
| Grouped Q4 rowtiles | R8 at R32, R6 where the exact tail schedule requires it |
| Grouped planar-Q6 | Mixed/grouped rows and exact root R8 owners |
| Grouped Q5 R8 | Exact T16 fallback; Q5 launches 240→144 and C8 +1.61% in its gate |
| Raw-Q5 MMQ R24/R32 | Eight-request production default; Q5 operation wall 45.514→38.027 ms and C8 87.186→89.377 tok/s in its gate |
| Adaptive source-Q5 I64/J16-J32 | Default-off staging candidate; all-48 retained-owner screen saves 0.606 ms at R24 and 1.724 ms at R32 |
| Selected CJK-aware proposal head | Immutable model-bound 131,072-row selection; full category gate |
| Fused packed state | Amortizes Conv/recurrent state pointer gathers |
| Direct resident verifier state | Removes redundant initial state import |
| Accepted-tail K/V-only repair | Skips dead NextN Q/output/FFN/prediction work; future-state exact |
| Device acceptance | No candidate D2H after target |
| Strict fallbacks | Registered for every fused or production-profile owner |

Candidate traces must prove expected symbols and preserve all applicable
retained owners outside the intended replacement.

---

## 6. Closed or bounded directions

A closed direction may reopen only with the materially different mechanism in
the final column. Renaming a kernel, changing launch constants, or rerunning a
single prompt is not a new mechanism.

| Direction | Existing verdict | Reopen condition |
| --- | --- | --- |
| Per-layer or whole-verifier HIP graphs | Dynamic geometry created idle gaps/recapture; complete wall regressed | Prewarmed argument-update graph or device-driven submission with no measured recapture |
| Packed accept readback | Fewer APIs, target flat, complete wall +0.71% | Device consumer removes the producer dependency or tokens join an already-required payload |
| Async root-token H2D | Target -0.25%, complete wall +0.26% | Eliminate the dependency, not merely change API type |
| Singleton Q4 WMMA R32 | 0.165–0.959x actual roles | Fundamentally different weight/output ownership, not row-capacity specialization |
| Planar-Q6 WMMA R32 | 0.141–0.678x actual roles | Mechanism preserving weight reuse/ILP while changing ownership |
| Q4 R8 col4 | 0.898–0.987x actual roles | Reduce weight traversal, not only accumulator registers |
| Wider paired planar-Q6 WG256 | 0.860–0.907x | Reduce weight work or thread count; do not only coarsen scheduling |
| Q4 grouped R12 | Five of seven roles lose; full-K 0.631x | Lower per-thread state with actual-role leaf wins |
| Q4 grouped-R8 R24 broad route | Leaf/marker promise did not survive both full-suite orders | New body or narrowly proven cycle mechanism; unchanged policy rerun is prohibited |
| Q4 production fused-rowtile composition | C8 -3.63% despite passing numerics | Fewer launches and positive C8 complete wall, not an isolated leaf win |
| Q6 parallel epilogue / scale hoist / DPP alternatives | Category or actual-role gates failed | New arithmetic-preserving mechanism with every required scope positive |
| Q6 root col4 | Direct actual-role loss | New weight traversal or reduction mechanism |
| Grouped Q4/Q6 down+residual store | Q4 0.983–0.996x; Q6 ~flat | Composite first wins actual R24/R28/R32 FFN-down operation wall |
| Fixed 40,960-row sliced draft head | recall@16 0.883→0.720; Japanese acceptance collapsed | Materially different immutable model-bound selection passing all categories and heldouts |
| Prompt/category/token-conditioned routing | Invalid benchmark gaming | Never reopen |

---

## 7. Evidence ladder for every implementation candidate

### L0 — source and lineage

- Read the current `docs/KERNELS.md` path map and run
  `python3 scripts/check_lineage.py --kind kernel --diff stat` before kernel
  work.
- Name the current registered owner, strict fallback, exact operation shapes,
  rows, quant layout, output dtype, and arithmetic class.
- Cite external source file+commit if borrowing an idea; implementation stays
  in this tree.

### L1 — RED and actual-weight leaf

- Add an exact/parent-parity RED test for T0/T1 changes, or the declared
  production-profile numerical RED for T2.
- Screen every affected actual Qwen3.8 role at R24 and/or R32 in alternating
  order with HIP-event operation-complete timing.
- Stop before marker/full-suite spend if the incumbent operation wall does not
  improve or the candidate misses its declared arithmetic contract.

### L2 — current-route marker admission

- Warm every JIT object outside `rocprofv3`; use the compiler-version file plus
  `--require-cached-build` inside the profiler.
- Trace the final `specdec2_perf_bridge.py --profile-child`, never the nested
  prompt-suite parent.
- Require expected kernel name, ownership/row counts, exact IDs and acceptance,
  no extra schedules, zero failure, and clean close.
- Measure target wall, device interval union, kernel family/symbol wall, launch
  count, and complete wall. A launch reduction without operation-complete gain
  does not advance.

### L3 — full C8 task/economics gate

Use two fresh-process orders over all ten prompts and heldouts. Candidate and
rollback must each be tracked-clean and use the same commit. Require:

- candidate positive in both aggregate orders;
- every category and heldout slice non-regressive in each order;
- generated IDs, acceptance, target rows, ownership, and lifecycle equal for
  exact candidates;
- all packets pass and drain to zero tracked ownership;
- no best-of selection and no omission of losing cells.

### L4 — profile-specific quality

Any T2 arithmetic change additionally passes `docs/EXECUTION-PROFILES.md`:
strict-teacher mean/tail/max KL and top-1 by category/shape/transition,
repeatability, request-block permutation/isolation, state/K/V, BF16-relative,
and task gates. The artifact records production and strict manifest hashes.

### L5 — shared-route regression and publication

- If shared code can affect other widths, run the full C5-C8 gate; otherwise run
  focused C8 plus the genuinely affected narrow tests.
- At a peer-threshold crossing, recapture hipEngine, current llama.cpp HIP, and
  Laurent HIP through the full standardized C1-C8 protocol in counterbalanced
  process order.
- Update the compact result artifact, `benchmarks/README.md`,
  `benchmarks/CHANGELOG.md`, campaign punchlist, `docs/KERNELS.md`, refactor
  ledger, and immutable worklog; commit immediately.

---

## 8. Milestone plan and punchlist

### C8-P0 — freeze and profile current source

Status: **complete at `b920ead8ee`.**

- [x] Confirm clean source, W7900 GPU 0, ROCm, model identity, and compiler key.
- [x] Warm JIT outside the profiler and require cached builds in the child.
- [x] Capture kernel, marker, HIP runtime, memory-copy, and allocation traces.
- [x] Confirm one physical C8 group, `[32,32,24]` target rows, no candidate D2H,
      no failures, exact AR/MTP IDs, and clean final drain.
- [x] Publish the compact diagnostic artifact and this plan.

Decision at P0: target Q4/Q6 first; do not reopen batching, DMA cosmetics,
ordinary attention, or host-captured graphs from stale attribution. P1
supersedes only the quant-family priority by exposing a new Q5 matrix-tile
mechanism; the closed directions remain closed.

### C8-P1 — peer and semantic-role census

Status: **complete.** Evidence:
[`peer/role census`](../benchmarks/results/2026-09-02-w7900-q4km-k3-c8-peer-role-census.json).

- [x] Map each top C8 symbol/geometry to semantic roles and layers: recurrent
      gate/QKV, full-attention Q/K/V/output, FFN gate/up/down, SSM output, root
      and proposal heads.
- [x] Split attribution by the two R32 cycles and the R24 accepted-tail cycle.
- [x] Record bytes/weight shape, calls, kernel sum, interval union, VGPR, LDS,
      workgroup, and registry variant per role.
- [x] Directly profile prewarmed current and Laurent `llama-server` processes at
      C8/K3; drive requests from a separate harness so profiler/JIT state does
      not propagate through a nested parent.
- [x] Audit llama.cpp source commits and logs for physical batch, candidate
      depth, accepted progress, graph usage, quant kernel family, and F16-KV
      attention ownership. The standard response already exposes aggregate
      draft/accept counters, so timed-path instrumentation was unnecessary.
- [x] Separate the exact current-peer deficit from Laurent's two arithmetic-
      different C8 cells; never infer equal work from equal requested K.
- [x] Publish a cross-engine mechanism artifact before claiming that a specific
      hipEngine kernel explains the 101.072 tok/s row.

The hipEngine trace now maps every dominant geometry. The two R32 cycles use
fused Q4 gate/up+SiLU plus grouped R8 Q4/Q5/Q6; the R24 tail replaces fused
Q4 with grouped R6 work while retaining grouped Q6 R8 and grouped Q5 R6. The
artifact records every role's source weights/bytes, calls, kernel sum, interval
union, registry variant, workgroup, VGPR, LDS, and scratch. The compact cycle
split is:

| Target cycle | Q4 grouped | Q4 fused | Q5 SSM out | Q6 grouped | root/proposal heads |
| --- | ---: | ---: | ---: | ---: | ---: |
| R32 #1 | 33.770 ms | 34.700 ms | 15.243 ms | 32.154 ms | 10.730 / 6.244 ms |
| R32 #2 | 35.014 ms | 35.683 ms | 15.686 ms | 32.888 ms | 10.685 / 6.235 ms |
| R24 tail | 75.928 ms | — | 14.227 ms | 26.015 ms | 8.357 / 6.247 ms |

A matched-output `general_en_explain` profile holds requested K3, physical C8,
visible output, generated/accepted drafts (**152/128; 84.21%**), F16 K/V, and
19 graph launches equal. Current→Laurent decode span is
**1231.717→1098.313 ms (-133.403 ms)** and kernel sum is
**986.749→863.917 ms (-122.832 ms)**. Q5 alone changes from 384 MMVQ calls
and **131.651 ms** to 384 MMQ calls and **32.018 ms**: **-99.633 ms**, or
**74.69%** of the decode-span delta. Each side executes eight target rounds,
so the Q5 role moves **16.456→4.002 ms/round**.

The source cause is Laurent commit `25748619d26231137fa3add44d0a42d2c73c6003`:
it flattens recurrent GDN `ssm_out` input to a batch-wide 2D matrix before the
Q5 projection, selecting matrix-matrix quant ownership instead of per-sequence
matrix-vector ownership. Applying the measured **0.2432x** Q5 ratio to
hipEngine's 15.052 ms/cycle predicts **3.661 ms/cycle**, an estimated
**11.392 ms/cycle saving**. hipEngine already groups physical rows, so the
candidate is a true matrix-tile/MMQ analogue—not a host reshape or another
launch-only grouping change.

The separate known-different `code_merge_intervals` probe is kept diagnostic:
current generates/accepts **152/128** drafts, Laurent **168/120**, and their
output hashes differ. Thus Laurent's arithmetic-different cells cannot quantify
an equal-work deficit. The matched-output cell establishes the Q5 mechanism
independently; current llama.cpp remains the exact-output closure baseline.
Laurent's adaptive-draft commit is present but disabled in both profile commands.

Exit met: **recurrent Q5 `ssm_out`** is ranked first, its estimated saving is
11.392 ms/cycle, and matrix-tiled Q5 ownership is a source-grounded mechanism
not closed by §6.

### C8-P2 — peer-derived Q5 matrix tile, then exact grouped Q4

First pool: **15.052 ms/cycle** across the 48 recurrent Q5 `ssm_out`
projections. P1 estimates a matrix-tile analogue at **3.661 ms/cycle**, leaving
**11.392 ms/cycle** of mechanism-backed upside. Current grouped R6/R8 stays the
strict fallback.

- [x] Add an actual-weight R24/R32 RED comparing a matrix-tiled Q5 candidate to
      the current grouped-R6/R8 active-row output and production profile.
- [x] Prototype batch-wide Q5 matrix ownership using a generic DP4A/Q8-activation
      or WMMA mechanism; do not hard-code model data or bypass the registry.
- [x] Measure all 48 `ssm_out` weights in alternating order and require a
      projected operation-complete cycle wall below the incumbent 15.052 ms.
- [x] Keep exact grouped R6/R8 registered as strict fallback and declare any
      changed reduction/activation arithmetic as T2 rather than relabeling it.
- [x] Run L2→L5 whenever the actual-weight screen survives. Both surviving Q5
      owners completed the ladder and are retained: raw D4S4 MMQ at R24/R32 and
      the FP32-metadata source-layout MMQ at R32, together
      **15.052→10.940 ms/cycle** of Q5 operation wall.
- [ ] Close the remaining **7.280 ms/cycle** gap to the peer-derived 11.392
      ms/cycle Q5 target, or record the measured resource/roofline reason it
      cannot close, before changing C8 priority.

The first actual-weight screen now covers both available arithmetic families
and a peer-derived geometry prototype. On W7900 GPU0 with
`blk.0.ssm_out.weight` `[5120,6144]`, the operation-complete raw D4S4 path was
**0.2532/0.2763 ms** at R24/R32 versus grouped **0.2686/0.3112 ms** in its
preliminary run: useful R32 evidence, but only a modest R24/R32 candidate rather
than the peer's 4x mechanism. A source-compatible I64/J32/K256 prototype then
reproduced its I128/J128 parent output bits and passed the broad production
floor against grouped output (R24/R32 mean KL **7.87e-06/4.61e-05**, top-1
**95.83%/100%**). In the more conservative alternating run it measured
**0.2833/0.1899 ms** operation-complete versus grouped **0.2486/0.3047 ms**:
a strong R32 result but an R24 regression, so it remains unregistered.

The important residual is now concrete rather than geometric. The prototype's
no-tail BF16 code object uses **206 VGPR, 29 SGPR, 29,440 B LDS, WG128**;
Laurent's matched Q5 I64/J32 kernel uses **136 VGPR, WG128**. Directly changing
ownership therefore does not reproduce the peer's instruction/load layout.

The simpler existing D4S4 owner nevertheless cleared the complete ladder and is
retained. The tracked committed-harness all-48 screen measures
**12.983→11.686 ms (1.111x, 45/48 wins)** at R24 and
**16.111→13.483 ms (1.195x, 48/48)** at R32. Current-route tracing physically
replaces 144 grouped-Q5 launches with 144 quantizers plus 144 MMQs and moves
three-cycle Q5 operation wall **45.514→38.027 ms (-16.45%)**, target marker
wall **508.980→490.306 ms (-3.67%)**, and cycle device union
**477.443→457.526 ms (-4.17%)**. The strict-teacher R24/R32 gates pass all
**480** rows at 100% top-1 with max KL **0.005018**, three deterministic
candidate repeats, and exact teardown. The promoted-default task gate improves
**87.186→89.377 tok/s (+2.51%)**, wins all 20 prompt/order cells and every
category/heldout slice, preserves all visible generated IDs, and drains.

That measured **2.496 ms/cycle** Q5 saving is retainable but only **21.91%** of
the peer-derived 11.392 ms/cycle target. Continue Q5 by porting or reproducing
Laurent's shared-layout/load-dot schedule to reduce register pressure; do not
change priority yet. `HIPENGINE_GGUF_C8_Q5_RAW_MMQ=0`, request counts other
than eight, peer backends, and scope misses retain exact grouped R6/R8
ownership.

An initial all-width L5 audit caught that row-only R24 selection also reached
C5-C7. Although C5/C6 improved, C6 had a -0.04% reverse-order category and C7
was noise-flat with several negative slices, so that broad policy was rejected.
The retained selector additionally requires exactly eight request blocks. Its
fresh C5-C8 negative-control gate preserves C5-C7 generated IDs and acceptance
exactly with pooled rates +0.79%/+0.44%/+0.41%; C8 remains +2.72% and positive
in every category/heldout slice in both orders. This scope repair preserves the
eight-request R24 accepted tail without claiming non-C8 performance.

The follow-up source-layout reproduction fixes the rejected prototype's two
mechanical defects without copying its bad geometry: packed-Q5 loads are
coalesced to one high-bit word plus four low words per lane/output block, the
four K32 subblocks are partially unrolled, and the fast variants compile at
83/115 VGPR with zero scratch instead of the fixed-J32 prototype's 206 VGPR.
The source-faithful FP16-metadata build was the fastest variant - its retained
marker packet moves Q5 operation wall **12.806→9.514 ms/cycle (-25.71%)** - and
won every actual weight, but it failed the campaign's slice rule: its task gate
changed one acceptance decision (1,256→1,248 accepted drafts) and regressed
`general_en` by 1.38%/1.90% across orders. Keeping the same I64/J32 K-major
tiling while storing weight scale/min and activation metadata in **FP32**
removed that independent precision downgrade at a measured cost of **1.426
ms/cycle** of Q5 operation wall. Against the retained raw owner, all 48 actual
R32 weights win (11.330→11.025 ms, 1.028x) with mean/max KL 5.57e-11/2.98e-9
and 100% top-1, while R24 becomes 0.974x with zero winning weights and
therefore keeps the qualified raw MMQ. The route is retained as a T2 R32-only
owner behind the zero-valued `HIPENGINE_GGUF_C8_Q5_SOURCE_MMQ` rollback;
grouped T16 remains the fallback underneath it.

Its retained ladder measures Q5 operation wall **12.794→10.940 ms/cycle
(-14.49%)**, cycle device union **-2.55%**, target marker wall **-1.25%**, and
target device union **-2.32%** with identical 144-consumer/144-producer launch
counts, 96 generated tokens, zero candidate D2H, and zero recoverable failures.
Across the two retained-owner packets the Q5 pool has gone
**15.052/15.171→12.794→10.940 ms/cycle**, realizing **36.10-37.14%** of the
peer-derived 11.392 ms/cycle target; the two grouped controls come from
different packets, so the ladder is quoted as a range. The R32 strict-teacher
packet passes 240 rows at mean/p95/p99/max KL
**0.000158/0.000837/0.001535/0.005678**, 99.583% top-1 (minimum category
97.917%), three bit-deterministic repeats, and exact teardown. The promoted
default is 88.953→**90.139 tok/s (+1.33%)** with 20/20 prompt cells, every
category/heldout slice positive in both orders, identical generated IDs *and*
identical acceptance ledgers (568 cycles / 1,592 proposed / 1,256 accepted),
and a clean drain.

Second pool: **48.237 ms/cycle** across grouped Q4 R6 and R8 symbols. P1
ranks the R24-only R6 tail slightly above the two R32 R8 cycles:
**75.928/144.711 ms (52.47%)** of grouped-Q4 complete-profile exposure.

- [x] Use P1 to distinguish R24-only R6 work from R32 R8 work and rank by total
      complete-workload exposure, not per-call latency.
- [x] Inspect current ISA/resource counters for weight loads, cache behavior,
      VALU, stalls, and occupancy on each actual role. Hardware PMC counters
      are uncollectible on this stack (rocprofv3 `--pmc`/`-i` hangs before the
      profiled app ever launches, and rocprofv3 traps SIGTERM so a plain
      `timeout` cannot kill it — use `timeout -k`/`pkill -9`; see worklog
      entry for iteration 9). Inspection was completed via available means:
      code-object resources (VGPR 192/224, workgroup 64, LDS 0 — in the P1
      census) plus a measured wall-vs-roofline analysis. From the complete
      profile (benchmarks/results/2026-09-02-w7900-q4km-k3-c8-current-profile.json)
      the grouped Q4 R6/R8 owners traverse their encoded weights at only
      ~90–164 GB/s (e.g. 17.70 MB recurrent gate at 0.1466–0.1509 ms →
      ~117–121 GB/s; 50.14 MB FFN gate/up at 0.3664 ms → ~137 GB/s), i.e.
      ~11–19% of the 864 GB/s theoretical peak and well below the ~232–258
      GB/s the llama.cpp reference achieves on model-wide streaming. The
      "already bandwidth-bound" closure is therefore rejected by measurement:
      the kernels have headroom, and the residual Q4 upside must come from a
      mechanism that removes weight bytes or work (or raises achieved BW), not
      scheduling-only changes.
- [ ] Prefer a mechanism that reduces encoded-weight traversal, duplicate
      dequantization, or synchronization while preserving each row's FMA and
      reduction order. Kernel-body inspection (iteration 15) confirms there is
      no duplicate traversal or dequantization to remove within a call: the
      two waves of a T16 tile own disjoint column halves so each packed byte
      is read once, scale/min metadata is read once per block, and the
      dequantized weights are reused across all ROW_TILE rows. The measured
      gap to the roofline band is therefore latency/occupancy-bound
      (VGPR 224 at workgroup 64 caps resident waves), so an exact candidate
      must cut work or bytes (e.g. narrower metadata, lower VGPR pressure),
      not merely reschedule loads.
- [ ] Do not repeat col4, grouped-R12, singleton-WMMA, pair-major, or unchanged
      grouped-R8 policy screens.
- [ ] Admit a multi-output or cross-role owner only if it reduces actual work;
      merely combining independent owners into a wider workgroup is closed.
- [ ] Keep the current grouped-R8/R6 registered chain as strict rollback.
- [ ] Run L0→L3 and L5; retain any exact same-suite win.

Exit: retained Q5 and/or exact Q4 target-wall reduction, or role-specific
structural blockers with measured roofline/resource evidence.

### C8-P3 — fused Q4 gate/up+SiLU row32 track

Current pool: **23.461 ms/cycle**, WG128, VGPR248, LDS 32 KiB.

- [x] Profile per-layer distribution and confirm the expected 64 calls on each
      R32 cycle versus no unintended ownership on the R24 tail.
- [x] Inspect whether LDS capacity, VGPR pressure, barriers, weight decode, or
      WMMA issue is binding; do not assume occupancy from resource counts alone.
      Measured offline (iter29, rocprofv3 PMC remains broken on this stack):
      the row32 instantiation `<false,false,1,2>` allocates VGPR 241
      (next_free_vgpr; table's 248 is the rocprof-aligned figure), SGPR 28,
      LDS 32,768 B. 241 VGPRs/wave × 64 lanes = 15,424 of 16,384 per SIMD →
      1 wave/SIMD → 1 block/CU → **2 active waves/CU (6.25%)**; LDS would allow
      4 blocks/CU, so VGPR pressure binds, not LDS. Disassembly (1,583 insns):
      3 barriers, 8 WMMA, 66 LDS, 87 VMEM, 1,100 vector — Q4 T16 decode
      dominates; barriers and WMMA issue are not binding. Achieved BW =
      104.5 MB/call / 0.5499 ms/call = **190 GB/s** (weights 187). Cross-check:
      the row48/row64 siblings at 3–4 active waves measured 1.36–1.61x faster
      per row on actual weights, confirming latency-hiding (occupancy) as the
      limiter. A lower-VGPR sibling is the live axis; the ~209 non-accumulator
      VGPRs are decode temporaries, so the authorized instruction-layout axis
      must narrow decode state (accumulator floor is 32 VGPRs: 2×float8×2).
- [ ] Screen a lower-pressure or lower-barrier row32 sibling only when its
      arithmetic boundary is declared. Candidate axes may include LDS lifetime,
      gate/up staging order, active-wave scheduling, or instruction layout.
- [ ] Compare against the current two-active-wave fused owner on actual gate/up
      weights before whole-model numerics.
- [ ] Exact/parent-parity candidates follow L0→L3. Reassociated T2 candidates
      follow L0→L5 and keep the exact fused owner as strict fallback.

Exit: retained complete-wall win, or evidence that the current fused owner is
at its measured occupancy/bandwidth floor.

### C8-P4 — planar-Q6 projection/head track

Current pool: **46.519 ms/cycle**.

- [x] Role-resolve the grouped-R8 symbol (**30.353 ms**) separately from F32
      rowtile/direct symbols (**16.166 ms combined**).
- [x] Determine whether F32 symbols are target root/full-vocabulary heads,
      projection tails, or another boundary before editing them.
- [ ] For grouped R8, seek reduced weight traversal/dequantization or an exact
      reduction schedule with lower operation wall; WMMA capacity changes and
      WG256 pairing are closed.
- [ ] For head-shaped work, measure weight traffic, output traffic, and top-1
      ownership. Fusing exact argmax is admissible only if it removes material
      traffic or launches while preserving tie/finite behavior and full-vocab
      correctness.
- [ ] Preserve full-head fallback and every mapped-ID/finite/tie guard.
- [ ] Run L0→L3/L5 for exact work; L4 is mandatory for changed arithmetic.

Exit: retained Q6 target-wall reduction, or measured bandwidth/reuse floor.

### C8-P5 — proposal track

Current pool: **22.286 ms/cycle**, including Q6 10.406 ms and Q4 5.398 ms.
Target work remains higher priority until P2–P4 are exhausted or P1 proves a
peer proposal advantage.

- [ ] Profile proposal roles and selected-head traffic separately from provider
      state update.
- [ ] Optimize the exact current 131,072-row model-bound head/kernel before
      considering a new vocabulary representation.
- [ ] Reopen vocabulary selection only with a materially different immutable
      artifact and a four-category feasibility screen that clears recall,
      Japanese acceptance, and mapped-ID guards before the full suite.
- [ ] Never tune a row list or threshold to benchmark token IDs.
- [ ] Preserve full-vocabulary fallback and record acceptance/progress, not only
      head latency.

Exit: retained full-suite proposal win with unchanged acceptance, or a quality-
bounded artifact-research handoff.

### C8-P6 — device-driven submission and residual host wall

Current exposed upper bounds: target wall-minus-device-union **24.924 ms** and
launch API **6.062 ms/cycle**. These overlap GPU work and are not additive.

- [ ] Reprofile after P2–P5; proceed only if exposed non-device wall grows or a
      source-grounded device-driven mechanism is available.
- [ ] Do not retry dynamic host graphs or API packing.
- [ ] A persistent/device command stream must support R24/R32 geometry,
      argument updates, cancellation/failure, strict fallback, and clean close
      without measured recapture.
- [ ] Require marker wall improvement in both orders before a full suite.

Exit: retained operation-complete win or explicit deferral because the ideal
launch-only ceiling cannot repay complexity.

### C8-P7 — acceptance and product economics

This track begins only after engine-cost work is exhausted or comparative P1
shows accepted progress—not target execution—is the dominant peer difference.

- [ ] Measure accepted visible tokens per cycle, conditional P1/P2/P3
      acceptance, rejected-cycle cost, and category/heldout distribution.
- [ ] Freeze any generic confidence/budget/circuit-breaker rule before running
      the full suite; no prompt/category/token conditioning.
- [ ] Preserve target-model correctness, transactional K/V/state, RNG,
      cancellation, and deterministic repeat behavior.
- [ ] Report K1/K2/K3/K4 screens honestly; only K3 can close this campaign's
      cross-engine K3 cell.
- [ ] If C8 becomes an automatic candidate, open a separate production
      promotion unit covering full economics, quality, blocking/SSE,
      cancellation, overload/recovery, negative scopes, and drain.

Exit: either a fixed generic K3 policy win or a documented reason acceptance
cannot be changed without quality/gaming risk.

### C8-P8 — final closure

- [ ] Run the final tracked-clean C8 candidate/control two-order D24 suite.
- [ ] Run all affected focused tests and the applicable execution-profile gate.
- [ ] Recapture the standardized hipEngine/current/Laurent C1-C8 matrix in
      counterbalanced order on the same physical host.
- [ ] Require hipEngine C8 >94.735 for exact-peer closure and >101.072 for
      strongest-peer closure, using fresh peer rows if they moved.
- [ ] Confirm C1-C7 non-regression for shared changes and preserve the automatic
      capacity-8 K0 policy unless separately promoted.
- [ ] Update scoreboard, changelog, artifacts, kernel catalog, refactor ledger,
      plan, worklog, and commit.
- [ ] Mark the campaign complete, or record the quantified structural blocker
      and the exact condition that would reopen it.

---

## 9. Canonical commands

### 9.1 Current-source attribution

Use the same profiler SDK/runtime library environment as the captured packet,
then prebuild outside the profiler:

```bash
SDK=/home/lhl/mambaforge/envs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$SDK:$SDK/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
HIPENGINE_HIP_ARCH=gfx1100 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
PYTHONPATH=. .venv/bin/python -u scripts/specdec2_perf_bridge.py \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --backend hip_gfx1100 --target-arch gfx1100 --quant-label Q4_K_M \
  --execution-profile production --scope train --limit 1 \
  --concurrency 8 --service-capacity 8 --budgets 3 \
  --max-tokens 12 --runs 1 --max-sequence-length 1024 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --output /tmp/he-current-c8-profile/warmup.json
```

Profile only the final child:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
HIPENGINE_HIP_ARCH=gfx1100 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
PYTHONPATH=. rocprofv3 \
  --kernel-trace --marker-trace --hip-runtime-trace \
  --memory-copy-trace --memory-allocation-trace --output-format csv \
  -d /tmp/he-current-c8-profile/trace -- \
  .venv/bin/python -u scripts/specdec2_perf_bridge.py \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --backend hip_gfx1100 --target-arch gfx1100 --quant-label Q4_K_M \
  --execution-profile production --scope train --limit 1 \
  --concurrency 8 --service-capacity 8 --budgets 3 \
  --max-tokens 12 --runs 1 --max-sequence-length 1024 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --roctx-markers --profile-child \
  --output /tmp/he-current-c8-profile/child.json

.venv/bin/python scripts/specdec2_rocprof_summary.py \
  --lane 8 /tmp/he-current-c8-profile/child.json \
  /tmp/he-current-c8-profile/trace \
  --output /tmp/he-current-c8-profile/summary.json
```

### 9.2 Direct peer mechanism profile

Run one peer at a time. The helper starts `llama-server` under delayed rocprof
collection, finishes a C8 prewarm before collection begins, and drives the
measured request from the unprofiled Python parent:

```bash
python3 scripts/llamacpp_c8_k3_rocprof.py \
  --server /home/lhl/external-qwen38-bench/llama.cpp-upstream/build-hip-gfx1100/bin/llama-server \
  --source /home/lhl/external-qwen38-bench/llama.cpp-upstream \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --prompt-id general_en_explain --port 18123 \
  --raw-root /tmp/qwen38-current-c8-profile \
  --output /tmp/qwen38-current-c8-profile.json \
  --label current-hip-c8-k3

python3 scripts/llamacpp_c8_k3_rocprof.py \
  --server /home/lhl/external-qwen38-bench/llama.cpp-laurent/build-hip-gfx1100/bin/llama-server \
  --source /home/lhl/external-qwen38-bench/llama.cpp-laurent \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --prompt-id general_en_explain --port 18124 \
  --raw-root /tmp/qwen38-laurent-c8-profile \
  --output /tmp/qwen38-laurent-c8-profile.json \
  --label laurent-hip-c8-k3
```

These one-prompt rows are mechanism evidence only. They do not replace the
binding ten-prompt D24 scores.

### 9.3 Binding full C8 gate

Each candidate supplies an explicit same-build rollback variable or registered
fallback. Run both process orders; placeholders are recorded as templates, not
as-run evidence:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=. \
<CANDIDATE_ENV>=<0-or-1> \
.venv/bin/python scripts/gguf_mtp_c1c8_server_bench.py \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --backend hip_gfx1100 --quant Q4_K_M \
  --execution-profile production \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --widths 8 --resident-capacity 8 --expected-mtp-widths 8 \
  --max-tokens 24 --candidate-budget 3 --batch-window-ms 20 \
  --max-sequence-length 1024 --generation2-diagnostic \
  --capture-prefill-attribution --correctness-contract ar_exact \
  --output <packet.json>
```

Do not wrap this parent suite in `rocprofv3`; it launches child processes.

---

## 10. Iteration record template

Every C8 candidate worklog/artifact records:

1. hypothesis and why current profile selects it;
2. source commit, dirty state, model identity, hardware, ROCm, command, and
   manifest hashes;
3. incumbent and candidate registry keys, exact shapes/roles, arithmetic class,
   and strict rollback;
4. RED result, actual-weight leaf rows, order, samples, exactness/numerics, and
   operation-complete timing;
5. marker target/cycle wall, device union, kernel calls/family/symbol wall,
   expected symbol, acceptance, target rows, IDs, and final memory;
6. full-suite per-order aggregate, every category/heldout slice, prompt wins,
   generated/acceptance equality, ownership, failure, and drain;
7. keep/revert decision and the precise reopen condition for a rejection;
8. docs/artifact/refactor updates and atomic commit.

Raw profiler CSV and JIT `.so` files remain uncommitted; compact artifacts keep
hashes and reproducible commands.

---

## 11. Stop rules

- Stop on any wrong physical group, target rows, candidate budget, provider
  owner, execution profile, or strict fallback.
- Stop and localize any state/K/V/control bug; do not relabel it numerical drift.
- Stop exact candidates on any BF16/parent mismatch outside their declared
  contract.
- Stop production candidates on any binding KL, top-1, determinism, isolation,
  category, heldout, BF16-relative, or task failure.
- Stop before full-suite spend when actual-weight operation-complete timing or
  current-route marker wall is non-positive.
- Revert candidates that win an aggregate but regress a required category or
  heldout slice.
- Do not rerun closed geometry or API ideas without the documented materially
  different mechanism.
- Do not claim closure from stale peers, best-of runs, one prompt, profiler
  throughput, cross-host rates, or a different K/V/quant/model configuration.
- Do not promote explicit C8/K3 automatically without its independent product
  gate.
