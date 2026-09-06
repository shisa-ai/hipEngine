# Qwen3.8-27B gfx1100: faster MTP on the concurrency engine

Status: follow-up implementation plan; no new performance result claimed.

## 1. Objective and scope

Make multi-token prediction (MTP) useful again for Qwen3.8-27B GGUF `Q4_K_M`
on the W7900 through the Generation-2 concurrency engine. Implement **native
physical C1 MTP** on that engine and reduce the complete speculative cycle cost
at C2-C8. Faster target verification is the leading hypothesis, not a substitute
for measuring drafting, acceptance, commit/repair, and exposed host overhead.

This follows [`CONCURRENCY2-GFX1100-MTP-CN-PROMOTION.md`](CONCURRENCY2-GFX1100-MTP-CN-PROMOTION.md)
and the [2026-09-06 depth sweep](../worklog/entries/20260906T070547.769218Z-lhl-gfx1100-mtp-ck-sweep-k0-291573.md).
The Qwen3.8 automatic policy now selects AR, meaning ordinary autoregressive
decode without MTP. The previous C2/K2 and C8/K3 automatic speed claims are
withdrawn. Their safety-qualified explicit routes remain available for testing.
Qwen3.6 policies were not swept and must remain unchanged.

Deliver four independently reviewable outcomes:

1. A functional, tested C1 path using the concurrency engine's provider, target
   frontier, resource claims, and commit/rollback ownership. A legacy singleton
   runner wrapped in the scheduler does **not** satisfy this requirement.
2. Measured reductions in MTP cycle cost, prioritized at C8/K3 and C2/K2, then
   across the width/depth matrix. Keep qualified smaller wins even if a cell
   still loses to AR; such a cell remains explicit-only.
3. Functional K4, K5, K6 and K7 on the same native concurrency path at C1-C8.
   Generalize depth-dependent storage and execution rather than moving a
   hardcoded K3 limit to K4. Correct execution is required even if deeper
   speculation is slower; a diagnosis alone does not complete this outcome.
4. Automatic MTP only for same-host, full-suite winning keys with complete
   numerical and lifecycle evidence. Use K0 elsewhere, with recorded reasons.

Aim for 1.10x matched AR in a first winning scope and investigate the cost of
1.20x. These are planning targets, not predictions or minimum promotion
thresholds. A smaller demonstrated non-regressive win is still promotable
under [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md). Do not slow AR, restore an
old AR implementation, or select a weak denominator to manufacture MTP gains.

Binding lane: physical host `epyc`, AMD Radeon Pro W7900, `gfx1100`, one GPU;
model `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf`, BF16 KV. Record the actual model
hash, device UUID/PCI identity, software versions, and source revision at start.
The RX 7900 XTX, gfx1151, and [TP2 campaign](QWEN38-27B-GFX1100-TP2.md) are
independent lanes. No two-GPU work belongs in this campaign.

Notation: C = realized active request count; N = resident owner capacity;
K = maximum draft candidates per request; K0 = no speculation. R is logical
target frontier rows, `sum(1 + k_i)` over active requests; for uniform depth,
`R = C * (K + 1)`. P is the actual padded row count dispatched to kernels.
These are different from proposal rows (normally C at each serial draft depth).
KV means key/value cache; GDN means Gated DeltaNet recurrent attention; EOS
means end-of-sequence. A physical group is one staged batch transaction, not
C calls to a whole-request singleton loop.

## 2. Starting evidence and the cost target

The [sweep artifact](../benchmarks/results/2026-09-06-gfx1100-qwen38-mtp-ck-matrix.json)
records 20 measured cells, all below their own AR arms and all 10/10 token-exact,
under the canonical ten-prompt D24, greedy, 20 ms batch-window protocol. C2
used capacity 2; C3-C7 used capacity 8. This is **not** the complete 21-cell
C2-C8 × K1-K3 rectangle: `C2K3` is absent from the artifact. C1 is unmeasured;
K4 stalled. Preserve those distinctions in the follow-up matrix.

The artifact is a starting decision record, not a new benchmark run. It does
not contain full raw invocations/model-hash/variant provenance for each repeat;
Packet 0 must recover that provenance or remeasure, not invent it. In
particular, do not turn its ratios into a current verifier timing attribution.

Use a ledger with matching measurement boundaries:

- `A`: wall time for one true AR group step at the same C/N/context/profile.
- `T`: complete MTP group-cycle wall, including proposal, target, acceptance,
  commit, provider repair, and exposed synchronization.
- `U`: committed output tokens summed across that group cycle.
- `g = U / C` for a stable full group; `s = g * A / T` is its speedup.
  For dynamic groups/tails, use actual token counts and elapsed intervals
  rather than averaging per-cycle ratios or assuming every row survives.

At fixed token yield and AR cost, the total-cycle fractional reduction needed
for target speedup `s_goal` is `d = 1 - s_current / s_goal`:

| Historical cell | Reported MTP/AR ratio | Total-cycle reduction to break even | To 1.10x | To 1.20x |
| --- | ---: | ---: | ---: | ---: |
| C8/K3 | 0.9902 | 0.98% | 9.98% | 17.48% |
| C2/K2 | 0.9147 | 8.53% | 16.85% | 23.78% |
| C3/K3 | 0.8905 | 10.95% | 19.05% | 25.79% |
| C5/K3 | 0.8031 | 19.69% | 26.99% | 33.08% |
| C7/K1 | 0.7026 | 29.74% | 36.13% | 41.45% |

These are calculations from the cited ratios, not measured kernel targets.
“Cycle costs 24.5% more than its yield” at C5 is not “reduce current cycle
by 24.5% to break even”; the denominators differ.

If target verification accounts for fraction `f` of complete cycle wall,
verification alone must shrink by `d / f`. For example, **assuming** `f=0.8`,
C8/K3 needs about 12.5% less verifier time to reach 1.10x, not 10%. If `d/f > 1`,
that component cannot deliver the goal by itself. Measure f on this revision.
The historical “verify cycle” includes more than the target kernel interval.

For uniform depth B=`K+1`, also record target-only efficiency
`eta = target_verify_wall / (B * A)` at fixed C. Do not substitute R for B and
accidentally divide by concurrency twice. A multi-row target should reuse
weights across positions; repeatedly streaming weights per row defeats that
advantage. Sequential GDN state evolution still has to preserve causality.

Record proposed, accepted, correction/root/bonus, discarded, and committed
counts independently. `1 + K * acceptance` is only a cross-check when acceptance
means accepted/proposed and every cycle proposes K with one extra committed
token. EOS, ragged depths, fallback, and final-horizon clipping invalidate that
shortcut. Measure tokens committed and wall directly.

## 3. What can transfer from gfx1151—and what cannot

Use in-tree source and evidence as references. Before a kernel port, read
[`KERNELS.md`](KERNELS.md), run `scripts/check_lineage.py`, and identify source
file + commit. Qualify W7900 layouts, dispatch and performance independently;
do not copy backend capabilities, row thresholds, queue settings, or results.

| Reference and source identity | Useful experiment | Boundary / known counterevidence |
| --- | --- | --- |
| [`gfx1151 scaling campaign`](QWEN38-GFX1151-SCALING-CAMPAIGN.md), M2j `9d37394f2`; `hipengine/runtime/gguf_linear.py` and backend linear registrations | Route actual verifier Q4/Q5/Q6 shapes to existing row-amortized owners before writing kernels. Reuse weights across rows and reduce repeated per-row launches. | gfx1100 already has its own rows6, mixed-Q6, exact-R8/R28/R32, and fused gate/up paths. First prove a missing or slower owner in the current trace. |
| Same campaign, C6/R12 dual-Q4 owner `ff2e8423b` | Paired gate/up and shape-specific multi-row composition rather than two weight passes. | W7900's old R36 exact unfused gate/up screen lost; unpadding can lose a faster fused owner. Compare complete projection pairs and full cycle, not row count alone. |
| Scaling campaign W0-W5 and Y2/Y3 | Copy the method: target row-cost curves, measured memory traffic, and component savings bounds. Recheck whether current verifier shapes repeatedly decode the same quantized weights. | Wide-verifier pipeline/fusion proposals were bounded or measured losers there; later high-row prefill Q6 pair-decode improvements are not verifier wins. Neither “double buffer it” nor “one giant fused layer” is a justified default experiment without fresh W7900 evidence. |
| [`gfx1151 active-C1 route`](../worklog/entries/20260903T033431.189163Z-lhl-qwen38-gfx1151-30a341.md), `b58a70c82`; `hipengine/generation/qwen35_gguf_mtp2.py` | Reuse provider ownership/transition tests; investigate why small-row packed verification diverged. | This fix selects `Qwen35GGUFTransactionalVerifier` when alone. It is a diagnostic oracle/design reference, **not** the required native concurrency C1 implementation. Do not copy its bypass as completion. |
| [`gfx1151 C1 blocker correction`](../worklog/entries/20260903T001800.768446Z-lhl-qwen38-gfx1151-af1880.md) | First-divergent-boundary probes for C1 packed verification. | The earlier WMMA row-floor explanation did not fix the capacity-dependent issue. Do not assume a missing flag or floor is the root cause. |
| Scaling campaign M3/M4; `hipengine/runtime/qwen35_gguf_nextn.py`, `hipengine/generation/qwen35_gguf_mtp2.py` | Prompt-hidden streaming, proposal-head reuse, batched drafting per depth, and exact provider catch-up. | Prompt priming may change acceptance; classify it separately from an exact kernel speedup and run category/heldout gates. Serial draft depths cannot be parallelized without changing dependencies. |
| [`gfx1151 RDNA3 closure`](../worklog/entries/20260905T035406.988087Z-lhl-gfx1151-rdna3-application-closure-ed0e7d.md) and [`W7900 transfer audit`](20260905-gfx1100-audit.md) | Audit head chunking, final-prefill masks, and producer reuse for actual reachability. | W7900 head rows8 was already implemented before that audit. Final-prefill mask is not a decode-cycle win; persistent workers were unreachable on the staged product owner. Do not rerun rejected candidates without a changed premise. |
| [`RDNA3 tuning guide`](RDNA3-TUNING-GUIDE.md#62-multi-row-quantized-linear-and-verifier-paths) | Compare vector/dot row reuse, matrix-tile owners, register pressure, memory traffic and complete wall. | gfx1151's integrated-memory behavior and matrix-instruction details are not W7900 evidence. Generic speculative speedup projections are not targets for this model. |

The recent singleton-indexed GDN optimization is **one token per independent
active sequence**, not permission to treat dependent verifier positions as
independent sequences. Likewise, selected-expert reuse in MoE is not a dense
Qwen3.8 optimization merely because it was part of the same tuning work.
Reuse a primitive only after proving that its data and state contract applies.

## 4. Ordered implementation packets

Each packet requires RED tests where practical, focused validation, an
immutable worklog entry, and a scoped commit. Commit a completed unit before
starting the next. Proposed instrumentation/tests below are deliverables,
not assertions that commands already exist. C1 is required even if C8 finds
an early win. K4-K7 implementation must not delay valid K1-K3 improvements,
but the campaign cannot close with deeper depths merely diagnosed or disabled.

### Packet 0 — Freeze public truth, provenance and reachable shapes

- [ ] Confirm Qwen3.8 automatic K0 at every width, explicit admission only inside
  its safety envelope, and unchanged Qwen3.6 controls. Capture decline reasons
  with `HIPENGINE_MTP2_TRACE_DECLINE=1` outside timed measurements.
- [ ] Record model/hash, host/GPU identity, ROCm/compiler, execution profile and
  variant hash, KV/state storage, graph mode, environment, warmup and per-run
  commands. Recover the sweep's per-repeat source or label missing provenance
  and remeasure. Retain the original record without rewriting history.
- [ ] Emit a route map per (C,N,K): due IDs and slots, admitted K, logical R,
  padded P, active mask, physical proposal groups, target passes, kernel owners,
  accept capacity, graph bucket, fallback count and final drain. Inspect exact
  row policies; do not assume C8/K3 still dispatches padded R36.
- [ ] Reproduce C8/K3, C2/K2 and one inefficient middle width first, with matched
  AR in both process orders. Inventory all 24 C1-C8 × K1-K3 cells explicitly
  as engaged, rejected before mutation, unmeasured, or failing. C2/K3 needs its
  own status; no full-matrix claim from the existing 20 rows. This is the
  starting inventory; the final implementation matrix is 56 C1-C8 × K1-K7
  cells, plus K0 controls.
- [ ] Preserve maximum-depth safety admission: `_admit` chooses
  `min(requested, qualified)`. Keep the width/depth performance policy separate
  from safety evidence. Depth is a tuning axis **inside a proved implementation
  and resource envelope**, not proof that arbitrary depth is safe; K4's hang
  makes this distinction binding.

Exit: reproducible controls and a complete reachability map, not an automatic
policy expansion. For new shapes, use a fail-closed, explicitly unqualified
test candidate path if needed; do not fabricate public evidence to collect
its own qualification data.

### Packet 1 — Attribute complete cycle cost and select candidates

- [ ] Extend existing MTP telemetry/profiler harnesses with proposal per depth,
  target projection/attention/GDN/head, accept, selected state commit, rollback,
  provider repair, and host synchronization boundaries. Separate prefill/priming,
  first cycle, steady cycles, final clipped cycle, and ordinary AR fallback.
- [ ] Profile C8/K3, C2/K2, C3/K3 and C5/K3; add C1/K1-K3 after Packet 2.
  Add K4-K7 row/cycle curves as Packet 5 enables them, including C8/K7.
  Record kernel interval union as well as family sums. A blocking API's duration
  may be waiting for queued GPU work, not evidence of expensive copying.
- [ ] Audit fast AR versus MTP owners by role/shape: quant payload/layout,
  activation dtype, row tile, graph capture, GDN schedule, head chunking, and
  scratch. Identify which AR improvements are missing from MTP **and are
  semantically applicable**, rather than copying all AR defaults.
- [ ] Measure a target-only row-cost curve using actual model weights and warm
  recurrent/KV snapshots. Sweep R2-R4 and the real active/padded boundaries up
  through C8/K3. Include ragged tails. Record launches, bytes read/written,
  registers/local scratch, and active versus padded computation.
- [ ] Reconcile cycle accounting to unprofiled wall; report instrumentation
  overhead. Use the measured component fractions to set millisecond budgets
  for break-even/1.10x/1.20x. Select one dominant-cost hypothesis per unit.

Exit: an artifact that names the next operation, its caller and kernel, its
fraction of wall, expected maximum saving, and a correctness oracle. If target
math is not dominant, optimize the actual bottleneck instead.

### Packet 2 — Implement physical C1 on the concurrency engine

Required architecture: one request is a valid one-row provider group and one
R2/R3/R4 target frontier under the same staged adapter/resource transaction as
C>1. It must work at resident N=1 and when one survivor occupies any slot of a
larger resident owner. Reusing qualified leaf kernels is allowed; switching to
a whole-request singleton scheduler/verifier to avoid packed-state correctness
is not this deliverable.

- [ ] Add CPU seam REDs for C1 eligibility, claims, provider construction,
  frontier packing, selected commit and K0 transitions. Make tests fail if the
  candidate calls `_ensure_active_singleton_target_verifier`,
  `Qwen35GGUFTransactionalVerifier`, or the legacy singleton execution route.
  Scope that assertion to this new path; do not break existing gfx1151 or
  Qwen3.6 uses of those helpers.
- [ ] Replace the blanket `bound > 1` exclusion in `partition_max_requests`
  only through an explicitly qualified physical-C1 plan. Audit `claims_fit`,
  `_singleton_only`, provider construction and `_execute_target_frontier_batch`
  together. Do not allow a multi-request due batch to become serial singleton
  MTP calls because a C1 safety row exists.
- [ ] Implement native C1 provider/prompt-hidden initialization and packed target
  execution. Validate root token, positions, pre/post-norm hidden taps,
  embedding, convolution/GDN state and KV against independent strict reference
  trajectories from the first divergent boundary. Use warm states, not only
  position-zero smoke tests. Do not mask the divergence with a dtype flag.
- [ ] Handle R2/R3/R4 with bounded graph/workspace keys, inactive padding and
  exact ownership. Measure existing exact small-row owners before adding a
  specialized kernel. Production small-row arithmetic requires independent
  numerical qualification; C2 evidence cannot certify C1.
- [ ] Test K1/K2/K3, all accepted, first/middle/final rejection, EOS at every
  depth, final-horizon clipping, cancellation and recoverable failure. Restore
  both provider and target state and continue with the correct hidden seed.
  Extend every depth-sensitive C1 test through K7 in Packet 5; K3-only C1
  coverage is an intermediate milestone, not final depth support.
- [ ] Test N=1,2,8 and every physical slot: delayed C1→C2→C1, C8→C1→C8,
  sparse survivors, refill and compaction; K0→MTP→K0→MTP at transaction
  boundaries. Preserve IDs, page ownership, output/usage and clean drain.
- [ ] Add explicit evidence only after safety/numerical gates pass; measure
  against native C1 AR at the **same resident capacity**. Automatic stays K0
  until economics pass. A healthy legacy singleton rate is a diagnostic
  comparison, never the concurrency engine denominator or completion proof.

Exit: engaged native physical C1 with route-proof tests, numerical/state gates,
full-suite explicit measurements and capacity-specific policy decisions.

### Packet 3 — Amortize target verification across real frontier rows

- [ ] Prioritize C8/K3 and C2/K2 using Packet 1, then include middle-width controls.
  Compare actual R/P dispatch, not only C/K labels. Eliminate unnecessary
  padding only if the replacement keeps or improves the complete fused owner.
- [ ] Audit Q4 gate/up and QKV pairs, Q5 state-output projection, Q6 recurrent
  QKV/full-attention V/down projections, and target output head. First enable an
  already-qualified better owner if missing; otherwise develop one in-tree
  row-reuse/tile candidate with a registered strict fallback.
- [ ] Avoid one weight stream per verifier row. Compare current grouped/vector,
  row-amortized and matrix-tile implementations at actual layouts. Price the
  benefit against register pressure, scratch spills and extra reduction work.
  Do not reopen old small-row or unfused rejections without a changed shape,
  caller, implementation, or measured bottleneck documented first.
- [ ] Separate independent per-row projections from causal GDN recurrence.
  Batch/reuse projections across chain positions, then evolve each request's
  recurrent state in order. Optimize recurrence setup/state traffic only with
  exact intermediate-prefix/selected-commit tests; never substitute the AR
  singleton-indexed primitive for an entire dependent chain.
- [ ] Reuse resident descriptors, scratch and graph executables when lifetimes
  permit. Key structural graph layouts by C/K/R/P, profile/variant and buffer
  identity; refresh and validate dynamic positions/masks on replay rather than
  recapturing every token or reusing stale inputs. Measure replay versus eager
  without collecting full logits in the timed product path. Keep valid inactive
  tokens/masks and causal row mapping.
- [ ] Retain measured exact or fully quality-gated production gains with
  non-regressive complete affected cycles and current AR controls. A sub-window
  gain remains useful even before automatic MTP becomes profitable.

### Packet 4 — Reduce draft/head/commit costs where the trace warrants it

- [ ] Batch proposal work across the current C requests at each depth; keep
  token IDs and hidden tensors device-resident. Reuse projection/head work only
  for identical semantic inputs. Do not reuse target and draft head results
  merely because their weights are shared.
- [ ] Audit prompt-hidden streaming/NextN priming and provider catch-up. Measure
  their startup cost separately from steady decode; classify any acceptance
  change and validate the full category suite and heldouts. No prompt-specific
  IDs, candidate reranking or fixed-suite branching.
- [ ] Audit vocabulary head evaluations per verified row. Every row needed to
  decide acceptance/correction/bonus must be scored; the prefill final-row mask
  cannot discard verifier intermediate scores. Greedy device argmax and exact
  proposal-head reuse are candidates only where the current trace shows waste.
- [ ] If measured material, reduce duplicate snapshots, commits, copies and
  scalar readbacks through selected-prefix journals and persistent storage.
  Rejection, abort, sparse slots and K0 catch-up must remain exact. Do not
  optimize already-absent device-to-host transfers from an old theory.
- [ ] Screen overlap only for truly independent work on the actual staged
  owner. Do not add legacy generation pools/persistent workers to a path that
  does not use them. Count resource contention and complete wall, not inclusive
  overlapping kernel-time sums.

### Packet 5 — Fix K4 and implement depth-generic execution through K7

K4-K7 are required functional deliverables, not optional profitable cells.
Repeated use of the existing draft block can form a deeper chain; there is no
campaign-level algorithmic requirement to stop at K3. Buffer sizes, graph
buckets, kernel bounds and state handling still need implementation and proof.
This does not promise unlimited depth or useful acceptance at K7.

The reported K4 hangs ended after 1200 s and 3000 s without a completed prompt.
Localize that failure first; do not repeat full 20–50 minute sweeps or open
untested depths publicly. Keep the K3 public safety limit until replacements
pass their gates, then expose qualified deeper depths for explicit execution.

- [ ] Build a watchdog-bounded, cached-build reproducer with progress markers
  for initialization, prompt prefill, proposal depth, frontier build, graph
  capture/replay, target completion, acceptance, commit and drain. Save host
  stack and last completed GPU event on timeout. Begin with a few cycles and
  a small safe case, not the whole prompt suite. Distinguish cache/JIT stalls,
  graph deadlocks and out-of-bounds kernels; preserve evidence before recovery.
- [ ] Trace depth from server/request parsing through serving admission,
  adapter validation, provider unrolling, frontier metadata, target dispatch,
  accept, selected commit and provider repair. Audit hardcoded ranges, fixed
  arrays, unrolled loops, cached graph keys, masks and native ABI limits. Update
  CLI/environment diagnostics and tests along with the implementation; accepting
  a budget in the parser does not prove the runner can execute it.
- [ ] Make geometry derive from declared C, per-request depths and actual P.
  Keep a single implementation-depth limit, initially extending support through
  7, separate from qualified maxima and automatic performance policy. Derive
  memory claims before allocation; reject unsupported/resource-exceeding shapes
  before mutation. Do not replace K3-specific constants with K7-specific arrays.
- [ ] Inspect logical/padded boundaries, including the following C8 examples.
  The adapter computes `physical_accept_max_rows` dynamically; a fixed 36-row
  bound is not an established root cause. Exact R32 is separately enabled for
  C8/K3, so record actual dispatch rather than inferring it from this table.

  | Depth | Logical R = 8 × (K+1) | P if rows6 padding applies |
  | --- | ---: | ---: |
  | K3 | 32 | 36 |
  | K4 | 40 | 42 |
  | K5 | 48 | 48 |
  | K6 | 56 | 60 |
  | K7 | 64 | 66 |

- [ ] Size target/accept payloads, position/token arrays, logits or selected
  scores, recurrent snapshots, journals and hidden-seed storage for all active
  and padded rows. Audit 32/64-bit mask and index boundaries: P66 cannot fit
  in one 64-bit row-validity mask. Check actual representations rather than
  assuming such a mask exists. Keep inactive rows valid but noncommitting.
- [ ] Implement K4, then K5/K6/K7, including C1 R5-R8 and C8 R40-R64. Preserve
  one physical staged target frontier; qualified tiled/chunked kernels are
  allowed, but serial whole-request singleton calls are not. Graph and eager
  paths must agree under their declared arithmetic contract and terminate.
- [ ] Add CPU geometry/admission REDs and guarded GPU fixtures for every depth
  1-7, every C1-C8, ragged per-request depths, nonzero/sparse slots, and padded
  boundaries. Use guard sentinels for buffer overruns. Assert requested,
  admitted and executed depth; an explicit K7 test that silently runs K3 fails.
  Existing `min(requested, qualified)` admission may still cap a request, but
  must report that cap and cannot count as deeper-depth execution evidence.
- [ ] At each new depth test rejection and EOS at every candidate position,
  zero/all accepted, correction/bonus handling, output-horizon clipping,
  cancellation, rollback, retry and following-cycle state. Verify both target
  and provider state. Exercise K7↔K1↔K0 and C1↔C2 / C8↔C1 transitions,
  including resident N=1/2/8 and clean final drain.
- [ ] Qualify K4-K7 numerical/task behavior and explicit execution through the
  actual service owner, then measure the full category/heldout suite. Preserve
  existing Qwen3.6/gfx1151 safety limits unless independently qualified; raising
  shared implementation capacity must not widen their public evidence.

Exit: K1-K7 all execute correctly in the declared C1-C8/context/capacity matrix,
with real K4-K7 engagement, bounded resources and no hangs. A deeper cell may
lose to K3/K0 and remain explicit-only, but low speed or acceptance is not a
reason to leave it non-functional. A concrete blocker is a progress handoff,
not campaign completion. Valid K1-K3 wins can ship before this packet closes.

### Packet 6 — Re-sweep, select by width, and close the public path

- [ ] Evaluate K0-K7 at C1-C8, admitting deeper test cells as Packet 5 qualifies
  them. Record all 56 positive-depth cells, including explicit losing depths;
  automatic selection may still choose K0-K3. Run capacity-8 realized-width
  curves and separate N=1/2 controls. Preserve canonical N=2/C2
  and N=8/C8 comparisons; never present an own-capacity curve as fixed-N=8.
- [ ] Select depth from immutable model/profile/shape evidence, not prompt
  identity or observed benchmark token IDs. Ragged/remaining-horizon budgets
  must stay at or below requested and qualified maxima. Recheck policy after
  any shared kernel change that speeds AR as well as MTP.
- [ ] Test automatic winning cells and automatic K0 losing/missing cells before
  provider mutation; retain explicit safety-qualified cells for diagnostics.
  Add stable decline reasons, requested/effective K and physical engagement
  telemetry. No public safety-evidence bypass and no duplicate width/depth gate.
- [ ] Run delayed arrival, width/depth switching, cancellation, sparse refill,
  failure, streaming/non-streaming output, usage and final-owner drain through
  the actual service owner. Measure throughput, latency and queueing together;
  a long speculative cycle must not hide a serving latency regression.
- [ ] Preserve Qwen3.6 and gfx1151 model/backend policies with focused regression
  tests. Update only the qualified Qwen3.8 gfx1100 evidence keys, benchmark
  rollups and public claims. Keep every unsupported or losing key K0.

## 5. Change surfaces and validation

Read the relevant implementation before editing these shared files:

| Surface | Expected work / existing test anchor |
| --- | --- |
| `hipengine/generation/qwen35_gguf_mtp2.py` | Partition/admission, physical C1, claims, provider/target transactions; `tests/test_qwen35_gguf_mtp2_seam.py`, `tests/test_qwen35_gguf_mtp2_accept_staging.py`. |
| `hipengine/generation/qwen35_gguf.py`, `hipengine/speculative/serving.py`, `hipengine/models/qwen35.py` | Owner creation, safety versus performance policy, depth and evidence; `tests/test_speculative_mtp_serving_capability.py`, `tests/test_specdec2_engine_frontier.py`. |
| `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/runtime/qwen35_gguf_nextn.py`, `hipengine/runtime/gguf_linear.py` | Frontier row cost, hidden/state lifecycle, draft and target owners; existing packed-state/oracle tests plus new C1 boundary fixtures. |
| `hipengine/kernels/hip_gfx1100/` | Four-axis registrations and device source in this checkout, with strict fallbacks and applicable CPU-reference/numerical/profiler gates. |
| `scripts/gguf_mtp_c1c8_server_bench.py`, `scripts/gguf_mtp_verifier_rocprof.py`, `scripts/mtp_cycle_accounting.py` | Extend route/cycle evidence; verify the harness reaches the actual staged product route before relying on it. |

Do not introduce torch into runtime, backend/quant string branches into model
or dispatch code, or a second scheduler. Coordinate edits to shared owner,
server API and test files; use an isolated worktree for expensive measurements
when concurrent edits would contaminate provenance.

Exact state/control obligations in every profile: request/slot/output identity,
positions/masks, `KVLiveSpans`, page ownership, causal state and selected-prefix
destinations, provider cursor/hidden seed, graph/buffer ownership, RNG, resource
claims and transaction boundaries. No amount of numerical tolerance permits
one request to inherit another's state.

Strict candidates satisfy their declared exact/parent arithmetic contracts.
Production candidates require the calibrated teacher mean/tail/max KL, top-1,
repeat determinism, isolation, BF16-relative and task gates in
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md). Generated-ID equality across
arithmetic widths is diagnostic where that contract permits it; it is not a
replacement for numerical gates, and ownership failures are never drift.
The CPU-reference KL ≤ 0.05 and top-1 ≥ 90% kernel floor alone cannot promote a
production path. Initial greedy-only support must reject unsupported sampling;
do not claim stochastic correctness from an argmax test.

For new kernels: consult the catalog/lineage first, add the RED fixture, run the
applicable oracle/profile gate, and capture the expected kernel name/duration
with `rocprofv3 --kernel-trace`. Prebuild outside profiling and require cached
builds with a compiler-version file. Profile final leaf processes, not the
suite parent. Hardware tests need explicit HIP availability guards; follow
[`TESTING.md`](TESTING.md) for focused bundles and milestone closure.

### Reproducible starting command

This uses existing parser options, but was **not executed for this plan**.
Resolve the physical W7900 ordinal before using `0`, warm the build outside
profiling, and record the complete environment and source. It reproduces a
current explicit C8/K3 arm and its true AR control, not historical values:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 \
  .venv/bin/python scripts/gguf_mtp_c1c8_server_bench.py \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --backend hip_gfx1100 --quant gguf_q4_k_m --execution-profile production \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --mtp-request-mode explicit --widths 8 --resident-capacity 8 \
  --expected-mtp-widths 8 --candidate-budget 3 --max-tokens 24 \
  --batch-window-ms 20 --correctness-contract ar_exact \
  --output /tmp/qwen38-gfx1100-c8-k3-better-mtp-baseline.json
```

For the automatic-K0 control, change request mode to `automatic`, expected MTP
widths to `none`, and output path. These explicit `ar_exact` controls preserve
the sweep contract; a later production-arithmetic candidate uses the full
profile evaluator rather than weakening this check silently. Do not fabricate
a runnable C1 command until Packet 2 makes its physical route reachable.

### Binding measurement matrix

- Full `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, all `code`,
  `general_en`, `general_ja`, `mixed_ja_en`, plus category-heldouts frozen before
  tuning. Single-prompt profiles are attribution only; no acceptance/speed
  promotion from them.
- Canonical D24 and 20 ms window for comparison; add sustained D128/D512
  horizons and short/512/2048-token contexts with documented deterministic
  context construction. Extend maximum sequence length consistently. Longer
  contexts/horizons outside the existing safety envelope need qualification
  before public admission, not a benchmark bypass presented as product support.
- At least three independent balanced pairs after warmup, both arm orders;
  retain raw counts/walls and per-run ratios as well as the aggregation rule.
  Repeat further when a near-break-even result is within uncertainty. A ratio
  of separate medians is not a confidence interval or a verifier-only timing.
- Report committed tokens, elapsed wall, complete cycle and stage intervals,
  AR step time, acceptance by depth, actual R/P, group/pass count, fallback and
  engagement. Include per-prompt/category/heldout results, time to first token,
  inter-token p50/p95/p99, per-request fairness, throughput and peak memory.
  Separate burst emission intervals from amortized time per committed token.
- Compare each candidate to current same-host true no-MTP AR and incumbent
  explicit MTP, holding model, profile, storage, C/N, K, context, horizon,
  sampling and graph policy fixed as applicable. Verifier-derived B0 is not AR.
  If the candidate improves AR too, report both changes and the new ratio.

Publish compact artifacts under `benchmarks/results/` with model hash, host,
hardware, exact commands, versions, clean source, variant manifests, repetitions,
correctness and route evidence. Every retained measurement updates benchmark
README date/row and changelog; run `scripts/sync_benchmark_readme.py --check`.
Track experimental flags, rejected forks and removal conditions in
[`REFACTOR.md`](REFACTOR.md). Do not change historical immutable entries.

## 6. Completion audit

- [ ] Native physical C1 is engaged on the concurrency owner at the qualified
  capacities/slots; tests prove no singleton verifier/scheduler substitution.
- [ ] All 56 C1-C8/K1-K7 cells execute at their actual requested depth inside
  the declared safety envelope, with current true AR controls, full-suite
  correctness and complete-cycle attribution. C1 also passes N=1/2 controls.
  No K4-K7 hang, silent downgrade or diagnosis-only outcome counts as completion;
  unsupported contexts/resource sizes outside that envelope fail closed.
- [ ] Each candidate is retained, rejected, or blocked by a named measured
  prerequisite; no open-ended “tune verification” handoff and no repeated
  rejected experiment without changed evidence.
- [ ] Every qualified non-regressive improvement is kept and enabled within
  scope. Automatic MTP selects only winning cells; losing/unqualified cells
  select K0 before mutation. No universal-width or 1.10x hurdle discards a win.
- [ ] Public lifecycle/profile gates, Qwen3.6/gfx1151 policy controls, benchmark
  rollups, architecture links and immutable handoff agree. If no cell beats AR,
  report that negative result honestly; functional C1 and useful cycle savings
  are separate completed deliverables, not a claimed predictive speedup.
