# MTP and CONCURRENCY2 Recovery — W7900

- Status: **C2 acceptance and R6 target repaired; still below AR; recovery active**
- Hardware lane: AMD Radeon Pro W7900 / `gfx1100` / host `epyc`
- Current source baseline: `dc919ca1f`
- Scope: Generation-2 AR plus dense-GGUF and packed-PARO SPECDEC2
- Normative architecture: [`CONCURRENCY2.md`](CONCURRENCY2.md),
  [`SPECDEC2.md`](SPECDEC2.md), and [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md)

This is the compact recovery campaign ledger. It does not replace the historical
mechanism notebooks in `MTP.md`, `MTP-gguf.md`, `CONCURRENCY.md`, or the immutable
worklogs. It ranks current work from matched current-source evidence.

## 1. Executive finding

There is no demonstrated raw Generation-2 AR throughput collapse under the old
fixed-C p512/d128 protocol:

| AR row | Old fixed-C | Current G2 | Raw ratio | Correctness status |
| --- | ---: | ---: | ---: | --- |
| C1 blocking | 72.169 | **77.176 tok/s** | **1.069x** | 3/3 exact |
| C1 SSE | 72.169 | **76.925 tok/s** | **1.066x** | 3/3 exact |
| physical C8 blocking | 158.542 | **161.882 tok/s** | **1.021x** | **16/24 strict rows; diagnostic only** |

The C8 raw rate is not a strict result. Varied D128 composition differs from an
independent C1 trajectory. Eager and graph submission produce the same
per-row mismatch schedule, queue1 and queue2 both reproduce it, and the first
captured hidden difference is decode step 1/layer 0. This is an execution-profile
and batch-composition arithmetic boundary, not evidence of scheduler, graph,
PM4, or host-service overhead.

SPECDEC2 has a different status:

- dense C1 K1/K2/K3 is exact at `1.272x/1.407x/1.439x` AR but remains
  2.7%-3.9% behind direct MTP;
- packed-PARO production C1/K1 is exact at `0.979x` AR / `0.960x` direct;
- gfx1100 physical C2/C4 capabilities remain false and automatic policy is K0;
- target-hidden/cursor repair plus exact R6 projection routing raises physical
  C2 from `0.544x` to `0.7156x` AR, with 10/10 exact and zero D2H/recovery.

Therefore the efficient recovery order is **profile contract -> C1 activation ->
PARO physical MTP -> remaining exact dense gate/up target work**, not generic
fusion or acceptance tuning.

Evidence:
[`AR/MTP recovery profile`](../benchmarks/results/2026-08-26-w7900-mtp-concurrency2-recovery-profile.json),
[`C2 root cause`](../benchmarks/results/2026-08-26-w7900-specdec2-c2-acceptance-root-cause.json),
[`C2 R6 target`](../benchmarks/results/2026-08-26-w7900-specdec2-c2-r6-target-rowtile-retained.json), and
[`gfx1100 SPECDEC2 closure`](../benchmarks/results/2026-08-25-w7900-specdec2-perf-campaign-closure.json).

## 2. What transfers from prior campaigns

### Retain and reuse

- One `EngineService`, Generation-2 fairness, resource claims, target frontier,
  transactions, committed output, cancellation, and K0 planning.
- Stable proposal/repair/accept/result slabs and zero hot allocation.
- Device candidate handoff, GPU accept, selected state/KV commit, and bounded
  provider repair.
- Dense N1R/N2/N3P graphs, packed PARO strict/production manifests, and strict
  fallbacks.
- Full category+heldout anti-gaming suite, profile gates, lifecycle/fault
  matrices, and current profiler children.
- Current AR physical owners as profiling controls, subject to an explicit
  execution-profile contract.

### Do not revive without a new premise

- The synchronous SPEC-C5 C=10 whole-generation wrapper. It is historical
  `0.579x` evidence, not the landed staged execution architecture.
- Singleton MTP loops presented as physical concurrency.
- Launch-count-only KV batching, generic fusion, extra queues, or graph/PM4
  tuning without operation-complete wall evidence.
- Dense physical target tuning while draft acceptance is near 12%-18%.
- Prompt/token-conditioned acceptance policy or single-prompt promotion.
- gfx1151 rates, thresholds, row policies, or manifests transferred to W7900.
  P7's after-root provider snapshots are reusable source, but its 4.0x repair /
  1.2%-1.4% request gain transfers only after an independent W7900 profile.

## 3. Current bottleneck ledger

| Priority | Lane | Measured bottleneck | Consequence |
| ---: | --- | --- | --- |
| 0 | AR C>1 | Legacy/default execution has no named gfx1100 strict/production/batch-invariant profile; D128 differs from C1 from layer 0 | Raw C8 speed is useful diagnostic capacity but cannot be labeled strict |
| 1 | Dense C1 SPECDEC2 | Activation: sampled staged target prefill + NextN priming `339.3 + 25.3 ms` vs direct prefill `305.9 ms`; staged cycle `416.6 ms` is already faster than direct decode `433.5 ms` | Optimize shared prompt/target-hidden/provider activation before target leaves |
| 2 | Packed PARO C1 | Production trails AR only 2.1%, with 80.92% accepted drafts and qualified T2 verifier | Best candidate for physical C2 once direct C2 ownership is refreshed |
| 3 | Dense physical C2 | Exact hybrid cuts target/accept 38.45% and raises physical throughput 31.93%, but remains 0.7156x AR; single-wave gate/up is now ~58 ms/cycle | Keep K0; require a materially different exact gate/up schedule and complete reprofile |
| 4 | Dense C1 residual | Required target/accept synchronization plus graph wall; no individual target primitive projects the campaign admission threshold | Reprofile only after activation contraction |
| 5 | Multi-slot provider arithmetic | Reject repair now follows C1; one full-accept continuation can still differ because physical provider KV/hidden is not C1-equivalent | Treat as profile/acceptance headroom, not the dominant 51% target-cost blocker |
| 6 | Packed physical C2/C4 | gfx1100 direct C2 exists, but request-major MTP proposal/frontier and c4/c8 owner symmetry are unqualified | Implement only after matched AR C2 control and state oracle |

## 4. Recovery phases

### R0 — Audit and matched profile — complete

- [x] Index current MTP/SPECDEC2/CONCURRENCY2 docs, 299 named worklogs, and
      retained/rejected artifacts.
- [x] Re-run old-protocol C1/C8 AR on current source.
- [x] Separate raw rate from strict profile eligibility.
- [x] Join final dense/PARO C1 SPECDEC2 evidence and target profiles.
- [x] Publish compact profile artifact.

### R1 — Define gfx1100 AR execution profiles

Goal: stop conflating byte-identical cross-width strictness with accepted
production arithmetic and batch-composition invariance.

1. Register explicit gfx1100 Qwen3.6 GGUF `strict` and `production` plans;
   `batch_invariant` stays unregistered and falls back to strict until its gate.
2. Bind every physical width/quantized projection/GDN/attention/sampler owner in
   the manifest with a strict fallback and evidence artifact.
3. Run C1/C2/C4/C8 p128/d8 and p512/d128:
   - exact control ownership and finite state;
   - strict same-C1 IDs where binding;
   - production strict-teacher KL/top-1/task thresholds;
   - batch-composition and neighbor-isolation as separate gates;
   - deterministic three-repeat IDs per declared width.
4. Publish only profile-qualified rates. Do not disable the fast C8 owner merely
   because legacy no-profile execution fails the strict cross-width gate.

Exit: a user can request strict, production, or batch-invariant and receive an
immutable manifest, declared fallback, and correctly labeled throughput.

### R2 — Recover dense C1 activation

Goal: make staged complete wall match or beat direct MTP without changing the
already-positive hot cycle.

1. Measure the same final target-hidden row and provider priming boundaries in
   direct and staged owners under one loaded model.
2. Reuse direct target prefill/output-normalized hidden capture instead of
   paying a second staged activation boundary.
3. Batch/stream provider prompt priming without prompt replay, extra host rows,
   or first-cycle allocation.
4. Require exact IDs/state/KV, carried-row ownership, all categories, p128/p512,
   lifecycle, and three counterbalanced runs.
5. Admit the change only if complete staged wall matches direct or materially
   reduces the measured activation owner; hot-cycle-only wins are insufficient.

Target: close the current 2.7%-3.9% staged-to-direct gap before another dense
kernel candidate. The merged P7 after-root snapshot path remains enabled source,
but its gfx1151 physical gain does not establish a W7900 C1 activation win.

### R3 — Recover packed PARO C1, then physical C2

Packed PARO is the higher-value physical MTP lane because C1 acceptance is
80.92% and complete wall is only 2.1% behind AR.

1. Refresh the current production/strict C1 AR/direct/staged packet.
2. Reprofile target verify/readback and activation under the qualified
   `decode_batched` production manifest; retain `c1_loop` strict fallback.
3. Freeze the existing gfx1100 direct selected-batch C2 AR/target oracle.
4. Add request-major C2/K1 proposal with one provider group, R4 target rows,
   row-specific KV/state journals, one group accept, and independent commit.
5. Gate reject/partial/full, staggered arrival/refill, neighbor cancellation,
   following-cycle continuity, and zero final ownership.
6. Run full categories only after a multi-prompt screen projects >=1.10x AR or a
   credible path to the repository promotion floor. C4 follows C2; do not start
   c4/c8 symmetry first.

### R4 — Dense physical C2 acceptance and R6 target repair — complete; economics blocked

The physical provider-refill premise was valid but incomplete. Differential
tracing proved initial candidates and GPU accept correct, then localized later
collapse to stale target-attached state: packed NextN cursors conflated consumed
position with the next input, and physical selected commit omitted the selected
pre-output-norm BF16 target trunk row. The retained repair owns both surfaces.

Full-suite results on W7900:

| Horizon | K | Acceptance | SPECDEC2 / AR | Exact |
| --- | ---: | ---: | ---: | ---: |
| D8 | 1 / 2 / 3 | 68.75% / 62.38% / 63.37% | 0.560x / 0.587x / 0.622x | 10/10 each |
| D24, repaired target | 2 / 3 | **76.92%** / 56.85% | **0.544x** / 0.534x | 10/10 each |
| D24, retained R6 hybrid | 2 | **74.28%** | **0.7156x** | 10/10 |

K2 is the fixed-cell winner. Complete profiling found shared-B standard-Q4
owned 1,282.637/1,802.702 ms (71.1%) of eight R6 cycle windows because its
four-wave body computes a 256-row tile for rows6. The retained hybrid routes
five actual shapes to a rows6 owner that is bit-identical to two C1 R3 rowtiles
and gate/up to the current-bit-exact single-wave sibling; shared-B remains the
registered miss fallback.

Clean full-suite C2 improves **16.974->22.393 tok/s (+31.93%)**, target/accept
**17.664->10.872 s (-38.45%)**, and **0.544x->0.7156x AR**, with 10/10 exact,
zero candidate D2H/recovery, and every category positive against the repaired
route. The unrestricted six-shape rowtile reached 0.861x AR but failed one
heldout strict ID and was narrowed. The next dense premise is a materially
new exact schedule for the remaining ~58 ms/cycle gate/up owner; the existing
dual-WMMA+SiLU composite is rejected at 0.795x its exact control. Physical
capability and automatic policy remain false/K0.

### R5 — Continuous product gate

Only after R1 plus at least one winning fixed speculative cell:

- C1/C2/C4/C8 fixed and ragged schedules;
- mixed AR/MTP neighbors, refill, cancellation, prefix, pressure, and soak;
- aggregate/per-request throughput, queue, TTFT, ITL, E2E, SLO goodput, memory;
- actual `(C,R,K,profile,provider)` decomposition and fallback reasons;
- full category+heldout/task quality against true same-protocol AR.

Automatic MTP requires every binding gate and >1.10x true AR in its declared
scope. The project target remains >1.3x. No wider scope inherits a C1 result.

## 5. Immediate execution queue

1. **R1 RED:** add gfx1100 Qwen3.6 GGUF profile registration and prove
   `batch_invariant -> strict` fallback before measuring another AR rate.
2. **R2 attribution:** run matched direct/staged activation markers on dense K2
   and design one shared target-hidden/NextN priming owner.
3. **Dense C2 gate/up profile:** start from the retained R6 hybrid and test only
   a materially different exact K5120/N17408 gate/up schedule. Do not retry the
   rejected dual-WMMA composite or the strict-failing all-rowtile scope.
4. **R3 control:** refresh packed-PARO direct selected-batch C2 under current
   production/strict manifests; implement physical PARO only if its C1 premise
   survives.

The next queue starts from measured target lowering and activation ownership;
it does not reopen acceptance heuristics, generic fusion, or prompt-conditioned
policy.
