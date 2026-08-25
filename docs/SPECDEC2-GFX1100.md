# SPECDEC2 S7 / gfx1100 Integration Plan

- Status: **active; C1 foundations plus W7900 performance P1-P3 checkpoint retained**
- Started: **2026-08-25**
- Hardware lanes: **AMD Radeon Pro W7900** (binding/default) and **RX 7900 XTX** (independent diagnostic)
- Shared architecture base: [`SPECDEC2.md`](SPECDEC2.md) S1-S6 at `82af2b6a4`
- Dense GGUF target: `/models/gguf/Qwen3.6-27B-Q4_K_M.gguf`, BF16 KV
- Packed PARO target/provider: `/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16`, W4A16 target + BF16 MTP sidecar, BF16 KV
- Normative gates: [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), [`TESTING.md`](TESTING.md), [`BENCHMARK.md`](BENCHMARK.md), [`KERNELS.md`](KERNELS.md)

This is the independent gfx1100 S7 capability ledger. It maps and qualifies
**both** dense GGUF MTP2 and packed-PARO MTP2 on the backend-neutral Generation-2
SPECDEC2 owner. The active implementation/performance punchlist is
[`SPECDEC2-PERF-GFX1100.md`](SPECDEC2-PERF-GFX1100.md). The lanes share frontier,
transaction, scheduler, and lifecycle records; they do not share provider
capabilities, physical buckets, numerical profiles, policies, or performance
evidence.

## 1. Frozen decisions

1. `EngineService` / `ResidentEngineLoop` remains the only request and visible-
   output owner.
2. AR is ordinary `K=0`; every automatic gfx1100 cell stays K0 until an exact
   local cost/policy artifact promotes it.
3. GGUF and PARO use separate exact four-axis capability identities.
4. Strict fallback remains registered for every candidate:
   - GGUF: exact AR plus strict staged/direct NextN oracle;
   - PARO: qualified corrected provider plus exact `c1_loop` target profile.
5. Start with C1. C2/C4 support requires physical cross-request proposal and
   target execution; request-serial loops cannot claim those cells.
6. W7900 evidence binds W7900 only. XTX is an independent diagnostic lane.
7. No gfx1151 absolute rate, K0 policy fingerprint, target row threshold, graph
   bucket, or FP16-state result transfers.

## 2. Starting controls

### 2.1 Backend-neutral

- [x] Read complete SPECDEC2 plan/research and gfx1151 S4-S6 closure evidence.
- [x] Raise package floor to Python 3.11 rather than adding `StrEnum` shims.
- [x] Run contracts, transactions, provider SPI, policy, simulator, frontier,
      engine-loop, and engine-service suites unchanged: **61 passed**.
- [x] Confirm complete claims, cancellation/recovery, fairness, multi-token
      publication, and final conservation are backend-neutral.

### 2.2 Dense GGUF control

`Qwen3.6-27B-Q4_K_M.gguf` is dense and contains all required NextN tensors.
The current direct B3 publication is exact and reaches `60.929 tok/s / 2.0684x`
its current true-AR baseline. It is a control, not staged SPECDEC2 evidence.

The landed `Qwen35GGUFMTP2Adapter` has no hardcoded gfx1151 arithmetic. It uses
shared raw-pointer NextN, transactional target, device accept, selected commit,
and gfx1100-source kernels. The package now exposes only C1 and resolves the
named strict dense-GGUF gfx1100 profile. Direct N2/N3P device proposal/target
owners are retained reuse controls; staged C1 still materializes proposal IDs on
the host and therefore has remaining device-boundary work.

### 2.3 Packed PARO control

The qualified standalone B1 production route is:

- final-normalized target-hidden provider input;
- selected-target-hidden reseed;
- borrowed full-vocabulary target W8A16 scorer;
- fast `decode_batched` production verifier;
- exact `c1_loop` strict profile/fallback.

Its canonical D24 evidence is exact `720/720`, `115.770 tok/s`, and `1.0446x`
true AR; its complete D64 numerical/repeat/task/state gate passes. This remains
a control until the same provider/target ownership executes through SPECDEC2.

`Qwen35ParoResidentModelRunner` now exposes the dedicated staged hooks used by
`Qwen35ParoMTP2Adapter`. Prompt final-normalized hidden rows stream directly into
NextN with one carried row, so there is no prompt-sized hidden owner. The target
verifier remains singleton and staged proposal/result ownership is not yet fully
device-resident; PARO still cannot reuse the dense adapter or claim C>1.

## 3. Capability map

| Lane | First exact capability | Proposal | Target frontier | Transaction | Initial policy |
| --- | --- | --- | --- | --- | --- |
| GGUF dense | `gfx1100/qwen_dense_gguf/mtp2/strict/c1/k1-k3` | Existing dense NextN; staged C1 host IDs with retained direct N3P device primitive | C1 R2/R3/R4, native/eager exact | provider journal + target reversible journal | explicit strict only; auto K0 |
| PARO packed | `gfx1100/qwen_paro/mtp2/production/c1/k1` | Corrected persistent BF16 sidecar; device scorer with staged bounded host-I32 handoff | C1 R2 fast packed scratch; strict c1-loop fallback | provider selected reseed + target packed scratch | production explicit/default only after staged parity; auto K0 during bring-up |

Future physical cells are separate gates:

- GGUF C2/C4, K1-K3, R4/R6/R8/R12/R16;
- PARO C2/C4, initially K1, R4/R8 using request-major candidates, per-request
  `KVLiveSpans`, parent-indexed Conv/GDN, and independent selected commit.

## 4. Phase graph

```text
S7.0 audit + backend-neutral proof
  -> G1 dense GGUF strict C1
  -> P1 packed PARO staged C1 K1
    -> SPECDEC2-PERF-GFX1100 P0-P10
      -> dense G2 and packed P2 physical C2/C4 + local policies
        -> S7 product/load closure and default decision per lane
```

G1 and P1 may proceed independently after this plan. G2 cannot use P1 evidence;
P2 cannot use G2 arithmetic or rates.

## 5. G1 — gfx1100 dense GGUF strict C1

### Implementation

- [x] Register a named gfx1100 dense-GGUF strict execution profile with exact
      FP32 recurrent state, BF16 KV, target capture, and stable manifest hash.
- [x] Expose only `GGUF_SPECDEC2_MTP2_C1`; keep C4 absent.
- [x] Generalize the existing c1 gate through model/backend arguments and retain
      exact W7900 provenance.
- [x] Confirm C1 capacity uses the staged adapter rather than direct whole-
      request generation.
- [x] Keep automatic policy K0; explicit strict K1/K2/K3 only.

### Gate

For K1/K2/K3:

- staged cold/warm IDs equal true AR and unchanged direct dense control;
- accept, selected hidden, Conv/GDN, touched K/V, positions/cursors, following AR,
  provider checkpoint/fingerprint, failure recovery, and teardown pass;
- native/eager route and exact strict fallback are reported;
- cached profiler proves provider, target, accept, and commit owners;
- complete same-host staged wall is compared against true AR and direct control.

**G1 foundation closed 2026-08-25:** K1/K2/K3 staged cold/warm IDs match true
AR and direct exact control; repeat provider fingerprints pass and all owners
drain. Warm staged/direct wall is `0.583/0.554/0.579`; K2/K3 beat warm AR on the
short d8 screen. This is not a full-suite speed claim, so automatic remains K0.
Evidence: [`gfx1100 GGUF C1 foundation`](../benchmarks/results/2026-08-25-w7900-specdec2-gguf-c1-foundation.json).

## 6. G2 — gfx1100 dense GGUF physical C2/C4

- [ ] Independently enable C4 only after physical proposal/target ownership is
      demonstrated on gfx1100.
- [ ] Cover C2/C4 K1-K3 and logical R4/R6/R8/R12/R16.
- [ ] Keep candidates device-resident through target; bounded D2H follows target.
- [ ] Prove proposal/target backbone counts are physical rather than one request
      loop per row.
- [ ] Validate per-request accept/selected state/KV, reject-neighbor, refill,
      cancellation, compaction, prefix COW, pressure, failure recovery, and drain.
- [ ] Measure target bucket/decomposition costs and build a gfx1100-only policy
      fingerprint. Select K0 for every losing or unqualified cell.

## 7. P1 — gfx1100 packed PARO staged C1 K1

### Implementation

- [x] Add a `Qwen35ParoMTP2Adapter` implementing bounded capability, claims,
      prepare/propose, target execute, commit/rollback, K0 catch-up, and close.
- [x] Add request-owned provider KV/cursor/checkpoint state and one carried
      device BF16 target-hidden row.
- [x] Stream each final-normalized target row into NextN during prefill without
      D2H, duplicate target weights, prompt-sized slab, or post-prefill replay.
- [x] Wrap `NativeMtpChainProposer` as one bounded C1 host-token proposal and
      pool the heavy proposer owner across requests.
- [x] Lower C1/K1 to R2 and call the qualified fast `decode_batched` verifier;
      preserve exact `c1_loop` through the strict profile/fallback.
- [x] Compose provider/target/result claims before mutation and expose manifest,
      candidate handoff, accept/commit, repair, recovery, and route telemetry.

### Gate

- staged IDs/accept/correction match the qualified direct fast control under the
  production profile;
- strict staged mode matches strict direct/AR where strict semantics require it;
- provider hidden/KV/cursor and target selected Conv/GDN/KV/position ownership
  pass reject/full-accept, forced rollback, following-AR, repeat, reuse, and
  teardown cases;
- complete canonical numerical/task/state packet remains inside the qualified
  production envelope;
- profiler proves corrected provider, fast target, accept, selected commit, and
  provider update; and
- complete staged wall is non-regressive versus the direct fast B1 control before
  any default owner changes.

**P1 foundation closed 2026-08-25:** production-fast and strict staged d8 IDs
match true AR, actual staged C1/K1 eager rows engage, streaming priming preserves
the shifted prompt contract, and warm requests reuse one proposer build. Warm
production staged wall is `0.731 s` versus `0.670 s` AR on the short screen, so
no performance/default promotion; automatic remains K0. Evidence:
[`gfx1100 PARO C1 foundation`](../benchmarks/results/2026-08-25-w7900-specdec2-paro-c1-foundation.json).

**Performance checkpoint retained 2026-08-25:** dense C1 K1/K2/K3 streaming is
exact at `1.259x/1.365x/1.419x` true AR but trails direct; packed production is
exact with zero allocation in 372/372 cycles but remains `0.933x` AR. Dense
proposal/repair stable slabs are exact and wall-neutral; graph first use remains
P4. p128/p512 streaming is exact but slower, and p4K/p16K selects K0 before provider
mutation.  No automatic/product scope promotes. Evidence:
[`gfx1100 P1-P3 checkpoint`](../benchmarks/results/2026-08-25-w7900-specdec2-perf-p1-p3-checkpoint.json).

## 8. P2 — gfx1100 packed PARO physical C2/C4 K1

PARO C>1 is not a quant alias of dense GGUF. It requires:

- physical request-major MTP proposal at C2/C4;
- target R4/R8 verifier buckets with device router/expert ownership;
- per-request canonical/provisional `KVLiveSpans`;
- parent-indexed Conv/GDN state;
- one device accept and independently selected commit per physical group;
- no cross-request expert/state/KV visibility; and
- complete same-host category/load economics against physical AR C2/C4.

Until those gates pass, C>1 PARO plans select K0 before mutation.

## 9. Profile and policy rules

- Dense GGUF strict and PARO strict/production manifests remain separate.
- Automatic policy resolution must be keyed by backend/model/quant/KV/profile/
  provider/policy fingerprint. The gfx1151 global K0 LUT is not reused.
- Exact generated IDs bind strict routes. Production PARO uses its calibrated
  numerical/task/state gate; cross-profile ID equality remains diagnostic.
- BF16-relative evidence is conditional on an available full-precision target.
- Every profile names exact registered variants and a strict fallback.

## 10. Required artifacts per retained cell

- physical host/GPU serial/backend/target arch;
- exact model/provider/quant/KV/profile/manifest hashes;
- C/K/logical R and honest physical proposal/target decomposition;
- provider/target/accept/commit/update/readback wall and kernel engagement;
- exact/profile quality, repeat, state/KV/isolation, lifecycle/memory;
- true same-protocol AR plus direct oracle wall;
- TTFT/ITL/E2E/SLO-goodput for any automatic/product claim; and
- compact artifact, benchmark rollup/changelog, and immutable worklog.

## 11. Stop rules

- Stop on any ownership/rollback/state/KV failure before performance work.
- Stop and add a numerical oracle before changing arithmetic.
- Do not expose C2/C4 from singleton loops.
- Do not infer gfx1100 capability from shared source or gfx1151 success.
- Reject losing cells after one attribution pass; retain K0 rather than weakening
  true AR or SLO gates.
