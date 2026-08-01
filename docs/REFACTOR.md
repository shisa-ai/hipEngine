# hipEngine Refactor / Dead-Path Ledger

This file tracks cleanup work that should happen after the fast/correct path is
proven. During optimization, temporary flags and fallback paths are useful for
bisection; after the optimal path stabilizes, they become dispatch confusion and
should be removed or collapsed.

## Policy

- Exact, same-suite non-regressive performance wins should become defaults.
- Keep opt-out flags only while they are useful for rollback, bisection, or a
  named validation gap.
- When a flag is left in place, record the removal trigger here.
- Do not remove unfused numerical fallbacks required by `AGENTS.md`; remove dead
  runtime dispatch branches and stale experiment toggles first.

## Laguna gfx1151 source-F16 non-temporal decode comparison seam — closed

- Added 2026-07-31 as a gfx1151 default plus
  `--compare-f16-nontemporal-decode` same-resident profiling seam after all
  four natural leaves improved **3.015-3.509%** and all seven exact
  p512/d128 pairs improved **22.792512 -> 22.855773 tok/s (+0.27755%)**.
- Keep the comparison switch through tracked-clean selector-unset publication
  and the next source-F16/kernel-span census, then remove the CLI switch.
  Retain constructor `false` and the cached-load registered composites as the
  exact peer-backend/rollback path.
- The standalone single/triple/quad non-temporal wrappers are leaf diagnostics,
  not production dispatch. After the census freezes the family attribution,
  collapse any wrappers no longer used by a retained benchmark or exact
  regression test while preserving the production composite variants.
- **Clean publication passed 2026-07-31:** selector-unset production advances
  **22.780604 -> 22.856155 tok/s (+0.33165%)**. Keep the comparison seam only
  through the pending source-F16/kernel-span census, then close it.
- **Closed 2026-07-31:** the post-promotion two-queue census confirms the
  expected source-F16 production symbols and lowers their inclusive family
  time **24.538908 -> 24.362893 ms/token (-0.717%)**. Removed the completed
  CLI/protocol comparison seam. Constructor `false`, cached-load registered
  composites, and exact leaf diagnostics remain the rollback and regression
  coverage.

## Laguna gfx1151 exact global local1024 selector

- Added 2026-07-31 as a default-off `LAGUNA_GLOBAL_LOCAL1024` capability and
  `--compare-global-local1024` profile seam after exact 513/576/639
  correctness, an all-positive 21x100 leaf, and cached resource tracing.
- If any same-resident p512/d128 pair regresses or changes state, remove the
  capability field, runtime branch, and comparison switch while retaining the
  exact registered primitive/leaf route. If every pair improves, promote the
  gfx1151 capability, publish tracked-clean selector-unset production, and
  remove the comparison switch after the next attention census. Keep local512
  as the exact non-dense/eviction/peer rollback.
- **Promotion gate passed 2026-07-31:** all seven exact resident pairs improve
  **22.358675 -> 22.383414 tok/s (+0.11065%)** with a
  **0.059368-ms/token** paired-median saving. The gfx1151 capability is
  promoted. Keep the comparison switch through tracked-clean publication and
  the post-promotion attention census, then remove it.
- **Closed 2026-07-31:** tracked-clean selector-unset production reaches
  **22.378602 tok/s**, and the 127-transition census cuts global
  **0.453932 -> 0.402996 ms/token (-11.221%)**. Remove the dedicated
  comparison switch; retain local512 as the exact non-dense,
  explicit-eviction, and peer-backend fallback.

## Laguna gfx1151 exact SWA local1024 selector

- Added 2026-07-31 as a default-off runtime capability and
  `--compare-swa-local1024` profile seam after the exact dense-ring candidate
  passed RED/GREEN, the real wrap oracle, a positive 21x100 leaf, and cached
  resource tracing.
- If any of seven same-resident p512/d128 pairs regresses or changes state,
  remove the capability field, profile switch, registered local1024 route,
  wrapper, export, and body specialization. If all seven improve, promote the
  gfx1151 capability, publish a tracked-clean selector-unset result, then
  remove the comparison switch after the next attention census. Keep
  local512 as the exact rollback.
- **Promotion gate passed 2026-07-31:** all seven exact resident pairs improve
  **22.273482 -> 22.356330 tok/s (+0.37195%)**. The gfx1151 capability is
  promoted.
- **Closed 2026-07-31:** tracked-clean selector-unset production reaches
  **22.335681 tok/s**, and the 127-transition census cuts SWA
  **0.893032 -> 0.721795 ms/token (-19.175%)**. Remove the dedicated
  comparison switch; retain local512 as the exact fallback for non-dense,
  explicit-eviction, and peer-backend cases.

## Laguna exact SWA producer-gate selector

- Added 2026-07-29 as a default-off gfx1151 capability plus session/profile
  comparison seam after the registered primitive passed byte-exact
  wrap/eviction, leaf, and cached resource gates.
- If matched resident p512/d128 decode is negative or indistinguishable, remove
  the runtime capability, cache field, session setter, and
  `--compare-swa-producer-gate` switch while retaining the exact registered
  primitive for diagnostics. If positive, promote the architecture capability,
  publish a clean selector-unset result, then remove the dedicated comparison
  switch after the next attention census. Keep producer-max without
  producer-gate as the exact rollback.
- **Promotion gate passed 2026-07-29:** all seven exact resident pairs improve
  **19.992650 -> 20.012052 tok/s (+0.097%)**. The gfx1151 architecture
  capability is promoted. Keep the comparison switch through the pending clean
  publication and post-promotion attention census, then remove it.
- **Clean publication passed:** selector-unset production is
  **20.003064 tok/s**, **+0.0835%** over the prior packet, with exact repeated
  state/lifecycle. The comparison switch now remains only for the pending
  post-promotion attention census and should be removed in that logical unit.
- **Closed 2026-07-29:** the census confirms SWA **-0.252%**, kernel sum
  **-0.043%**, unchanged resources, and the intended producer-gate symbol. The
  profile comparison flag, session setter, and setter-only test are removed;
  retain the architecture capability, fail-closed cache field, registered
  primitive, and producer-max rollback.

## Laguna exact SWA producer-maximum selector

- Added 2026-07-29 as a default-off gfx1151 capability plus session/profile
  comparison seam after the separately registered primitive passed byte-exact
  wrap/eviction, leaf, and cached resource gates.
- If matched resident p512/d128 decode is negative, remove the runtime
  capability, cache field, session setter, and profile comparison switch while
  retaining the exact registered primitive for diagnostics. If positive,
  promote the architecture capability, publish a clean selector-unset result,
  then remove the dedicated comparison switch after one later attention
  re-profile. Keep the prior mixed32/exp32 kernel as the exact rollback.
- **Promotion gate passed 2026-07-29:** all seven resident pairs improve
  **19.684442 -> 19.996117 tok/s (+1.583%)** with exact state/lifecycle.
  Keep the comparison switch through the pending clean publication and
  post-promotion attention census, then remove it.
- **Clean publication passed:** selector-unset production is
  **19.983610 tok/s** with exact state/lifecycle. The comparison switch now
  remains only for the pending post-promotion attention census and should be
  removed in that logical unit.
- **Closed 2026-07-29:** the census confirms SWA **-21.36%** and kernel sum
  **-1.46%**. The profile comparison flag and session setter are removed in
  the census unit; retain only the architecture capability, current owner,
  and registered mixed32/exp32 rollback.

## Laguna long-context F32 hipBLASLt rollback routes

- Added 2026-07-27. The first capacity-sized full-score owner improved
  mandatory 128K **22.088%**, but widened the complete BF16 K/V prefix and
  materialized an F32 `[48,128,C]` score tile, costing **4.298 GB** scratch.
- The retained successor uses exact online softmax state across 4K key blocks.
  It improves the full-score owner another **12.521%** at mandatory 128K and
  cuts scratch **96.655%** to **143,753,216 bytes**. gfx1151 now selects
  `LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT`; the generic `KVLiveSpans` chain
  remains the required fallback.
- Remove the capacity-sized full-score owner, its
  `LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT` capability/session setter, and
  ceiling-only algorithm-override API after one later long-context publication
  confirms the block route is sufficient for rollback. Keep the bounded owner,
  its explicit rollback while SWA/query-chunk work is active, and the generic
  numerical fallback. Retire the dedicated ceiling harness after the
  block-size/algorithm policy no longer needs bisection.

## Laguna rolling-SWA hipBLASLt production route

- Added and promoted 2026-07-28 after exact wrap-oracle plus
  4K/16K/64K/128K gates. The fixed owner gathers a 639-key union and owns
  **33,554,432 bytes** scratch; decode, verifier, partial, evicted, and
  nonconsecutive paths retain the registered generic fallback.
- `LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT` and the session setter remain useful
  while LC-3 widens attention query chunks. Collapse the explicit selector
  after the wider-query route either absorbs this owner or proves the M128
  route remains independently optimal. Do not remove the span-aware fallback.

## Laguna global M2048 attention-query route

- Added and promoted 2026-07-28 after exhaustive M128..M2048 global/SWA
  screens and positive 4K/16K/64K/128K complete-model gates. The production
  capability `LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS=2048` applies only to
  complete global matrix chunks; SWA and partial tails remain M128.
- Keep the explicit constructor/profile override through LC-4/LC-5 so dense
  addressing and larger matrix chunks can be bisected independently. Collapse
  it into the final chunk policy after those stages choose their retained
  geometry. Never remove the generic span-aware fallback.

## Laguna dense-contiguous global-cache widen

- Added and promoted on gfx1151 in LC-4 after the exact 4K block sub-window
  improved **0.250249 -> 0.234780 ms (-6.181%)** and the complete-model gates
  remained neutral with exact state through 128K.
- Keep `prefill_dense_contiguous_cache`, the profile comparison switch, and
  the generic span-aware block kernel through LC-5/LC-6 for rollback and
  non-dense semantics. After one release plus the final six-shape sweep,
  collapse the selector if no regression appears. Never remove or weaken the
  complete `KVLiveSpans` fallback for continuation, verifier, eviction, SWA,
  decode, or unmeasured backends.

## Laguna M4096/M8192 explicit matrix diagnostics

- LC-5 lifts the constructor/profile validation ceiling to M8,192 but keeps
  gfx1151 production at M2,048. M4,096 fails mandatory 128K
  **149.684 -> 147.939 tok/s (-1.166%)** while adding **1.756 GB** scratch;
  M8,192 is directionally dominated and adds **5.268 GB**.
- **Closed 2026-07-28:** LC-6 did not reproduce a capacity-dependent short
  prefill loss, so no bucketed policy is justified. M4,096/M8,192 constructor,
  profile, and test support is removed; production and the validation ceiling
  are both M2,048. The LC-5 artifact preserves the rejected evidence.

## Laguna MoE shared/routed branch-concurrency candidate

- Added 2026-07-26 as an exact, default-off session and profile-harness
  candidate. For rows greater than one, the always-on shared expert runs on a
  nonblocking secondary stream while router selection and routed experts run
  on the caller stream. Two timing-disabled events preserve the producer and
  final-combine dependencies. Decode and the established single-stream route
  remain unchanged.
- The production-style Q4 GPU composition fixture is BF16-byte identical to
  the sequential route. The performance gate must use
  `GPU_MAX_HW_QUEUES=2` for both arms so queue count is not a confound, then
  prove an actual overlap in a cached trace rather than inferring it from wall
  time.
- If the clean queue-matched A/B is negative, remove the constructor option,
  profile flag, secondary-stream lifecycle, event plumbing, helper split, and
  concurrency fixture immediately. If it is positive, promote through an
  architecture capability only after complete-state/quality/lifecycle and
  traced-overlap gates; then remove the explicit positive selector after clean
  publication while retaining the required sequential fallback.
- The candidate trigger passes: seven queue-matched complete-state pairs move
  **560.837 -> 567.577 tok/s (+1.202%, 7/7 wins)**, clean default-off
  512/1K/4K reaches **565.457/525.733/443.027 tok/s**, and tracing proves
  **100.390/101.241 ms (99.16%)** of secondary kernel time overlaps caller
  kernels while kernel span falls **12.727 ms**. Promote through the gfx1151
  capability and two-queue process policy; retain the explicit selector only
  through clean selector-unset publication and one later MoE checkpoint.
- Promotion is complete in code: gfx1151 defaults the capability when at least
  two process queues are available, the process default is now two, an
  explicit one-queue override automatically selects the sequential fallback,
  and constructor/CLI rollback remains. Clean selector-unset publication and
  the follow-up MoE checkpoints are now complete; defer selector cleanup only
  through the lower-priority secondary-stream screen.
- Initial eager-concurrency selector-unset publication passed at
  **565.447/526.711/443.444 tok/s**. The production trace observes two
  queues/two streams, overlaps **76.883/77.763 ms (98.87%)** of secondary
  kernel time, and cuts the prior kernel span **11.265 ms**. Keep the explicit
  constructor/CLI and one-queue rollbacks through the scheduling cleanup while
  retaining the automatic sequential safety fallback.
- The delayed-launch checkpoint is closed. `before_down` preserved complete
  state but regressed **566.394 -> 565.011 tok/s (-0.244%, 2/7 wins)** and
  introduced a **535.465 tok/s** low tail. Its row-launcher, session, CLI, and
  test selectors are removed; no temporary phase-selection debt remains.
- The follow-up after-router checkpoint temporarily adds
  `moe_shared_after_router` at the row launcher, session, and long-profile
  CLI. It records the secondary dependency after router selection while
  retaining gate/up-plus-down overlap. The gate passes at **+0.073%, 5/7
  wins**, and tracing verifies a **0.310-ms** span reduction. Promote through
  a gfx1151 capability, then collapse redundant positive selection after clean
  selector-unset publication.
- Promotion now resolves `LAGUNA_MOE_SHARED_AFTER_ROUTER` from the gfx1151
  package; explicit constructor/CLI false remains rollback. Remove redundant
  public positive selection after clean selector-unset 512/1K/4K publication,
  while retaining the false rollback through the next MoE scheduling audit.
- Clean selector-unset publication passes at
  **566.839/527.381/444.447 tok/s**. Collapse redundant positive selection
  during the next MoE scheduling cleanup; retain the explicit false rollback
  while lower-priority secondary-stream scheduling is screened.
- The lower-priority checkpoint adds general HIP priority-range/query and
  priority-stream creation helpers plus a temporary
  `moe_shared_low_priority` session/CLI selector. gfx1151 reports
  `(least=+1, greatest=-1)`. Remove the Laguna selector if the complete-state
  gate fails; the general HIP helpers may remain because they are exact,
  tested runtime primitives.
- The checkpoint passes **+0.494%, 6/7 wins** and tracing cuts kernel span
  **7.255 ms**. Promote the least-priority selection through a gfx1151
  capability. After clean selector-unset publication, remove redundant
  positive CLI/session selection while retaining explicit priority-0 rollback
  through one later scheduling checkpoint.
- Promotion now resolves `LAGUNA_MOE_SHARED_LOW_PRIORITY` from the gfx1151
  package. `--no-moe-shared-low-priority` and an explicit constructor false
  remain exact priority-0 rollback. Clean selector-unset 512/1K/4K publication
  is the remaining promotion gate.
- Clean selector-unset publication passes at
  **568.849/527.113/444.508 tok/s**; 1K/4K are flat within
  **-0.051%/+0.014%**. The eager-priority1 checkpoint is now closed at
  **-0.198%, 1/7 wins**. Collapse redundant positive selection during the next
  MoE scheduling cleanup while retaining explicit priority-0 and sequential
  rollbacks.

## Laguna selected one-plane FP32-scale MMQ modes

- Added 2026-07-25 as explicit session-local LAP-3/LAP-4 candidates. The
  original `mmq64x32_d4_f32` route used one FP32 scale per 32 activations for
  selected Q4 gate/up and Q4/Q6 down. It crossed 350 tok/s but failed the
  complete quality gate at max KL 0.0767056. The repaired
  `mmq128x32_d8_f32` gate/up route stores one FP32 scale per 16 activations in
  the same 160-byte block and pairs with `mmq64x32_d4_f32` down. It retains
  direct/exact selected and grouped-down paths as required fallbacks;
  D4x2/D4x3 remain diagnostics.
- Promoted on gfx1151 after the clean category gate passed at max KL
  0.040724836, 317/320 top-1, 2.615x aggregate prefill, flat decode, and exact
  lifecycle recovery. Clean selector-unset publication now passes at
  **354.820 tok/s** median with all three pp512 samples above 350, and the final
  cached trace names the D8 128-column consumer. Retain the explicit
  exact/D4x2/D4x3 rollback modes for one release; the rollback window is now
  the sole removal condition for redundant experiment selectors. Remove the
  rejected D4 gate default candidate at that boundary. Pre-admission pp512
  samples were 353.951/356.082/356.473 tok/s.

## Laguna `f16_prefill_mode=hipblaslt_range_direct`

- Added 2026-07-25 as an explicit session-local LAP-6 candidate. Rows greater
  than one cast each BF16 producer row once into existing FP16-sized scratch,
  use an exact power-of-two row scale, run cached zero-workspace hipBLASLt
  FP16-input/F16-weight contractions with FP32 output, and restore the scale
  before the next boundary. Decode and the retained custom projection route
  remain unchanged.
- Promoted as the gfx1151 package capability by the clean compounded category
  gate. Clean selector-unset pp512/milestone publication and the final
  hipBLASLt trace now pass. Retain exact tiled as rollback for one release;
  remove the session selector after that rollback window.
- The 2026-07-26 exact range-qualified follow-up makes
  `hipblaslt_norm_direct` the gfx1151 default. Actual norm metadata proves a
  **16.34623** FP16 input bound; full logits, final/pre-final hidden, KV,
  cursor, token, and token-logit hashes match `hipblaslt_scaled`. Seven
  explicit pairs improve **502.348 -> 505.887 tok/s (+0.704%)**. Keep
  `hipblaslt_scaled` only as a one-release rollback for the range-qualified
  producer boundary, then collapse it while retaining the general scaled cast
  for the unnormalized attention-output projection.
- The second 2026-07-26 range proof computes per-layer value/gate maximum row
  L2 norms from the resident F16 source and applies Cauchy-Schwarz, FP32 dot,
  BF16-rounding, and 2x online-attention safety factors. The worst gated
  attention-output bound is **7,957.539**, still **4.116x** inside FP16 after
  the runtime's separate 2x admission reserve. `hipblaslt_range_direct` now
  removes the remaining output row reduction/scale and is the gfx1151 default;
  keep `hipblaslt_norm_direct` and `hipblaslt_scaled` only for the same
  one-release rollback window, then collapse both selectors.

## Laguna packed-query F32 hipBLASLt attention selector

- Added 2026-07-27 for the quality-gated two-call dense-initial attention
  route. `prefill_attention_hipblaslt_packed_queries=false` restores the
  admitted eight-QK/eight-PV route; unsafe attention shapes retain their
  established `KVLiveSpans` fallbacks independently.
- The route passes the tuned leaf, seven-pair pp512, CPU-reference,
  pp512-all-exact KL, cached trace, and clean selector-unset gates. Production
  improves **623.050/563.399/462.430 -> 629.101/566.858/463.903 tok/s**;
  tracing cuts pp512 attention **82.763 -> 73.330 ms** and dispatches
  **4,145 -> 2,417**.
- Retain explicit false through the next selected-projection checkpoint, then
  collapse the redundant positive selector/session rebuild path while keeping
  the whole-BLAS-route rollback and every unsafe-shape `KVLiveSpans` fallback.

## Laguna direct packed-query producer rollback

- Added 2026-07-27 as
  `prefill_attention_hipblaslt_packed_query_producer=false`. The gfx1151
  capability writes only the three qualified dense-initial M128 query tiles
  head-major from fused RMSNorm/RoPE; false restores the exact standalone
  generic-to-head-major query transpose. Generic row zero and every unsafe
  attention route are independent fallbacks, not part of this rollback.
- Eleven complete pp512 pairs are state-exact and improve
  **647.210 -> 650.651 tok/s (+0.532%, 7/11 wins)** with **1.557 ms** paired
  median wall saved. Clean tracing removes all **144 / 4.907-ms** query
  transposes, cuts dispatches **2,273 -> 2,129**, and improves the exact
  producer-plus-pack boundary **20.530 -> 16.666 ms (-18.82%)**.
  Selector-unset publication reaches **654.249/579.699/468.608 tok/s**.
  Keep explicit false through the next optimization checkpoint, then remove
  this one-purpose session selector and A/B comparison while retaining the
  generic producer for unqualified rows.

## Laguna wave-per-row causal-softmax rollback

- Added 2026-07-27 as
  `prefill_attention_hipblaslt_wave_rows_softmax=false` plus the session
  setter. The gfx1151 capability selects a local32 wave-per-row reduction;
  false restores the local256 block/LDS reduction. Unsafe shapes never enter
  either BLAS route and retain their established `KVLiveSpans` kernels.
- The candidate improves the qualified 48-layer attention model
  **72.738 -> 62.755 ms (-13.73%)** and seven-pair pp512
  **614.668 -> 620.032 tok/s (+0.873%, 6/7 wins)**. All-exact KL improves
  **0.002097 -> 0.001796**, top-1 stays 2930, and tracing reports
  local32/VGPR24/SGPR128/LDS0/scratch0.
- Clean selector-unset publication now passes at
  **632.618/568.845/464.606 tok/s**, and the required later corrected
  attention-family trace measures **69.983 ms** at unchanged **2,417**
  dispatches. The removal preconditions are satisfied. Remove the session
  setter and one-purpose A/B harness in the next rollback-cleanup unit while
  retaining the block256 kernel as the numerical fallback until the
  packed-query rollback window closes.

## Laguna `dense_q4_prefill_mode=wmma_pack8`

- Added 2026-07-25 as an explicit session-local LAP-5 candidate. Rows from 16
  use wave32 WMMA consumers over existing rank-2 Q4 pack8 bytes/effective FP32
  scale/min planes and raw Q6_K bytes. Q4 remains 64x16; Q6 now defaults to
  the exact scratch-free 16x32 tile after the 64x16 body exposed
  VGPR256/236 B-thread spilling. They add no sidecar, leave decode unchanged,
  and retain exact pack8/raw FMA paths as fallbacks.
- Promoted as the gfx1151 package capability by the clean compounded category
  gate. Clean selector-unset pp512/milestone publication and the final Q4/Q6
  WMMA trace now pass. Keep the exact fallback for one release; remove the
  session selector after that rollback window.
- `HIPENGINE_GGUF_Q6_K_DENSE_WMMA_TILE=64x16` is the temporary Q6 tile
  rollback added 2026-07-26. Remove the environment branch after the clean
  16x32 publication and one release window remain non-regressive; explicit
  `tile_m`/`tile_n` arguments stay for tests and microbenchmarks.
- `HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE=64x16` is the temporary Q4 pack8
  shape-policy rollback added 2026-07-26. Remove it after clean publication,
  a confirming family trace, and one non-regressive release window; explicit
  tile arguments remain part of the primitive test/microbenchmark surface.

## Priority Cleanup (do first)

**Revalidate the gfx1151 GGUF graph default on the current stack and retire
unrelated legacy graph blocks.** SOL-G5 reintroduced a state-bound runtime graph
with a complete transition key. gfx1151 advertises its measured 128-step
admission; gfx1100 now advertises a separately measured 24-step admission after
passing all 24 hidden/GDN/KV/token transitions on W7900. Non-streaming c1 greedy
generation uses the graph only at each backend's admitted horizon, with
`HIPENGINE_GGUF_DECODE_GRAPH=0` retained for rollback. The explicit
`qwen35_gguf_bench.py --graph-replay-decode` surface is required for the current
default decision, while `scripts/gguf_mtp_bench.py --target-graph-verify` /
`--target-graph-batched-verify` remains separate stale diagnostic plumbing.

SOL-G4 provides the correct comparison floor: clean p512/d128 eager is
`20.290 ms/token`, while a 24-step marker profile contains `18.402 ms/token` of
GPU kernels (`88.62%` of profiled host wall). SOL-G5's clean production route
at `7f611fe3` passed 128 launches and measured a capture-inclusive
`20.334 -> 20.311 ms/token` (+0.112% throughput) edge. The 2026-07-12 TheRock
HIP 7.15 refresh is still 128/128 exact but rejects the graph wall on both the
scalar parent (`20.5230 -> 20.5736 ms/token`) and wave/block candidate
(`20.4723 -> 20.5324`). Do not remove the rollback flag or broaden graph
admission. A separate scoped decision must either reproduce a current graph win
or restore eager as the gfx1151 production selector; the wave/block kernel
itself helps both routes and is not the cause of this graph-policy result.

The gfx1100 decision is independently strong: clean W7900 p512/d24 SOL-G5 at
`833921ce` passed 24/24 byte-exact transitions and measured capture-inclusive
`30.5364 -> 12.5139 ms/token` (**2.4402x**, five runs). Per-token recapture was
only `22.22 tok/s`; the retained route is one state-bound capture followed by 24
validated relaunches, not recapture. Keep the gfx1100 admission at 24 until a
shorter-horizon audit establishes a lower break-even.

## Cleanup Ledger

| Area | Debt | Current status | Removal trigger |
| --- | --- | --- | --- |
| Laguna source-F16 prefill selector | `HIPENGINE_LAGUNA_F16_PREFILL=auto|gemv|tiled|wmma_comp_swa` exposes the promoted scoped AR-O2 route plus exact LPF-1 rollback. | gfx1151 `auto` now selects compensated WMMA only for QKV/gate/O on the 36 SWA layers from M16. The clean gate moves weighted prefill 53.388->69.037 tok/s (+29.313%) at max KL 0.043888 and 318/320 top-1; every category, Poolside oracle, determinism, and lifecycle gate passes. All 12 full-attention layers, M2-15, rows=1, and unmeasured backends remain exact. The rejected direct WMMA route stays removed after max KL 0.097062. | After one release window and a defaults-only gfx1151 refresh, remove redundant positive `wmma_comp_swa` selector semantics and keep at most explicit `tiled`/`gemv` rollback. Never revive direct `wmma`, broaden the compensated scope without an independent full quality gate, or remove exact rows=1/unsupported fallbacks. |
| Laguna SWA token4-exact decode rollback | `LagunaGGUFResidentSession(..., swa_decode_variant=...)` and `allocate_laguna_kv_cache(..., swa_decode_variant=...)` retain baseline `swa_context_spans` beside gfx1100's backend-qualified `swa_context_token4_exact_spans` default. | Promoted on gfx1100. The exact 4-wave/4-slot schedule passes all focused wrap/eviction/KV/runner gates; clean SWA improves 49.60% short and 52.80-53.03% at 512/1K/near-4K. The full category gate moves h32 decode 38.840->43.081 tok/s and E2E 11.448->11.760 with prefill within -0.223%; gfx1151 and unmeasured backends remain baseline. | After one release window and a defaults-only gfx1100 refresh, remove public positive candidate selection and keep at most explicit baseline rollback. Never remove the registered baseline or change unmeasured backend defaults without independent evidence. |
| Laguna SWA wave32-exact prefill rollback | `LagunaGGUFResidentSession(..., swa_prefill_variant=...)` and `allocate_laguna_kv_cache(..., swa_prefill_variant=...)` retain explicit wave32 selection beneath gfx1151's online-qrow2 default. | LPF-5 first promoted wave32 on gfx1151: it reconstructs the original stride-64/32/16..1 FP32 tree, passes the 508..515 fixture byte-exactly, improves the leaf 20.434->9.229 ms (2.214x), and moves exact full-model 512/1K/4K prefill +8.31%/+12.85%/+14.06%. Exact context-qualified qrow2 later overlaid only M128/start>=128 slices; online qrow2 now owns measured gfx1151 SWA prefill after its full quality gate. Wave32 remains explicit rollback and the exact short/partial fallback inside context-qualified qrow2. Unmeasured backends retain prior defaults. | After one release window plus the post-prefill DFlash refresh, remove redundant public positive wave32 selection if the exact-qrow2 rollback no longer needs it; keep the registered wave32 fallback for unsupported/short rows. Never remove unmeasured-backend fallbacks without independent evidence. |
| Laguna SWA exact-qrow2 prefill rollback | Explicit `swa_context_rows_qrow2_exact_spans`, context-qualified `swa_context_rows_qrow2_m128_c128_exact_spans`, and retained wave32 variants coexist beneath the promoted online-qrow2 default. | The exact context-qualified route applies qrow2 only to complete M128 attention slices at absolute start>=128. Its final exact three-repeat gate improves 512/1K/4K prefill 0.893%/1.212%/1.040%; its complete category gate is exact and non-regressive at 0.999652x prefill and 0.999917/0.999999x h16/h32 E2E. It now serves as the primary exact rollback; empty-context, short, partial, and verifier rows delegate to wave32. gfx1100 and unmeasured backends are unchanged. | After one release window and a defaults-only gfx1151 512/1K/4K refresh, remove redundant positive direct-qrow2 selection if no bisection needs it; keep wave32 as the unsupported/short-row fallback and context-qualified exact qrow2 as the primary numerical rollback. Never broaden exact qrow2 below M128 or before start 128 without independent crossover and full gates. |
| Laguna global qrow2 online-prefill rollback | Explicit `global_context_rows_spans` remains beside gfx1151's promoted `global_context_rows_qrow2_online_spans` default. | The online route streams one BF16 K/V row across two adjacent queries without whole-context score LDS. It improves repeated 512/1K/4K **2.472%/5.444%/21.854%**; the complete category gate improves weighted prefill **0.315%** and h16/h32 E2E **0.184%/0.125%** with max KL `0.030836`, top-1 317/320, every category positive, and Poolside/repeats/lifecycle passing. gfx1100/unmeasured backends stay exact. Evidence: `benchmarks/results/2026-07-23-gfx1151-laguna-global-qrow2-online-retained.json`. | After one release window and a defaults-only 512/1K/4K plus category refresh, remove redundant positive explicit-online selection if no bisection needs it; keep `global_context_rows_spans` as the required exact fallback and explicit rollback. Never enable on another backend without independent quality/performance evidence. |
| Laguna SWA qrow2 online-prefill rollback | Explicit context-qualified exact qrow2 and wave32 routes coexist beneath gfx1151's promoted `swa_context_rows_qrow2_online_spans` default. | One wave replaces exact qrow2's two ring scans with online max/denominator/output state. M128/full-window and start508 wrap improve **3.093x/2.904x**; repeated 512/1K/4K improves **6.828%/9.364%/10.766%**. The complete category gate improves weighted prefill **1.086%** and h16/h32 E2E **0.616%/0.420%** with max KL `0.042924`, top-1 316/320, every category positive, and Poolside/repeats/lifecycle passing. gfx1100/unmeasured backends retain prior defaults. Evidence: `benchmarks/results/2026-07-23-gfx1151-laguna-swa-qrow2-online-retained.json`. | After one release window and a defaults-only 512/1K/4K plus category refresh, remove redundant positive explicit-online selection if no bisection needs it; keep exact context-qualified qrow2 and wave32 as rollback/fallback. Never enable on another backend without independent quality/performance evidence. |
| Laguna prompt prefill fallback | `LagunaGGUFResidentSession.prefill(..., use_bulk=False)` keeps the original token-serial prompt path beside default chunked rows. | Multi-length state/target-AR gates are complete. LPF-4 established exact bounded 128-row bulk chunks; AR-O3 now defaults gfx1151 to matrix512/attention128 after clean repeated 512/1K/4K gains of 6.266%/5.862%/4.943% with complete logits/hidden/KV/span/cursor/repeat/lifecycle equality. Unmeasured backends retain matrix128. Token-serial remains only as the independent state oracle and rollback through the post-prefill DFlash refresh; `forward_token()` decode is unchanged. Evidence: `benchmarks/results/2026-07-23-gfx1151-laguna-matrix-chunk-retained.json`. | Move token-serial prefill to a correctness-only helper (or remove the public selector) after bulk passes the full prompt-length/context matrix, retained target-AR performance is non-regressive, and verifier accept/rollback gates no longer need prefill bisection. |
| Laguna source-F16 tiled prefill selector | `HIPENGINE_LAGUNA_F16_PREFILL=auto|gemv|tiled` exposes explicit selection around the separately registered LPF-1 row/column tile. | Promoted gfx1151 default from two rows. Clean same-session rows 2..128 are exact and all faster (2.0538x weighted); the two-repeat category gate moves prefill 23.333->48.560 tok/s, TTFT 3.481->1.692 s, and h32 E2E 5.719->8.717 with neutral decode and all correctness/lifecycle gates. The reassociated WMMA control remains removed after changing three trajectories. `gemv` is the release rollback; rows=1/unsupported backends always retain registry-driven GEMV. | After one release window and a defaults-only gfx1151 refresh, remove the positive `tiled`/`auto` experiment semantics and keep at most one clear `gemv` rollback until release confidence permits removing the env selector entirely. Never remove the registered rows=1/unsupported-backend GEMV fallback. |
| Laguna SWA token4-exact decode rollback | `LagunaGGUFResidentSession(..., swa_decode_variant=...)` and `allocate_laguna_kv_cache(..., swa_decode_variant=...)` retain baseline `swa_context_spans` beside gfx1100's backend-qualified `swa_context_token4_exact_spans` default. | Promoted on gfx1100. The exact 4-wave/4-slot schedule passes all focused wrap/eviction/KV/runner gates; clean SWA improves 49.60% short and 52.80-53.03% at 512/1K/near-4K. The full category gate moves h32 decode 38.840->43.081 tok/s and E2E 11.448->11.760 with prefill within -0.223%; gfx1151 and unmeasured backends remain baseline. D10 token8 was exact and improved clean profiles, but failed the aggregate/every-category h16 gate and was removed with no rollback debt. | After one release window and a defaults-only gfx1100 refresh, remove public positive token4 selection and keep at most explicit baseline rollback. Never remove the registered baseline or change unmeasured backend defaults without independent evidence. |
| Laguna SWA wave32-exact prefill rollback | `LagunaGGUFResidentSession(..., swa_prefill_variant=...)` and `allocate_laguna_kv_cache(..., swa_prefill_variant=...)` retain explicit baseline selection beside the gfx1151 backend-qualified LPF-5 default. | Promoted on gfx1151. It reconstructs the original stride-64/32/16..1 FP32 tree, passes the 508..515 fixture byte-exactly, improves the leaf 20.434->9.229 ms (2.214x), and moves exact full-model 512/1K/4K prefill +8.31%/+12.85%/+14.06%. Unmeasured backends default to baseline. | After one release window plus the post-prefill DFlash refresh, remove public positive candidate selection and keep at most one explicit baseline rollback if needed. Never remove the registered baseline from unmeasured backends without independent evidence. |
| Laguna prompt prefill fallback | `LagunaGGUFResidentSession.prefill(..., use_bulk=False)` keeps the original token-serial prompt path beside default chunked rows. | Multi-length state/target-AR gates are complete. LPF-4 now defaults public sessions to bounded 128-row bulk chunks: the clean canonical gate is exact and moves paired prefill 48.541->49.641 tok/s with every category non-regressive. The 512/1K/4K matrix is now complete and exact with wave32 SWA; token-serial remains only as the independent state oracle and rollback through the post-prefill DFlash refresh. `forward_token()` decode is unchanged. | Move token-serial prefill to a correctness-only helper (or remove the public selector) after bulk passes the full prompt-length/context matrix, retained target-AR performance is non-regressive, and verifier accept/rollback gates no longer need prefill bisection. |
| Laguna grouped-down diagnostic selector | `LagunaGGUFResidentSession.set_selected_down_mode(direct|grouped_smallm|adaptive_grouped_smallm|grouped_smallm_fused|adaptive_grouped_smallm_fused)` retains explicit A/B selection beside the backend-qualified exact grouped-combine Q4/Q6 down default. | Exact grouped combine is promoted on gfx1151 from 32 rows after a bit-exact 1.249-1.313x sub-window screen, 0.99972x five-repeat complete-model screen, and exact/non-regressive category gate. The quality-rejected M16/M32 runtime route, scratch, selectors, and harnesses are removed; its registered leaf oracle remains. gfx1100 and rows below 32 retain direct. | Keep `adaptive_grouped_smallm` as the explicit unfused grouped rollback through one release window. Then remove redundant positive `grouped_smallm*` experiment modes and keep at most the adaptive unfused rollback plus clear `direct` for unsupported backends/short rows. Never remove direct fallback without independent evidence. |
| Laguna D9 MoE-tail next-RMS rollback | `LagunaGGUFResidentSession(..., use_moe_tail_next_rmsnorm=False)` restores the exact registered BF16 add/add/F32-weight-RMS chain around the gfx1100/gfx1151 c=1 composite; rows>1 and unsupported backends keep that chain. | Promoted on gfx1100 c=1. Synthetic and all-47-boundary actual Q2 XL state are exact; clean short/512/1K/near-4K kernel sum, span, and profiled child rows improve, and the complete category gate moves h32 decode 46.409->47.132 tok/s with every decode/E2E row positive. Independently promoted on gfx1151: native hidden17/3072 is byte-exact, cached resources are local256/VGPR16/LDS1024/scratch0, and counterbalanced p512/d128 moves rollback 14.529573/14.525706 to 14.555265 tok/s while removing 94 launches/token. | Keep for one release/bisection window and the current DFlash/MTP refresh, then remove the constructor selector while retaining the architecturally required unfused chain. |
| Laguna D12 raw-Q5 wave32x2 selectors | `LagunaGGUFResidentSession(..., use_q5_wave32x2_output=..., use_q5_wave32x2_query_gate=...)` independently rolls the gfx1100 defaults back to the exact pack8 siblings for c=1 attention output and unequal query/gate. | Promoted together on gfx1100. Formal 50-warmup/15x200 actual-weight leaves improve 13.63-24.80% in HIP events; clean short/512/1K/near-4K kernel sum/span/child all improve; and the counterbalanced canonical gate moves h32 decode 47.046->48.987 tok/s with every category positive and prefill neutral. Synthetic/production leaves and full shared-weight state are exact; cached trace is local32/VGPR96/LDS0/scratch0. gfx1151, rows>1, shape/registry misses, and explicit disable retain pack8. | After one release/bisection window and the current DFlash/MTP refresh, remove positive constructor-selection semantics and keep at most a clear pack8 rollback. Never remove pack8 as the required fallback. |
| Laguna exact Q5 fixed-metadata rollback | `LagunaGGUFResidentSession(..., use_q5_fixed_meta_output=False, use_q5_fixed_meta_query_gate=False)` and `laguna_target_ar_bench.py --disable-q5-fixed-meta-{output,query-gate}` independently restore the retained D12 coefficient-publication siblings around the gfx1100 fixed-metadata default. | Promoted. Two wave-uniform 128-bit metadata loads remove 32 coefficient exchanges and reduce logical VGPR **89 -> 72** without a sidecar. First/last actual output/query-gate rows improve event **19.80-25.19%** / wall **17.59-24.07%** at exact bits; full state/lifecycle and cached c=1 trace pass. Both clean process orders improve Q5 **22.68-23.12%**, kernel sum **2.35-6.34%**, span **2.17-5.58%**, and child throughput **2.26-4.41%** at every context. Both complete 18-prompt orders move h32 **54.476 -> 57.711 tok/s (+5.938%)** with every train/heldout category decode and E2E row positive. | Retain explicit role-scoped `False` for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered coefficient-publication siblings and pack8 routes for rollback, rows>1, shape/registry misses, and unsupported backends. |
| Laguna shared-Q5 fixed-metadata rollback | `LagunaGGUFResidentSession(..., use_q5_shared_fixed_meta=False)` and `laguna_target_ar_bench.py --disable-q5-shared-fixed-meta` restore the registered local128 pack8 BF16 pair around gfx1100's local32 fixed-metadata default for sparse layers 1-46. | Promoted. First/last actual pairs are BF16-bit exact and improve event **27.45-27.61%** / wall **26.88-27.48%**. Full state/lifecycle and default-vs-rollback pass; cached tracing records exactly 46 calls/token at local32/VGPR80/LDS0/scratch0 and unchanged 723 dispatches/token. Both clean orders improve shared-pair/kernel/span/child **45.99-47.13% / 1.32-3.02% / 1.43-2.62% / 0.89-3.33%**. Both complete category orders move h32 **59.500 -> 60.942 tok/s (+2.425%)** with every train/heldout category decode positive. Layer 47 Q6, rows>1, key misses, and unsupported backends retain existing routes. | Retain explicit `False` for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered local128 pack8 pair for rollback, rows>1, shape/registry misses, and unsupported backends. |
| Laguna mixed attention-projection quad rollback | `LagunaGGUFResidentSession(..., use_mixed_q6_fixed_meta_attention=False)` / `laguna_target_ar_bench.py --disable-mixed-q6-fixed-meta-attention` restores generic Q6 blocks inside the retained mixed quad; `use_mixed_q5_q6_attention=False` / `--disable-mixed-q5-q6-attention` restores the exact Q5/Q6 pair plus layer-47 Q8 singleton chain. | Promoted. Layers 0-46 use Q5 fixed-metadata wave32 owners plus generic Q6 local128 blocks; layer 47 uses exact generic Q6/Q8 local128 blocks. Actual first/last inclusive screens improve event **4.52-16.57%** and wall **3.65-14.23%**; full logits, 48 hidden/47 routed boundaries, active K/V/`KVLiveSpans`, reset, and lifecycle are exact. Cached trace records 48 calls and **772 -> 723 dispatches/token**. Both clean process orders improve projection work **2.02-3.35%**, kernel sum **0.09-0.35%**, span **0.69-1.56%**, and child throughput **1.06-2.92%**. Both complete category orders move h32 **57.833 -> 58.425 tok/s (+1.024%)** with every train/heldout category decode positive and all guards passing. The subsequent fixed-Q6 sibling preserves launch structure and exact state while improving actual projection wall **8.50-38.85%**, clean projection/kernel/span **8.08-10.10% / 0.73-1.26% / 0.57-1.49%**, and two-order h32 decode **58.466 -> 59.211 tok/s (+1.275%)**; gfx1100 now defaults that sibling. | Retain both explicit `False` rollback levels for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the generic-Q6 mixed quad plus registered exact pair/singleton chain for rollback, rows>1, shape/registry misses, and unsupported backend defaults. |
| Laguna all-local32 mixed-projection rollback | `LagunaGGUFResidentSession(..., use_mixed_local32_fixed_meta_attention=False)` and `laguna_target_ar_bench.py --disable-mixed-local32-fixed-meta-attention` restore the registered local128 fixed-Q6 Q5/Q6 quad around gfx1100's local32 default; registry miss also restores that route, including layer 47's Q6/Q8 tuple. | Promoted. Global/SWA production outputs, full state, and default-vs-rollback are bit-exact; first/last actual layers improve event **11.39-14.77%** and wall **11.24-15.72%**. Both clean orders improve projection/kernel/span/child **7.00-8.12% / 0.49-2.12% / 0.45-2.77% / 0.20-1.29%**. Both category orders move h32 **60.900 -> 61.732 tok/s (+1.367%)** with every train/heldout category positive at unchanged 723 model kernels/token. | Retain explicit `False` for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered local128 fixed-Q6 mixed quad and pair/singleton chain for layer 47, rows>1, key misses, and unsupported backends. |
| Laguna Q4 LM-head local32 fixed-metadata rollback | `LagunaGGUFResidentSession(..., use_q4_lm_head_local32_fixed_meta=False)` and `laguna_target_ar_bench.py --disable-q4-lm-head-local32-fixed-meta` restore the registered local128 c=1 BF16/F32 LM head around gfx1100's local32 default; bulk-prefill/verifier projection, gfx1151, rows>1, and registry miss also retain local128. | Promoted. The full 100,352-logit actual head and default-vs-rollback model state are bit-exact; actual event/wall improve **30.04%/21.44%**. Both clean orders improve LM-head/kernel-sum time **29.07-30.79% / 0.34-1.10%** at unchanged 723 kernels/token. Both complete category orders move paired h32 **61.675 -> 61.992 tok/s (+0.512%)** with every train/heldout category positive. | Retain explicit `False` for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered local128 body for bulk/verifier, rows>1, key misses, and unsupported backends. |
| Laguna P0 IQ3 serial fallback | `LagunaGGUFResidentSession(..., iq3_c1_down_schedule="serial_weighted")` remains the second-level c=1 rollback behind the retained wave10 default and explicit wave4 rollback; unsupported backends continue to default serial. | Historical P0 promotion moved h32 **48.780 -> 50.254 tok/s (+3.022%)** with exact full state and every context/category positive. Wave4 is now the immediate fallback for the later retained wave10 composite. The slower `row4_reduce` runtime mode is removed; the exact sign-bit sibling remains diagnostic after failing clean span/child guards. The measured tile4 leaf remains explicit for DFlash verifier rows. | Fold this entry into the wave10 rollback after one release/defaults-only refresh. Never remove the registered serial producer/composite or unmeasured-backend fallback without independent evidence. |
| Laguna IQ3 wave10-fused rollback | `LAGUNA_IQ3_WAVE10_FUSED=True` defaults gfx1100 c=1 decode to the exact local320 producer/reducer composite; `LagunaGGUFResidentSession(..., iq3_c1_down_schedule="wave4_reduce")` and `scripts/laguna_target_ar_bench.py --iq3-c1-down-schedule wave4_reduce` restore the registered producer-plus-reducer chain. Exact-key miss, rows/prefill, gfx1151, and unsupported backends retain prior registered schedules. | Promoted. Shared-weight full state/default-vs-wave4 is exact through 16 transitions. Cache-only tracing records **45 candidate + two unchanged reducers / 678 model kernels/token**, local320/VGPR88/LDS512/scratch0. Every short/512/1K/near-4K clean order improves inclusive IQ3 and kernel sum with span/child guards passing. Both complete 18-prompt orders move h32 **62.318 -> 63.270 tok/s (+1.528%)**, every train/heldout category improves at both horizons, and E2E/prefill/TTFT guards pass. | Retain explicit wave4 rollback for one release/bisection window and a defaults-only matched-completion refresh, then remove positive `wave10_fused` experiment semantics while permanently keeping the registered wave4+weighted and serial fallbacks for key miss, rows/prefill, and unsupported backends. |
| Laguna P2 exact split attention rollback | `LagunaGGUFResidentSession(..., use_split_attention=False)`, `allocate_laguna_kv_cache(..., use_split_attention=False)`, and `laguna_target_ar_bench.py --disable-split-attention` restore the registered global/SWA token4 readers; explicit threshold arguments retain focused A/B control. | Promoted on gfx1100 at global `>=127` and SWA `>=65`. Actual layer event/wall gates and full state are exact; clean short/512/1K/near-4K attention improves **15.66-23.28%**, kernel sum **2.67-16.11%**, span **4.65-14.63%**, and child throughput **1.19-17.58%**. The complete two-order 18-prompt gate moves h32 **50.093 -> 51.436 tok/s (+2.681%)** and E2E **+0.496%**, with every train/heldout category positive. Two reusable buffers add 1,572,864 bytes. gfx1151 and unmeasured backends retain fallback. | After one release/bisection window and P2.2 adjudication, remove positive threshold experiment semantics and keep at most the clear disable rollback. Always retain the registered exact global/token4 readers for below-threshold and unsupported routes. |
| Laguna P2 exact SWA tile16 rollback | `use_swa_split_tile16=False` / `--disable-swa-split-tile16` restores P2.1's one-slot score producer above gfx1100's retained live-257 tile16 default; explicit threshold remains for focused A/B. | Promoted. Two process orders improve pooled 512/1K/near-4K SWA attention **0.571%/0.344%/0.208%** and total attention **0.461%/0.272%/0.056%** with unchanged dispatches/memory. The complete fallback gate and a 150-transition default-vs-rollback gate are exact/non-regressive. | Fold into the P2 rollback row after one release/defaults-only long-context refresh; retain the registered P2.1 producer permanently for below-threshold and unsupported routes. |
| Laguna P4.1 split-reducer gate-fusion rollback | `LagunaGGUFResidentSession(..., use_split_gate_fusion=False)`, `allocate_laguna_kv_cache(..., use_split_gate_fusion=False)`, and `laguna_target_ar_bench.py --disable-split-gate-fusion` restore the exact split reducer plus standalone softplus BF16 gate. | Promoted on gfx1100 split paths. First/last actual layers at live 128/257 are bit-exact and improve inclusive event **3.00-10.05%** and wall **2.89-9.60%**; full state/lifecycle are exact. The two-order category gate moves h32 **51.497 -> 51.825 tok/s (+0.637%)** with every train/heldout category decode row positive and no new allocation. Below-threshold, registry miss, gfx1151, and unsupported backends always use the registered unfused chain. | After one release/bisection window and a defaults-only matched-completion refresh, remove the positive constructor/CLI selection while retaining the architecturally required unfused registered fallback. |
| Laguna current-P4 head RMSNorm+RoPE+KV rollback | `LagunaGGUFResidentSession(..., use_head_kv_fusion=False)` and `scripts/laguna_target_ar_bench.py --disable-head-kv-fusion` restore the registered two-launch head-plus-writer chain around the gfx1100 default. | Promoted on gfx1100 c=1. Global/SWA boundary fixtures, 48 hidden/47 routed boundaries, active K/V and every span field, reset, and lifecycle are exact; first/last actual layers improve inclusive event **33.05-39.36%** and wall **33.41-39.13%**. Pooled short and all long clean kernel/span/child rows improve; the two-order 18-prompt gate moves h32 **51.872 -> 52.391 tok/s (+1.001%)** with every train/heldout category decode positive and all guards passing. Rows/prefill, P2/P4.1 bodies, gfx1151, and unsupported backends remain unchanged. | Retain `False`/CLI rollback for one release and a defaults-only matched-completion refresh, then remove positive constructor selection while permanently keeping the registered unfused chain for rows/prefill and unsupported backends. |
| Laguna SWA split wave-local reducer rollback | `LagunaGGUFResidentSession(..., use_swa_split_wave_local=False)` and `scripts/laguna_target_ar_bench.py --disable-swa-split-wave-local` restore the exact shared-statistics gated reducers around the gfx1100 wave-local default. | Promoted. Full logits, 48 hidden/47 routed boundaries, active K/V and every span byte, reset, and lifecycle are exact. First/last actual layers improve **4.84-18.96%**; two clean process orders improve reducer/SWA **4.63-5.22% / 4.24-4.55%**, kernel sum **0.94-1.98%**, and span **0.61-1.69%** at every context. The complete two-order gate moves h32 **52.211 -> 52.514 tok/s (+0.580%)** with every category decode positive and all guards passing. | Retain explicit false for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered shared-statistics reducer for unsupported backends and rollback. |
| Laguna exact IQ2 expanded-magnitude rollback | `LagunaGGUFResidentSession(..., use_iq2_grid64=False)` / `scripts/laguna_target_ar_bench.py --disable-iq2-grid64` restores the compact-grid c=1 IQ2 gate/up+SiLU tile2 route around gfx1100's expanded-magnitude default; rows>1 always retain compact grid. | Promoted. A 4 KiB canonical magnitude table replaces the retained 1 KiB packed-code reconstruction without duplicating persistent weights. First/last actual layers improve **30.00-33.73%**; full state is exact. Two clean process orders improve IQ2/kernel/span/child at every context, and both complete 18-prompt orders move h32 **52.650 -> 54.540 tok/s (+3.590%)** with every train/heldout category decode and E2E row positive. gfx1151 and unsupported backends remain compact-grid. | Retain explicit `False` for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered compact-grid route for rows>1, unsupported backends, and exact fallback. |
| Laguna DFlash IQ3 selected-down tile4 | `LagunaGGUFResidentSession(..., iq3_selected_down_tile=1|4)` and `scripts/laguna_dflash_category_bench.py --iq3-selected-down-tile` retain tile1 plus the explicit gfx1100 tile4 verifier path. | Retained explicit-only. Tile4 is exact, improves the profiled 45-call family 33.66%, and moves counterbalanced h32/h128 DFlash decode +4.725%/+4.536% with every category/heldout decode and E2E positive. Tile2 is removed. Automatic DFlash and public Q2 routing stay off because tile4 DFlash is still only 0.6915x/0.6338x true AR; a global session default would broaden to unmeasured models/backends and non-verifier rows. | Remove the positive selector and tile4 leaf if explicit Q2 DFlash ceases to be maintained. Otherwise keep it only in the measured gfx1100 verifier scope until a complete-suite automatic route is admitted; retain tile1 permanently for unsupported backends/shapes and ordinary target paths. |
| Qwen tokenizer EOS discovery | The PARO generator falls back to looking up `<|im_end|>` and `<|endoftext|>` because `tokenizers.Tokenizer.from_file()` does not expose `generation_config.json` EOS metadata. | The fallback recognizes both Qwen EOS ids, preserves explicit scalar/sequence metadata first, and deliberately avoids unrelated model-family markers. | Remove the string lookup once the Qwen model plugin owns and supplies normalized BOS/EOS/PAD metadata from `generation_config.json` / `tokenizer_config.json` to generation and sampling. |
| PARO width-plan execution and rollback flags | The greedy generator and `scripts/qwen35_batch_retained_bench.py` each execute `BatchWidthPartitionPlan` groups and collect similar telemetry; `_generate_batch_sampled()` remains diagnostic. `HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS` and the legacy-named `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE` remain explicit rollback opt-outs. | G3 retains an identity-matched schema-2 gfx1151 c2/c4/c8 profile. G4/G5 attach it to the shared stable-slot owner; blocking F1, native/serial SSE, live admission, c8 repeatability, and a no-flag OpenAI c4 confirmation are exact. gfx1151 backend capabilities now enable the hash/sync-checked packaged profile by default even outside a repository CWD; explicit `=0` still selects the serial fallback. gfx1100 remains package-default off pending owner c4/c8 symmetry. | After one release window and a defaults-only gfx1151 refresh, remove the two positive opt-in semantics and rename/collapse them to one clear rollback selector; consolidate duplicate generator/benchmark plan execution after gfx1100 owner c4/c8 is independently retained. Keep profile identity checks, unsupported-shape partitions, and exact serial fallback permanently. |
| PARO selected-MoE compatibility alias | `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE` and CLI `--batch-decode-moe-path selected_c1` remain compatibility aliases even though true per-row paths still use `selected_c1_per_row_*` labels. | Runtime metadata emits canonical `moe_decode_path=selected_batch` plus `moe_selected_batch_layers`; retained validators accept either exact grouped-compact or selected-batch layers while still requiring zero per-row fallback layers. The canonical env/CLI spelling is `SELECTED_BATCH_MOE` / `selected_batch`, and the old alias has lower precedence. Clean current-revision gfx1100 c2 is now retained at **121.923 tok/s** selected-batch versus c1 **116.022** and serial c2 **100.925**; all-layer/lifecycle/category/profiler gates pass and `auto` resolves selected-batch. Evidence: `benchmarks/results/2026-07-18-gfx1100-paro-g2-selected-batch-c2-retained.json`. | Remove the legacy env/CLI alias after one compatibility window from the retained c2 promotion. Keep explicit per-row fallback names and counters permanently. |
| PARO short-context c>N attention | The optimized 1,024-thread `bf16_context_batch_spans` kernel remains registered beside the default 256-thread dense-order `bf16_context_batch_c1_exact_spans` route. | The optimized route is faster but introduces FP32 reduction drift that changes the full-model c2 trajectory; the exact route preserves physical batch block rows and matches dense c1 bit-for-bit. There is no runtime env flag and dispatch selects the exact variant through the registry. | Remove the duplicate optimized route or replace it only after a c-aware implementation is bit-exact to dense c1 at the 513-token model shape, passes full L40 c2 token/hidden/state/KV gates, and is non-regressive in repeated p512/d128 timing. |
| PARO gfx1151 greedy decode graph | `capture_decode_graph()` remains available, but public greedy c1 consults a resident-session architecture policy before capture. | The 2026-07-13 gfx1151 p512/d8 graph/eager fixture gate rejected generated-token equality, and graph wall was slightly slower (`0.133842` vs `0.133048` s). gfx1151 therefore uses canonical eager resident steps; gfx1100 retains its previously validated graph path. | Repair and re-admit gfx1151 only after a current-stack graph/eager gate matches generated tokens plus hidden/Conv/GDN/KV state across long replay and improves capture-inclusive wall. If no repair is planned, move the rejected graph probe to a diagnostic-only harness and keep eager as the permanent gfx1151 policy. |
| PARO ragged packed prefill | Ragged compact slabs automatically use per-segment linear/full-attention prefill; `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR` and `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN` retain explicit bisection routes. | SOL-P2 found the packed segmented/varlen route changed the first ragged row's persistent state and, across all 40 layers, its generated tokens. `per_segment_ragged_exact` is the production-safe fallback; equal-length slabs retain `packed_segments`/`packed_varlen`. | Repair the ragged packed kernels, then remove the automatic fallback only after c8 ragged generated-token plus all-linear-state/all-full-KV identity passes independent c1 on gfx1151 and gfx1100 and an end-to-end prefill comparison is non-regressive. Remove the force flags after the repaired route survives one release window. |
| PARO short-prompt prefill | The public generator uses token-serial c1 steps when prompt tokens are fewer than `linear_conv_kernel_dim`; low-level `prefill_native()` remains strict. | Release fallback. A no-env gfx1151 wheel-path smoke generated from the one-token prompt `Hello`; normal prompts still use native prefill. | Replace the serial fallback only after a dedicated native short-prompt kernel matches c1 generated tokens plus recurrent/KV state for lengths 1-3. |
| GGUF public AR profile | `HIPENGINE_GGUF_DECODE_REPACK=0` remains a rollback opt-out; low-level WMMA/GEMV selectors remain available for benchmark bisection. | Release default: T16 decode-repack is on, and public generate/stream sessions pass the resolved backend plus `use_wmma_prefill=True` and `use_gemv_decode=True`. A no-env gfx1151 Q4_K_M smoke generated one token through `LLM(model)`. | Remove the decode-repack opt-out after one release window and a defaults-only gfx1100 refresh. Keep raw layouts only where a quant/kernel lacks a T16 fallback or a retained diagnostic requires them. |
| GGUF duplicate AR loop ownership | `Qwen35GGUFResidentModelRunner` owns public blocking and OpenAI submit/poll execution, but `_generate_ar_serving_slots()` remains as a direct control/oracle and explicit compatibility fallback. | D1–E3/F1 now prove one shared model-owning loop through exact arbitrary-C burst/live admission, cancellation, SSE, shutdown, real KV ownership, observability, and retained server scaling on both gfx11 targets. The direct loop still supplies the independent c1/native-width oracle used by the retained E1/E2/E3 packets. | Both gfx11 triggers are met. During F2, move the oracle into an explicit test/benchmark helper and remove production call sites after one release window; keep registry-resolved unsupported-shape fallbacks. |
| GGUF RadixCache production admission | `HIPENGINE_PREFIX_CACHE=radix` is a real model-loop opt-in. It prefers an active source's exact-current positive 256-token boundary, then may restore a bounded cache-owned device snapshot after normal source reclaim. Stochastic sampled reuse remains off, while deterministic `processed_argmax` forced-tool rows now restore prefixes with full-vocabulary logits and unchanged host processors. Shared-prefix suffix tokens use exact c1 steps because packed one-row suffix arithmetic passes KL/top-1 but is not byte-identical to c1 state/KV. | Host/runtime RED/GREEN proves same-backing page refcounts/COW, non-contiguous block-table gather/scatter, exact hybrid-state clone, suffix-only execution, source-first reclaim, and final drain. The active-current gfx1151 and gfx1100 gates are byte-exact; the clean gfx1151 paired p256+s1 packet moves already-live continuation TTFT **249.269 -> 21.188 ms (11.765x, -91.50%)**, with live pages **4 -> 3** (5,242,880 bytes) and zero paired HIP-current median savings. The completed-source gate then resets/unbinds the source before admission, restores all 66,846,720 snapshot bytes, keeps output/all Conv/GDN/live-KV/four teacher-forced steps byte-exact (`KL=0`, top-1 `100%`), and proves cache refs/eviction **1->1->2->1->0**. Its clean paired economics move TTFT **249.446 -> 22.013 ms (11.332x, -91.18%)** with 3/3 exact snapshot hits; unique continuation pages stay **2 -> 2**, while exact cache residency is **72,089,600 bytes** and paired HIP current is **+62,914,560 bytes**. The gfx1100 active/completed correctness transfer also passes output, all Conv/GDN/live-KV bytes, four teacher-forced steps (`KL=0`, top-1 100%), refcount/COW, snapshot eviction, and final drain. Deterministic processed-argmax p2048/p8192 active/completed gates additionally preserve a two-token forced sequence plus the five-ID response trajectory, reuse 8/32 pages, and bound completed residency at 108,789,760/234,618,880 bytes before exact eviction/final drain. The final W7900 A2 packet closes lifecycle/pressure but rejects agentic promotion: radix hits only 0/12, 3/24, and 3/18 C1 turns, regresses active-SSE goodput 64.19%/65.63%/26.64%, and worsens tool-ready latency 181.90%/196.09%/38.81%. Default stays `off`; radix is explicit diagnostic-only. | Do not default-enable the measured latest-boundary policy. Reconsider only after a model-general LCP/snapshot redesign passes the full frozen C1/C4/C8 suite, same-seed sampled output/state gates, and lifecycle economics without prompt-conditioned tuning. If no such redesign is scheduled after A3/A4, move production radix admission to a benchmark/diagnostic helper and remove the runtime flag; keep exact c1 suffix and ownership tests as references. |
| GGUF LCP-1 convolution prefill | `HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE=baseline|tile32x128` selects between the production global-read convolution and the registered exact shared-token route. | gfx1151 selects `tile32x128` automatically. The clean 512/4K 82-part and wall gates pass, the 4K body falls `954.134 -> 49.790 ms`, and all six right-sized prefill rows improve `+1.10%..+24.04%` with unchanged memory. gfx1100 remains on `baseline` pending hardware transfer. The production implementation is the required unfused fallback and explicit rollback. | Remove the explicit mode selector after one release window if the gfx1100 transfer remains stable. Never remove the exact production fallback. |
| GGUF packed-AR singleton-indexed GDN | Backend capability `GGUF_GDN_INDEXED_SINGLETON_DECODE` selects a one-token-per-row indexed sibling while retaining the arbitrary-length segmented recurrence. | gfx1151 defaults to the singleton sibling after independent-c1 byte equality and exact p512/d64 trajectories; gfx1100 remains on segmented GDN pending hardware transfer. The runtime manifest records `indexed_singleton` versus `segments` explicitly. | Remove the gfx1100 capability split only after an independent W7900 c2/c4/c8 correctness/performance gate. Keep the segmented implementation permanently as the arbitrary-length fallback. |
| GGUF selected-MoE duplicate-expert reuse | `HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE=0`, `HIPENGINE_GGUF_T16_SELECTED_DOWN_PAIRREUSE=0`, and `HIPENGINE_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE=0` roll physical-C8 Q4T16 gate/up plus Q5/Q6T16 down back to per-selected-lane kernels; backend capabilities keep gfx1100 unchanged. | gfx1151 pairs consecutive dynamic expert-ID occurrences inside 128-thread blocks while preserving each row's reduction order. Q4 gate/up and Q5/Q6 down share weights across independent per-row accumulators. Lower widths and unpaired IDs retain exact fallback behavior. | Remove the three env opt-outs after one release window plus defaults-only gfx1151 direct/server refreshes and an independent gfx1100 transfer. Keep per-lane kernels for unsupported widths and as required fallbacks. |
| GGUF F32-weight cooperative c1 router | `HIPENGINE_GGUF_ROUTER_F32W_COOP=0` retains the separate expert-logits/shared-logit/top-k chain as an explicit rollback around the default-on gfx1100 cooperative route. `HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER=0` temporarily restores the selected-ID counter alias plus per-layer host reset instead of the default dedicated self-resetting four-byte counter. | Production-shape logits, selected IDs, and routing weights are byte-exact. Clean commit `4c743994` first improved 4K graph decode **97.234 -> 98.273 tok/s (+1.07%)**. The persistent follow-up removes exactly 40 reset nodes/token, improves the cache-cycled fused leaf **14.667 -> 10.444 us (-28.79%)**, and cleanly improves the 4K graph gate **98.812 -> 100.446 tok/s (+1.65%)**, with all IDs/final values exact and only eight added tracked bytes. The unfused chain remains the required numerical fallback for unsupported hidden/backend/quant shapes. | Remove both env opt-outs after one defaults-only gfx1100 refresh remains non-regressive. Keep registry-driven fallback resolution, not experiment toggles. |
| GGUF long-context split-K reduction | `HIPENGINE_GGUF_PAGED_ATTN_PARALLEL_REDUCE=0` and the minimum-context override retain the serial split reduction for rollback/A/B around the gfx1100 prepare-plus-coalesced-output route. | Promoted gfx1100 default from 32K after the clean LCP-D2 gate: 32K 1+3 decode **84.525 -> 85.561 tok/s (+1.23%)**, clean 64K/128K confirmations **+3.95%/+7.80%**, max long-context KL **1.904e-6**, top-1 100%, exact IDs, unchanged memory. gfx1151 remains serial without independent evidence. | Keep the serial implementation as the required fallback. Remove the env opt-out after one release window plus a final defaults-only gfx1100 six-shape refresh; retain backend capability scoping until gfx1151 is independently gated. |
| GGUF MTP server packed verifier | `_MTP_SERVING_TARGET_BATCH_MAX_SLOTS = 4` chunks c>N server target verification instead of sending all active slots to one packed target forward. | Default serving policy after the first packed verifier landing and the stream-draft/stream-verify follow-ups. c=2/c=4 packed target verify wins, but one 8-slot packed batch is a measured rejected regime (`11.58 tok/s`, `target_verify_batch_ms=63733.783`). The current c=8 stream path still chunks verify at 4 slots and reaches **52.18 tok/s**, with verifier still dominant (`slots_verify_phase_ms=12345.442`). | Remove or raise the cap only after rows>=16 packed verifier and resident-draft row-count/cold-slot behavior are tuned and a c=8 natural24 rerun beats the chunked stream path without correctness or latency regressions. |
| GGUF packed verifier GPU-event instrumentation | `HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS` records HIP events through `Qwen35GGUFResidentSession.verify_target_blocks_batch()` and compact-MoE leaves. | Default-off diagnostic. It exposed c=8 server verifier GPU leaves on 2026-07-05 but adds event overhead (`47.17 tok/s` in the compact-WMMA event run), so it is not a retained speed path. | Keep only while c>N MTP verifier tuning is active; remove or move behind a dedicated profiling helper once the packed verifier bottleneck is closed. |
| GGUF compact-WMMA tight no-read scope | `HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS=0` restores the scalar `wmma_total` D2H read around the gfx1100 default capped at 4,096 selected rows; gfx1151 remains scalar by package policy. | The old rejected c=8 probe overlaunched `selected_rows * 16`. LCP-2B instead proves the exact worst-case tile count `A + floor((S-A)/16)`, clears unused tile ids to `-1`, and at pp512 removes 40 D2H boundaries while selected-Q4 kernel time stays flat (**40.631 -> 40.728 ms**) and matched queue idle falls **15.163 -> 11.634 ms**. | Keep the scalar fallback as the required exact-count route. Remove the env opt-out after one defaults-only gfx1100 six-shape refresh; retain backend scoping until gfx1151 is independently gated. |
| GGUF selected-WMMA launch-bounds tuning | `HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS` remains an R&D build flag for selected-WMMA kernels. | Default unchanged after the 2026-07-05 c>N server probe. `=2` was flat at c=8 (**52.55/52.23 tok/s**); `=4` helped c=8 (**53.22/53.44**) but regressed c=4 (**49.20/49.04** vs retained **49.65**), so no default promotion. | Keep as kernel R&D only; do not promote without a c=2/c=4/c=8 same-protocol rerun that is non-regressive at every concurrency. |
| GGUF AR server packed decode | `HIPENGINE_GGUF_AR_PACKED_DECODE` is a default-on rollback opt-out around decode-shaped packed resident target passes for c>N GGUF greedy AR serving. | E1/E2 retain eager/graph c8 p512/d128, ragged, sparse, cancellation, and **748 packed-native / 0 row-local / 0 copies** on both targets. E3/F1 adds exact C13 c8+sparse-c8, middle-hole/new admission, and real SSE logical c1/c8/c9/c13/serial-c13 at **25.583/136.122/88.592/111.380/31.708 tok/s** on gfx1100 and **15.701/86.338/57.127/72.522/42.764** on gfx1151, with zero packed-route fallback and **189/189** exact requests per packet. | Both gfx11 E3/F1 triggers are met. Keep the opt-out for one release window, then remove the env switch during F2 while retaining registry-resolved scalar fallback for unsupported shapes. |
| GGUF AR server packed prefill | `HIPENGINE_GGUF_AR_PACKED_PREFILL` is a default-on rollback opt-out around packed final-row prompt prefill for c>N GGUF greedy AR serving. Row-bounded multi-round prefill still computes and samples each intermediate chunk tail even though only the final prompt result is returned. | Packed linear/MoE stays multi-row and full attention preserves slot-local arithmetic across bounded rounds. E1/E2/E3 cover c8 and C13 eager/graph, ragged, sparse masks, cancellation/admission, and exact token/hidden/Conv/GDN/live-KV state. Each gfx11 F1 packet sends 512 exact prompt IDs/request through real prefill work and preserves all **189** prompt rows, usage counts, and outputs. | Remove intermediate chunk-tail output-norm/LM-head sampling after a final-slot mask preserves hidden-seed/MTP contracts and is profiler-non-regressive. Keep the opt-out for one release window, then remove it during F2 on both gfx11 targets while retaining scalar fallback only for unsupported shapes. |
| GGUF fair prefill burst rollback | `HIPENGINE_FAIR_PREFILL_BURST_CHUNKS` and `--fair-prefill-burst-chunks` retain strict one-chunk alternation as an explicit rollback around bounded fair-prefill bursting. | gfx1151 Q4_K_M defaults to at most two consecutive 256-token chunks only while at least two prompts still need prefill; lone staggered arrivals keep one-chunk alternation. The static p512/C8 host contract removes six duplicate partial-width ticks, accepted real-Uvicorn C8 improves over the retained exact SSE/live rows, and `continuous_fixed` moves ITL p99 **0.5068 -> 0.2949 s** while all 12 rows and SLO checks pass. Other backend/quant packages remain at one chunk. | Remove the env/CLI rollback after one release window and a defaults-only full F4 production matrix plus C1/C2/C4/C8 packet remain exact, SLO-clean, and non-regressive. Keep workload-derived pressure/fairness logic rather than backend branches in the scheduler. |
| GGUF MTP server packed prefill | `HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL` is a default-on opt-out around packed prompt prefill for eligible c=2/c=4 GGUF MTP serving batches. | Default-on after the 2026-07-06 steady-state natural24 rerun. The path reuses packed prompt rows and returns FP32 prompt hidden rows for MTP catch-up, moving server MTP **46.75/49.65/52.18 -> 59.94/66.60/54.88 tok/s** at c=2/c=4/c=8. It keeps the four-slot safety cap: c=8's first wave still uses serial prompt open and only the trailing c=2 wave uses packed prefill. Startup now warms hidden-seed packed prefill at widths 2/4 when MTP serving is enabled, moving fresh c=2 to **56.59 tok/s** and warm c=2/c=4 to **59.71/65.57 tok/s**. | Keep the opt-out until one more c=2/c=4/c=8 rerun confirms the default. Do not remove the four-slot cap until c=8 full packed prefill is non-regressive; pool-filling eight startup slots was rejected (**35.25 tok/s** c=8 rerun, **76.5 GiB** used). |
| GGUF MTP server startup warmup | `HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP` is an internal server-scoped env marker set only during startup scratch probing when `--speculative-mtp-serving` is not `off`. | Added after the 2026-07-06 cold-start audit. It lets the GGUF backend warm MTP hidden-seed packed prefill plus one tiny packed verifier at supported widths 2/4 without changing the generic `prepare_request_scratch(...)` hook signature. It removes the worst c=2 first-request MTP cliff but deliberately does not attempt unsupported width-8 packed prefill. | Replace this env handoff with an explicit backend scratch-preparer option if the startup hook grows typed capabilities. Keep it while MTP serving is opt-in/auto and c=2/c=4 cold-start evidence remains positive; remove or narrow it if startup memory/time becomes a production blocker. |
| GGUF MTP server deferred verifier scatter | `HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER` is a default-on opt-out around delaying packed target verifier state scatter until after the accept decision. | Default-on after the 2026-07-06 resetfix rerun. It keeps owner-side packed verifier state live and commits only accepted hidden/full-attention KV/linear-state rows, moving retained no-env natural24 MTP **70.06/77.29/76.46 -> 70.53/78.76/79.61 tok/s** at c=2/c=4/c=8 with unchanged economy (`draft=165`, `accepted=141`, accept rate **0.8545**, **250** target rows). Reset now invalidates packed verifier/decode session metadata so startup verifier warmup cannot leave stale packed KV write-position bookkeeping for the first real request. | Remove the env opt-out after one more no-env c=2/c=4/c=8 server rerun plus API tests show no regression. Keep the eager-scatter branch only while it is useful for bisecting packed verifier state lifecycle bugs. |
| GGUF llama-compat native MTP cycle | Benchmark flags `--native-spec-target-cycle`, `--native-spec-device-accept-commit`, `--native-spec-complete-cycle`, and `--native-spec-proposal-graph` retain N1 target-only, N2 device-policy, N3 complete-adapter, and N3P proposal-submission boundaries over separate reusable B1/B2 graph buckets. | Explicit accuracy-traded gfx1100 scope. N1 remains the 122.667 tok/s topline over llama.cpp's 115.444 floor. N2 owns strict acceptance, selected hidden/Conv/GDN state, target cursor, and one bounded payload. N3 makes proposal, N2, MTP-KV repair, reseed, and cursor/result accounting one public single-request adapter call; its clean committed gate is exact for 240 IDs / 96 cycles and aggregate-neutral at 118.592 vs clean N2 117.557 tok/s (+0.88%). N3P stages dynamic proposal inputs into fixed runner buffers and replays one B1/B2 NextN graph before the unchanged target graph. A detached clean publication is exact at 118.183 tok/s / 8.610 ms-output; a same-source pair is 117.589 N3P vs 116.793 N3 tok/s, and cached eight-cycle tracing replaces 542 `hipLaunchKernel` plus 80 synchronous `hipMemcpy` calls with eight proposal `hipGraphLaunch` calls. N3P still has two native graph submissions per public cycle, not one combined submission, and gfx1151 remains unregistered. | Migrate the provider-neutral adapter to PARO/DFlash (N4), independently gate gfx1151, and decide whether combining proposal+target behind one native boundary yields a measurable gain. Remove benchmark-only N1/N2/N3 splits after cross-provider full-suite/profiler gates. Keep independent B1/B2 buckets and exact unsupported-shape fallback. |
| PARO MTP / DFlash native target graph | `HIPENGINE_PARO_NATIVE_SPEC_TARGET_GRAPH=1` explicitly routes eligible shared single-request B1/B2/B3/B4/B5/B8 target graph replays through the registered gfx1100 `w4_paro/native_v1_target_graph` NativeSpecCycle boundary. `HIPENGINE_PARO_NATIVE_SPEC_TARGET_COMMIT=0` temporarily rolls capture-width-zero FP16 PARO MTP back from selected linear-state commit/device cursor ownership to the neutral N4+ provider commit. | N4 remains globally default-off. Inside explicit N4, PARO selected commit is default-on after three exact 240-ID arms, every split/category, accepted-row/B2 state gates, and consistent capture-adjusted wins. It uses graph-owned combined state-pointer tables and declares `VERIFY|ACCEPT|COMMIT|UPDATE_CURSORS`; BF16 DFlash hidden/KV repair, proposal, and scheduler result construction remain unchanged. Cached profiles mechanically change **80.6875 -> 75.6875 APIs**, **2 -> 1 sync**, **36.1875 -> 34.1875 host launches**, and **1248.5 -> 1247.5 kernels**; mean wall is neutral. Pointer/shape/stream/inactive drift retains validation/direct fallback, and gfx1151 remains unregistered. No AR speed row/global default is retained. | Remove `HIPENGINE_PARO_NATIVE_SPEC_TARGET_COMMIT` after one release window and a defaults-only explicit-N4 refresh remains exact and mechanically non-regressive. Keep the global N4 switch until complete PARO proposal ownership and independent full category+heldout PARO/DFlash gates have a measured complete-wall advantage over direct graph control; gate gfx1151 separately. Remove the bound-control experimental split once that provider boundary is the only admitted implementation. Do not remove or downgrade the current model artifact. |
| GGUF AR server stream decode | `HIPENGINE_GGUF_AR_STREAM_DECODE` is a default-on fallback/opt-out around per-slot HIP stream decode when packed AR is disabled or unavailable, and also gates the historical parallel c4+c4 chunk path above four rows. | Direct physical-c8 beats c4+c4 on both targets. F1 measures the same-loop packed-off serial C13 bridge at **31.708/42.764 tok/s** versus grouped packed C13 **111.380/72.522 tok/s (3.513x/1.696x)** on gfx1100/gfx1151; every static/live packed route records zero serial/resident fallback, including real c8→c13 admission. | Both gfx11 triggers are met. Keep the bridge as an explicit unsupported-shape/test oracle and rollback for one release window, then remove the public env opt-out during F2 while retaining backend capability fallback. |
| GGUF AR server stream prefill | `HIPENGINE_GGUF_AR_STREAM_PREFILL` is a default-off diagnostic around launching AR prompt prefill and top-1 sampling on each slot's decode stream before the stream-decode loop. | Rejected on the 2026-07-05 natural24 c=8 rerun: AR stream prefill measured **47.44 tok/s**, below the retained stream-decode baseline **47.70 tok/s**. It also raised prompt-prefill wall (`prefill_stream_batch_ms=15076.522`, `prefill_ms=8665.634`) on the short natural24 workload, so concurrent prefill appears to contend more than it helps. | Remove the env and async prefill plumbing unless a longer-prompt c>N sweep or a true batched prefill implementation proves it non-regressive versus the default stream-decode path. Do not promote this path without a c=2/c=4/c=8 rerun that beats retained AR. |
| GGUF MTP server stream draft | `HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT` is a default-on opt-out around per-slot HIP stream draft proposal for c>N GGUF MTP serving. | Default-on after the 2026-07-05 natural24 server run. It moves packed-verifier c=2/c=4/c=8 MTP **45.57/47.48/47.18 -> 46.75/49.65/48.72 tok/s** and beat the then-current stream-AR rows by **1.058x/1.063x/1.021x**; the later packed-AR route supersedes that same-server AR comparison, so current MTP is below AR. It currently creates a `ThreadPoolExecutor` per draft phase and the c=8 wall remains verifier-heavy. | Remove the env opt-out and/or replace the per-cycle executor with a persistent scheduler after c>N server reruns and profiler evidence show the stream path is non-regressive. Keep the opt-out while tuning verifier wall and thread-pool overhead. |
| GGUF MTP server stream verify | `HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY` is a default-on opt-out around running independent chunk-4 packed target verifier chunks on separate owner-session streams at c>N. | Default-on after the 2026-07-05 natural24 c=8 rerun. It keeps the four-slot verifier cap but overlaps the two c=8 chunks, moving warm c=8 MTP **48.72 -> 52.18 tok/s** and `slots_verify_phase_ms` **15354.902 -> 12345.442**. c=4 remains on the single-chunk path. | Remove the env opt-out after c=2/c=4/c=8 reruns and profiler evidence show the stream verify path is non-regressive; replace it if a tuned rows>=16 verifier or true batched scheduler supersedes chunk-stream overlap. |
| GGUF row-shaped target executor | `Qwen35GGUFResidentSession.step_rows()` still loops physical slots through the retained c=1 layer path; `verify_rows()` likewise advances each linear-chain row serially. They are explicit correctness/speculative bridges rather than the default independent-decode route. | UD-Q3_K_M C=2/4/8 uses native indexed Conv/GDN, batched `KVLiveSpans` attention, selected-row MoE, row lm-head/argmax, and shape-specific HIP graphs; synchronous greedy prompt lists select the scheduler-owned native path. The serial executor remains the independent layer/full-logit oracle. Task #31's transactional MTP verifier deliberately reuses the exact c=1 row arithmetic and journals each row rather than weakening correctness. | Keep `step_rows()` as an explicit unsupported-shape/bisection oracle. Replace the eager serial MTP chain only when a native dependency-aware root+candidate forward preserves the B=1/2/3 scalar-logit and state/KV commit gates and is same-suite non-regressive; do not remove unfused numerical fallbacks. |
| GGUF MTP eager verifier | `Qwen35GGUFTransactionalVerifier` allocates stable full-shape buffers by scheduler graph key, but the target chain itself is eager and snapshots fragmented Conv/GDN state per row; `captured=false` is explicit. Draft prompt prefill also runs token-serial because every prior target-hidden row must seed blk.40 KV. | Correctness route only. Candidate-only/shared ABI, verify-chain spans, GPU accept, rollback, reject/partial/full commit, exact greedy output, and expected profiler symbols pass. Matched GPU1 B=1/2/3 ratios are `0.544x/0.346x/0.271x`, so no public route or env default was added. | Remove the diagnostic runtime only if it ceases to serve as the oracle. Promote/capture only after a dependency-aware native verifier and resident draft prefill beat AR on the same prompt while preserving every transactional gate. |
| GGUF prompt-list scheduling | The promoted scheduler is synchronous and greedy-only: it does not persist across `generate()` calls, serve HTTP arrivals, cancel/disconnect rows, share prefixes, grow/shrink an elastic KV pool, or batch non-greedy sampling. Prefill into a reclaimed slot is still per-request rather than a packed multi-prompt slab. | Stable request ids, 2–8 physical slots, EOS/length reclaim, downward recurrent-state/live-KV compaction, mixed readmission/prefill between decode steps, C/context/mask graph buckets, row argmax, and admission/completion timestamps are retained and exact. Fixed C=2/4/8 scaling is accepted; one in-call 4-request/capacity-2 schedule exercises three admission waves, three compactions, and two readmissions with `serial_decode_fallback=false`. | Move this ownership into the persistent server engine loop, add cancellation/disconnect and elastic/prefix-shared KV, then native per-row non-greedy sampling and packed mixed-prompt prefill. Keep the synchronous path until the persistent route passes the same exactness/provenance gates. |
| UD-Q3_K_M grouped raw-IQ prefill | `HIPENGINE_GGUF_IQ_GROUPED_PREFILL=0` is the temporary rollback around the default-on expert-major scalar IQ3/IQ4 route; direct selected kernels remain the required fallback. | Promoted after mixed-prompt native-row-bulk parity (`KL=0`, top-1 `1.0`), exact 512/4K trajectories, zero scratch/copies, raw-IQ time `994.668 -> 613.995 ms` (-38.27%), and total kernel sum `4396.145 -> 4078.667 ms` (-7.22%). The formal 512/128 headline is correctly labeled flat within spread (`16.648 -> 16.685 tok/s`, +0.22%); promotion retains the verified sub-window win, not a noisy topline claim. WMMA and scalar RT2 remain rejected/test-only. | Keep the opt-out for one rollback/bisection window through task #20/#15 D0. Remove the env branch after the next defaults-only 512/4K milestone rerun confirms the grouped symbols and direct fallback tests remain healthy. |
| GGUF MoE-tail plus next-input RMSNorm | `HIPENGINE_GGUF_MOE_TAIL_NEXT_RMS=0` is the temporary rollback around the default-on decode chain. The retained route fuses only the already-rounded selected-aggregate ABI; slot-weighted layers preserve the feature-parallel combine followed by the exact unfused RMSNorm because the one-block weighted composite regressed cold production-layer time. | Q3 GPU1 counterbalanced graph decode improves `100.195 -> 101.216 tok/s` (+1.02%) at 512/128 and `107.366 -> 108.383 tok/s` (+0.95%) at 4K/128, with all five paired deltas positive, exact layer-limit 1/4/40 buffers, exact eager/graph trajectory, unchanged final IDs/logits/memory, and no bulk-prefill route change. Q4/PARO slot-weighted paths keep their existing two-kernel math. | Keep the opt-out for one Q3/Q4 defaults-only milestone and task-#16 overlap bisection window. Then remove the env branch while retaining the registered unfused combine→RMSNorm fallback and kernel-level BF16/FP16 tests. |
| GGUF 24GB capacity diagnostics | `HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING` offloads the Q8_0 token embedding from device residency and performs exact host Q8_0→BF16 embedding copies. | Default-off diagnostic. It proves Q4_K_M `128K/128` can fit on GPU1 (`23.400 GiB` tracked / `23.913 GiB` sampled) but disables GGUF HIP decode graph replay, so decode falls to `11.141 tok/s`; not a promoted path. | Remove or demote to a one-off harness after a retained 24GB `128K/128` path keeps device-side graph-class decode, likely via GGUF INT8/full-attention KV or another device-side embedding/cache strategy. |
| GGUF INT8 KV diagnostics | GGUF accepts explicit `--kv-storage int8_per_token_head` for resident full-attention KV, reusing the PARO per-token/head INT8 write/decode kernels plus layer-local temporary BF16 prefill-oracle caches. Short contexts (`<=8192` rounded max positions) retain an additional BF16 mirror cache so primary short gates use exact BF16 decode while still exercising INT8 writes. Long contexts now default to `HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS=8` as a correctness fallback; lower prefixes, pure INT8, key-only (`HIPENGINE_GGUF_INT8_KV_KEY_ONLY=1`), block16 scale granularity (`HIPENGINE_GGUF_INT8_KV_BLOCK16=1`), and custom non-contiguous BF16 masks via `HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS` require `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1` for diagnostics. | Default remains BF16 unless explicit INT8 KV is requested. GPU1 Q4_K_M pure INT8-only diagnostic fit and ran `128K/128` with graph-class decode (`760.724` prefill / `64.923` decode tok/s, `22.911 GiB` tracked / `23.472 GiB` sampled), but W7900 BF16-vs-INT8 no-mirror correctness rejects pure INT8 at `4K/1` (`KL=0.275781`, top-1 agreement `0.5`). The 2026-06-24 layer-local prefill-oracle fix shows the old prefix `3..8` prefill failures were partly a shared-oracle chunk-outer bug; current prefix `8` passes full `128K/128` (`KL mean=0.01448`, top-1 `0.96124`, no persistent BF16 mirror), while prefix `7` still fails `128K/16` top-1. Non-contiguous 3-INT8-layer masks that skip the known-sensitive full-attention layer 7 (`{6,8,9}` and `{5,8,9}` INT8) also failed `128K/16`, so no custom mask is promoted. The real HIP key-only diagnostic is primitive-correct, but prefix `0` fails `4K/1`, prefix `6` fails `128K/16`, and prefix `7` saves less memory than admitted prefix-8 per-token/head while raising prefill peak; no key-only path is promoted. The real HIP block16 diagnostic is primitive-correct too, but forced-long W7900 `4K/1` BF16-vs-block16 gates fail top-1 at prefix `0`, `6`, `7`, and `8`; no block16 path is promoted. Prefix 8 per-token/head is correctness-admitted but not a retained 24GB throughput row. | Remove the short BF16 mirror, BF16-prefix/custom-mask/key-only/block16 envs, and unverified-long env only after an all-INT8 or more compact calibrated KV format preserves GGUF BF16 logits at `4K` and `128K/128` long-context gates and completes a retained 24GB `128K/128` throughput benchmark. |
| GGUF selected-prefill diagnostics | `HIPENGINE_GGUF_T16_DS4_PREFILL` guarded runtime route for resident `gguf_q4_k_t16_v1` DS4/Q8_1 selected-prefill. | Default-off diagnostic. Full-model Q4_K_S GPU1 gate showed useful prefill speed (`1833.185 -> 1989.578 tok/s` at `512/128`, `2159.561 -> 2372.228 tok/s` at `4K/128`) but changed final token IDs versus default (`220/570 -> 3241/1510`) and added `+0.070 GiB` opt-in activation scratch. The scratch is allocated only when the flag is enabled, so default memory/IDs remain unchanged. | Remove or demote to a microbench/test-only path unless a later exact-enough Q8_1/DS4 calibration path preserves default final IDs/logits on `512/128`, `4K/128`, and the `128K/128` promotion gate while keeping memory bounded. |
| GGUF selected-prefill diagnostics | Microbench-only raw-Q4_K/Q8_1 selected-prefill variants in `gguf_q4_k_q8_1_selected_prefill` (`q8-1-dot`, `q8-1-ds4-dot`, `q8-1-ds4-wmma`, `q8-1-ds4-wmma32`, `q8-1-ds4-wmma64`, `q8-1-ds4-preview-wmma32`, `q8-1-ds4-wmma32-ldspack`, and rejected `q8-1-ds4-wmma32-lds`). | Diagnostic-only, not model runtime defaults. The 2026-06-16 DS4 WMMA path is useful as a fragment/math reference. Expanded-Q4 LDS staging regressed `8.210 -> 18.257 ms/call`; packed-Q4 LDS staging recovered to `11.438 ms/call` but still lost to raw WMMA32; the pre-unpacked preview path measured `12.020 ms/call` with higher fixture memory; four-wave WMMA64 was only flat/sub-1% better than WMMA32. These same-shape staging/pre-unpack/independent-wave probes should not become dispatch paths without a later shared-tile reuse win. | After a real GGUF MMQ/T16 prefill path is promoted or the llama.cpp parity detour closes, demote negative variants to tests or remove them from the microbench to avoid a permanent zoo of rejected kernels. |
| Qwen3.5/PARO INT8 KV prefill | `HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION`, `HIPENGINE_QWEN35_INT8_PREFILL_STREAMING_MIN_TOKENS`, `HIPENGINE_QWEN35_INT8_PREFILL_LOW_MEMORY_TOTAL_GIB`, and `HIPENGINE_QWEN35_INT8_PREFILL_ORACLE_RESERVE_MIB` gate direct streaming INT8 prefill after the 2026-06-15 GPU1 sweep found a severe 128K prefill regression. | Default `auto`: use the fast temporary BF16-oracle/AOTriton bridge unless prompts are at least `224Ki` rows **and** device/meminfo pressure says the oracle is unsafe (`total <= 26 GiB` or free memory cannot cover oracle bytes plus a 1 GiB reserve). This keeps W7900/Strix-style larger-memory runs on the fast path while preserving the 24GB 262K scratch win. Direct streaming is not throughput-promoted (`1020.723 -> 23.425 tok/s` at 128K/128). | Remove the gate after a memory-safe fast INT8 prefill path matches the BF16-oracle/AOTriton speed envelope at `512/128`, `4K/128`, and `128K/128` while retaining the 262K no-oracle memory gate; otherwise demote direct streaming to a dedicated diagnostic/scratch-probe path. |
| Qwen3.5 GGUF/PARO tail-four mixed KV | Explicit `tail4_hadamard_group32` policy plus the BF16-attention/packed-retention prefill bridge; PARO keeps `HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION=streaming` as the direct-INT8 diagnostic. | Explicit-only and **not supported/default**. Clean therock-7.15 GGUF gates pass all 11 prompts at 512/8 and 4K/16 plus bounded `mixed_v1` at 128K/16, save exactly 18.75% persistent K/V, and retain zero persistent BF16 shadow. Default still rejects: 4K production prefill/decode regress 0.67%/0.75%, 128K decode regresses 3.82%, and an inferred four-layer BF16 prefill transient raises allocator high water by 0.532 GiB despite 0.470 GiB lower live owned memory. Native PARO separately fails one 512/8 prompt and two 4K/16 prompts; its quality-preserving 256 Ki XTX request-scratch probe OOMs. | Keep as a reproducible diagnostic. For GGUF, remove the 1.002 GiB prefill transient and optimize long-context group32 attention before repeating the full gate; for PARO, require a quality-safe layout and physical capacity pass. Remove rejected direct/oracle plumbing only after a promoted design supersedes it. |
| Qwen3.5 GGUF/PARO native sampler | `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` is the PARO rollback opt-out; `HIPENGINE_QWEN35_NATIVE_SAMPLER=1` enables the separate GGUF correctness candidate. | PARO remains default-on for supported c1 and scheduler-owned c>N serial-per-slot rows. GGUF is explicit-only: supported c1 and dense compatible c>N rows use native row/rows selection with zero full-vocabulary D2H; forced/dynamic/unsupported combinations remain host-backed. The W7900 c4 correctness artifact passes fixed-seed/state/KV/finish/ownership gates, but A3 stops before timing: native-eligible auto tools fail the frozen turn-1 strict-envelope oracle, while 4/4 exact forced-tool turns report `host_logits_sample` / `native_gpu_unsupported_request`. No active-SSE/tool-ready claim exists. | The A3 non-promotion trigger has fired. In a dedicated cleanup unit, remove the GGUF env branch and resident workspace wiring while retaining the standalone sampler/kernel correctness gates, unless a model-general native forced/dynamic-queue or constrained auto-tool design is approved and separately correctness-gated first. Separately remove the PARO opt-out after its release window. |
| GGUF MoE decode graph | `HIPENGINE_GGUF_MOE_GRAPH` opt-in around per-layer rows==1 MoE FFN capture/replay (`hipengine/runtime/moe_graph.py` + `_run_decode_layer_graphed`). | Default-off. Proven bit-exact on 35B-A3B Q4_K_M (KL=0, 40 captures / 3800 replays / 0 rejects). Cuts the FFN ~64% in launch count (~440->40/token) but multi-trial (3x32 steps) shows a **consistent ~0.84% wall regression** (eager 18.12 vs graph 18.27 ms/step, non-overlapping p10-p90 bands) — NOT noise. The host was already ahead of the GPU (decode is bandwidth/compute bound), so removing launches reclaims no wall and the graph launch overlaps marginally worse with the surrounding eager attention. Fails the non-regressive gate, so NOT promoted. Kept as the validated A/B lever proving launch-count is not the decode bottleneck. Artifact `benchmarks/results/2026-06-28-moe-graph-rows1-ab.json`, WORKLOG 2026-06-28. | Remove the flag + `MoeGraphCache` wiring once the bandwidth-bound conclusion is accepted and no rows=4 verifier-graph follow-up is planned. Re-A/B (and only then reconsider default-on) if a future per-GEMV bandwidth cut shifts the bottleneck back to host-dispatch and makes the launch saving net-positive on wall. Keep `moe_graph.py` + its unit gate as a reusable tool even if the runner wiring is removed. |
| GGUF row-compact MoE GEMV | `HIPENGINE_GGUF_ROW_COMPACT_GEMV` opt-in around `_try_run_post_attention_moe_rows_compact_gemv` for rows>1 selected-MoE verifier blocks. | Default-off. Rechecked on 2026-07-01 against the current llama-compat B2 dp4a all-sync smoke after direct-state cleanup. It regressed badly: B2 **36.05 tok/s**, cycle **27.765 ms/output**, `target_block_verify_total` **24.277 ms/output**; the new `target_block_linear_attn_ffn_moe_compact_gemv` bucket alone cost **8.977 ms/output**. Current split selected-MoE GEMVs are faster for this verifier shape. Artifact `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-rowcompact-allsync-smoke.json`, WORKLOG 2026-07-01. | Remove the runtime env gate or demote the compact row-GEMV path to tests/microbench-only unless a new compact scheduler/kernel beats the split selected-MoE path on a full-suite `llama-compat-device-chain-dp4a` B2 run with unchanged acceptance. |
| GGUF dense Q8 dp4a sidecar | `HIPENGINE_GGUF_Q8_0_RAW_SIDECAR` materialization sidecar plus `HIPENGINE_GGUF_DENSE_Q8_DP4A` / `--verify-dense-q8-dp4a` and `HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL` / `--verify-dense-q8-dp4a-all`, routed by `_try_launch_dense_q8_pair_dp4a`, `_try_launch_dense_q8_single_dp4a`, and `_try_launch_dense_q8_triple_dp4a` for rows>1 verifier blocks. | Default-off. Added for the llama.cpp replication lane. The original route paid a q8_1 quantize launch plus two singleton dense Q8 GEMV launches and lost on B2 smoke; the rowtile-pair retry improved smoke/all-sync verifier timing but full-suite regressed **60.36 -> 59.42 tok/s**, cycle **16.587 -> 16.852 ms/output**, acceptance **0.583 -> 0.559**, target rows/output **1.250 -> 1.322**, and verifier drain **13.023 -> 13.093 ms/output**. The broader all-sidecar route adds raw singleton and Q/K/V triple wrappers and cuts the block profile dense-Q8 bucket **11.420 -> 8.902 ms/block** / kernel **26.053 -> 23.427 ms/block**; full-suite improves speed **60.36 -> 60.89 tok/s** and verifier drain **13.023 -> 12.742 ms/output**, but acceptance regresses **0.583 -> 0.567** and draft acceptance **0.700 -> 0.655**. Later retained lanes add Q8 shared dual, X8 draft lm-head, and F32 `ssm_out` on top of this all-sidecar base. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-{smoke,full}.json`, and earlier denseq8 rowtile-pair artifacts, WORKLOG 2026-07-01. | Keep only as part of the named accuracy-traded llama-compat route while parity work is active. Remove loose env/bench/suite variants after the current llama-compat audit unless a true llama-style Q8 layout/scheduler beats the active compat lane on the full suite, or unless the compat acceptance contract is explicitly changed. |
| GGUF verifier F32 dense-Q8 dp4a diagnostic | `HIPENGINE_GGUF_DENSE_Q8_DP4A_F32` / `--verify-dense-q8-dp4a-f32` plus suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm{,-allsync}` route rows>1 direct-state F32 `ssm_out` through F32 q8_1 quantization plus the raw-Q8 dp4a singleton body. | Default-off globally; retained only for the accuracy-traded llama-compat lane. Isolated block profile moved host **32.470 -> 30.936 ms/block** and kernel **23.893 -> 22.881 ms/block**; same-session smoke moved **70.74 -> 71.43 tok/s** with identical acceptance; full-suite B2 moved **61.31 -> 63.63 tok/s**, cycle **16.331 -> 15.735 ms/output**, verifier drain **12.662 -> 12.158 ms/output**, acc/output **0.567 -> 0.578**, and target rows/output **1.299 -> 1.266**. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all-x8top1-f32ssm.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-f32ssm-{control-smoke,smoke,full}.json`, WORKLOG 2026-07-01. | Keep only as part of the named compat route while the safe verifier transaction gap is audited. Do not promote to exact default unless an exact/non-regressive replacement exists. Collapse this flag behind the final named compat route or remove it during post-compat cleanup if a later verifier rewrite supersedes direct F32 q8_1/raw-Q8 dp4a. |
| GGUF verifier shared-Q8 dp4a diagnostic | `HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED` / `--verify-dense-q8-dp4a-shared` plus suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-sharedq8{,-allsync}` route verifier shared-expert `ffn_gate_shexp`/`ffn_up_shexp`/`ffn_down_shexp` through the raw-Q8 q8_1/dp4a helpers. | Default-off and rejected on the then-active llama-compat B2 lane. Isolated block profile moved kernel time **23.893 -> 23.648 ms/block** and smoke improved **70.64 -> 71.66 tok/s**, cycle **14.181 -> 13.978 ms/output**, verifier drain **11.377 -> 11.183 ms/output** with identical smoke acceptance. Full-suite rejected it: then-active `denseq8all-x8top1` **61.31 tok/s**, cycle **16.331 ms/output**, acc/output **0.567**, target rows/output **1.299**, verifier **12.662 ms/output**; sharedq8 **59.63 tok/s**, cycle **16.793 ms/output**, acc/output **0.556**, target rows/output **1.333**, verifier **13.038 ms/output**. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all-x8top1-{refresh,sharedq8}.json` and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-sharedq8-{control-smoke,smoke,full}.json`. | Remove the env/bench/suite route during post-compat flag cleanup unless a later fused shared-expert body or launch-collapsed shared route beats the active compat lane on the full suite with unchanged acceptance/economy. Do not promote this per-projection q8_1/dp4a shared path. |
| GGUF resident MTP draft Q8 shared dual | `HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL` opt-out around the default-on raw-Q8 dual F32/F32 GEMV for resident draft shared gate/up projections. | Default-on. Added 2026-07-01 for the llama-compat lane and exact resident draft path. It is bit-exact vs two single `gguf_q8_0_gemv_f32_f32_out` launches (`tests/test_gguf_k_gemv.py::test_q8_0_dual_f32_matches_two_single_gemvs`). Draft rocprof A/B reduced `gguf_k_prefill_out` from 16 -> 12 calls/cycle and added `gguf_k_dual_prefill_out` 2 calls/cycle; same-session smoke improved **69.44 -> 70.20 tok/s** with identical acceptance, and full-suite llama-compat improved **60.96 -> 61.19 tok/s** with unchanged acceptance/economy. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q8shared-{control,dual}.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-q8shareddual-full.json`, WORKLOG 2026-07-01. | Remove the opt-out branch and make the dual call unconditional after the next full-suite default exact and semantic-safe llama-compat parity reruns stay non-regressive, unless a later draft rewrite supersedes the shared-expert path. |
| GGUF resident MTP draft dense-Q8 dp4a stage selector | `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A` plus `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES` / `--resident-mtp-draft-dense-q8-dp4a-stages` route resident draft F32 dense projections through F32->q8_1 plus raw-Q8 dp4a float-output wrappers by stage. | Default-off globally and retained only in the accuracy-traded llama-compat lane with `stages=draft`. The legacy all-stage route, including initial KV seeding stages, regressed full-suite B2 **64.41 -> 64.14 tok/s** with worse acc/output and target rows/output. The draft-only selector preserved row economy and moved the then-active unsafe direct-state compat row **74.39 -> 75.15 tok/s**, cycle **13.463 -> 13.325 ms/output**, and `draft_initial` **2.204 -> 2.066 ms/output** with unchanged acc/output **0.621**, draft acceptance **0.820**, and target rows/output **1.136**. That performance row is now superseded as an exact-state claim. The current llama-style directcommit replication row is **60.56 tok/s**, cycle **16.534 ms/output**, verifier drain **14.071 ms/output**, replay/commit **0.043 ms/output**, target rows/output **1.172**, and zero replay rows; the serial-state exact control remains **51.85 tok/s** / **19.308 ms/output**. Artifacts `benchmarks/results/2026-07-02-ar-mtp-llama-compat-draftdenseq8-draftonly-full.json`, `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-full.json`, `benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json`, `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-draftdenseq8-draftonly-gpuevents.json`, WORKLOG 2026-07-02. | Keep only as part of the named compat route while the llama-replication lane is under parity audit. Do not treat the unsafe 75.15 row as a cleanup/promote trigger. Collapse/remove the selector after the final compat route is settled or a verifier/draft rewrite supersedes this route. |
| GGUF resident MTP draft Q6 top-1 X8 sidecar | `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=x8` / `--resident-mtp-draft-q6-top1-stage1-shape x8` plus suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1{,-allsync}` for an X8-packed Q6_K draft lm-head top-1 sidecar. | Default-off globally; retained only for the accuracy-traded llama-compat lane. It materializes `output.weight[:vocab]` into contiguous groups of eight GGUF Q6_K rows and routes the q8_1/dp4a top-1 stage1 through `gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1`. Correctness passes against the q8_1/Q6_K oracle. Same-session smoke improved **71.53 -> 71.76 tok/s** with identical acceptance; draft rocprof moved stage1 **3.603 -> 3.558 ms/cycle**; full-suite compat moved **61.19 -> 61.31 tok/s**, cycle **16.364 -> 16.331 ms/output**, and `draft_initial` **3.378 -> 3.352 ms/output** with unchanged acceptance/economy. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-{control-smoke,smoke,full}.json`, `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1.json`, WORKLOG 2026-07-01. | Keep only as part of the named llama-replication route while the safe verifier transaction gap remains under analysis. Remove/demote the X8 sidecar and route variants if a later fused draft lm-head/sampler or different Q6_K body/layout supersedes it, or if parity closure decides the accuracy-traded llama-compat lane should not retain separate draft lm-head sidecars. Do not promote to exact default without exactness/full-suite correctness evidence. |
| GGUF selected-down X8 repack | `HIPENGINE_GGUF_SELECTED_X8_REPACK` materialization gate plus bench flag `--selected-down-x8-repack {off,q5,q6,both}` for Q5_K/Q6_K selected-down X8 q8_1/dp4a replacement layouts. | Default-off globally. Retained only for the accuracy-traded llama-compat B2 lane with `q6`; first-class suite route is `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6`. Full suite **59.63 -> 60.36 tok/s**, `cycle_wall_ms_per_output` **16.793 -> 16.587**, and `target_block_verify_total` **13.178 -> 13.023 ms/output**. q5/both remains rejected for that route (`64.81 tok/s` smoke vs q6-only `69.03 tok/s`), so Q5_K selected-down stays on T16. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-full.json`, `...x8q6-allsync-smoke.json`, route smoke `...x8q6-route-smoke.json`, and `benchmarks/results/2026-07-01-llama-compat-b2-x8-selected-down-dp4a-current-micro.json`. | Remove/demote q5/both materialization from performance paths unless a future full-suite route beats q6-only with unchanged acceptance. Do not promote to exact default without exactness/full-suite correctness evidence. Once the compat lane is final, consider collapsing the env gate behind the named route and leaving raw env use to tests/microbenches. |
| GGUF selected gate/up X8 repack | `HIPENGINE_GGUF_SELECTED_GATE_UP_X8` materialization gate plus bench flag `--selected-gate-up-x8` and suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup{,-allsync}` for Q4_K selected gate/up X8 q8_1/dp4a replacement layouts. | Default-off and rejected on the current retained llama-compat B2 lane. Same-session smoke regressed **67.62 -> 59.08 tok/s**, cycle **14.810 -> 16.948 ms/output**, and target verifier drain **12.005 -> 14.117 ms/output** with identical smoke acceptance (`acc/output=0.667`, draft acceptance `1.000`). All-sync attribution shows the loss is the selected gate/up GEMV body: linear-attn gate/up GEMV **1.408 -> 3.050 ms/output** and full-attn gate/up GEMV **0.462 -> 1.015 ms/output**, while q8_1 quantize is unchanged/slightly lower. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup{,-control}-smoke.json` and `...x8gateup{,-control}-allsync-smoke.json`, WORKLOG 2026-07-01. | Remove the bench/suite route during the post-compat flag cleanup unless a different Q4 X8 scheduler/body beats retained T16 dp4a on the same async/full-suite route with unchanged acceptance. Future selected-MoE work should compare against llama.cpp `mul_mat_vec_q_moe` rather than broadening this X8 gate/up path. |
| GGUF selected gate/up raw materialization | `HIPENGINE_GGUF_SELECTED_GATE_UP_RAW` materialization gate plus bench flag `--selected-gate-up-raw` and suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup{,-allsync}` for keeping Q4_K selected gate/up experts in raw GGUF layout under decode-repack. With `--verify-dp4a`, the runtime uses the raw selected-dual q8_1/dp4a body instead of the retained T16 replacement-layout body. | Default-off and rejected on the current retained llama-compat B2 lane. Same-session smoke regressed **68.55 -> 62.04 tok/s**, cycle **14.612 -> 16.142 ms/output**, and target verifier drain **11.792 -> 13.328 ms/output** with identical smoke acceptance (`acc/output=0.667`, draft acceptance `1.000`). All-sync attribution shows the loss is the selected gate/up GEMV body: linear-attn gate/up GEMV **1.422 -> 2.153 ms/output** and full-attn gate/up GEMV **0.461 -> 0.729 ms/output**, while q8_1 quantize changes by only ~0.01 ms/output. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup{,-control}-smoke.json` and `...rawgateup{,-control}-allsync-smoke.json`, WORKLOG 2026-07-01. | Remove the bench/suite route during the post-compat flag cleanup unless a new raw-GGUF scheduler/body beats retained T16 dp4a on the same async/full-suite route with unchanged acceptance. Do not retry a mechanical raw `mul_mat_vec_q_moe` body copy for selected gate/up; the measured path is slower than retained T16. |
| GGUF decode-graph rollback and benches | `HIPENGINE_GGUF_DECODE_GRAPH=0` keeps eager as a production rollback; `scripts/qwen35_gguf_bench.py --graph-replay-decode` remains an explicit measurement surface; `scripts/gguf_mtp_bench.py` still has separate `--target-graph-verify` / `--target-graph-batched-verify` modes. | gfx1100 defaults to the state-bound graph for non-streaming c1 greedy windows with at least 24 remaining transitions after clean W7900 p512/d24 SOL-G5 passed 24/24 state/token checks and moved capture-inclusive wall **30.5364 -> 12.5139 ms/token (2.4402x)**. The current gfx1151 one-queue recheck also retains graph: exact 128-token trajectories improve over eager **+1.00%/+0.86%** at 512/4K across 1+3 and **+0.36%** in a bounded 128K confirmation. The final production refresh publishes exact graph rows through 64K; its missing 128K topline is a prefill lifecycle blocker, not graph rejection. The gfx1151 resident BF16 owner now captures state-bound packed C2/C4/C8 graphs per complete wave and replays them for greedy blocking/SSE. Physical C8 improves blocking **75.702 -> 83.771 tok/s (+10.66%)**; low-width capture costs **2.24%/1.58%** blocking at C2/C4 but improves the primary SSE surface, so graph remains default-on with eager rollback. Sampled, short, INT8-KV, host-embedding, and per-layer-MoE-graph routes remain eager. MTP graph modes are not part of this result. | Keep the opt-out through one release window and the next matched gfx1100/gfx1151 refresh. Then make admitted graph routes unconditional if no release regression appears. Before removing the rollback, either reuse state-safe captures across low-width waves or otherwise eliminate their fixed capture tax without regressing SSE. Remove or isolate stale MTP graph modes separately. |
| GGUF AR-baseline timing contract | `gguf_true_ar_category_bench.py` and `scripts/gguf_ar_mtp_suite.py` request state-bound graph admission and record the effective per-prompt route. | Repaired for backend-qualified production timing: gfx1100 horizons >=24 report `graph_replay` with capture/instantiate/close included; unsupported/short horizons honestly report `eager_step`. This replaces the stale unconditional eager artifact while preserving an explicit no-graph diagnostic. The older attachment validator still has a fixed graph-only requirement and does not yet model backend/horizon admission. | Teach the attachment validator to consume the recorded backend capability/effective route instead of one global graph-only constant, then remove duplicate suite-side protocol logic. Preserve anti-gaming rejection of raw/non-production timing. |
| MTP P1 verifier | `HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL` opt-out around the promoted split-output dual W4 shared-gate/up route. | Default-on after 2026-06-11 D32 9-prompt exact A/B: same acceptance, verify `22.98 -> 22.37 ms/cycle`. | After the next retained MTP gate with defaults-on passes at the target sprint shape, remove the opt-out or demote it to a test-only override. |
| MTP P1 verifier | `HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED` opt-out around promoted `f32_to_fp16 + paro_rotate1` fusion. | Default-on after raw-bit RED test and 2026-06-11 D32 9-prompt exact A/B; removes 30 launches/pass and contributes to the stacked `-0.60 ms/cycle` suite delta. | After the next retained MTP gate with defaults-on passes, collapse the old runtime dispatch branch if no other path still needs it. |
| MTP P1 verifier | `HIPENGINE_SELECTED_MOE_DOWN_STAGED` opt-in around the superseded staged selected SiLU/down-rotate + down GEMV path. | Flipped default-off on 2026-06-11 after current graph-auto D32 9-prompt exact A/B: identical acceptance, cycle `27.648 -> 27.408 ms/cycle`, verify `22.377 -> 22.131 ms/cycle`. The 2026-06-12 graph-off current-best compound retest also stayed exact but regressed ratio `0.8252x -> 0.8204x`, cycle `21.661 -> 21.763 ms/cycle`, and verify `16.511 -> 16.628 ms/cycle`. The staged path remains available with `=1` for bisection and historical comparison. | After the next retained MTP gate with defaults-on passes, remove the staged runtime branch or demote it to a kernel test-only path unless a new barrier-free implementation beats the fallback. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS` opt-out around skipped discarded expert-topk host reads, update-only lm-head/argmax results, and final draft snapshot saves. | Default-on after 2026-06-11 D32 9-prompt exact gates: same acceptance/visible tokens, read/result skip moved actual speed `0.664x -> 0.670x`, cycle wall `27.94 -> 27.68 ms`, proposal/update `2.145 -> 2.052 ms`; final-snapshot skip then stayed exact `9/9`, skipped `142` D2D snapshot saves, and trimmed proposal/update `2.052 -> 2.045 ms` with flat actual ratio within noise. | After the next retained MTP gate with defaults-on passes, remove the opt-out or demote it to a test-only override. Keep the functional code path; it is the desired proposer behavior. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_PACK_TOKEN_POSITION` opt-out around the packed token+position metadata H2D copy. | Default-on after 2026-06-11 same-tree D32 9-prompt exact A/B: exact `9/9`, identical acceptance, wall `26.922 -> 26.869 ms/cycle`, proposal/update `1.9766 -> 1.9758 ms/cycle`; ratio is noisy/down because the AR control changed. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep the packed one-copy metadata path. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_ROUTE0_ACCUM_INIT` opt-out around route-0 FP32 MoE accumulator initialization. | Default-on after 2026-06-11 D32 9-prompt exact A/B: exact `9/9`, identical acceptance, standalone `moe_accum` memset removed by route 0 overwrite, cycle wall `27.081246 -> 27.079143 ms/cycle`, proposal/update `1.96299 -> 1.95303 ms/cycle`; ratio is noisy/down because AR changed. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep route-0 accumulator initialization as the only proposer MoE accumulation path. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE` opt-out around direct sidecar K/V cache writes. | Default-on after 2026-06-11 D32 9-prompt exact A/B: exact `9/9`, identical acceptance, K rotary and V projection producers write directly into cache slots instead of temp buffers followed by two D2D copies per advance, proposal/update `1.9955 -> 1.9801 ms/cycle`; total wall was flat/noisy-negative because verify moved independently. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep direct cache writes as the only proposer K/V materialization path. |
| MTP/DFlash verifier accept | `HIPENGINE_VERIFY_ACCEPT_PACKED_PAYLOAD` opt-out around the packed accept-summary D2H payload. | Default-on after 2026-06-11 MTP D32 9-prompt exact same-tree A/B: exact `9/9`, identical accepted lengths and active budgets, cycle wall `27.279 -> 27.122 ms/cycle`, verify `22.162 -> 21.997 ms/cycle`; packs seven tiny D2H reads into one int32 payload read while keeping the legacy output buffers for commit/compatibility. | After one follow-up defaults-only MTP gate and one DFlash chain smoke pass, remove the opt-out or demote it to a test-only override. Keep the packed payload path as the default verifier accept API. |
| MTP/DFlash verifier metadata | `HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA` opt-out around the packed token/position/context metadata H2D path. | Default-on after 2026-06-11 MTP D32 9-prompt exact A/B: exact `9/9`, identical accepted lengths and active budgets, actual ratio `0.68417x -> 0.68898x`, cycle wall `27.02196 -> 26.99252 ms/cycle`, verify `21.87984 -> 21.85918 ms/cycle`; replaces five tiny per-cycle H2D submissions with one packed int64 copy plus `unpack_verify_chain_dynamic_metadata_i64_kernel`. Rocprof confirms the kernel; a 27B dense DFlash D16 one-prompt shared-path smoke passed. | After one follow-up defaults-only MTP gate and the next full 27B DFlash hardening/defaults-only gate pass, remove the opt-out and keep the packed metadata path. |
| MTP/DFlash verifier commit | `HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED` opt-out around the chunked linear-state commit copy grid. | Default-on after 2026-06-11 MTP D32 9-prompt exact A/B: exact `9/9`, identical accepted lengths and active budgets, verify `21.8518 -> 21.8308 ms/cycle`; rocprof moved `linear_state_pair_commit` `0.250 -> 0.203 ms/pass`, total verifier kernel `14.395 -> 14.341 ms/pass`, and host marker `18.301 -> 18.263 ms/pass`. Whole-cycle wall was neutral/noisy, so this is retained as a verifier sub-window micro-slice. A 27B dense DFlash D16 shared-path smoke passed. | After one follow-up defaults-only MTP gate and one DFlash chain smoke/defaults-only gate pass, remove the opt-out and keep the chunked 64 KiB commit grid. |
| MTP verifier host cache | `HIPENGINE_VERIFY_SCRATCH_CACHE` opt-out around fixed-shape verifier scratch object caching. | Default-on after 2026-06-11 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates, wall `27.0958 -> 26.7015 ms/cycle`, verify `21.9328 -> 21.5511 ms/cycle`, and actual ratio `0.6860x -> 0.6987x`. Graph-auto profile showed only a small steady replay host change (`18.290 -> 18.275 ms/pass`), while graph-off control showed the raw Python rebuild win (`33.469 -> 32.988 ms/pass`). | After the next retained MTP defaults-only gate passes with scratch, tensor lookup, and resident view caches enabled, remove the opt-out and keep the workspace-validated scratch cache as the only verifier scratch reservation path. |
| MTP verifier host cache | `HIPENGINE_WEIGHT_TENSOR_LOOKUP_CACHE` opt-out around immutable model tensor lookup memoization on each decode state. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates, wall `26.6621 -> 26.6433 ms/cycle`, verify `21.5290 -> 21.4984 ms/cycle`, and actual ratio `0.69160x -> 0.69200x`. Graph-auto profile was neutral/noisy (`18.218 -> 18.236 ms/pass`), while graph-off isolated the raw Python lookup win (`34.757 -> 32.288 ms/pass`). | After the next retained MTP defaults-only gate passes, remove the opt-out or demote it to a test-only override; keep raw-name tensor lookup memoization as the default host path. |
| MTP verifier host cache | `HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE` opt-out around resident non-owning Tensor view caching. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates and identical per-prompt accepted lengths, wall `26.6424 -> 26.4259 ms/cycle`, verify `21.5059 -> 21.2785 ms/cycle`, and actual ratio `0.69239x -> 0.69857x`. Graph-auto profile was neutral/noisy (`18.235 -> 18.244 ms/pass`), while graph-off isolated raw host improvement (`32.52 -> 31.70 ms/pass`). | After the next retained MTP defaults-only gate passes, remove the opt-out or demote it to a test-only override; keep cached `_slot_linear_state`, `_slot_full_cache`, and `_full_cache_all_slots` views as the default host path. |
| MTP verifier scratch policy | `HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED` opt-out around c1/grouped verifier MLP scratch selection. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates plus identical accepted lengths/active budgets, wall `26.3089 -> 25.6898 ms/cycle`, verify `21.1757 -> 20.5228 ms/cycle`, and actual ratio `0.7003x -> 0.7172x`. Graph-auto profile kept `932` calls/pass and moved host `18.314 -> 18.246 ms/pass`; graph-off host moved `32.445 -> 32.273 ms/pass`. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep verifier MLP scratch reservation aligned with `_verify_moe_grouped_min_tokens()` as the only path. |
| MTP verifier host cache | `HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP` opt-out around generation-stamped verifier scratch cache hits. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates plus identical accepted lengths/active budgets, wall `25.7085 -> 25.5955 ms/cycle`, verify `20.5460 -> 20.4342 ms/cycle`, and actual ratio `0.7145x -> 0.7252x`. Graph-auto profile kept `932` calls/pass and moved host `18.322 -> 18.298 ms/pass`; graph-off host moved `32.659 -> 31.971 ms/pass`. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep generation-stamped cache-hit validation as the only verifier scratch cache path. |
| MTP graph-off verifier scratch | `HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY` opt-out around keeping verifier-shaped scratch live between MTP verify cycles. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical accepted lengths/active budgets, graph-off batched wall `37.207 -> 24.076 ms/cycle`, verify `32.069 -> 18.933 ms/cycle`, and actual ratio `0.4969x -> 0.7730x`. Rocprof showed this is host-only cleanup: calls unchanged `932/pass`, kernel `14.332 -> 14.330 ms/pass`, host `32.505 -> 18.272 ms/pass`. The follow-on `decode_batched + graph_off + skip` row is current best at `0.8252x`, `21.661 ms/cycle`, `16.511 ms` verify. | After the next retained MTP defaults-only gate passes and the c1/AR handoff path has explicit coverage for `canonicalize_after=True`, remove the env opt-out or demote it to a test-only override; keep the `canonicalize_after` API only where handoff semantics need it. |
| MTP verifier rejected gate | `HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED` opt-in for fused full-attention Q/Gate split plus FP16-to-FP32 key cast. | Default-off; bit-exact GPU parity vs `qwen35_split_qgate_fp16 + fp16_to_f32`, exact quicksort, and profile-positive for launch count (`932 -> 922` calls/pass, host `18.269 -> 18.234 ms/pass`), but two exact 9-prompt D32 A/B pairs with identical acceptance regressed average wall/verify (`26.925 -> 27.010 ms/cycle`, `21.754 -> 21.828 ms/cycle`). | Remove the runtime gate or demote it to a kernel test-only path after the break-even sprint unless a broader full-layer composite reuses the kernel and beats the prompt-suite gate. |
| MTP/DFlash verifier rejected gate | `HIPENGINE_VERIFY_ACCEPT_UPDATES_POSITION` opt-in for writing resident base-slot position/context from the packed accept kernel. | Default-off; exact quicksort and D32 `9/9`, and the locked profile removed one scalar `set_decode_position_i64` launch/pass (`932 -> 931` calls/pass), but profile host window worsened (`0.2004 -> 0.2010 s` over 11 passes) and the prompt suite regressed wall/verify (`27.038 -> 27.135 ms/cycle`, `21.884 -> 21.969 ms/cycle`). | Remove the runtime gate or fold it into a broader accept/commit composite only if that composite beats the exact prompt-suite gate. |
| MTP/DFlash verifier LM-head | `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD` opt-in for fused W8A16 LM-head + argmax rows. | Default-off; 2026-06-11/12 MTP retests regressed, and clean gfx1151 SOL-S7 now reconfirms exact-but-slower DFlash (`9.676 -> 9.177 tok/s`, -5.16%). The fused body outweighs the saved argmax launch; synchronized accept-readback is only 0.042% of wall. | Removal trigger is satisfied. Demote the runtime branch to a kernel-test-only parity path in a scoped cleanup; do not expose it as a tuning option unless a new schedule beats unfused on gfx1100 and gfx1151. |
| MTP proposer/verifier overlap | `HIPENGINE_MTP_OVERLAP_VERIFY_COMMIT_PROPOSER` opt-in for running proposer update on a side stream while verifier commit drains. | Default-off; 2026-06-12 D32 9-prompt exact A/B kept identical acceptance and hid some verify/commit time (`16.166 -> 16.028 ms/cycle`) but regressed proposer/update more (`1.243 -> 1.438 ms/cycle`), moving wall `19.441 -> 19.506 ms/cycle` and ratio `0.9216x -> 0.9184x`. | Remove the harness flag after the current break-even sprint unless a broader proposer graph/update redesign makes the side stream positive on the exact prompt suite. Keep only generic stream plumbing if another retained path uses it. |
| MTP D64 state-drift diagnostics | `scripts/mtp_chain_e2e_smoke.py --ar-fallback-after-mtp-cycles` diagnostic override. | Default-off; added to bracket the D64 `translation` token-34 resident-state drift by forcing target AR after a fixed number of MTP verifier cycles. It is diagnostic evidence plumbing, not an acceptance policy. | Remove or move to a dedicated debug harness after the D64 target-state audit is fixed and artifacted, or if a per-layer state comparator supersedes forced-cycle bisection. |
| MTP GDN state-drift diagnostics | `HIPENGINE_GDN_CHAIN_TLOOP_VTILE` temporary env selector for chain GDN t-loop VTILE 1 vs retained VTILE 4. | Default path remains VTILE=4. `VTILE=1` helped localize the first accepted-row D64 handoff (`force_after=2/3` exact) but did not fix `force_after=4`, so it is not a promoted speed or correctness path. | Remove after the D64 chain GDN/materialized-state bug is fixed or after a narrower per-layer comparator identifies a different root cause. Do not leave this as a permanent user-facing tuning flag. |
| MTP verifier rejected gate | `HIPENGINE_MOE_FUSED_ROTATE` opt-in for M13.B.1 selected-dual rotate+GEMV. | Default-off; 2026-05-23 W7900 gate stayed exact and removed 40 rotate launches/pass, but regressed total kernel time `17.32 -> 29.76 ms/pass` because the fused kernel repeated the rotation per `(out_pack,row)` block. | Remove or demote to kernel test-only after break-even path stabilizes; only keep runtime access if a new non-redundant staged design replaces it. |
| MTP verifier rejected gate | `HIPENGINE_SELECTED_MOE_STAGED_ROTATE` opt-in for M13.B.3 staged selected gate/up rotate+GEMV. | Default-off; staged/keyed gate-up path is exact but later W7900 verifier-window measurement regressed kernel time (`15.344 -> 15.611 ms/pass`) despite launch-count reduction. | Remove or demote to kernel test-only unless a no-spin/no-barrier-reset design beats the unfused chain on the current D32 prompt suite. |
| MTP verifier rejected gate | `HIPENGINE_SHARED_EXPERT_FUSED_ROTATE` opt-in for M13.B.2 shared-expert rotate+dual GEMV. | Default-off; exact, but the saved rotate launch was replaced by a barrier reset in the original path and the keyed-barrier follow-up was neutral (`15.350 -> 15.365 ms/pass`). | Remove or demote to kernel test-only unless a future full-layer C-dispatch path can reuse it without adding per-launch synchronization overhead. |
| MTP verifier rejected gate | `HIPENGINE_FUSED_RMSNORM_ROTATE` opt-in for M15.4 fused input RMSNorm + PARO rotate2. | Default-off; current-stack retest on 2026-06-11 stayed exact but regressed verifier kernel `13.41 -> 14.09 ms/pass` and host window `18.45 -> 19.05 ms/pass`. | After the MTP break-even path is stable, remove the runtime gate or demote it to a kernel test-only path unless a new implementation avoids the one-block RMSNorm occupancy trap. |
| MTP verifier docs | Older "default-off diagnostic" notes for P1 gates can become stale as promoted defaults land. | `docs/MTP.md`, `benchmarks/README.md`, and `WORKLOG.md` carry historical rows plus current status. | During each MTP sprint commit, update current-status language and leave old measurements only as dated history. |
| GGUF decode graph replay | Session-bound graphs retain a full key and cumulative transition budget; callers can still explicitly capture diagnostic windows outside the production selector. | SOL-G5 resolves the old uncapped debt: the context cap is inferred from `position + max_replay_steps` or validated when explicit; the key covers backend/model/weights/buffers/KV/layers/route/sampler/recording/state generation; replay rejects cursor drift or budget overflow. Current gfx1151 HIP passes 128/128 byte-exact checkpoints. | Keep the strict cap/key checks permanently. Remove only redundant legacy graph arguments after the MTP diagnostic modes are retired and all callers use the state-bound API. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --no-target-graph-verify` eager target-verification opt-out. | Default path now uses capped resident decode-graph target verification with fp32 hidden-seed capture; the eager opt-out is useful for bisection and correctness/perf comparison but is >2x slower on the B1 full suite. | Remove or move to a dedicated debug harness once the capped target graph has survived the next MTP break-even sprint and a multi-token GGUF graph correctness gate is part of the regular validation bundle. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --no-mtp-draft-warmup` cold-start diagnostic opt-out. | Default path now runs one stateless untimed draft warmup so MTP timing matches the true-AR warmup protocol; the cold-start opt-out is useful only for measuring wrapper/library/weight-cache first-use cost. | Remove or move to a dedicated cold-start harness once the MTP draft runtime has persistent device buffers and per-process cold-start cost is documented separately from steady-state tok/s. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --root-topk-accept >1`, `--sibling-topk-accept >1`, and `--topk-branch-redraft` exact-verified coverage tree. | Default-off after the 2026-06-23 speed-first reset to B1 linear greedy: K4096 B5 raised accepted/output but regressed draft efficiency (`0.000227`) and tok/s versus B1 linear. Keep it only for coverage diagnostics and llama.cpp parity investigations. | Remove or move to a dedicated experiment harness if the real GGUF MTP verifier path/adaptive policy supersedes top-k tree diagnostics, or if repeated full-suite speed-first gates show the tree remains speed-negative. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --mtp-context-replay` slow default-off prompt-catchup replay path. | Useful to bracket llama.cpp-style MTP prompt catch-up while the resident bulk target path exposes only the final hidden row. Current smoke is acceptance-negative and should not be the default benchmark path. | Remove or move to a dedicated debug harness after bulk prefill exposes all-row fp32 hidden seeds and the real MTP KV/RoPE path is implemented, or if that path supersedes replay diagnostics. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --mtp-device-kv-cache` default-off device-resident MTP dense KV context. | Implemented for B1 llama.cpp-parity investigation: MTP attention writes post-RoPE K/V to persistent device buffers, draft steps attend over the cache, and accepted target rows use a KV-write-only commit path. Smoke is much faster than host replay/prefix diagnostics but still below the retained no-cache default, so it is opt-in only. | Promote to default only if a same-protocol full-suite row improves raw tok/s, accepted/output, and draft_acceptance; otherwise move to a dedicated debug harness or replace with a paged/KVLiveSpans implementation once bulk hidden-row capture lands. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-graph-batched-verify` full-accept-required verifier graph replay experiment. | Default-off; records generated target IDs and FP32 hidden seeds for a whole strict verifier block. The 2026-06-25 merge-sort B3 smoke stayed exact but was speed-neutral/slower because target kernels still execute sequentially inside the graph and hidden-row recording adds overhead. | Remove unless a true rollback-safe block verifier starts using the recorded hidden rows, or promote only after full-suite evidence shows a speed win over one-step graph replay without requiring prompt-specific full acceptance. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-verify[=bulk/native]` rollback-safe target continuation block verifier. | Default-off; snapshots linear-attention conv/recurrent state with persistent buffers, runs row-bulk target continuation over `[prev]+drafts`, records target IDs + FP32 hidden seeds, and restores/replays the consumed prefix on partial accept. Tiny B3/B5 verifier blocks now default to `--no-target-block-wmma-prefill` because selected/WMMA prefill regressed the B3+32k smoke (`37.8 tok/s`); the GEMV verifier fallback lifts the same smoke to `48.1 tok/s` with `15/15` accepts, but B5 partial rollback remains slow. | Keep as a correctness scaffold and small-B scheduler harness. Promote only if verifier `ar_decode_ms` beats one-step graph replay on full-suite/heldout without reducing acceptance; otherwise replace the linear-attention/rollback pieces with a dedicated small-B continuation kernel and do not re-enable selected-prefill for tiny verifier blocks without evidence. |
| GGUF NativeSpecCycle target route | `scripts/gguf_mtp_bench.py --native-spec-target-cycle` and the `native_v1_b2_target_graph` provider route, now admitting B1/B2 shape buckets. | Retained for explicit accuracy-traded `llama-compat` on both admitted gfx11 backends. One fixed-address graph per B1/B2 bucket uses live device token/position/context/cursor metadata. gfx1100 clean repeats reach **123.332/122.667 tok/s**, preserve every prior 240-ID/96-cycle trajectory and 80.45% acceptance, and clear the 115.444 tok/s llama.cpp floor. The independent gfx1151 transfer reaches **80.132 tok/s** for N1 and **80.099** for public N3 versus **70.020** direct commit (+14.39% N3); all 240 IDs/97 cycle semantics, the real N1/N2 state/KV oracle, and the cached profiler gate pass. Exact/default MTP remains unchanged, and the gfx1151 N3P proposal graph remains unregistered. | Keep target-only N1 as the proven performance/rollback boundary while gfx1100 N3 remains below it. Remove the user-facing target-only flag only when complete-cycle ownership is non-regressive on both backends. Do not route exact/default MTP or automatic exact generation through the accuracy-traded contract. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-direct-state-commit` and suite routes `resident-strict-block-direct-commit` / `resident-hybrid-strict-block-direct-cap32k` / `resident-hybrid-strict-block-direct-native-cap32k` / `resident-b1-branch-safe-direct-cap32k-device-seed`. | Default-off diagnostic. It materializes per-row GGUF linear-attention Conv/GDN state during target block verification and commits captured verifier rows directly when the verifier mode is serial-exact/native, or default `bulk` with a short verifier block (`end < 1024`). Native mode is scalar-byte-exact through row 1 after the row-serial full-attention fallback was fixed to use absolute continuation positions and capture row states. Optimized bulk mode now follows the admitted peer-aligned reassociated GDN contract rather than scalar byte identity. Under the production no-copy capture mode (`HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1`), W7900 gates prove B5 row 0 is byte-identical to an independent B1 bulk prefix and a deferred B2 row-1 commit is byte-identical to normal B2 final state; serial-exact/native remain the scalar-exact controls. 2026-06-29 smoke is still not positive: pure strict B3 **37.20 tok/s = 0.678x AR**, hybrid direct B3 **49.01 tok/s = 0.893x AR**, native hybrid direct B3 **48.17 tok/s = 0.875x AR**, and B1 branch-safe direct **26.66 tok/s = 0.4849x AR**. | Keep only as rollback-slot verifier scaffolding. Remove or move to a dedicated experiment harness unless a future full-suite/heldout row uses exact direct commit inside a verifier-amortization path that beats true AR. Do not promote from row-level exactness or smoke-level diagnostics alone. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-replay-state-commit`. | Default-off diagnostic. It scores strict target blocks with the selected block verifier without direct linear-state capture, then restores and replays the accepted prefix through `verify_target_block_serial_exact()` for resident state. The corrected 2026-07-02 13-cycle F32 selected-intermediate run proves the transaction wiring (`target_verify_replay_rows=38`, `target_verify_direct_commit_rows=0`) but rejects it as a replication path: it diverges early at cycle 2 (`[40798, 1590]`) and falls to **31.14 tok/s** because every accepted prefix pays serial replay. | Remove after the direct-state capture path is made semantically identical to the intended block scoring path, or move this to a dedicated debug harness if it remains useful as a state-lifecycle comparator. Do not promote; it is intentionally slower and negative semantically. |
| GGUF MTP capture-path diagnostics | Default-off env probes `HIPENGINE_GGUF_VERIFY_CAPTURE_F32_CHAIN_CONV`, `HIPENGINE_GGUF_VERIFY_CAPTURE_REGULAR_CHAIN_GDN`, `HIPENGINE_GGUF_VERIFY_CAPTURE_BF16_GDN_OUT`, `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN`, `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV`, and `HIPENGINE_GGUF_VERIFY_CAPTURE_SCORE_PREFILL`, plus diagnostic Conv/GDN row-state wrappers `qwen35_linear_attn_conv_prefill_f32_state_rows` and `qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows`. | Added 2026-07-02 to split the forced pair-12 direct-state mismatch. The diagnostic artifact `benchmarks/results/2026-07-02-mtp-capture-path-diagnostics.json` rejects the simple token-output fixes: BF16 GDN output, prefill-shaped Conv/GDN state rows, and score-prefill/chain-commit all still sample `[15495, 539, 1151]` with row-1 margin **+0.29526**. The later lifecycle comparator shows `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1` is required for byte-identical full-accept captured state, so `--llama-compat` now forces that env while the other capture flags remain diagnostic-only. The 2026-07-18 W7900 gate confirmed the old default-off chain-capture test route can produce sentinel/NaN output; the production prefill-GDN route is finite and makes B5 row 0 exactly prefix-equivalent to independent B1 across token, 2,048 hidden values, and 16,711,680 Conv/GDN state floats. Added 2026-07-03 `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV` to test the raw-state finding that selected Conv state has much larger default-vs-prefill-GDN pairwise drift than recurrent state. | Remove or move the unused capture probes to a dedicated debug harness after the verifier FP32 hidden/KV history contract is aligned against llama.cpp. Keep the prefill-GDN capture mechanism as part of llama-compat until a narrower always-prefix-equivalent row-state capture supersedes it. Do not expose the other flags as user tuning; they are negative diagnostics unless the hybrid chain-Conv/prefill-GDN route wins the forced-pair and suite gates. |
| GGUF MTP state-lifecycle diagnostics | `scripts/gguf_mtp_forced_target_probe.py --state-lifecycle-compare` hashes post-cycle FP32 hidden seed plus per-layer Conv/GDN resident state for replay-state vs direct-state verifier policies. | Default-off diagnostic. Added 2026-07-02 for the active llama-compat trace. Base direct capture first mismatches replay at cycle 0 despite identical visible tokens `[12305, 198, 727]`. With `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1`, cycles 0-2 become byte-identical and the first mismatch moves to cycle 3, a partial/reject cycle where both policies emit `[65342]` but direct commit diverges from `serial_exact_accepted_prefix` in hidden seed and all linear states. The partial-commit policy fix keeps direct commit for full accepts but serial-replays the accepted prefix on rejected bulk blocks; `benchmarks/results/2026-07-02-mtp-state-lifecycle-prefillgdn-partialfix-compare.json` is clean through cycle 12 (`first_mismatch: null`). The retained serial state-only replay comparator is also clean and improves the exact control **50.96 -> 51.85 tok/s**. The new directcommit comparator `benchmarks/results/2026-07-02-mtp-state-lifecycle-directcommit-partial-compare.json` intentionally diverges from serial replay at cycle 3 while emitting the same visible token `[65342]`; that is expected for the llama-replication lane, whose full-suite row is **60.56 tok/s** / **16.534 ms/output** with zero replay rows. | Keep while the split contract remains useful: serial-state for exact-state control, directcommit for llama-replication timing. Remove or move to a dedicated debug harness after parity closure picks the final compat transaction policy and a narrower per-state/KV-tail comparator supersedes this broad hash check. |
| GGUF MTP diagnostics | `HIPENGINE_FUSED_LINEAR_STATE_COMMIT` opt-out around the fused captured Conv/GDN row commit in `Qwen35GGUFResidentSession._commit_verify_linear_state_row`; `HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED` selects the existing chunked commit copy grid. | Default-on for direct-commit diagnostic paths only. It reuses the DFlash `linear_state_pair_commit_*` kernels to replace per-layer D2D copies when all live GGUF linear layers have captured verifier rows; it falls back to the legacy per-layer copies for non-uniform state sizes and rejects partial captured-row state through the legacy all-or-error check. Focused unit coverage and the GGUF verifier state exactness gate passed on 2026-06-29. No e2e speed claim is retained for this row; it is rollback-slot scaffolding for a future verifier-amortization path. | Remove the opt-out after a full-suite verifier-amortization path that uses direct commit beats true AR, or delete the GGUF direct-commit experiment if it never contributes to such a path. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-verify-mode serial-exact` and `Qwen35GGUFResidentSession.verify_target_block_serial_exact()`. | Default-off diagnostic. It consumes verifier block rows with the token-serial decode scheduler, stages FP32 hidden rows, and optionally records per-row Conv/GDN state. The focused wrong-branch gate proves direct row-0 commit is bit-exact after `[prev, wrong_child]`, unlike the current row-bulk capture path, but this deliberately does not amortize target weight loads. | Keep only as the rollback-slot correctness oracle while developing the row-bulk/amortized verifier. Remove or move to a dedicated debug harness after an exact transactional verifier beats true AR on the full suite, or if a better per-row oracle supersedes it. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-b1-branch-safe-block-verify` and suite route `resident-b1-branch-safe-block-cap32k-device-seed`. | Default-off rejected diagnostic. It probes a B1/root-topK block verifier that batches `[prev, draft0]`, uses row 1 only for strict draft top-1 accepts, and restores/replays row 0 for root-topK branch accepts/rejects. Row-0 direct commit is now exact in the direct route, but both branch-safe variants are smoke-negative: restore/replay B1 AR **54.93 tok/s**, MTP **31.11 tok/s = 0.566x AR**; direct row-0 B1 AR **54.97 tok/s**, MTP **26.66 tok/s = 0.4849x AR**; accepted/output **0.400**. | Remove or move to a dedicated experiment harness unless a future verifier-row lifecycle plus full-suite row beats true AR. Do not re-run as a goal path while branch-safe B1 remains slower than the retained serial/cap32k routes. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --draft-p-min` on the resident draft host-logits path, `--record-draft-confidence` artifact-only top-1 probability capture, and suite route `resident-strict-context-block-pmin08`. | Default-off rejected diagnostic. The resident path supports probability-threshold diagnostics by reading full logits only when `draft_p_min > 0`, computing top-1 softmax probability, and stopping before appending a weak draft. `--record-draft-confidence` records the same top-1 probability in raw cycle artifacts without changing acceptance; the 2026-06-29 full 10-prompt B1 diagnostic (`/tmp/hipengine-draft-confidence-b1.json`) showed strict top-1 `p>=0.999` was clean (`13/13`) but only **21.7%** recall of strict hits, while `p>=0.98` had **28/29** strict hits. The strict-context/block p=0.8 route is not competitive: smoke AR **55.00 tok/s**, B3 **38.44 tok/s = 0.6991x AR**, accepted/output **0.571**. | Keep the covered probability/confidence plumbing only for diagnostics. Remove the suite route and `--record-draft-confidence`, or move them to a dedicated experiment harness, unless a future full-suite/heldout row proves confidence gating helps a structural verifier-amortization path beat true AR. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --mtp-draft-vocab-cap N` hot-token draft LM-head cap. | Default full-vocab (`0`). A 32k cap improved the corrected merge-sort B3 smoke from `42.29` to `44.51 tok/s` with unchanged `15/15` strict accepts, but acceptance/quality are prompt-sensitive and the cap is not yet suite-validated. | Either promote a cap after full-suite train/heldout/category validation shows non-regressive acceptance and better true-AR speed ratio, or keep it as an explicit experiment knob and document that retained default remains full vocab. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --adaptive-full-vocab-after-cap-miss` and the suite/workbench `resident-cap32k-recover` route. | Default-off diagnostic. It instantiates a second full-vocab resident draft runner and switches to it after a generic capped-vocab zero-accept miss, suppressing permanent AR fallback for that miss. 2026-06-29 partial suite recovered the known cap32k B1 collapse (`accepted/output 17/37 -> 19/39`) but still measured only **52.45 tok/s = 0.958x true AR**; full suite measured **51.71 tok/s = 0.9478x AR**, `mtp_beats_ar=false`; smaller caps were not goal-closing. | Remove or move to a dedicated experiment harness once the real resident MTP lifecycle/verifier-amortization path lands, unless a future full-suite + heldout row proves capped recovery beats true AR without prompt-sensitive cap tuning. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --resident-mtp-device-seed` and suite route `resident-cap32k-device-seed`. | Default-off structural diagnostic. It seeds the resident draft from the target session's device-resident fp32 hidden seed pointer, mirroring llama.cpp's resident `pending_h` lifecycle direction and avoiding the pending-seed host round trip. 2026-06-29 full suite: AR **54.59 tok/s**, best MTP B1 **52.08 tok/s = 0.9540x AR**, accepted/output unchanged vs cap32k recovery at **78/178 = 0.438**, `mtp_beats_ar=false`. | Promote into the final resident lifecycle only if a future full-suite + heldout row beats true AR, or if the real `GGUFMTPDraftContext` absorbs the device-seed path. Otherwise remove or move it to an experiment harness after the verifier-amortization path lands. |
| GGUF MTP diagnostics | Suite route `resident-cap32k-device-seed-kv` plus `scripts/gguf_mtp_bench.py` path combining `--resident-mtp-device-seed` and `--mtp-device-kv-cache`. | Default-off rejected diagnostic. The route uses new device verifier-row staging and device-base accepted-row KV commit plumbing, but without llama.cpp prompt/context catch-up it collapses draft acceptance: B3 smoke **38.94 tok/s = 0.7124x AR**, draft_acceptance **0.032**; B1 smoke **39.73 tok/s = 0.7235x AR**, draft_acceptance **0.017**. | Keep `Qwen35GGUFResidentSession.stage_current_hidden_seed_as_verify_row()` and `Qwen35GGUFResidentMTPDraftRunner.write_kv_rows_from_device_seed_base()` as lifecycle primitives. Remove or move the no-context-replay suite route unless a future prompt-catch-up resident lifecycle uses it and beats true AR on full suite. |
| GGUF MTP diagnostics | Suite route `resident-context-cap32k-device-seed` plus `scripts/gguf_mtp_bench.py` compatibility path combining `--resident-mtp-device-seed`, `--mtp-context-replay`, and `--mtp-device-kv-cache`. | Default-off rejected structural diagnostic. It wires llama.cpp shifted prompt catch-up, resident device `pending_h`, staged verifier rows, and device MTP KV, but the target verifier remains serial: B1 smoke **50.84 tok/s = 0.9257x AR**, accepted/output **0.400**; B3 smoke **46.97 tok/s = 0.856x AR**, accepted/output **0.571**. | Keep only as a compatibility scaffold while building real target-pass amortization. Remove or move to an experiment harness once a block/graph verifier path owns the llama.cpp lifecycle and beats true AR on the full suite. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --adaptive-strict-block-probe` and suite route `resident-hybrid-strict-block-cap32k`. | Default-off rejected diagnostic. It starts with strict top-1 block-promotion probing and falls back generically to root-topK B1 + cap32k recovery when the strict probe under-accepts. 2026-06-29 full suite: AR **54.58 tok/s**, best MTP B3 **50.91 tok/s = 0.9328x AR**, accepted/output **94/194 = 0.485**, `mtp_beats_ar=false`; worse than cap32k recovery B1 **51.71 tok/s = 0.9478x AR**. | Remove or move to a dedicated experiment harness unless a future full-suite + heldout row proves a strict-block hybrid beats true AR. Do not promote based on smoke/partial closeness. |
| GGUF contiguous prefill metadata | `HIPENGINE_GGUF_PREFILL_DEVICE_METADATA=0|1` selects six synchronous per-chunk H2D copies or the stream-ordered `prepare_prefill_chunk_metadata` kernel; gfx1100 and gfx1151 package policies select the device path through 4,096 prompt tokens. | Scoped default-on on both gfx11 backends. gfx1151 clean 512/1K/4K prefill improves **+1.56%/+0.90%/+0.53%** with 83/83 exact state. W7900 control tracing confirms **167.213 us** of six-copy HIP API wall at 512; exact device metadata improves balanced 512/4K aggregate prefill **+0.41%/+2.43%** (paired **+0.26%/+2.26%**), with non-regressive decode and unchanged memory/IDs. gfx1151 explicit 128K still enters its low-power no-progress state, and gfx1100 long-context behavior is unmeasured, so both retain synchronous metadata above 4K. | Keep `=0` and the synchronous branch for one release and indefinitely for >4K until completed architecture-local long-context gates prove otherwise. After one release, remove the short-context host branch/selector only if both backends remain non-regressive; do not widen the ceiling from short-context evidence. |
| GGUF Q8T16 prefill LCP-3 four-wave | `HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE=0|1`, the `wmma_prefill_4wave_bf16_bf16_out` registry variant, and gfx1151 automatic wrapper retain a four-wave duplicate plus the GPF-5A two-wave rollback. | Default-on on gfx1151 through 64K. Clean 512/4K full-model state is 83/83 exact and five-pair prefill improves **1214.510 -> 1220.993 (+0.53%)** and **1269.030 -> 1288.986 tok/s (+1.57%)**; dominant 4K projection micros improve **7.50%-14.08%**. gfx1100 remains production. | After one rollback release and gfx1100 transfer evidence, replace the two-wave body/automatic wrapper and remove its selector if no regression appears. Keep the production long-context path while the predecessor's measured 128K regression remains. |
| GGUF F32 router LCP-4A | Callable `qwen35_router_logits_bf16_f32w_auto_256` duplicates only the Python launch default for the existing exact token-tiled kernel; both gfx1100 and gfx1151 registry bindings now select it. | Default-on on both gfx11 backends. gfx1151 clean 512/4K prefill improves **+2.76%/+3.28%** with 83/83 exact state; W7900 improves **2689.171 -> 2795.242 (+3.94%)** and **2955.867 -> 3070.905 tok/s (+3.89%)**, with a bit-exact 4K primitive, flat graph decode, and unchanged memory. | Keep the 512-thread base callable for one rollback release, then collapse the named wrapper into the normal F32-router binding if no release regression appears. Both architecture-local gates are complete. |
| Laguna gfx1151 router token tile | `LAGUNA_ROUTER_LOGITS_MODE` and session setter retain exact token-tile-4/16 diagnostics around the promoted token-tile-8 route. | Clean seven-pair pp512 improves **497.625 -> 503.349 tok/s (+1.150%)**, wins every pair, keeps every tile-8 sample above 500, and closes the production 500 gate. Production-shape F32 logits, selected IDs, routing weights, complete MoE BF16 output, and token 2930 are exact; cached tracing cuts router **30.658 -> 23.315 ms**. | After one later defaults-only clean refresh remains non-regressive, remove the session setter and token-tile-16 route. Keep token-tile-4 as the low-row/unmeasured-backend fallback. |
| GGUF prefill router select LCP-4B | `HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS=64|128|256|512` overrides both backends' promoted 128-thread package capability for bulk prefill only; decode retains its independent launch. | Default-on at 128 on both gfx11 backends. gfx1151 clean 512/4K prefill improves **+0.34%/+0.36%** and the named family falls **12.539 -> 3.741 ms (-70.17%)**. W7900 aggregate prefill improves **+0.32%/+0.81%** (paired medians **+0.30%/+0.12%**) with bit-exact selected IDs/routing weights, flat graph decode, and unchanged memory. The faster 64-thread gfx1151 primitive remains ineligible due to state divergence. | Keep explicit 512 rollback for one release, then remove the env resolver and retain unconditional 128-thread package capabilities if no release regression appears. Never expose/promote 64 without a genuinely different exact implementation. |
| gfx1151 persistent prefill flight recorder | `Qwen35GGUFResidentSession.prefill_flight_recorder_path`, the `qwen35_readme_sweep.py --prefill-flight-recorder{,-granularity}` CLI, file-backed mapped-host ring, and `prefill_flight_recorder_mark_i64_kernel`. | Default-off diagnostic for the unresolved repeated-128K silent no-progress state. `chunk` records every host layer submission but emits one retirement marker per reset/outer chunk; `layer` is explicit high-perturbation refinement. A 512/1 gfx1151 smoke is exact and rocprof confirms the marker, but recorder runs cannot support performance claims or lifecycle incidence estimates. | Keep through the fixed-kernel/firmware three-policy investigation and upstream triage. After the 128K gate is lifecycle-safe, either generalize it into a maintained runtime tracing facility with a schema contract or remove the session/CLI wiring and marker kernel; do not leave an undocumented diagnostic hot-path branch indefinitely. |
| gfx1151 HIP hardware-queue workaround | `configure_hip_process_environment()` sets `GPU_MAX_HW_QUEUES=1` before loading `libamdhip64` when all recognized visible HIP arches map to gfx1151. Existing user values win; gfx1100 and mixed recognized arches are unchanged. | Risk-reducing default after a clean same-command 128K A/B: ROCm's documented four-queue default stalls in first warmup, while one queue once completes warmup+3 at **499.755 / 500.210/500.873/500.687 prefill tok/s**, exact IDs, and is non-regressive at 512/4K (**+0.35%/+0.46% prefill**). It is not lifecycle-safe: later current-production, router-rollback, and SDMA-disabled full 128K gates all complete warmup then reproduce the stall. A clean HIP 7.13 versus 7.15 matrix also reproduces under both stacks: 7.13 completes two gates but stalls on a third after measured pass 1; 7.15 stalls in both controls. Upstream initial/follow-up evidence is posted to ROCm#5107; cross-stack evidence is in `benchmarks/results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json`. | Keep one queue as risk reduction until a fixed gfx11 firmware/runtime independently completes the bounded 128K 1+3 lifecycle gate. Preserve explicit `GPU_MAX_HW_QUEUES=4` for rollback/diagnosis and provenance. Do not add `HSA_ENABLE_SDMA=0`: one 1+1 screen passes but the full gate fails. Remove the workaround only after fixed-stack evidence, not after an intermittent passing run. |
| GGUF GDN prefill | `HIPENGINE_GGUF_GDN_PREFILL_MODE=auto|exact|fused|chain|chain_k2|chain_peer_wave32|chain_peer_cluster8|chain_tile64|chain_tile32|chain_wave32|chain_wave32_tree|chain_lds64|chain_lds32|chain_lds32_direct|chain_lds32_direct_nonvolatile` is the fail-closed rollback/bisection selector. Explicit `chain` is the GGUF-only exact split; `chain_k2` exposes the registered PARO normalized-Q/K two-wave recurrence; `chain_peer_wave32` exposes GPF-9C's llama.cpp-HIP-shaped normalized-Q/K XOR reduction; `chain_peer_cluster8` exposes GPF-9D's llama.cpp-Vulkan-shaped eight-lane clustered reduction; tile64/tile32 and the older wave32 routes are historical controls. The semantic harness sets `HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE=1` internally before session allocation so materialized candidates receive dedicated Q/K/V scratch rather than production direct-route null views. gfx1100 `auto` now selects `chain_peer_wave32`; gfx1151 selects the byte-exact compiler-cacheable `chain_lds32_direct_nonvolatile` route. | On gfx1151, LCP-2A preserves six-case state plus 250/250 natural transitions exactly and improves balanced 512/1K/4K prefill +34.76%/+36.63%/+36.58%. Under the prospective 18-prompt 0.05/0.90 contract, clean W7900 rejects K2 at KL `0.059031` and register-tree at `0.068757`; both pass top-1 and decode. GPF-9C passes at KL `0.041737`, top-1 `445/450`, and non-regressive decode, but originally missed the 512 floor. LCP-5A removes HIP 7.2 spills from the exact selected-Q5/Q6 T16 prefill leaves, moving clean-target pp512 peer kernels/span to **184.513/194.886 ms**, faster than llama.cpp HIP's **203.301/212.236 ms**. The clean selector-unset 512/4K screen reaches **2588.231/2757.752 tok/s**, clears both floors, keeps IDs stable, and preserves the liveness arena at **21.670 GiB** tracked peak. The final W7900 strict-exact convergence screen selects nonvolatile direct-LDS32: it halves VGPR **64 -> 32**, cuts the 512 trace-family median **74.39%**, and improves volatile-direct full-model 512/4K prefill **+73.01%/+82.46%** with byte-exact primitive state, flat decode, and unchanged compact-scratch memory. The new architecture-scoped `exact` alias exposes that rollback without changing gfx1100 peer-wave production. GPF-9D's clustered route remains rejected on strict decode. | Retain explicit volatile `chain_lds32_direct` for one release beneath the promoted `exact` nonvolatile route, then remove it if no exact-route regression appears. Remove `chain_k2`, `chain_wave32_tree`, and rejected `chain_peer_cluster8` after the final gfx1151 peer transfer decision; retain fused, `chain`, `exact`, and peer-wave as production/correctness/bisection routes. Collapse `HIPENGINE_GGUF_GDN_PREFILL_MODE` to `auto|exact|fused|chain|chain_peer_wave32` after that cleanup. |
| GGUF Q4T16 selected-prefill GPF-3A | `HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE=auto|baseline|shared_x`, explicit baseline/shared-X registry variants, and replay switch `scripts/qwen35_gguf_moe_replay.py --q4-t16-shared-x` retain duplicate Q4T16 compact32 bodies. Both gfx1151 and gfx1100 `auto` now select `shared_x`. | BF16/FP16 fixture bytes are exact; tiny trace is `44.725 -> 33.343 us` (-25.45%), real Q4 gate/up replay is `114.633 -> 97.082 ms` (-15.31%), and clean gfx1151 full-model 512/1K/4K prefill improves +3.11%/+2.42%/+1.94%. The W7900 predeclared borderline repeat improves 512/4K +0.877%/+0.822%, preserves byte-exact logits/trajectories, and improves aggregate decode wall 0.081%. | Retain explicit baseline rollback for one release window after the gfx1100 automatic-route publication, then collapse the losing body/alias and remove the env/replay switches. |
| GGUF Q8T16 prefill GPF-5A | `HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE=0|1` selects production/two-wave on gfx1100 and remains the first gfx1151 rollback beneath the promoted four-wave route; request-scoped package ceilings restore production above 65,536 gfx1151 prompt tokens and above 4,096 gfx1100 prompt tokens. | Exact 32-column waves share 1 KiB LDS; 80 VGPR/zero scratch. Published gfx1151 512-64K prefill is **889.904/919.598/762.940/648.948/546.296 tok/s (+1.01% to +8.57%)**; same-commit 128K rejects two-wave **382.041 vs 392.219 tok/s (-2.59%)**. Independent W7900 focus improves 512/4K **645.901/676.444 -> 654.872/683.164 tok/s (+1.389%/+0.993%)** with exact primitive bytes, so gfx1100 is intentionally capped at the measured 4K scope. | Keep env rollback and both architecture ceilings for one release. Expand the gfx1100 ceiling only after a clean hardware-local long-context A/B; collapse the duplicate only if a replacement schedule passes 128K too. |
| GGUF small-B linear dispatch | `HIPENGINE_GGUF_Q4K_ROWTILE` / `q4k_rowtile_session(False)` opt-out for the weight-amortized raw row-tile GEMV (`rowtile_*` variants for Q4_K/Q5_K/Q6_K/Q8_0, rows 2..8, WMMA off). | Default-on. Bit-exact vs the per-row prefill alias and ~3x faster on the dense projection shape at B=4 (microbench); fires ~250x in the B3 verifier. End-to-end verifier is flat within noise because dense projections are only ~11-17% of the verifier (the MoE selected-expert GEMV is the ~54% bottleneck). The opt-out exists for bisection only. | Make the rowtile path unconditional (drop the env/session opt-out) after the current GGUF-MTP verifier path in `docs/MTP-LLAMACPP-PARITY.md` has a same-protocol full-suite non-regression row; keep the per-row kernel only as the rows==1 / rows>8 path. |
| GGUF selected-MoE dp4a diagnostic | `HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A` opt-in around q8_1 activation quantization plus sudot4 for raw Q4_K selected-dual gate/up and the T16 rows>1 split gate/up path; `HIPENGINE_GGUF_T16_SELECTED_DP4A` opt-in for the broader T16 selected diagnostic, currently Q4 split plus Q5 selected-down; `HIPENGINE_GGUF_RAW_SELECTED_DP4A` opt-in for the raw no-decode-repack Q4/Q5/Q6 selected-MoE diagnostic bundle. | Default-off. The raw Q4 fallback launches correctly with caller-owned q8_1 workspace but production B3 decode-repack does not hit it; isolated raw Q4 POC measured `0.946 ms -> 0.357 ms`. Raw Q5/Q6 selected-down is also positive in isolation (`0.0916 -> 0.0395 ms` Q5, `0.0419 -> 0.0259 ms` Q6 including q8_1 quantize) and improves no-decode-repack B3 `31.63 -> 39.61 tok/s`, but still trails default decode-repack B3 `51.31 tok/s`. The active T16 split path cuts that row-bulk kernel (`~172 us -> ~142 us` in the two-cycle trace), but B3 remains flat (`49.31 tok/s`, warm `50.60`). Q5T16 selected-down also launches and is `1.10x` faster in isolation (`0.0335 ms -> 0.0306 ms` including quant), but the c1-shaped synthetic top-1 is `0.875` and B3 regresses to `47.62 tok/s` (warm `48.44`). The callable T16 fused-SiLU dp4a variant and Q6T16 selected-down dp4a are intentionally not routed. X8 selected-down is tracked in the dedicated row above because q6-only now has llama-compat evidence while q5/both remains rejected. **2026-06-28 GPU-bound re-test (post lib-cache): `HIPENGINE_GGUF_T16_SELECTED_DP4A` clean interleaved A/B (3 runs x 12 cycles, warm) on the full resident-draft B3 bench is flat-negative `48.60 -> 48.42 tok/s` (-0.4%), acceptance identical — dp4a wins at the kernel level (-35% MoE GEMV) and +5% on the verify-isolated harness but does NOT move the full-bench wall, i.e. the full B3 verifier is host-dispatch-bound (~875 launches), not GPU-kernel-bound. dp4a stays default-off; the lever is launch-count reduction (#9 + fusion). Artifact `benchmarks/results/2026-06-28-verifier-dp4a-fullbench-b3-ab.json`.** | Remove these raw/T16 flags unless a later GGML-style q8_1/x4 layout clears the quality gate and improves the same B3/full-suite protocol. Promote only the production-compatible route; keep raw no-decode-repack diagnostics separate from production T16 routing. |
| GGUF Q5 T16 selected-down one-wave diagnostic | `HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS` q5-only override for the T16 selected-down q8_1/dp4a direct kernel. Unset inherits `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS` and therefore the retained 64-thread selected-MoE scheduler; valid diagnostic values are `32`, `64`, and `128`. | Default-off. The llama.cpp-shaped one-wave Q5 check improved the isolated selected-down microbench at rows=16 on gfx1151: prequantized dot **0.03608 -> 0.03305 ms**, quantize+dot **0.04031 -> 0.03685 ms**, KL mean **0.00398**, KL max **0.03093**, top-1 **0.9375**. The Q4 control stayed on the 64-thread path (`t16_dp4a_dot_prequantized` **0.04007 ms**), so the override only affected Q5. The real compat smoke rejected it: `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6` with q5t32 measured **68.14 tok/s / 14.776 ms/output**, worse than same-route pack8/q6 smoke around **69.06 tok/s / 14.501 ms/output**, with identical smoke acceptance. Artifacts `benchmarks/results/2026-07-01-llama-compat-b2-q5-t16-selected-down-dp4a-t64-rerun-micro.json`, `benchmarks/results/2026-07-01-llama-compat-b2-q5-t16-selected-down-dp4a-q5t32-micro.json`, `benchmarks/results/2026-07-01-llama-compat-b2-q4-t16-selected-dual-dp4a-q5t32-control-micro.json`, and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q5t32-smoke.json`. | Remove this q5-specific override after the selected-MoE scheduler/layout is replaced, or sooner if the next verifier body/layout optimization leaves q5t32 rejected on the same async/full-suite protocol. Do not promote it into the active `llama-compat` route. |
| Env flag surface | Benchmark and diagnostic flags still cover rejected or superseded experiments. | The 2026-07-10 release audit removed env requirements from public backend/quant/GGUF fast-path selection. Accuracy-traded MTP, unsafe kernels, profiler synchronization, and rejected paths remain explicit diagnostics. | Move rejected runtime flags into benchmark-only configuration as each associated MTP/PARO investigation closes; retain correctness fallbacks for one release window after a default promotion. |
| GGUF MTP resident draft | `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE` opt-out around the device-resident selected-MoE down + combine in `mtp_resident_draft.py` (`apply_moe_down_combine`: `silu_mul_separate_out_bf16` + `gguf_q5_k_selected_gemv_bf16_bf16_out` + `weighted_sum_shared_gate_combine_residual_out_bf16_f32w`). The `=0` legacy path keeps the host-readback per-expert Python down loop for bisection. | Default-on after 2026-06-28 B3/c5 A/B: exact-acceptance (drafts byte-identical, accepted_per_output identical in all 4 categories code/general_en/general_ja/mixed_ja_en) and tok/s consistently up per-category (~+0.7-1.4%); removes 2 blocking D->H and ~24 launches/depth from the MoE-down section. Matches verifier bf16 precision. Artifact `benchmarks/results/2026-06-28-resident-mtp-draft-device-moe-down-ab.json`. | Drop the opt-out and delete the legacy host-loop branch (and the then-unused `gate_f32`/`up_f32`/`inter_f32`/`down_out`/`scaled`/`gated_shared` buffers) after sub-win B (device argmax + embedding gather) lands and a same-protocol full-suite MTP row is retained. **Sub-win B landed 2026-06-29 (device-chain row below); the legacy-host-loop deletion + buffer cleanup is now unblocked once a retained full-suite MTP row exists.** |
| GGUF MTP resident draft device-chain | `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN` opt-in (default off) for the device-chained draft in `_propose_chain_device`; explicit bench flag `--resident-mtp-device-chain` prewarms the path for llama.cpp replication routes. Each depth's top-1 is device-gathered from a resident FP32 embedding table (`gather_f32_rows_by_i32id`, exact copy of `token_embd_f32[:vocab]`, 268MB cached upload), top-k is accumulated on device, one drain + readback happens at chain end, and rope/pos/ctx are precomputed once. | Default-OFF. BIT-EXACT vs the legacy host loop (0/5 top-1 + topk-row divergence unit gate; e2e B3 drafts + total_accepted 37/1008 identical). Original B3 evidence was flat (`39.81 -> 39.79 tok/s`) because the draft is GPU-compute-bound, not host/sync-bound. Llama-replication evidence on 2026-06-30: prewarm removes the short-run 268MB upload artifact (`draft_device_chain_ensure_embed_table` **11.888 -> 0.000 ms/output**) but full-suite compat dp4a only moves **52.48 -> 52.79 tok/s**. Split timing shows `draft_topk_readback` is almost all GPU drain (`draft_device_chain_drain` **3.830 ms/output**) and not D2H (`draft_topk_d2h` **0.008 ms/output**). 2026-07-01 sync-stage attribution moves that drain into section buckets: `draft_run_lm_head` **1.882 ms/output**, `draft_run_attention` **0.718**, `draft_run_ffn_up_shared` **0.557**, `draft_device_topk_gather` **0.357**. Artifacts `benchmarks/results/2026-06-29-resident-mtp-draft-device-chain.json`, `benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full.json`, `benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full-split.json`, and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-draftsync-full.json`. | Keep as an explicit llama.cpp replication diagnostic while the compat draft/verifier lifecycle is being matched. Do not promote just to remove host copies; promote/collapse only if a future resident draft fusion or verifier-side change cuts the **GPU drain** and improves the same full-suite protocol. Otherwise delete the flag/route during the post-MTP flag cleanup; the `_run_one` device-pointer refactor + gather kernel stay regardless. |
| GGUF MTP resident draft Q6 top-1/gather | `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER` opt-out around the exact Q6_K lm-head top-1 specialization in `mtp_resident_draft.py` for resident device-chain `top_k == 1`. The new `pack8_gemv_decode_bf16_top1_gather_f32` kernel writes the selected id/value and optionally gathers the next FP32 embedding row, replacing full-logits materialization + separate top-k + gather in the llama-compat draft path. | Default-on after 2026-07-01 same-tree full-suite A/B on `llama-compat-device-chain-dp4a`: **52.60 -> 53.34 tok/s** (+1.4%), `cycle_wall_ms_per_output` **19.033 -> 18.772**, `draft_initial` **4.033 -> 3.712**, acceptance unchanged (`acc/output 0.561`, draft acceptance `0.640`). Unit gate proves identical selected id/value/embedding row vs the old logits -> top-k -> gather chain. Sync-stage attribution confirms `draft_device_topk_gather` **0.357 -> 0.001 ms/output** while verifier remains ~14.66 ms/output. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-q6top1-full.json`, disabled control `...q6top1-control-full.json`, and sync-stage `...q6top1-draftsync-full.json`. | Keep the opt-out for short-term A/B while the llama-compat replication lane is still active. Make the top-1/gather path unconditional for resident device-chain `top_k == 1` after the next compat verifier-layer optimization validates against the same full-suite route, then remove the env flag and old top-k/gather branch for that case. |
| GGUF MTP resident draft Q6 top-1 q8_1/dp4a | `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A` / `--resident-mtp-draft-q6-top1-dp4a` opt-in around the llama-compat resident draft Q6_K lm-head top-1/gather path. It q8_1-quantizes the BF16 head input and calls `pack8_gemv_decode_q8_1_dp4a_top1_gather_f32`, matching llama.cpp's quantized matvec economy more closely than the exact raw-Q6 path. The diagnostic `HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS` / `--resident-mtp-draft-q6-top1-stage1-threads {64,128}` exists only to A/B the stage1 scheduler; suite routes `...x8q6-t64` and `...x8q6-t64-allsync` force 64 threads. | Default-off, accuracy-traded. 2026-07-01 full-suite `llama-compat-device-chain-dp4a-q6top1dp4a` B2: **58.83 -> 59.63 tok/s** (+1.36%), `cycle_wall_ms_per_output` **17.019 -> 16.793**, `draft_initial` **3.564 -> 3.293 ms/output**, acceptance unchanged (`acc/output 0.578`, draft acceptance 0.685). All-sync smoke confirmed `draft_run_lm_head` **1.471 -> 1.253 ms/output**; the q6-X8 stage split now attributes the retained 128-thread aggregate to Q6 top-1 stage1 **1.218 ms/output** plus stage2/gather **0.041 ms/output**, while norm+cast+q8_1 quantize is **0.030 ms/output**. The 64-thread scheduler check is rejected on the real route: all-sync stage1 **1.218 vs 1.246 ms/output** and same-session async **69.06 tok/s / 14.501 ms** vs t64 **68.79 tok/s / 14.557 ms**, identical acceptance. Unit gate matches a CPU q8_1/Q6_K oracle; rocprofv3 confirms both 128-thread and 64-thread stage1 kernels ran. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-full.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-top1split128-allsync-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64-top1split-allsync-smoke.json`, and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64-smoke.json`. | Keep the q8_1/dp4a draft head as an explicit llama-compat diagnostic until full-suite heldouts decide whether the accuracy-traded draft head should be folded into the named compat route. Do not promote to the exact/default resident draft path. Remove the t64 thread flag/routes after the next draft lm-head optimization because the scheduler-width copy is rejected; standalone activation-quantization/cast and final reduce/gather tweaks are already ruled out, so future work should target Q6_K stage1 compute/layout or a broader llama-style MMVQ layout. |

## Post-Optimal-Path Cleanup Targets

These are not optimization tasks for the current sprint. They are the cleanup
pass to run once a path is fast and correct enough that the benchmark defaults
should be boring.

| Path | Cleanup target | Keep | Remove / collapse trigger |
| --- | --- | --- | --- |
| 35B MTP chain verifier | Collapse the sprint-era stack of env flags into the default dispatch path and document the current optimal B=1 chain route, while keeping B=2/B=3 available for adaptive-density policy experiments. | Numerical fallbacks, exactness tests, and rollback toggles that are still needed for one release window. | Retained `>1.0x` same-suite row plus one follow-up defaults-only rerun after the adaptive-policy decision. |
| 35B MTP tree/top-k | Keep tree code default-off until it beats chain on the same wall and prompt suite; do not let tree-specific dispatch obscure the chain hot path. | Tree correctness tests and graph replay scaffolding. | If tree remains negative after the verifier wall cut, demote branch/top-k runtime flags to explicit experiment scripts. |
| 27B dense DFlash | Separate deployable online routing from profile-history diagnostics. The current positive production row is the online whole-cycle confidence gate; older prompt-history route/terminal-tail rows are retained evidence, not the default API shape. | Online gate config, oracle/calibration tooling, exact AR comparisons. | After the DFlash hardening rerun and decode API update, trim profile-history routing from the main hot path or move it behind an explicit research harness. |
| DFlash drafter/verifier flags | Audit `HIPENGINE_DFLASH_DRAFTER_DENSE`, `HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM`, and `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD`. | Default-on exact dense WMMA if the fresh 27B gate confirms it; tests for rejected fused kernels. | Fresh 27B DFlash rerun decides: promote exact positive flags to defaults, remove negative runtime branches, or demote them to test-only overrides. |
| Benchmark commands | Stop requiring long flag piles once defaults represent the optimal path. | Flags that select workload shape, model, quant, and explicit experiments. | After MTP/DFlash defaults-only rows are retained, update benchmark docs to show default commands first and move historical A/B flags into dated notes. |

## `--verify-dp4a` / `*-pmin05-dp4a` route (default OFF, opt-in accuracy-traded)
- Added 2026-06-30. Bench flag `--verify-dp4a` (gguf_mtp_bench.py) + suite route
  `resident-b1-probe-block-direct-cap32k-minrows2-pmin05-dp4a` enable llama.cpp-style
  dp4a (q8_1) selected-expert verify GEMVs. **Default off; accuracy-degrading.**
- Purpose: let users who accept llama's precision loss get max accuracy-traded MTP
  perf (~61.6 tok/s / 1.13x B5). FAILS the ja correctness gate (greedy top-1 0.700 <
  0.90). Does NOT match llama HIP MTP (67.3) — dp4a is necessary but not sufficient.
- Remove when: either a Vulkan backend supersedes the perf goal, or the project
  decides to drop dp4a experimentation entirely. Until then it is the documented
  opt-in for the dp4a/accuracy tradeoff. See docs/MTP-LLAMACPP-PARITY.md "COFFIN NAIL".

## `--llama-compat` / `llama-compat*` routes (default OFF, semantic diagnostic)
- Added 2026-06-30. Bench flag `--llama-compat` forces the closest hipEngine
  replica of llama.cpp MTP semantics: B2, `draft_p_min=0`, full draft vocab,
  shifted MTP context replay, device MTP KV, no adaptive B1 probe/fallback, and
  one target block verifier per cycle. Suite routes `llama-compat` and
  `llama-compat-dp4a` are fixed to B2 so the artifact label matches the forced
  child `draft_n_max`. Follow-up replication routes
  `llama-compat-device-chain{,-dp4a}` and
  `llama-compat-device-seed-chain{,-dp4a}` add prewarmed resident device-chain
  drafting and optional resident target `pending_h` starts without changing the
  shipped default path.
- Purpose: isolate whether the remaining llama HIP MTP gap is semantic-policy
  mismatch versus implementation/backend cost. The exact compat route is
  precision-preserving; `llama-compat-dp4a` adds the already-known
  accuracy-traded q8_1/dp4a regime. Full-suite B2 evidence landed the same day:
  exact compat **51.16 tok/s = 0.934x AR**, dp4a compat **52.48 tok/s = 0.958x
  AR**, prewarmed device-chain dp4a **52.79 tok/s = 0.965x AR**, and
  device-seed-chain dp4a **52.53 tok/s = 0.960x AR**. All remove the serial B1
  probe and keep acc/output ~0.56, but lose to AR because the compat
  draft/context + block verifier lifecycle is too costly. Split instrumentation
  shows device-chain `draft_topk_readback` is almost all GPU drain
  (`draft_device_chain_drain` **3.830 ms/output**) and not D2H
  (`draft_topk_d2h` **0.008 ms/output**). Follow-up Q6 top-1/gather plus
  direct-state verifier cleanup lifts the best compat dp4a B2 diagnostic row to
  **55.41 tok/s = 1.014x AR**, with unchanged acceptance. The new
  `llama-compat-device-chain-dp4a-allsync` route adds
  `--resident-mtp-draft-sync-stage-timings` and
  `--target-block-sync-stage-timings` for attribution only; its buckets show the
  remaining verifier cost is target linear-attention/MoE operation time, not
  snapshot/commit bookkeeping. The current active replication route is
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit`:
  **60.56 tok/s**, **16.534 ms/output**, **1.1055x AR**, acc/output **0.609**,
  draft acceptance **0.780**, target rows/output **1.172**, verifier drain
  **14.071 ms/output**, replay/commit **0.043 ms/output**, and zero replay rows.
  The semantic-safe serial-state row remains the exact control at **51.85 tok/s**
  / **19.308 ms/output**.
- Remove / promote when: after full-suite stage-bucket evidence decides the
  question. Current evidence says these are replication diagnostics, not default
  promotion candidates; keep only the smallest route set needed for future parity
  audits, or delete the routes during the next MTP flag cleanup unless another
  llama.cpp semantic delta is identified.

## `--resident-mtp-draft-sync-stage-timings` (default OFF, attribution-only)
- Added 2026-07-01. Bench flag inserts `hipDeviceSynchronize()` boundaries inside
  the resident MTP draft `_run_one()` path when `--record-cycle-stage-timings` is
  enabled. Suite route `llama-compat-device-chain-dp4a-draftsync` wires it into the
  llama.cpp replication lane.
- Purpose: split the previous `draft_device_chain_drain` bucket into
  `draft_run_project`, `draft_run_qkv_kvwrite`, `draft_run_attention`,
  `draft_run_ffn_up_shared`, `draft_run_moe_down_combine`, `draft_run_lm_head`, and
  `draft_device_topk_gather`. The flag changes timing by adding synchronization and
  is not a performance path.
- Remove when: the resident draft LM-head/top-k or verifier layer-time follow-up has
  its own lower-overhead profiler/rocprof attribution, or after the llama.cpp
  replication lane is closed. Until then keep it only as a named diagnostic route.

## `--target-block-sync-stage-timings` (default OFF, attribution-only)
- Added 2026-07-01. Bench flag inserts `hipDeviceSynchronize()` boundaries inside
  the target block verifier when `--record-cycle-stage-timings` is enabled. Suite
  route `llama-compat-device-chain-dp4a-allsync` combines it with resident draft
  sync timings for one-pass draft+verifier attribution.
- Purpose: split `target_block_linear_attn_layers` and
  `target_block_full_attn_layers` into operation buckets (`norm_qkv_gate`,
  `chain_gdn`, selected-MoE expert gate/up/down, shared expert, combine, and
  full-attn KV/attention/output sections). The flag changes timing by adding
  synchronization and is not a performance path.
- Remove when: a lower-overhead verifier profiler/rocprof harness can produce the
  same operation split, or after the llama.cpp replication lane is closed.

## `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL` / `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM` / `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS` / `HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA` / `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT` / `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE` / `HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN` / `HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE` / `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN` / `HIPENGINE_GGUF_VERIFY_F32_POST_NORM` (default OFF, semantic diagnostic)
- Added 2026-07-02. Env flag keeps target-block verifier residual outputs in
  FP32 for an opt-in llama.cpp parity probe while preserving BF16 mirrors for
  existing projection kernels. It adds FP32 add/RMSNorm and MoE combine helpers.
  The follow-up diagnostic also feeds layer-entry attention RMSNorm from FP32
  residual rows when available, but intentionally does not claim full llama.cpp
	  graph parity. `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM=1` materializes
	  layer-entry attention RMSNorm into FP32 scratch, casts a BF16 mirror for
	  unsupported consumers, and routes dense-Q8 dp4a QKV / QKV+gate consumers from
	  the FP32 tensor when the F32 dense-Q8 diagnostic is already active.
	  `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS=1` additionally routes
	  compatible row-bulk linear-attention Q8 `attn_qkv`/`attn_gate` projections
	  into FP32 scratch through the raw-Q8 dp4a F32-output dual wrapper, casts BF16
	  mirrors for existing downstream kernels, and emits explicit BF16 mirror
	  capture keys. It also routes dense-F32 `ssm_alpha`/`ssm_beta` through the
	  registry-dispatched F32-input/F32-output dense GEMV route when available,
	  while preserving BF16 mirrors for existing downstream consumers.
	  `HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA=1` additionally routes row-bulk
	  linear-attention `ssm_alpha`/`ssm_beta` from that FP32 attention-norm tensor to
  mirror llama.cpp's `build_layer_attn_linear` source shape.
  `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT=1` keeps the row-bulk linear-attention
  `ssm_out` projection output in FP32 through the post-attention residual/RMSNorm
  add while preserving the BF16 mirror for existing captures and downstream
  kernels. It also keeps row-bulk full-attention `attn_output` in FP32 through
  the same residual/RMSNorm helper when a raw Q8 sidecar BF16-input/F32-output
  path is available.
  `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE=1` keeps the selected-expert weighted
  sum in FP32 inside the F32-residual MoE combine instead of BF16-rounding that
  selected sum before adding residual and sigmoid-gated shared output.
  `HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN=1` requires the F32 MoE combine
  diagnostic and routes compatible X8 Q5/Q6 selected-down GEMV outputs into an
  FP32 scratch buffer before combining selected rows with BF16 shared output.
  `HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE=1` requires the F32 MoE
  combine + selected-down stack, computes selected `silu(gate) * up` into FP32
  scratch, preserves the BF16 mirror, and feeds the FP32 activation into
  selected-down.
  `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN=1` requires the F32 MoE combine and
  selected-down stack, routes shared-expert down output into FP32 scratch,
  preserves the BF16 mirror, and combines FP32 selected rows with FP32 shared
  rows.
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM=1` extends the probe by materializing
  post-attention RMSNorm into FP32 scratch and independently gating router /
  selected-q8 / shared-q8 consumers with
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_{ROUTER,SELECTED_Q8,SHARED_Q8}`.
- Purpose: test the current semantic hypothesis that accumulated BF16 verifier
  layer-boundary drift is enough to flip near-tie target decisions versus
  llama.cpp's F32 target `l_out` graph tensors. The diagnostic artifact
  `benchmarks/results/2026-07-02-mtp-target-f32-residual-diagnostic.json`
  confirms the lever is active: the old cycle-12 trace cannot replay unchanged
  because cycle 2 flips from exact `[40798, 25, 1103]` / accepted 2 to
  FP32-residual `[40798, 1590, 1103]` / accepted 1. The follow-up artifact
  `benchmarks/results/2026-07-02-mtp-target-f32-residual-attnnorm-diagnostic.json`
  reaches the old cycle-12 branch but still accepts `539`, with the wrong
  `539 - 26126` margin increasing from **+0.11822** to **+0.14309** versus
  llama.cpp **-0.00896**. The attention-norm-output dense-Q8 split
  (`benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-output-denseq8-diagnostic.json`,
  control
  `benchmarks/results/2026-07-02-mtp-target-f32-residual-bulk-control-diagnostic.json`)
  moves the bulk pair-12 `539 - 26126` margin from **+0.31369** to
  **+0.18198**, still opposite llama.cpp. The linear-attention output-to-residual
  split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-denseq8-diagnostic.json`
  moves that margin only **+0.18198 -> +0.17663**, so the `ssm_out` BF16 round is
  not the main missing semantic lever. The alpha/beta F32 input split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-denseq8-diagnostic.json`
  is byte-identical to the attention-output slice and leaves the row-1 margin at
  **+0.17663**, ruling out `ssm_alpha`/`ssm_beta` projection input precision for
  the active branch. The full-attention output-to-residual split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-fullattnout-denseq8-diagnostic.json`
  still samples `[15495, 539, 1151]`, accepts 2, and worsens row-1
  `539 - 26126` to **+0.27480**, so full-attention `attn_output` BF16 output
  rounding is ruled out as well. The MoE selected-sum accumulator split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-denseq8-diagnostic.json`
  still samples `[15495, 539, 1151]` and accepts 2, but narrows row-1
  `539 - 26126` from **+0.27480** to **+0.03385**. That makes the combine
  selected-sum BF16 boundary semantically active, but still not sufficient to
  match llama.cpp's **-0.00896** margin. The selected-down output split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-denseq8-diagnostic.json`
  keeps compatible X8 selected-down rows in FP32 and narrows the same margin to
  **+0.00536** (`26.06115 - 26.05580`), still on the wrong side of llama.cpp by
  about **0.0143 logits**. The selected-intermediate split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-selectedintermediate-denseq8-diagnostic.json`
  is the first pair-12 side-matching slice: sampled tokens become
  `[15495, 26126, 1151]`, accepted drafts fall to 1, and row-1 `539 - 26126`
  moves to **-0.00303** (`26.04795 - 26.05098`), close to llama.cpp's
  **-0.00896**. This confirms the selected SwigLU/intermediate BF16 boundary
  is a parity contract to fold into a cohesive llama-compat verifier mode if
  longer-trace/full-suite acceptance validates it. The shared-down output split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-shareddown-denseq8-diagnostic.json`
  still samples `[15495, 539, 1151]` and accepts 2, and widens the same margin
  to **+0.03043** (`26.12703 - 26.09660`), ruling out isolated shared-down
  output precision as the missing parity fix. Combining MoE combine with
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM=1` failed prior-cycle replay at cycle 2
  (`[40798, 1590, 1103]` vs trace `[40798, 25, 1103]`), so the combination is
  not a pair-12 parity result. The post-norm split artifact
  `benchmarks/results/2026-07-02-mtp-target-f32-postnorm-split-diagnostic.json`
  shows the combined router+selected-q8 consumer path breaks the old trace at
  cycle 7; selected-q8 alone flips row 1 (`413 - 4071` **+0.13053 -> -0.14458**),
  while router-only reaches pair 12 but worsens `539 - 26126` to **+0.33520**.
  This makes these flags instrumentation only, not a candidate promotion.
- Remove when: either a fuller F32 verifier graph path supersedes this partial
  residual-boundary slice, or parity work decides llama.cpp's F32 graph
  semantics are not the target. Do not promote this flag as a speed path.

## `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS` (diagnostic rollback)
- Added 2026-07-01. Host-side launch switch for the selected T16 q8_1/dp4a
  verifier kernels used by `--verify-dp4a` / `llama-compat-device-chain-dp4a`.
  Default is now `64`; setting the env var to `128` restores the old launch
  shape.
- Purpose: keep a rollback/A-B hook for the first selected-MoE scheduler change
  that survived async/full-suite validation. Full-suite `llama-compat` B2 moved
  **55.45 -> 58.83 tok/s** and `target_block_verify_total`
  **14.025 -> 13.134 ms/output** on gfx1151.
- Remove when: either the selected-MoE scheduler is replaced by a llama-style
  `mul_mat_vec_q_moe` port or two later full-suite compat runs confirm 64 is
  stable enough that the 128-thread rollback path is no longer useful.

## `HIPENGINE_GGUF_Q8_T16_THREADS` (diagnostic rejected)
- Added 2026-07-01. Host-side launch switch for Q8_0 T16 single/pair/triple
  GEMV wrappers. Default/unset keeps the existing 128-thread launch; setting
  the env var to `64` exercises a smaller workgroup for verifier projections.
- Purpose: test whether the llama-compat verifier hot leaf
  `attn_qkv+attn_gate` is losing time because Q8T16 pair projection uses the
  wrong launch width. The focused qwen35 pair microbench rejected 64 threads:
  rows 2/3/4 measured **197.77/224.80/251.96 us** at 64 threads versus
  **179.26/207.05/237.02 us** at 128 threads. `rocprofv3` confirmed the 64-thread
  override launched with `Workgroup_Size_X=64`.
- Remove when: the Q8T16 verifier pair work moves to a different llama-style
  kernel body/schedule, or when the parity sprint no longer needs this A/B hook.
  It is not a performance path and should not be promoted.

## `gguf_q8_0_t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out` (diagnostic rejected)
- Added 2026-07-01. Callable T16 Q8_0 dual-split pair kernel that consumes
  GGML q8_1 activation blocks and uses `sudot4`, intended to test whether the
  llama.cpp Q8_0×Q8_1 arithmetic recipe transfers to the existing Q8T16
  `attn_qkv+attn_gate` verifier pair layout.
- Purpose: isolate the kernel-body question after the 64-thread launch-width
  check failed. Correctness passed against a q8_1 CPU oracle plus KL/top-1 gate,
  and `rocprofv3` confirmed `q8_0_t16_dual_split_q8_1_dp4a_kernel<unsigned short>`
  launched with `Workgroup_Size_X=128`. Performance rejected the route: for the
  qwen35 pair shape, rows 2/3/4 exact 128-thread pair is
  **181.50/207.98/236.26 us**, while quantize+dp4a is
  **304.78/448.32/558.14 us** and prequantized dp4a is
  **303.05/452.51/566.29 us**.
- Remove when: the parity sprint moves Q8 verifier work to a true llama-style
  mmvq/T16 replacement layout or row-amortized verifier kernel. This callable is
  evidence, not a performance path; do not route it into `llama-compat` runtime.

## `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE` / `HIPENGINE_GGUF_Q8_T16_PAIR_COL8` (gfx1151 C8 rollback)
- The 2026-07-01 rowtile diagnostic was repaired to 128 threads so it preserves
  the production reduction partition, then scoped by backend metadata to
  gfx1151 physical C8. C2/C4 and gfx1100 retain per-row Q8T16.
- F3I keeps rowtile4's two row groups but computes eight output columns per
  block. This halves live accumulators and moves static resources from 136 to
  72 VGPR and 1,024 to 512 B LDS, with zero scratch. Exact direct C8 improves
  **151.015 -> 152.164 tok/s (+0.76%)** in implementation evidence; matched
  blocking/SSE/delayed server rows all improve **+0.38%/+0.54%/+0.46%**.
- `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE=0` rolls the physical-C8 owner back to
  the per-row kernel. `HIPENGINE_GGUF_Q8_T16_PAIR_COL8=0` keeps row
  amortization but restores the prior 16-column body. Explicit `PAIR_ROWTILE=1`
  outside the backend-certified session remains a diagnostic and does not
  automatically select col8.
- Remove the two rollbacks after one release window plus defaults-only gfx1151
  direct/server refreshes and an independent gfx1100 transfer. Keep the old
  per-row and 16-column rowtile bodies as unsupported-width fallbacks.

## `HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK` (gfx1151 C8 rollback)
- The physical-C8 Q6T16 lm-head reads the full ~417 MiB head twice. The prior
  **6+2** partition uses 200/88 VGPR; gfx1151 now defaults to exact **5+3** at
  168/104 VGPR. Isolated wall improves **4.865 -> 4.815 ms (-1.02%)** and
  clean direct C8 improves **152.192 -> 152.709 tok/s (+0.340%)**.
- `HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK=6` restores 6+2. gfx1100 package
  metadata remains 6 pending independent W7900 transfer. Values outside [2,6]
  are invalid.
- Remove after one release window plus defaults-only gfx1151 direct/server
  refresh and gfx1100 transfer. Keep `_small_b_rowtile_chunks` and all 2-6 row
  kernels because they remain exact fallbacks for arbitrary packed widths.

## `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL` (diagnostic rejected)
- Added 2026-07-01 as a default-off runtime hook for broad exact Q8T16 verifier
  row-amortization. Setting `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL=1` routes qwen35
  `rows>1, in=2048` singleton, pair, and triple Q8T16 projections through
  rowtile4 wrappers where available. It also enables the pair rowtile diagnostic
  unless `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE=0` is set explicitly. Suite route:
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q8rowtileall`.
- The original verifier decision remains rejected. Correctness passes against
  the exact singleton/pair/triple wrappers and its B2 profile moved dense Q8
  **11.420 -> 10.811 ms/block**, but async llama-compat moved **68.78 -> 68.54
  tok/s** with identical acceptance.
- F3 re-evaluated the original 64-thread bodies for native packed AR c2/c4/c8.
  A clean p512/d64 screen looked positive, but p512/d128 changed one prompt's
  trajectory at every packed width. The later model-hidden oracle localized the
  cause to 64-thread reduction order, not cross-row weight reuse. Moving all
  rowtiles to the production 128-thread partition makes the first-transition
  model-hidden gate exact, but broad all-projection model wall remains neutral
  or negative: C2/C4/C8 **77.940/107.798/133.377 tok/s** versus retained
  **78.552/108.050/133.251**. Therefore all-projection promotion stays rejected;
  only the separately scoped physical-C8 pair above is eligible.
- `GGUF_Q8_T16_DECODE_ROWTILE_ALL` is false on both gfx11 backend packages and
  `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL=1` remains diagnostic-only at 128 threads.
  Remove this broad hook when the investigation is archived. Do not promote it
  for AR or MTP without a new full-horizon correctness and performance gate.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=row` / row Q6 top-1 routes (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape row` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-row` /
  `...-row-allsync` exercise a llama.cpp-shaped Q6_K draft lm-head top-1
  stage: one output row per block, two wave32 warps, and a signed
  `__vsubss4`/dot4 Q6_K MMVQ body. Default stays `pack8`.
- Purpose: test whether the remaining draft-side Q6_K top-1 gap is caused by
  hipEngine's pack8 output-row geometry rather than the vector-dot body itself.
  Correctness passes against the q8_1/Q6_K oracle, but performance rejects the
  route: all-sync row stage1 is only slightly faster than pack8
  (**1.202 vs 1.218 ms/output**) while row stage2/gather grows
  **0.041 -> 0.252 ms/output** because it reduces over `vocab` instead of
  `vocab/8`; async smoke regresses **69.06 tok/s / 14.501 ms** to
  **66.95 tok/s / 14.958 ms** with identical acceptance.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused row-stage/top-1 reduce makes this row-shape diagnostic
  obsolete. It is evidence, not a performance route.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=pack8_scalehoist` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape pack8_scalehoist` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-scalehoist` /
  `...-scalehoist-allsync` exercise a Q6_K draft lm-head top-1 stage that keeps
  the retained pack8 `vocab/8` final reduce but hoists each Q6_K block's
  `d*scale[16]` values into shared memory.
- Purpose: test whether the remaining q8_1/dp4a Q6_K draft stage1 cost is from
  repeated Q6 scale loads rather than the dot body or output geometry. Correctness
  passes against the q8_1/Q6_K oracle, and `rocprofv3` confirms
  `gguf_q6_k_pack8_gemv_q8_1_dp4a_top1_scalehoist_stage1_kernel` launches.
  Same-session smoke rejected it: retained `x8q6` rerun **68.65 tok/s**,
  cycle **14.589 ms/output**, `draft_initial` **2.482 ms/output** vs
  scalehoist **68.54 tok/s**, cycle **14.610 ms/output**, `draft_initial`
  **2.485 ms/output**, with identical acceptance.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused top-1/sampler path supersedes this evidence hook. It is
  not a performance route and should not update the active llama-compat headline.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=pack8_llama` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape pack8_llama` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack8llama` /
  `...-pack8llama-allsync` exercise a Q6_K draft lm-head top-1 stage that keeps
  the retained pack8 `vocab/8` final reduce but uses the llama.cpp Q6_K MMVQ
  vecdot decomposition inside stage1.
- Purpose: test whether the remaining q8_1/dp4a Q6_K draft stage1 cost is the
  pack8 dot body rather than final-reduce geometry. Correctness passes against
  the q8_1/Q6_K oracle for fused and split stage1+stage2 paths. Same-session
  all-sync moved the intended leaf **1.220 -> 1.205 ms/output**, but async B2
  smoke rejected the route: retained control **68.88 tok/s**, cycle
  **14.541 ms/output**, `draft_initial` **2.487 ms/output** vs pack8_llama
  **67.92 tok/s**, cycle **14.747 ms/output**, `draft_initial`
  **2.493 ms/output**, with identical acceptance.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused top-1/sampler path supersedes this evidence hook. It is
  not a performance route and should not update the active llama-compat headline.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=pack16` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape pack16` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack16`,
  `...-denseq8all-pack16`, and `...-denseq8all-pack16-allsync` exercise a
  Q6_K draft lm-head top-1 stage that keeps the retained pack reduction but
  doubles the output group from 8 to 16 vocab rows per block.
- Purpose: test whether the current draft Q6_K stage1 cost is dominated by q8_1
  activation reloads and final-reduce entries rather than register pressure in
  the per-output Q6 body. Correctness passes against the q8_1/Q6_K oracle for
  fused and split stage1+stage2 paths. Same-session denseq8all smoke rejected it:
  retained control **71.74 tok/s**, cycle **13.961 ms/output**,
  `draft_initial` **2.479 ms/output** vs pack16 **71.72 tok/s**, cycle
  **13.963 ms/output**, `draft_initial` **2.487 ms/output**, with identical
  acceptance. Draft rocprof confirms the kernel-family loss:
  `gguf_q6_k_pack16_gemv_q8_1_dp4a_top1_stage1` is **3.684 ms/cycle** vs the
  retained pack8 stage1 **3.603 ms/cycle**.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused top-1/sampler path supersedes this evidence hook. It is
  not a performance route and should not update the active llama-compat headline.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=x8_dscale` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape x8_dscale` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8dscale-f32ssm`
  / `...-x8dscale-f32ssm-allsync` exercise the retained X8-packed Q6_K draft
  lm-head top-1 layout with an extra X8-aligned FP32 `d*scale` sidecar.
- Purpose: test whether the remaining retained X8 Q6_K top-1 cost is dominated
  by repeatedly unpacking/multiplying Q6 block scales inside the dot body.
  Correctness passes against the q8_1/Q6_K oracle for fused and split
  stage1+stage2 paths, but draft-chain rocprof rejects the route:
  `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8dscale.json`
  reports host wall **6.805 -> 8.023 ms/cycle**, kernel time
  **6.427 -> 7.615 ms/cycle**, and `draft_lm_head_q6_top1`
  **3.648 -> 4.859 ms/cycle** versus the retained X8 artifact. Extra FP32
  sidecar memory traffic/register pressure is worse than recomputing scales.
- Remove when: a later draft Q6_K top-1 body/layout or fused top-1/sampler route
  supersedes the current X8 evidence set. This route is evidence only; do not
  promote or rerun full-suite unless a separate change materially alters the
  dscale memory path.

## `--fused-b1-block-probe` / `resident-fused-b1-block-direct-cap32k-minrows2-pmin05`
- Added 2026-06-30. Bench flag `--fused-b1-block-probe` keeps the retained
  adaptive B1-probe policy, but lets B1 probe cycles verify `[prev, draft0]` with
  one strict two-row target block instead of the serial target step loop. The suite
  route mirrors `resident-b1-probe-block-direct-cap32k-minrows2-pmin05` plus this
  flag. **Default off** until a full-suite row proves it improves wall time.
- Purpose: test the first queued llama.cpp-parity fix from
  `docs/MTP-LLAMACPP-PARITY.md`: remove or shrink `target_serial_verify_step`
  without merely shifting the same cost into `target_block_verify_total`.
- Remove / promote when: promote into the retained route only if exact full-suite
  B5 beats the current default and stage buckets show serial verifier cost falls
  below ~2 ms/output with no acceptance regression. Otherwise delete the flag/route
  after the parity A/B is recorded.

## `--target-block-direct-partial-replay-mode`
- Added 2026-07-02. Bench and forced-target-probe flag with choices
  `serial-exact` (default), `serial-state-only`, `direct-commit`,
  `bulk-state-only`, and `native-state-only`. It only affects direct-state block verification when a
  bulk verifier block is rejected or partially accepted. The retained
  `serial-state-only` mode restores the snapshot, advances the accepted prefix
  through `verify_target_block_serial_exact(..., advance_state_only=True)`, and
  skips replay LM-head sampling; target tokens still come from the original
  full-block scoring pass. The active llama-replication `direct-commit` mode
  commits the captured verifier row on rejected/partial blocks, matching
  llama.cpp's normal MTP accept lifecycle rather than serial-prefix replay. The
  rejected bulk/native modes replay the accepted
  prefix with `verify_target_block(..., advance_state_only=True)`, with
  `native-state-only` using the native row-serial-attention verifier only for
  that state replay.
- Purpose: reduce the semantic-safe `llama-compat` replay/commit bucket while
  preserving state lifecycle, and provide a separate llama-style replication
  lane. `--llama-compat` promotes unspecified `serial-exact` replay to
  `direct-commit`; explicit serial-state, bulk, and native diagnostic modes remain
  opt-in.
- Result: `direct-commit` is retained for the llama-compat replication lane. The
  full-suite row
  `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-full.json`
  moves **51.85 -> 60.56 tok/s**, cycle **19.308 -> 16.534 ms/output**,
  verifier drain **16.891 -> 14.071 ms/output**, replay/commit
  **2.489 -> 0.043 ms/output**, and replay rows **38 -> 0** versus the
  serial-state control. The lifecycle diagnostic
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-directcommit-partial-compare.json`
  intentionally diverges from serial replay at cycle 3 with matching visible
  token `[65342]`; that is expected for llama-replication, not an exact-state
  claim.
- Exact-control result: `serial-state-only` is retained as the semantic-safe
  control.
  The lifecycle comparator artifact
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-serial-state-only-partial-replay-compare.json`
  reports `first_mismatch: null`, and the full-suite row
  `benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json`
  moves **50.96 -> 51.85 tok/s**, cycle **19.645 -> 19.308 ms/output**,
  verifier drain **17.222 -> 16.891 ms/output**, and replay/commit
  **2.775 -> 2.489 ms/output** with unchanged acceptance/economy.
- Rejected diagnostics: artifact
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-bulk-state-only-partial-replay-compare.json`
  reports `first_mismatch` at cycle 3. The visible token still matches
  `[65342]`, but `bulk_state_only_replay` diverges from
  `serial_exact_accepted_prefix` in hidden seed plus Conv/GDN state across 61
  fingerprints. The active-shape native replay artifact
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-native-state-only-partial-replay-active-compare.json`
  also fails at cycle 3 with matching visible token `[65342]` but 59
  hidden/linear-state mismatches.
- Remove when: parity closure picks the final compat transaction policy. If
  `direct-commit` remains the llama-replication path, collapse the route/flag
  surface so only the named compat mode and the exact serial-state control remain.
  If exact-state semantics become the compat target, delete directcommit as a
  perf diagnostic. Do not promote bulk/native state-only replay into any retained
  route.

## `--verify-lm-head-q6-top1-dp4a` / verifier lm-head X8 sidecar
- Added 2026-07-01. Bench flag `--verify-lm-head-q6-top1-dp4a` sets
  `HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR=1` before materialization and
  `HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A=1` at runtime. Suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-vlmheadtop1`
  and `...-allsync` exercise it on top of the active llama-compat lane.
- Purpose: test whether the verifier-side `target_block_lm_head_sample`
  bucket can copy the draft-side q8_1/dp4a Q6_K top-1 economy by skipping full
  verifier logits plus argmax. This is accuracy-traded and default-off; exact
  verifier lm-head sampling remains the shipped behavior.
- Remove / promote when: promote only inside the llama-compat replication lane
  if a full-suite B2 row moves total wall and `target_block_lm_head_sample`
  toward the llama.cpp verifier target without unacceptable row-economy loss.
  Delete the route if smoke/full-suite shows the extra X8 sidecar/top-1 path does
  not move `target_block_verify_total` or if a later fused verifier sampler
  supersedes it.

## `HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL`
- Added 2026-07-02. Bench flag `--resident-mtp-draft-router-row-parallel`,
  draft-profile flag `--router-row-parallel`, and suite route
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow`
  route the resident MTP draft F32 router projection through the row-parallel
  `qwen35_router_logits_f32_f32w` kernel instead of the generic one-block
  `hipengine_mtp_linear_f32` path.
- Purpose: retained llama.cpp-replication optimization. Full-suite B2 moved
  **63.63 -> 64.41 tok/s** and cycle **15.735 -> 15.547 ms/output** with
  unchanged acceptance/economy. Draft-chain sync attribution moved
  `draft_run_ffn_router_linear` **0.508 -> 0.048 ms/cycle**.
- Remove / promote when: make this the unconditional resident draft router
  projection once it is either promoted beyond the llama-compat lane or no
  longer needs A/B isolation. Delete the env/CLI flag and old generic-router
  fallback route after the next parity checkpoint no longer needs the direct
  control.

## `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A` (diagnostic rejected)
- Added 2026-07-02. Bench flag `--resident-mtp-draft-dense-q8-dp4a`,
  draft-profile flag `--dense-q8-dp4a`, and suite route
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8`
  route resident draft dense Q8_0 F32 projections through F32->q8_1 plus
  raw-Q8 dp4a float-output wrappers.
- Purpose: test whether copying the verifier/llama.cpp q8_1/raw-Q8 dp4a economy
  into the draft dense projections (`eh_proj`, Q/K/V, attention output, shared
  gate/up, shared down) closes the non-Q6 draft drain gap.
- Result: draft-chain rocprof moved the intended kernel bucket, but full-suite
  B2 rejected the route: active router-row **64.41 tok/s / 15.547 ms/output**
  vs draftdenseq8 **64.14 tok/s / 15.612 ms/output**, with worse acceptance and
  target rows/output.
- Remove when: the next parity checkpoint no longer needs this negative
  evidence, or if a future fused dense-draft design replaces the standalone
  quantize+dp4a calls. Do not promote the flag as-is.

## `HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED` (diagnostic rejected)
- Added 2026-07-02. Bench flag
  `--resident-mtp-draft-selected-silu-down-fused`, draft-profile flag
  `--selected-silu-down-fused`, and suite routes ending in `-siludown` route
  selected MoE `silu(gate)*up` directly into a Q5_K selected-down GEMV.
- Purpose: test a llama.cpp-shaped fused GLU/down idea without changing row
  economy or draft precision. The fused kernel is bit-exact versus the existing
  BF16 chain (`silu_mul_separate_out_bf16` + `gguf_q5_k_selected_gemv_bf16_bf16_out`).
- Result: rejected by draft-chain profile before full-suite. It removes one
  launch, but the fused Q5 body is slower: active router-row control
  **5.973 ms/cycle kernel / 7.044 ms/cycle host** vs fused
  **6.054 ms/cycle kernel / 7.206 ms/cycle host**; selected-down family
  **0.325 -> 0.391 ms/cycle**.
- Remove when: the next parity checkpoint no longer needs this negative evidence,
  or if a different fused Q5 selected-down body replaces it and wins the draft
  parent profile. Do not promote the flag as-is.

## `HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING`
- Added 2026-07-03. Default-off verifier diagnostic that seeds the target
  verifier F32 residual buffer from host-dequantized `token_embd.weight` rows
  when `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1` is enabled. The BF16 token
  embedding launch still runs and still populates the BF16 mirror path.
- Purpose: isolate llama.cpp MTP parity for layer-0 target input construction.
  The first task-9/cycle-3/row-2 run closed `hidden_in` and
  `attn_norm_f32_scratch` to exact llama.cpp parity, but it did not flip the
  bonus token; the remaining split moved to F32-input projection/dequant and
  later residual/LM-head amplification.
- Remove / promote when: remove once the projection/dequant split is resolved
  and this host-side diagnostic is no longer needed. Promote only by replacing
  it with a real device-side F32 embedding path if full-suite llama-compat
  evidence shows it is required and non-regressive; do not keep host dequant/H2D
  in a timing route.

## `--record-draft-stage-stats`
- Added 2026-07-02. Bench flag that records compact FP32 summaries for resident
  MTP draft sub-stage tensors and dense MTP K/V cache rows in
  `draft_hidden_state_trace`. Default-off extensions
  `--record-draft-cache-rows` and `--record-draft-attention-debug` add selected
  history rows plus host-recomputed dense-attention score/weight diagnostics.
  It forces host-chain resident drafting when enabled so intermediate buffers
  can be read back, and is not a timing route.
- Purpose: diagnose the remaining llama.cpp parity miss after hidden-state
  tracing narrowed the first divergence to the depth-0 MTP block. The first use
  found and fixed the resident MTP RoPE dimension mismatch (`qk_head_dim=256`
  vs model `rope.dimension_count=64`); the attention-debug extension ruled out
  hipEngine's dense-attention kernel math for the seq-position-49 divergence.
- Remove when: llama.cpp tensor/KV parity is either achieved or superseded by a
  more complete graph-tensor trace facility. Keep it default-off until then.

## `--record-target-topk-scores`
- Added 2026-07-03. Bench flag that asks `verify_target_block()` to copy the
  already-materialized full target lm-head logits back to host for block-verifier
  rows and serialize compact `target_lm_head_score_rows` with top-k plus
  candidate-token scores. `--target-score-candidate-tokens` adds explicit
  llama.cpp near-tie tokens to the candidate list. When score rows are present,
  the same diagnostic also emits compact `target_hidden_seed_rows` summaries so
  the scored verifier hidden row can be lined up with llama.cpp `verify_h`
  traces without dumping full hidden vectors in the normal artifact.
- Purpose: diagnose the active llama.cpp parity miss on target verifier
  near-ties without relying on forced-target replay. The first smoke artifact
  `benchmarks/results/2026-07-03-mtp-target-score-capture-smoke.json` populated
  three live target verifier rows on the active `llama-compat` direct-commit
  shape. The hidden-seed follow-up artifact
  `benchmarks/results/2026-07-03-mtp-mixed-ja-en-translate-target-hidden-scores-live.json`
  captures the live task-9/cycle-3/row-2 hidden summary for the `8940` vs `668`
  rank flip.
- Remove when: target hidden-to-logit parity is closed or replaced by a broader
  cross-engine tensor trace. Keep it default-off; the extra full-logit D2H copy
  makes it invalid for retained timing claims.

## `HIPENGINE_GGUF_AR_PACKED_DECODE` (default-on packed decode)
- Added 2026-07-05 as a packed-verifier AR diagnostic, then replaced by the
  retained decode-shaped packed AR path. The current default-on route calls
  `Qwen35GGUFResidentSession.step_batch_native(..., scatter_state=False)` for
  prepared multi-prompt GGUF greedy AR, keeps packed multi-slot state canonical
  across decode cycles, and scatters back only before stream/scalar fallback or
  a changed chunk layout.
- Purpose: provide the first useful GGUF AR c>N server backend after fixing
  default-route request coalescing while preserving each request's canonical
  state. The exact route uses c1 per-slot linear-attention state slices and
  keeps full attention plus MoE/FFN row-batched. Deferred flush copies the full
  live KV prefix instead of only the last dirty row.
- Result: E1/E2 retain direct physical c8 at **127.902/246.872 aggregate
  tok/s** on gfx1151/gfx1100 with complete token/hidden/state/KV and profiler
  gates. E3/F1 then retain C13 as c8+sparse-c8, middle-hole/new admission, and
  real p512/128-output SSE logical c1/c8/c9/c13/serial-c13 at
  **15.701/86.338/57.127/72.522/42.764 tok/s** on gfx1151 and
  **25.583/136.122/88.592/111.380/31.708** on gfx1100. Each packet preserves
  all **189** server prompt/output rows, and every packed static/live route
  records zero serial/resident fallback.
- Remove when: the one-release rollback window ends on both gfx11 targets; both
  correctness/profiler/repetition/live-server triggers are met. Keep only
  registry-resolved unsupported-shape fallback.

## `HIPENGINE_GGUF_AR_PACKED_PREFILL` (default-on packed prompt prefill)
- Added 2026-07-05 after the packed-decode AR route exposed prompt prefill as
  the remaining c>N server AR limiter. The current default-on route calls
  `Qwen35GGUFResidentSession.prefill_batch_native(...)` for multi-prompt GGUF
  greedy AR, packs prompt rows slot-major, scatters the resulting KV/recurrent
  state back to each resident session, and samples only each slot's final
  prompt row before entering packed decode.
- Purpose: remove the serial per-slot prompt-prefill wall from coalesced
  OpenAI-server AR batches without reusing the MTP verifier result contract.
  The rejected verifier-as-prefill probe sampled/copied all prompt rows and
  measured c=8 **50.56 tok/s**, below the retained packed-decode baseline.
- Result: the 2026-07-13 state audit found the old packed full-attention decode
  reduction first changed BF16 layer output at layer 31, then Conv/GDN state at
  layer 32 and live KV at layer 35. Packed prefill now uses the span-aware paged
  prefill reduction below 512 rows; if any slot crosses the AOTriton threshold,
  full attention runs slot-locally with c1 math while linear/MoE remains packed.
  Steady c4 and ragged `[512,64,64,64]` are token/Conv/GDN/live-KV exact. The old
  **65.91/82.41/63.17 tok/s** row predates these gates and exact generated-ID
  accounting; it is historical rather than a current retained baseline.
- Result addendum: E3/F1 covers exact C13 prompt/state/KV plus **189/189** real
  server prompt-ID/usage/output rows per gfx11 target under repeated p512 bursts
  and live c8→c13 admission.
- Remove when: the one-release rollback window ends on both gfx11 targets. Keep
  fallback for total prompt slabs beyond the current packed hidden-row guard.

## `HIPENGINE_GGUF_MTP_SERVER_VERIFY_FINAL_STATE_FASTPATH`
- Added 2026-07-06 as a default-off MTP serving diagnostic after the first
  no-capture packed-verifier probe changed MTP economy. The corrected version
  keeps packed slot segments through Conv/GDN prefill, mutates per-slot packed
  final linear state directly, and falls back to accepted-prefix replay for
  partial/reject cycles.
- Purpose: test whether skipping per-row linear-state capture can beat the
  retained captured-row verifier once the no-capture path is semantically
  equivalent for packed c>N serving.
- Result: rejected on AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) with
  Qwen3.6-35B-A3B `UD-Q4_K_M`, natural24 `max_tokens=24`, 5 ms server batch
  window. c=4 measured **66.75 tok/s** in
  `benchmarks/results/2026-07-06-hipengine-server-mtp-natural24-c4-bw5-finalstate-fastpath2.json`
  versus retained **76.83 tok/s** for
  `benchmarks/results/2026-07-06-hipengine-server-mtp-natural24-c4-bw5-rowtilechunk-verify.json`.
  Acceptance stayed identical (**0.8545**, draft **165**, accepted **141**), but
  `target_state_commit_ms` rose **10.443 -> 405.559 ms** because
  partial/reject cycles must replay the consumed prefix without captured rows.
- Remove when: a compact selected-row capture path exists, or if no follow-up
  uses the segment-aware no-capture kernel. The flag must stay default-off and
  must not be used for retained timing claims.

## `HIPENGINE_GGUF_MTP_SERVER_ROLLING_SLOTS`
- Added 2026-07-06 as a default-off MTP serving diagnostic while trying to lift
  the four-request MTP route cap without using a true width-8 verifier. The
  route keeps at most four live resident slots, opens replacements in warmed
  widths when possible, and can hold a stable packed-verifier owner session so
  replacement slots do not allocate owner workspaces mid-batch.
- Purpose: test whether c=8 can avoid the fixed two-backend-group barrier while
  preserving the retained four-slot packed verifier shape.
- Result: rejected on AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) with
  Qwen3.6-35B-A3B `UD-Q4_K_M`, natural24 `max_tokens=24`, 5 ms server batch
  window. Naive rolling measured **11.22 tok/s** at c=8; the stable-owner /
  warmed-width variant improved to **61.23 tok/s**, still below retained
  **79.61 tok/s**. Economy stayed normal (`draft=165`, `accepted=141`, accept
  rate **0.8545**), but replacement slot opening/prefill exposed
  **14.613 s** aggregate `slots_open_ms`. The default MTP route cap remains
  four; guarded default c=8 rerun measured **78.91 tok/s**.
- Remove when: a true cap>4 MTP scheduler can pre-open/reuse replacement slots
  without exposing slot-open/prefill wall, or after the next c>N MTP scheduler
  direction supersedes it. The flag must stay default-off and must not be used
  for retained timing claims.

## `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_SUFFIX_ROW_CHUNK_*`
- Added 2026-07-09 as a default-off PARO c>N diagnostic:
  `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_SUFFIX_ROW_CHUNK_SIZE`,
  `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_SUFFIX_ROW_CHUNK_LAYERS`, and
  `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_SUFFIX_ROW_CHUNK_INCLUDE_GATE`.
  The path keeps batch full-attention QKV/append/context, then chunks either
  O/post/MoE or gate/O/post/MoE over row sub-batches. It is exposed through
  `scripts/qwen35_batch_retained_bench.py` and
  `scripts/qwen35_batch_hidden_bisect.py`, records suffix-rowchunk metadata in
  `last_batch_decode_execution`, and blocks native-caware claims.
- Purpose: isolate the remaining gfx1151 c6 full-attention rowchunk tax after
  context-only rowchunking rejected. It tests whether the green selected
  full-layer rowchunk bridge is paying for post-context suffix work.
- Result: rejected on AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) with
  `Qwen3.6-35B-A3B-PARO-packed`, `w4_paro`, rows=6, prompt=512,
  decode=16, selected-c1 MoE, forced small-batch shared expert, and suffix
  rowchunk2 on layers `3,7,11,15,19,23,27,31`. After batch context+gate:
  **106.864 tok/s**, median **53.609 ms**, generated-token red at token 9
  (`12` vs c1 `27`). Including gate in the suffix chunks: **107.508 tok/s**,
  median **53.189 ms**, same token-9 failure. Compact summary:
  `benchmarks/results/2026-07-09-hipengine-qwen35-c6-suffix-rowchunk-rejects-summary.json`.
- Remove when: c6 full-attention rowchunk isolation moves to lower-level
  hidden/KV source tracing or a retained green non-rowchunk c6 path exists.
  Keep the flags default-off and do not use them for retained timing claims.

## `scripts/qwen35_batch_hidden_bisect.py --compare-full-attn-rowchunk-boundary`
- Added 2026-07-09 as a default-off PARO c>N diagnostic mode. It compares two
  native rows=6 batch variants directly: no-rowchunk full attention versus the
  selected full-layer rowchunk repair, using the existing hidden-bisect summary
  machinery but labelling the rowchunk repair as the comparison peer instead of
  an independent c1 oracle.
- Purpose: isolate whether the remaining c6 full-attention rowchunk tax comes
  from KV append/page placement, context-only work, suffix work, or from a
  whole-layer numerical boundary introduced by rowchunking.
- Result: the L8 trace showed layer 3 full-layer rowchunk output drift still
  under tolerance (`0.000122 max_abs`), layer 4 `attn_input` first over
  tolerance (`0.001953`), and layer 7 `attn_input_pre_qkv` at `0.0078125`.
  The corrected combined context+suffix rowchunk probe records both
  `native_context_row_chunks` and `native_suffix_row_chunks_include_gate` on
  layers `3,7,11,15,19,23,27,31`, but remains generated-token red at token 2
  (`220` vs c1 `17`) and slower than the current green selected full-layer
  rowchunk bridge (`103.998 tok/s`, median `54.865 ms`). Compact summary:
  `benchmarks/results/2026-07-09-hipengine-qwen35-c6-rowchunk-boundary-combined-summary.json`.
- Remove when: the c6 full-layer rowchunk tax is either fixed or the scheduler
  avoids live c6 groups in retained/default operation. Do not use this
  comparison mode for throughput claims.

## `HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM`

- Added 2026-07-12 as a default-on PARO prefill rollback/bisection control.
  The retained path creates one lazy nonblocking stream plus two reusable HIP
  events for AOTriton query rows above the proven-safe 256-row bucket, and
  keeps all pre/post work on the caller stream. Setting the flag to `0`
  restores the old same-stream dispatch without changing model math.
- Purpose: isolate AOTriton's high-scratch dispatch from the queue used by
  later low-scratch linear-attention convolution kernels. On gfx1151 the same
  captured convolution changes from about `1.83 ms` after same-stream
  4096-row AOTriton to `119 us` when AOTriton uses the isolated queue. The
  clean 256-row AOTriton trace uses much less scratch (`992/1008` versus
  `2560` bytes) and does not trigger the cliff. The first
  final clean matched prefill A/Bs improve 4K `885.141 -> 1089.031` (+23.03%),
  32K `765.316 -> 906.145` (+18.40%), 64K `621.691 -> 716.775` (+15.29%),
  and 128K `418.838 -> 474.641 tok/s` (+13.32%). Decode stays within
  `-0.16%..+0.12%`, tracked peak is unchanged, and every shape matches sampled
  seed, final hidden, all 30 Conv/GDN state families, and all 10 live K/V
  families. The 1K 256-query negative control never enters isolation and is
  not promoted. Retained evidence:
  `benchmarks/results/2026-07-12-gfx1151-paro-aotriton-stream-isolation.json`.
- A clean W7900/gfx1100 transfer matrix measured the earlier global-isolation
  policy in balanced 15-sample legs. Isolated prefill changes by
  `+1.638%/+0.495%/+0.192%` and total measured wall falls by
  `1.653%/0.127%/0.562%` at 512/1K/4K, with byte-exact hidden/state/KV at every
  shape. The merged threshold intentionally leaves the 256-query 512/1K path
  same-stream; its 4K/4096-query result directly validates the scoped gfx1100
  default. Retained evidence:
  `benchmarks/results/2026-07-12-w7900-gfx1100-paro-gfx1151-transfer.json`.
- GPF-4 extends the same selector to GGUF but is **default-off on both
  backends** after the final stability gate rejected its provisional gfx1151
  capability. Clean focus is byte-exact and often fast, but automatic-route
  32K collapses once to 294.254 tok/s, its fresh 1+5 replacement fails to
  finish the warmup after 481 s, and 128K measured run 2 stays GPU-active past
  1200 s. A same-stream 32K control immediately completes normally. Keep `=1`
  only as a diagnostic; do not remove the same-stream GGUF route. Evidence:
  `benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf4-candidate-focus.json`,
  `benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf4-clean-promotion.json`,
  and
  `benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf4-final-rejected.json`.
- Remove when: either upstream removes the queue-scratch cliff so isolation is
  unnecessary, or a replacement isolation design passes the full stability
  gate and survives one release cycle. Then delete the rejected duplicate
  route and selector; keep only the proven scheduling policy.

## Laguna D8 row-vector selected gate/up rollback

- Added 2026-07-26 after the exact wave-column MMQ128x32 consumer became the
  gfx1151 production default. The older `mmq128x32_d8_f32_rowvec` mode remains
  an explicit session-level rollback. Clean selector-unset pp512 improves
  **385.602 -> 432.355 tok/s**, cached tracing measures **434.994 tok/s**, and
  the canonical absolute-quality gate passes unchanged at maximum KL
  **0.049542582** and **316/320** top-1.
- Removal trigger is now satisfied by the retained Q4-down checkpoint at
  **448.203 tok/s**. Remove the explicit gfx1151 session rollback during the
  immediate post-publication cleanup; keep the underlying scalar-staged D8
  body only if it still serves a tested unmeasured-backend fallback.

## Laguna Q4 down row-vector rollback

- Added 2026-07-26 after the exact Q4-only wave-column MMQ64x32 consumer
  became the gfx1151 production default. The older
  `mmq64x32_d4_f32_rowvec` mode remains an explicit session-level rollback.
  The implementation-worktree four-mode screen improves Q4-wave/Q6-row
  **433.791 -> 448.945 tok/s (+3.493%)**; Q6 wave columns regress and remain
  rejected. Clean publication confirms **433.081 -> 448.203 tok/s
  (+3.492%)**, max KL **0.049542582**, and the complete trace/lifecycle gate.
- Removal trigger is satisfied. Remove the explicit all-row-vector selector
  during immediate cleanup while retaining the row-vector kernel internally
  for production Q6 unless a different direct-consume mapping wins its own
  quant-isolated gate.

## Laguna Q4 gate/up pair-decode wave-column rollback

- Added 2026-07-26 when direct per-column T16 decode became the next gfx1151
  candidate default. The older `mmq128x32_d8_f32_wavecols` mode remains an
  explicit session-level rollback beside
  `mmq128x32_d8_f32_wavecols_direct`. The candidate preserves resident bytes
  and arithmetic, is BF16-byte identical, improves the actual layer-1 leaf
  **8.107 -> 6.916 ms**, and improves the implementation-worktree pp512 screen
  **447.582 -> 472.533 tok/s**.
- Removal trigger is satisfied: clean selector-unset pp512 improves
  **449.020 -> 474.363 tok/s**, direct all-exact quality/lifecycle pass, and
  cached all-family tracing names the intended direct template. Remove the
  explicit pair-decode session route during immediate cleanup; keep no
  duplicate positive selector solely for historical benchmarking.

## Laguna Q4-down pair-decode wave-column rollback

- Added 2026-07-26 when direct per-column Q4-down decode became the next
  gfx1151 candidate default. The older
  `mmq64x32_d4_f32_wavecols_q4` mode remains explicit rollback beside
  `mmq64x32_d4_f32_wavecols_direct_q4`. The candidate is BF16-byte identical
  and improves the implementation-worktree pp512 screen
  **473.774 -> 483.409 tok/s** while Q6 stays on row-vector production.
- Removal trigger is satisfied: clean selector-unset pp512 improves
  **473.963 -> 480.629 tok/s**, direct all-exact quality/lifecycle pass, and
  cached tracing cuts the Q4 consumer **90.280 -> 71.378 ms**. Remove the
  explicit Q4 pair-decode selector during immediate cleanup. Do not couple
  this cleanup to the independently retained Q6 row-vector kernel.

## Laguna rejected Q6 down wave-column template branch

- Added 2026-07-26 during the Q4/Q6 direct-consume split. Q6 quartet
  decode/shuffle is no longer exported or runtime-selectable after regressing
  actual pp512 **433.791 -> 428.184 tok/s (-1.293%)**, but its compile-time
  template branch remains beside the retained Q4 implementation for now.
- Removal trigger is satisfied by the Q4-only production checkpoint. Delete
  the dead compile-time branch during immediate cleanup. Do not re-export it
  without a new mapping and an independently positive Q6 gate.

## Laguna Q6 selected-down 32-row rollback

- Added 2026-07-26 when the exact 64-row/local128 Q6 selected-down body became
  the gfx1151 candidate default. The older
  `mmq64x32_d4_f32_wavecols_direct_q4` session mode retains Q4 direct
  wave-column decode and Q6 32-row row-vector consumption as an explicit
  rollback. The actual runtime-bound Q6 leaf improves **5.260 -> 5.161 ms**,
  dirty pp512 improves **490.105 -> 491.335 tok/s**, and outputs are BF16-bit
  exact.
- Clean publication is complete at **489.110 -> 492.640 tok/s (+0.722%)**
  with all seven paired wins and cached attribution at **493.509 tok/s**.
  Remove the explicit 32-row Q6 session selector after one subsequent retained
  selected-expert checkpoint. Keep the underlying 32-row template if it
  remains the fallback for non-gfx1151 backends or shapes below the 64-row
  admission threshold.

## Laguna serial stable MoE compaction rollback

- Added 2026-07-26 when gfx1151 changed from one-workgroup serial
  expert-major compaction to the byte-identical per-expert parallel
  count/prefix/ballot-scatter path.
- The session-level `serial` selector and registered one-pass kernels remain
  for rollback and unmeasured backends. The production-shape leaf improves
  **0.348880 -> 0.058969 ms (-83.10%)**, and clean publication improves
  **490.824 -> 497.408 tok/s (+1.341%)** with all seven paired wins. Cached
  attribution independently reaches **500.325 tok/s**.
- Remove the explicit gfx1151 serial session selector after the next retained
  exact MoE metadata checkpoint. Keep the serial registry primitive if another
  backend still requires it as an unfused/reference fallback.

## Laguna Q4 gate/up activation-double-buffer candidate selector

- Added 2026-07-26 for the exact
  `mmq128x32_d8_f32_wavecols_direct_doublebuf` session candidate. It
  ping-pongs the 1.5 KB activation tile, removes one of two barriers per K32,
  preserves resident bytes and arithmetic order, and is BF16 byte-exact.
  The actual natural-M512 inclusive leaf improves **6.995 -> 6.907 ms
  (-1.258%)**.
- If clean complete-state A/B is positive, make the double-buffer body the
  gfx1151 package default and remove the redundant positive selector after its
  clean publication; retain the current direct body only as the explicit
  synchronization rollback through one later expert-family checkpoint. If the
  complete-state A/B is negative, remove the double-buffer export, selector,
  harness mode, and template branch immediately.
- The first trigger is satisfied: clean pp512 improves **505.970 -> 507.405
  tok/s (+0.284%)** with **5/7** pair wins and complete-state exactness, and
  the package default is promoted. Remove the redundant positive selector and
  A/B harness after clean selector-unset publication; keep the prior direct
  body as rollback through one subsequent expert-family checkpoint.
- The publication trigger is now satisfied: selector-unset production is
  **505.084 tok/s** median / **504.984 tok/s** minimum, matched A/B remains
  **+0.284%**, and cached tracing cuts gate/up **318.559 -> 314.378 ms**.
  Remove the redundant positive selector and one-purpose A/B harness in the
  next cleanup unit; keep the direct synchronization rollback until the next
  retained expert-family checkpoint.

## Laguna cached-only attention scheduling rollback

- Added 2026-07-26 with the gfx1151
  `LAGUNA_PREFILL_KV_PREAPPEND` capability and the optional
  `prefill_kv_preappend` session override. Complete M128 global tiles and
  pre-wrap SWA tiles append before cached-only qrow4 attention; partial,
  wrapped, verifier, gfx1100, and unmeasured paths preserve the established
  attend-then-append sequence.
- The implementation-tree gate is exact and positive at pp512
  (**507.391 -> 528.771 tok/s, +4.214%**) and at 1K/4K
  (**1.103x/1.047x**). Keep the explicit rollback through clean
  selector-unset publication and one later attention-family checkpoint.
- Clean selector-unset publication is now satisfied at **526.451 tok/s**
  median / **526.288 tok/s** minimum; 1K/4K improve **3.218%/5.701%**, and
  cached tracing cuts attention **219.709 -> 176.580 ms (-19.63%)**.
  The one-later-attention-checkpoint trigger remains open.
- After both triggers pass, remove the public constructor override while
  retaining the backend capability/default and the automatic safety
  fallbacks. Do not remove the source-qualified kernels: wrapped SWA and
  staged verifier transactions still require them.
- The next exact checkpoint adds `LAGUNA_PREFILL_CACHED_META` and the optional
  `prefill_cached_meta` session/cache rollback. It selects metadata-only qrow4
  for every safe SWA tile and global tiles from position 128, retaining the
  established cached body at global position 0. Matched seven-pair pp512
  improves **533.507 -> 542.785 tok/s (+1.739%, 7/7 wins)** and every compared
  output/state digest is exact. Clean selector-unset publication is now
  satisfied at **542.088 tok/s** median / **542.022 tok/s** minimum, with
  1K/4K at **478.856/387.725 tok/s** and traced attention
  **175.802 -> 160.123 ms (-8.92%)**. Remove the public positive selector after
  one later attention-family checkpoint; retain source-qualified registered
  fallbacks for global start 0, partial tiles, wrapped SWA, verifier
  transactions, and unmeasured backends.
- The next exact global-only checkpoint adds
  `LAGUNA_PREFILL_GLOBAL_QROW6` and the optional
  `prefill_global_qrow6` session/cache rollback. It applies only to complete
  preappended global M128 tiles from position 128; global position 0 and every
  SWA tile retain qrow4. Seven complete-state pairs improve
  **546.056 -> 548.774 tok/s (+0.498%, 7/7 wins)** with identical
  logits/hidden/KV/token/cursor. Remove the public positive selector after
  clean selector-unset publication plus one later attention checkpoint. The
  first trigger is satisfied at **547.064/513.180/428.628 tok/s**, with traced
  attention **158.702 -> 152.406 ms (-3.97%)**. Keep automatic qrow4
  fallbacks and do not reintroduce the rejected SWA qrow6 surface.
- The next exact checkpoint adds `LAGUNA_PREFILL_DENSE_INITIAL` and optional
  `prefill_dense_initial` session/cache rollback. It selects only consecutive
  complete M128 initial tiles with no wrap or explicit eviction, preserving
  cached-metadata/current-source fallbacks everywhere else. Seven matched
  complete-state pairs improve **552.144 -> 559.539 tok/s (+1.339%, 5/7
  wins)** with identical logits/hidden/KV/token/cursor. Remove the public
  positive selector after clean selector-unset publication plus one later
  attention checkpoint; keep the automatic safety fallbacks permanently.
  The publication trigger is satisfied at **559.290/523.090/439.044 tok/s**,
  with traced attention **153.226 -> 141.846 ms (-7.43%)**. Keep the rollback
  through one later attention checkpoint.

## Laguna Q6 qmicro resident-layout rollback

- Added 2026-07-26 with the gfx1151 `LAGUNA_Q6_QMICRO` capability and optional
  `q6_qmicro=False` resident-session rollback. Sparse `ffn_down_exps` Q6
  payloads convert from the existing legacy cache once before upload; the
  cache fingerprint, quant key, byte count, root lm-head, and other backends
  stay unchanged.
- The actual-weight leaf is BF16-byte exact and improves natural-M512
  selected prefill **1.65%** plus top-10 exact decode **6.99%**. Clean
  selector-unset publication is now satisfied at **530.447 tok/s** median /
  **525.864 tok/s** minimum, with 1K/4K at **473.118/381.375 tok/s** and
  traced Q6 **126.594 -> 123.473 ms (-2.465%)**. Keep the explicit rollback
  through one later selected-down checkpoint.
- After both triggers pass, remove the public constructor override while
  retaining the backend capability, legacy consumer instantiations required
  by other backends, and the host legacy-to-qmicro cache adapter. If clean
  publication regresses, disable the capability and remove the qmicro runtime
  default before starting another selected-down premise.

## Laguna fused selected-SiLU pack rollback

- Added 2026-07-26 with the gfx1151
  `LAGUNA_FUSED_SELECTED_SILU_PACK` capability and session-local
  `set_fused_selected_silu_pack(...)` rollback. The exact composite preserves
  the standalone SiLU BF16 boundary, reuses selected-down scratch, and leaves
  the required standalone SiLU plus activation-quant primitive fallback
  registered.
- Primitive and production Q4_K/Q6_K MoE equality pass. Seven complete-state
  pp512 pairs are exact and win **7/7**, paired geometric throughput improves
  **0.651%**, and tracing removes 47 launches while cutting the target window
  **10.301 -> 6.377 ms (-38.09%)**. Clean selector-unset publication then
  improves **543.807 -> 546.100 tok/s (+0.422%)** and records the intended 47
  fused/zero standalone calls. Keep the session rollback through one later
  selected-down checkpoint.
- After that trigger passes, remove the public setter while retaining the
  backend capability/default and registered unfused numerical fallback. If
  a later checkpoint regresses, disable the capability before another
  selected-down experiment.

## Laguna Q6 compact-activation rollback

- Added 2026-07-26 with gfx1151 `LAGUNA_Q6_COMPACT_ACTIVATION` and the optional
  `q6_compact_activation=False` resident-session rollback. The Q6-specific
  activation cache omits unused Q8 sum metadata and stores its bounded K16
  quant sums as int16, reducing LDS **5,632 -> 5,120 B** without changing
  arithmetic or resident bytes.
- The actual leaf improves **3.36%**, fifteen complete-state pp512 pairs
  improve **550.584 -> 552.807 tok/s (+0.404%, 15/15 wins)**, and tracing
  cuts Q6 **125.380 -> 119.566 ms (-4.64%)**. Keep the explicit rollback
  through clean selector-unset publication and one later selected-down
  checkpoint.
- Clean selector-unset publication is satisfied at
  **550.625/517.017/431.789 tok/s**, and the production trace records
  **119.384 ms** Q6 plus **71.641 ms** Q4. The one-later-selected-down
  checkpoint trigger remains open.
- After both triggers pass, remove the public constructor override and the
  one-purpose A/B harness. Retain the ordinary 48-byte activation cache only
  where other Q4/Q6 template instantiations or unmeasured backends require it.

## Laguna Q6 half-row activation rollback

- Added 2026-07-26 with gfx1151 `LAGUNA_Q6_HALF_ROW_ACTIVATION` and optional
  `q6_half_row_activation=False` session rollback. It is constrained to the
  qmicro/compact-cache/64-row/local128 Q6 body and changes only activation
  staging ownership.
- The actual 23-layer screen improves **21/23** layers and
  **111.798 -> 111.490 ms (-0.276%)** with exact BF16 output. Complete pp512
  A/B is exact and positive at **552.562 -> 553.018 tok/s (+0.083%)**.
- Keep the rollback through clean selector-unset publication and one later
  selected-down checkpoint. Then remove the public constructor override and
  the one-purpose A/B harness while retaining the backend capability/default.
- Clean selector-unset publication is satisfied at
  **549.150/514.956/430.300 tok/s**; it is headline-neutral, while the clean
  trace preserves the Q6 sub-window win at **119.384 -> 118.568 ms (-0.684%)**.
  The one-later-selected-down checkpoint remains open.

## Laguna Q6 padded-activation rollback

- Added 2026-07-26 with gfx1151
  `LAGUNA_Q6_SKIP_PADDED_ACTIVATION` and optional
  `q6_skip_padded_activation=False` session rollback. It is constrained to
  the exact qmicro/compact/half-row/64-row Q6 body.
- The candidate skips only LDS stores and K16 sums for padded slots that the
  guarded dot and output loops never consume. The actual 23-layer screen is
  exact and improves **112.008 -> 111.806 ms (-0.180%, 19/23 layers)**;
  complete pp512 is exact at **552.983 -> 553.559 tok/s (+0.104%)**.
- Keep the rollback through clean selector-unset publication and one later
  selected-down checkpoint. Then remove the constructor override and
  one-purpose A/B harness while retaining the backend default.
- Clean selector-unset publication is satisfied at
  **551.459/517.307/432.099 tok/s**. The repeated 23-layer and complete-state
  A/B evidence remains positive; the one-later-selected-down checkpoint is
  still open.

## Laguna Q6 qmicro-permute rollback

- Added 2026-07-26 with gfx1151 `LAGUNA_Q6_QMICRO_PERMUTE` and optional
  `q6_qmicro_permute=False` session/profile rollback. It is constrained to
  the exact qmicro/compact/half-row/skip-padded/64-row Q6 body and changes
  neither resident bytes nor arithmetic order.
- The actual layer-1 leaf is BF16-byte exact and improves
  **4.872 -> 4.741 ms (-2.67%)**. Seven complete-state pp512 pairs improve
  **567.998 -> 569.563 tok/s (+0.276%, 5/7 wins)**. Cached tracing observes
  all **115** intended calls with scratch 0 and reduces their total
  **1,138.893 -> 1,124.852 ms (-1.23%)** across 512/1K/4K.
- Keep the rollback through clean selector-unset publication and one later
  selected-down checkpoint. Then remove the constructor/profile override
  while retaining the backend capability and scalar-unpack fallback for
  unmeasured backends and non-production template shapes.

## Laguna Q6 planar-qmicro candidate rollback

- Added 2026-07-26 with gfx1151
  `LAGUNA_Q6_QMICRO_PLANAR` plus the temporary
  `q6_qmicro_planar=True` session/profile selector. It changes only the
  byte order inside each existing 12-byte qmicro record and selects matching
  prefill, direct-decode, and grouped-small-M consumers.
- The exact actual-weight leaf is positive. Two opposite owner-order
  complete-state blocks are aggregate-neutral (+0.013% combined mean,
  +0.139% combined median) with exact logits/hidden/KV/token/cursor, so the
  repo's verified-subwindow rule retains and enables the capability. Keep
  `q6_qmicro_planar=False` only through clean selector-unset publication and
  one later selected-down checkpoint, then remove the constructor/profile
  override and obsolete production permute selector. Preserve legacy and
  interleaved qmicro fallbacks for unmeasured backends and rollback caches.
- Clean selector-unset publication is satisfied at
  **573.354/530.351/446.189 tok/s**, improving every preceding production
  length. The one-later-selected-down checkpoint remains open.

## Laguna source-F16 boundary-fusion rollback

- Added 2026-07-26 with session-local
  `set_f16_boundary_fusion(...)`. The exact range-direct specialization folds
  each separate BF16-to-FP16 cast into its RMSNorm or softplus-gate producer
  while preserving the BF16 rounding boundary bit-for-bit. The registered
  unfused primitives remain the required fallback.
- Primitive composition is exact and improves the two production M512
  boundaries by **1.877x/1.425x**. Seven matched pp512 pairs improve
  **554.909 -> 559.320 tok/s (+0.795%, 6/7 wins)** with identical token and
  logit. The gfx1151 package capability now enables the exact path by default;
  clean selector-unset publication passes at
  **559.554/523.912/440.809 tok/s**. A second seven-pair gate preserves
  logits, hidden states, complete KV, token/logit, and cursor exactly.
  Cached tracing removes all **96** standalone pp512 casts and records
  **1,696** dispatches.
- After promotion and one later source-F16 checkpoint, remove the public
  setter and one-purpose A/B harness while retaining the backend capability,
  fused registered kernels, and registered unfused numerical fallback.

## Laguna Q6 selected-down integer-WMMA leaf selector

- Added 2026-07-26 as the optional `integer_wmma` wrapper argument and matching
  actual-weight leaf A/B mode. An unspecified value selects integer WMMA only
  for the already-constrained planar-qmicro/compact/half-row/skip-padded
  row64 path; `False` preserves the packed-dot comparator.
- The CPU-reference gate and actual layer-1 leaf are BF16-byte exact. Twenty-one
  counter-rotated natural-M512 pairs improve **4.7654 -> 4.5655 ms (-4.20%,
  21/21 wins)** with zero resident-byte change.
- Keep the explicit comparator through clean selector-unset publication and
  one refreshed selected-down family trace. Then remove the one-purpose leaf
  A/B mode and make the admitted production specialization unconditional
  inside the planar-row64 wrapper while retaining packed-dot bodies for all
  other template shapes and unmeasured backends.
- Clean selector-unset publication is satisfied at
  **576.137/543.213/459.054 tok/s**, improving every preceding production
  length with deterministic tokens, exact positions, and complete allocation
  return. The refreshed trace cuts selected down
  **189.049 -> 181.583 ms (-3.95%)** and the 115-call Q6 window
  **1,124.852 -> 792.625 ms (-29.54%)**.
- The immediately queued activation-fragment hoist is also admitted at the
  leaf gate: **4.5645 -> 4.5126 ms (-1.136%, 20/21 wins)**, zero BF16
  mismatches, and unchanged local128/VGPR96/LDS5120B/scratch0 resources.
  Clean selector-unset publication is satisfied at
  **577.396/545.366/459.716 tok/s**, improving every required length. Keep
  `wmma_hoist_activation=False` only through one refreshed family trace. Then
  remove both one-purpose
  wrapper selectors and collapse the admitted hoisted specialization into the
  unconditional planar-row64 gfx1151 production route.
- The refreshed family trace is now satisfied: the 115-call Q6 window improves
  **792.625 -> 779.709 ms (-1.63%)** with unchanged resources. Selector
  collapse is unblocked and should accompany the next accepted Q6 body or the
  post-campaign cleanup, whichever comes first.

## Laguna Q6 WMMA weight-prefetch rollback

- Added 2026-07-27 as gfx1151
  `LAGUNA_Q6_WMMA_PREFETCH_WEIGHT`, the optional
  `q6_wmma_prefetch_weight` resident-session field/setter, and two
  one-purpose benchmark modes. It applies only to the admitted
  planar-qmicro/compact/half-row/skip-padded/integer-WMMA/activation-hoist
  row64 specialization.
- The candidate overlaps the next K32 planar-qmicro record plus `d`/scale
  global loads with current fragment compute. It adds no resident bytes,
  second LDS plane, arithmetic change, or scratch. The exact layer leaf
  improves **4.518 -> 4.104 ms (-9.156%)**; seven complete-state pairs
  improve **618.294 -> 623.900 tok/s (+0.907%)**. The resource cost is
  VGPR **96 -> 104** with LDS fixed at 5,120 bytes.
- Clean selector-unset publication is satisfied at **636.073 tok/s** pp512,
  and the subsequent all-family trace cuts the exact 23-call Q6 body
  **112.746 -> 101.963 ms (-9.564%)**. Collapse the prefetch and
  already-satisfied activation-hoist selectors into the unconditional
  gfx1151 production specialization with the next retained Q6 body or
  post-campaign cleanup; retain the non-prefetch template only for unmeasured
  backends and non-production geometries.
- The retained successor adds `LAGUNA_Q6_WMMA_PREFETCH_ACTIVATION`, the
  `q6_wmma_prefetch_activation` session field/setter, and a candidate wrapper
  solely for A/B rollback. It improves clean pp512
  **636.073 -> 639.114 tok/s** and the 23-call Q6 body
  **101.963 -> 100.367 ms**, with VGPR104 -> 112 and unchanged
  LDS/scratch/resident bytes. Collapse weight prefetch, activation prefetch,
  and activation hoist into one unconditional gfx1151 production symbol after
  the current 700 campaign no longer needs the three independent bisection
  points; preserve one non-prefetch fallback for unmeasured geometries.
- The published exact successor adds
  `LAGUNA_Q6_PRECOMPUTED_ACTIVATION_SUMS`, a resident-session rollback, and
  distinct producer/consumer exports. It reuses Q6's unused D4 raw-sum field
  for two `int16` K16 quant sums without changing the 160-byte ABI. The
  actual inclusive leaf improves **4.1501 -> 4.1162 ms (-0.818%)** and 11
  complete-state pp512 pairs save **1.407 ms** at the paired median with
  **6/11 wins**. Clean selector-unset 512/1K/4K subsequently improves
  **645.803/575.942/468.311 -> 647.207/576.799/468.431 tok/s**, while the
  traced 23-call Q6 body falls **100.367 -> 99.459 ms (-0.905%)**. Both
  promotion gates pass. Keep the rollback through the current 700 campaign,
  then collapse this and the three older Q6 WMMA selectors into one
  unconditional gfx1151 production symbol.

## Laguna Q4 projection-role quality candidate

- Added 2026-07-27 as default-off selected gate/up modes
  `mmq128x32_role_gate_d4_up_d8` and
  `mmq128x32_role_gate_d8_up_d4`, plus a reusable multi-arm pp512 harness.
  Both use uniform role kernels, two existing-size activation planes, and no
  resident weight sidecar. The exact separate-input fused SiLU/down pack
  preserves the established BF16 boundary.
- D4-gate/D8-up clears the economic gate at **15.329 ms** saved in seven
  paired complete pp512 medians and reproduces its unfused complete-state
  hashes. Its clean 320-step comparison is rejected at max KL **0.061203**
  despite **317/320** top-1. Keep both role modes only through the alternate
  D8-gate/D4-up clean comparison.
- D8-gate/D4-up also fails, at max KL **0.203467** despite **317/320**
  top-1. Projection-wide modes are closed. Keep the shared role primitives
  only through the M512+ shape screen and long-prompt full-logit gate. If no
  shape-qualified route or bounded producer-row repair is retained, remove
  both role exports, the separate fused pack, plan fields, runtime modes, leaf
  modes, and their focused tests. Retain the generic multi-arm harness only if
  another selected-projection comparison reuses it.
- The default-off M512 bucket now keeps production D8 below 512 rows and
  selects D4-gate/D8-up only at M512+. Its seven-pair pp512 median saves
  **13.676 ms**. The short no-change gate passes at the exact production
  **316/320** top-1 and max KL **0.049542582**. Keep the selector and category
  lanes only through the extended-512 full-logit gate. On rejection, remove
  the M512 selector and both lanes before starting producer-row repair; on
  admission, remove both projection-wide runtime modes and collapse the M512
  selector into the gfx1151 backend capability after clean publication.
- The extended-512 gate rejects the bucket at max KL **1.379757** across nine
  failing streams. The M512 runtime selector plus its short/long category
  comparisons, lane, and selector-specific tests are removed. Keep the shared
  role kernels, activation packs, exact separate-input fused boundary, and
  generic long-prompt extension helper only because the bounded off-path
  `laguna_q4_role_risk_calibration.py` screen consumes them. Remove the role
  primitives, separate fused boundary, extension helper, and calibration
  harness if its global activation-only sweep has no economically viable
  operating point at no more than **25%** repaired producer rows.
- The fixed absmax rule passes calibration/heldout transfer, but sparse route
  expansion rejects a second weight pass. A default-off
  `mmq128x32_absmax2_layer_gate_d4_up_d8` mode therefore keeps only the
  layer-uniform case: safe layers run specialized D4-gate/D8-up and risky
  layers run production dual D8. Its seven-pair pp512 median saves
  **3.426 ms (+0.416%)**. Keep the risk pack, risky/safe conditional exports,
  conditional-layout fused pack, runtime mode, and `q4_layer_risk_absolute`
  lane only through the clean extended-512 quality gate. On rejection remove
  all of them plus the now-exhausted role primitives and calibration harness;
  on admission collapse the mode into the gfx1151 capability after clean
  selector-unset publication.
- The clean extended-512 gate rejects the layer candidate at max KL
  **1.265492** despite **314/320** top-1. The removal trigger is satisfied:
  the layer-risk pack/selector/conditional exports, both role-split modes and
  single-role exports, separate-input fused pack, category lanes and extension
  helper, calibration harness, and focused tests are removed. The reusable
  multi-arm pp512 harness remains. This cleanup closes the Q4
  projection-role candidate; no temporary runtime selector survives.

## Laguna Q4 raw-nibble P8 prefetch rollback

- Added 2026-07-27 as gfx1151 mode
  `mmq128x32_d8_f32_wavecols_direct_doublebuf_rawprefetch_ge512` and P8
  wrapper export. It carries only the next K32 interval's eight raw T16
  nibble words, preserves the existing D8 activation double buffer, and
  selects P8 only at producer chunks of at least 512 rows.
- The exact actual M512 leaf improves **6.8727 -> 6.7389 ms (-1.948%)**;
  seven complete-state pp512 pairs improve
  **636.367 -> 640.003 tok/s (+0.571%, 7/7 wins)**. M256 regresses
  **0.211%**, so the non-prefetch specialization remains required below the
  threshold rather than serving only as historical rollback.
- After clean selector-unset publication and one refreshed trace, collapse
  the user-visible positive mode into the gfx1151 shape capability. Keep one
  tested non-prefetch specialization for sub-512/unmeasured geometries, and
  remove the one-purpose leaf/complete-state comparison modes when they no
  longer serve campaign bisection.
- The clean publication trigger is satisfied at
  **643.554/573.066/466.290 tok/s**, improving all required lengths with
  exact lifecycle. The refreshed trace trigger is also satisfied: selected
  Q4 gate/up falls **337.395 -> 333.701 ms** and the P8 resource tuple is
  confirmed. Keep the explicit old/new symbols only through the immediately
  following compact-metadata-prefetch A/B; then collapse the winning >=512
  shape policy as described above.
- Compact metadata prefetch is rejected and fully removed at
  **6.7265 -> 7.0330 ms (+4.556%)**, so that bisection dependency is closed.
  Collapse the retained P8/non-P8 shape capability in the next cleanup unit;
  preserve the non-P8 specialization below 512 because it is measured faster.

## Laguna Q4 selected-down raw-nibble P8 rollback

- Added 2026-07-27 as the gfx1151 shape mode
  `mmq64x64_d4_f32_q6_wavecols_direct_rawprefetch_q4_ge512`. It reuses the
  exact gate/up P8 mechanism in the single-output 64x32/local64 Q4-down body,
  adds no resident bytes/LDS/scratch, and preserves the previous body below
  512 producer rows.
- The M512 trace cuts 72 Q4-down launches
  **217.416 -> 212.090 ms (-2.450%)**. Seven complete-state pp512 pairs
  improve **639.574 -> 643.166 tok/s (+0.562%, 7/7 wins)** with every output
  and state digest exact. Clean selector-unset publication is
  **643.141/573.717/466.913 tok/s**, with pp512 flat within run variance.
- The clean publication trigger is satisfied. After a refreshed all-family
  trace, collapse the explicit positive selector into the gfx1151 shape
  capability. Preserve the
  non-prefetch symbol for sub-512 rows and rollback/bisection because the
  admitted route is deliberately shape-qualified.

## Laguna Q4 precomputed activation-sum rollback

- Added 2026-07-27 as an exact gfx1151 M512+ default behind
  `LAGUNA_Q4_PRECOMPUTED_ACTIVATION_SUMS`. The D8 producer writes exact K16
  integer sums to a bounded activation-only `int16` sidecar, and the retained
  P8 consumer loads them instead of rebuilding them in 16 output-column
  workgroups.
- The actual leaf improves **0.434%/0.933%** at M256/M512. Eleven
  complete-state pp512 pairs save **2.491 ms** at the paired median with
  **9/11 wins** and exact state. The maximum scratch plan grows by
  **786,432 bytes**; resident weights do not grow.
- Keep the capability, setter, old consumer, and two-symbol pack/consumer
  split through the active 700 campaign for rollback and attribution. After
  clean cached trace plus selector-unset 512/1K/4K publication, collapse the
  positive path into the gfx1151 M512 shape policy. Remove the sidecar and
  candidate symbols if clean publication regresses any required length or the
  traced gate/up family does not improve.
- The clean trace/publication trigger is satisfied: selected gate/up falls
  **334.229 -> 330.720 ms (-1.050%)**, pp512 reaches **649.791 tok/s**, and
  the aggregate-flat 1K result is positive in same-process paired wall by
  **4.428 ms (7/11 wins)**. Keep rollback through the 700 campaign, then
  collapse the positive path as described above.

## Laguna packed attention-output rollback

- Added 2026-07-27 as the gfx1151 default candidate behind
  `LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_OUTPUT_GATE`. Qualified
  dense-initial M128 library PV tiles write directly to head-major context
  slices; one exact packed-aware softplus gate restores generic BF16 or
  FP16-via-BF16 output. The old output transpose remains available through
  the session constructor for clean A/B and rollback.
- Eleven complete-state pp512 pairs are exact and improve independent medians
  **645.735 -> 647.920 tok/s (+0.338%)**, but paired-median wall is only
  **-0.057 ms** with **6/11 wins**. Keep the capability, optional
  `unpack_output`, generic gate, packed gate, and dedicated A/B harness only
  through the immediate clean trace. Remove the candidate if the trace does
  not eliminate all **144** output-transpose launches or if the replacement
  gate repays the saved sub-window. On admission, preserve the generic route
  for fallback attention and unmeasured backends, but collapse the dedicated
  A/B harness after the active 700 campaign.
- The repaired clean trace satisfies admission: all **144** output transposes
  are absent, dispatches fall **2,417 -> 2,273**, and transpose plus gate
  improves **11.240 -> 10.318 ms (-8.20%)**. Keep the generic route because
  fallback attention and unmeasured backends require generic output; remove
  only the dedicated A/B harness and constructor override after the active
  700 campaign no longer needs this bisection point.

## Laguna prefill-only routed-width selector

- Added 2026-07-27 as the explicit diagnostic
  `LagunaGGUFResidentSession.set_prefill_moe_top_k`. It changes only the
  logical plan attached to the already-full-size row scratch; the separately
  owned c=1 scratch keeps the model-declared top-10 route.
- A five-repetition pp512 screen measures top-10/top-9/top-8 medians
  **648.578/684.313/720.130 tok/s**. Each mode is internally deterministic
  and selects token 2930, while cross-mode hidden/logit/KV hashes differ as
  expected for approximate routing.
- Top-8 is rejected at max KL **0.671401** and top-9 is also rejected at
  **0.452960**, both with **314/320** top-1. The removal trigger is satisfied:
  the runtime setter, multi-arm harness mode, both category
  comparisons/configurations, and focused tests are removed. Production
  remains model-declared top-10 for prefill and decode.

## Laguna gfx1151 head/KV transfer selector

- Added 2026-07-28 as a default-off gfx1151 screen for the retained gfx1100
  current-P4 head-RMSNorm + partial-RoPE + BF16 KV-write composites. The
  registry aliases and `--head-kv-fusion` profiler selector exist only to run
  the architecture-local p512/d128 and trace gates.
- If the candidate is exact and improves the complete decode median, promote
  `LAGUNA_HEAD_KV_FUSION` and keep explicit False as the unfused rollback
  through the next decode campaign. If it is neutral or negative, restore the
  gfx1151 alias exclusions and remove the profiler selector and candidate
  tests instead of carrying a dead backend path.
- The promotion trigger is satisfied: clean p512/d128 improves
  **11.466687 -> 11.483587 tok/s (+0.147%)**, and all three candidate samples
  beat all three controls with identical trajectory/state. Keep explicit
  False through the next split-attention transfer screen, then collapse the
  positive CLI selector if it is no longer needed for attribution.

## Laguna gfx1151 split-attention transfer selector

- Added 2026-07-28 as a default-off gfx1151 screen for the retained gfx1100
  exact global/SWA split-attention bundle. `--decode-split-attention` requests
  global/SWA/tile16 thresholds **127/65/257**, gated reducers, and the
  wave-local SWA reducer together; package defaults remain unsplit.
- If the clean p512/d128 candidate is exact and faster, promote the complete
  threshold/capability bundle and retain `--no-decode-split-attention` through
  one follow-up context crossover check. If the full-model gate is neutral or
  negative, restore all ten gfx1151 alias exclusions and remove the profiler
  selector/test rather than carrying partial split registrations.
- The promotion trigger is satisfied: clean p512/d128 improves
  **11.485885 -> 14.533955 tok/s (+26.538%)**, saves **2.318892 seconds** over
  127 calls, and preserves the complete trajectory/lifecycle. Keep the
  explicit serial rollback until a short context-crossover check confirms the
  inherited **127/65/257** thresholds on gfx1151.

## Laguna source-F16 decode fixed-K selector

- Added 2026-07-28 as
  `HIPENGINE_LAGUNA_F16_DECODE=auto|gemv|onebarrier|fixedk`.
  gfx1151 `auto` now selects exact K3072/K6144/K9216 fixed-K single/triple
  siblings for rows==1. Explicit `onebarrier` retains the prior exact generic-K
  default, `gemv` retains the two-barrier root, and peer backends retain GEMV.
- The primitive preserves every output byte and improves all six natural
  roles **0.57-1.71%**, with weighted leaf family
  **31.316 -> 31.097 ms/token (-0.698%)**. If full-state, cached trace, and
  clean p512/d128 gates transfer positively, keep `gemv` only through the
  active decode campaign and then collapse the gfx1151 positive selection
  into the default dispatch. Remove the candidate and selector if any
  production gate regresses.
- The promotion trigger is satisfied: seven same-session p512/d128 pairs move
  **14.758912 -> 14.800191 tok/s (+0.280%)**, every candidate beats every
  control, all generated hashes match, and cached tracing records exactly
  **18,288 = 144 x 127** candidate calls with zero retained decode GEMVs.
  Keep explicit `gemv` through LD-2, then collapse the positive selector as
  described above.
- The fixed-K successor also satisfies promotion: the six-role family moves
  **30.952 -> 24.482 ms/token (-20.90%)**, seven exact production pairs move
  **14.786076 -> 16.391201 tok/s (+10.856%)**, and cache-only tracing records
  **18,288/18,288** fixed-K calls with no fallback. Keep `onebarrier` for one
  release/bisection window; then remove positive `fixedk` selector semantics
  and make it unconditional for qualified gfx1151 K widths. Permanently retain
  the generic registered path for unsupported K/rows/backends.

## Laguna selected natural tile8 decode selector

- Added 2026-07-28 as
  `LagunaGGUFResidentSession(..., use_selected_natural_tile8_decode=False)`
  plus `set_selected_natural_tile8_decode(...)` and
  `--compare-selected-natural-tile8-decode`. False restores the exact
  16-column natural gate/up owner; non-natural shapes and peer backends retain
  the registered generic/natural fallbacks.
- The promotion trigger is satisfied. Actual-weight tile8 improves
  **5.35-7.13%** with zero BF16 mismatches. Seven p512/d128 pairs improve
  **16.991621 -> 17.007001 tok/s (+0.091%)** with 7/7 wins and exact state.
  Cache-only tracing records all **5,969** intended calls, zero fallback, and
  local128/VGPR96/LDS512/scratch0.
- Keep the false rollback and comparison flag through one decode campaign
  checkpoint. Then remove positive selector semantics and make tile8
  unconditional for the qualified gfx1151 shape. Permanently retain the
  natural tile16 and generic registered owners for unsupported
  shapes/backends.

## Laguna gfx1151 SWA exponential decode selector

- Added 2026-07-29 as a default-off resident and category measurement seam for
  raw-native and bounded range-reduction SWA exponentials. Both candidates
  changed only the exponential in saturated 72Q/8KV/D128/SWA512 decode; the
  retained compiler `expf` direct-store body was always the production path.
- Raw native exponential is rejected: seven production pairs improve
  **19.130955 -> 19.309790 tok/s (+0.935%)**, but the complete category gate
  reaches max KL **1.452698** despite **566/576** top-1. Its runtime field,
  setter, and comparison names were removed.
- Bounded accurate-style range reduction is also rejected: seven production
  pairs improve only **19.164777 -> 19.229973 tok/s (+0.340%)**, while the
  complete category gate reaches max KL **1.888082** despite **558/576**
  top-1. This satisfies the cleanup trigger: remove the runtime field, setter,
  both comparison lanes, and the two dead registered primitive/leaf controls.
  Production remains on compiler `expf`; no exponential rollback debt remains.

## Laguna gfx1151 exact SWA exp-domain selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512` field,
  `LagunaGGUFResidentSession.set_decode_swa_assume_exp(...)`, and
  `--compare-swa-assume-exp`. It leaves compiler `expf` intact and only asserts
  the exact score-minus-maximum domain at saturated
  72Q/8KV/D128/SWA512. The retained direct-store body is the rollback.
- Promote only if seven tracked-clean p512/d128 pairs improve and every
  generated ID/state field remains exact. On promotion, make the gfx1151
  capability select the assumed-domain sibling, retain explicit false rollback
  through the decode campaign, then collapse the positive selector after one
  checkpoint. On rejection, remove the field, setter, comparison lane, and
  registered candidate primitive.
- Promotion gate satisfied 2026-07-29: **19.140826 -> 19.245912 tok/s
  (+0.549%, -0.285 ms/token)** with 7/7 wins, complete sample separation, and
  exact IDs/state. gfx1151 now defaults the capability true. Keep explicit
  false rollback through this decode campaign; then remove positive selector
  semantics while retaining the generic registered rollback.

## Laguna gfx1151 mixed32 SWA decode selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512`
  field, `LagunaGGUFResidentSession.set_decode_swa_mixed32(...)`, and
  `--compare-swa-mixed32`. False restores the exact 24-block GQA3
  assumed-domain owner; shorter/non-natural shapes and peer backends retain
  their existing registered routes.
- Promotion gate satisfied: the exact 32-block 2+2+2+3 owner improves the
  leaf **5.41%** and all seven resident p512/d128 pairs
  **19.268862 -> 19.371717 tok/s (+0.534%)** with identical trajectories and
  allocation lifecycle.
- Keep the explicit false rollback through clean production publication and
  the next attention campaign checkpoint. Then remove positive selector
  semantics and make mixed32 unconditional for the qualified gfx1151 shape,
  while permanently retaining the registered 24-block GQA3 and generic
  fallbacks.

## Laguna gfx1151 mixed32 exact exp4 selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512`
  field, `LagunaGGUFResidentSession.set_decode_swa_mixed32_exp4(...)`, and
  `--compare-swa-mixed32-exp4`. False restores the retained serial-exp
  mixed32 sibling; it never selects a different geometry, arithmetic order,
  or non-natural route.
- Promotion gate satisfied: the exact four-lane issue schedule improves the
  leaf **0.091487 -> 0.089135 ms (-2.57%)** and all seven resident p512/d128
  pairs **19.368030 -> 19.432503 tok/s (+0.333%, -0.171 ms/token)** with
  complete sample separation and identical trajectories/state/lifecycle.
- Keep the explicit false rollback through clean production publication and
  the next attention checkpoint. Then collapse the positive selector into the
  qualified gfx1151 mixed32 dispatch while permanently retaining the
  registered serial-exp mixed32 sibling for compiler/codegen bisection.

## Laguna gfx1151 mixed32 exact exp8 selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512`
  field, `LagunaGGUFResidentSession.set_decode_swa_mixed32_exp8(...)`, and
  `--compare-swa-mixed32-exp8`. False restores retained exp4; it changes no
  geometry, arithmetic order, or non-natural route.
- Promotion gate satisfied: the exact eight-lane issue schedule improves the
  leaf **0.089191 -> 0.083755 ms (-6.09%)**, stable cached trace
  **83.557 -> 78.667 us (-5.85%)**, and all seven resident p512/d128 pairs
  **19.427449 -> 19.510986 tok/s (+0.430%, -0.220 ms/token)** with complete
  sample separation and identical trajectories/state/lifecycle.
- Clean selector-unset production is published at **19.515697 tok/s**
  (**+0.470% / -0.241 ms/token** over clean exp4), with exact repeated state.
  Keep the explicit false rollback through the next attention checkpoint.
  Then collapse the positive selector into the qualified gfx1151 mixed32
  dispatch while retaining registered exp4 for compiler/codegen bisection.

## Laguna gfx1151 mixed32 exact exp16 selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512`
  field, `LagunaGGUFResidentSession.set_decode_swa_mixed32_exp16(...)`, and
  `--compare-swa-mixed32-exp16`. False restores retained exp8; it changes no
  geometry, arithmetic order, or non-natural route.
- Promotion gate satisfied: exact sixteen-lane issue improves the leaf
  **0.083740 -> 0.082224 ms (-1.81%)**, stable cached trace
  **78.814 -> 77.265 us (-1.97%)**, and all seven resident p512/d128 pairs
  **19.506557 -> 19.523370 tok/s (+0.0862%, -0.0441 ms/token)** with complete
  sample separation and identical trajectories/state/lifecycle.
- Clean selector-unset production is published at **19.530105 tok/s**
  (**+0.0738% / -0.0378 ms/token** over clean exp8), with exact repeated
  state. Keep the explicit false rollback through the next attention
  checkpoint. Then collapse the positive selector into the qualified gfx1151
  mixed32 dispatch while retaining registered exp8 for compiler/codegen
  bisection.

## Laguna gfx1151 mixed32 exact exp32 selector

- Added 2026-07-29 as the session-scoped
  `LagunaKVCache.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512`
  field, `LagunaGGUFResidentSession.set_decode_swa_mixed32_exp32(...)`, and
  `--compare-swa-mixed32-exp32`. False restores retained exp16.
- Promotion gate satisfied: exact wave32 issue improves the leaf
  **0.082313 -> 0.081551 ms (-0.93%)**, stable trace
  **77.185 -> 76.838 us (-0.45%)**, and all seven resident pairs
  **19.524103 -> 19.538164 tok/s (+0.0720%, -0.0369 ms/token)** with complete
  separation and exact state.
- Clean selector-unset production is aggregate-flat at **19.530839 tok/s**
  (**+0.0038%** versus exp16), so retain on the stronger separated A/B and
  leaf/trace evidence. Keep false rollback through the next wall census.
  Then collapse positive selector semantics while retaining exp16 as the
  compiler/codegen rollback. Wave32 closes the natural issue-width screen.

## Laguna gfx1151 exact global exp-domain selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape`
  field, `LagunaGGUFResidentSession.set_decode_global_assume_exp(...)`, and
  `--compare-global-assume-exp`. It changes no arithmetic and applies only to
  natural gated capacity-4096 global decode through live4000; the retained
  direct-store body is the rollback.
- Promote only if the tracked-clean counterbalanced p512/d128 screen is
  exact and positive. On promotion, default the gfx1151 capability true and
  retain explicit false rollback through the decode campaign. On rejection,
  remove the field, setter, comparison lane, capability, and registered
  primitive rather than carrying a dead global fork.
- Promotion gate satisfied 2026-07-29: all seven pairs improve
  **19.235596 -> 19.243968 tok/s (+0.0435%, -0.0226 ms/token)** with exact
  IDs/state/lifecycle. gfx1151 now defaults the capability true. Keep explicit
  false rollback through this campaign, then collapse positive selector
  semantics while retaining the generic-domain registered rollback.

## Laguna gfx1151 exact global exp32 selector

- Added 2026-07-29 as the session-scoped
  `LagunaKVCache.global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape`
  field, `LagunaGGUFResidentSession.set_decode_global_exp32(...)`, and
  `--compare-global-exp32`. False restores the retained serial-issue
  assume-exp sibling.
- The primitive is byte-exact at live 513/576/639 and improves the leaf
  **2.25%/3.22%/3.79%**. Promote only if cached tracing is spill-free and the
  tracked-clean counterbalanced p512/d128 screen is exact and positive.
- Promotion gate satisfied 2026-07-29: cached tracing improves
  **88.486 -> 85.601 us (-3.26%)** at VGPR56/scratch0, and all seven resident
  pairs improve **19.547209 -> 19.556569 tok/s (+0.0479%)** with exact state.
- After the decode campaign, collapse positive selector semantics while
  retaining the serial-issue primitive as the compiler/codegen rollback.

## Laguna gfx1151 exact global mixed32 selector

- Added 2026-07-29 as the session-scoped
  `LagunaKVCache.global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape`
  field, `LagunaGGUFResidentSession.set_decode_global_mixed32(...)`, and
  `--compare-global-mixed32`. False restores the retained 24-block GQA2
  exp32 owner.
- Promotion gate satisfied: live513/576/639 leaves improve
  **5.19%/8.39%/8.39%**, the cached trace names the intended
  grid8192/local256/VGPR56/scratch0 specialization, and all seven resident
  pairs improve **19.641357 -> 19.668893 tok/s (+0.1402%, -0.0713
  ms/token)** with byte-exact attention and exact generated state.
- Clean selector-unset production is **19.667705 tok/s**, **+0.1917% /
  -0.0975 ms/token** over the preceding packet, with exact repeated state.
- The 40-block ordinary-grid point is rejected at live576/live639 and removed.
  The pair-local256 plus singleton-local128 split-launch screen is also removed
  after exact leaves regress **142.26-154.66%**. Keep the explicit false
  rollback through the next clean attention census and the resulting
  single-launch screen. Then collapse positive selector semantics while
  retaining the 24-block GQA2 exp32 primitive as the exact
  compiler/occupancy rollback.

## Laguna gfx1151 exact global producer-max selector

- Added 2026-07-29 as the default-off session-scoped
  `LagunaKVCache.global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape`
  field, `LagunaGGUFResidentSession.set_decode_global_producer_max(...)`, and
  `--compare-global-producer-max`. False restores the retained global
  mixed32/exp32 owner.
- The primitive is byte-exact at live513/576/639, improves the leaves
  **4.50%/4.89%/4.88%**, removes one barrier, and lowers VGPR **56 -> 48**
  without added LDS or scratch.
- Promote only if seven matched resident p512/d128 pairs preserve exact
  trajectories/state/lifecycle and pass the positive complete-model gate.
  After clean selector-unset publication, remove the comparison flag and
  session setter while retaining the architecture capability and registered
  mixed32/exp32 rollback.
- Promotion gate passed 2026-07-29: all seven pairs improve
  **19.978296 -> 19.993586 tok/s (+0.0765%, -0.0383 ms/token)** with complete
  sample separation and exact state/lifecycle. The gfx1151 capability is now
  default-on.
- Clean selector-unset publication passes at **19.986371 tok/s**, with exact
  repeated state and the capability active. Remove the comparison CLI flag
  and session setter at the next attention-census commit; retain the
  architecture capability, cache field, and registered mixed32/exp32
  rollback.
- **Closed 2026-07-29:** remove `--compare-global-producer-max`, its session
  setter, and the setter-only test after clean publication. The default-on
  gfx1151 capability, fail-closed cache field, registered producer-max
  primitive, and mixed32/exp32 rollback remain.

## Laguna gfx1151 SWA producer-value-tail diagnostic

- Added 2026-07-30 as a separately registered exact mixed40 attention
  primitive. Finished tail probability-producer waves copy the last 64
  staged-V vectors on pair owners and the last 32 on singleton owners,
  balancing the copy loop without changing attention arithmetic.
- The 21x100 leaf improves **3.270%** at unchanged kernel resources, but seven
  resident p512/d128 pairs reject standalone ownership at
  **20.509962 -> 20.507264 tok/s (-0.01316%)**, only **4/7** wins.
- Keep the symbol only through the current attention campaign for a compounded
  producer/denominator/PV scheduling screen. Remove it, its wrapper, registry
  key, leaf selector, and focused test call if no compound candidate uses it;
  never make it the default independently.

## Laguna gfx1151 Q4T16 dual-interleaved expert production rollback

- Originated 2026-07-30 as an unregistered exact leaf primitive and
  actual-weight harness mode. It is now registered and production-owned. It is
  byte-neutral only when it replaces both resident gate/up T16 matrices;
  keeping it as a sidecar would add about **43.76 GB** across 47 routed layers.
- The exact byte-neutral D8 MMQ128x32 prefill consumer now wins M128/M256/M512
  **2.460%/2.034%/1.187%**, so keep the dual-layout host helpers, decode and
  prefill symbols/wrappers, leaf modes, and focused fixtures through resident
  integration. Remove the two ordinary expert gate/up allocations when the
  paired allocation becomes canonical; never add a decode-only production
  sidecar. Remove the temporary comparison selectors and legacy cache
  acceptance after tracked-clean decode plus the prefill sweep publish.
- Resident integration is now default and byte-neutral. The temporary rollback
  is `scripts/laguna_long_context_profile.py --ordinary-q4-expert-t16` plus
  `LagunaGGUFResidentSession(...,
  use_q4_expert_t16_dual_interleaved=False)`. Same-revision production improves
  **22.130173 -> 22.260802 tok/s** with exact state and unchanged
  **79,022,522,196-byte** residency. Tracked-clean publication reaches
  **22.262504 tok/s** decode and **656.990 tok/s** pp512.
- The existing repacked cache still stores ordinary gate and up T16 payloads,
  so default load interleaves them on the host and rises
  **92.084 -> 142.902 seconds**. Retain the rollback until a versioned paired
  cache payload can be written without duplicating the roughly **43.76 GB**
  resident expert pair. After that cache migration and one clean decode/prefill
  recertification, remove the ordinary CLI/session seam and ordinary paired
  materializer fallback.

## Laguna gfx1151 dense/shared dual-interleaved T16 selector

- Added 2026-07-30 as
  `LagunaGGUFResidentSession.set_q4_decode_t16_dual_interleaved(...)` and
  `--compare-q4-decode-t16-dual-interleaved`. The gfx1151 capability defaults
  on; false restores the exact resident-pack8 fused gate/up path.
- The temporary same-resident diagnostic duplicated the old separate sidecars
  only for seven counterbalanced pairs. All seven candidates win
  **21.898558 -> 21.954474 tok/s (+0.25534%)** with exact state.
  Production removes the duplicates: 48 paired payloads replace 96 separate
  payloads at the same **214,597,632-byte** auxiliary residency.
- Remove the comparison CLI and session setter after tracked-clean
  selector-unset publication and the first post-retention wall census. Retain
  the architecture capability, paired materializer contract, registered tile2
  owner, and pack8 exact fallback.

## Laguna gfx1151 selected-Q4 scalar-Q rollback

- Added 2026-07-30 after exact adjacent-column T16 Q-payload reuse improved the
  actual gate/up leaf **1.945%** and all seven p512/d128 model pairs.
  Production keeps the existing
  `selected_dual_t16_natural_tile8_parallel_silu...` variant name; the
  separately registered `...parallel_silu_scalarq...` sibling restores the
  pre-change one-nibble-per-load body exactly.
- Keep scalar-Q through clean selector-unset publication and one later decode
  wall census. Remove its launcher, wrapper, registry key, and comparison-only
  exactness call if pair-Q remains positive and spill-free after that census.
- **Clean publication passed:** selector-unset production improves
  **20.800509 -> 20.830515 tok/s (+0.14426%)** with exact repeated state.
  Keep scalar-Q only through the next decode wall census.

## Laguna gfx1151 dense/shared Q4 T16 sidecar selector

- Added 2026-07-30 as
  `LagunaGGUFResidentSession.set_q4_decode_t16_sidecar(...)` and
  `--compare-q4-decode-t16-sidecar`. The gfx1151 capability defaults on;
  false restores the resident-pack8 fused gate/up plus SiLU path.
- The additive 96-weight sidecar passes exact actual-weight leaves and all
  seven same-resident p512/d128 pairs, improving
  **21.311596 -> 21.852204 tok/s (+2.53669%)**.
- Remove the comparison CLI and session setter after tracked-clean
  selector-unset publication and its first wall census. Retain the
  architecture capability, materialized T16 sidecars, and registered pack8
  fallback.
- Tracked-clean publication passes at **21.851538 tok/s**, with all three
  trajectories exact and the capability active. Remove the comparison seam
  after the pending post-sidecar wall census.
- The post-sidecar census passes at **43.972461 ms/token** kernel sum and
  **46.007636 ms/token** span. The comparison seam is now eligible for
  removal; retain only the capability, T16 materialization, and pack8 fallback.

## Laguna gfx1151 source-F16 attention-quad selector

- Added 2026-07-30 as
  `LagunaGGUFResidentSession.set_f16_attention_quad_decode(...)` and
  `--compare-f16-attention-quad-decode`. The candidate flattens the exact
  fixed-K Q/K/V triple and per-head gate singleton into one grid for c=1;
  rows>1, unsupported layouts, explicit disable, and unmeasured backends
  retain the registered triple-plus-single chain.
- Remove the comparison CLI and session setter after the same-resident
  production gate is either rejected or published from a tracked-clean
  default. If retained, keep the four-axis `linear_quad` registration,
  gfx1151 capability, and unfused fallback.
- The same-resident gate retains the capability at
  **21.944420 -> 22.026384 tok/s (+0.37351%)**, with all seven candidates
  positive and complete state exact. Remove the comparison seam after the
  pending tracked-clean selector-unset publication and wall census.
- Publication and census pass at **22.031913 tok/s** and exactly
  **625 dispatches/token**. The comparison CLI and session setter are removed;
  retain the constructor override, capability, registered quad, and unfused
  fallback.

## Laguna gfx1151 source-F16 projection/head/KV selector

- Added 2026-07-30 as
  `LagunaGGUFResidentSession.set_f16_projection_head_kv_decode(...)` and
  `--compare-f16-projection-head-kv-decode`. The candidate preserves one exact
  fixed-K block per Q/K/V/gate output column and lets the final producer for
  each head run the established RMSNorm/RoPE/BF16-KV body through bounded
  resident completion counters.
- Remove the comparison CLI and session setter after tracked-clean
  selector-unset production and a complete 127-transition census either
  publish or reject the default. If retained, keep the constructor override,
  four-axis composite registrations, gfx1151 capability, counter scratch, and
  exact quad-plus-head/KV fallback.
- The provisional same-resident gate is exact and mechanically positive at
  **22.016010 -> 22.017120 tok/s (+0.00504%)**, saving
  **0.002932 ms/token** by paired median with five of seven wins. Publication
  must confirm 48 composite calls/token, zero separate head/KV calls, and no
  material tracked-clean wall regression before this seam is removed.
- Publication/census passes mechanically: tracked-clean throughput is
  aggregate-flat at **22.007742 tok/s (-0.1097%)**, while tracing confirms
  **625 -> 577 dispatches/token**, zero old quad/head-KV launches,
  **45.699715 -> 45.660100 ms/token** span, and
  **2.006962 -> 1.882766 ms/token** span-minus-kernel time. The comparison
  CLI and session setter are now eligible for immediate removal.
- The comparison CLI, setter, and comparison-protocol fields are removed.

## Laguna gfx1151 source-F16 output/add/RMSNorm selector

- Added 2026-07-30 as
  `LagunaGGUFResidentSession.set_f16_output_add_rmsnorm_decode(...)` and
  `--compare-f16-output-add-rmsnorm-decode`. The candidate preserves every
  fixed-K attention-output projection block and lets only the final producer
  execute the established residual-add/RMSNorm tree.
- Seven exact same-resident p512/d128 pairs all improve
  **22.005296 -> 22.062263 tok/s (+0.25888%)**, saving
  **0.113153 ms/token** by paired median with unchanged residency.
- Remove the comparison CLI, setter, and protocol fields after tracked-clean
  selector-unset production and a complete 127-transition census publish or
  reject the default. If retained, keep the constructor override, four-axis
  composite registration, gfx1151 capability, shared completion scratch, and
  exact fixed-K projection plus add/RMSNorm fallback.
- Publication passes at **22.063262 tok/s (+0.25227%)** and exactly
  **529 dispatches/token**, with kernel span
  **45.660100 -> 45.543776 ms/token**. The comparison CLI, setter, and
  protocol fields are now eligible for immediate removal.
- The comparison CLI, setter, and protocol fields are removed. Retain the
  constructor rollback, capability, registered composite, shared counter
  scratch, and exact unfused fallback.

## Laguna gfx1151 selected-down parallel-weighted selector

- Added 2026-07-30 as
  `LagunaGGUFResidentSession.set_selected_down_natural_parallel_weighted_decode(...)`.
  The candidate preserves all ten route-parallel natural Q4/planar-Q6
  producers and uses tile-local completion counters to own the exact
  slot-order weighted reducer.
- Seven exact same-resident p512/d128 pairs all improve
  **22.071805 -> 22.139076 tok/s (+0.30479%)**, saving
  **0.141227 ms/token** by paired median. Natural Q4/Q6 leaves improve
  **3.940%/3.752%**.
- Remove the session setter after tracked-clean selector-unset publication and
  a complete 127-transition census either publish or reject the default. If
  retained, keep the constructor rollback, architecture capability,
  four-axis composite registrations, bounded counter scratch, and exact
  route-parallel projection plus standalone weighted-sum fallback.
- Publication and census pass at **22.119461 tok/s** and exactly
  **482 dispatches/token**. The session setter is now eligible for immediate
  removal.
- The session setter is removed. Retain only the constructor rollback,
  architecture capability, composite registrations, bounded counter scratch,
  and exact two-launch fallback.

## Laguna gfx1151 shared-down/D9 native host-batch selector

- Added 2026-07-30 as
  `LagunaGGUFResidentSession.set_shared_down_moe_tail_host_batch(...)` and
  `--compare-shared-down-moe-tail-host-batch`. The candidate preserves both
  existing GPU launches and combines only their Python/ctypes host boundary.
- Seven exact same-resident p512/d128 pairs all improve
  **22.146074 -> 22.154405 tok/s (+0.03762%)**, saving
  **0.020358 ms/token** by paired median.
- Remove the comparison CLI, setter, and protocol fields after tracked-clean
  selector-unset production and a cached dispatch census publish or reject
  the default. If retained, keep the constructor rollback, gfx1151 capability,
  four-axis host-batch registrations, native shim, and exact separate-call
  fallback.
- Publication and census pass at **22.141787 tok/s** and exactly **482 model
  kernels/token**. The comparison CLI, setter, and protocol fields are now
  eligible for immediate removal.
- The comparison CLI, setter, and protocol fields are removed. Retain the
  constructor rollback, gfx1151 capability, four-axis host-batch
  registrations, native shim, and exact separate-call fallback.

## Laguna gfx1151 Q4T16 shared-down selector

- Added 2026-07-31 as
  `LagunaGGUFResidentSession.set_q4_shared_down_t16_decode(...)` and
  `--compare-q4-shared-down-t16-decode`. The gfx1151 capability defaults on;
  false restores the exact expanded-pack8 projection while preserving the
  retained native shared-down-to-D9 host batch.
- All seven same-resident p512/d128 candidates win
  **22.377298 -> 22.563488 tok/s (+0.83205%)** with exact trajectory,
  state, residency, and lifecycle. The 24 sidecars add **43,646,976 bytes**.
- Remove the comparison CLI and session setter after tracked-clean
  selector-unset production plus the first cached 127-transition census.
  Retain the constructor rollback, architecture capability, sidecar
  materialization, registered T16 single-output and host-batch keys, and
  expanded-pack8 fallback.
- Tracked-clean publication passes at **22.555437 tok/s** and tracing records
  **24 T16 / zero pack8** shared-down calls per token. The comparison CLI and
  session setter are removed. Retain the constructor rollback, capability,
  sidecars, registered owners, and expanded-pack8 fallback.

## Laguna gfx1151 router-projection wave-0 selector

- Added 2026-07-31 as
  `LagunaGGUFResidentSession.set_router_projection_wave0_tree(...)` and
  `--compare-router-projection-wave0-tree`. The gfx1151 capability defaults
  on; false restores the exact scalar local256 projection while preserving
  the separate correction-bias selector.
- Six of seven same-resident p512/d128 candidates improve
  **22.572873 -> 22.579029 tok/s (+0.02727%)**, saving
  **0.012080 ms/token** by paired median with unchanged trajectory, residency,
  and launch count.
- Remove the comparison CLI and session setter after tracked-clean
  selector-unset production. Retain the constructor rollback, architecture
  capability, registered wave-level owner, and scalar local256 fallback.
- Tracked-clean publication passes at **22.581875 tok/s**. The comparison CLI,
  setter, and protocol fields are removed. Retain only constructor rollback,
  the gfx1151 capability, registered wave-level owner, and scalar fallback.

## Laguna gfx1151 Q4 selected-down paired-coefficient selector

- Added 2026-07-31 as the constructor override
  `use_selected_down_q4_paircoeff_weighted_decode` and direct profiling flag
  `--selected-down-q4-paircoeff-weighted-decode`. The gfx1151 capability
  defaults on; `false` restores the exact scalar-column route-parallel weighted
  Q4 owner.
- The exact actual-weight leaf improves **0.967%**, profiling contracts
  allocated VGPR **104 -> 80**, and seven same-resident p512/d128 pairs improve
  **22.762554 -> 22.793632 tok/s (+0.13653%, 5/7 wins)**.
- Remove the direct profiling flag after tracked-clean selector-unset
  publication. Retain the constructor rollback, gfx1151 capability, both
  four-axis registrations, and exact scalar-column fallback.
- Tracked-clean publication passes at **22.780604 tok/s / 43.896992
  ms/token**. The direct profiling flag is removed. Retain only the constructor
  rollback, gfx1151 capability, both registrations, and exact scalar-column
  fallback.

## Laguna gfx1151 decode shared-stream priority selector

- Added 2026-07-31 as
  `LagunaGGUFResidentSession.set_moe_decode_shared_normal_priority(...)`,
  `--moe-decode-shared-normal-priority`, and the focused
  `scripts/laguna_moe_shared_priority_ab.py` harness. The gfx1151 capability
  defaults on; false routes c=1 back to the retained least-priority prefill
  stream.
- Seven exact same-session p512/d128 pairs all improve
  **22.869628 -> 22.891888 tok/s (+0.09733%)**, saving
  **0.038251 ms/token** by paired median, while both arms keep prefill on the
  least-priority stream.
- Remove the session setter, direct profiling CLI/protocol field, and focused
  one-off harness after tracked-clean selector-unset publication. Retain the
  constructor rollback, gfx1151 capability, separate c=1 stream, and the
  all-low-priority fallback.
- Tracked-clean publication passes at
  **22.891692 tok/s / 43.683971 ms/token**. The setter, direct CLI/protocol
  field, and one-off harness are removed. Retain only the constructor rollback,
  gfx1151 capability, separate c=1 stream, and all-low-priority fallback.

## Laguna gfx1151 selected gate/up halfdot rollback

- Added 2026-07-31 as constructor override
  `use_selected_halfdot_decode`. The gfx1151 capability defaults on; `false`
  restores the exact interleaved pair-coefficient F32 owner without changing
  resident bytes or the surrounding fused SiLU boundary.
- The quality-gated candidate improves the actual layer-1 leaf
  **0.110406 -> 0.090545 ms (-17.989%)** and all seven same-resident p512/d128
  pairs **22.999793 -> 23.084044 tok/s (+0.36631%)**. Recurrent
  candidate-vs-exact quality is max KL **0.008202** and top-1 **93.75%**.
- The temporary session setter and comparison CLI are removed after
  default-on production measures **23.089693 tok/s**. Retain the constructor
  rollback, gfx1151 capability, both four-axis registrations, exact fallback,
  and diagnostic leaf/quality selectors until a later exact path matches or
  exceeds the candidate.

## Laguna gfx1151 long-global mixed32 reducer prototype

- Added 2026-08-01 while constructing LC-D3. The internal-only
  `laguna_global_attention_split_exact_gated_mixed32_vstage64_reduce_kernel`
  established exact exp32 probability caching, local512 idle-wave V64
  prefetch, and paired-query V reuse, but it was superseded before runtime
  integration by the faster GQA6 score + normalization + D32 PV route.
- It has no C wrapper, Python export, registry key, runtime selector, or
  production dispatch and therefore cannot run. Remove the unused body after
  the clean GQA6 production confirmation, when LC-D3's next score-plane
  rewrite reopens this source file. Do not retain it as a rollback; the
  registered generic exact split path is the rollback.

## Laguna gfx1151 LC-D3 long-global geometry and context-split prototypes

- Added 2026-08-02 while screening the second and third LC-D3 milestones. The
  HIP source and leaf harness retain internal D64/V64, D64/V32, local256
  D32/V64, direct-qhead D32, GQA6 score-tile4, and context-split/merge
  variants. The 4,096-token context path is now production-selected on
  gfx1151 only from global layer 32; exact deferred-normalization D32/V64 is
  the earlier-layer and peer-backend fallback.
- Every exact geometry alternative is slower at a mandatory depth. The
  all-layer context-split route is faster but fails the same-state 127-step
  quality gate at maximum KL **0.687034** versus the **0.05** ceiling; an
  8,192-token sibling fails at **0.776134**, and exact sparse BF16-boundary
  repair is slower than the retained GQA6 owner. Only the measured late-four
  scope is admitted, at maximum KL **0.042569/0.007344** and 254/254 top-1
  across two independent prompts.
- Remove rejected geometries, the unused repair kernel/mask calculation, the
  8,192-token experiment, and non-registered symbols after one clean release
  plus the next exact/precision LC-D3 iteration. Preserve the registered
  4,096-token production owner, its gfx1151 minimum-layer capability, compact
  evidence, and same-state quality harness. The exact normalized/deferred
  GQA6 siblings and generic complete-`KVLiveSpans` split route remain required
  rollback/fallback paths.
