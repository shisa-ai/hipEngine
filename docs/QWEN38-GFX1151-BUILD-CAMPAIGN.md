# Qwen3.8-27B gfx1151 Build Campaign

Status: **open 2026-09-02**

This is the implementation successor to the analysis campaigns. It builds the
mechanisms the prior work identified but did not implement:

- [`QWEN38-GFX1151-STRUCTURAL-DIFFERENTIAL-CAMPAIGN.md`](QWEN38-GFX1151-STRUCTURAL-DIFFERENTIAL-CAMPAIGN.md)
  — closed 2026-09-02 with measured blockers; the Z0-Z4 attribution, external
  parity, mechanism bounds, and candidate declarations this campaign builds
  from.
- [`QWEN38-GFX1151-SCALING-CAMPAIGN.md`](QWEN38-GFX1151-SCALING-CAMPAIGN.md) —
  closed W/Y evidence and the retained Y2/Y3 prefill owners B1 transfers.
- [`QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md`](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md)
  — standardized external matrix, routes, and the complete-wall protocol.
- [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) — normative
  strict/production/batch-invariant contracts every retention gate uses.
- [`BENCHMARK.md`](BENCHMARK.md) — evidence policy and anti-gaming rules.

Owner: build loop. Scope: the frozen product key — physical gfx1151 (Ryzen AI
MAX+ 395 / Radeon 8060S), `Qwen3.8-27B` `standard_q4_k_m` (SHA-256
`7e78da5d…c6fe169`), BF16 KV, production profile, the common ten-prompt suite
(SHA-256 `fac920be…1d86084a`), raw greedy, no prompt cache, declared D24
serving scope unless a unit measures another scope.

## 1. Why this campaign exists

The structural campaign closed with measured blockers instead of
implementations. Its own reviewer then found a contradiction in the closeout
premise, the human authorized building (including T3 depth policy), and the
entry conditions changed. Everything below is grounded in committed artifacts;
derived values are labeled.

### 1.1 The double-counted pass budget (reviewer finding, 2026-09-02)

Z0's pass-budget artifact measured a 526.7-714.4 ms/cycle "non-target stage"
at C5-C8 and concluded target-owner work had a **zero feasible pass budget**
(measured: [`z0-pass-budgets`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z0-pass-budgets.json)).
Two weeks later, the M3 attribution
([`c8`](../benchmarks/results/2026-09-02-gfx1151-qwen38-z4-m3-c8-accept-boundary-attribution.json),
[`wide`](../benchmarks/results/2026-09-02-gfx1151-qwen38-z4-m3-wide-accept-boundary-closure.json))
measured that the dominant component of that stage — the 490.8-667.1 ms
`accept_ms` window — is **96.21%+ traced target-kernel execution**, with only
1.31-2.18% genuine host residue. The budget therefore counted target execution
as non-target cost and then concluded there was no room to reduce target cost.

**Derived, not measured:** reassigning the accept window's kernel time to the
target stage leaves roughly 160 ms true host cost at C8; an A+B cycle at the
F3 prefill-owner anchor (~230-260 ms target pass versus today's ~790 ms) lands
near ~480 ms/cycle versus today's ~789 — a ~35-40% C8 cycle-wall cut, plausible
parity range. B0 re-derives these budgets properly; B1 measures them.

### 1.2 Entry state (measured, from the structural campaign closeout)

| Cell | hipEngine | External leader | Gap |
| --- | ---: | ---: | ---: |
| Prefill C2 | 178.660 tok/s | Laurent 196.824 | -9.23% |
| Prefill C8 | 283.540 tok/s | Laurent 297.325 | -4.64% |
| MTP C8 | 43.421 tok/s K1 | Stock HIP 56.222 K3 | -22.8% |
| MTP C7 | 33.106 tok/s K3 | Stock HIP 46.084 | -28.2% |
| MTP C5 | 27.980 tok/s K3 | Mainline Vulkan 32.713 | -14.5% |
| MTP C2 | 28.441 tok/s K3 | Laurent 32.221 | -11.7% |
| MTP C1 | 15.753 tok/s K3 | Laurent 21.126 | -25.4% |
| MTP C6 (control) | 37.074 tok/s K1 | Laurent 37.154 | -0.22% |

F3's measured anchor (structural §7.1): the verify R32 target pass costs
~790 ms while the already-retained exact Y2/Y3 prefill owners execute the same
GEMMs on the same T16 tensors at ~230-260 ms — a 3-9x per-family gap (Q6
515.6 vs ~50+lm-head ms; Q5 `ssm_out` 104.7 vs ~30 ms; Q4 137.8 vs
~110-130 ms).

### 1.3 Build menu and bounds

| Step | Mechanism | Class | Measured/derived bound | Status |
| --- | --- | --- | ---: | --- |
| B1 | A+B verifier owner transfer + Q6 lm-head one-sweep | T0 registry / T2 | ~35-40% C8 cycle (derived from measured F3 + M3); parity plausible | blocked only by the refuted B0 premise |
| B2 | P1 sole-T16 input-F16 Q4/Q5 family | T1 | 19.8%/20.1% C2/C8 prefill wall (derived from Laurent's measured quant delta) | blocked on missing kernel family — build it |
| B3 | M1 C1 request-owned shadow-session lifecycle | T2 | 41.76% C1 complete-suite (measured screen) | blocked on missing lifecycle ABI — build it |
| B4 | M2 C2 draft depth K3→K2/K1 | T3 | 17.79% C2 (derived) | **authorized 2026-09-02**, incl. automatic promotion via full packet |
| B5 | E integer MMQ M=17-48 | T2 | stock HIP 56.222 tok/s existence anchor | conditional on B1's measured pass |

### 1.4 Authorization and policy deltas from the structural campaign

- **M2 depth policy is authorized (human, 2026-09-02).** K3→K2/K1 at C2 runs
  as an explicit experiment; promotion — including automatic default —
  requires the complete T3 route/economics/lifecycle packet, complete
  same-suite non-regression, category/heldout plus heldout split, and a true
  same-protocol AR control. The structural campaign's "unauthorized" lock is
  superseded by this entry, not edited in place.
- **Retention bar (human, 2026-09-02).** Retain any candidate that is
  non-regressive on the **complete same-suite wall** and passes its full
  applicable profile gate with a registered strict fallback. External parity
  is a stretch goal, not a retention requirement.
- Everything else in structural §3 carries over unchanged: exact
  control/ownership in every profile; T1/T2 eligible under the complete
  production gate; strict remains the oracle; production quality binding;
  no relabeling defects; anti-gaming absolute.

## 2. Punchlist

Order is fixed: B0 → B1 → B2 → B3 → B4 → B5. Every item closes with a
retained result or a measured, named blocker with exact command, physical
host, model/prompt provenance, execution profile, and compact artifact. Retain
per §1.4; commit each validated unit atomically with its worklog entry.

### B0 — pass-budget accounting correction and mechanism-A reopening

- [x] Re-derive the C5-C8 one-group K3 pass budgets reassigning the
  accept-window traced target-kernel time (96.21% C8 per M3 attribution) to
  the target stage. Publish a compact artifact with per-cell corrected
  non-target stage, residual, and the A+B target-pass budget at today's ~790
  ms F3 pass and the ~230-260 ms prefill-owner anchor (anchor labeled
  derived). All inputs must come from committed Z0/M3/F3 artifacts; no new GPU
  run is required.

  Corrected budgets: [`2026-09-02-gfx1151-qwen38-b0-corrected-pass-budgets.json`](../benchmarks/results/2026-09-02-gfx1151-qwen38-b0-corrected-pass-budgets.json)
  (generated by `scripts/qwen38_b0_corrected_budgets.py`, no GPU run).
  Reassigned target-kernel execution hidden in the Z0 non-target stage is
  516.3/580.8/672.5/715.3 ms/cycle at C5/C6/C7/C8 (C6 coverage interpolated
  from C5/C7, labeled derived). Corrected target-pass budgets residual
  removed/kept: C5 346.7/278.8, C6 344.6/264.0, C7 289.4/204.5, C8
  227.1/152.8 ms against the derived F3 owner anchor 230-260 ms. Mechanism
  A+B entry is positive at C5/C6/C7 (residual removed) and C5/C6 (residual
  kept); C8 is a near-miss needing owners at or below ~227 ms. With today's
  ~790 ms owners no cell reaches parity even at zero boundary host cost.
- [x] Amend the structural campaign doc with a dated decision entry reopening
  the mechanism A/B entry conditions and linking the corrected budget
  artifact. Do not edit committed artifacts or immutable worklog entries; the
  correction is additive.

  Appended structural campaign §9 "Decision log (post-close)" entry dated
  2026-09-02: names the double-count, links the corrected artifact and the
  successor build campaign, and supersedes only the Z0 entry-condition
  decision. The committed Z0 artifact and worklog entries are unchanged.

### B1 — mechanism A+B: verifier-side owner transfer (T0/T2)

- [x] Map every verifier R17-R32 packed subshape (including mixed
  R20/R24/R32) to the retained exact Y2/Y3 owner bodies
  (`<3,1,2>` Q6 standard/planar, Q5 one-sweep route, best Q4 rows17-48 owner)
  versus the pre-Y2 `shared4` bodies currently selected by
  `GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS`
  (`hipengine/kernels/hip_gfx1151/__init__.py:1705`). Record the four-axis
  keys; no backend/quant dispatch branches.

  Owner map: [`2026-09-02-gfx1151-qwen38-b1-verifier-owner-map.json`](../benchmarks/results/2026-09-02-gfx1151-qwen38-b1-verifier-owner-map.json)
  (`scripts/qwen38_b1_verifier_owner_map.py`, host-only; sha-pinned to the
  committed M3 C5/C7 raw telemetry + rocprof traces, launch-attributed via
  hipLaunchKernel correlation IDs, reconciling with the M3 wide closure
  within 1.65%). Measured current owners at R20/R28 passes: Q6 planar
  per-row direct GEMV 209.4/292.7 ms (39-42%), Q4 generic wmma-prefill
  bodies ~132+24 ms, Q6 std GEMV 90.0/125.6 ms, Q5 selected-direct
  67.3/92.9 ms, lm-head rowtile re-sweep 23.9/33.4 ms; totals 540.6/702.0
  ms. No retained Y2/Y3 owner is selected by the verifier today. Derived
  A+B projection at R28 is ~265 ms (F3 per-tensor leaves), inside the B0
  corrected C7 budget of 289.4 ms. Transfer surface: the four quant-family
  routers at rows 17-48 plus the default-off wide-q6 table hook; four-axis
  keys recorded per family in the artifact.
- [~] Register the transfer under verifier keys with the current owners as
  registered strict fallback, including B: the Q6 lm-head as one sweep at
  R > 8 (replacing the per-rowtile re-sweep, F4). RED-first where practical:
  strict exact/parent-parity tests per transferred owner on R17-R32 shapes
  before the routing default flips.

  A-transfer registered (default-off, 2026-09-02): root cause of the GEMV
  verifier was the hard constant `_MTP_SERVING_TARGET_USE_WMMA_PREFILL =
  False` (9cceedbcc, a July small-B perf decision on pre-Y2 bodies). It is
  now the env-gated `mtp_serving_target_use_wmma_prefill()` switch
  (`HIPENGINE_GGUF_MTP_SERVING_TARGET_WMMA_PREFILL`, default off = current
  per-row GEMV owners remain the strict fallback) consumed by all five MTP
  serving verify-job sites (four in `qwen35_gguf.py`, packed-batch job in
  `qwen35_gguf_mtp2.py`). No dispatch branch, no kernel signature, and no
  registry key changed: enabling routes verify rows>1 through the already
  registered `t16_wmma_prefill_bf16_bf16_out` four-axis variants — the same
  retained exact band routers prefill uses. RED coverage:
  `tests/test_mtp_serving_target_wmma_transfer.py` (12 tests: default-off
  fallback, env values, both serving modules consume the switch, prefill
  band variants registered for all four verifier quants on hip_gfx1151).
  Ledger entry RF-B1A in `docs/REFACTOR.md`.

  Remaining for this item (B-full): the one-sweep lm-head at R>8. The batched
  verifier lm-head already amortizes the head read via exact rowtile chunks
  capped at the primitive's 2-8 row band
  (`_verify_lm_head_rowtile_chunked`, `HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK`
  in [2,8]); one sweep at R20-R32 needs the rowtile primitive widened or a
  WMMA lm-head body with its own exactness RED — a named kernel unit before
  the B1 item-4 retention measurement.

  B-full feasibility bound (2026-09-02): widening the exact planar rowtile
  template is structurally infeasible — `float acc[ROW_TILE][kTileCols]`
  in `q6_k_t16_qmicro_planar_gemv_rowtile_col8_kernel` needs 128 accumulator
  VGPRs at ROW_TILE=16 and 256 at 32 (RDNA3 caps at 256 total per thread),
  so a one-sweep body must be a new design (WMMA sweep or split-row
  accumulation), not a template widening. Measured share after the A
  transfer: 23.9-33.4 ms per wide pass, ~7.1 cycles/cell → an estimated
  2.4-3.2% of one-group complete wall (derived). B1's measured retention
  stands without it; the B-full kernel unit remains open here with this
  named prerequisite and is deprioritized behind B2 (P1 prefill bound
  19.8/20.1% vs B-full ~2-3%).
- [x] `rocprofv3 --kernel-trace` smoke (prebuilt `.so`) confirming the new
  owners execute under expected names with plausible durations at R20/R24/R32.

  Smoke artifact: [`2026-09-02-gfx1151-qwen38-b1-transfer-smoke.json`](../benchmarks/results/2026-09-02-gfx1151-qwen38-b1-transfer-smoke.json)
  (driver `/tmp/q38-b1-run/driver.py`, analysis
  `scripts/qwen38_b1_transfer_smoke.py`; two arms + rocprof'd env-on arm on
  physical host `gfx1151`, production profile, single canonical prompt,
  widths 5/7, K3/D24, cached build — no kernel code changed so the pinned JIT
  cache stayed valid). Correctness: `ar_self_exact`, `mtp_self_exact`, and
  `ar_mtp_equal` pass on every arm, and generated IDs are identical env-off
  vs env-on at both widths. Owner routing (launch-attributed, R28 pass):
  wmma-band owners 210.0 ms of 264.6 ms total; per-row GEMV owners 0 ms.
  Measured pass medians: R20 540.6 → 245.9 ms, R28 702.0 → 264.6 ms versus
  the B1 owner map. Warm-arm diagnostic walls (single prompt, no claim):
  C5 MTP 21.381 → 34.560 tok/s (+61.6%), C7 23.545 → 42.837 (+81.9%). Q4
  (~132+24 ms) and the lm-head rowtile re-sweep (33.4 ms) are unchanged, as
  expected before B-full.
- [x] Measure one-pass K3/R32 and K2/R24 at C5-C8 on the full ten-prompt suite
  under the production profile (exact commands, host identity, manifest
  hashes): complete wall, target-pass kernel time, corrected cycle economics
  versus the B0 budget. Retain only if complete same-suite wall is
  non-regressive and the applicable gates pass; update
  `benchmarks/README.md`/`CHANGELOG.md` and a compact artifact only on a
  retained public number. Strict fallback must remain selectable.

  Measurement complete 2026-09-02 (retention packet:
  [`...b1-transfer-full-suite.json`](../benchmarks/results/2026-09-02-gfx1151-qwen38-b1-transfer-full-suite.json),
  [`...b1-transfer-logits-equality.json`](../benchmarks/results/2026-09-02-gfx1151-qwen38-b1-transfer-logits-equality.json);
  seven arms on physical host `gfx1151`, production profile, ten-prompt
  suite, cached build, `HIPENGINE_GGUF_MTP_SERVING_TARGET_WMMA_PREFILL=1`
  on-arms):

  - **One-group protocol (the D24 declared scope, Z0 budget protocol):**
    complete-suite MTP means 22.476→35.135 (+56.3%) at C5, 23.568→38.651
    (+64.0%) at C6, 24.215→41.032 (+69.5%) at C7, 25.983→44.870 (+72.7%)
    at C8; 40/40 cells exact; every generated ID identical off-vs-on;
    MTP/own-AR ratio 0.54-0.63 → 0.96-0.99. K2 arm: 32.7/36.8/39.1/43.2 —
    K3 wins at every width with the new owners. Versus the entry-table
    leaders: C5 now **leads** mainline Vulkan (+7.4%), C6 leads Laurent
    (+4.0%), C7/C8 remain behind stock HIP (−11.0%/−20.2%).
  - **Production-admission suite: measured inert** (−0.1% to +0.0%).
    Structural, not noise: admission caps groups at 4, so verify passes are
    R2-R16 (histogram in the artifact) and the rows 17-48 transfer never
    executes on that path. C5's pre-existing exactness anomaly reproduces
    identically on both arms (not caused by the transfer).
  - **§6 teacher-forced logits gate:** 336 full-vocabulary rows across
    widths 5/7/8 plus a mixed-category offset window: top-1 agreement 100%,
    mean/p95/max KL 2.8e-5-5.7e-5 / 2.0e-4-3.0e-4 / 4.9e-4-6.5e-4 — 17-77x
    inside every binding envelope threshold; no row above the 2e-2 review
    line. Not bit-identical (T1-class body swap), consistent with identical
    free-running IDs everywhere.
  - **Determinism:** deterministic repeat arm IDs identical; smoke rep
    identical; in-process A/B logits stable.

  Retention executed 2026-09-02 (commit c27abc15d): the switch is now
  profile-scoped and **default-on for the production execution profile**;
  strict and any profile fallback keep the GEMV verifier oracle and
  manifest unchanged. Confirmed on the default path with no env set:
  one-group C5-C8 34.969/38.425/41.615/47.642 tok/s (matches the env-on
  arm), production-admission C7 spot 34.049 (inert as measured), and a
  strict-profile C7 spot at 20.317 tok/s with 10/10 exact running the
  unchanged GEMV chain. The env remains an explicit override for bisection
  (`1` forces the transfer on any profile, `0` restores the GEMV owners).
  Retained public numbers and the rollup row are recorded in
  `benchmarks/README.md`/`CHANGELOG.md` with this artifact as source.

### B2 — P1: sole-T16 input-F16 Q4/Q5 kernel family (T1)

- [ ] RED full-output numerics first: fixtures at rows72/288 and all
  production shapes for input-F16 Q4/Q5 T16 matmul versus the current BF16
  owners (strict oracle `kernels/cpu_reference/` where applicable). Tests fail
  before implementation.
- [ ] Implement the input-F16 siblings as new four-axis variants under
  `hipengine/kernels/hip_gfx1100/quant/` (shared gfx11 bodies, peer-registered
  on gfx1151 per `docs/KERNELS.md`): BF16→F16 activation cast
  workspace/ownership, unchanged weight representation, raw device-pointer
  kernel signatures, current BF16 owners as registered strict fallback.
- [ ] Profile the expected kernels (prebuilt `.so`, `rocprofv3`), then run the
  complete C2/C8 same-suite prefill gates and the applicable production
  numerical gate (strict-teacher mean/p95/p99/max KL and top-1 by
  category/shape/transition, deterministic repeat, isolation, BF16-relative
  and task gates as applicable). Retain per §1.4.

### B3 — M1: request-owned C1 shadow-session lifecycle (T2)

- [ ] RED-first lifecycle contract before adapter changes: request-owned
  shadow-row ABI covering second target session, provider checkpoint, hidden
  row, KV/recurrent state, commit owner, cancellation, compaction, and
  teardown — with failing tests pinning each ownership surface. The shadow is
  input-independent physical padding for any admitted C1 request; no
  prompt/token/candidate branch is allowed.
- [ ] Implement the C1 shadow-row route on the qualified physical C2
  production path; publish one row, discard/reclaim the shadow. Current C1
  route remains the registered strict fallback.
- [ ] Full T2 production packet: full-logit strict-teacher gates, lifecycle,
  isolation, deterministic repeat, category/heldout, task gates, and the
  complete C1 suite wall versus current C1 (bound: 41.76% screen, acceptance
  15.33% C1 versus 78.89% C2). Retain per §1.4.

### B4 — M2: C2 draft-depth K3→K2/K1 (T3, authorized)

- [ ] Screen K2 and K1 at C2 under an explicit experiment configuration on
  the full suite: complete wall, acceptance, cycle count, proposal/target
  trade, versus K3 current-head. True same-protocol AR control included.
- [ ] If complete wall improves: assemble the complete T3
  route/economics/lifecycle packet (per §1.4 authorization) —
  category/heldout splits, deterministic repeats, isolation, lifecycle,
  economics — and promote (including automatic default) only if every gate
  passes and same-suite wall is non-regressive. Otherwise record the measured
  blocker with the artifact.

### B5 — E: integer MMQ for M=17-48 (T2, conditional)

- [ ] Only after B1's measured target-pass result: extend/screen the gfx1151
  `mmq128x32`/`mmq64x64` bodies below rows512 on the three Q6 shapes,
  `ssm_out`, and the two binding Q4 shapes, **against the new A owners**, not
  `selected-wmma`. Proceed to registration and full gates only if the screen
  beats A's measured pass; otherwise record the measured bound and stop.

## 3. Dead ends (measured; do not spend on them)

- Accept-window/host-dataflow cleanup: M3 measured 1.31-2.18% complete-wall
  ceilings against 18.1-35.2% entry thresholds.
- Scheduling/co-scheduling: W3 measured null.
- Micro-tuning the current verify owners: F3 shows them 3-9x off the prefill
  owners on identical GEMMs. Replace, don't polish.
- Launch-count reduction: Laurent uses more dispatches (3,588 vs 1,955).
- Prompt chunking: hipEngine already owns one grouped tick (Z0 tick fact).
- Isolated owner morphology without a new measured premise (structural §2).

## 4. Campaign close criteria

Close when every punchlist item is `[x]` or `[~]` with a durable measured
disposition: a retained validated mechanism, a measured bound that proves
further work unjustified, or a named blocker with its prerequisite. Automatic
serving remains width-4 fail-closed (K0) throughout unless a complete
route/economics/lifecycle packet admits a change. No performance claim is
published without the full evidence policy of
[`BENCHMARK.md`](BENCHMARK.md).

## 5. Evidence map

- Structural campaign artifacts: see
  [`QWEN38-GFX1151-STRUCTURAL-DIFFERENTIAL-CAMPAIGN.md`](QWEN38-GFX1151-STRUCTURAL-DIFFERENTIAL-CAMPAIGN.md)
  §8 and its closeout
  [`2026-09-02-gfx1151-qwen38-structural-differential-closeout.json`](../benchmarks/results/2026-09-02-gfx1151-qwen38-structural-differential-closeout.json).
- Retained Y2 owners:
  [`y2-q6-shared3r1-retained`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q6-shared3r1-retained.json);
  Y3: [`y3-planar-q6-pair-decode-retained`](../benchmarks/results/2026-09-02-gfx1151-qwen38-y3-planar-q6-pair-decode-retained.json).
- M1 screen: [`z4-m1-c1-shadow-screen`](../benchmarks/results/2026-09-02-gfx1151-qwen38-z4-m1-c1-shadow-screen.json).
- P1 blocker spec: [`z4-p1-activation-f16-blocker`](../benchmarks/results/2026-09-02-gfx1151-qwen38-z4-p1-activation-f16-blocker.json).
- M3 attribution: [`z4-m3-c8`](../benchmarks/results/2026-09-02-gfx1151-qwen38-z4-m3-c8-accept-boundary-attribution.json),
  [`z4-m3-wide`](../benchmarks/results/2026-09-02-gfx1151-qwen38-z4-m3-wide-accept-boundary-closure.json).
- Candidate declarations and RED gate:
  [`z3-candidate-declarations`](../benchmarks/results/2026-09-02-gfx1151-qwen38-z3-candidate-declarations.json),
  `scripts/qwen38_z3_candidate_gate.py`.
- W0 row curve / stage telemetry:
  [`w0-sweep-economics`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w0-sweep-economics.json).
