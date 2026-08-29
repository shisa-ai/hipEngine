# Qwen3.8-27B gfx1151 External-Parity Campaign (Umbrella)

Status: active. Opened 2026-08-28. Owner: parity punchlist loop
(`docs/QWEN38-GFX1151-PARITY-CAMPAIGN.md`, lane `parity`).

## 1. Goal

On the physical gfx1151 host (AMD Ryzen AI MAX+ 395 w/ Radeon 8060S), for
`Qwen3.8-27B` `standard_q4_k_m` (sha256 `7e78da5d…c6fe169`) under the frozen
common-suite protocol, **match or beat the best external engine row for
prefill, AR decode, and built-in MTP K3 at every concurrency C1-C8**, or close
the cell with a measured, named blocker. Scope decision 1A (2026-08-28):
standard `Q4_K_M` first; FP4/ROCmFPX, Unsloth UD formats, and ngram replay are
out of scope for parity targets. **DFlash2 is in scope as a revisit item** by
lead decision: other engines leverage it effectively; if hipEngine cannot, the
named blocker is expected to coincide with a weakness whose removal also lifts
our MTP (the multi-row verify amortization wall is the shared suspect).

Campaign method per order decision 2A: umbrella plan → source/route
attribution → short-prompt prefill → AR gaps → MTP → DFlash2 → closure.

## 2. Frozen comparators (2026-08-28 survey artifact)

Source: [`2026-08-28-gfx1151-qwen38-external-reproduction-survey.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-external-reproduction-survey.json),
`standardized_q4_k_m_comparison`, complete-wall tok/s (higher is better),
greedy, no prompt cache, 10 common-suite prompts, 24 output tokens per request
(prefill arms: 1). Frozen rows and commits:

- hipEngine `a9b801d5` (HIP/gfx1151)
- stock mainline Vulkan `4e97ac86`
- stock HIP `9d57ce45`
- LaurentZuijdwijk Vulkan `c28d538d`
- Nathanw Vulkan `0eb52805`
- q38rocm normal-MTP Vulkan `5d097740`

Frozen best-external target per cell (complete-wall tok/s) and hipEngine's
required uplift:

| Phase | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill winner | Nathan 138.95 | Laurent 194.07 | stock HIP 180.09 | stock HIP 192.54 | stock HIP 217.39 | Laurent 243.52 | Laurent 245.61 | Laurent 296.82 |
| hipEngine prefill | 71.55 | 74.87 | 104.51 | 124.35 | 141.45 | 152.20 | 162.08 | 169.96 |
| Required uplift | +94.3% | +159.2% | +72.4% | +54.8% | +53.7% | +60.0% | +51.5% | +74.6% |
| AR winner | Nathan 11.34 | Laurent 20.06 | hipEngine 20.73 | hipEngine 26.22 | hipEngine 31.19 | hipEngine 34.56 | hipEngine 36.59 | Laurent 45.75 |
| hipEngine AR | 9.64 | 14.25 | 20.73 | 26.22 | 31.19 | 34.56 | 36.59 | 39.06 |
| Required uplift | +17.6% | +40.8% | lead | lead | lead | lead | lead | +17.1% |
| MTP K3 winner | Laurent 21.28 | Laurent 32.38 | Laurent 27.52 | mainline 27.02 | mainline 32.74 | Laurent 36.02 | Laurent 42.30 | stock HIP 54.83 |
| hipEngine MTP K3 | 7.07 | 17.40 | 20.07 | 16.51 | 12.97 | 16.85 | 17.66 | 16.07 |
| Required uplift | +200.8% | +86.1% | +37.1% | +63.6% | +152.5% | +113.8% | +139.5% | +241.2% |

Standing context that shapes the plan:

- hipEngine isolated gfx1151 prefill on **this parity model** is 402 tok/s
  @512 / ~87 tok/s @45 (forced-bulk owners, 2026-08-29 attribution), while the
  server matrix shows 71.5 tok/s at C1 — the first prefill wall was suspected
  to be serving-path ownership, but **P1.3 measurement disproved that**: the
  C1 request wall equals the isolated packed-prefill wall and is GPU-bound in
  the small-row T16 wmma prefill GEMM family (see the corrected table below).
  Note: the ~1294 tok/s @512 LCP row cited at campaign open is a different
  model (Qwen3.6-35B-A3B) and never applied to Qwen3.8-27B.
- hipEngine AR is at parity with stock llama.cpp implied-bandwidth on the
  identical quant (12.332 vs 12.27 tok/s, survey renormalization); our MTP
  cycle efficiency is 47-54% vs external 70-77% at matched K3.
- DFlash2 loss attribution was corrected 2026-08-22 (`docs/DFLASH.md`):
  DFlash2 B3 is at **acceptance parity** (2.80 vs 2.85 accepted tokens/cycle);
  the deficit is cost — ~96 ms/cycle drafter+select vs MTP's 2.4 ms proposal,
  and 166 ms/4-row verify with a `_PACK8_ROWTILE_MAX_ROWS = 4` admission cliff
  (the reverted rowtile-8 halving 620→310 ms was AR-divergent and never
  root-caused). This is the same multi-row amortization wall the MTP campaign's
  E2 attacks, and it gates the T3 adaptive-K / B4-clamp reopen.
- E1b current state (2026-08-29): strict C1/K3 natural25 remains automatic at
  **18.191 vs 11.062 tok/s (1.6445x)**. Production C2/K3 is now an exact
  explicit diagnostic at **25.749 vs 17.986 (1.4316x)** and remains automatic
  K0 pending refreshed evidence. E1a+E1b+E2 lift C3/K3 to **29.564 vs 24.042
  (1.2297x)**, 7.45% above the frozen external row. Exact Q6 R8+R8 closes C4
  external parity at **27.450 tok/s (+1.61%)**. C3/C4 remain automatic K0
  pending their full production/serving gates.

## 3. Punchlist

Rules: every cell closes either at `>= frozen winner` under the frozen
protocol with the applicable correctness gate, or with a **measured, named
blocker** recorded in this doc and the unit worklog entry (`[~]`). All perf
claims follow the AGENTS.md evidence policy and anti-gaming rules (full-suite
validation, no single-prompt tuning). Kernel/math changes carry their
`docs/EXECUTION-PROFILES.md` gates. Commit each completed unit immediately.

### P0 — umbrella plan and frozen comparators

- [x] P0.1 Publish this doc; freeze the comparator matrix and required-uplift
  table; commit with the campaign worklog entry and `docs/PLAN.md` pointer.

### P1 — attribution: engine-only deltas under a flag/commit-controlled protocol

- [x] P1.1 Protocol parity audit — done 2026-08-29
  ([`artifact`](../benchmarks/results/2026-08-29-parity-p1-protocol-attribution.json)).
  All six frozen rows ran **identical flags** (`-b 2048 -ub 512 --no-mmap -c
  8192 -np 8`); only device selectors differ. The mainline-Vulkan prefill
  deficit (514 vs ~323-332 ms/request C1) is code/backend, not config. Suite
  prompts are 35-67 tokens, so C1-C8 prefill rows measure **fixed per-request
  serving cost** (GPU floor ~66 ms/prompt); hipEngine sits at ~628
  ms/request vs the ~323-332 ms winner cluster.
- [x] P1.2 Commit bisect — closed 2026-08-29 by pinned-diff classification +
  stock-HIP control (same artifact); full rebuild bisect declined as
  not-evidence-bearing. Every fork delta active on standard `Q4_K_M` is
  Vulkan coopmat matmul tuning (LDS pad/bank conflicts/wave32/f16 operands,
  GDN concat-transpose tiling) — Vulkan catching up to HIP. Stock HIP reaches
  the prefill winner cluster unforked (within 1.6-12.8% per cell), so **no
  fork kernel port is required for hipEngine parity**; port candidates park
  under P2.3 unless a below-serving-boundary wall is proven.
- [x] P1.3 hipEngine serving-path attribution — closed 2026-08-29
  ([`artifact`](../benchmarks/results/2026-08-29-parity-p13-c1-prefill-attribution.json)).
  Per C1 request (~45-token prompt): total 628 ms = packed-route prefill
  ~534 ms **GPU-bound** (rocprof: 1380 launches, 534/560 ms busy; top kernel
  `gguf_q4_t16_dense_wmma_prefill_shared_b` 360.6 ms at M=45, ~20x
  bandwidth-ideal) + packed-vs-bulk route overhead ~111 ms + serving stack
  0-30 ms. Serving/ownership is NOT the wall; the small-row T16 wmma prefill
  GEMM family and the route overhead are. Queues (1 vs 2) are neutral; the
  published 1294 tok/s row was the wrong model (Qwen3.6-35B-A3B).

### P2 — prefill C1-C8 parity

P1.3 decomposition per C1 request: ~534 ms GPU-bound prefill (small-row T16
wmma GEMMs) + ~111 ms route overhead + ~30 ms serving; winner cluster total
~323-332 ms.

- [x] P2.1 Close the route + small-M GEMM walls; target C1 complete-wall
  `>= 138.95` tok/s with the exact/correctness gate green. Route small slabs
  through the best owners (kill the ~111 ms packed-vs-bulk overhead), then
  lift the small-M T16 wmma prefill family (360.6 ms at M=45; target
  `<= ~130 ms` via N-split/split-K workgroup partitioning or low-M tile
  variants) under the strict/production gates.
  - Progress 2026-08-29 (second retained unit): arrival-aware solo batch
    dispatch (2 ms slice idle-solo; full window for ≥2 queued or busy) —
    cumulative server prefill C1 71.55→93.26 (+30.4%), C4 124.35→137.84
    (+10.8%), exact outputs
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-solo-dispatch-window-retained.json)).
  - Progress 2026-08-29 (third retained unit): reclaim-path `zero_states`
    template-scan removal (direct memsets, identical bytes) — cumulative
    C1 71.55→102.21 (+42.8%), C2 74.87→97.18 (+29.8%), exact outputs
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-zerostates-reclaim-retained.json)).
  - Closure 2026-08-29 (seventh retained unit): current telemetry falsified
    the uniform host-wall hypothesis—35-48-token requests were 285-291 ms,
    but rows60/67 were 449/596 ms. General (not prompt-specific) rows49-64
    and rows65-80 Q4/Q5/Q6 low-VGPR bands cut isolated rows60/67 by
    24.2%/31.9%. Frozen C1 reaches **147.11 tok/s**, +105.6% cumulative and
    5.9% above the 138.95 comparator, with all ten exact cells green
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-row49-80-prefill-parity-retained.json)).
- [~] P2.2 C2-C8 prefill parity at the frozen protocol; each cell closes at
  its frozen winner (194.07/180.09/192.54/217.39/243.52/245.61/296.82) or a
  measured named blocker. Slab rows scale with width, so the small-M fix
  carries part of C2-C4; the rest is multi-row route efficiency.
  - Re-freeze 2026-08-29: C1-C8 is now
    **146.81/139.77/153.39/174.64/199.28/214.90/226.75/228.38 tok/s**,
    all 80 cells exact. Every width improves 34.4-105.2% from frozen
    hipEngine; only C1 is at parity. Remaining gaps are C2-C8
    **28.0/14.8/9.3/8.3/11.8/7.7/23.1%**. Largest deficits are C2 and C8;
    profile their current packed prefill owners before extending row bands
    above 80
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-prefill-c1c8-refreeze.json)).
  - Attribution 2026-08-29: fully grouped C2 rows134 and C8 rows536 are
    98.4%/99.3% GPU-bound; Q4 owners consume 59.4%/59.8% of kernel sum, with
    Q6 planar second. Batcher arrivals are already within 0.72/3.18 ms; a
    10 ms solo slice regresses C1/C2 2.5%/2.2% for only +1.0% C8 and is
    rejected. Next screen existing Q4/Q5/Q6 owners at rows96-536; retain the
    2 ms slice
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-prefill-c2-c8-attribution.json)).
  - Progress 2026-08-29: exact periodic rows81-144 Q4/Q5/Q6 bands plus Q6
    shared4 reuse lift C3/C4/C6/C7/C8 to **165.60/188.64/225.40/234.54/248.14
    tok/s**. The initial C5 218.70 row was a grouping outlier; clean pre-unit
    and candidate repeats are neutral at 212.216/212.207, so C5 remains open.
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-highrow-prefill-reuse-retained.json)).
  - Progress 2026-08-29: new exact 32-column Q4 shared-B owners (VGPR
    256→224) produce a one-run C4 **196.23 tok/s** crossing and improve
    C2/C3/C6/C7/C8. C5 is controlled neutral versus the pre-unit repeat and
    remains 2.5% short. Rows385+ still use parent shared-B, preserving the C8
    high-row blocker
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-q4-shared2-prefill-retained.json)).
  - Repeatability 2026-08-29: C7 clean runs are
    **237.53/237.42/237.60 tok/s**, median 237.53 (3.3% below 245.61); C4
    clean runs are **189.212/189.207/189.229**, median 189.212 (1.7% below
    192.54). Their favorable one-run rows are superseded; both remain open
    ([`C7 artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-prefill-c7-repeatability.json),
    [`C4 artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-prefill-c4-repeatability.json)).
    **Partial blocker:** only C1 is repeatably at parity. C2-C8 remain
    GPU-bound by high-row Q4/device work plus physical-grouping variance;
    selector and scheduler ladders above are exhausted. Continue with a new
    high-row Q4 algorithm/fusion after P3, not more threshold tuning.
- [x] P2.3 Small-row T16 wmma prefill GEMM kernel family (Q4/Q5/Q6 T16
  dense + qmicro planar wmma prefill): low-M efficiency work per
  `docs/KERNELS.md` + strict/production gates + rocprof trace evidence per
  AGENTS.md. This is now first-priority kernel work (the wall is below the
  serving boundary), not conditional on P2.1/P2.2.
  - Progress 2026-08-29 (first retained unit): Q4 dense low-M band (rows
    17-64, six shapes) routed to the single-wave owner — server prefill C1
    71.55→84.59 (+18.2%), C2 74.87→90.86 (+21.4%), isolated 45-token
    -18.9%, bit-exact, RED-first, 76/76 family tests
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-lowm-dense-q4t16-prefill-retained.json)).
  - Progress 2026-08-29 (fourth retained unit): low-VGPR 16-column Q4T16
    owners (VGPR 248→96/80, bit-exact) for rows 17-48 — cumulative C1
    71.55→122.09 (+70.6%), C2 74.87→115.58 (+54.3%); isolated 45-token
    prefill 0.515→0.331 s; rocprof: Q4 family 263→174.9 ms
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-lowvgpr-q4t16-prefill-retained.json)).
  - Progress 2026-08-29 (fifth retained unit): the same low-VGPR treatment
    routes Q6 qmicro-planar rows 17-48 through 16-column owners (VGPR
    184→88, bit-exact). The measured Q6 family falls 103.1→57.7 ms and
    cumulative server prefill reaches C1 71.55→132.54 (+85.2%) and C2
    74.87→133.52 (+78.3%); C1 remains 4.6% below the 138.95 target
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-lowvgpr-q6t16-prefill-retained.json)).
  - Closure 2026-08-29 (sixth retained unit): shape-scoped Q5 16-column
    owners (VGPR 200→96/112, bit-exact) cut Q5 35.79→28.74 ms and lift C1
    132.54→134.37 (+1.4%; +87.8% cumulative). C2 is controlled neutral:
    every slab exceeds the selector's row-48 cap, and same-protocol candidate
    versus plain is 125.74 versus 125.67 tok/s. P2.3 closes the original
    rows17-48 family. The subsequent P2.1 attribution found and retained the
    general rows49-80 extension; P2.2 owns the C1-C8 re-freeze
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-lowvgpr-q5t16-prefill-retained.json)).

### P3 — AR decode C1/C2/C8 (defend the C3-C7 lead)

- [~] P3.1 Close C1 `9.637 -> >= 11.336`, C2 `14.247 -> >= 20.056`, C8
  `39.057 -> >= 45.751` complete-wall tok/s, each with a measured win or named
  blocker, and no regression vs frozen C3-C7 (20.731/26.216/31.19/34.564/
  36.592).
  - Re-freeze 2026-08-29: current C1-C8 AR is
    **11.060/16.993/24.032/30.623/35.896/40.585/44.220/47.781 tok/s**, with
    all 80 cells exact. C3-C8 beat their frozen targets by 4.4-20.8%, defending
    the C3-C7 lead and closing C8. C1 remains 2.4% short and C2 remains 15.3%
    short; profile those two current decode routes before changing kernels or
    dispatch
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-ar-c1c8-refreeze.json)).
  - Progress 2026-08-29: cached marker traces show C1 at **81.77 ms wall /
    78.96 ms kernel sum / 592 launches**, while C2 is **97.28 / 81.56 ms /
    880 launches**. A model/quant/physical-width policy now admits only dense
    standard-Q4 packed C2 to HIP graph capture at all 23 D24 transitions.
    Clean C2 improves **16.993→18.072 tok/s (+6.35%)** with all ten cells exact;
    C1 stays eager and non-regressive at **11.122 tok/s**. Remaining gaps are
    C1 1.9% and C2 9.9%
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-packed-c2-short-graph-retained.json)).
  - Partial closure 2026-08-29: C1 is 96.6% device-kernel time and needs
    ~1.77 ms/transition; the prior Q4/Q5/Q6/norm/attention/activation/arena
    ladders are exhausted, so its named blocker is a new operation-complete
    serial dataflow. C2 graph replay is **85.26 ms wall / 81.38 ms kernels**;
    free capture projects only 18.67 tok/s. Its graph-record step plus the
    P2.2 GPU-bound prefill gap explain ~250.6 of the 262.7 ms/request deficit.
    C2's named blocker is the P2.2 high-row Q4/prefill algorithm or a new row-2
    operation-complete fusion. Do not reopen graph thresholds or multistep
    replay without a new premise
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-ar-c1-c2-blockers.json)).

### P4 — MTP K3 C1-C8 parity via the acceptance campaign

- [ ] P4.1 Execute `docs/QWEN38-Q4KM-MTP-ACCEPTANCE.md` E0-E5 as written;
  its own binding gates and statistical discipline apply unchanged.
  - E0 complete 2026-08-29: common K3 C1/C2/C3 is
    **7.826/19.146/21.382 tok/s** versus true AR
    **11.115/18.205/24.119**; all 30 cells are exact and acceptance remains
    78.89%. C3 fixed K1/K2/K3 is **22.414/20.804/21.382 tok/s**. Current
    profiling measures **746.7 ms** prompt prime and a **41.26 ms/cycle** Q6
    proposal head, so E1a then E1b remains the order
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json)).
  - E1a retained 2026-08-29: post-output-norm physical-C3 prompt streaming
    reduces complete MTP wall **33.673→26.501 s** and raises K3
    **21.382→27.169 tok/s (+27.06%)**, with exact 471/597 acceptance and every
    category positive
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e1a-prompt-streaming-retained.json)).
  - E1b retained 2026-08-29: exact physical proposal-head row reuse moves C3
    **27.169→29.198 tok/s (+7.47%)**, 6.12% above the frozen comparator, and C2
    **19.146→21.690 (+13.29%)**. Acceptance remains exactly 471/597 and 314/398;
    all 20 cells and every category pass
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e1b-proposal-head-rowtile-retained.json)).
  - Post-E1 Amdahl 2026-08-29: C3 target remains **190.12 ms/cycle** and
    **95.62% kernel-bound**; Q4/Q6/Q5 consume **102.71/50.19/12.98 ms** and
    87.25% of target wall. E2 Q4 true-R12 work passes its entry condition and
    remains the next shared-width/DFlash lever
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-post-e1-amdahl.json)).
  - E2 Q4 closeout 2026-08-29: exact true-R12 col4 and shared-weight owners are
    respectively **9.1-44.7%** and **72.9-169.7% slower** than R8+R4 across all
    six actual shapes, with zero paired wins. Q4 keeps R8+R4
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e2-q4-true-r12-rejected.json)).
  - E2 Q6 retained 2026-08-29: standard K5120/N10240 true-R12 is exact and
    moves leaf **0.6915→0.4468 ms (-35.38%)**, traced family
    **98.12→75.21 ms (-23.35%)**, and clean C3 **29.198→29.409 tok/s (+0.72%)**.
    Both planar R12 shapes lose and keep R8+R4
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e2-standard-q6-true-r12-retained.json)).
  - E2 Q5 retained/closeout 2026-08-29: K6144/N5120 true-R12 is exact and
    moves leaf **0.2654→0.1940 ms (-26.91%)**, traced family
    **97.11→78.07 ms (-19.60%)**, and clean C3 **29.409→29.564 tok/s (+0.53%)**.
    E2 is complete; choose E5/E6 promotion or wider-width re-freeze next
    ([`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e2-q5-true-r12-retained.json)).
- [~] P4.2 Frozen MTP parity targets: C1 `>= 21.277`, C2 `>= 32.378`, C3
  `>= 27.515`, C4 `>= 27.015`, C5 `>= 32.74`, C6 `>= 36.023`, C7 `>= 42.304`,
  C8 `>= 54.834`; each cell closes with a measured win or named blocker.
  Post-E2 re-freeze:

  | C | MTP tok/s | True AR tok/s | MTP/AR | External gap | Status / named blocker |
  | ---: | ---: | ---: | ---: | ---: | --- |
  | 1 | 7.814 | 11.024 | 0.7088x | -63.28% | production-D24 keeps direct/scalar proposal + replay; strict-natural25 evidence does not transfer |
  | 2 | **25.749** | 17.986 | **1.4316x** | -20.48% | blocked: target is 60.38% of child wall/95.38% kernel-bound; Q4 740 ms |
  | 3 | **29.564** | 24.042 | **1.2297x** | **+7.45%** | **closed win** |
  | 4 | **27.450** | 30.120 | 0.9114x | **+1.61%** | **closed win:** exact Q6 R8+R8 replaces R16 direct/WMMA |
  | 5 | **17.970** | 35.391 | 0.5078x | -45.11% | exact Q6 R8+R8+R4 retained; replay + Q4/Q5 operation-complete R20 target wall |
  | 6 | **26.904** | 39.797 | 0.6760x | -25.31% | exact Q6 R8x3 retained; replay + Q4/Q5 operation-complete R24 target wall |
  | 7 | **28.205** | 43.341 | 0.6508x | -33.33% | exact Q6 R8x3+R4 retained; replay + Q4/Q5 operation-complete R28 target wall |
  | 8 | **27.393** | 47.229 | 0.5800x | -50.04% | exact Q6 R8x4 retained; replay + Q4/Q5 operation-complete R32 target wall |

  All 80 cells are exact/engaged/budget-conformant, and acceptance is identical
  at 78.894% for every width. Each width is one physical group, so the wide
  blocker is execution-cost scaling, not acceptance or request-serial lowering.
  A follow-up trace shows wide proposals already lower to rows4 plus a remainder,
  so E1b covers them; rows5-8 policy keys cannot engage. Next: independent
  C2 streaming is subsequently retained at **25.749 tok/s (+19.31%)**; C4 is
  rejected after changing acceptance. C2 closes on a measured R8 blocker:
  target/accept/commit/provider is **1.205/1.996 s**, 95.38% kernel-bound, with
  Q4/Q6/Q5 **740/246/64 ms**. Qualified shapes are already one-sweep; narrow-Q4
  shared-B is numerically unqualified and insufficient alone. Exact Q6 chunks
  subsequently improve clean C5-C8 by **17.89%/29.16%/21.34%/39.41%**, remove
  wide direct-Q6 calls, and preserve all outputs and acceptance. The widths
  remain 25-50% below external and below AR; refresh wide target attribution,
  then screen Q4/Q5 operation-complete work under their exact/production gates
  ([`matrix`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-post-e2-c1c8-refreeze.json),
  [`proposal correction`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-wide-proposal-policy-rejected.json),
  [`C2 streaming`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-c2-prompt-streaming-retained.json),
  [`C2 blocker`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-c2-post-streaming-blocker.json)).
  C4 attribution measures target/accept/commit/provider at **3.533/4.861 s
  (72.68%)**, 97.78% kernel-bound. Exact Q6 R8+R8 then cuts the Q6 family
  **1.847→0.478 s (-74.10%)** and clean C4 **20.384→27.450 tok/s (+34.66%)**,
  1.61% above the frozen comparator, with exact 628/796 acceptance. Across
  C5-C8, target enqueue plus its following synchronization/commit/readback
  owner consumes 57.5-69.5% of wall. Exact Q6 R20-R32 chunks then improve clean
  C5-C8 by **17.89%/29.16%/21.34%/39.41%**, while a C8 trace removes direct Q6
  and reduces whole-process Q6 kernel time by about 69.95%. Every width remains
  below AR and 25-50% below external; refreshed Q4/Q5 target attribution is next
  ([`wide blockers`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-c4-c8-target-blockers.json),
  [`C4 win`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-c4-q6-r16-retained.json),
  [`wide Q6`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-q6-r20-r32-retained.json)).
- [ ] P4.3 Reopen T3 adaptive-K and the B4 clamp once verifier rowtile work
  lands (per the CONCURRENCY2 supersession note); require full-suite
  acceptance/speed validation per anti-gaming.

### P5 — DFlash2 revisit (acceptance parity proven; cost is the wall)

- [ ] P5.1 Root-cause the reverted rowtile-8 AR divergence (verify 620→310 ms
  halving, AR-divergent, never root-caused; `docs/DFLASH.md` top open item).
- [ ] P5.2 Attack the drafter+select cost wall (~96 ms/cycle vs MTP proposal
  2.4 ms) sharing the E2 high-row amortization work; measure before/after on
  the common suite.
- [ ] P5.3 Decision cell: promote DFlash2 on any phase×concurrency where it
  beats both our MTP K3 and AR under the frozen protocol, or record the
  measured named weakness and map each blocker to its MTP-shared fix.

### P6 — closure and rollup

- [ ] P6.1 Full standardized matrix re-run: all six external rows re-measured
  same-host/protocol plus the final hipEngine row; artifact under
  `benchmarks/results/`.
- [ ] P6.2 Rollup: `benchmarks/README.md` row + `Last updated`,
  `benchmarks/CHANGELOG.md` one-liners per retained win, campaign closeout
  worklog entry, and `docs/PLAN.md` status refresh.

## 4. Non-goals

- No FP4/ROCmFPX, Unsloth UD-Q4_K_XL/Q5/Q6/Q8, or ngram-replay parity rows
  (decision 1A; separate later tracks).
- No new external vendoring; external repos stay read-only references under
  `/home/lhl/.local/state/hipengine-external-survey/repos/`.
- No benchmark gaming: no prompt-conditioned tuning; every acceptance/speed
  claim validates on the full multi-prompt suite plus category heldouts.
