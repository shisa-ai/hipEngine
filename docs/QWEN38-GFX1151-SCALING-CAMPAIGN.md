# Qwen3.8-27B gfx1151 Scaling Campaign (MTP batch scaling + prefill)

Status: **punchlist closed; post-closeout review corrections recorded
2026-08-31; bounded C6/C8 K1 successor loop closed on 2026-09-01; extension W
(dataflow-wall successor punchlist) opened 2026-09-01, W0 instrumentation done
and W1 closed measured-blocked; extension Y (prefill sweep-multiplicity punchlist) opened 2026-09-01, no Y
unit measured yet**.
Successor to the closed
[`external-parity campaign`](QWEN38-GFX1151-PARITY-CAMPAIGN.md).
Owner: scaling loop.

Scope: the same frozen product key as the parity campaign — physical gfx1151
(Ryzen AI MAX+ 395 / Radeon 8060S), `Qwen3.8-27B` `standard_q4_k_m`
(sha256 `7e78da5d…c6fe169`), BF16 KV, production profile, common ten-prompt
suite (sha256 `fac920be…1d86084a`), raw greedy, no prompt cache, K3, D24.

This campaign does **not** reopen AR decode. AR leads C3-C8 and its C1/C2
blockers stay closed under the parity campaign's named-blocker rule.

## 1. Goal

Two measured objectives, in priority order:

1. **MTP must scale with concurrency.** Today it does not: MTP reaches its peak
   at C3 and collapses at C5 while AR keeps scaling. Target is
   `MTP >= 1.15x own AR` at every width C1-C8, or a measured named blocker per
   cell.
2. **Close the two prefill cells that hold 60% of the prefill deficit** — C2
   and C8 — or name their blockers. The other six widths are already within
   4.3-9.5% and are explicitly *not* the target.

Scope note: this campaign targets MTP at C1-C8 and prefill at C2/C8 while
preserving the existing AR lead. It does not claim literal leadership in all 24
survey cells; AR C1/C2 and prefill C1/C3-C7 remain explicit non-goals.

## 2. Frozen entry state (2026-08-30 six-engine matrix)

Source: [`final matrix`](../benchmarks/results/2026-08-30-gfx1151-qwen38-final-six-engine-c1c8.json),
complete-wall tok/s.

| C | Prefill | vs best ext | AR | MTP | vs best ext | MTP / own AR | AR scale | MTP scale |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 146.758 | -4.33% | 11.021 | 7.809 | -63.03% | 0.7086x | 1.00x | 1.00x |
| 2 | 142.806 | **-32.60%** | 17.918 | 25.740 | -20.12% | 1.4365x | 1.63x | 3.30x |
| 3 | 174.878 | -9.12% | 23.659 | 29.468 | **+4.99%** | 1.2455x | 2.15x | 3.77x |
| 4 | 188.942 | -5.75% | 30.083 | 29.385 | **+9.02%** | 0.9768x | 2.73x | 3.76x |
| 5 | 211.737 | -6.36% | 35.544 | 18.657 | -42.97% | 0.5249x | 3.23x | 2.39x |
| 6 | 226.616 | -9.50% | 39.906 | 28.195 | -24.11% | 0.7065x | 3.62x | 3.61x |
| 7 | 240.672 | -4.60% | 43.112 | 29.305 | -36.41% | 0.6797x | 3.91x | 3.75x |
| 8 | 247.216 | -19.17% | 45.936 | 28.577 | -49.17% | 0.6221x | 4.17x | 3.66x |

Two signatures drive the whole campaign:

- **MTP reaches a ~29.5 tok/s ceiling from C3 onward and collapses at C5**
  (29.468/29.385/18.657/28.195/29.305/28.577) while AR climbs 23.659 ->
  45.936. Every external engine finishes materially higher at C8 than at C3;
  ours does not scale beyond its C3 peak.
- **Prefill C2 is lower than C1** (142.806 < 146.758) where every external
  engine rises steeply (stock HIP 146.2 -> 186.9; Laurent 149.1 -> 211.9).
  C2 and C8 hold **32.4%** and **27.5%** of the absolute prefill deficit; the
  other six widths together hold 40.1%.

## 3. Diagnosis

Findings A-C are new attributions from current source and retained telemetry.
D-F restate measured facts already in the tree so the punchlist has one entry
point.

### A. The MTP wall is operation-complete row cost, then a width-4 partition

Two compounding defects, which must be measured separately:

**A1 — the operation-complete speculative cycle costs more per target row than
AR decode costs per row.** From the retained wide telemetry
([`wide blockers`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-c4-c8-target-blockers.json)),
a complete C8 subgroup cycle is **688.1 ms** for R16, i.e. ~43 ms per target
row. The matched AR cycle at C8 serves 8 rows at ~21.8 ms per row (derived from
the matrix's complete-wall 45.936 tok/s, not from cycle telemetry). The current
operation-complete speculative cost is therefore about **2x higher per row**
even though a wide verify pass should amortize one weight sweep over 4x the
rows. The 688.1 ms is not a pure target-kernel measurement; M0 replaces both
sides with matched cycle accounting and separates target, accept/commit, and
other ownership. This is the "multi-row verify amortization wall" the parity
campaign named as the shared MTP/DFlash2 suspect (P5.3).

**A2 — every width above 4 runs sequential complete cycles.**
`hipengine/generation/engine_loop.py:2336` `_maybe_run_partitioned_speculative_decode`
loops over subgroups and calls `_maybe_run_speculative_cycle` once per
subgroup, so C5-C8 execute 2 full proposal+verify cycles per tick. The bound
comes from eleven width-4 cap expressions in
`hipengine/generation/qwen35_gguf_mtp2.py` at
`:504/:780/:871/:894/:915/:934/:1028/:1052/:1159/:1314/:3413`. The capability
gate `GGUF_SPECDEC2_MTP2_C4` in
`hipengine/kernels/hip_gfx1151/__init__.py:1750` also stops at C4. A separate
C4-only accept owner at `qwen35_gguf_mtp2.py:1865-1885` fixes
`max_rows=16`, `max_requests=4`, and two request buffers to shape `(4,)`;
`TargetVerifyBufferOwner.bind()` rejects larger batches. Retained telemetry
confirms the resulting subgroup shapes `4`, `4+1`, `4+2`, `4+3`, `4+4` at
C4-C8.

This is a deliberate CONCURRENCY2 D4 decision ("physical through C4 and
decomposed into bounded C4 frontiers above it"), correct as a functional
milestone and now the binding scaling defect. It is a scheduler/ownership
bound, not a kernel bound. **The C5 `4+1` shape explains its matrix-worst
0.5249x-AR result**: the trailing R4 group pays a near-full pass for a quarter
of the rows.

*Status 2026-08-30 (M1):* the eleven caps and the `GGUF_SPECDEC2_MTP2_C4`
gate are gone - one capability-owned profile-keyed bound owns the width, the
accept owner sizes through C8/R32, and the server batch route adopts the
engine bound. The partition still runs at C5-C8 only because the measured
wide single cycle regresses C6-C8 (missing wide-row owners and accept
scaling, M1 blocker artifact); the default bound remains 4.

The width-4 cap, its eleven cap expressions, and the C4 accept owner are tracked
as [`REFACTOR.md`](REFACTOR.md) **RF-OI5** with removal tied to M1's outcome
(punchlist R1, closed at campaign open). A1 is a measured performance defect,
not refactor debt, and stays owned here.

### B. C1 production MTP has a strong missing-coverage hypothesis

Production D24 C1 is **7.809 tok/s / 0.7086x AR**, while our strict C1/K3
natural25 route is **18.191 vs 11.062 = 1.6445x**, and the gfx1100
`llama-compat` route reaches **1.2679x own AR** on reusable native target
graphs ([`MTP-LLAMACPP-PARITY.md`](MTP-LLAMACPP-PARITY.md)). Those controls
show that related routes can win; they do not establish production-D24
capability because the execution profile/horizon or backend differs. Together
with the missing production keys below, they make coverage the leading
hypothesis. Concretely, C1 is absent from exactly the two policy tables that
produced the C3 wins:

- `GGUF_SPECDEC2_PHYSICAL_PROMPT_STREAMING_POLICIES`
  (`hip_gfx1151/__init__.py:1754`) admits widths **`(2, 3)`** only, so C1 still
  pays full prompt replay. E1a's streaming was worth **+27.06%** at C3.
- `GGUF_SPECDEC2_PROPOSAL_LM_HEAD_ROWTILE_POLICIES` (`:1765`) admits rows
  **2/3/4** only, so C1's proposal head keeps the direct/scalar producer. E1b
  was worth **+7.47%** at C3.

The resolver `_physical_prompt_streaming_widths()` currently rejects width 1
at `qwen35_gguf_mtp2.py:270-271`; adding a package-policy key alone raises an
error. M3 therefore owns the resolver change and its policy-miss guards, not
only the two package keys.

E0 measured **746.7 ms** prompt prime and a **41.26 ms/cycle** proposal head on
this route, so both keys are sized to matter at C1.

### C. At campaign entry, Q4 was a major verify cost with no owner above R12

`GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ROWS = {6, 8, 9, 12}`
(`:1604`). Q6 R16 is admitted through
`GGUF_T16_TARGET_VERIFIER_ROWTILE_CHUNK_ROWS_BY_QUANT` (`:1580`); Q5 R16 is a
separately qualified entry in `GGUF_T16_TARGET_VERIFIER_TRUE_ROWTILE_VARIANTS`.
Neither table gives Q4 an owner above R12. The C4 trace puts Q6 at 1846.5 ms and
Q4 at 1059.0 ms of 3454.9 ms target kernel time; the C2 blocker records Q4 at
740 ms of a 95.38% kernel-bound target. Q4 first hit a leaf-only stop, then an
operation-complete revisit. In that revisit, weighted GPU work rose **55.31%**
and the one-prompt C4 screen regressed **6.39%**. Widening the group (A2)
without a Q4 owner above R12 will move the wall, not remove it, so **A2 must be
measured with per-quant attribution** and C's entry condition is A2's trace, not
another blind R16 retry.

### D. Prefill is now two cells, not a trend

Six of eight widths sit within 4.3-9.5% of the best external engine. The
campaign's prefill scope is only:

- **C2 (-32.60%, 32.4% of the deficit).** hipEngine prefill *falls* from C1 to
  C2 while all five external engines rise 20-42%. That is a grouping/dispatch
  signature at width 2, not the high-row Q4 blocker the parity campaign named
  for C8. It was never independently attributed: the P2.2 attribution traced
  "fully grouped C2 rows134" and found it 98.4% GPU-bound with Q4 owners at
  59.4%, but did not explain the absence of any C1 -> C2 scaling.
- **C8 (-19.17%, 27.5% of the deficit).** This is the named high-row Q4/device
  blocker. It is real and terminal under threshold/scheduler tuning; the parity
  campaign's own closing instruction was "continue with a new high-row Q4
  algorithm/fusion, not more threshold tuning."

### E. The recorded prefill kernel efficiency number is stale

[`P1.3`](../benchmarks/results/2026-08-29-parity-p13-c1-prefill-attribution.json)
traced M=45 at **1380 launches / 534.2 ms GPU busy**, with
`gguf_q4_t16_dense_wmma_prefill_shared_b_bf16` at **360.56 ms / 298 launches /
1.21 ms avg**. That one family alone was **4.66x** the **77.4 ms** full-model
17.1 GB sweep anchor at the retained **221 GB/s** practical read roof
([`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md)); this is a screening anchor, not
a family-byte measurement. [P2.3's low-VGPR
work](../benchmarks/results/2026-08-29-gfx1151-qwen38-lowvgpr-q5t16-prefill-retained.json)
raised C1 server prefill from 71.55 to 134.37 tok/s, and [P2.1's subsequent
rows49-80 extension](../benchmarks/results/2026-08-29-gfx1151-qwen38-row49-80-prefill-parity-retained.json)
reached 147.11 tok/s; the final matrix records 146.76. Intermediate post-P2.3
M=45 and post-P2.1 row-67 traces exist, but there is no current-head,
frozen-protocol trace of the target C2/C8 paths. Any new prefill kernel work
must re-trace those paths first.

### F. The integer-MMQ prefill continuation was specified and never done

[`GGUF-PREFILL-OPTIMIZATION.md`](GGUF-PREFILL-OPTIMIZATION.md) item 19
(`LCP-3B`) rejected a direct prequantized Q8_1 x Q8T16 integer-WMMA prefill
body (44.66% slower) with an explicit continuation condition: any retry "must
reproduce the actual shared MMQ tile/decomposition, not route raw dp4a or retry
direct T16." That continuation was never attempted. MMQ sources exist only
under `kernels/hip_gfx1100/` (`gguf_k_mmq_prefill.hip`,
`gguf_q8_0_mmq_prefill.hip`, `gguf_iq2_xs_mmq_prefill.hip`);
`kernels/hip_gfx1151/` contains **no `.hip` files at all** and inherits gfx1100
bodies through policy.

**Size it before building it.** On gfx1151, INT8 WMMA is **59.4 TOP/s — the
same rate as FP16/BF16 WMMA**, not double (only INT4 reaches 118.8 TOP/s)
per [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md). MMQ's payoff here is removing
dequant ALU and halving activation register/LDS traffic, **not** a matmul-rate
win. If P2's trace does not show dequant ALU dominance, F does not open.

## 4. Punchlist

Rules unchanged from the parity campaign: every cell closes at its target under
the frozen protocol with the applicable `docs/EXECUTION-PROFILES.md` gate, or
with a **measured, named blocker** recorded here and in the unit worklog entry.
Full-suite plus category-heldout validation for every acceptance/speed claim; no
prompt-conditioned tuning. Commit each completed unit immediately.

**"All 80 cells exact"** means the parity campaign's gate: 10 prompts x 8
widths, each cell passing exact-generated-ID/output, engaged-route, and
budget-conformance checks (the closeout's "exactness/route/budget gates").
**Known evidence gap:** the frozen entry artifact embeds the full `protocol`
block and raw-source hashes but **not the driver command lines**; the parity
closeout entry does not record them either. Prefill and AR/MTP came from
separate raw sources. M0's artifact and worklog entry must therefore
re-establish and record the exact reproducible command set for the C1-C8
prefill, AR, MTP, and instrumentation/profile runs before any M-track perf claim
is made.

### X — external MTP batching survey (cheap, de-risks M1)

- [x] X1 **Done 2026-08-30.** Read all pinned checkouts under
  `/home/lhl/.local/state/hipengine-external-survey/repos/` (llama.cpp
  mainline `4e97ac86`, mike pin `152d337f`, Vulkan/ROCmFPX fork pins
  `laurent/`, `nathan/`, `q38rocm/`) and the unpinned upstreams from
  commit-pinned URLs (vLLM V1 `8c51b926…`, SGLang EAGLE `e51a3ae6…`). The
  comparison table of batch dimension, per-cycle model passes, and width
  caps — cited by file and commit, no vendoring, no code port — is in
  [`EXTERNAL-MTP-BATCHING.md`](EXTERNAL-MTP-BATCHING.md). Verdict: every
  surveyed engine flattens draft verification for all in-flight requests
  into **one** target forward bounded by a generic token budget
  (`n_batch`/`max_num_batched_tokens`), not a small fixed width; A2's
  per-subgroup sequential cycles have no external analogue. This supports
  proceeding with M1.

### M — MTP scaling (primary track)

- [x] M0 **Done 2026-08-30.** Re-frozen the C1-C8 prefill/AR/MTP protocol at
  head with the exact reproducible command set, model/prompt/profile-manifest
  hashes, and raw-source hashes recorded in
  [`2026-08-30-gfx1151-qwen38-mtp-scaling-m0-refreeze-instrumentation.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-mtp-scaling-m0-refreeze-instrumentation.json),
  plus per-cycle accounting from append-only pass-ms samples
  (`scripts/mtp_cycle_accounting.py`). Headline: 80/80 exact/engaged/budget,
  identical 78.894% acceptance, exactly **2.00 physical target passes per
  cycle at C5-C8** (width-4 partition confirmed per tick), C8 operation-complete
  34.44 ms/committed-token vs matched AR 21.58 ms/row while the target kernel
  itself costs only 2.63 ms/row - A1's gap lives in the operation-complete
  cycle (proposal + accept-interval sync), not target math. C1 production-D24
  baseline for M3: **7.841 tok/s**, streaming engaged on 0 requests, direct
  proposal head. No perf claim.
- [x] M1 **Single-group wide verify.** **Mechanism done 2026-08-30; wide
  default measured and blocked.** All eleven scattered caps were replaced by
  one capability-owned profile-keyed bound
  (`GGUF_SPECDEC2_MTP2_PHYSICAL_MAX_REQUESTS`; the `GGUF_SPECDEC2_MTP2_C4`
  gate is renamed `GGUF_SPECDEC2_MTP2_PHYSICAL`), the accept owner/spec/
  remaining/payload tensors re-key through C8/R32, and the server
  explicit-MTP batch route now adopts the engine-published bound instead of
  its old hardcoded 4 (an outer wall the campaign inventory missed: without
  it the engine never receives a due-group wider than 4). With the bound at
  8 the frozen protocol ran **80/80 exact, engagement-complete, identical
  78.894% acceptance, 1.00 physical target passes/cycle at every width** -
  every M1 binding gate passed. The intermediate targets missed: C5 won
  (+18.99%, `18.912 -> 22.503`, still below 33) but C6/C7/C8 regressed
  (`-12.49/-19.56/-9.59%`), so the production default reverted to the
  certified width-4 bound and the mechanism waits behind M2. The rows5-8
  proposal head keeps the direct producer - measured cost: proposal member
  sums `+198..+264%` (C8 `19.3 -> 70.3 s`). rocprof attribution (M2's entry
  trace): at R>16 the wide pass loses every R<=16 owner - Q6-planar direct
  gemv `4265.6 ms / 116 launches (~36.8 ms each)`, `qk_t16_selected_direct_
  gemv` 1473 ms, Q4 `wmma_prefill<false,2>` 1868 ms - and the single
  eight-request accept interval costs +69.6%. Full numbers:
  [`m1-wide-cycle-blocked`](../benchmarks/results/2026-08-30-gfx1151-qwen38-mtp-m1-wide-cycle-blocked.json).
  [`Entry`](../worklog/entries/20260830T142143.717427Z-lhl-qwen38-m1-wide-bound-blocker-8b3ed2.md)
- [x] M2 **Per-row verify cost (A1). Done 2026-08-31: explicit C4 gate PASS
  (reviewed all-ten 35.474 >= 34.596); C5-C8 K3 closed with measured blockers
  and C6/C8 K1 follow-ups identified below.** Using M1's per-quant attribution, close the
  gap between MTP ms/target-row and AR ms/row at matched width.
  **Interim (2026-08-31):** the M2i hip-API/copy/kernel trace closed every
  host-side explanation (the accept window is 98% GPU-busy; the drain is Q4/Q5/Q6
  verify math). M2j promoted the bit-exact low-VGPR/shared-B2W2 Q4 owners to
  physical rows 2-16 (C1 **+75.5% to 13.759**, C4 **+15.8% to 34.201 = 1.138x
  AR, 1.14% short of the 34.596 gate**, C5-C8 **26.059/31.885/32.395/33.491**;
  raw suite 80/80 exact/engaged/budget, including 48 non-heldout cells).
  M2k screened Q5/Q6 siblings: the admitted rowtiles already win
  every bit-exact cell and association-different WMMA alternatives save <5 ms
  per pass - strict-owner recovery is exhausted at the certified bound.
  [`M2j`](../benchmarks/results/2026-08-31-gfx1151-qwen38-mtp-q4-verify-owner-retained.json)
  M1+M2 jointly
  own the final `MTP >= 1.15x own AR` gates: C4 `>= 34.596`, C5 `>= 40.876`,
  C6 `>= 45.892`, C7 `>= 49.579`, and C8 `>= 52.827` tok/s, or a measured named
  blocker for each missed cell. Q4 owners above R12 (C) open here **only if**
  M1's trace names Q4 as the binding class; the leaf-only stop and
  operation-complete revisit set the entry condition — weighted GPU work must
  not rise.
  **Successor update 2026-09-01:** exact R16 Q4 owners are now retained for
  K17408/N5120 and K5120/N1024 only. Four peer shapes remain on their measured
  winners. These scoped owners lift C8 K1 to 43.421 tok/s but do not close the
  compound target; the remaining named blocker is the multi-family
  packed-verifier dataflow wall recorded in the post-audit successor ledger.
  **C5-C8 K3 closure (reviewed 2026-08-31, measured named blocker per cell):**
  the tracked-clean all-ten current-head refresh measures
  **27.980/32.807/33.106/35.423** vs gates 40.876/45.892/49.579/52.827
  (80/80 C1-C8 exact/engaged/budget). Every width>=5 K3 cycle is sub-group
  interleaving at 0.75-0.81x own AR whose per-cycle floor includes the
  GPU-busy verify drain; M1's one-pass K3 loses the R<=16 owners and M2k
  exhausts exact siblings at the certified bound.
  **Review correction:** this is not a terminal all-depth result. A new
  one-pass K1 screen keeps target rows at R10/R12/R14/R16 and measures
  C5-C8 **22.537/35.383/24.167/39.260**: C6 **+7.85%** and C8 **+10.83%**
  over split K3, while C5/C7 regress. All 40 cells are exact/engaged/budget
  with 95.65% acceptance. C6/C8 K1 remain below own AR and need a
  width×depth admission implementation plus the full production gate; they
  supersede the prior "no remaining mechanism" claim. See the
  [review artifact](../benchmarks/results/2026-08-31-gfx1151-qwen38-reviewed-current-head-c1c8.json).
- [x] M3 **C1 coverage (B). Done 2026-08-31:** all clauses closed below; the
  18.191 aspiration residual is -14.0% on the compact non-heldout metric and
  -13.4% on the reviewed all-ten aggregate, named to the shared accept-window
  verify math (same wmma-family structural rate as M2/P2). Extend `_physical_prompt_streaming_widths()` to
  admit width 1 without broadening the unqualified `>4` range; add the width-1
  package-policy key and qualify the rows1 proposal rowtile owner. Re-screen the
  reusable native target graph for the production route. Use M0's refreshed
  production-D24 C1 result as the matched baseline. Target `>= 18.191` (the
  strict natural25 result, used as an aspiration rather than a comparator),
  stretch `>= 21.126` (external). Resolver, policy-miss, and strict-C1 tests
  must prove strict automatic behavior is unchanged.
  **Interim (2026-08-31):** validator + width-1 production policy key retained
  (streaming screen: IDs/acceptance/route/budget identical to replay, +24.1%;
  the promotion artifact's six-non-heldout arithmetic headline is **15.646**;
  the reviewed all-ten aggregate is **7.841 -> 15.753 tok/s, 1.418x own AR,
  80/80 C1-C8 exact/engaged/budget**); the rows1 proposal-head clause needs no wrapper change (rows1
  lm-head is the qualified decode GEMV and multi-row proposals stay in the
  admitted rows2-8 band); the native-graph re-screen closes no-capture on the
  staged route (0 graph buckets; N2/N3 belongs to the llama_compat adapter and
  M2i shows the window GPU-busy). The residual to the 18.191 aspiration lives
  in the shared accept-window verify math (C1 accept-member 111
  ms/cycle vs 33.6 ms/pass target kernels).
  [`M3`](../benchmarks/results/2026-08-31-gfx1151-qwen38-mtp-c1-streaming-width1-retained.json)
- [x] M4 **C4 prompt-streaming acceptance blocker. Done 2026-08-31; scope
  corrected in review.** The explicit diagnostic binds `mtp_self_exact` IDs +
  route + budget, but acceptance changed (92/121 vs 93/120), so this is a T3
  explicit scope under `EXECUTION-PROFILES.md`, not an automatic production
  numerical/task promotion. The six-non-heldout arithmetic headline is
  **34.182 -> 35.618**; the reviewed all-ten aggregate is **35.474 tok/s,
  1.177x own AR**, and 80/80 C1-C8 cells pass. Production key (1,2,3,4) is
  retained for explicit MTP; automatic C4 remains K0.
  [`M4`](../benchmarks/results/2026-08-31-gfx1151-qwen38-mtp-c4-streaming-retained.json)
- [x] M5 **Concurrency-aware admission. Done 2026-08-31, named blocker;
  artifact gap repaired in review.** Current-head all-ten economics:
  sub-group K3 costs **0.75-0.81x own AR** at widths 5-8
  (27.980/32.807/33.106/35.423 vs AR 35.778/40.343/43.974/47.194, 40/40
  exact/engaged/budget).
  The engine-surface whole-batch AR route (`GGUF_SPECDEC2_MTP2_BATCH_ROUTE_ABOVE_REQUESTS`,
  seam-tested) is implemented but measured inert on the server-bench surface:
  admission caps explicit-MTP groups at 4 upstream, so over-width batches never
  reach the partitioner. Production demotion therefore needs an **admission-route
  change plus a frozen-protocol amendment of the C5-C8 engagement contract**
  (route ceiling = AR, still 10-12% under the 1.15x gates) - a campaign-owner
  decision, not silent scope. An interim +67-90% reading was retracted as a
  `--max-tokens 512` protocol artifact. See RF-M5 in
  [`REFACTOR.md`](REFACTOR.md), the M5 entry, and the committed
  [review artifact](../benchmarks/results/2026-08-31-gfx1151-qwen38-reviewed-current-head-c1c8.json).

### P — prefill (secondary track, two cells only)

- [x] P1 **C2 root-cause. Done 2026-08-31, named blocker.** Prefill ticks are
  **100.0% GPU-busy** (paired hipEvents, 55 calls, median 281.4 ms wall = GPU).
  Rows 35-48 all cost ~278 ms regardless of rows: the Q4 `wmma_prefill` owner
  streams ~16.3 GB of weights per tick at ~57 GB/s, and grouped ticks re-stream
  per 16-row M-tile instead of amortizing (rows256 = 1418 ms, ~4.4 ms/row).
  Removing the equal-length grouping gate fires ragged groups (rows72-96 at
  ~410 ms vs 2x278 serial) but moves C2 only -3.0% / C3 +4.8% - inside drift;
  the experiment was reverted byte-identical. Grouping/scheduling is measured
  **not** the blocker; the floor is the prefill owner's weight streaming, which
  is exactly P2/P3's algorithmic target. See
  [`blocker artifact`](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-c2-scaling-blocker.json).
- [x] P2 **C8 high-row Q4. Done 2026-08-31, named blocker; P3 gate OPENED.**
  Current-head `rocprofv3 --kernel-trace` of the frozen width-8 path: Q4 WMMA
  family = **60%** of trace GPU (16.8 s of 27.9 s). Engine ticks are pure
  kernel time (rows288: 1080.0 ms wall / 1080.0 ms hipEvent, **gapshare
  0.0%**), and the standalone shared_b owner runs rows256-1024 at
  **19-24 TF/s (~35% of the 60 TF/s MFMA peak)** with weight streaming far
  under DRAM bandwidth: the wall is the **in-loop Q4 dequant ALU/LDS/issue
  pass**, not bandwidth, gaps, or scheduling. Named blocker recorded;
  P3's opening conditions (dequant dominance + written sizing assuming
  INT8 WMMA == BF16 at 59.4 TOP/s; projection 247 -> ~330 >= 305.847)
  are met. See
  [`trace blocker`](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-c8-current-trace-blocker.json).
- [x] P3 **Integer-MMQ continuation. Done 2026-08-31, closed measured
  negative.** Both opening conditions were met (P2 dequant dominance + written
  sizing), so the mandated reproduction was screened rather than re-invented:
  items 20/21 already built the shared MMQ tile/decomposition (T16-backed
  MMQ128 +118.81%; source MMQ128 over prequantized `block_q8_0` spill-free and
  still +3.95%/+5.32% slower). Re-screen at the current head (H5120, dual
  18432, rows256): selected-wmma 10.71 ms / 9.03 TF/s vs best integer body
  13.21 ms (0.81x), others 0.23-0.68x. INT8 WMMA == BF16 rate and the
  dequant-free bodies hit the same LDS-staging/issue wall, so the >=2x
  Q4-family gain the 305.847 target needs is unreachable by data-format
  change. See [`negative`](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-p3-int-wmma-negative.json).

### R — refactor ledger

- [x] R1 **Done and corrected on 2026-08-30.** The width-4 MTP partition,
  eleven `qwen35_gguf_mtp2.py` cap expressions, C4-only accept owner,
  per-subgroup sequential loop, and `GGUF_SPECDEC2_MTP2_C4` gate are recorded
  as [`REFACTOR.md`](REFACTOR.md) RF-OI5 with an explicit removal condition tied
  to M1's outcome, success or blocker.

## 4a. Post-closeout review (2026-08-31)

The review re-ran both standardized hipEngine matrices at tracked-clean
`b768516f2` and audited the implementation path:

- Public survey rates use total tokens / summed wall over all ten prompts.
  M2j/M3/M4 compact headlines used an arithmetic mean over six non-heldout
  prompts; their raw sources still contain all 80 cells. Public comparison now
  uses 15.753/28.441/30.541/35.474/27.980/32.807/33.106/35.423 K3 tok/s.
- NextN proposal depth is serial because each draft token conditions on the
  prior draft, but every depth batches the physical request group. C1-C4 target
  verification is one flattened packed forward. C5-C8 production-server K3
  remains two serial complete groups because admission caps explicit MTP at 4.
- One full-width K1 pass is not a universal solution, but C6/R12 and C8/R16
  are real +7.85%/+10.83% follow-ups. The next policy must resolve both width
  and K before mutation; prompt/content-dependent selection remains forbidden.
- C2's external-gap blocker is the measured operation-complete R8 target plus
  proposal work, not an acceptance-rate ceiling.
- M4 is retained only as an explicit diagnostic T3 scope; automatic C2-C8
  remains K0 pending the complete production numerical/task/serving gates.

## 4b. Historical audit ledger (exact Git provenance)

Audit cut: `a3ffbd8f8790fa4530523210b0941b89d9bfbdca`. Timestamps below are exact
Git **committer timestamps** from `%cI`, including their `+09:00` offset; they
are not inferred from artifact filenames or worklog front matter. Full hashes
are intentional so a later dissection can reproduce the repository state
without resolving an abbreviated hash.

### Campaign and punchlist provenance

| Scope | Exact commit and committer timestamp | Role and retained result | Durable evidence |
| --- | --- | --- | --- |
| Campaign definition | `b40bc9edb37b3beb86df26d703f1b03e1746da8d` — `2026-08-30T18:53:48+09:00`; `04d45c70e86b400340e71eefecbf006e53cd41dc` — `2026-08-30T18:53:48+09:00`; `d2c3721ff6b1e7f24a2b4aa96ad0fe7b9f3319c8` — `2026-08-30T19:01:46+09:00`; `112a50f9a4241e7f533c36b52a10a2f75a970a0e` — `2026-08-30T19:24:34+09:00` | Opened, independently hardened, corrected, then finalized the scope and evidence rules before measured units began. | [`opening entry`](../worklog/entries/20260830T075721.949466Z-lhl-qwen38-scaling-campaign-8fc9d4.md), [`scope correction`](../worklog/entries/20260830T095634.730287Z-lhl-qwen38-scaling-campaign-correction-b14b99.md), [`final correction`](../worklog/entries/20260830T102020.938305Z-lhl-qwen38-scaling-campaign-final-correction-e05ba0.md) |
| R1 refactor ledger | `04d45c70e86b400340e71eefecbf006e53cd41dc` — `2026-08-30T18:53:48+09:00` | Introduced RF-OI5 with the width-cap inventory and an M1-dependent removal condition. Later commits update the condition; the historical insertion remains this commit. | [`REFACTOR.md` RF-OI5](REFACTOR.md) |
| X1 external batching | `17bff8d283ff9e5405831e7f91c9c766b5830dc1` — `2026-08-30T19:43:21+09:00` | Closed the commit-pinned reading survey; found that surveyed engines flatten request verification into one token-budget-bounded target forward. | [`survey`](EXTERNAL-MTP-BATCHING.md), [`entry`](../worklog/entries/20260830T104252.968383Z-lhl-qwen38-x1-external-batching-a2a8b9.md) |
| M0 instrumentation and re-freeze | `db360a4128ab233f3a0fc2f14c034148f9e33b5d` — `2026-08-30T19:52:52+09:00`; `7ad92a2690af3aa220188040e72679e3c0aac124` — `2026-08-30T19:55:53+09:00`; `2bc7c7742b90fdff124edc4f210550ea704dfd3c` — `2026-08-30T20:40:45+09:00` | Added append-only pass telemetry, added the cycle-accounting extractor, then committed the reproducible C1-C8 baseline and compact artifact. | [`artifact`](../benchmarks/results/2026-08-30-gfx1151-qwen38-mtp-scaling-m0-refreeze-instrumentation.json), [`entry`](../worklog/entries/20260830T114000.239909Z-lhl-qwen38-m0-refreeze-instrumentation-77f32c.md) |
| M1 physical C8 mechanism | `84f3744d8c8705875fa14886541ba8787fcc9c6f` — `2026-08-30T21:38:22+09:00`; `0e7b4f31a6ccd8c18aeb1251c0e90ce81695d853` — `2026-08-30T22:27:15+09:00` | Replaced scattered caps with one capability-owned bound and propagated it to server admission. | [`implementation entry`](../worklog/entries/20260830T115724.853542Z-lhl-qwen38-m1-single-group-wide-verify-68d526.md) |
| M1 measured default decision | `f78556bf9118fa80cf7b9ce9c0ccc28d52357eb5` — `2026-08-30T23:23:12+09:00`; `47cb994f76ae2e96ba305b14e7126518f7d80ffe` — `2026-08-30T23:42:22+09:00` | Restored the certified production bound of four after C6-C8 K3 regressions, then committed the blocker artifact and rollup. The physical-C8 mechanism remains available behind policy. | [`artifact`](../benchmarks/results/2026-08-30-gfx1151-qwen38-mtp-m1-wide-cycle-blocked.json), [`decision entry`](../worklog/entries/20260830T142143.717427Z-lhl-qwen38-m1-wide-bound-blocker-8b3ed2.md) |
| M2a rows5-8 proposal owner | `e681dfa4d577fe5a4429042f85288e9a257e71ad` — `2026-08-31T00:25:43+09:00` | Qualified the existing exact rowtile producer for wide NextN proposal heads. It removes one M1 fallback but did not make global K3 C6-C8 non-regressive. | [`entry`](../worklog/entries/20260830T152531.233311Z-lhl-qwen38-m2-proposal-head-owner-9f7913.md) |
| M2b-M2g accept-window diagnostics | No retained code or standalone Git evidence commit. Results were local probes under `.worklog/` and `/tmp`; reverted experiments are summarized by the later M2h entry. | Marker decomposition, cross-thread copy probe, and stack capture rejected the initial host/pageable-copy explanation. This is a historical reproducibility limitation: the committed entry records hashes and observations, but the raw probes are not durable repository artifacts. | [`M2h entry`](../worklog/entries/20260830T180550.268739Z-lhl-qwen38-m2h-pinned-null-23353c.md) |
| M2h pinned staging | `f982f4d2d4a2ee41c6d4db5d91ebe6fabf4fa670` — `2026-08-31T03:06:13+09:00` | Recorded the measured-null pinned-host experiment after reverting its implementation. The copy waited on queued GPU work; host staging was not the blocker. | [`entry`](../worklog/entries/20260830T180550.268739Z-lhl-qwen38-m2h-pinned-null-23353c.md) |
| M2i HIP API/copy/kernel attribution | No dedicated commit or compact artifact. Raw `rocprofv3` outputs lived at `/tmp/m2i-trace/gfx1151/`; their interpretation first became durable in M2j. | Measured the accept window at about 98% GPU busy and named Q4/Q5/Q6 verification work rather than host idle time. Later review must treat this as cited raw-only evidence, not as a self-contained retained artifact. | [`M2j entry`](../worklog/entries/20260830T195930.540628Z-lhl-qwen38-m2j-q4-owner-51ae23.md) |
| M2j Q4 rows2-16 owners | `9d37394f2509e4f32f521755216f25fa353a3fc7` — `2026-08-31T04:59:59+09:00` | Promoted bit-exact low-VGPR/shared-B2W2 owners and retained the measured C1-C8 lift. | [`artifact`](../benchmarks/results/2026-08-31-gfx1151-qwen38-mtp-q4-verify-owner-retained.json), [`entry`](../worklog/entries/20260830T195930.540628Z-lhl-qwen38-m2j-q4-owner-51ae23.md) |
| M2k Q5/Q6 screen | `b17178215e1993680b19a45886386a354df94add` — `2026-08-31T05:02:57+09:00` | Recorded exact sibling screens as measured null/negative; no production code changed. | [`entry`](../worklog/entries/20260830T200247.265778Z-lhl-qwen38-m2k-q5q6-screen-4a2805.md) |
| M3 width-1 streaming | `aeb391dbbb2adea1520923b71a53635ebbdbd252` — `2026-08-31T03:05:23+09:00`; `c1612d2c468c622705df2a1f4415a2b319856a35` — `2026-08-31T05:43:50+09:00`; `46b117e106d558de1a9a9b8d6ba3523c0c51af5c` — `2026-08-31T13:45:40+09:00` | Added validator support, promoted production Q4_K_M C1 streaming, then marked the checklist item closed. | [`artifact`](../benchmarks/results/2026-08-31-gfx1151-qwen38-mtp-c1-streaming-width1-retained.json), [`entry`](../worklog/entries/20260830T212000.000000Z-lhl-qwen38-m3-c1-streaming-retained-77c41b.md) |
| M4 width-4 streaming | `38e781b719f06793f8469182c4ce04060b94893f` — `2026-08-31T06:42:13+09:00`; `46b117e106d558de1a9a9b8d6ba3523c0c51af5c` — `2026-08-31T13:45:40+09:00`; corrected by `a3ffbd8f8790fa4530523210b0941b89d9bfbdca` — `2026-08-31T17:18:47+09:00` | Retained explicit width-4 streaming and closed the checklist item. The review correction reclassified changed-acceptance M4 as explicit T3 rather than an automatic production numerical/task promotion. | [`artifact`](../benchmarks/results/2026-08-31-gfx1151-qwen38-mtp-c4-streaming-retained.json), [`entry`](../worklog/entries/20260830T223000.000000Z-lhl-qwen38-m4-c4-streaming-decision-4e91c2.md), [`correction`](../worklog/entries/20260831T081311.905286Z-lhl-qwen38-cn-review-0462c4.md) |
| P1 prefill C2 | `88160d1c863262ec213ab3179f15966d631519a5` — `2026-08-31T06:58:21+09:00` | Closed with the measured Q4 weight-stream/repeated-M-tile floor; the scheduling experiment was reverted. | [`artifact`](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-c2-scaling-blocker.json), [`entry`](../worklog/entries/20260831T001500.000000Z-lhl-qwen38-p1-prefill-c2-blocker-2a7f3d.md) |
| P2 prefill C8 | `64fec87a87291004d49fd205bb842d7febebe84f` — `2026-08-31T07:08:00+09:00` | Closed with the dequantization/LDS/issue blocker and opened P3 under its written condition. | [`artifact`](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-c8-current-trace-blocker.json), [`entry`](../worklog/entries/20260831T011000.000000Z-lhl-qwen38-p2-c8-trace-p3-open-9c4d7e.md) |
| P3 integer MMQ | `07528c87603ec061a4e830cfee79901895cc367e` — `2026-08-31T07:11:43+09:00` | Closed measured-negative using already-built shared-MMQ bodies; no integer body beat selected WMMA. | [`artifact`](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-p3-int-wmma-negative.json), [`entry`](../worklog/entries/20260831T015500.000000Z-lhl-qwen38-p3-int-wmma-closed-5f8a1b.md) |
| M5 whole-batch AR seam | `1f1f360e307e971c85ff83e2cebe75d59b999aed` — `2026-08-31T13:45:16+09:00`; artifact/test gap repaired by `a3ffbd8f8790fa4530523210b0941b89d9bfbdca` — `2026-08-31T17:18:47+09:00` | Added the engine-surface over-width AR route. Server admission made it inert; the review added an engine-loop integration test and committed the missing measured evidence. | [`entry`](../worklog/entries/20260831T053000.000000Z-lhl-qwen38-m5-routing-blocker-decision-6b2e9f.md), [`review artifact`](../benchmarks/results/2026-08-31-gfx1151-qwen38-reviewed-current-head-c1c8.json) |
| Original closeout | `b768516f2b387c8a83f4c97376ca243d663f9c02` — `2026-08-31T13:45:58+09:00` | Marked 11/11 items closed. Its immutable entry contains stale metric-boundary, C2, M4, and terminal-mechanism statements; use the correction row below when interpreting it. | [`entry`](../worklog/entries/20260831T061000.000000Z-lhl-qwen38-campaign-closeout-checkpoint-3d7c8a.md) |
| Post-closeout correction and C=N review | `a3ffbd8f8790fa4530523210b0941b89d9bfbdca` — `2026-08-31T17:18:47+09:00` | Re-ran current-head all-ten AR/MTP and prefill, corrected metric boundaries and scopes, documented flattened C1-C4 verification, and identified C6/C8 one-pass K1 candidates without promotion. | [`artifact`](../benchmarks/results/2026-08-31-gfx1151-qwen38-reviewed-current-head-c1c8.json), [`entry`](../worklog/entries/20260831T081311.905286Z-lhl-qwen38-cn-review-0462c4.md) |

### Primary compact-artifact digests at this audit cut

These digests cover committed compact artifacts, not the larger `/tmp` raw
sources whose path, size, and digest are recorded inside the artifacts.

| Scope | SHA-256 |
| --- | --- |
| M0 | `e9626eac320a010a1021a77792066953a70a8a0097bd32d0bbcf07131bd6ab8c` |
| M1 | `689506c43eae8cd6b7236629c71836e9d5bb3b24148ade04ee1081415e77b1a6` |
| M2j | `9ee7953ef5c98052e76152b715604e9d0484c6cbd61e9e02f93995fb7e0b9057` |
| M3 | `681cd7c0436a6e6505cbb81553221533979cfa013c2331add743ba179ffede62` |
| M4 | `ddfe8f0bdb50cfb035c517ae03d72e6aeb4054b701c90bd2ff157412aef98078` |
| P1 | `5cc4697da615cde0a71e81ead66f83ecef40d44b45dff30d685bee71e5c46200` |
| P2 | `6c2f81540c175fb540b4148be4e2765926a1e0971b267709f72c702fc64af7ae` |
| P3 | `3c7676004c67f364a16842bb0d8c0a23175f72240368ea84111393eaba83b0a9` |
| Post-closeout review / M5 repair | `a3c494fd555baec66e2b7689bb4bc619a0012af251bd1e7788c82ae9f9a25fa6` |

### Audit limitations and revisit rules

1. M2b-M2g and M2i are not self-contained repository evidence. Preserve their
   conclusions as historical attribution, but re-profile on the exact checkout
   before using them for a new performance claim.
2. Worklog timestamps describe when an entry was authored; the table above uses
   Git committer timestamps to identify when evidence became durable.
3. The compact M2j/M3/M4 artifacts include all prompt cells in their raw
   sources, but their original prose headlines used six non-heldout arithmetic
   means. Cross-engine comparisons must use the all-ten total-token/summed-wall
   boundary in the post-closeout review artifact.
4. M4 is an explicit T3 result because acceptance changed. Generated-ID equality
   alone cannot promote it to automatic production.
5. Every successor optimization must append a new immutable entry and artifact;
   do not rewrite the historical entries or reinterpret raw-only evidence as a
   retained gate.

### Post-audit successor ledger

| Scope | Exact commit and committer timestamp | Result | Durable evidence |
| --- | --- | --- | --- |
| C6/C8 K1 direct verifier state | `5f7b3cb6b1b193c134ca93799c76be30e3a7084e` — `2026-09-01T01:14:52+09:00` | Bound read-only packed-verifier roots to the stable resident Conv/GDN slab, removing 9,216 C8 imports / 8.0 GB. Clean same-process MTP improves C6 35.458→35.966 (+1.43%) and C8 41.842→42.594 tok/s (+1.80%); both remain ~0.899x AR and unpromoted. | [`artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-c6c8-direct-verifier-state-retained.json), [`implementation entry`](../worklog/entries/20260831T161329.846325Z-lhl-qwen38-c6c8-direct-verify-state-043f5a.md), [`publication entry`](../worklog/entries/20260831T162416.883411Z-lhl-qwen38-c6c8-direct-state-publication-bc13f3.md) |
| C6/K1 R12 dual-Q4 verifier owner | `ff2e8423bdd109a6b90f4d19c40dc0b4d3c26dff` — `2026-09-01T02:22:09+09:00` | Replaced the R8+R4 gate/up chain with one exact two-wave R12 WMMA+SiLU owner. Clean C6 improves 35.956→37.130 tok/s (+3.26%, every category positive); C8 is an unchanged control. C6 remains 0.9289x AR and unpromoted; the pair owner stays excluded at R16. | [`artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-c6-r12-dual-wmma-retained.json), [`implementation entry`](../worklog/entries/20260831T171926.381120Z-lhl-qwen38-c6-r12-dual-wmma-4f715e.md), [`publication entry`](../worklog/entries/20260831T173126.288409Z-lhl-qwen38-c6-r12-dual-wmma-publication-1be814.md) |
| C8/K1 R16 shared-B2R1 down owner | `6eb922c90dd4bb4d528e9d1b5272d90db504f133` — `2026-09-01T03:31:27+09:00` | Replaced the 128-row-capacity shared-B2W2 K17408/N5120 down owner with an exact 32-row-capacity shared-B specialization. Leaf 0.735→0.534 ms; clean C8 42.571→43.225 tok/s (+1.54%, every category positive). C6 is unchanged; C8 remains 0.9148x AR and unpromoted. | [`artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-c8-r16-shared-b2r1-retained.json), [`implementation entry`](../worklog/entries/20260831T182934.577633Z-lhl-qwen38-c8-r16-shared-b2r1-3dd8b5.md), [`publication entry`](../worklog/entries/20260831T184018.797123Z-lhl-qwen38-c8-r16-shared-b2r1-publication-23f53d.md) |
| C8/K1 R16 shared-B2R1 narrow-V extension | `1f4687cab17ac8dc341e12134d2870221429eb4f` — `2026-09-01T04:13:10+09:00` | Extended the exact owner only to K5120/N1024 after rejecting four peer shapes. Leaf 0.191→0.110 ms; clean C8 43.234→43.421 tok/s (+0.43%, every category positive). C6 is unchanged; C8 remains 0.9192x AR and unpromoted. | [`artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-c8-r16-shared-b2r1-narrow-retained.json), [`implementation entry`](../worklog/entries/20260831T191134.431488Z-lhl-qwen38-c8-r16-shared-b2r1-narrow-b88453.md), [`publication entry`](../worklog/entries/20260831T192213.298947Z-lhl-qwen38-c8-r16-shared-b2r1-narrow-publication-dc9c0b.md) |
| C6/C8 K1 ten-iteration closeout | `be72258c12af255a992e083fca1c0990c9877d05` — `2026-09-01T04:26:52+09:00` is the final retained source before this documentation unit. | Four exact units cumulatively improve clean C6 35.458→37.074 tok/s (+4.56%) and C8 41.842→43.421 (+3.78%); the loop metric rises 0.88256→0.91779 (+3.99%). The 1.15x target remains blocked: the clean endpoint needs another 19.2%/20.1% full-wall reduction at C6/C8. Existing exact Q4 R16 owner geometries are exhausted, Q5 true R16 is retained, and exact Q6 R8+R8 is the measured winner. Reopen with a multi-family packed-verifier dataflow, not another single-owner morphology. Automatic serving remains K0. | [`closeout artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-c6c8-k1-ten-iteration-closeout.json), [`entry`](../worklog/entries/20260831T194251.679968Z-lhl-qwen38-c6c8-k1-ten-iteration-closeout-f9ae7d.md) |

## 5. Order

Planned order was `X1` -> `M0` -> `M1` -> `M3` -> `M2` -> `M4` -> `P1` ->
`P2` -> `M5` -> `P3` (`R1` closed at campaign open). Historical execution
**deviated**: M2 proposal/Q4 owner work landed before M3. The evidence remains
independently scoped, but the prior claim that every item ran in recorded order
was inaccurate. X1 and M0 were completed before the expensive M1 unit.

## 6. Non-goals

- No AR decode reopening. C3-C8 lead; C1/C2 blockers stay closed.
- No prefill work at C1/C3-C7. Those six widths are within 4.3-9.5% and are not
  worth a kernel campaign.
- No FP4/ROCmFPX, Unsloth UD, or ngram-replay parity rows (parity campaign
  decision 1A stands).
- No external vendoring or fork porting. X1 is a reading pass that produces a
  table, not code.
- No benchmark gaming: no prompt-conditioned tuning; every acceptance/speed
  claim validates on the full multi-prompt suite plus category heldouts, and
  every MTP speedup claim uses a true no-MTP AR baseline from the same protocol.

## 7. Document ownership

- [`QWEN38-GFX1151-PARITY-CAMPAIGN.md`](QWEN38-GFX1151-PARITY-CAMPAIGN.md) owns
  the closed parity punchlist and its terminal blockers; this campaign owns
  their successors.
- [`QWEN38-Q4KM-MTP-ACCEPTANCE.md`](QWEN38-Q4KM-MTP-ACCEPTANCE.md) owns the
  E0-E6 implementation ladder and E6 automatic promotion.
- [`CONCURRENCY2-GFX1151-MTP-DYNAMIC-ADMISSION.md`](CONCURRENCY2-GFX1151-MTP-DYNAMIC-ADMISSION.md)
  owns the D4 bounded-C4 decomposition that M1 supersedes.
- [`GGUF-PREFILL-OPTIMIZATION.md`](GGUF-PREFILL-OPTIMIZATION.md) owns prefill
  kernel lineage and the LCP-3B continuation condition.
- [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), [`TESTING.md`](TESTING.md),
  and [`BENCHMARK.md`](BENCHMARK.md) own gates and evidence format.
- [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md) owns the hardware constraints
  used to size P3.

## 8. Extension W (opened 2026-09-01): breaking the multi-family packed-verifier dataflow wall

Successor to the closed C6/C8 K1 loop
([closeout](../benchmarks/results/2026-09-01-gfx1151-qwen38-c6c8-k1-ten-iteration-closeout.json)).
Same frozen product key and rules as sections 1-4. Objective unchanged:
`min(C6, C8) MTP >= 1.15x own AR` — C6 `>= 45.894`, C8 `>= 54.326` tok/s —
which requires removing another **19.2% / 20.1%** of the full-suite MTP wall
(7.46 s / 8.88 s). The closeout proved no single registered exact family can
supply that; this extension changes the cycle's economics instead of its
leaves.

### 8.1 Why the wall exists (cycle-economics analysis)

All numbers below are from the closeout artifact and the reviewed matrix
([L9-equivalent](../benchmarks/results/2026-08-31-gfx1151-qwen38-reviewed-current-head-c1c8.json)),
same host, model, and protocol.

**K1 has a low arithmetic ceiling and we pay double for it.** At the clean
endpoint, C8/K1 commits 1,540/1,610 = 95.65% of single candidates, i.e.
**~1.956 committed tokens per request per cycle**. Even a free proposal +
accept path therefore caps K1 at ~1.96x own AR. Measured 0.9192x means each
speculative cycle costs **~2.13 matched-AR-step equivalents**
(C6: 1.956/0.9290 = 2.11). Reaching 1.15x at K1 requires the cycle to cost
<= 1.70 AR-steps — a 19-20% cycle-cost cut, exactly the closeout's wall gap.
Leaf morphology is exhausted at that scale; the extra ~0.43 AR-steps are
distributed across proposal, R12/R16 verify, and accept/commit stages.

**The stronger lever is committed tokens per weight sweep, and rows-scaling
verify cost blocks it.** In the bandwidth-bound regime, AR at C8 commits 8
tokens per model sweep. K1 commits ~15.7 per cycle; K3 at the historical
78.894% acceptance would commit ~3.37/request = **~27 per cycle** — enough to
absorb even today's cycle overhead — but measured K3 C5-C8 runs at only
0.75-0.81x own AR because the R20-R32 verify shapes lose every R<=16 owner
(M1 trace: Q6-planar direct gemv ~36.8 ms/launch, Q4 falling back to
`wmma_prefill`) and wide proposal sums ballooned +198..+264%. Our verifier is
a ladder of row-tile-specialized exact owners: unbeatable leaf-for-leaf at
R<=16, but its aggregate cost grows with rows instead of staying sweep-bound,
so depth cannot amortize.

**External engines prove the target rate is physically reachable on this
host.** From the standardized `Q4_K_M` matrix
([survey](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md)): stock HIP llama.cpp K3
reaches **56.222 tok/s at C8 = 1.854x its own AR** (and 46.084 = 1.736x at
C7); Laurent reaches 50.837 (1.114x) at C8 and 37.154 (1.312x) at C6. Stock
HIP's absolute C8 MTP exceeds our 54.326 target while streaming the same
17.1 GB of weights on the same 256 GB/s part. Per
[X1](EXTERNAL-MTP-BATCHING.md), every surveyed engine (a) verifies all
in-flight requests in **one** target forward through a general M-row
GEMM/MMQ path whose cost is nearly row-invariant at these widths, (b) batches
the drafter across all slots per depth step, and (c) caps by token budget,
not width. At C6 no external absolute MTP reaches our 45.894 target (our AR
is far higher than theirs), so C6 rests on the ratio evidence plus
tokens-per-sweep arithmetic, not on an external existence proof.

**What we need to be able to scale (the requirement, stated once):**

1. verify-pass cost must become approximately **row-invariant** across
   R8-R32 (sweep-bound, not row-bound), per quant family;
2. the proposal pass must stay a small fraction of a target sweep at wide
   rows and deeper K;
3. accept/selected-state/KV commit must not add an operation-complete stage
   per cycle.

Once (1)-(3) hold, depth (K2/K3) converts directly into committed tokens per
sweep and the 1.15x gates follow arithmetically. Without (1), no further leaf
or single-family work can close 19-20% — that is the named blocker restated
as a requirement.

### 8.2 W punchlist

Rules unchanged (frozen protocol, full-suite + heldouts, no
prompt-conditioned tuning, exact ownership + strict fallback, complete
production gates before any promotion, automatic C6/C8 serving stays K0
throughout). Additional extension rule from the closeout's reopen condition:
**every W implementation unit needs a sized full-wall bound before code** —
a measured projection of the wall seconds it can remove; unsized candidates
are not started.

- [x] W0 **Sweep-economics instrumentation. Done 2026-09-01.** Target-submit
  and complete target/accept-drain timestamps now correlate resident telemetry
  with `rocprofv3` kernel traces and the gfx1151 `FETCH_SIZE` counter. The
  frozen all-ten C6/C8 K1 refresh passes 20/20 exact/engaged/budget cells and
  measures **2.1069 / 2.1423 AR-step equivalents per speculative cycle**
  versus **1.7013** required for 1.15x. (This corrects the opening item's unit:
  2.11/2.13 is per cycle, not per committed token.) The diagnostic R8-R32
  curve passes 6/6 cells but fails flatness: R32/R8 family time is
  **2.27x Q4 / 12.59x Q5 / 15.69x Q6**. Actual video-memory fetch at R32 is
  **1.07x / 34.62x / 26.69x** each family's resident target bytes; Q5/Q6
  re-sweep while Q4's bytes are already flat but its time is not. Applying
  W1's 1.25x ceiling sizes **33.9 s C6/R24 / 44.8 s C8/R32** of family time
  across the historical 71-cycle full suite, well above the remaining K1
  wall gaps. W1 entered from this bound and is now closed measured-blocked below.
  W2/W5's overlapping non-family upper bound is
  2.47/3.39 s; W4's post-W1 proposal excess is only ~0.85/0.44 s. No perf
  claim. [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w0-sweep-economics.json).
- [x] W1 **Row-invariant wide verify owners (R20-R32) — measured blocked.**
  Screened all existing Q4/Q5/Q6 bodies, retained a default-off two-wave Q6
  owner, and exhausted bounded wave/slab geometries. The owner passes full T2
  C6/C8 gates and cuts R32 Q6 device time 33.5%, but measures 3.432x R32/R8
  versus the <=1.25x gate. Passing requires 63.6% more R32 reduction. Even
  perfect overlap of serial decode/LDS and WMMA stages bottoms at 1.716x, so
  double buffering cannot close W1. Full-suite owner combinations were neutral
  or negative. W1 reopens only if W2 changes the stage/multi-family lower bound;
  strict fallback and automatic K0 remain unchanged.
  [`Row-curve evidence`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w1-q6-two-wave-rowcurve.json)
  [`Pipeline bound`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w1-pipeline-bound.json).
- [x] W2 **Single-sweep multi-family layer dataflow — measured blocked.**
  W1 exhausted within-tensor one-decode owners. Q4/Q5/Q6 projection tensors
  are distinct, and the runtime already attempts mixed pair/triple and gate/up
  owners, so cross-family co-scheduling cannot share mandatory weight bytes.
  Current R32 target wall is 51.223 ms host / 49.016 ms summed kernels; even
  deleting the full 2.207 ms gap saves only 4.31% of target and less of full
  wall, versus 19.2%/20.1% required. The gfx1100 B4 FFN megakernel was 2.66x
  slower on GPU, so a giant fused retry is not justified by this bound.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w2-multifamily-bound.json).
- [x] W3 **Depth reopen behind W1 — dependency blocked.** W1's written
  flatness prerequisite failed: the best Q6 owner is 3.432x R32/R8 versus
  1.25x required. Current valid K3 remains below retained K1 at both widths
  (C6 0.839x vs 0.929x; C8 0.771x vs 0.919x own AR). A width-depth admission
  table cannot change target arithmetic, so deeper K cannot reopen until a
  successor meets W1 flatness or invalidates that dependency.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w3-depth-dependency-bound.json).
- [x] W4 **Proposal economics at wide rows and depth — measured blocked.**
  W0 measures 34.46/30.14 ms proposal per C6/C8 cycle. Completely free
  proposal over 71 cycles saves only 2.45/2.14 s, versus 7.46/8.88 s wall
  gaps; the portion above the hypothetical post-W1 <=15% gate is only
  0.85/0.44 s. W1's flat-target denominator is itself blocked. Hot-vocabulary
  or fused top-1 policy would be T3 and can reduce acceptance, while zero cost
  still cannot close the objective, so implementation is not justified.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w4-proposal-bound.json).
- [x] W5 **Accept/commit stage fusion — below-threshold blocker.** W0 measures
  distinct provider-update plus selected-commit ownership at 0.845/0.993 s
  over 71 C6/C8 cycles, only 1.42%/1.36% of K3 wall and below W5's written 2%
  drop threshold. Even free stages cover only 11.3%/11.2% of the remaining
  gaps. The larger accept interval is 98% GPU-busy verifier drain already
  owned by W1; crediting it here would double-count target math. No fusion is
  justified.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w5-accept-commit-bound.json).
- [x] W6 **Width-6/8 prompt-streaming engagement — compound blocked.** C6
  streaming repeatedly measured 38.616/38.599/38.602 tok/s (~+4.0%), while C8
  never engaged. Even granting C8 the same 4% gain lifts retained K1 ratios
  only to 0.966x/0.956x at C6/C8, still below 1.15x. Implementing C8 engagement
  cannot close the compound objective; reverted C6 evidence remains preserved.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w6-prompt-streaming-bound.json).
- [x] W7 **(Conditional) multi-candidate/tree drafts — not opened.** W1's
  flatness prerequisite failed at 3.432x versus 1.25x, candidate budget 1
  remains the certified protocol, and no production-qualified prompt-independent
  tree policy exists. Extra candidates would add rows to the blocked verifier,
  so W7 stays closed until a successor satisfies both prerequisites.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w7-conditional-close.json).

### 8.3 Order and success criteria

`W0` -> `W1` -> (`W2` and `W4` in parallel, both sized by W0) -> `W3` ->
`W5`/`W6` opportunistic -> `W7` conditional. W6 may run any time.

The extension closes when `min(C6, C8) >= 1.15x own AR` on the frozen
protocol with all exactness/route/budget cells passing, or when every W item
carries a measured named blocker whose sized bounds sum below the remaining
gap. Any promotion to automatic serving additionally requires the complete
production numerical, determinism, isolation, task, lifecycle, and serving
gates — the W punchlist alone never changes admission policy.

## 9. Extension Y (opened 2026-09-01): breaking the prefill sweep-multiplicity wall

Successor to the closed P track (P1/P2 named blockers, P3 measured negative).
Same frozen product key and rules as sections 1-4 and 8. Objective: close the
campaign's goal-2 prefill cells to best-external parity —
**C2 `>= 211.888`** and **C8 `>= 305.847`** prefill tok/s (Laurent's frozen
2026-08-30 rates; the C8 value is P2's existing named target) — from the
reviewed refresh's 139.8 / 247.3, i.e. **+51.6% / +23.7%**. The
[survey](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md) refresh shows hipEngine
trailing the best external prefill at all eight widths (-4.2..-11.7% at
C1/C3-C7, -34.0%/-19.1% at C2/C8); gates stay on C2/C8 per section 1, and
section 6's per-width non-goal stands — no width-specific kernels for the
other six, but their collateral movement under shared-owner changes is
recorded in each retained unit's artifact.

### 9.1 Why prefill trails (byte-multiplicity analysis)

Re-reading the three P closures together names one structural defect rather
than three independent walls.

**Y0 corrects the original `ceil(rows/16)` attribution.**
[P1](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-c2-scaling-blocker.json)
correctly measured GPU-busy ticks and repeated weight traffic, but its
whole-family `ceil(rows/16)` inference did not account for the mixed shared-B
owners already present at current head. Y0 matches each active source-weight
launch group to its `rocprofv3` M-grid. Q4's byte-weighted sweep multiplicity
is **1.00/1.30/1.34/2.02/2.09/1.17/2.87/3.00/4.00** at rows
16/35/48/72/96/256/288/536/1024—not 1/3/3/5/6/16/18/34/64. Shared-B
variants therefore amortize many shapes, including rows256, while the
rows288+ policy/shape mix still repeats Q4 traffic. Per-tile Q4 bandwidth
falls from 104 GB/s at rows16 to 27 GB/s at rows1024 as the issue wall binds.
[`Y0 artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-prefill-y0-sweep-multiplicity.json).

The remaining defect is still **tiles instead of one sweep**, but it is
shape- and owner-dependent rather than uniform. This is the same class of
dataflow defect extension W found in the verify path; Y1 must flatten the
remaining Q4 multiplicity without replacing shared-B owners that already
reach approximately one sweep.

**The dequant/LDS/issue wall is the second wall behind the first.**
[P2](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-c8-current-trace-blocker.json)
measured the standalone owner at 19-24 TF/s (~35% of peak) at rows256+; that
wall binds only once multiplicity ~= 1 makes large-M tiles compute-bound.
[P3](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-p3-int-wmma-negative.json)
closed **data-format-only** change (INT8 WMMA == BF16 at 59.4 TOP/s;
dequant-free bodies hit the same LDS-staging/issue structure at 0.81x). P3
did **not** close loop-order/dataflow change, and did not test INT4 — the
only raised tensor roof on gfx1151 (118.8 TOP/s).

**Measured Y0 bounds supersede the screening arithmetic.** At C2-like
rows35/48, perfect Q4 single-sweep dataflow can remove only **13.1-14.3%**
of tick wall; perfect Q4+Q5+Q6 single-sweep dataflow removes **26.1-27.5%**,
below the **34.0%** wall reduction needed for 139.8->211.888 tok/s. C2
therefore requires post-dataflow issue/fusion work even if Y1/Y2 are played
perfectly. At C8-like rows288, Q4 alone has a **36.6%** wall bound and all
quant families **62.9%**, above the **19.1%** wall reduction needed for
247.3->305.847 tok/s. These are upper bounds, not performance claims.

**The requirement, stated once:** prefill weight bytes swept per tick per
quant family must be ~= 1.0 sweeps regardless of grouped rows (multiplicity
flatness in M), with dequantization performed once per weight tile per sweep.

### 9.2 Y punchlist

Rules unchanged (frozen protocol, exact ownership + strict fallback,
applicable `docs/EXECUTION-PROFILES.md` gate per unit, no
prompt-conditioned tuning, sized full-wall bound before code).

- [x] Y0 **Prefill sweep-multiplicity instrumentation (entry gate for all of
  Y).** The committed capture/analyzer publishes Q4/Q5/Q6/GDN/other timing,
  active swept bytes, byte-weighted multiplicity, per-tile GB/s, effective
  TF/s, tick wall vs HIP-event span, and Y1/Y2 full-wall bounds at all nine
  required row points. It rejects the blanket `ceil(rows/16)` premise:
  current shared-B owners make Q4 multiplicity shape-dependent (1.00-4.00),
  while rows288 still leaves a 36.6% Q4 wall bound. Raw profiler sources stay
  under `/tmp` with hashes in the durable artifact; no performance claim.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-prefill-y0-sweep-multiplicity.json).
- [x] Y1 **Single-sweep (B-stationary) Q4 prefill dataflow.** Restructure the
  Q4 prefill owner family so each weight tile is fetched and dequantized once
  per tick and looped across all M-tiles (workgroup-owns-weight-tile with an
  in-kernel M-loop; persistent variants allowed). Gates: measured
  multiplicity <= ~1.25 from rows16 to rows1024 (flatness in M), exact
  BF16-out parity or production-profile numerical RED, strict fallback
  registered, frozen C1-C8 prefill row re-run with C2/C8 movement recorded.
  **Shared lineage with W1:** M1's wide-verify trace shows Q4 falling back to
  `wmma_prefill<false,2>` at R>16 — a multiplicity-1 M-loop GEMM body is the
  same shape W1 needs for row-invariant R20-R32 verify owners. Build the tile
  machinery once, register it under both keys. First rows288 policy screen:
  forcing the existing 48-column/256-row parent over the retained periodic
  owner regresses complete tick wall **2.69%** despite fewer M-grid sweeps;
  threshold rollback is rejected, so Y1 proceeds to a new one-sweep body.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y1-row288-parent-rejected.json).
  The first new body keeps rows<=384 FP32 accumulators in LDS across K256
  slabs and is bit-exact/one-sweep, but rows288 K5120/N6144 regresses
  **55.1%** versus shared-B2W2; FP32 partial spill traffic is rejected.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y1-lds-accum384-rejected.json).
  Register-resident `<3 output tiles, 8 waves, 3 row tiles/wave>` succeeds
  narrowly: rows288-384 on three measured shapes is one-sweep and bit-exact,
  improves nine leaf cells **6.25-16.24%**, and cuts complete rows288 tick wall
  **2.65%**. Tracked-clean C1-C8 collateral preserves 160/160 control/candidate
  ID rows; C8 combined prompt throughput improves **0.83%** to 239.658 tok/s.
  Retained as a partial Y1 win; rows385-1024 remain open.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y1-q4-b3w8r3-partial-retained.json).
  A 384-thread `<3,12,3>` extension improves the rows513/536/576 down-projection
  leaves **8.58-16.28%** but regresses complete rows536 tick wall **0.73%**;
  scope-reverted. Y1 above rows384 therefore remains a cross-family persistent-
  scheduling problem, not a leaf-only geometry problem.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y1-b3w12r3-full-wall-rejected.json).
  **Closure blocker:** rows1024 exhausts exact one-workgroup ownership at
  gfx1151's 1024-thread/32-wave limit: `<3,16,4>` and hardware-limit
  `<3,32,2>` regress **43.5%/57.6%** despite bit-exact gridY1 execution;
  cross-workgroup ownership requires synchronization or FP32 partial spill,
  whose measured prototype regresses 55.1%. The retained band moves frozen C8
  only +0.83%, so the blocked remainder cannot cover the 19.1% target gap.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y1-rows1024-workgroup-limit.json).
- [ ] Y2 **Sibling-family multiplicity (Q5/Q6/GDN).** The C8 trace remainder
  (Q6 5.13 s, Q5 2.03 s, GDN 1.82 s, other 2.08 s of 27.9 s) becomes the
  binding share after Y1. Extend the single-sweep dataflow per Y0's measured
  multiplicities; opens per family only where Y0 sizes it above ~2% of the
  remaining wall. First Q6 rows288 one-sweep body screens 29.0x faster on
  standard K5120/N10240 but fails strict parity (300,641 BF16 differences,
  max 0.0078125); scope-reverted. It may reopen only as declared T2 with the
  complete production gate.
  [`Artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q6-shared8r3-exact-rejected.json).
  Corrected direct shared4 comparison recovers strict parity: standard Q6
  rows257-384 shared8r3 is bit-exact, 1.95x faster at rows288, and cuts the
  complete rows288 tick wall **3.43%**; retained as a partial Y2 win. Planar
  peers are flat and unchanged. Standard Q5 K6144/N5120 rows257-384 now uses
  the same exact `<8,3,2>` one-sweep geometry: rows288 is bit-exact and
  **1.54x** faster at the leaf, reducing the complete tick
  **1100.915->1068.494 ms (-2.94%)** with the same token. GDN's measured
  64.37 ms (5.48% of rows288 wall) has **zero Y2 multiplicity bound**: all
  five stages launch exactly once per recurrent layer, and recurrence already
  uses the retained compact peer-wave owner (1.42-1.52x over strict direct at
  rows512-4096). Further GDN work enters Y3. Remaining Q5/Q6 row bands keep Y2
  open. Q6 rows33-48 now uses an exact `<3,1,2>` single-sweep owner across
  all three physical shapes: six rows35/48 leaves are bit-exact and improve
  **1.12-5.57x**; complete walls fall **281.276->265.883 ms (-5.47%)** and
  **286.328->269.313 ms (-5.94%)** with unchanged tokens. Tracked-clean
  C1-C8 collateral passes **160/160 exact**; aggregate wall improves **1.74%
  AR** and **0.15% MTP** (individual MTP widths are -0.39% to +0.20% wall).
  Standard Q6 rows49-96 now uses exact `<6,1,2>`: four rows49/64/72/96
  leaves improve **1.31-1.68x**, and complete rows72/96 walls fall
  **408.827->396.426 ms (-3.03%)** and **432.039->423.395 ms (-2.00%)**
  with unchanged tokens. Its tracked-clean C1-C8 collateral is **160/160
  exact** and improves aggregate wall **0.57% AR / 0.17% MTP**. Standard
  Q6 rows385-1024 now reuses exact `<8,3,2>` at gridY2/3: rows536/1024
  leaves improve **1.68x/1.66x**, and complete walls fall
  **1867.136->1810.768 ms (-3.02%)** and **3150.993->3046.379 ms (-3.32%)**
  with unchanged tokens. Tracked-clean C1-C8 collateral is **160/160 exact**
  and improves aggregate wall **0.60% AR / 0.26% MTP**.
  [`Q6 high-row artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-standard-q6-shared8r3-partial-retained.json),
  [`Q6 mid-row artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q6-shared3r1-retained.json),
  Planar Q6 cannot mirror the six-wave owner: `<6,1,2>` fails its first
  actual-weight launch with **HIP 719**. A four-wave `<4,2,2>` replacement
  succeeds across both physical planar shapes at rows65-96: leaves are exact
  and improve **1.53-3.98x**; complete rows72/96 walls fall **5.78%/5.60%**,
  tokens unchanged. Trace: 120 gridY1 hits, 128 threads, VGPR176, LDS16 KiB,
  scratch0. Tracked-clean collateral is 160/160 exact and improves MTP wall
  **0.32%**, but regresses aggregate AR wall **1.05%**; route/export removed.
  Planar Q6 rows65-96 is blocked by complete-suite wall regression.
  [`Q6 rows49-96 artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q6-shared6r1-retained.json),
  [`planar Q6 six-wave blocker`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-planar-q6-shared6r1-launch-blocker.json),
  [`Q6 high-grid artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q6-shared8r3-high-retained.json),
  Q5 rows65-96 exact `<6,1,2>` improves four leaves **1.31-1.35x** and
  complete rows72/96 walls **2.06%/0.86%**, but its tracked-clean collateral,
  while **160/160 exact** and -0.37% MTP wall, regresses aggregate AR wall
  **0.19%**; the route and export are removed. Q5 high-row shared8r3 is
  exact and 1.24x/1.23x faster at rows536/1024,
  but complete rows536 wall regresses **0.07%**. Rows1024 improves its tick
  **3050.717->3045.508 ms (-0.17%)**, but tracked-clean C1-C8 collateral,
  while **160/160 exact** and -0.11% MTP wall, regresses aggregate AR wall
  **0.62%**; both high-row routes are rejected and removed.
  [`Q5 artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-standard-q5-shared8r3-partial-retained.json),
  A narrower Q5 rows65-80 admission is exact and lowers rows72 wall **1.55%**,
  but tracked-clean collateral regresses aggregate wall **0.53% AR / 0.04%
  MTP** despite 160/160 exact cells. Its route/export are removed; this family
  is blocked by complete-suite wall regression rather than local leaf speed.
  [`Q5 rows65-80 artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q5-shared6r1-rows65-80-retained.json),
  [`Q5 rows65-96 artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q5-shared6r1-retained.json),
  [`Q5 rows1024 artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q5-shared8r3-r1024-retained.json),
  [`GDN blocker`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-gdn-no-multiplicity-blocker.json).
  Current-head re-trace still sizes Y2 above threshold: rows72/96 Q5 is
  **2.00 sweeps** and Q6 **2.07/2.26**; rows256 Q5/Q6 is **4.00/7.39**;
  rows536 **4.00/11.66**; rows1024 **4.00/22.67**. Rows256 is the next
  highest-feasibility band because existing one-sweep bodies may apply.
  [`Current ledger`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-current-head-sweep-ledger.json).
  Existing exact Q5 and standard-Q6 `<8,3,2>` owners now include rows256:
  leaves improve **1.55x/1.83x** and their combined complete wall falls
  **887.889->827.647 ms (-6.78%)** with unchanged token; trace confirms both
  at gridY1. Tracked-clean C1-C8 collateral is **160/160 exact** and improves
  aggregate wall **0.79% AR / 0.05% MTP**.
  [`Rows256 artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q5q6-rows256-retained.json).
  Post-retention rows256 Q5 is now **1.00 sweep**; Q6 remains **4.84 sweeps /
  223.79 ms (26.76% wall)**, decomposed as 48 standard gridY1, 64 planar
  shared4 gridY8, and 16 planar plain gridY4 hits. Its family-wide removable
  ceiling is **177.54 ms**. An exact planar `<4,3,2>` owner now covers both
  physical planar shapes at rows256: leaves improve **1.94x/2.48x**, complete
  wall falls **827.488 -> 735.999 ms (-11.06%)**, and the token is unchanged.
  Trace confirms 80 gridY2 hits, 128 threads, VGPR176, LDS16 KiB, scratch0.
  Tracked-clean C1-C8 collateral is **160/160 exact**; aggregate wall changes
  **-0.23% AR / +0.01% MTP** (per-width MTP range -0.51% to +0.28%), passing
  the collateral guard. Post-retention rows256 Q6 is **1.73 sweeps / 132.22
  ms (17.80% wall)** with gridY1/gridY2 only; its remaining one-sweep ceiling
  is **55.68 ms**. Exact wide-only `<4,4,2>` reaches gridY1 and improves its
  leaf **1.45x**; narrow `<4,4,2>` is rejected at **0.94x** and remains on
  `<4,3,2>`. Complete wall falls **734.190 -> 706.580 ms (-3.76%)**, token
  unchanged. Trace: 64 gridY1 hits, 128 threads, VGPR176, LDS16 KiB, scratch0.
  Tracked-clean C1-C8 collateral is **160/160 exact** with aggregate wall
  **+0.18% AR / -0.05% MTP** (per-width MTP -0.41% to +0.18%), passing the
  collateral guard. Final rows256 multiplicity is **1.00 Q5 / 1.08 Q6**.
  The sole residual is 16 narrow-planar gridY2 hits totaling **3.87 ms / 0.54%
  wall**, below Y2's 2% opening threshold; exact gridY1 `<4,4,2>` loses locally
  at 0.94x. Rows256 is therefore closed at the practical exact sweep floor;
  Y2 remains open on other row bands.
  [`Rows256 final ledger`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-rows256-final-ledger.json).
  [`Planar rows256 artifact`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-planar-q6-shared4r3-rows256-retained.json).
  [`Rows256 residual`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-rows256-post-retention-ledger.json).
  A current high-row audit prevents premature Y2 closure: rows288/536/1024
  Q6 remains **4.40/8.18/14.48 sweeps** and **248.25/468.03/870.54 ms**, with
  family-wide removable ceilings **191.83/410.82/810.42 ms**. The retained
  standard owner is active, but planar gridY9/17/32 dominates; exact planar
  high-row ownership remains an open implementation unit.
  [`High-row current ledger`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-high-row-current-ledger.json).
- [ ] Y3 **Post-dataflow issue-wall attack.** With multiplicity ~= 1, large-M
  tiles hit P2's 19-24 TF/s dequant/LDS/issue wall. Re-trace, then attack at
  the algorithm/fusion level (pipelined dequant/WMMA overlap, LDS-staging
  restructure, dual-issue scheduling) per the parity campaign's closing
  instruction — a new high-row Q4 algorithm/fusion, not more threshold
  tuning. Entry: the post-Y1 trace shows compute/issue-bound.
- [ ] Y4 **(Conditional) INT4-WMMA Q4 body.** The only raised tensor roof on
  gfx1151 (118.8 vs 59.4 TOP/s). Opens only if Y3's trace shows
  tensor-rate-bound; P3's INT8 negative is standing evidence that a format
  change without fixing the LDS/issue structure loses, so Y4 never opens
  before Y3 closes.
- [ ] Y5 **(Conditional) non-GEMM prefill tail.** P1.3 traced 1380 launches /
  534.2 ms at M=45 with ~174 ms outside the wmma family. Size with Y0; drop
  if under ~2% of the remaining wall.

### 9.3 Order, priority, and success criteria

`Y0` -> `Y1` -> (`Y2` and `Y3` as sized) -> `Y4`/`Y5` conditional.

MTP (extension W) remains the campaign's priority-1 track and W0 stays the
next unit. Y touches disjoint owners (prefill bodies vs verify/proposal/
accept), so Y units may interleave with W without contention; the W1/Y1
shared tile machinery is the deliberate exception and should be built once.

The extension closes when C2 `>= 211.888` and C8 `>= 305.847` prefill tok/s
on the frozen protocol (same prefill-dominant boundary: prompt tokens over
barrier-to-last-completion wall, one generated token, API overhead included
for every engine), or when every Y item carries a measured named blocker
whose sized bounds sum below the remaining gap. External rows stay frozen at
their 2026-08-30 pinned commits; any lead claim against moving upstream
requires a re-measured external matrix.
