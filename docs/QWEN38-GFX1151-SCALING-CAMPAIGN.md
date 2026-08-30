# Qwen3.8-27B gfx1151 Scaling Campaign (MTP batch scaling + prefill)

Status: **open**, 2026-08-30. Successor to the closed
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

### C. Q4 is a major verify cost and has no rowtile owner above R12

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
- [ ] M1 **Single-group wide verify.** Lift the width-4 partition so one cycle
  covers all due requests. Replace the eleven scattered cap expressions with
  one capability-owned bound; update `partition_max_requests`, `claims_fit`,
  `proposal_widths`, `max_requests`/`max_frontier_rows`, and replace or rename
  the misleading `GGUF_SPECDEC2_MTP2_C4` gate. Resize and re-key
  `_batch_accept_resources()` through C8/R32: its `TargetVerifyBufferSpec`,
  remaining-decode tensor, and packed-payload tensor must all cover eight
  requests and 32 rows, with allocation/reuse/teardown/pressure tests. Re-add
  the R20-R32 target row buckets removed as unengaged. Because one proposal now
  has rows5-8, qualify the rows5-8 proposal-head owner or explicitly measure and
  record the direct fallback cost; do not assume the rows2-4 policy transfers.
  Binding gates: exact control/ownership in every profile, all 80 cells exact,
  acceptance unchanged at 78.894%, registered strict fallback, and C1-C4
  non-regressive. Targets: C5 `18.657 -> >= 33`, C8 `28.577 -> >= 45` (its own
  AR is 45.936). These are **intermediate** partition-lift targets — roughly
  own-AR parity — not the campaign goal.
- [ ] M2 Per-row verify cost (A1). Using M1's per-quant attribution, close the
  gap between MTP ms/target-row and AR ms/row at matched width. M1+M2 jointly
  own the final `MTP >= 1.15x own AR` gates: C4 `>= 34.596`, C5 `>= 40.876`,
  C6 `>= 45.892`, C7 `>= 49.579`, and C8 `>= 52.827` tok/s, or a measured named
  blocker for each missed cell. Q4 owners above R12 (C) open here **only if**
  M1's trace names Q4 as the binding class; the leaf-only stop and
  operation-complete revisit set the entry condition — weighted GPU work must
  not rise.
- [ ] M3 **C1 coverage** (B). Extend `_physical_prompt_streaming_widths()` to
  admit width 1 without broadening the unqualified `>4` range; add the width-1
  package-policy key and qualify the rows1 proposal rowtile owner. Re-screen the
  reusable native target graph for the production route. Use M0's refreshed
  production-D24 C1 result as the matched baseline. Target `>= 18.191` (the
  strict natural25 result, used as an aspiration rather than a comparator),
  stretch `>= 21.126` (external). Resolver, policy-miss, and strict-C1 tests
  must prove strict automatic behavior is unchanged.
- [ ] M4 C4 prompt-streaming acceptance blocker. Streaming at C4 changed
  acceptance 628/796 -> 624/800 and was rejected. Decide explicitly whether the
  binding contract is exactness of the replayed prompt or of the acceptance
  count, then either qualify a streaming variant that preserves it or record the
  contract as the terminal blocker.
- [ ] M5 Concurrency-aware admission. **Only after M1-M3.** Route by measured
  physical concurrency and cycle economics, not a global switch: our own C4 sits
  at 0.9768x AR and the external MikeVeerman result is 2.23x at C1 and 0.84x at
  C4. This is admission policy work, not acceptance tuning.

### P — prefill (secondary track, two cells only)

- [ ] P1 **C2 root-cause.** Explain why prefill does not scale C1 -> C2 when AR
  does. Trace the C2 packed prefill grouping directly against C1 and C3 under
  one committed protocol; the existing C2 attribution measured GPU-boundness but
  never the missing scaling. Target `142.806 -> >= 211.888` or a named blocker.
- [ ] P2 **C8 high-row Q4.** Re-trace the current head first (E: the final
  frozen C2/C8 paths lack a current trace, and P1.3's 360 ms figure predates the
  retained low-VGPR work). Then attack the named high-row Q4/device
  algorithm with a real algorithmic change — N-split/split-K partitioning or a
  new fusion — under the strict/production gates plus a `rocprofv3
  --kernel-trace` entry. Target `247.216 -> >= 305.847` or a named blocker.
  No threshold or scheduler ladders; those are exhausted and recorded as such.
- [ ] P3 Integer-MMQ continuation (F). Conditional on P2's trace showing dequant
  ALU dominance **and** a written sizing estimate that assumes INT8 WMMA equals
  BF16 WMMA rate on gfx1151. Must reproduce the shared MMQ tile/decomposition
  per item 19's recorded condition. If P2 shows a bandwidth or occupancy wall
  instead, P3 closes unopened with that reason.

### R — refactor ledger

- [x] R1 **Done and corrected on 2026-08-30.** The width-4 MTP partition,
  eleven `qwen35_gguf_mtp2.py` cap expressions, C4-only accept owner,
  per-subgroup sequential loop, and `GGUF_SPECDEC2_MTP2_C4` gate are recorded
  as [`REFACTOR.md`](REFACTOR.md) RF-OI5 with an explicit removal condition tied
  to M1's outcome, success or blocker.

## 5. Order

`X1` -> `M0` -> `M1` -> `M3` -> `M2` -> `M4` -> `P1` -> `P2` -> `M5` -> `P3`,
(`R1` closed at campaign open). X1 and M0 are cheap and de-risk the
expensive M1 unit. M3 is placed early because it tests the leading coverage
hypothesis and is the largest single-cell deficit in the matrix (-63.03%).

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
