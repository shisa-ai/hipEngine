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

- hipEngine isolated gfx1151 prefill is **~1294/1358/1366 tok/s at 512/1K/4K**
  (LCP tranche, `docs/GGUF-PREFILL-OPTIMIZATION.md`), while the server matrix
  shows 71.5 tok/s at C1 — the first prefill wall is expected to be
  serving-path ownership/activation/API boundary, not bulk prefill kernels.
  CONCURRENCY2 T0.2 already implicated legacy prompt activation and cold
  provider streaming (981.9 ms open, 189 allocations, 0.052 ms GPU).
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
- Post-CONCURRENCY2 state: production C2/K3/context1-128 is automatic at
  17.031 vs 14.887 tok/s (1.1441x AR); realized-singleton C1/K3 is 15.769
  (+59.65%); C3+ and scope misses remain K0.

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

- [ ] P1.1 Protocol parity audit: recover the exact per-row server flags and
  harness boundaries for all six standardized rows (batch/ubatch/context/
  threads/poll/cache flags), then publish a flag-matched control matrix
  separating config deltas from code deltas. The mainline-Vulkan 87.43 vs
  forks ~136-139 C1-prefill cluster is a suspected flag artifact and must be
  resolved before any porting decision.
- [ ] P1.2 Commit bisect: Laurent `d222767c..c28d538d` and Nathan
  `add19980..0eb52805` on the standardized matrix under P1.1-matched flags;
  attribute each prefill/AR/MTP delta to a commit or a flag; record port/
  non-port verdicts with source file + commit per AGENTS.md lineage rules.
- [ ] P1.3 hipEngine serving-path attribution: quantify the C1 prefill
  71.5 tok/s server vs ~1294 tok/s isolated LCP leaf gap (ownership,
  activation, API boundary, packing, timing owner) with a measured breakdown
  that names the top walls; extends CONCURRENCY2 T0.2.

### P2 — prefill C1-C8 parity

- [ ] P2.1 Close the serving-ownership gap from P1.3; target C1 complete-wall
  `>= 138.95` tok/s with the exact/correctness gate green.
- [ ] P2.2 C2-C8 prefill parity at the frozen protocol; each cell closes at
  its frozen winner (194.07/180.09/192.54/217.39/243.52/245.61/296.82) or a
  measured named blocker.
- [ ] P2.3 Bulk-prefill kernel follow-through (port/adapt external wins from
  P1.2) only for walls P1.3 places below the serving boundary; each retained
  kernel change carries the strict/production gate and rocprof trace per
  `docs/KERNELS.md`.

### P3 — AR decode C1/C2/C8 (defend the C3-C7 lead)

- [ ] P3.1 Close C1 `9.637 -> >= 11.336`, C2 `14.247 -> >= 20.056`, C8
  `39.057 -> >= 45.751` complete-wall tok/s, each with a measured win or named
  blocker, and no regression vs frozen C3-C7 (20.731/26.216/31.19/34.564/
  36.592).

### P4 — MTP K3 C1-C8 parity via the acceptance campaign

- [ ] P4.1 Execute `docs/QWEN38-Q4KM-MTP-ACCEPTANCE.md` E0-E5 as written;
  its own binding gates and statistical discipline apply unchanged.
- [ ] P4.2 Frozen MTP parity targets: C1 `>= 21.277`, C2 `>= 32.378`, C3
  `>= 27.515`, C4 `>= 27.015`, C5 `>= 32.74`, C6 `>= 36.023`, C7 `>= 42.304`,
  C8 `>= 54.834`; each cell closes with a measured win or named blocker.
  Working theory: cycle efficiency 47-54% → 70-77% via E1/E2.
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
