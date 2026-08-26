# CONCURRENCY2 gfx1151 MTP Tuning Campaign

- Status: **plan frozen 2026-08-26; no GPU work started**
- Hardware lane: **Radeon 8060S / `hip_gfx1151`** only (two 8060S hosts are
  independent lanes; W7900 is the separate
  [`MTP-CONCURRENCY2-DUAL-PROMOTION.md`](MTP-CONCURRENCY2-DUAL-PROMOTION.md)
  campaign)
- Primary model: **Qwen3.8-27B `Q4_K_M`, BF16 KV** (SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`)
- Normative dependencies: [`PLAN.md`](PLAN.md), [`BENCHMARK.md`](BENCHMARK.md),
  [`TESTING.md`](TESTING.md), [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
  [`SPECDEC2.md`](SPECDEC2.md), [`SPECDEC2-PERF.md`](SPECDEC2-PERF.md),
  [`QWEN38-Q4KM-MTP-SERVING.md`](QWEN38-Q4KM-MTP-SERVING.md),
  [`MTP-LLAMACPP-PARITY.md`](MTP-LLAMACPP-PARITY.md), and
  [`OLMX-IDEAS.md`](OLMX-IDEAS.md). Coordination surface:
  [`GFX1151-TUNING-LANDSCAPE.md`](GFX1151-TUNING-LANDSCAPE.md).

This is the ordered tuning punchlist for closing the measured gap between
hipEngine's exact MTP serving on gfx1151 and (a) our own direct-leaf MTP speed
and (b) external Strix Halo MTP results. It is a measurement-first campaign:
every phase names its evidence and exit gate before implementation.

## 1. Why this campaign exists — three measured gaps

### G1 — C1 serving is ~39% below our own direct leaf

Same model, same host, same natural25 protocol family:

| Surface | tok/s | vs true AR | Source |
| --- | ---: | ---: | --- |
| Direct-leaf exact B3 (no prefill in timer) | **21.157528** | 1.8095x | `2026-08-26-gfx1151-qwen38-current-main-ar-mtp.json` |
| Public serving C1/B3, operation-complete | **12.940** | 1.4337x | `2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0.json` |

The difference is prefill amortization, activation, transaction ownership, and
HTTP/service boundaries — not model arithmetic. That is the largest single
recoverable block at c1.

### G2 — MTP is flat at c1-c8 while AR scales

Fixed-prompt p512/d128 blocking-HTTP diagnostic
(`2026-08-26-gfx1151-qwen38-mtp-old-vs-concurrency2-c1-c8.json`):

| Route | c1 | c2 | c4 | c8 |
| --- | ---: | ---: | ---: | ---: |
| OLD MTP (B2 serial-exact) | 7.907 | 7.127 | 7.068 | 7.013 |
| CONCURRENCY2 MTP (B3 native) | 13.596 | 12.935 | 12.948 | 12.552 |

CONCURRENCY2 is +72% to +83% over OLD but gains nothing from width, while the
indicative AR comparison (Q4_K_S, quant-mismatched — T0.1 fixes this) scales
10.9 → 29.8 tok/s and passes MTP from c2 onward. External engines confirm both
endpoints are real: llama.cpp/Vulkan MTP crosses over near c3-c4 (MikeVeerman),
while the CIRU vLLM stack keeps MTP K1 scaling c1→c6 (26.79 → 63.51 aggregate
tok/s). Flat-at-13 is a defect, not physics.

### G3 — physical SPECDEC2 acceptance collapses at C2/C4

From `SPECDEC2-PERF.md` (Q4_K_S): C1/K2 accepts **80.48%** and reaches 1.442x
AR; C2/C4 accept only **18.43%** and sit at 0.362x/0.319x AR. The retained
small-M Q4 WMMA owner improved C2/C4 +15.10%/+11.69% and is still below AR
(`2026-08-27-gfx1151-specdec2-smallm-q4-wmma-retained.json`). Acceptance, not
target cost alone, is the first-order c>1 problem.

### What "our llama-like MTP was much faster" refers to

The historical fast rows are **not** the qualified exact route:

- `llama-compat` B2, gfx1151: **69.50 tok/s (1.2776x own AR)** — explicit
  accuracy-traded direct-partial-commit/dp4a lane; it changes AR IDs and can
  never select `auto` (`MTP-LLAMACPP-PARITY.md`, `SOL-OPTIMIZATION.md` R2).
- W7900 `llama-compat`: 122.67 tok/s vs llama.cpp's 115.44 floor.
- Exact/default natural24 on gfx1151 was 52.13/52.04/50.65 tok/s at B1/B2/B5 —
  competitive, but on the old direct harness, not the serving path.

This campaign tunes the **exact** serving/SPECDEC2 route. `llama-compat`
remains a diagnostic control, never a promotion target.

## 2. External source review (commit-pinned, read-only)

| Project | Commit | Mechanism worth transferring | Not transferable |
| --- | --- | --- | --- |
| `julianmb/q38rocm` | `5d097740` | Draft-budget sweep discipline (`n_max` 3-6 × `p_min` 0.50-0.55); default `DRAFT_N=4, DRAFT_P=0.0`; Vulkan-Wave64 vs ROCm MTP note (36 vs 28 tok/s) | Vulkan numbers (we are HIP-native); ROCmFP4 artifacts; q8_0/turbo4 KV (our INT8 KV failed quality gates) |
| `KyaniteLabs/qwen38-27b-strix-halo` | `7fa3ca81` | ngram-mod stacked on MTP (`n12`, `n-min 24`) as a free prompt-derived second provider; time-per-task framing over raw tok/s; `HSA_ENABLE_SDMA=0 HSA_XNACK=1` hang guards | Their c30 148-163 tok/s rows (different model file/engine/power); n16 deep-draft waste already measured |
| `LaurentZuijdwijk/llama.cpp` | `c28d538` | **EMA adaptive draft length** (`common/speculative.cpp`: per-seq acceptance EMA, size next draft to `ema+1`, additive probe on full accept, censor-aware decay): adaptive json 65.57 vs bare 13.99 tok/s; fixed `n=7` collapses acceptance to 18-25% while adaptive holds 96% | FP4/DFlash2 absolute rates; Vulkan gate names |
| `MikeVeerman/qwen38-27-Strix-Halo-bench` | `cc527064` | Clean c1-c4 MTP-vs-AR protocol: c1 +2.11x, c2 +28%, c3 +4%, c4 **−22%** — crossover evidence | Q8_K_XL/Vulkan absolute rates; bit-nondeterminism tolerance (we require exact IDs) |
| `hogeheer499/strix-halo-guide` | `029320fb` | Paired no-spec control + acceptance + prompt-class reporting discipline; 35B n2/n3 sweet spots | 35B/Vulkan leaderboard rows |
| `jcbtc/Ling-3.0-Flash-CIRU-int4-Strix-native` (vLLM fork) | `838616875` (on vLLM `d35eb6c4`) | (a) **silent skinny-GEMM Wave32 dispatch cost 28.16x** — audit that serving shapes never silently fall to a wrong owner; (b) **multi-token verifier routing repair was +27.3%** — same family as G2/G3; (c) MTP K1 **scales c1→c6** (26.79→63.51 aggregate) with per-request acceptance — the G2 target state; (d) **`max_num_batched_tokens=8192`** vs the 2048 spec-decode default materially changed aggregate TG — audit our batch window/chunk analogs | MLA/LSE layout fixes (Qwen3.8 is GQA+GDN, not MLA); Ling-specific W4A16 geometry guards |

## 3. Punchlist

### T0 — baselines and attribution (measurement only; no behavior change)

- [ ] **T0.1 Matched-quant AR c1-c8.** Run the Q4_K_M true-AR c1-c8 blocking
  sweep on the same harness as the MTP diagnostic. The current AR comparison
  row is Q4_K_S and cannot be the MTP denominator. Exit: artifact + the
  definitive AR/MTP crossover table replacing the indicative one.
- [ ] **T0.2 C1 serving-vs-direct attribution (G1).** Operation-complete
  breakdown of serving C1/B3 against the direct leaf: prefill share, NextN
  activation/catch-up, proposal/target/accept/commit owners, HTTP boundary.
  Include a rocprof marker pass and assert the expected kernel owners actually
  execute at serving shapes (Ling skinny-GEMM lesson: no silent wrong-owner
  dispatch). Exit: named owner list with ms/token and a ranked gap closure
  order.
- [ ] **T0.3 c>1 acceptance root cause (G3).** Per-request acceptance and
  cycle telemetry at c2/c4/c8 on the current serving/SPECDEC2 path. Classify
  provider-state (NextN catch-up across requests) vs target-frontier batching
  vs verify-width arithmetic. Exit: one named root cause with reproducing
  artifact.
- [ ] **T0.4 Full-suite MTP c1-c8 baseline.** Replace the fixed-prompt
  diagnostic with the complete mtpbench category suite + heldouts at c1-c8
  under exact-ID and route-engagement checks. Required before any retained
  claim from T1+.

### T1 — close the C1 serving gap (G1)

- [ ] **T1.1 Streaming NextN prompt priming (OI-3) on the serving path.**
  Eliminate the retained full prompt-hidden slab + serial post-prefill draft
  catch-up; stream the exact shifted fold during target prefill. Approved as
  an idea in `OLMX-IDEAS.md`; this is the first implementation candidate.
- [ ] **T1.2 Zero-hot-allocation audit for the Q4_K_M serving key.** P3 closed
  this for Q4_K_S SPECDEC2; verify the promoted Q4_K_M serving route has zero
  cycle-local malloc/free after warmup.
- [ ] **T1.3 Device-chain coverage check.** Confirm N2/N3P device proposal /
  accept / selected-commit engagement on the Q4_K_M route at C1 (P4 retained
  it for Q4_K_S); fence or repair any host round-trip that remains.
- [ ] **T1.4 Candidate-budget sweep on serving C1.** B1/B2/B3/B4 under the
  full suite. External and internal evidence both point at B2/B3;
  `HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET` default 3, budget 4 known to regress.
  Exit: one measured default per qualified key, not a heuristic.

### T2 — c>1 MTP economics (G2/G3)

- [ ] **T2.1 Repair the C2/C4 acceptance collapse** identified by T0.3, or
  record the concrete blocker. Provider repair and per-request acceptance must
  survive physical batching (llama.cpp server model: one target batch,
  independent per-slot accept).
- [ ] **T2.2 Target-owner continuation.** Only with a new premise, extend the
  retained small-M Q4 WMMA direction to the remaining R6/R8/R12/R16 families;
  prior campaign gates apply (no retry of rejected composites).
- [ ] **T2.3 Measured K=0 crossover policy.** From T0.1+T0.4, derive the
  honest per-width verdict and admit `K=0 above crossover` with a stable typed
  reason (vLLM/SGLang adaptive-tier equivalent). If T2.1 fails, this is the
  product endpoint for wide cells; MTP at c1, AR above the measured crossover.
- [ ] **T2.4 Scheduler-budget audit.** Check our speculative batch window /
  chunk-token analogs of Ling's `max_num_batched_tokens=8192` finding
  (`HIPENGINE_MAX_PREFILL_CHUNK_TOKENS`, generation batch window) under MTP at
  c1-c8; measure, don't assume.

### T3 — adaptive candidate budget (after T1/T2 land)

- [ ] **T3.1 EMA adaptive K over independent B1/B2/B3 graph buckets.**
  Port the LaurentZuijdwijk mechanism as a per-request, censor-aware
  acceptance EMA sized to `ema+1` with additive probe on full accept. OI-2's
  reopen precondition (independent buckets, transitions passing) is now met.
  Default-off until the full suite proves it; the policy consumes only a
  predeclared online acceptance statistic — never prompt text/IDs/category
  (SPECDEC2-PERF invariant 11).
- [ ] **T3.2 K(C) policy table from measured cells.** Adaptive behavior by
  realized width comes from the T0/T2 measurement table, not a hand-written
  curve.

### T4 — research / parked (no promotion path in this campaign)

- [ ] **T4.1 ngram-mod stacking spike (KyaniteLabs mechanism).** Prompt-derived
  drafts are free and complementary to MTP; evaluate as a separate provider
  behind the registry. Content-based drafting is a general mechanism (allowed);
  any policy keyed to prompt identity is forbidden.
- [ ] **T4.2 KV-quant (q8_0 / turbo4-class):** parked — our INT8 KV lane
  failed quality gates; external llama.cpp claims do not transfer.
- [ ] **T4.3 DFlash2:** parked — measured 8.85 tok/s = 0.66x AR on this exact
  target (`QWEN38-27B-DFLASH2-CAMPAIGN.md`).
- [ ] **T4.4 ROCr idle busy-spin check (Ling release note):** verify our host
  stack does not carry the idle-CPU spin; close as environment hygiene.

## 4. Non-negotiable rules

1. Full mtpbench category suite + heldouts for every acceptance/speed/quality
   claim; fixed-prompt results are diagnostics and are never retained rows.
2. True same-protocol, matched-quant AR denominator for every MTP ratio;
   verifier `off`/B0 rows are diagnostic only.
3. No prompt-, token-, category-, or heldout-conditioned policy anywhere.
4. Same-host evidence only; nothing transfers to/from W7900 or between 8060S
   hosts.
5. Every candidate names its strict fallback before implementation; production
   arithmetic passes the complete profile gate.
6. Retained win: compact artifact + `benchmarks/README.md` +
   `benchmarks/CHANGELOG.md` + worklog. Rejected: compact artifact + worklog
   only. Reruns follow the focused-repair policy in `AGENTS.md`.

## 5. Definition of done

The campaign closes when either:

- **(a)** serving C1 MTP reaches direct-leaf parity territory (≳1.6x AR,
  i.e. ~19+ tok/s operation-complete) under the full suite, **and** each of
  c2-c8 has a measured verdict — MTP promoted where it beats matched AR under
  every binding gate, honest `K=0` elsewhere; or
- **(b)** a phase produces a concrete measured blocker (named owner, ms/token,
  reproducing artifact) that invalidates the premise, in which case the K0
  crossover policy ships with exact reasons and the blocker is recorded in
  `docs/REFACTOR.md`/the relevant campaign ledger.
