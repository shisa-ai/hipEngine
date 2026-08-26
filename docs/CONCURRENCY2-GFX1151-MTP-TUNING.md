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

> **T0 update (2026-08-27):** T0.1 and T0.3 are complete; see §1a for the
> definitive matched-quant crossover table and the measured root causes. The
> gap statements below are the pre-T0 framing kept for context.

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

CONCURRENCY2 is +72% to +83% over OLD but gains nothing from width.
The definitive matched-quant AR comparison (§1a) scales 10.3 → 31.8 tok/s and
passes MTP from c2 onward. External engines confirm both
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

## 1a. T0 measured findings (2026-08-27, gfx1151, Q4_K_M)

Artifact:
[`2026-08-27-gfx1151-qwen38-q4km-ar-vs-mtp-c1-c8.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-q4km-ar-vs-mtp-c1-c8.json)
(diagnostic only — fixed synthetic prompt; T0.4 remains the retained gate).

### Definitive matched-quant crossover (blocking HTTP, p512/d128, Q4_K_M)

| Route | c=1 | c=2 | c=3 | c=4 | c=5 | c=6 | c=7 | c=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CONCURRENCY2 AR Q4_K_M (tok/s) | 10.323 | 17.714 | 23.296 | 27.240 | 29.293 | 31.799 | 26.398 | 28.431 |
| CONCURRENCY2 MTP Q4_K_M (tok/s) | 13.596 | 12.935 | 12.956 | 12.948 | 13.091 | 12.957 | 12.665 | 12.552 |
| MTP vs AR | **+31.7%** | −27.0% | −44.4% | −52.5% | −55.3% | −59.3% | −52.0% | −55.9% |

Crossover is **between c1 and c2**. Q4_K_M AR tracks the earlier Q4_K_S SSE
diagnostic within ~3% at c1-c6; the c7 dip (26.40 vs c6 31.80 at 0.05% stdev)
is reproducible and unexplained — width-bucketing suspect. MTP accept-per-draft
is stable: 0.7395 at c1, 0.639-0.669 at c2-c8 — acceptance is **not** the
serving-route problem at width (contrast SPECDEC2's physical 18.43% at C2/C4,
which is a different surface and still open).

### G1 root cause correction: legacy prompt activation, not target prefill

T1.0 supersedes the original additive interpretation of the legacy response
telemetry. Packed chunk wall is copied into each member request; wide-cell
`prefill_ms`/`target_verify_ms` values cannot be summed across requests.
Targeted same-host probes isolate p512 target prefill at **1.27-1.34 s
(384-402 tok/s)** across plain, GDN-capture, hidden-seed-return, D2H, and full
legacy-MTP-env variants; the hidden D2H itself is 0.3 ms. There is no prefill
kernel regression to fix.

The forced legacy route instead retains the full prompt-hidden slab and calls
`Qwen35GGUFResidentMTPDraftRunner.write_kv_rows()` once per prompt row after
target prefill. That serial NextN catch-up is the missing activation wall. The
staged MTP2 owner already uses retained exact OI-3 streaming prompt priming
(p512 TTFT 13.079 -> 10.356 s, -20.82%; slab -> one 10,240-byte row). Current
public requests are blocked before MTP with typed reason
`execution_profile_not_qualified`; T0.4 must exercise/qualify staged MTP2 rather
than tune the legacy fallback. Evidence:
[`T1.0 attribution`](../benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t10-prefill-attribution.json).

### G2 mechanism: forced legacy scheduler; staged owner not yet qualified

- The forced diagnostic route is phase serial: draft live slots -> verify
  chunks of <=4 slots -> commit live slots.
- Legacy response telemetry books whole chunk wall to every member request, so
  prior per-position/additive estimates are withdrawn. A definitive staged
  target-owner ranking requires T0.4 route engagement followed by T0.2 rocprof.
- Tokens/step ≈ 2.9-3.2 and accept-per-draft 0.64-0.74 on the forced legacy
  route — drafter quality was not its flat-line owner.
- Current public explicit MTP routes to AR (`execution_profile_not_qualified`),
  so it has no MTP economics until the staged execution profile passes its
  production correctness/task gate.

### Correctness observation (diagnostic)

Batched AR flips greedy near-ties on the repeated-token prompt at c≥4 (up to
12/24 rows at c8; the Q4_K_S diagnostic showed the same pattern), while
target-verified MTP had **0 mismatches at every width**. MTP is the more
batch-invariant route on this diagnostic; T0.4's natural-prompt suite is the
binding gate.

### T0.4 production result (2026-08-27)

Capacity-matched public serving, canonical ten-prompt/four-category+heldout
suite, max24, B2:

| Width | Effective route | AR tok/s | MTP tok/s | MTP vs AR | Engagement |
| ---: | --- | ---: | ---: | ---: | ---: |
| c1 | staged MTP2 B3 | 9.730 | **14.287** | **+46.84%** | 10/10 |
| c2-c8 | typed K0/AR | 15.124-35.296 | n/a | n/a | 0/70 |

C1 is non-regressive in every category: code +48.33%, general_en +52.15%,
general_ja +39.56%, mixed_ja_en +46.44%; heldouts +48.66%. Aggregate
accept-per-draft is 79.29%. Exact AR/MTP IDs on 10/10 c1 cells and all 70 K0
cells are diagnostic, not the binding production criterion. T1.4 later exposed
that the original label B2 was adapter-default physical B3; this table uses the
truthful budget-conformed B3 rerun.

The Q4_K_M production gate passes **1,170** strict-teacher rows: KL
mean/p95/p99/max `1.239e-4 / 3.503e-4 / 1.110e-3 / 0.049788`, top-1
`1167/1170 = 99.744%`, finite logits, all three repeats deterministic,
neighbor/permutation isolation, and indexed manifest gates. The sole review
row passes manually (same top-1/rank 1, margin 3.168, top-k overlap 1.0,
teacher NLL improves). BF16-relative/external task scoring are normative N/A
because quant/KV/target capability are unchanged; the full category suite is
non-regressive.

This is a **candidate, not a promotion**: actual-owner rocprof,
streaming/device-chain/allocation telemetry, and dynamic lifecycle/SLO remain.
Requested-MTP K0 controls also show wide timing anomalies (c5 -35.1%, c6
-30.6% vs paired AR), so T2.3/T2.4 must prove true-AR batching/economics.
Evidence:
[`T0.4 production suite`](../benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t04-production-suite.json).

### Punchlist impact

T0.1/T0.3/T0.4/T1.0 are done. T0.2 rocprof and T1.1-T1.3 telemetry are now the
critical path before any c1 promotion; T1.4 then measures B1-B4. Public
production policy remains MTP at no automatic cell (K0) until those gates pass.

## 2. External source review (commit-pinned, read-only)

| Project | Commit | Mechanism worth transferring | Not transferable |
| --- | --- | --- | --- |
| `julianmb/q38rocm` | `5d097740` | Draft-budget sweep discipline (`n_max` 3-6 × `p_min` 0.50-0.55); default `DRAFT_N=4, DRAFT_P=0.0`; Vulkan-Wave64 vs ROCm MTP note (36 vs 28 tok/s) | Vulkan numbers (we are HIP-native); ROCmFP4 artifacts; q8_0/turbo4 KV (our INT8 KV failed quality gates) |
| `KyaniteLabs/qwen38-27b-strix-halo` | `7fa3ca81` | ngram-mod stacked on MTP (`n12`, `n-min 24`) as a free prompt-derived second provider; time-per-task framing over raw tok/s; `HSA_ENABLE_SDMA=0 HSA_XNACK=1` hang guards | Their c30 148-163 tok/s rows (different model file/engine/power); n16 deep-draft waste already measured |
| `LaurentZuijdwijk/llama.cpp` | `c28d538` | **EMA adaptive draft length** (`common/speculative.cpp`: per-seq acceptance EMA, size next draft to `ema+1`, additive probe on full accept, censor-aware decay): adaptive json 65.57 vs bare 13.99 tok/s; fixed `n=7` collapses acceptance to 18-25% while adaptive holds 96% | FP4/DFlash2 absolute rates; Vulkan gate names |
| `MikeVeerman/qwen38-27-Strix-Halo-bench` | `cc527064` | Clean c1-c4 MTP-vs-AR protocol: c1 +2.11x, c2 +28%, c3 +4%, c4 **−22%** — crossover evidence | Q8_K_XL/Vulkan absolute rates; different correctness contract (our production profile binds numerical/task quality and repeatability; generated-ID equality is diagnostic) |
| `hogeheer499/strix-halo-guide` | `029320fb` | Paired no-spec control + acceptance + prompt-class reporting discipline; 35B n2/n3 sweet spots | 35B/Vulkan leaderboard rows |
| `jcbtc/Ling-3.0-Flash-CIRU-int4-Strix-native` (vLLM fork) | `838616875` (on vLLM `d35eb6c4`) | (a) **silent skinny-GEMM Wave32 dispatch cost 28.16x** — audit that serving shapes never silently fall to a wrong owner; (b) **multi-token verifier routing repair was +27.3%** — same family as G2/G3; (c) MTP K1 **scales c1→c6** (26.79→63.51 aggregate) with per-request acceptance — the G2 target state; (d) **`max_num_batched_tokens=8192`** vs the 2048 spec-decode default materially changed aggregate TG — audit our batch window/chunk analogs | MLA/LSE layout fixes (Qwen3.8 is GQA+GDN, not MLA); Ling-specific W4A16 geometry guards |

## 3. Punchlist

### T0 — baselines and attribution (measurement only; no behavior change)

- [x] **T0.1 Matched-quant AR c1-c8** — done 2026-08-27; see §1a.
  Crossover between c1 and c2; MTP only wins c1 (+31.7%).
- [ ] **T0.2 C1 serving-vs-direct attribution (G1).** Operation-complete
  breakdown of serving C1/B3 against the direct leaf: prefill share, NextN
  activation/catch-up, proposal/target/accept/commit owners, HTTP boundary.
  Include a rocprof marker pass and assert the expected kernel owners actually
  execute at serving shapes (Ling skinny-GEMM lesson: no silent wrong-owner
  dispatch). Exit: named owner list with ms/token and a ranked gap closure
  order.
- [x] **T0.3 c>1 legacy acceptance attribution** — accept-per-draft is stable
  (0.64-0.67) on the forced legacy route; its flat line is not an acceptance
  collapse. Wide timing fields are chunk-wall copies and are not additive.
  The staged SPECDEC2 18.43% C2/C4 collapse remains open as T2.1.
- [x] **T0.4 Full-suite staged MTP c1-c8 baseline.** C1 production staged MTP2
  is +46.75% over production AR and passes the full 1,170-row production
  numerical/determinism/isolation gate; c2-c8 stay typed K0 in 70/70 cells.
  Promotion remains blocked on T0.2/T1.1-T1.3 lifecycle/ownership evidence.

### T1 — close the C1 serving gap (G1)

- [x] **T1.0 Prefill attribution.** Isolated target p512 is 384-402 tok/s;
  capture/hidden-seed/D2H/full-MTP-env variants differ <0.4%. The apparent
  ~113 tok/s field belonged to the forced legacy activation owner and packed
  telemetry bookkeeping, not a prefill kernel. No kernel candidate admitted.
- [ ] **T1.1 Streaming NextN prompt priming (OI-3) on public serving.** Shared
  source is already retained and fully gated in staged MTP2; the legacy forced
  route still uses the slab/catch-up loop. Completion means T0.4 proves the
  public staged route engages OI-3 and passes the production profile; do not
  duplicate the sink or optimize the legacy fallback.
- [ ] **T1.2 Zero-hot-allocation audit for the Q4_K_M serving key.** P3 closed
  this for Q4_K_S SPECDEC2; verify the promoted Q4_K_M serving route has zero
  cycle-local malloc/free after warmup.
- [ ] **T1.3 Device-chain coverage check.** Confirm N2/N3P device proposal /
  accept / selected-commit engagement on the Q4_K_M route at C1 (P4 retained
  it for Q4_K_S); fence or repair any host round-trip that remains.
- [x] **T1.4 Candidate-budget sweep on serving C1.** Truthful physical budgets
  under the full suite: B1 12.236 tok/s (+25.59%, 95.65% accept), B2 14.055
  (+44.54%, 91.87%), B3 **14.287 (+46.84%, 79.29%)**. Every category is
  positive. B3 beats B2 1.65% and remains the measured default; B4 clamps to
  B3 and is not a distinct candidate. Generated IDs are diagnostic; T0.4's
  production numerical gate is binding.

### T2 — c>1 MTP economics (G2/G3)

- [ ] **T2.1 Repair the SPECDEC2-physical C2/C4 acceptance collapse**
  (18.43% vs 80.48% at C1, Q4_K_S surface; the serving route does not show
  it). Provider repair and per-request acceptance must survive physical
  batching (llama.cpp server model: one target batch, independent per-slot
  accept).
- [ ] **T2.2 Target-owner continuation.** Only with a new premise, extend the
  retained small-M Q4 WMMA direction to the remaining R6/R8/R12/R16 families;
  prior campaign gates apply (no retry of rejected composites).
- [ ] **T2.3 Measured K=0 crossover policy.** T0.4 fixes the candidate table:
  c1/B2 staged MTP2, c2-c8 K0. Before admission, prove requested-MTP K0 uses
  true-AR batching/economics (current controls regress c5/c6 35.1%/30.6%) and
  emit stable typed reasons. T2.1 may reopen only independently qualified
  physical c2/c4 cells.
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
5. Every candidate names its strict fallback before implementation. A speed
   gain promotes when the complete **production execution-profile** numerical,
   deterministic/isolation, BF16-relative, and task gates pass; generated-ID
   bit equality with strict/AR is diagnostic and is not a promotion requirement.
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
