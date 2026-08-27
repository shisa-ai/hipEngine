# CONCURRENCY2 gfx1151 MTP Tuning Campaign

- Status: **campaign completed 2026-08-27 via definition-of-done branch (b); no product cell promoted, automatic remains K0**
- Follow-up intake: **documentation-only CR-S0–CR-S3 compact-rollback sweep proposed 2026-08-27; no GPU run or product claim, and the original closure remains binding**
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

## Final outcome

- Capacity-matched production c1/B3: **14.287 vs 9.730 tok/s (+46.84%)**;
  every category positive, 79.29% accept/draft.
- Production correctness (not bit exactness): 1,170 strict-teacher rows, KL
  mean/p95/p99/max `1.239e-4/3.503e-4/1.110e-3/0.049788`, top-1 99.744%,
  deterministic repeats/isolation/manifest gates pass.
- C2-c8: typed K0; repaired physical C2/C4 still only 0.6975x/0.5843x AR.
- No promotion: c1 is 1.468x, below the 1.6x/~19 tok/s target, and only engages
  on capacity-1. The normal concurrency owner cannot select c1 MTP then c2+
  K0 before mutation; physical streaming/refill/survivor lifecycle is
  unqualified. FP16 device proposal also lacks eager selected commit.
- Definition-of-done **branch (b)** applies. Automatic stays K0, B3 remains the
  explicit diagnostic candidate, and blockers/reopen triggers are recorded in
  `docs/REFACTOR.md`. Completion audit:
  [`campaign closeout`](../benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-campaign-closeout.json).

## 1. Why this campaign exists — three measured gaps

> **Final note (2026-08-27):** all T0-T4 items are complete. The gap statements
> below are entry framing kept for context; §1a and the punchlist carry the
> superseding results.

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

All T0-T4 items are complete. No product cell promotes: capacity-matched c1/B3
is the sole positive candidate, while automatic remains K0 under the recorded
capacity/dynamic-lifecycle blockers.

## 2. External source review (commit-pinned, read-only)

| Project | Commit | Mechanism worth transferring | Not transferable |
| --- | --- | --- | --- |
| `julianmb/q38rocm` | `5d097740` | Draft-budget sweep discipline (`n_max` 3-6 × `p_min` 0.50-0.55); default `DRAFT_N=4, DRAFT_P=0.0`; Vulkan-Wave64 vs ROCm MTP note (36 vs 28 tok/s) | Vulkan numbers (we are HIP-native); ROCmFP4 artifacts; q8_0/turbo4 KV (our INT8 KV failed quality gates) |
| `KyaniteLabs/qwen38-27b-strix-halo` | `7fa3ca81` | ngram-mod stacked on MTP (`n12`, `n-min 24`) as a free prompt-derived second provider; time-per-task framing over raw tok/s; `HSA_ENABLE_SDMA=0 HSA_XNACK=1` hang guards | Their c30 148-163 tok/s rows (different model file/engine/power); n16 deep-draft waste already measured |
| `LaurentZuijdwijk/llama.cpp` | `c28d538` | **EMA adaptive draft length** (`common/speculative.cpp`: per-seq acceptance EMA, size next draft to `ema+1`, additive probe on full accept, censor-aware decay): adaptive json 65.57 vs bare 13.99 tok/s; fixed `n=7` collapses acceptance to 18-25% while adaptive holds 96% | FP4/DFlash2 absolute rates; Vulkan gate names |
| `MikeVeerman/qwen38-27-Strix-Halo-bench` | `cc527064` | Clean c1-c4 MTP-vs-AR protocol: c1 +2.11x, c2 +28%, c3 +4%, c4 **−22%** — crossover evidence | Q8_K_XL/Vulkan absolute rates; different correctness contract (our production profile binds numerical/task quality and repeatability; generated-ID equality is diagnostic) |
| `hogeheer499/strix-halo-guide` | `029320fb` | Paired no-spec control + acceptance + prompt-class reporting discipline; 35B n2/n3 sweet spots | 35B/Vulkan leaderboard rows |
| `jcbtc/Ling-3.0-Flash-CIRU-int4-Strix-native` (vLLM fork) | `838616875` (on vLLM `d35eb6c4`) | (a) **silent skinny-GEMM Wave32 dispatch cost 28.16x** — audit that serving shapes never silently fall to a wrong owner; (b) **multi-token verifier routing repair was +27.3%** — same family as G2/G3; (c) MTP K1 **scales c1→c6** (26.79→63.51 aggregate) with per-request acceptance — the G2 target state; (d) **`max_num_batched_tokens=8192`** vs the 2048 spec-decode default materially changed aggregate TG — audit our batch window/chunk analogs | MLA/LSE layout fixes (Qwen3.8 is GQA+GDN, not MLA); Ling-specific W4A16 geometry guards |
| Eaman [`MTP Compact Rollback`](https://store.piffa.net/lm/bug/mtp_compact_rollback.md) for llama.cpp | llama.cpp `3737e41370da1830a44c663f9929a0f27591ffa6`; feature `6cb89357`; [standalone patch](https://store.piffa.net/lm/bug/mtp_compact_rollback_3737e41.patch) SHA-256 `759ad384...526ba` | Decouple draft maximum `N` from immediate recurrent rollback depth `D`; pre-reserve one per-slot on-device pre-verify checkpoint; direct-rollback shallow suffixes and restore/replay already selected target inputs after deeper rejection; expose replay-event/token and exact memory-reservation telemetry | Published context/TG rows are explicitly directional, not matched A/B evidence; RDNA2, multi-GPU, quantized-KV, and N5/N7 rates do not transfer. Final flag is `--spec-mtp-cr-depth`; `--spec-mtp-rs-depth` is stale. Its adaptive controller is already covered by T3. |

## 3. Punchlist

### T0 — baselines and attribution (measurement only; no behavior change)

- [x] **T0.1 Matched-quant AR c1-c8** — done 2026-08-27; see §1a.
  Crossover between c1 and c2; MTP only wins c1 (+31.7%).
- [x] **T0.2 C1 serving-vs-direct attribution (G1).** Cached production
  c1/B3 profile leaf: 2,807.7-ms arm; target prefill/activation 1,680.8 ms
  (59.9%), including cold provider streaming open **981.9 ms / 189 allocations /
  only 0.052 ms GPU**; eight cycles 1,068.3 ms (proposal 147.1, target/accept/
  commit/provider 919.3); reclaim 51.6 ms. Existing pooling reduces provider
  open to **0.13-0.15 ms** on warm repeats (1.735/1.739-s complete wall), so it
  is not steady debt. All 16 repeat cycles have zero malloc/free. Expected Q4/
  Q5/Q6 rows4/rows2 owners execute; no R6/R8/R12/R16 dispatch. Evidence:
  [`T0.2 profile`](../benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t02-actual-owner-profile.json).
- [x] **T0.3 c>1 legacy acceptance attribution** — accept-per-draft is stable
  (0.64-0.67) on the forced legacy route; its flat line is not an acceptance
  collapse. Wide timing fields are chunk-wall copies and are not additive.
  The staged SPECDEC2 18.43% C2/C4 collapse remains open as T2.1.
- [x] **T0.4 Full-suite staged MTP c1-c8 baseline.** C1 production staged MTP2
  is +46.84% over production AR and passes the full 1,170-row production
  numerical/determinism/isolation gate; c2-c8 stay typed K0 in 70/70 cells.
  Final promotion is blocked by capacity>1/dynamic lifecycle ownership.

### T1 — close the C1 serving gap (G1)

- [x] **T1.0 Prefill attribution.** Isolated target p512 is 384-402 tok/s;
  capture/hidden-seed/D2H/full-MTP-env variants differ <0.4%. The apparent
  ~113 tok/s field belonged to the forced legacy activation owner and packed
  telemetry bookkeeping, not a prefill kernel. No kernel candidate admitted.
- [x] **T1.1 Streaming NextN prompt priming (OI-3) on public serving.** 10/10
  production B3 requests stream all 449 prompt rows, retain exactly one
  10,240-byte carried row each, and report no prompt fallback/full slab.
- [x] **T1.2 Zero-hot-allocation audit for the Q4_K_M serving key.** Hot-cycle
  proposal/repair workspace is persistent at stable pointers/shape `[1,5120]`
  across all requests after warmup. Each request still allocates/frees
  83,794,462 bytes of verifier/admission state, but active ownership returns to
  zero and this does not scale with its 6-8 cycles. Warm repeats confirm provider
  open at 0.13-0.15 ms and zero allocations in all 16 cycles; existing pooling
  is effective.
- [x] **T1.3 Device-chain coverage check / fence.** Qualified production uses
  70 eager cycles with zero device handoff/GPU accept/selected-commit calls.
  An oracle-only FP16 device-proposal/eager-target candidate launched one
  handoff/request, then failed precommit (`FP16 recurrent state device proposal
  requires eager selected commit`), recovered all 10 to AR, and regressed
  6.77%. Candidate source was removed; retain/fence the qualified eager host
  proposal owner. Evidence:
  [`T1.1-T1.3 ownership`](../benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t11-t13-ownership.json).
- [x] **T1.4 Candidate-budget sweep on serving C1.** Truthful physical budgets
  under the full suite: B1 12.236 tok/s (+25.59%, 95.65% accept), B2 14.055
  (+44.54%, 91.87%), B3 **14.287 (+46.84%, 79.29%)**. Every category is
  positive. B3 beats B2 1.65% and remains the measured default; B4 clamps to
  B3 and is not a distinct candidate. Generated IDs are diagnostic; T0.4's
  production numerical gate is binding.

### T2 — c>1 MTP economics (G2/G3)

- [x] **T2.1 Repair the SPECDEC2-physical C2/C4 acceptance collapse.** Newer
  retained P9 target-reseed evidence supersedes the 18.43% row: production
  acceptance is 95.0%/89.82%/77.73% at K1/K2/K3 with zero candidate D2H.
  Economics still fail: best K3 is only 0.6975x C2 / 0.5843x C4 true AR.
  Close provider repair; C2/C4 remain K0 unless T2.2 finds a new target-cost
  premise. Evidence:
  [`T2.1 closeout`](../benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t21-acceptance-closeout.json).
- [x] **T2.2 Target-owner continuation closeout.** T0.2 public c1/B3 executes
  only rows4 plus rows2 tails; R6/R8/R12/R16 dispatch count is zero. Physical
  C2/C4 remains K0 and best retained K3 is only 0.6975x/0.5843x AR. No current
  dispatch satisfies the >=1.15x leaf / >=0.5-ms-token admission premise; do
  not implement wider kernels. Evidence:
  [`T2.2 closeout`](../benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t22-target-owner-closeout.json).
- [x] **T2.3 Measured K=0 crossover policy.** Automatic remains K0 with stable
  typed reasons; no cell promotes. Capacity-matched c1/B3 is +46.84%, but the
  HTTP plan sees `len(prompts)`, not concurrent independent children, while the
  resident owner rejects c1 on capacity>1/unqualified physical streaming.
  Dynamic c1-MTP/c2+-K0 therefore needs a future resident realized-group +
  refill/survivor lifecycle design. Forced backend K0 is not acceptable (c5
  diagnostic -35.1%); fail closed before mutation.
- [x] **T2.4 Scheduler-budget audit.** Full-suite c1/B3 chunks 16/32/64/256
  yield 11.162/11.533/13.932/**14.287** MTP tok/s at unchanged 79.29%
  acceptance. Retain 256. Batch-window overlap is N/A while only c1 MTP is
  admitted; c2+ must stay on true-AR ownership. Evidence:
  [`T2.3/T2.4`](../benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t23-t24-policy-scheduler.json).

### T3 — adaptive candidate budget (after T1/T2 land)

- [x] **T3.1 EMA adaptive K closeout.** OI-2 already implemented the
  request-owned content-agnostic acceptance/cycle-wall EMA over independent
  B1/B2/B3 buckets; all nine transitions and deterministic repeats pass. It
  loses fixed B3 0.58% primary / 1.72% repeat and regresses train/code/general
  English. New truthful serving evidence also keeps B3 1.65% above B2. Retain
  transition diagnostics; reject/default-off the controller.
- [x] **T3.2 K(C) policy table not admitted.** There is no adaptive winner and
  c2+ has no MTP product cell; a curve would be hand-written without eligible
  evidence. Automatic remains K0. Evidence:
  [`T3 closeout`](../benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t31-adaptive-closeout.json).

### T4 — research / parked (no promotion path in this campaign)

- [x] **T4.1 ngram-mod stacking spike (KyaniteLabs mechanism).** Full-suite
  content-agnostic simulation rejects a provider: best short ngram has 46.9%
  coverage / 28.4% accept / 1.37 visible tokens per cycle; deep n12 reaches
  1.45 visible/cycle but only 12.2% accept and 630 verifier rows. MTP B3 is
  3.43 visible/cycle / 79.29% accept. Heldout coverage is weaker; no code
  candidate.
- [x] **T4.2 KV-quant (q8_0 / turbo4-class):** closed parked — our INT8 KV
  lane failed quality gates; external llama.cpp claims do not transfer.
- [x] **T4.3 DFlash2:** closed parked — measured 8.85 tok/s = 0.66x AR on this
  exact target (`QWEN38-27B-DFLASH2-CAMPAIGN.md`).
- [x] **T4.4 ROCr idle busy-spin check (Ling release note):** initialized HIP
  runtime idle for 15 s used 0.00 CPU ticks / 0% of one core; no busy-spin.
  Evidence:
  [`T4 closeout`](../benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t4-research-closeout.json).

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

**Closure: branch (b).** The exact blockers and production evidence are in the
Final outcome above and the closeout artifact. Automatic K0 is the shipped
safe policy; this campaign does not claim a product MTP promotion.

## 6. Proposed follow-up: CR compact-rollback capacity sweep

Status: **planned from source review only; no GPU work was run for this intake.**
This is not a reopening of the completed throughput campaign. The Eaman source
adds one genuinely new capacity mechanism, but its own report labels the
headline context/TG table directional. The sweep below must establish a local
product premise before implementation or measurement.

### Intake decision and local source map

The transferable mechanism is **target recurrent-state depth decoupling**, not
its external rates. llama.cpp normally retains one recurrent rollback state per
possible draft position. Compact Rollback retains only `D`, keeps one reusable
pre-verify device checkpoint, and reconstructs state from already selected
target inputs after a rejection deeper than `D`; it does not redraft or
resample.

hipEngine already has most of the control plumbing but not the compact storage
policy:

- `_initial_state_only_journal_applies()` and producer capture retain one
  pre-verify Conv/GDN snapshot for the native B3 target owner;
- `_ensure_verify_linear_state_row_buffers(rows)` still allocates Conv/GDN
  state for every verify row so a selected row can commit directly; and
- `verify_target_block(..., advance_state_only=True)` already supports
  target-state replay without the LM-head/sample work.

The known **83,794,462-byte** per-request admission/provider allocation is only
an aggregate. Do not call it rollback savings until CR-S0 attributes every
plane. The currently fingerprinted public key also rejects context 68, while
the external claim concerns 56K–85K context. A local rollback-memory reduction
has no product value unless it unlocks a retained context or residency cell.

### CR-S0 — premise, ownership, and baseline (measurement before code)

- [ ] **CR-S0.0 Product-premise gate.** Name the exact currently inadmissible
  context/resident-capacity cell that fewer rollback rows could unlock. Confirm
  the limiting `ResourceClaimSet` plane and the route's context capability. If
  another plane or the context-68 product fingerprint remains limiting, stop;
  do not build a memory feature for a short-only route.
- [ ] **CR-S0.1 Exact state-plane byte attribution.** For physical B1/B2/B3,
  report live Conv/GDN state, initial checkpoint, captured verify-row states,
  hidden rows, target accept buffers, NextN provider state, and unrelated
  admission workspace separately. Record row capacity, allocation lifetime,
  pointers, current/peak bytes, and bytes per additional rollback row.
- [ ] **CR-S0.2 Admission claim and fallback.** Model full-depth and compact-D1
  bytes as exact pre-mutation resource claims. Reserve persistent per-request
  buffers before admission, fail closed on a missing claim/allocation, and name
  the unchanged full-row journal as strict fallback. No first-cycle/lazy
  allocation may consume headroom after admission.
- [ ] **CR-S0.3 Matched baselines.** Freeze same-host AR and full-row B1/B2/B3
  results on the full category suite plus heldouts. Separately freeze a
  memory-pressure context/capacity row; never compare speed from two different
  fitted contexts as an old→new throughput result.

### CR-S1 — RED and bounded D1 prototype

- [ ] **CR-S1.1 RED every commit outcome.** Cover accepted counts `0..K` for
  B1/B2/B3 across multiple cycles, full accept, shallow/deep rejection,
  correction/root transition, short remaining horizon/EOS, cancellation,
  injected failure, refill, and neighbor/permutation isolation. Request/slot,
  target/provider cursor, Conv/GDN, KV/`KVLiveSpans`, transaction, and output
  ownership are exact. Production arithmetic uses the normative numerical,
  deterministic/isolation, BF16-relative, and task gates; generated-ID
  bit-equality remains diagnostic.
- [ ] **CR-S1.2 Compact target-state journal.** Keep the existing full-row
  implementation unchanged as fallback. A D1 candidate retains only the one
  eligible near-tail direct-commit row plus the reusable initial device
  checkpoint. A deeper rejection restores the initial state and replays only
  the already selected target input prefix required by the current cursor
  contract, using the state-only path where valid. It must never rerun proposal,
  acceptance, or sampling, and rejected full-attention KV remains unreachable
  through live-span metadata.
- [ ] **CR-S1.3 Persistent ownership and telemetry.** Expose direct rollbacks,
  deep-replay events/tokens/target rows, replay wall and kernel time, checkpoint
  copy time, row-state bytes saved, actual claim bytes, and host-fallback count.
  Steady cycles must allocate/free zero bytes; reclaim must return all
  request-owned bytes exactly once.

### CR-S2 — economics, capacity, and lifecycle gates

- [ ] **CR-S2.1 Diagnostic screen.** One natural prompt may screen full rows vs
  D1 at B1/B2/B3 for byte savings, replay frequency, and replay cost. It is not
  retainable performance evidence and cannot tune prompt/category policy.
- [ ] **CR-S2.2 Binding fixed-context A/B.** Run the complete category suite +
  heldouts at one common context, with true same-protocol AR, full-row MTP, and
  D1 MTP. Report aggregate and category tok/s, complete wall, acceptance by
  position, full-accept rate, replay distribution, and all production-profile
  correctness gates.
- [ ] **CR-S2.3 Separate capacity A/B.** Under one declared memory cap, report
  maximum admissible context and resident request count for full rows and D1,
  then benchmark both at their largest **common** context. A larger fitted
  context is capacity evidence, not a throughput speedup.
- [ ] **CR-S2.4 Dynamic lifecycle.** Two independent request checkpoints must
  survive cancel/refill/survivor/reorder and repeated admission with zero
  cross-talk, delayed OOM, sticky host fallback, or residual ownership. This is
  a prerequisite for the existing T2.3 realized-group blocker; it does not by
  itself qualify c>1 MTP economics.

Default promotion requires a complete same-suite non-regression or a newly
admissible product cell with passing SLO and correctness gates. A capacity-only
win with a speed cost may remain an explicit capacity profile, but must not
replace automatic B3. Reject D1 if saved state bytes do not increase context or
residency, if replay erases the product benefit, or if any lifecycle/gate fails.

### CR-S3 — conditional follow-ups only

- [ ] **CR-S3.1 Replay-aware budget policy, conditional.** T3 already rejected
  adaptive K. Reopen only if retained D1 changes the measured cost surface; use
  content-agnostic per-request acceptance, replay, and cycle-wall telemetry.
  Do not import the external heuristic as a new default.
- [ ] **CR-S3.2 Deeper N4–N7, conditional.** Do not copy the external N5/N7
  settings. Current B4 clamps to B3 and native rows>4 lack the T2.2 owner
  premise. Screen a deeper budget only after a registered rows5+ target owner
  clears the existing leaf/ms-token admission gate and the full-row memory is
  the demonstrated limiter.
- [ ] **CR-S3.3 Keep unrelated controls out.** `--hip-fa-force-vec` is a
  quantized-KV llama.cpp capacity/PP trade and does not apply to this BF16-KV
  product key; route it through a separately qualified KV-backend campaign.
  `--pipeline-parallel` is multi-GPU scheduler policy and is N/A on the single
  Radeon 8060S lane. Neither reopens T4.2 or creates a CONCURRENCY2 candidate.
