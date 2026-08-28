# Qwen3.8-27B `Q4_K_M` MTP acceptance-economics campaign

- Status: **planned; A0-A3 scoped, no implementation started**
- Created: 2026-08-28
- Hardware lane: **Radeon 8060S / `hip_gfx1151`**
- Primary product key: **Qwen3.8-27B `Q4_K_M`, BF16 KV, production profile**
- Opening position: **C3/K3 retained diagnostic at 19.934 vs 20.788 tok/s true AR (0.9589x) with 78.89% draft acceptance; automatic C3 remains K0**
- Promotion gate: **`>=1.10x` true same-protocol AR, all categories non-regressive, outputs exact, complete production correctness**
- Predecessors (do not reopen; extend only):
  [`QWEN38-Q4KM-MTP-SERVING.md`](QWEN38-Q4KM-MTP-SERVING.md),
  [`CONCURRENCY2-GFX1151-MTP-DYNAMIC-ADMISSION.md`](CONCURRENCY2-GFX1151-MTP-DYNAMIC-ADMISSION.md)
  including its post-closure D6-R1/R2/R3 and C3 rowtile extensions
- Opening evidence:
  [`C3 rowtiles retained`](../benchmarks/results/2026-08-28-gfx1151-qwen38-c3-production-rowtiles-retained.json),
  [`llama.cpp 1:1 comparison`](../benchmarks/results/2026-08-28-gfx1151-qwen38-llamacpp-1to1.json),
  [`external survey`](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md)
- Normative dependencies: [`PLAN.md`](PLAN.md), [`TESTING.md`](TESTING.md),
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
  [`BENCHMARK.md`](BENCHMARK.md), [`CONCURRENCY2.md`](CONCURRENCY2.md)
- Sibling campaign (separate artifact, no shared gates):
  [`QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md`](QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md)

## 1. Objective and definition of success

The verifier kernel-ownership defect that made physical-C3 targets pay a
per-row cliff is closed: R12 target wall fell 398.77-399.48 -> 206.62-208.27 ms
and R9 fell 495.37 -> 195.16-196.37 ms, yet C3/K3 still trails true AR at
0.9589x. The remaining gap is **proposal economics**: our draft tokens are
accepted materially less often than reference implementations on the same or
equivalent workloads, and our depth policy has not been re-ranked on the now
amortized verifier.

This campaign attacks only the proposal side. Success is any of:

1. one or more C>=2 cells clear the `>=1.10x` true-AR gate and are promoted
   with exact outputs and category non-regression; or
2. a fully measured rejection: the acceptance gap is explained and closed to
   the extent it is content-independent, depth is re-ranked, and the remaining
   deficit is attributed to named, measured causes (e.g. irreducible draft
   quality at this quant), recording why no cell qualifies.

A faster route that changes greedy outputs, games the fixed suite, or skips
the true-AR baseline is not a result of this campaign.

## 2. Measured evidence basis

All anchors below are same-host measured values; nothing is inferred.

### 2.1 Our current C3/K3 acceptance (retained route, D24 ten-prompt suite)

| Scope | Acceptance |
| --- | ---: |
| Overall | **78.89%** (471/597) |
| Code | 79.75% (189/237) |
| General English | 84.21% (96/114) |
| **General Japanese** | **68.18% (90/132)** |
| Mixed Japanese/English | 84.21% (96/114) |

### 2.2 External reference acceptance from the reproduction survey

| Reference | Conditions | Acceptance |
| --- | --- | ---: |
| llama.cpp HIP build 10438, same `Q4_K_M` SHA-256, C1/K3, D24 suite, F16 KV ([L3](../benchmarks/results/2026-08-28-gfx1151-qwen38-llamacpp-1to1.json)) | same model file | **90.16%** |
| hipEngine same packet, C1/K3, BF16 KV | same model file | 78.57% |
| Latest mainline Vulkan, `UD-Q4_K_M`, K3, D24 common suite ([survey §10](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md)) | different artifact | 95.45% |
| MikeVeerman stock pin, `UD-Q8_K_XL`, C1->C4 | different artifact/quant | 72.9% -> 66.4% |

Three decisive readings:

- The same-model gap is **11.59 points** (90.16% vs 78.57%) under nearly
  matched protocol. The survey itself names proposal-state alignment as the
  lead.
- Acceptance is nearly flat in concurrency (72.9% -> 66.4% over C1 -> C4), so
  our C3 deficit is not width-explained.
- The cross-artifact 95.45% row is diagnostic only; it is not a binding
  target.

### 2.3 Cycle economics (retained route)

- Steady R12 target: 206.62-208.27 ms; R9: 195.16-196.37 ms.
- Proposal: ~61.9 ms/cycle and host ~10 ms from the pre-tail-close trace;
  **must be re-measured on the final route in A0** before any proposal-side
  work is sized.
- Tokens/cycle scale ~linearly with acceptance at fixed cycle cost. Scaling
  the suite's own counters (471/597 accepted drafts, 720 tokens, fixed cycle
  cost): at 90% acceptance the suite completes with ~9% fewer wasted cycles
  ⇒ ~21.8 tok/s ≈ **1.05x AR**; at 95% ⇒ ~22.6 tok/s ≈ **1.09x AR**.
  Acceptance alignment is therefore the primary lever but likely needs the
  depth/proposal-cost levers beside it to clear 1.10x; both must be measured,
  not assumed.

### 2.4 Depth evidence

- Externally, K4 beat K3/K6/K8 wherever depth was swept on this model family
  (`q38rocm` strict K4 fastest; yandaq K4 31.18 > K6 28.75 > K8 20.99).
- Laurent's adaptive K3-K7 recovered 18.8% over fixed K7 but stayed 6.7%
  below fixed K3: adaptive depth defends against bad deep drafts; it has not
  beaten a well-chosen fixed depth.
- Our strict one-wave WMMA owner already covers physical R16, so a C3/K4
  (R15->R16 tile) verify is cheap to qualify relative to the rowtile work
  already landed.

## 3. What already exists — do not reimplement

- Acceptance telemetry with per-position, per-category, train/heldout scopes
  and all three denominators (draft, position, conditional position) — commit
  `456cc1aaf`.
- D7 static/dynamic admission: concurrency-routed policy, pure K0 fallback,
  lifecycle/pressure/cancellation/SSE ownership. This campaign changes no
  admission machinery; it only feeds it better evidence.
- Physically qualified and numerically gated C3 K1/R6, K2/R9, K3/R12 cells
  plus strict fallbacks and manifest hashes.
- llama.cpp raw-prompt acceptance oracle runs from the survey campaign
  (natural25 and exact raw-prompt packets).
- FP16 recurrent state with FP32 rollback and request-boundary provider
  teardown contracts.
- Sequential contamination and repetition guards in the external-suite
  harness.

## 4. Phase plan

### A0 — re-baseline the retained route (cheap, GPU-light)

- [ ] One cached-only `rocprofv3` decomposition of the final C3/K3 route:
      proposal / target families / accept-commit / host split. Confirm or
      replace the 61.9 ms proposal and 10 ms host anchors.
- [ ] Confirm per-position/per-category acceptance telemetry covers the exact
      cells this campaign will compare.
- [ ] Publish the campaign baseline artifact (current acceptance table +
      cycle decomposition) under `benchmarks/results/`.

Exit: baseline artifact committed. No code changes.

### A1 — proposal-state alignment audit (CPU-first, no code changes)

Compare our GGUF NextN/MTP2 proposal path against the pinned llama.cpp
speculative implementation (survey pins `152d337fa`, `4e97ac86`, and the
build-10438 lineage used in [L3](../benchmarks/results/2026-08-28-gfx1151-qwen38-llamacpp-1to1.json))
and produce a written ranked defect/hypothesis list covering:

- [ ] Draft input construction: exact hidden-state handoff to the MTP head
      (post-norm vs pre-norm state, embedding concatenation, position
      handling).
- [ ] Draft state lifecycle on accept vs reject: rollback vs
      recompute-from-target, and whether rejected-position state ever feeds
      the next proposal.
- [ ] Draft-token KV writes: which positions are written, at what precision,
      and whether the verifier reads the same values the target would have
      produced.
- [ ] Draft-head sampling/argmax semantics: tie-breaking, logits source,
      any temperature or repetition handling leaking into greedy paths.
- [ ] Request-boundary reset of all proposal state (survey follow-up #2).
- [ ] KV precision confound: bound the F16-vs-BF16 acceptance difference on
      our side with one controlled A/B before attributing the gap to state
      alignment.

Exit: each hypothesis carries a content-independence argument and a
falsifying measurement. Nothing is implemented from an unranked list.

### A2 — alignment fixes and acceptance measurement

- [ ] Implement only content-independent fixes, each as its own unit with
      its own gate; record a `REFACTOR.md` entry for any retained flag.
- [ ] Binding gate per fix: ten-prompt suite outputs remain **exact**
      (greedy verification is unchanged, so token IDs must be identical);
      acceptance and economics re-measured on the full multi-category suite
      plus heldouts with the true-AR baseline; sequential multi-prompt
      contamination gate passes.
- [ ] Acceptance ladder target: close toward the 90.16% same-model
      reference. The 95.45% cross-artifact row stays diagnostic.
- [ ] Keep every exact non-regressive speed win per the ground rules even if
      the 1.10x gate is not reached; automatic policy stays K0 until a cell
      clears the gate.

Exit: measured acceptance delta per fix with attribution, or a documented
dead end per hypothesis.

### A3 — depth re-rank on the amortized verifier

- [ ] Numerically qualify the C3/K4 (R15->R16) cell: canonical + heldout D24
      full-logit gates, three deterministic repeats, isolation, teardown —
      same protocol as K1/K2/K3.
- [ ] Measure C3 K1/K2/K3/K4 economics on the final route (same host,
      counterbalanced arms, exact token counts, complete wall).
- [ ] Adaptive depth is diagnostic only in this campaign; do not promote
      adaptive policies here regardless of result.
- [ ] Promote any cell clearing `>=1.10x` with category non-regression;
      otherwise publish the ranked table and the reason no cell qualifies.

Exit: ranked depth table; policy updated only through the D7 admission owner.

### A4 — closure

- [ ] Update `benchmarks/README.md` / `CHANGELOG.md` / artifacts for every
      retained result; worklog entries for each unit; `docs/KERNELS.md` /
      `EXECUTION-PROFILES.md` only if arithmetic or ownership changed.
- [ ] If C>=2 cells remain below 1.10x after A2+A3, record the named,
      measured causes and the deferred follow-ons below with their entry
      criteria.

## 5. Deferred follow-ons (out of scope; entry criteria only)

| Follow-on | Entry criterion | Notes |
| --- | --- | --- |
| Tree-drafted proposals (EAGLE-3/SpecExec-shaped) | A2+A3 leave C3 below 1.10x **and** A0 decomposition shows verifier headroom | `KVLiveSpans` already carries variable live spans/evict masks; needs its own campaign (tree mask, branching proposal, tree acceptance). |
| DFlash2 sidecar draft model | See §9; reopen requires the matched N3 protocol and an amortized multi-row verify | In-tree campaign already closed diagnostic once — §9 records what changed since closure. |
| Stochastic / sampler-matched acceptance | Product decision to serve non-greedy distributions | Changes output semantics, not just speed; out of scope for greedy product keys. |
| ngram composition | Closed | Rejected in [L5](../benchmarks/results/2026-08-28-gfx1151-qwen38-ngram-mtp-composition-closeout.json); replay-scale wins only. |
| `UD-Q4_K_M` artifact work | Sibling campaign | See [`QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md`](QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md). |

## 6. RED contract inventory

| Layer | Required RED |
| --- | --- |
| Draft state | unit tests vs CPU reference for draft logits at fixed target states (accept path, reject path, first cycle, post-refill) |
| Draft input | construction test pinning hidden-state source, norm placement, embedding concat, positions |
| Boundary | proposal state fully reset between requests; no cross-request leakage (survey follow-up #2) |
| Telemetry | acceptance scopes/denominators correct under partial cycles, K4, and mixed categories |
| Depth | K4 manifest/evaluator cells fail-closed like K1-K3; every scope miss retains its prior owner |
| Economics | true-AR arm present in every retained comparison; complete wall and decode reported separately |

## 7. Anti-gaming rules (binding)

- Acceptance or speed improvements must come from content-independent
  mechanisms and be validated on the full multi-prompt mtpbench category
  suite plus heldouts. Single-prompt acceptance deltas are diagnostic only.
- No prompt-conditioned, token-ID-conditioned, or category-conditioned
  branches anywhere in proposal, verification, or admission.
- Every MTP speed claim carries a true no-MTP autoregressive baseline from
  the same protocol; verifier-derived `off`/`B0` rows are diagnostic only.
- Greedy output exactness is a gate, not a metric: a fix that changes token
  IDs fails unless it goes through the full production-profile numerical
  machinery, which this campaign does not plan to open.

## 8. Deliverables

- Campaign baseline artifact (A0) and per-unit `benchmarks/results/*.json`.
- Ranked A1 alignment audit with measurements.
- Acceptance ladder table (before/after per fix, per category).
- C3 K1-K4 economics ranking table.
- Updated benchmark rollups and policy fingerprints via the D7 admission
  owner for any promoted cell.

## 9. Appendix: what a DFlash2 revival looks like on this architecture

This is an architectural preview, not a commitment. DFlash2 is not
hypothetical for this tree: the
[`QWEN38-27B-DFLASH2-CAMPAIGN.md`](QWEN38-27B-DFLASH2-CAMPAIGN.md)
closed **diagnostic** on 2026-08-19 (B3 optimum 8.85 tok/s = 0.66x AR), and
its 2026-08-22 correction rewrote the attribution: DFlash2 was at **acceptance
parity** with exact MTP (2.80 vs 2.85 tokens/cycle; 0.70 vs 0.74 per verify
row). The deficit was entirely cost — ~96 ms/cycle of drafter+select and a
verify running at 2.14 sweeps/cycle at 4 rows and **8.01 sweeps at 8 rows**
(the `_PACK8_ROWTILE_MAX_ROWS = 4` admission cliff). The corrected record
explicitly leaves the door open and names the reverted rowtile-8 divergence
as the top open item.

### 9.1 What changed since that closure

The C3 rowtile campaign landed after that record was written, and it fixed
the same defect family on the production path:

- Row-independent Q4/Q5/Q6 owners now amortize physical R6/R9/R12/R16 instead
  of re-reading weights per row: the retained C3 route verifies 12 rows in
  206.62-208.27 ms ≈ 2.7 sweeps (~0.22 sweeps/row) where the DFlash2 record's
  8-row verify paid 8.01 sweeps (~1.0/row).
- The R9/R12 decompositions (R7+R2, R8+R4) plus same-width isolation and
  per-row state-commit ownership are exactly the machinery N2 said was
  missing when the 2026-05 rowtile-8 attempt diverged and was reverted
  unexplained.
- N1's falsifiable prediction (an exact multi-row rowtile lands a deep verify
  at <=1.5 sweeps) is now half-answered on the MTP side; the DFlash2
  `verify_target_block` harness still needs its own N1 curve because its
  tap-capture path differs.

The structural blocker (verify row cliffs) is therefore no longer a valid
reason to keep DFlash2 closed. The remaining structural costs are the ~96 ms
drafter+select (predicted ~5x headroom: <60 GB/s effective today against the
3.584 GiB drafter residency) and the fixed-B proposal policy (MTP declines
  low-confidence drafts; DFlash2 always pays every row — N4).

### 9.2 Why external DFlash2 routes won where this tree lost

The externally successful DFlash2 routes are llama.cpp-family implementations:
Laurent's fork (FP4 target + 1.03 GB `Q4_0` sidecar, adaptive K3-7: 34.483
token-weighted common-suite tok/s, 60.43% acceptance; 56.532 valid structured
JSON fresh-process) and the PieBru recipes on Nathanw/mainline (UD Q5/Q6/Q8
targets 20.9-31.5 GB + 2.06 GB `Q8_0` sidecar: DFlash decode 30.659/26.470/
23.044 vs AR 10.695/8.778/7.275 tok/s = **2.86x/3.01x/3.17x**, acceptance
53.19%/42.92%/43.94%). `q38rocm` (35.575 arithmetic) is **not** a DFlash2
route — it is built-in MTP K4 on a custom FP4 format; on the one shared FP4
target, Laurent's DFlash2 beat it (34.483 vs 32.969 token-weighted).

Their cycles decompose cleanly against ours at closure. Sweep times: their
targets at ~94-137 ms per full weight read; our Q4_K_M at 77.4 ms:

| Mechanism | External winners | Ours at 2026-08-19 closure | Effect |
| --- | --- | --- | --- |
| Verify weight traffic | **~1.0 sweep/cycle at any B** — the whole draft block is one batched forward in a single compute graph | **2.14 sweeps at B=4, 8.01 at B=8** — per-row owners plus the `_PACK8_ROWTILE_MAX_ROWS = 4` admission cliff re-read weights per row | >1 full sweep saved per cycle at B4; ~7 sweeps at B8 |
| Drafter | 1.0-2.1 GB sidecar on the engine's ordinary efficient decode path — a few ms per cycle | 3.584 GiB residency at **<60 GB/s effective** (unfused/launch-bound forward + unfused select) — **~96 ms/cycle** | ~10x drafter cost gap; the single largest deficit |
| Acceptance | 43.9-60.4% per draft token (≈2.5-2.8 tokens/cycle at their B) | 0.70 tokens/verify row (2.80/cycle) — **parity** | Never the differentiator |
| Timing regime | ~0.9 sweeps/cycle implied end-to-end (Q5 at B3: 1+3x0.5319 = 2.60 tokens/cycle ⇒ ~85 ms cycles vs ~94 ms sweeps) | 264 ms cycles = verify 166 + drafter/select 96 | 8.85 tok/s = 0.66x AR despite parity acceptance |
| Closure comparison hygiene | single engine, single protocol | cross-file (Q4_K_S vs Q4_K_M), cross-harness, 25 vs 40 token budgets, tap capture inside our timed region only | Part of our 2.7x deficit was measurement mismatch (N3), not mechanism |

The one-paragraph answer: **they won because their draft block and verify ride
one native batched graph — weights stream once per cycle — and their drafter
is a small sidecar on the same efficient path; we lost because our DFlash2
harness verified through per-row owners behind a 4-row admission bound and ran
an unfused 3.584 GiB drafter forward. The algorithm (acceptance) was never
the problem, and is measured at parity.**

### 9.3 Why a revival can succeed now

Each closure-time deficit now has a landed fix or a named, sized mechanism:

1. **Verify sweeps — fixed on the production path.** The 2026-08-28 C3
   rowtile campaign amortized physical R6/R9/R12/R16 to ~0.22 sweeps per
   marginal row with exact, isolation-proven, row-independent owners. The
   8-row cliff that produced 8.01 sweeps is the same defect family, now
   closed. N1's falsifiable prediction (deep verify at <=1.5 sweeps) is
   half-answered on the MTP side; the DFlash2 `verify_target_block` harness
   needs its own N1 curve because tap capture changes the path.
2. **Drafter cost — now ordinary kernel work.** The same
   localize-profile-fuse-gate loop that just removed the verifier cliffs
   applies to the drafter: hoisted/fused forward + fused select, strict CPU
   oracle first (D2 kernels), against the recorded >120 GB/s effective
   prediction for the 3.584 GiB residency. This is cost engineering, not
   drafting-quality research — acceptance parity means nothing about the
   drafter's *outputs* needs to change.
3. **State ownership — the missing methodology exists.** The May rowtile-8
   AR divergence was reverted unexplained; the C3 campaign's same-width
   isolation, per-row state-commit, and manifest-hash machinery is the
   toolkit N2 said was missing.
4. **Policy — already owned.** D7 admission routes by concurrency economics;
   a qualified DFlash2 provider enters as another measured cell, and the
   external C1-slack vs C>=2-saturation crossover (Mike: 2.23x at C1, 0.84x
   at C4) is the same admission problem D7 already solves.

Falsifiable projection (not a claim): at measured parity acceptance 2.80
tokens/cycle, an amortized ~1.1-sweep verify (~85 ms on Q4_K_M) plus a drafter
at the predicted >120 GB/s (~32 ms; ~10-15 ms with a sidecar-class drafter)
and ~5 ms select gives ~105-122 ms cycles ⇒ **~23-27 tok/s vs the campaign's
own in-session AR 13.4 = 1.7-2.0x at C1** — before any N4 adaptive-gate gain.
N1-N3 exist to confirm or kill exactly this arithmetic under one matched
protocol.

### 9.4 Architecture mapping (no new scheduler, no new verifier)

DFlash2 slots into seams this campaign's predecessors already own:

| Concern | Existing owner DFlash2 reuses |
| --- | --- |
| Provider identity/lifecycle | Generation-2 `SpecRequestPlan` provider groups, activation/catch-up, refill, teardown — DFlash2 is another speculative provider beside GGUF NextN/MTP2, selected by the D7 admission owner from measured cells |
| Draft execution | Four-axis registry: drafter backbone (conv + selector + multi-row output head) registers as ordinary layer/quant kernels; strict exact parents first, production variants only through the profile gate |
| Verify | The **same** packed target verifier, rowtiles, and `KVLiveSpans` ABI used by MTP — DFlash2's parent-indexed tree rows are variable live spans, which is what the ABI was designed for; no verifier fork |
| State | FP16/FP32 rollback + request-boundary reset contracts; Laurent's cross-request state leak (survey §5) is the named failure mode the contamination gates must cover |
| Memory | Second resident payload (external sidecars measure 1.0-2.1 GB; our in-tree drafter residency is 3.584 GiB) accounted through the resident memory contracts |
| Numerics | DFlash's 5-layer tap capture rides as an execution-profile variant; strict profile keeps the untapped parent exact |

### 9.5 Reopen shape, in order

The old campaign's N-numbering stays authoritative; the order is unchanged,
but N1/N2 now lean on in-tree owners:

1. **N1 verify row curve** for `verify_target_block` rows 1..8 with and
   without tap capture — cheap, standalone, settles sweeps/row on the actual
   DFlash2 harness.
2. **N2 state-ownership root-cause** of the rowtile-8 divergence using the
   landed row-independent owners and isolation methodology instead of a
   fresh attempt.
3. **N3 matched-protocol rerun** — one target file (Q4_K_M), one harness, one
   timing boundary, tap on/off, same suite/budget for AR / MTP / DFlash2.
   Without N3, acceptance parity and the 55 ms verify delta remain
   cross-protocol readings.
4. **Native drafter kernels** (conv + selector + output head) against the
   >120 GB/s effective-bandwidth prediction, strict-exact with CPU oracle.
5. **N4 adaptive proposal gate** from selector scores/top-16 margins so the
   chain stops paying for rows it expects to reject.

### 9.6 External anchors and honesty rules

The DFlash2 external anchors are the Laurent and PieBru/Nathanw rows above:
2.86-3.17x served on UD Q5/Q6/Q8 and 34.483 token-weighted on the shared FP4
target — **diagnostic upside evidence only**: different artifacts, formats,
sidecars, and C1-focused protocols we have not qualified. Any revival claim
requires the same bindings as this campaign — true same-protocol AR baseline,
full mtpbench categories plus heldouts, exact outputs under greedy verify,
sequential-request contamination gates, and `>=1.10x` before any automatic
cell exists. A second diagnostic closure with clean attribution is an
acceptable outcome; the 2026-08-22 correction explicitly forbids closing on
"the gap is unclosable".
