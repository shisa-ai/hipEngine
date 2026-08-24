# PARO MTP Review and Repair Spike

- **Date:** 2026-08-24
- **Review commit:** `4d32e6e2e3d8406acfbdabfd31b97e3c7274583a`
- **Implementation base:** `050b97936047f8c43ed76dbc690fb9d7d7482c07`
- **Model:** `/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16`
- **Hardware:** AMD Radeon Pro W7900, `hip_gfx1100`
- **Status:** qualified production B1 default: corrected target-contract provider + fast `decode_batched` verifier; registered strict `c1_loop` profile/fallback retained

| Surface | Current evidence | Assessment | Next action |
| --- | --- | --- | --- |
| Fast verifier (`decode_batched`) | Complete D64 gate: 10 canonical prompts, 704 rows, 30 captures; `702/704 = 99.716%` top-1, KL mean/p95/p99/max `0.000344/0.001390/0.007830/0.017676`, all binding scopes and three-repeat hashes pass. Paired task review `10/10`; applicable B1 state/lifecycle pass. | **Production-qualified T2 default** for B1 graph-off fixed-chain. | Keep strict `c1_loop` as registered profile/fallback and use fast for production/default. |
| Strict verifier (`c1_loop`) | D24 three-run weighted `104.929 tok/s`, `0.9495x` true AR; exact `720/720`. D64 strict remains the exact oracle. | Sound strict profile/fallback, 10.33% slower than qualified fast MTP overall. | Preserve as explicit strict profile and automatic production rollback. |
| Shared acceptance at D24 | Fast and strict have identical traces: `79/151 = 52.32%` draft acceptance, `79/240 = 32.92%` accepted/output. | Strict arithmetic is not the acceptance problem. The old `7.48%` N4 result predates grouped PARO heads and is stale for current economics. | Use current canonical suite and longer horizon for every decision. |
| Target-hidden input | PARO passed the pre-final-norm last-layer BF16 tap. nano-vLLM-amd `5d8f496da5e3` and vLLM `470229c37efa` pass final output-normalized target hidden. | Contract mismatch repaired; the isolated five-fixture fixed-chain native/reference gate passes all four categories plus reject/full-accept coverage. | Keep the corrected contract; no clean-slate provider rewrite is justified by proposal quality. |
| Proposer reseed | Target hidden previously seeded prompt prefill only; cycle repair continued from MTP-owned hidden. | Selected-target-hidden reseed is implemented. Native/reference drift begins at BF16 FC reduction order, not input fusion; full-vocab KL remains <= `1.44e-4` with 100% top-1/top-5 agreement in the declared fixed-chain fixtures. | Preserve the scoped numerical evidence; do not claim strict bitwise hidden or branching top-8 parity. |
| Draft LM head | Private F16 head scored rows `[0, 65536)` while target AR/verifier uses resident full-vocab W8A16 over 248,320 rows. | Borrowed ownership/lifecycle passed and removes 970 MiB. Isolated uv-native fused, rescored, fresh-scratch, and materialized W8 top-1 all agree; full-shape fused/unfused kernel parity passes. | Keep the borrowed scorer in the explicit route; the remaining D24 loss is verifier economics. |
| Vocab-cap impact | `32/72` D24 rejections were outside-cap; general Japanese was `24/30`. Full F16 raised acceptance but usually added enough wall to regress economics. | Target W8A16 spike removes cap failures efficiently and raises D24 pooled acceptance to 80.92%. | Keep cap/private-F16 only as opt-out until promotion gates close. |
| Route selection | Strict manifest `3199678e...5723` selects c1-loop exact; production manifest `92073760...9561` selects fast decode-batched with exact strict fallback. Omitting both profile and manual chain mode selects production fast. | Qualified default and rollback are reproducible and fail closed outside B1 graph-off fixed-chain. | Remove temporary env/manual duplicate selection after one release window. |
| Shared engine infrastructure | `TargetVerifyBatch`, `KVLiveSpans`, accept/commit, journals, NativeSpecCycle, and strict fallbacks are functional. | No evidence supports discarding the engine. | Preserve these boundaries during the spike and any later provider replacement. |
| Provider-contract production route | Fast D24 three-run: exact `720/720`, deterministic `80.92%` draft acceptance, weighted `115.770` versus `110.830` true-AR tok/s (`1.0446x`). Fast beats strict MTP `10.33%` overall and `8.03%-12.96%` in every category. | Correct and economically positive qualified production composition. | Default to corrected provider + fast verifier; retain strict profile/fallback. |

## Scope

This document owns the **PARO target + BF16 MTP sidecar** provider review. It does
not redefine GGUF `llama-compat`, dense GGUF NextN, or DFlash policy. Those paths
are controls and infrastructure references only.

Normative policy remains in:

- [`PLAN.md`](PLAN.md)
- [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md)
- [`TESTING.md`](TESTING.md)
- [`BENCHMARK.md`](BENCHMARK.md)

Detailed historical MTP experiments remain in [`MTP.md`](MTP.md). The compact
review artifact is
[`2026-08-24-w7900-paro-mtp-fast-strict-review.json`](../benchmarks/results/2026-08-24-w7900-paro-mtp-fast-strict-review.json).

## What The Review Changed

### The old acceptance diagnosis is superseded

The July current-packed N4 row accepted `16/214 = 7.48%`. It was measured before
`52973ce02` corrected packed PARO GDN V-head mapping from the GGUF tiled mapping
to canonical grouped `repeat_interleave` semantics. Before that repair, the
complete PARO runtime measured mean KL `12.07584` and `0%` top-1 against the
independent ParoQuant runtime. After repair it measures mean KL `0.00115135` and
`98.89%` top-1.

The current D24 canonical run is therefore the valid baseline. Fast and strict
produce identical IDs and acceptance. Any proposal-quality or verifier-speed
work must start from that fact.

### Short exact IDs are not enough for fast

The fast route saves `1.346 ms/cycle` at D24 without changing its 240 observed
IDs. Nevertheless, its selected state is already different after cycle 1. The
D64 heldout first changes a generated ID immediately after the old 24-token
window. Fast is a plausible T2 implementation-drift candidate, but only the
strict teacher-forced production envelope—not a short free-running equality
sample—can qualify it.

### The provider is now the first repair target

The verifier comparison holds proposal and acceptance traces constant, while the
provider audit finds mismatched hidden and scoring boundaries shared by both
routes. More verifier fusion before repairing those boundaries would optimize a
cycle whose draft decisions are not generated under the intended model contract.

## Bounded Repair Spike

The spike is one provider unit with three ordered slices. Each slice keeps the
existing path as an opt-out until its complete gate passes.

### S1 — coherent route contract and RED tests

1. Declare `strict` and `production_candidate` PARO MTP plans.
2. Bind the complete strict set together; partial strict flag combinations are
   diagnostic-only and cannot claim a named profile.
3. Add RED contracts for:
   - final output-normalized target-hidden capture;
   - selected target-hidden reseed after reject, partial accept, and full accept;
   - target scorer ownership and full-vocabulary coverage;
   - exact strict fallback resolution.

### S2 — target-hidden lifecycle repair

1. Capture the target's final output-normalized hidden row during prompt prefill.
2. Capture/select the corresponding final-normalized verifier row at accept.
3. Reseed the next proposal from that selected target hidden. Do not substitute
   the MTP block's own previous hidden for the target observation.
4. Preserve independent MTP attention K/V and exact target transaction ownership.
5. Compare native proposal top-1/top-k and hidden/KV state against the existing
   torch/reference forward on identical captured rows.

### S3 — matched full-vocabulary scoring

1. Let the proposer borrow the target session's resident W8A16 head and scale.
2. Score the full 248,320-token vocabulary through the existing fused top-1 path;
   do not materialize a second full F16 head or add a vocab-sized D2H.
3. Remove cap-caused Japanese misses from the candidate path.
4. Keep cap65536 and private-F16 paths only as diagnostic controls until the
   complete gate decides cleanup.

## Spike Gate

### Focused correctness

- Native and torch/reference proposal top-1/top-k match on the same target hidden.
- Reject, partial accept, and full accept reseed from the correct selected target
  row.
- Strict state remains exact for all Conv/GDN and live K/V records.
- Target correction, accepted count, commit row, cursors, and GPU/CPU acceptance
  remain exact.

### Canonical quality and lifecycle

- Canonical ten-prompt category+heldout suite, train/heldout/category rollups.
- D24 and D64, including thinking-off heldouts.
- Three same-schedule candidate repeats for any production arithmetic claim.
- Full-vocabulary strict-teacher mean/p95/p99/max KL and scoped top-1 gates from
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md).
- Determinism, finite state/logits, rejection rollback, session reuse, and clean
  teardown.

### Economics

Compare against both current routes and true AR:

- proposal/update wall;
- verify and complete cycle wall;
- accepted/draft and accepted/output;
- per-category cap and top-1 miss counts;
- prompt-mean and total-time MTP/AR;
- duplicate-head allocation removed.

A local acceptance increase is not a win if complete full-suite economics or a
heldout/category gate regresses.

## Spike Result — 2026-08-24

S2 and S3 were implemented together behind
`HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT=1`, fail-closed to B1, graph-off, fixed
chain policy without fallback/tree/overlap:

- prompt prefill captures final output-normalized BF16 target hidden;
- verify returns the selected final-normalized target row;
- proposer repair reseeds from that selected target hidden;
- proposal top-1 borrows the target-owned full-vocabulary W8A16 head/scales;
- the spike does not load the private 970 MiB F16 head.

Current evidence:

| Gate | Control | Spike | Verdict |
| --- | ---: | ---: | --- |
| D24 canonical exact IDs | `240/240` | `240/240` | pass |
| D24 pooled draft acceptance | `79/151 = 52.32%` | `106/131 = 80.92%` | +28.60 points |
| D24 accepted/output | 32.92% | 44.17% | +11.25 points |
| D24 weighted MTP | 97.12 tok/s | 109.97 tok/s | +13.23% |
| D24 total-time MTP/AR | `0.8700x` | `0.9907x` | +0.1207x |
| Strict D64 heldout exact IDs | `256/256` | `256/256` | pass |
| Strict D64 pooled acceptance | 39.78% | 84.67% | +44.89 points |
| Strict D64 weighted MTP | 92.46 tok/s | 113.43 tok/s | +22.68% |
| Strict D64 total-time MTP/AR | `0.8339x` | `1.0220x` | crosses break-even |

Fast D64 differs from strict in two narrow heldout top-2 decisions, but complete
paired task review finds both continuations non-inferior. The complete canonical
strict-teacher/repeat/state packet qualifies those T2 differences for production;
strict generated-ID equality remains diagnostic rather than a universal gate.

The spike remains explicit rather than promoted. Qualification status as of
2026-08-24:

| Gate | Status | Evidence / consequence |
| --- | --- | --- |
| Native/reference proposal parity | **passed for declared fixed-chain scope** | Five fixtures cover four categories plus known reject/full-accept transitions: 100% top-1 and top-5-set agreement, max full-vocab KL `1.44e-4`, hidden cosine >= `0.9999898`. Drift begins at BF16 FC reduction order. Exact hidden bytes and branching top-8 are not claimed; borrowed scoring does not expose branching top-k. |
| Strict D24 three-repeat control | **correctness passed; strict economics rejected** | `30/30` prompt runs and `720/720` IDs exact with deterministic traces. Strict MTP/AR is `0.9495x`; this blocks strict as production default but preserves it as exact profile/fallback. Qualified fast economics are recorded below. |
| Borrowed-pointer lifecycle and memory | **passed** | Closed-owner launches fail before use; close/reuse is stable; borrowed scoring saves `1,017,114,848` bytes versus the private F16 head. Teardown has one bounded, non-growing 8-byte runtime residue, reported explicitly rather than called exact-zero. |
| Registered strict/production route manifest | **passed; fast selected** | Strict hash `3199678e604d...5723`; production hash `920737601cca...9561`. Production selects fast decode-batched and names strict c1-loop as exact fallback. |
| Fast-verifier production numerical/task gate | **passed** | Complete W7900 matrix: 10 prompts, 704 rows, 30 captures; top-1 `99.716%`, mean/p95/p99/max KL `0.000344/0.001390/0.007830/0.017676`, all binding scopes and repeat hashes pass. Paired task `10/10`; applicable B1 state/lifecycle passes three identical runs. |
| Fast D24 economics | **passed; production default** | Exact `720/720`; weighted `115.770` MTP versus `110.830` true AR (`1.0446x`). Fast improves strict MTP `10.33%` overall and every category by `8.03%-12.96%`. |

The source-bound production contract is complete: the full canonical suite,
three fixed-schedule repeats, frozen numerical thresholds, predeclared task
review, and applicable B1 state/KV/rollback/isolation/teardown checks all pass.
Generic c4/c8 multi-request dynamics remain outside this B1 route's declared
scope. BF16-relative comparison is inapplicable because the local packed model
directory has no full-precision target teacher.

Compact evidence:
[`provider-contract spike`](../benchmarks/results/2026-08-24-w7900-paro-mtp-provider-contract-spike.json),
[`native/reference parity`](../benchmarks/results/2026-08-24-w7900-paro-mtp-native-reference-parity.json),
and [`borrowed-pointer lifecycle`](../benchmarks/results/2026-08-24-w7900-paro-mtp-lifecycle-gate.json).

## Decision After The Spike

### Retain the stitched PARO provider when

- corrected target-hidden and scoring contracts pass native/reference parity;
- strict remains exact;
- a production candidate passes every numerical/task/lifecycle gate; and
- the complete canonical suite beats true AR with a credible margin.

### Return to a clean-slate provider when

The current S2+S3 spike does **not** trigger a clean slate. Reconsider only if,
after the remaining promotion gates and scorer scheduling cleanup, the matched
provider still fails either of these:

1. proposal quality remains too low to beat true AR under a full-suite optimal
   fixed/adaptive budget; or
2. matching the intended target-hidden/scoring contract costs more than an
   integrated NextN implementation would plausibly save.

A clean slate means replacing the **stitched target/sidecar provider artifact and
proposal owner**, preferably with an integrated target+NextN artifact. It does
not mean replacing the four-axis registry, `KVLiveSpans`, NativeSpecCycle ABI,
strict target verifier, accept/commit transaction, or scheduler integration.

## Evidence Ledger

| Evidence | Result |
| --- | --- |
| Review artifact | `benchmarks/results/2026-08-24-w7900-paro-mtp-fast-strict-review.json` |
| Review worklog | `worklog/entries/20260823T222811.535835Z-lhl-paro-mtp-review-042298.md` |
| Spike artifact | `benchmarks/results/2026-08-24-w7900-paro-mtp-provider-contract-spike.json` |
| Spike worklog | `worklog/entries/20260824T063019.079917Z-lhl-paro-mtp-spike-3108e9.md` |
| Registered route manifests | strict `3199678e604d...5723`, fast production `920737601cca...9561` |
| Native/reference fixed-chain parity | `benchmarks/results/2026-08-24-w7900-paro-mtp-native-reference-parity.json`; `2a66339f9` |
| Borrowed-pointer lifecycle pass | `benchmarks/results/2026-08-24-w7900-paro-mtp-lifecycle-gate.json`; `bf849a150` |
| Strict D24 three-run decision | `benchmarks/results/2026-08-24-w7900-paro-mtp-strict-d24-3run.json` |
| Fast-verifier complete numerical/repeat | `benchmarks/results/2026-08-24-w7900-paro-fast-verifier-full-numerical-repeat-review.json`; `7bd5524da` |
| Fast-verifier complete task | `benchmarks/results/2026-08-24-w7900-paro-fast-verifier-complete-task-review.json`; `058caeb04` |
| Fast-verifier B1 state/lifecycle | `benchmarks/results/2026-08-24-w7900-paro-fast-state-lifecycle-review.json`; `cdad810e2` |
| Fast production economics/default | `benchmarks/results/2026-08-24-w7900-paro-fast-d24-3run-default.json` |
| Grouped PARO head repair | `52973ce02` |
| Fast/strict review commit | `4d32e6e2e` |
| nano-vLLM reference | `/home/lhl/amd-gpu-tuning/nano-vllm-amd@5d8f496da5e3`, read-only |
| vLLM reference | `/home/lhl/vllm/vllm-main@470229c37efa`, read-only |
