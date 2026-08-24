# PARO MTP Review and Repair Spike

- **Date:** 2026-08-24
- **Review commit:** `4d32e6e2e3d8406acfbdabfd31b97e3c7274583a`
- **Implementation base:** `050b97936047f8c43ed76dbc690fb9d7d7482c07`
- **Model:** `/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16`
- **Hardware:** AMD Radeon Pro W7900, `hip_gfx1100`
- **Status:** bounded provider-repair spike retained explicit/default-off; fixed-chain parity/lifecycle/manifest/repeat correctness pass, but registered strict D24 economics are `0.9495x` true AR; fast verifier is review-positive but remains unselected pending the full production gate

| Surface | Current evidence | Assessment | Next action |
| --- | --- | --- | --- |
| Fast verifier (`decode_batched`) | Four-category D64 strict-teacher review: `278/280 = 99.286%` top-1, all KL tails and every 97% scope pass; two root-row top-2 swaps have identical top-5 sets. Paired 64-token task review passes `4/4`. | Review-positive T2 candidate, **not yet production-qualified**: Wilson 95% top-1 interval `[97.433%, 99.804%]`, only one prompt/category, no three-repeat or complete applicable B1 state/KV/rollback/isolation packet. BF16-relative evidence is conditional and no full-precision target teacher is present locally. | Keep registered but uncertified/unselected; run the complete canonical PARO suite/heldouts, repeats, and applicable state/task gates before promotion. |
| Strict verifier (`c1_loop`) | D24 canonical: exact `10/10`, `240/240`; `90.405 tok/s`, `0.8114x`, `16.727 ms/cycle`. State exact at cycles 1/2/4/8. D64 four-heldout: exact `256/256`, `0.8418x`. | Sound oracle/fallback, currently too slow. | Preserve as selected fallback while the fast candidate completes the full T2 campaign. |
| Shared acceptance at D24 | Fast and strict have identical traces: `79/151 = 52.32%` draft acceptance, `79/240 = 32.92%` accepted/output. | Strict arithmetic is not the acceptance problem. The old `7.48%` N4 result predates grouped PARO heads and is stale for current economics. | Use current canonical suite and longer horizon for every decision. |
| Target-hidden input | PARO passed the pre-final-norm last-layer BF16 tap. nano-vLLM-amd `5d8f496da5e3` and vLLM `470229c37efa` pass final output-normalized target hidden. | Contract mismatch repaired; the isolated five-fixture fixed-chain native/reference gate passes all four categories plus reject/full-accept coverage. | Keep the corrected contract; no clean-slate provider rewrite is justified by proposal quality. |
| Proposer reseed | Target hidden previously seeded prompt prefill only; cycle repair continued from MTP-owned hidden. | Selected-target-hidden reseed is implemented. Native/reference drift begins at BF16 FC reduction order, not input fusion; full-vocab KL remains <= `1.44e-4` with 100% top-1/top-5 agreement in the declared fixed-chain fixtures. | Preserve the scoped numerical evidence; do not claim strict bitwise hidden or branching top-8 parity. |
| Draft LM head | Private F16 head scored rows `[0, 65536)` while target AR/verifier uses resident full-vocab W8A16 over 248,320 rows. | Borrowed ownership/lifecycle passed and removes 970 MiB. Isolated uv-native fused, rescored, fresh-scratch, and materialized W8 top-1 all agree; full-shape fused/unfused kernel parity passes. | Keep the borrowed scorer in the explicit route; the remaining D24 loss is verifier economics. |
| Vocab-cap impact | `32/72` D24 rejections were outside-cap; general Japanese was `24/30`. Full F16 raised acceptance but usually added enough wall to regress economics. | Target W8A16 spike removes cap failures efficiently and raises D24 pooled acceptance to 80.92%. | Keep cap/private-F16 only as opt-out until promotion gates close. |
| Route selection | Registered strict and production manifests now bind the corrected proposer to strict B1 graph-off verification; fast is registered only as uncertified. | Exact fallback/provenance is reproducible; unsupported provider scope fails closed. | Keep fast unselected until its full production numerical/task gate passes. |
| Shared engine infrastructure | `TargetVerifyBatch`, `KVLiveSpans`, accept/commit, journals, NativeSpecCycle, and strict fallbacks are functional. | No evidence supports discarding the engine. | Preserve these boundaries during the spike and any later provider replacement. |
| Provider-contract spike | Registered strict D24 three-run: exact `720/720`, deterministic `321/390 = 82.31%` acceptance, but weighted `104.929` versus `110.515` true-AR tok/s (`0.9495x`). Strict D64 heldout remains exact `256/256` at `1.0220x`. | Correct provider, workload-dependent economics; D24 fails overall and three category non-regression gates. | Keep `HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT=1` explicit/default-off; optimize strict verifier wall rather than redesigning the provider. |

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

Fast D64 differs from strict `general_en_explain` at output index 24
(`13 -> 4016`), so provider repair alone does not authorize fast arithmetic.
The expanded strict-teacher and paired-task review is favorable, however: the
completed fast continuation is fluent and technically valid, and the Japanese
swap is also task-equivalent. Fast remains a separate incomplete T2 campaign;
strict remains the selected fallback.

The spike remains explicit rather than promoted. Qualification status as of
2026-08-24:

| Gate | Status | Evidence / consequence |
| --- | --- | --- |
| Native/reference proposal parity | **passed for declared fixed-chain scope** | Five fixtures cover four categories plus known reject/full-accept transitions: 100% top-1 and top-5-set agreement, max full-vocab KL `1.44e-4`, hidden cosine >= `0.9999898`. Drift begins at BF16 FC reduction order. Exact hidden bytes and branching top-8 are not claimed; borrowed scoring does not expose branching top-k. |
| Three same-schedule D24 repetitions | **correctness passed; economics rejected** | `30/30` prompt runs and `720/720` IDs exact with deterministic traces. Weighted MTP/AR is `0.9495x`; only general-English beats AR, while code/general-Japanese/mixed regress. No default promotion. |
| Borrowed-pointer lifecycle and memory | **passed** | Closed-owner launches fail before use; close/reuse is stable; borrowed scoring saves `1,017,114,848` bytes versus the private F16 head. Teardown has one bounded, non-growing 8-byte runtime residue, reported explicitly rather than called exact-zero. |
| Registered strict/production route manifest | **passed** | Strict hash `3199678e604d...5723`; production hash `9ea22c030d76...8876`. Both select strict verification; fast D64 is registered only as an uncertified/unselected candidate. |
| Fast-verifier production numerical/task gate | **review-positive; incomplete** | Expanded W7900 matrix covers 280 rows/four categories: top-1 `99.286%`, mean/p95/p99/max KL `0.000377/0.001647/0.009209/0.013702`, and all scopes pass. Two narrow top-2 swaps alter exact decisions, but predeclared paired task review passes `4/4`. Confidence straddles 99%; the complete canonical PARO suite, three repeats, and applicable B1 state/task evidence are absent. The exact 18-prompt/450-row calibration count is not a binding cardinality, and BF16-relative evidence applies only if a full-precision target teacher exists. |

The actual next-gate contract is source-bound rather than cardinality-bound:

1. keep the frozen mean/p95/p99/max KL and 99% overall / 97% per-scope limits;
2. run every prompt in `benchmarks/prompts/mtpbench-code-general-ja.jsonl`
   across all four categories and its applicable heldouts, per project anti-gaming
   policy and `EXECUTION-PROFILES.md` section 6.4;
3. repeat the same manifest/schedule at least three times;
4. cover the route's applicable B1 state surfaces: repeated reject/full-accept,
   selected-state/KV ownership, rollback/isolation, long horizon, reuse, and
   teardown. Generic c4/c8 multi-request dynamics are outside this B1 route's
   declared scope unless that scope expands; and
5. run BF16-relative non-inferiority only if a full-precision target teacher is
   available. The local packed model directory has a packed target and BF16 MTP
   sidecar, but no full-precision target teacher.

`18 prompts/450 rows` describes the threshold calibration and other campaign
packets; it was not written as a universal candidate sample-size requirement.

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
| Registered route manifests | `74c69f59b`; strict `3199678e604d...5723`, production `9ea22c030d76...8876` |
| Native/reference fixed-chain parity | `benchmarks/results/2026-08-24-w7900-paro-mtp-native-reference-parity.json`; `2a66339f9` |
| Borrowed-pointer lifecycle pass | `benchmarks/results/2026-08-24-w7900-paro-mtp-lifecycle-gate.json`; `bf849a150` |
| Strict D24 three-run decision | `benchmarks/results/2026-08-24-w7900-paro-mtp-strict-d24-3run.json` |
| Fast-verifier four-category review | `benchmarks/results/2026-08-24-w7900-paro-fast-verifier-four-category-review.json`; rubric `108a6ae05`; aggregate `f73f286bc` |
| Grouped PARO head repair | `52973ce02` |
| Fast/strict review commit | `4d32e6e2e` |
| nano-vLLM reference | `/home/lhl/amd-gpu-tuning/nano-vllm-amd@5d8f496da5e3`, read-only |
| vLLM reference | `/home/lhl/vllm/vllm-main@470229c37efa`, read-only |
