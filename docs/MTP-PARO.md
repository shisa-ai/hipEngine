# PARO MTP Review and Repair Spike

- **Date:** 2026-08-24
- **Review commit:** `4d32e6e2e3d8406acfbdabfd31b97e3c7274583a`
- **Implementation base:** `4d32e6e2e3d8406acfbdabfd31b97e3c7274583a`
- **Model:** `/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16`
- **Hardware:** AMD Radeon Pro W7900, `hip_gfx1100`
- **Status:** bounded provider-repair spike approved; clean-slate decision deferred until the spike gate

| Surface | Current evidence | Assessment | Next action |
| --- | --- | --- | --- |
| Fast verifier (`decode_batched`) | D24 canonical: exact `10/10`, `240/240`; `97.950 tok/s`, `0.8775x` AR, `15.382 ms/cycle`. State differs at cycle 1 (`57/60` linear, `20/20` K/V). Thinking-off D64 `general_en_explain` first diverges at output 24 (`13 -> 4016`). | Useful T2 production candidate, **not** strict or production-qualified. | Keep explicit/default-off; qualify against strict only after proposer repair. |
| Strict verifier (`c1_loop`) | D24 canonical: exact `10/10`, `240/240`; `90.405 tok/s`, `0.8114x`, `16.727 ms/cycle`. State exact at cycles 1/2/4/8. D64 four-heldout: exact `256/256`, `0.8418x`. | Sound oracle/fallback, currently too slow. | Preserve registered primitives; optimize only after the shared proposer is corrected. |
| Shared acceptance at D24 | Fast and strict have identical traces: `79/151 = 52.32%` draft acceptance, `79/240 = 32.92%` accepted/output. | Strict arithmetic is not the acceptance problem. The old `7.48%` N4 result predates grouped PARO heads and is stale for current economics. | Use current canonical suite and longer horizon for every decision. |
| Target-hidden input | PARO passes the pre-final-norm last-layer BF16 tap. nano-vLLM-amd `5d8f496da5e3` and vLLM `470229c37efa` pass final output-normalized target hidden. | Provider contract mismatch. | Capture final-normalized target hidden for prompt prefill and each selected verifier row. |
| Proposer reseed | Target hidden seeds prompt prefill only. After verify/commit, `advance_with_previous_hidden()` continues from MTP-owned hidden. | Provider lifecycle contradicts the selected-target-hidden handoff specified in `docs/MTP.md`. | Reseed proposal from the selected target hidden after every cycle. |
| Draft LM head | Private F16 head scores rows `[0, 65536)` while target AR/verifier uses resident full-vocab W8A16 over 248,320 rows. | Decision-surface mismatch, guaranteed multilingual misses, and duplicate ownership. | Reuse the target resident W8A16 full-vocab scorer/top-1 result. |
| Vocab-cap impact | `32/72` D24 rejections are outside-cap; general Japanese is `24/30`. Full F16 raises acceptance but usually adds enough wall to regress economics. | Cap is a correctness-of-opportunity defect; universal F16 is the wrong repair. | Test shared W8A16 full-vocab scoring against both cap65536 and full-F16 controls. |
| Route selection | Strict and fast are composed manually from several environment variables. | Unsupported hybrid arithmetic is easy to invoke and hard to reproduce. | Add one cold-path strict/production route plan and manifest. |
| Shared engine infrastructure | `TargetVerifyBatch`, `KVLiveSpans`, accept/commit, journals, NativeSpecCycle, and strict fallbacks are functional. | No evidence supports discarding the engine. | Preserve these boundaries during the spike and any later provider replacement. |

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

## Decision After The Spike

### Retain the stitched PARO provider when

- corrected target-hidden and scoring contracts pass native/reference parity;
- strict remains exact;
- a production candidate passes every numerical/task/lifecycle gate; and
- the complete canonical suite beats true AR with a credible margin.

### Return to a clean-slate provider when

After S2+S3, the matched provider still fails either of these:

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
| Grouped PARO head repair | `52973ce02` |
| Fast/strict review commit | `4d32e6e2e` |
| nano-vLLM reference | `/home/lhl/amd-gpu-tuning/nano-vllm-amd@5d8f496da5e3`, read-only |
| vLLM reference | `/home/lhl/vllm/vllm-main@470229c37efa`, read-only |
