# PARO Transfer Dashboard From GGUF/MTP Work

Last reviewed: 2026-07-11.

This file is the current PARO transfer queue. The verbatim investigation and
server-tuning notebook is preserved in
[`PARO-GGUF-MTP-TRANSFER-HISTORY.md`](PARO-GGUF-MTP-TRANSFER-HISTORY.md).
Historical throughput tables and completion language in that notebook remain
diagnostic unless the canonical scoreboard explicitly retains them.

## Current Routing And Evidence

| Surface | Current state | Consequence |
| --- | --- | --- |
| gfx1151 PARO single request | Exact direct/HTTP 512/128 token parity passed on Radeon 8060S with the committed fixture. This was a correctness/identity gate, not a throughput run. | Use the shared exact-token route for future direct/server comparisons. |
| gfx1151 native c2-c8 | Blocked by the independent-c1 oracle. Clean P1 (`a18ff7bc`) rejects every native width at generated token index 2 and localizes the first selected-c1 drift to layer-4 linear state/input; clean P2 (`6f1910c9`) proves ragged sparse c8-to-c1 on the serial route. | Production greedy and sampled batches use exact width-1 sessions. Schema-1 timing rows have `performance_claim=false` and cannot select routing. |
| gfx1100 native c>N | Older direct retained evidence exists, but it predates the unified exact-ID/provenance/server matrix and is not a gfx1151 baseline. | Keep architectures separate and rerun the current contract on W7900 before changing gfx1100 claims. |
| PARO MTP/DFlash | Clean gfx1151 S4 evidence now covers coarse, synchronized, and graph-shape buckets on the curated 35B target/drafter pair. Exact replay is only 0.14825x AR; branch-copy is faster but correctness-red. | Keep exact replay and default-off DFlash. Do not transfer commit/fusion/group changes until drafter quality and exact native state change the premise. |

The exact-token gate artifacts are
[`direct`](../benchmarks/results/2026-07-11-sol-e5-gfx1151-paro-direct-exact-p512-d128.json)
and
[`HTTP`](../benchmarks/results/2026-07-11-sol-e5-gfx1151-paro-http-exact-p512-d128.json).
The native-batch blocker is recorded in
[`2026-07-10...true-c1-shrinking-gates.json`](../benchmarks/results/2026-07-10-gfx1151-paro-true-c1-shrinking-gates.json).

## Active Queue

| Order | Work | Status | Exit gate |
| ---: | --- | --- | --- |
| 1 | Localize the gfx1151 native c8 divergence (`SOL-P1`) | Accepted/closed | Clean P1 localizes the first state/input drift to layer-4 linear attention before the token-index-2 failure; P2 proves the production true-c1 lifecycle. |
| 2 | Build the unified exact direct/server matrix (`SOL-M1`) | Accepted | Manifest/schema v1 joins exact tokens, scoped timings, route/backend/verifier shapes, request latency, memory, and profiler summaries for PARO/GGUF without manual denominators. |
| 3 | Rerun PARO server c1/c2/c4/c8 with raw IDs | Covered by P1/P2 production classification; separate HTTP throughput remains diagnostic-only | Production c>N is explicitly width-1 until a general native algorithm changes P1. |
| 4 | Reopen native c1-c8, sparse, ragged, and shrinking gates | Parked after P1/P2 | Reactivate only when a general native c>N algorithm passes independent-c1 state/token equality. |
| 5 | Profile PARO DFlash verifier buckets | Accepted as diagnostic evidence; speed rejected | Clean S4 artifact ranks all required buckets, records exact output, and reports graph misses/hits by shape. |
| 6 | Revisit LM-head/sample fusion or graph shapes | Rejected/parked | Fused target LM-head is 5.16% slower; readbacks are immaterial. Exact replay prevents graph reuse, while graph-reusing branch-copy fails at token 1. |

Do not resume width-specific c3/c5/c7 tuning or c>8 exploration before the
common c8 divergence is fixed. Do not promote a native route from the legacy
batch-shaped oracle.

## Portable Lessons

These GGUF lessons can transfer after PARO-specific evidence:

- exact raw-token inputs and all-choice accounting;
- explicit queue/backend/verifier shapes and timing ownership;
- startup/cache-shape observability;
- route caps selected from same-protocol evidence;
- accepted-row commit/scatter and rejected-tail lifecycle accounting;
- shape-keyed graph hit/miss/fallback buckets;
- separating full-request wall from backend decode and verifier wall.

These device paths do not transfer directly to PARO:

- GGUF `Q*_K`, Q8/q8_1/dp4a, T16/X8, and Q6_K rowtile kernels;
- llama.cpp compatibility precision/state trades;
- GGUF-specific no-copy GDN capture layout;
- direct partial-commit semantics except as a lifecycle comparison target.

PARO uses `w4_paro` AWQ/WMMA FP16/BF16 activation paths. Porting a GGUF quant
kernel without a matched PARO layout and profiled bottleneck is a category
error.

## Profiling Buckets Required Before Verifier Work

- `draft_propose`
- `metadata_upload`
- `target_verify_attention`
- `target_verify_moe_projection`
- `lm_head_top1`
- `accept_summary`
- `commit_scatter`
- `graph_replay`
- `host_sync_readback`
- `scheduler_wall`

Use synchronized phase buckets only for attribution; their added waits make
them ineligible as throughput claims.

## References

- [Dated PARO transfer notebook](PARO-GGUF-MTP-TRANSFER-HISTORY.md)
- [Canonical benchmark scoreboard](../benchmarks/README.md)
- [Concurrency roadmap and punchlist](CONCURRENCY.md)
- [DFlash design](DFLASH.md)
- [Optimization punchlist](SOL-OPTIMIZATION.md)
