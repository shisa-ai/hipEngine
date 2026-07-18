# PARO Transfer Dashboard From GGUF/MTP Work

Last reviewed: 2026-07-18.

This file is the current PARO transfer queue. The verbatim investigation and
server-tuning notebook is preserved in
[`PARO-GGUF-MTP-TRANSFER-HISTORY.md`](PARO-GGUF-MTP-TRANSFER-HISTORY.md).
Historical throughput tables and completion language in that notebook remain
diagnostic unless the canonical scoreboard explicitly retains them.

## Current Routing And Evidence

| Surface | Current state | Consequence |
| --- | --- | --- |
| gfx1151 PARO single request | Exact direct/HTTP 512/128 token parity passed on Radeon 8060S with the committed fixture. This was a correctness/identity gate, not a throughput run. | Use the shared exact-token route for future direct/server comparisons. |
| gfx1151 native c2-c8 | G3 supersedes P1 with independent-c1-exact physical c2/c4/c8, all-layer state/KV, sparse lifecycle, category/heldout, profiler, and repeated direct scaling. G4/G5 attach those widths to the shared resident owner; blocking F1 and native/serial SSE are retained, and a no-flag OpenAI c4 gate loads the packaged profile outside repository CWD. | Greedy W4/BF16-KV c2/c4/c8 are package-default. c3/c5/c6/c7 are exact profile partitions, not native-width claims. Sampled native groups, context >=1024, non-BF16 KV, and graph replay remain separate gates. |
| gfx1100 native c>N | Direct selected-batch c2 is retained under the unified contract, but public/OpenAI remains width-1 and physical c4/c8 owner symmetry is not closed. | Keep architectures separate; transfer the gfx1151 owner/width gates on W7900 before changing gfx1100 public routing. |
| PARO MTP/DFlash | Clean gfx1151 S4 evidence now covers coarse, synchronized, and graph-shape buckets on the curated 35B target/drafter pair. Exact replay is only 0.14825x AR; branch-copy is faster but correctness-red. | Keep exact replay and default-off DFlash. Do not transfer commit/fusion/group changes until drafter quality and exact native state change the premise. |

The exact-token gate artifacts are
[`direct`](../benchmarks/results/2026-07-11-sol-e5-gfx1151-paro-direct-exact-p512-d128.json)
and
[`HTTP`](../benchmarks/results/2026-07-11-sol-e5-gfx1151-paro-http-exact-p512-d128.json).
The historical native-batch blocker is recorded in
[`2026-07-10...true-c1-shrinking-gates.json`](../benchmarks/results/2026-07-10-gfx1151-paro-true-c1-shrinking-gates.json); current retained evidence is
[`G3 direct`](../benchmarks/results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json),
[`G5 F1`](../benchmarks/results/2026-07-18-gfx1151-paro-g5-f1-server-scaling.json), and
[`G5 SSE`](../benchmarks/results/2026-07-18-gfx1151-paro-g5-sse-server-scaling.json).

## Active Queue

| Order | Work | Status | Exit gate |
| ---: | --- | --- | --- |
| 1 | Localize the gfx1151 native c8 divergence (`SOL-P1`) | Superseded/closed by G3 | Exact multi-row Marlin-K projections plus sparse compact metadata close physical c2/c4/c8; all direct gates are retained. |
| 2 | Build the unified exact direct/server matrix (`SOL-M1`) | Accepted | Manifest/schema v1 joins exact tokens, scoped timings, route/backend/verifier shapes, request latency, memory, and profiler summaries for PARO/GGUF without manual denominators. |
| 3 | Rerun PARO server c1/c2/c4/c8 with exact IDs | Accepted by G5 | Blocking F1 keeps 68/68 rows exact; SSE keeps 100/100 plus 72/72 c8 stress rows exact, with separate timing scopes. |
| 4 | Reopen native c1-c8, sparse, ragged, and shrinking gates | Accepted for physical c2/c4/c8 on gfx1151 | Keep c3/c5/c6/c7 as exact partitions; next transfer is gfx1100 owner c4/c8, then broader sampling/context/KV. |
| 5 | Profile PARO DFlash verifier buckets | Accepted as diagnostic evidence; speed rejected | Clean S4 artifact ranks all required buckets, records exact output, and reports graph misses/hits by shape. |
| 6 | Revisit LM-head/sample fusion or graph shapes | Rejected/parked | Fused target LM-head is 5.16% slower; readbacks are immaterial. Exact replay prevents graph reuse, while graph-reusing branch-copy fails at token 1. |

Do not relabel c3/c5/c6/c7 partitions as native widths or extend beyond the
G3 position/KV/sampling envelope without independent-c1 state/token evidence.
Never promote a route from the legacy batch-shaped oracle.

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
