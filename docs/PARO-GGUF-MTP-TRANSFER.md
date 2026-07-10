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
| gfx1151 native c2-c8 | Blocked by the independent-c1 oracle. At clean `0c184517`, serial c8-to-c1 passes every row; native c8 diverges on every row at generated token index 2. | Production greedy and sampled batches use exact width-1 sessions. Schema-1 timing rows have `performance_claim=false` and cannot select routing. |
| gfx1100 native c>N | Older direct retained evidence exists, but it predates the unified exact-ID/provenance/server matrix and is not a gfx1151 baseline. | Keep architectures separate and rerun the current contract on W7900 before changing gfx1100 claims. |
| PARO MTP/DFlash | Coarse verifier buckets and shape-keyed graph attribution exist. The current public DFlash row is retained only under its recorded legacy gate. | Collect one clean, current real-model profile before moving verifier math or defaults. |

The exact-token gate artifacts are
[`direct`](../benchmarks/results/2026-07-11-sol-e5-gfx1151-paro-direct-exact-p512-d128.json)
and
[`HTTP`](../benchmarks/results/2026-07-11-sol-e5-gfx1151-paro-http-exact-p512-d128.json).
The native-batch blocker is recorded in
[`2026-07-10...true-c1-shrinking-gates.json`](../benchmarks/results/2026-07-10-gfx1151-paro-true-c1-shrinking-gates.json).

## Active Queue

| Order | Work | Status | Exit gate |
| ---: | --- | --- | --- |
| 1 | Localize the gfx1151 native c8 divergence (`SOL-P1`) | Open, highest priority | Teacher-forced hidden, linear-state, KV, and token comparisons identify the first mismatching layer/substage at token index 2. |
| 2 | Build the unified exact direct/server matrix (`SOL-M1`) | Accepted | Manifest/schema v1 joins exact tokens, scoped timings, route/backend/verifier shapes, request latency, memory, and profiler summaries for PARO/GGUF without manual denominators. |
| 3 | Rerun PARO server c1/c2/c4/c8 with raw IDs | Awaiting the first clean matrix run; width-1 production route is safe | Same fixture/model/quant/target, all-choice exact output, clean provenance, owned timing, and explicit width/queue shapes. |
| 4 | Reopen native c1-c8, sparse, ragged, and shrinking gates | Blocked on item 1 | Every row matches independent `prefill_native()+step()`; profiler proves the intended native kernels ran. |
| 5 | Profile PARO DFlash verifier buckets | Open | A clean real-model artifact ranks draft, target attention/MoE, LM-head/top1, accept, commit/scatter, graph, sync, and scheduler wall. |
| 6 | Revisit LM-head/sample fusion or graph shapes | Evidence-gated | The current profile identifies a material bucket and the replacement passes exactness plus end-to-end A/B. |

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
- [Concurrency design and history](CONCURRENCY.md)
- [DFlash design](DFLASH.md)
- [Optimization punchlist](SOL-OPTIMIZATION.md)
