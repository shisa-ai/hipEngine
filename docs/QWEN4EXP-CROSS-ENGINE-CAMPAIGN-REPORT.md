# Qwen4Exp cross-engine gap campaign report

Status: 2026-10-12. This report records the current evidence and the remaining
requirements; it is not a production-promotion report.

## Apples-to-apples baseline

Same four-shard `Qwen3.8-Flash-Next UD-Q4_K_XL` GGUF, canonical fixture,
Radeon 8060S/gfx1151 host, and pass-1 protocol:

| Engine | p512 | p1024 | p4096 | p4096 decode |
| --- | ---: | ---: | ---: | ---: |
| hipEngine production | 305.1 | 328.9 | 334.4 | 20.6 |
| halo-box master `6aa81b88e` | 759.3 | 1109.1 | 1446.4 | 25.1 |
| gufo `v0.12.0` | 1095.0 | 1482.2 | 1736.4 | 28.1 |

At p4096 the prefill gaps are 4.33x for halo-box and 5.19x for gufo.
Quality characterization against the strict hipEngine teacher is 15/18,
11/18, and 13/18 strict-exact respectively. Halo-box has top-1 agreement
1.00 with aligned top-20 mean KL 0.241; gufo has no logprob surface for G2.

Source artifact:
`benchmarks/results/2026-10-12-qwen4exp-cross-engine-pass1/README.md`.

## Implementation inventory

The implementation matrix is maintained in
[`QWEN4EXP-CROSS-ENGINE-GAP-MATRIX.md`](QWEN4EXP-CROSS-ENGINE-GAP-MATRIX.md).
The source review found that hipEngine already has grouped MoE metadata,
Q8_1 activation reuse, shape-specific HC routes, and promoted dense/GDN
owners. Gufo's concrete differentiators include a count-aware MMQ column-tile
objective, a parallel expert-ID map, and a wave64 gate/up path with a fused
SwiGLU epilogue. Halo-box's Qwen4Exp source is marked incomplete as a reference;
its pass-1 result remains a characterization, not a porting specification.

## Controlled hipEngine diagnostics

- QSA head-tile screen: all variants bit-identical. Head-pair wins at rows=32
  but loses to head-quad by 8.8% at rows=256; no global tile replacement is
  justified.
- Routed-MoE synthetic screen: tile16 is about 2.2x faster than tile32 for
  both balanced and sparse 256-row distributions.
- Real grouped `code-p4096` capture: 1,152 prefix records, 5,120 routed lanes
  per record, mean nonempty bucket 16.04 lanes.
- Real generic WMMA capture: 2,184 records, tile capacity 800, 7,088–9,440
  row tiles per record, mean 8,255.65.
- Gufo policy replay on the captured histogram selects column width 16, but
  this is not directly comparable to hipEngine's row-tile geometry.

Artifacts:
`benchmarks/results/2026-10-12-qwen4exp-cross-engine-gap-qsa-head-tile.json`,
`benchmarks/results/2026-10-12-qwen4exp-cross-engine-gap-qsa-head-tile-rows256.json`,
`benchmarks/results/2026-10-12-qwen4exp-cross-engine-gap-moe-main-kernel.json`,
`benchmarks/results/2026-10-12-qwen4exp-cross-engine-gap-routing-code-p4096-grouped-summary.json`,
`benchmarks/results/2026-10-12-qwen4exp-cross-engine-gap-routing-code-p4096-wmma-summary.json`,
and
`benchmarks/results/2026-10-12-qwen4exp-cross-engine-gap-routing-tile-policy-replay.json`.

## Correctness gates

The existing pass-1 quality gates cover greedy identity, aligned top-20
fidelity where available, determinism, and the 18-prompt task battery. The
hipEngine production contract additionally requires exact control/ownership,
calibrated mean/p95/p99/max KL, top-1 by scope/category, isolation, BF16
relative checks, and task quality. No new candidate has been promoted, so no
candidate-specific production gate is claimed here.

## Remaining blockers

The campaign is not complete. The missing evidence is:

1. A same-operand external replay that measures gufo or halo-box kernels
   against the captured hipEngine operands/histograms.
2. A candidate implementation change selected from that replay.
3. Full candidate correctness and canonical p512/p1024/p4096 remeasurement.
4. A final keep/reject decision based on both performance and quality.

Until those items land, the current conclusion is source-level and diagnostic:
hipEngine has already adopted several external-style mechanisms, QSA and
routed-expert geometry remain the highest-value areas, and no input-agnostic
change has yet been proven safe and faster under the complete production
contract.
