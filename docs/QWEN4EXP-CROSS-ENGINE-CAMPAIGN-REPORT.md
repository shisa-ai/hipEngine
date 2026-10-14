# Qwen4Exp cross-engine gap campaign report

Status: 2026-10-13. This report records the cross-engine evidence and the
resulting direction; it is not a production-promotion report.

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

## Conclusion and results

The family gaps were explained by arithmetic class and, for routed experts
and attention, by kernel design. gufo and halo-box run F16/BF16 operands
through WMMA for every family at every layer with no repair pass, while
hipEngine's `production` profile keeps exact F32, iu8-plus-repair, and scalar
FMA paths wherever a reduced-precision route exceeds its calibrated envelope.
No same-operand replay inside an external engine was needed.

The work landed first in an opt-in `fast` execution profile, measured
in-tree against the gufo kernel-time attribution, and became the `production`
plan later on 2026-10-13 UTC after the complete composition passed the
production gate.
Same host and protocol as the baseline table above:

| Engine / profile | p512 | p1024 | p4096 | p4096 decode |
| --- | ---: | ---: | ---: | ---: |
| hipEngine `production` before the composition | 306.6 | 331.4 | 337.2 | 20.6 |
| hipEngine `production` with the composition | 769.2 | 932.9 | 1108.4 | 20.6 |
| hipEngine `production`, plus two bit-identical prefill kernels | 779.4 | 977.9 | 1190.3 | 20.6 |
| hipEngine `production` on 2026-10-14, after the hyper-connection, GDN and expert-glue kernels | 1028.5 | 1276.5 | 1562.2 | 20.5 |
| halo-box master `6aa81b88e` (2026-10-12) | 759.3 | 1109.1 | 1446.4 | 25.1 |
| gufo `v0.12.0` (2026-10-12) | 1095.0 | 1482.2 | 1736.4 | 28.1 |

| Family, kernel ms per code-p4096 prefill | Before | `production` on 2026-10-13 | gufo |
| --- | ---: | ---: | ---: |
| Routed experts | 4621 | 694 | 812 |
| Dense projections | 3045 | 1283 | 841 |
| Attention, including block scoring and the indexer projections | 1152 | 333 | 113 |
| GDN | 750 | 347 | 208 |
| Hyper-connection elementwise, router, reduce, conversions, other | 2618 | 1001 | 399 |
| **Total** | **12186** | **3657** | **2373** |

On 2026-10-14 the total is 2564 ms: routed experts 699, dense projections
659, the hyper-connection read and write 544, GDN 271, attention with block
scoring and the indexer 211, router, expert reduce and conversions 167
([per-kernel table and the changes behind it](../benchmarks/results/2026-10-14-qwen4exp-prefill-hc-gdn-batch/README.md);
these families are drawn by kernel role, so the hyper-connection projections
count under the hyper-connection row rather than under dense projections).

Quality of the 2026-10-14 plan against the strict hipEngine teacher: 13/18
strict-exact on the pass-1 battery with aligned top-1 agreement 1.00
(halo-box 11/18, gufo 13/18 under the same scoring), and on the
full-vocabulary layer2 gate mean KL 6.30e-04, max 0.010, top-1 99.56%, 13/18
strict-exact tasks.

hipEngine's prefill rate is now 0.94x / 0.86x / 0.90x of gufo's at p512 /
p1024 / p4096 and ahead of halo-box at every length; gufo's decode rate is
1.37x hipEngine's. The largest remaining prefill items are attention block
scoring and the indexer, the GDN recurrence, and the expert glue; at 1024
tokens the routed-expert kernels are bound by reading every expert's
weights. Decode has not been worked on. A new reduced-precision route lands in `fast` first and
becomes a production route when the complete composition passes the
production gate.
