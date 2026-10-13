# Qwen4Exp cross-engine gap matrix

Status: 2026-10-13. Scope: the same `Qwen3.8-Flash-Next` `UD-Q4_K_XL`
GGUF on gfx1151. This is an investigation plan, not a claim that an external
engine's arithmetic is acceptable for hipEngine production.

The pass-1 p4096 prefill attribution is the prioritization signal:

| Family | hipEngine / gufo kernel-time ratio | hipEngine status | Current evidence | First controlled experiment |
| --- | ---: | --- | --- | --- |
| QSA attention | 10.2x | Dedicated QSA kernels exist; parity with external indexed/flash path is unresolved. | External engines use specialized fused/indexed attention paths; exact kernel and arithmetic differences still need source-to-source mapping. | Same Q/K/V/index tensors through each attention stage; split projection, index selection, top-k, gather, and attention reduction. |
| Routed experts | 5.7x | Qwen4Exp runner already uses parallel group count/prefix/scatter metadata and grouped low-precision routes; exact gufo wave64 gate/up equivalence is unresolved. | gufo uses grouped expert gather, one fused routed gate/up path with a SwiGLU epilogue, and batched expert execution; its family bucket includes fused work. | Compare identical routed IDs and activations for separate gate/up/down, paired gate/up, grouped rows, and fused gate/up+activation. |
| Hyper-connection | 4.9x | Shape-specific GR and Q8 activation-reuse paths are present; same-operand cross-engine decomposition is missing. | gufo has shape-specific vector and projection paths; hipEngine has qualified GR owners but not a cross-engine same-operand decomposition. | Compare HC mix, branch projection, combine, and the K=320 down shape separately at rows 1/8/64/4096. |
| Dense projections | 3.6x | Checked dense Q8 DPP/MMQ promotions are live; remaining gap is dispatch/library geometry, not missing basic Q8 support. | hipEngine narrowed this gap with the checked dense Q8 DPP promotion; external builds use mature GGML HIP MMQ/BLAS dispatch. | Replay the same Q8/Q4 weights and activations across candidate kernels, including launch geometry and input quantization. |
| GDN | 2.9x | Tiled16_multi GDN prefill is promoted for its certified suffix; taxonomy and direct equivalence remain unresolved. | External attribution taxonomy may fold some work into other families; no direct equivalence is established. | Compare projection, recurrent update, state movement, and output projection with identical state and chunk boundaries. |
| Router/reduce | diagnostic | gufo's fused expert path changes launch and reduction ownership; direct ratios are not comparable yet. | Not separately attributed. | Attribute routing, ID-map construction, weighted reduction, and residual combine separately. |
| Launch scheduling | cross-cutting | gufo and halo-box run fewer/fused launches in several families; hipEngine's dispatch gaps are material in decode. | Not separately attributed. | Request-delimited launch census and host/device timeline for identical p4096 and 128 decode workloads. |
| Memory movement | cross-cutting | gufo groups expert rows and reuses pre-quantized Q8_1 activations; residency/layout effects are not isolated. | Not separately attributed. | Count H2D/D2H, temporary allocations, quantization bytes, and expert-weight reads per stage. |
| Graph/runtime behavior | cross-cutting | hipEngine has explicit graph/eager policies; external server runtimes have different capture and scheduling behavior. | Not separately attributed. | Compare warm steady-state request traces with graph/capture status and dispatch gaps, without mixing server overhead into kernel claims. |

## What explains the gap

The family ratios above are dominated by arithmetic class, not by tile policy
or launch geometry. gufo and halo-box run F16/BF16 operands through WMMA for
every family at every layer with no repair pass; hipEngine's `production`
profile runs exact F32 dense at the low layers, iu8 WMMA plus a sparse exact
repair for routed experts, exact F32 hyper-connections, and scalar exact FMA
attention. The per-block source comparison is in
[`QWEN4EXP-OPTIMIZATION-LEDGER.md`](QWEN4EXP-OPTIMIZATION-LEDGER.md) section
2.1. Where hipEngine runs the same operand class it is competitive: its F16
dense kernel measures 40.65 TFLOP/s on the 1024x2560x10240 packet shape.

Closing the gap therefore means moving each family to the F16-operand class.
That work lands in the opt-in `fast` execution profile
([`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md)); `production` keeps its
calibrated envelope. No same-operand replay inside an external engine is
required: each family is measured in-tree against the gufo kernel-time column
above.

## Routed-expert tile evidence

The routed-expert tile investigation is complete on the hipEngine side and
did not identify a tile-policy change worth making.

- **Capture point.** Q4 prefill takes the grouped path: it builds
  `group_counts`/`group_expert_start`, runs
  `qwen35_moe_group_scatter_gather_lowp`, then selects the generic
  `qwen35_moe_wmma_tile_map` owner for this shape (the MMQ32 map is selected
  only when `q4_k_mmq_prefill` is true). `scripts/qwen4exp_routing_capture.py`
  wraps only the pair-prefill keys and records nothing for this route; the
  valid observation points are the grouped scheduler's buffers.
- **Grouped histogram, `code-p4096`.** 1,152 prefix records, 5,120 routed
  lanes per record, mean nonempty-expert bucket 16.04 lanes, count-aware
  maxima up to 512
  (`benchmarks/results/2026-10-12-qwen4exp-cross-engine-gap-routing-code-p4096-grouped-summary.json`).
- **WMMA tile map, `code-p4096`.** 2,184 records, tile capacity 800,
  7,088-9,440 row tiles per record, mean 8,255.65
  (`benchmarks/results/2026-10-12-qwen4exp-cross-engine-gap-routing-code-p4096-wmma-summary.json`).
- **Synthetic geometry screen.** Tile16 is about 2.2x faster than tile32 on
  balanced and sparse 256-row distributions; production already runs 16-row
  tiles.
- **gufo policy replay.** gufo minimizes a cost over column widths 16..80 in
  steps of 16 using observed per-expert counts and a fixed panel penalty
  (`executor.cpp:1395-1405`, `qfn_mmq.hip.cpp:42-62`). Replayed on the
  captured histogram it selects width 16. gufo's policy is a column-tile
  width and hipEngine's map is an expert-row tile, so the replay is a
  consistency check rather than a comparison of the same dimension.

## In-tree experiment entry points


The repository already has focused experiments that can serve as the hipEngine
side of the comparison:

- `scripts/qwen4exp_qsa_head_tile_microbench.py` isolates QSA K/V reuse across
  heads 1/2/4 while requiring bit-identical outputs.
- `scripts/qwen4exp_moe_main_kernel_cost.py`,
  `scripts/qwen4exp_moe_gemv_decode_screen.py`, and
  `scripts/qwen4exp_routing_distribution_capture.py` cover routed expert cost,
  decode shapes, and routing distributions.
- `scripts/qwen4exp_gr_operand_replay.py` and
  `scripts/qwen4exp_gr_projection_screen.py` cover same-operand HC/GR shapes.
- `scripts/qwen4exp_delimited_prefill_attribution.py` and
  `scripts/qwen4exp_per_prefill_attribution.py` provide request-delimited
  family attribution for the canonical workload.

## gufo mechanisms to port

The gufo Qwen3.8 Flash-Next ROCm implementation currently exposes these
candidate mechanisms in `src/models/qwen38_flash_next/kernels/rocm/mmq/`:

- A three-pass parallel deterministic expert-ID map builder (`mmid.hip.cpp`)
  replacing a serial assignment-row loop while preserving the output contract.
- Grouped expert row construction and gather/convert/GEMM/scatter for routed
  projections, including per-expert row counts and active-expert batching.
- A wave64 gate/up kernel that computes two output rows at once, shares input
  loads, and applies the SwiGLU epilogue before writing routed slots
  (`moe_wave64.hip.cpp`).
- Reusable Q8_1 activation packing and a pre-quantized routed path, avoiding
  repeated conversion when the caller already owns the required layout.
- Shape-specific HC-down handling for the 320x10240 projection with paired
  K-iterations and recursive token-shape dispatch (`qfn_mmq.hip.cpp`).
- Runtime-selected routed tile columns based on the observed expert-count
  distribution rather than one fixed tile width.

These are candidates, not promotions. Each must be ported or reproduced in
hipEngine's registry-based backend tree, tested against the CPU reference and
strict fallback, and admitted only through the production profile gates.

## Correctness review to run for every candidate

1. Control and ownership: exact token IDs, routing IDs, KV/state hashes, and
   batch-composition invariance where applicable.
2. Numerical envelope: strict-teacher mean/p95/p99/max KL and top-1 by category,
   transition, and scope; BF16-relative checks where required.
3. Repeatability and isolation: repeated runs, fresh process, and candidate
   off/on state teardown.
4. End-to-end quality: the 18-prompt strict-exact battery, including the
   prefill-to-c1 and prefill-last scopes that rejected prior MoE candidates.
5. Canonical throughput: p512/p1024/p4096 prefill plus the separate 128-step
   decode leg under the same host and residency protocol.

A faster external implementation is evidence about a technique, not a quality
waiver. Distribution differences must be reported separately from throughput;
bit identity is diagnostic for production candidates unless the declared profile
requires exact parity.
