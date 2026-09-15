# Flash-Next Per-Operation Prefill Cost, and Why Tuning Is Not the Lever

One 4096-token prefill of `code-p4096` at the production default, measured
2026-09-16 on Framework `gfx1151` (machine `55ea6c509d0b49eea8de7094a1023668`,
**AMD Radeon 8060S, Strix Halo APU**), chunk1024, BF16 KV, warm PLE.

The prefill takes **22046 ms**. This artifact says where that time goes, what the
hardware ceiling actually is, and what the first experiment on the largest
identified projection showed.

## Hardware basis

Every rate below is quoted against this machine, not the W7900 that
`docs/ROOFLINE.md` documents. The two differ enough that mixing them changes the
answer by a factor of two.

| Property | gfx1151 (measured here) | gfx1100 W7900 (docs/ROOFLINE.md) |
| --- | ---: | ---: |
| Compute units | 40 | 96 |
| Clock | 2.9 GHz | 2.5 GHz |
| FP32 FMA peak | **29696 GFLOP/s** | 61300 GFLOP/s |
| Memory bandwidth | 256 GB/s LPDDR5X | 864 GB/s GDDR6 |
| MALL | 32 MB | 96 MB |

The dense projections in this prefill are scalar FP32 FMA kernels. Their honest
denominator is the FP32 rate, not the 123 TFLOP/s BF16 matrix-core rate, because
they issue `v_fmac_f32` and not `v_mmac`.

## How the numbers were obtained

Three records, all from the same run and the same model file:

1. A role-marked `rocprofv3` capture (`--profile --role-markers`). Owner entry
   points push a ROCTX range naming the tensor
   (`qwen4exp_role:linear:layers.15.attn_q`), and every dispatch is attributed to
   the innermost enclosing range. **Attribution is 100%:** 9652 kernels,
   22368.1 ms attributed over a 22932.5 ms window, 0 ms unattributed.
2. A launch census (`--launch-census`) recording quant, `K`, `N`, rows and launch
   count behind each owner call. It is reset after warmup, so graph
   instantiation and allocator growth are not attributed to the measured pass.
3. `model-shapes.json`, the GGUF tensor geometry and expert routing config. This
   supplies shapes for operations that do not go through the GGUF quant launch
   path — MoE, QSA, GR, GDN, indexer — which the census alone cannot cover.

The census is not needed for the MoE: its shapes come from the model file and
its row count from `expert_used_count = 10` (512 experts, 10 selected per token,
expert FFN width 640). Every role in the table below therefore has a real shape.

## Cost by operation

`rows` is per launch. For MoE it is `1024 tokens x 10 experts`.

| Operation | ms | % | K | N | rows | n | GFLOP | GFLOP/s | % of FP32 peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `moe:expert_gate` gate+up | 2710 | 12.3 | 2560 | 640 | 10240 | 188 | 6308 | 2328 | 7.8 |
| `linear:attn_qkv` | 2585 | 11.7 | 2560 | 10240 | 1024 | 144 | 7731 | 2991 | 10.1 |
| `linear:ssm_out` | 1960 | 8.9 | 6144 | 2560 | 1024 | 144 | 4639 | 2366 | 8.0 |
| `linear:attn_gate` | 1580 | 7.2 | 2560 | 6144 | 1024 | 144 | 4639 | 2935 | 9.9 |
| `moe:expert_gate` down | 1482 | 6.7 | 640 | 2560 | 10240 | 172 | 5771 | 3893 | 13.1 |
| `qsa_prefill:attn_q` | 1421 | 6.4 | — | — | — | — | — | — | — |
| `gr_read:hc_attn_down` | 1271 | 5.8 | — | — | — | — | — | — | — |
| `gr_read:hc_ffn_down` | 1268 | 5.8 | — | — | — | — | — | — | — |
| `linear:attn_q` | 1249 | 5.7 | 2560 | 12288 | 1024 | 48 | 3092 | 2475 | 8.3 |
| `moe:expert_gate` routing/scatter | 931 | 4.2 | — | — | — | — | — | — | — |
| `gdn:attn_qkv` | 824 | 3.7 | — | — | — | — | — | — | — |
| `moe:expert_gate` repair | 1370 | 6.2 | — | — | — | — | — | — | — |
| `linear:attn_output` | 656 | 3.0 | 6144 | 2560 | 1024 | 48 | 1546 | 2358 | 7.9 |
| `linear:hc_ffn_down` | 550 | 2.5 | 10240 | 320 | 1024 | 192 | 1288 | 2345 | 7.9 |
| `linear:hc_attn_down` | 549 | 2.5 | 10240 | 320 | 1024 | 192 | 1288 | 2345 | 7.9 |
| `linear:shared_down` | 298 | 1.3 | 640 | 2560 | 1024 | 192 | 644 | 2165 | 7.3 |
| `linear:shared_gate` | 269 | 1.2 | 2560 | 640 | 1024 | 192 | 644 | 2395 | 8.1 |
| `linear:index_q` | 228 | 1.0 | — | — | — | — | — | — | — |
| `linear:shared_up` | 187 | 0.8 | 2560 | 640 | 1024 | 192 | 644 | 3442 | 11.6 |
| `moe:expert_gate` down (Q8_0 layers) | 106 | 0.5 | 640 | 2560 | 10240 | 20 | 671 | 6361 | 21.4 |

By operation class and by role family:

| Class | ms | % |
| --- | ---: | ---: |
| matmul | 14368 | 65.2 |
| non-matmul | 6308 | 28.6 |
| risk or repair | 1370 | 6.2 |

| Role family | ms | % |
| --- | ---: | ---: |
| `linear` | 10553 | 47.9 |
| `moe` | 6599 | 29.9 |
| `gr_read` | 2539 | 11.5 |
| `qsa_prefill` | 1421 | 6.4 |
| `gdn` | 824 | 3.7 |
| `prefill_boundary` | 94 | 0.4 |

The MoE bucket decomposes into gate+up 2710 ms, down 1588 ms, repair 1370 ms and
routing/scatter/reduction 931 ms. Repair is 20.8% of the MoE, which is why
"repair is only a small share" is not a safe assumption for this block.

Two rows deserve a caveat. `gr_read:hc_attn_down` and `gr_read:hc_ffn_down` are
non-matmul time in a role whose sibling `linear:` role *is* a matmul: the same
tensor is read twice, and the gather costs 1271 + 1268 = 2539 ms against 549 +
549 = 1098 ms for the matmuls that consume it. Whether that read can be folded
into the projection is an open question, not a measured win.

`risk_or_repair` rows carry milliseconds but no rate: the risk and repair passes
touch only the rows the risk heuristic flagged, so computing a rate from the
full-row shape would overstate the work. Their `gflop` field is labelled as an
upper bound for the same reason.

## The dense projections are one kernel family

Every dense projection above — 10553 ms of `linear:` roles, of which 10070 ms is
matmul — runs through a single kernel family,
`gguf_k_prefill_out_coltile_rowbatch_kernel`. That is the whole dense-projection
cost, in one place, which makes it the right target.

**They run at 2650 GFLOP/s, 8.9% of the FP32 peak.**

## First experiment: the tuning path is exhausted

`attn_qkv` was chosen because it is the largest fully identified projection:
2585 ms, K=2560, N=10240, Q8_0, 1024 rows, 144 launches.

`scripts/qwen4exp_dense_projection_ab.py` ran every registered coltile/rowbatch
instantiation on identical operands — five real GDN layers (0, 9, 22, 34, 46),
row counts 1/8/64/512/1024, weights rotating per layer so the matrix does not
stay cache-resident, compilation outside every timed region.

Result: **the production instantiation is already the fastest of the family.**
The next best is 12–14% slower. The tile-shape design space is exhausted.

The sweep also caught an error in its own setup. The kernel name alone does not
identify the instantiation, because `WAVE_SCALE` is a template parameter and both
instantiations share the kernel name. The trace's template argument list is
`float, float, 8, 8, 4, true`, so production is the *wave-scale* instantiation.
Reading the base name and assuming otherwise made the first run report a 12%
"win" that was really the shipped configuration beating a superseded one.

`wave_scale` itself is a real mechanism and worth recording: it replaces the
per-lane `k / 32` block index with `__builtin_amdgcn_readfirstlane`, making the
block pointer and the FP16 scale load wave-uniform. The inner loop drops from 171
to 145 instructions, and the output is bit-identical (`max_abs_diff = 0.0` across
every layer and row count in the sweep).

## Why the tuning path is exhausted: the instruction mix

`scripts/qwen4exp_dense_projection_loop_mix.py` compiles the kernel and counts
issue slots in the innermost loop.

| | Production instantiation |
| --- | ---: |
| Instructions per k-iteration | 145 |
| FMA-class | 40 (27.6%) |
| Non-FMA | 105 (72.4%) |
| Iterations per thread | 20 |
| VGPR / SGPR / LDS | 72 / 58 / 512 B |
| Private segment (spills) | 0 |
| Register-only occupancy | ≤16 waves/SIMD |

The 105 non-FMA instructions are 32 for dequantization (8 scale loads, 8 int8
loads, 8 int8→float converts, 8 dequant multiplies), 26 for address arithmetic,
15 for `s_waitcnt`, 7 for `s_delay_alu`, and the remainder for control flow.

The consequence is a hard ceiling on the design:

| | GFLOP/s | % of FP32 peak |
| --- | ---: | ---: |
| FP32 peak | 29696 | 100 |
| Issue-limited ceiling of this design | **6554** | **22.1** |
| Measured, dense projections | 2650 | 8.9 |

The kernel reaches **40% of its own issue ceiling**. There are no spills and
occupancy is already at the register-only maximum, so the missing 60% is
memory-latency stall, not an occupancy problem.

This splits the headroom into two parts with different answers:

- **Within the design (about 2.5x):** recoverable by hiding the stalls behind
  the existing instruction stream. Real, but bounded by the 22.1% ceiling.
- **Beyond the ceiling (about 4.5x):** requires reducing the 72.4% of issue
  slots that do not multiply. Tiling differently cannot do it; the mix is fixed
  by dequantizing each weight element in the inner loop.

## Status

**Rejected: further tuning of the dense projection.** The tile family is swept,
occupancy is maximal, there are no spills, and the design ceiling is 22.1% of
peak against 8.9% measured. Nothing in this artifact is a retained performance
claim.

Not yet measured, in the order that would settle the remaining questions:

1. A practical roofline probe on this GPU — achieved FP32 FMA rate and achieved
   bandwidth for a pure stream — to confirm the 29696 GFLOP/s peak is reachable
   at all, and to size the latency-hiding headroom empirically rather than by
   inference from the instruction mix.
2. Real activation ranges from a boundary capture, replacing the deterministic
   per-layer activations this sweep used.
3. The `gr_read` 2539 ms of non-matmul tensor reads, which no projection change
   would touch.
