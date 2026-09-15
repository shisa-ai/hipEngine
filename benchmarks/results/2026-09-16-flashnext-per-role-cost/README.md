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

| Probe | GFLOP/s | Share |
| --- | ---: | ---: |
| Datasheet FP32 peak | 29696 | 100% |
| Register-resident FMA, measured | 28521 | 96.0% |
| Dense projections, measured in-model | 2650 | 9.3% of measured peak |

Those dense projections run through a single kernel family,
`gguf_k_prefill_out_coltile_rowbatch_kernel`.

**They run at 2650 GFLOP/s, 9.3% of the measured FP32 peak.**

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

## Why the tuning path is exhausted: the tile space, not the kernel

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

### A wrong ceiling, and the measured one

Reading those counts as issue slots — one instruction per cycle, so 40 of 145
means 27.6% — gives a ceiling of 22.1% of peak. **That model is wrong for RDNA3,
which co-issues integer and FP32 work**, and it understated the ceiling by more
than a factor of two. `scripts/qwen4exp_roofline_probe.py` measures the bound
directly instead of inferring it:

| Probe | GFLOP/s | Share |
| --- | ---: | ---: |
| Datasheet FP32 peak (40 CU × 128 × 2 × 2.9 GHz) | 29696 | 100% |
| Register-resident FMA, measured | **28521** | 96.0% of datasheet |
| 40 FMA + 105 integer per iteration, measured | **16467** | 57.7% of measured FMA peak |
| Dense projections, measured in-model | 2650 | 16.1% of the mix ceiling |

Two things follow. The datasheet peak is real on this part — a register-resident
kernel reaches 96% of it, so percent-of-peak has an honest denominator. And a
kernel holding this exact instruction mix sustains 16467 GFLOP/s, so the mix is
not what limits the projection.

**The dense projection runs at 16.1% of what its own instruction mix can
sustain.** The tile shapes are exhausted, but the kernel is nowhere near its
mix ceiling: the missing 6x is memory latency and scheduling, not tiling, and it
does not require changing the arithmetic.

Stream bandwidth, for the same reason: 211.1 GB/s measured against 256 GB/s
datasheet, 82.5%, in line with the 75-85% a well-written streaming kernel
expects.

### The memory accounting does not close

One number in this artifact is not explained and is flagged rather than
asserted. `attn_qkv` reads 27.85 MB of weights, 10.49 MB of activations and
writes 41.94 MB of F32 output per 1024-row launch — 80.3 MB in 17.32 ms, or
4.6 TB/s, which exceeds both the 211 GB/s measured stream rate and the 32 MB
MALL. The bytes cannot all be crossing DRAM in the measured window.

Either the working set is substantially cache-resident across repetitions in a
way the byte model does not capture, or the effective traffic is smaller than
the buffer sizes suggest. A `rocprofv3` memory-counter capture of the same
launch would settle it, and until then the 16.1% figure should be read as
"16.1% of the mix ceiling on a kernel whose memory behaviour is not yet
characterized" rather than as a pure latency-hiding gap.

## Status

**Rejected: further tile-shape tuning of the dense projection.** The family is
swept, occupancy is maximal, there are no spills. **Not rejected: making the
same arithmetic substantially faster.** The kernel sits at 16.1% of its own
instruction-mix ceiling, and the measured probe says that ceiling is real, so a
memory-latency and scheduling effort has roughly 6x of headroom without touching
the arithmetic. Nothing in this artifact is a retained performance claim.

Not yet measured, in the order that would settle the remaining questions:

1. A `rocprofv3` memory-counter capture of the same `attn_qkv` launch, to close
   the byte accounting above. Until it runs, the size of the latency-hiding gap
   is bounded but not attributed.
2. Where the stalls actually are: an occupancy-and-stall breakdown of the real
   kernel, since the probe kernel that reaches 57.7% has no memory traffic.
3. Real activation ranges from a boundary capture, replacing the deterministic
   per-layer activations the sweep used.
4. The `gr_read` 2539 ms of non-matmul tensor reads, which no projection change
   would touch.
