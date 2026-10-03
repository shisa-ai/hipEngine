# DPP wave reduction in the Q8 prefill leaf

**Result: replacing the five-level `__shfl_down` wave reduction with the VALU-only
DPP sequence is bit-identical and 1.5-1.7% faster on the production shape, and the
first measurement of the launch order itself shows the grid axes are free but
swapping them loses 2.1-2.9%.**

Date 2026-10-04. Host gfx1151 (Strix Halo, Radeon 8060S, machine
`55ea6c509d0b49eea8de7094a1023668`). Model `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`,
UD-Q4_K_XL, real Q8_0 `attn_qkv.weight` tensors of five layers (0, 9, 22, 34, 46),
`2560 -> 10240`, 1024 rows, 20 repetitions, compilation outside every timed
region. Harness `scripts/qwen4exp_dense_projection_ab.py`.

## Why this was tried: the disassembly, not the model

The kernel is 6.2x off what its own instruction mix sustains (3327 GFLOP/s measured
against the 16467 GFLOP/s mix probe), and every tile in the space is already slower
than the production one. Disassembling the production leaf
(`coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out`, gfx1151, `llvm-objdump
--mcpu=gfx1151`) gives 378 static instructions, of which the largest single class
is not a load at all:

| opcode class | count | what it is |
| --- | ---: | --- |
| `ds_bpermute_b32` | 128 | the wave reduction's `__shfl_down` levels, one LDS access each |
| `s_waitcnt` | 113 | mostly `lgkmcnt` - waiting on those LDS accesses |
| `v_add_f32_e32` + `v_dual_add_f32` | 130 | the reduction's additions |
| `ds_swizzle_b32` | 32 | the same tree, the offset-1 level |
| `v_dual_fmac_f32` + `v_fma_mix_f32` | 55 | the actual multiply-accumulate |

The reduction tree is `4 rows x 8 cols x 5 levels = 160` shuffle ops per thread for
32 accumulators, and it is roughly 40% of the static instruction stream.

## The change

`h7c_reduce_wave_accumulators_dpp` already exists in this file for the q6_k prefill
kernels. It folds the same pairs with one `v_permlanex16_b32` and four VALU
data-parallel ops:

* `v_permlanex16` pairs lane i with i+16 - the offset-16 level;
* `v_mov_dpp row_shr:8`, `row_shr:4`, `row_shr:2` pair with i-8, i-4, i-2, which is
  what `__shfl_down` at 8, 4, 2 pairs with, because a lane's own value plus a
  partner's is the same sum whichever direction the partner is named from;
* `v_add_f32_dpp row_shl:1` pairs lane 0 with lane 1 - the offset-1 level.

**The tree is the same tree**, so this is an implementation swap and not an
arithmetic change. It is exposed as `DPP_REDUCTION` on the same kernel body and as
`coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out`.

## Measured

Same-host A/B on the production shape, 20 repetitions per variant per layer:

| layer | pipelined (shuffle tree) | DPP | delta |
| ---: | ---: | ---: | ---: |
| 0 | 16.1395 ms | 15.8936 ms | −1.52% |
| 9 | 16.1421 ms | 15.8694 ms | −1.69% |
| 22 | 16.1266 ms | 15.8689 ms | −1.60% |
| 34 | 16.1177 ms | 15.8438 ms | −1.70% |
| 46 | 16.1191 ms | 15.8465 ms | −1.69% |
| **mean** | **16.1290 ms** | **15.8644 ms** | **−1.64%** |

`bit_identical_to_production: true` with `max_abs_diff_vs_production` 0.0 on all
five layers, and the pipelined parent reproduces its recorded 16.1369 ms mean.

Static census of the compiled profile (`scripts/qwen4exp_repair_register_census.py`):

| instantiation | VGPR | SGPR | static instructions | waves/CU |
| --- | ---: | ---: | ---: | ---: |
| pipelined (parent) | 76 | 48 | 378 | 25 |
| pipelined + DPP | 76 | 48 | **309** | 25 |
| pipelined + row-first grid | 76 | 48 | 378 | 25 |
| unscaled pipelined | 83 | 36 | 382 | 23 |

Identical registers and identical occupancy, 69 fewer instructions: the win is the
removed LDS round trips, not a resource change. It is small because the tree was
mostly overlapped - which is the same finding the 6.2x issue gap gives from the
other side, and it is why the epilogue was worth one attempt and not a campaign.

## The launch order, measured for the first time

`ROW_FIRST` swaps which grid axis carries the column tile. The default order
(`blockIdx.x` = column tile) sweeps all 1280 weight slices of one row batch before
any is needed again, so a re-read is separated by the whole 27.85 MB weight tensor
plus the 10.49 MB activation against a 32 MB MALL; `ROW_FIRST` keeps one 21.8 KB
weight slice resident across the row sweep instead. It is free - identical VGPR,
identical instruction count, bit-identical output - and it is **2.1-2.9% slower**:

| layer | default order | row-first | delta |
| ---: | ---: | ---: | ---: |
| 0 | 16.1348 ms | 16.5252 ms | +2.42% |
| 9 | 16.1176 ms | 16.4779 ms | +2.24% |
| 22 | 16.1219 ms | 16.5836 ms | +2.86% |
| 34 | 16.1129 ms | 16.4500 ms | +2.09% |
| 46 | 16.1222 ms | 16.4538 ms | +2.06% |

`bit_identical_to_production: true` on all five layers.

**This is the third independent refutation of the traffic reading of this kernel
family.** If the weight re-read were the binding cost, keeping the slice resident
could not lose; the streaming column-major order is better because it is what the
hardware prefetches. Together with the GR up result (`2026-10-04`, halving a
modelled 90.3% re-read costs 16%) and the dead-store result (`HC-3`, removing 4.2%
of modelled traffic costs 2%), the consistent reading is that these kernels are
latency-bound and their cost is instruction structure and stall exposure, not bytes.
The 6.2x gap is therefore not closed by any locality change, and the honest next
target is the operand representation - see the note on `dense_wide256` below.

## Routing and exactness through the engine

Promoted as the preferred leaf behind `HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION`
(default on, `=0` restores the pipelined leaf). The rewrite is
`is_registered`-guarded, so the chain degrades DPP -> pipelined -> wave-scale parent
-> unscaled exact parent, and the flag is in the dispatch memoization key so a flip
inside one process cannot replay the other arm's leaf.

`scripts/qwen4exp_row4_state_gate.py --route-package q8-dpp-reduction --case-id
code-p4096 --full-kv`, three arms (`enabled=0`, `enabled=1`, `enabled=0`):
`status: passed`, `exact: true`, and `state_sha256`, `layout_sha256`,
`prefill_logits_sha256`, `step_logits_sha256` and `full_kv_sha256` are identical
across all three arms. **Engagement is recorded, not assumed:** the DPP leaf's
call count is 0 in both off arms and **1072** in the on arm, all in prefill and none
in decode - the same 1072 the pipeline promotion recorded, so the new leaf is
reached for every dense Q8 projection in the case.

`tests/test_gpu_qwen4exp_q8_dpp_reduction.py`, 25 tests with the existing pipeline
file: bit-exactness against the pipelined leaf and against the wave-scale
grandparent on the raw `uint32` bits at eight shapes (including `1023 x 2560 x
10240`), the row-first grid order at two shapes, registry resolution, the dispatch
rewrite chain and its degradation when the DPP key is absent, and the CPU-reference
outer gate on the narrow shapes.

## Limits

Per-kernel microbenchmark, not a full-engine rate: the prefill wall effect is
**not** measured here. On the recorded role attribution (dense projections 10070 ms
of a 22046 ms `code-p4096` prefill) a 1.64% leaf gain is about 165 ms, or 0.75% of
the prefill, and that is arithmetic on a different capture rather than a
measurement. One host, one arch, one quant, one tensor suffix, five layers, one
shape; the state gate's own `step_seconds` are single diagnostic steps and are not
used as a rate.

## Reproduce

```bash
python3 scripts/qwen4exp_dense_projection_ab.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --rows 1024 --repetitions 20 \
  --variants coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out,\
coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out,\
coltile8_rowbatch4_wave_scale_pipelined_rowfirst_f32_f32_out \
  --output /tmp/dpp-ab.json

python3 scripts/qwen4exp_row4_state_gate.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --route-package q8-dpp-reduction --case-id code-p4096 --full-kv \
  --output /tmp/dpp-gate.json
```

## What this does not do

The gap this leaf sits inside is 11.7x on the operation (17.361 ms against the
comparator's 1.492 ms in the cross-engine replay) and 4.6-7.0x on Flash-Next prefill
against the fastest same-file engine. A 1.64% leaf gain does not touch either. The
measured route to the former is already in the tree: `dense_wide256_f32_f32_out`
runs the same operation in **2.570 ms against 17.319 ms, 6.74x**, on identical
resources and launch geometry, and is registered but not a default because f16
operands put it at 2.83e-03 against the exact float64 reference and **no production
gate has been run**. That gate is the next action, not another pass over this leaf.
