# QSA prefill: the 9.01x is recoverable cost, not a bucket artifact

**Date:** 2026-09-22
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL / BF16 KV
**Case:** `code-p4096`, engine prefill chunk 1024
**Artifact:** `artifact.json`
**New engine runs:** none. Derived from the existing shipped-default attribution
capture and from the comparator's own source and trace.

## Question

`qsa_attention` is the worst ratio in the gap table: 1402.5 ms against the
comparator's 155.7 ms, 9.01x, worth 1246.8 ms of the 12.97 s total gap. Is that
recoverable cost, or are the two engines bucketing different work?

## The selected-position work is identical

Both engines select the same positions, by the same formula, with the same
budget:

- **hipEngine:** `indexer.top_k` = 2048 and `compress_ratios` = 4 at the 12 QSA
  layers, so `qsa_dense_equivalent_max_tokens` = 2048 + 4 - 1 = **2051**
  (`hipengine/loading/qwen4exp_gguf.py:107-109`).
- **Comparator:** `llama.cpp-hip/src/models/qwen4exp.cpp:664-665`, in
  `build_qsa_top_k` — *"the reference returns indexer_top_k + compress_ratio - 1:
  whole blocks plus the tail"*, `width = min(n_kv, indexer_top_k + r - 1)` =
  **2051**.

Same formula, same budget, same selected positions. There is no work difference
to explain the ratio.

## The operation boundary does not explain it either

The engine's whole QSA family, measured kernel by kernel in the attribution
capture (48 launches = 12 layers x 4 chunks of 1024):

| kernel | ms | launches | grid | block |
| --- | ---: | ---: | --- | ---: |
| `qsa_sparse_attention_h256_wave_rows_f32_kernel<true, 4>` | 910.89 | 48 | `(192, 1021, 1)` | 32 |
| `qsa_score_f32_kernel` | 212.01 | 48 | `(25092096, 1, 1)` | 32 |
| `qsa_split_norm_rope_rows_f32_kernel<false>` | 46.43 | 96 | `(6656, 1024, 1)` | 256 |
| `qsa_topk_expand_f32_i64_kernel` | 34.47 | 48 | `(1045504, 1, 1)` | 1024 |
| `qsa_gate_context_f32_kernel` | 28.40 | 96 | `(6291456, 1, 1)` | 256 |
| `qsa_norm_rope_rows_f32_kernel<false>` | 3.74 | 96 | `(1024, 1024, 1)` | 256 |
| `qsa_scatter_index_keys_f32_kernel` | 0.64 | 96 | `(131072, 1, 1)` | 256 |
| `qsa_pool_norm_rope_f32_kernel` | 0.63 | 48 | `(65536, 1, 1)` | 256 |
| **total** | **1237.2** | | | |

The comparator's `qsa_attention` bucket holds its attention plus `rope_multi`
and `flash_attn_mask_to_KV_max`. The engine's `qsa_score` (212.0 ms) is arguably
indexer work, which the comparator files under `indexer` — and there the two
engines are already comparable (17.6 ms against 15.0 ms, 1.17x). Moving it is
the most generous re-bucketing available and it lands at **6.58x**, not parity.

## The residual gap is kernel technology and reuse

| | engine | comparator |
| --- | --- | --- |
| kernel | `qsa_sparse_attention_h256_wave_rows_f32<true,4>` | `flash_attn_ext_f16<256,256,16,4,...>` |
| form | 32-lane wave, scalar `__fmul_rn`/`fmaf` | WMMA flash attention |
| LDS | **0 bytes** | staged tiles |
| VGPR | 112 | — |
| effective rate | **2.04 TFLOP/s** | **11.93 TFLOP/s** |

The 1858 GFLOP of attention work in these 12 layers (6.30M selected row-key
pairs per layer, computed from the 2051 cap) takes 910.9 ms in the engine's
kernel and 155.7 ms in the comparator's. **The comparator's rate is above this
host's ~9.5 TFLOP/s fp32 peak**, so it can only be reached with BF16 tensor
cores; the engine's scalar path runs at 21% of that peak.

The engine's kernel is also not DRAM-bound: one chunk's K/V is 8.4 MB against a
32 MB MALL, so the re-reads are cache traffic. With `HEADS=4` it amortizes each
K/V load across 4 query heads while 24 query heads share each of the 2 KV heads
— up to 6x more reuse is available, and there is no LDS staging to capture it.

## Decision

**The boundary-artifact hypothesis is rejected.** The 9.01x is real recoverable
cost, and the two levers are ranked in the artifact:

1. **K/V reuse and LDS staging** — raise the amortization from 4 query heads
   toward 24 by staging the selected K/V tile in shared memory and tiling rows.
   This is bit-exact per row, so it needs no numerical gate. Raising `HEADS`
   alone is the trap: VGPR 112 at `HEADS=4` becomes roughly 184 at `HEADS=8`,
   and the routed-MoE 32-row tile precedent shows a spilled variant going
   4.8-8.1x *slower*.
2. **Tensor cores for QK^T and P·V** — where the comparator's advantage comes
   from, and it changes arithmetic, so it needs the production numerical gate
   plus a registered strict fallback.
3. **An occupancy and tiling sweep** on the existing wave-row path, measured as a
   kernel microbenchmark. **Done, and the axis is exhausted** - see below.

The engine-level A/B that would retain a prefill-level number is blocked: an idle
`hipengine serve` (pid 4175913) holds 98.6 GB of GTT against a 95.7 GB
requirement. Kernel microbenchmarks are unaffected, and for a bit-exact
restructuring cycle wall plus bit-identity is the applicable evidence.

**Acceptance status:** the boundary question is settled (recoverable cost), and
the cheap parameter lever is measured to be exhausted. The task's
retained-speedup branch is *not* met - it needs the structural work: LDS staging
with row tiling (bit-exact per row, no numerical gate) or tensor cores (changes
arithmetic, needs the production gate).

## The head-tile axis is exhausted (measured)

The three registered wave-row variants differ *only* in how many query heads
share one selected K/V load, and they are bit-identical (verified: all three
outputs compare equal as uint32). Timing them on the real geometry prices the
K/V amortization axis directly:

| variant | heads | blocks/row | ms at 1021 rows | vs shipped | effective K/V GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `heads_1_page256` | 1 | 24 | 55.052 | 0.351x | 934 |
| `heads_2_head_pair` | 2 | 12 | 20.792 | 0.929x | 1237 |
| `heads_4_head_quad` | 4 | 6 | **19.316** | **1.000x** | 666 |

The harness reproduces the shipped kernel's traced per-launch cost (19.32 ms
against 18.98 ms, 1.8% off), so the comparison is trustworthy. **The shipped
`quad` setting is already the fastest of the three**, so the head-tile parameter
axis offers nothing.

The kernel is also not simply K/V-traffic-bound at `HEADS=4`: `HEADS=1` carries
4x the K/V traffic and is 2.85x slower, but `HEADS=2` carries 2x the traffic and
is only 7% slower, and the effective K/V rate at `HEADS=4` (666 GB/s) is far
above DRAM - the re-reads are MALL/L2 traffic. The residual cost is the scalar
FMA issue rate and occupancy.

One trap is worth recording: at a 64-row launch the ranking *inverts* and
`HEADS=2` looks 1.32x faster than `HEADS=4`. It is not; that grid is simply too
small to be representative.

## Open observation

The traced launch of `qsa_sparse_attention_h256_wave_rows_f32_kernel<true,4>` has
grid (192, 1021, 1) with block 32, but the kernel indexes
`head = blockIdx.x * HEADS` and `out_base = (row * query_heads + head) * 256`,
which implies a launch-time `query_heads` of 768 while the model metadata carries
`attention.head_count = 24` (and the QSA layer's `attn_q` emits 12288 = 6144
query + 6144 gate). This does not affect the work comparison, which rests on the
selection cap, but the launch geometry must be understood before any grid or
tiling change.

## Limits

Kernel time from a profiled capture, not wall time; neither side is a throughput
claim. The comparator is the same case on the same host but not the same
arithmetic (BF16 against F16). The FLOP and traffic figures are computed from the
selection rule and geometry, not measured with hardware counters. No performance
claim and no default change results from this packet.
