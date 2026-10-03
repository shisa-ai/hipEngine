# gfx1151 selected Q4_K gate+up: 16-column -> tile8 owner

Date 2026-10-03. Model `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, quant `gguf_ud_q4_k_m`,
host gfx1151 Strix Halo Radeon 8060S, shipping AR route.

`decode_ffn_moe_selected` was the largest remaining bandwidth-efficiency gap in
the decode step: 597.8 MiB/token at 170.4 GB/s, 72% of the 237 GB/s ceiling,
3.678 ms of a 17.81 ms step. Its Q4_K gate+up leaf ran at VGPR 200 with only 32
resident workgroups, so it was register/occupancy-limited rather than
bandwidth-limited. A new owner covering 8 columns per workgroup instead of 16
halves register pressure, doubles resident workgroups, and is byte-identical.

## Mechanism

rocprofv3 at the production 2048x512x8 shape (`rocprof-mechanism.json`):

| kernel | VGPR | LDS | block | grid.x blocks | dispatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q4_k_t16_selected_dual_silu_direct_gemv_kernel` (16-col) | 200 | 1024 | 128 | 32 | 3 |
| **`q4_k_t16_selected_dual_silu_tile8_gemv_kernel` (new)** | **88** | **512** | 128 | **64** | 200 |

Note that a `Grid_Size_X` value in a rocprofv3 kernel-trace CSV is total
threads, not a block count; divide by `Workgroup_Size_X`.

## Byte-identity

Both owners launch `dim3 block(128)`. The 16-column owner is declared
`__launch_bounds__(256, 1)` but its host wrapper passes 128, and that is what
makes this a work-partitioning change rather than an arithmetic one:

| | 16-column owner | tile8 owner |
| --- | --- | --- |
| K ownership | `k_in_block = tid; k += blockDim.x` | `k_in_block = tid; k += kThreads` |
| K-block loop | `for blk < blocks_per_row` | same |
| wave reduce | `GGUF_T16_WAVE_REDUCE` per column | same |
| cross-wave sum | `sub_groups = blockDim.x >> 5` = 4, ordered | `kThreads >> 5` = 4, same order |
| BF16 round-trip, fused SiLU | present | same |

`col_base = (out_group & 1) * kTileCols` maps the two half-tiles back onto the
16-column layout. The nibble and coefficient loads become one aligned `uint32`
per buffer per column pair, which changes how values are fetched, not which
values are combined or in what order.

Verified three ways:

- `tests/test_gpu_gguf_t16_selected_gemv_decode.py::test_qwen_c1_selected_q4_tile8_matches_col16_production_bits`
  runs both owners at the production shape and asserts
  `np.testing.assert_array_equal`. HIP-guarded, so no-ROCm runners skip it.
- A 120-config sweep over heavy-tail inputs with scales 0.5-120: **491,520
  outputs, 0 mismatches, worst ULP 0**.
- A real-model check: 512-token prefill plus 16 greedy steps produced
  **byte-identical final logits** (993,280 bytes) and the same token ids.

## Real decode step

Same-session A/B toggling the real flag `HIPENGINE_GGUF_SELECTED_Q4_GATE_UP_TILE8`,
so both arms run one tree and one session shape. One persistent session, warmup
1, measured 1, prompt 512, 32 decode tokens, GPU stage recorder
(`real-step-ab.json`):

| measurement | baseline | patched | delta |
| --- | ---: | ---: | ---: |
| `decode_ffn_moe_selected` | 125.727 ms/32tok | 117.140 | **-6.83%** |
| decode | 53.4247 tok/s | 54.3652 | **+1.760%** |
| prefill | 1410.50 tok/s | 1410.47 | -0.002% (flat) |

The stage saving is 268.4 us/token against an 18.7 ms step, a +1.434%
prediction, and +1.760% was measured, so the transfer to the step is complete.
An independent delegated run read 125.715 -> 117.215 ms/32tok (-6.76%) and
53.598 -> 54.327 tok/s (+1.36%), so the stage delta reproduced across two
sessions.

## Tests

204 passed, exit 0: `tests/test_gpu_gguf_t16_selected_gemv_decode.py` plus
`tests/test_unit_qwen35_gguf_compact_moe_gemv_routing.py`. A wider delegated run
reported 246 passed across those two plus four more routing and unit files.
Prefill is unchanged, so this change cannot regress it.

## Rollback

`HIPENGINE_GGUF_SELECTED_Q4_GATE_UP_TILE8=0` restores the 16-column owner, which
stays registered as the strict fallback. `docs/REFACTOR.md` carries the removal
condition, including a note that the selector's `_env_flag` read is not memoized
and should be hoisted when the selector is dropped.

## Limits

One host, one model, one quant, one profile, one prompt family. Decode tok/s is
a single measured run per arm; the -6.83% stage delta is the load-bearing number
and it reproduced across two sessions. The change is byte-identical with
identical reduction order, so it is not gfx1151-specific by mechanism, but the
magnitude is measured on gfx1151 only and is **unverified on gfx1100/W7900**.
