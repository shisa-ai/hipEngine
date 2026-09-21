# MoE main-kernel 32-row M tile: traffic halves, registers spill

**Date:** 2026-09-22
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop, ROCm therock10-staging-20260828
**Artifact:** `artifact.json` (this directory)
**Script:** `scripts/qwen4exp_moe_main_kernel_cost.py`
**Command:**

```bash
/tmp/run-gpu.sh -u scripts/qwen4exp_moe_main_kernel_cost.py \
    --rows 2560,10240,40960 --distribution balanced,uniform \
    --tile-rows 16,32 --repetitions 15 \
    --output benchmarks/results/2026-09-22-moe-j32-tile/artifact.json
```

## Question

The routed-MoE gate/up and down main kernels read expert weights at a measured
104-126 GB/s regardless of row count. Weight traffic is
`Σ ceil(rows_per_expert / TILE_ROWS) × tensor_bytes`, so the per-expert M tile
height sets how many times each expert's weights are read. A 32-row tile should
halve that whenever an expert holds 16 or fewer rows per tile.

## Result

Both kernels were widened to `TILE_ROWS=32` (templated on the existing 16-row
owner, registered as
`selected_wmma_iu8_risk_j32_prefill_bf16_bf16_out` /
`selected_wmma_iu8_risk_j32_prefill_bf16_bf16_out`). Outputs are bit-identical
to the 16-row owner and the risk queues match exactly, verified on gfx1151.

The traffic prediction is confirmed and the speed prediction is **wrong by 8x**:

| rows | dist | tiles 16→32 | gate ms 16→32 | down ms 16→32 | gate GB/s 16→32 |
| --- | --- | --- | --- | --- | --- |
| 2560 | balanced | 512 → 512 | 8.72 → **70.45** | 5.63 → **50.67** | 108.2 → 13.4 |
| 10240 | balanced | 857 → 512 | 14.80 → **71.94** | 8.93 → **51.71** | 106.7 → 13.1 |
| 40960 | balanced | 2801 → 1536 | 45.27 → **200.94** | 27.33 → **148.21** | 114.1 → 14.1 |
| 2560 | uniform | 512 → 512 | 8.76 → **70.48** | 5.88 → **50.69** | 107.8 → 13.4 |
| 10240 | uniform | 1024 → 512 | 17.22 → **71.73** | 10.21 → **51.89** | 109.6 → 13.2 |
| 40960 | uniform | 2560 → 1536 | 41.70 → **200.14** | 25.26 → **147.89** | 113.2 → 14.2 |

Tile counts fall exactly as the traffic model says (857 → 512 at 20 rows/expert,
2801 → 1536 at 80 rows/expert, i.e. 1.67x and 1.82x fewer weight reads), but the
kernels run 4.8-8.1x slower and achieved bandwidth collapses to ~13 GB/s.

## Mechanism (measured, not inferred)

`rocprofv3 --kernel-trace` on the same geometry:

| kernel | VGPR | Scratch | LDS/block | threads/block |
| --- | --- | --- | --- | --- |
| `..._risk_prefill_kernel<16>` | 192 | **0** | 16 KB | 128 |
| `..._risk_prefill_kernel<32>` | **256** (arch max) | **492 B** | 32 KB | 128 |

Doubling the tile height doubles the per-lane row state: the kernel holds
`sum`, `kahan_err` and `abs_terms` per (row, column) pair, i.e. 3 floats per
row, and each lane owns `TILE_ROWS / 8 × 8` rows. At 16 rows that is 48
registers; at 32 rows it is 96, on top of 32 accumulator registers
(`acc[4 planes][8]`). `__launch_bounds__(128, 2)` lets the compiler use the
whole CU register file (2 blocks x 128 threads x 256 VGPRs = 65536) and it still
spills 492 bytes per thread, so the true demand is above the 256-VGPR
architectural ceiling. Every spill and reload sits in the K loop, so the kernel
becomes memory-bound on its own spills instead of on weights.

The arithmetic itself is unaffected: the 32-row tile reads the same staged
activations and issues the same `wmma_i32_16x16x16_iu8_w32` sequence per row
group, so outputs and risk queues are bit-identical.

## Conclusion

The per-expert padding amplification is real and the tile count is the right
metric, but the fix cannot be "the same block geometry with twice the rows" on
this kernel: the per-row fp32 risk state does not fit in registers. A viable
32-row tile needs the register geometry to stay at 16 rows per lane, for
example an 8-warp block (256 threads) covering 32 rows x 128 columns as two row
groups x four column groups, so the two row groups share one weight read
through L2 while each warp keeps the current 16-row register footprint. That is
a redesign, not a parameter change, and it is recorded in `docs/REFACTOR.md`.

Until then the 16-row tile remains the production geometry. The 32-row variants
are registered and tested (bit-identical, both kernels) but are not wired into
any dispatch default.

## Correctness evidence

- `tests/test_gpu_qwen4exp_q4_iu8_j32_tile.py` — 18 passed on gfx1151: bit
  identity against the 16-row owner for gate/up and down across six expert-count
  shapes and two risk multipliers, identical risk queues, registry binding, and
  a chain test that runs the exact repair after the 32-row tile.
- The first run of this packet exposed a real bug in the widened down kernel:
  the `lds_row_atrisk` clear guard still read `tid < 16`, leaving entries 16-31
  as uninitialized shared memory. Outputs stayed bit-identical (the flag only
  gates the risk queue) but the queue grew from 106 to 14522 entries. Fixed by
  parameterizing the guard to `tid < TILE_ROWS`; this is exactly the failure
  mode the RED test now covers.
