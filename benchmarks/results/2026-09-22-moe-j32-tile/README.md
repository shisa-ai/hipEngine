# MoE main-kernel 32-row M tile: the traffic model is right, the sharing is not

**Date:** 2026-09-22
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop, ROCm therock10-staging-20260828
**Artifacts:** `artifact.json` (128-thread variant), `artifact-8warp.json` (8-warp
variant), `dram-counters.json`
**Script:** `scripts/qwen4exp_moe_main_kernel_cost.py`
**Command:**

```bash
/tmp/run-gpu.sh -u scripts/qwen4exp_moe_main_kernel_cost.py \
    --rows 2560,10240,40960 --distribution balanced,uniform \
    --tile-rows 16,32 --repetitions 15 \
    --output benchmarks/results/2026-09-22-moe-j32-tile/artifact-8warp.json
```

## Question

The routed-MoE gate/up and down main kernels read expert weights at a measured
104-126 GB/s regardless of row count. Weight traffic is
`Σ ceil(rows_per_expert / TILE_ROWS) × tensor_bytes`, so the per-expert M tile
height sets how many times each expert's weights are read. A 32-row tile should
halve that whenever an expert holds 16 or fewer rows per tile, and the two
row groups of a tile read the *same* weight bytes, so the second read should be
served by cache.

## Result: the tile count halves, the traffic does not

Both kernels were widened to `TILE_ROWS=32`, registered as
`selected_wmma_iu8_risk_j32_prefill_bf16_bf16_out`. Outputs are bit-identical to
the 16-row owners and the risk queues match exactly, verified on gfx1151.

The predicted tile-count reduction is exact. The predicted time reduction does
not appear; the 32-row tile is slower in every configuration:

| config | tiles 16→32 | gate ms 16→32 | down ms 16→32 | gate GB/s 16→32 |
| --- | --- | --- | --- | --- |
| 2560 balanced | 512 → 512 | 8.74 → **19.45** | 5.66 → **12.16** | 108.0 → 48.5 |
| 10240 balanced | 857 → 512 | 14.84 → **20.25** | 9.04 → **12.58** | 106.4 → 46.6 |
| 40960 balanced | 2801 → 1536 | 45.41 → **57.39** | 27.78 → **36.58** | 113.7 → 49.3 |
| 2560 uniform | 512 → 512 | 8.79 → **19.47** | 5.79 → **12.16** | 107.3 → 48.5 |
| 10240 uniform | 1024 → 512 | 17.24 → **20.29** | 10.33 → **12.65** | 109.5 → 46.5 |
| 40960 uniform | 2560 → 1536 | 41.74 → **57.05** | 25.55 → **36.55** | 113.0 → 49.6 |

`gate GB/s` is the *modeled* rate, i.e. `tiles × tensor_bytes / time`. It halves
with the tile height because the model assumes the second row group's read is
free. It is not.

## Mechanism, part 1: the naive widening spills (fixed)

The first implementation kept the 128-thread block geometry and doubled the
tile, which doubles the per-lane fp32 risk state (`sum`, `kahan_err`,
`abs_terms` per (row, column)) from 48 to 96 registers on top of 32 accumulator
registers. `rocprofv3 --kernel-trace`:

| variant | threads | VGPR | Scratch | LDS/block |
| --- | --- | --- | --- | --- |
| 16-row (owner) | 128 | 192 | 0 | 16 KB |
| 32-row, 128 threads | 128 | 256 (arch max) | 492 B | 32 KB |
| 32-row, 8 warps | 256 | 208 | 0 | 32 KB |

`__launch_bounds__(128, 2)` already permits the whole CU register file
(2 × 128 × 256 = 65536) and the compiler still spilled, so demand exceeded the
256-VGPR ceiling. That version ran 4.8-8.1x slower (`artifact.json`).

The fix keeps 16 rows per lane and doubles the warps instead: an 8-warp block
covers 32 rows × 128 columns as two row groups × four column groups, so
`warp & 3` picks the column group and `warp >> 2` the row group. Per-lane state
stays at 48 registers and the spill disappears (VGPR 208, scratch 0), which
takes the loss from 8x to 1.3-2.2x. It is still a loss.

## Mechanism, part 2: the shared read misses cache (measured)

`rocprofv3 --pmc GCEA_RDRAM_SIZE_REQ_sum` on the same geometry
(`dram-counters.json`), per dispatch at 40960 rows, balanced:

| kernel | counter units/dispatch | modeled weight bytes | real/modeled |
| --- | --- | --- | --- |
| down 16-row | 2.9007e7 | 3.442 GB | 1.08 (at 128 B/unit) |
| down 8-warp 32-row | 2.7179e7 | 1.887 GB | 1.84 |

The unit size cancels in the ratio, which is the robust reading: **real DRAM
read traffic falls 6.3% (0.937x), not the modeled 45% (0.55x).** The j16
calibration (118.7 B/unit against a 128-byte sector) confirms the counter is
measuring real read traffic rather than something else.

So the two row groups do read the same weight bytes, but not within the cache's
reuse window: warps 0-3 and 4-7 sit on different SIMD units, iterate `sub` and
`chunk` independently, and each streams the expert's full slice (1.2 MB for the
down tensor). With 8 warps × 1.2 MB in flight per block and 16 CUs, the working
set is far beyond L2, so both row groups fetch from DRAM. The 32-row tile then
pays the same traffic as the 16-row tile plus the cost of one block per CU
instead of two, taller padding (6.4x vs 3.2x at 2560 rows), and a bigger staging
pass.

## Conclusion

The tile-count traffic model is **necessary but not sufficient**: it counts
*distinct* weight reads, and a taller tile makes the reads shared rather than
distinct. Sharing only pays if it is explicit. A 32-row tile that actually
banked the traffic reduction would have to stage the chunk's quantized weights
in LDS once per block and have all 8 warps read them from there, instead of
each lane gathering its own column's blocks from global memory. That is a
different weight path, not a tile-height parameter.

The 16-row tile remains the production geometry. The 32-row variants are
registered and tested but wired into no dispatch default; `docs/REFACTOR.md`
records the LDS-staging follow-up and the removal trigger.

## Correctness evidence

- `tests/test_gpu_qwen4exp_q4_iu8_j32_tile.py` — 18 passed on gfx1151: bit
  identity against the 16-row owner for gate/up and down across six expert-count
  shapes and two risk multipliers, identical risk queues, registry binding, and
  a chain test that runs the sparse exact repair after a 32-row tile.
- `tests/test_gpu_qwen4exp_q51_iu8_exact.py` +
  `tests/test_gpu_qwen4exp_q4_iu8_exact.py` — 40 passed: the production 16-row
  paths still match their parents on adversarial and nonfinite inputs.
- The first run of the packet exposed a real bug in the widened down kernel: the
  `lds_row_atrisk` clear guard still read `tid < 16`, leaving entries 16-31 as
  uninitialized shared memory. Outputs stayed bit-identical (the flag only gates
  the risk queue) but the queue grew from 106 to 14522 entries. Fixed by
  parameterizing the guard to `tid < TILE_ROWS`.
