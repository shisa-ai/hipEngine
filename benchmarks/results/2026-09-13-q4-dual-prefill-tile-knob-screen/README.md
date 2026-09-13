# Q4 dense WMMA prefill: tile-geometry and occupancy knobs

This screen answers one question before any new kernel variant is written for the
`gfx1151` Q4_K_M dense WMMA prefill owner: which tile-geometry knob can still
move its wall time? The owner is the fused gate/up dual-SiLU kernel
`gguf_q4_t16_dense_dual_wmma_prefill_silu_bf16_kernel<false, false, 4, 4>`, which
holds 33.0-36.5% of prefill kernel time at 512/1024/4096 prompt tokens
(`benchmarks/results/2026-09-13-qwen38-gfx1151-prefill-kernel-profile`).

## Result

**Both tile axes lose when they shrink, and occupancy cannot be moved by tile
geometry alone.** A candidate variant therefore has to raise amortization, not
occupancy.

| Knob | Current value | Effect of shrinking, measured |
| --- | --- | --- |
| Rows per block (tokens) | 256 | 0.88x at 128 rows, 0.47x at 64, 0.44x at 48, 0.40x at 32 |
| Columns per block (weights) | 32 | 1.27-1.39x slower at 32 columns and 1.7x at 16 columns versus 48 columns, measured on the single-matrix sibling at the same shape |
| Waves per block | 4 | rows fall with the wave count while the 128-thread launch stays fixed, so idle wave capacity is wasted: 48 rows (3 waves) is 0.44x and 32 rows (2 waves) 0.40x |
| LDS footprint | 32 KiB | 16 KiB halves LDS use but not resident waves: the compiler still allocates 217 VGPR, and 217 VGPR allows 2 waves per SIMD |

## What was measured

All timings are medians of event-timed bursts on the Radeon 8060S (`gfx1151`,
40 CUs), one warmup plus three measured repetitions per cell. Every geometry arm
in this kernel family shares the K16 WMMA association and BF16 store, so outputs
must be bit-identical; the row-tile screen confirms that in all 20 cells and the
column-tile screen in all 12 non-reference cells.

### Row tile at prefill rows

`scripts/qwen38_packet4_q4_dual_silu_row_leaf.py` on the real
`Qwen3.8-27B-Q4_K_M` `blk.0.ffn_gate`/`blk.0.ffn_up` tiles, shape 5120x17408.

| Rows | 256-row parent | 128 rows | 64 rows | 48 rows | 32 rows |
| --- | --- | --- | --- | --- | --- |
| 256 | 3.270 ms | 3.895 | 7.121 | 8.280 | 8.277 |
| 512 | 6.629 ms | 7.687 | 14.003 | 15.178 | 16.358 |
| 1024 | 13.129 ms | 15.037 | 27.819 | 30.360 | 32.526 |
| 2048 | 26.023 ms | 29.868 | 55.562 | 59.229 | 64.884 |
| 4096 | 52.245 ms | 59.456 | 110.963 | 118.227 | 129.304 |

Per-block time shows why: at 4096 rows one block costs 6.00 us at 256 rows but
3.42 us at 128 rows, 3.19 us at 64 rows and 1.86 us at 32 rows. Work that does
not depend on rows per block — the weight decode (2 matrices x 32 columns x 256
weights per K block) and the operand loads — is then spread over fewer rows, so
the per-row cost rises as the tile shrinks.

### Column tile at the same shape

`scripts/gguf_q4_t16_dense_prefill_owner_microbench.py` on synthetic Q4_K tiles,
single-matrix owners, shape 5120x17408. The 16-column owners are 32-thread
single-wave blocks, so they confound the column tile with block size; the 32- and
48-column owners are 4- or 8-wave blocks.

| Owner | Columns | Rows/block | Threads | 512 rows (ms) | 4096 rows (ms) | LDS | VGPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `shared_b` | 48 | 256 | 128 | 3.878 | 30.819 | 24 KiB | 256 |
| `shared_b3w8r3` | 48 | 384 | 256 | 5.484 | 35.286 | 24 KiB | 224 |
| `shared_b2w2` | 32 | 128 | 64 | 4.648 | 39.109 | 16 KiB | 224 |
| `shared_b2w4` | 32 | 256 | 128 | 4.551 | 42.970 | 16 KiB | 224 |
| `lowvgpr` | 16 | 32 | 32 | 6.778 | 56.910 | 0 | 80 |
| `lowvgpr48` | 16 | 48 | 32 | 6.234 | 52.145 | 0 | 96 |
| `default` | 48 | 64 | 32 | 7.504 | 57.958 | 0 | 248 |

Wider blocks read the activation matrix fewer times: 363 column blocks at 48
columns against 544 at 32 and 1088 at 16, for a 5120-wide K per row. The 32- and
48-column rows of that table are occupancy-neutral (16 KiB and 24 KiB both allow
2 workgroups per CU at 128 threads), so the 1.27-1.39x gap is not an occupancy
effect.

## Occupancy

Every registered dual gate/up SiLU variant reports 248 allocated VGPR and 32768
LDS bytes, which pins all of them at 2 workgroups per CU and 2 waves per SIMD.
LDS is fixed because the shared storage is a union whose weight member
(2 matrices x 256 K x 32 columns x 2 B) does not depend on rows or row tiles per
wave; VGPR is fixed because the kernel's `__launch_bounds__(128, 1)` lets the
compiler spend the registers freed by smaller accumulators.

That matches the wider family: the fastest prefill owners in the companion
profile all sit at 2 waves per SIMD, while the owner with the most resident waves
(q6, 96 VGPR, 5 waves per SIMD) is the slowest Q6 prefill owner. LDS headroom
alone does not raise resident waves — `shared_b2w4` has 16 KiB LDS (4 workgroups
per CU of headroom) and still allocates 224 VGPR, so it stays at 2 waves per
SIMD.

## Comparison with the llama.cpp fork

The fork (`654803517b06da47f5210553a661bf6c80deb97f`,
`ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`) adds 128-thread dense MMQ entries
with a 64-row weight tile beside the 256-thread 128-row entries, keeping the
128-token tile. Its axes map to this kernel as: fork `I` (weight rows) to
columns per block, fork `J` (tokens) to rows per block, and fork thread count to
32 x waves. The change is therefore a halved weight axis plus halved threads. On
this host and shape the measured equivalents of that direction are 1.14x slower
(rows halved at 32 columns) and 1.27-1.39x slower (columns halved at 256 rows),
so the fork's tile change does not transfer to this kernel.

## Candidate direction

The measured per-block cost is dominated by work that rows per block amortize
and by operand traffic that a wider block amortizes, so the candidate that
follows from this screen raises amortization while holding the resources that are
already at the family norm. Ranked, with the full rationale in
`artifact.json`:

1. **16 columns x 512 rows per block, row tiles per wave 8, 4 waves, 128
   threads.** Halves the per-block decode for the same compute and halves
   b-fragment loads per WMMA, at the same 32 KiB LDS, 128 accumulator VGPRs and
   2 waves per SIMD. A source-level instruction count predicts about 3072 warp
   instructions per K block for the same 8.39 MFLOP against 3840 for the parent.
   Risk: 16-column blocks double activation traffic per FLOP.
2. **16 columns x 256 rows, 4 waves, 128 threads, launch bounds min 4 blocks per
   CU.** The only arm that can raise resident waves: 16 KiB LDS allows 4
   workgroups per CU, and the register cap forces the 128 VGPR budget that 4
   waves per SIMD requires.
3. **32 columns x 512 rows, 8 waves, 256 threads** with a register-local SiLU
   epilogue (2 x 512 x 32 x 2 B would exceed the LDS union). Keeps the
   measured-good column tile and only doubles rows per block.
4. **Decode restructure at unchanged geometry.** The decode reads one packed
   byte per thread per two weights and is the largest single block of
   non-WMMA instructions.

Rejected by measurement: every shrink of rows per block, shrinking the column
tile at fixed rows, and occupancy gains that rely on LDS savings without a
register cap.

## Files

| File | Contents |
| --- | --- |
| `artifact.json` | Assembled result: knob table, sweeps, ladder resources, findings, candidates. |
| `assemble.py` | Regenerates `artifact.json` from the three inputs below. |
| `resource_table.json` | Per-kernel VGPR/SGPR/LDS for all 49 kernels of the family, from compiler metadata. |
| `rowtile_prefill.json` | Dual gate/up SiLU row-tile sweep on real model tiles. |
| `col_tile_owners.json` | Single-matrix column-tile sweep. |

## Reproduce

```bash
# ROCm environment: conda activate therock, then export LD_LIBRARY_PATH from
# python -m rocm_sdk path --root
cd /home/lhl/hipEngine

# 1. Per-kernel resources (compiles the kernel family for gfx1151, ~15 s)
python3 scripts/gguf_prefill_kernel_resources.py \
  --source hipengine/kernels/hip_gfx1100/quant/gguf_k_t16_selected_prefill.hip \
  --arch gfx1151 --waves-per-workgroup 4 \
  --json benchmarks/results/2026-09-13-q4-dual-prefill-tile-knob-screen/resource_table.json

# 2. Row-tile sweep on real gate/up tiles (5 row counts, ~4 min)
PYTHONPATH=. .venv/bin/python scripts/qwen38_packet4_q4_dual_silu_row_leaf.py \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --rows 256 512 1024 2048 4096 \
  --burst 4 --repetitions 3 --warmups 2 \
  --output benchmarks/results/2026-09-13-q4-dual-prefill-tile-knob-screen/rowtile_prefill.json

# 3. Column-tile sweep on the single-matrix siblings (~1 min)
PYTHONPATH=. .venv/bin/python scripts/gguf_q4_t16_dense_prefill_owner_microbench.py \
  --rows 512 4096 --in-features 5120 --out-features 17408 --warmup 3 --iters 10 \
  --json benchmarks/results/2026-09-13-q4-dual-prefill-tile-knob-screen/col_tile_owners.json

# 4. Assemble (idempotent apart from provenance.generated_at)
python3 benchmarks/results/2026-09-13-q4-dual-prefill-tile-knob-screen/assemble.py
```

## Limits

Single host, single model, one warmup and three measured repetitions per cell.
The column-tile direction is measured on single-matrix siblings because the dual
kernel cannot express a 16-column block without a new instantiation. Occupancy
figures come from compiler metadata, not from a hardware counter. The row-tile
sweep times one launch at a time; it does not model end-to-end prefill.
