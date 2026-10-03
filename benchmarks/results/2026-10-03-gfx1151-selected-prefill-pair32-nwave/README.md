# gfx1151 selected-expert prefill: multi-wave block + LDS activation staging (R12 ladder step 1)

The two selected-expert prefill kernels of Qwen3.6-35B-A3B `UD-Q4_K_M` on
gfx1151 launched one wave32 per output tile with no shared memory: one
`(16-row, 32-column)` block for the fused gate+up leg and one
`(16-row, 16-column)` block for the down leg, each dequantizing its Q4_K/Q5_K
weight fragment once per 16-row tile. This change gives both legs a
`NUM_WAVES`-wave block that stages a 32-row activation tile in 1 KiB of LDS and
pairs two 16-row tiles per wave, so each dequantized weight fragment feeds two
WMMA accumulators.

Both bodies are **bit-identical** to the kernels they pair, on all four T16
quants (Q4_K, Q5_K, Q5_K-qmicro, Q6_K) x BF16/FP16 and four shapes including
uneven, empty-first, and expert-straddling row splits.

## Result

Same protocol, one process per arm, `HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE=baseline|pair32`.

| Run | Baseline (tok/s) | pair32 (tok/s) | Change |
| --- | ---: | ---: | ---: |
| 1 | 1450.8 | 1603.7 | **+10.54%** |
| 2 (committed tree) | 1454.0 | 1639.3 | **+12.75%** |

Both runs measured 4096-token prefill, `--public-ar-profile`, chunk 1024,
`--warmup-runs 1 --measured-runs 3`; every arm reports
`effective_use_wmma_prefill=true`. All three pair32 samples are at or above
1600 tok/s in run 1 (1602.7 / 1603.7 / 1604.6) and 1638.9 / 1639.3 / 1640.0 in
run 2.

The retained headline is the **conservative** pair, run 1: **+10.54%**, against
the pre-declared primary target of +10.0% (>= 1600 tok/s).

Run 2 is the same A/B on the committed source and reads 2.2% higher on the
pair32 arm while its baseline arm moved 0.2%. That between-session drift on the
pair32 arm is not explained here and is the reason the lower pair is the one
carried forward.

## Expert-leg rate

The R12 design basis prices the two expert legs at **8452.5 GFLOP / 993.40 ms =
8.51 TFLOP/s** and sets the primary target at **>= 11.3 TFLOP/s**, which needs
242.6 ms cut from the 993.40 ms leg total.

Run 1's wall drops by 269.2 ms and only these two kernels changed, so the legs
fall to 724.2 ms. By the design basis's own FLOP accounting that is
**11.49 TFLOP/s**, clearing the 11.3 TFLOP/s bar.

## Isolated legs (real shape, 256 experts, 32 rows/expert)

`compact_rows = 8192`, one launch per variant, 10 warmup + 50 measured calls.

| Leg | Baseline | pair32 | Speedup | Baseline TFLOP/s | pair32 TFLOP/s | Bit-exact |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| gate+up (K=2048, N=512+512) | 3.809 ms | 2.910 ms | **1.309x** | 9.02 | 11.81 | yes |
| down Q4_K (K=512, N=2048) | 1.983 ms | 1.520 ms | **1.305x** | 8.66 | 11.31 | yes |
| down Q5_K qmicro | 2.409 ms | 2.190 ms | 1.100x | 7.13 | 7.84 | yes |

The Q4_K down row is the one the shipping model uses. The Q5_K-qmicro sibling is
registered and bit-exact but only 1.10x here; it is not on this model's path.

## Mechanism

`rocprofv3 --kernel-trace` over both variants in one process:

| Leg | Kernel | Calls | us/call | VGPR | LDS | block | grid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| gate+up | `..._dual_wmma_prefill_compact32_kernel<u16,false>` | 6 | 3824.3 | 56 | 0 | 32 | 1024 x 512 |
| gate+up | `..._dual_wmma_prefill_pair32_nwave_kernel<u16,false,2>` | 6 | 2892.6 | 88 | 1024 | 64 | 2048 x 256 |
| down | `..._wmma_prefill_compact_kernel<u16,4,false>` | 7 | 1969.0 | 48 | 0 | 32 | 4096 x 512 |
| down | `..._wmma_prefill_pair32_nwave_kernel<u16,4,false,4>` | 7 | 1510.8 | 104 | 1024 | 128 | 4096 x 256 |

Both paired kernels run under their expected names with plausible durations.
Grid halving in `y` is the two-row-tile pairing; `grid.x` doubling for the
gate+up leg is the 32-column block that `NUM_WAVES=2` implies.

## Commands

```bash
source /tmp/reprof/env.sh; export PYTHONPATH=/home/lhl/hipEngine
for m in baseline pair32; do
  HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE=$m $PY scripts/qwen35_gguf_bench.py \
    --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --quant gguf_ud_q4_k_m \
    --prompt-length 4096 --decode-tokens 0 --warmup-decode-tokens 0 \
    --warmup-runs 1 --measured-runs 3 --public-ar-profile \
    --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
    --compiler-version-file /tmp/reprof/compiler.txt --require-cached-build \
    --json /tmp/reprof/r12/ab2-$m.json
done
```

Bit-exactness gate:

```bash
$PY -m pytest tests/test_gpu_gguf_k_t16_selected_wmma_prefill.py \
  -k pair32 -q
```

## Limits

- One host (integrated Strix Halo APU, not the gfx1100/W7900 default hardware),
  one model, one quant, one prompt family, one profile, one process per arm.
- gfx1100/W7900 is unverified. The change is bit-identical with an unchanged K
  traversal and operand order, so it is not gfx1151-specific by mechanism, but
  the magnitude is not measured there.
- The pair32 arm showed 2.2% between-session drift across the two A/B pairs
  while the baseline arm moved 0.2%; the conservative pair is retained.
- `pair32` is **not** the automatic default. The gfx1151 selected-prefill
  LDS-staging family already has one repeated-128K lifecycle-stall precedent
  (`compact32_shared_x`), and the new bodies have not been through that gate.
  See `docs/REFACTOR.md`.
- Isolated-leg TFLOP/s are single-uniform-launch rates; the model's 41 layers
  have different shapes, so the design basis's per-layer profile reads 9.49
  TFLOP/s for gate+up against 9.02 here.

## Files

- `pair32-nwave.json` — end-to-end A/B pairs, isolated legs, design-basis targets.
- `rocprof-kernel-trace.json` — `rocprofv3 --kernel-trace` summary for both legs.
