# GR-up K320/full-row DPP screen

The separately registered `K=320`, `hidden=2560`, four-branch, complete
rowbatch-4 specialization is **1.084–1.103x faster** than the promoted GR-up
DPP owner across 12 real-weight, complete GR operation cases. Geometric mean
speedup is **1.092x** (8.42% less operation time). Both materialized F32 gate
and mixed outputs are bit-identical in the focused GPU gate and in every screen
case. The candidate is default-off and the generic promoted DPP owner remains
the fallback.

## Measurement basis

- Physical host: `gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`,
  AMD Radeon 8060S (Strix Halo), accelerator-performance profile and high GPU
  clock policy.
- Model: Qwen3.8-Flash-Next UD-Q4_K_XL at
  `/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL`; real Q8_0
  GR-up weight banks at layers 0 and 22.
- Shapes: rows 512/1024/4096, K=320, N=10240, four branches, H=2560;
  four weight tensors (`blk.0.hc_attn_up`, `blk.0.hc_ffn_up`,
  `blk.22.hc_attn_up`, `blk.22.hc_ffn_up`). Synthetic F32 activation and
  normalized inputs use the screen's seeded generator.
- Parent: `gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_dpp_f32`.
  Candidate: `gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32`.
  Both are prebound complete fused GR-up operations; this is not a projection-only
  shortcut.
- Timing: 20 synchronized pairs per case, alternating parent-first and
  candidate-first. All 12 cases improve in both order strata. The screen was
  run from a dirty working tree because it precedes the candidate commit;
  the exact source state is recorded in the JSON artifact.
- Toolchain: TheRock ROCm 10.0.0, HIP 7.15.26333, AMD clang 23.0.0git,
  compiler commit `8f497e0992fb7513f7f78a6f6b6f1056c375e961`.

| Rows | Arm cases | Mean speedup range |
| ---: | ---: | ---: |
| 512 | 4 | 1.084–1.090x |
| 1024 | 4 | 1.089–1.093x |
| 4096 | 4 | 1.092–1.103x |

The exact per-case means, order strata, weight hashes and command are in
[`2026-10-06-qwen4exp-gr-k320-full-rows-screen.json`](../2026-10-06-qwen4exp-gr-k320-full-rows-screen.json).

## Contract and validation

The specialization is separately registered under `linear+gr_gated_mean`. It
is selected only behind `HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS=1`, after the
promoted DPP route has selected, and only when rows are >256, divisible by 4,
four branches and low rank 320. Other shapes retain the selected DPP key.
If the candidate registration is absent, dispatch falls back to DPP. The
production binder and default are unchanged.

The focused unit route/fallback/profile bundle passes 33 tests. The HIP GPU
bundle passes 10/10 tests on gfx1151: both backends resolve parent/DPP/K320
keys, parent and candidate outputs are exact for small and full shapes, and
K320 rows 512/1024/4096 match the DPP parent bits. The native build completes
with the current TheRock compiler. No model state/KV/logit gate or default
promotion is claimed from this operation screen alone.

## Implementation boundary

The candidate uses compile-time K320 full-row admission to remove generic
row-validity and variable-trip control in the activation loop and epilogue.
It preserves ascending per-lane K FMA order, DPP reduction, four-wave
publication, `expf`, branch mean and both outputs. The wrapper rejects K other
than 320, H other than 2560, incomplete row groups and non-DPP/non-wave-scale
requests before loading the library. No row-batch widening, staging, operand
rounding, gate-store removal or output-layout change is included.

## Reproduce

```bash
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt
uv run --no-sync python scripts/qwen4exp_q8_wave_scale_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" --gr-composite \
  --gr-parent gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_dpp_f32 \
  --gr-candidate gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32 \
  --rows 512 1024 4096 --pairs 20 \
  --tensor blk.0.hc_attn_up.weight --tensor blk.0.hc_ffn_up.weight \
  --tensor blk.22.hc_attn_up.weight --tensor blk.22.hc_ffn_up.weight \
  --output /tmp/qwen4exp-gr-k320-screen.json
```
