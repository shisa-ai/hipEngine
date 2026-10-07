# Dense Q8_0 checked-int32/full-four-row DPP coltile screen

The separately registered diagnostic sibling
`coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out`
is **1.111–1.506x faster** than the current dense DPP owner
`coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out` across 48 real-weight,
complete dense Q8_0 projection cases. Geometric mean speedup is **1.409x**
(29.0% less operation time); 48/48 case medians improve and both first/second
arm orders win every case. The full output tensor is bit-identical to the
parent before and after the timing loop (one full-tensor `uint32` comparison at
each end, not a per-pair check). The candidate
is default-off, no production selector points at it, and the generic DPP parent
remains the fallback for tails, overflows and unsupported shapes.

This is a diagnostic operation screen, not a model rate or default promotion.

## Measurement basis

- Physical host: `gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`,
  AMD Radeon 8060S (Strix Halo), `tuned-adm` profile `accelerator-performance`.
- Model: Qwen3.8-Flash-Next UD-Q4_K_XL at
  `/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL`; real Q8_0
  weight banks at layers 0 and 22.
- Shapes (K x N): `attn_qkv` 2560x10240, `attn_gate` 2560x6144, `ssm_out`
  6144x2560, `ffn_gate_shexp`/`ffn_up_shexp` 2560x640, `ffn_down_shexp`
  640x2560, `hc_attn_down`/`hc_ffn_down` 10240x320; rows 512/1024/4096.
- Parent: `gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out`.
  Candidate:
  `gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out`.
- Timing: 20 synchronized pairs per case, alternating parent-first and
  candidate-first. Each sample wraps the complete host launch plus stream
  synchronize; compilation happens before any timed region. The exactness
  verdict is a full-tensor `uint32` comparison once before and once after the
  timing loop.
- Activations are synthetic (seeded per layer/rows, standard normal scaled by
  0.1); the weight bytes are the real model tensors. A natural-input owner/state
  qualification is a separate gate and is not claimed here.
- Toolchain: TheRock ROCm 10.0.0, HIP 7.15.26333, AMD clang 23.0.0git,
  compiler commit `8f497e0992fb7513f7f78a6f6b6f1056c375e961`.

| Rows | Cases | Geomean speedup | Range |
| ---: | ---: | ---: | ---: |
| 512 | 16 | 1.419x | 1.240–1.504x |
| 1024 | 16 | 1.423x | 1.241–1.506x |
| 4096 | 16 | 1.384x | 1.111–1.474x |

The exact per-case medians, order strata, weight hashes and command are in
[`screen.json`](screen.json). Raw per-sample JSON stays outside Git under
`/tmp/qwen4exp-dense-dpp-checked-20261007/`.

## Contract and validation

The sibling is separately registered under `linear`/`gguf_q8_0`. Its Python and
C++ admission both require a Q8_0 block-aligned `in_features`, `out_features`
divisible by 8, rows divisible by 4, and complete signed-int32 activation
(`rows * in_features`), output (`rows * out_features`), raw-weight byte
(`out_features * row_bytes`) and grid products. A tail, overflow or unsupported
shape raises before the library loads; the generic DPP parent keeps serving it.

The focused GPU gate passes 25/25 cases on gfx1151: bit-exact parity with the
DPP parent at K32/96/160/2560/6144/10240, subnormal/wide/mixed Q8 scales,
subnormal/cancellation/wide activations, output guard bands, input immutability,
a nondefault stream, tail-row refusal with parent fallback, grandparent parity
and the CPU-reference floor (max KL <= 0.05, top-1 >= 90%). The CPU unit gate
passes 55 cases including the admission bounds and refusal-before-build. The
existing DPP-reduction and pipelined-sibling gates still pass, so the old
instantiations are unchanged.

## Implementation boundary

The sibling narrows the pipelined leaf's weight, activation and output offsets
from int64 to int32 (`index_t = int`) and removes the per-row validity control
for complete four-row groups (`FULL_ROWS`). It preserves the Q8 scale decode,
ascending per-lane K FMA order, DPP reduction, four-wave publication, F32
output, tile shape and grid order, so every admitted shape is bit-identical.
The generic DPP parent, the strict wave-scale grandparent and the shuffle-tree
pipeline are untouched. No row-batch widening, staging, operand rounding,
output-layout change or default edit is included.

## Cached resource and trace evidence

From the cached gfx1151 device code object (`cache_key`
`6f0b72061932469996e61bcd1c1ffde555a9c2dd0cb78600e238993838e19ce3`):

| Symbol (template tail) | VGPR | SGPR | Private | Static instrs | Carry pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPP parent `...,true,false,true,long,false` | 76 | 50 | 0 | 872 | 42 |
| checked sibling `...,true,false,true,int,true` | 73 | 54 | 0 | 703 | 13 |
| shuffle pipeline `...,true,false,false,long,false` | 76 | 50 | 0 | 1035 | 42 |

The sibling drops 169 static instructions (19.4%) and 29 int64 carry pairs
(69%) while keeping the DPP reduction (32 `v_permlanex16` + 128
`v_add_f32_dpp`). A cached `rocprofv3 --kernel-trace` on `attn_qkv` rows 512
shows both instantiations engaged on the same 163840x128 grid, 3 launches each,
with median device durations 8.014 ms (parent) and 5.425 ms (sibling),
scratch 0. Trace CSV SHA256
`c50c7f2ea5a532b5d2117cd29c5c8bea76c8d51c3e6979122dd05acd17b48d2f`.
Full rows are in [`resources.json`](resources.json).

## Reproduce

```bash
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt
uv run --no-sync python scripts/qwen4exp_dense_dpp_checked_full_rows_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --layers 0,22 --rows 512,1024,4096 --pairs 20 \
  --output /tmp/qwen4exp-dense-dpp-checked-20261007/screen.json
```

The harness pins `HIPENGINE_REQUIRE_CACHED_BUILD=1`, so prebuild the family with
the pinned compiler-version file before profiling. Trace:

```bash
rocprofv3 --kernel-trace --output-format csv --output-file dense-checked \
  --output-directory /tmp/qwen4exp-dense-dpp-checked-20261007/trace -- \
  uv run --no-sync python scripts/qwen4exp_dense_dpp_checked_full_rows_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --layers 0 --rows 512 --pairs 2 --tensors attn_qkv.weight \
  --output /tmp/qwen4exp-dense-dpp-checked-20261007/trace/trace-run.json
```

## Limitations

- The screen ran from a dirty working tree (the candidate commit follows it);
  the source state is recorded in `screen.json`.
- Synthetic activations only. A natural-input owner/state qualification and a
  production numerical-envelope review are required before any default change.
- One host, one model, two layers and one quant. No model rate is claimed.
- No speedup is predicted outside the admitted complete four-row shapes.
