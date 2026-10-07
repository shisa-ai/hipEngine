# Dense Q8_0 checked-int32/full-four-row DPP coltile screen

The separately registered diagnostic sibling
`coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out`
is **1.188–1.506x faster** than the current dense DPP owner
`coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out` across 48 real-weight,
complete dense Q8_0 projection cases. Geometric mean speedup is **1.414x**
(29.2% less operation time); 48/48 case medians improve and both first/second
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
| 512 | 16 | 1.420x | 1.240–1.506x |
| 1024 | 16 | 1.424x | 1.238–1.506x |
| 4096 | 16 | 1.398x | 1.188–1.473x |

The exact per-case medians, order strata, weight hashes, source-file/build
hashes and command are in [`screen.json`](screen.json). Raw per-sample JSON
stays outside Git under
`/tmp/qwen4exp-dense-dpp-checked-repair-20261007/screen.samples.json`
(SHA256 `c6e8bced2fe2f9998c8e1c28902d43e2add0a5e01f38243592fe4a2a99785c95`); the
committed `screen.json` records the raw full-output hash
`raw_artifact_sha256` and the raw-sample hash `raw_samples_sha256`.

## Contract and validation

The sibling is separately registered under `linear`/`gguf_q8_0`. Its Python and
C++ admission both require a Q8_0 block-aligned `in_features`, `out_features`
divisible by 8, rows divisible by 4, and complete signed-int32 float activation
byte (`rows * in_features * 4`), float output byte (`rows * out_features * 4`),
raw-weight byte (`out_features * row_bytes`) and grid products. The C++ launcher
validates the scalar dimensions before forming the packed row byte count, so a
block-aligned near-INT64_MAX `in_features` cannot overflow signed int64. A tail,
overflow or unsupported shape raises before the library loads; the generic DPP
parent keeps serving it.

The focused GPU gate passes 25/25 cases on gfx1151: bit-exact parity with the
DPP parent at K32/96/160/2560/6144/10240, subnormal/wide/mixed Q8 scales,
subnormal/cancellation/wide activations, output guard bands, input immutability,
a nondefault stream, tail-row refusal with parent fallback, grandparent parity,
raw-ABI refusal of block-aligned near-INT64_MAX K and extreme int64 rows/N before
any pointer dereference, and the CPU-reference floor (max KL <= 0.05, top-1 >=
90%). The CPU unit gate passes 59 cases including the admission bounds and
refusal-before-build. The existing DPP-reduction and pipelined-sibling gates
still pass, so the old instantiations are unchanged.

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
`6a62e39137715b18337383ba5cb7c2896351b2367a11946a6e18a25700de5be9`, loaded
library SHA256 `e5d1cf9572a225ce5920e133616b3a474e4393f62942dd2f6c8562cc7d09bd86`,
device object SHA256
`f38d594b938c58819c81ce5f265578889081755db627a5aa5ae79894e33233c2`):

| Symbol (template tail) | VGPR | SGPR | Private | Static instrs | Carry pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPP parent `...,true,false,true,long,false` | 76 | 50 | 0 | 872 | 42 |
| checked sibling `...,true,false,true,int,true` | 73 | 54 | 0 | 703 | 13 |
| shuffle pipeline `...,true,false,false,long,false` | 76 | 50 | 0 | 1035 | 42 |

The sibling drops 169 static instructions (19.4%) and 29 int64 carry pairs
(69%) while keeping the DPP reduction (32 `v_permlanex16` + 128
`v_add_f32_dpp`). A cached `rocprofv3 --kernel-trace` on `attn_qkv` rows 512
shows both instantiations engaged on the same 163840x128 grid, 3 launches each,
with median device durations 8.009 ms (parent) and 5.428 ms (sibling),
scratch 0. Trace CSV SHA256
`7e959844bed65c38ef78cdce8657d9e6bede96946fbce7bda26c86a9046ca240`.
Full rows are in [`resources.json`](resources.json).

## Repair and provenance

The repair commit `83b028c60b719988e973b4b2d0d7e72bbb74ea5a` tightened the
Python/C++ admission to float activation/output byte ranges and validates scalar
dimensions before forming the packed row byte count. The earlier `b91209bc8`
screen recorded only a dirty head with no diff or source-file hashes, so its
source provenance was incomplete; this packet was regenerated on the
tracked-clean repair commit with per-file source hashes
(`kernel_hip` `8716c2f8…`, `wrapper_py` `e6f7d0d6…`, `harness_py` `64704876…`),
loaded-library and build-cache identity, and raw-sample hashes. The repair is
host-side admission only; regenerated device resources match the prior counts
and trace durations are similar, not identical.

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
  --output /tmp/qwen4exp-dense-dpp-checked-repair-20261007/screen.json
```

The harness pins `HIPENGINE_REQUIRE_CACHED_BUILD=1`, so prebuild the family with
the pinned compiler-version file before profiling. Trace:

```bash
rocprofv3 --kernel-trace --output-format csv --output-file dense-checked-repair \
  --output-directory /tmp/qwen4exp-dense-dpp-checked-repair-20261007/trace -- \
  uv run --no-sync python scripts/qwen4exp_dense_dpp_checked_full_rows_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --layers 0 --rows 512 --pairs 2 --tensors attn_qkv.weight \
  --output /tmp/qwen4exp-dense-dpp-checked-repair-20261007/trace/trace-run.json
```

## Limitations

- The screen ran on the tracked-clean repair commit
  `83b028c60b719988e973b4b2d0d7e72bbb74ea5a`; the source state, source-file
  hashes and build identity are recorded in `screen.json`.
- Synthetic activations only. A natural-input owner/state qualification and a
  production numerical-envelope review are required before any default change.
- One host, one model, two layers and one quant. No model rate is claimed.
- No speedup is predicted outside the admitted complete four-row shapes.
