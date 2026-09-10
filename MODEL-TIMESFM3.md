# MODEL-TIMESFM3.md — TimesFM 3.0 500M on hipEngine

Status: **complete (GPU decode, initial implementation; no optimization
campaign yet)**. Per-model record following `MODEL-TIMESFM.md`.

## Model

**Checkpoint:** `google/timesfm-3.0-pytorch` (Hugging Face safetensors, 445
tensors, ~331M tensor parameters, FP32 storage, TimesFM Non-Commercial
License v1.0).

TimesFM 3.0 is a **non-autoregressive** multivariate patched forecaster.
Inputs are `(batch, variates, patches, patch_len)` — targets plus optional
past-only and past-future covariates. A 192-input ReLU ResidualBlock
tokenizer (values + the future-covariate roll + both mask channels) feeds 20
`MixingTransformer` layers: causal sequence attention over the patch axis
(RoPE, QK RMSNorm, per-dim query scaling, scores × √head_dim), non-causal
**variate attention** across up to 32 variates, then a ReLU FFN — each with
pre/post RMSNorm pairs. A single `Linear(1280, 64×9)` head emits 9 quantiles
(median at index 4) per patch; horizon patches are appended fully masked
(CPM), predictions are stitched from overlapping forecast patches, RevIN
stats at CPM positions are refined iteratively from the model's own median
forecasts, and linear detrending is applied to context (and re-added to
forecasts) when it reduces residual std below 0.5× original.

Decoding is **one forward pass** — no AR loop, no KV cache.

**Geometry:** 1280 dims, 20 layers, 16 heads × 80 head dim, patch 32,
output patch 64 (rolls 2), context up to 16K, 9 quantiles, max 32 variates.

**Numerics that differ from 2.5:** torch `nn.RMSNorm` semantics
(`eps = finfo(float32).eps` ≈ 1.19e-7, not 1e-6), ReLU activations (not
Swish), no biases except the output head, SDPA score scaling × √head_dim,
fully-masked attention rows produce zeros (CPU SDPA semantics).

## hipEngine implementation

Torch-free HIP implementation on the gfx1151 backend (AMD Radeon 8060S,
Strix Halo iGPU) — same device class as the 2.5 decoder.

| Piece | Path |
| --- | --- |
| Model contract / spec | `hipengine/models/timesfm3.py` |
| Checkpoint loader | `hipengine/loading/timesfm3.py` |
| GPU decoder (host orchestration) | `hipengine/runtime/timesfm3_decode.py` |
| New HIP kernels (var attention, eps scatter, relu) | `hipengine/kernels/hip_gfx1100/timesfm3/timesfm3.{hip,py}` |
| Reused 2.5 kernels (norms, rope, flash, scatter, transpose) | `hipengine/kernels/hip_gfx1100/timesfm/` |
| CPU reference + oracle | `hipengine/kernels/cpu_reference/timesfm3.py`, `scripts/timesfm3_oracle_torch.py` |
| Bench + correctness guard | `scripts/timesfm3_gpu_bench.py` |
| Tests | `tests/test_timesfm3_{model_contract,cpu_reference,gpu_decode}.py` |

Host/device split: padding, detrending, running stats, the future-covariate
roll, CPM RevIN refinement, stitching, and the final RevIN reversal run as
host numpy (O(batch × variates × patches) scalar work); everything
O(rows × hidden) runs on device. Sequence attention reuses the 2.5 flash
kernel with `B = batch × variates` independent sequences and one scratch
cache pair shared across layers (no AR loop); the new `var_attention`
kernel does non-causal variate attention with per-(batch, variate)
leading-mask counts. The SDPA √head_dim score scale is folded into the
K-side norm weight (both precisions).

**Precision modes:**

- **`fp16` (production default):** FP16 storage and rocBLAS `gemm_ex`
  FP16/FP32-accum GEMMs; gate: max ≤ 2%, mean ≤ 0.5% of per-series signal
  scale vs the torch oracle on both fixtures (measured max 0.42% / mean
  0.12%).
- **`fp32` (strict):** rocBLAS SGEMM + FP32 kernels; parity vs the torch
  oracle fixture at max 1.1e-5.

Both validated by `python3 scripts/timesfm3_gpu_bench.py --check`.

## Performance (initial implementation, no optimization campaign)

Workload: batch 8, 3 variates (2 targets + 1 past-future covariate), context
8192, horizon 512 — one non-autoregressive forward pass (258 context + 17
horizon patches; 6,600 rows). Same host throughout (zbook, Ryzen AI MAX+
PRO 395, Radeon 8060S).

| Path | Precision | Decode time | Speed vs hipEngine fp16 |
| --- | --- | ---: | ---: |
| **hipEngine GPU** | fp16 (production) | **0.397 s** | 1× |
| hipEngine GPU | fp32 (strict) | 3.39 s | 8.5× slower |
| Torch reference, same GPU | fp32 | 1.33 s | **3.36× slower than hipEngine fp16** |

rocprofv3 kernel breakdown (fp16, one decode, ~345 ms of device time):
GEMMs 134 ms, var-q/k head_rmsnorm 61 ms, head_perdim 39 ms, sequence flash
attention 39 ms, var_attention 21 ms, norms/elementwise ~45 ms, qkv scatter
10 ms.

**Known optimization headroom (not yet chased):** the var-q/k normalization
path (`head_rmsnorm` ×2 + `head_perdim` = 100 ms, 29% of decode) runs as
three separate generic kernels; fusing them into `var_attention` (the 2.5
`qkv_norm_scatter` pattern) is the first target. The fp32 strict path is
slower than torch fp32 (the naive per-row `attention_f32` kernel at
prefill scale) — acceptable for a rollback-only path.

## Notes

- `tests/fixtures/cpu_reference/timesfm_3p0_decode{,_edge}.npz` hold the
  torch-oracle decode outputs (base: multivariate + front-pad; edge:
  unaligned context/horizon, true univariate, rank-2 global mask,
  detrend on/off, forward+freeze case).
- The review-fix history for the CPU tier (rank-agnostic pad, freeze
  semantics, use_sdpa pin) is in `worklog/entries/` (458688).
