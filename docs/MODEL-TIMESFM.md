# MODEL-TIMESFM.md — TimesFM 2.5 200M on hipEngine

Status: **complete (GPU decode, production quality)**. This file is the
per-model record: what the model is, how hipEngine runs it, the measured
performance against reference implementations, and the optimization history
with the exact gains. Future model ports get their own `docs/MODEL-<NAME>.md`
following this format.

## Model

**Checkpoint:** `google/timesfm-2.5-200m-pytorch` (Hugging Face
safetensors, ~231M tensor parameters, FP32 storage).

TimesFM 2.5 is Google's patched-decoder time-series forecaster. It ingests
a raw series (plus a boolean mask channel), normalizes it per RevIN-style
running statistics, tokenizes it into non-overlapping patches, and runs a
pre-norm transformer whose outputs are projected by three heads:

- **Point head:** backcast (input reconstruction) + forecast horizon of
  128 points per patch.
- **Quantile head:** 1024-horizon output at 10 quantiles (median is
  duplicated; index 5 is the autoregressive decode channel).

Decoding longer horizons is autoregressive over patches: each step feeds
the last patch's point forecast back through the stack until the full
horizon is produced.

**Geometry (from `config.json`):**

| Parameter | Value |
| --- | --- |
| Hidden size | 1280 |
| Layers | 20 pre-norm transformer |
| Attention heads × head dim | 16 × 80 (fused QKV, RoPE, QK norms, per-dim softplus query scaling) |
| Patch length / input dims | 32 (value + mask channels → 64) |
| Context length | 16384 |
| Point horizon / quantile horizon | 128 / 1024 × 10 |
| RMSNorm epsilon | 1e-6 |

**Attention semantics** (correctness-relevant): unscaled dot-product
scores with a combined causal + front mask; a query row whose keys are all
masked attends uniformly (1/S) — this comes from the reference's
finite-negative fill and must be reproduced by every path.

## hipEngine implementation

Torch-free HIP implementation on the gfx1151 backend (AMD Radeon 8060S,
Strix Halo iGPU). Weights load from the loader's FP32 buffers; the
production decoder converts them to FP16 on device at init.

| Piece | Path |
| --- | --- |
| Model contract / spec | `hipengine/models/timesfm.py` |
| Checkpoint loader | `hipengine/loading/timesfm.py` |
| GPU decoder (host orchestration) | `hipengine/runtime/timesfm_decode.py` |
| HIP kernels (norms, RoPE+scatter, flash attention, heads) | `hipengine/kernels/hip_gfx1100/timesfm/timesfm.{hip,py}` |
| CPU reference + oracle | `hipengine/kernels/cpu_reference/timesfm.py` |
| Bench + correctness guard | `scripts/timesfm_gpu_bench.py` |
| Tests | `tests/test_unit_timesfm_model_contract.py`, `tests/test_live_timesfm_cpu_reference.py`, `tests/test_gpu_timesfm_gpu_decode.py` |

**Precision modes:**

- **`fp16` (production default):** FP16 storage and WMMA/rocBLAS FP16
  matrix cores with FP32 accumulation. Exact control/ownership semantics;
  gate: strict-teacher KL/top-1 plus a max-error budget of 2% of signal
  scale against the torch oracle (measured ≤ 0.86%).
- **`fp32` (strict):** FP32 SGEMM path with strict parity to the oracle
  fixture (atol 5e-4). Used by `--check` as the exactness contract and as
  the rollback path.

Both modes are validated per change by
`PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/timesfm_gpu_bench.py --check`.

## Performance

Workload: batch 8, context 8192, horizon 512 (one full decode: prefill of
256 patches + 3 autoregressive steps × 20 layers).

Two physical Strix Halo hosts are recorded as separate lanes. They run the
same `gfx1151` backend, the same code, and the same workload; the difference
between them is host power/thermal headroom, so their absolute rates must not
be read as an old→new delta. The torch and CPU reference rows exist only on
the zbook lane (the Framework Desktop ROCm environment has no torch install).

**HP ZBook Ultra G1a lane** (power/thermal limited; same host throughout):

| Path | Precision | Decode time | Speed vs hipEngine fp16 |
| --- | --- | ---: | ---: |
| **hipEngine GPU** | fp16 (production) | **0.082 s** | 1× |
| hipEngine GPU | fp32 (strict, atol 5e-4) | 0.49 s | 6.0× slower |
| Torch reference, same GPU | fp32 | 0.707 s | 8.6× slower |
| Torch reference, CPU | fp32 | 9.52 s | 116× slower |
| hipEngine NumPy CPU reference | fp32 | 9.13 s | 111× slower |

**Framework Desktop lane** (host `gfx1151`, machine ID
`55ea6c509d0b49eea8de7094a1023668`; median of five independent process
invocations, 0.062065 / 0.061938 / 0.062128 / 0.062122 / 0.062066 s):

| Path | Precision | Decode time | Speed vs hipEngine fp16 |
| --- | --- | ---: | ---: |
| **hipEngine GPU** | fp16 (production) | **0.062 s** | 1× |

The Framework Desktop runs this workload 25% faster than the zbook (0.062 s
vs 0.082 s). That is a host-level difference between two machines with the
same GPU, not a code improvement.

Reproduce with
`PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/timesfm_gpu_bench.py --metric`
(best of N decodes; the same command with `--check` runs the fixture-parity
guard).
Full artifacts per step live in `benchmarks/results/gfx1151-timesfm-*.json`.
Lane baselines: zbook
[`gfx1151-timesfm-quadtile-flash-2026-09-09.json`](../benchmarks/results/gfx1151-timesfm-quadtile-flash-2026-09-09.json),
Framework Desktop
[`2026-09-11-framework-desktop-timesfm-2p5-decode-lane.json`](../benchmarks/results/2026-09-11-framework-desktop-timesfm-2p5-decode-lane.json).

## Optimization history

All rows are the same workload on the same hardware; each step's change is
cumulative. "Δ" is the improvement over the previous row.

| # | Change | Decode (s) | Δ (ms) | Δ (%) |
| --- | --- | ---: | ---: | ---: |
| 1 | First working GPU path (rocBLAS SGEMM + naive kernels, FP32) | 1.096 | — | — |
| 2 | FP16 storage, FP32-accum GEMMs (matrix cores) | 0.492 | −604 | −55% |
| 3 | Attention onto batched GEMMs (QK^T / AV) | 0.368 | −124 | −25% |
| 4 | Block-per-row cooperative kernel reductions | 0.174 | −194 | −53% |
| 5 | Host-path vectorization (running stats, last-patch D2H) | 0.141 | −33 | −19% |
| 6 | RoPE timescale table + head-kernel block tuning | 0.122 | −19 | −13% |
| 7 | Quantile-head work reduction (last patch only) | 0.115 | −7 | −6% |
| 8 | Merged qkv norm+scatter kernel | 0.104 | −11 | −10% |
| 9 | RoPE folded into the merged kernel | 0.099 | −5 | −5% |
| 10 | WMMA flash attention (fused causal softmax, no scores buffer) | 0.0959 | −3 | −3% |
| 11 | transpose-heads rewrite + rocBLAS solution autotune (shape-keyed) | 0.0915 | −4 | −5% |
| 12 | flash_short: 4-wave kv split for AR queries | 0.0913 | −0.2 | −0.2% |
| 13 | Warp-per-head qkv scatter (shuffle reductions) | 0.0909 | −0.4 | −0.5% |
| 14 | Pair-tile prefill flash rounds (reductions amortized over 2 kv tiles) | 0.0874 | −3.5 | −4% |
| 15 | Pair-tile flash_short | 0.0861 | −1.3 | −1.5% |
| 16 | Scatter constant folding (qscale vector; kernel was transcendental-bound) | 0.0844 | −1.7 | −2% |
| 17 | Quad-tile prefill flash rounds | 0.0823 | −2.1 | −2.4% |

Total: **1.096 s → 0.082 s (13.3×)**. Rows 2, 3, 4 were the structural
wins; rows 10–17 were a second campaign targeting kernel-level overhead
after the profile showed the path was no longer host-bound.

### Negative results (measured, reverted, kept for the record)

| Attempt | Result |
| --- | --- |
| HIP graph capture of the forward | Net loss: graph-node execution overhead exceeded the Python launch savings (0.104–0.109 s vs 0.097–0.099 s eager), and any capture attempt slowed later eager runs ~8% on this ROCm stack |
| Split-K AR flash (separate partial+merge kernels) | 146 µs vs 113 µs per call — per-chunk fixed cost exceeded the parallelism win |
| Custom WMMA small-GEMM for AR shapes | 62 µs vs 49 µs autotuned rocBLAS (qkv); flat overall once shape-keyed solution autotuning exists |
| Quad-tile flash_short | 100.1 µs vs 97.4 µs — LDS pressure from the doubled staging buffer |
| Whole-layer fusion experiment (rope+norm+scatter, 256-float register arrays) | 110.7 vs 90.1 ms per 3 decodes — register spills |

## Where the remaining time goes

rocprofv3 breakdown at 0.082 s (per decode): GEMMs ~51 ms (prefill GEMMs
at ~38–42 TF/s, the practical FP16 matrix-core peak on this device; AR
GEMMs at the autotuned rocBLAS floor), prefill flash ~13 ms, AR flash
~5.8 ms, qkv scatter ~4.3 ms, norm/elementwise ~6.5 ms (effective memory
bound). Every remaining lever identified is either measured-dead (above),
at a hardware bound, or carries register-spill risk for under 1.5 ms.

## Notes for future model ports

- Keep the correctness gate first: a CPU-reference oracle fixture plus a
  strict-parity FP32 path and a tolerance-gated production FP16 path, both
  runnable from one bench command.
- Profile before optimizing: the phase-1 wins (precision, matrix cores,
  block reductions) and the phase-2 wins (kernel overhead) came from
  entirely different bottlenecks, found by rocprofv3 kernel traces.
- Watch for constant-foldable math in hot kernels (the softplus-over-
  constants bug cost ~2 ms until noticed) and for per-element transcendentals.
- Record negative results with numbers; they steered three of the later
  decisions here.
