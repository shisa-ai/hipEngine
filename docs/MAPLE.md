# MAPLE — Ternary MoE Inference on hipEngine

Date: 2026-08-05 (branch `maple`)

Target model: [`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview)
(bf16 source of truth) via the official 2-bit ternary MLX checkpoint
[`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx)
(**5.31 GB**, the deployment artifact this engine runs).

Reference implementations:

- HF transformers: `modeling_maple.py` / `configuration_maple.py` in the bf16 repo
  (Triton/FlashAttention CUDA path; used here as the math oracle).
- MLX: [`deepgrove-ai/mlx-lm-deepgrove`](https://github.com/deepgrove-ai/mlx-lm-deepgrove)
  (`mlx_lm/models/maple.py` model, `mlx_lm/ternary.py` converter). Apple-only, so it is a
  *format* reference, not a runnable oracle on gfx1151.

## Architecture (from config + both references)

| Field | Value |
| --- | --- |
| Layers | 24, all MoE (`first_k_dense_replace=0`) |
| Hidden / head_dim | 2048 / 128 |
| Attention | GQA 16 Q-heads / 4 KV-heads, QK-RMSNorm (per-head, over head_dim) |
| Layer pattern | 3:1 `sliding_attention` (window 512) : `full_attention` |
| RoPE | **SWA layers only** (global layers are NoPE), partial factor 0.5 → rotary_dim 64, rotate-half pairing (j, j+32), θ=10000 |
| MoE | 256 experts, top-8, fp32 router logits → softmax over all → top-8 → renormalize (`norm_topk_prob`) |
| Expert MLP | `silu(min(gate, 7)) * clip(up, -7, 7)` → down (clamped SwiGLU, trained-in) |
| MoE intermediate | 512 per expert; no shared experts |
| Norms | RMSNorm, eps 1e-6, fp32 internal + fp32 weight multiply, single bf16 rounding |
| Vocab | 151936 (Qwen2 BPE), untied lm_head, EOS 151645 `<|im_end|>` |
| Activations | bf16 with fp32 accumulation; router always fp32 (`router_dtype: fp32`) |

## Checkpoint format (`maple-preview-2bit-mlx`, MLX safetensors)

Converted by `mlx_lm/ternary.py` from the bf16 repo (quantization-aware trained, so
thresholding recovers exact ternary values).

- **Ternary projections** (`self_attn.{q,k,v,o}_proj`, `mlp.switch_mlp.{up,gate,down}_proj`):
  `weight` = `uint32 [.., out, in/16]`, 16 2-bit codes per word, **LSB first**, code = value+1;
  `row_alpha` = `bf16 [.., out]`, one scale per output row. Dequant: `w = alpha * (code - 1)`,
  values ∈ {−α, 0, +α}. Experts are pre-stacked `[256, out, in/16]`.
- **Embeddings + lm_head**: MLX affine 4-bit, group 64 (`weight` uint32 8 codes/word LSB first,
  `scales`/`biases` bf16 per group): `w = q * s + b`.
- **Dense bf16**: `mlp.gate.weight` [256, 2048] (router, fp32 compute), all norm weights,
  `q_norm`/`k_norm` [128].
- `model-flashhead.safetensors` (approximate head) is **not used** by the exact path.

## Scope of the basic implementation (this branch, gfx1151 first)

- Batch-1 text generation, greedy/exact sampling first, `rows=1` decode, token-by-token
  prefill through the same step (exact, slow but simple; a real prefill path is follow-up).
- Contiguous per-layer KV cache: SWA layers keep the last 512 tokens (window mask), global
  layers append up to a configured max context.
- Exact 4-bit lm_head (no FlashHead).
- Weights resident on GPU in checkpoint-native packed layouts; the fused Q/K/V and selected
  expert kernels consume the original split tensors without a host-side repack.

### Landed kernels (shared gfx11 sources, native gfx1151 retarget)

| Family / wrapper | Purpose |
| --- | --- |
| `quant/maple_ternary.py::maple_ternary_gemv_bf16` | y[r] = α_r · Σ x_j·(code−1), fp32 accum, bf16 x/out |
| `quant/maple_ternary.py::maple_ternary_qkv_gemv_bf16` | fused launch over the original split Q/K/V packed tensors |
| `quant/maple_ternary.py::maple_affine4_gemv_f32` | lm_head: fp32 logits from 4-bit affine rows |
| `quant/maple_ternary.py::maple_affine4_embed_bf16` | dequantize one embedding row → bf16 hidden |
| `attention/maple_attention.py::maple_qknorm_rope_kv_write_bf16` | fused per-head RMSNorm(q,k), optional partial RoPE, and KVLiveSpans write |
| `attention/maple_attention.py::maple_attention_decode_bf16` | GQA decode attention over KVLiveSpans; SWA/global behavior comes from span capacity |
| `moe/maple_moe.py::maple_router_topk_bf16` | fp32 router GEMV + softmax + stable top-8 + renorm |
| `quant/maple_ternary.py::maple_selected_ternary_{dual_,}gemv_bf16` | selected-expert gate/up and down projection without expert unpacking |
| `moe/maple_moe.py::maple_clamped_swiglu_bf16` | silu(min(g,7))·clip(u,±7) |
| `moe/maple_moe.py::maple_weighted_residual_bf16` | h ← h + Σ_e w_e·y_e (fp32 combine, one rounding) |
| reused `linear/lm_head.py::argmax_f32` | two-stage greedy argmax over fp32 logits |

Reused existing kernels: `hipengine_paro_rmsnorm_out_bf16` (standard RMSNorm semantics),
`hipengine_paro_add_rmsnorm_out_bf16` (fused residual+norm between sublayers, one bf16
rounding of the sum, matching the reference's residual adds).

## Correctness gates

1. `hipengine/kernels/cpu_reference/maple.py` is the torch-free NumPy oracle for packed
   dequantization and Maple forward math.
2. Primitive gfx1151 fixtures establish BF16-bit exact standard RMSNorm,
   QK-norm+partial-RoPE (including KV write), and clamped SwiGLU. Router top-k IDs are
   exact and its fp32 softmax-renormalized weights are within one ULP of the oracle.
3. `scripts/maple_correctness.py` runs token-serial, teacher-forced decode in separate
   hipEngine and oracle processes. It requires KL ≤ 0.05, top-1 agreement ≥ 90%, and
   independently checks that the device greedy sampler equals exact argmax of copied FP32
   logits.
4. The external oracle is the pinned remote model
   `deepgrove/maple-preview@ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07` under its
   checkpoint-pinned `transformers==4.57.1` with `trust_remote_code=True`. The remote
   model code is unmodified. A scalar-tested pure-Torch shim supplies its hard-required
   FlashAttention API on ROCm.
5. The implementation gate uses matched weights: the remote model's embeddings and head
   are replaced with BF16 dequantization of the exact deployment checkpoint's affine4
   tensors. This avoids conflating runtime correctness with an intentional quantization
   change. The untouched dense-source comparison remains a separately reported quality
   diagnostic; thresholds are never weakened or prompt-conditioned.

### Retained gfx1151 result (18-position formatted chat prompt)

| Oracle | Max KL | Mean KL | Top-1 | Result |
| --- | ---: | ---: | ---: | --- |
| Independent packed-formula Torch | 0.013508 | 0.001679 | 18/18 | pass |
| HF `trust_remote_code`, checkpoint-matched affine4 endpoints | **0.004719** | **0.000723** | **18/18** | pass |
| Untouched dense HF source (quantization-quality diagnostic) | 0.149840 | 0.024479 | 16/18 | fail, retained diagnostic |

The dense-source gap localizes to the deployment checkpoint's affine4 embedding/head:
using dense endpoints with the same packed projection math reduces the diagnostic to max
KL 0.033023 and 17/18 top-1. This is not relabeled as a pass. Full evidence, commands,
and both outcomes are in
`benchmarks/results/2026-08-05-gfx1151-maple-ternary2-correctness.json`.

The 40.4 GB dense checkpoint has nine shards / 18,651 tensors. Verify at least 45 GB free
on the filesystem selected by `--hf-cache-dir`, not merely on `$HOME`. On UMA gfx1151,
`--hf-offload-experts` keeps dense control/attention resident and mmap-loads only routed
experts from the original safetensors; a tiny resident-vs-offloaded model is logit-bit
exact. Reproduction:

```bash
python3 -m venv --system-site-packages /tmp/maple-hf-oracle-venv
/tmp/maple-hf-oracle-venv/bin/python -m pip install 'transformers==4.57.1'

TOKENS='151644,872,198,7985,825,2805,11652,911,54380,12408,13,151645,198,151644,77091,198,151667,198'
PYTHONPATH=$PWD /tmp/maple-hf-oracle-venv/bin/python scripts/maple_correctness.py \
  --model deepgrove/maple-preview-2bit-mlx --backend hip_gfx1151 \
  --token-ids "$TOKENS" --oracle hf \
  --hf-model deepgrove/maple-preview \
  --hf-revision ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07 \
  --hf-cache-dir /path/with/45GB/free --hf-local-files-only \
  --hf-offload-experts --hf-offload-dir /path/to/offload \
  --hf-match-packed-affine4 --json /tmp/maple-hf-correctness.json
```

## Bring-up completion audit

The post-correctness-fix public API smoke constructs
`LLM("deepgrove/maple-preview-2bit-mlx", backend="auto", quant="auto")`, runs
the formatted prompt twice in one resident process, and checks the complete
public route rather than calling `MapleRunner` directly. Both runs produce the
same 37 greedy tokens and coherent completed answer, stop on real EOS 151645,
and leave zero tracked allocations after `LLM.close()`. The measured 4.365 s
cold and 0.703 s resident-repeat walls are diagnostics only, not throughput
claims. Compact evidence is in
`benchmarks/results/2026-08-05-gfx1151-maple-public-e2e-smoke.json`.

| Bring-up deliverable | Concrete evidence | Status |
| --- | --- | --- |
| Official checkpoint identity and exact packed manifest | `hipengine/loading/maple.py`; 463 tensors / 5,308,186,624 bytes; `tests/test_maple_loading.py` | complete |
| Pinned model geometry and attention/MoE semantics | `hipengine/models/maple.py`; `tests/test_maple_model_contract.py` | complete |
| Torch-free model-ID → backend/quant/generator route | `hipengine/generation/maple.py`, `hipengine/quant/maple_ternary.py`; public smoke resolves `hip_gfx1151` / `maple_ternary2` | complete |
| Exact checkpoint-native GPU primitives | Maple quant, attention, and MoE families in `hipengine/kernels/hip_gfx1100/`, native-compiled for gfx1151; CPU-reference fixtures and `rocprofv3` traces in `docs/KERNELS.md` | complete |
| Resident token-serial prefill and decode | `hipengine/runtime/maple.py`; post-fix public smoke exercises prompt prefill, 37 decode tokens, reset/reuse, EOS, and close | complete |
| Numerical implementation gate | packed max KL 0.013508 / 18-of-18 top-1; matched HF max KL 0.004719 / 18-of-18 top-1; exact device argmax | complete |
| Public free-running behavior after the global-span fix | deterministic coherent answer, real EOS, identical resident repeat, allocator returns to zero | complete |
| Documentation, measurements, and atomic history | `WORKLOG.md`, this file, `docs/KERNELS.md`, compact result artifacts, commits `55bc253ff` through `7b82c60bf` plus the completion-audit commit | complete |

The untouched dense-BF16 quality diagnostic remains explicitly failed at max KL
0.149840 / 16-of-18 top-1 because the deployment checkpoint intentionally uses
affine4 embeddings and head. It is not an implementation-correctness failure and
its thresholds were not weakened; see the correctness artifact for attribution.

## Current performance baseline (gfx1151, basic bring-up)

All numbers below are from the accepted public smoke
`benchmarks/results/2026-08-05-gfx1151-maple-public-e2e-smoke.json` plus the
kernel shapes in this tree, on Radeon 8060S / gfx1151 (40 CU, 80 SIMD32,
~256 GB/s, 59.4 TFLOP/s BF16-WMMA, 118.8 TOP/s INT4-WMMA).

| Metric | Value |
| --- | ---: |
| Cold request wall (incl. model load) | 4.365 s |
| Resident repeat wall (18-token prefill + 36 decode) | 0.703 s |
| Effective output rate (incl. prefill) | 52.6 tok/s |
| Avg model-forward latency | ~13.0 ms |
| Inferred token-serial prefill | ~234 ms / 18 tokens |
| Inferred decode rate | ~76.8 tok/s |

**Why decode is slow today — launch-bound, not compute/bandwidth bound.** Each
token forward launches ~271 HIP kernels through Python ctypes (24 layers × 11
per-layer kernels, plus span update, embedding, final norm, affine4 lm_head,
and two-stage argmax). At 76.8 tok/s ≈ 13 ms/token, that is ~48 µs of
host/dispatch cost per launch — the dominant term. The active weight traffic per
token is only ~9.5 MB (ternary packs 0.25 bytes/element), so the pure bandwidth
floor is ~37 µs/token and compute is even lower. Decode is therefore limited by
dispatch/launch overhead, not by the hardware doing the math.

**Why prefill is slow today.** Prefill is token-serial: `runner.prefill()` loops
`step()` over the prompt, so each prompt token pays the same ~13 ms forward with
no weight reuse across rows. A 4K prompt costs ~52 s of model-forward time. The
correct fix is a true batched `[T, hidden]` prefill that reuses weights across
prompt rows and is compute-bound (see `docs/MAPLE-PERF.md`).

## Open follow-ups (not required for the basic bring-up)

- True chunked prefill (batched GEMM + masked attention).
- FlashHead approximate head (`lm_head_flash.*`) as an opt-in fast path.
- Wider batch / server integration, CUDA-peer backend, ternary WMMA prefill kernels.
