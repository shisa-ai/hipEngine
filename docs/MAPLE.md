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
- Weights resident on GPU in checkpoint-native packed layouts (no repack beyond q/k/v row
  concatenation at load).

### Kernel plan (`hipengine/kernels/hip_gfx1100/maple/`, gfx1151 reuses via arch retarget)

| Kernel | Purpose |
| --- | --- |
| `maple_ternary_gemv_bf16` | y[r] = α_r · Σ x_j·(code−1), fp32 accum, bf16 x/out |
| `maple_affine4_gemv_f32` | lm_head: fp32 logits from 4-bit affine rows |
| `maple_affine4_embed_bf16` | dequantize one embedding row → bf16 hidden |
| `maple_qknorm_rope_bf16` | fused per-head RMSNorm(q,k) + partial RoPE (rope_dim 0 for NoPE) |
| `maple_attn_decode_bf16` | GQA decode attention, optional 512 window, online softmax |
| `maple_router_topk_bf16` | fp32 router GEMV + softmax + top-8 + renorm |
| `maple_clamped_swiglu_bf16` | silu(min(g,7))·clip(u,±7) |
| `maple_weighted_residual_bf16` | h ← h + Σ_e w_e·y_e (fp32 combine, one rounding) |
| `maple_argmax_f32` | greedy argmax over fp32 logits |

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

## Open follow-ups (not in the basic slice)

- True chunked prefill (batched GEMM + masked attention).
- FlashHead approximate head (`lm_head_flash.*`) as an opt-in fast path.
- Wider batch / server integration, CUDA-peer backend, ternary WMMA prefill kernels.
