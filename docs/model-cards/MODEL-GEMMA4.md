---
status: current
owns: Gemma 4 model support: architecture contract, format decision, kernel gap audit, and bring-up plan.
---
# MODEL-GEMMA4.md — Gemma 4 MoE on hipEngine

Status: **reviewed; port not started** (2026-09-22). The architecture contract
and format decision below are settled; no hipEngine code exists yet.

Driver checkpoint is `shisa-ai/shisa-de-1`, a full fine-tune of
`google/gemma-4-26B-A4B-it`. Gemma 4 is the architecture to implement; shisa-de-1
is the checkpoint that motivated it. This card covers the text decoder only —
the vision and audio towers are separate `mmproj` artifacts and are out of scope.

## Recommendation

**Ship GGUF, quantized, on the Laguna path.** Concretely:

| Artifact | Size | Target | Role |
| --- | --- | --- | --- |
| BF16 GGUF | ~50.5 GB | neither GPU | Conversion intermediate and llama.cpp oracle source only |
| **Q8_0** | ~26.9 GB | W7900 (44.98 GiB) | **Primary deployment artifact** — near-lossless |
| **Q4_K_M** | ~17 GB | W7900 and 7900 XTX (23.98 GiB) | 24 GB deployment, and the quantization-drift study |

Safetensors is not a deployment path. The BF16 checkpoint is 51.6 GB, which
exceeds the W7900's 44.98 GiB of VRAM, and hipEngine is single-GPU. Its role is
the torch reference oracle.

The reason the format question has a clear answer here is that the two hardest
things about this port — mixed sliding/full attention with two RoPE contracts,
and a 128-expert MoE with an always-on parallel dense branch — are already
implemented in hipEngine for Laguna S 2.1. Gemma 4 is a Laguna-shaped problem
with a different per-layer schedule.

Quantization is not a free choice for this checkpoint specifically. shisa-de-1 is
a *decision model*: its entire output contract is the next-token logit
distribution restricted to option letters. Quantization error is therefore an
accuracy question, not a throughput question, and Q8_0 is worth its size. See
"Quantization sensitivity" below.

## Architecture contract

Everything in this section is read from the pinned `config.json` and confirmed
against the GGUF metadata produced by `convert_hf_to_gguf.py` in this session.

| Field | Value |
| --- | --- |
| HF architectures | `Gemma4ForConditionalGeneration`; text `model_type=gemma4_text` |
| GGUF architecture | `gemma4` |
| Parameters | 25.2B total, 3.8B active; 1013 tensors, 51,612,009,332 bytes |
| Layers | 30, all MoE |
| Attention schedule | 25 `sliding_attention` (window 1024) + 5 `full_attention` at layers 5, 11, 17, 23, 29 |
| Hidden size | 2816 |
| Q heads | 16 everywhere |
| KV heads | 8 on sliding layers, 2 on full layers (per-layer GGUF array) |
| Head dim | 256 on sliding layers, 512 on full layers (per-layer) |
| Dense MLP width | 2112, `gelu_pytorch_tanh`, parallel gate/up |
| Expert count | 128, top-8, `moe_intermediate_size` 704, `gelu_pytorch_tanh` |
| Vocabulary | 262144, tied embeddings |
| Context | 262144 |
| RMS norm eps | 1e-6 |
| Final logit softcap | 30.0 |
| RoPE, sliding | theta 10000, `default`, `n_rot` 256 (full head dim) |
| RoPE, full | theta 1e6, `proportional` with `partial_rotary_factor` 0.25, `n_rot` 512 |
| Attention scaling | **1.0** — Gemma 4 applies no `1/sqrt(head_dim)` |
| Embedding scaling | **`× sqrt(2816)`** applied to token embeddings |
| KV sharing | none (`num_kv_shared_layers: 0`) |
| Per-layer input | disabled (`hidden_size_per_layer_input: 0`) |
| Vision / audio | present in the checkpoint, separate `mmproj` GGUF, out of scope |

### Forward pass

The reference implementation is `src/models/gemma4.cpp` in upstream llama.cpp.
Per layer, in order:

1. `x_attn = RMSNorm(x, attn_norm)`
2. `Q = Wq @ x_attn`, reshaped to `(16, head_dim)`
3. `Q = RMSNorm(Q, q_norm)` — per head, learned weight
4. `Q = RoPE(Q, n_rot, freq_base, freq_factors)`
5. `K = Wk @ x_attn`
6. **`V = Wv @ x_attn` if `v_proj` exists, else `V = K`** — 25 of 30 layers have `v_proj`
7. `K = RMSNorm(K, k_norm)` — per head, learned weight
8. **`V = RMSNorm(V)`** — per head, *no learned weight*
9. `K = RoPE(K, ...)`; V is not rotated
10. `attn = softmax(Q Kᵀ × 1.0) V`, masked to a 1024-token window on sliding layers
11. `cur = Wo @ attn`, then `cur = RMSNorm(cur, post_attention_norm)`
12. `attn_out = cur + x`
13. Dense branch: `RMSNorm(attn_out, ffn_norm)` → GELU parallel MLP (2112) → `RMSNorm(·, post_ffw_norm_1)`
14. Router: `RMSNorm(attn_out)` → `× 1/sqrt(2816)` → `× ffn_gate_inp.scale` → `ffn_gate_inp` → softmax over top-8 of 128, renormalized
15. Expert branch: `RMSNorm(attn_out, pre_ffw_norm_2)` → 8 selected experts, GELU, `down_proj` output `× ffn_down_exps.scale[e]` → `RMSNorm(·, post_ffw_norm_2)`
16. `cur = dense_branch + expert_branch`
17. `cur = RMSNorm(cur, post_feedforward_layernorm)`
18. `cur = cur + attn_out`
19. **`cur = cur × layer_scalar`**

Then: `RMSNorm(cur, norm)` → tied embedding matmul → **`30 × tanh(logits / 30)`**.

Three details are easy to get wrong and each breaks the readout silently:

- The router reads `attn_out` through its **own** RMSNorm, not the normalized
  input the experts receive.
- `V` is normalized without a learned weight, and only `K` is rotated. Treating
  K and V as one tensor through RoPE gives wrong logits.
- The final softcap is part of the model, not a sampling detail. Omitting it
  changes every reported logprob.

### MoE tensor layout

The checkpoint stores experts as two fused 3-D tensors per layer, not per-expert
tensors:

```
blk.N.ffn_gate_up_exps.weight   {2816, 1408, 128}   gate and up, fused
blk.N.ffn_down_exps.weight      {704,  2816, 128}
blk.N.ffn_down_exps.scale       {128}               per-expert output scale
blk.N.ffn_gate_inp.weight       {2816, 128}         router
blk.N.ffn_gate_inp.scale        {2816}              per-hidden router input scale
```

hipEngine's existing expert sidecar slots are `ffn_gate_exps`, `ffn_up_exps`,
`ffn_down_exps` — three separate rank-3 tensors. Gemma 4's fused `gate_up` is a
fourth layout. Either add a `ffn_gate_up_exps` slot or de-interleave to gate/up
at load; this is a concrete decision to make before writing the loader.

## Why GGUF

1. **Safetensors does not fit.** 51.6 GB against 44.98 GiB of VRAM on the
   W7900, with one GPU supported. There is no expert-offload path that makes
   this a deployment artifact.
2. **GGUF is hipEngine's deepest path.** Q4_K/Q8_0 kernels carry the T16, x8,
   and QMicro repack variants, the MoE `group_scatter` and MMQ prefill work, and
   the expert pack8 sidecar. That is where the tuning already is.
3. **GGUF buys an independent oracle.** llama.cpp has a reference `gemma4`
   implementation. Converting gives a second implementation to diff against,
   which is the strongest available check for a brand-new architecture.
4. **Conversion drops the vision tower.** The text GGUF is the decoder alone;
   vision and audio go to a separate `mmproj` file we do not need.
5. **Quant choice becomes a knob.** With the readout-is-logits constraint, being
   able to trade size for fidelity is the point.

### Quantization sensitivity

The serving contract reads `logprobs` at the answer position and restricts to the
option letters. On the model card's worked example the margin between the chosen
letter and the runner-up is about 6 nats (`A = -0.0056`, `B = -6.1306`), so a
Q4_K_M argmax flip is unlikely on easy cases. Two things are more exposed:

- **`noul` probabilities.** These are a softmax over two rows and are reported as
  confidence. The model card already fits a temperature (`T_noul = 1.69`) to make
  them usable, and quantization noise perturbs the same quantity.
- **Close calls.** The published option-order study shows the model already
  shifts answers with presentation (15 points on an 11-option probe). Margins
  that small are where quant noise can matter.

So: measure the letter-logprob delta between BF16, Q8_0, and Q4_K_M on a fixed
prompt set before picking a default. Q8_0 is the safe default; Q4_K_M is the
24 GB option and should be validated rather than assumed.

## Kernel gap audit

Measured by reading the tree, not by running it. "Reuse" means the primitive
exists and is shaped for this use; it does not mean it has been exercised on
Gemma 4 geometry.

### Reuse

| Gemma 4 need | Existing asset |
| --- | --- |
| Mixed sliding + full attention | Laguna `sliding_attention_decode` / `full_attention_decode` |
| Sliding-window mask | `sliding_window` parameter on the Laguna attention kernels |
| Token-granular sliding KV | `KVLiveSpans.sliding_ring` mode, bf16 storage |
| Two RoPE contracts per layer | Laguna `rope.freq_base` / `rope.freq_base_swa` |
| Attention scale as a value | attention kernels take `float scale`; pass 1.0 |
| QK-norm | Laguna `attn_q_norm` / `attn_k_norm`, `qwen35_head_rmsnorm_*` |
| Router + always-on parallel branch | Laguna `laguna_sigmoid_router_topk` + shared expert + combine |
| Top-8 of 128 expert routing | MoE router kernels take `num_experts` / `top_k` |
| 3-D stacked expert tensors | `ffn_*_exps` rank-3 loading and pack8 sidecar |
| Q4_K_M / Q8_0 storage and repack | `GGUF_Q4_K`, `GGUF_Q8_0`, T16 and x8 variants |
| Tied embeddings | `qwen35_gguf` lm_head/token_embd aliasing |
| Top-k logprob readout | `native_sampler` `temperature_top_logprobs_rows_i32` |

### New work

| Gap | Notes |
| --- | --- |
| **`K = V` on 5 full-attention layers** | No precedent. `v_proj` is absent; V is K after the K projection and before k_norm/RoPE |
| **Weightless RMSNorm on V** | Existing norms all apply a learned weight |
| **GELU-tanh MLP and expert activation** | Main path is SiLU; GELU exists only in `evie`/vision kernels, f32 |
| **Per-layer head_dim and KV head count** | 256/8 sliding, 512/2 full. All current models are uniform |
| **`layer_scalar`** | Per-layer output multiply |
| **Fused `ffn_gate_up_exps`** | New rank-3 expert layout |
| **`ffn_gate_inp.scale` and `ffn_down_exps.scale`** | Per-hidden router input scale and per-expert output scale |
| **Embedding `× sqrt(n_embd)`** | Gemma-family trait |
| **Proportional RoPE** | llama.cpp emulates it with a `freq_factors` tensor (1.0 for 64 dims, 1e30 for 192). Implementing proportional directly avoids carrying that tensor |
| **Softmax MoE gating with renormalization** | Laguna's router is sigmoid; the softmax path exists for Qwen3.5 |
| **`gemma4` tokenizer** | New GGUF tokenizer model with its own pre-tokenization. `hipengine/tokenization/gguf.py` currently accepts only `gpt2` with `qwen35`/`laguna` pre-tokenizers |
| **GPU final-logit softcap** | The softcap math exists in CPU reference only |
| **7 RMSNorms per layer** | Unusually norm-dense FFN region; scratch sizing must account for it |

The last row is a sizing trap rather than a math problem: the FFN region applies
five separate RMSNorms plus two in attention, against two for a Qwen block.

## Bring-up ladder

Follow the Surya precedent — CPU reference first, GPU after, with the oracle
independent of the implementation under test.

1. **Torch oracle.** Install `transformers >= 5.16.1` (5.17.0 is current; the
   local maximum is 5.13.0, which predates `gemma4`). Load the BF16 checkpoint
   and reproduce the model card's worked example: the 161-token scaffold prompt
   must give `A = -0.0056`, `B = -6.1306`, `C = -7.5056`, `D = -7.3806`. This
   single check validates softcap, dual RoPE, `K = V`, V-norm, the MoE norms,
   `layer_scalar`, and embedding scaling at once. Do this before writing any
   hipEngine code.
2. **GGUF conversion and cross-check.** Convert with llama.cpp's `conversion/gemma.py`
   and confirm the llama.cpp logprobs match the torch oracle on the same prompt.
   Two independent implementations agreeing is what makes the contract safe to
   port.
3. **Quantize and measure drift.** Q8_0 and Q4_K_M, then re-read the letter
   logprobs on a fixed prompt set. This produces the number that decides the
   default artifact.
4. **hipEngine CPU reference.** Torch-free, with a tiny deterministic fixture
   and a per-layer oracle, matching the `cpu_reference/` pattern.
5. **HIP kernels.** Attention first (K=V, per-layer geometry, window), then the
   MoE block, then the softcap and logprob readout.
6. **Public surface.** `LLM.generate()` with `top_logprobs`, and the
   `v1/completions` `logprobs` route the model card's serving contract uses.

Steps 1–3 are cheap and answer most of the open questions. Step 5 is the bulk
of the work.

## Open questions

- **`n_rot = 512` on full layers.** The GGUF declares a full-width rotary
  dimension with a `freq_factors` tensor that neutralizes 384 of the 512 dims.
  Whether hipEngine should carry that tensor or implement proportional RoPE
  directly is unresolved; the direct implementation looks cleaner.
- **Window vs. ring capacity.** Gemma 4's window is 1024 tokens. Laguna's
  `sliding_ring` requires `max_live_count == capacity` and bf16 storage, so the
  ring geometry needs checking against a 1024 window.
- **`K = V` and KV cache layout.** On full layers K and V are equal before
  k_norm and RoPE, then diverge. Whether to store one tensor and derive the other,
  or store both, is a layout decision that affects cache size on 5 of 30 layers.
- **Softcap placement.** Applying it inside the logits kernel versus as a
  separate pass changes the logprob readout path; the sampling kernels currently
  assume they receive final logits.

## References

- Checkpoint: `shisa-ai/shisa-de-1` (Apache-2.0), full fine-tune of `google/gemma-4-26B-A4B-it`
- Reference implementation: llama.cpp `src/models/gemma4.cpp`
- Converter: llama.cpp `conversion/gemma.py`, `Gemma4Model`
- Closest existing hipEngine port: [Laguna campaign](../campaigns/LAGUNA.md)
- Work items already tracked in `TODO.md` for the Gemma 4 plugin
