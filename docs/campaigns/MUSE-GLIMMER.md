---
status: current
owns: Muse Glimmer 30B gfx1151 bring-up campaign: text/vision/DFlash implementation plan, correctness gates, and the pre-registered llama.cpp prefill+decode win condition.
---
# Muse Glimmer 30B implementation campaign (gfx1151)

Last updated: 2026-11-07

Status: **P0 baseline retained; P1 text, P2 vision, P3 DFlash bring-up landed
with correctness gates green. Gate B (DFlash) PASSED: hipEngine 21.6 tok/s vs
llama.cpp draft-dflash 21.30 on the mtpbench suite, exact greedy IDs == true AR
(rolled up). Gate A (AR perf) IN PROGRESS: prefill 333/333/300 vs 394/386/376
and decode 11.13/11.03/10.83 vs 12.24/12.17/12.05 at 512/1K/4K. The WMMA GQA
flash-prefill attention (2.1-2.9x over scalar) and prefill chunk 1024 lifted
prefill from the 320/300/240 checkpoint; the Q4/Q6 projection kernels (~35% of
bf16 peak) are the remaining prefill gap and decode is memory-bound GEMV near
roofline.**
Lead sequencing (2026-11-06): verify vision end-to-end after the perf changes,
then capture the llama.cpp `draft-dflash` mtpbench baseline and make hipEngine
DFlash beat it (Gate B, done), then return to the Gate A AR kernel campaign
(in progress). Gate A is not yet passed.

Lead directive (2026-11-06): implement full Muse Glimmer inference in hipEngine —
text, vision stack, and MTP/DFlash — with an initial correctness pass, and do not
call the campaign done until hipEngine beats llama.cpp in **both prefill and
decode speed** on this host. Work happens on branch `muse-glimmer`.

This document is the campaign authority for Glimmer bring-up. Architecture-wide
rules stay in [`PLAN.md`](../PLAN.md); kernel/evidence rules in
[`OPTIMIZATION.md`](../OPTIMIZATION.md); numerical contracts in
[`EXECUTION-PROFILES.md`](../EXECUTION-PROFILES.md); benchmark protocols in
[`BENCHMARK.md`](../BENCHMARK.md). The first hardware lane is the local
Radeon 8060S / `gfx1151` host (Ryzen AI MAX+ PRO 395, 128 GiB unified memory).
No gfx1100/W7900 result may be inferred from this host.

## 1. Artifacts

All three artifacts are local under `/models/gguf/Muse-Glimmer-30B-GGUF/`:

| Artifact | Path | Size | Role |
| --- | --- | ---: | --- |
| Target | `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` | 16.76 GB | 52-layer dense text model, Q4_K_M (file_type 15) |
| Vision | `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` | 1.40 GB | CLIP-style vision tower + projector, Q4_K_M |
| Draft | `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` | 1.63 GB | 5-layer DFlash block drafter, Q4_K_M |

Upstream identity: HF `meta-models/Muse-Glimmer-30B`, architecture
`MuseGlimmerForConditionalGeneration`; drafter
`MuseGlimmerAssistantModel` (`meta-models/Muse-Glimmer-30B-assistant`).
Reference implementation: `~/llama.cpp/llama.cpp-hip` (read-only peer),
`src/models/muse-glimmer.cpp`, `conversion/muse_glimmer.py`,
`tools/mtmd/clip.cpp` (`clip_graph_muse_glimmer`). The existing launcher
`~/llama.cpp/run-glimmer-server.py` documents the production llama.cpp
configuration: `-ngl 99 -c 8192 -np 1 --mmproj … --spec-type draft-dflash
--spec-draft-model …`.

Two resident llama.cpp Glimmer servers (ports 18080 and 8000) were stopped with
lead approval on 2026-11-06 to free the device for clean measurement.

## 2. Architecture contract

Decoded from GGUF metadata (`hipengine.loading.gguf.scan_gguf`) and the
llama.cpp reference graph. These values are the implementation contract; the
loader must read them from the artifact, not hardcode them.

### 2.1 Text target (`muse-glimmer`)

| Parameter | Value |
| --- | --- |
| Layers | 52 |
| `n_embd` | 6656 |
| `n_ff` | 19968 (dense SwiGLU, 3× expansion) |
| Heads / KV heads | 32 / 2 (GQA 16:1) |
| `head_dim` (K and V) | 128 (Q/O projection width 4096 ≠ `n_embd`) |
| Context | 131072 |
| RoPE base | 500000 (no separate SWA base key present) |
| Sliding window | 2048 |
| SWA pattern | `[T,T,T,F] × 13` — 39 SWA layers, 13 full layers |
| Positional encoding | **RoPE on SWA layers only; NoPE on full layers** |
| RoPE layout | NORM (interleaved); Q/K un-permuted at conversion |
| Pre-norm RMS eps | 1e-5, `weight+1` folded at conversion |
| Post-norm RMS eps | **1e-8**, plain weight multiply (attn and FFN) |
| QK-norm | Per-head RMS; `q_norm.weight = qk_scale_factor` (synthesized), `k_norm.weight = 1` |
| Attention gate | `sigmoid(attn_gate @ pre_attn_normed) * attn_out`, before `o_proj` |
| Embedding | token lookup followed by **weightless RMSNorm** |
| Logits | `lm_head`, then `× logit_scale (0.1961161345243454)`, then tanh softcap at 20.0 |
| Vocab | 202048, GPT-2 BPE, pre-tokenizer `llama4` |
| Special tokens | BOS 200000, EOS 200001, EOT 200008, pad 200018; `add_bos_token=true` |
| Layer-residual taps | per-layer inputs exposed for DFlash (`LLM_KV_TARGET_LAYERS`) |

Per-layer dataflow (llama.cpp `muse-glimmer.cpp` graph):

```text
inpL ─ attn_norm(+1 folded, eps 1e-5) ─ QKV proj ─ QK-norm ─ [RoPE if SWA]
     ─ SDPA (kq_scale = 1/sqrt(128)) ─ × sigmoid(gate_proj(attn_normed))
     ─ o_proj ─ RMSNorm(eps 1e-8) × attn_post_norm ─ + residual
     ─ ffn_norm(+1 folded) ─ SwiGLU(gate,up,down) ─ RMSNorm(eps 1e-8)
     × ffn_post_norm ─ + residual
final: output_norm(+1 folded) ─ lm_head ─ × logit_scale ─ softcap(20)
```

### 2.2 Vision (`clip` mmproj, projector type `muse-glimmer`)

| Parameter | Value |
| --- | --- |
| Tower | 50 layers, `n_embd` 1536, `n_ff` 8960, 16 heads |
| Patch size | 14; nominal image 896 (`pos_emb_height × patch × merge`) |
| Dynamic resolution | yes — nominal size is only an initialization hint |
| Spatial merge | 2 (2×2 patch merge before projector) |
| Projector | 3-layer MLP (`vision_adapter.fc1/fc2`, `vision_projection`) → 6656 |
| Normalization | mean/std 0.5; LayerNorm eps 1e-5 |
| Attention pattern | 3 sparse (window) + 1 global repeating (`muse_glimmer_sparse_factor=4`) |
| Temporal | `patch_temporal=2`; conversion sums the pt slabs into one conv2d kernel |
| Position embeddings | interpolated to the dynamic grid (rpos_w/rpos_h), spatial permutations for merge/downsample |

The reference graph builder is `clip_graph_muse_glimmer` in
`tools/mtmd/clip.cpp`; image-token placement and the chat-side media markers
come from the mtmd muse-glimmer handler and the ATEM Jinja template
(`models/templates/muse-glimmer.jinja`, also embedded as
`tokenizer.chat_template` in the target GGUF).

### 2.3 DFlash draft (`dflash`)

| Parameter | Value |
| --- | --- |
| Layers | 5, same `n_embd` 6656 / `n_ff` 19968 |
| Heads / KV heads | 32 / 8 |
| `head_dim` | 128 |
| Block size | 16 (masked block diffusion drafting) |
| Target layers | `[2, 14, 26, 38, 50]` — **inputs** to those target layers (outputs of layers 1/13/25/37/49) |
| Attention | all 5 layers SWA 2048 |
| RoPE layout | **NEOX (rotate_half)** — the draft keeps HF layout; target is NORM |
| Mask token | 201818 |
| Vocab/tokenizer | shared with target (same 202048-row GPT-2 BPE) |

hipEngine already ships a DFlash path for Laguna
(`hipengine/generation/laguna_dflash.py`, `kernels/*/speculative/dflash*`,
`kernels/cpu_reference/dflash2.py`); the block-draft/verify/commit scaffolding
and speculative-provider registry are reuse candidates, not rewrites.

## 3. Reuse map and new work

### 3.1 P1 runner design (decided 2026-11-06)

New `MuseGlimmerResidentSession` (runtime) composed from registered kernels;
no Qwen/Laguna/Gemma runner branches. Primitive map verified in-tree:

| Step | Kernel surface | Notes |
| --- | --- | --- |
| Projections (Q4_K/Q6_K/Q5_K) | `runtime/gguf_linear.launch_gguf_linear`, `LAYOUT_RAW_GGUF` | raw quant bytes uploaded as-is; dispatch resolves GEMV decode / prefill owners per quant |
| Embedding | raw Q4_K `embedding/lookup_bf16_out` consumer contract | `resolve_embedding_consumer_contract` |
| Pre/post RMSNorm | `gemma4_rmsnorm_f32w_bf16` (plain weight, eps arg) | Gemma4 family applies weight as-is — required, since Glimmer weights are +1-folded; Qwen gguf_ops norms apply (1+w) and must NOT be used |
| Embedding norm | `gemma4_rmsnorm_weightless_bf16` | eps 1e-5 |
| QK-norm + RoPE | `gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight` (NEOX) | **load-time re-permutation**: invert the conversion's `_unpermute_for_rope` on Q/K projection rows so existing NEOX rotary equals ggml NORM interleaved rope; dot products invariant under consistent Q/K permutation; unit-test against CPU reference interleaved rope |
| NoPE full layers | skip the rotary step | config-driven from `swa_pattern` |
| Attention | gemma4 attention family (bf16 prefill block + decode, keep-mask) | generic head_dim; SWA via keep mask `key > query - window`; decode split/flash routes for long full layers; GQA 32:2 |
| Sigmoid attn gate | `gguf_gate_mul_bf16` | `out = attn * sigmoid(gate)`; gate projected from pre-attn normed hidden |
| SwiGLU | `silu_mul_separate_out_bf16` (fused/paro_silu) | gate/up parallel, then down |
| Residual adds | `gguf_bf16_add` / gemma4 add fusion | post-norm eps 1e-8 then add |
| Head | `launch_gguf_linear` (Q5_K) → scale → `gemma4_logit_softcap_f32` | scale 0.19611613 then cap 20 |
| KV | contiguous per-layer f16 bring-up cache | migrate to KVLiveSpans paged path in P4 if the serving/DMS surface requires it; record as campaign decision, not a silent shortcut |

Bring-up is eager (no graph replay); performance phase adds the production
owners. Weight re-permutation is a load-time layout transform, not arithmetic.

### 3.2 P2 vision design (decided 2026-11-06)

Graph decoded from `tools/mtmd/models/muse-glimmer.cpp` + the
`PROJECTOR_TYPE_MUSE_GLIMMER` host branch in `clip.cpp`:

1. Preprocess: `muse_glimmer_grid_size` in 28px (patch×merge) units, aspect-
   preserving floor/ceil candidate pick, cap 4096 post-merge tokens; Lanczos
   stretch resize (no padding); normalize (p/255 − 0.5)/0.5.
2. conv2d patchify 14×14 stride 14, no bias; raster patch order.
3. Learned 32×32 position table bilinear-resized (half-pixel, align_corners
   False, clamped) to the patch grid; added.
4. 32×32-window grouping permutation (`sp_perm`); 1-indexed width/height RoPE
   positions travel in permuted order.
5. pre-LN (LayerNorm eps 1e-5, weight+bias) → 50 ViT layers: ln1 → Q/K/V
   (1536², bias) → 2D **interleaved** RoPE on Q/K (dims [0,48) width,
   [48,96) height; 24 pairs/half; base 10000) → bidirectional attention
   (scale 1/√96; sparse layers block-diagonal window mask, every 4th + last
   global) → out proj+bias → residual; ln2 → up (8960) + bias → **erf** GELU →
   down+bias → residual. post_ln.
6. Inverse permutation; 2×2 pixel-shuffle, channel-outer concat → [n_out, 6144].
7. Adapter mm.0 (6144→4096) → erf GELU → mm.1 (4096→4096) → erf GELU →
   mm.2 (4096→6656); BF16 weights, no biases.

Token placement: `<|image_start|>` 200080 + n_out placeholder rows +
`<|image_end|>` 200081; placeholder id `<|image|>` 200090; ATEM template
marker `<|patch|>` 200092. Oracle prompt chunks verified:
[BOS, "Describe this image: ", 200080] + n_out + [200081].

**Precision finding:** on RDNA3 ggml routes Q4_K projections to the MMQ
integer kernel for *all* batch sizes (int8-quantized activations — changed
arithmetic vs the f32 architecture), and dequantizes other quants to F16 for
cuBLAS by default. `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` fixes the cuBLAS path
but cannot reach MMQ, so GPU-config embeddings diverge from any f32
implementation far beyond rounding (~1%/Q4_K layer, compounding through the
amplifying residual stream to ~60% by layer 49). The **semantic oracle is the
llama.cpp CPU backend** (`GLIMMER_CPU=1` in the capture tool): dequantize →
f32 vec_dot. The hipEngine tower computes f32 (rocBLAS sgemm + FP32
primitives, the gemma4_vision pattern) and gates against that oracle.
Cross-engine generation sanity against the GPU server is a *sanity* check
(consistent descriptions), not an embedding-parity gate.

Implementation: CPU reference first (oracle-gated), then GPU runner, then
generator `generate_multimodal_detailed` with embedding-row overrides at the
placeholder spans (the gemma4 `image_input` pattern). Overrides enter before
the weightless embedding norm — llama.cpp's external-embedding path feeds the
graph the same way (the norm applies to image rows too).

**Amplification finding (gate design):** stage-by-stage bisection against the
oracle's graph nodes (GPU-runner injection hook) shows every stage matches to
f32 rounding; injecting the oracle's `pre_ln` makes the full downstream tower
agree to 0.004. The residual upstream difference is zero-mean f32 noise
(std ~6e-4), and this network amplifies it ~10^4x on smooth inputs: injecting
random noise of the same std into the oracle's own pre_ln reproduces the
observed output spread (2.9–14.9 vs our 14.4 on the 896² gradient; the 448²
and 672×448 fixtures stay within 0.023/0.125). Per-element parity is therefore
not the semantic gate; the live tiers gate on cosine ≥ 0.998, mean/std drift
≤ 0.01, and a catastrophic max-err bound. Image-conditioned generation sanity
vs llama.cpp is the product-level check.

### 3.3 Subsystem map

| Need | Existing surface | New work |
| --- | --- | --- |
| GGUF container/metadata | `hipengine/loading/gguf.py` | tensor-map for `muse-glimmer` post-norm/gate names |
| Q4_K_M dense linear | `hipengine/quant/gguf_q4_k.py`, `runtime/gguf_linear.py`, gfx1151 row48/row64/row128 prefill owners + GEMV decode | shape admission for 6656/4096/19968 widths |
| Tokenizer | `hipengine/tokenization/gguf.py` (per-model `pre` variants) | `llama4` pre-tokenizer variant + Glimmer special tokens |
| Attention | `KVLiveSpans` paged attention, Laguna SWA variants | NoPE full layers (no rotary), SWA 2048 pattern, GQA 32:2 head_dim 128 |
| Norm/gate primitives | `norm/rmsnorm`, `head_rmsnorm`, Laguna `attention_gate` (softplus) | weightless embedding RMSNorm, eps-1e-8 post-norm, **sigmoid** gate variant, logit scale+softcap |
| Rotary | `rotary/` (NORM and NEOX consumers exist across models) | per-layer RoPE/NoPE selection from `sliding_window_pattern` |
| Model plugin + generator | `models/registry.py`, `generation/registry.py`, `LLM._construct_text_generator` | `MuseGlimmerGGUFModel` plugin + registered generator/resident runner |
| Vision | `runtime/gemma4_vision.py` pattern (FP32 primitives + rocBLAS), `kernels/*/vision/` | Glimmer tower graph: dynamic grid, 3:1 window/global attention, merge permutations, 3-layer MLP |
| DFlash | `generation/laguna_dflash.py`, speculative provider registry, `dflash_*` kernels | Glimmer draft loader (NEOX rope), residual taps at [2,14,26,38,50], block-16 cycle |
| Serving | `hipengine serve`, OpenAI surface | chat template (ATEM), media handling, provider wiring |

No identity gating: admission is by declared capability (architecture string,
quant, shapes), never by artifact name/hash. New fused composites get registered
strict unfused fallbacks. No `import torch` on any runtime path.

## 4. Phases and acceptance

Bring-up follows the implementation-stage discipline: full-model tests stay at
512/1K until the basic text AR + DFlash + serving + vision gates pass; context
escalation beyond 4K is rate-gated later, mirroring
[`BENCHMARK.md`](../BENCHMARK.md) "Implementation-stage context escalation".

### P0 — llama.cpp baseline (before any hipEngine perf claim) — DONE

Retained baseline (clean worktree @ `44498df1a`, artifact
`benchmarks/results/2026-11-06-gfx1151-muse-glimmer-llamacpp-hip-baseline.json`):
llama.cpp HIP build 12005 `e0f142d5d`, `-ngl 99 -fa 1 -ctk f16 -ctv f16`,
5 repetitions, GTT peak sampling, Radeon 8060S / gfx1151.

| Workload | Prefill tok/s | Decode tok/s | Peak GTT Δ GiB |
| --- | ---: | ---: | ---: |
| 512/128 | 394.404 | 12.240 | 15.868 |
| 1K/128 | 385.784 | 12.167 | 15.895 |
| 4K/128 | 376.332 | 12.047 | 15.984 |

These are the Gate A targets. A first same-day run reproduced within ~0.5%
(396.5/12.25 at 512) before the clean rerun; it is kept only as a dirty-
provenance diagnostic outside the tree.

- `llama-bench` AR (no draft) via `scripts/llamacpp_bench_with_peak.py`,
  HIP build `~/llama.cpp/llama.cpp-hip/build/bin/llama-bench`,
  `-ngl 99 -fa 1 -ctk f16 -ctv f16`, workloads **512/128, 1K/128, 4K/128**,
  5 repetitions, GTT memory domain.
- Server diagnostic with `--spec-type draft-dflash` + mmproj at the production
  configuration (natural prompts, `temperature=0`): pending, recorded as an
  external diagnostic per `BENCHMARK.md` cross-engine rules when P3 lands.
- Compact JSON artifact under `benchmarks/results/` with full provenance
  (host, build, commit, commands, samples). This is the pre-registered baseline.

### P1 — Text bring-up + initial correctness

1. Model plugin `muse_glimmer_gguf` registered for architectures
   `("muse-glimmer", "MuseGlimmerForConditionalGeneration")`.
2. Loader/tensor map; CPU-reference layer oracle (NumPy) for one SWA layer, one
   NoPE full layer, and the final head (scale+softcap), RED-first.
3. gfx1151 resident runner composed from registered kernels; strict unfused
   chain for every new fused composite.
4. Tokenizer `llama4` variant; round-trip fixture against llama.cpp
   `llama-tokenize` output.
5. **Initial correctness gate:** greedy generated IDs and final-logit agreement
   vs llama.cpp on committed fixture prompts (512/1K), finite logits, and the
   outer kernel floor (KL ≤ 0.05, top-1 ≥ 90% vs `cpu_reference`) for new
   kernels. Exact llama.cpp ID equality is the bring-up target; any divergence
   is localized layer-by-layer against the CPU oracle before proceeding.
6. Exercised through `hipengine.LLM.generate()` (not only a harness), with the
   selected variant/fallback reported.

### P2 — Vision stack

1. mmproj loader + tower graph (dynamic grid, 3:1 window/global attention,
   position interpolation, spatial merge permutations, 3-layer MLP projector).
2. Image preprocessing parity (mean/std 0.5, patch 14, pt=2 slab sum) and
   image-token placement per the ATEM template/mtmd handler.
3. Correctness: projected embedding rows vs llama.cpp mtmd on fixture images
   (tolerance declared before measuring), then image-conditioned greedy output
   sanity vs llama.cpp on ≥3 fixture images at two resolutions (one non-square).
4. Reached through the public multimodal surface (`LLM.generate_multimodal*` /
   `hipengine serve`), not a test-only route.

### P3 — DFlash speculative decode

Design record (2026-11-06): the draft is a 5-layer pre-norm decoder (no gate,
no post-norms; raw HF norm weights applied as-is — the draft conversion folds
nothing) with NEOX RoPE (HF layout kept). Dual mode per `dflash.cpp`:
injection (target taps [2,14,26,38,50] concat → fc → enc norm → per-layer K/V
→ K head-norm+rope → draft KV) and token mode ([anchor, MASK×15] through the
shared target embd, **non-causal** attention over [context + block], shared
target lm_head; mask-row logits are the draft). The driver reads block rows
1..15; verification is target-corrected greedy with contiguous-KV truncate
rollback. On partial acceptance the verify row at the mismatch processed the
rejected draft token — its KV/logits are invalid for the committed sequence,
so the corrected token is reprocessed fresh (truncate to the accepted prefix,
decode the correction, inject its taps). **Batch-invariance contract:** the
bring-up verify runs the single-row decode path per token because multi-row
projection dispatch is not batch-invariant (bf16 rounding flips near-tie
argmaxes; measured ≤0.05 logit shift). Exactness requires verify-written KV to
be bit-identical to AR. The batched-invariant block verify (row-independent
GEMV + decode-equivalent attention) is P4 work — until it lands, block verify
costs k forwards and DFlash cannot beat AR on speed. Oracle tooling must call
`llama_set_causal_attn(ctx_dft, false)` —
without it the mask rows diverge while the anchor row matches. Draft logit
parity vs llama.cpp is distributional (MMQ int8 drift, as with text/vision);
the binding gate is exact greedy ID equality vs true AR.

1. Draft loader (NEOX rope, KV heads 8, mask token), target residual capture at
   layers [2,14,26,38,50], block-16 draft/verify/commit through the speculative
   provider registry.
2. Correctness: exact greedy ID equality vs true AR on the ten-prompt public
   gate pattern used by Laguna DFlash, then the full multi-prompt mtp-bench
   category suite (`benchmarks/prompts/mtpbench-code-general-ja.jsonl`).
3. Economics: true no-MTP AR baseline from the same protocol (never a
   verifier-derived `off` row); acceptance/speed reported full/train/heldout.
4. Ship-on rule: a working DFlash path lands enabled on the default path unless
   a named concrete cause (observed failure/regression with a clearing command)
   is recorded in `docs/REFACTOR.md`.

### P4 — Performance: beat llama.cpp (win condition)

Profile-driven (`rocprofv3` kernel-trace, prebuilt caches) optimization of the
shipping path. Pre-registered gates, same host + same GGUF + same shapes:

- **Gate A (binding, AR):** hipEngine prefill **and** decode tok/s exceed the
  P0 `llama-bench` HIP baseline at **each** of 512/128, 1K/128, 4K/128.
  hipEngine rows run through the hermetic gfx1151 wrapper discipline
  (`BENCHMARK.md`), warmup + ≥3 measured repetitions, median synchronized wall,
  clean-worktree provenance, correctness gate attached.
- **Gate B (product, speculative):** hipEngine with DFlash enabled beats the
  llama.cpp server `draft-dflash` decode rate on the same natural-prompt suite,
  using the cross-engine decode timing boundary (`BENCHMARK.md`); prefill is
  covered by Gate A. **PASSED (2026-11-06):** hipEngine DFlash **21.6 tok/s**
  (budget 4, 3 repeats, per-repeat median 21.48) vs llama.cpp **21.30** on the
  ten-prompt mtpbench suite, exact greedy IDs == true AR, 1.98x AR. The
  batched-invariant block verify (Q4/Q5/Q6 parent-parity rowtile projections,
  bit-identical KV + decode-equivalent attention) and the folded cycle
  (corrected boundary token processed inside the next verify, no separate
  forward) took DFlash from 2.59 -> 21.6 tok/s. Empirical budget sweep peaks
  at 4 (shipping default). Artifact
  `benchmarks/results/2026-11-06-gfx1151-muse-glimmer-hipengine-dflash-mtpbench.json`.
- Every retained win updates `benchmarks/README.md`, `benchmarks/CHANGELOG.md`,
  and a `benchmarks/results/` artifact; arithmetic-changing defaults pass the
  `EXECUTION-PROFILES.md` production gate.
- 16K/256 and deeper shapes are stretch rows after Gates A/B, rate-gated per
  the escalation ladder.

### P5 — Serving closure and rollup

`hipengine serve` with the ATEM chat template, vision inputs, and DFlash on the
default path; one real request per surface confirming the intended route ran;
docs (`MODELS.md`, model card, `KERNELS.md` catalog rows), worklog entries, and
benchmark rollup complete. Campaign closes only when Gates A and B pass.

## 5. Risks and open details

- **NoPE full layers:** positional information flows only through the 39 SWA
  layers; the KV cache/position plumbing must not apply rotary on full layers.
- **RoPE layout split:** target NORM vs draft NEOX. A layout mismatch produces
  plausible-looking but wrong logits; the CPU oracle must pin both.
- **Post-norm eps 1e-8** differs from pre-norm 1e-5; weightless embedding norm
  and the scale→softcap order are additional exact-match traps.
- **Vision dynamic grid:** window/global attention pattern and merge
  permutations must match `clip_graph_muse_glimmer` at non-nominal resolutions;
  fixture images must include a non-square size.
- **DFlash residual semantics:** `target_layers[k]` is the input to layer k
  (conversion already applied the +1 offset); verify against the reference
  before wiring taps.
- **`llama4` pre-tokenizer:** not yet implemented in
  `hipengine/tokenization/gguf.py`; regex/split behavior must be pinned against
  `llama-tokenize` fixtures.
- **Device sharing:** resolved for P0 (servers stopped with approval); later
  phases re-check GTT occupancy before measurement.

## 6. References

- `~/llama.cpp/llama.cpp-hip/src/models/muse-glimmer.cpp` (reference graph)
- `~/llama.cpp/llama.cpp-hip/conversion/muse_glimmer.py` (tensor folds, vocab, DFlash metadata)
- `~/llama.cpp/llama.cpp-hip/tools/mtmd/clip.cpp` (`clip_graph_muse_glimmer`)
- `~/llama.cpp/run-glimmer-server.py` (production llama.cpp configuration)
- [`LAGUNA.md`](LAGUNA.md) (DFlash bring-up precedent), [`../KERNELS.md`](../KERNELS.md),
  [`../BENCHMARK.md`](../BENCHMARK.md), [`../OPTIMIZATION.md`](../OPTIMIZATION.md)

## 7. Campaign scorecard (2026-11-07)

Consolidated status; per-attempt provenance lives in `worklog/entries/`
(`topic: muse-glimmer`). Bring-up → current on the P0 host, artifact, and shapes:

| Metric | Bring-up | Current | llama.cpp (P0) | Current vs target |
| --- | ---: | ---: | ---: | ---: |
| AR decode 512/1K/4K | 2.38 | 11.13 / 11.03 / 10.83 | 12.24 / 12.17 / 12.05 | 91 / 91 / 90% |
| AR prefill 512/1K/4K | <0.5 (3×512 unfinished in 50 min) | 333 / 333 / 300 | 394 / 386 / 376 | 84 / 86 / 80% |
| DFlash decode | 2.59 (sequential verify) | 21.64 | 21.30 | 102% (Gate B passed) |

AR decode is 4.68× bring-up; DFlash is 8.36× bring-up and 1.98× true AR
(acceptance 0.49, 2.93 tokens per target forward). Gate A still needs prefill
+61/+53/+76 tok/s and decode +1.1/+1.1/+1.2 at 512/1K/4K.

### 7.1 Landed improvements

- T16 resident weight layouts + flash-decoding attention (the decode foundation).
- HIP graph decode replay + graph-safe KV append (removed the per-step host
  dispatch bubble).
- WMMA GQA flash-prefill attention kernel, 2.1–2.9× over the scalar attention
  (cosine 1.0); the WAVES=1 variant added prefill +4.5/+7.8/+19.4%.
- Prefill chunk 512→1024 (+3–4% at 1K/4K); with the attention wins, prefill
  320/299/241 → 333/333/300.
- DFlash batched-invariant block verify (bit-identical KV) + folded correction
  cycle: 2.59 → 21.6 tok/s; WMMA prefill drafter; `candidate_budget=4` (sweep
  optimum, shipping default).

### 7.2 Closed optimization directions (measured, reverted — do not re-walk)

Projection (the prefill gap):

- int8 shared-B WMMA prefill — register-bound.
- int8 WMMA projection (three designs) — correct, 0.08–0.54× (AGPR/VGPR
  accumulator spill + per-subblock scale traffic).
- int8 dp4a MMQ port (llama.cpp's single-local-int register discipline) —
  correct, ~4× slower than bf16 WMMA (5.5 vs 22 TF).
- bf16 WMMA leaf sweep — exhausted; `shared_b` fastest at ~22 TF.
- Software-pipelined double-buffer — 0.73× (51 KB LDS halves occupancy 2→1).
- XOR-swizzled LDS layout — 0.86× (bank conflict rose to 82%).
- LDS stride sweep — 272 optimal; 264 cut conflict 48.5%→5.56% but ran slower.
- Activation per-subblock LDS tiling — 0.24× (8× sync overhead).
- Wide-N configs (activation re-reads ∝ 1/OUT_TILES) — the faster `<6,4,2>` is
  NaN-incorrect; `shared_b` is correct only at `<3,4,4>`; `<8,4,1>` exceeds the
  64 KB LDS limit.

Decode:

- Fusion ceiling ~3% (casts 0.2%, norms 3%); GEMV is 80.8% of decode and
  memory-bound, so fusion cannot close the +10% gap.
- Hot-path Python — 0.344 ms/step non-GPU overhead (not the bottleneck; the
  decode graph already absorbs launch overhead).

### 7.3 Root cause (PMC-profiled)

The projection is L2-bandwidth-bound: each row block's activation (~3.4 MB) is
re-read by all 416 output-column blocks (`MemUnitBusy` 71%,
`ALUStalledByLDS` 0.019 — the LDS bank conflict is not the bottleneck).
Decode is GEMV-bandwidth-bound (199 vs 221 GB/s, ~90% of roofline). The
remaining Gate A gap is a fundamentally better GEMM data-feed than these ports
replicate, not a missing tuning trick.

## 8. Reopened optimization plan (2026-11-07)

Two external reviews reopened the campaign: the §7 "empirically closed"
conclusion was over-strong. Corrections, scoped to what was actually tested:

- **The wide-N NaNs are a loader coverage bug, not a design limit.**
  `gguf_k_t16_selected_prefill.hip:996` assigns `decode_pair = threadIdx.x >> 2`
  with no stride, so 128 threads initialize only 64 weight columns; any
  `OUT_TILES >= 5` reads uninitialized LDS. The finite/NaN split
  (`<3,4,4>`/`<4,4,x>` finite, `<6,4,2>`/`<7,4,1>`/`<6,4,4>` NaN) matches exactly.
  The 6.23 ms `<6,4,2>` timing is invalid (it skipped decode work). Wider tiling
  is an open question once the loader is fixed.
- **"L2-bandwidth-bound" was a hypothesis, not a measured roofline.**
  `MemUnitBusy` is not L2 bytes/sec; no TCC hit/miss or DRAM request counter was
  collected. The limiting level (load issue, conversion instructions,
  dependencies, registers, cache) is not yet isolated.
- **The decode breakdown sums to ~85.5%; ~14.5% is unaccounted** — larger than
  the 9% gap. The "fusion ceiling ~3%" claim is not supported until every
  dispatch is reconciled.
- **RDNA3 has no AGPRs**; the int8 "AGPR spill" wording was wrong (VGPR/private
  scratch). The int8 measurements stand; the diagnosis label does not.
- **The inner loop converts activation bf16→f16 per fragment per column block**
  (`load_compact_row_half16<uint16_t>`), stealing VALU issue slots from WMMA.

Prioritized experiments (session tasks #10–#20), each reporting which hypothesis
it supports or falsifies; cover real attention/FFN shapes, row tails, and chunk
boundaries — not only ffn_gate:

1. Fix the shared-B cooperative-loader coverage; correctness-first wide-N sweep.
2. Pre-cast activations to F16 once per projection (remove inner-loop conversion).
3. Complete whole-prefill + decode breakdown; find the missing 14.5% decode.
4. Q6 then Q4 dequantize + library-GEMM controls (the F16 wide route), including
   dequant/scratch/output-conversion cost; pre-dequantized GEMM as a diagnostic
   upper bound only.
5. Fast weight dequant (dword loads, nibble→f16 magic number, vector LDS stores)
   plus more rows per block.
6. Replace asserted rooflines with measured shape-specific limits (TCC/DRAM
   counters, cache-read ceiling, resident-byte census, allocation-type check).
7. Decode shape-specific bandwidth + projection grouping (narrow K/V fill, Q/K/V/
   gate and gate/up concatenation).
8. Fix timing-harness boundary (matched model-step vs user-facing generation).
9. 2D-tiled GEMM (2D wave grid, K-slab 32–64, ~32 KB 128×128 tile, staged
   activations) only if 1–5 do not close the gap.
10. Targeted fusion/grouping + compact-layout candidates.
11. Rollup: numerical gates, a real `LLM.generate()`/serving run confirming
    selection, benchmark + campaign updates, re-run Gate A on a matched boundary.

## 9. Results of the reopened plan (2026-11-07)

Every plan item was tackled; the headline is that the prefill gap was an
implementation defect plus a missing activation pre-cast, not a hardware wall.

**Landed (committed, correctness-gated):**

- Fixed the shared-B cooperative-loader coverage bug (the wide-N NaNs). The
  default `<3,4,4>` is byte-identical; wider tiles are now correct.
- Wired F16 activation pre-cast staging into the runner prefill (the inner-loop
  bf16→f16 conversion was stealing WMMA VALU issue slots; VALUInsts fell 59%).
- Shape-banded the F16-staged sibling to the 8-wave `<5,8,2>` wide tile for
  `out_features >= 1024` (narrow outputs keep `<3,4,4>`).
- Corrected the AR bench timing boundary (named model-step vs user-facing).

**Measured prefill (AR bench, was 333/333/300; laptop thermals float the
absolute rates, the ordering is stable):**

| Shape | hipEngine now | llama.cpp | ratio |
| --- | ---: | ---: | ---: |
| 512 | 436 | 394 | 111% |
| 1024 | 442 | 386 | 114% |
| 4096 | 387 | 376 | 103% |

Numerical gates pass: runner logits match the CPU reference and llama.cpp
oracle, greedy generation coherent, and DFlash IDs == true AR on the ten-prompt
suite plus the public speculative surface. Long-context sanity (4K/8K/16K/32K)
runs without OOM; prefill tapers with attention growth (≈346/321/315/242 at
4K/8K/16K/32K in one thermal state), decode stays ≈10–11 tok/s.

**Decode:** at 98% of the measured 211 GB/s read ceiling (L2 hit 33.8%,
DRAM-bound); the +10% gap is structural — T16 +2.78% resident-byte expansion,
the synchronized step boundary, and the lm_head — not fusible overhead. The
argmax sampling impact is ~0.04% (timing-label fix, not a real cost).

**Q6 ffn_down full-GEMM route (integrated):** the single-wave Q6 WMMA owner
(11 TF) is replaced on the shipping prefill path by a registered planar-Q6T16
dequantize-to-F16 + hipBLASLt full-GEMM sibling (22.7 TF), with a shared weight
plane and a cached per-shape best algorithm; `prefill_with_taps` shares the
same F16 staging so DFlash stays exact. This cleared the 4K shape and Gate A
prefill at 512/1K/4K. A one-time per-shape algorithm-selection warmup runs on
the first prefill (JIT-style). The 2D-tiled GEMM redesign was dropped as
superseded by these targeted wins.

**Gate status:** Gate A prefill passes at 512/1K/4K; Gate A decode is still
≈91% (bandwidth-roofline structural gap). Gate B (DFlash) passes and remains
exact. Basic validation complete; no large weakness found.

## 10. Decode and long-context pass (2026-11-08)

Directional diagnostics on the zbook (power-capped at a 70 W PPT, sclk ~2.4
GHz under load), r=1; formal Gate A rows are a separate clean rerun.

**Decode measurement corrections.** The "211 GB/s ceiling" was a DtoD copy; a
read-only grid-stride kernel reaches ~240 GB/s on this host, but a read through
the GEMV access pattern (one wave streaming each tile's K range) tops out near
222 GB/s, and the in-graph decode GEMVs reach ~219 GB/s. The pre-pass step
(90.9 ms at 512 ctx) was kernels 84.1 + intra-graph gaps 6.1 (1204 launches,
~4-5 us per boundary) + host 0.67 ms.

**Landed, all bit-identical to the paths they replace** (DFlash verify owners
still match AR decode):

- Grouped Q4T16 tile16 GEMV (Q/K/V/gate and gate/up as one launch each) and a
  hoisted planar Q6T16 GEMV: step 89.0 -> 84.1 ms.
- Fused decode glue (QK-norm + rope + KV append; post-norm + residual + next
  pre-norm with all reads issued up front; gated attention combine): 1204 ->
  554 launches, 84.1 -> 81.4 ms.
- One staging upload and in-place softcap per step; 8-key unrolled decode
  attention (32K ctx 91.6 -> 89.7 ms/step; this one is a production
  association change, shared by decode and verify).
- Multi-token WMMA flash prefill (eight query tokens share each K/V tile):
  prefill 4K 387 -> 432, 16K 321 -> 393, 32K 265 -> 354 tok/s. The unused
  per-chunk keep masks are gone.

**Measured, not kept:** split-K Q4 decode GEMV with consistent verify siblings
(1.04-1.07x isolated, identical in-graph at the ~219 GB/s ceiling); vector
header loads; 128 span-sized attention splits; 16 Q heads per decode-attention
block; 16 tokens per prefill-attention block.

**Sweep (current code, r=1, 512 decode tokens per row):**

| Prompt | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 512 | 443.9 | 12.21 |
| 1K | 453.9 | 12.10 |
| 4K | 427.7 | 11.92 |
| 8K | 412.9 | 11.81 |
| 16K | 392.3 | 11.58 |
| 32K | 357.5 | 11.11 |
| 128K (130,560) | 221.5 | 8.74 |

Needle-in-a-haystack through `LLM.generate()` recalls the passphrase at 32K
(depth 0.5) and 128K (130,713 tokens, depth 0.25, ~98K tokens back), so
long-range recall through the NoPE full layers works. `<|eom|>` (200007) is not
a stop token in either engine; chat-templated turns end on `<|eot|>`.

**DFlash (mtpbench, budget 4, r=1):** 21.64 tok/s, exact IDs == true AR on
all ten prompts, acceptance 49.6%, 2.95 tokens per target forward; the AR
control is now 12.05 tok/s (1.80x). DFlash did not move because block verify
runs eagerly through the exact multi-row owners rather than the fused decode
graph; carrying the fused glue and grouped projections into verify is the
DFlash lever.

**Remaining decode levers:** decode attention is ~2.5x its bandwidth floor at
32K and dominates the 128K decode drop (a lane-per-key restructure of the
shared decode/verify kernel); T16 Q4 scale/min compaction (+2.78% bytes);
folding silu into the down GEMV input and merging the split/combine launches.
