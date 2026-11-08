---
status: current
owns: Independent clean-room hipEngine implementation design for the DSpine parallel drafter (adjacent causal injection + transition-cache decoding) on Qwen3.8-27B, and its RED test plan.
---
# DSpine independent implementation design (Qwen3.8-27B)

Design note for tasks #16–18 of the
[`Qwen3.8-27B specdec campaign`](../campaigns/QWEN38-27B-SPECDEC-OPTIMIZATION-CAMPAIGN.md).
**Clean-room:** we reviewed the DSpine paper (arXiv:2609.36173) and the Tencent
reference source for *mechanics and weight layout only*; the hipEngine kernels,
runtime, and loading are written independently against our four-axis registry.
We load the Apache-2.0 `tencent/Qwen3.8-27B-DSpine-b16` trained weights (the model
artifact), not the reference code. No reference source is vendored.

## 1. What DSpine is (mechanics)

A DFlash-family parallel drafter: one block-parallel forward proposes a block of
`M+1` positions (anchor `x_0` + `M` masked candidates); the target verifies in one
pass. DSpine's difference from DFlash2 is **how intra-block token dependencies are
formed**: instead of DFlash2's grouped conv + post-backbone path selector, DSpine
injects each predecessor's *predicted feature* into its successor **at every
layer**, then conditions final token selection on the *decoded* predecessor via a
transition cache.

Components (inference only; training losses are not implemented):

1. **Backbone.** `fc` (concat of `num_target_layers` target taps → hidden) →
   `hidden_norm` → `L` decoder layers → final `norm`. Each layer is Qwen3-style:
   `input_layernorm`, attention (`q/k/v/o_proj`, per-head `q_norm/k_norm`),
   `post_attention_layernorm`, SwiGLU MLP. Attention is **block-causal** (position
   `t` attends to all target-context K/V + block positions `0..t`), **no sliding
   window**. Context K/V are projected from the fc'd target features every layer
   (cacheable across cycles, as DFlash2 already does).
2. **Unified transfer space (codebook `C`).** A fixed `[vocab, code_size]` table
   derived from the target LM head (l2-normalize rows → PCA whitening to
   `code_size` directions → renormalize). A known token `x` enters as
   `sqrt(code_size)·C_x`. The checkpoint ships `C` as `selected_prefix_bridge.codebook`.
3. **Adjacent causal injection (the "code bridge"), after every layer `l`.** With
   pre-injection layer output `u` (BF16):
   - `rms = sqrt(mean(u²))` (f32, clamped); `n = u/(rms+eps)` (BF16).
   - `p = n @ R_lᵀ` (f32), `z = sqrt(C)·p/(‖p‖+eps)` (BF16) — the predicted feature.
     `R_l` = `early_read_weights[l]` (layers `< L-1`) or `read_weight` (last layer).
   - Position 0 (anchor) uses `z_0 = sqrt(C)·C_{x_0}` instead of its projected feature.
   - `packed = z @ [G_q; G_p; A]ᵀ` → split into `q,s,v` (each `rank`-dim). The three
     projections are packed into one matmul (`gate.weight`, `predecessor_gate.weight`,
     `prev.weight` stacked).
   - `gate_t = sigmoid(q_t + s_{t-1} + b)`; `message_t = gate_t ⊙ v_{t-1}` (position 0
     message = 0).
   - `delta = message @ W_lᵀ` (`feature_write_weights[l]`); `out = u + rms·delta`
     (non-anchor positions only). Then the next layer's input norm is applied.
   - Shared across layers: `A=prev`, `G_q=gate`, `G_p=predecessor_gate`, `b=gate.bias`.
     Per-layer: `R_l`, `W_l`. The terminal `out` projection is zero (disabled).
4. **Predecessor-conditioned decoding (transition cache).** After the final layer,
   `norm` + shared LM head give each position's top-`K` candidates (`head_topk`,
   power of two, e.g. 16). A decoded predecessor token `s` enters as `sqrt(C)·C_s`
   and redoes the **last-layer** write from the cached pre-injection state
   (`stash`: raw, rms, packed-q). To avoid sequential scoring, precompute two
   vocabulary-wide tables once:
   - `TG[v] = (E_v ⊙ norm_w) @ W_{L-1}` (`[V, rank]`),
   - `TSV[v] = (sqrt(C)·C_v) @ [G_p; A]ᵀ` (`[V, 2·rank]`).
   Then the conditional score for sender candidate `i` → receiver candidate `j` is
   `base_j + rms · (sigmoid(q + TSV[i][:rank] + b) ⊙ TSV[i][rank:]) · TG[j]`, where
   `base_j = (raw ⊙ norm_w)·E_j`. Greedy selects the chain by walking the `[n,K,K]`
   argmax table left-to-right (no sequential matmuls). Sampling builds the same
   table and walks it under the request's top-k/top-p/min-p filters.

The reference MLX port keeps the bridge **BF16** (the refinement tail depends on
its exact rounding) and optionally quantizes the backbone (`fc` + layers) to INT8.
We follow the same split.

## 2. hipEngine module layout

Mirror the DFlash2/Muse-Glimmer drafter structure; reuse the DFlash2 foundation
(`hipengine/speculative/dflash2_native.py`, `kernels/.../speculative/dflash_drafter.hip`,
`dflash2.hip`) for the backbone primitives.

| Piece | Path (new) | Reuses |
| --- | --- | --- |
| Config + weight loading | `hipengine/loading/dspine.py` | `loading/dflash.py` patterns; safetensors index |
| CPU oracle (RED) | `hipengine/kernels/cpu_reference/dspine.py` | numpy reference of §1 math |
| Drafter runtime | `hipengine/runtime/dspine.py` | `runtime/muse_glimmer_dflash.py` session/tap pattern |
| Injection + refine kernels | `hipengine/kernels/hip_gfx1100/speculative/dspine.{hip,py}` | DFlash2 dense WMMA, norms |
| Block-causal attention | reuse `muse_glimmer_draft_attention` (task #7) | non-causal→block-causal mask variant |
| Speculative provider | `hipengine/generation/dspine.py` | `generation/muse_glimmer_dflash.py` |
| Target verify + state | existing `qwen35_gguf` verify_target_block | MTP/DFlash2 verify path |

Registry keys (four-axis, never identity-gated): `(backend, "dspine_injection",
quant, variant)` for the fused bridge write, `(backend, "dspine_refine", quant,
variant)` for the transition-table scoring, plus the reused `linear`/attention
keys for the backbone. The drafter is admitted by declared capability (config
architecture `DSpineDraftModel`, block-causal, feature-writes), not model name.

## 3. Backbone reuse and the linear-attention (ReplaySSM) question

The DSpine **drafter** uses full block-causal attention (no SSM in the drafter);
its backbone maps onto the DFlash2 dense-WMMA + attention primitives. The
**target** Qwen3.8-27B is the hybrid (linear-attention/GDN layers). The
"ReplaySSM-equivalent" is a *target-verify* concern: verifying a draft block must
advance the target's recurrent state for accepted rows and roll it back for
rejected rows. hipEngine's `qwen35_gguf` runtime already does this for MTP/DFlash2
(`qwen35_gdn_recurrent_*` kernels + state snapshot/rollback). DSpine reuses that
verify path unchanged; no new SSM kernel is needed for the drafter. Confirm the
existing rollback handles the DSpine block geometry (block-causal, variable accepted
prefix) in the RED exactness test.

## 4. Kernel plan (profile-first, one lever per unit)

1. **Fused bridge write** (`dspine_injection`): one kernel doing RMS-read →
   `R_l` projection → normalize → packed `[G_q;G_p;A]` matmul → sigmoid gate →
   predecessor message → `W_l` write → residual → next input norm. The projections
   are small (`hidden↔code_size↔rank`); the predecessor shift (`t-1`) is a register
   shuffle / shared-memory neighbor read. Keep the bridge BF16 (rounding-sensitive).
2. **Block-causal attention**: adapt the task-#7 `muse_glimmer_draft_attention`
   WMMA kernel from non-causal to block-causal (mask `key ≤ position` within the
   block, all context visible). 32Q/8KV/d128 geometry matches.
3. **Transition tables + chain walk** (`dspine_refine`): precompute `TG`/`TSV`
   once per request (vocab-wide, chunked); per cycle build the `[n,K,K]` score
   table and walk the greedy chain. The top-`K` candidate selection reuses the
   Muse Glimmer single-wave Q5 head lever where the head quant matches.
4. **Backbone projections**: reuse the DFlash2 WMMA dense kernels (already
   default-WMMA); INT8 backbone quant is a follow-up speed lever (task #18).

## 5. RED test plan (CPU oracle first)

`tests/test_unit_dspine_cpu_oracle.py` (CPU, no GPU) pins the §1 math against a
numpy oracle written from the paper equations (not the reference code):

1. **Transfer space**: codebook row norm == sqrt(C) convention; anchor code
   substitution at position 0.
2. **Single-layer injection**: given random `u`, `R_l`, packed gate/message,
   `W_l`, the oracle computes `out` and the next-norm; assert the (future) kernel
   matches to BF16 tolerance. Include the anchor-zero-message and non-anchor
   residual cases, and the `t-1` predecessor shift at block boundaries.
3. **Block-causal mask**: position `t` attends only to context + `0..t`; verify no
   leakage from later block positions (the training-correctness invariant).
4. **Transition-cache equivalence**: the parallel `TG`/`TSV` table score equals a
   direct serial last-write refinement (redo the last-layer write per sender) to
   tolerance; greedy chain walk == serial argmax decoding.
5. **End-to-end drafter oracle**: a small synthetic DSpine config (2 layers, tiny
   hidden/vocab) runs the full forward + refinement; pin output ids.

GPU RED tests (`tests/test_gpu_dspine_*.py`, HIP-guarded) then pin each kernel to
the CPU oracle (outer floor KL ≤ 0.05 / top-1 ≥ 90%; bridge BF16 tighter). The
binding live gate (`tests/test_live_dspine_exactness.py`): DSpine greedy IDs ==
true AR on the ten-prompt suite + full mtpbench, acceptance within a pre-registered
band vs the DFlash2/MTP baselines, public `LLM.generate_speculative`/serve route
confirmed.

## 6. Validation gates and order

- Correctness: AR-exact IDs (verify unchanged), acceptance within band, full
  mtpbench (anti-gaming), true no-speculation AR baseline from the same protocol.
- Speed: back-to-back A/Bs + rocprof kernel attribution (ZBook thermally throttled;
  final publishable numbers on the stable-clock Desktop). Compare acceptance length
  and drafter-cost-per-committed-token vs DFlash2 (9.72 tok/s) and MTP (18.05).
- Order: (#16 design + CPU oracle) → (#17 drafter + provider, exactness) →
  (#18 speed + INT8 backbone). The win case is DSpine's higher acceptance
  amortizing the drafter cost over more committed tokens (paper: +10% acceptance
  over DFlash2 at parity round cost); measure it on our hardware, do not assume it.

## 7. Sources (mechanics reviewed, clean-room)

- Paper: arXiv:2609.36173v2 (DSpine), §§4.1–4.3 + Appendix A (architecture,
  transfer space, injection equations, transition cache, hyperparameters).
- Reference layout (weight names/config contract only): github.com/tencent/dspine
  `dspine_mlx/dspine_mlx/{model,bridge,config,propose}.py`. Weights:
  `tencent/Qwen3.8-27B-DSpine-b16` (Apache 2.0). No reference code is copied.