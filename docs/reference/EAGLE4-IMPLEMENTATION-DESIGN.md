---
status: current
owns: Independent clean-room hipEngine implementation design for the EAGLE-4 feature-autoregression drafter (single-layer second-to-top feature draft, cost-aware dynamic tree, efficient low-rank LM head) on Qwen3.8-27B, and its RED test plan.
---
# EAGLE-4 independent implementation design (Qwen3.8-27B)

Design note for tasks #19–21 of the
[`Qwen3.8-27B specdec campaign`](../campaigns/QWEN38-27B-SPECDEC-OPTIMIZATION-CAMPAIGN.md).
**Clean-room:** we reviewed the EAGLE-4 architecture doc, paper, and the
`EAGLE-Corp/Qwen3.8-27B-EAGLE4` artifact's *config and weight layout only* for
mechanics. EAGLE-4 is **source-available, not open source** (commercial competing
use needs a license; revenue-bounded otherwise), so **no reference code is
vendored** — the hipEngine kernels, runtime, loading, and tree policy are written
independently against our four-axis registry. We load the trained weight artifact
(the model), not the reference implementation.

## 1. What EAGLE-4 is (mechanics)

Feature-level autoregression in the EAGLE-1/2/3 lineage, with a cost-aware dynamic
tree and an efficient LM head. Unlike DSpine/DFlash2 (parallel block drafters with
a **5-layer** backbone), the EAGLE-4 drafter is a **single Transformer layer**
(~0.44B params). This directly attacks the structural deficit the campaign found
in DFlash2/DSpine: the multi-layer drafter forward cost. The drafter proposes a
**tree** autoregressively rather than a fixed parallel block.

One decode round (mechanics reviewed from the architecture doc):

1. **Seed / extend.** Run the drafter over the last accepted tokens to update its
   KV and produce the root feature + first candidates.
2. **Tree growth.** From the root, expand the best `K` frontier nodes per level.
   Each draft step reads the target's hidden states, attends to the drafter's
   sliding window **plus the row's tree ancestors**, and scores the vocabulary
   with the efficient head (§3).
3. **Rank + select.** A node's score = parent score × its draft probability. The
   verify set is the best `width − 1` nodes plus the root, ordered parent-first
   (every prefix is itself a tree).
4. **Cost-aware budget.** A cost model picks draft depth and verify width from
   measured draft/verify cost tables (batch size, context length) and draft
   confidence — the same conclusion as the campaign budget finding: the optimum
   is cost/acceptance-driven, not fixed or context-only.
5. **Target verify.** One target forward scores every verify node with a **tree
   mask** (each row attends to committed context + its ancestors). Greedy accept
   walks the target argmax from the root; the accepted path + one bonus token are
   committed; accepted KV/state move to chain positions.

### Artifact architecture (config + weight inspection, clean-room)

`config.json`: `LlamaForCausalLMEagle4`, `num_hidden_layers=1`, `hidden_size=5120`,
`target_hidden_size=5120`, `num_attention_heads=40`, `num_key_value_heads=40`
(MHA, not GQA), `head_dim=128`, `intermediate_size=16384`, `sliding_window=4096`,
`rope_theta=1e7`, `rms_norm_eps=1e-6`, `fc_norm=true`, `norm_output=true`.
`eagle_config.eagle_aux_hidden_state_layer_ids=[3,31,59]`,
`use_aux_hidden_state=true`, `num_aux_hidden_states=3`.
`efficient_lm_head_config={type: low_rank, rank: 128, num_candidates: 128}`.

Weight tensors (BF16):

| Tensor | Shape | Role |
| --- | --- | --- |
| `fc.weight` | (5120, 15360) | fusion: 3×5120 → 5120 |
| `fc_norm.{0,1,2}.weight` | (5120,) | per-chunk RMSNorm of the 3 fusion inputs |
| `layers.0.input_layernorm.weight` | (5120,) | pre-attention norm |
| `layers.0.hidden_norm.weight` | (5120,) | feature norm (EAGLE second-to-top input) |
| `layers.0.self_attn.{q,k,v}_proj.weight` | (5120, 10240) | attention over a 2×5120 concatenated input |
| `layers.0.self_attn.o_proj.weight` | (5120, 5120) | attention output |
| `layers.0.post_attention_layernorm.weight` | (5120,) | pre-MLP norm |
| `layers.0.mlp.{gate,up}_proj.weight` | (16384, 5120) | SwiGLU |
| `layers.0.mlp.down_proj.weight` | (5120, 16384) | SwiGLU down |
| `norm.weight` | (5120,) | output feature norm |
| `efficient_lm_head_down_proj.weight` | (128, 5120) | hidden → rank-128 |
| `efficient_lm_head.weight` | (248320, 128) | low-rank vocab estimate |
| `lm_head.weight` | (248320, 5120) | full-rank shortlist scoring |
| `embed_tokens.weight` | (248320, 5120) | drafter token embedding |

**RED-oracle questions (task #20, verify against the reference forward):** the
exact assignment of the three `fc` input chunks (target aux hidden states
[3,31,59] vs. the second-to-top feature vs. token embedding) and how the aux
states combine; the composition of the 10240-dim attention input (the
concatenation of the normalized hidden and the fused input feature); and whether
`hidden_norm` applies to the target feature before fusion. These are pinned by the
CPU oracle RED test against a reference forward trace before any GPU kernel lands.

## 2. hipEngine module layout

| Piece | Path (new) | Reuses |
| --- | --- | --- |
| Config + weight loading | `hipengine/loading/eagle4.py` | `hipengine/loading/safetensors.py` index; DSpine loader pattern |
| CPU oracle (RED) | `hipengine/kernels/cpu_reference/eagle4.py` | `hipengine/kernels/cpu_reference/dflash2.py` attention/ops |
| Drafter runtime | `hipengine/runtime/eagle4.py` | `hipengine/speculative/dflash2_native.py` context/attention helpers |
| Fusion + efficient-head kernels | `hipengine/kernels/hip_gfx1100/speculative/eagle4.{hip,py}` | DFlash dense WMMA, norms, single-wave head rowtile |
| Tree policy + cost budget | `hipengine/speculative/eagle4_tree.py` | `hipengine/speculative/frontier.py`, `candidate_budget.py`, `chain.py` |
| Speculative provider | `hipengine/generation/qwen35_gguf_eagle4.py` | `hipengine/generation/qwen35_gguf_dspine.py` generic-child attachment |
| Target tree verify | existing `qwen35_gguf` verify + tree mask | `hipengine/speculative/verify_graph.py` |

Registry keys (four-axis, never identity-gated): `(backend, "eagle4_fusion",
quant, variant)`, `(backend, "eagle4_efficient_head", quant, variant)`,
`(backend, "eagle4_tree_attention", quant, variant)`. Capability-gated attachment
on the draft `eagle_config`/`efficient_lm_head_config` metadata + hidden-size
match — no model-name/path/hash allowlist (AGENTS.md "Never key admission on
identity"). Provider key `(provider="eagle4", target_model="qwen3_5_gguf",
backend, quant)`, registered in `register_builtin_speculative_providers`.

## 3. Efficient LM head (the headline drafter-cost lever)

Full-vocabulary logits dominated the vendor draft forward (65.4% of 574 µs). The
low-rank head avoids the full `hidden @ lm_head.T` over 248320 vocab on every draft
step:

1. `r = hidden @ efficient_lm_head_down_proj.T` → rank-128 vector.
2. `approx_logits = r @ efficient_lm_head.weight.T` → 248320 cheap estimate
   (rank-128, ~128/5120 of the full head FLOPs).
3. Retrieve the top `num_candidates=128` (sharded top-k).
4. Re-score **only** those 128 rows exactly with `lm_head.weight[cand]` → shortlist
   logits → draft probabilities for tree expansion.

This builds on the Muse Glimmer draft-head single-wave rowtile work. The retrieve
step is a top-k over the low-rank estimate (reuse `dflash2_top16_rows` generalized
to 128, or a sharded top-k); the shortlist scoring is a small gathered dense
(128×5120). The draft proposes, so the head owes no bit-exactness — the target
verify keeps output AR-exact.

## 4. Kernel plan (profile-first, one lever per unit)

1. **CPU oracle** of the full single-layer forward + efficient head + tree scoring
   (RED), pinned to a reference forward trace. No GPU until green.
2. **Fusion + single-layer forward** on the reused DFlash dense WMMA / rmsnorm /
   silu / sliding-attention primitives (MHA 40/40, head_dim 128, window 4096). The
   10240-dim attention input is a concatenation handled by the projection layout.
3. **Efficient head** kernels (low-rank estimate + sharded top-128 + gathered
   shortlist scoring). Profile vs the full head; this is the largest drafter lever.
4. **Tree attention + verify mask** for the drafter sliding window + ancestors,
   and the target tree-mask verify. Reuse the speculative tree infrastructure.
5. **Cost-aware budget** from a hipEngine profiling pass (draft/verify cost vs
   depth/width/context), mirroring `eagle-harness` but with our measured tables.
6. **Graph capture** of the draft step + verify width (Muse Glimmer pattern) once
   the eager path is exact and profiled.

## 5. RED test plan (CPU oracle first)

- `tests/test_unit_eagle4_cpu_oracle.py`: fusion chunk math, single-layer forward,
  efficient-head retrieve+shortlist, and tree node scoring vs an independent
  serial numpy reference; greedy draft path == reference on synthetic weights.
- `tests/test_gpu_eagle4_kernels.py`: each GPU kernel RED-pinned to the CPU oracle
  (fusion, efficient head, tree attention), HIP-availability guarded.
- `tests/test_gpu_eagle4_efficient_head.py`: low-rank retrieve + shortlist scores
  match the full-head top-128 within tolerance; candidate set equality.
- Live exactness gate: EAGLE-4 greedy IDs == true AR on the ten-prompt mtpbench
  suite via the resident-session verify surface, then public
  `LLM.generate_speculative` (the DSpine generic-child route).

## 6. Validation gates and order

1. CPU oracle green (RED) → 2. GPU kernels pinned → 3. single-prompt EAGLE-4 == AR
→ 4. full ten-prompt suite AR-exact + acceptance band → 5. public-surface route →
6. speed vs MTP/DFlash2/DSpine baselines on the full suite (profile-first,
back-to-back A/B under thermal throttling) → 7. rollup + worklog + commit. The
acceptance band and AR baseline are pre-registered from the task #13 rows before
speed tuning (anti-gaming). INT8/quant drafter variants only if they improve speed
without breaking the exactness/acceptance gates.

## 7. Sources (mechanics reviewed, clean-room)

- EAGLE-4 architecture doc + paper (github.com/eagle-hq/EAGLE-4), reviewed for
  mechanics only; source-available license forbids vendoring.
- `EAGLE-Corp/Qwen3.8-27B-EAGLE4` config.json + safetensors weight layout (the
  trained artifact hipEngine loads).
- Campaign §3.2 (vendor evidence, comparison framing) and the Muse Glimmer
  draft-head / verify-graph / budget findings (hipEngine analogues).