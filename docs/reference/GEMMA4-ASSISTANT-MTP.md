---
status: current
owns: The `gemma4-assistant` MTP head's forward specification (per-layer step order, KV binding, attention scale), where it is implemented, how a caller reaches it, what it measures, and its open defects. Read before changing the head's forward or its draft/verify loop.
---
# Gemma 4 assistant head (MTP)

The Gemma 4 26B-A4B MTP head is a **separate GGUF artifact with its own
architecture string**, not tensors inside the target. This document is the
resolved specification: the algorithm, the hyperparameters, the tensor layout,
and which KV the head's four blocks attend against. The sections after the
specification describe the implementation, how to reach it, what it measures,
and what is still open.

The head is implemented and reachable through the public API for greedy
requests on `hip_gfx1100` and `hip_gfx1151`. Two things in this document are
not as the code's own docstrings describe them, and both are under
"Open defects": the verify pass is not arithmetically identical to one-token
decode, and sampled (temperature above zero) requests are refused.

## The artifact

| field | value |
| --- | --- |
| path | `/models/gguf/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` |
| size | 461,766,816 bytes |
| `general.architecture` | **`gemma4-assistant`** (the target is `gemma4`) |
| tensors | 49 |
| blocks | 4 (`blk.0` … `blk.3`) |
| linear type | Q8_0, norms F32 |

Upstream also ships `-BF16` and `-F16` variants under `MTP/`.

## Hyperparameters

| key | value | meaning |
| --- | --- | --- |
| `block_count` | 4 | layers in the head |
| `embedding_length` | 1024 | the head's own width |
| `embedding_length_out` | **2816** | the target's width |
| `feed_forward_length` | 8192 | head FFN width |
| `attention.head_count` | 16 | query heads |
| `attention.head_count_kv` | 8 per layer | GQA |
| `attention.key_length` / `value_length` | 512 | full-attention head dim |
| `attention.key_length_swa` / `value_length_swa` | 256 | sliding-window head dim |
| `attention.sliding_window` | 1024 | window |
| `attention.sliding_window_pattern` | per layer | which layers are SWA |
| `attention.shared_kv_layers` | 4 | see "KV binding" |
| `rope.dimension_count` / `_swa` | 512 / 256 | rotary dims |
| `rope.freq_base` / `_swa` | 1e6 / 1e4 | two RoPE bases |
| `nextn_predict_layers` | 4 | equals `block_count` |

`embedding_length_out = 2816` is what makes `2 * 2816 = 5632` the pre-projection
input width. Three independent sources agree on that number: the reference's
tensor shape, the artifact's `nextn.pre_projection` dims, and this metadata key.

## Tensor layout

11 tensors per block, identical names across all four:

```
attn_norm  layer_output_scale  attn_q  attn_q_norm  attn_output
post_attention_norm  ffn_norm  ffn_gate  ffn_up  ffn_down  post_ffw_norm
```

Plus five global tensors: `token_embd` (Q8_0, 285 MB), `output_norm`,
`rope_freqs`, `nextn.pre_projection` (Q8_0, `[5632, 1024]`, 21.9 MB), and
`nextn.post_projection` (Q8_0, `[1024, 2816]`, 3.1 MB).

**There is no `attn_k` and no `attn_v` in any block.** That is what makes the KV
binding question the only open one, and it is now answered below.

`output` is the same tensor as `token_embd` (`TENSOR_DUPLICATED` in the
reference's loader), so the head's vocabulary projection is tied.

## The forward

From `llama.cpp@17252c769`, `src/models/gemma4-assistant.cpp`. `n_embd_backbone`
is 2816, `n_embd` is 1024.

```
x  = backbone_tok_embd[token]       # [2816]   the *target* model's embedding
x  = x * sqrt(n_embd_backbone)      # 2816 ** 0.5
xh = concat(x, h_backbone)          # [5632]   h_backbone is the target's hidden state
cur = nextn_proj_pre @ xh           # [1024]

for il in 0 .. 3:
    is_swa       = sliding_window_pattern[il]
    n_embd_head  = 256 if is_swa else 512
    freq_base    = 1e4 if is_swa else 1e6
    n_rot        = 256 if is_swa else 512
    freq_factors = None if is_swa else rope_freqs

    normed = rmsnorm(cur, attn_norm)
    Q      = wq @ normed                        # [16 * n_embd_head]
    Q      = reshape(Q, n_embd_head, 16)
    Q      = rmsnorm(Q, attn_q_norm)            # per head, over n_embd_head
    Q      = rope(Q, pos, freq_factors, n_rot, freq_base, freq_scale)

    attn     = attention(Q, K, V, scale = 1.0)  # K, V are the backbone's; see below
    attn     = wo @ attn                        # [n_embd]
    attn     = rmsnorm(attn, post_attention_norm)
    attn_out = attn + cur

    ffn_in = rmsnorm(attn_out, ffn_norm)
    ffn    = down @ (gelu_tanh(gate @ ffn_in) * (up @ ffn_in))
    ffn    = rmsnorm(ffn, post_ffw_norm)

    cur = (ffn + attn_out) * layer_output_scale[il]

cur    = rmsnorm(cur, output_norm)
logits = head_token_embd @ cur      # [262144]
h_next = nextn_proj_post @ cur      # [2816]  the next-token hidden state
```

Five details that are easy to get wrong:

- **`f_attention_scale = 1.0f`.** The head does *not* apply `1/sqrt(head_dim)`;
  the reference sets the scale explicitly in `load_arch_hparams`.
- **The norm order is post-norm, not pre-norm.** `attn_post_norm` normalizes the
  attention *output* before the residual add, and `post_ffw_norm` normalizes the
  FFN output before its residual add. Only `attn_norm` and `ffn_norm` are
  pre-norms.
- **`attn_q_norm` is applied to Q only**, per head, over `n_embd_head`. There is
  no K norm.
- **`wo` sits between the attention and `attn_post_norm`.** The `attn_output`
  tensor is `[n_embd_head*n_head, n_embd]`, so the projection contracts the
  concatenated heads back to `n_embd` *before* `post_attention_norm` (which is
  `[n_embd]`) sees it. Applying the norm to the unconcatenated `[n_head*n_embd_head]`
  attention output would be a shape error against a `[n_embd]` weight.
- **The input embedding and the output projection are different tables.** The
  graph reads `model_other->tok_embd` -- the *backbone's* embedding, width 2816 --
  for its input, and uses the head's own `token_embd` (width 1024, duplicated as
  `output`) only for the logits. The head's `token_embd.weight` is never used as
  an input embedding. So the forward needs the backbone's embedding table *and*
  the head's, and a port that gathers the head's own table for the input would
  produce a `[1024]` vector where `[2816]` is required.

## KV binding — the resolved question

The head creates no `wk`/`wv`, and `attention.shared_kv_layers` is read in
`load_arch_hparams` but never used in `load_arch_tensors`. The answer is in the
cache construction, `src/llama-model.cpp` around line 2578:

```c
if (arch == LLM_ARCH_GEMMA4_ASSISTANT) {
    llama_memory_t mem_other = llama_get_memory(cparams.ctx_other);
    share = [&](int32_t il) {
        const llama_model * model_other = llama_get_model(cparams.ctx_other);
        if (hparams.is_swa(il)) return llama_model_n_layer(model_other) - 2;
        return llama_model_n_layer(model_other) - 1;
    };
    res = new llama_kv_cache_iswa(..., mem_other, filter, reuse, share);
}
```

So, with a 30-layer backbone:

- **Every one of the four head blocks attends against the backbone's last two
  layers' KV.** A sliding-window head block reads **backbone layer 28**; a
  full-attention head block reads **backbone layer 29**.
- The head's K/V tensors are views, not copies: `llama_kv_cache` pushes a *copy
  of the layer struct* for a shared layer, so `k`/`v` point at the same device
  buffers the backbone wrote.
- `cparams.ctx_other` is the target context. `llama-context.cpp` throws
  `"Gemma4Assistant requires ctx_other to be set"` when it is missing, so a
  draft context cannot be built without the target's context handle.

**Consequence for the port:** the head needs no KV allocation, no eviction
policy, and no `KVLiveSpans` plumbing of its own. It needs read access to two of
the backbone's KV layer buffers, which means the backbone runner has to expose
them by layer index, and the draft step has to be positioned at the same token
position the backbone last wrote.

**The shared cache needs no GQA adaptation, and that is checkable.** The head's
`attention.head_count_kv` is `[8, 8, 8, 2]` per block, and the backbone's
per-layer pattern is `[8, 8, 8, 8, 8, 2]` repeating over its 30 layers. Layer 28
is `28 mod 6 == 4`, so 8 KV heads and sliding-window; layer 29 is
`29 mod 6 == 5`, so 2 KV heads and full attention. The head's three SWA blocks
read layer 28 and ask for 8 KV heads; its full block reads layer 29 and asks for
2. Both sides also carry `head_count = 16` with `key_length` 512 and
`key_length_swa` 256.

So for the two layers the head reads, the head's attention geometry is
**identical** to the backbone layer's -- same Q head count, same KV head count,
same head width. The head's attention can therefore call the backbone's own
attention kernel against the backbone's KV buffers with no head remapping and no
head-count conversion. A port that treated `head_count_kv` as a property of the
head alone would be free to pick a value the cache does not carry; the equality
above is what makes 8 and 2 correct rather than arbitrary, and a test should
assert it against both artifacts rather than trusting this paragraph.

**The rope geometry is likewise recoverable, not assumed.** The head's
`rope_freqs.weight` is 256 entries: the first **64** are exactly `1.0` and the
remaining 192 are `1e30`. That is the same span-marker encoding the backbone's
artifact uses, which `gemma4_rotated_pair_count` already decodes -- so the
full-attention block rotates **64** pairs of its 256 with the exponent scale of
`head_dim = 512` (`rope_type="proportional"`), and the three SWA blocks rotate
all **128** of their 128 pairs at `head_dim = 256`. A reading that treated
`freq_factors` as a multiplier rather than a span marker would divide the
inverse frequencies by `1e30` and silently produce a table that matches
rotation-by-zero only by accident.

**The head's rope configs are the backbone layers' own, verified against both
artifacts.** Decoding the backbone gives layer 28 as
`(default, head_dim=256, rotated_pairs=128, freq_base=1e4)` and layer 29 as
`(proportional, head_dim=512, rotated_pairs=64, freq_base=1e6)`. Decoding the
head's own metadata and `rope_freqs` independently gives exactly those two
configs. So the head does not merely *tolerate* the backbone's tables -- it wants
the same ones, and a port can build each head block's rope from the backbone's
`rope_for_layer` for the layer it reads instead of deriving a second schedule.
The equality is also the check: if the two ever disagree, one of them is wrong,
and it is cheaper to notice that here than in an acceptance-rate number.

## Implementation

| Piece | Where |
| --- | --- |
| Config decoder, expected shapes, tensor-map validation | `hipengine/loading/gemma4_assistant_gguf.py` |
| Device residency for all 49 tensors (461.8 MB file, 440.5 MB on device) | `hipengine/loading/gemma4_assistant_device.py` |
| Target/head pairing checks before either is loaded | `hipengine/loading/gemma4_mtp_gguf.py` |
| The head's forward and the drafter that chains it | `hipengine/runtime/gemma4_assistant.py` (`Gemma4AssistantHead`, `Gemma4MtpDrafter`) |
| Draft/verify loop, telemetry, provider registration | `hipengine/generation/gemma4_mtp.py` (provider `gemma4_mtp`) |
| Generator hook that attaches a provider | `hipengine/generation/gemma4_gguf.py` (`attach_speculative_provider`) |

The head reads backbone layers 28 and 29 through the runner, uses the backbone's
embedding table for its input and its own for the logits, and builds its rope
tables from the backbone layers it reads, as specified above.

### The cycle

One cycle drafts up to `candidate_budget` tokens with the head, then verifies
them in one backbone forward of `[token, *drafts]` that returns
`len(drafts) + 1` logit rows (`Gemma4Runner.forward(..., logits_rows=...,
verification=True)`). It accepts the longest run of drafts that match the
verify rows' argmax and commits one more token from the row after the last
accepted draft, so a cycle commits between 1 and `candidate_budget + 1` tokens.
The backbone is rewound to the committed position and the next cycle's seed
hidden row is taken from the verify pass.

- `drafts[0]` predicts the token after `token`, not `token` itself.
- The last cycle leaves one row of headroom so a request never emits more than
  `max_tokens`.
- `candidate_budget` is 1 to 8.
- Requests with `temperature != 0` or `top_k` outside `{0, 1}` are refused with
  an error naming the limitation.

### Reaching it

```python
llm = hipengine.LLM(
    model="<target>.gguf",
    speculative_provider="gemma4_mtp",
    draft_model="<mtp head>.gguf",
    speculative_candidate_budget=4,
)
outputs = llm.generate_speculative_mtp_detailed([prompt], sampling_params)
```

`LLM.generate()` stays autoregressive unless the request sets
`speculative_mtp=True`; without an attached provider that request raises
instead of returning plain text. Server routing of the flag is described in
`docs/API.md`. `scripts/gemma4_mtp_probe.py` compares plain and MTP generation
through this API over prompt suites and writes a paired artifact.

## What it measures

Radeon Pro W7900 (`hip_gfx1100`), target `UD-Q4_K_XL`, head `Q8_0`, production
profile, BF16 KV, 18 prompts from the two suites under `benchmarks/prompts/`
(code, English, Japanese, mixed), two samples each, 128 greedy tokens. Rates
are whole-call tokens per second including the prefill of a short prompt, plain
call first, measured 2026-10-13 with `scripts/gemma4_mtp_probe.py`.

| `candidate_budget` | Plain tok/s | MTP tok/s | Ratio | Code | English | Japanese | Mixed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 74.3 | 94.2 | 1.27x | 1.33x | 1.24x | 1.23x | 1.25x |
| 4 | 74.0 | 103.3 | 1.40x | 1.51x | 1.34x | 1.28x | 1.40x |
| 6 | 74.2 | 102.2 | 1.38x | 1.50x | 1.32x | 1.21x | 1.43x |
| 8 | 73.7 | 72.8 | 0.99x | 1.07x | 0.97x | 0.85x | 1.01x |

A budget of 4 is the best measured setting on this device. Artifact:
`benchmarks/results/2026-10-13-gemma4-gfx1100-mtp-status.json`. The head has not
been measured on `hip_gfx1151` on the current tree.

## Open defects

1. **The verify pass is not the arithmetic of one-token decode.** The provider's
   docstring and its telemetry field
   `verify_equivalence: bit_identical_to_target_ar_greedy` state that a verify
   row equals the single-token forward for that position. It does not. After
   the same prefill, forwarding K tokens one at a time and forwarding them as
   one K-row block with `verification=True` gives no bitwise-equal row in either
   the production or the strict profile (prompts of 120, 600 and 1100 tokens, K
   of 3 and 7; row KL from 5.8e-8 to 3.7e-2 in production). The first row of the
   block already differs after layer 0, so the cause is how a multi-row block
   is computed rather than context; the operator that introduces it has not
   been isolated. In the table above, MTP output is token-identical to plain
   greedy on 11 of 18 prompts and diverges on the same 7 at every budget; under
   the strict profile at budget 4 it is identical on 14 of 18. Each MTP output
   is still a greedy continuation under the block arithmetic. Either the verify
   block has to run row-invariant arithmetic, or the equivalence claim has to
   be replaced by the production envelope in `docs/EXECUTION-PROFILES.md` with
   its own measurement. The telemetry label is wrong until one of those lands.
2. **A budget of 8 gives no speedup** where 6 gives 1.38x. The acceptance and
   cycle-length telemetry was not examined for that row, so whether the cost is
   in the nine-row verify block or in acceptance is unmeasured.
3. **Sampled requests are refused.** `hipengine/speculative/sampling.py`
   already implements exact speculative sampling (accept with `min(1, p/q)`,
   resample the residual on rejection), and `hipengine/generation/mtp_sampled_accept.py`
   applies it for the Qwen MTP route. The Gemma provider does not call it. Two
   pieces are missing: a sampled accept walk over the verify rows in
   `hipengine/generation/gemma4_mtp.py`, and a drafter that samples from the
   head's distribution and reports the probability it drew with, since a
   drafter that always offers its top token has `q = 1` and is accepted only
   with probability `p`. llama.cpp added the same pair of changes for its MTP
   and simple draft routes (`ggml-org/llama.cpp` pull request 27694). Its
   RTX 5090 tables report mean accepted length up about 6% to 12% over
   temperatures 0.5 to 1.0 for the five configurations read here (two MTP, three
   draft-model), and no change at temperature 0, where both arms emit the same
   tokens.

## References

- Algorithm and tensor shapes: `llama.cpp@17252c769`,
  `src/models/gemma4-assistant.cpp` — `load_arch_hparams`, `load_arch_tensors`,
  and the graph builder.
- KV binding: same tree, `src/llama-model.cpp` (~line 2578) and
  `src/llama-kv-cache.cpp` (`layer_share_cb`).
- `17252c769` is the commit `benchmarks/README.md` records for the gfx1151
  llama.cpp baseline row, so the reference and the baseline are from one tree.
