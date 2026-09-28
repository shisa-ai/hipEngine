---
status: current
owns: The `gemma4-assistant` MTP head's complete forward specification — per-layer step order, KV binding, attention scale, and the list of what is still unimplemented. Read before writing the head's forward or its draft/verify loop.
---
# Gemma 4 assistant head (MTP)

The Gemma 4 26B-A4B MTP head is a **separate GGUF artifact with its own
architecture string**, not tensors inside the target. This document is the
resolved specification: the algorithm, the hyperparameters, the tensor layout,
and the one design question that was open — which KV the head's four blocks
attend against. It exists so the port is mechanical rather than a re-derivation.

Nothing here is implemented. `hipengine/generation/gemma4_gguf.py` declares
`supports_speculative_mtp = False`, and that declaration is what the work has to
make true, not a gate that blocks the work.

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

## What is implemented

`hipengine/loading/gemma4_assistant_gguf.py` (311 lines, landed in b42a56c20):
the config decoder, `expected_gemma4_assistant_shapes`, the required-tensor-name
list, `validate_gemma4_assistant_tensor_map`, and `build_gemma4_assistant_tensor_map`.
It follows `hipengine/loading/qwen35_gguf_nextn.py` as the template.

`hipengine/loading/gemma4_assistant_device.py` (landed in 8bb1c31c1): the
device-residency half. `plan_gemma4_assistant_device_specs` plans all 49
tensors, `materialize_gemma4_assistant_device_weights` uploads them
all-or-nothing, and `load_gemma4_assistant_device_weights` does both from a
path. Every tensor is reachable by a forward slot (`blocks.2.attn_q`,
`nextn_pre_projection`), so the forward does not build GGUF names. The real
461.8 MB head loads to 440.5 MB of device allocations, 23 raw Q8_0 blocks and 26
dense F32, with the planned total equal to the artifact's own tensor bytes.

## What is not

1. **The forward.** Four dense blocks at width 1024 with the step order above.
   Every primitive exists in this engine; nothing here needs a new kernel. The
   head's blocks are not `gemma4_layer_forward_bf16` -- that does pre-norm with a
   KV write and a `1/sqrt(head_dim)` scale, where this does post-norm with a
   shared-KV read and scale 1.0 -- so the forward composes
   `launch_gguf_linear`, `gemma4_rmsnorm_f32w_bf16`, the rope tables and the
   attention kernel directly. It also needs the *backbone's* embedding table for
   its input and its own for the logits, and the backbone's layer-28/29 rope
   tables, both of which the backbone runner has to expose.
2. **The KV read path.** Exposing backbone layer 28/29 K and V to the head and
   making the draft step's position agree with them.
3. **The draft/verify loop.** Draft k tokens with the head, verify the whole
   prefix in one target forward, accept the longest matching run.
4. **The adapter registration.** `register_gguf_mtp2_adapter(key, factory)` in
   `hipengine/generation/qwen35_gguf_mtp2_registry.py` already has `dense_nextn`
   and `moe_nextn` for qwen35; the head is a `dense_nextn` shape. Then
   `supports_speculative_mtp` in `hipengine/generation/gemma4_gguf.py` can
   become true.
5. **An acceptance-rate measurement.** `benchmarks/README.md` records MTP decode
   at 25.34 against 12.27 tok/s per request, **2.07x**, for a different model on
   this host. That is the number to compare against and the reason to expect a
   decode win, but this head's acceptance rate on Gemma 4 is unmeasured, and a
   fast head with low acceptance buys little.

## References

- Algorithm and tensor shapes: `llama.cpp@17252c769`,
  `src/models/gemma4-assistant.cpp` — `load_arch_hparams`, `load_arch_tensors`,
  and the graph builder.
- KV binding: same tree, `src/llama-model.cpp` (~line 2578) and
  `src/llama-kv-cache.cpp` (`layer_share_cb`).
- `17252c769` is the commit `benchmarks/README.md` records for the gfx1151
  llama.cpp baseline row, so the reference and the baseline are from one tree.
