"""The Gemma 4 assistant (MTP) head forward pass.

The head is a four-block draft model that predicts the *next* token from the
target model's last hidden state. It is not a decoder in its own right: it holds
no KV cache, and each of its blocks attends against one of the backbone's last
two layers. The contract is specified in
``docs/reference/GEMMA4-ASSISTANT-MTP.md``; this module implements the forward
from it, and the two facts that are easiest to get wrong are called out at their
call sites below (``wo`` before the post-attention norm, and the input embedding
being the backbone's rather than the head's).

Where this differs from :func:`gemma4_layer_forward_bf16`
-------------------------------------------------------

The head's block is *not* the backbone's block with different weights, so it does
not call the backbone's fused layer:

* the norm order is **post-norm** on both branches -- ``attn_post_norm``
  normalizes the attention output *before* the residual add, and ``post_ffw_norm``
  normalizes the FFN output before its residual add, where the backbone adds then
  normalizes;
* there is no KV write, and K/V come from the backbone;
* the attention scale is 1.0 rather than the backbone's geometry scale (which is
  also 1.0 for Gemma 4, but the head has no geometry object to read it from);
* ``attn_q_norm`` is a per-head norm with no K counterpart.

So the forward composes the primitives directly. Every one of them already
existed; no new kernel was needed.

Batch scope
-----------

This runs one token per call. A draft/verify loop that batches proposals must run
under the ``batch_invariant`` execution profile: the backbone selects its expert
routes by batch width, so a draft pass and a verify pass of the same tokens can
otherwise use different arithmetic. See
``worklog/entries/20260928T010526.507968Z-lhl-gemma4-mmq-divergence-profile-scope-b58b30.md``.
"""

from __future__ import annotations


from dataclasses import dataclass

import numpy as np

from hipengine.kernels.cpu_reference.gemma4 import Gemma4RopeConfig

from hipengine.loading.gemma4_assistant_gguf import Gemma4AssistantConfig
from hipengine.loading.gemma4_gguf import Gemma4GGUFConfig

_BF16_BYTES = 2
_F32_BYTES = 4

# The two backbone layers the head reads, counted back from the end. A
# sliding-window head block reads ``n_layer - 2`` and a full-attention block
# reads ``n_layer - 1``; ``docs/reference/GEMMA4-ASSISTANT-MTP.md`` records the
# cache-construction code that fixes this, and
# ``tests/test_unit_gemma4_assistant_kv_binding.py`` asserts the geometry of the
# two layers against both artifacts.
_SWA_LAYER_FROM_END = 2
_FULL_LAYER_FROM_END = 1


@dataclass(frozen=True)
class Gemma4AssistantGeometry:
    """Resolved attention geometry for one head block.

    ``rope`` is the *bound backbone layer's* rope config rather than a second
    schedule derived from the head's metadata: the two were verified equal, so
    the head reads the backbone's and there is one definition.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope: Gemma4RopeConfig
    sliding_window: int | None
    kv_layer: int

    @property
    def q_width(self) -> int:
        return self.num_heads * self.head_dim


def gemma4_assistant_geometry(
    config: Gemma4AssistantConfig,
    backbone: Gemma4GGUFConfig,
) -> tuple[Gemma4AssistantGeometry, ...]:
    """Resolve the head's per-block geometry against a backbone config.

    Fails closed if the head's block does not match the backbone layer it binds
    to. The equality is a property of how the two artifacts were built rather
    than of either file, so it is checked here at construction instead of being
    assumed and producing silently wrong attention.
    """

    block_count = int(config.block_count)
    if block_count <= 0:
        raise ValueError("the assistant head has no blocks")
    layers = int(backbone.block_count)
    if layers < 2:
        raise ValueError(f"a {layers}-layer backbone has no last two layers to share")
    if block_count > layers:
        raise ValueError(
            f"assistant head has {block_count} blocks, more than the backbone's "
            f"{layers} layers; its blocks cannot each bind a distinct last layer"
        )

    geometry: list[Gemma4AssistantGeometry] = []
    for block_id in range(block_count):
        is_swa = bool(config.is_swa[block_id])
        layer_id = layers - (_SWA_LAYER_FROM_END if is_swa else _FULL_LAYER_FROM_END)
        head_dim = int(config.key_length_swa if is_swa else config.key_length)
        kv_heads = int(config.n_head_kv[block_id])

        if bool(backbone.is_sliding(layer_id)) != is_swa:
            raise ValueError(
                f"assistant block {block_id} binds backbone layer {layer_id}, which is "
                f"{'sliding' if backbone.is_sliding(layer_id) else 'full'} attention "
                f"while the head block is {'sliding' if is_swa else 'full'}"
            )
        if int(backbone.head_count(layer_id)) != int(config.n_head):
            raise ValueError(
                f"assistant block {block_id} has {config.n_head} Q heads but backbone "
                f"layer {layer_id} has {backbone.head_count(layer_id)}"
            )
        if int(backbone.head_count_kv_for(layer_id)) != kv_heads:
            raise ValueError(
                f"assistant block {block_id} declares {kv_heads} KV heads but backbone "
                f"layer {layer_id} carries {backbone.head_count_kv_for(layer_id)}"
            )
        if int(backbone.head_dim(layer_id)) != head_dim:
            raise ValueError(
                f"assistant block {block_id} has head width {head_dim} but backbone "
                f"layer {layer_id} has {backbone.head_dim(layer_id)}"
            )

        geometry.append(
            Gemma4AssistantGeometry(
                num_heads=int(config.n_head),
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                rope=backbone.rope_for_layer(layer_id),
                sliding_window=int(config.sliding_window) if is_swa else None,
                kv_layer=layer_id,
            )
        )
    return tuple(geometry)


def gemma4_assistant_keep_mask(
    position: int,
    keys: int,
    *,
    sliding_window: int | None,
) -> np.ndarray:
    """Return the ``(1, keys)`` uint8 keep-mask for one query row.

    Causal, and windowed when the block is a sliding one: key ``k`` is kept when
    ``k <= position`` and, with a window, when ``k > position - window``. The
    head is single-token, so there is no row-to-row variation to encode and this
    is the same predicate ``_keep_mask`` uses for the backbone.
    """

    if position < 0:
        raise ValueError(f"position must be non-negative, got {position}")
    if keys <= 0:
        raise ValueError(f"keys must be positive, got {keys}")
    index = np.arange(keys, dtype=np.int64)
    keep = index <= position
    if sliding_window is not None:
        keep &= index > position - int(sliding_window)
    return keep.astype(np.uint8).reshape(1, keys)


__all__ = [
    "Gemma4AssistantGeometry",
    "gemma4_assistant_geometry",
    "gemma4_assistant_keep_mask",
]
