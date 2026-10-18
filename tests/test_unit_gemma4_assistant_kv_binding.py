"""The Gemma 4 MTP head's binding to the backbone's last two layers.

The head creates no ``attn_k``/``attn_v`` and no KV cache of its own. Every head
block attends against one of the backbone's last two layers: a sliding-window
block against layer ``n_layer - 2`` and a full-attention block against layer
``n_layer - 1``. This module asserts that binding holds between the two real
artifacts, because nothing in either file states it -- it is a property of how
the two were built, and the reference expresses it only in the cache
construction, not in the head's own tensors.

Three equalities, each of which the forward depends on:

* the head block's Q head count, KV head count and head width equal the backbone
  layer's, so the shared cache needs no head remapping;
* the head's sliding-window flag matches the backbone layer's, so a block reads a
  layer with the same attention kind;
* the head's rope geometry equals the backbone layer's, so the head can read the
  backbone's tables instead of deriving a second schedule.

No HIP is involved: both sides are metadata plus one 1 KB ``rope_freqs`` tensor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import scan_gguf
from hipengine.loading.gemma4_assistant_gguf import (
    gemma4_assistant_config_from_metadata,
)
from hipengine.loading.gemma4_gguf import gemma4_gguf_config_from_metadata

BACKBONE = Path(
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)
HEAD = Path("/models/gguf/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it-Q8_0.gguf")

_needs_artifacts = pytest.mark.skipif(
    not (BACKBONE.exists() and HEAD.exists()),
    reason="the Gemma 4 backbone and MTP head are not both downloaded",
)


def _both():
    backbone = gemma4_gguf_config_from_metadata(scan_gguf(BACKBONE))
    head = gemma4_assistant_config_from_metadata(scan_gguf(HEAD).metadata)
    return backbone, head


def _bound_layer(backbone, head, block_id: int) -> int:
    """The backbone layer head block ``block_id`` attends against."""

    return backbone.block_count - 2 if head.is_swa[block_id] else backbone.block_count - 1


@_needs_artifacts
def test_every_head_block_binds_to_a_backbone_layer_with_the_same_geometry() -> None:
    """Q heads, KV heads and head width all agree, block for block."""

    backbone, head = _both()
    assert backbone.block_count == 30, (
        "the layer-28/29 binding below assumes a 30-layer backbone; if the "
        "backbone changed, the two layers the head reads changed with it"
    )

    for block_id in range(head.block_count):
        layer_id = _bound_layer(backbone, head, block_id)
        head_dim = head.key_length_swa if head.is_swa[block_id] else head.key_length
        assert backbone.is_sliding(layer_id) == head.is_swa[block_id], (
            f"head block {block_id} reads backbone layer {layer_id}, whose "
            f"attention kind disagrees with the head block's"
        )
        assert backbone.head_count(layer_id) == head.n_head, block_id
        assert backbone.head_count_kv_for(layer_id) == head.n_head_kv[block_id], (
            f"head block {block_id} reads backbone layer {layer_id}, which carries "
            f"{backbone.head_count_kv_for(layer_id)} KV heads, but the head block "
            f"declares {head.n_head_kv[block_id]}"
        )
        assert backbone.head_dim(layer_id) == head_dim, block_id


@_needs_artifacts
def test_the_two_bound_layers_differ_in_kv_head_count() -> None:
    """The binding is not trivially satisfied by one uniform layer.

    If both bound layers had the same KV head count, the equality above would
    pass for a head that ignored its own per-block ``head_count_kv`` entirely.
    The backbone's pattern is ``[8, 8, 8, 8, 8, 2]`` repeating, so the two bound
    layers are 8 and 2, and the head's ``[8, 8, 8, 2]`` has to track that.
    """

    backbone, head = _both()
    swa_layer = _bound_layer(backbone, head, 0)
    full_layer = _bound_layer(backbone, head, head.block_count - 1)
    assert backbone.head_count_kv_for(swa_layer) == 8
    assert backbone.head_count_kv_for(full_layer) == 2
    assert head.n_head_kv == (8, 8, 8, 2)


@_needs_artifacts
def test_head_rope_geometry_equals_the_backbone_layers_it_reads() -> None:
    """The head wants the backbone's own tables, not merely compatible ones."""

    backbone, head = _both()
    for block_id in range(head.block_count):
        layer_id = _bound_layer(backbone, head, block_id)
        rope = backbone.rope_for_layer(layer_id)
        if head.is_swa[block_id]:
            assert rope.head_dim == head.key_length_swa
            # A sliding-window block passes no freq_factors, so every pair of its
            # half-width table rotates.
            assert rope.rotated_pairs == head.key_length_swa // 2
            assert rope.rope_type == "default"
        else:
            assert rope.head_dim == head.key_length
            # The head's own rope_freqs encodes 64 rotated pairs of 256, which is
            # what makes this "proportional" rather than a full rotation.
            assert rope.rotated_pairs == 64
            assert rope.rotated_pairs < head.key_length // 2
            assert rope.rope_type == "proportional"
