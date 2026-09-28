"""Geometry resolution and keep-mask for the Gemma 4 assistant (MTP) head.

The head binds each of its four blocks to one of the backbone's last two layers.
``gemma4_assistant_geometry`` resolves that binding and refuses to resolve one
that does not match, because the match is a property of how the two artifacts
were built rather than of either file: nothing in the head says "read layer 28",
and nothing in the backbone says "an assistant head reads me".

The keep-mask half is the head's own, not the backbone's: the head is
single-token, so its mask has one row, but it is still causal and still windowed
on the three sliding blocks.

No HIP: geometry is metadata, and the mask is numpy.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.gguf import scan_gguf
from hipengine.loading.gemma4_assistant_gguf import (
    gemma4_assistant_config_from_metadata,
)
from hipengine.loading.gemma4_gguf import gemma4_gguf_config_from_metadata
from hipengine.runtime.gemma4_assistant import (
    gemma4_assistant_geometry,
    gemma4_assistant_keep_mask,
)

BACKBONE = Path(
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)
HEAD = Path("/models/gguf/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it-Q8_0.gguf")

_needs_artifacts = pytest.mark.skipif(
    not (BACKBONE.exists() and HEAD.exists()),
    reason="the Gemma 4 backbone and MTP head are not both downloaded",
)


def _configs():
    backbone = gemma4_gguf_config_from_metadata(scan_gguf(BACKBONE))
    head = gemma4_assistant_config_from_metadata(scan_gguf(HEAD).metadata)
    return backbone, head


@_needs_artifacts
def test_geometry_binds_three_sliding_blocks_to_layer_28_and_one_to_29() -> None:
    backbone, head = _configs()
    geometry = gemma4_assistant_geometry(head, backbone)

    assert len(geometry) == 4
    assert [block.kv_layer for block in geometry] == [28, 28, 28, 29]
    assert [block.sliding_window for block in geometry] == [1024, 1024, 1024, None]

    # Head width follows the attention kind, not the head's own single value.
    assert [block.head_dim for block in geometry] == [256, 256, 256, 512]
    assert [block.num_kv_heads for block in geometry] == [8, 8, 8, 2]
    assert all(block.num_heads == 16 for block in geometry)
    assert [block.q_width for block in geometry] == [4096, 4096, 4096, 8192]

    # The rope configs are the bound layers' own, so the sliding blocks share one
    # schedule and the full block has its own.
    assert geometry[0].rope == geometry[1].rope == geometry[2].rope
    assert geometry[0].rope != geometry[3].rope
    assert geometry[0].rope.rotated_pairs == 128
    assert geometry[3].rope.rotated_pairs == 64


@_needs_artifacts
def test_geometry_refuses_a_head_block_the_backbone_layer_does_not_match() -> None:
    """A binding that does not match is an error, not silently wrong attention."""

    backbone, head = _configs()

    # A head that asks for the wrong KV head count on the full block.
    wrong_kv = replace(head, n_head_kv=(8, 8, 8, 8))
    with pytest.raises(ValueError, match="KV heads"):
        gemma4_assistant_geometry(wrong_kv, backbone)

    # A head that asks for the wrong Q head count.
    wrong_q = replace(head, n_head=8)
    with pytest.raises(ValueError, match="Q heads"):
        gemma4_assistant_geometry(wrong_q, backbone)

    # A head whose full block is marked sliding, so it binds layer 28 while
    # still declaring the full block's 2 KV heads. Any refusal is the point: the
    # specific message is not asserted because the backbone's two bound layers
    # differ in kind *and* in KV head count, so flipping the kind alone trips
    # whichever check runs first. Isolating the kind check would need a backbone
    # whose two layers agree on head counts and differ only in kind, which no
    # real Gemma 4 artifact is.
    wrong_kind = replace(head, is_swa=(True, True, True, True))
    with pytest.raises(ValueError):
        gemma4_assistant_geometry(wrong_kind, backbone)


@_needs_artifacts
def test_geometry_refuses_a_backbone_with_no_last_two_layers() -> None:
    backbone, head = _configs()
    with pytest.raises(ValueError, match="last two layers"):
        gemma4_assistant_geometry(head, replace(backbone, block_count=1))


def test_keep_mask_is_causal_and_windowed() -> None:
    """The mask keeps past keys, drops future ones, and honours the window."""

    # No window: every key up to the position, nothing after.
    mask = gemma4_assistant_keep_mask(3, 6, sliding_window=None)
    assert mask.shape == (1, 6)
    assert mask.tolist() == [[1, 1, 1, 1, 0, 0]]

    # A window of 2 at position 5 over 6 keys keeps keys 4 and 5 only: key 3 is
    # exactly `position - window` and the predicate is strict there.
    mask = gemma4_assistant_keep_mask(5, 6, sliding_window=2)
    assert mask.tolist() == [[0, 0, 0, 0, 1, 1]]

    # The first token keeps only itself, window or not.
    assert gemma4_assistant_keep_mask(0, 4, sliding_window=None).tolist() == [[1, 0, 0, 0]]
    assert gemma4_assistant_keep_mask(0, 4, sliding_window=1024).tolist() == [[1, 0, 0, 0]]

    # A window wider than the context is the unwindowed mask.
    assert gemma4_assistant_keep_mask(3, 4, sliding_window=1024).tolist() == [[1, 1, 1, 1]]


def test_keep_mask_matches_the_predicate_over_a_range() -> None:
    """Check the whole space, not the gate points.

    The windowed predicate has an off-by-one at `position - window` that a test
    at only one position would miss, so this walks every position and a range of
    windows around the boundary.
    """

    for position in range(0, 9):
        for window in (1, 2, 3, 8, 16):
            keys = 10
            mask = gemma4_assistant_keep_mask(position, keys, sliding_window=window)[0]
            expected = np.array(
                [1 if (k <= position and k > position - window) else 0 for k in range(keys)],
                dtype=np.uint8,
            )
            assert mask.tolist() == expected.tolist(), (position, window)
            # A sliding row always keeps its own position.
            assert mask[position] == 1, (position, window)


def test_keep_mask_refuses_an_empty_or_negative_request() -> None:
    with pytest.raises(ValueError):
        gemma4_assistant_keep_mask(-1, 4, sliding_window=None)
    with pytest.raises(ValueError):
        gemma4_assistant_keep_mask(0, 0, sliding_window=None)
