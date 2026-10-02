"""How the WMMA full-layer attention divides its key walk.

The rule is a pure function of the geometry, so it is tested here without a
device: `plan_gemma4_attention_wmma_full_slices` decides, and the launcher only
lays out the workspace and sizes the grid. What the numbers are for is recorded
with the function; what matters here is the boundary between the shapes that get
divided and the shapes that must not be.

The split is a reassociation of the softmax -- each slice takes its own maximum
and the combine rescales every slice to the largest of them -- so it is worth
taking only where the unsplit grid cannot fill the machine. Every prefill shape
keeps the arithmetic it has always run, and these cases pin that.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma_full import (
    GQA_HEADS,
    K_BATCH,
    QUERY_ROWS,
    TARGET_BLOCKS,
    plan_gemma4_attention_wmma_full_scratch_bytes,
    plan_gemma4_attention_wmma_full_slices,
)

# The full layers of gemma-4-26B-A4B: head_dim 512, 16 query heads, 2 KV heads,
# GQA ratio 8, so `gqa_tiles` is 8 / 2 = 4.
FULL_HEADS = 16
FULL_KV_HEADS = 2
GQA_TILES = (FULL_HEADS // FULL_KV_HEADS + GQA_HEADS - 1) // GQA_HEADS
BASE_BLOCKS = FULL_KV_HEADS * GQA_TILES  # query_tiles == 1


def _slices(tokens: int, keys: int) -> int:
    return plan_gemma4_attention_wmma_full_slices(
        tokens=tokens, keys=keys, num_heads=FULL_HEADS, num_kv_heads=FULL_KV_HEADS
    )


def test_the_decode_grid_is_short_enough_to_divide():
    """A one-tile query block presents `BASE_BLOCKS` blocks against the target.

    This is the case the split exists for: 8 blocks at this geometry, each
    walking every key, measured at 75 GB/s where 64 blocks reach 225.
    """

    assert BASE_BLOCKS == 8
    assert BASE_BLOCKS < TARGET_BLOCKS
    assert _slices(1, 262144) > 1


def test_a_single_query_tile_divides_to_about_the_target():
    """The slice count is what it takes to reach the target, not more."""

    slices = _slices(1, 262144)
    assert slices == (TARGET_BLOCKS + BASE_BLOCKS - 1) // BASE_BLOCKS
    assert BASE_BLOCKS * slices >= TARGET_BLOCKS


@pytest.mark.parametrize("tokens", [QUERY_ROWS + 1, 64, 128, 512])
def test_a_multi_tile_query_block_is_never_divided(tokens):
    """A prefill block already fills the machine, so it keeps its own arithmetic.

    A 512-row block is 32 query tiles, which is 256 blocks at this geometry --
    twice the target. Dividing it would reassociate the softmax for no gain and
    would move every prefill number this kernel has.
    """

    assert _slices(tokens, 262144) == 1


@pytest.mark.parametrize("tokens", [1, QUERY_ROWS])
def test_a_single_query_tile_divides_for_any_short_walk(tokens):
    assert _slices(tokens, 65536) > 1


@pytest.mark.parametrize("keys", [1, K_BATCH - 1, K_BATCH])
def test_a_walk_shorter_than_one_batch_per_slice_is_not_divided(keys):
    """A slice with no full K batch in it has nothing to overlap.

    `keys // K_BATCH` is the cap, so a walk that cannot give every slice a tile
    stays whole rather than launching blocks that walk nothing.
    """

    slices = _slices(1, keys)
    assert slices == max(1, min((TARGET_BLOCKS + BASE_BLOCKS - 1) // BASE_BLOCKS, keys // K_BATCH))


def test_the_scratch_covers_every_slice_of_every_plane():
    """One accumulator float per (column, dimension), two state floats per column."""

    slices = _slices(1, 262144)
    nbytes = plan_gemma4_attention_wmma_full_scratch_bytes(
        slices, num_heads=FULL_HEADS, num_kv_heads=FULL_KV_HEADS, head_dim=512
    )
    columns = QUERY_ROWS * GQA_HEADS
    expected = slices * BASE_BLOCKS * (columns * 512 + columns * 2) * 4
    assert nbytes == expected
    # At this geometry a 256K decode needs a few tens of MB, not the gigabytes a
    # whole-context workspace would.
    assert 1e6 < nbytes < 1e8
