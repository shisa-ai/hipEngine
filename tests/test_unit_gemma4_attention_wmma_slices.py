"""How the WMMA full-layer attention picks its query-block shape and divides its walk.

Both rules are pure functions of the geometry, so they are tested here without a
device: `shape_for_tokens` decides which compiled object a query block runs, and
`plan_gemma4_attention_wmma_full_slices` decides how far that object divides its
key walk. What the numbers are for is recorded with each function; what matters
here is the boundaries between the shapes and between the divided and undivided
walks.

The split is a reassociation of the softmax -- each slice takes its own maximum
and the combine rescales every slice to the largest of them -- so it is worth
taking only where the unsplit grid cannot fill the machine. Every prefill shape
keeps the arithmetic it has always run, and these cases pin that.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma_full import (
    DECODE_SHAPE,
    PREFILL_SHAPE,
    K_BATCH,
    plan_gemma4_attention_wmma_full_scratch_bytes,
    plan_gemma4_attention_wmma_full_slices,
    shape_for_tokens,
)

# The full layers of gemma-4-26B-A4B: head_dim 512, 16 query heads, 2 KV heads,
# GQA ratio 8.
FULL_HEADS = 16
FULL_KV_HEADS = 2
GQA_RATIO = FULL_HEADS // FULL_KV_HEADS


def _base_blocks(shape, tokens: int) -> int:
    query_tiles = (tokens + shape.query_rows - 1) // shape.query_rows
    return query_tiles * FULL_KV_HEADS * shape.gqa_tiles


def _slices(tokens: int, keys: int) -> int:
    return plan_gemma4_attention_wmma_full_slices(
        tokens=tokens, keys=keys, num_heads=FULL_HEADS, num_kv_heads=FULL_KV_HEADS
    )


class TestShapeSelection:
    """A one-row query block is a decode, and the shapes are not interchangeable."""

    def test_a_decode_takes_the_shape_that_reads_each_band_once(self):
        """The decode shape's whole reason is `gqa_tiles == 1`."""

        assert shape_for_tokens(1) is DECODE_SHAPE
        assert DECODE_SHAPE.gqa_heads == GQA_RATIO
        assert DECODE_SHAPE.gqa_tiles == 1

    def test_the_prefill_shape_rereads_each_band_once_per_gqa_tile(self):
        assert PREFILL_SHAPE.gqa_heads == 2
        assert PREFILL_SHAPE.gqa_tiles == 4

    def test_a_block_that_fits_the_decode_width_takes_it(self):
        """Two rows is the decode shape's width, so two rows is still a decode."""

        assert shape_for_tokens(DECODE_SHAPE.query_rows) is DECODE_SHAPE
        assert shape_for_tokens(DECODE_SHAPE.query_rows + 1) is PREFILL_SHAPE

    @pytest.mark.parametrize("tokens", [3, 16, 64, 512, 4096])
    def test_anything_wider_is_a_prefill(self, tokens):
        assert shape_for_tokens(tokens) is PREFILL_SHAPE

    @pytest.mark.parametrize("shape", [PREFILL_SHAPE, DECODE_SHAPE])
    def test_both_shapes_are_a_whole_number_of_wmma_tiles(self, shape):
        """`kColumns` is the WMMA M dimension, so it must divide by sixteen."""

        assert shape.columns % 16 == 0
        assert shape.columns == shape.query_rows * shape.gqa_heads
        assert shape.column_groups * 16 == shape.columns
        assert shape.waves == shape.column_groups * shape.dim_groups
        assert shape.threads == 32 * shape.waves

    def test_both_shapes_run_the_same_threads_from_different_groupings(self):
        """The two shapes reach 128 threads by different routes, on purpose.

        `kThreads` is what sets how many loads the staging loop keeps in flight,
        and a pure-read probe over the walk's own address sequence put the
        residual there rather than in the KV layout: 2,048 threads reached 158.8
        GB/s at the walk's 32x64 grid while 1024x256 reached 231.0, and a flat
        contiguous read at the walk's grid was *slower* than the strided one.

        The decode shape has half the prefill shape's column groups, so it needs
        twice the dim groups to reach the same threads -- which is also the
        geometry the staging loop's own comment describes ("128 threads cover
        two rows per iteration", `kRowStep` 2).
        """

        assert PREFILL_SHAPE.column_groups == 2 * DECODE_SHAPE.column_groups
        assert DECODE_SHAPE.dim_groups == 2 * PREFILL_SHAPE.dim_groups
        assert PREFILL_SHAPE.threads == DECODE_SHAPE.threads == 128
        assert 512 % DECODE_SHAPE.dim_groups == 0
        assert 512 % PREFILL_SHAPE.dim_groups == 0

    def test_the_decode_shape_fits_in_shared_memory_and_the_prefill_shape_does_too(self):
        """The 64 KB budget is what the sixteen-row decode shape would have blown.

        `(kColumns * kQStride + kKBatch * kKvStride) * 2` against 65536 bytes:
        32.5 KB for the decode shape, 48.8 KB for the prefill shape, and 81.2 KB
        for the sixteen-row, eight-head shape that is not built.
        """

        assert DECODE_SHAPE.shared_bytes == (16 * 520 + K_BATCH * 520) * 2
        assert PREFILL_SHAPE.shared_bytes == (32 * 520 + K_BATCH * 520) * 2
        assert DECODE_SHAPE.shared_bytes < 65536
        assert PREFILL_SHAPE.shared_bytes < 65536
        assert (16 * 8 * 520 + K_BATCH * 520) * 2 > 65536


class TestSlicePlanning:
    def test_the_decode_grid_is_short_enough_to_divide(self):
        """Two blocks at this geometry, against the decode shape's own target."""

        assert _base_blocks(DECODE_SHAPE, 1) == 2
        assert _base_blocks(DECODE_SHAPE, 1) < DECODE_SHAPE.split_target_blocks
        assert _slices(1, 262144) > 1

    def test_a_single_query_tile_divides_to_about_the_target(self):
        """The slice count is what it takes to reach the target, not more.

        Each shape divides against its own target, because the two move
        different amounts of traffic per block.
        """

        for shape, tokens in ((DECODE_SHAPE, 1), (PREFILL_SHAPE, 16)):
            base = _base_blocks(shape, tokens)
            target = shape.split_target_blocks
            expected = (target + base - 1) // base
            assert base * expected >= target
            assert base * (expected - 1) < target
            assert _slices(tokens, 262144) == expected

    def test_the_decode_shape_does_not_inherit_the_prefill_shapes_target(self):
        """The two shapes' block targets are separate, measured numbers.

        This test used to assert the opposite -- that the decode shape divides
        four times as far as the prefill shape, on the reasoning that it reads
        each band once where the prefill shape reads it four times and so has
        four times less to overlap. That reasoning double-counts. The shape
        change already cut each block's traffic to a quarter; multiplying the
        slice count by four then cut each slice's work to a sixteenth. Sweeping
        the slice count directly, on one full layer at tokens=1, puts the decode
        shape's optimum at 16 slices (32 blocks) and measures the old 64 (128
        blocks) at 1.066x slower at 262,144 keys and 1.578x slower at 16,384.

        Both shapes now land on 16 slices, from different targets and different
        base block counts -- which is the coincidence, not the rule.
        """

        decode = _base_blocks(DECODE_SHAPE, 1)
        prefill = _base_blocks(PREFILL_SHAPE, PREFILL_SHAPE.query_rows)
        assert prefill == decode * 4
        assert DECODE_SHAPE.split_target_blocks < PREFILL_SHAPE.split_target_blocks
        # The decode shape presents a quarter of the blocks and asks for a
        # quarter of the target, so the two land on the same slice count.
        assert _slices(1, 262144) == _slices(PREFILL_SHAPE.query_rows, 262144) == 16
        # The decode shape's target is sized for its own base blocks, not for the
        # prefill shape's: 32 blocks is what 16 slices of 2 base blocks reaches.
        assert DECODE_SHAPE.split_target_blocks == 16 * decode

    @pytest.mark.parametrize("tokens", [PREFILL_SHAPE.query_rows + 1, 64, 128, 512])
    def test_a_multi_tile_query_block_is_never_divided(self, tokens):
        """A prefill block already fills the machine, so it keeps its own arithmetic.

        A 512-row block is 32 query tiles, which is 256 blocks at this geometry --
        twice the target. Dividing it would reassociate the softmax for no gain
        and would move every prefill number this kernel has.
        """

        assert _slices(tokens, 262144) == 1

    @pytest.mark.parametrize("keys", [1, K_BATCH - 1, K_BATCH])
    def test_a_walk_shorter_than_one_batch_per_slice_is_not_divided(self, keys):
        """A slice with no full K batch in it has nothing to overlap.

        `keys // K_BATCH` is the cap, so a walk that cannot give every slice a
        tile stays whole rather than launching blocks that walk nothing.
        """

        want = (DECODE_SHAPE.split_target_blocks + 2 - 1) // 2
        assert _slices(1, keys) == max(1, min(want, keys // K_BATCH))


class TestScratch:
    """One accumulator float per (column, dimension), two state floats per column."""

    def test_the_decode_workspace_covers_every_slice_of_every_plane(self):
        slices = _slices(1, 262144)
        nbytes = plan_gemma4_attention_wmma_full_scratch_bytes(
            slices, tokens=1, num_heads=FULL_HEADS, num_kv_heads=FULL_KV_HEADS, head_dim=512
        )
        columns = DECODE_SHAPE.columns
        expected = slices * FULL_KV_HEADS * DECODE_SHAPE.gqa_tiles * (columns * 512 + columns * 2) * 4
        assert nbytes == expected
        assert 1e6 < nbytes < 1e8

    def test_the_decode_workspace_is_smaller_than_the_prefill_workspace(self):
        """Half the columns and a quarter of the planes, at the same slice count."""

        decode = plan_gemma4_attention_wmma_full_scratch_bytes(
            _slices(1, 262144), tokens=1, num_heads=FULL_HEADS,
            num_kv_heads=FULL_KV_HEADS, head_dim=512,
        )
        prefill = plan_gemma4_attention_wmma_full_scratch_bytes(
            _slices(PREFILL_SHAPE.query_rows, 262144), tokens=PREFILL_SHAPE.query_rows,
            num_heads=FULL_HEADS, num_kv_heads=FULL_KV_HEADS, head_dim=512,
        )
        assert decode < prefill
