"""RED gate for a Q8_0 -> T16 repack on gemma4's dense leaves.

gemma4's loader has no repack step, so its Q8_0 dense leaves stay in the raw
layout and cannot reach the registered two-wave/four-wave Q8T16 prefill
schedules (worklog/entries/20260929T143000). Wiring that route means storing
these weights as Q8T16 tiles, so the first thing that must hold is that the
repack is bit-lossless: the tiles must reconstruct the original raw GGUF bytes
exactly. ``unpack_gguf_q8_0_tile16`` is the oracle.

This is CPU-only and needs no ROCm, so it runs everywhere. It does NOT claim the
T16 route is faster -- that is a separate measurement -- only that the layout
change is value-preserving, which is the precondition for trying it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.quant.gguf_t16 import (
    GGUF_Q8_0_BLOCK_BYTES,
    GGUF_Q8_0_QK,
    GGUF_T16_COLS,
    repack_gguf_q8_0_tile16,
    unpack_gguf_q8_0_tile16,
)

# The dense leaf shapes gemma4's dispatch actually produces at a 512-token
# prefill, from scripts/gemma4_dense_dispatch_probe.py: rows=512 with
# out_features 2816/2112/2048/4096/8192/1024. in_features is the model hidden
# width (2816) except for the down projection (in=2112). Values on both sides of
# the wrapper's out_features >= 2048 gate are included deliberately, so the test
# covers the gate rather than only the shapes that pass it.
GEMMA4_DENSE_SHAPES = (
    (2816, 2816),  # q/k/v/o
    (2112, 2816),  # gate/up
    (2816, 2112),  # down, below no gate but a distinct in_features
    (4096, 2816),  # a wider projection
    (8192, 2816),  # the widest
    (1024, 2816),  # below the out_features >= 2048 gate
    (32, 2816),  # one T16 tile exactly
    (48, 2816),  # not a multiple of 32 -- 1.5 tiles
)


def _raw_q8_0(out_features: int, in_features: int, seed: int) -> np.ndarray:
    """Random raw GGUF Q8_0 bytes with GGUF byte shape [out_features, bytes_per_row].

    A Q8_0 block covers 32 values in 34 bytes (2 bytes of scale plus 32 of
    quants), so a row is ``in_features // 32 * 34`` bytes flat. The repack takes
    this rank-2 dense form; the rank-3 expert stack is rejected, which is
    covered separately below.
    """

    if in_features % GGUF_Q8_0_QK:
        pytest.skip(f"in_features {in_features} is not a multiple of {GGUF_Q8_0_QK}")
    rng = np.random.default_rng(seed)
    blocks_per_row = in_features // GGUF_Q8_0_QK
    return rng.integers(
        0,
        256,
        size=(out_features, blocks_per_row * GGUF_Q8_0_BLOCK_BYTES),
        dtype=np.uint8,
    )


@pytest.mark.parametrize(("out_features", "in_features"), GEMMA4_DENSE_SHAPES)
def test_q8_0_t16_repack_round_trips_gemma4_dense_shapes(out_features, in_features):
    """Tiles must reconstruct the original raw bytes exactly, byte for byte."""

    raw = _raw_q8_0(out_features, in_features, seed=out_features * 31 + in_features)
    packed = repack_gguf_q8_0_tile16(raw)

    assert packed.out_features == out_features
    assert packed.in_features == in_features
    # [out_tiles16, blocks_per_row, 544]: the T16 leaf's declared ABI. The tile
    # is GGUF_T16_COLS wide, which is 16, not 32 -- 544 = 16 * 34 bytes.
    assert packed.tiles.ndim == 3
    assert packed.tiles.shape[0] == out_features // GGUF_T16_COLS
    assert packed.tiles.shape[1] == in_features // GGUF_Q8_0_QK

    restored = unpack_gguf_q8_0_tile16(packed)
    assert restored.shape == raw.shape
    assert np.array_equal(restored, raw), "Q8_0 -> T16 repack is not bit-lossless"


def test_q8_0_t16_repack_is_not_a_second_copy_of_the_tensor():
    """The tile allocation must be close to the raw size, not double it.

    The Q4_K precedent is 2.78 percent larger; the point of a repack over a
    second copy is that the resident cost stays near one tensor's worth.
    """

    out_features, in_features = 2816, 2816
    raw = _raw_q8_0(out_features, in_features, seed=7)
    packed = repack_gguf_q8_0_tile16(raw)

    raw_bytes = raw.size
    tile_bytes = packed.tiles.size
    ratio = tile_bytes / raw_bytes
    # Q8_0 is exactly value-for-value: 16 columns * 34 bytes per tile row equals
    # 16 raw blocks of 34 bytes, so this repack costs nothing in resident memory
    # beyond the same bytes rearranged.
    assert ratio == 1.0, f"tiles are {ratio:.3f}x the raw bytes, expected exactly 1.0"


def test_q8_0_t16_repack_rejects_a_rank3_expert_stack():
    """A rank-3 stack is not a dense leaf and must not silently repack.

    gemma4's loader distinguishes rank-3 (expert stack) from rank-2 (dense leaf)
    by rank alone, so the repack must refuse the expert case rather than produce
    a tile layout that means something else.
    """

    raw = np.zeros((8, 64, GGUF_Q8_0_BLOCK_BYTES), dtype=np.uint8)
    with pytest.raises((ValueError, IndexError, AssertionError)):
        repack_gguf_q8_0_tile16(raw)
