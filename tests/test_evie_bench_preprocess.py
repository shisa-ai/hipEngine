"""Regression tests for the EVIE bench's torch-free page preprocessing.

Covers the two layout contracts of `_preprocess_page` that the GPU
fixture cannot expose (it uses real-processor outputs):

1. spatial: patch rows in 2x2 merge-block order;
2. per-patch flattening: (channel, temporal, 16, 16) per the Conv3d
   weight layout (`hipengine/kernels/cpu_reference/evie.py` reshapes
   patches to (-1, 3, 2, 16, 16)).

Both are checked with RGB-differentiated content (equal-channel images
hide channel permutations) against an independently constructed expected
array — the recipe from review entry 821af6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evie_hip_bench import _preprocess_page  # noqa: E402

INFO = {"image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0]}


def _rgb_tiled_image(size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for rp in range(size // 16):
        for cp in range(size // 16):
            v = 4 * rp + cp
            img[rp * 16 : (rp + 1) * 16, cp * 16 : (cp + 1) * 16, 0] = v
            img[rp * 16 : (rp + 1) * 16, cp * 16 : (cp + 1) * 16, 1] = 80 + v
            img[rp * 16 : (rp + 1) * 16, cp * 16 : (cp + 1) * 16, 2] = 160 + v
    return img


def _expected_patches(img: np.ndarray) -> np.ndarray:
    chw = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    rows = []
    # 2x2 merge-block order: block raster, then inner raster
    for br in range(h // 32):
        for bc in range(w // 32):
            for r in range(2):
                for c in range(2):
                    pr, pc = br * 2 + r, bc * 2 + c
                    tile = chw[:, pr * 16 : (pr + 1) * 16, pc * 16 : (pc + 1) * 16]
                    # per-patch flatten: (channel, temporal, 16, 16)
                    rows.append(np.stack([tile, tile], axis=1).reshape(-1))
    return np.array(rows)


def test_patch_contents_channel_temporal_layout() -> None:
    img = _rgb_tiled_image(64)
    got, grid = _preprocess_page(img, INFO)
    expected = _expected_patches(img)
    assert got.shape == expected.shape
    assert grid.tolist() == [[1, 4, 4]]
    np.testing.assert_allclose(got, expected, atol=0)


def test_white_image_temporal_slots_identical() -> None:
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    got, _ = _preprocess_page(img, INFO)
    two = got.reshape(-1, 2, 3 * 16 * 16)
    np.testing.assert_array_equal(two[:, 0], two[:, 1])


def test_channel_permutation_is_detectable() -> None:
    """Guard the guard: unequal RGB values must expose a wrong channel order."""

    img = _rgb_tiled_image(64)
    good, _ = _preprocess_page(img, INFO)
    # channel-swapped variant of the same data must NOT match
    bad = good.reshape(-1, 3, 2, 16, 16).swapaxes(1, 2).reshape(-1, 3 * 2 * 16 * 16)
    assert not np.allclose(good, bad)


@pytest.mark.parametrize("size", [64, 96])
def test_patch_row_count_and_grid(size: int) -> None:
    img = _rgb_tiled_image(size)
    got, grid = _preprocess_page(img, INFO)
    ph, pw = size // 16, size // 16
    assert got.shape == (ph * pw, 3 * 2 * 16 * 16)
    assert grid.tolist() == [[1, ph, pw]]
