"""Unit tier: Muse Glimmer vision reference primitives (hermetic, no artifact)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.muse_glimmer_vision import (
    bilinear_resize_pos,
    downsample_permutation,
    erf_approx,
    gelu_erf,
    grid_size,
    layer_norm,
    pixel_shuffle,
    rope_2d,
    window_mask,
    window_permutations,
)


def test_erf_approx_error_bound() -> None:
    """A&S 7.1.26 stays within 2e-7 of libm erf over the active range."""

    x = np.linspace(-8, 8, 20001).astype(np.float32)
    ref = np.array([math.erf(float(v)) for v in x], dtype=np.float64)
    err = np.abs(erf_approx(x).astype(np.float64) - ref)
    assert err.max() < 2e-7, err.max()


def test_gelu_erf_matches_definition() -> None:
    x = np.array([-3.0, -1.0, 0.0, 0.5, 2.0], dtype=np.float32)
    ref = np.array([0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in x])
    assert np.max(np.abs(gelu_erf(x) - ref)) < 1e-6


def test_grid_size_aligned_identity() -> None:
    # 896x896 -> 32x32 post-merge tokens, under the cap: unchanged.
    assert grid_size(896, 896, 28, 4096) == (896, 896)
    assert grid_size(672, 448, 28, 4096) == (672, 448)


def test_grid_size_downscales_over_cap() -> None:
    # 2800x2800 = 100x100 = 10000 tokens > 4096 cap -> scale to ~64x64.
    w, h = grid_size(2800, 2800, 28, 4096)
    assert (w // 28) * (h // 28) <= 4096
    assert w == h  # square stays square


def test_grid_size_preserves_aspect() -> None:
    w, h = grid_size(1600, 900, 28, 4096)
    assert abs((h / w) - (900 / 1600)) < 0.05


def test_bilinear_resize_identity_and_half_pixel() -> None:
    table = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)  # 4x4 grid
    same = bilinear_resize_pos(table, 4, 4, 4)
    assert np.array_equal(same, table)
    # 2x2 destination from 4x4 source: half-pixel centers sample at 0.5,1.5...
    out = bilinear_resize_pos(table, 4, 2, 2)
    src = table.reshape(4, 4, 3)
    # dst (0,0): src coord (0.5*2-0.5)=0.5 -> average of 4 neighbors at 0/1
    expect = (src[0, 0] + src[0, 1] + src[1, 0] + src[1, 1]) / 4
    assert np.allclose(out[0], expect, atol=1e-6)


def test_window_permutations_cover_and_inverse() -> None:
    sp, inv, pw, ph, slens = window_permutations(5, 4, 32)
    # grid smaller than one window: identity grouping, one window of 20
    assert np.array_equal(sp, np.arange(20))
    assert np.array_equal(inv, np.arange(20))
    assert slens == [20]
    assert pw[7] == (7 % 5) + 1
    assert ph[7] == (7 // 5) + 1
    # multi-window grid: 40x33 -> 2x2 windows, partial edges
    sp2, inv2, _, _, slens2 = window_permutations(40, 33, 32)
    assert sorted(sp2.tolist()) == list(range(40 * 33))
    assert np.array_equal(inv2[sp2], np.arange(40 * 33))
    assert slens2 == [32 * 32, 32 * 8, 32 * 1, 8 * 1]


def test_window_mask_block_diagonal() -> None:
    mask = window_mask(5, [2, 3])
    assert mask[0, 1] == 0.0
    assert mask[2, 4] == 0.0
    assert mask[0, 2] == -np.inf
    assert mask[4, 1] == -np.inf


def test_pixel_shuffle_channel_outer() -> None:
    # 4x4 grid, C=2, f=2 -> 4 outputs of 8 features; feature[c*4+s]
    x = np.arange(16 * 2, dtype=np.float32).reshape(16, 2)
    out = pixel_shuffle(x, 4, 4, 2)
    assert out.shape == (4, 8)
    dsp = downsample_permutation(4, 4, 2)
    for o in range(4):
        for s in range(4):
            for c in range(2):
                assert out[o, c * 4 + s] == x[dsp[o * 4 + s], c]


def test_rope_2d_halves_and_interleave() -> None:
    # head_dim 8: dims [0,4) width, [4,8) height; 2 pairs per half.
    x = np.random.default_rng(1).standard_normal((3, 2, 8)).astype(np.float32)
    pw = np.array([1, 2, 3], dtype=np.int64)
    ph = np.array([4, 5, 6], dtype=np.int64)
    out = rope_2d(x, pw, ph, 8, 10000.0)
    inv = 10000.0 ** (-np.arange(2, dtype=np.float32) * (2.0 / 4))
    for t in range(3):
        for h in range(2):
            for half, pos in ((0, pw[t]), (4, ph[t])):
                for i in range(2):
                    ang = pos * inv[i]
                    c, s = math.cos(ang), math.sin(ang)
                    a = x[t, h, half + 2 * i]
                    b = x[t, h, half + 2 * i + 1]
                    assert abs(out[t, h, half + 2 * i] - (a * c - b * s)) < 2e-5
                    assert abs(out[t, h, half + 2 * i + 1] - (a * s + b * c)) < 2e-5


def test_layer_norm_population_variance_and_bias() -> None:
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    w = np.array([2.0, 1.0, 0.5], dtype=np.float32)
    b = np.array([0.1, -0.1, 0.2], dtype=np.float32)
    out = layer_norm(x, w, b, 1e-5)
    mean, var = 2.0, 2.0 / 3.0
    ref = (x - mean) / math.sqrt(var + 1e-5) * w + b
    assert np.allclose(out, ref, atol=1e-6)