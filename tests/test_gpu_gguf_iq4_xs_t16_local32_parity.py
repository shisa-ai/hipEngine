"""The IQ4_XS T16-layout decode owner must be bit-identical to the raw owner.

The T16 tile changes only how the qs plane is arranged: raw stores
``[group][byte]`` per column, T16 stores ``[group][element][column pair]`` so one
u32 load at a fixed element serves eight output columns. The kernel keeps the
raw owner's lane ownership, FMA order, shuffle tree and cross-wave sum, so the
contract is exact equality of the bf16 outputs, not a tolerance.

That makes this the RED contract for the layout's first consumer: a wrong-but-
self-consistent nibble or column-pair convention would still round-trip through
the packer's own inverse, but it cannot reproduce the raw owner's numbers.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense
from hipengine.kernels.hip_gfx1100.quant.gguf_iq4_xs_t16_local32 import (
    launch_iq4_xs_t16_local32,
)
from hipengine.quant.gguf_t16 import repack_gguf_iq4_xs_tile16

FIXTURE = Path(__file__).parent / "fixtures/gguf_ud"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


def bf16(x):
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((b + 0x7fff + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def _real_raw():
    """Real IQ4_XS GGUF bytes: 64 columns of a K=17408 fixture row."""

    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if entry is None:
        pytest.skip("no IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"IQ4_XS fixture K={k} is not block-aligned")
    return np.ascontiguousarray(source[np.arange(64) % len(source)]), k


def _random_raw(columns: int, k: int, seed: int):
    """Arbitrary payload bytes with a finite per-block scale.

    Random bytes exercise every nibble and every scale in -32..31, including the
    saturated ends. The f16 ``d`` field is forced finite so no output is NaN,
    which would make an exact comparison vacuous.
    """

    bpr = k // 256
    raw = np.random.default_rng(seed).integers(
        0, 256, (columns, bpr * 136), dtype=np.uint8
    )
    blocks = raw.reshape(columns, bpr, 136)
    blocks[:, :, 0:2] = np.array([0x00, 0x3C], dtype=np.uint8)  # f16 1.0, little endian
    return np.ascontiguousarray(blocks.reshape(columns, bpr * 136))


@pytest.fixture(scope="module")
def device():
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )

    bufs = []

    def dev(a):
        b = malloc(a.nbytes)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(np.ascontiguousarray(a)), a.nbytes)
        return b

    def read(buffer, nbytes):
        out = np.zeros(nbytes, dtype=np.uint8)
        copy_device_to_host(host_array_ptr(out), buffer, nbytes)
        return out

    try:
        yield dev, read, malloc
    finally:
        for b in reversed(bufs):
            free(b)


def _run_both(raw, k, rows, dev, read, malloc, *, x=None):
    """Run the raw owner and the T16 owner on the same weights and activations."""

    columns = len(raw)
    tiles = repack_gguf_iq4_xs_tile16(raw).tiles
    if x is None:
        x = bf16(np.random.default_rng(11).normal(0, 0.1, (rows, k)))
    x = np.ascontiguousarray(x)
    assert x.shape == (rows, k), x.shape

    x_b = dev(x)
    raw_b = dev(raw)
    tile_b = dev(tiles)
    ref_b = malloc(rows * columns * 2)
    got_b = malloc(rows * columns * 2)

    lib = gguf_iq_dense.build_gguf_iq_dense()
    if rows == 1:
        gguf_iq_dense.launch_local32(x_b.ptr, raw_b.ptr, ref_b.ptr, 1, k, columns,
                                     library=lib)
    else:
        gguf_iq_dense.launch_local32_rows(x_b.ptr, raw_b.ptr, ref_b.ptr, rows, k,
                                          columns, library=lib)
    launch_iq4_xs_t16_local32(x_b.ptr, tile_b.ptr, got_b.ptr, rows, k, columns,
                              library=lib)
    return (read(ref_b, rows * columns * 2), read(got_b, rows * columns * 2))


@pytest.mark.parametrize("columns,k", ((64, 17408), (64, 5120)))
def test_t16_owner_is_bit_identical_to_the_raw_owner(columns, k, device):
    dev, read, malloc = device
    raw, fixture_k = _real_raw()
    raw = raw[:columns]
    if fixture_k != k:
        raw = _random_raw(columns, k, seed=k)
    ref, got = _run_both(raw, k, 1, dev, read, malloc)
    assert np.array_equal(ref.view(np.uint16), got.view(np.uint16)), (
        "the T16 consumer must reproduce the raw owner's bf16 outputs exactly"
    )


def test_t16_owner_is_bit_identical_on_random_payloads(device):
    """Every nibble and every scale in -32..31, not just the fixture's values."""

    dev, read, malloc = device
    raw = _random_raw(64, 5120, seed=29)
    ref, got = _run_both(raw, 5120, 1, dev, read, malloc)
    assert np.array_equal(ref.view(np.uint16), got.view(np.uint16))


@pytest.mark.parametrize("rows", (2, 3, 4))
def test_t16_rows_sibling_matches_the_raw_rows_sibling(rows, device):
    """rows > 1 is the verifier sibling: same parity contract as rows == 1."""

    dev, read, malloc = device
    raw = _random_raw(64, 5120, seed=rows)
    ref, got = _run_both(raw, 5120, rows, dev, read, malloc)
    assert np.array_equal(ref.view(np.uint16), got.view(np.uint16))


@pytest.mark.parametrize("rows", (2, 3, 4))
def test_t16_rows_sibling_repeats_the_rows_one_owner(rows, device):
    """Each row of the sibling must equal the rows==1 owner on that same row."""

    dev, read, malloc = device
    k = 5120
    raw = _random_raw(64, k, seed=rows)
    x = bf16(np.random.default_rng(11).normal(0, 0.1, (rows, k)))
    multi, _ = _run_both(raw, k, rows, dev, read, malloc, x=x)
    multi = multi.view(np.uint16).reshape(rows, 64)
    for r in range(rows):
        single, _ = _run_both(raw, k, 1, dev, read, malloc, x=x[r:r + 1])
        assert np.array_equal(single.view(np.uint16).reshape(64), multi[r]), (
            f"row {r} drifted from the rows==1 owner"
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        dict(rows=0),
        dict(rows=5),
        dict(in_features=4992),
        dict(out_features=8),
        dict(x_ptr=0),
        dict(qweight_ptr=0),
        dict(out_ptr=0),
    ),
)
def test_rejects_invalid_arguments_before_building(kwargs, monkeypatch):
    monkeypatch.setattr(
        gguf_iq_dense, "_default_library", lambda: pytest.fail("built before validation")
    )
    with pytest.raises(ValueError):
        launch_iq4_xs_t16_local32(
            kwargs.pop("x_ptr", 1), kwargs.pop("qweight_ptr", 2),
            kwargs.pop("out_ptr", 3),
            kwargs.pop("rows", 1),
            kwargs.pop("in_features", 5120),
            kwargs.pop("out_features", 17408),
        )
