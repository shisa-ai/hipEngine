"""GPU tier: grouped Q4T16 and hoisted planar Q6T16 GEMVs are bit-identical
to their parent decode owners.

Strict parent-parity contract: every grouped Q4 output equals the
``dense_single_local32`` owner and every hoisted Q6 output equals the
``q6_k_t16_qmicro_planar`` owner, bit for bit, on random T16 tiles with finite
fp16 scales. Shapes cover the Glimmer projections, other K/N families, uneven
segment mixes, every wave count, and multi-row Q6.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP = True
except OSError:
    _HIP = False

pytestmark = pytest.mark.skipif(not _HIP, reason="ROCm runtime not available")

_Q4_T16_BLOCK_BYTES = 2368
_Q6_T16_BLOCK_BYTES = 3360


def _tiles(rng, in_features, out_features, block_bytes, d_bytes):
    n_tiles = (out_features // 16) * (in_features // 256)
    tiles = rng.integers(0, 256, size=(n_tiles, block_bytes), dtype=np.uint8)
    # Finite, moderate fp16 d (and dmin for Q4) at the head of every tile.
    d = (0x1C00 | rng.integers(0, 0x400, size=(n_tiles, d_bytes // 2))).astype("<u2")
    tiles[:, :d_bytes] = d.view(np.uint8).reshape(n_tiles, d_bytes)
    return tiles


def _bf16_input(rng, n):
    vals = rng.standard_normal(n).astype(np.float32)
    return (vals.view(np.uint32) >> 16).astype(np.uint16)


class _Dev:
    def __init__(self):
        from hipengine.core.hip import get_hip_runtime

        self.rt = get_hip_runtime()
        self.bufs = []

    def upload(self, array):
        from hipengine.core.memory import copy_host_array_to_device, malloc

        buf = malloc(array.nbytes, runtime=self.rt)
        copy_host_array_to_device(buf, np.ascontiguousarray(array), runtime=self.rt)
        self.bufs.append(buf)
        return buf

    def empty(self, nbytes):
        from hipengine.core.memory import malloc

        buf = malloc(nbytes, runtime=self.rt)
        self.bufs.append(buf)
        return buf

    def read_u16(self, buf, n):
        from hipengine.core.memory import copy_device_to_host

        out = np.empty(n, dtype=np.uint16)
        self.rt.device_synchronize()
        copy_device_to_host(out.ctypes.data, buf, runtime=self.rt)
        return out

    def close(self):
        from hipengine.core.memory import free

        for buf in self.bufs:
            free(buf, runtime=self.rt)


@pytest.fixture
def dev():
    d = _Dev()
    yield d
    d.close()


@pytest.mark.parametrize(
    "in_features,out_list",
    [
        (6656, (4096, 256, 256, 4096)),  # Glimmer attention Q/K/V/gate
        (6656, (4096, 256, 4096)),  # Glimmer attention when V is Q6
        (6656, (19968, 19968)),  # Glimmer FFN gate/up
        (4096, (6656,)),  # attn_output
        (19968, (6656,)),  # Q4 ffn_down
        (5120, (6144, 1024, 1024)),  # another hidden-size family
        (256, (16, 48, 32, 16)),  # one superblock, tiny uneven tails
    ],
)
@pytest.mark.parametrize("waves", [1, 2, 4, 8])
def test_grouped_q4_matches_local32_owner(dev, in_features, out_list, waves) -> None:
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_grouped_gemv import (
        gguf_q4_k_t16_tile16_grouped_gemv,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
    )

    rng = np.random.default_rng(in_features * 31 + len(out_list) + waves)
    x = dev.upload(_bf16_input(rng, in_features))
    segments, refs = [], []
    for out_features in out_list:
        w = dev.upload(_tiles(rng, in_features, out_features, _Q4_T16_BLOCK_BYTES, 64))
        ref = dev.empty(out_features * 2)
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
            x.ptr, w.ptr, ref.ptr, 1, in_features, out_features, runtime=dev.rt
        )
        out = dev.empty(out_features * 2)
        segments.append((w.ptr, out.ptr, out_features))
        refs.append((ref, out, out_features))
    gguf_q4_k_t16_tile16_grouped_gemv(x.ptr, segments, in_features, waves=waves, runtime=dev.rt)
    for ref, out, n in refs:
        expected = dev.read_u16(ref, n)
        got = dev.read_u16(out, n)
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [(1, 19968, 6656), (1, 6656, 256), (1, 6656, 4096), (3, 19968, 6656), (2, 256, 48)],
)
def test_hoisted_q6_matches_planar_owner(dev, rows, in_features, out_features) -> None:
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_grouped_gemv import (
        gguf_q6_k_t16_qmicro_planar_hoisted_bf16_bf16_out,
    )

    rng = np.random.default_rng(rows * 7 + in_features + out_features)
    x = dev.upload(_bf16_input(rng, rows * in_features))
    w = dev.upload(_tiles(rng, in_features, out_features, _Q6_T16_BLOCK_BYTES, 32))
    ref = dev.empty(rows * out_features * 2)
    out = dev.empty(rows * out_features * 2)
    gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
        x.ptr, w.ptr, ref.ptr, rows, in_features, out_features, runtime=dev.rt
    )
    gguf_q6_k_t16_qmicro_planar_hoisted_bf16_bf16_out(
        x.ptr, w.ptr, out.ptr, rows, in_features, out_features, runtime=dev.rt
    )
    np.testing.assert_array_equal(
        dev.read_u16(out, rows * out_features), dev.read_u16(ref, rows * out_features)
    )


def test_grouped_q4_rejects_bad_segments(dev) -> None:
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_grouped_gemv import (
        gguf_q4_k_t16_tile16_grouped_gemv,
    )

    with pytest.raises(ValueError):
        gguf_q4_k_t16_tile16_grouped_gemv(0, [], 256, runtime=dev.rt)
    with pytest.raises(ValueError):
        gguf_q4_k_t16_tile16_grouped_gemv(0, [(1, 1, 16)] * 5, 256, runtime=dev.rt)
    with pytest.raises(ValueError):
        gguf_q4_k_t16_tile16_grouped_gemv(0, [(1, 1, 24)], 256, runtime=dev.rt)
    with pytest.raises(ValueError):
        gguf_q4_k_t16_tile16_grouped_gemv(0, [(1, 1, 16)], 256, waves=3, runtime=dev.rt)
