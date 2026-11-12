"""Multi-row mixed-pair parity against scalar launches and a CPU safety oracle."""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
except OSError:
    pytest.skip("HIP runtime unavailable", allow_module_level=True)

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
    build_gguf_iq4_q4_pair,
    gguf_iq4_q4_pair_silu_bf16_bf16_out,
    gguf_q4_iq4_pair_silu_bf16_bf16_out,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
from tests._gguf_synthetic_weights import make_q4_k_weight


def _bf16_bits(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_float(x):
    return (x.astype(np.uint32) << 16).view(np.float32)


def mixed_pair_inputs(rows: int, k: int, n: int):
    fixture = Path(__file__).parent / "fixtures/gguf_ud"
    entries = json.loads((fixture / "real_rows.json").read_text())["entries"]
    entry = next(e for e in entries if e["type"] == "IQ4_XS" and e["shape"][1] == 5120)
    with np.load(fixture / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"].reshape(-1, 136)
    iq4 = np.ascontiguousarray(np.resize(source, (n * (k // 256), 136)))
    q4 = np.tile(make_q4_k_weight(16, k), (n // 16, 1))
    tiles = np.ascontiguousarray(repack_gguf_q4_k_tile16(q4[None, ...]).tiles)
    x = np.random.default_rng(7301).normal(0, 0.1, (rows, k))
    return _bf16_bits(x), iq4, tiles


def _check_cpu_floor(arrays, actual, *, reverse, k, n):
    x_bits, iq4, _ = arrays
    x = _bf16_float(x_bits)
    iq = gguf_quant_gemv(x, iq4.reshape(n, -1), GGMLQuantizationType.IQ4_XS)
    q4 = gguf_quant_gemv(
        x, np.tile(make_q4_k_weight(16, k), (n // 16, 1)), GGMLQuantizationType.Q4_K,
    )
    gate, up = (q4, iq) if reverse else (iq, q4)
    gate = _bf16_float(_bf16_bits(gate))
    up = _bf16_float(_bf16_bits(up))
    with np.errstate(over="ignore"):
        expected = _bf16_float(_bf16_bits(gate / (1 + np.exp(-gate)) * up))
    candidate = _bf16_float(actual)
    assert np.isfinite(candidate).all()

    def log_softmax(value):
        value = value.astype(np.float64)
        value -= value.max(axis=-1, keepdims=True)
        return value - np.log(np.exp(value).sum(axis=-1, keepdims=True))

    ref_log = log_softmax(expected)
    got_log = log_softmax(candidate)
    assert np.max(np.sum(np.exp(ref_log) * (ref_log - got_log), axis=-1)) <= 0.05
    assert np.mean(expected.argmax(axis=-1) == candidate.argmax(axis=-1)) >= 0.90


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 7, 8, 13])
@pytest.mark.parametrize("k,n", [(256, 16), (2048, 64), (5120, 64), (5120, 8192)])
def test_mixed_pair_batch_matches_scalar_rows(reverse, rows, k, n):
    runtime = get_hip_runtime()
    arrays = mixed_pair_inputs(rows, k, n)
    bufs = []
    try:
        for array in (*arrays, np.zeros((rows, n), np.uint16), np.zeros((rows, n), np.uint16)):
            buf = malloc(array.nbytes, runtime=runtime)
            bufs.append(buf)
            copy_host_array_to_device(buf, array, runtime=runtime)
        x, iq4, q4, ref, got = bufs
        fn = (gguf_q4_iq4_pair_silu_bf16_bf16_out if reverse
              else gguf_iq4_q4_pair_silu_bf16_bf16_out)
        a, b = (q4, iq4) if reverse else (iq4, q4)
        lib = build_gguf_iq4_q4_pair(load=True)
        for row in range(rows):
            fn(x.ptr + row * k * 2, a.ptr, b.ptr, ref.ptr + row * n * 2,
               1, k, n, library=lib, runtime=runtime)
        expected = np.empty((rows, n), np.uint16)
        actual = np.empty_like(expected)
        for _ in range(2):
            fn(x.ptr, a.ptr, b.ptr, got.ptr, rows, k, n, library=lib, runtime=runtime)
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(expected), ref, expected.nbytes, runtime=runtime)
            copy_device_to_host(host_array_ptr(actual), got, actual.nbytes, runtime=runtime)
            np.testing.assert_array_equal(actual, expected)
        # The CPU floor is separate from scalar-parent equality. Narrow N keeps
        # the reference inexpensive while covering each split-K wave count.
        if n <= 64:
            _check_cpu_floor(arrays, actual, reverse=reverse, k=k, n=n)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)


@pytest.mark.parametrize("rows", [0, -1, 65536])
@pytest.mark.parametrize("fn", [gguf_iq4_q4_pair_silu_bf16_bf16_out,
                               gguf_q4_iq4_pair_silu_bf16_bf16_out])
def test_mixed_pair_rejects_unsupported_grid_rows(fn, rows):
    with pytest.raises(ValueError, match="1 <= rows <= 65535"):
        fn(1, 2, 3, 4, rows, 256, 16)
