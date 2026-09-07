"""Guarded GPU depth fixtures for the Q4_K pack8 decode GEMV rowtile.

Campaign Packet-5 contract: guarded GPU fixtures for every depth 1-7 with
guard sentinels for buffer overruns, asserting the executed depth. The C1
singleton native verify now issues rows 2-8 batches through this kernel,
so each fixture:

1. surrounds the batch rows with sentinel guard rows (pre/post), proving
   the kernel never writes outside the declared row count;
2. asserts bit-exact row independence: one rows-K batch launch produces
   byte-identical outputs to K single-row launches. This is the contract
   that makes chunked rowtile dispatch exact by construction;
3. validates every row against the CPU Q4_K GEMV oracle.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_pack8_gemv import (
    build_gguf_q4_k_pack8_gemv,
    gguf_q4_k_pack8_gemv_decode_bf16_bf16_out,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q4_k_weight

_DEPTHS = (2, 3, 4, 5, 6, 7, 8)
_IN_FEATURES = 512
_OUT_FEATURES = 256
_SENTINEL = np.uint16(0xDEAD)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _f32_to_bf16_u16(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded


def _bf16_u16_to_f32(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


@pytest.fixture(scope="module")
def q4_k_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q4_k_pack8_gemv(load=True)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("depth", _DEPTHS)
def test_pack8_decode_batch_rows_are_guarded_and_row_independent(
    depth: int, q4_k_library
) -> None:
    """Rows-K batch equals K single-row launches; sentinels stay untouched."""

    runtime = q4_k_library
    qweight = make_q4_k_weight(_OUT_FEATURES, _IN_FEATURES)
    rng = np.random.default_rng(0xC1D0 + depth)
    x_f32 = rng.normal(0.0, 0.3, size=(depth, _IN_FEATURES)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x_f32)
    x_ref = _bf16_u16_to_f32(x_bf16)

    # Guarded layout: [sentinel row | depth batch rows | sentinel row].
    guarded_x = np.full((depth + 2, _IN_FEATURES), _SENTINEL, dtype=np.uint16)
    guarded_x[1 : depth + 1] = x_bf16
    guarded_out = np.full((depth + 2, _OUT_FEATURES), _SENTINEL, dtype=np.uint16)

    x_buf = malloc(guarded_x.nbytes)
    w_buf = malloc(qweight.nbytes)
    out_buf = malloc(guarded_out.nbytes)
    try:
        copy_host_to_device(x_buf, host_array_ptr(guarded_x), guarded_x.nbytes)
        copy_host_to_device(w_buf, host_array_ptr(qweight), qweight.nbytes)
        copy_host_to_device(out_buf, host_array_ptr(guarded_out), guarded_out.nbytes)
        gguf_q4_k_pack8_gemv_decode_bf16_bf16_out(
            x_buf.ptr + _IN_FEATURES * 2,  # skip the leading sentinel row
            w_buf.ptr,
            out_buf.ptr + _OUT_FEATURES * 2,
            depth,
            _IN_FEATURES,
            _OUT_FEATURES,
            library=q4_k_library,
        )
        copy_device_to_host(
            host_array_ptr(guarded_out), out_buf, guarded_out.nbytes
        )
    finally:
        for buf in (x_buf, w_buf, out_buf):
            free(buf)

    # Guard sentinels: no write before row 0 or after the last batch row.
    assert np.all(guarded_x[0] == _SENTINEL)
    assert np.all(guarded_x[-1] == _SENTINEL)
    assert np.all(guarded_out[0] == _SENTINEL)
    assert np.all(guarded_out[-1] == _SENTINEL)

    batch_out = _bf16_u16_to_f32(guarded_out[1 : depth + 1])

    # Row independence: K single-row launches must match the batch bit-exact.
    single_rows = []
    for row in range(depth):
        single = _run_single_row(
            x_bf16[row : row + 1], qweight, q4_k_library
        )
        single_rows.append(single)
    np.testing.assert_array_equal(
        batch_out, _bf16_u16_to_f32(np.concatenate(single_rows, axis=0))
    )

    # CPU oracle per row.
    expected = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q4_K)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(batch_out, expected_bf16, atol=1.0e-3, rtol=1.0e-2)


def _run_single_row(x_row: np.ndarray, qweight: np.ndarray, library) -> np.ndarray:
    x_buf = malloc(x_row.nbytes)
    w_buf = malloc(qweight.nbytes)
    out_arr = np.zeros((1, _OUT_FEATURES), dtype=np.uint16)
    out_buf = malloc(out_arr.nbytes)
    try:
        copy_host_to_device(x_buf, host_array_ptr(x_row), x_row.nbytes)
        copy_host_to_device(w_buf, host_array_ptr(qweight), qweight.nbytes)
        gguf_q4_k_pack8_gemv_decode_bf16_bf16_out(
            x_buf.ptr,
            w_buf.ptr,
            out_buf.ptr,
            1,
            _IN_FEATURES,
            _OUT_FEATURES,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, w_buf, out_buf):
            free(buf)
