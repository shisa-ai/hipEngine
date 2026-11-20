"""Two-row GEMV gate: per-row bit-identity with the single-row kernel.

The AR decode spends about 79% of a step in the projection GEMV families, and the
LM head alone is 756 MB of BF16 weight per call for one row. The two-row kernel
issues that weight stream once for two rows, so it is only usable if a row's result
is unchanged: each row must keep the single-row kernel's K traversal and local-256
reduction order. This gate asserts exactly that, bit for bit, on the production
shapes and on odd row counts, plus a NumPy FP64 outer check that both agree with the
mathematics. The BF16-output twins cover the AR's ``o`` and ``down`` projections and
the dual ``gate``/``up`` projection, which feed bf16 rounding into the next stage.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_array_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear import dense_gemv


def _has_hip() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


requires_hip = pytest.mark.skipif(not _has_hip(), reason="ROCm/HIP runtime is not available")


def _upload(array: np.ndarray):
    host = np.ascontiguousarray(array)
    buffer = malloc(max(host.nbytes, 8))
    copy_host_array_to_device(buffer, host)
    return buffer


def _download(buffer, shape, dtype):
    out = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), buffer)
    return out


def _bf16(values) -> np.ndarray:
    from hipengine.runtime.yue2_nar import to_bf16_bits

    return to_bf16_bits(np.asarray(values, dtype=np.float32))


def _widen(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _run_pair(rows: int, in_features: int, out_features: int, seed: int):
    rng = np.random.default_rng(seed)
    x = _bf16(rng.standard_normal((rows, in_features)) * 0.5)
    weight = _bf16(rng.standard_normal((out_features, in_features)) * 0.05)
    x_buf = _upload(x)
    w_buf = _upload(weight)

    single = _upload(np.zeros((rows, out_features), dtype=np.float32))
    for row in range(rows):
        dense_gemv.dense_gemv_bf16_f32_out(
            x_buf.ptr + row * in_features * 2, w_buf.ptr, single.ptr + row * out_features * 4,
            1, in_features, out_features,
        )
    paired = _upload(np.zeros((rows, out_features), dtype=np.float32))
    dense_gemv.dense_gemv_bf16_f32_out_rowtile2(
        x_buf.ptr, w_buf.ptr, paired.ptr, rows, in_features, out_features,
    )
    return (
        _download(single, (rows, out_features), np.float32),
        _download(paired, (rows, out_features), np.float32),
        _widen(x).astype(np.float64),
        _widen(weight).astype(np.float64),
    )


@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (2, 2048, 2048),      # q / o projection at the AR's hidden size
        (2, 2048, 184704),    # the LM head, the shape this kernel exists for
        (2, 2048, 1024),      # k / v projection
        (3, 256, 512),        # odd row count: the second block runs one row
        (1, 512, 256),        # a single row must still match
    ],
)
@requires_hip
def test_rowtile2_is_bit_identical_per_row(rows, in_features, out_features):
    single, paired, x, weight = _run_pair(rows, in_features, out_features, seed=rows * 31 + out_features)
    assert np.array_equal(single, paired), (
        f"max abs difference {np.abs(single - paired).max()} on "
        f"rows={rows} in={in_features} out={out_features}"
    )
    expected = x @ weight.T
    rel = np.abs(paired - expected).max() / max(np.abs(expected).max(), 1e-12)
    assert rel < 2e-3, f"relative error {rel} against the FP64 reference"


@requires_hip
def test_rowtile2_rejects_bad_shapes():
    x_buf = _upload(np.zeros((2, 64), dtype=np.uint16))
    w_buf = _upload(np.zeros((8, 64), dtype=np.uint16))
    out_buf = _upload(np.zeros((2, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        dense_gemv.dense_gemv_bf16_f32_out_rowtile2(
            x_buf.ptr, w_buf.ptr, out_buf.ptr, 0, 64, 8,
        )
    with pytest.raises(ValueError):
        dense_gemv.dense_gemv_bf16_f32_out_rowtile2(
            x_buf.ptr, w_buf.ptr, out_buf.ptr, 2, 64, 8, threads=100,
        )


def _run_bf16_pair(rows: int, in_features: int, out_features: int, seed: int):
    rng = np.random.default_rng(seed)
    x = _bf16(rng.standard_normal((rows, in_features)) * 0.5)
    weight = _bf16(rng.standard_normal((out_features, in_features)) * 0.05)
    x_buf = _upload(x)
    w_buf = _upload(weight)

    single = _upload(np.zeros((rows, out_features), dtype=np.uint16))
    for row in range(rows):
        dense_gemv.dense_gemv_out_bf16(
            x_buf.ptr + row * in_features * 2, w_buf.ptr, single.ptr + row * out_features * 2,
            1, in_features, out_features,
        )
    paired = _upload(np.zeros((rows, out_features), dtype=np.uint16))
    dense_gemv.dense_gemv_out_bf16_rowtile2(
        x_buf.ptr, w_buf.ptr, paired.ptr, rows, in_features, out_features,
    )
    return (
        _download(single, (rows, out_features), np.uint16),
        _download(paired, (rows, out_features), np.uint16),
        _widen(x).astype(np.float64),
        _widen(weight).astype(np.float64),
    )


@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (2, 2048, 2048),      # o projection at the AR's hidden size
        (2, 6144, 2048),      # down projection
        (2, 2048, 256),       # k / v projection
        (3, 256, 512),        # odd row count: the second block runs one row
        (1, 512, 256),        # a single row must still match
    ],
)
@requires_hip
def test_bf16_rowtile2_is_bit_identical_per_row(rows, in_features, out_features):
    single, paired, x, weight = _run_bf16_pair(rows, in_features, out_features, seed=rows * 37 + out_features)
    assert np.array_equal(single, paired), (
        f"max abs difference {np.abs(_widen(single) - _widen(paired)).max()} on "
        f"rows={rows} in={in_features} out={out_features}"
    )
    # BF16 inputs carry 2^-9 relative rounding, and the max over every output of a
    # 2048-term dot lands above that, so this bound is looser than the F32-output
    # twin's. Bit-identity above is the contract; this only rejects a kernel that
    # computes the wrong product.
    expected = _widen(paired).astype(np.float64)
    rel = np.abs(expected - x @ weight.T).max() / max(np.abs(expected).max(), 1e-12)
    assert rel < 5e-3, f"relative error {rel} against the FP64 reference"


def _run_dual_bf16_pair(rows: int, in_features: int, out_features: int, seed: int):
    rng = np.random.default_rng(seed)
    x = _bf16(rng.standard_normal((rows, in_features)) * 0.5)
    weight_a = _bf16(rng.standard_normal((out_features, in_features)) * 0.05)
    weight_b = _bf16(rng.standard_normal((out_features, in_features)) * 0.05)
    x_buf = _upload(x)
    wa_buf = _upload(weight_a)
    wb_buf = _upload(weight_b)

    single = _upload(np.zeros((rows, 2 * out_features), dtype=np.uint16))
    for row in range(rows):
        dense_gemv.dense_dual_gemv_out_bf16(
            x_buf.ptr + row * in_features * 2, wa_buf.ptr, wb_buf.ptr,
            single.ptr + row * 2 * out_features * 2,
            1, in_features, out_features, out_features,
        )
    paired = _upload(np.zeros((rows, 2 * out_features), dtype=np.uint16))
    dense_gemv.dense_dual_gemv_out_bf16_rowtile2(
        x_buf.ptr, wa_buf.ptr, wb_buf.ptr, paired.ptr, rows, in_features, out_features, out_features,
    )
    return (
        _download(single, (rows, 2 * out_features), np.uint16),
        _download(paired, (rows, 2 * out_features), np.uint16),
        _widen(x).astype(np.float64),
        _widen(weight_a).astype(np.float64),
        _widen(weight_b).astype(np.float64),
    )


@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (2, 2048, 6144),      # gate / up at the AR's MLP width
        (3, 256, 512),        # odd row count: the second block runs one row
        (1, 512, 256),        # a single row must still match
    ],
)
@requires_hip
def test_dual_bf16_rowtile2_is_bit_identical_per_row(rows, in_features, out_features):
    single, paired, x, weight_a, weight_b = _run_dual_bf16_pair(
        rows, in_features, out_features, seed=rows * 41 + out_features
    )
    assert np.array_equal(single, paired), (
        f"max abs difference {np.abs(_widen(single) - _widen(paired)).max()} on "
        f"rows={rows} in={in_features} out={out_features}"
    )
    expected = _widen(paired).astype(np.float64)
    reference = np.concatenate((x @ weight_a.T, x @ weight_b.T), axis=1)
    rel = np.abs(expected - reference).max() / max(np.abs(expected).max(), 1e-12)
    assert rel < 5e-3, f"relative error {rel} against the FP64 reference"


@requires_hip
def test_bf16_rowtile2_rejects_bad_shapes():
    x_buf = _upload(np.zeros((2, 64), dtype=np.uint16))
    w_buf = _upload(np.zeros((8, 64), dtype=np.uint16))
    out_buf = _upload(np.zeros((2, 8), dtype=np.uint16))
    with pytest.raises(ValueError):
        dense_gemv.dense_gemv_out_bf16_rowtile2(x_buf.ptr, w_buf.ptr, out_buf.ptr, 0, 64, 8)
    with pytest.raises(ValueError):
        dense_gemv.dense_gemv_out_bf16_rowtile2(
            x_buf.ptr, w_buf.ptr, out_buf.ptr, 2, 64, 8, threads=100,
        )
    with pytest.raises(ValueError):
        dense_gemv.dense_dual_gemv_out_bf16_rowtile2(
            x_buf.ptr, w_buf.ptr, w_buf.ptr, out_buf.ptr, 0, 64, 8, 8,
        )
