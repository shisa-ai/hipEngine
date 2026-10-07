"""Checked-int32/full-four-row dense Q8_0 coltile DPP sibling: GPU parity.

The diagnostic sibling narrows the dense pipelined-DPP leaf's offsets to int32
and removes per-row validity control for complete four-row groups. Neither change
touches the Q8 scale decode, the ascending per-lane K FMA order, the DPP wave
reduction, the four-wave publication or the F32 output, so the sibling must be
bit-identical to the generic DPP parent wherever it is admitted. Tail rows and
unsupported shapes keep the parent. The narrow shapes also carry the
CPU-reference outer floor.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PARENT = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
SIBLING = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out"
WAVE_SCALE = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out"


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not hip_available(), reason="HIP unavailable")

GUARD = 64


def _make_operands(rows: int, k: int, n: int, seed: int, kind: str = "normal"):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (n, k // 32, 34), dtype=np.uint8)
    if kind == "subnormal_scale":
        scales = np.full((n, k // 32), np.float16(2.0**-24), dtype=np.float16)
    elif kind == "wide_scale":
        scales = rng.uniform(1.0, 8.0, (n, k // 32)).astype(np.float16)
    elif kind == "mixed_scale":
        scales = np.where(
            (np.arange(n)[:, None] + np.arange(k // 32)[None, :]) % 2 == 0,
            np.float16(2.0**-20),
            np.float16(4.0),
        ).astype(np.float16)
    else:
        scales = rng.uniform(0.0001, 0.003, (n, k // 32)).astype(np.float16)
    raw[..., :2] = scales.view(np.uint8).reshape(n, k // 32, 2)
    if kind == "subnormal":
        x = (rng.normal(0, 1, (rows, k)) * 2.0**-30).astype(np.float32)
    elif kind == "cancellation":
        base = rng.normal(0, 1, (rows, k)).astype(np.float32)
        x = base.copy()
        x[:, 1::2] = -base[:, 0::2]
    elif kind == "wide":
        x = (rng.normal(0, 1, (rows, k)) * 1.0e4).astype(np.float32)
    else:
        x = rng.normal(0, 0.1, (rows, k)).astype(np.float32)
    return x, raw.reshape(n, -1)


def _run(fn, x, raw, rows, k, n, runtime, allocations, stream=0, guard=GUARD):
    dx, dw = [_upload(v, runtime, allocations) for v in (x, raw)]
    host_out = np.full(rows * n + 2 * guard, np.float32(0xA55A), dtype=np.float32)
    dout = _upload(host_out, runtime, allocations)
    fn(
        dx.ptr,
        dw.ptr,
        dout.ptr + guard * 4,
        rows,
        k,
        n,
        runtime=runtime,
        stream=stream,
    )
    if stream:
        runtime.stream_synchronize(stream)
    else:
        runtime.device_synchronize()
    full = _download(dout, host_out.shape, np.float32, runtime)
    np.testing.assert_array_equal(full[:guard], host_out[:guard])
    np.testing.assert_array_equal(full[-guard:], host_out[-guard:])
    out = full[guard:-guard].reshape(rows, n).copy()
    # Inputs must be untouched.
    for d, host in ((dx, x), (dw, raw)):
        np.testing.assert_array_equal(_download(d, host.shape, host.dtype, runtime), host)
    return out


_CASES = [
    (4, 32, 8),
    (8, 32, 8),
    (12, 96, 24),
    (4, 160, 32),
    (20, 2560, 32),
    (128, 2560, 256),
    (512, 2560, 6144),
    (1024, 2560, 10240),
    (4, 6144, 2560),
    (4, 10240, 320),
]


@pytest.mark.parametrize("rows,k,n", _CASES)
def test_sibling_matches_dpp_parent_bit_exactly(rows: int, k: int, n: int) -> None:
    runtime = get_hip_runtime()
    x, raw = _make_operands(rows, k, n, seed=20261006)
    allocations = []
    try:
        parent = _run(getattr(q8, PARENT), x, raw, rows, k, n, runtime, allocations)
        for _ in range(2):
            actual = _run(getattr(q8, SIBLING), x, raw, rows, k, n, runtime, allocations)
            np.testing.assert_array_equal(
                actual.view(np.uint32), parent.view(np.uint32)
            )
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.parametrize(
    "kind,k,n",
    [
        ("subnormal", 2560, 32),
        ("cancellation", 2560, 32),
        ("wide", 2560, 32),
        ("subnormal_scale", 160, 24),
        ("wide_scale", 96, 24),
        ("mixed_scale", 32, 8),
    ],
)
def test_sibling_matches_parent_on_extreme_operands(kind, k, n) -> None:
    rows = 8
    runtime = get_hip_runtime()
    x, raw = _make_operands(rows, k, n, seed=7, kind=kind)
    allocations = []
    try:
        parent = _run(getattr(q8, PARENT), x, raw, rows, k, n, runtime, allocations)
        actual = _run(getattr(q8, SIBLING), x, raw, rows, k, n, runtime, allocations)
        np.testing.assert_array_equal(actual.view(np.uint32), parent.view(np.uint32))
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


def test_sibling_matches_parent_on_nondefault_stream() -> None:
    runtime = get_hip_runtime()
    stream = runtime.stream_create()
    rows, k, n = 64, 2560, 10240
    x, raw = _make_operands(rows, k, n, seed=11)
    allocations = []
    try:
        parent = _run(
            getattr(q8, PARENT), x, raw, rows, k, n, runtime, allocations, stream=stream
        )
        actual = _run(
            getattr(q8, SIBLING), x, raw, rows, k, n, runtime, allocations, stream=stream
        )
        np.testing.assert_array_equal(actual.view(np.uint32), parent.view(np.uint32))
    finally:
        runtime.stream_destroy(stream)
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.parametrize("rows,k,n", [(2, 2560, 32), (6, 2560, 32), (1023, 2560, 10240)])
def test_tail_rows_are_refused_by_the_sibling_and_kept_by_the_parent(rows, k, n) -> None:
    runtime = get_hip_runtime()
    x, raw = _make_operands(rows, k, n, seed=13)
    allocations = []
    try:
        with pytest.raises(ValueError, match="int32|four-row|divisible"):
            _run(getattr(q8, SIBLING), x, raw, rows, k, n, runtime, allocations)
        # The generic DPP parent still owns the tail and must run.
        parent = _run(getattr(q8, PARENT), x, raw, rows, k, n, runtime, allocations)
        assert parent.shape == (rows, n)
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.parametrize("rows,k,n", [(4, 32, 8), (12, 96, 24), (20, 2560, 32)])
def test_narrow_shapes_pass_the_cpu_reference_floor(rows, k, n) -> None:
    runtime = get_hip_runtime()
    x, raw = _make_operands(rows, k, n, seed=20261006)
    allocations = []
    try:
        actual = _run(getattr(q8, SIBLING), x, raw, rows, k, n, runtime, allocations)
        expected = x @ dequantize_gguf_data(raw, GGMLQuantizationType.Q8_0).T
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

        def log_softmax(a):
            a = a.astype(np.float64)
            a -= a.max(-1, keepdims=True)
            return a - np.log(np.exp(a).sum(-1, keepdims=True))

        p, q = log_softmax(expected), log_softmax(actual)
        assert np.max(np.sum(np.exp(p) * (p - q), axis=-1)) <= 0.05
        assert np.mean(expected.argmax(-1) == actual.argmax(-1)) >= 0.9
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


def test_sibling_is_bit_exact_to_the_wave_scale_grandparent() -> None:
    runtime = get_hip_runtime()
    rows, k, n = 512, 2560, 6144
    x, raw = _make_operands(rows, k, n, seed=17)
    allocations = []
    try:
        grandparent = _run(
            getattr(q8, WAVE_SCALE), x, raw, rows, k, n, runtime, allocations
        )
        actual = _run(getattr(q8, SIBLING), x, raw, rows, k, n, runtime, allocations)
        np.testing.assert_array_equal(
            actual.view(np.uint32), grandparent.view(np.uint32)
        )
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


def test_raw_abi_refuses_overflow_before_touching_pointers() -> None:
    library = q8.build_gguf_k_gemv(load=True)
    fn = getattr(library, "hipengine_gguf_q8_0_gemv_" + SIBLING.removeprefix("gguf_q8_0_gemv_"))
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    for rows, k, n in [
        (2, 2560, 10240),  # tail group
        (4, 31, 10240),  # not a Q8_0 block multiple
        (4, 2560, 7),  # not a COL_TILE multiple
        (4, 2**32, 10240),
        (4, 2560, 2**32),
        (2**32, 2560, 10240),
        (1000000, 2560, 128),
        (16777216, 32, 8),  # activation elements fit, float bytes overflow
        (16777212, 32, 40),  # output elements fit, float bytes overflow
        # Block-aligned near-INT64_MAX K must be refused before the launcher
        # computes the packed row byte count (which would overflow signed int64).
        (4, (2**63 - 1) // 32 * 32, 10240),
        (4, (2**63 - 1) // 32 * 32 + 32, 10240),
        # Extreme int64 rows and N.
        (2**62, 2560, 10240),
        (4, 2560, 2**62),
    ]:
        assert fn(1, 2, 3, rows, k, n, 128, 0) == 1
