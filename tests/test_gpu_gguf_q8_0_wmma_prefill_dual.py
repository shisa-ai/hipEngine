"""Correctness fixtures for the P9.C1 Q8_0 dual gate+up WMMA prefill kernel.

Same per-tile compute as ``gguf_q8_0_prefill_wmma_kernel`` but consumes two
weight tensors and emits the row-major concatenated layout
``[rows, out_features_a + out_features_b]`` that ``silu_mul_dual_out_*``
consumes. Each col_tile lies entirely in either gate or up; the wrapper
enforces ``out_features_a % tile_m == 0`` and ``out_features_b % tile_m == 0``.
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    build_gguf_q8_0_prefill,
    gguf_q8_0_wmma_prefill_bf16_bf16_out,
    gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out,
    gguf_q8_0_wmma_prefill_dual_gate_up_fp16_fp16_out,
    register_gguf_q8_0_prefill_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q8_0_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def q8_0_prefill_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q8_0_prefill(load=True)


# ---------------------------------------------------------------------------
# Registry / wrapper contract.
# ---------------------------------------------------------------------------


def test_p9_c1_dual_registry_keys_resolve() -> None:
    register_gguf_q8_0_prefill_kernels()
    for variant in (
        "wmma_prefill_dual_gate_up_bf16_bf16_out",
        "wmma_prefill_dual_gate_up_fp16_fp16_out",
    ):
        fn = resolve(backend="hip_gfx1100", layer="linear", quant="gguf_q8_0", variant=variant)
        assert fn is not None, f"missing registry entry: {variant}"


def test_p9_c1_dual_wrapper_validates_args() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out(
            0, 0, 0, 0, 0, 32, 64, 64,
        )
    with pytest.raises(ValueError, match="in_features must be divisible by Q8_0 block size 32"):
        gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out(
            0, 0, 0, 0, 1, 31, 64, 64,
        )
    # The gate/up boundary constraint: out_features_a/b must align to tile_m.
    with pytest.raises(ValueError, match=r"out_features_a=24 must be a multiple of tile_m=32"):
        gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out(
            0, 0, 0, 0, 32, 32, 24, 64, tile_m=32, tile_n=32,
        )
    with pytest.raises(ValueError, match=r"out_features_b=24 must be a multiple of tile_m=32"):
        gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out(
            0, 0, 0, 0, 32, 32, 64, 24, tile_m=32, tile_n=32,
        )


# ---------------------------------------------------------------------------
# Correctness vs CPU oracle.
# ---------------------------------------------------------------------------


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _run_dual_bf16(x_bf16, wa, wb, rows, in_f, oa, ob, tile_m, tile_n, library) -> np.ndarray:
    xbuf = malloc(x_bf16.nbytes); copy_host_to_device(xbuf, host_array_ptr(x_bf16), x_bf16.nbytes)
    wabuf = malloc(wa.nbytes); copy_host_to_device(wabuf, host_array_ptr(wa), wa.nbytes)
    wbbuf = malloc(wb.nbytes); copy_host_to_device(wbbuf, host_array_ptr(wb), wb.nbytes)
    out_arr = np.zeros((rows, oa + ob), dtype=np.uint16); obuf = malloc(out_arr.nbytes)
    try:
        gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out(
            xbuf.ptr, wabuf.ptr, wbbuf.ptr, obuf.ptr, rows, in_f, oa, ob,
            tile_m=tile_m, tile_n=tile_n, library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), obuf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (xbuf, wabuf, wbbuf, obuf):
            free(b)


def _run_single_bf16(x_bf16, w, rows, in_f, out_f, tile_m, tile_n, library) -> np.ndarray:
    xbuf = malloc(x_bf16.nbytes); copy_host_to_device(xbuf, host_array_ptr(x_bf16), x_bf16.nbytes)
    wbuf = malloc(w.nbytes); copy_host_to_device(wbuf, host_array_ptr(w), w.nbytes)
    out_arr = np.zeros((rows, out_f), dtype=np.uint16); obuf = malloc(out_arr.nbytes)
    try:
        gguf_q8_0_wmma_prefill_bf16_bf16_out(
            xbuf.ptr, wbuf.ptr, obuf.ptr, rows, in_f, out_f,
            tile_m=tile_m, tile_n=tile_n, library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), obuf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (xbuf, wbuf, obuf):
            free(b)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("tile_m,tile_n", [(16, 16), (32, 32), (64, 32), (16, 32)])
def test_p9_c1_dual_matches_two_singles(tile_m, tile_n, q8_0_prefill_library) -> None:
    """The dual kernel must be bit-exact with two single-kernel calls.

    Same tile, same accumulation order: launching the dual is mathematically
    equivalent to launching the single kernel twice (one per side) and
    concatenating the outputs. Anchors the math invariant for future tile
    tuning.
    """

    ROWS, IN, OUTA, OUTB = 32, 256, max(tile_m, 32), max(tile_m, 32)
    rng = np.random.default_rng(tile_m * 7 + tile_n * 11)
    wa = make_q8_0_weight(OUTA, IN)
    wb = make_q8_0_weight(OUTB, IN)
    x = rng.normal(0.0, 0.3, size=(ROWS, IN)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    dual = _run_dual_bf16(x_bf16, wa, wb, ROWS, IN, OUTA, OUTB, tile_m, tile_n, q8_0_prefill_library)
    single_a = _run_single_bf16(x_bf16, wa, ROWS, IN, OUTA, tile_m, tile_n, q8_0_prefill_library)
    single_b = _run_single_bf16(x_bf16, wb, ROWS, IN, OUTB, tile_m, tile_n, q8_0_prefill_library)
    np.testing.assert_array_equal(dual[:, :OUTA], single_a)
    np.testing.assert_array_equal(dual[:, OUTA:], single_b)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features_a,out_features_b",
    [
        (32, 256, 32, 32),
        (32, 512, 64, 64),
        (64, 1024, 128, 256),
        (128, 2048, 256, 512),
        (512, 2048, 4096, 4096),  # qwen35moe shared-expert gate+up shape
    ],
)
def test_p9_c1_dual_matches_cpu_oracle(rows, in_features, out_features_a, out_features_b, q8_0_prefill_library) -> None:
    rng = np.random.default_rng(rows * 13 + in_features + out_features_a)
    wa = make_q8_0_weight(out_features_a, in_features)
    wb = make_q8_0_weight(out_features_b, in_features)
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual = _run_dual_bf16(
        x_bf16, wa, wb, rows, in_features, out_features_a, out_features_b,
        tile_m=None, tile_n=None, library=q8_0_prefill_library,
    )
    actual_f32 = _bf16_u16_to_f32(actual)
    expected = np.zeros_like(actual_f32)
    expected[:, :out_features_a] = gguf_quant_gemv(x_ref, wa, GGMLQuantizationType.Q8_0)
    expected[:, out_features_a:] = gguf_quant_gemv(x_ref, wb, GGMLQuantizationType.Q8_0)
    # WMMA computes in fp16; round expected through BF16 first so we are
    # comparing kernel math, not output rounding.
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    # BF16 output rounds to 1 ULP at the output magnitude; allow ~1 BF16 ULP
    # absolute plus a small relative slack for the WMMA FP16 accumulation.
    np.testing.assert_allclose(actual_f32, expected_bf16, atol=4.0e-3, rtol=1.0e-2)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_p9_c1_dual_fp16_path(q8_0_prefill_library) -> None:
    ROWS, IN, OUTA, OUTB = 32, 512, 64, 128
    rng = np.random.default_rng(101)
    wa = make_q8_0_weight(OUTA, IN); wb = make_q8_0_weight(OUTB, IN)
    x_f16 = rng.normal(0.0, 0.3, size=(ROWS, IN)).astype(np.float16)

    xbuf = malloc(x_f16.nbytes); copy_host_to_device(xbuf, host_array_ptr(x_f16), x_f16.nbytes)
    wabuf = malloc(wa.nbytes); copy_host_to_device(wabuf, host_array_ptr(wa), wa.nbytes)
    wbbuf = malloc(wb.nbytes); copy_host_to_device(wbbuf, host_array_ptr(wb), wb.nbytes)
    out_arr = np.zeros((ROWS, OUTA + OUTB), dtype=np.float16); obuf = malloc(out_arr.nbytes)
    try:
        gguf_q8_0_wmma_prefill_dual_gate_up_fp16_fp16_out(
            xbuf.ptr, wabuf.ptr, wbbuf.ptr, obuf.ptr, ROWS, IN, OUTA, OUTB,
            tile_m=32, tile_n=32, library=q8_0_prefill_library,
        )
        copy_device_to_host(host_array_ptr(out_arr), obuf, out_arr.nbytes)
    finally:
        for b in (xbuf, wabuf, wbbuf, obuf):
            free(b)

    x_ref = x_f16.astype(np.float32)
    expected = np.zeros((ROWS, OUTA + OUTB), dtype=np.float32)
    expected[:, :OUTA] = gguf_quant_gemv(x_ref, wa, GGMLQuantizationType.Q8_0)
    expected[:, OUTA:] = gguf_quant_gemv(x_ref, wb, GGMLQuantizationType.Q8_0)
    actual = out_arr.astype(np.float32)
    expected_f16 = expected.astype(np.float16).astype(np.float32)
    # FP16 output has a wider mantissa than BF16; rounding ULP is much
    # smaller, but the WMMA FP16 accumulation still introduces small drift.
    np.testing.assert_allclose(actual, expected_f16, atol=2.0e-3, rtol=5.0e-3)
