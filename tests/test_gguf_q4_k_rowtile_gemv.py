"""Correctness for the small-B weight-amortized GGUF Q4_K row-tile GEMV.

The row-tile kernel (`gguf_q4_k_prefill_out_rowtile_kernel`) launches one block
per output column and accumulates into ROW_TILE row accumulators, dequantizing
each Q4_K weight element once instead of once per row.  It is the verifier
small-B (rows in [2, 8]) replacement for the per-row
`gguf_q4_k_prefill_out_kernel`.

Gate:
1. Bit-exact vs the per-row kernel (same per-thread k order, same wave/cross-wave
   reduction order) for bf16->bf16, bf16->f32, and f32->f32 across rows 2..8 and
   several shapes.
2. Within fp tolerance of the CPU Q4_K dequantized matmul oracle.
3. Wrapper contract / registry surface (no GPU).
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_gemv_bf16_bf16_out,
    gguf_q4_k_gemv_bf16_f32_out,
    gguf_q4_k_gemv_f32_f32_out,
    gguf_q4_k_gemv_rowtile_bf16_bf16_out,
    gguf_q4_k_gemv_rowtile_bf16_f32_out,
    gguf_q4_k_gemv_rowtile_f32_f32_out,
)
from hipengine.kernels.registry import resolve
from tests.test_gguf_q4_k_gemv import make_q4_k_weight

QK_K = 256


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(arr: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 bit pattern (uint16)."""
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    lsb = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + lsb
    return (u32 >> 16).astype(np.uint16)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _q4_k_dequant_row(weight_row: np.ndarray, in_features: int) -> np.ndarray:
    """CPU dequant mirroring gguf_q4_k_weight() for one output column row."""
    blocks = in_features // QK_K
    out = np.empty(in_features, dtype=np.float32)
    for blk in range(blocks):
        base = blk * 144
        d = weight_row[base : base + 2].view(np.float16).astype(np.float32)[0]
        dmin = weight_row[base + 2 : base + 4].view(np.float16).astype(np.float32)[0]
        scales = weight_row[base + 4 : base + 16]
        qs = weight_row[base + 16 : base + 144]
        for sub in range(8):
            if sub < 4:
                sc = scales[sub] & 0x3F
                mn = scales[4 + sub] & 0x3F
            else:
                idx = sub - 4
                sc = (scales[8 + idx] & 0x0F) | ((scales[idx] >> 2) & 0x30)
                mn = (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30)
            pair = sub >> 1
            for lane in range(32):
                packed = qs[pair * 32 + lane]
                q = (packed >> 4) if (sub & 1) else (packed & 0x0F)
                out[blk * QK_K + sub * 32 + lane] = d * sc * float(q) - dmin * mn
    return out


def _cpu_reference(x: np.ndarray, qweight: np.ndarray, in_features: int, out_features: int) -> np.ndarray:
    w = np.stack([_q4_k_dequant_row(qweight[c], in_features) for c in range(out_features)], axis=0)
    return x.astype(np.float32) @ w.T.astype(np.float32)


# ---------------------------------------------------------------------------
# No-GPU surface
# ---------------------------------------------------------------------------


def test_rowtile_registry_binds() -> None:
    for variant in ("rowtile_f32_f32_out", "rowtile_bf16_f32_out", "rowtile_bf16_bf16_out"):
        fn = resolve(backend="hip_gfx1100", layer="linear", quant="gguf_q4_k", variant=variant)
        assert callable(fn)


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------

_SHAPES = [(256, 16), (512, 48), (768, 128), (1024, 64)]
_ROWS = [2, 3, 4, 5, 8]


def _run_device(wrapper, x_host: np.ndarray, qweight: np.ndarray, out_host: np.ndarray) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_gemv(load=True)
    rows, in_features = x_host.shape
    out_features = out_host.shape[1]
    bufs = []
    try:
        x_dev = malloc(x_host.nbytes, runtime=runtime)
        qw_dev = malloc(qweight.nbytes, runtime=runtime)
        out_dev = malloc(out_host.nbytes, runtime=runtime)
        bufs.extend((x_dev, qw_dev, out_dev))
        copy_host_to_device(x_dev, host_array_ptr(np.ascontiguousarray(x_host)), runtime=runtime)
        copy_host_to_device(qw_dev, host_array_ptr(np.ascontiguousarray(qweight)), runtime=runtime)
        wrapper(x_dev.ptr, qw_dev.ptr, out_dev.ptr, rows, in_features, out_features,
                library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_host), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
    return out_host


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("in_features,out_features", _SHAPES)
def test_rowtile_bf16_bf16_bit_exact_vs_per_row(rows, in_features, out_features) -> None:
    rng = np.random.default_rng(1234 + rows * 31 + in_features)
    qweight = make_q4_k_weight(out_features, in_features)
    x_f32 = rng.standard_normal((rows, in_features)).astype(np.float32) * 0.1
    x_bits = _bf16_bits(x_f32).reshape(rows, in_features)

    ref = _run_device(
        gguf_q4_k_gemv_bf16_bf16_out, x_bits, qweight,
        np.zeros((rows, out_features), dtype=np.uint16),
    ).copy()
    got = _run_device(
        gguf_q4_k_gemv_rowtile_bf16_bf16_out, x_bits, qweight,
        np.zeros((rows, out_features), dtype=np.uint16),
    ).copy()
    np.testing.assert_array_equal(got, ref)  # bit-exact


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("in_features,out_features", _SHAPES)
def test_rowtile_bf16_f32_bit_exact_vs_per_row(rows, in_features, out_features) -> None:
    rng = np.random.default_rng(5678 + rows * 17 + out_features)
    qweight = make_q4_k_weight(out_features, in_features)
    x_f32 = rng.standard_normal((rows, in_features)).astype(np.float32) * 0.1
    x_bits = _bf16_bits(x_f32).reshape(rows, in_features)

    ref = _run_device(
        gguf_q4_k_gemv_bf16_f32_out, x_bits, qweight,
        np.zeros((rows, out_features), dtype=np.float32),
    ).copy()
    got = _run_device(
        gguf_q4_k_gemv_rowtile_bf16_f32_out, x_bits, qweight,
        np.zeros((rows, out_features), dtype=np.float32),
    ).copy()
    np.testing.assert_array_equal(got, ref)
    # And within tolerance of the CPU dequant oracle.
    cpu = _cpu_reference(_bf16_to_f32(x_bits), qweight, in_features, out_features)
    np.testing.assert_allclose(got, cpu, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", _ROWS)
def test_rowtile_f32_f32_bit_exact_vs_per_row(rows) -> None:
    in_features, out_features = 512, 80
    rng = np.random.default_rng(99 + rows)
    qweight = make_q4_k_weight(out_features, in_features)
    x = (rng.standard_normal((rows, in_features)).astype(np.float32) * 0.1)

    ref = _run_device(
        gguf_q4_k_gemv_f32_f32_out, x, qweight,
        np.zeros((rows, out_features), dtype=np.float32),
    ).copy()
    got = _run_device(
        gguf_q4_k_gemv_rowtile_f32_f32_out, x, qweight,
        np.zeros((rows, out_features), dtype=np.float32),
    ).copy()
    np.testing.assert_array_equal(got, ref)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_rowtile_rejects_unsupported_rows() -> None:
    # rows outside [2, 8] must be rejected by the launcher (per-row / WMMA path
    # handles those), surfacing as a launch error.
    in_features, out_features = 256, 16
    qweight = make_q4_k_weight(out_features, in_features)
    for rows in (1, 9):
        x = np.zeros((rows, in_features), dtype=np.float32)
        with pytest.raises(Exception):
            _run_device(
                gguf_q4_k_gemv_rowtile_f32_f32_out, x, qweight,
                np.zeros((rows, out_features), dtype=np.float32),
            )
