"""Selected Q5 T16 local32 decode owner tests.

The selected-expert ABI must preserve expert and row ordering while changing
only the c=1 accumulation geometry.  These tests use real T16 repacking over
synthetic raw Q5_K blocks and compare the candidate with the existing direct
selected GEMV.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.backends import backend_package_capability, load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as t16_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.quant.gguf_t16 import (
    repack_gguf_q5_k_qmicro_tile16,
    repack_gguf_q5_k_tile16,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _make_q5_k_raw(
    experts: int, out_features: int, in_features: int, seed: int
) -> np.ndarray:
    assert in_features % 256 == 0
    assert out_features % 32 == 0
    blocks_per_row = in_features // 256
    rng = np.random.default_rng(seed)
    raw = np.zeros((experts, out_features, blocks_per_row, 176), dtype=np.uint8)
    d = np.array([0.010, 0.020, 0.031, 0.008], dtype=np.float16)
    dmin = np.array([0.004, 0.009, 0.001, 0.016], dtype=np.float16)
    raw[..., 0:2] = d[
        rng.integers(0, 4, raw.shape[:2])
    ].view(np.uint8).reshape(experts, out_features, 1, 2)
    raw[..., 2:4] = dmin[
        rng.integers(0, 4, raw.shape[:2])
    ].view(np.uint8).reshape(experts, out_features, 1, 2)
    raw[..., 4:16] = rng.integers(
        0, 256, (*raw.shape[:2], blocks_per_row, 12), dtype=np.uint8
    )
    raw[..., 16:] = rng.integers(
        0, 256, (*raw.shape[:2], blocks_per_row, 160), dtype=np.uint8
    )
    return raw.reshape(experts, out_features, blocks_per_row * 176)


def test_selected_q5_local32_policy_is_shape_scoped() -> None:
    load_backend_kernel_package("hip_gfx1100")
    policy = backend_package_capability(
        "hip_gfx1100", "GGUF_T16_SELECTED_C1_VARIANTS_BY_QUANT_SHAPE", None
    )
    # The real selected-down shape of Qwen3.6-35B-A3B-UD-Q4_K_M: moe_inter 512
    # into hidden 2048, keyed (in_features, out_features) as the launcher names
    # them. The MoE decode path materializes those weights in the qmicro planar
    # layout, so the qmicro quant key is the admission; the plain-T16 key and a
    # dense Qwen3.6-27B FFN shape must not resolve here.
    assert policy == {
        "gguf_q5_k_qmicro_t16_v1": {
            (512, 2_048): "selected_t16_local32_gemv_decode_bf16_bf16_out",
        }
    }

    from hipengine.runtime.qwen35_gguf_runner import (
        _gguf_t16_selected_c1_variant,
    )

    variant = "selected_t16_local32_gemv_decode_bf16_bf16_out"
    # Production single-token dispatch: x_rows == rows == top_k.
    assert (
        _gguf_t16_selected_c1_variant(
            "hip_gfx1100",
            "gguf_q5_k_qmicro_t16_v1",
            x_rows=8,
            rows=8,
            in_features=512,
            out_features=2_048,
        )
        == variant
    )
    # The plain-T16 quant key is not this admission.
    assert (
        _gguf_t16_selected_c1_variant(
            "hip_gfx1100",
            "gguf_q5_k_t16_v1",
            x_rows=8,
            rows=8,
            in_features=512,
            out_features=2_048,
        )
        is None
    )
    # Batched dispatch: several tokens, each with its own selected rows.
    assert (
        _gguf_t16_selected_c1_variant(
            "hip_gfx1100",
            "gguf_q5_k_qmicro_t16_v1",
            x_rows=2,
            rows=16,
            in_features=512,
            out_features=2_048,
        )
        == variant
    )
    # Not a multiple: the kernel's row-to-x mapping is undefined.
    assert (
        _gguf_t16_selected_c1_variant(
            "hip_gfx1100",
            "gguf_q5_k_qmicro_t16_v1",
            x_rows=3,
            rows=8,
            in_features=512,
            out_features=2_048,
        )
        is None
    )
    # Dense Qwen3.6-27B FFN-down shape and the transposed pair are not selected
    # admissions.
    assert (
        _gguf_t16_selected_c1_variant(
            "hip_gfx1100",
            "gguf_q5_k_qmicro_t16_v1",
            x_rows=8,
            rows=8,
            in_features=17_408,
            out_features=5_120,
        )
        is None
    )
    assert (
        _gguf_t16_selected_c1_variant(
            "hip_gfx1100",
            "gguf_q5_k_qmicro_t16_v1",
            x_rows=8,
            rows=8,
            in_features=2_048,
            out_features=512,
        )
        is None
    )
    assert (
        _gguf_t16_selected_c1_variant(
            None,
            "gguf_q5_k_qmicro_t16_v1",
            x_rows=8,
            rows=8,
            in_features=512,
            out_features=2_048,
        )
        is None
    )


def test_selected_q5_local32_is_registered() -> None:
    load_backend_kernel_package("hip_gfx1100")
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q5_k_t16_v1",
            "selected_t16_local32_gemv_decode_bf16_bf16_out",
        )
    )


@pytest.mark.parametrize(
    "x_rows,rows,in_features,out_features,experts",
    [
        # Shared-x contract: one x row reused by every selected row.
        (1, 8, 512, 64, 3),
        # Production dispatch contract: x_rows == rows == top_k, so every
        # selected row has its own x row. This is the shape Qwen3.6-35B-A3B
        # actually launches; the shared-x-only version of this kernel was
        # numerically correct here and still wrong for the real path.
        (8, 8, 512, 64, 3),
        # Batched contract: two tokens, each with four selected rows.
        (2, 8, 512, 64, 3),
        # The real production geometry (moe_inter 512 -> hidden 2048).
        (8, 8, 512, 2_048, 4),
    ],
)
def test_selected_q5_local32_matches_direct_selected_gemv(
    x_rows: int, rows: int, in_features: int, out_features: int, experts: int
) -> None:
    runtime = get_hip_runtime()
    raw = _make_q5_k_raw(experts, out_features, in_features, seed=0x7A51)
    tiles = repack_gguf_q5_k_tile16(raw).tiles
    selected = np.array(
        [(index * 2 + 1) % experts for index in range(rows)], dtype=np.int64
    )
    rng = np.random.default_rng(0x7A52)
    # One distinct x row per x row index, so a kernel that ignores the row
    # mapping cannot accidentally match.
    x_bits = _bf16_bits(rng.normal(0.0, 0.2, (x_rows, in_features)))
    ref_bits = np.zeros((rows, out_features), dtype=np.uint16)
    got_bits = np.zeros_like(ref_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        selected_dev = malloc(selected.nbytes, runtime=runtime)
        tiles_dev = malloc(tiles.nbytes, runtime=runtime)
        ref_dev = malloc(ref_bits.nbytes, runtime=runtime)
        got_dev = malloc(got_bits.nbytes, runtime=runtime)
        buffers.extend((x_dev, selected_dev, tiles_dev, ref_dev, got_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(
            selected_dev, host_array_ptr(selected), runtime=runtime
        )
        copy_host_to_device(tiles_dev, host_array_ptr(tiles), runtime=runtime)
        library = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        gguf_q5_k_t16_selected_gemv_bf16_bf16_out(
            x_dev.ptr,
            selected_dev.ptr,
            tiles_dev.ptr,
            ref_dev.ptr,
            x_rows,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        t16_gemv.gguf_q5_k_t16_selected_local32_gemv_bf16_bf16_out(
            x_dev.ptr,
            selected_dev.ptr,
            tiles_dev.ptr,
            got_dev.ptr,
            x_rows,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(ref_bits), ref_dev, ref_bits.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(got_bits), got_dev, got_bits.nbytes, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    ref = (ref_bits.astype(np.uint32) << 16).view(np.float32)
    got = (got_bits.astype(np.uint32) << 16).view(np.float32)
    assert np.isfinite(got).all()
    scale = max(float(np.abs(ref).max()), 1e-30)
    assert float(np.abs(ref - got).max()) / scale <= 5e-4
    assert float(np.corrcoef(ref.ravel(), got.ravel())[0, 1]) >= 0.9999


def test_selected_q5_local32_rejects_undefined_row_mapping() -> None:
    """rows must be a whole multiple of x_rows, matching the direct owner."""

    with pytest.raises(ValueError, match="divisible by x_rows"):
        t16_gemv.gguf_q5_k_t16_selected_local32_gemv_bf16_bf16_out(
            1,
            2,
            3,
            4,
            3,
            8,
            3,
            512,
            64,
        )


@pytest.mark.parametrize(
    "x_rows,rows,in_features,out_features,experts",
    [
        (1, 8, 512, 64, 3),
        (8, 8, 512, 64, 3),
        (2, 8, 512, 64, 3),
        # The real production geometry and quant layout: the MoE decode path
        # materializes Q5 selected-down weights as qmicro planar tiles.
        (8, 8, 512, 2_048, 4),
    ],
)
def test_selected_q5_qmicro_local32_matches_production_tile8(
    x_rows: int, rows: int, in_features: int, out_features: int, experts: int
) -> None:
    """The qmicro-planar local32 owner must match the production qmicro owner."""

    runtime = get_hip_runtime()
    raw = _make_q5_k_raw(experts, out_features, in_features, seed=0x7A61)
    tiles = repack_gguf_q5_k_qmicro_tile16(raw).tiles
    selected = np.array(
        [(index * 2 + 1) % experts for index in range(rows)], dtype=np.int64
    )
    rng = np.random.default_rng(0x7A62)
    x_bits = _bf16_bits(rng.normal(0.0, 0.2, (x_rows, in_features)))
    ref_bits = np.zeros((rows, out_features), dtype=np.uint16)
    got_bits = np.zeros_like(ref_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        selected_dev = malloc(selected.nbytes, runtime=runtime)
        tiles_dev = malloc(tiles.nbytes, runtime=runtime)
        ref_dev = malloc(ref_bits.nbytes, runtime=runtime)
        got_dev = malloc(got_bits.nbytes, runtime=runtime)
        buffers.extend((x_dev, selected_dev, tiles_dev, ref_dev, got_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(
            selected_dev, host_array_ptr(selected), runtime=runtime
        )
        copy_host_to_device(tiles_dev, host_array_ptr(tiles), runtime=runtime)
        library = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        # Production owner for gguf_q5_k_qmicro_t16_v1 at tile_cols=8.
        t16_gemv.gguf_q5_k_qmicro_t16_selected_gemv_bf16_bf16_out(
            x_dev.ptr,
            selected_dev.ptr,
            tiles_dev.ptr,
            ref_dev.ptr,
            x_rows,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        t16_gemv.gguf_q5_k_qmicro_t16_selected_local32_gemv_bf16_bf16_out(
            x_dev.ptr,
            selected_dev.ptr,
            tiles_dev.ptr,
            got_dev.ptr,
            x_rows,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(ref_bits), ref_dev, ref_bits.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(got_bits), got_dev, got_bits.nbytes, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    ref = (ref_bits.astype(np.uint32) << 16).view(np.float32)
    got = (got_bits.astype(np.uint32) << 16).view(np.float32)
    assert np.isfinite(got).all()
    scale = max(float(np.abs(ref).max()), 1e-30)
    assert float(np.abs(ref - got).max()) / scale <= 5e-4
    assert float(np.corrcoef(ref.ravel(), got.ravel())[0, 1]) >= 0.9999
