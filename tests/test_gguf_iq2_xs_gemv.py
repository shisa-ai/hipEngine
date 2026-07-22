"""Raw IQ2_XS selected-MoE decode kernels for Laguna-shaped gate/up weights."""

from __future__ import annotations

import ctypes
import inspect
import os
import re
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_gemv import (
    build_gguf_iq_gemv,
    gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_iq2_xs_selected_gemv_bf16_bf16_out,
    plan_gguf_iq_gemv_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data

QK_K = 256
IQ2_XS_BLOCK_BYTES = 74
_QUANT_SOURCE_DIR = (
    Path(__file__).parents[1]
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "quant"
)
_HIP_SOURCE = _QUANT_SOURCE_DIR / "gguf_iq_gemv.hip"
_PREFILL_SOURCE = _QUANT_SOURCE_DIR / "gguf_iq_selected_prefill.hip"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def iq_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    require_cached = os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1"
    return build_gguf_iq_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached,
    )


def _f32_to_bf16_u16(array: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(array, dtype=np.float32)
    bits = f32.view(np.uint32).copy()
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    return ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)


def _bf16_u16_to_f32(array: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(array, dtype=np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).copy()


def _make_x(rows: int, features: int) -> np.ndarray:
    values = np.arange(rows * features, dtype=np.int32).reshape(rows, features)
    return ((values % 31) - 15).astype(np.float32) / np.float32(64.0)


def _make_iq2_xs_weight(
    num_experts: int, out_features: int, in_features: int, *, seed: int
) -> np.ndarray:
    if in_features % QK_K:
        raise ValueError("in_features must be divisible by 256")
    rng = np.random.default_rng(seed)
    blocks = in_features // QK_K
    out = np.empty(
        (num_experts, out_features, blocks * IQ2_XS_BLOCK_BYTES), dtype=np.uint8
    )
    block_view = out.reshape(num_experts, out_features, blocks, IQ2_XS_BLOCK_BYTES)
    scales = np.float16(
        0.0009765625
        * (1 + np.arange(num_experts * out_features * blocks, dtype=np.int32) % 7)
    ).reshape(num_experts, out_features, blocks)
    block_view[..., :2] = scales[..., None].view(np.uint8).reshape(
        num_experts, out_features, blocks, 2
    )
    block_view[..., 2:66] = rng.integers(
        0, 256, size=block_view[..., 2:66].shape, dtype=np.uint8
    )
    block_view[..., 66:74] = rng.integers(
        0, 256, size=block_view[..., 66:74].shape, dtype=np.uint8
    )
    return out


def _selected_reference(
    x_bf16: np.ndarray, selected: np.ndarray, qweight: np.ndarray
) -> np.ndarray:
    x = _bf16_u16_to_f32(x_bf16)
    dequant = dequantize_gguf_data(qweight, GGMLQuantizationType.IQ2_XS)
    rows = int(selected.size)
    lanes_per_x_row = rows // x.shape[0]
    out = np.zeros((rows, qweight.shape[1]), dtype=np.float32)
    for row, expert_value in enumerate(selected):
        expert = int(expert_value)
        if 0 <= expert < qweight.shape[0]:
            out[row] = dequant[expert] @ x[row // lanes_per_x_row]
    return _f32_to_bf16_u16(out)


def _run_selected(
    library,
    x_bf16: np.ndarray,
    selected: np.ndarray,
    qweight: np.ndarray,
    *,
    out_features: int,
) -> np.ndarray:
    out = np.zeros((selected.size, out_features), dtype=np.uint16)
    buffers = []
    try:
        x_buf = malloc(x_bf16.nbytes)
        selected_buf = malloc(selected.nbytes)
        weight_buf = malloc(qweight.nbytes)
        out_buf = malloc(out.nbytes)
        buffers.extend([x_buf, selected_buf, weight_buf, out_buf])
        copy_host_to_device(x_buf, host_array_ptr(x_bf16), x_bf16.nbytes)
        copy_host_to_device(selected_buf, host_array_ptr(selected), selected.nbytes)
        copy_host_to_device(weight_buf, host_array_ptr(qweight), qweight.nbytes)
        gguf_iq2_xs_selected_gemv_bf16_bf16_out(
            x_buf.ptr,
            selected_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            x_rows=x_bf16.shape[0],
            rows=selected.size,
            num_experts=qweight.shape[0],
            in_features=x_bf16.shape[1],
            out_features=out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def _run_dual(
    library,
    x_bf16: np.ndarray,
    selected: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    *,
    out_features: int,
) -> np.ndarray:
    out = np.zeros((selected.size, out_features), dtype=np.uint16)
    buffers = []
    try:
        arrays = [x_bf16, selected, gate, up]
        allocated = [malloc(array.nbytes) for array in arrays]
        out_buf = malloc(out.nbytes)
        buffers.extend([*allocated, out_buf])
        for buffer, array in zip(allocated, arrays, strict=True):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out(
            allocated[0].ptr,
            allocated[1].ptr,
            allocated[2].ptr,
            allocated[3].ptr,
            out_buf.ptr,
            x_rows=x_bf16.shape[0],
            rows=selected.size,
            num_experts=gate.shape[0],
            in_features=x_bf16.shape[1],
            out_features=out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer)


@pytest.mark.parametrize("in_features", [256, 3072])
def test_iq2_xs_selected_matches_cpu_oracle(iq_library, in_features: int) -> None:
    x = _f32_to_bf16_u16(_make_x(2, in_features))
    selected = np.asarray([0, 2, 1, -1], dtype=np.int64)
    weight = _make_iq2_xs_weight(3, 19, in_features, seed=0x12A5 + in_features)
    expected = _selected_reference(x, selected, weight)
    actual = _run_selected(iq_library, x, selected, weight, out_features=19)
    np.testing.assert_array_equal(actual, expected)


def test_iq2_xs_selected_covers_laguna_3072x1024_shape(iq_library) -> None:
    in_features = 3072
    out_features = 1024
    x = _f32_to_bf16_u16(_make_x(1, in_features))
    selected = np.asarray([0], dtype=np.int64)
    weight = _make_iq2_xs_weight(1, out_features, in_features, seed=0x1A6A)
    expected = _selected_reference(x, selected, weight)
    actual = _run_selected(
        iq_library, x, selected, weight, out_features=out_features
    )
    np.testing.assert_array_equal(actual, expected)


def test_iq2_xs_dual_silu_is_exact_to_single_projection_boundary(iq_library) -> None:
    in_features = 3072
    out_features = 23
    x = _f32_to_bf16_u16(_make_x(2, in_features))
    selected = np.asarray([1, 0, 0, 1], dtype=np.int64)
    gate = _make_iq2_xs_weight(2, out_features, in_features, seed=0x1A6B)
    up = _make_iq2_xs_weight(2, out_features, in_features, seed=0x1A6C)
    gate_bits = _run_selected(iq_library, x, selected, gate, out_features=out_features)
    up_bits = _run_selected(iq_library, x, selected, up, out_features=out_features)
    gate_f32 = _bf16_u16_to_f32(gate_bits)
    up_f32 = _bf16_u16_to_f32(up_bits)
    expected = _f32_to_bf16_u16(
        gate_f32 * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate_f32))) * up_f32
    )
    actual = _run_dual(
        iq_library, x, selected, gate, up, out_features=out_features
    )
    np.testing.assert_array_equal(actual, expected)


def test_iq2_xs_grid_supports_exact_branchless_magnitude_decoder() -> None:
    direct_source = _HIP_SOURCE.read_text()
    prefill_source = _PREFILL_SOURCE.read_text()
    match = re.search(
        r"IQ2_XS_GRID_PACKED\[512\] = \{(?P<body>.*?)\n\};",
        direct_source,
        re.DOTALL,
    )
    assert match is not None
    packed = [int(value, 16) for value in re.findall(r"0x([0-9a-f]+)u", match["body"])]
    assert len(packed) == 512
    codes = [(value >> shift) & 3 for value in packed for shift in range(0, 16, 2)]
    assert [codes.count(code) for code in range(4)] == [2114, 1142, 840, 0]
    assert [8 + 17 * code + (code >> 1) for code in range(3)] == [8, 25, 43]
    assert "iq2_xs_signed_magnitude" in direct_source
    assert "8U + 17U * code + (code >> 1)" in direct_source
    assert "iq2_xs_pair16_dot" in direct_source
    assert "const int pairs16 = in_features / 16" in direct_source
    assert "const int quads32 = in_features / 32" not in direct_source
    nested = "code == 0U ? 8.0f"
    assert nested not in direct_source
    assert nested not in prefill_source


def test_iq2_xs_registry_build_and_raw_pointer_contract() -> None:
    assert (
        inspect.signature(gguf_iq2_xs_selected_gemv_bf16_bf16_out)
        .parameters["threads"]
        .default
        == 64
    )
    assert (
        inspect.signature(gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out)
        .parameters["threads"]
        .default
        == 64
    )
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq2_xs",
        variant="selected_gemv_decode_bf16_bf16_out",
    ) is gguf_iq2_xs_selected_gemv_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq2_xs",
        variant="selected_dual_silu_gemv_decode_bf16_bf16_out",
    ) is gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out
    artifact = plan_gguf_iq_gemv_build(compiler_version="test-compiler")
    assert _HIP_SOURCE in artifact.sources
    source = _HIP_SOURCE.read_text()
    assert "torch::Tensor" not in source
    assert "IQ2_XS_BLOCK_BYTES = 74" in source
    assert "IQ2_XS_GRID_PACKED[512]" in source


def test_iq2_xs_wrappers_validate_contract_before_loading() -> None:
    with pytest.raises(ValueError, match="divisible by 256"):
        gguf_iq2_xs_selected_gemv_bf16_bf16_out(
            1,
            2,
            3,
            4,
            x_rows=1,
            rows=1,
            num_experts=1,
            in_features=3071,
            out_features=1024,
        )
    with pytest.raises(ValueError, match="divisible by x_rows"):
        gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out(
            1,
            2,
            3,
            4,
            5,
            x_rows=2,
            rows=3,
            num_experts=1,
            in_features=3072,
            out_features=1024,
        )
