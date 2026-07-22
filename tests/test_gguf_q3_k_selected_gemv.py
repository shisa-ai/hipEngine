"""CPU-oracle and raw-pointer HIP gates for GGUF Q3_K selected experts."""

from __future__ import annotations

import base64
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_q3_k_gemv import (
    build_gguf_q3_k_gemv,
    gguf_q3_k_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_q3_k_selected_gemv_bf16_bf16_out,
    plan_gguf_q3_k_gemv_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data

_QK_K = 256
_BLOCK_BYTES = 110
_FIXTURE = Path(__file__).parent / "fixtures" / "gguf" / "q3km_iq_dequant_oracle.json"
_SOURCE = Path(__file__).parents[1] / "hipengine/kernels/hip_gfx1100/quant/gguf_q3_k_gemv.hip"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.fixture(scope="module")
def q3_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return build_gguf_q3_k_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


def _f32_to_bf16(array: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(array, dtype=np.float32)
    bits = f32.view(np.uint32).copy()
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    return ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)


def _bf16_to_f32(array: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(array, dtype=np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(bits.shape).copy()


def _make_x(rows: int, features: int) -> np.ndarray:
    values = np.arange(rows * features, dtype=np.int32).reshape(rows, features)
    return ((values % 31) - 15).astype(np.float32) / np.float32(64.0)


def _make_weight(experts: int, outputs: int, inputs: int, *, seed: int = 0x33) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = inputs // _QK_K
    out = rng.integers(0, 256, size=(experts, outputs, blocks * _BLOCK_BYTES), dtype=np.uint8)
    for block in range(blocks):
        start = block * _BLOCK_BYTES
        scale = np.asarray([np.float16(0.00390625 * (block + 1))], dtype=np.float16).view(np.uint8)
        out[..., start + 108 : start + 110] = scale
    return out


def _real_weight_row() -> tuple[np.ndarray, int]:
    fixture = json.loads(_FIXTURE.read_text())
    entry = next(item for item in fixture["tensors"] if item["ggml_type"] == "Q3_K")
    raw = np.frombuffer(base64.b64decode(entry["row_bytes_b64"]), dtype=np.uint8).copy()
    shape = tuple(int(value) for value in entry["expected_shape"])
    return raw.reshape(1, shape[0], raw.size // shape[0]), shape[1]


def _selected_cpu(
    x_bf16: np.ndarray,
    selected: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    x = _bf16_to_f32(x_bf16)
    lanes_per_row = selected.size // x.shape[0]
    out = np.zeros((selected.size, weight.shape[1]), dtype=np.float32)
    for row, expert_raw in enumerate(selected):
        expert = int(expert_raw)
        if expert < 0 or expert >= weight.shape[0]:
            continue
        x_row = 0 if x.shape[0] == 1 else row // lanes_per_row
        dequant = dequantize_gguf_data(weight[expert], GGMLQuantizationType.Q3_K)
        out[row] = np.matmul(x[x_row], dequant.T).astype(np.float32)
    return _f32_to_bf16(out)


def _device(array: np.ndarray, buffers: list) -> object:
    host = np.ascontiguousarray(array)
    buf = malloc(host.nbytes)
    copy_host_to_device(buf, host_array_ptr(host), host.nbytes)
    buffers.append(buf)
    return buf


def _run_single(
    library,
    x_bf16: np.ndarray,
    selected: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    x_bf16 = np.ascontiguousarray(x_bf16, dtype=np.uint16)
    selected = np.ascontiguousarray(selected, dtype=np.int64)
    weight = np.ascontiguousarray(weight, dtype=np.uint8)
    out = np.zeros((selected.size, weight.shape[1]), dtype=np.uint16)
    buffers: list = []
    try:
        x_buf = _device(x_bf16, buffers)
        selected_buf = _device(selected, buffers)
        weight_buf = _device(weight, buffers)
        out_buf = malloc(out.nbytes)
        buffers.append(out_buf)
        gguf_q3_k_selected_gemv_bf16_bf16_out(
            x_buf.ptr,
            selected_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            x_rows=x_bf16.shape[0],
            rows=selected.size,
            num_experts=weight.shape[0],
            in_features=x_bf16.shape[1],
            out_features=weight.shape[1],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)
    return out


def _run_dual(
    library,
    x_bf16: np.ndarray,
    selected: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    out = np.zeros((selected.size, gate.shape[1]), dtype=np.uint16)
    buffers: list = []
    try:
        x_buf = _device(x_bf16, buffers)
        selected_buf = _device(selected, buffers)
        gate_buf = _device(gate, buffers)
        up_buf = _device(up, buffers)
        out_buf = malloc(out.nbytes)
        buffers.append(out_buf)
        gguf_q3_k_selected_dual_silu_gemv_bf16_bf16_out(
            x_buf.ptr,
            selected_buf.ptr,
            gate_buf.ptr,
            up_buf.ptr,
            out_buf.ptr,
            x_rows=x_bf16.shape[0],
            rows=selected.size,
            num_experts=gate.shape[0],
            in_features=x_bf16.shape[1],
            out_features=gate.shape[1],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)
    return out


def test_q3_k_selected_registry_build_and_raw_pointer_contract() -> None:
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_q3_k",
        variant="selected_gemv_decode_bf16_bf16_out",
    ) is gguf_q3_k_selected_gemv_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_q3_k",
        variant="selected_dual_silu_gemv_decode_bf16_bf16_out",
    ) is gguf_q3_k_selected_dual_silu_gemv_bf16_bf16_out
    artifact = plan_gguf_q3_k_gemv_build(compiler_version="test")
    assert artifact.output_path.name == "gguf_q3_k_gemv.so"
    assert artifact.profile.name == "decode"
    assert build_gguf_q3_k_gemv(dry_run=True, compiler_version="test").output_path == artifact.output_path
    source = _SOURCE.read_text()
    assert "torch::Tensor" not in source
    assert "const uint8_t* __restrict__ qweight" in source


def test_q3_k_selected_wrapper_rejects_invalid_shape_before_loading() -> None:
    with pytest.raises(ValueError, match="divisible by 256"):
        gguf_q3_k_selected_gemv_bf16_bf16_out(
            1, 2, 3, 4, x_rows=1, rows=1, num_experts=1, in_features=255, out_features=1
        )
    with pytest.raises(ValueError, match="divisible by x_rows"):
        gguf_q3_k_selected_gemv_bf16_bf16_out(
            1, 2, 3, 4, x_rows=2, rows=3, num_experts=1, in_features=256, out_features=1
        )


@pytest.mark.parametrize("x_rows", [1, 2])
def test_q3_k_selected_gpu_matches_independent_cpu_dequant_oracle(q3_library, x_rows: int) -> None:
    x_bf16 = _f32_to_bf16(_make_x(x_rows, 512))
    selected = np.asarray([2, 0, 2, 1, 1, 0], dtype=np.int64)
    weight = _make_weight(3, 13, 512)
    actual = _run_single(q3_library, x_bf16, selected, weight)
    expected = _selected_cpu(x_bf16, selected, weight)
    actual_f32 = _bf16_to_f32(actual)
    expected_f32 = _bf16_to_f32(expected)
    max_rel = float(np.max(np.abs(actual_f32 - expected_f32) / np.maximum(np.abs(expected_f32), 1.0)))
    assert max_rel <= 0.02
    assert evaluate_logits(expected_f32, actual_f32).passed


def test_q3_k_selected_real_blk40_row_matches_cpu_oracle(q3_library) -> None:
    weight, inputs = _real_weight_row()
    x_bf16 = _f32_to_bf16(_make_x(1, inputs))
    selected = np.asarray([0], dtype=np.int64)
    actual = _run_single(q3_library, x_bf16, selected, weight)
    expected = _selected_cpu(x_bf16, selected, weight)
    actual_f32 = _bf16_to_f32(actual)
    expected_f32 = _bf16_to_f32(expected)
    max_rel = float(np.max(np.abs(actual_f32 - expected_f32) / np.maximum(np.abs(expected_f32), 1.0)))
    assert max_rel <= 0.02


def test_q3_k_dual_silu_matches_unfused_selected_math(q3_library) -> None:
    x_bf16 = _f32_to_bf16(_make_x(1, 512))
    selected = np.asarray([2, 0, 1], dtype=np.int64)
    gate = _make_weight(3, 11, 512, seed=0x33)
    up = _make_weight(3, 11, 512, seed=0x44)
    gate_out = _run_single(q3_library, x_bf16, selected, gate)
    up_out = _run_single(q3_library, x_bf16, selected, up)
    gate_f32 = _bf16_to_f32(gate_out)
    up_f32 = _bf16_to_f32(up_out)
    expected = _f32_to_bf16(
        gate_f32 * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate_f32).astype(np.float32))) * up_f32
    )
    actual = _run_dual(q3_library, x_bf16, selected, gate, up)
    np.testing.assert_array_equal(actual, expected)
