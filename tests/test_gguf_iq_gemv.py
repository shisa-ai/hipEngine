"""Raw IQ3_XXS/IQ4_XS selected-MoE GEMV correctness gates.

The GPU candidates are independently checked against the torch-free GGUF
NumPy dequantizers.  The fused IQ3 gate/up path is also checked against the
single-primitives plus the existing BF16 SiLU rounding contract, and the
routing-weighted IQ4 down composite is checked against selected-single outputs
followed by a host implementation of the registered weighted-sum fallback.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_gemv import (
    build_gguf_iq_gemv,
    gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
    gguf_iq3_xxs_selected_gemv_tile2_bf16_bf16_out,
    gguf_iq3_xxs_selected_gemv_tile4_bf16_bf16_out,
    gguf_iq3_xxs_weighted_selected_down_bf16_bf16_out,
    gguf_iq4_xs_selected_gemv_bf16_bf16_out,
    gguf_iq4_xs_weighted_selected_down_bf16_bf16_out,
    iq3_selected_default_threads,
    iq_weighted_down_default_threads,
    plan_gguf_iq_gemv_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data

QK_K = 256
IQ3_XXS_BLOCK_BYTES = 98
IQ4_XS_BLOCK_BYTES = 136
_FIXTURE = Path(__file__).parent / "fixtures" / "gguf" / "q3km_iq_dequant_oracle.json"
_HIP_SOURCE = (
    Path(__file__).parents[1]
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "quant"
    / "gguf_iq_gemv.hip"
)


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
    nan_mask = np.isnan(f32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    out = ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)
    out[nan_mask] = np.uint16(0x7FC0)
    return out.reshape(f32.shape)


def _bf16_u16_to_f32(array: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(array, dtype=np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(bits.shape).copy()


def _make_x(rows: int, features: int) -> np.ndarray:
    values = np.arange(rows * features, dtype=np.int32).reshape(rows, features)
    return ((values % 29) - 14).astype(np.float32) / np.float32(32.0)


def _make_iq3_weight(num_experts: int, out_features: int, in_features: int) -> np.ndarray:
    if in_features % QK_K:
        raise ValueError("in_features must be divisible by 256")
    rng = np.random.default_rng(0x1A3)
    blocks = in_features // QK_K
    out = np.empty(
        (num_experts, out_features, blocks * IQ3_XXS_BLOCK_BYTES), dtype=np.uint8
    )
    for expert in range(num_experts):
        for row in range(out_features):
            for block in range(blocks):
                start = block * IQ3_XXS_BLOCK_BYTES
                scale = np.float16(0.001953125 * (1 + (expert + row + block) % 5))
                out[expert, row, start : start + 2] = np.asarray(
                    [scale], dtype=np.float16
                ).view(np.uint8)
                out[expert, row, start + 2 : start + IQ3_XXS_BLOCK_BYTES] = rng.integers(
                    0, 256, size=96, dtype=np.uint8
                )
    return out


def _make_iq4_weight(num_experts: int, out_features: int, in_features: int) -> np.ndarray:
    if in_features % QK_K:
        raise ValueError("in_features must be divisible by 256")
    rng = np.random.default_rng(0x1A4)
    blocks = in_features // QK_K
    out = np.empty(
        (num_experts, out_features, blocks * IQ4_XS_BLOCK_BYTES), dtype=np.uint8
    )
    for expert in range(num_experts):
        for row in range(out_features):
            for block in range(blocks):
                start = block * IQ4_XS_BLOCK_BYTES
                scale = np.float16(0.0009765625 * (1 + (expert + row + block) % 7))
                out[expert, row, start : start + 2] = np.asarray(
                    [scale], dtype=np.float16
                ).view(np.uint8)
                out[expert, row, start + 2 : start + 8] = rng.integers(
                    0, 256, size=6, dtype=np.uint8
                )
                out[expert, row, start + 8 : start + IQ4_XS_BLOCK_BYTES] = rng.integers(
                    0, 256, size=128, dtype=np.uint8
                )
    return out


def _real_rows(qtype_name: str) -> tuple[np.ndarray, int]:
    payload = json.loads(_FIXTURE.read_text())
    tensor = next(item for item in payload["tensors"] if item["ggml_type"] == qtype_name)
    raw = np.frombuffer(base64.b64decode(tensor["row_bytes_b64"]), dtype=np.uint8).copy()
    expected_shape = tuple(int(value) for value in tensor["expected_shape"])
    row_bytes = raw.size // expected_shape[0]
    return raw.reshape(1, expected_shape[0], row_bytes), expected_shape[1]


def _selected_reference(
    x_bf16: np.ndarray,
    selected: np.ndarray,
    qweight: np.ndarray,
    qtype: GGMLQuantizationType,
) -> np.ndarray:
    x = _bf16_u16_to_f32(x_bf16)
    rows = int(selected.size)
    lanes_per_x_row = rows // x.shape[0]
    out_features = qweight.shape[1]
    out = np.zeros((rows, out_features), dtype=np.float32)
    for row, expert_value in enumerate(selected):
        expert = int(expert_value)
        if expert < 0 or expert >= qweight.shape[0]:
            continue
        x_row = 0 if x.shape[0] == 1 else row // lanes_per_x_row
        weight = dequantize_gguf_data(qweight[expert], qtype)
        out[row] = np.matmul(x[x_row].astype(np.float32), weight.T).astype(np.float32)
    return _f32_to_bf16_u16(out)


def _device_buffer(array: np.ndarray, buffers: list) -> object:
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes)
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes)
    buffers.append(buffer)
    return buffer


def _run_selected(
    launch,
    library,
    *,
    x_bf16: np.ndarray,
    selected: np.ndarray,
    qweight: np.ndarray,
    threads: int,
) -> np.ndarray:
    x_bf16 = np.ascontiguousarray(x_bf16, dtype=np.uint16)
    selected = np.ascontiguousarray(selected, dtype=np.int64)
    qweight = np.ascontiguousarray(qweight, dtype=np.uint8)
    out = np.zeros((selected.size, qweight.shape[1]), dtype=np.uint16)
    buffers: list = []
    try:
        x_buf = _device_buffer(x_bf16, buffers)
        selected_buf = _device_buffer(selected, buffers)
        weight_buf = _device_buffer(qweight, buffers)
        out_buf = malloc(out.nbytes)
        buffers.append(out_buf)
        launch(
            x_buf.ptr,
            selected_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            x_rows=x_bf16.shape[0],
            rows=selected.size,
            num_experts=qweight.shape[0],
            in_features=x_bf16.shape[1],
            out_features=qweight.shape[1],
            threads=threads,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        for buffer in buffers:
            free(buffer)
    return out


def _run_iq3_fused(
    library,
    *,
    x_bf16: np.ndarray,
    selected: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    x_bf16 = np.ascontiguousarray(x_bf16, dtype=np.uint16)
    selected = np.ascontiguousarray(selected, dtype=np.int64)
    gate = np.ascontiguousarray(gate, dtype=np.uint8)
    up = np.ascontiguousarray(up, dtype=np.uint8)
    out = np.zeros((selected.size, gate.shape[1]), dtype=np.uint16)
    buffers: list = []
    try:
        x_buf = _device_buffer(x_bf16, buffers)
        selected_buf = _device_buffer(selected, buffers)
        gate_buf = _device_buffer(gate, buffers)
        up_buf = _device_buffer(up, buffers)
        out_buf = malloc(out.nbytes)
        buffers.append(out_buf)
        gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out(
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
            threads=256,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        for buffer in buffers:
            free(buffer)
    return out


def _run_iq_weighted(
    launch,
    library,
    *,
    x_bf16: np.ndarray,
    selected: np.ndarray,
    routing: np.ndarray,
    qweight: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    x_bf16 = np.ascontiguousarray(x_bf16, dtype=np.uint16)
    selected = np.ascontiguousarray(selected, dtype=np.int64)
    routing = np.ascontiguousarray(routing, dtype=np.float32)
    qweight = np.ascontiguousarray(qweight, dtype=np.uint8)
    tokens, top_k = selected.shape
    out = np.zeros((tokens, qweight.shape[1]), dtype=np.uint16)
    buffers: list = []
    try:
        x_buf = _device_buffer(x_bf16, buffers)
        selected_buf = _device_buffer(selected, buffers)
        routing_buf = _device_buffer(routing, buffers)
        weight_buf = _device_buffer(qweight, buffers)
        out_buf = malloc(out.nbytes)
        buffers.append(out_buf)
        launch(
            x_buf.ptr,
            selected_buf.ptr,
            routing_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            tokens=tokens,
            top_k=top_k,
            num_experts=qweight.shape[0],
            in_features=x_bf16.shape[1],
            out_features=qweight.shape[1],
            threads=threads,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        for buffer in buffers:
            free(buffer)
    return out


def test_iq3_hip_table_matches_pinned_llamacpp_table() -> None:
    source = _HIP_SOURCE.read_text()
    initializer = source.split("IQ3_XXS_GRID[256] = {", 1)[1].split("};", 1)[0]
    values = np.asarray(
        [int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{8})u", initializer)],
        dtype="<u4",
    )
    assert values.size == 256
    assert hashlib.sha256(values.tobytes()).hexdigest() == (
        "46e35f5a997efdee6c99ce57854c8a0d4f0ff8ca57e5e8a60c0793ea580acf5d"
    )


def test_iq_gemv_registry_and_build_plan() -> None:
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant="selected_gemv_decode_bf16_bf16_out",
    ) is gguf_iq3_xxs_selected_gemv_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant="selected_dual_silu_gemv_decode_bf16_bf16_out",
    ) is gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant="selected_gemv_decode_tile2_bf16_bf16_out",
    ) is gguf_iq3_xxs_selected_gemv_tile2_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant="selected_gemv_decode_tile4_bf16_bf16_out",
    ) is gguf_iq3_xxs_selected_gemv_tile4_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq4_xs",
        variant="selected_gemv_decode_bf16_bf16_out",
    ) is gguf_iq4_xs_selected_gemv_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant="selected_weighted_down_gemv_decode_bf16_bf16_out",
    ) is gguf_iq3_xxs_weighted_selected_down_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq4_xs",
        variant="selected_weighted_down_gemv_decode_bf16_bf16_out",
    ) is gguf_iq4_xs_weighted_selected_down_bf16_bf16_out

    artifact = plan_gguf_iq_gemv_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_iq_gemv.so"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert any(path.name == "gguf_iq_gemv.hip" for path in artifact.sources)
    dry_run = build_gguf_iq_gemv(dry_run=True, compiler_version="test-compiler")
    assert dry_run.output_path == artifact.output_path


def test_iq_gemv_wrappers_validate_contract_before_loading() -> None:
    with pytest.raises(ValueError, match="divisible by 256"):
        gguf_iq3_xxs_selected_gemv_bf16_bf16_out(
            1, 2, 3, 4, x_rows=1, rows=1, num_experts=1, in_features=255, out_features=1
        )
    with pytest.raises(ValueError, match="divisible by x_rows"):
        gguf_iq4_xs_selected_gemv_bf16_bf16_out(
            1, 2, 3, 4, x_rows=2, rows=3, num_experts=1, in_features=256, out_features=1
        )
    with pytest.raises(ValueError, match="threads"):
        gguf_iq3_xxs_selected_gemv_bf16_bf16_out(
            1,
            2,
            3,
            4,
            x_rows=1,
            rows=1,
            num_experts=1,
            in_features=256,
            out_features=1,
            threads=32,
        )
    with pytest.raises(ValueError, match="top_k"):
        gguf_iq4_xs_weighted_selected_down_bf16_bf16_out(
            1,
            2,
            3,
            4,
            5,
            tokens=1,
            top_k=0,
            num_experts=1,
            in_features=256,
            out_features=1,
        )
    assert iq3_selected_default_threads(in_features=256) == 256
    assert iq3_selected_default_threads(in_features=1024) == 128
    assert iq3_selected_default_threads(in_features=2048) == 256
    with pytest.raises(ValueError, match="divisible by 256"):
        iq3_selected_default_threads(in_features=1023)
    assert iq_weighted_down_default_threads(top_k=8, in_features=512) == 128
    assert iq_weighted_down_default_threads(top_k=8, in_features=2048) == 256


@pytest.mark.parametrize(
    ("qtype", "make_weight", "launch", "threads"),
    [
        (
            GGMLQuantizationType.IQ3_XXS,
            _make_iq3_weight,
            gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
            256,
        ),
        (
            GGMLQuantizationType.IQ4_XS,
            _make_iq4_weight,
            gguf_iq4_xs_selected_gemv_bf16_bf16_out,
            128,
        ),
    ],
)
@pytest.mark.parametrize("x_rows", [1, 2])
def test_selected_iq_gpu_matches_cpu_dequant_oracle(
    iq_library, qtype, make_weight, launch, threads, x_rows
) -> None:
    x_bf16 = _f32_to_bf16_u16(_make_x(x_rows, 512))
    selected = np.asarray([2, 0, 2, 1, 1, 0], dtype=np.int64)
    qweight = make_weight(3, 11, 512)
    actual = _run_selected(
        launch,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        qweight=qweight,
        threads=threads,
    )
    expected = _selected_reference(x_bf16, selected, qweight, qtype)
    actual_f32 = _bf16_u16_to_f32(actual)
    expected_f32 = _bf16_u16_to_f32(expected)
    max_rel = float(
        np.max(np.abs(actual_f32 - expected_f32) / np.maximum(np.abs(expected_f32), 1.0))
    )
    assert max_rel <= 0.02
    result = evaluate_logits(expected_f32, actual_f32)
    assert result.passed, result


@pytest.mark.parametrize(
    ("qtype_name", "qtype", "launch", "threads"),
    [
        (
            "IQ3_XXS",
            GGMLQuantizationType.IQ3_XXS,
            gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
            256,
        ),
        (
            "IQ4_XS",
            GGMLQuantizationType.IQ4_XS,
            gguf_iq4_xs_selected_gemv_bf16_bf16_out,
            128,
        ),
    ],
)
def test_selected_iq_real_fixture_rows_match_cpu_oracle(
    iq_library, qtype_name, qtype, launch, threads
) -> None:
    qweight, in_features = _real_rows(qtype_name)
    x_bf16 = _f32_to_bf16_u16(_make_x(1, in_features))
    selected = np.asarray([0], dtype=np.int64)
    actual = _run_selected(
        launch,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        qweight=qweight,
        threads=threads,
    )
    expected = _selected_reference(x_bf16, selected, qweight, qtype)
    actual_f32 = _bf16_u16_to_f32(actual)
    expected_f32 = _bf16_u16_to_f32(expected)
    max_rel = float(
        np.max(np.abs(actual_f32 - expected_f32) / np.maximum(np.abs(expected_f32), 1.0))
    )
    assert max_rel <= 0.02


@pytest.mark.parametrize(
    "tiled_launch",
    [
        gguf_iq3_xxs_selected_gemv_tile2_bf16_bf16_out,
        gguf_iq3_xxs_selected_gemv_tile4_bf16_bf16_out,
    ],
)
@pytest.mark.parametrize("tokens", [1, 2, 5, 8])
def test_iq3_selected_output_tiles_are_bit_exact_to_tile1(
    iq_library, tiled_launch, tokens: int
) -> None:
    top_k = 10
    in_features = 1024
    out_features = 19
    x_bf16 = _f32_to_bf16_u16(_make_x(tokens, in_features))
    selected = np.asarray(
        [(-1 if lane % 17 == 0 else (7 * lane + 3) % 5) for lane in range(tokens * top_k)],
        dtype=np.int64,
    )
    qweight = _make_iq3_weight(5, out_features, in_features)
    tile1 = _run_selected(
        gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        qweight=qweight,
        threads=128,
    )
    tiled = _run_selected(
        tiled_launch,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        qweight=qweight,
        threads=128,
    )
    np.testing.assert_array_equal(tiled, tile1)


def test_iq3_laguna_k1024_local128_is_bit_exact_to_local256(iq_library) -> None:
    x_bf16 = _f32_to_bf16_u16(_make_x(2, 1024))
    selected = np.asarray([2, 0, 2, 1, 1, 0], dtype=np.int64)
    qweight = _make_iq3_weight(3, 19, 1024)
    local256 = _run_selected(
        gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        qweight=qweight,
        threads=256,
    )
    local128 = _run_selected(
        gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        qweight=qweight,
        threads=128,
    )
    np.testing.assert_array_equal(local128, local256)


def test_selected_iq_invalid_expert_emits_zero(iq_library) -> None:
    x_bf16 = _f32_to_bf16_u16(_make_x(1, 256))
    qweight = _make_iq3_weight(2, 5, 256)
    selected = np.asarray([-1, 2], dtype=np.int64)
    actual = _run_selected(
        gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        qweight=qweight,
        threads=256,
    )
    np.testing.assert_array_equal(actual, np.zeros_like(actual))


@pytest.mark.parametrize("x_rows", [1, 2])
def test_iq3_fused_matches_single_primitives_and_bf16_silu(
    iq_library, x_rows
) -> None:
    x_bf16 = _f32_to_bf16_u16(_make_x(x_rows, 512))
    selected = np.asarray([2, 0, 2, 1, 1, 0], dtype=np.int64)
    gate = _make_iq3_weight(3, 13, 512)
    up = _make_iq3_weight(3, 13, 512).copy()
    for block_start in range(0, up.shape[-1], IQ3_XXS_BLOCK_BYTES):
        payload = up[..., block_start + 2 : block_start + IQ3_XXS_BLOCK_BYTES]
        up[..., block_start + 2 : block_start + IQ3_XXS_BLOCK_BYTES] = np.roll(
            payload, shift=17, axis=-1
        )
    gate_out = _run_selected(
        gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        qweight=gate,
        threads=256,
    )
    up_out = _run_selected(
        gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        qweight=up,
        threads=256,
    )
    gate_f32 = _bf16_u16_to_f32(gate_out)
    up_f32 = _bf16_u16_to_f32(up_out)
    expected = _f32_to_bf16_u16(
        gate_f32
        * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate_f32).astype(np.float32)))
        * up_f32
    )
    actual = _run_iq3_fused(
        iq_library, x_bf16=x_bf16, selected=selected, gate=gate, up=up
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("tokens", [1, 2])
def test_iq4_weighted_down_matches_selected_single_fallback(iq_library, tokens: int) -> None:
    top_k = 8
    in_features = 512
    out_features = 17
    qweight = _make_iq4_weight(4, out_features, in_features)
    x_bf16 = _f32_to_bf16_u16(_make_x(tokens * top_k, in_features))
    selected = np.asarray(
        [[3, 0, 3, 1, 2, 1, -1, 2], [1, 4, 3, 0, 2, 3, 2, 0]], dtype=np.int64
    )[:tokens]
    routing = np.asarray(
        [
            [0.18, 0.16, 0.14, 0.13, 0.12, 0.11, 0.09, 0.07],
            [0.22, 0.18, 0.15, 0.12, 0.11, 0.09, 0.08, 0.05],
        ],
        dtype=np.float32,
    )[:tokens]
    single = _run_selected(
        gguf_iq4_xs_selected_gemv_bf16_bf16_out,
        iq_library,
        x_bf16=x_bf16,
        selected=selected.reshape(-1),
        qweight=qweight,
        threads=128,
    )
    single_f32 = _bf16_u16_to_f32(single).reshape(tokens, top_k, out_features)
    fallback = np.zeros((tokens, out_features), dtype=np.float32)
    for token in range(tokens):
        for slot in range(top_k):
            fallback[token] += routing[token, slot] * single_f32[token, slot]
    fallback_bf16 = _f32_to_bf16_u16(fallback)
    actual = _run_iq_weighted(
        gguf_iq4_xs_weighted_selected_down_bf16_bf16_out,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        routing=routing,
        qweight=qweight,
    )
    np.testing.assert_array_equal(actual, fallback_bf16)


@pytest.mark.parametrize("tokens", [1, 2])
def test_iq3_weighted_down_matches_local128_selected_single_fallback(
    iq_library, tokens: int
) -> None:
    top_k = 10
    in_features = 1024
    out_features = 17
    qweight = _make_iq3_weight(5, out_features, in_features)
    x_bf16 = _f32_to_bf16_u16(_make_x(tokens * top_k, in_features))
    selected = np.asarray(
        [
            [4, 0, 3, 1, 2, 1, -1, 2, 4, 0],
            [1, 5, 3, 0, 2, 4, 2, 0, 3, 1],
        ],
        dtype=np.int64,
    )[:tokens]
    routing = np.asarray(
        [
            [0.31, 0.27, 0.23, 0.19, 0.15, 0.11, 0.09, 0.07, 0.05, 0.03],
            [0.29, 0.25, 0.21, 0.18, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04],
        ],
        dtype=np.float32,
    )[:tokens]
    single = _run_selected(
        gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
        iq_library,
        x_bf16=x_bf16,
        selected=selected.reshape(-1),
        qweight=qweight,
        threads=128,
    )
    single_f32 = _bf16_u16_to_f32(single).reshape(tokens, top_k, out_features)
    fallback = np.zeros((tokens, out_features), dtype=np.float32)
    for token in range(tokens):
        for slot in range(top_k):
            fallback[token] = (
                fallback[token]
                + routing[token, slot] * single_f32[token, slot]
            ).astype(np.float32)
    fallback_bf16 = _f32_to_bf16_u16(fallback)
    actual = _run_iq_weighted(
        gguf_iq3_xxs_weighted_selected_down_bf16_bf16_out,
        iq_library,
        x_bf16=x_bf16,
        selected=selected,
        routing=routing,
        qweight=qweight,
    )
    np.testing.assert_array_equal(actual, fallback_bf16)
