"""Integer-WMMA prefill candidate for populated raw IQ2_XS experts."""

from __future__ import annotations

import ctypes
import os
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
from hipengine.kernels.hip_gfx1100.quant.gguf_iq2_xs_mmq_prefill import (
    build_gguf_iq2_xs_mmq_prefill,
    build_iq2_xs_mmq32_metadata,
    gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill import (
    build_gguf_iq_selected_prefill,
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
    build_gguf_q8_0_mmq_prefill,
    gguf_q8_0_mmq128_quantize_bf16_d4,
    q8_mmq_d4_nbytes,
)
from hipengine.kernels.registry import resolve
from tests.test_gguf_iq2_xs_selected_prefill import _weights
from tests.test_gguf_iq_gemv import _f32_to_bf16_u16, _make_x
from tests.test_gguf_iq_selected_prefill import (
    _bf16_u16_to_f32,
    _compact_meta,
    _run_dual_grouped,
)

_SOURCE = (
    Path(__file__).parents[1]
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "quant"
    / "gguf_iq2_xs_mmq_prefill.hip"
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def libraries():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    require_cached = os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1"
    return (
        build_gguf_iq2_xs_mmq_prefill(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        ),
        build_gguf_iq_selected_prefill(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        ),
        build_gguf_q8_0_mmq_prefill(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        ),
    )


def _copy(array: np.ndarray):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes)
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes)
    return buffer


def _run_candidate(
    libraries,
    *,
    x_bf16: np.ndarray,
    counts: list[int],
    gate: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    mmq_library, _, quant_library = libraries
    compact = _compact_meta(counts)
    mmq = build_iq2_xs_mmq32_metadata(counts)
    out = np.empty((compact.compact_rows, 2 * gate.shape[1]), dtype=np.uint16)
    buffers = []
    try:
        x_buf = _copy(x_bf16)
        starts_buf = _copy(compact.expert_start_compact)
        mmq_starts_buf = _copy(mmq.expert_start_mmq)
        tile_buf = _copy(mmq.tile_expert)
        gate_buf = _copy(gate)
        up_buf = _copy(up)
        xq_buf = malloc(q8_mmq_d4_nbytes(compact.compact_rows, x_bf16.shape[1]))
        out_buf = malloc(out.nbytes)
        buffers.extend(
            [
                x_buf,
                starts_buf,
                mmq_starts_buf,
                tile_buf,
                gate_buf,
                up_buf,
                xq_buf,
                out_buf,
            ]
        )
        gguf_q8_0_mmq128_quantize_bf16_d4(
            x_buf.ptr,
            xq_buf.ptr,
            compact.compact_rows,
            x_bf16.shape[1],
            library=quant_library,
        )
        gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out(
            xq_buf.ptr,
            starts_buf.ptr,
            mmq_starts_buf.ptr,
            tile_buf.ptr,
            gate_buf.ptr,
            up_buf.ptr,
            out_buf.ptr,
            compact_rows=compact.compact_rows,
            in_features=x_bf16.shape[1],
            out_features=gate.shape[1],
            num_experts=len(counts),
            mmq_total_rows=mmq.mmq_total_rows,
            library=mmq_library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def test_iq2_xs_mmq32_metadata_and_registry_contract() -> None:
    meta = build_iq2_xs_mmq32_metadata([0, 1, 31, 32, 33])
    np.testing.assert_array_equal(meta.expert_start_mmq, [0, 0, 32, 64, 96, 160])
    np.testing.assert_array_equal(meta.tile_expert, [1, 2, 3, 4, 4])
    assert meta.mmq_total_rows == 160
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq2_xs",
        variant="selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out",
    ) is gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out
    source = _SOURCE.read_text()
    assert "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32" in source
    assert "torch::Tensor" not in source


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_iq2_xs_mmq32_passes_populated_expert_quality_gate(libraries) -> None:
    counts = [1, 15, 16, 17, 31, 32, 33, 64]
    compact = _compact_meta(counts)
    in_features = 3072
    out_features = 32
    x = _f32_to_bf16_u16(_make_x(compact.compact_rows, in_features))
    gate, up = _weights(len(counts), out_features, in_features)
    expected = _run_dual_grouped(
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
        libraries[1],
        x_bf16=x,
        meta=compact,
        gate=gate,
        up=up,
        wmma=False,
    )
    actual = _run_candidate(
        libraries,
        x_bf16=x,
        counts=counts,
        gate=gate,
        up=up,
    )
    expected_f32 = _bf16_u16_to_f32(expected)
    actual_f32 = _bf16_u16_to_f32(actual)
    assert np.all(np.isfinite(actual_f32))
    max_rel = float(
        np.max(np.abs(actual_f32 - expected_f32) / np.maximum(np.abs(expected_f32), 1.0))
    )
    assert max_rel <= 0.05
    result = evaluate_logits(expected_f32, actual_f32)
    assert result.passed, result


def test_iq2_xs_mmq32_wrapper_validation() -> None:
    meta = build_iq2_xs_mmq32_metadata([1])
    with pytest.raises(ValueError, match="multiple of 32"):
        gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out(
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            compact_rows=1,
            in_features=3072,
            out_features=32,
            num_experts=1,
            mmq_total_rows=meta.mmq_total_rows - 1,
        )
