"""Grouped q8_1 DP4A planar-Q5 sibling: contract, numerics, determinism.

C8 reduced-dequantization candidate: a grouped-row dp4a kernel for the Q5_K
qmicro-planar layout (rows 8-64). Arithmetic class is integer dp4a decode
(changed vs the raw mmq32 d4s4_f32 owner, which remains the registered strict
fallback until a production-profile L4 campaign). The numerical contract
mirrors the Q6 grouped dp4a test: q8-quantized CPU oracle agreement
(rtol 2e-2), outer KL/top-1 floor vs the exact reference, and run-to-run
determinism. The d4s4_f32 activation producer (gguf_q8_1_d4s4_f32_quantize_
bf16, 160-byte blocks) feeds both the candidate and the owner.
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
from hipengine.kernels.hip_gfx1100.quant.gguf_k_mmq_prefill import (
    build_gguf_k_mmq_prefill,
    gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out,
    gguf_q8_1_d4s4_f32_quantize_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_qmicro_planar_gemv import (
    build_gguf_q5_k_qmicro_planar_gemv,
    gguf_q5_k_qmicro_planar_q8_1_dp4a_grouped_bf16_bf16_out,
    gguf_q5_k_qmicro_planar_q8_1_dp4a_grouped_bf16_f32_out,
)
from hipengine.kernels.cpu_reference import gguf_q5_k_gemv
from hipengine.quant.gguf_t16 import (
    convert_gguf_q5_k_qmicro_tile16_to_planar,
    repack_gguf_q5_k_qmicro_tile16,
)
from tests.test_gguf_q6_k_t16_planar_q8_1_gemv import HIP_AVAILABLE
from tests.test_gguf_x8_selected_gemv import (
    _bf16_bits,
    _bf16_to_f32,
    _softmax_kl,
    _top1,
    _weights,
)

D4S4_F32_BLOCK_BYTES = 160


@pytest.fixture(scope="module")
def mmq_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_k_mmq_prefill(load=True)


@pytest.fixture(scope="module")
def planar_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q5_k_qmicro_planar_gemv(load=True)


def _planar_tiles(qweight: np.ndarray) -> np.ndarray:
    qmicro = repack_gguf_q5_k_qmicro_tile16(qweight)
    planar = convert_gguf_q5_k_qmicro_tile16_to_planar(qmicro)
    return np.ascontiguousarray(planar.tiles)


def _run_grouped(
    x_bits: np.ndarray,
    qweight: np.ndarray,
    planar_library,
    f32_out: bool = False,
):
    from hipengine.core.hip import get_hip_runtime

    quant_library = build_gguf_k_mmq_prefill(load=True)
    runtime = get_hip_runtime()
    rows, in_features = x_bits.shape
    out_features = int(qweight.shape[1])
    tiles = _planar_tiles(qweight)
    buffers = []

    try:
        def upload(value: np.ndarray):
            value = np.ascontiguousarray(value)
            buffer = malloc(value.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(value), runtime=runtime)
            return buffer

        x_buf = upload(x_bits)
        tiles_buf = upload(tiles)
        xq_buf = malloc(rows * (in_features // 128) * D4S4_F32_BLOCK_BYTES, runtime=runtime)
        out_buf = malloc(rows * out_features * 4, runtime=runtime)
        buffers.append(xq_buf)
        buffers.append(out_buf)

        gguf_q8_1_d4s4_f32_quantize_bf16(
            x_buf.ptr,
            xq_buf.ptr,
            rows,
            in_features,
            library=quant_library,
            runtime=runtime,
        )
        wrapper = (
            gguf_q5_k_qmicro_planar_q8_1_dp4a_grouped_bf16_f32_out
            if f32_out
            else gguf_q5_k_qmicro_planar_q8_1_dp4a_grouped_bf16_bf16_out
        )
        wrapper(
            xq_buf.ptr,
            tiles_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=planar_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        dtype = np.float32 if f32_out else np.uint16
        first = np.empty((rows, out_features), dtype=dtype)
        copy_device_to_host(
            host_array_ptr(first), out_buf, first.nbytes, runtime=runtime
        )
        wrapper(
            xq_buf.ptr,
            tiles_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=planar_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        second = np.empty_like(first)
        copy_device_to_host(
            host_array_ptr(second), out_buf, second.nbytes, runtime=runtime
        )
        return first, second
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _run_owner(x_bits: np.ndarray, qweight: np.ndarray, library) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rows, in_features = x_bits.shape
    out_features = int(qweight.shape[1])
    buffers = []
    try:
        def upload(value: np.ndarray):
            value = np.ascontiguousarray(value)
            buffer = malloc(value.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(value), runtime=runtime)
            return buffer

        x_buf = upload(x_bits)
        weight_buf = upload(qweight[0])
        xq_buf = malloc(rows * (in_features // 128) * D4S4_F32_BLOCK_BYTES, runtime=runtime)
        out_buf = malloc(rows * out_features * 4, runtime=runtime)
        buffers.append(xq_buf)
        buffers.append(out_buf)
        gguf_q8_1_d4s4_f32_quantize_bf16(
            x_buf.ptr, xq_buf.ptr, rows, in_features, library=library, runtime=runtime
        )
        gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out(
            xq_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        out = np.empty((rows, out_features), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), out_buf, runtime=runtime)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_wrapper_rejects_bad_rows() -> None:
    for rows in (0, 4, 7, 12, 72):
        with pytest.raises(ValueError):
            gguf_q5_k_qmicro_planar_q8_1_dp4a_grouped_bf16_bf16_out(
                1,
                1,
                1,
                rows,
                512,
                16,
                library=ctypes.CDLL("libc.so.6"),
            )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_runner_planar_dp4a_gate_defaults_off() -> None:
    from hipengine.runtime.qwen35_gguf_runner import (
        _gguf_c8_q5_planar_dp4a_enabled,
    )

    assert _gguf_c8_q5_planar_dp4a_enabled("hip_gfx1100", request_count=8) is False


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [8, 16, 24, 32])
def test_grouped_dp4a_matches_bf16_reference_with_outer_floor(
    rows: int, mmq_library, planar_library
) -> None:
    """BF16-reconstructed-x CPU reference + outer KL/top-1 floor.

    Oracle convention matches tests/test_gguf_k_mmq_direct_dp4a_q5.py: the
    d4s4_f32 quantized-activation path (owner and candidate) is compared to
    the exact-weight CPU GEMV over the BF16-reconstructed activations.
    """
    rng = np.random.default_rng(0x6A350500 + rows)
    in_features, out_features = 512, 256
    qweight = _weights(
        "q5",
        out_features=out_features,
        in_features=in_features,
        experts=1,
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32) + 0.002
    )
    first, second = _run_grouped(x_bits, qweight, planar_library, f32_out=True)
    np.testing.assert_array_equal(first, second)  # deterministic

    x_f32 = _bf16_to_f32(x_bits)
    reference = gguf_q5_k_gemv(x_f32, qweight[0])
    candidate_f32 = np.asarray(first, dtype=np.float32)

    # Owner parity: the registered raw mmq32 d4s4_f32 owner on the same
    # quantized activations. The candidate reassociates only the f32
    # accumulation, so it must track the owner to float noise.
    owner_f32 = _run_owner(x_bits, qweight, mmq_library)
    np.testing.assert_allclose(candidate_f32, owner_f32, rtol=1.0e-4, atol=1.0e-4)

    # Outer smoke/safety floor vs the exact-weight CPU reference (the same
    # class the owner's own d4s4 error sits inside).
    _, kl_max = _softmax_kl(reference, candidate_f32)
    assert kl_max <= 0.05
    assert _top1(reference, candidate_f32) >= 0.90


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_grouped_dp4a_bf16_out_deterministic(mmq_library, planar_library) -> None:
    rng = np.random.default_rng(0x6A350577)
    in_features, out_features = 512, 256
    qweight = _weights(
        "q5",
        out_features=out_features,
        in_features=in_features,
        experts=1,
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(16, in_features)).astype(np.float32) + 0.002
    )
    first, second = _run_grouped(x_bits, qweight, planar_library, f32_out=False)
    np.testing.assert_array_equal(first, second)
    assert int(np.isfinite(_bf16_to_f32(first)).all())
