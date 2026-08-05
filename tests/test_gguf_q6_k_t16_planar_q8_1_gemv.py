"""Correctness gates for planar-Q6 Q8_1/sudot4 c1 and verifier row reuse."""

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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    build_gguf_q6_k_t16_gemv,
    gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_residual_bf16_out,
    gguf_q6_k_t16_qmicro_planar_q8_1_threads,
    register_gguf_q6_k_t16_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_x8_selected_gemv import (
    build_gguf_x8_selected_gemv,
    gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
from hipengine.quant.gguf_x8 import repack_gguf_q6_k_x8
from tests.test_gguf_x8_selected_gemv import (
    _bf16_bits,
    _bf16_to_f32,
    _exact_oracle,
    _q8_oracle,
    _softmax_kl,
    _top1,
    _weights,
)

Q8_1_BLOCK = 32
Q8_1_BLOCK_BYTES = 36


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
    return (
        build_gguf_q6_k_t16_gemv(load=True),
        build_gguf_q4_k_gemv(load=True),
        build_gguf_x8_selected_gemv(load=True),
    )


def _run_planar_and_x8(
    x_bits: np.ndarray,
    qweight: np.ndarray,
    *,
    threads: int,
    residual: np.ndarray | None = None,
    libraries,
) -> tuple[np.ndarray, np.ndarray]:
    from hipengine.core.hip import get_hip_runtime

    q6_library, q4_library, x8_library = libraries
    runtime = get_hip_runtime()
    rows, in_features = x_bits.shape
    out_features = int(qweight.shape[1])
    planar = repack_gguf_q6_k_tile16_qmicro_planar(qweight).tiles
    x8 = repack_gguf_q6_k_x8(qweight).tiles
    selected = np.zeros(rows, dtype=np.int64)
    candidate = np.zeros((rows, out_features), dtype=np.uint16)
    oracle = np.zeros_like(candidate)
    buffers = []
    try:
        def upload(value: np.ndarray):
            value = np.ascontiguousarray(value)
            buffer = malloc(value.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_host_to_device(
                buffer,
                host_array_ptr(value),
                value.nbytes,
                runtime=runtime,
            )
            return buffer

        x_buf = upload(x_bits)
        planar_buf = upload(planar)
        x8_buf = upload(x8)
        selected_buf = upload(selected)
        residual_buf = upload(residual) if residual is not None else None
        xq_buf = malloc(
            rows * (in_features // Q8_1_BLOCK) * Q8_1_BLOCK_BYTES,
            runtime=runtime,
        )
        candidate_buf = malloc(candidate.nbytes, runtime=runtime)
        oracle_buf = malloc(oracle.nbytes, runtime=runtime)
        buffers.extend((xq_buf, candidate_buf, oracle_buf))
        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr,
            xq_buf.ptr,
            rows,
            in_features,
            library=q4_library,
            runtime=runtime,
        )
        if residual_buf is None:
            gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out(
                xq_buf.ptr,
                planar_buf.ptr,
                candidate_buf.ptr,
                rows,
                in_features,
                out_features,
                threads=threads,
                library=q6_library,
                runtime=runtime,
            )
        else:
            gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_residual_bf16_out(
                xq_buf.ptr,
                planar_buf.ptr,
                residual_buf.ptr,
                candidate_buf.ptr,
                rows,
                in_features,
                out_features,
                threads=threads,
                library=q6_library,
                runtime=runtime,
            )
        gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out(
            xq_buf.ptr,
            selected_buf.ptr,
            x8_buf.ptr,
            oracle_buf.ptr,
            rows,
            rows,
            1,
            in_features,
            out_features,
            threads=threads,
            library=x8_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(candidate),
            candidate_buf,
            candidate.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(oracle),
            oracle_buf,
            oracle.nbytes,
            runtime=runtime,
        )
        return candidate, oracle
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def test_planar_q8_1_registry_and_shape_policy() -> None:
    register_gguf_q6_k_t16_gemv_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer="linear_q8_1",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_q8_1_dp4a_gemv_bf16_bf16_out",
    ) is gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear_q8_1+residual",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_q8_1_dp4a_gemv_bf16_residual_bf16_out",
    ) is gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_residual_bf16_out
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear_q8_1",
            quant="gguf_q6_k_t16_qmicro_planar_v1",
            variant="t16_q8_1_dp4a_gemv_bf16_bf16_out",
            missing="none",
        )
        is None
    )

    for rows in (1, 2, 3, 4):
        assert gguf_q6_k_t16_qmicro_planar_q8_1_threads(
            rows, 17_408, 5_120
        ) == 256
        assert gguf_q6_k_t16_qmicro_planar_q8_1_threads(
            rows, 5_120, 10_240
        ) == 64
    assert gguf_q6_k_t16_qmicro_planar_q8_1_threads(5, 17_408, 5_120) == 0
    assert gguf_q6_k_t16_qmicro_planar_q8_1_threads(4, 5_120, 248_320) == 0


def test_planar_q8_1_wrappers_validate_contract() -> None:
    with pytest.raises(ValueError, match="rows must be in"):
        gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out(
            1, 2, 3, 5, 512, 256
        )
    with pytest.raises(ValueError, match="64, 128, or 256"):
        gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out(
            1, 2, 3, 4, 512, 256, threads=32
        )
    with pytest.raises(ValueError, match="rows must be in"):
        gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_residual_bf16_out(
            1, 2, 3, 4, 1, 512, 256
        )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("threads", [64, 256])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_planar_q8_1_matches_x8_arithmetic_and_cpu_quality(
    rows: int,
    threads: int,
    libraries,
) -> None:
    rng = np.random.default_rng(0x6A380000 + rows * 1000 + threads)
    in_features, out_features = 512, 256
    qweight = _weights(
        "q6",
        out_features=out_features,
        in_features=in_features,
        experts=1,
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
        + 0.002
    )
    candidate, x8_oracle = _run_planar_and_x8(
        x_bits,
        qweight,
        threads=threads,
        libraries=libraries,
    )
    np.testing.assert_array_equal(candidate, x8_oracle)

    x_f32 = _bf16_to_f32(x_bits)
    x_rows = np.arange(rows, dtype=np.int64)
    selected = np.zeros(rows, dtype=np.int64)
    q8_reference = _q8_oracle("q6", x_f32, x_rows, selected, qweight)
    candidate_f32 = _bf16_to_f32(candidate)
    np.testing.assert_allclose(candidate_f32, q8_reference, rtol=2.0e-2, atol=2.0e-2)

    exact_reference = _exact_oracle("q6", x_f32, x_rows, selected, qweight)
    _, kl_max = _softmax_kl(exact_reference, candidate_f32)
    assert kl_max <= 0.05
    assert _top1(exact_reference, candidate_f32) >= 0.90


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_planar_q8_1_residual_matches_unfused_approximate_chain(libraries) -> None:
    rows, in_features, out_features = 4, 512, 256
    threads = 256
    rng = np.random.default_rng(0x6A38F00D)
    qweight = _weights(
        "q6",
        out_features=out_features,
        in_features=in_features,
        experts=1,
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
    )
    residual = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, out_features)).astype(np.float32)
    )
    projected, _ = _run_planar_and_x8(
        x_bits,
        qweight,
        threads=threads,
        libraries=libraries,
    )
    fused, _ = _run_planar_and_x8(
        x_bits,
        qweight,
        threads=threads,
        residual=residual,
        libraries=libraries,
    )
    expected = _bf16_bits(_bf16_to_f32(projected) + _bf16_to_f32(residual))
    np.testing.assert_array_equal(fused, expected)
