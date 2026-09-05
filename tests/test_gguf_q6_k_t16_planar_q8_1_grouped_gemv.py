"""Grouped q8_1 DP4A planar-Q6 sibling: registry, contract, and numerics.

C8-P4 reduced-dequantization candidate (iteration 33): a grouped-row variant
of the registered rows 1-4 dp4a kernel for production rows 8-64. Arithmetic
class is integer dp4a decode (changed vs the BF16 grouped-R8 owner, which
remains the registered strict fallback until a production-profile L4
campaign). The numerical contract mirrors the registered rows 1-4 dp4a test:
q8-quantized CPU oracle agreement (rtol 2e-2), outer KL/top-1 floor vs the
exact reference, and run-to-run determinism.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
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
    gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_grouped_bf16_bf16_out,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
from tests.test_gguf_q6_k_t16_planar_q8_1_gemv import HIP_AVAILABLE
from tests.test_gguf_x8_selected_gemv import (
    _bf16_bits,
    _bf16_to_f32,
    _exact_oracle,
    _q8_oracle,
    _softmax_kl,
    _top1,
    _weights,
)

Q8_1_BLOCK_BYTES = 36


def _hip_available() -> bool:
    try:
        import ctypes as _ctypes

        _ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
class TestGroupedPlanarQ8_1Registry:
    def test_symbol_registered(self) -> None:
        from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
            register_gguf_q6_k_t16_gemv_kernels,
        )

        register_gguf_q6_k_t16_gemv_kernels()
        entry = resolve(
            backend="hip_gfx1100",
            layer="linear_q8_1",
            quant="gguf_q6_k_t16_qmicro_planar_v1",
            variant="t16_q8_1_dp4a_gemv_grouped_bf16_bf16_out",
        )
        assert entry is not None

    def test_wrapper_rejects_bad_rows(self) -> None:
        for rows in (0, 4, 7, 12, 72):
            with pytest.raises(ValueError):
                gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_grouped_bf16_bf16_out(
                    1,
                    1,
                    1,
                    rows,
                    256,
                    16,
                    library=ctypes.CDLL("libc.so.6"),
                )


def _run_grouped(
    x_bits: np.ndarray,
    qweight: np.ndarray,
    libraries,
):
    from hipengine.core.hip import get_hip_runtime

    q6_library, q4_library = libraries
    runtime = get_hip_runtime()
    rows, in_features = x_bits.shape
    out_features = int(qweight.shape[1])
    planar = np.ascontiguousarray(
        repack_gguf_q6_k_tile16_qmicro_planar(qweight).tiles
    )
    buffers = []

    try:
        def upload(value: np.ndarray):
            value = np.ascontiguousarray(value)
            buffer = malloc(value.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_device_to_host  # noqa: B018 - keep import referenced
            from hipengine.core.memory import copy_host_to_device

            copy_host_to_device(
                buffer,
                host_array_ptr(value),
                value.nbytes,
                runtime=runtime,
            )
            return buffer

        x_buf = upload(x_bits)
        planar_buf = upload(planar)
        xq_buf = malloc(
            rows * (in_features // 32) * Q8_1_BLOCK_BYTES,
            runtime=runtime,
        )
        out_buf = malloc(rows * out_features * 2, runtime=runtime)
        buffers.append(xq_buf)
        buffers.append(out_buf)

        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr,
            xq_buf.ptr,
            rows,
            in_features,
            library=q4_library,
            runtime=runtime,
        )
        gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_grouped_bf16_bf16_out(
            xq_buf.ptr,
            planar_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=q6_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        first = np.empty((rows, out_features), dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(first),
            out_buf,
            first.nbytes,
            runtime=runtime,
        )
        gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_grouped_bf16_bf16_out(
            xq_buf.ptr,
            planar_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=q6_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        second = np.empty_like(first)
        copy_device_to_host(
            host_array_ptr(second),
            out_buf,
            second.nbytes,
            runtime=runtime,
        )
        return first, second
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.fixture(scope="module")
def grouped_libraries():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q6_k_t16_gemv(load=True), build_gguf_q4_k_gemv(load=True)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_decode_wrapper_session_routes_to_grouped_dp4a(grouped_libraries) -> None:
    """The production decode wrapper honors the owner-controlled session."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out,
        q6_dp4a_grouped_target_session,
    )

    q6_library, _q4_library = grouped_libraries
    runtime = get_hip_runtime()
    rows, in_features, out_features = 32, 512, 256
    qweight = _weights(
        "q6",
        out_features=out_features,
        in_features=in_features,
        experts=1,
    )
    rng = np.random.default_rng(0x6A380200)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32) + 0.002
    )
    planar = np.ascontiguousarray(
        repack_gguf_q6_k_tile16_qmicro_planar(qweight).tiles
    )
    buffers = []

    try:
        def upload(value: np.ndarray):
            value = np.ascontiguousarray(value)
            buffer = malloc(value.nbytes, runtime=runtime)
            buffers.append(buffer)
            from hipengine.core.memory import copy_host_to_device

            copy_host_to_device(
                buffer,
                host_array_ptr(value),
                value.nbytes,
                runtime=runtime,
            )
            return buffer

        x_buf = upload(x_bits)
        tiles_buf = upload(planar)
        out_buf = malloc(rows * out_features * 2, runtime=runtime)
        workspace = malloc(rows * (in_features // 32) * 36, runtime=runtime)
        buffers.extend((out_buf, workspace))

        def run_default() -> np.ndarray:
            gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
                x_buf.ptr,
                tiles_buf.ptr,
                out_buf.ptr,
                rows,
                in_features,
                out_features,
                library=q6_library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            host = np.empty((rows, out_features), dtype=np.uint16)
            copy_device_to_host(
                host_array_ptr(host), out_buf, host.nbytes, runtime=runtime
            )
            return host

        exact = run_default()
        with q6_dp4a_grouped_target_session(
            workspace_ptr=int(workspace.ptr),
            workspace_nbytes=int(workspace.nbytes),
            enabled=True,
        ):
            candidate_first = run_default()
            candidate_second = run_default()
        exact_again = run_default()

        np.testing.assert_array_equal(exact, exact_again)
        np.testing.assert_array_equal(candidate_first, candidate_second)
        assert int(np.count_nonzero(candidate_first != exact)) > 0
        x_f32 = _bf16_to_f32(x_bits)
        x_rows = np.arange(rows, dtype=np.int64)
        selected = np.zeros(rows, dtype=np.int64)
        q8_reference = _q8_oracle("q6", x_f32, x_rows, selected, qweight)
        np.testing.assert_allclose(
            _bf16_to_f32(candidate_first),
            q8_reference,
            rtol=2.0e-2,
            atol=2.0e-2,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_runner_dp4a_grouped_gate_is_qwen38_q4km_c8_only(monkeypatch) -> None:
    from hipengine.runtime.qwen35_gguf_runner import (
        _GGUF_C8_Q6_DP4A_GROUPED_ENV,
        _gguf_c8_q6_dp4a_grouped_enabled,
    )

    kwargs = {
        "request_count": 8,
        "model_name": "Qwen3.8-27B",
        "file_type_name": "MOSTLY_Q4_K_M",
    }
    monkeypatch.delenv(_GGUF_C8_Q6_DP4A_GROUPED_ENV, raising=False)
    assert _gguf_c8_q6_dp4a_grouped_enabled("hip_gfx1100", **kwargs) is False

    monkeypatch.setenv(_GGUF_C8_Q6_DP4A_GROUPED_ENV, "1")
    assert _gguf_c8_q6_dp4a_grouped_enabled("hip_gfx1100", **kwargs) is True
    assert _gguf_c8_q6_dp4a_grouped_enabled(
        "hip_gfx1100", **(kwargs | {"request_count": 7})
    ) is False
    assert _gguf_c8_q6_dp4a_grouped_enabled(
        "hip_gfx1100", **(kwargs | {"model_name": "Qwen3.6-27B"})
    ) is False
    assert _gguf_c8_q6_dp4a_grouped_enabled(
        "hip_gfx1100", **(kwargs | {"file_type_name": "MOSTLY_Q4_K_S"})
    ) is False
    assert _gguf_c8_q6_dp4a_grouped_enabled("hip_gfx1151", **kwargs) is False


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [8, 16, 32])
def test_grouped_dp4a_matches_q8_oracle_with_outer_floor(
    rows: int, grouped_libraries
) -> None:
    rng = np.random.default_rng(0x6A380100 + rows)
    in_features, out_features = 512, 256
    qweight = _weights(
        "q6",
        out_features=out_features,
        in_features=in_features,
        experts=1,
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32) + 0.002
    )
    first, second = _run_grouped(x_bits, qweight, grouped_libraries)
    np.testing.assert_array_equal(first, second)  # deterministic

    x_f32 = _bf16_to_f32(x_bits)
    x_rows = np.arange(rows, dtype=np.int64)
    selected = np.zeros(rows, dtype=np.int64)
    q8_reference = _q8_oracle("q6", x_f32, x_rows, selected, qweight)
    candidate_f32 = _bf16_to_f32(first)
    np.testing.assert_allclose(candidate_f32, q8_reference, rtol=2.0e-2, atol=2.0e-2)

    exact_reference = _exact_oracle("q6", x_f32, x_rows, selected, qweight)
    _, kl_max = _softmax_kl(exact_reference, candidate_f32)
    assert kl_max <= 0.05
    assert _top1(exact_reference, candidate_f32) >= 0.90
