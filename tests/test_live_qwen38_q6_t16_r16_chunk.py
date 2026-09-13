"""Actual-weight RED for Qwen3.8 exact Q6 R16 rowtile chunking."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"missing {MODEL}")


def _bf16(value: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(value, dtype=np.float32)
    return (value.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def test_qwen38_q6_r16_chunks_match_current_parent(
    monkeypatch: pytest.MonkeyPatch,
    hip_test_target_arch: str,
) -> None:
    if hip_test_target_arch != "gfx1151":
        pytest.skip("Qwen3.8 Q6 R16 target candidate is gfx1151-only")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is unavailable")
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")

    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_t16_gemv as kernels
    from hipengine.runtime.gguf_linear import (
        launch_gguf_linear,
        target_verifier_production_q4_rowtile_session,
        target_verifier_rowtile_session,
    )
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    runtime = get_hip_runtime()
    rows = 16
    guard_words = 64
    guard_value = np.uint16(0x55AA)
    rng = np.random.default_rng(0xE616)
    with Qwen35GGUFResidentSession(
        MODEL,
        backend="hip_gfx1151",
        max_sequence_length=32,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        weights = session.runner.weights
        cases: dict[tuple[str, int, int], object] = {}
        for layer in weights.layers:
            for weight in layer.weights.values():
                quant = str(weight.spec.quant_key)
                if quant not in {
                    "gguf_q6_k_t16_v1",
                    "gguf_q6_k_t16_qmicro_planar_v1",
                }:
                    continue
                out_features, in_features = map(int, weight.spec.source.shape)
                cases.setdefault((quant, in_features, out_features), weight)
        assert set(cases) == {
            ("gguf_q6_k_t16_v1", 5120, 10240),
            ("gguf_q6_k_t16_qmicro_planar_v1", 5120, 1024),
            ("gguf_q6_k_t16_qmicro_planar_v1", 17408, 5120),
        }

        for (quant, in_features, out_features), weight in cases.items():
            hidden_host = _bf16(
                rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
            )
            hidden = malloc(hidden_host.nbytes, runtime=runtime)
            reference = malloc(rows * out_features * 2, runtime=runtime)
            guarded = malloc((rows * out_features + 2 * guard_words) * 2, runtime=runtime)
            guarded_host = np.full(
                (rows * out_features + 2 * guard_words,), guard_value, dtype=np.uint16
            )
            candidate_ptr = guarded.ptr + guard_words * 2
            try:
                runtime.memcpy(hidden.ptr, host_array_ptr(hidden_host), hidden_host.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
                runtime.memcpy(guarded.ptr, host_array_ptr(guarded_host), guarded_host.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
                with (
                    target_verifier_rowtile_session(True),
                    target_verifier_production_q4_rowtile_session(True),
                ):
                    launch_gguf_linear(
                        weight,
                        hidden.ptr,
                        reference.ptr,
                        rows,
                        in_features,
                        out_features,
                        backend="hip_gfx1151",
                        use_wmma_prefill=False,
                        runtime=runtime,
                    )
                if quant == "gguf_q6_k_t16_v1":
                    candidate = kernels.gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out
                else:
                    candidate = kernels.gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out
                tiles = weight.allocation("tiles").tensor.ptr
                for row_base in (0, 8):
                    candidate(
                        hidden.ptr + row_base * in_features * 2,
                        tiles,
                        candidate_ptr + row_base * out_features * 2,
                        8,
                        in_features,
                        out_features,
                        runtime=runtime,
                    )
                runtime.device_synchronize()
                expected = np.empty((rows, out_features), dtype=np.uint16)
                actual = np.empty_like(expected)
                runtime.memcpy(host_array_ptr(expected), reference.ptr, expected.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                runtime.memcpy(host_array_ptr(actual), candidate_ptr, actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                runtime.memcpy(host_array_ptr(guarded_host), guarded.ptr, guarded_host.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                np.testing.assert_array_equal(actual, expected, err_msg=str((quant, in_features, out_features)))
                assert np.all(guarded_host[:guard_words] == guard_value)
                assert np.all(guarded_host[-guard_words:] == guard_value)
            finally:
                for buffer in (hidden, reference, guarded):
                    free(buffer, runtime=runtime)
