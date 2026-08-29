"""Actual-weight RED for a Qwen3.8 exact Q5 T16 true-R12 target owner."""

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


def test_qwen38_q5_true_r12_matches_r8_r4_parent(
    monkeypatch: pytest.MonkeyPatch,
    hip_test_target_arch: str,
) -> None:
    if hip_test_target_arch != "gfx1151":
        pytest.skip("Qwen3.8 true-R12 target owner is gfx1151-only")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is unavailable")
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")

    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as kernels
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    candidate = getattr(
        kernels,
        "gguf_q5_k_t16_gemv_rowtile12_col8_bf16_bf16_out",
        None,
    )
    assert callable(candidate), "true-R12 Q5 owner is not implemented"
    parent = kernels.gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out
    runtime = get_hip_runtime()
    rows, in_features, out_features = 12, 6144, 5120
    guard_words = 64
    guard_value = np.uint16(0x55AA)
    hidden_host = _bf16(
        np.random.default_rng(0xE512)
        .normal(0.0, 0.2, size=(rows, in_features))
        .astype(np.float32)
    )

    with Qwen35GGUFResidentSession(
        MODEL,
        backend="hip_gfx1151",
        max_sequence_length=32,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        weight = session.runner.weights.layers[0].weights["ssm_out"]
        assert weight.spec.quant_key == "gguf_q5_k_t16_v1"
        assert tuple(weight.spec.source.shape) == (out_features, in_features)
        tiles = weight.allocation("tiles").tensor.ptr
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
            parent(hidden.ptr, tiles, reference.ptr, 8, in_features, out_features, runtime=runtime)
            parent(
                hidden.ptr + 8 * in_features * 2,
                tiles,
                reference.ptr + 8 * out_features * 2,
                4,
                in_features,
                out_features,
                runtime=runtime,
            )
            candidate(hidden.ptr, tiles, candidate_ptr, rows, in_features, out_features, runtime=runtime)
            runtime.device_synchronize()
            expected = np.empty((rows, out_features), dtype=np.uint16)
            actual = np.empty_like(expected)
            runtime.memcpy(host_array_ptr(expected), reference.ptr, expected.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            runtime.memcpy(host_array_ptr(actual), candidate_ptr, actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            runtime.memcpy(host_array_ptr(guarded_host), guarded.ptr, guarded_host.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            np.testing.assert_array_equal(actual, expected)
            assert np.all(guarded_host[:guard_words] == guard_value)
            assert np.all(guarded_host[-guard_words:] == guard_value)
        finally:
            for buffer in (hidden, reference, guarded):
                free(buffer, runtime=runtime)
