"""Actual-weight RED for Qwen3.8 exact Q5 true-R16 target owner."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"missing {MODEL}")


def test_qwen38_q5_true_r16_matches_r8_r8(
    monkeypatch: pytest.MonkeyPatch,
    hip_test_target_arch: str,
) -> None:
    if hip_test_target_arch != "gfx1151":
        pytest.skip("Qwen3.8 Q5 true-R16 candidate is gfx1151-only")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is unavailable")
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")

    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        gguf_q5_k_t16_gemv_rowtile16_col8_bf16_bf16_out as candidate,
        gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out as parent,
    )
    from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
        register_qwen35_paged_attn_decode_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    register_qwen35_paged_attn_decode_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    runtime = get_hip_runtime()
    rows, in_features, out_features = 16, 6_144, 5_120
    rng = np.random.default_rng(0xE516)
    hidden_f32 = np.ascontiguousarray(
        rng.normal(0.0, 0.2, size=(rows, in_features)), dtype=np.float32
    )
    hidden_host = (hidden_f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    guard_words = 64
    guard = np.uint16(0x55AA)
    guarded_host = np.full(rows * out_features + 2 * guard_words, guard, dtype=np.uint16)
    hidden = malloc(hidden_host.nbytes, runtime=runtime)
    expected_buf = malloc(rows * out_features * 2, runtime=runtime)
    guarded_buf = malloc(guarded_host.nbytes, runtime=runtime)
    try:
        runtime.memcpy(hidden.ptr, host_array_ptr(hidden_host), hidden_host.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        runtime.memcpy(guarded_buf.ptr, host_array_ptr(guarded_host), guarded_host.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        with Qwen35GGUFResidentSession(
            MODEL,
            backend="hip_gfx1151",
            max_sequence_length=32,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as session:
            tiles = session.runner.weights.layers[0].weights["ssm_out"].allocation("tiles").tensor.ptr
            parent(hidden.ptr, tiles, expected_buf.ptr, 8, in_features, out_features, runtime=runtime)
            parent(
                hidden.ptr + 8 * in_features * 2,
                tiles,
                expected_buf.ptr + 8 * out_features * 2,
                8,
                in_features,
                out_features,
                runtime=runtime,
            )
            actual_ptr = guarded_buf.ptr + guard_words * 2
            candidate(
                hidden.ptr,
                tiles,
                actual_ptr,
                rows,
                in_features,
                out_features,
                runtime=runtime,
            )
            runtime.device_synchronize()
        expected = np.empty((rows, out_features), dtype=np.uint16)
        actual = np.empty_like(expected)
        runtime.memcpy(host_array_ptr(expected), expected_buf.ptr, expected.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        runtime.memcpy(host_array_ptr(actual), actual_ptr, actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        runtime.memcpy(host_array_ptr(guarded_host), guarded_buf.ptr, guarded_host.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        np.testing.assert_array_equal(actual, expected)
        assert np.all(guarded_host[:guard_words] == guard)
        assert np.all(guarded_host[-guard_words:] == guard)
    finally:
        for buffer in (hidden, expected_buf, guarded_buf):
            free(buffer, runtime=runtime)
