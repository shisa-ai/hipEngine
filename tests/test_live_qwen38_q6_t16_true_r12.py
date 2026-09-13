"""Actual-weight RED for the retained Qwen3.8 standard-Q6 true-R12 owner."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest


MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason=f"local GGUF fixture not found: {MODEL}",
)


def _f32_to_bf16_bits(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    return (contiguous.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def test_qwen38_standard_q6_true_r12_matches_r8_r4_parent(
    monkeypatch: pytest.MonkeyPatch,
    hip_test_target_arch: str,
) -> None:
    if hip_test_target_arch != "gfx1151":
        pytest.skip("Qwen3.8 true-R12 target owner is qualified only on gfx1151")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")

    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")

    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        gguf_q6_k_t16_gemv_rowtile12_col8_bf16_bf16_out as candidate,
        gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out as parent,
    )
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    runtime = get_hip_runtime()
    rows = 12
    in_features = 5120
    out_features = 10240
    guard_words = 64
    guard_value = np.uint16(0x55AA)
    rng = np.random.default_rng(0xE612)
    hidden_host = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )

    with Qwen35GGUFResidentSession(
        MODEL,
        backend="hip_gfx1151",
        max_batch_size=1,
        max_sequence_length=32,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        assert session.runner is not None
        weights = session.runner.weights
        assert weights is not None
        weight = weights.layers[0].weights["attn_qkv"]
        assert weight.spec.quant_key == "gguf_q6_k_t16_v1"
        assert tuple(weight.spec.source.shape) == (out_features, in_features)
        tiles = weight.allocation("tiles").tensor.ptr

        hidden = malloc(hidden_host.nbytes, runtime=runtime)
        reference = malloc(rows * out_features * 2, runtime=runtime)
        guarded = malloc((rows * out_features + 2 * guard_words) * 2, runtime=runtime)
        guarded_host = np.full(
            (rows * out_features + 2 * guard_words,),
            guard_value,
            dtype=np.uint16,
        )
        candidate_ptr = guarded.ptr + guard_words * 2
        try:
            runtime.memcpy(
                hidden.ptr,
                host_array_ptr(hidden_host),
                hidden_host.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )
            runtime.memcpy(
                guarded.ptr,
                host_array_ptr(guarded_host),
                guarded_host.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )
            parent(
                hidden.ptr,
                tiles,
                reference.ptr,
                8,
                in_features,
                out_features,
                runtime=runtime,
            )
            parent(
                hidden.ptr + 8 * in_features * 2,
                tiles,
                reference.ptr + 8 * out_features * 2,
                4,
                in_features,
                out_features,
                runtime=runtime,
            )
            candidate(
                hidden.ptr,
                tiles,
                candidate_ptr,
                rows,
                in_features,
                out_features,
                runtime=runtime,
            )
            runtime.device_synchronize()

            reference_host = np.empty((rows, out_features), dtype=np.uint16)
            candidate_host = np.empty((rows, out_features), dtype=np.uint16)
            runtime.memcpy(
                host_array_ptr(reference_host),
                reference.ptr,
                reference_host.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            runtime.memcpy(
                host_array_ptr(candidate_host),
                candidate_ptr,
                candidate_host.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            runtime.memcpy(
                host_array_ptr(guarded_host),
                guarded.ptr,
                guarded_host.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            np.testing.assert_array_equal(candidate_host, reference_host)
            assert np.all(guarded_host[:guard_words] == guard_value)
            assert np.all(guarded_host[-guard_words:] == guard_value)
        finally:
            for buffer in (hidden, reference, guarded):
                free(buffer, runtime=runtime)
