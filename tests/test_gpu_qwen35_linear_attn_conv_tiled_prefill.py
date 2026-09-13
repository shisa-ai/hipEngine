from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_prefill_f32,
    qwen35_linear_attn_conv_prefill_f32_tile32x128,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module", autouse=True)
def _build_for_detected_target(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment

    with hip_target_arch_environment(hip_test_target_arch):
        yield


class _Buf:
    def __init__(self, nbytes: int) -> None:
        self.buffer = malloc(nbytes)

    @property
    def ptr(self) -> int:
        return self.buffer.ptr

    def free(self) -> None:
        if self.buffer is not None:
            free(self.buffer)
            self.buffer = None


def _to_device(array: np.ndarray) -> _Buf:
    array = np.ascontiguousarray(array)
    buf = _Buf(array.nbytes)
    copy_host_to_device(buf.buffer, host_array_ptr(array), array.nbytes)
    return buf


def _from_device(buf: _Buf, shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty(shape, dtype=np.float32)
    copy_device_to_host(host_array_ptr(out), buf.buffer, out.nbytes)
    return out


def _run(
    launch,
    hidden: np.ndarray,
    state: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tokens, channels = hidden.shape
    kernel_size = state.shape[1]
    hidden_buf = _to_device(hidden)
    state_buf = _to_device(state)
    weight_buf = _to_device(weight)
    out_buf = _Buf(hidden.nbytes)
    try:
        launch(
            hidden_buf.ptr,
            state_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            tokens,
            channels,
            kernel_size,
        )
        return (
            _from_device(out_buf, hidden.shape),
            _from_device(state_buf, state.shape),
        )
    finally:
        for buf in (hidden_buf, state_buf, weight_buf, out_buf):
            buf.free()


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("tokens", [4, 31, 32, 33, 512, 4096])
def test_tile32x128_prefill_matches_production_output_and_final_state(tokens: int) -> None:
    rng = np.random.default_rng(12000 + tokens)
    channels = 257
    kernel_size = 4
    hidden = rng.normal(0.0, 0.5, size=(tokens, channels)).astype(np.float32)
    state = rng.normal(0.0, 0.25, size=(channels, kernel_size)).astype(np.float32)
    weight = rng.normal(0.0, 0.2, size=(channels, kernel_size)).astype(np.float32)

    expected_out, expected_state = _run(
        qwen35_linear_attn_conv_prefill_f32,
        hidden,
        state,
        weight,
    )
    candidate_out, candidate_state = _run(
        qwen35_linear_attn_conv_prefill_f32_tile32x128,
        hidden,
        state,
        weight,
    )

    np.testing.assert_array_equal(candidate_out, expected_out)
    np.testing.assert_array_equal(candidate_state, expected_state)
