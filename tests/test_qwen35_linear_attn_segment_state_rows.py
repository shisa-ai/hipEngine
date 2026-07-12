from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_prefill_f32,
    qwen35_linear_attn_conv_prefill_f32_state_rows,
    qwen35_linear_attn_conv_prefill_segments_f32,
    qwen35_linear_attn_conv_prefill_segments_f32_state_rows,
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


def _from_device(buf: _Buf, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), buf.buffer, out.nbytes)
    return out


def _run_single_conv_state_rows(
    hidden: np.ndarray,
    conv_state: np.ndarray,
    conv_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens, channels = hidden.shape
    kernel_size = conv_state.shape[1]
    hidden_buf = _to_device(hidden)
    state_buf = _to_device(conv_state)
    weight_buf = _to_device(conv_weight)
    out_buf = _Buf(tokens * channels * np.dtype(np.float32).itemsize)
    rows_buf = _Buf(tokens * channels * kernel_size * np.dtype(np.float32).itemsize)
    try:
        qwen35_linear_attn_conv_prefill_f32_state_rows(
            hidden_buf.ptr,
            state_buf.ptr,
            rows_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            tokens,
            channels,
            kernel_size,
        )
        out = _from_device(out_buf, (tokens, channels), np.float32)
        rows = _from_device(rows_buf, (tokens, channels, kernel_size), np.float32)
        state_after = _from_device(state_buf, conv_state.shape, np.float32)
        return out, rows, state_after
    finally:
        for buf in (hidden_buf, state_buf, weight_buf, out_buf, rows_buf):
            buf.free()


def _run_packed_conv_state_rows(
    hidden: np.ndarray,
    conv_state_slots: np.ndarray,
    conv_weight: np.ndarray,
    cu_seqlens: np.ndarray,
    state_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens, channels = hidden.shape
    kernel_size = conv_state_slots.shape[2]
    hidden_buf = _to_device(hidden)
    state_buf = _to_device(conv_state_slots)
    weight_buf = _to_device(conv_weight)
    cu_buf = _to_device(np.ascontiguousarray(cu_seqlens, dtype=np.int32))
    state_indices_buf = _to_device(np.ascontiguousarray(state_indices, dtype=np.int64))
    out_buf = _Buf(tokens * channels * np.dtype(np.float32).itemsize)
    rows_buf = _Buf(tokens * channels * kernel_size * np.dtype(np.float32).itemsize)
    try:
        qwen35_linear_attn_conv_prefill_segments_f32_state_rows(
            hidden_buf.ptr,
            state_buf.ptr,
            rows_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            cu_buf.ptr,
            state_indices_buf.ptr,
            tokens,
            int(len(cu_seqlens) - 1),
            channels,
            kernel_size,
        )
        out = _from_device(out_buf, (tokens, channels), np.float32)
        rows = _from_device(rows_buf, (tokens, channels, kernel_size), np.float32)
        state_after = _from_device(state_buf, conv_state_slots.shape, np.float32)
        return out, rows, state_after
    finally:
        for buf in (hidden_buf, state_buf, weight_buf, cu_buf, state_indices_buf, out_buf, rows_buf):
            buf.free()


def _run_single_conv_mutating(
    hidden: np.ndarray,
    conv_state: np.ndarray,
    conv_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tokens, channels = hidden.shape
    kernel_size = conv_state.shape[1]
    hidden_buf = _to_device(hidden)
    state_buf = _to_device(conv_state)
    weight_buf = _to_device(conv_weight)
    out_buf = _Buf(tokens * channels * np.dtype(np.float32).itemsize)
    try:
        qwen35_linear_attn_conv_prefill_f32(
            hidden_buf.ptr,
            state_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            tokens,
            channels,
            kernel_size,
        )
        out = _from_device(out_buf, (tokens, channels), np.float32)
        state_after = _from_device(state_buf, conv_state.shape, np.float32)
        return out, state_after
    finally:
        for buf in (hidden_buf, state_buf, weight_buf, out_buf):
            buf.free()


def _run_packed_conv_mutating(
    hidden: np.ndarray,
    conv_state_slots: np.ndarray,
    conv_weight: np.ndarray,
    cu_seqlens: np.ndarray,
    state_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tokens, channels = hidden.shape
    kernel_size = conv_state_slots.shape[2]
    hidden_buf = _to_device(hidden)
    state_buf = _to_device(conv_state_slots)
    weight_buf = _to_device(conv_weight)
    cu_buf = _to_device(np.ascontiguousarray(cu_seqlens, dtype=np.int32))
    state_indices_buf = _to_device(np.ascontiguousarray(state_indices, dtype=np.int64))
    out_buf = _Buf(tokens * channels * np.dtype(np.float32).itemsize)
    try:
        qwen35_linear_attn_conv_prefill_segments_f32(
            hidden_buf.ptr,
            state_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            cu_buf.ptr,
            state_indices_buf.ptr,
            tokens,
            int(len(cu_seqlens) - 1),
            channels,
            kernel_size,
        )
        out = _from_device(out_buf, (tokens, channels), np.float32)
        state_after = _from_device(state_buf, conv_state_slots.shape, np.float32)
        return out, state_after
    finally:
        for buf in (hidden_buf, state_buf, weight_buf, cu_buf, state_indices_buf, out_buf):
            buf.free()


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_conv_prefill_segments_state_rows_match_per_segment_capture() -> None:
    rng = np.random.default_rng(1234)
    tokens = 7
    channels = 11
    kernel_size = 4
    hidden = rng.normal(0.0, 0.5, size=(tokens, channels)).astype(np.float32)
    conv_state_slots = rng.normal(0.0, 0.25, size=(2, channels, kernel_size)).astype(np.float32)
    conv_weight = rng.normal(0.0, 0.2, size=(channels, kernel_size)).astype(np.float32)
    cu_seqlens = np.asarray([0, 4, 7], dtype=np.int32)
    state_indices = np.asarray([0, 1], dtype=np.int64)

    packed_out, packed_rows, packed_state_after = _run_packed_conv_state_rows(
        hidden,
        conv_state_slots,
        conv_weight,
        cu_seqlens,
        state_indices,
    )

    expected_out = np.empty_like(packed_out)
    expected_rows = np.empty_like(packed_rows)
    for segment, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)):
        seg_out, seg_rows, seg_state_after = _run_single_conv_state_rows(
            hidden[int(start) : int(end)],
            conv_state_slots[segment],
            conv_weight,
        )
        expected_out[int(start) : int(end)] = seg_out
        expected_rows[int(start) : int(end)] = seg_rows
        np.testing.assert_array_equal(seg_state_after, conv_state_slots[segment])

    np.testing.assert_allclose(packed_out, expected_out, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(packed_rows, expected_rows, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(packed_state_after, conv_state_slots)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_conv_prefill_segments_mutating_c6_one_token_segments_match_independent_c1() -> None:
    rng = np.random.default_rng(5678)
    tokens = 6
    channels = 17
    kernel_size = 4
    hidden = rng.normal(0.0, 0.5, size=(tokens, channels)).astype(np.float32)
    conv_state_slots = rng.normal(0.0, 0.25, size=(6, channels, kernel_size)).astype(np.float32)
    conv_weight = rng.normal(0.0, 0.2, size=(channels, kernel_size)).astype(np.float32)
    cu_seqlens = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    state_indices = np.asarray([0, 2, 4, 1, 3, 5], dtype=np.int64)

    packed_out, packed_state_after = _run_packed_conv_mutating(
        hidden,
        conv_state_slots,
        conv_weight,
        cu_seqlens,
        state_indices,
    )

    expected_out = np.empty_like(packed_out)
    expected_state_after = conv_state_slots.copy()
    for segment, slot in enumerate(state_indices.tolist()):
        seg_out, seg_state_after = _run_single_conv_mutating(
            hidden[segment : segment + 1],
            conv_state_slots[int(slot)],
            conv_weight,
        )
        expected_out[segment : segment + 1] = seg_out
        expected_state_after[int(slot)] = seg_state_after

    np.testing.assert_allclose(packed_out, expected_out, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(packed_state_after, expected_state_after)
