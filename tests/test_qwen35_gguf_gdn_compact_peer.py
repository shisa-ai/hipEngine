"""Focused contract tests for compact peer-wave32 GDN prefill.

The candidate preserves the admitted peer-wave32 recurrence arithmetic while
materializing normalized Q/K once per K head instead of once per V head.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    build_qwen35_linear_attn_gdn,
    qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32,
    qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32,
    qwen35_gdn_prefill_rmsnorm_gate_bf16,
    qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16,
    qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16,
    register_qwen35_linear_attn_gdn_kernels,
)
from hipengine.kernels.registry import resolve

_SOURCE = Path(__file__).parents[1] / "hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _f32_to_bf16_u16(array: np.ndarray) -> np.ndarray:
    f32 = np.asarray(array, dtype=np.float32, order="C")
    u32 = f32.view(np.uint32).copy()
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16
    return rounded.astype(np.uint16).reshape(f32.shape)


def _upload(array: np.ndarray, buffers: list) -> object:
    host = np.ascontiguousarray(array)
    buffer = malloc(host.nbytes)
    copy_host_to_device(buffer, host_array_ptr(host), host.nbytes)
    buffers.append(buffer)
    return buffer


def _allocate(nbytes: int, buffers: list) -> object:
    buffer = malloc(nbytes)
    buffers.append(buffer)
    return buffer


def _download(buffer, shape: tuple[int, ...], dtype) -> np.ndarray:
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), buffer, host.nbytes)
    return host


def test_compact_peer_source_has_k_head_qk_abi() -> None:
    source = _SOURCE.read_text(encoding="utf-8")
    assert "qwen35_linear_attn_prefill_prepare_compact_peer_normalized_kernel" in source
    assert "qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_kernel" in source
    assert "token * num_k_heads + k_head" in source


def test_compact_peer_kernels_are_registry_resolved_without_replacing_default() -> None:
    register_qwen35_linear_attn_gdn_kernels()
    prepare = resolve(
        backend="hip_gfx1100",
        layer="linear_attn_prefill_prepare",
        quant="gguf_qwen35",
        variant="f32_compact_peer_normalized_bf16",
    )
    recurrent = resolve(
        backend="hip_gfx1100",
        layer="gdn_prefill_recurrent",
        quant="gguf_qwen35",
        variant="f32_compact_normalized_wave32_xor",
    )
    assert prepare is qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16
    assert recurrent is qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_compact_peer_matches_peer_wave32_output_and_state_bits() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_gdn(load=True)
    rng = np.random.default_rng(36027)
    tokens = 7
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 128
    head_v_dim = 128
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_width = 2 * key_dim + value_dim

    conv = rng.normal(0.0, 0.2, (tokens, conv_width)).astype(np.float32)
    alpha = _f32_to_bf16_u16(rng.normal(0.0, 0.3, (tokens, num_v_heads)))
    beta_lowp = _f32_to_bf16_u16(rng.normal(0.0, 0.3, (tokens, num_v_heads)))
    gate = _f32_to_bf16_u16(rng.normal(0.0, 0.2, (tokens, value_dim)))
    dt_bias = rng.normal(0.0, 0.1, (num_v_heads,)).astype(np.float32)
    a_log = rng.normal(0.0, 0.1, (num_v_heads,)).astype(np.float32)
    norm = rng.normal(1.0, 0.05, (head_v_dim,)).astype(np.float32)
    initial_state = rng.normal(
        0.0, 0.02, (num_v_heads, head_k_dim, head_v_dim)
    ).astype(np.float32)

    buffers: list = []
    try:
        conv_dev = _upload(conv, buffers)
        alpha_dev = _upload(alpha, buffers)
        beta_lowp_dev = _upload(beta_lowp, buffers)
        gate_dev = _upload(gate, buffers)
        dt_bias_dev = _upload(dt_bias, buffers)
        a_log_dev = _upload(a_log, buffers)
        norm_dev = _upload(norm, buffers)
        control_state = _upload(initial_state, buffers)
        candidate_state = _upload(initial_state, buffers)

        f32_bytes = np.dtype(np.float32).itemsize
        control_q = _allocate(tokens * num_v_heads * head_k_dim * f32_bytes, buffers)
        control_k = _allocate(tokens * num_v_heads * head_k_dim * f32_bytes, buffers)
        candidate_q = _allocate(tokens * num_k_heads * head_k_dim * f32_bytes, buffers)
        candidate_k = _allocate(tokens * num_k_heads * head_k_dim * f32_bytes, buffers)
        control_v = _allocate(tokens * value_dim * f32_bytes, buffers)
        candidate_v = _allocate(tokens * value_dim * f32_bytes, buffers)
        scalar_bytes = tokens * num_v_heads * f32_bytes
        control_beta = _allocate(scalar_bytes, buffers)
        control_decay = _allocate(scalar_bytes, buffers)
        candidate_beta = _allocate(scalar_bytes, buffers)
        candidate_decay = _allocate(scalar_bytes, buffers)
        recurrent_bytes = tokens * value_dim * f32_bytes
        control_recurrent = _allocate(recurrent_bytes, buffers)
        candidate_recurrent = _allocate(recurrent_bytes, buffers)
        out_bytes = tokens * value_dim * np.dtype(np.uint16).itemsize
        control_out = _allocate(out_bytes, buffers)
        candidate_out = _allocate(out_bytes, buffers)

        qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16(
            conv_dev.ptr,
            alpha_dev.ptr,
            beta_lowp_dev.ptr,
            dt_bias_dev.ptr,
            a_log_dev.ptr,
            control_q.ptr,
            control_k.ptr,
            control_v.ptr,
            control_beta.ptr,
            control_decay.ptr,
            tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
            runtime=runtime,
        )
        qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32(
            control_q.ptr,
            control_k.ptr,
            control_v.ptr,
            control_beta.ptr,
            control_decay.ptr,
            control_state.ptr,
            control_recurrent.ptr,
            tokens,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
            runtime=runtime,
        )
        qwen35_gdn_prefill_rmsnorm_gate_bf16(
            control_recurrent.ptr,
            gate_dev.ptr,
            norm_dev.ptr,
            control_out.ptr,
            1.0e-6,
            tokens,
            num_v_heads,
            head_v_dim,
            library=library,
            runtime=runtime,
        )

        qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16(
            conv_dev.ptr,
            alpha_dev.ptr,
            beta_lowp_dev.ptr,
            dt_bias_dev.ptr,
            a_log_dev.ptr,
            candidate_q.ptr,
            candidate_k.ptr,
            candidate_v.ptr,
            candidate_beta.ptr,
            candidate_decay.ptr,
            tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
            runtime=runtime,
        )
        qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32(
            candidate_q.ptr,
            candidate_k.ptr,
            candidate_v.ptr,
            candidate_beta.ptr,
            candidate_decay.ptr,
            candidate_state.ptr,
            candidate_recurrent.ptr,
            tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
            runtime=runtime,
        )
        qwen35_gdn_prefill_rmsnorm_gate_bf16(
            candidate_recurrent.ptr,
            gate_dev.ptr,
            norm_dev.ptr,
            candidate_out.ptr,
            1.0e-6,
            tokens,
            num_v_heads,
            head_v_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        out_shape = (tokens, num_v_heads, head_v_dim)
        state_shape = (num_v_heads, head_k_dim, head_v_dim)
        np.testing.assert_array_equal(
            _download(candidate_out, out_shape, np.uint16),
            _download(control_out, out_shape, np.uint16),
        )
        np.testing.assert_array_equal(
            _download(candidate_state, state_shape, np.float32).view(np.uint32),
            _download(control_state, state_shape, np.float32).view(np.uint32),
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer)
