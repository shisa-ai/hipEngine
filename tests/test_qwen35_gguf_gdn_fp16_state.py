"""RED: fp16-state GDN prefill->decode flow vs host fp16-state replay.

Production fp16-state route (HIPENGINE_GGUF_FP16_RECURRENT_STATE): the GDN
recurrent state is stored fp16 with fp32 accumulation.  This test runs the
device chain-prefill fp16-state kernel (writes fp16 leaf states) followed by
the segments-decode fp16-state kernel (reads/writes fp16 state) and compares
against a host fp16-state recurrence replay of the same token sequence.

The strict FP32-state path (default) is covered by
test_qwen35_linear_attn_decode_batch_indexed.py; this test only exercises the
fp16-state instantiations and is skipped without ROCm/HIP.
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
from hipengine.kernels.hip_gfx1100.linear_attn import (
    build_qwen35_linear_attn_gdn,
    qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16_fp16state,
    qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state,
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


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32).copy()
    bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (bits >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(
        np.float32
    )


def _silu(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.asarray(values / (np.float32(1.0) + np.exp(-values)), dtype=np.float32)


def _softplus(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        values > np.float32(20.0),
        values,
        np.log1p(np.exp(values)),
    ).astype(np.float32)


class _Buffers:
    def __init__(self) -> None:
        self._buffers = []

    def from_host(self, array: np.ndarray):
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes)
        self._buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes)
        return buffer

    def empty(self, nbytes: int):
        buffer = malloc(int(nbytes))
        self._buffers.append(buffer)
        return buffer

    def close(self) -> None:
        for buffer in reversed(self._buffers):
            free(buffer)
        self._buffers.clear()


def _from_device(buffer, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), buffer, out.nbytes)
    return out


def _host_fp16_state_recurrence(
    conv_out: np.ndarray,
    gate_bits: np.ndarray,
    a_bits: np.ndarray,
    b_bits: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    norm_weight: np.ndarray,
    state_f32: np.ndarray,
    *,
    eps: float,
    num_k_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """GDN recurrence with fp16 round-trip of the state each token, fp32 math.

    Returns (per-token outputs, final fp32 state).
    """
    tokens = conv_out.shape[0]
    num_v_heads, head_k_dim, head_v_dim = state_f32.shape
    key_dim = num_k_heads * head_k_dim
    gate = _bf16_bits_to_f32(gate_bits).reshape(tokens, num_v_heads, head_v_dim)
    a = _bf16_bits_to_f32(a_bits).reshape(tokens, num_v_heads)
    b = _bf16_bits_to_f32(b_bits).reshape(tokens, num_v_heads)
    value = conv_out[:, 2 * key_dim:].reshape(tokens, num_v_heads, head_v_dim)
    beta = np.asarray(
        np.float32(1.0) / (np.float32(1.0) + np.exp(-b)), dtype=np.float32
    )
    decay = np.asarray(
        np.exp(-np.exp(a_log[None, :]) * _softplus(a + dt_bias[None, :])),
        dtype=np.float32,
    )
    state = state_f32.copy()
    outputs = np.empty((tokens, num_v_heads * head_v_dim), dtype=np.float32)
    for t in range(tokens):
        state_rt = np.float16(state).astype(np.float32)
        for v_head in range(num_v_heads):
            k_head = v_head % num_k_heads
            q_raw = conv_out[t, k_head * head_k_dim:(k_head + 1) * head_k_dim]
            k_raw = conv_out[
                t, key_dim + k_head * head_k_dim:key_dim + (k_head + 1) * head_k_dim
            ]
            q_norm = max(float(np.sqrt(np.sum(q_raw * q_raw, dtype=np.float32))), 1.0e-6)
            k_norm = max(float(np.sqrt(np.sum(k_raw * k_raw, dtype=np.float32))), 1.0e-6)
            q_normed = np.asarray(
                q_raw * np.float32(1.0 / q_norm) * np.float32(1.0 / np.sqrt(head_k_dim)),
                dtype=np.float32,
            )
            k_normed = np.asarray(k_raw * np.float32(1.0 / k_norm), dtype=np.float32)
            sv = state_rt[v_head]
            kv_mem = k_normed @ (sv * decay[t, v_head])
            delta = (value[t, v_head] - kv_mem) * beta[t, v_head]
            new_state = sv * decay[t, v_head] + np.outer(k_normed, delta).astype(np.float32)
            state[v_head] = new_state.astype(np.float32)
            out_acc = (q_normed @ state[v_head]).astype(np.float32)
            inv_rms = np.float32(1.0) / np.sqrt(
                np.sum(out_acc * out_acc, dtype=np.float32) / np.float32(head_v_dim)
                + np.float32(eps)
            )
            outputs[t, v_head * head_v_dim:(v_head + 1) * head_v_dim] = (
                out_acc * inv_rms * norm_weight * _silu(gate[t, v_head])
            )
    return outputs.reshape(tokens, num_v_heads * head_v_dim), state


def test_fp16_state_chain_prefill_then_segments_decode_matches_host_replay() -> None:
    if not HIP_AVAILABLE:
        pytest.skip("ROCm/HIP not available")
    rng = np.random.default_rng(20260819)
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 64
    head_v_dim = 32
    key_dim = num_k_heads * head_k_dim
    qkv_width = 2 * key_dim + num_v_heads * head_v_dim
    prefill_tokens = 6
    decode_tokens = 1
    eps = 1.0e-6

    prefill_conv = rng.normal(0.0, 0.35, size=(prefill_tokens, qkv_width)).astype(np.float32)
    prefill_gate = _f32_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(prefill_tokens, num_v_heads * head_v_dim)).astype(np.float32)
    )
    prefill_a = _f32_to_bf16_bits(
        rng.normal(-0.1, 0.25, size=(prefill_tokens, num_v_heads)).astype(np.float32)
    )
    prefill_b = _f32_to_bf16_bits(
        rng.normal(0.0, 0.3, size=(prefill_tokens, num_v_heads)).astype(np.float32)
    )
    decode_conv = rng.normal(0.0, 0.35, size=(decode_tokens, qkv_width)).astype(np.float32)
    decode_gate = _f32_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(decode_tokens, num_v_heads * head_v_dim)).astype(np.float32)
    )
    decode_a = _f32_to_bf16_bits(
        rng.normal(-0.1, 0.25, size=(decode_tokens, num_v_heads)).astype(np.float32)
    )
    decode_b = _f32_to_bf16_bits(
        rng.normal(0.0, 0.3, size=(decode_tokens, num_v_heads)).astype(np.float32)
    )
    dt_bias = rng.normal(0.0, 0.2, size=(num_v_heads,)).astype(np.float32)
    a_log = rng.normal(-0.75, 0.2, size=(num_v_heads,)).astype(np.float32)
    norm_weight = rng.normal(1.0, 0.1, size=(head_v_dim,)).astype(np.float32)
    state_f32 = rng.normal(0.0, 0.1, size=(num_v_heads, head_k_dim, head_v_dim)).astype(np.float32)

    # Host fp16-state replay over the full sequence (prefill + decode).
    full_conv = np.concatenate([prefill_conv, decode_conv], axis=0)
    full_gate = np.concatenate([prefill_gate, decode_gate], axis=0)
    full_a = np.concatenate([prefill_a, decode_a], axis=0)
    full_b = np.concatenate([prefill_b, decode_b], axis=0)
    host_out, host_state = _host_fp16_state_recurrence(
        full_conv, full_gate, full_a, full_b, dt_bias, a_log, norm_weight,
        state_f32, eps=eps, num_k_heads=num_k_heads,
    )
    host_decode_out = host_out[prefill_tokens:]
    host_final_state = host_state

    state_elem = num_v_heads * head_k_dim * head_v_dim
    out_elem = num_v_heads * head_v_dim
    state_bytes = state_elem * np.dtype(np.float16).itemsize

    require_cached = False
    library = build_qwen35_linear_attn_gdn(load=True, require_cached=require_cached)
    bufs = _Buffers()
    try:
        prefill_conv_dev = bufs.from_host(prefill_conv)
        prefill_gate_dev = bufs.from_host(prefill_gate)
        prefill_a_dev = bufs.from_host(prefill_a)
        prefill_b_dev = bufs.from_host(prefill_b)
        decode_conv_dev = bufs.from_host(decode_conv)
        decode_gate_dev = bufs.from_host(decode_gate)
        decode_a_dev = bufs.from_host(decode_a)
        decode_b_dev = bufs.from_host(decode_b)
        dt_dev = bufs.from_host(dt_bias)
        al_dev = bufs.from_host(a_log)
        nw_dev = bufs.from_host(norm_weight)
        # fp16 state buffers
        leaf_dev = bufs.empty(prefill_tokens * state_elem * np.dtype(np.float16).itemsize)
        slot_dev = bufs.empty(state_elem * np.dtype(np.float16).itemsize)
        state_h = np.float16(state_f32)
        copy_host_to_device(slot_dev, host_array_ptr(state_h), state_h.nbytes)
        acc_dev = bufs.empty(prefill_tokens * out_elem * np.dtype(np.float32).itemsize)
        prefill_out_dev = bufs.empty(prefill_tokens * out_elem * np.dtype(np.float32).itemsize)
        decode_out_dev = bufs.empty(decode_tokens * out_elem * np.dtype(np.float32).itemsize)

        # 1) Chain prefill (fp16 state): writes per-token leaf states.
        qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16_fp16state(
            prefill_conv_dev.ptr,
            prefill_gate_dev.ptr,
            prefill_a_dev.ptr,
            prefill_b_dev.ptr,
            dt_dev.ptr,
            al_dev.ptr,
            nw_dev.ptr,
            slot_dev.ptr,
            leaf_dev.ptr,
            acc_dev.ptr,
            prefill_out_dev.ptr,
            eps,
            prefill_tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
        )
        # 2) The final prefill leaf state (leaf[prefill_tokens-1]) is the fp16
        #    decode slot; run decode directly on that leaf address.
        leaf_slot_dev = leaf_dev.ptr + (prefill_tokens - 1) * state_bytes
        # 3) Segments decode (fp16 state) reads leaf[prefill_tokens-1] directly.
        cu_seqlens = np.asarray([0, 1], dtype=np.int32)
        state_indices = np.asarray([0], dtype=np.int64)
        cu_dev = bufs.from_host(cu_seqlens)
        si_dev = bufs.from_host(state_indices)
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state(
            decode_conv_dev.ptr,
            decode_gate_dev.ptr,
            decode_a_dev.ptr,
            decode_b_dev.ptr,
            dt_dev.ptr,
            al_dev.ptr,
            nw_dev.ptr,
            leaf_slot_dev,
            decode_out_dev.ptr,
            cu_dev.ptr,
            si_dev.ptr,
            decode_tokens,
            decode_tokens,
            eps,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
        )
        ctypes.CDLL("libamdhip64.so").hipDeviceSynchronize()
        dev_decode_out = _from_device(
            decode_out_dev, (decode_tokens, out_elem), np.float32
        )
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import MemcpyKind

        _runtime = get_hip_runtime()
        dev_final_raw = np.empty((state_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_final_raw),
            leaf_slot_dev,
            dev_final_raw.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_final_state = dev_final_raw.view(np.float16).astype(np.float32).reshape(
            num_v_heads, head_k_dim, head_v_dim
        )
    finally:
        bufs.close()

    # The device reads fp16 state, so the reference state must be the fp16
    # rounding of the fp32 host state (both accumulate in fp32).
    np.testing.assert_allclose(
        dev_decode_out, host_decode_out, rtol=1.0e-3, atol=1.0e-3
    )
    # State within ~2 fp16 ULPs of the host fp32 state.
    np.testing.assert_allclose(
        dev_final_state, host_final_state, rtol=2.0e-3, atol=2.0e-3
    )


def _host_compact_wave32_xor_fp16_state(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    beta: np.ndarray,
    decay: np.ndarray,
    state_f32: np.ndarray,
    *,
    num_k_heads: int,
    state_storage_dtype: np.dtype = np.dtype(np.float16),
) -> tuple[np.ndarray, np.ndarray]:
    """Host replay of compact wave32-xor with configurable state storage."""

    tokens = query.shape[0] // num_k_heads
    num_v_heads, head_k_dim, head_v_dim = state_f32.shape
    state = np.asarray(state_f32, dtype=state_storage_dtype).astype(np.float32)
    outputs = np.empty((tokens * num_v_heads, head_v_dim), dtype=np.float32)
    for v_head in range(num_v_heads):
        k_head = v_head % num_k_heads
        sv = state[v_head].copy()  # (head_k_dim, head_v_dim) fp32
        for t in range(tokens):
            q = query[t * num_k_heads + k_head]  # (head_k_dim,)
            k = key[t * num_k_heads + k_head]  # (head_k_dim,)
            v = value[t * num_v_heads + v_head]  # (head_v_dim,)
            d = decay[t * num_v_heads + v_head]
            b = beta[t * num_v_heads + v_head]
            kv = (k[:, None] * sv).sum(axis=0)  # (head_v_dim,)
            delta = (v - np.float32(d) * kv) * np.float32(b)
            sv = np.float32(d) * sv + k[:, None] * delta[None, :]
            attn = (q[:, None] * sv).sum(axis=0)
            outputs[t * num_v_heads + v_head] = attn * np.float32(
                0.08838834764831845
            )
        state[v_head] = sv
    final_state = np.asarray(state, dtype=state_storage_dtype).astype(np.float32)
    return outputs, final_state


def test_fp16_state_compact_wave32_xor_prefill_writer_matches_host_replay() -> None:
    if not HIP_AVAILABLE:
        pytest.skip("ROCm/HIP not available")
    rng = np.random.default_rng(20260819)
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 128
    head_v_dim = 32
    tokens = 5
    eps = 1.0e-6  # unused by the recurrence; retained for parity with prepare

    query = rng.normal(0.0, 0.3, size=(tokens * num_k_heads, head_k_dim)).astype(np.float32)
    key = rng.normal(0.0, 0.3, size=(tokens * num_k_heads, head_k_dim)).astype(np.float32)
    value = rng.normal(0.0, 0.3, size=(tokens * num_v_heads, head_v_dim)).astype(np.float32)
    beta = rng.uniform(0.1, 1.0, size=(tokens * num_v_heads,)).astype(np.float32)
    decay = rng.uniform(0.5, 0.99, size=(tokens * num_v_heads,)).astype(np.float32)
    state_f32 = rng.normal(0.0, 0.1, size=(num_v_heads, head_k_dim, head_v_dim)).astype(np.float32)

    host_out, host_state = _host_compact_wave32_xor_fp16_state(
        query, key, value, beta, decay, state_f32, num_k_heads=num_k_heads
    )

    out_elem = num_v_heads * head_v_dim
    state_elem = num_v_heads * head_k_dim * head_v_dim
    state_bytes = state_elem * np.dtype(np.float16).itemsize

    library = build_qwen35_linear_attn_gdn(load=True)
    bufs = _Buffers()
    try:
        query_dev = bufs.from_host(query)
        key_dev = bufs.from_host(key)
        value_dev = bufs.from_host(value)
        beta_dev = bufs.from_host(beta)
        decay_dev = bufs.from_host(decay)
        state_dev = bufs.empty(state_bytes)
        state_h = np.float16(state_f32)
        copy_host_to_device(state_dev, host_array_ptr(state_h), state_h.nbytes)
        out_dev = bufs.empty(tokens * out_elem * np.dtype(np.float32).itemsize)

        from hipengine.kernels.hip_gfx1100.linear_attn import (
            qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_fp16state,
        )

        qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_fp16state(
            query_dev.ptr,
            key_dev.ptr,
            value_dev.ptr,
            beta_dev.ptr,
            decay_dev.ptr,
            state_dev.ptr,
            out_dev.ptr,
            tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
        )
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import MemcpyKind

        _runtime = get_hip_runtime()
        dev_out = np.empty((tokens * out_elem,), dtype=np.float32)
        _runtime.memcpy(
            host_array_ptr(dev_out),
            out_dev.ptr,
            dev_out.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_state_raw = np.empty((state_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_state_raw),
            state_dev.ptr,
            dev_state_raw.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_state = dev_state_raw.view(np.float16).astype(np.float32).reshape(
            num_v_heads, head_k_dim, head_v_dim
        )
    finally:
        bufs.close()

    dev_out = dev_out.reshape(tokens, num_v_heads, head_v_dim)
    host_out_r = host_out.reshape(tokens, num_v_heads, head_v_dim)
    np.testing.assert_allclose(dev_out, host_out_r, rtol=1.0e-3, atol=1.0e-3)
    np.testing.assert_allclose(dev_state, host_state, rtol=1.0e-3, atol=1.0e-3)


def _host_compact_wave32_xor_segments_fp16_state(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    beta: np.ndarray,
    decay: np.ndarray,
    state_slots_f32: np.ndarray,
    cu_seqlens: np.ndarray,
    state_indices: np.ndarray,
    *,
    num_k_heads: int,
    state_storage_dtype: np.dtype = np.dtype(np.float16),
) -> tuple[np.ndarray, np.ndarray]:
    """Independent per-segment compact-peer replay over indexed FP16 slots."""

    num_slots, num_v_heads, head_k_dim, head_v_dim = state_slots_f32.shape
    del num_slots, head_k_dim
    total_tokens = int(cu_seqlens[-1])
    outputs = np.empty(
        (total_tokens, num_v_heads, head_v_dim),
        dtype=np.float32,
    )
    final_slots = np.asarray(
        state_slots_f32,
        dtype=state_storage_dtype,
    ).astype(np.float32).copy()
    for segment, state_slot in enumerate(state_indices.tolist()):
        start = int(cu_seqlens[segment])
        end = int(cu_seqlens[segment + 1])
        segment_out, segment_state = _host_compact_wave32_xor_fp16_state(
            query[start * num_k_heads : end * num_k_heads],
            key[start * num_k_heads : end * num_k_heads],
            value[start * num_v_heads : end * num_v_heads],
            beta[start * num_v_heads : end * num_v_heads],
            decay[start * num_v_heads : end * num_v_heads],
            final_slots[int(state_slot)],
            num_k_heads=num_k_heads,
            state_storage_dtype=state_storage_dtype,
        )
        outputs[start:end] = segment_out.reshape(
            end - start,
            num_v_heads,
            head_v_dim,
        )
        final_slots[int(state_slot)] = segment_state
    return outputs, final_slots


@pytest.mark.parametrize(
    ("state_variant", "state_dtype"),
    (("f32", np.dtype(np.float32)), ("fp16state", np.dtype(np.float16))),
)
def test_compact_wave32_xor_segmented_prefill_matches_host_replay(
    state_variant: str,
    state_dtype: np.dtype,
) -> None:
    """Packed c>N prefill keeps each indexed state register-resident."""

    if not HIP_AVAILABLE:
        pytest.skip("ROCm/HIP not available")
    rng = np.random.default_rng(20260822)
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 128
    head_v_dim = 32
    total_tokens = 7
    cu_seqlens = np.asarray([0, 3, 7], dtype=np.int32)
    state_indices = np.asarray([2, 0], dtype=np.int64)
    segments = len(state_indices)
    num_slots = 3

    query = rng.normal(
        0.0, 0.3, size=(total_tokens * num_k_heads, head_k_dim)
    ).astype(np.float32)
    key = rng.normal(
        0.0, 0.3, size=(total_tokens * num_k_heads, head_k_dim)
    ).astype(np.float32)
    value = rng.normal(
        0.0, 0.3, size=(total_tokens * num_v_heads, head_v_dim)
    ).astype(np.float32)
    beta = rng.uniform(0.1, 1.0, size=(total_tokens * num_v_heads,)).astype(
        np.float32
    )
    decay = rng.uniform(0.5, 0.99, size=(total_tokens * num_v_heads,)).astype(
        np.float32
    )
    state_f32 = rng.normal(
        0.0,
        0.1,
        size=(num_slots, num_v_heads, head_k_dim, head_v_dim),
    ).astype(np.float32)
    host_out, host_state = _host_compact_wave32_xor_segments_fp16_state(
        query,
        key,
        value,
        beta,
        decay,
        state_f32,
        cu_seqlens,
        state_indices,
        num_k_heads=num_k_heads,
        state_storage_dtype=state_dtype,
    )

    state_elem = num_slots * num_v_heads * head_k_dim * head_v_dim
    out_elem = total_tokens * num_v_heads * head_v_dim
    library = build_qwen35_linear_attn_gdn(load=True)
    bufs = _Buffers()
    try:
        query_dev = bufs.from_host(query)
        key_dev = bufs.from_host(key)
        value_dev = bufs.from_host(value)
        beta_dev = bufs.from_host(beta)
        decay_dev = bufs.from_host(decay)
        state_h = np.asarray(state_f32, dtype=state_dtype)
        state_dev = bufs.empty(state_h.nbytes)
        copy_host_to_device(state_dev, host_array_ptr(state_h), state_h.nbytes)
        out_dev = bufs.empty(out_elem * np.dtype(np.float32).itemsize)
        cu_dev = bufs.from_host(cu_seqlens)
        indices_dev = bufs.from_host(state_indices)

        from hipengine.kernels.hip_gfx1100.linear_attn import (
            qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_f32,
            qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_fp16state,
        )

        kernel = (
            qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_f32
            if state_variant == "f32"
            else qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_fp16state
        )
        kernel(
            query_dev.ptr,
            key_dev.ptr,
            value_dev.ptr,
            beta_dev.ptr,
            decay_dev.ptr,
            state_dev.ptr,
            out_dev.ptr,
            cu_dev.ptr,
            indices_dev.ptr,
            total_tokens,
            segments,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
        )
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import MemcpyKind

        runtime = get_hip_runtime()
        dev_out = np.empty((out_elem,), dtype=np.float32)
        runtime.memcpy(
            host_array_ptr(dev_out),
            out_dev.ptr,
            dev_out.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_state_raw = np.empty((state_elem,), dtype=state_dtype)
        runtime.memcpy(
            host_array_ptr(dev_state_raw),
            state_dev.ptr,
            dev_state_raw.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
    finally:
        bufs.close()

    dev_out = dev_out.reshape(total_tokens, num_v_heads, head_v_dim)
    dev_state = dev_state_raw.astype(np.float32).reshape(
        num_slots,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    np.testing.assert_allclose(dev_out, host_out, rtol=1.0e-3, atol=1.0e-3)
    np.testing.assert_allclose(dev_state, host_state, rtol=1.0e-3, atol=1.0e-3)


def test_fp16_state_fused_f32_bf16_out_decode_matches_host_replay() -> None:
    """The Qwen3.8 Q4_K_S topline decode kernel: fused gate + BF16 cast.

    The dense ``ssm_out`` projection quantizes as ``gguf_q5_k_t16_v1``, which
    resolves the fused ``gdn_recurrent_rmsnorm_gate+cast`` owner on the decode
    path.  This test exercises the fp16-state instantiation (fp32 accumulate,
    fp16 round-trip state) and checks the FP32 output, BF16 output, and final
    state against a host fp16-state replay.
    """
    if not HIP_AVAILABLE:
        pytest.skip("ROCm/HIP not available")
    rng = np.random.default_rng(20260819)
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 64
    head_v_dim = 32
    key_dim = num_k_heads * head_k_dim
    qkv_width = 2 * key_dim + num_v_heads * head_v_dim
    decode_tokens = 1  # the decode gate kernel is single-token per launch
    eps = 1.0e-6

    conv = rng.normal(0.0, 0.35, size=(decode_tokens, qkv_width)).astype(np.float32)
    gate = _f32_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(decode_tokens, num_v_heads * head_v_dim)).astype(np.float32)
    )
    a = _f32_to_bf16_bits(
        rng.normal(-0.1, 0.25, size=(decode_tokens, num_v_heads)).astype(np.float32)
    )
    b = _f32_to_bf16_bits(
        rng.normal(0.0, 0.3, size=(decode_tokens, num_v_heads)).astype(np.float32)
    )
    dt_bias = rng.normal(0.0, 0.2, size=(num_v_heads,)).astype(np.float32)
    a_log = rng.normal(-0.75, 0.2, size=(num_v_heads,)).astype(np.float32)
    norm_weight = rng.normal(1.0, 0.1, size=(head_v_dim,)).astype(np.float32)
    state_f32 = rng.normal(0.0, 0.1, size=(num_v_heads, head_k_dim, head_v_dim)).astype(np.float32)

    host_out, host_state = _host_fp16_state_recurrence(
        conv, gate, a, b, dt_bias, a_log, norm_weight, state_f32,
        eps=eps, num_k_heads=num_k_heads,
    )

    out_elem = num_v_heads * head_v_dim
    state_elem = num_v_heads * head_k_dim * head_v_dim
    state_bytes = state_elem * np.dtype(np.float16).itemsize

    library = build_qwen35_linear_attn_gdn(load=True)
    bufs = _Buffers()
    try:
        conv_dev = bufs.from_host(conv)
        gate_dev = bufs.from_host(gate)
        a_dev = bufs.from_host(a)
        b_dev = bufs.from_host(b)
        dt_dev = bufs.from_host(dt_bias)
        al_dev = bufs.from_host(a_log)
        nw_dev = bufs.from_host(norm_weight)
        state_dev = bufs.empty(state_bytes)
        state_h = np.float16(state_f32)
        copy_host_to_device(state_dev, host_array_ptr(state_h), state_h.nbytes)
        out_dev = bufs.empty(decode_tokens * out_elem * np.dtype(np.float32).itemsize)
        out_bf16_dev = bufs.empty(decode_tokens * out_elem * np.dtype(np.uint16).itemsize)

        from hipengine.kernels.hip_gfx1100.linear_attn import (
            qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out_fp16state,
        )

        qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out_fp16state(
            conv_dev.ptr,
            gate_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_dev.ptr,
            al_dev.ptr,
            nw_dev.ptr,
            state_dev.ptr,
            out_dev.ptr,
            out_bf16_dev.ptr,
            eps,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
        )
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import MemcpyKind

        _runtime = get_hip_runtime()
        dev_out = np.empty((decode_tokens * out_elem,), dtype=np.float32)
        _runtime.memcpy(
            host_array_ptr(dev_out),
            out_dev.ptr,
            dev_out.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_out_bf16 = np.empty((decode_tokens * out_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_out_bf16),
            out_bf16_dev.ptr,
            dev_out_bf16.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_state_raw = np.empty((state_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_state_raw),
            state_dev.ptr,
            dev_state_raw.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_state = dev_state_raw.view(np.float16).astype(np.float32).reshape(
            num_v_heads, head_k_dim, head_v_dim
        )
    finally:
        bufs.close()

    np.testing.assert_allclose(
        dev_out.reshape(decode_tokens, out_elem),
        host_out,
        rtol=1.0e-3,
        atol=1.0e-3,
    )
    expected_bf16 = _f32_to_bf16_bits(host_out).reshape(-1)
    np.testing.assert_array_equal(dev_out_bf16, expected_bf16)
    np.testing.assert_allclose(dev_state, host_state, rtol=2.0e-3, atol=2.0e-3)



def _host_fp16_state_recurrence_rows(
    conv_out: np.ndarray,
    gate_bits: np.ndarray,
    a_bits: np.ndarray,
    b_bits: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    norm_weight: np.ndarray,
    state_f32: np.ndarray,
    *,
    eps: float,
    num_k_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """decode-order fp16-state recurrence with per-token row capture.

    Mirrors ``_host_fp16_state_recurrence`` but additionally captures the
    fp16-rounded per-token new-state rows written by the decode-order
    state-rows writer.  Returns (per-token outputs, rows_f32).
    """
    tokens = conv_out.shape[0]
    num_v_heads, head_k_dim, head_v_dim = state_f32.shape
    key_dim = num_k_heads * head_k_dim
    gate = _bf16_bits_to_f32(gate_bits).reshape(tokens, num_v_heads, head_v_dim)
    a = _bf16_bits_to_f32(a_bits).reshape(tokens, num_v_heads)
    b = _bf16_bits_to_f32(b_bits).reshape(tokens, num_v_heads)
    value = conv_out[:, 2 * key_dim:].reshape(tokens, num_v_heads, head_v_dim)
    beta = np.asarray(
        np.float32(1.0) / (np.float32(1.0) + np.exp(-b)), dtype=np.float32
    )
    decay = np.asarray(
        np.exp(-np.exp(a_log[None, :]) * _softplus(a + dt_bias[None, :])),
        dtype=np.float32,
    )
    state = np.float16(state_f32).astype(np.float32)
    outputs = np.empty((tokens, num_v_heads * head_v_dim), dtype=np.float32)
    rows = np.empty((tokens, num_v_heads, head_k_dim, head_v_dim), dtype=np.float32)
    for t in range(tokens):
        sv = state.copy()
        for v_head in range(num_v_heads):
            k_head = v_head % num_k_heads
            q_raw = conv_out[t, k_head * head_k_dim:(k_head + 1) * head_k_dim]
            k_raw = conv_out[
                t, key_dim + k_head * head_k_dim:key_dim + (k_head + 1) * head_k_dim
            ]
            q_norm = max(float(np.sqrt(np.sum(q_raw * q_raw, dtype=np.float32))), 1.0e-6)
            k_norm = max(float(np.sqrt(np.sum(k_raw * k_raw, dtype=np.float32))), 1.0e-6)
            q_normed = np.asarray(
                q_raw * np.float32(1.0 / q_norm) * np.float32(1.0 / np.sqrt(head_k_dim)),
                dtype=np.float32,
            )
            k_normed = np.asarray(k_raw * np.float32(1.0 / k_norm), dtype=np.float32)
            kv_mem = k_normed @ (sv[v_head] * decay[t, v_head])
            delta = (value[t, v_head] - kv_mem) * beta[t, v_head]
            new_state = sv[v_head] * decay[t, v_head] + np.outer(
                k_normed, delta
            ).astype(np.float32)
            state[v_head] = np.float16(new_state).astype(np.float32)
            rows[t, v_head] = np.float16(new_state).astype(np.float32)
            out_acc = (q_normed @ new_state).astype(np.float32)
            inv_rms = np.float32(1.0) / np.sqrt(
                np.sum(out_acc * out_acc, dtype=np.float32) / np.float32(head_v_dim)
                + np.float32(eps)
            )
            outputs[t, v_head * head_v_dim:(v_head + 1) * head_v_dim] = (
                out_acc * inv_rms * norm_weight * _silu(gate[t, v_head])
            )
    return outputs.reshape(tokens, num_v_heads * head_v_dim), rows


def test_fp16_state_decode_order_state_rows_no_copy_matches_host_replay() -> None:
    """fp16-state decode-order state-rows prefill writer vs host replay.

    The production packed-prefill route's decode-order row-state capture is
    fp16-incompatible in the strict gate; this fp16-state no-copy writer stores
    the initial and per-token captured state as fp16 (fp32 accumulate).
    """
    if not HIP_AVAILABLE:
        pytest.skip("ROCm/HIP not available")
    rng = np.random.default_rng(20260819)
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 64
    head_v_dim = 32
    key_dim = num_k_heads * head_k_dim
    qkv_width = 2 * key_dim + num_v_heads * head_v_dim
    tokens = 6
    eps = 1.0e-6

    conv = rng.normal(0.0, 0.35, size=(tokens, qkv_width)).astype(np.float32)
    gate = _f32_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(tokens, num_v_heads * head_v_dim)).astype(np.float32)
    )
    a = _f32_to_bf16_bits(
        rng.normal(-0.1, 0.25, size=(tokens, num_v_heads)).astype(np.float32)
    )
    b = _f32_to_bf16_bits(
        rng.normal(0.0, 0.3, size=(tokens, num_v_heads)).astype(np.float32)
    )
    dt_bias = rng.normal(0.0, 0.2, size=(num_v_heads,)).astype(np.float32)
    a_log = rng.normal(-0.75, 0.2, size=(num_v_heads,)).astype(np.float32)
    norm_weight = rng.normal(1.0, 0.1, size=(head_v_dim,)).astype(np.float32)
    state_f32 = rng.normal(0.0, 0.1, size=(num_v_heads, head_k_dim, head_v_dim)).astype(np.float32)

    host_out, host_rows = _host_fp16_state_recurrence_rows(
        conv, gate, a, b, dt_bias, a_log, norm_weight, state_f32,
        eps=eps, num_k_heads=num_k_heads,
    )

    state_elem = num_v_heads * head_k_dim * head_v_dim
    out_elem = num_v_heads * head_v_dim
    state_bytes = state_elem * np.dtype(np.float16).itemsize

    library = build_qwen35_linear_attn_gdn(load=True)
    bufs = _Buffers()
    try:
        conv_dev = bufs.from_host(conv)
        gate_dev = bufs.from_host(gate)
        a_dev = bufs.from_host(a)
        b_dev = bufs.from_host(b)
        dt_dev = bufs.from_host(dt_bias)
        al_dev = bufs.from_host(a_log)
        nw_dev = bufs.from_host(norm_weight)
        init_dev = bufs.empty(state_bytes)
        state_h = np.float16(state_f32)
        copy_host_to_device(init_dev, host_array_ptr(state_h), state_h.nbytes)
        rows_dev = bufs.empty(tokens * state_elem * np.dtype(np.float16).itemsize)
        out_dev = bufs.empty(tokens * out_elem * np.dtype(np.uint16).itemsize)

        from hipengine.kernels.hip_gfx1100.linear_attn import (
            qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy_fp16state,
        )

        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy_fp16state(
            conv_dev.ptr,
            gate_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_dev.ptr,
            al_dev.ptr,
            nw_dev.ptr,
            init_dev.ptr,
            rows_dev.ptr,
            out_dev.ptr,
            eps,
            tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
        )
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import MemcpyKind

        _runtime = get_hip_runtime()
        dev_out = np.empty((tokens * out_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_out),
            out_dev.ptr,
            dev_out.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_rows_raw = np.empty((tokens * state_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_rows_raw),
            rows_dev.ptr,
            dev_rows_raw.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
    finally:
        bufs.close()

    dev_out_f32 = _bf16_bits_to_f32(dev_out).reshape(tokens, num_v_heads * head_v_dim)
    dev_rows = dev_rows_raw.view(np.float16).astype(np.float32).reshape(
        tokens, num_v_heads, head_k_dim, head_v_dim
    )
    np.testing.assert_allclose(
        dev_out_f32, host_out, rtol=5.0e-2, atol=5.0e-3
    )
    np.testing.assert_allclose(dev_rows, host_rows, rtol=2.0e-3, atol=2.0e-3)


def test_fp16_state_decode_order_segments_state_rows_no_copy_matches_host_replay() -> None:
    """fp16-state segment-aware decode-order state-rows writer vs host replay.

    Packed rows with per-segment initial-state slots; each segment's first
    token reads its own fp16 initial slot and every token captures an fp16
    row-state.  The initial slots must not be mutated (no-copy).
    """
    if not HIP_AVAILABLE:
        pytest.skip("ROCm/HIP not available")
    rng = np.random.default_rng(20260819)
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 64
    head_v_dim = 32
    key_dim = num_k_heads * head_k_dim
    qkv_width = 2 * key_dim + num_v_heads * head_v_dim
    tokens = 7
    eps = 1.0e-6
    cu_seqlens = np.asarray([0, 4, 7], dtype=np.int32)
    state_indices = np.asarray([0, 1], dtype=np.int64)
    num_slots = 2

    conv = rng.normal(0.0, 0.35, size=(tokens, qkv_width)).astype(np.float32)
    gate = _f32_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(tokens, num_v_heads * head_v_dim)).astype(np.float32)
    )
    a = _f32_to_bf16_bits(
        rng.normal(-0.1, 0.25, size=(tokens, num_v_heads)).astype(np.float32)
    )
    b = _f32_to_bf16_bits(
        rng.normal(0.0, 0.3, size=(tokens, num_v_heads)).astype(np.float32)
    )
    dt_bias = rng.normal(0.0, 0.2, size=(num_v_heads,)).astype(np.float32)
    a_log = rng.normal(-0.75, 0.2, size=(num_v_heads,)).astype(np.float32)
    norm_weight = rng.normal(1.0, 0.1, size=(head_v_dim,)).astype(np.float32)
    init_slots_f32 = rng.normal(
        0.0, 0.1, size=(num_slots, num_v_heads, head_k_dim, head_v_dim)
    ).astype(np.float32)

    # Host per-segment replay stitched into global token/row layout.
    expected_out = np.empty((tokens, num_v_heads * head_v_dim), dtype=np.float32)
    expected_rows = np.empty(
        (tokens, num_v_heads, head_k_dim, head_v_dim), dtype=np.float32
    )
    for segment, (start, end) in enumerate(
        zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)
    ):
        seg_out, seg_rows = _host_fp16_state_recurrence_rows(
            np.ascontiguousarray(conv[int(start):int(end)]),
            np.ascontiguousarray(gate[int(start):int(end)]),
            np.ascontiguousarray(a[int(start):int(end)]),
            np.ascontiguousarray(b[int(start):int(end)]),
            dt_bias,
            a_log,
            norm_weight,
            init_slots_f32[segment],
            eps=eps,
            num_k_heads=num_k_heads,
        )
        expected_out[int(start):int(end)] = seg_out
        expected_rows[int(start):int(end)] = seg_rows

    state_elem = num_v_heads * head_k_dim * head_v_dim
    out_elem = num_v_heads * head_v_dim
    state_bytes = num_slots * state_elem * np.dtype(np.float16).itemsize

    library = build_qwen35_linear_attn_gdn(load=True)
    bufs = _Buffers()
    try:
        conv_dev = bufs.from_host(conv)
        gate_dev = bufs.from_host(gate)
        a_dev = bufs.from_host(a)
        b_dev = bufs.from_host(b)
        dt_dev = bufs.from_host(dt_bias)
        al_dev = bufs.from_host(a_log)
        nw_dev = bufs.from_host(norm_weight)
        init_dev = bufs.empty(state_bytes)
        state_h = np.float16(init_slots_f32)
        copy_host_to_device(init_dev, host_array_ptr(state_h), state_h.nbytes)
        rows_dev = bufs.empty(tokens * state_elem * np.dtype(np.float16).itemsize)
        out_dev = bufs.empty(tokens * out_elem * np.dtype(np.uint16).itemsize)
        cu_dev = bufs.from_host(cu_seqlens)
        si_dev = bufs.from_host(state_indices)

        from hipengine.kernels.hip_gfx1100.linear_attn import (
            qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_fp16state,
        )

        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_fp16state(
            conv_dev.ptr,
            gate_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_dev.ptr,
            al_dev.ptr,
            nw_dev.ptr,
            init_dev.ptr,
            rows_dev.ptr,
            out_dev.ptr,
            cu_dev.ptr,
            si_dev.ptr,
            eps,
            tokens,
            int(len(cu_seqlens) - 1),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
        )
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import MemcpyKind

        _runtime = get_hip_runtime()
        dev_out = np.empty((tokens * out_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_out),
            out_dev.ptr,
            dev_out.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_rows_raw = np.empty((tokens * state_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_rows_raw),
            rows_dev.ptr,
            dev_rows_raw.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_init_raw = np.empty((num_slots * state_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_init_raw),
            init_dev.ptr,
            dev_init_raw.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
    finally:
        bufs.close()

    dev_out_f32 = _bf16_bits_to_f32(dev_out).reshape(tokens, num_v_heads * head_v_dim)
    dev_rows = dev_rows_raw.view(np.float16).astype(np.float32).reshape(
        tokens, num_v_heads, head_k_dim, head_v_dim
    )
    dev_init = dev_init_raw.view(np.float16).astype(np.float32).reshape(
        num_slots, num_v_heads, head_k_dim, head_v_dim
    )
    np.testing.assert_allclose(
        dev_out_f32, expected_out, rtol=5.0e-2, atol=5.0e-3
    )
    np.testing.assert_allclose(dev_rows, expected_rows, rtol=2.0e-3, atol=2.0e-3)
    # no-copy: the fp16 initial slots must be untouched.
    np.testing.assert_array_equal(
        dev_init, np.float16(init_slots_f32).astype(np.float32)
    )


def test_fp16_state_decode_order_segments_inplace_matches_host_replay() -> None:
    """fp16-state in-place segmented decode-order prefill writer vs host replay.

    This is the packed AR multi-slot writer: each segment owns a per-slot state
    (``state_indices``) and the kernel mutates that slot's fp16 state in place
    across the segment's tokens.  The host replays the same fp16 round-trip
    recurrence per slot and compares per-token outputs plus the final mutated
    per-slot state.
    """
    if not HIP_AVAILABLE:
        pytest.skip("ROCm/HIP not available")
    rng = np.random.default_rng(20260820)
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 64
    head_v_dim = 32
    key_dim = num_k_heads * head_k_dim
    qkv_width = 2 * key_dim + num_v_heads * head_v_dim
    tokens = 7
    eps = 1.0e-6
    cu_seqlens = np.asarray([0, 4, 7], dtype=np.int32)
    state_indices = np.asarray([0, 1], dtype=np.int64)
    num_slots = 2

    conv = rng.normal(0.0, 0.35, size=(tokens, qkv_width)).astype(np.float32)
    gate = _f32_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(tokens, num_v_heads * head_v_dim)).astype(np.float32)
    )
    a = _f32_to_bf16_bits(
        rng.normal(-0.1, 0.25, size=(tokens, num_v_heads)).astype(np.float32)
    )
    b = _f32_to_bf16_bits(
        rng.normal(0.0, 0.3, size=(tokens, num_v_heads)).astype(np.float32)
    )
    dt_bias = rng.normal(0.0, 0.2, size=(num_v_heads,)).astype(np.float32)
    a_log = rng.normal(-0.75, 0.2, size=(num_v_heads,)).astype(np.float32)
    norm_weight = rng.normal(1.0, 0.1, size=(head_v_dim,)).astype(np.float32)
    init_slots_f32 = rng.normal(
        0.0, 0.1, size=(num_slots, num_v_heads, head_k_dim, head_v_dim)
    ).astype(np.float32)

    # Host per-segment replay mutating each slot's state in place.
    expected_out = np.empty((tokens, num_v_heads * head_v_dim), dtype=np.float32)
    final_slots_f32 = np.empty_like(init_slots_f32)
    for segment, (start, end) in enumerate(
        zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)
    ):
        slot = int(state_indices[segment])
        seg_init = np.float16(init_slots_f32[slot]).astype(np.float32)
        seg_out, seg_final = _host_fp16_state_recurrence(
            np.ascontiguousarray(conv[int(start):int(end)]),
            np.ascontiguousarray(gate[int(start):int(end)]),
            np.ascontiguousarray(a[int(start):int(end)]),
            np.ascontiguousarray(b[int(start):int(end)]),
            dt_bias,
            a_log,
            norm_weight,
            seg_init,
            eps=eps,
            num_k_heads=num_k_heads,
        )
        expected_out[int(start):int(end)] = seg_out
        final_slots_f32[slot] = seg_final

    state_elem = num_v_heads * head_k_dim * head_v_dim
    out_elem = num_v_heads * head_v_dim
    state_bytes = num_slots * state_elem * np.dtype(np.float16).itemsize

    library = build_qwen35_linear_attn_gdn(load=True)
    bufs = _Buffers()
    try:
        conv_dev = bufs.from_host(conv)
        gate_dev = bufs.from_host(gate)
        a_dev = bufs.from_host(a)
        b_dev = bufs.from_host(b)
        dt_dev = bufs.from_host(dt_bias)
        al_dev = bufs.from_host(a_log)
        nw_dev = bufs.from_host(norm_weight)
        state_dev = bufs.empty(state_bytes)
        state_h = np.float16(init_slots_f32)
        copy_host_to_device(state_dev, host_array_ptr(state_h), state_h.nbytes)
        out_dev = bufs.empty(tokens * out_elem * np.dtype(np.uint16).itemsize)
        cu_dev = bufs.from_host(cu_seqlens)
        si_dev = bufs.from_host(state_indices)

        from hipengine.kernels.hip_gfx1100.linear_attn import (
            qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_fp16state,
        )

        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_fp16state(
            conv_dev.ptr,
            gate_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_dev.ptr,
            al_dev.ptr,
            nw_dev.ptr,
            state_dev.ptr,
            out_dev.ptr,
            cu_dev.ptr,
            si_dev.ptr,
            eps,
            tokens,
            int(len(cu_seqlens) - 1),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
        )
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import MemcpyKind

        _runtime = get_hip_runtime()
        dev_out = np.empty((tokens * out_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_out),
            out_dev.ptr,
            dev_out.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_state_raw = np.empty((num_slots * state_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_state_raw),
            state_dev.ptr,
            dev_state_raw.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
    finally:
        bufs.close()

    dev_out_f32 = _bf16_bits_to_f32(dev_out).reshape(tokens, num_v_heads * head_v_dim)
    dev_state = dev_state_raw.view(np.float16).astype(np.float32).reshape(
        num_slots, num_v_heads, head_k_dim, head_v_dim
    )
    np.testing.assert_allclose(
        dev_out_f32, expected_out, rtol=5.0e-2, atol=5.0e-3
    )
    np.testing.assert_allclose(
        dev_state, final_slots_f32, rtol=2.0e-3, atol=2.0e-3
    )


@pytest.mark.parametrize("indexed_variant", ("plain", "shared_statecache24"))
def test_fp16_state_indexed_singleton_matches_host_replay(
    indexed_variant: str,
) -> None:
    """fp16-state indexed one-token-per-row decode kernel vs host replay.

    The packed AR decode advances one token per active row; each row owns a
    per-slot state (``state_indices``) that the kernel reads and mutates in
    place with fp16 storage.  The host replays the same fp16 round-trip
    recurrence per row.
    """
    if not HIP_AVAILABLE:
        pytest.skip("ROCm/HIP not available")
    rng = np.random.default_rng(20260821)
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 64
    head_v_dim = 32
    key_dim = num_k_heads * head_k_dim
    qkv_width = 2 * key_dim + num_v_heads * head_v_dim
    if indexed_variant == "plain":
        rows = 4
        state_indices = np.asarray([4, 1, 5, 0], dtype=np.int64)
        num_slots = 6
    else:
        # rows=8 is the first shape that executes the gfx1151
        # shared-statecache24 body rather than delegating to the plain wrapper.
        rows = 8
        state_indices = np.asarray([8, 1, 9, 0, 7, 2, 6, 3], dtype=np.int64)
        num_slots = 10
    eps = 1.0e-6

    conv = rng.normal(0.0, 0.35, size=(rows, qkv_width)).astype(np.float32)
    gate = _f32_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(rows, num_v_heads * head_v_dim)).astype(np.float32)
    )
    a = _f32_to_bf16_bits(
        rng.normal(-0.1, 0.25, size=(rows, num_v_heads)).astype(np.float32)
    )
    b = _f32_to_bf16_bits(
        rng.normal(0.0, 0.3, size=(rows, num_v_heads)).astype(np.float32)
    )
    dt_bias = rng.normal(0.0, 0.2, size=(num_v_heads,)).astype(np.float32)
    a_log = rng.normal(-0.75, 0.2, size=(num_v_heads,)).astype(np.float32)
    norm_weight = rng.normal(1.0, 0.1, size=(head_v_dim,)).astype(np.float32)
    init_slots_f32 = rng.normal(
        0.0, 0.1, size=(num_slots, num_v_heads, head_k_dim, head_v_dim)
    ).astype(np.float32)

    expected_out = np.empty((rows, num_v_heads * head_v_dim), dtype=np.float32)
    # Untouched slots keep their fp16-rounded initial state on device.
    final_slots_f32 = np.float16(init_slots_f32).astype(np.float32).copy()
    for row, slot in enumerate(state_indices):
        slot = int(slot)
        seg_init = np.float16(init_slots_f32[slot]).astype(np.float32)
        seg_out, seg_final = _host_fp16_state_recurrence(
            conv[row : row + 1],
            gate[row : row + 1],
            a[row : row + 1],
            b[row : row + 1],
            dt_bias,
            a_log,
            norm_weight,
            seg_init,
            eps=eps,
            num_k_heads=num_k_heads,
        )
        expected_out[row] = seg_out[0]
        final_slots_f32[slot] = seg_final

    state_elem = num_v_heads * head_k_dim * head_v_dim
    out_elem = num_v_heads * head_v_dim
    state_bytes = num_slots * state_elem * np.dtype(np.float16).itemsize

    library = build_qwen35_linear_attn_gdn(load=True)
    bufs = _Buffers()
    try:
        conv_dev = bufs.from_host(conv)
        gate_dev = bufs.from_host(gate)
        a_dev = bufs.from_host(a)
        b_dev = bufs.from_host(b)
        dt_dev = bufs.from_host(dt_bias)
        al_dev = bufs.from_host(a_log)
        nw_dev = bufs.from_host(norm_weight)
        state_dev = bufs.empty(state_bytes)
        state_h = np.float16(init_slots_f32)
        copy_host_to_device(state_dev, host_array_ptr(state_h), state_h.nbytes)
        out_dev = bufs.empty(rows * out_elem * np.dtype(np.float32).itemsize)
        si_dev = bufs.from_host(state_indices)

        from hipengine.kernels.hip_gfx1100.linear_attn import (
            qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state,
            qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state,
        )

        indexed_kernel = (
            qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state
            if indexed_variant == "plain"
            else qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state
        )
        indexed_kernel(
            conv_dev.ptr,
            gate_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_dev.ptr,
            al_dev.ptr,
            nw_dev.ptr,
            state_dev.ptr,
            out_dev.ptr,
            si_dev.ptr,
            rows,
            eps,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
        )
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import MemcpyKind

        _runtime = get_hip_runtime()
        dev_out = np.empty((rows * out_elem,), dtype=np.float32)
        _runtime.memcpy(
            host_array_ptr(dev_out),
            out_dev.ptr,
            dev_out.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        dev_state_raw = np.empty((num_slots * state_elem,), dtype=np.uint16)
        _runtime.memcpy(
            host_array_ptr(dev_state_raw),
            state_dev.ptr,
            dev_state_raw.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
    finally:
        bufs.close()

    dev_out_f32 = dev_out.reshape(rows, num_v_heads * head_v_dim)
    dev_state = dev_state_raw.view(np.float16).astype(np.float32).reshape(
        num_slots, num_v_heads, head_k_dim, head_v_dim
    )
    np.testing.assert_allclose(
        dev_out_f32, expected_out, rtol=5.0e-2, atol=5.0e-3
    )
    np.testing.assert_allclose(
        dev_state, final_slots_f32, rtol=2.0e-3, atol=2.0e-3
    )
