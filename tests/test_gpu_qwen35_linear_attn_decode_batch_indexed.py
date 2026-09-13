from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import (
    gdn_prefill_recurrent_segments,
    linear_attn_conv_prefill_segments,
)
from hipengine.kernels.hip_gfx1100.linear_attn import (
    build_qwen35_linear_attn_conv,
    build_qwen35_linear_attn_gdn,
    qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16_fp16state,
    qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_fp16state,
    qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16_fp16state,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out_fp16state,
    qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16,
    qwen35_linear_attn_conv_decode_bf16,
    qwen35_linear_attn_conv_decode_indexed_bf16,
    register_qwen35_linear_attn_conv_kernels,
    register_qwen35_linear_attn_gdn_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve
from hipengine.runtime.qwen35_gguf_runner import (
    _resolve_gguf_linear_attention_decode_batch_plan,
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


def _gdn_cpu_reference(
    conv_out: np.ndarray,
    gate_bits: np.ndarray,
    a_bits: np.ndarray,
    b_bits: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    norm_weight: np.ndarray,
    recurrent_state: np.ndarray,
    cu_seqlens: np.ndarray,
    state_indices: np.ndarray,
    *,
    eps: float,
    num_k_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, _ = conv_out.shape
    _, num_v_heads, head_k_dim, head_v_dim = recurrent_state.shape
    key_dim = num_k_heads * head_k_dim
    gate = _bf16_bits_to_f32(gate_bits).reshape(rows, num_v_heads, head_v_dim)
    a = _bf16_bits_to_f32(a_bits).reshape(rows, num_v_heads)
    b = _bf16_bits_to_f32(b_bits).reshape(rows, num_v_heads)
    query = np.empty((rows, num_v_heads, head_k_dim), dtype=np.float32)
    key = np.empty_like(query)
    value = conv_out[:, 2 * key_dim :].reshape(rows, num_v_heads, head_v_dim)
    beta = np.asarray(np.float32(1.0) / (np.float32(1.0) + np.exp(-b)), dtype=np.float32)
    decay = np.asarray(
        np.exp(-np.exp(a_log[None, :]) * _softplus(a + dt_bias[None, :])),
        dtype=np.float32,
    )
    for row in range(rows):
        for v_head in range(num_v_heads):
            k_head = v_head % num_k_heads
            q_raw = conv_out[
                row, k_head * head_k_dim : (k_head + 1) * head_k_dim
            ].astype(np.float32)
            k_raw = conv_out[
                row,
                key_dim + k_head * head_k_dim : key_dim + (k_head + 1) * head_k_dim,
            ].astype(np.float32)
            q_norm = max(float(np.sqrt(np.sum(q_raw * q_raw, dtype=np.float32))), 1.0e-6)
            k_norm = max(float(np.sqrt(np.sum(k_raw * k_raw, dtype=np.float32))), 1.0e-6)
            query[row, v_head] = np.asarray(
                q_raw * np.float32(1.0 / q_norm) * np.float32(1.0 / np.sqrt(head_k_dim)),
                dtype=np.float32,
            )
            key[row, v_head] = np.asarray(k_raw * np.float32(1.0 / k_norm), dtype=np.float32)

    recurrent, state = gdn_prefill_recurrent_segments(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        cu_seqlens,
        state_indices,
    )
    out = np.empty_like(recurrent, dtype=np.float32)
    for row in range(rows):
        for v_head in range(num_v_heads):
            row_values = recurrent[row, v_head]
            inv_rms = np.float32(
                1.0
                / np.sqrt(
                    np.sum(row_values * row_values, dtype=np.float32)
                    / np.float32(head_v_dim)
                    + np.float32(eps)
                )
            )
            out[row, v_head] = np.asarray(
                row_values * inv_rms * norm_weight * _silu(gate[row, v_head]),
                dtype=np.float32,
            )
    return out.reshape(rows, num_v_heads * head_v_dim), state


def _kl_and_top1(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    ref -= np.max(ref, axis=1, keepdims=True)
    cand -= np.max(cand, axis=1, keepdims=True)
    ref_p = np.exp(ref)
    cand_p = np.exp(cand)
    ref_p /= np.sum(ref_p, axis=1, keepdims=True)
    cand_p /= np.sum(cand_p, axis=1, keepdims=True)
    kl = np.sum(ref_p * (np.log(ref_p) - np.log(cand_p)), axis=1)
    top1 = np.mean(np.argmax(reference, axis=1) == np.argmax(candidate, axis=1))
    return float(np.max(kl)), float(top1)


def test_indexed_decode_kernels_register_gguf_batch_variants() -> None:
    clear_registry_for_tests()
    register_qwen35_linear_attn_conv_kernels()
    register_qwen35_linear_attn_gdn_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_conv_decode",
            quant="gguf_qwen35",
            variant="bf16_indexed",
        )
        is qwen35_linear_attn_conv_decode_indexed_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_recurrent_rmsnorm_gate",
            quant="gguf_qwen35",
            variant="bf16_segments",
        )
        is qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_recurrent_rmsnorm_gate",
            quant="gguf_qwen35",
            variant="bf16_indexed_singleton",
        )
        is qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16
    )
    fp16_variants = {
        (
            "gdn_recurrent_rmsnorm_gate",
            "gguf_qwen35",
            "bf16_fp16state",
        ): qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16_fp16state,
        (
            "gdn_recurrent_rmsnorm_gate+cast",
            "gguf_q5_k_t16_v1",
            "bf16_lowp_f32_bf16_out_fp16state",
        ): qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out_fp16state,
        (
            "gdn_chain_recurrent_rmsnorm_gate",
            "gguf_qwen35",
            "bf16_c1_exact_state_rows_tloop_fp16state",
        ): qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16_fp16state,
        (
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_compact_normalized_wave32_xor_fp16state",
        ): qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_fp16state,
    }
    for (layer, quant, variant), expected in fp16_variants.items():
        assert (
            resolve(
                backend="hip_gfx1100",
                layer=layer,
                quant=quant,
                variant=variant,
            )
            is expected
        )

    gfx1100_plan = _resolve_gguf_linear_attention_decode_batch_plan("hip_gfx1100")
    gfx1151_plan = _resolve_gguf_linear_attention_decode_batch_plan("hip_gfx1151")
    assert (
        gfx1100_plan.gdn_indexed_singleton
        is qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16
    )
    assert gfx1100_plan.gdn_decode_path == "indexed_singleton"
    assert (
        gfx1151_plan.gdn_indexed_singleton
        is qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16
    )
    assert gfx1151_plan.gdn_decode_path == "indexed_singleton"
    assert callable(gfx1100_plan.gdn_segments)
    assert callable(gfx1151_plan.gdn_segments)


def test_indexed_decode_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        qwen35_linear_attn_conv_decode_indexed_bf16(0, 0, 0, 0, 0, 0, 0, 4)
    with pytest.raises(ValueError, match="segments must be positive"):
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1.0e-6, 1, 2, 8, 4
        )
    with pytest.raises(ValueError, match="rows must be positive"):
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 1, 2, 8, 4
        )
    with pytest.raises(ValueError, match="rows must be positive"):
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 1, 2, 8, 4
        )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_sparse_indexed_conv_gdn_matches_independent_c1_and_cpu_reference() -> None:
    rng = np.random.default_rng(20260716)
    rows = 4
    state_slots = 6
    state_indices = np.asarray([4, 1, 5, 0], dtype=np.int64)
    cu_seqlens = np.arange(rows + 1, dtype=np.int32)
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 8
    head_v_dim = 4
    key_dim = num_k_heads * head_k_dim
    channels = 2 * key_dim + num_v_heads * head_v_dim
    kernel_size = 4
    eps = 1.0e-6

    hidden_f32 = rng.normal(0.0, 0.35, size=(rows, channels)).astype(np.float32)
    hidden_bits = _f32_to_bf16_bits(hidden_f32)
    hidden_quantized = _bf16_bits_to_f32(hidden_bits)
    conv_state = rng.normal(
        0.0, 0.2, size=(state_slots, channels, kernel_size)
    ).astype(np.float32)
    conv_weight = rng.normal(0.0, 0.15, size=(channels, kernel_size)).astype(np.float32)

    gate_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(rows, num_v_heads * head_v_dim)).astype(np.float32)
    )
    a_bits = _f32_to_bf16_bits(
        rng.normal(-0.1, 0.25, size=(rows, num_v_heads)).astype(np.float32)
    )
    b_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.3, size=(rows, num_v_heads)).astype(np.float32)
    )
    dt_bias = rng.normal(0.0, 0.2, size=(num_v_heads,)).astype(np.float32)
    a_log = rng.normal(-0.75, 0.2, size=(num_v_heads,)).astype(np.float32)
    norm_weight = rng.normal(1.0, 0.1, size=(head_v_dim,)).astype(np.float32)
    recurrent_state = rng.normal(
        0.0,
        0.1,
        size=(state_slots, num_v_heads, head_k_dim, head_v_dim),
    ).astype(np.float32)

    buffers = _Buffers()
    try:
        hidden_dev = buffers.from_host(hidden_bits)
        conv_weight_dev = buffers.from_host(conv_weight)
        state_indices_dev = buffers.from_host(state_indices)
        cu_seqlens_dev = buffers.from_host(cu_seqlens)
        serial_conv_state_dev = buffers.from_host(conv_state)
        batch_conv_state_dev = buffers.from_host(conv_state)
        serial_conv_out_dev = buffers.empty(rows * channels * np.dtype(np.float32).itemsize)
        batch_conv_out_dev = buffers.empty(rows * channels * np.dtype(np.float32).itemsize)

        require_cached = os.environ.get("HIPENGINE_TEST_REQUIRE_CACHED_BUILD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        conv_library = build_qwen35_linear_attn_conv(
            load=True,
            require_cached=require_cached,
        )
        conv_state_stride = channels * kernel_size * np.dtype(np.float32).itemsize
        hidden_stride = channels * np.dtype(np.uint16).itemsize
        conv_out_stride = channels * np.dtype(np.float32).itemsize
        for row, slot in enumerate(state_indices.tolist()):
            qwen35_linear_attn_conv_decode_bf16(
                hidden_dev.ptr + row * hidden_stride,
                serial_conv_state_dev.ptr + int(slot) * conv_state_stride,
                conv_weight_dev.ptr,
                serial_conv_out_dev.ptr + row * conv_out_stride,
                channels,
                kernel_size,
                library=conv_library,
            )
        qwen35_linear_attn_conv_decode_indexed_bf16(
            hidden_dev.ptr,
            batch_conv_state_dev.ptr,
            conv_weight_dev.ptr,
            batch_conv_out_dev.ptr,
            state_indices_dev.ptr,
            rows,
            channels,
            kernel_size,
            library=conv_library,
        )

        gate_dev = buffers.from_host(gate_bits)
        a_dev = buffers.from_host(a_bits)
        b_dev = buffers.from_host(b_bits)
        dt_bias_dev = buffers.from_host(dt_bias)
        a_log_dev = buffers.from_host(a_log)
        norm_weight_dev = buffers.from_host(norm_weight)
        serial_recurrent_state_dev = buffers.from_host(recurrent_state)
        batch_recurrent_state_dev = buffers.from_host(recurrent_state)
        indexed_recurrent_state_dev = buffers.from_host(recurrent_state)
        cached_recurrent_state_dev = buffers.from_host(recurrent_state)
        serial_gdn_out_dev = buffers.empty(
            rows * num_v_heads * head_v_dim * np.dtype(np.float32).itemsize
        )
        batch_gdn_out_dev = buffers.empty(
            rows * num_v_heads * head_v_dim * np.dtype(np.float32).itemsize
        )
        indexed_gdn_out_dev = buffers.empty(
            rows * num_v_heads * head_v_dim * np.dtype(np.float32).itemsize
        )
        cached_gdn_out_dev = buffers.empty(
            rows * num_v_heads * head_v_dim * np.dtype(np.float32).itemsize
        )
        gdn_library = build_qwen35_linear_attn_gdn(
            load=True,
            require_cached=require_cached,
        )
        recurrent_state_stride = (
            num_v_heads * head_k_dim * head_v_dim * np.dtype(np.float32).itemsize
        )
        lowp_scalar_stride = num_v_heads * np.dtype(np.uint16).itemsize
        gate_stride = num_v_heads * head_v_dim * np.dtype(np.uint16).itemsize
        gdn_out_stride = num_v_heads * head_v_dim * np.dtype(np.float32).itemsize
        for row, slot in enumerate(state_indices.tolist()):
            qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
                serial_conv_out_dev.ptr + row * conv_out_stride,
                gate_dev.ptr + row * gate_stride,
                a_dev.ptr + row * lowp_scalar_stride,
                b_dev.ptr + row * lowp_scalar_stride,
                dt_bias_dev.ptr,
                a_log_dev.ptr,
                norm_weight_dev.ptr,
                serial_recurrent_state_dev.ptr + int(slot) * recurrent_state_stride,
                serial_gdn_out_dev.ptr + row * gdn_out_stride,
                eps,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                library=gdn_library,
            )
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16(
            batch_conv_out_dev.ptr,
            gate_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_bias_dev.ptr,
            a_log_dev.ptr,
            norm_weight_dev.ptr,
            batch_recurrent_state_dev.ptr,
            batch_gdn_out_dev.ptr,
            cu_seqlens_dev.ptr,
            state_indices_dev.ptr,
            rows,
            rows,
            eps,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=gdn_library,
        )
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16(
            batch_conv_out_dev.ptr,
            gate_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_bias_dev.ptr,
            a_log_dev.ptr,
            norm_weight_dev.ptr,
            indexed_recurrent_state_dev.ptr,
            indexed_gdn_out_dev.ptr,
            state_indices_dev.ptr,
            rows,
            eps,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=gdn_library,
        )
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16(
            batch_conv_out_dev.ptr,
            gate_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_bias_dev.ptr,
            a_log_dev.ptr,
            norm_weight_dev.ptr,
            cached_recurrent_state_dev.ptr,
            cached_gdn_out_dev.ptr,
            state_indices_dev.ptr,
            rows,
            eps,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=gdn_library,
        )

        ctypes.CDLL("libamdhip64.so").hipDeviceSynchronize()
        serial_conv_out = _from_device(serial_conv_out_dev, (rows, channels), np.float32)
        batch_conv_out = _from_device(batch_conv_out_dev, (rows, channels), np.float32)
        serial_conv_after = _from_device(
            serial_conv_state_dev, conv_state.shape, np.float32
        )
        batch_conv_after = _from_device(batch_conv_state_dev, conv_state.shape, np.float32)
        serial_gdn_out = _from_device(
            serial_gdn_out_dev, (rows, num_v_heads * head_v_dim), np.float32
        )
        batch_gdn_out = _from_device(
            batch_gdn_out_dev, (rows, num_v_heads * head_v_dim), np.float32
        )
        indexed_gdn_out = _from_device(
            indexed_gdn_out_dev, (rows, num_v_heads * head_v_dim), np.float32
        )
        cached_gdn_out = _from_device(
            cached_gdn_out_dev, (rows, num_v_heads * head_v_dim), np.float32
        )
        serial_recurrent_after = _from_device(
            serial_recurrent_state_dev, recurrent_state.shape, np.float32
        )
        batch_recurrent_after = _from_device(
            batch_recurrent_state_dev, recurrent_state.shape, np.float32
        )
        indexed_recurrent_after = _from_device(
            indexed_recurrent_state_dev, recurrent_state.shape, np.float32
        )
        cached_recurrent_after = _from_device(
            cached_recurrent_state_dev, recurrent_state.shape, np.float32
        )
    finally:
        buffers.close()

    np.testing.assert_array_equal(batch_conv_out, serial_conv_out)
    np.testing.assert_array_equal(batch_conv_after, serial_conv_after)
    np.testing.assert_array_equal(batch_gdn_out, serial_gdn_out)
    np.testing.assert_array_equal(batch_recurrent_after, serial_recurrent_after)
    np.testing.assert_array_equal(indexed_gdn_out, serial_gdn_out)
    np.testing.assert_array_equal(indexed_recurrent_after, serial_recurrent_after)
    np.testing.assert_array_equal(cached_gdn_out, serial_gdn_out)
    np.testing.assert_array_equal(cached_recurrent_after, serial_recurrent_after)

    cpu_conv_out, cpu_conv_after = linear_attn_conv_prefill_segments(
        hidden_quantized,
        conv_state,
        conv_weight,
        cu_seqlens,
        state_indices,
    )
    cpu_gdn_out, cpu_recurrent_after = _gdn_cpu_reference(
        cpu_conv_out,
        gate_bits,
        a_bits,
        b_bits,
        dt_bias,
        a_log,
        norm_weight,
        recurrent_state,
        cu_seqlens,
        state_indices,
        eps=eps,
        num_k_heads=num_k_heads,
    )
    np.testing.assert_allclose(batch_conv_out, cpu_conv_out, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_array_equal(batch_conv_after, cpu_conv_after)
    np.testing.assert_allclose(batch_gdn_out, cpu_gdn_out, rtol=2.0e-5, atol=2.0e-5)
    np.testing.assert_allclose(
        batch_recurrent_after, cpu_recurrent_after, rtol=2.0e-6, atol=2.0e-6
    )

    inactive_slots = sorted(set(range(state_slots)) - set(state_indices.tolist()))
    np.testing.assert_array_equal(batch_conv_after[inactive_slots], conv_state[inactive_slots])
    np.testing.assert_array_equal(
        batch_recurrent_after[inactive_slots], recurrent_state[inactive_slots]
    )

    probe = rng.normal(
        0.0, 0.25, size=(num_v_heads * head_v_dim, 17)
    ).astype(np.float32)
    cpu_logits = cpu_gdn_out @ probe
    batch_logits = batch_gdn_out @ probe
    max_kl, top1 = _kl_and_top1(cpu_logits, batch_logits)
    assert max_kl <= 0.05
    assert top1 >= 0.90

    rounded_gdn = _bf16_bits_to_f32(_f32_to_bf16_bits(batch_gdn_out))
    assert np.any(rounded_gdn != batch_gdn_out)
    assert np.any((rounded_gdn @ probe) != batch_logits)


def test_indexed_decode_plan_switches_to_fp16_kernels_under_flag(monkeypatch) -> None:
    """Under the fp16-state flag the indexed decode keeps its fast singleton path.

    The fp16-state route must not run the FP32-state indexed-singleton kernels
    against the half-sized state buffer.  The batch plan must resolve the
    fp16-state siblings (indexed singleton + segments) under the flag, and
    keep the strict gfx1151 singleton path unchanged without it.
    """
    from hipengine.kernels.hip_gfx1100.linear_attn import (
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16,
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state,
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16,
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state,
    )
    from hipengine.runtime.qwen35_gguf_runner import (
        _resolve_gguf_linear_attention_decode_batch_plan,
    )

    monkeypatch.delenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", raising=False)
    strict_plan = _resolve_gguf_linear_attention_decode_batch_plan("hip_gfx1151")
    explicit_fp16_plan = _resolve_gguf_linear_attention_decode_batch_plan(
        "hip_gfx1151",
        use_fp16_state=True,
    )
    assert strict_plan.gdn_decode_path == "indexed_singleton"
    assert (
        strict_plan.gdn_indexed_singleton
        is qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16
    )
    assert strict_plan.gdn_segments is qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16

    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "1")
    fp16_plan = _resolve_gguf_linear_attention_decode_batch_plan("hip_gfx1151")
    explicit_strict_plan = _resolve_gguf_linear_attention_decode_batch_plan(
        "hip_gfx1151",
        use_fp16_state=False,
    )
    assert fp16_plan.gdn_decode_path == "indexed_singleton"
    assert explicit_fp16_plan == fp16_plan
    assert explicit_strict_plan == strict_plan
    assert (
        fp16_plan.gdn_indexed_singleton
        is qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state
    )
    assert (
        fp16_plan.gdn_segments
        is qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state
    )
