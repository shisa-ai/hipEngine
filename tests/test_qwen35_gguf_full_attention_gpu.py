from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.fused import gguf_rmsnorm_bf16_f32_weight
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import gguf_q6_k_embedding_bf16_out
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.gguf_linear import GGUF_OUTPUT_F32, launch_gguf_linear
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner
import hipengine.runtime.qwen35_gguf_runner as gguf_runner

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def test_qwen35_gguf_full_attention_gpu_prelude_matches_cpu_oracle() -> None:
    """Compare the first full-attention layer's GPU prelude/KV path to the old CPU bridge."""

    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    reader = GGUFReader(MODEL)
    q_norm = np.asarray(reader.tensor_data("blk.3.attn_q_norm.weight"), dtype=np.float32)
    k_norm = np.asarray(reader.tensor_data("blk.3.attn_k_norm.weight"), dtype=np.float32)
    buffers = []
    with Qwen35GGUFFullStackRunner(MODEL) as runner:
        scratch_gpu = gguf_runner._FullStackScratch.allocate(runner, runtime=runtime)
        scratch_cpu = gguf_runner._FullStackScratch.allocate(runner, runtime=runtime)
        buffers.extend((*scratch_gpu.buffers, *scratch_cpu.buffers))
        token_gpu = malloc(np.dtype(np.int64).itemsize, runtime=runtime)
        token_cpu = malloc(np.dtype(np.int64).itemsize, runtime=runtime)
        hidden_gpu_a = malloc(runner.hidden_size * 2, runtime=runtime)
        hidden_gpu_b = malloc(runner.hidden_size * 2, runtime=runtime)
        hidden_cpu_a = malloc(runner.hidden_size * 2, runtime=runtime)
        hidden_cpu_b = malloc(runner.hidden_size * 2, runtime=runtime)
        out_gpu = malloc(runner.hidden_size * 2, runtime=runtime)
        out_cpu = malloc(runner.hidden_size * 2, runtime=runtime)
        buffers.extend((token_gpu, token_cpu, hidden_gpu_a, hidden_gpu_b, hidden_cpu_a, hidden_cpu_b, out_gpu, out_cpu))
        histories: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {3: ([], [])}
        try:
            scratch_gpu.zero_states(runtime)
            scratch_cpu.zero_states(runtime)
            for position, token_id in enumerate((760, 4087)):
                src_gpu = _run_prefix_to_first_full_attention(
                    runner, scratch_gpu, token_gpu, hidden_gpu_a, hidden_gpu_b, token_id, position, runtime
                )
                runner._run_full_attention_layer(3, src_gpu.ptr, out_gpu.ptr, scratch_gpu, position=position)
                src_cpu = _run_prefix_to_first_full_attention(
                    runner, scratch_cpu, token_cpu, hidden_cpu_a, hidden_cpu_b, token_id, position, runtime
                )
                _run_full_attention_layer_cpu_bridge(
                    runner,
                    3,
                    src_cpu.ptr,
                    out_cpu.ptr,
                    scratch_cpu,
                    position=position,
                    q_norm=q_norm,
                    k_norm=k_norm,
                    histories=histories,
                    runtime=runtime,
                )
            runtime.device_synchronize()
            gpu_bits = np.empty((1, runner.hidden_size), dtype=np.uint16)
            cpu_bits = np.empty_like(gpu_bits)
            copy_device_to_host(host_array_ptr(gpu_bits), out_gpu, runtime=runtime)
            copy_device_to_host(host_array_ptr(cpu_bits), out_cpu, runtime=runtime)
            gpu = bf16_to_float32(gpu_bits)
            cpu = bf16_to_float32(cpu_bits)
            np.testing.assert_allclose(gpu, cpu, rtol=2.5e-2, atol=3.0e-2)
            logits_gpu = _lm_head_logits(runner, out_gpu, runtime=runtime)
            logits_cpu = _lm_head_logits(runner, out_cpu, runtime=runtime)
            assert int(np.argmax(logits_gpu)) == int(np.argmax(logits_cpu))
            assert _kl(logits_cpu.reshape(-1), logits_gpu.reshape(-1)) <= 0.05
        finally:
            pass
    for buffer in reversed(buffers):
        free(buffer, runtime=runtime)


def test_qwen35_gguf_full_attention_prefill_threshold_and_oracle() -> None:
    """Native GQA prefill is selected at threshold and matches the CPU bridge on the final row."""

    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    reader = GGUFReader(MODEL)
    q_norm = np.asarray(reader.tensor_data("blk.3.attn_q_norm.weight"), dtype=np.float32)
    k_norm = np.asarray(reader.tensor_data("blk.3.attn_k_norm.weight"), dtype=np.float32)
    with Qwen35GGUFFullStackRunner(MODEL) as runner:
        hidden_rows = _prefix_hidden_rows(runner, (760, 4087), runtime)
        native = runner.run_full_attention_prefill_layer(3, hidden_rows, attn_aotriton_min_tokens=3)
        bulk = runner.run_full_attention_prefill_layer(3, hidden_rows, attn_aotriton_min_tokens=2)
        assert native.mode == "native_sequential"
        assert not native.used_aotriton
        assert bulk.mode == "native_gqa_bf16"
        assert not bulk.used_aotriton
        cpu_bits = _cpu_bridge_prefill_outputs(runner, hidden_rows, q_norm, k_norm, runtime)
        # The native GQA prefill kernel mirrors the resident decode attention
        # numerics, including BF16 gate/output projection. The last prompt row is
        # the prefill row consumed by generation; require it to stay close and
        # preserve the lm-head distribution gate.
        np.testing.assert_allclose(
            bf16_to_float32(bulk.hidden_bits[-1:]),
            bf16_to_float32(cpu_bits[-1:]),
            rtol=0.20,
            atol=0.15,
        )
        logits_bulk = _logits_from_host_bits(runner, bulk.hidden_bits[-1:], runtime=runtime)
        logits_cpu = _logits_from_host_bits(runner, cpu_bits[-1:], runtime=runtime)
        assert int(np.argmax(logits_bulk)) == int(np.argmax(logits_cpu))
        assert _kl(logits_cpu.reshape(-1), logits_bulk.reshape(-1)) <= 0.05


def _prefix_hidden_rows(runner: Qwen35GGUFFullStackRunner, token_ids: tuple[int, ...], runtime) -> np.ndarray:
    scratch = gguf_runner._FullStackScratch.allocate(runner, runtime=runtime)
    token_buf = malloc(np.dtype(np.int64).itemsize, runtime=runtime)
    hidden_a = malloc(runner.hidden_size * 2, runtime=runtime)
    hidden_b = malloc(runner.hidden_size * 2, runtime=runtime)
    hidden_rows = np.empty((len(token_ids), runner.hidden_size), dtype=np.uint16)
    try:
        scratch.zero_states(runtime)
        for position, token_id in enumerate(token_ids):
            src = _run_prefix_to_first_full_attention(runner, scratch, token_buf, hidden_a, hidden_b, token_id, position, runtime)
            copy_device_to_host(host_array_ptr(hidden_rows[position : position + 1]), src, runtime=runtime)
    finally:
        for buffer in reversed((hidden_b, hidden_a, token_buf, *scratch.buffers)):
            free(buffer, runtime=runtime)
    return hidden_rows


def _cpu_bridge_prefill_outputs(
    runner: Qwen35GGUFFullStackRunner,
    hidden_rows: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    runtime,
) -> np.ndarray:
    scratch = gguf_runner._FullStackScratch.allocate(runner, runtime=runtime)
    hidden_buf = malloc(runner.hidden_size * 2, runtime=runtime)
    out_buf = malloc(runner.hidden_size * 2, runtime=runtime)
    out = np.empty_like(hidden_rows)
    histories: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {3: ([], [])}
    try:
        scratch.zero_states(runtime)
        for position, row in enumerate(hidden_rows):
            row_bits = np.ascontiguousarray(row.reshape(1, runner.hidden_size), dtype=np.uint16)
            copy_host_to_device(hidden_buf, host_array_ptr(row_bits), runtime=runtime)
            _run_full_attention_layer_cpu_bridge(
                runner,
                3,
                hidden_buf.ptr,
                out_buf.ptr,
                scratch,
                position=position,
                q_norm=q_norm,
                k_norm=k_norm,
                histories=histories,
                runtime=runtime,
            )
            copy_device_to_host(host_array_ptr(out[position : position + 1]), out_buf, runtime=runtime)
    finally:
        for buffer in reversed((out_buf, hidden_buf, *scratch.buffers)):
            free(buffer, runtime=runtime)
    return out


def _logits_from_host_bits(runner: Qwen35GGUFFullStackRunner, hidden_bits: np.ndarray, *, runtime) -> np.ndarray:
    hidden = np.ascontiguousarray(hidden_bits, dtype=np.uint16)
    hidden_buf = malloc(hidden.nbytes, runtime=runtime)
    try:
        copy_host_to_device(hidden_buf, host_array_ptr(hidden), runtime=runtime)
        return _lm_head_logits(runner, hidden_buf, runtime=runtime)
    finally:
        free(hidden_buf, runtime=runtime)


def _run_prefix_to_first_full_attention(
    runner: Qwen35GGUFFullStackRunner,
    scratch,
    token_buf,
    hidden_a,
    hidden_b,
    token_id: int,
    position: int,
    runtime,
):
    assert runner.weights is not None
    scratch.set_full_attention_position(position, runtime)
    token_arr = np.asarray([int(token_id)], dtype=np.int64)
    copy_host_to_device(token_buf, host_array_ptr(token_arr), runtime=runtime)
    gguf_q6_k_embedding_bf16_out(
        token_buf.ptr,
        runner.weights.root("token_embedding").allocation().tensor.ptr,
        hidden_a.ptr,
        rows=1,
        hidden_size=runner.hidden_size,
        vocab_size=runner.vocab_size,
        runtime=runtime,
    )
    src = hidden_a
    dst = hidden_b
    for layer_id in range(3):
        runner._run_linear_attention_layer(layer_id, src.ptr, dst.ptr, scratch)
        src, dst = dst, src
    return src


def _run_full_attention_layer_cpu_bridge(
    runner: Qwen35GGUFFullStackRunner,
    layer_id: int,
    hidden_ptr: int,
    out_ptr: int,
    scratch,
    *,
    position: int,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    histories: dict[int, tuple[list[np.ndarray], list[np.ndarray]]],
    runtime,
) -> None:
    assert runner.weights is not None
    layer = runner.weights.layer(layer_id)
    cfg = runner.weights.config
    gguf_rmsnorm_bf16_f32_weight(
        hidden_ptr,
        layer.weight("attn_norm").allocation().tensor.ptr,
        scratch.norm.ptr,
        rows=1,
        hidden_size=runner.hidden_size,
        eps=cfg.rms_norm_eps,
        runtime=runtime,
    )
    launch_gguf_linear(
        layer.weight("attn_q"),
        scratch.norm.ptr,
        scratch.full_q.ptr,
        rows=1,
        in_features=runner.hidden_size,
        out_features=2 * runner.q_width,
        runtime=runtime,
    )
    launch_gguf_linear(
        layer.weight("attn_k"),
        scratch.norm.ptr,
        scratch.full_k.ptr,
        rows=1,
        in_features=runner.hidden_size,
        out_features=runner.kv_width,
        runtime=runtime,
    )
    launch_gguf_linear(
        layer.weight("attn_v"),
        scratch.norm.ptr,
        scratch.full_v.ptr,
        rows=1,
        in_features=runner.hidden_size,
        out_features=runner.kv_width,
        runtime=runtime,
    )
    runtime.device_synchronize()
    q_full = _copy_bf16_device_to_f32(scratch.full_q, 2 * runner.q_width, runtime=runtime)
    key = _copy_bf16_device_to_f32(scratch.full_k, runner.kv_width, runtime=runtime)
    value = _copy_bf16_device_to_f32(scratch.full_v, runner.kv_width, runtime=runtime)
    context = _host_full_attention(runner, q_full, key, value, q_norm, k_norm, histories[layer_id], position=position)
    context_bits = float_array_to_bf16_bits(context.reshape(1, runner.q_width))
    copy_host_to_device(scratch.full_gated, host_array_ptr(context_bits), runtime=runtime)
    launch_gguf_linear(
        layer.weight("attn_output"),
        scratch.full_gated.ptr,
        scratch.attn_out.ptr,
        rows=1,
        in_features=runner.q_width,
        out_features=runner.hidden_size,
        runtime=runtime,
    )
    runner._run_post_attention_ffn(layer_id, hidden_ptr, scratch.attn_out.ptr, out_ptr, scratch)


def _host_full_attention(
    runner: Qwen35GGUFFullStackRunner,
    q_full: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    histories: tuple[list[np.ndarray], list[np.ndarray]],
    *,
    position: int,
) -> np.ndarray:
    assert runner.weights is not None
    cfg = runner.weights.config
    head_dim = cfg.key_length
    q_full = q_full.reshape(cfg.head_count, 2 * head_dim)
    query = _rmsnorm(q_full[:, :head_dim], 1.0 + q_norm, cfg.rms_norm_eps)
    gate = q_full[:, head_dim:]
    key = _rmsnorm(key.reshape(cfg.head_count_kv, head_dim), 1.0 + k_norm, cfg.rms_norm_eps)
    value = value.reshape(cfg.head_count_kv, cfg.value_length)
    query = _apply_rope(query, position, cfg.rope_dimension_count, cfg.rope_freq_base)
    key = _apply_rope(key, position, cfg.rope_dimension_count, cfg.rope_freq_base)
    k_history, v_history = histories
    k_history.append(key.copy())
    v_history.append(value.copy())
    keys = np.stack(k_history, axis=0)
    values = np.stack(v_history, axis=0)
    out = np.empty((cfg.head_count, cfg.value_length), dtype=np.float32)
    group = cfg.head_count // cfg.head_count_kv
    scale = 1.0 / np.sqrt(float(head_dim))
    for head in range(cfg.head_count):
        kv_head = head // group
        scores = keys[:, kv_head, :] @ query[head]
        scores = scores.astype(np.float32) * scale
        scores = scores - np.max(scores)
        probs = np.exp(scores).astype(np.float32)
        probs /= np.sum(probs)
        out[head] = probs @ values[:, kv_head, :]
    return out * _sigmoid(gate)


def _lm_head_logits(runner: Qwen35GGUFFullStackRunner, hidden_buf, *, runtime) -> np.ndarray:
    logits = np.empty((1, runner.vocab_size), dtype=np.float32)
    logits_buf = malloc(logits.nbytes, runtime=runtime)
    try:
        launch_gguf_linear(
            runner.weights.root("lm_head"),
            hidden_buf.ptr,
            logits_buf.ptr,
            rows=1,
            in_features=runner.hidden_size,
            out_features=runner.vocab_size,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(logits), logits_buf, runtime=runtime)
    finally:
        free(logits_buf, runtime=runtime)
    return logits


def _copy_bf16_device_to_f32(buffer, elements: int, *, runtime) -> np.ndarray:
    bits = np.empty((elements,), dtype=np.uint16)
    copy_device_to_host(host_array_ptr(bits), buffer, runtime=runtime)
    return bf16_to_float32(bits)


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    inv_rms = 1.0 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + np.float32(eps))
    return x32 * inv_rms * weight.astype(np.float32)


def _apply_rope(x: np.ndarray, position: int, rotary_dim: int, base: float) -> np.ndarray:
    out = np.array(x, dtype=np.float32, copy=True)
    half = rotary_dim // 2
    dims = np.arange(half, dtype=np.float32)
    inv_freq = np.power(np.float32(base), -dims / np.float32(half))
    angles = np.float32(position) * inv_freq
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    first = out[..., :half].copy()
    second = out[..., half:rotary_dim].copy()
    out[..., :half] = first * cos - second * sin
    out[..., half:rotary_dim] = second * cos + first * sin
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    positive = x32 >= 0
    out = np.empty_like(x32)
    out[positive] = 1.0 / (1.0 + np.exp(-x32[positive]))
    exp_x = np.exp(x32[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _kl(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    ref = reference_logits.astype(np.float64)
    cand = candidate_logits.astype(np.float64)
    ref = ref - np.max(ref)
    cand = cand - np.max(cand)
    p = np.exp(ref)
    q = np.exp(cand)
    p /= np.sum(p)
    q /= np.sum(q)
    eps = 1.0e-30
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True
