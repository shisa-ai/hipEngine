"""Numerical dtype handoffs between GGUF prefill and direct INT8 attention."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import qwen35_gguf_full_attention_bulk as bulk_mod
from hipengine.runtime import qwen35_gguf_runner as runner_mod


def _bf16_bits(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_values(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize("rows", [1, 3, 17])
@pytest.mark.parametrize("kernel", ["sequential", "flash", "wmma", "hadamard", "native"])
def test_direct_int8_prefill_converts_gate_and_output_at_kernel_boundary(
    monkeypatch, rows, kernel,
):
    """Run the real layer orchestration with independent numerical leaf doubles."""

    heads, kv_heads, dim = 24, 4, 256
    count = rows * heads * dim
    buffers = {}

    def buffer(size=count):
        values = np.zeros(size, dtype=np.uint16)
        buffers[values.ctypes.data] = values
        return SimpleNamespace(ptr=values.ctypes.data)

    gate_values = np.resize(np.array([-3, -1, 0, 0.125, 1, 3], np.float32), count)
    gate = buffer()
    gated = buffer()
    cfg = SimpleNamespace(
        head_count=heads, head_count_kv=kv_heads, key_length=dim,
        rope_dimension_count=64, rms_norm_eps=1e-6,
    )
    weight = SimpleNamespace(allocation=lambda: SimpleNamespace(tensor=buffer(1)))
    layer = SimpleNamespace(weight=lambda _name: weight)
    metadata = SimpleNamespace(
        granularity="hadamard_group32" if kernel == "hadamard" else "per_token_head",
        scale_dtype=DType.FP32, k_scale=buffer(1), v_scale=buffer(1),
    )
    spans = SimpleNamespace(
        storage_dtype=DType.BF16 if kernel == "native" else DType.INT8_PER_TOKEN_HEAD,
        scale_metadata=None if kernel == "native" else metadata,
    )
    scratch = SimpleNamespace(
        rows=rows, start=0, max_positions=rows + 7, block_size=256,
        append_spans=spans, prefill_spans=spans, full_gate=gate, full_gated=gated,
        **{name: buffer() for name in (
            "norm", "full_q", "full_k", "full_v", "full_query_raw", "full_key_raw",
            "full_query", "full_key", "positions_tensor", "key_cache", "value_cache",
            "attn_out",
        )},
        full_attn_split_batch_rows=rows, full_attn_split_count=1,
        full_attn_split_partial=buffer(), full_attn_split_m=buffer(), full_attn_split_l=buffer(),
        retained_key_cache=None, retained_value_cache=None,
    )
    runtime = object()
    casts = []
    launches = []
    consumed = []

    def bf16_to_fp16(src, dst, n, **kwargs):
        assert kwargs["stream"] == 7 and kwargs["runtime"] is runtime
        values = _bf16_values(buffers[src][:n]).astype(np.float16)
        buffers[dst][:n] = values.view(np.uint16)
        casts.append("gate")

    def fp16_to_bf16(src, dst, n, **kwargs):
        assert kwargs["stream"] == 7 and kwargs["runtime"] is runtime
        values = buffers[src][:n].view(np.float16).astype(np.float32)
        buffers[dst][:n] = _bf16_bits(values)
        casts.append("output")

    def split(_q, _query, gate_ptr, *_args, **_kwargs):
        buffers[gate_ptr][:] = _bf16_bits(gate_values)

    # The registered direct leaves consume FP16 gates. Hadamard also returns
    # FP16; the three per-token leaves return BF16. Model those ABI contracts
    # numerically instead of asserting that a particular cast helper is called.
    def attention(*args, **kwargs):
        assert kwargs["stream"] == 7 and kwargs["runtime"] is runtime
        actual_gate = (
            _bf16_values(buffers[args[3]][:count]) if kernel == "native"
            else buffers[args[5]][:count].view(np.float16).astype(np.float32)
        )
        np.testing.assert_array_equal(actual_gate, gate_values)
        output = np.float32(0.375) / (1 + np.exp(-actual_gate))
        out_index = 4 if kernel == "native" else 6
        buffers[args[out_index]][:count] = (
            output.astype(np.float16).view(np.uint16)
            if kernel == "hadamard" else _bf16_bits(output)
        )
        launches.append(kernel)

    expected = np.float32(0.375) / (1 + np.exp(-gate_values))
    if kernel == "hadamard":
        expected = expected.astype(np.float16).astype(np.float32)
    expected = _bf16_values(_bf16_bits(expected))

    def linear(_weight, x_ptr, _out_ptr, **_kwargs):
        np.testing.assert_array_equal(_bf16_values(buffers[x_ptr]), expected)
        consumed.append(True)

    noop = lambda *_args, **_kwargs: None

    def patch(name, value):
        monkeypatch.setattr(runner_mod, name, value)
        if hasattr(bulk_mod, name):
            monkeypatch.setattr(bulk_mod, name, value)

    for name in (
        "gguf_rmsnorm_bf16_f32_weight", "bf16_to_f32",
        "qwen35_write_paged_kv_mixed_value_bf16_prompt_spans",
        "gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight",
    ):
        patch(name, noop)
    patch("launch_gguf_linear_triple", lambda *_a, **_k: True)
    patch("qwen35_split_qgate_bf16", split)
    patch("_gguf_int8_kv_prompt_write_fn", lambda _m: noop)
    patch("_gguf_int8_prefill_kernel", lambda **_k: kernel)
    patch("_ensure_full_attn_split_rows", noop)
    patch("_try_launch_dense_q8_single_dp4a", lambda *_a, **_k: False)
    patch("launch_gguf_linear", linear)
    patch("bf16_to_fp16", bf16_to_fp16)
    patch("fp16_to_bf16", fp16_to_bf16)
    for name in (
        "qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_spans",
        "qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_flash_spans",
        "qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_wmma_spans",
        "qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans",
    ):
        patch(name, attention)
    runner = SimpleNamespace(
        weights=SimpleNamespace(config=cfg, layer=lambda _i: layer), runtime=runtime,
        backend="hip_gfx1100", hidden_size=5120, q_width=heads * dim,
        kv_width=kv_heads * dim, _cast_library=lambda: object(),
        _paged_kv_write_library=lambda: object(), _paged_attn_decode_library=lambda: object(),
        _run_post_attention_ffn_rows=noop, _full_attn_prefill_native_fn=lambda: attention,
    )

    used_aotriton = runner_mod.Qwen35GGUFFullStackRunner._run_full_attention_prefill_layer_aotriton(
        runner, 0, 1, 2, scratch, cos_table_ptr=3, sin_table_ptr=4,
        max_positions=rows + 7, stream=7, allow_aotriton=False,
    )

    assert used_aotriton is False
    assert launches == [kernel] and consumed == [True]
    assert casts == (
        [] if kernel == "native" else ["gate", "output"] if kernel == "hadamard" else ["gate"]
    )
