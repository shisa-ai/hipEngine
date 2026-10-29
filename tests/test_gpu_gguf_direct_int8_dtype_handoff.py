"""Real-device GGUF BF16 adapters around the FP16 Hadamard attention leaf."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_gpu_qwen35_int8_prefill_attention_gpu import (
    _alloc,
    _attention_lib,
    _download,
    _f32_to_bf16_f32,
    _free,
    _hip_available,
    _runtime,
    _upload,
)

pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("scale_dtype", ["fp32", "fp16"])
def test_gguf_hadamard_prefill_bf16_adapters_match_cpu(_runtime, _attention_lib, rows, scale_dtype):
    from hipengine.core.device import Device
    from hipengine.core.dtype import DType
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.cpu_reference import (
        paged_attn_decode_int8_hadamard_group32,
        quantize_kv_int8_hadamard_group32,
    )
    from hipengine.kernels.hip_gfx1100.attention import (
        qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans,
    )
    from hipengine.kernels.hip_gfx1100.convert import bf16_to_fp16, fp16_to_bf16
    from hipengine.kvcache import KVLiveSpans, KVScaleMetadata

    rng = np.random.default_rng(404)
    blocks, block_size, q_heads, kv_heads, dim = 2, 4, 4, 2, 256
    query = _f32_to_bf16_f32(rng.normal(0, 0.1, (rows, q_heads, dim)).astype(np.float32))
    key = rng.normal(0, 0.1, (blocks, block_size, kv_heads, dim)).astype(np.float32)
    value = rng.normal(0, 0.1, key.shape).astype(np.float32)
    np_scale_dtype = np.float32 if scale_dtype == "fp32" else np.float16
    qk, qv, ks, vs = quantize_kv_int8_hadamard_group32(key, value, scale_dtype=np_scale_dtype)
    gate = np.resize(np.array([-3, -1, 0, 0.125, 1, 3], np.float32), query.shape)
    gate_bits = (gate.view(np.uint32) >> 16).astype(np.uint16)
    table = np.tile(np.arange(blocks, dtype=np.int32), (rows, 1))
    positions = np.arange(rows, dtype=np.int64) + 2
    counts = np.full(rows, 2 + rows, dtype=np.int64)
    attention = paged_attn_decode_int8_hadamard_group32(
        query, qk, qv, ks, vs, positions + 1, block_table=table,
        block_size=block_size, scale=dim ** -0.5,
    )
    expected = _f32_to_bf16_f32(attention) / (1 + np.exp(-gate))
    expected = _f32_to_bf16_f32(expected.astype(np.float16).astype(np.float32))
    bufs = []
    try:
        uploaded = [_upload(_runtime, bufs, array) for array in
                    (query, qk, qv, ks, vs, gate_bits, table, counts, positions)]
        q_dev, k_dev, v_dev, ks_dev, vs_dev, gate_dev, table_dev, counts_dev, pos_dev = uploaded
        out_dev = _alloc(_runtime, bufs, gate_bits.nbytes)
        device = Device("hip", 0)
        dtype = DType.FP32 if scale_dtype == "fp32" else DType.FP16
        metadata = KVScaleMetadata(
            k_scale=Tensor.from_handle(ks_dev.ptr, ks.shape, dtype, device),
            v_scale=Tensor.from_handle(vs_dev.ptr, vs.shape, dtype, device),
            scale_dtype=dtype, granularity="hadamard_group32",
        )
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(table_dev.ptr, table.shape, DType.INT32, device),
            live_counts=Tensor.from_handle(counts_dev.ptr, counts.shape, DType.INT64, device),
            row_positions=Tensor.from_handle(pos_dev.ptr, positions.shape, DType.INT64, device),
            max_live_count=int(counts.max()), storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            span_role="prefill", scale_metadata=metadata,
        )
        bf16_to_fp16(gate_dev.ptr, gate_dev.ptr, gate.size, runtime=_runtime)
        qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans(
            q_dev.ptr, k_dev.ptr, v_dev.ptr, ks_dev.ptr, vs_dev.ptr,
            gate_dev.ptr, out_dev.ptr, spans, rows, int(counts.max()),
            block_size, q_heads, kv_heads, dim, dim, 1, dim ** -0.5,
            library=_attention_lib, runtime=_runtime,
        )
        fp16_to_bf16(out_dev.ptr, out_dev.ptr, gate.size, runtime=_runtime)
        actual_bits = _download(_runtime, out_dev, query.shape, np.uint16)
        actual = (actual_bits.astype(np.uint32) << 16).view(np.float32)
    finally:
        _free(_runtime, bufs)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0.03, atol=0.002)
