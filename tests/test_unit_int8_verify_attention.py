from dataclasses import replace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.attention import paged_attn_decode as attention
from hipengine.kernels.registry import resolve
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata


def _spans():
    device = Device("hip", 0)
    metadata = KVScaleMetadata(
        Tensor.from_handle(0x1000, (4, 256, 4), DType.FP32, device),
        Tensor.from_handle(0x2000, (4, 256, 4), DType.FP32, device),
        scale_dtype=DType.FP32,
    )
    return KVLiveSpans.paged_uniform(
        block_table=Tensor.from_handle(0x3000, (2,), DType.INT32, device),
        live_counts=Tensor.from_handle(0x4000, (4,), DType.INT64, device),
        max_live_count=258,
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        scale_metadata=metadata,
        span_role="verify_chain",
    )


def _check(spans):
    return attention._check_int8_qwen35_gqa_batch_shape(
        spans, 4, 256, 2, 256, 24, 4, 256,
        k_scale_ptr=0x1000, v_scale_ptr=0x2000, shared_table=True,
    )


def test_verify_attention_accepts_one_table_for_four_causal_rows():
    assert _check(_spans()) == 2


@pytest.mark.parametrize("role", ["decode", "prefill", "verify_tree"])
def test_verify_attention_rejects_other_span_roles(role):
    with pytest.raises(ValueError, match="verify_chain"):
        _check(replace(_spans(), span_role=role))


def test_verify_attention_rejects_per_request_tables():
    spans = _spans()
    table = Tensor.from_handle(0x3000, (4, 2), DType.INT32, Device("hip", 0))
    with pytest.raises(ValueError, match="single shared"):
        _check(replace(spans, base_offsets=table))


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_verify_attention_and_c1_fallback_are_registered(backend):
    load_backend_kernel_package(backend)
    assert resolve(
        backend=backend, layer="paged_attn_decode", quant="int8_per_token_head",
        variant="per_token_head_gqa_splitk_gate_bf16_verify_chain_spans",
    ) is attention.qwen35_paged_attn_verify_int8_gqa_splitk_gate_bf16_spans
    assert callable(resolve(
        backend=backend, layer="paged_attn_decode", quant="int8_per_token_head",
        variant="gqa_splitk_gate_bf16_spans",
    ))
