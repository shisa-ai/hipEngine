from __future__ import annotations

from dataclasses import dataclass

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from scripts.qwen35_gguf_bench import _decode_scratch_breakdown


class _Buffer:
    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = ptr
        self.nbytes = nbytes


@dataclass(frozen=True)
class _BulkScratch:
    key_cache: object | None
    value_cache: object | None
    append_spans: KVLiveSpans
    retained_key_cache: object | None = None
    retained_value_cache: object | None = None
    retained_append_spans: KVLiveSpans | None = None


def _tensor(ptr: int, shape: tuple[int, ...], dtype: DType) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _bf16_append_spans() -> KVLiveSpans:
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (4,), DType.INT32),
        live_counts=_tensor(0x2000, (1,), DType.INT64),
        max_live_count=255,
        storage_dtype=DType.BF16,
        span_role="prefill",
    )


def _scale_metadata() -> KVScaleMetadata:
    return KVScaleMetadata(
        k_scale=_tensor(0x3000, (4, 256, 2), DType.FP16),
        v_scale=_tensor(0x4000, (4, 256, 2), DType.FP16),
        scale_dtype=DType.FP16,
    )


def test_gguf_full_attention_prefill_scratch_retains_bf16_cache_by_default() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.kv_storage_dtype = DType.BF16
    retained_key = _Buffer(0x5000, 32)
    retained_value = _Buffer(0x6000, 32)
    session.scratch = type(
        "Scratch",
        (),
        {"full_cache": lambda self, layer_id: (retained_key, retained_value)},
    )()
    bulk = _BulkScratch(key_cache=None, value_cache=None, append_spans=_bf16_append_spans())

    layer_scratch = session._full_attention_prefill_scratch_for_layer(bulk, 7)

    assert layer_scratch.key_cache is retained_key
    assert layer_scratch.value_cache is retained_value
    assert layer_scratch.retained_key_cache is None
    assert layer_scratch.retained_value_cache is None
    assert layer_scratch.retained_append_spans is None


def test_gguf_int8_full_attention_prefill_uses_bf16_oracle_and_retained_int8_cache() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD
    oracle_key = _Buffer(0x5000, 32)
    oracle_value = _Buffer(0x6000, 32)
    retained_key = _Buffer(0x7000, 16)
    retained_value = _Buffer(0x8000, 16)
    metadata = _scale_metadata()
    session.scratch = type(
        "Scratch",
        (),
        {
            "full_cache": lambda self, layer_id: (retained_key, retained_value),
            "full_scale_metadata": lambda self, layer_id: metadata,
        },
    )()
    bulk = _BulkScratch(key_cache=oracle_key, value_cache=oracle_value, append_spans=_bf16_append_spans())

    layer_scratch = session._full_attention_prefill_scratch_for_layer(bulk, 7)

    assert layer_scratch.key_cache is oracle_key
    assert layer_scratch.value_cache is oracle_value
    assert layer_scratch.retained_key_cache is retained_key
    assert layer_scratch.retained_value_cache is retained_value
    assert layer_scratch.retained_append_spans is not None
    assert layer_scratch.retained_append_spans.storage_dtype is DType.INT8_PER_TOKEN_HEAD
    assert layer_scratch.retained_append_spans.scale_metadata is metadata
    assert layer_scratch.append_spans.storage_dtype is DType.BF16


def test_gguf_decode_scratch_breakdown_reports_int8_kv_scales_separately() -> None:
    key = _Buffer(0x1000, 100)
    value = _Buffer(0x2000, 100)
    k_scale = _Buffer(0x3000, 12)
    v_scale = _Buffer(0x4000, 12)
    other = _Buffer(0x5000, 7)
    scratch = type(
        "Scratch",
        (),
        {
            "buffers": (key, value, k_scale, v_scale, other),
            "full_key_caches": (key,),
            "full_value_caches": (value,),
            "full_k_scale_caches": (k_scale,),
            "full_v_scale_caches": (v_scale,),
            "layer_conv_states": (),
            "layer_recurrent_states": (),
            "kv_storage_dtype": DType.INT8_PER_TOKEN_HEAD,
            "kv_scale_dtype": DType.FP16,
        },
    )()

    breakdown = _decode_scratch_breakdown(scratch)

    assert breakdown["total_bytes"] == 231
    assert breakdown["kv_storage_dtype"] == "int8_per_token_head"
    assert breakdown["kv_scale_dtype"] == "fp16"
    assert breakdown["by_component_bytes"]["full_attention_kv_cache"] == 200
    assert breakdown["by_component_bytes"]["full_attention_kv_scales"] == 24
    assert breakdown["by_component_bytes"]["decode_workspace_other"] == 7
