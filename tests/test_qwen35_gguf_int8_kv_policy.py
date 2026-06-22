from __future__ import annotations

from dataclasses import dataclass

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata
from hipengine.runtime.qwen35_gguf_runner import (
    _GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV,
    _GGUF_INT8_BF16_PREFIX_FULL_ATTENTION_ENV,
    _GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS,
    Qwen35GGUFResidentSession,
    _gguf_int8_bf16_prefix_full_attention_layers,
    _gguf_int8_effective_scale_dtype,
    _validate_gguf_int8_kv_context,
)
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


def test_gguf_int8_hybrid_prefill_uses_bf16_primary_when_layer_has_no_scale_metadata() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD
    oracle_key = _Buffer(0x5000, 32)
    oracle_value = _Buffer(0x6000, 32)
    retained_key = _Buffer(0x7000, 64)
    retained_value = _Buffer(0x8000, 64)
    session.scratch = type(
        "Scratch",
        (),
        {
            "full_cache": lambda self, layer_id: (retained_key, retained_value),
            "full_scale_metadata": lambda self, layer_id: None,
        },
    )()
    bulk = _BulkScratch(key_cache=oracle_key, value_cache=oracle_value, append_spans=_bf16_append_spans())

    layer_scratch = session._full_attention_prefill_scratch_for_layer(bulk, 3)

    assert layer_scratch.key_cache is retained_key
    assert layer_scratch.value_cache is retained_value
    assert layer_scratch.retained_key_cache is None
    assert layer_scratch.retained_value_cache is None
    assert layer_scratch.retained_append_spans is None


def test_gguf_int8_short_prefill_prefers_bf16_mirror_cache_when_available() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD
    oracle_key = _Buffer(0x5000, 32)
    oracle_value = _Buffer(0x6000, 32)
    mirror_key = _Buffer(0x6100, 64)
    mirror_value = _Buffer(0x6200, 64)
    retained_key = _Buffer(0x7000, 16)
    retained_value = _Buffer(0x8000, 16)
    metadata = _scale_metadata()
    session.scratch = type(
        "Scratch",
        (),
        {
            "full_cache": lambda self, layer_id: (retained_key, retained_value),
            "full_bf16_mirror_cache": lambda self, layer_id: (mirror_key, mirror_value),
            "full_scale_metadata": lambda self, layer_id: metadata,
        },
    )()
    bulk = _BulkScratch(key_cache=oracle_key, value_cache=oracle_value, append_spans=_bf16_append_spans())

    layer_scratch = session._full_attention_prefill_scratch_for_layer(bulk, 7)

    assert layer_scratch.key_cache is mirror_key
    assert layer_scratch.value_cache is mirror_value
    assert layer_scratch.retained_key_cache is retained_key
    assert layer_scratch.retained_value_cache is retained_value
    assert layer_scratch.retained_append_spans is not None
    assert layer_scratch.retained_append_spans.storage_dtype is DType.INT8_PER_TOKEN_HEAD
    assert layer_scratch.retained_append_spans.scale_metadata is metadata


def test_gguf_int8_context_guard_allows_short_mirror_without_env(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, raising=False)

    _validate_gguf_int8_kv_context(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=8192,
    )


def test_gguf_int8_context_guard_allows_long_hybrid_without_env(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, raising=False)

    _validate_gguf_int8_kv_context(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=8448,
        bf16_prefix_full_attention_layers=_GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS,
    )


def test_gguf_int8_context_guard_blocks_unverified_long_without_prefix_or_env(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, raising=False)

    try:
        _validate_gguf_int8_kv_context(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            max_positions=8448,
            bf16_prefix_full_attention_layers=0,
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected long pure GGUF INT8 KV context to be blocked")

    assert "BF16 full-attention prefix" in message
    assert "diagnostic-only" in message
    assert _GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV in message


def test_gguf_int8_long_hybrid_promotes_fp32_scales(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, raising=False)

    assert (
        _gguf_int8_effective_scale_dtype(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            max_positions=131328,
            requested_scale_dtype=DType.FP16,
            bf16_prefix_full_attention_layers=_GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS,
        )
        is DType.FP32
    )
    assert (
        _gguf_int8_effective_scale_dtype(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            max_positions=8192,
            requested_scale_dtype=DType.FP16,
            bf16_prefix_full_attention_layers=0,
        )
        is DType.FP16
    )


def test_gguf_int8_context_guard_allows_long_diagnostic_with_env(monkeypatch) -> None:
    monkeypatch.setenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, "1")

    _validate_gguf_int8_kv_context(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=131328,
    )


def test_gguf_int8_long_hybrid_prefix_default_and_env_override(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, raising=False)
    monkeypatch.delenv(_GGUF_INT8_BF16_PREFIX_FULL_ATTENTION_ENV, raising=False)

    assert (
        _gguf_int8_bf16_prefix_full_attention_layers(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            max_positions=131328,
        )
        == _GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS
    )

    monkeypatch.setenv(_GGUF_INT8_BF16_PREFIX_FULL_ATTENTION_ENV, "6")
    assert (
        _gguf_int8_bf16_prefix_full_attention_layers(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            max_positions=131328,
        )
        == 6
    )

    monkeypatch.setenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, "1")
    assert (
        _gguf_int8_bf16_prefix_full_attention_layers(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            max_positions=131328,
        )
        == 0
    )


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
