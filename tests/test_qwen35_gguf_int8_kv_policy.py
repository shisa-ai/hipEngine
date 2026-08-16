from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata
from hipengine.runtime.qwen35_gguf_runner import (
    _GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV,
    _GGUF_INT8_BF16_FULL_ATTENTION_LAYERS_ENV,
    _GGUF_INT8_BF16_PREFIX_FULL_ATTENTION_ENV,
    _GGUF_INT8_KV_BLOCK16_ENV,
    _GGUF_INT8_KV_KEY_ONLY_ENV,
    _GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS,
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen35GGUFResidentSession,
    _gguf_aotriton_head_major_buffers,
    _gguf_int8_bf16_full_attention_layer_indices,
    _gguf_int8_bf16_prefix_full_attention_layers,
    _gguf_int8_effective_scale_dtype,
    _gguf_int8_kv_append_write_fn,
    _gguf_int8_kv_decode_gate_fn,
    _gguf_int8_kv_prompt_write_fn,
    _gguf_int8_kv_scale_granularity,
    _gguf_int8_kv_value_bf16_enabled,
    _plan_gguf_int8_prefill_lifetime,
    _plan_gguf_int8_prefill_lifetime_for_session,
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
    prefill_spans: KVLiveSpans | None = None
    retained_key_cache: object | None = None
    retained_value_cache: object | None = None
    retained_append_spans: KVLiveSpans | None = None
    int8_kv_value_bf16: bool = False
    head_major_key_cache: object | None = None
    head_major_value_cache: object | None = None
    head_major_kv_capacity: int = 0
    head_major_kv_admitted: bool = False
    head_major_kv_dense_prefix: bool = True


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


def _block16_scale_metadata() -> KVScaleMetadata:
    return KVScaleMetadata(
        k_scale=_tensor(0x3000, (4, 256, 2, 16), DType.FP32),
        v_scale=_tensor(0x4000, (4, 256, 2, 16), DType.FP32),
        scale_dtype=DType.FP32,
        granularity="block16",
    )


def _hadamard_group32_scale_metadata() -> KVScaleMetadata:
    return KVScaleMetadata(
        k_scale=_tensor(0x3000, (4, 256, 2, 8), DType.FP16),
        v_scale=_tensor(0x4000, (4, 256, 2, 8), DType.FP16),
        scale_dtype=DType.FP16,
        granularity="hadamard_group32",
    )


def test_qwen38_pure_int8_prefill_plan_trades_layer_local_oracles_for_one_hidden_plane() -> None:
    plan = _plan_gguf_int8_prefill_lifetime(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=33_024,
        scratch_rows=4_096,
        hidden_size=5_120,
        head_count_kv=4,
        key_length=256,
        full_attention_layers=16,
        bf16_full_attention_layers=0,
        has_bf16_mirror=False,
        hidden_buffer_count=1,
    )

    assert plan.mode == "layer_outer_shared_oracle"
    assert plan.int8_full_attention_layers == 16
    assert plan.oracle_pair_bytes == 135_266_304
    assert plan.layer_local_oracle_bytes == 2_164_260_864
    assert plan.chunk_hidden_bytes == 41_943_040
    assert plan.layer_outer_hidden_bytes == 338_165_760
    assert plan.chunk_token_bytes == 32_768
    assert plan.layer_outer_token_bytes == 264_192
    assert plan.projected_peak_delta_bytes == -1_732_540_416
    assert plan.projected_peak_saving_bytes == 1_732_540_416
    assert plan.required_hidden_capacity == 33_024
    assert plan.oracle_buffer_count == 1


def test_qwen38_tail4_prefill_plan_still_saves_peak_with_shared_oracle() -> None:
    plan = _plan_gguf_int8_prefill_lifetime(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=33_024,
        scratch_rows=4_096,
        hidden_size=5_120,
        head_count_kv=4,
        key_length=256,
        full_attention_layers=16,
        bf16_full_attention_layers=12,
        has_bf16_mirror=False,
        hidden_buffer_count=1,
    )

    assert plan.mode == "layer_outer_shared_oracle"
    assert plan.int8_full_attention_layers == 4
    assert plan.projected_peak_delta_bytes == -109_344_768
    assert plan.projected_peak_saving_bytes == 109_344_768


def test_gguf_int8_prefill_plan_keeps_layer_local_route_when_full_hidden_costs_more() -> None:
    plan = _plan_gguf_int8_prefill_lifetime(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=131_328,
        scratch_rows=4_096,
        hidden_size=7_168,
        head_count_kv=2,
        key_length=256,
        full_attention_layers=10,
        bf16_full_attention_layers=8,
        has_bf16_mirror=False,
        hidden_buffer_count=1,
    )

    assert plan.mode == "chunk_outer_layer_local_oracles"
    assert plan.int8_full_attention_layers == 2
    assert plan.projected_peak_delta_bytes == 1_556_056_064
    assert plan.projected_peak_saving_bytes == 0
    assert plan.required_hidden_capacity == 4_096
    assert plan.oracle_buffer_count == 2


def test_gguf_int8_prefill_plan_preserves_short_bf16_mirror_route() -> None:
    plan = _plan_gguf_int8_prefill_lifetime(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=8_192,
        scratch_rows=4_096,
        hidden_size=5_120,
        head_count_kv=4,
        key_length=256,
        full_attention_layers=16,
        bf16_full_attention_layers=0,
        has_bf16_mirror=True,
        hidden_buffer_count=1,
    )

    assert plan.mode == "bf16_mirror"
    assert plan.required_hidden_capacity == 4_096
    assert plan.oracle_buffer_count == 0
    assert plan.projected_peak_delta_bytes == 0


def test_qualified_short_no_mirror_session_does_not_inherit_mirror_lifetime() -> None:
    session = SimpleNamespace(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        kv_scale_granularity="per_token_head",
        int8_kv_no_mirror_qualified=True,
        int8_bf16_full_attention_layer_indices=(),
        scratch=SimpleNamespace(max_positions=512),
        runner=SimpleNamespace(
            hidden_size=5_120,
            weights=SimpleNamespace(
                config=SimpleNamespace(
                    layer_types=(LINEAR_ATTENTION,) * 48 + (FULL_ATTENTION,) * 16,
                    head_count_kv=4,
                    key_length=256,
                )
            ),
        ),
    )

    plan = _plan_gguf_int8_prefill_lifetime_for_session(session, scratch_rows=512)

    assert plan.mode == "layer_outer_shared_oracle"
    assert plan.oracle_buffer_count == 1
    assert plan.required_hidden_capacity == 512


def test_qwen38_session_plan_selects_full_hidden_capacity_without_model_name_gate() -> None:
    session = SimpleNamespace(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        kv_scale_granularity="per_token_head",
        int8_bf16_full_attention_layer_indices=(),
        scratch=SimpleNamespace(max_positions=33_024),
        runner=SimpleNamespace(
            hidden_size=5_120,
            weights=SimpleNamespace(
                config=SimpleNamespace(
                    layer_types=(LINEAR_ATTENTION,) * 48 + (FULL_ATTENTION,) * 16,
                    head_count_kv=4,
                    key_length=256,
                )
            ),
        ),
    )

    plan = _plan_gguf_int8_prefill_lifetime_for_session(session, scratch_rows=4_096)

    assert plan.mode == "layer_outer_shared_oracle"
    assert plan.required_hidden_capacity == 33_024
    assert plan.oracle_buffer_count == 1


def test_qwen38_shared_oracle_plan_reuses_one_pair_across_layers(monkeypatch) -> None:
    allocations: list[_Buffer] = []

    def fake_malloc(nbytes: int, *, runtime) -> _Buffer:
        buffer = _Buffer(0x10_0000 + len(allocations) * 0x10_0000, int(nbytes))
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr("hipengine.runtime.qwen35_gguf_runner.malloc", fake_malloc)
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = SimpleNamespace()
    session.scratch = SimpleNamespace(max_positions=33_024)
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(config=SimpleNamespace(head_count_kv=4, key_length=256))
    )
    session._int8_prefill_oracle_buffers = {}
    session._int8_prefill_lifetime_plan = _plan_gguf_int8_prefill_lifetime(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=33_024,
        scratch_rows=4_096,
        hidden_size=5_120,
        head_count_kv=4,
        key_length=256,
        full_attention_layers=16,
        bf16_full_attention_layers=0,
        has_bf16_mirror=False,
        hidden_buffer_count=1,
    )

    first = session._int8_prefill_oracle_cache_for_layer(3)
    second = session._int8_prefill_oracle_cache_for_layer(7)

    assert first == second
    assert len(allocations) == 2
    assert len(session._int8_prefill_oracle_buffers) == 1


def test_layer_local_oracle_plan_keeps_distinct_pairs(monkeypatch) -> None:
    allocations: list[_Buffer] = []

    def fake_malloc(nbytes: int, *, runtime) -> _Buffer:
        buffer = _Buffer(0x10_0000 + len(allocations) * 0x10_0000, int(nbytes))
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr("hipengine.runtime.qwen35_gguf_runner.malloc", fake_malloc)
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = SimpleNamespace()
    session.scratch = SimpleNamespace(max_positions=131_328)
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(config=SimpleNamespace(head_count_kv=2, key_length=256))
    )
    session._int8_prefill_oracle_buffers = {}
    session._int8_prefill_lifetime_plan = _plan_gguf_int8_prefill_lifetime(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=131_328,
        scratch_rows=4_096,
        hidden_size=7_168,
        head_count_kv=2,
        key_length=256,
        full_attention_layers=10,
        bf16_full_attention_layers=8,
        has_bf16_mirror=False,
        hidden_buffer_count=1,
    )

    first = session._int8_prefill_oracle_cache_for_layer(3)
    second = session._int8_prefill_oracle_cache_for_layer(7)

    assert first != second
    assert len(allocations) == 4
    assert len(session._int8_prefill_oracle_buffers) == 2


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


def test_gguf_int8_full_attention_prefill_uses_layer_local_bf16_oracle_and_retained_int8_cache() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD
    session.int8_kv_value_bf16 = True
    shared_oracle_key = _Buffer(0x5000, 32)
    shared_oracle_value = _Buffer(0x6000, 32)
    layer_oracle_key = _Buffer(0x5100, 64)
    layer_oracle_value = _Buffer(0x6200, 64)
    retained_key = _Buffer(0x7000, 16)
    retained_value = _Buffer(0x8000, 16)
    head_major_key = _Buffer(0x9000, 128)
    head_major_value = _Buffer(0xA000, 128)
    metadata = _scale_metadata()
    session.scratch = type(
        "Scratch",
        (),
        {
            "full_cache": lambda self, layer_id: (retained_key, retained_value),
            "full_scale_metadata": lambda self, layer_id: metadata,
        },
    )()
    session._int8_prefill_oracle_cache_for_layer = lambda layer_id: (layer_oracle_key, layer_oracle_value)
    bulk = _BulkScratch(
        key_cache=shared_oracle_key,
        value_cache=shared_oracle_value,
        append_spans=_bf16_append_spans(),
        head_major_key_cache=head_major_key,
        head_major_value_cache=head_major_value,
        head_major_kv_capacity=64,
        head_major_kv_admitted=True,
    )

    layer_scratch = session._full_attention_prefill_scratch_for_layer(bulk, 7)

    assert layer_scratch.key_cache is layer_oracle_key
    assert layer_scratch.value_cache is layer_oracle_value
    assert layer_scratch.retained_key_cache is retained_key
    assert layer_scratch.retained_value_cache is retained_value
    assert layer_scratch.retained_append_spans is not None
    assert layer_scratch.retained_append_spans.storage_dtype is DType.INT8_PER_TOKEN_HEAD
    assert layer_scratch.retained_append_spans.scale_metadata is metadata
    assert layer_scratch.append_spans.storage_dtype is DType.BF16
    assert layer_scratch.int8_kv_value_bf16 is True
    assert _gguf_aotriton_head_major_buffers(layer_scratch, context_len=64) == (
        head_major_key,
        head_major_value,
        64,
    )


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


def test_gguf_tail4_hadamard_prefill_uses_bf16_attention_oracle_and_packed_retention() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD
    session.int8_kv_value_bf16 = False
    oracle_key = _Buffer(0x5100, 64)
    oracle_value = _Buffer(0x6200, 64)
    retained_key = _Buffer(0x7000, 16)
    retained_value = _Buffer(0x8000, 16)
    metadata = _hadamard_group32_scale_metadata()
    session.scratch = type(
        "Scratch",
        (),
        {
            "full_cache": lambda self, layer_id: (retained_key, retained_value),
            "full_bf16_mirror_cache": lambda self, layer_id: None,
            "full_scale_metadata": lambda self, layer_id: metadata,
        },
    )()
    session._int8_prefill_oracle_cache_for_layer = lambda layer_id: (oracle_key, oracle_value)
    bulk = _BulkScratch(key_cache=None, value_cache=None, append_spans=_bf16_append_spans())

    layer_scratch = session._full_attention_prefill_scratch_for_layer(bulk, 27)

    assert layer_scratch.key_cache is oracle_key
    assert layer_scratch.value_cache is oracle_value
    assert layer_scratch.retained_key_cache is retained_key
    assert layer_scratch.retained_value_cache is retained_value
    assert layer_scratch.append_spans.storage_dtype is DType.BF16
    assert layer_scratch.retained_append_spans is not None
    assert layer_scratch.retained_append_spans.storage_dtype is DType.INT8_PER_TOKEN_HEAD
    assert layer_scratch.retained_append_spans.scale_metadata is metadata


def test_gguf_int8_short_prefill_prefers_bf16_mirror_cache_when_available() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD
    session.int8_kv_value_bf16 = False
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
    assert layer_scratch.int8_kv_value_bf16 is False


def test_gguf_int8_key_only_env_is_diagnostic_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_KV_KEY_ONLY_ENV, raising=False)
    assert _gguf_int8_kv_value_bf16_enabled(kv_storage_dtype=DType.BF16) is False
    assert _gguf_int8_kv_value_bf16_enabled(kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD) is False

    monkeypatch.setenv(_GGUF_INT8_KV_KEY_ONLY_ENV, "1")
    assert _gguf_int8_kv_value_bf16_enabled(kv_storage_dtype=DType.BF16) is False
    assert _gguf_int8_kv_value_bf16_enabled(kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD) is True


def test_gguf_int8_block16_env_is_diagnostic_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_KV_BLOCK16_ENV, raising=False)
    assert (
        _gguf_int8_kv_scale_granularity(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            requested_granularity="per_token_head",
        )
        == "per_token_head"
    )
    assert (
        _gguf_int8_kv_scale_granularity(
            kv_storage_dtype=DType.BF16,
            requested_granularity="block16",
        )
        == "per_token_head"
    )

    monkeypatch.setenv(_GGUF_INT8_KV_BLOCK16_ENV, "1")
    assert (
        _gguf_int8_kv_scale_granularity(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            requested_granularity="per_token_head",
        )
        == "block16"
    )


def test_gguf_int8_block16_metadata_routes_to_block16_kernels() -> None:
    per_token = _scale_metadata()
    block16 = _block16_scale_metadata()
    hadamard = _hadamard_group32_scale_metadata()

    assert _gguf_int8_kv_prompt_write_fn(per_token).__name__.endswith("per_token_head_prompt_spans")
    assert _gguf_int8_kv_append_write_fn(per_token).__name__.endswith("per_token_head_spans")
    assert _gguf_int8_kv_decode_gate_fn(per_token).__name__.endswith("int8_gqa_splitk_gate_bf16_spans")
    assert _gguf_int8_kv_prompt_write_fn(block16).__name__.endswith("block16_prompt_spans")
    assert _gguf_int8_kv_append_write_fn(block16).__name__.endswith("block16_spans")
    assert _gguf_int8_kv_decode_gate_fn(block16).__name__.endswith("block16_gqa_splitk_gate_bf16_spans")
    assert _gguf_int8_kv_prompt_write_fn(hadamard).__name__.endswith("hadamard_group32_prompt_spans")
    assert _gguf_int8_kv_append_write_fn(hadamard).__name__.endswith("hadamard_group32_spans")
    assert _gguf_int8_kv_decode_gate_fn(hadamard).__name__.endswith(
        "hadamard_group32_gqa_splitk_gate_bf16_spans"
    )


def test_gguf_int8_block16_prefill_retained_spans_keep_scale_metadata() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD
    session.int8_kv_value_bf16 = False
    layer_oracle_key = _Buffer(0x5100, 64)
    layer_oracle_value = _Buffer(0x6200, 64)
    retained_key = _Buffer(0x7000, 16)
    retained_value = _Buffer(0x8000, 16)
    metadata = _block16_scale_metadata()
    session.scratch = type(
        "Scratch",
        (),
        {
            "full_cache": lambda self, layer_id: (retained_key, retained_value),
            "full_scale_metadata": lambda self, layer_id: metadata,
        },
    )()
    session._int8_prefill_oracle_cache_for_layer = lambda layer_id: (layer_oracle_key, layer_oracle_value)
    bulk = _BulkScratch(key_cache=None, value_cache=None, append_spans=_bf16_append_spans())

    layer_scratch = session._full_attention_prefill_scratch_for_layer(bulk, 7)

    assert layer_scratch.retained_append_spans is not None
    assert layer_scratch.retained_append_spans.storage_dtype is DType.INT8_PER_TOKEN_HEAD
    assert layer_scratch.retained_append_spans.scale_metadata is metadata
    assert layer_scratch.retained_append_spans.scale_metadata.granularity == "block16"


def test_gguf_tail4_hadamard_context_guard_allows_screened_long_layout(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, raising=False)

    _validate_gguf_int8_kv_context(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=262400,
        bf16_prefix_full_attention_layers=6,
        storage_layout="tail4_hadamard_group32",
    )


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

    assert "at least" in message
    assert "BF16-prefix full-attention layers" in message
    assert "diagnostic-only" in message
    assert _GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV in message


def test_gguf_int8_context_guard_blocks_too_small_long_prefix_without_env(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, raising=False)

    try:
        _validate_gguf_int8_kv_context(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            max_positions=131328,
            bf16_prefix_full_attention_layers=_GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS - 1,
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected low-prefix GGUF INT8 KV context to be blocked")

    assert "Prefixes below" in message
    assert _GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV in message


def test_gguf_hadamard_long_hybrid_keeps_requested_fp16_scales(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, raising=False)

    assert (
        _gguf_int8_effective_scale_dtype(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            max_positions=262400,
            requested_scale_dtype=DType.FP16,
            bf16_prefix_full_attention_layers=6,
            scale_granularity="hadamard_group32",
        )
        is DType.FP16
    )


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
        == 6
    )

    monkeypatch.delenv(_GGUF_INT8_BF16_PREFIX_FULL_ATTENTION_ENV, raising=False)
    assert (
        _gguf_int8_bf16_prefix_full_attention_layers(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            max_positions=131328,
        )
        == 0
    )


def test_gguf_int8_full_layer_set_defaults_to_prefix_and_allows_ranges(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, raising=False)
    monkeypatch.delenv(_GGUF_INT8_BF16_PREFIX_FULL_ATTENTION_ENV, raising=False)
    monkeypatch.delenv(_GGUF_INT8_BF16_FULL_ATTENTION_LAYERS_ENV, raising=False)

    assert _gguf_int8_bf16_full_attention_layer_indices(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=131328,
        full_attention_layers=10,
    ) == tuple(range(_GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS))

    monkeypatch.setenv(_GGUF_INT8_BF16_FULL_ATTENTION_LAYERS_ENV, "0-2,5,7")
    assert _gguf_int8_bf16_full_attention_layer_indices(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=131328,
        full_attention_layers=10,
    ) == (0, 1, 2, 5, 7)

    monkeypatch.setenv(_GGUF_INT8_BF16_FULL_ATTENTION_LAYERS_ENV, "none")
    assert _gguf_int8_bf16_full_attention_layer_indices(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=131328,
        full_attention_layers=10,
    ) == ()


def test_gguf_int8_context_guard_blocks_custom_full_layer_set_without_env(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, raising=False)

    try:
        _validate_gguf_int8_kv_context(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            max_positions=131328,
            bf16_prefix_full_attention_layers=_GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS,
            bf16_full_attention_layer_indices=(0, 1, 2, 5, 7, 8, 9),
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected custom GGUF INT8 KV layer set to be blocked")

    assert "custom BF16 full-attention layer sets" in message
    assert _GGUF_INT8_BF16_FULL_ATTENTION_LAYERS_ENV in message
    assert _GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV in message


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
            "kv_storage_layout": "tail4_hadamard_group32",
            "kv_scale_dtype": DType.FP16,
            "kv_scale_granularity": "block16",
        },
    )()

    breakdown = _decode_scratch_breakdown(scratch)

    assert breakdown["total_bytes"] == 231
    assert breakdown["allocation_mode"] == "dedicated"
    assert breakdown["physical_owner_count"] == 5
    assert breakdown["kv_storage_dtype"] == "int8_per_token_head"
    assert breakdown["kv_storage_layout"] == "tail4_hadamard_group32"
    assert breakdown["kv_scale_dtype"] == "fp16"
    assert breakdown["kv_scale_granularity"] == "block16"
    assert breakdown["by_component_bytes"]["full_attention_kv_cache"] == 200
    assert breakdown["by_component_bytes"]["full_attention_kv_scales"] == 24
    assert breakdown["by_component_bytes"]["full_attention_bf16_mirrors"] == 0
    assert breakdown["by_component_bytes"]["decode_workspace_other"] == 7
