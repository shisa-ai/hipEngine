"""Gemma 4 direct per-token/head INT8 decode consumer: registry/ABI and oracle.

The GPU consumer is exercised in
``tests/test_gpu_gemma4_int8_attention_consumer.py``. This module covers the two
things that do not need a GPU: the four-axis registration / span-ABI refusal
contract (which must fire before any build or launch), and the independent CPU
representation oracle's mask semantics on the quantized representation.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference import (
    gemma4_attention_decode_int8_per_token_head,
    quantize_kv_int8_per_token_head,
)
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata

_DEVICE = Device("hip", 0)
_NP_SCALE = {"fp32": np.float32, "fp16": np.float16}


def _tensor(
    ptr: int, shape: tuple[int, ...], dtype: DType, strides: tuple[int, ...] | None = None
) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, _DEVICE, strides=strides)


def _make_spans(
    *,
    base_shape: tuple[int, ...] = (2,),
    live_shape: tuple[int, ...] = (1,),
    block_size: int = 4,
    num_kv_heads: int = 2,
    scale_shape: tuple[int, ...] | None = None,
    scale_strides: tuple[int, ...] | None = None,
    base_strides: tuple[int, ...] | None = None,
    scale_dtype: str = "fp32",
    granularity: str = "per_token_head",
    storage_dtype=DType.INT8_PER_TOKEN_HEAD,
    live_dtype=DType.INT64,
    spans_mode: str = "uniform",
    with_scale_metadata: bool = True,
    token_positions: Tensor | None = None,
    evict_mask: Tensor | None = None,
    row_positions: Tensor | None = None,
) -> KVLiveSpans:
    if scale_shape is None:
        scale_shape = (2, block_size, num_kv_heads)
    metadata = (
        KVScaleMetadata(
            k_scale=_tensor(0x3000, scale_shape, scale_dtype, scale_strides),
            v_scale=_tensor(0x4000, scale_shape, scale_dtype, scale_strides),
            scale_dtype=scale_dtype,
            granularity=granularity,
        )
        if with_scale_metadata
        else None
    )
    return KVLiveSpans(
        base_offsets=_tensor(0x1000, base_shape, DType.INT32, base_strides),
        live_counts=_tensor(0x2000, live_shape, live_dtype),
        max_live_count=8,
        token_positions=token_positions,
        evict_mask=evict_mask,
        storage_dtype=storage_dtype,
        spans_mode=spans_mode,
        scale_metadata=metadata,
        row_positions=row_positions,
    )


# --------------------------------------------------------------------------
# CPU representation oracle
# --------------------------------------------------------------------------


def _paged_fixture(
    *,
    num_kv_heads: int,
    head_dim: int,
    context: int,
    block_size: int,
    seed: int,
    scale_dtype: str = "fp32",
):
    """Quantize random K/V rows and scatter them into a shuffled paged cache."""

    rng = np.random.default_rng(seed)
    key = rng.uniform(-4.0, 4.0, size=(context, num_kv_heads, head_dim)).astype(np.float32)
    value = rng.uniform(-4.0, 4.0, size=(context, num_kv_heads, head_dim)).astype(np.float32)
    key[0, 0] = 0.0  # zero row -> zero scale, zero payload
    value[0, 0] = 0.0
    qk, qv, ks, vs = quantize_kv_int8_per_token_head(
        key, value, scale_dtype=_NP_SCALE[scale_dtype]
    )
    blocks = (context + block_size - 1) // block_size + 1
    key_cache = np.full((blocks, block_size, num_kv_heads, head_dim), 127, np.int8)
    value_cache = np.full_like(key_cache, -128)
    k_scale = np.full((blocks, block_size, num_kv_heads), np.float32(3.0), _NP_SCALE[scale_dtype])
    v_scale = np.full_like(k_scale, np.float32(3.0))
    # Shuffled physical pages: logical block 0 -> physical 1, logical 1 -> physical 0.
    block_table = np.asarray(list(range(blocks))[::-1], dtype=np.int32)
    for slot in range(context):
        logical_block, offset = divmod(slot, block_size)
        physical = int(block_table[logical_block])
        key_cache[physical, offset] = qk[slot]
        value_cache[physical, offset] = qv[slot]
        k_scale[physical, offset] = ks[slot].astype(_NP_SCALE[scale_dtype])
        v_scale[physical, offset] = vs[slot].astype(_NP_SCALE[scale_dtype])
    query = rng.uniform(-2.0, 2.0, size=(num_kv_heads * 2, head_dim)).astype(np.float32)
    return query, key_cache, value_cache, k_scale, v_scale, block_table


def _manual_attention(
    query,
    key_cache,
    value_cache,
    k_scale,
    v_scale,
    block_table,
    context,
    *,
    block_size,
    scale=1.0,
    token_positions=None,
    evict_mask=None,
    row_position=None,
    sliding_window=None,
):
    """Independent float64 reference over the same quantized representation."""

    key = key_cache.astype(np.float64) * k_scale.astype(np.float64)[..., None]
    value = value_cache.astype(np.float64) * v_scale.astype(np.float64)[..., None]
    q = query.astype(np.float64)
    num_q_heads, head_dim = q.shape
    num_kv_heads = key.shape[2]
    kv_group = num_q_heads // num_kv_heads
    query_position = context - 1 if row_position is None else int(row_position)
    out = np.empty_like(q)
    for head in range(num_q_heads):
        kv_head = head // kv_group
        logits = []
        vectors = []
        for slot in range(context):
            logical_block, offset = divmod(slot, block_size)
            physical = int(block_table[logical_block])
            k = key[physical, offset, kv_head]
            v = value[physical, offset, kv_head]
            pos = slot if token_positions is None else int(token_positions[slot])
            visible = 0 <= pos <= query_position
            if sliding_window is not None:
                visible = visible and pos > query_position - sliding_window
            if evict_mask is not None:
                visible = visible and not bool(evict_mask[slot])
            logits.append(float(k @ q[head]) * scale if visible else -np.inf)
            vectors.append(v)
        logits = np.asarray(logits)
        if not np.isfinite(logits).any():
            out[head] = 0.0
            continue
        m = np.max(logits)
        w = np.exp(logits - m)
        w = w / w.sum()
        out[head] = sum(w[i] * vectors[i] for i in range(context))
    return out


@pytest.mark.parametrize("scale_dtype", ["fp32", "fp16"])
@pytest.mark.parametrize("num_kv_heads,head_dim", [(8, 4), (2, 8)])
def test_gemma4_int8_oracle_matches_independent_manual_reference(
    num_kv_heads: int, head_dim: int, scale_dtype: str
) -> None:
    query, key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        context=5,
        block_size=4,
        seed=0xD12 + num_kv_heads + head_dim,
        scale_dtype=scale_dtype,
    )
    got = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 5, block_size=4
    )
    want = _manual_attention(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 5, block_size=4
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_gemma4_int8_oracle_all_masked_returns_zeros() -> None:
    query, key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=2, head_dim=4, context=5, block_size=4, seed=0xA11
    )
    evict = np.ones(5, dtype=bool)
    got = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 5,
        block_size=4, evict_mask=evict,
    )
    np.testing.assert_array_equal(got, np.zeros_like(got))
    # A query position before every key masks the whole row too.
    got = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 5,
        block_size=4, row_position=-1,
    )
    np.testing.assert_array_equal(got, np.zeros_like(got))


def test_gemma4_int8_oracle_window_and_positions_exclude_keys() -> None:
    query, key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=2, head_dim=4, context=6, block_size=4, seed=0x1D0
    )
    positions = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    got = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 6,
        block_size=4, token_positions=positions, row_position=5, sliding_window=3,
    )
    want = _manual_attention(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 6,
        block_size=4, token_positions=positions, row_position=5, sliding_window=3,
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    # Only slots 3, 4, 5 are inside the window; 0, 1, 2 are excluded.
    unwindowed = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 6,
        block_size=4, token_positions=positions, row_position=5,
    )
    assert not np.allclose(got, unwindowed)


def test_gemma4_int8_oracle_zero_scale_row_is_well_defined() -> None:
    num_kv_heads, head_dim, context, block_size = 2, 4, 4, 4
    key = np.ones((context, num_kv_heads, head_dim), dtype=np.float32)
    value = np.ones((context, num_kv_heads, head_dim), dtype=np.float32)
    key[1] = 0.0
    value[1] = 0.0
    qk, qv, ks, vs = quantize_kv_int8_per_token_head(key, value)
    assert ks[1].max() == 0.0 and vs[1].max() == 0.0
    key_cache = np.zeros((1, block_size, num_kv_heads, head_dim), np.int8)
    value_cache = np.zeros_like(key_cache)
    k_scale = np.zeros((1, block_size, num_kv_heads), np.float32)
    v_scale = np.zeros_like(k_scale)
    key_cache[0], value_cache[0] = qk, qv
    k_scale[0], v_scale[0] = ks, vs
    query = np.ones((num_kv_heads * 2, head_dim), dtype=np.float32)
    table = np.asarray([0], dtype=np.int32)
    got = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, table, context, block_size=block_size
    )
    want = _manual_attention(
        query, key_cache, value_cache, k_scale, v_scale, table, context, block_size=block_size
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    # The zero-scale slot contributes no value, so every output sits strictly
    # below the unit reconstructed value it would take if the slot were absent.
    assert np.all((got > 0.0) & (got < 1.0))


def test_gemma4_int8_oracle_empty_span_returns_zeros() -> None:
    query = np.ones((4, 8), dtype=np.float32)
    key_cache = np.full((1, 1, 2, 8), 127, np.int8)
    value_cache = np.full_like(key_cache, -128)
    k_scale = np.full((1, 1, 2), 1.0, np.float32)
    v_scale = np.full_like(k_scale, 1.0)
    block_table = np.asarray([0], dtype=np.int32)
    got = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 0, block_size=1
    )
    np.testing.assert_array_equal(got, np.zeros_like(got))


def test_gemma4_int8_oracle_negative_span_is_refused() -> None:
    query = np.ones((4, 8), dtype=np.float32)
    key_cache = np.full((1, 1, 2, 8), 127, np.int8)
    value_cache = np.full_like(key_cache, -128)
    k_scale = np.full((1, 1, 2), 1.0, np.float32)
    v_scale = np.full_like(k_scale, 1.0)
    block_table = np.asarray([0], dtype=np.int32)
    with pytest.raises(ValueError, match="must not be negative"):
        gemma4_attention_decode_int8_per_token_head(
            query, key_cache, value_cache, k_scale, v_scale, block_table, -1, block_size=1
        )


def test_gemma4_int8_oracle_nonpositive_window_is_global() -> None:
    query, key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=2, head_dim=4, context=6, block_size=4, seed=0x91D
    )
    base = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 6,
        block_size=4, row_position=5,
    )
    zero = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 6,
        block_size=4, row_position=5, sliding_window=0,
    )
    negative = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 6,
        block_size=4, row_position=5, sliding_window=-5,
    )
    np.testing.assert_array_equal(zero.view(np.uint8), base.view(np.uint8))
    np.testing.assert_array_equal(negative.view(np.uint8), base.view(np.uint8))


def test_gemma4_int8_oracle_masked_poisoned_scale_does_not_leak() -> None:
    query, key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=2, head_dim=4, context=5, block_size=4, seed=0xA01
    )
    evict = np.zeros(5, dtype=bool)
    evict[0] = True  # slot 0 is masked
    clean = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 5,
        block_size=4, evict_mask=evict,
    )
    # Poison the masked slot's physical K/V scales with non-finite values. A
    # masked slot must not participate, so the result stays finite and equal.
    physical = int(block_table[0]) * 4
    bad_k = k_scale.copy()
    bad_v = v_scale.copy()
    bad_k[physical // 4, physical % 4, :] = np.inf
    bad_v[physical // 4, physical % 4, :] = np.nan
    with np.errstate(invalid="ignore", over="ignore"):
        poisoned = gemma4_attention_decode_int8_per_token_head(
            query, key_cache, value_cache, bad_k, bad_v, block_table, 5,
            block_size=4, evict_mask=evict,
        )
    assert np.all(np.isfinite(poisoned))
    np.testing.assert_array_equal(poisoned.view(np.uint8), clean.view(np.uint8))


def test_gemma4_int8_oracle_visible_nonfinite_logit_is_nan() -> None:
    query, key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=2, head_dim=4, context=5, block_size=4, seed=0xB02
    )
    physical = int(block_table[1]) * 4 + 0  # slot 4, visible
    bad_k = k_scale.copy()
    bad_k[physical // 4, physical % 4, :] = np.inf
    with np.errstate(invalid="ignore", over="ignore"):
        got = gemma4_attention_decode_int8_per_token_head(
            query, key_cache, value_cache, bad_k, v_scale, block_table, 5, block_size=4
        )
    # A visible non-finite logit is a failure, not a silently masked key.
    assert np.all(np.isnan(got))


# --------------------------------------------------------------------------
# Registry / span-ABI refusal (must fire before any build)
# --------------------------------------------------------------------------


def test_gemma4_int8_attention_registers_direct_key() -> None:
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    gemma4_attention_int8.register_gemma4_int8_attention_kernels(replace=True)
    key = KernelKey(
        "hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "gemma4_direct_spans"
    )
    assert is_registered(key)
    resolved = resolve(
        backend="hip_gfx1100",
        layer="paged_attn_decode",
        quant="int8_per_token_head",
        variant="gemma4_direct_spans",
    )
    assert resolved is gemma4_attention_int8.gemma4_attention_decode_int8_per_token_head_spans


def _no_build(monkeypatch) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8

    def _raise(**_kwargs):
        raise AssertionError("build must not be called before span validation")

    monkeypatch.setattr(gemma4_attention_int8, "build_gemma4_attention_int8", _raise)


def _call(spans, **kwargs) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8

    params = dict(block_size=4, num_q_heads=16, num_kv_heads=8, head_dim=256, max_context_len=4)
    params.update(kwargs)
    gemma4_attention_int8.gemma4_attention_decode_int8_per_token_head_spans(
        0, 0, 0, 0, spans, **params
    )


def test_gemma4_int8_attention_rejects_grouped_granularity_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8, 16), granularity="block16"
    )
    with pytest.raises(ValueError, match="per_token_head scale granularity"):
        _call(spans)


def test_gemma4_int8_attention_rejects_non_uniform_spans_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        base_shape=(1, 1, 2),
        live_shape=(1, 1, 2),
        block_size=4,
        num_kv_heads=8,
        scale_shape=(2, 4, 8),
        spans_mode="per_head_variable",
    )
    with pytest.raises(ValueError, match="uniform spans"):
        _call(spans)


def test_gemma4_int8_attention_rejects_wrong_storage_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        block_size=4,
        num_kv_heads=8,
        scale_shape=(2, 4, 8),
        storage_dtype=DType.BF16,
        with_scale_metadata=False,
    )
    with pytest.raises(ValueError, match="int8_per_token_head storage spans"):
        _call(spans)


def test_gemma4_int8_attention_rejects_unsupported_geometry_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(block_size=4, num_kv_heads=4, scale_shape=(2, 4, 4))
    with pytest.raises(ValueError, match="supports GQA geometries"):
        _call(spans, num_kv_heads=4, head_dim=128)


def test_gemma4_int8_attention_rejects_scale_shape_mismatch_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(block_size=4, num_kv_heads=8, scale_shape=(2, 3, 8))
    with pytest.raises(ValueError, match="scale shape must match"):
        _call(spans)


def test_gemma4_int8_attention_rejects_short_block_table_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        base_shape=(1,), block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8)
    )
    with pytest.raises(ValueError, match="block table is too short"):
        _call(spans, max_context_len=8)


def test_gemma4_int8_attention_rejects_oversized_context_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        base_shape=(4000,), block_size=4, num_kv_heads=2, scale_shape=(4000, 4, 2)
    )
    with pytest.raises(ValueError, match="shared memory"):
        _call(spans, max_context_len=16000, num_kv_heads=2, head_dim=512)


def test_gemma4_int8_attention_rejects_multi_row_live_counts_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        base_shape=(2,), live_shape=(2,), num_kv_heads=8, scale_shape=(2, 4, 8)
    )
    with pytest.raises(ValueError, match="live_counts must hold exactly one count"):
        _call(spans)


def test_gemma4_int8_attention_rejects_multi_row_row_positions_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    # KVLiveSpans already aligns row_positions to live_counts; this bypasses
    # that guard to prove the wrapper itself also refuses a multi-row position.
    spans = _make_spans(
        base_shape=(2,), num_kv_heads=8, scale_shape=(2, 4, 8)
    )
    object.__setattr__(spans, "row_positions", _tensor(0x5000, (2,), DType.INT64))
    with pytest.raises(ValueError, match="row_positions must hold exactly one position"):
        _call(spans)


def test_gemma4_int8_attention_rejects_2d_page_table_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        base_shape=(2, 1), num_kv_heads=8, scale_shape=(2, 4, 8)
    )
    with pytest.raises(ValueError, match="1-D single-row base_offsets"):
        _call(spans)


def test_gemma4_int8_attention_rejects_non_contiguous_scale_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        base_shape=(2,),
        num_kv_heads=8,
        scale_shape=(2, 4, 8),
        scale_strides=(64, 16, 1),  # contiguous would be (32, 8, 1)
    )
    with pytest.raises(ValueError, match="contiguous k_scale"):
        _call(spans)


def _patch_readback(monkeypatch, live_count: int, physical: list[int]) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8

    def _fake(_spans, _used_blocks, _runtime, _stream=0):
        return live_count, None, physical

    monkeypatch.setattr(gemma4_attention_int8, "_read_device_metadata", _fake)


def test_gemma4_int8_attention_rejects_out_of_range_physical_page(monkeypatch) -> None:
    _no_build(monkeypatch)
    _patch_readback(monkeypatch, live_count=4, physical=[5])  # cache has 2 blocks
    spans = _make_spans(base_shape=(2,), num_kv_heads=8, scale_shape=(2, 4, 8))
    with pytest.raises(ValueError, match="outside \\[0, 2\\)"):
        _call(spans, runtime=object())


def test_gemma4_int8_attention_rejects_negative_physical_page(monkeypatch) -> None:
    _no_build(monkeypatch)
    _patch_readback(monkeypatch, live_count=4, physical=[-1])
    spans = _make_spans(base_shape=(2,), num_kv_heads=8, scale_shape=(2, 4, 8))
    with pytest.raises(ValueError, match="outside \\[0, 2\\)"):
        _call(spans, runtime=object())


def test_gemma4_int8_attention_rejects_over_capacity_live_count(monkeypatch) -> None:
    _no_build(monkeypatch)
    _patch_readback(monkeypatch, live_count=5, physical=[0])  # max_context_len=4
    spans = _make_spans(base_shape=(2,), num_kv_heads=8, scale_shape=(2, 4, 8))
    with pytest.raises(ValueError, match="exceeds max_context_len"):
        _call(spans, runtime=object())


def test_gemma4_int8_attention_rejects_negative_live_count(monkeypatch) -> None:
    _no_build(monkeypatch)
    _patch_readback(monkeypatch, live_count=-1, physical=[0])
    spans = _make_spans(base_shape=(2,), num_kv_heads=8, scale_shape=(2, 4, 8))
    with pytest.raises(ValueError, match="is negative"):
        _call(spans, runtime=object())


def test_gemma4_int8_attention_readback_orders_against_supplied_stream() -> None:
    """A supplied stream must be synchronized before the synchronous D2H readback.

    ``hipMemcpy`` is not ordered against a non-blocking producer stream, so the
    checked path must sync it first. This is the unit RED for that ordering.
    """

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8 as mod

    calls: list[tuple[str, int]] = []

    class _FakeRuntime:
        def stream_synchronize(self, stream: int) -> None:
            calls.append(("sync", int(stream)))

        def memcpy(self, dst, src, nbytes, kind) -> None:
            calls.append(("memcpy", int(src)))

    spans = _make_spans(base_shape=(2,), num_kv_heads=8, scale_shape=(2, 4, 8))
    mod._read_device_metadata(spans, 1, _FakeRuntime(), 7)
    assert calls[0] == ("sync", 7), calls
    assert calls[1][0] == "memcpy", calls
    assert [name for name, _ in calls].count("sync") == 1, calls

    # No supplied stream -> no synchronization, only the synchronous readback.
    calls.clear()
    mod._read_device_metadata(spans, 1, _FakeRuntime(), 0)
    assert calls and all(name == "memcpy" for name, _ in calls), calls


def test_gemma4_int8_oracle_visible_nonfinite_v_is_nan_independent_of_weight() -> None:
    num_kv_heads, head_dim, context, block_size = 1, 2, 2, 2
    query = np.ones((1, head_dim), dtype=np.float32)
    key_cache = np.zeros((1, block_size, num_kv_heads, head_dim), np.int8)
    value_cache = np.zeros_like(key_cache)
    k_scale = np.ones((1, block_size, num_kv_heads), np.float32)
    v_scale = np.ones_like(k_scale)
    table = np.asarray([0], dtype=np.int32)
    value_cache[0, 0, 0, :] = 1
    value_cache[0, 1, 0, :] = 1

    # Positive weight (both logits zero): a visible inf V must be NaN, not inf.
    bad_v = v_scale.copy()
    bad_v[0, 0, 0] = np.inf
    with np.errstate(invalid="ignore", over="ignore"):
        got = gemma4_attention_decode_int8_per_token_head(
            query, key_cache, value_cache, k_scale, bad_v, table, context,
            block_size=block_size,
        )
    assert np.all(np.isnan(got)), "positive-weight visible inf V must be NaN"

    # Underflow-zero weight: slot 1 logit ~ -1000, still a visible failure.
    key_cache[0, 1, 0, :] = 127
    k_scale[0, 1, 0] = np.float32(-1000.0 / (127 * head_dim))
    for poison in (np.inf, np.nan):
        bad_v = v_scale.copy()
        bad_v[0, 1, 0] = poison
        with np.errstate(invalid="ignore", over="ignore"):
            got = gemma4_attention_decode_int8_per_token_head(
                query, key_cache, value_cache, k_scale, bad_v, table, context,
                block_size=block_size,
            )
        assert np.all(np.isnan(got)), poison

    # Masked V poison stays excluded and finite.
    evict = np.ones(context, dtype=bool)
    bad_v = v_scale.copy()
    bad_v[0, 0, 0] = np.inf
    with np.errstate(invalid="ignore", over="ignore"):
        got = gemma4_attention_decode_int8_per_token_head(
            query, key_cache, value_cache, k_scale, bad_v, table, context,
            block_size=block_size, evict_mask=evict,
        )
    np.testing.assert_array_equal(got, np.zeros_like(got))
