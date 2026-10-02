"""Gemma 4 direct per-token/head INT8 prefill consumer: registry/ABI and oracle.

The GPU prefill consumer is exercised in
``tests/test_gpu_gemma4_int8_attention_prefill_consumer.py``. This module covers
the two things that do not need a GPU: the four-axis registration / span-ABI
refusal contract (which must fire before any build or launch), and the
independent CPU representation oracle's multi-row semantics on the quantized
representation.

Declared multi-row semantics under test: every query row attends over ONE
SHARED LOGICAL KV PREFIX through a single 1-D page table; a per-row (row-major)
page table is a separate prefix and is refused before build.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference import (
    gemma4_attention_prefill_int8_per_token_head,
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
    rows: int = 3,
    base_shape: tuple[int, ...] = (2,),
    live_shape: tuple[int, ...] | None = None,
    block_size: int = 4,
    num_kv_heads: int = 8,
    scale_shape: tuple[int, ...] | None = None,
    scale_strides: tuple[int, ...] | None = None,
    base_strides: tuple[int, ...] | None = None,
    scale_dtype: str = "fp32",
    granularity: str = "per_token_head",
    storage_dtype=DType.INT8_PER_TOKEN_HEAD,
    live_dtype=DType.INT64,
    spans_mode: str = "uniform",
    with_scale_metadata: bool = True,
    with_row_positions: bool = True,
    row_shape: tuple[int, ...] | None = None,
    token_positions: Tensor | None = None,
    evict_mask: Tensor | None = None,
) -> KVLiveSpans:
    if live_shape is None:
        live_shape = (rows,)
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
    if with_row_positions:
        row_positions = _tensor(0x5000, row_shape if row_shape is not None else (rows,), DType.INT64)
    else:
        row_positions = None
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
    return key_cache, value_cache, k_scale, v_scale, block_table


def _manual_prefill(
    query,
    key_cache,
    value_cache,
    k_scale,
    v_scale,
    block_table,
    live_counts,
    *,
    block_size,
    row_positions=None,
    scale=1.0,
    token_positions=None,
    evict_mask=None,
    sliding_window=None,
):
    """Independent float64 per-row reference over the same quantized representation."""

    key = key_cache.astype(np.float64) * k_scale.astype(np.float64)[..., None]
    value = value_cache.astype(np.float64) * v_scale.astype(np.float64)[..., None]
    q = query.astype(np.float64)
    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = key.shape[2]
    kv_group = num_q_heads // num_kv_heads
    out = np.empty_like(q)
    for row in range(rows):
        context = int(live_counts[row])
        query_position = context - 1 if row_positions is None else int(row_positions[row])
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
                logits.append(float(k @ q[row, head]) * scale if visible else -np.inf)
                vectors.append(v)
            logits = np.asarray(logits)
            if not np.isfinite(logits).any():
                out[row, head] = 0.0
                continue
            m = np.max(logits)
            w = np.exp(logits - m)
            w = w / w.sum()
            out[row, head] = sum(w[i] * vectors[i] for i in range(context))
    return out


@pytest.mark.parametrize("scale_dtype", ["fp32", "fp16"])
@pytest.mark.parametrize("num_kv_heads,head_dim", [(8, 4), (2, 8)])
@pytest.mark.parametrize("rows", [1, 3, 7])
def test_gemma4_int8_prefill_oracle_matches_independent_manual_reference(
    num_kv_heads: int, head_dim: int, scale_dtype: str, rows: int
) -> None:
    context = 7
    key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        context=context,
        block_size=4,
        seed=0xD12 + num_kv_heads + head_dim + rows,
        scale_dtype=scale_dtype,
    )
    rng = np.random.default_rng(0x900 + rows)
    query = rng.uniform(-2.0, 2.0, size=(rows, num_kv_heads * 2, head_dim)).astype(np.float32)
    live_counts = np.asarray([context] * rows, dtype=np.int64)
    row_positions = np.asarray([context - 1] * rows, dtype=np.int64)
    got = gemma4_attention_prefill_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, live_counts,
        block_size=4, row_positions=row_positions,
    )
    want = _manual_prefill(
        query, key_cache, value_cache, k_scale, v_scale, block_table, live_counts,
        block_size=4, row_positions=row_positions,
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert got.shape == (rows, num_kv_heads * 2, head_dim)


def test_gemma4_int8_prefill_oracle_distinct_row_positions_and_prefixes() -> None:
    context = 8
    key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=2, head_dim=4, context=context, block_size=4, seed=0x5EED
    )
    query = np.random.default_rng(3).uniform(
        -2.0, 2.0, size=(3, 4, 4)
    ).astype(np.float32)
    live_counts = np.asarray([8, 5, 3], dtype=np.int64)
    row_positions = np.asarray([7, 2, 0], dtype=np.int64)
    got = gemma4_attention_prefill_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, live_counts,
        block_size=4, row_positions=row_positions,
    )
    want = _manual_prefill(
        query, key_cache, value_cache, k_scale, v_scale, block_table, live_counts,
        block_size=4, row_positions=row_positions,
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    # Distinct row positions produce distinct rows, not a broadcast.
    assert not np.allclose(got[0], got[1])
    assert not np.allclose(got[1], got[2])


def test_gemma4_int8_prefill_oracle_per_row_empty_and_all_masked() -> None:
    context = 6
    key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=2, head_dim=4, context=context, block_size=4, seed=0xE0
    )
    query = np.ones((4, 4, 4), dtype=np.float32)
    # Row 0 empty, row 1 all-masked (position before every slot), rows 2/3 normal.
    live_counts = np.asarray([0, 6, 6, 6], dtype=np.int64)
    row_positions = np.asarray([-1, -1, 5, 5], dtype=np.int64)
    got = gemma4_attention_prefill_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, live_counts,
        block_size=4, row_positions=row_positions,
    )
    np.testing.assert_array_equal(got[0], np.zeros_like(got[0]))
    np.testing.assert_array_equal(got[1], np.zeros_like(got[1]))
    assert np.any(got[2] != 0.0) and np.any(got[3] != 0.0)
    want = _manual_prefill(
        query, key_cache, value_cache, k_scale, v_scale, block_table, live_counts,
        block_size=4, row_positions=row_positions,
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_gemma4_int8_prefill_oracle_window_positions_and_eviction() -> None:
    context = 8
    key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=2, head_dim=4, context=context, block_size=4, seed=0x1D0
    )
    query = np.random.default_rng(11).uniform(-2.0, 2.0, size=(2, 4, 4)).astype(np.float32)
    positions = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    evict = np.zeros(8, dtype=bool)
    evict[2] = True
    live_counts = np.asarray([8, 6], dtype=np.int64)
    row_positions = np.asarray([7, 5], dtype=np.int64)
    got = gemma4_attention_prefill_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, live_counts,
        block_size=4, row_positions=row_positions, token_positions=positions,
        evict_mask=evict, sliding_window=4,
    )
    want = _manual_prefill(
        query, key_cache, value_cache, k_scale, v_scale, block_table, live_counts,
        block_size=4, row_positions=row_positions, token_positions=positions,
        evict_mask=evict, sliding_window=4,
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_gemma4_int8_prefill_oracle_masked_poisoned_scale_does_not_leak() -> None:
    context = 6
    key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=2, head_dim=4, context=context, block_size=4, seed=0xA01
    )
    query = np.ones((2, 4, 4), dtype=np.float32)
    live_counts = np.asarray([6, 6], dtype=np.int64)
    row_positions = np.asarray([5, 5], dtype=np.int64)
    evict = np.zeros(6, dtype=bool)
    evict[0] = True
    clean = gemma4_attention_prefill_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, live_counts,
        block_size=4, row_positions=row_positions, evict_mask=evict,
    )
    physical = int(block_table[0]) * 4
    bad_k = k_scale.copy()
    bad_v = v_scale.copy()
    bad_k[physical // 4, physical % 4, :] = np.inf
    bad_v[physical // 4, physical % 4, :] = np.nan
    with np.errstate(invalid="ignore", over="ignore"):
        poisoned = gemma4_attention_prefill_int8_per_token_head(
            query, key_cache, value_cache, bad_k, bad_v, block_table, live_counts,
            block_size=4, row_positions=row_positions, evict_mask=evict,
        )
    assert np.all(np.isfinite(poisoned))
    np.testing.assert_array_equal(poisoned.view(np.uint8), clean.view(np.uint8))


def test_gemma4_int8_prefill_oracle_visible_nonfinite_is_head_local_nan() -> None:
    context = 4
    key_cache, value_cache, k_scale, v_scale, block_table = _paged_fixture(
        num_kv_heads=2, head_dim=4, context=context, block_size=4, seed=0xB02
    )
    query = np.ones((2, 4, 4), dtype=np.float32)
    live_counts = np.asarray([4, 4], dtype=np.int64)
    row_positions = np.asarray([3, 3], dtype=np.int64)
    # Poison the visible K scale of KV head 0 (physical slot 0), so query heads
    # 0..1 (GQA group 2) fail and heads 2..3 stay finite, in both rows.
    physical = int(block_table[0]) * 4
    bad_k = k_scale.copy()
    bad_k[physical // 4, physical % 4, 0] = np.inf
    with np.errstate(invalid="ignore", over="ignore"):
        got = gemma4_attention_prefill_int8_per_token_head(
            query, key_cache, value_cache, bad_k, v_scale, block_table, live_counts,
            block_size=4, row_positions=row_positions,
        )
    assert np.all(np.isnan(got[:, :2]))
    assert np.all(np.isfinite(got[:, 2:]))


def test_gemma4_int8_prefill_oracle_negative_count_is_refused() -> None:
    query = np.ones((2, 4, 8), dtype=np.float32)
    key_cache = np.full((1, 1, 2, 8), 127, np.int8)
    value_cache = np.full_like(key_cache, -128)
    k_scale = np.full((1, 1, 2), 1.0, np.float32)
    v_scale = np.full_like(k_scale, 1.0)
    block_table = np.asarray([0], dtype=np.int32)
    with pytest.raises(ValueError, match="must not be negative"):
        gemma4_attention_prefill_int8_per_token_head(
            query, key_cache, value_cache, k_scale, v_scale, block_table,
            np.asarray([1, -1], dtype=np.int64), block_size=1,
            row_positions=np.asarray([0, 0], dtype=np.int64),
        )


def test_gemma4_int8_prefill_oracle_row_count_must_match_rows() -> None:
    query = np.ones((3, 4, 8), dtype=np.float32)
    key_cache = np.full((1, 1, 2, 8), 127, np.int8)
    value_cache = np.full_like(key_cache, -128)
    k_scale = np.full((1, 1, 2), 1.0, np.float32)
    v_scale = np.full_like(k_scale, 1.0)
    block_table = np.asarray([0], dtype=np.int32)
    with pytest.raises(ValueError, match="one entry per query row"):
        gemma4_attention_prefill_int8_per_token_head(
            query, key_cache, value_cache, k_scale, v_scale, block_table,
            np.asarray([1, 1], dtype=np.int64), block_size=1,
            row_positions=np.asarray([0, 0, 0], dtype=np.int64),
        )


# --------------------------------------------------------------------------
# Registry / span-ABI refusal (must fire before any build)
# --------------------------------------------------------------------------


def test_gemma4_int8_prefill_registers_shared_prefix_key() -> None:
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    gemma4_attention_int8.register_gemma4_int8_attention_kernels(replace=True)
    key = KernelKey(
        "hip_gfx1100", "paged_attn_prefill", "int8_per_token_head", "gemma4_direct_spans"
    )
    assert is_registered(key)
    resolved = resolve(
        backend="hip_gfx1100",
        layer="paged_attn_prefill",
        quant="int8_per_token_head",
        variant="gemma4_direct_spans",
    )
    assert resolved is gemma4_attention_int8.gemma4_attention_prefill_int8_per_token_head_spans


def _no_build(monkeypatch) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8

    def _raise(**_kwargs):
        raise AssertionError("build must not be called before span validation")

    monkeypatch.setattr(gemma4_attention_int8, "build_gemma4_attention_int8", _raise)


def _no_metadata(monkeypatch) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8

    def _raise(*_args, **_kwargs):
        raise AssertionError("device metadata must not be read before the row bound")

    monkeypatch.setattr(gemma4_attention_int8, "_read_device_metadata_prefill", _raise)


def _call(spans, **kwargs) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8

    params = dict(
        rows=3, max_context_len=4, block_size=4, num_q_heads=16, num_kv_heads=8, head_dim=256
    )
    params.update(kwargs)
    gemma4_attention_int8.gemma4_attention_prefill_int8_per_token_head_spans(
        0, 0, 0, 0, spans, **params
    )


def test_gemma4_int8_prefill_rejects_grouped_granularity_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8, 16), granularity="block16"
    )
    with pytest.raises(ValueError, match="per_token_head scale granularity"):
        _call(spans)


def test_gemma4_int8_prefill_rejects_non_uniform_spans_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        base_shape=(1, 1, 2),
        live_shape=(1, 1, 2),
        row_shape=(1,),
        block_size=4,
        num_kv_heads=8,
        scale_shape=(2, 4, 8),
        spans_mode="per_head_variable",
    )
    with pytest.raises(ValueError, match="uniform spans"):
        _call(spans)


def test_gemma4_int8_prefill_rejects_wrong_storage_before_build(monkeypatch) -> None:
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


def test_gemma4_int8_prefill_rejects_unsupported_geometry_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(block_size=4, num_kv_heads=4, scale_shape=(2, 4, 4))
    with pytest.raises(ValueError, match="supports GQA geometries"):
        _call(spans, num_kv_heads=4, head_dim=128)


def test_gemma4_int8_prefill_rejects_scale_shape_mismatch_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(block_size=4, num_kv_heads=8, scale_shape=(2, 3, 8))
    with pytest.raises(ValueError, match="scale shape must match"):
        _call(spans)


def test_gemma4_int8_prefill_rejects_short_block_table_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(base_shape=(1,), block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8))
    with pytest.raises(ValueError, match="block table is too short"):
        _call(spans, max_context_len=8)


def test_gemma4_int8_prefill_rejects_oversized_context_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        base_shape=(4000,), block_size=4, num_kv_heads=2, scale_shape=(4000, 4, 2)
    )
    with pytest.raises(ValueError, match="shared memory"):
        _call(spans, max_context_len=16000, num_kv_heads=2, head_dim=512)


def test_gemma4_int8_prefill_rejects_oversized_rows_before_metadata_and_build(
    monkeypatch,
) -> None:
    """A row count past the HIP grid.y maximum must be refused before any device work.

    ``2**32 + 1`` would wrap to ``1`` under the launcher's ``unsigned int`` cast,
    silently launching one row. The refusal must fire before the device metadata
    readback and before any build.
    """

    _no_build(monkeypatch)
    _no_metadata(monkeypatch)
    spans = _make_spans(block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8))
    with pytest.raises(ValueError, match="at most 65535 query rows"):
        _call(spans, rows=2**32 + 1)


def test_gemma4_int8_prefill_allows_max_row_boundary(monkeypatch) -> None:
    """The 65535-row boundary is inclusive: it reaches metadata, not the bound refusal."""

    _no_build(monkeypatch)
    _no_metadata(monkeypatch)
    spans = _make_spans(
        rows=65535, live_shape=(65535,), row_shape=(65535,), block_size=4,
        num_kv_heads=8, scale_shape=(2, 4, 8),
    )
    with pytest.raises(AssertionError, match="device metadata must not be read"):
        _call(spans, rows=65535, max_context_len=4)


def test_gemma4_int8_prefill_rejects_separate_per_row_prefix_before_build(monkeypatch) -> None:
    """A row-major (per-row) page table is a separate prefix and must be refused."""

    _no_build(monkeypatch)
    spans = _make_spans(
        base_shape=(3, 2), block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8)
    )
    with pytest.raises(ValueError, match="shared 1-D base_offsets"):
        _call(spans)


def test_gemma4_int8_prefill_rejects_live_count_count_mismatch_before_build(monkeypatch) -> None:
    """A single-row decode span (one count) must not be read as a 3-row prefill."""

    _no_build(monkeypatch)
    spans = _make_spans(
        rows=3, live_shape=(1,), row_shape=(1,), block_size=4, num_kv_heads=8,
        scale_shape=(2, 4, 8),
    )
    with pytest.raises(ValueError, match="one live count per query row"):
        _call(spans)


def test_gemma4_int8_prefill_rejects_missing_row_positions_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        rows=3, block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8),
        with_row_positions=False,
    )
    with pytest.raises(ValueError, match="requires explicit row_positions"):
        _call(spans)


def test_gemma4_int8_prefill_rejects_row_position_count_mismatch_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    # KVLiveSpans aligns row_positions to live_counts, so bypass it to prove the
    # wrapper itself also refuses a mismatched per-row position count.
    spans = _make_spans(rows=3, block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8))
    object.__setattr__(spans, "row_positions", _tensor(0x5000, (2,), DType.INT64))
    with pytest.raises(ValueError, match="one row position per query row"):
        _call(spans)


def test_gemma4_int8_prefill_rejects_non_contiguous_scale_before_build(monkeypatch) -> None:
    _no_build(monkeypatch)
    spans = _make_spans(
        block_size=4,
        num_kv_heads=8,
        scale_shape=(2, 4, 8),
        scale_strides=(64, 16, 1),  # contiguous would be (32, 8, 1)
    )
    with pytest.raises(ValueError, match="contiguous k_scale"):
        _call(spans)


def _patch_readback(monkeypatch, counts: list[int], positions: list[int], physical: list[int]) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8

    def _fake(_spans, _rows, _used_blocks, _runtime, _stream=0):
        return counts, positions, physical

    monkeypatch.setattr(gemma4_attention_int8, "_read_device_metadata_prefill", _fake)


def test_gemma4_int8_prefill_rejects_out_of_range_shared_page(monkeypatch) -> None:
    _no_build(monkeypatch)
    _patch_readback(monkeypatch, counts=[4, 4, 4], positions=[3, 3, 3], physical=[5])
    spans = _make_spans(block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8))
    with pytest.raises(ValueError, match="outside \\[0, 2\\)"):
        _call(spans, runtime=object())


def test_gemma4_int8_prefill_rejects_negative_per_row_count(monkeypatch) -> None:
    _no_build(monkeypatch)
    _patch_readback(monkeypatch, counts=[4, -1, 4], positions=[3, 0, 3], physical=[0])
    spans = _make_spans(block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8))
    with pytest.raises(ValueError, match="live_counts\\[1\\]=-1 is negative"):
        _call(spans, runtime=object())


def test_gemma4_int8_prefill_rejects_over_capacity_per_row_count(monkeypatch) -> None:
    _no_build(monkeypatch)
    _patch_readback(monkeypatch, counts=[4, 5, 4], positions=[3, 4, 3], physical=[0])
    spans = _make_spans(block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8))
    with pytest.raises(ValueError, match="live_counts\\[1\\]=5 exceeds max_context_len"):
        _call(spans, runtime=object())


def test_gemma4_int8_prefill_readback_orders_against_supplied_stream() -> None:
    """A supplied stream must be synchronized before the synchronous D2H readback."""

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8 as mod

    calls: list[tuple[str, int]] = []

    class _FakeRuntime:
        def stream_synchronize(self, stream: int) -> None:
            calls.append(("sync", int(stream)))

        def memcpy(self, dst, src, nbytes, kind) -> None:
            calls.append(("memcpy", int(src)))

    spans = _make_spans(block_size=4, num_kv_heads=8, scale_shape=(2, 4, 8))
    mod._read_device_metadata_prefill(spans, 3, 1, _FakeRuntime(), 7)
    assert calls[0] == ("sync", 7), calls
    assert calls[1][0] == "memcpy", calls
    assert [name for name, _ in calls].count("sync") == 1, calls

    calls.clear()
    mod._read_device_metadata_prefill(spans, 3, 1, _FakeRuntime(), 0)
    assert calls and all(name == "memcpy" for name, _ in calls), calls
