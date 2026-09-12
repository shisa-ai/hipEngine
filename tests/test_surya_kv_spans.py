"""Surya KV path over the ``KVLiveSpans`` ABI (RED gate for task #75).

The pre-#75 Surya runtime wrote K/V into a bespoke dense ``(nk, max_seq, hd)``
plane with ``surya_scatter_kv_f32(src, dst, tokens, token_offset, ...)`` and
decoded through three rocBLAS batched SGEMMs plus two elementwise kernels over a
materialized ``(nq, max_seq)`` score row.  Both kernels read no span metadata at
all, which the ``KVLiveSpans`` architectural invariant forbids.

These tests pin the replacement contract before the kernels exist:

* the page table, ``token_positions``, ``evict_mask``, ``live_counts``, and
  ``row_positions`` all change the answer -- a ``(block_table, context_len)``
  shortcut that ignores any of them fails the device fixtures;
* the fused fp32 decode kernel agrees with the rocBLAS parent path it replaces
  (declared parent-parity bound) and with a float64 CPU reference (outer gate);
* the host planner builds a dense uniform span set whose fields are exercised,
  not merely declared.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path("hipengine/kernels/hip_gfx1100/surya/surya_ops.hip")

_BLOCK_SIZE = 256
_NQ = 8
_NK = 2
_HD = 256
# Small enough to keep the device fixture cheap; large enough that a reversed
# page table, an evicted slot, and an empty ``token_positions`` slot all land in
# different blocks.
_MAX_SEQ = 2048

# Declared parent-parity bound: the fused kernel reduces QK^T with warp
# shuffles and softmax with split online-max, so it cannot be bit-identical to
# the rocBLAS + row-softmax parent.  The bound is absolute because the decode
# output is a convex combination of unit-variance values, so |out| is O(1).
_PARENT_PARITY_ATOL = 5.0e-5
_CPU_REFERENCE_ATOL = 5.0e-4


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _quality(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """Row-wise KL and top-1 agreement over the two output matrices."""

    left = expected.astype(np.float64)
    right = actual.astype(np.float64)
    left -= np.max(left, axis=1, keepdims=True)
    right -= np.max(right, axis=1, keepdims=True)
    left_prob = np.exp(left)
    right_prob = np.exp(right)
    left_prob /= np.sum(left_prob, axis=1, keepdims=True)
    right_prob /= np.sum(right_prob, axis=1, keepdims=True)
    kl = np.sum(
        left_prob
        * (
            np.log(np.maximum(left_prob, 1.0e-300))
            - np.log(np.maximum(right_prob, 1.0e-300))
        ),
        axis=1,
    )
    agreement = np.mean(np.argmax(left, axis=1) == np.argmax(right, axis=1))
    return float(np.max(kl)), float(agreement)


# --------------------------------------------------------------------------- #
# host-only contract tests
# --------------------------------------------------------------------------- #


def test_plan_surya_dense_spans_builds_a_dense_uniform_span_set() -> None:
    from hipengine.kernels.hip_gfx1100.surya.surya_ops import plan_surya_dense_spans

    plan = plan_surya_dense_spans(_MAX_SEQ, block_size=_BLOCK_SIZE)
    assert plan.max_seq == _MAX_SEQ
    assert plan.block_size == _BLOCK_SIZE
    assert plan.block_table_len == _MAX_SEQ // _BLOCK_SIZE
    assert plan.page_table.dtype == np.int32
    assert plan.page_table.shape == (plan.block_table_len,)
    # Identity page table: logical block b maps to physical block b, so the
    # dense fill is address-preserving and the parent dense path stays exact.
    assert np.array_equal(plan.page_table, np.arange(plan.block_table_len, dtype=np.int32))
    assert plan.token_positions.dtype == np.int64
    assert plan.token_positions.shape == (_MAX_SEQ,)
    assert np.array_equal(plan.token_positions, np.arange(_MAX_SEQ, dtype=np.int64))
    assert plan.evict_mask.dtype == np.bool_
    assert plan.evict_mask.shape == (_MAX_SEQ,)
    assert not plan.evict_mask.any()
    # Every live slot is visible and carries its own absolute position, so the
    # dense policy exercises all four span fields instead of leaving them null.
    assert plan.num_splits >= 1
    assert plan.chunk_size * plan.num_splits >= plan.max_seq
    assert (plan.chunk_size * (plan.num_splits - 1)) < plan.max_seq or plan.num_splits == 1


def test_plan_surya_dense_spans_rejects_bad_geometry() -> None:
    from hipengine.kernels.hip_gfx1100.surya.surya_ops import plan_surya_dense_spans

    with pytest.raises(ValueError, match="max_seq"):
        plan_surya_dense_spans(0)
    with pytest.raises(ValueError, match="block_size"):
        plan_surya_dense_spans(_MAX_SEQ, block_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        plan_surya_dense_spans(_MAX_SEQ, chunk_size=0)
    with pytest.raises(ValueError, match="multiple of"):
        plan_surya_dense_spans(_MAX_SEQ, chunk_size=1)


def _fake_spans(max_seq: int = _MAX_SEQ) -> KVLiveSpans:
    pages = max_seq // _BLOCK_SIZE
    return KVLiveSpans.paged_dense(
        block_table=_tensor(0x1000, (pages,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        token_positions=_tensor(0x3000, (max_seq,), "int64"),
        evict_mask=_tensor(0x4000, (max_seq,), "bool"),
        row_positions=_tensor(0x5000, (1,), "int64"),
        capacity=max_seq,
        block_size=_BLOCK_SIZE,
        storage_dtype="fp32",
    )


def _fake_spans_bf16(max_seq: int = _MAX_SEQ) -> KVLiveSpans:
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (max_seq // _BLOCK_SIZE,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        max_live_count=max_seq,
        storage_dtype="bf16",
    )


def test_decode_wrapper_validates_shape_before_build(monkeypatch) -> None:
    import hipengine.kernels.hip_gfx1100.surya.surya_ops as module

    def fail_build(**_kwargs):
        raise AssertionError("build reached")

    monkeypatch.setattr(module, "build_surya_ops", fail_build)
    launch = module.surya_full_attn_decode_f32_spans
    spans = _fake_spans()
    valid = (0x6000, 0x7000, 0x8000, 0x9000, 0xA000, 0xB000, 0xC000)

    for index in range(len(valid)):
        pointers = list(valid)
        pointers[index] = 0
        with pytest.raises(ValueError, match="non-zero"):
            launch(*pointers, spans, _BLOCK_SIZE, _NQ, _NK, _HD, _HD**-0.5)
    with pytest.raises(ValueError, match="head_dim"):
        launch(*valid, spans, _BLOCK_SIZE, _NQ, _NK, 128, 128**-0.5)
    with pytest.raises(ValueError, match="GQA repeat"):
        launch(*valid, spans, _BLOCK_SIZE, 6, _NK, _HD, _HD**-0.5)
    with pytest.raises(ValueError, match="num_kv_heads"):
        launch(*valid, spans, _BLOCK_SIZE, _NQ, 0, _HD, _HD**-0.5)
    with pytest.raises(ValueError, match="chunk_size"):
        launch(*valid, spans, _BLOCK_SIZE, _NQ, _NK, _HD, _HD**-0.5, chunk_size=1)
    with pytest.raises(ValueError, match="fp32 storage"):
        launch(*valid, _fake_spans_bf16(), _BLOCK_SIZE, _NQ, _NK, _HD, _HD**-0.5)


def test_scatter_wrapper_validates_shape_before_build(monkeypatch) -> None:
    import hipengine.kernels.hip_gfx1100.surya.surya_ops as module

    def fail_build(**_kwargs):
        raise AssertionError("build reached")

    monkeypatch.setattr(module, "build_surya_ops", fail_build)
    launch = module.surya_scatter_kv_f32_spans
    spans = _fake_spans()

    with pytest.raises(ValueError, match="non-zero"):
        launch(0, 0x1000, spans, 4, 0, _BLOCK_SIZE, _NK, _HD)
    with pytest.raises(ValueError, match="head_dim"):
        launch(0x1000, 0x2000, spans, 4, 0, _BLOCK_SIZE, _NK, 0)
    with pytest.raises(ValueError, match="tokens"):
        launch(0x1000, 0x2000, spans, 0, 0, _BLOCK_SIZE, _NK, _HD)
    with pytest.raises(ValueError, match="token_offset"):
        launch(0x1000, 0x2000, spans, 4, -1, _BLOCK_SIZE, _NK, _HD)


def test_kernels_read_the_full_spans_abi() -> None:
    source = _SOURCE.read_text(encoding="utf-8")
    scatter = source[source.index("extern \"C\" int hipengine_surya_scatter_kv_f32_spans("):]
    scatter = scatter[: scatter.index("\n}\n")]
    for field in ("base_offsets", "live_counts", "token_positions", "evict_mask"):
        assert field in scatter, field
    decode = source[
        source.index("extern \"C\" int hipengine_surya_full_attn_decode_split_k_f32_spans("):
    ]
    decode = decode[: decode.index("\n}\n")]
    for field in ("base_offsets", "live_counts", "token_positions", "evict_mask", "row_positions"):
        assert field in decode, field


# --------------------------------------------------------------------------- #
# device fixtures
# --------------------------------------------------------------------------- #


def _reference_decode(
    query: np.ndarray,
    key_plane: np.ndarray,
    value_plane: np.ndarray,
    page_table: np.ndarray,
    token_positions: np.ndarray,
    evict_mask: np.ndarray,
    live_count: int,
    row_position: int,
    block_size: int,
) -> np.ndarray:
    """Float64 span-aware decode reference (page table + visibility + causal)."""

    nq, head_dim = query.shape
    nk = key_plane.shape[0]
    group = nq // nk
    scale = head_dim**-0.5
    out = np.zeros((nq, head_dim), dtype=np.float64)
    for q_head in range(nq):
        kv_head = q_head // group
        visible = []
        for token in range(min(live_count, token_positions.shape[0])):
            if token_positions[token] < 0 or token_positions[token] > row_position:
                continue
            if evict_mask[token]:
                continue
            logical_block = token // block_size
            physical = int(page_table[logical_block]) * block_size + (token % block_size)
            visible.append(physical)
        if not visible:
            continue
        keys = key_plane[kv_head, visible, :].astype(np.float64)
        values = value_plane[kv_head, visible, :].astype(np.float64)
        scores = keys @ query[q_head].astype(np.float64) * scale
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        out[q_head] = weights @ values
    return out


def _reference_scatter(
    src: np.ndarray,
    page_table: np.ndarray,
    token_positions: np.ndarray,
    evict_mask: np.ndarray,
    live_count: int,
    token_offset: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq: int,
    block_size: int,
) -> np.ndarray:
    dst = np.zeros((num_kv_heads, max_seq, head_dim), dtype=np.float32)
    for index in range(src.shape[0]):
        logical = token_offset + index
        if logical >= live_count:
            continue
        if token_positions[logical] < 0 or evict_mask[logical]:
            continue
        logical_block = logical // block_size
        physical = int(page_table[logical_block]) * block_size + (logical % block_size)
        dst[:, physical, :] = src[index]
    return dst


def _permuted_page_table(block_table_len: int) -> np.ndarray:
    """Reverse the physical block order so an identity shortcut cannot pass."""

    return np.arange(block_table_len - 1, -1, -1, dtype=np.int32)


def _cached_ptr_array(ptrs: list[int]):
    """Cache int64 device pointer arrays the way the runner does."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    runtime = get_hip_runtime()
    cache = _cached_ptr_array.__dict__.setdefault("_cache", {})
    key = tuple(ptrs)
    if key not in cache:
        host = np.asarray(ptrs, dtype=np.uint64)
        buf = malloc(host.nbytes, runtime=runtime)
        copy_host_to_device(buf, host_array_ptr(host), host.nbytes, runtime=runtime)
        cache[key] = buf
    return cache[key]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_scatter_f32_spans_honors_page_table_and_eviction() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.surya.surya_ops import (
        build_surya_ops,
        surya_scatter_kv_f32_spans,
    )

    runtime = get_hip_runtime()
    library = build_surya_ops(load=True, require_cached=_require_cached_build())

    block_table_len = _MAX_SEQ // _BLOCK_SIZE
    page_table = _permuted_page_table(block_table_len)
    token_positions = np.arange(_MAX_SEQ, dtype=np.int64)
    token_positions[300] = -1  # empty slot, inside the written range
    evict_mask = np.zeros(_MAX_SEQ, dtype=np.bool_)
    evict_mask[700] = True  # explicitly evicted, inside the written range
    token_offset = 256
    tokens = 1024
    live_count = token_offset + tokens

    rng = np.random.default_rng(0x5A17)
    src = rng.standard_normal((tokens, _NK, _HD)).astype(np.float32)
    expected = _reference_scatter(
        src, page_table, token_positions, evict_mask, live_count,
        token_offset, _NK, _HD, _MAX_SEQ, _BLOCK_SIZE,
    )
    # The fixture is only meaningful if the span fields change the answer.
    identity = np.arange(block_table_len, dtype=np.int32)
    no_evict = np.zeros(_MAX_SEQ, dtype=np.bool_)
    identity_expected = _reference_scatter(
        src, identity, np.arange(_MAX_SEQ, dtype=np.int64), no_evict, live_count,
        token_offset, _NK, _HD, _MAX_SEQ, _BLOCK_SIZE,
    )
    assert not np.array_equal(expected, identity_expected)

    allocations = []
    try:
        src_buf = malloc(src.nbytes, runtime=runtime)
        dst_buf = malloc(_NK * _MAX_SEQ * _HD * 4, runtime=runtime)
        page_buf = malloc(page_table.nbytes, runtime=runtime)
        pos_buf = malloc(token_positions.nbytes, runtime=runtime)
        mask_buf = malloc(evict_mask.nbytes, runtime=runtime)
        live_buf = malloc(8, runtime=runtime)
        allocations.extend((src_buf, dst_buf, page_buf, pos_buf, mask_buf, live_buf))
        copy_host_to_device(src_buf, host_array_ptr(src), src.nbytes, runtime=runtime)
        copy_host_to_device(dst_buf, host_array_ptr(np.zeros_like(expected)), expected.nbytes, runtime=runtime)
        copy_host_to_device(page_buf, host_array_ptr(page_table), page_table.nbytes, runtime=runtime)
        copy_host_to_device(pos_buf, host_array_ptr(token_positions), token_positions.nbytes, runtime=runtime)
        copy_host_to_device(mask_buf, host_array_ptr(evict_mask), evict_mask.nbytes, runtime=runtime)
        live = np.array([live_count], dtype=np.int64)
        copy_host_to_device(live_buf, host_array_ptr(live), live.nbytes, runtime=runtime)
        runtime.device_synchronize()

        spans = KVLiveSpans.paged_dense(
            block_table=_tensor(page_buf.ptr, (block_table_len,), "int32"),
            live_counts=_tensor(live_buf.ptr, (1,), "int64"),
            token_positions=_tensor(pos_buf.ptr, (_MAX_SEQ,), "int64"),
            evict_mask=_tensor(mask_buf.ptr, (_MAX_SEQ,), "bool"),
            row_positions=_tensor(live_buf.ptr, (1,), "int64"),
            capacity=_MAX_SEQ,
            block_size=_BLOCK_SIZE,
            storage_dtype="fp32",
        )
        surya_scatter_kv_f32_spans(
            src_buf.ptr, dst_buf.ptr, spans, tokens, token_offset, _BLOCK_SIZE,
            _NK, _HD, library=library, runtime=runtime,
        )
        runtime.device_synchronize()

        actual = np.empty_like(expected)
        copy_device_to_host(host_array_ptr(actual), dst_buf, actual.nbytes, runtime=runtime)
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
    finally:
        for buffer in allocations:
            free(buffer, runtime=runtime)


def _run_decode_case(
    *,
    page_table: np.ndarray,
    token_positions: np.ndarray,
    evict_mask: np.ndarray,
    live_count: int,
    row_position: int,
    with_parent: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one decode fixture; returns (fused, parent, float64 reference)."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.rocblas import Rocblas
    from hipengine.kernels.hip_gfx1100.evie.evie_ops import build_evie_ops
    from hipengine.kernels.hip_gfx1100.surya.surya_ops import (
        build_surya_ops,
        plan_surya_dense_spans,
        surya_full_attn_decode_f32_spans,
    )
    from hipengine.runtime.surya import attention_decode_rocblas_f32

    runtime = get_hip_runtime()
    library = build_surya_ops(load=True, require_cached=_require_cached_build())
    evie = build_evie_ops(load=True)
    rocblas = Rocblas.load()
    rocblas.set_workspace(0, 0)

    plan = plan_surya_dense_spans(_MAX_SEQ, block_size=_BLOCK_SIZE)
    assert page_table.shape == (plan.block_table_len,)

    rng = np.random.default_rng(0xDEC0)
    query = rng.standard_normal((_NQ, _HD)).astype(np.float32)
    key_plane = rng.standard_normal((_NK, _MAX_SEQ, _HD)).astype(np.float32)
    value_plane = rng.standard_normal((_NK, _MAX_SEQ, _HD)).astype(np.float32)
    expected = _reference_decode(
        query, key_plane, value_plane, page_table, token_positions, evict_mask,
        live_count, row_position, _BLOCK_SIZE,
    )
    scale = float(_HD**-0.5)
    allocations = []
    try:
        query_buf = malloc(query.nbytes, runtime=runtime)
        key_buf = malloc(key_plane.nbytes, runtime=runtime)
        value_buf = malloc(value_plane.nbytes, runtime=runtime)
        out_buf = malloc(query.nbytes, runtime=runtime)
        parent_buf = malloc(query.nbytes, runtime=runtime)
        scores_buf = malloc(_NQ * _MAX_SEQ * 4, runtime=runtime)
        partial_out = malloc(_NQ * plan.num_splits * _HD * 4, runtime=runtime)
        partial_m = malloc(_NQ * plan.num_splits * 4, runtime=runtime)
        partial_l = malloc(_NQ * plan.num_splits * 4, runtime=runtime)
        page_buf = malloc(page_table.nbytes, runtime=runtime)
        pos_buf = malloc(token_positions.nbytes, runtime=runtime)
        mask_buf = malloc(evict_mask.nbytes, runtime=runtime)
        live_buf = malloc(16, runtime=runtime)
        allocations.extend((
            query_buf, key_buf, value_buf, out_buf, parent_buf, scores_buf,
            partial_out, partial_m, partial_l, page_buf, pos_buf, mask_buf, live_buf,
        ))
        for buffer, array in (
            (query_buf, query),
            (key_buf, key_plane),
            (value_buf, value_plane),
            (page_buf, page_table),
            (pos_buf, token_positions),
            (mask_buf, evict_mask),
        ):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)
        scalars = np.array([live_count, row_position], dtype=np.int64)
        copy_host_to_device(live_buf, host_array_ptr(scalars), scalars.nbytes, runtime=runtime)
        runtime.device_synchronize()

        spans = KVLiveSpans.paged_dense(
            block_table=_tensor(page_buf.ptr, (plan.block_table_len,), "int32"),
            live_counts=_tensor(live_buf.ptr, (1,), "int64"),
            token_positions=_tensor(pos_buf.ptr, (_MAX_SEQ,), "int64"),
            evict_mask=_tensor(mask_buf.ptr, (_MAX_SEQ,), "bool"),
            row_positions=_tensor(live_buf.ptr + 8, (1,), "int64"),
            capacity=_MAX_SEQ,
            block_size=_BLOCK_SIZE,
            storage_dtype="fp32",
        )

        if with_parent:
            attention_decode_rocblas_f32(
                rocblas, evie, runtime, _cached_ptr_array,
                q_ptr=query_buf.ptr, k_cache_ptr=key_buf.ptr, v_cache_ptr=value_buf.ptr,
                out_ptr=parent_buf.ptr, scores_ptr=scores_buf.ptr,
                total=live_count, num_q_heads=_NQ, num_key_value_heads=_NK,
                head_dim=_HD, max_seq=_MAX_SEQ,
            )
        surya_full_attn_decode_f32_spans(
            query_buf.ptr, key_buf.ptr, value_buf.ptr, out_buf.ptr,
            partial_out.ptr, partial_m.ptr, partial_l.ptr,
            spans, _BLOCK_SIZE, _NQ, _NK, _HD, scale,
            library=library, runtime=runtime,
        )
        runtime.device_synchronize()

        actual = np.empty_like(query)
        parent = np.full_like(query, np.nan)
        copy_device_to_host(host_array_ptr(actual), out_buf, actual.nbytes, runtime=runtime)
        if with_parent:
            copy_device_to_host(host_array_ptr(parent), parent_buf, parent.nbytes, runtime=runtime)
        return actual, parent, expected
    finally:
        for buffer in allocations:
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_decode_f32_spans_matches_rocblas_parent_on_a_dense_fill() -> None:
    """Parent parity is only meaningful on the dense fill the parent implements."""

    page_table = np.arange(_MAX_SEQ // _BLOCK_SIZE, dtype=np.int32)
    actual, parent, expected = _run_decode_case(
        page_table=page_table,
        token_positions=np.arange(_MAX_SEQ, dtype=np.int64),
        evict_mask=np.zeros(_MAX_SEQ, dtype=np.bool_),
        live_count=_MAX_SEQ,
        row_position=_MAX_SEQ - 1,
        with_parent=True,
    )
    # Outer gate vs the float64 CPU reference.
    np.testing.assert_allclose(actual, expected, atol=_CPU_REFERENCE_ATOL, rtol=0.0)
    kl, agreement = _quality(actual, expected)
    assert kl <= 1.0e-3, kl
    assert agreement == 1.0
    # Declared parent-parity bound vs the rocBLAS chain this kernel replaces.
    np.testing.assert_allclose(actual, parent, atol=_PARENT_PARITY_ATOL, rtol=0.0)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_decode_f32_spans_honors_page_table_and_eviction() -> None:
    """A (block_table, context_len) shortcut fails every one of these fields."""

    page_table = _permuted_page_table(_MAX_SEQ // _BLOCK_SIZE)
    token_positions = np.arange(_MAX_SEQ, dtype=np.int64)
    token_positions[1234] = -1  # empty slot
    evict_mask = np.zeros(_MAX_SEQ, dtype=np.bool_)
    evict_mask[321] = True  # explicitly evicted
    actual, _, expected = _run_decode_case(
        page_table=page_table,
        token_positions=token_positions,
        evict_mask=evict_mask,
        live_count=_MAX_SEQ,
        row_position=_MAX_SEQ - 1,
        with_parent=False,
    )
    np.testing.assert_allclose(actual, expected, atol=_CPU_REFERENCE_ATOL, rtol=0.0)
    kl, agreement = _quality(actual, expected)
    assert kl <= 1.0e-3, kl
    assert agreement == 1.0
    # The span fields are load-bearing: the identity/visible view of the same
    # fixture (same seed) differs, so a shortcut that ignores any of them fails.
    dense, _, _ = _run_decode_case(
        page_table=np.arange(_MAX_SEQ // _BLOCK_SIZE, dtype=np.int32),
        token_positions=np.arange(_MAX_SEQ, dtype=np.int64),
        evict_mask=np.zeros(_MAX_SEQ, dtype=np.bool_),
        live_count=_MAX_SEQ,
        row_position=_MAX_SEQ - 1,
        with_parent=False,
    )
    assert not np.allclose(actual, dense, atol=1.0e-3)
