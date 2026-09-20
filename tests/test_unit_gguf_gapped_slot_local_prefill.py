"""Gapped device-KV sessions keep the slot-local prefill executor via gather.

A prefix hit whose device-KV pages land non-contiguously used to drop the
whole packed slab onto the native paged prefill kernel, which is a large
per-token multiple of the AOTriton slot-local route. The gapped gather route
instead keeps the slot-local executor and swaps the slot's identity spans for
spans carrying the session's real chunk-local block table: the paged KV
write, the native fallback kernel, and the head-major gather all address the
pool pages through it, and AOTriton reads the gathered dense head-major
buffers exactly as the contiguous route does.

Contracts under test (CPU, fake device):

- ``_gguf_gapped_slot_local_prefill_admitted`` only admits BF16-KV,
  non-direct-INT8 slabs with the gather env enabled; the kill-switch and the
  INT8-retained/direct populations keep the packed fallback.
- ``_gguf_gapped_slot_block_table_rows`` tiles the allocation's chunk-local
  page ids into one equal block table per query row.
- ``_gguf_gapped_slot_local_prefill_scratch`` swaps ``append_spans`` /
  ``prefill_spans`` to those block-table spans, clears
  ``head_major_kv_dense_prefix`` so the head-major copy walks the block
  table, and leaves every other scratch field untouched.
- The swapped spans satisfy the parent kernel-launch shape contracts
  (paged KV write batch, prefill GQA, head-major copy).
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    _check_prefill_gqa_shape,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    _check_head_major_copy_shape,
    _check_write_batch_shape,
)
from hipengine.kvcache.spans import KVLiveSpans
from hipengine.runtime import qwen35_gguf_runner as gguf_runner


_TEST_DEVICE = "cpu_reference"


def _tensor(ptr: int, shape: tuple[int, ...], dtype: DType) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, _TEST_DEVICE)


def _allocation(block_ids: tuple[int, ...], chunk_start: int) -> SimpleNamespace:
    return SimpleNamespace(
        block_ids=tuple(int(block_id) for block_id in block_ids),
        chunk_start_block_id=int(chunk_start),
    )


def _gapped_session(block_ids: tuple[int, ...], chunk_start: int, storage: DType) -> SimpleNamespace:
    return SimpleNamespace(
        _device_kv_allocation=_allocation(block_ids, chunk_start),
        kv_storage_dtype=storage,
    )


@dataclass
class _FakeSlotScratch:
    start: int
    rows: int
    positions_tensor: Tensor
    context_counts_tensor: Tensor
    append_spans: KVLiveSpans
    prefill_spans: KVLiveSpans
    head_major_kv_dense_prefix: bool = True
    head_major_kv_admitted: bool = True
    head_major_key_cache: object = None
    head_major_value_cache: object = None
    head_major_kv_capacity: int = 0
    touched: str = "kept"


def _fake_slot_scratch(*, start: int, rows: int, blocks: int) -> _FakeSlotScratch:
    positions = _tensor(0x1000, (rows,), DType.INT64)
    context_counts = _tensor(0x2000, (rows,), DType.INT64)
    identity = _tensor(0x3000, (rows, blocks), DType.INT32)
    return _FakeSlotScratch(
        start=start,
        rows=rows,
        positions_tensor=positions,
        context_counts_tensor=context_counts,
        append_spans=KVLiveSpans.paged_uniform(
            block_table=identity,
            live_counts=positions,
            max_live_count=start + rows - 1,
            storage_dtype=DType.BF16,
            row_positions=positions,
            span_role="prefill",
        ),
        prefill_spans=KVLiveSpans.paged_uniform(
            block_table=identity,
            live_counts=context_counts,
            max_live_count=start + rows,
            storage_dtype=DType.BF16,
            row_positions=positions,
            span_role="prefill",
        ),
    )


def test_gather_env_name_is_pinned() -> None:
    assert gguf_runner._GGUF_GAPPED_GATHER_ENV == "HIPENGINE_GGUF_GAPPED_GATHER"


def test_gapped_bf16_slab_admits_the_gather_route() -> None:
    session = _gapped_session((7, 9, 12, 3), chunk_start=2, storage=DType.BF16)
    assert (
        gguf_runner._gguf_gapped_slot_local_prefill_admitted(
            direct_int8_prefill=False,
            sessions=(session,),
            contiguous_base_rows=(None,),
        )
        is True
    )


def test_fast_route_checks_the_request_context_not_resident_capacity() -> None:
    """Demand-sized gather buffers admit any resident class up to the 64K cap."""

    from hipengine.kernels.backends import load_backend_kernel_package

    load_backend_kernel_package("hip_gfx1151")
    available = dict(backend="hip_gfx1151", kv_width=512)
    # A 262,144-position resident class is fine: buffers are sized to the
    # request's own context.
    assert (
        gguf_runner._gguf_gapped_slot_local_fast_route_available(
            context_tokens=12_288, **available
        )
        is True
    )
    # Past the validated head-major allocation class the fast route is gone.
    assert (
        gguf_runner._gguf_gapped_slot_local_fast_route_available(
            context_tokens=100_000, **available
        )
        is False
    )


@pytest.mark.parametrize("context_tokens", [257, 511, 513, 1224])
def test_gapped_gather_does_not_require_contiguous_layout_optimization(
    monkeypatch: pytest.MonkeyPatch, context_tokens: int,
) -> None:
    """A layout-required gather must not inherit the contiguous optimization default."""
    monkeypatch.delenv(gguf_runner._GGUF_AOTRITON_HEAD_MAJOR_KV_ENV, raising=False)
    monkeypatch.delenv(gguf_runner._GGUF_GAPPED_GATHER_ENV, raising=False)
    assert not gguf_runner._gguf_aotriton_head_major_kv_enabled("hip_gfx1100")
    scratch = replace(
        _fake_slot_scratch(start=256, rows=2, blocks=8),
        head_major_kv_admitted=False,
    )
    allocations = []

    def allocate(nbytes, *, runtime):
        buffer = DeviceBuffer(0x100000 * (len(allocations) + 1), nbytes)
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", allocate)
    session = SimpleNamespace(runner=SimpleNamespace(backend="hip_gfx1100"))
    result = gguf_runner._gguf_gapped_slot_head_major_scratch(
        session, scratch, context_tokens=context_tokens, kv_width=1024, runtime=object(),
    )
    assert result.head_major_kv_admitted
    assert result.head_major_kv_capacity >= context_tokens
    assert result.head_major_key_cache.nbytes == 4096 * 1024 * 2
    assert result.head_major_value_cache.nbytes == 4096 * 1024 * 2
    repeated = gguf_runner._gguf_gapped_slot_head_major_scratch(
        session, scratch, context_tokens=context_tokens, kv_width=1024, runtime=object(),
    )
    assert repeated.head_major_key_cache is result.head_major_key_cache
    assert repeated.head_major_value_cache is result.head_major_value_cache
    assert len(allocations) == 2


def test_gapped_gather_preserves_explicit_head_major_rollback(monkeypatch):
    monkeypatch.setenv(gguf_runner._GGUF_AOTRITON_HEAD_MAJOR_KV_ENV, "0")
    assert not gguf_runner._gguf_gapped_slot_local_fast_route_available(
        backend="hip_gfx1100", context_tokens=1224, kv_width=1024,
    )


def test_gapped_gather_requires_a_registered_copy_kernel(monkeypatch):
    monkeypatch.setattr(gguf_runner, "resolve", lambda **kwargs: None)
    assert not gguf_runner._gguf_gapped_slot_local_fast_route_available(
        backend="hip_gfx1100", context_tokens=1224, kv_width=1024,
    )


@pytest.mark.parametrize("slot_view", [False, True])
def test_bf16_gather_buffers_are_freed_at_session_teardown(monkeypatch, slot_view):
    cls = gguf_runner.Qwen35GGUFResidentSession
    session = object.__new__(cls)
    for descriptor in fields(cls):
        if descriptor.default is not MISSING:
            setattr(session, descriptor.name, descriptor.default)
        elif descriptor.default_factory is not MISSING:
            setattr(session, descriptor.name, descriptor.default_factory())
    session.runtime = SimpleNamespace(device_synchronize=lambda: None)
    session.runner = None
    session._moe_graph = None
    session._gapped_slot_block_table_cache = ((), DeviceBuffer(101, 16))
    session._gapped_head_major_cache = (
        4096, DeviceBuffer(102, 8192), DeviceBuffer(103, 8192),
    )
    freed = []
    monkeypatch.setattr(gguf_runner, "free", lambda buffer, **kwargs: freed.append(buffer.ptr))
    if slot_view:
        session._resident_batch_owner = object()
        session._close_resident_slot_view_buffers(runtime=session.runtime)
    else:
        session.close()
    assert sorted(freed) == [101, 102, 103]
    assert session._gapped_slot_block_table_cache is None
    assert session._gapped_head_major_cache is None


def test_gather_bucket_rounding_cannot_exceed_explicit_memory_cap(monkeypatch):
    monkeypatch.setenv(gguf_runner._GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS_ENV, "1224")
    monkeypatch.setenv(gguf_runner._GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_BYTES_ENV, "5013504")
    sizes = []

    def allocate(nbytes, **kwargs):
        sizes.append(nbytes)
        return DeviceBuffer(0x100000 * len(sizes), nbytes)

    monkeypatch.setattr(gguf_runner, "malloc", allocate)
    result = gguf_runner._gguf_gapped_slot_head_major_scratch(
        SimpleNamespace(runner=SimpleNamespace(backend="hip_gfx1100")),
        replace(_fake_slot_scratch(start=1024, rows=200, blocks=6), head_major_kv_admitted=False),
        context_tokens=1224, kv_width=1024, runtime=object(),
    )
    assert result.head_major_kv_admitted
    assert result.head_major_kv_capacity == 1224
    assert sum(sizes) == 5013504


def test_gapped_slab_falls_back_when_the_kill_switch_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(gguf_runner._GGUF_GAPPED_GATHER_ENV, "0")
    session = _gapped_session((7, 9), chunk_start=0, storage=DType.BF16)
    assert (
        gguf_runner._gguf_gapped_slot_local_prefill_admitted(
            direct_int8_prefill=False,
            sessions=(session,),
            contiguous_base_rows=(None,),
        )
        is False
    )


def test_direct_int8_gapped_slab_keeps_the_packed_route() -> None:
    session = _gapped_session((7, 9), chunk_start=0, storage=DType.BF16)
    assert (
        gguf_runner._gguf_gapped_slot_local_prefill_admitted(
            direct_int8_prefill=True,
            sessions=(session,),
            contiguous_base_rows=(None,),
        )
        is False
    )


def test_int8_retained_gapped_session_keeps_the_packed_route() -> None:
    session = _gapped_session((7, 9), chunk_start=0, storage=DType.INT8_PER_TOKEN_HEAD)
    assert (
        gguf_runner._gguf_gapped_slot_local_prefill_admitted(
            direct_int8_prefill=False,
            sessions=(session,),
            contiguous_base_rows=(None,),
        )
        is False
    )


def test_mixed_slab_admits_only_when_every_gapped_session_is_bf16() -> None:
    contiguous = _gapped_session((4, 5, 6), chunk_start=4, storage=DType.BF16)
    gapped_bf16 = _gapped_session((8, 10, 2), chunk_start=8, storage=DType.BF16)
    gapped_int8 = _gapped_session((9, 11), chunk_start=9, storage=DType.INT8_PER_TOKEN_HEAD)
    kwargs = {"direct_int8_prefill": False}
    assert (
        gguf_runner._gguf_gapped_slot_local_prefill_admitted(
            sessions=(contiguous, gapped_bf16),
            contiguous_base_rows=(1280, None),
            **kwargs,
        )
        is True
    )
    assert (
        gguf_runner._gguf_gapped_slot_local_prefill_admitted(
            sessions=(contiguous, gapped_int8),
            contiguous_base_rows=(1280, None),
            **kwargs,
        )
        is False
    )


def test_block_table_rows_tile_chunk_local_page_ids_per_query_row() -> None:
    session = _gapped_session((17, 18, 19, 30, 41), chunk_start=17, storage=DType.BF16)
    table = gguf_runner._gguf_gapped_slot_block_table_rows(session, rows=4)
    expected_row = np.asarray([0, 1, 2, 13, 24], dtype=np.int32)
    assert table.shape == (4, 5)
    for row_index in range(4):
        np.testing.assert_array_equal(table[row_index], expected_row)


def test_block_table_rows_reject_invalid_rows() -> None:
    session = _gapped_session((17, 18), chunk_start=17, storage=DType.BF16)
    with pytest.raises(ValueError):
        gguf_runner._gguf_gapped_slot_block_table_rows(session, rows=0)
    with pytest.raises(RuntimeError):
        gguf_runner._gguf_gapped_slot_block_table_rows(
            SimpleNamespace(_device_kv_allocation=None, kv_storage_dtype=DType.BF16),
            rows=4,
        )


def _block_table_tensor(table: np.ndarray) -> Tensor:
    return Tensor.from_handle(
        0x4000,
        tuple(int(dim) for dim in table.shape),
        DType.INT32,
        _TEST_DEVICE,
    )


def test_gapped_scratch_swap_replaces_spans_and_clears_dense_prefix() -> None:
    scratch = _fake_slot_scratch(start=512, rows=64, blocks=8)
    table = gguf_runner._gguf_gapped_slot_block_table_rows(
        _gapped_session((3, 4, 5, 6, 7, 8, 9, 10), chunk_start=3, storage=DType.BF16),
        rows=64,
    )
    swapped = gguf_runner._gguf_gapped_slot_local_prefill_scratch(
        scratch,
        block_table=_block_table_tensor(table),
    )
    assert swapped is not scratch
    assert swapped.head_major_kv_dense_prefix is False
    # Untouched fields survive the swap.
    assert swapped.start == scratch.start
    assert swapped.rows == scratch.rows
    assert swapped.positions_tensor is scratch.positions_tensor
    assert swapped.context_counts_tensor is scratch.context_counts_tensor
    assert swapped.head_major_kv_admitted == scratch.head_major_kv_admitted
    assert swapped.touched == "kept"
    # Append spans track the per-row position boundary; prefill spans track
    # the per-row context count, both over the real block table.
    assert swapped.append_spans.base_offsets is not scratch.append_spans.base_offsets
    assert swapped.append_spans.live_counts is scratch.positions_tensor
    assert swapped.append_spans.row_positions is scratch.positions_tensor
    assert swapped.append_spans.max_live_count == 512 + 64 - 1
    assert swapped.append_spans.storage_dtype == DType.BF16
    assert swapped.append_spans.spans_mode == "uniform"
    assert swapped.append_spans.span_role == "prefill"
    assert swapped.prefill_spans.live_counts is scratch.context_counts_tensor
    assert swapped.prefill_spans.row_positions is scratch.positions_tensor
    assert swapped.prefill_spans.max_live_count == 512 + 64
    assert swapped.prefill_spans.base_offsets is swapped.append_spans.base_offsets


def test_swapped_spans_satisfy_parent_kernel_shape_contracts() -> None:
    block_size = 256
    scratch = _fake_slot_scratch(start=2048, rows=64, blocks=12)
    session = _gapped_session(tuple(range(40, 52)), chunk_start=40, storage=DType.BF16)
    table = gguf_runner._gguf_gapped_slot_block_table_rows(session, rows=64)
    swapped = gguf_runner._gguf_gapped_slot_local_prefill_scratch(
        scratch,
        block_table=_block_table_tensor(table),
    )
    end = scratch.start + scratch.rows
    # Paged KV write (append spans).
    assert (
        _check_write_batch_shape(
            swapped.append_spans,
            scratch.rows,
            block_size,
            num_kv_heads=2,
            head_dim=256,
        )
        == 12
    )
    # Native paged prefill kernel (prefill spans).
    assert (
        _check_prefill_gqa_shape(
            swapped.prefill_spans,
            scratch.rows,
            end,
            block_size,
            num_q_heads=16,
            num_kv_heads=2,
            head_dim=256,
        )
        == 12
    )
    # Head-major gather (prefill spans; output capacity covers the context).
    table_len, table_row, live_index, has_slot_metadata = _check_head_major_copy_shape(
        swapped.prefill_spans,
        end,
        output_capacity=end,
        block_size=block_size,
        num_kv_heads=2,
        head_dim=256,
    )
    assert table_len == 12
    assert table_row == scratch.rows - 1
    assert live_index == scratch.rows - 1
    assert has_slot_metadata is False
