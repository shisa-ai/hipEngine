from __future__ import annotations

import ctypes
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipError, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    build_qwen35_paged_kv_write,
    qwen35_copy_paged_kv_bf16_to_head_major_dense_prefix_spans,
    qwen35_copy_paged_kv_bf16_to_head_major_spans,
)
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.kvcache import KVLiveSpans
from hipengine.runtime.qwen35_gguf_runner import (
    _gguf_aotriton_head_major_buffers,
    _gguf_aotriton_head_major_kv_enabled,
    _try_allocate_gguf_aotriton_head_major_kv_scratch,
)
from scripts.qwen35_gguf_bench import (
    _bulk_prefill_scratch_census,
    _session_buffer_breakdown,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _tensor(ptr: int, shape: tuple[int, ...], dtype: DType) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def test_head_major_copy_is_registered_through_attention_registry() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer="paged_kv_copy",
        quant="bf16",
        variant="head_major_spans",
    ) is qwen35_copy_paged_kv_bf16_to_head_major_spans
    assert resolve(
        backend="hip_gfx1100",
        layer="paged_kv_copy",
        quant="bf16",
        variant="head_major_dense_prefix_spans",
    ) is qwen35_copy_paged_kv_bf16_to_head_major_dense_prefix_spans
    assert KernelKey(
        "hip_gfx1100", "paged_kv_copy", "bf16", "head_major_spans"
    )
    assert resolve(
        backend="hip_gfx1151",
        layer="paged_kv_copy",
        quant="bf16",
        variant="head_major_spans",
    ) is qwen35_copy_paged_kv_bf16_to_head_major_spans
    assert resolve(
        backend="hip_gfx1151",
        layer="paged_kv_copy",
        quant="bf16",
        variant="head_major_dense_prefix_spans",
    ) is qwen35_copy_paged_kv_bf16_to_head_major_dense_prefix_spans


def test_head_major_policy_defaults_only_on_gfx1151_and_ignores_retired_boolean(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV", "0")
    assert _gguf_aotriton_head_major_kv_enabled("hip_gfx1151") is True
    monkeypatch.setenv("HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV", "1")
    assert _gguf_aotriton_head_major_kv_enabled("hip_gfx1100") is False


def test_head_major_scratch_is_reported_separately_from_bulk_scratch() -> None:
    base = DeviceBuffer(0x1000, 100)
    key = DeviceBuffer(0x2000, 20)
    value = DeviceBuffer(0x3000, 20)
    session = SimpleNamespace(
        _buffers=(base, key, value),
        _bulk_prefill_scratch=SimpleNamespace(
            rows=8,
            max_positions=16,
            head_major_kv_capacity=16,
            head_major_key_cache=key,
            head_major_value_cache=value,
            buffers=(base, key, value),
        ),
    )

    breakdown = _session_buffer_breakdown(session)

    assert breakdown["by_component_bytes"]["bulk_prefill_scratch"] == 100
    assert breakdown["by_component_bytes"]["aotriton_head_major_kv_scratch"] == 40
    assert breakdown["aotriton_head_major_kv_capacity"] == 16
    assert breakdown["by_component_bytes"]["session_buffer_other"] == 0


def test_bulk_prefill_scratch_census_exposes_logical_aliases_and_lifetimes() -> None:
    arena = DeviceBuffer(0x1000, 100)
    scratch = SimpleNamespace(
        rows=8,
        max_positions=16,
        allocation_mode="liveness_aliased",
        allocation_offsets={"field_a": (0, 80), "field_b": (0, 80)},
        allocation_lifetimes={
            "field_a": (("linear", 0, 1),),
            "field_b": (("full", 1, 2),),
        },
        allocation_groups={"field_a": "owner_slot_00", "field_b": "owner_slot_00"},
        allocation_inplace_aliases={"field_b": "field_a"},
        allocation_subranges={
            "field_a": (
                (0, 40, (("linear", 0, 2),)),
                (40, 40, (("full", 1, 2),)),
            )
        },
        field_a=DeviceBuffer(0x1000, 80),
        field_b=DeviceBuffer(0x1000, 80),
        metadata=DeviceBuffer(0x1050, 20),
        buffers=(arena,),
        head_major_key_cache=None,
        head_major_value_cache=None,
    )

    census = _bulk_prefill_scratch_census(scratch)

    assert census["physical_owner_bytes"] == 100
    assert census["logical_field_bytes"] == 180
    assert census["logical_alias_overhead_bytes"] == 80
    assert census["by_field_bytes"]["field_a"] == 80
    assert census["allocation_offsets"]["field_b"] == {"offset_bytes": 0, "nbytes": 80}
    assert census["allocation_lifetimes"]["field_a"] == [
        {"route": "linear", "start_stage": 0, "end_stage": 1}
    ]
    assert census["allocation_groups"] == {
        "field_a": "owner_slot_00",
        "field_b": "owner_slot_00",
    }
    assert census["allocation_inplace_aliases"] == {"field_b": "field_a"}
    assert census["allocation_subranges"]["field_a"][0] == {
        "relative_offset_bytes": 0,
        "nbytes": 40,
        "lifetimes": [
            {"route": "linear", "start_stage": 0, "end_stage": 2}
        ],
    }


def test_head_major_scratch_allocation_denial_frees_partial_and_falls_back(monkeypatch) -> None:
    import hipengine.runtime.qwen35_gguf_runner as runner_module

    allocated = DeviceBuffer(0x5000, 4096)
    calls = 0
    freed: list[DeviceBuffer] = []

    def fake_malloc(nbytes: int, *, runtime):
        nonlocal calls
        calls += 1
        if calls == 1:
            return replace(allocated, nbytes=nbytes)
        raise HipError(2, "forced capacity denial")

    monkeypatch.setattr(runner_module, "malloc", fake_malloc)
    monkeypatch.setattr(runner_module, "free", lambda buffer, *, runtime: freed.append(buffer))
    result = _try_allocate_gguf_aotriton_head_major_kv_scratch(
        backend="hip_gfx1151",
        capacity_tokens=257,
        kv_width=512,
        runtime=object(),
    )

    assert result is None
    assert freed == [replace(allocated, nbytes=257 * 512 * DType.BF16.itemsize)]
    scratch = type(
        "Scratch",
        (),
        {
            "head_major_key_cache": None,
            "head_major_value_cache": None,
            "head_major_kv_capacity": 0,
            "head_major_kv_admitted": True,
        },
    )()
    assert _gguf_aotriton_head_major_buffers(scratch, context_len=257) is None


def test_head_major_scratch_prices_long_capacity_against_common_budget(monkeypatch) -> None:
    import hipengine.runtime.qwen35_gguf_runner as runner_module

    runtime = SimpleNamespace(mem_get_info=lambda: (1, 1))
    monkeypatch.setattr(
        runner_module,
        "malloc",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not allocate")),
    )

    assert (
        _try_allocate_gguf_aotriton_head_major_kv_scratch(
            backend="hip_gfx1151",
            capacity_tokens=131328,
            kv_width=512,
            runtime=runtime,
        )
        is None
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("context_len", [1, 255, 256, 257])
def test_paged_bf16_head_major_copy_matches_permuted_numpy_oracle(context_len: int) -> None:
    runtime = get_hip_runtime()
    library = build_qwen35_paged_kv_write(load=True)
    block_size, kv_heads, head_dim = 256, 2, 64
    logical_blocks = (context_len + block_size - 1) // block_size
    physical_blocks = max(3, logical_blocks + 1)
    block_table = np.asarray([2, 0, 1][:logical_blocks], dtype=np.int32)
    live_count = np.asarray([context_len], dtype=np.int64)
    physical_shape = (physical_blocks * block_size, kv_heads, head_dim)
    rng = np.random.default_rng(20260804 + context_len)
    key = rng.integers(0, 65536, size=physical_shape, dtype=np.uint16)
    value = rng.integers(0, 65536, size=physical_shape, dtype=np.uint16)
    output_capacity = context_len + 3
    sentinel = np.uint16(0xA55A)
    key_out = np.full((kv_heads, output_capacity, head_dim), sentinel, dtype=np.uint16)
    value_out = np.full_like(key_out, sentinel)
    buffers: list[DeviceBuffer] = []

    def upload(array: np.ndarray) -> DeviceBuffer:
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
        return buffer

    try:
        key_buf = upload(key)
        value_buf = upload(value)
        table_buf = upload(block_table)
        live_buf = upload(live_count)
        key_out_buf = upload(key_out)
        value_out_buf = upload(value_out)
        spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(table_buf.ptr, block_table.shape, DType.INT32),
            live_counts=_tensor(live_buf.ptr, live_count.shape, DType.INT64),
            max_live_count=context_len,
            storage_dtype=DType.BF16,
            span_role="prefill",
        )

        qwen35_copy_paged_kv_bf16_to_head_major_spans(
            key_buf.ptr,
            value_buf.ptr,
            key_out_buf.ptr,
            value_out_buf.ptr,
            spans,
            context_len,
            output_capacity,
            block_size,
            kv_heads,
            head_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(key_out), key_out_buf, runtime=runtime)
        copy_device_to_host(host_array_ptr(value_out), value_out_buf, runtime=runtime)

        logical_rows = np.concatenate(
            [
                int(block_table[logical]) * block_size
                + np.arange(block_size, dtype=np.int64)
                for logical in range(logical_blocks)
            ]
        )[:context_len]
        expected_key = np.transpose(key[logical_rows], (1, 0, 2))
        expected_value = np.transpose(value[logical_rows], (1, 0, 2))
        assert np.array_equal(key_out[:, :context_len], expected_key)
        assert np.array_equal(value_out[:, :context_len], expected_value)
        assert np.all(key_out[:, context_len:] == sentinel)
        assert np.all(value_out[:, context_len:] == sentinel)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("value_bf16", [False, True])
def test_append_targets_independent_bf16_pages(hip_test_target_arch, value_bf16) -> None:
    from hipengine.kernels.backends import HIP_TARGET_ARCH_BACKEND, hip_target_arch_environment

    write = resolve(backend=HIP_TARGET_ARCH_BACKEND[hip_test_target_arch], layer="paged_kv_write",
                    quant="bf16", variant=("mixed_bf16_page_ptrs_spans" if value_bf16 else "f32_page_ptrs_spans"))
    runtime = get_hip_runtime()
    block_size, width = 256, 8
    sentinel = np.uint16(0xA55A)
    buffers = []

    def upload(array):
        result = malloc(array.nbytes, runtime=runtime)
        buffers.append(result)
        copy_host_to_device(result, host_array_ptr(array), runtime=runtime)
        return result

    try:
        keys = [upload(np.full((block_size, width), sentinel, dtype=np.uint16)) for _ in range(2)]
        vals = [upload(np.full((block_size, width), sentinel, dtype=np.uint16)) for _ in range(2)]
        kt = upload(np.array([p.ptr for p in keys], dtype=np.uint64))
        vt = upload(np.array([p.ptr for p in vals], dtype=np.uint64))
        table = upload(np.array([1, 0], dtype=np.int32))
        position = upload(np.array([0], dtype=np.int64))
        k = upload(np.arange(width, dtype=np.float32))
        source_v = -np.arange(1, width + 1, dtype=np.float32)
        v = upload((source_v.view(np.uint32) >> 16).astype(np.uint16) if value_bf16 else source_v)
        spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(table.ptr, (2,), DType.INT32),
            live_counts=_tensor(position.ptr, (1,), DType.INT64),
            max_live_count=block_size + 1, storage_dtype=DType.BF16,
        )
        with hip_target_arch_environment(hip_test_target_arch):
            for pos in (255, 256):
                host_pos = np.array([pos], dtype=np.int64)
                copy_host_to_device(position, host_array_ptr(host_pos), runtime=runtime)
                write(k.ptr, v.ptr, kt.ptr, vt.ptr, spans, block_size, 1, width, runtime=runtime)
            runtime.device_synchronize()
        for planes, source in ((keys, np.arange(width, dtype=np.float32)),
                               (vals, -np.arange(1, width + 1, dtype=np.float32))):
            for page, row in zip(planes, (0, 255), strict=True):
                out = np.empty((block_size, width), dtype=np.uint16)
                copy_device_to_host(host_array_ptr(out), page, runtime=runtime)
                expected = np.full_like(out, sentinel)
                expected[row] = (source.view(np.uint32) >> 16).astype(np.uint16)
                np.testing.assert_array_equal(out, expected)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_prompt_append_targets_independent_bf16_pages(hip_test_target_arch) -> None:
    from hipengine.kernels.backends import HIP_TARGET_ARCH_BACKEND, hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
        qwen35_write_paged_kv_mixed_value_bf16_prompt_spans as reference,
    )

    write = resolve(backend=HIP_TARGET_ARCH_BACKEND[hip_test_target_arch], layer="paged_kv_write",
                    quant="bf16", variant="mixed_bf16_prompt_page_ptrs_spans")
    runtime = get_hip_runtime()
    buffers = []
    block_size, width = 256, 8
    positions = np.array([255, 256, 257, -1], dtype=np.int64)
    keys = np.arange(len(positions) * width, dtype=np.float32).reshape(-1, width)
    values = ((-keys - 1).view(np.uint32) >> 16).astype(np.uint16)
    sentinel = np.full((block_size, width), 0xA55A, dtype=np.uint16)

    def upload(array):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    try:
        positions_gpu = upload(positions)
        row_tables = upload(np.tile(np.array([1, 0], dtype=np.int32), (len(positions), 1)))
        spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(row_tables.ptr, (len(positions), 2), DType.INT32),
            live_counts=_tensor(positions_gpu.ptr, (len(positions),), DType.INT64),
            max_live_count=258, storage_dtype=DType.BF16,
        )
        pages = [[upload(sentinel), upload(sentinel)] for _ in range(2)]
        tables = [upload(np.asarray([page.ptr for page in plane], dtype=np.uint64))
                  for plane in pages]
        contiguous = [upload(np.stack((sentinel, sentinel))) for _ in range(2)]
        key_src, value_src = upload(keys), upload(values)
        with hip_target_arch_environment(hip_test_target_arch):
            for fn, destinations in ((reference, contiguous), (write, tables)):
                fn(key_src.ptr, value_src.ptr, destinations[0].ptr, destinations[1].ptr,
                   spans, len(positions), block_size, 1, width, runtime=runtime)
            runtime.device_synchronize()
        for plane, parent, rows in zip(
            pages, contiguous, ((keys.view(np.uint32) >> 16).astype(np.uint16), values),
            strict=True,
        ):
            expected = np.stack((sentinel, sentinel))
            expected[1][255] = rows[0]
            expected[0][:2] = rows[1:3]
            parent_out = np.empty_like(expected)
            copy_device_to_host(host_array_ptr(parent_out), parent, runtime=runtime)
            np.testing.assert_array_equal(parent_out, expected)
            for page, expected_page in zip(plane, parent_out, strict=True):
                out = np.empty_like(sentinel)
                copy_device_to_host(host_array_ptr(out), page, runtime=runtime)
                np.testing.assert_array_equal(out, expected_page)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("head_dim", [128, 256])
def test_bf16_decode_through_independent_pages_matches_contiguous(head_dim, hip_test_target_arch):
    from hipengine.kernels.backends import HIP_TARGET_ARCH_BACKEND, hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
        qwen35_paged_full_attn_decode_context_bf16_spans as reference,
    )

    from hipengine.kvcache.device_global import GlobalDeviceKVPool
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    decode = resolve(backend=HIP_TARGET_ARCH_BACKEND[hip_test_target_arch], layer="paged_attn_decode",
                     quant="bf16", variant="context_page_ptrs_spans")
    runtime = get_hip_runtime()
    buffers = []
    pool = None
    allocation = None
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = runtime
    rng = np.random.default_rng(947)

    def upload(array):
        result = malloc(array.nbytes, runtime=runtime)
        buffers.append(result)
        copy_host_to_device(result, host_array_ptr(array), runtime=runtime)
        return result

    try:
        values = rng.uniform(-0.5, 0.5, (2, 2, 256, head_dim)).astype(np.float32)
        planes = (values.view(np.uint32) >> 16).astype(np.uint16)
        contiguous = [upload(plane) for plane in planes]
        roles = ("key_payload", "value_payload")
        pointers = {
            role: tuple(upload(page).ptr for page in reversed(plane))
            for role, plane in zip(roles, planes, strict=True)
        }
        # Deliberately not device tables: the resident path must snapshot the
        # lease's payload page addresses, not borrow this pool publication.
        pool = GlobalDeviceKVPool(
            page_bytes=planes[:, 0].nbytes,
            backend_fingerprint=hip_test_target_arch,
            generation=1,
            backing=object(),
            plane_page_pointers=pointers,
            pointer_table_pointers={role: 1 for role in roles},
            metadata_descriptor_pointer=1,
            close_storage=lambda: None,
        )
        allocation = pool.allocate(0, 2)
        session._device_kv_pool = pool
        session._device_kv_allocation = allocation
        view = session.device_kv_page_view()
        tables = [view.plane(role) for role in roles]
        table = upload(np.array([1, 0], dtype=np.int32))
        count = upload(np.array([257], dtype=np.int64))
        query = upload(rng.uniform(-0.5, 0.5, (2, head_dim)).astype(np.float32))
        a, b = (upload(np.zeros((2, head_dim), dtype=np.float32)) for _ in range(2))
        spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(table.ptr, (2,), DType.INT32),
            live_counts=_tensor(count.ptr, (1,), DType.INT64),
            max_live_count=257, storage_dtype=DType.BF16,
        )
        page_spans = replace(spans, base_offsets=_tensor(view.block_table.ptr, (2,), DType.INT32))
        with hip_target_arch_environment(hip_test_target_arch):
            for fn, inputs, output, live_spans in (
                (reference, contiguous, a, spans), (decode, tables, b, page_spans),
            ):
                fn(query.ptr, inputs[0].ptr, inputs[1].ptr, output.ptr, live_spans,
                   257, 256, 2, 1, head_dim, head_dim ** -0.5, runtime=runtime)
            runtime.device_synchronize()
        expected = np.empty((2, head_dim), dtype=np.float32)
        got = np.empty_like(expected)
        copy_device_to_host(host_array_ptr(expected), a, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), b, runtime=runtime)
        assert np.any(expected != 0)
        np.testing.assert_array_equal(got, expected)
    finally:
        session._release_device_kv_page_view()
        if pool is not None:
            if allocation is not None:
                pool.release(allocation.request_id)
            pool.close()
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("context_len", [1, 255, 256, 257])
def test_dense_and_generic_head_major_copy_are_byte_exact(context_len: int) -> None:
    runtime = get_hip_runtime()
    library = build_qwen35_paged_kv_write(load=True)
    block_size, kv_heads, head_dim = 256, 2, 64
    blocks = (context_len + block_size - 1) // block_size
    block_table = np.arange(blocks, dtype=np.int32)
    live_count = np.asarray([context_len], dtype=np.int64)
    source_shape = (blocks * block_size, kv_heads, head_dim)
    rng = np.random.default_rng(20260840 + context_len)
    key = rng.integers(0, 65536, size=source_shape, dtype=np.uint16)
    value = rng.integers(0, 65536, size=source_shape, dtype=np.uint16)
    output_shape = (kv_heads, context_len, head_dim)
    generic_key = np.zeros(output_shape, dtype=np.uint16)
    generic_value = np.zeros(output_shape, dtype=np.uint16)
    dense_key = np.zeros(output_shape, dtype=np.uint16)
    dense_value = np.zeros(output_shape, dtype=np.uint16)
    buffers: list[DeviceBuffer] = []

    def upload(array: np.ndarray) -> DeviceBuffer:
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
        return buffer

    try:
        key_buf = upload(key)
        value_buf = upload(value)
        table_buf = upload(block_table)
        live_buf = upload(live_count)
        generic_key_buf = upload(generic_key)
        generic_value_buf = upload(generic_value)
        dense_key_buf = upload(dense_key)
        dense_value_buf = upload(dense_value)
        spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(table_buf.ptr, block_table.shape, DType.INT32),
            live_counts=_tensor(live_buf.ptr, live_count.shape, DType.INT64),
            max_live_count=context_len,
            storage_dtype=DType.BF16,
            span_role="prefill",
        )
        common = (spans, context_len, context_len, block_size, kv_heads, head_dim)
        qwen35_copy_paged_kv_bf16_to_head_major_spans(
            key_buf.ptr,
            value_buf.ptr,
            generic_key_buf.ptr,
            generic_value_buf.ptr,
            *common,
            library=library,
            runtime=runtime,
        )
        qwen35_copy_paged_kv_bf16_to_head_major_dense_prefix_spans(
            key_buf.ptr,
            value_buf.ptr,
            dense_key_buf.ptr,
            dense_value_buf.ptr,
            *common,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        for host, device in (
            (generic_key, generic_key_buf),
            (generic_value, generic_value_buf),
            (dense_key, dense_key_buf),
            (dense_value, dense_value_buf),
        ):
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
        assert np.array_equal(generic_key, dense_key)
        assert np.array_equal(generic_value, dense_value)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
