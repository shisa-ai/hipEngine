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
    _GGUF_AOTRITON_HEAD_MAJOR_KV_ENV,
    _GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS_ENV,
    _gguf_aotriton_head_major_buffers,
    _gguf_aotriton_head_major_kv_enabled,
    _try_allocate_gguf_aotriton_head_major_kv_scratch,
)
from scripts.qwen35_gguf_bench import _session_buffer_breakdown


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


def test_head_major_policy_defaults_only_on_gfx1151_and_keeps_env_rollback(monkeypatch) -> None:
    monkeypatch.delenv(_GGUF_AOTRITON_HEAD_MAJOR_KV_ENV, raising=False)
    assert _gguf_aotriton_head_major_kv_enabled("hip_gfx1151") is True
    assert _gguf_aotriton_head_major_kv_enabled("hip_gfx1100") is False

    monkeypatch.setenv(_GGUF_AOTRITON_HEAD_MAJOR_KV_ENV, "0")
    assert _gguf_aotriton_head_major_kv_enabled("hip_gfx1151") is False
    monkeypatch.setenv(_GGUF_AOTRITON_HEAD_MAJOR_KV_ENV, "1")
    assert _gguf_aotriton_head_major_kv_enabled("hip_gfx1100") is True


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

    monkeypatch.setenv(_GGUF_AOTRITON_HEAD_MAJOR_KV_ENV, "1")
    monkeypatch.setattr(runner_module, "malloc", fake_malloc)
    monkeypatch.setattr(
        runner_module,
        "free",
        lambda buffer, *, runtime: freed.append(buffer),
    )

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


def test_head_major_scratch_rejects_capacity_above_validated_64k_class(monkeypatch) -> None:
    import hipengine.runtime.qwen35_gguf_runner as runner_module

    monkeypatch.setenv(_GGUF_AOTRITON_HEAD_MAJOR_KV_ENV, "1")
    monkeypatch.setenv(_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS_ENV, "65792")
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
            runtime=object(),
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
