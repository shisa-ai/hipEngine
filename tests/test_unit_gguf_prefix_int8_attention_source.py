"""Resumed INT8 prefill must not attend over an unwritten BF16 oracle."""

from __future__ import annotations

import ctypes
import inspect
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata
from hipengine.runtime import qwen35_gguf_runner as gguf


@dataclass(frozen=True)
class _Scratch:
    append_spans: KVLiveSpans
    prefill_spans: KVLiveSpans
    start: int
    rows: int = 1
    key_cache: DeviceBuffer | None = None
    value_cache: DeviceBuffer | None = None
    retained_key_cache: DeviceBuffer | None = None
    retained_value_cache: DeviceBuffer | None = None
    retained_append_spans: KVLiveSpans | None = None
    retained_decode_spans: KVLiveSpans | None = None
    retained_decode_kernel: object | None = None
    int8_kv_value_bf16: bool = False


def _tensor(array: np.ndarray, dtype: DType) -> Tensor:
    return Tensor.from_handle(array.ctypes.data, array.shape, dtype, Device("hip", 0))


def _buffer(array: np.ndarray) -> DeviceBuffer:
    return DeviceBuffer(array.ctypes.data, array.nbytes)


def _fixture(monkeypatch, *, selector: str, blocks: tuple[int, ...], start: int, mirrored: bool):
    shape = (4, 256, 1, 256)
    key = np.zeros(shape, dtype=np.int8)
    value = np.zeros(shape, dtype=np.int8)
    key_scale = np.full(shape[:3], np.nan, dtype=np.float32)
    value_scale = np.full(shape[:3], np.nan, dtype=np.float32)
    oracle = tuple(np.full(shape, 0x7FC1, dtype=np.uint16) for _ in range(2))
    mirror = tuple(np.full(shape, 0x7FC1, dtype=np.uint16) for _ in range(2))
    # Two distinguishable prefix pages; every other page remains invalid.
    for page, k, v, ks, vs, kb, vb in (
        (blocks[0], 4, -8, 0.5, 0.25, 0x4000, 0xC000),
        (blocks[1], 8, 12, 1.25, 0.5, 0x4120, 0x40C0),
    ):
        key[page], value[page] = k, v
        key_scale[page], value_scale[page] = ks, vs
        mirror[0][page], mirror[1][page] = kb, vb
    metadata = KVScaleMetadata(
        k_scale=_tensor(key_scale, DType.FP32),
        v_scale=_tensor(value_scale, DType.FP32),
        scale_dtype=DType.FP32,
    )
    retained = (_buffer(key), _buffer(value))
    oracle_pair = tuple(_buffer(array) for array in oracle)
    mirror_pair = tuple(_buffer(array) for array in mirror) if mirrored else None
    table = np.asarray([blocks if selector == "packed" else (0, 1, 2)], dtype=np.int32)
    position = np.asarray([start], dtype=np.int64)
    count = position + 1
    append = KVLiveSpans.paged_uniform(
        block_table=_tensor(table, DType.INT32),
        live_counts=_tensor(position, DType.INT64),
        max_live_count=start,
        row_positions=_tensor(position, DType.INT64),
        storage_dtype=DType.BF16,
        span_role="prefill",
    )
    scratch = _Scratch(
        append_spans=append,
        prefill_spans=replace(
            append, live_counts=_tensor(count, DType.INT64), max_live_count=start + 1,
        ),
        start=start,
    )
    state = SimpleNamespace(
        full_cache=lambda layer: retained,
        full_scale_metadata=lambda layer: metadata,
        full_bf16_mirror_cache=lambda layer: mirror_pair,
        kv_layout=SimpleNamespace(int8_kv_value_bf16=False),
    )
    allocations = []

    def malloc(nbytes, *, runtime):
        array = np.full(nbytes, 0xCD, dtype=np.uint8)
        allocations.append(array)
        return _buffer(array)

    def upload(buffer, host_ptr, nbytes, *, runtime):
        ctypes.memmove(buffer.ptr, host_ptr, nbytes)

    monkeypatch.setattr(gguf, "malloc", malloc)
    monkeypatch.setattr(gguf, "copy_host_to_device", upload)
    session = object.__new__(gguf.Qwen35GGUFResidentSession)
    session.__dict__.update(
        scratch=state,
        runtime=SimpleNamespace(),
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        int8_kv_value_bf16=False,
        dms_prefill_mode="off",
        _position=start,
        _device_kv_allocation=SimpleNamespace(block_ids=blocks, chunk_start_block_id=0),
        _int8_prefill_retained_block_table=None,
        _int8_prefill_lifetime_plan=SimpleNamespace(mode="layer_outer_shared_oracle"),
        _int8_prefill_oracle_cache_for_layer=lambda layer: oracle_pair,
    )
    return SimpleNamespace(
        session=session, state=state, scratch=scratch, metadata=metadata,
        retained=retained, oracle=oracle_pair, mirror=mirror_pair,
        # Keep every host allocation alive while its fake device pointer is used.
        owners=(key, value, key_scale, value_scale, oracle, mirror, table, position, count, allocations),
    )


def _select(fixture, selector: str, *, resumed: bool):
    if selector == "packed":
        method = fixture.session._packed_full_attention_scratch_for_layer
        args = (fixture.scratch, fixture.state, 0)
        kwargs = {"allow_direct_int8_prefill": True}
    else:
        method = fixture.session._full_attention_prefill_scratch_for_layer
        args = (fixture.scratch, 0)
        kwargs = {}
    # Exercise the old selector for a numerical RED, rather than stopping at
    # TypeError before the proposed per-call keyword is implemented.
    if resumed and "use_retained_prefix" in inspect.signature(method).parameters:
        kwargs["use_retained_prefix"] = True
    return method(*args, **kwargs)


def _prefix_samples(scratch) -> np.ndarray:
    spans = scratch.prefill_spans
    table = np.ctypeslib.as_array(
        (ctypes.c_int32 * spans.base_offsets.numel).from_address(spans.base_offsets.ptr)
    )
    samples = []
    for logical_page in (0, 1):
        physical_row = int(table[logical_page]) * 256
        pair = []
        for buffer, scale in (
            (scratch.key_cache, None if spans.scale_metadata is None else spans.scale_metadata.k_scale),
            (scratch.value_cache, None if spans.scale_metadata is None else spans.scale_metadata.v_scale),
        ):
            if spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD:
                quant = ctypes.c_int8.from_address(buffer.ptr + physical_row * 256).value
                factor = ctypes.c_float.from_address(scale.ptr + physical_row * 4).value
                pair.append(quant * factor)
            else:
                bits = ctypes.c_uint16.from_address(buffer.ptr + physical_row * 256 * 2).value
                pair.append(np.asarray([bits << 16], dtype=np.uint32).view(np.float32)[0])
        samples.append(pair)
    return np.asarray(samples)


@pytest.mark.parametrize("selector", ["slot_local", "packed"])
@pytest.mark.parametrize("blocks", [(1, 2, 3), (1, 3, 2)], ids=["contiguous", "gapped"])
def test_resumed_no_mirror_prefix_reads_retained_int8_and_fp32_scales(monkeypatch, selector, blocks):
    fixture = _fixture(monkeypatch, selector=selector, blocks=blocks, start=512, mirrored=False)
    selected = _select(fixture, selector, resumed=True)

    np.testing.assert_array_equal(
        _prefix_samples(selected), [[2.0, -2.0], [10.0, 6.0]],
        err_msg="resumed prefix attention selected unwritten oracle rows or wrong payload/scale pages",
    )
    assert (selected.key_cache, selected.value_cache) == fixture.retained
    for spans in (selected.append_spans, selected.prefill_spans):
        assert spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
        assert spans.scale_metadata is fixture.metadata
        assert spans.scale_metadata.scale_dtype == DType.FP32
        table = np.ctypeslib.as_array(
            (ctypes.c_int32 * spans.base_offsets.numel).from_address(spans.base_offsets.ptr)
        )
        np.testing.assert_array_equal(table, blocks)
    assert selected.append_spans.max_live_count == 512
    assert selected.prefill_spans.max_live_count == 513
    assert selected.retained_key_cache is None
    assert selected.retained_value_cache is None
    assert selected.retained_append_spans is None


@pytest.mark.parametrize("selector", ["slot_local", "packed"])
@pytest.mark.parametrize("blocks", [(1, 2, 3), (1, 3, 2)], ids=["contiguous", "gapped"])
def test_cold_no_mirror_prefill_preserves_bf16_oracle(monkeypatch, selector, blocks):
    fixture = _fixture(monkeypatch, selector=selector, blocks=blocks, start=0, mirrored=False)
    selected = _select(fixture, selector, resumed=False)

    assert (selected.key_cache, selected.value_cache) == fixture.oracle
    assert selected.prefill_spans.storage_dtype == DType.BF16
    assert selected.prefill_spans.scale_metadata is None
    assert (selected.retained_key_cache, selected.retained_value_cache) == fixture.retained
    assert selected.retained_append_spans.scale_metadata is fixture.metadata


@pytest.mark.parametrize("selector", ["slot_local", "packed"])
@pytest.mark.parametrize("blocks", [(1, 2, 3), (1, 3, 2)], ids=["contiguous", "gapped"])
def test_resumed_mirrored_prefill_preserves_bf16_mirror(monkeypatch, selector, blocks):
    fixture = _fixture(monkeypatch, selector=selector, blocks=blocks, start=512, mirrored=True)
    selected = _select(fixture, selector, resumed=True)

    assert (selected.key_cache, selected.value_cache) == fixture.mirror
    assert selected.prefill_spans.storage_dtype == DType.BF16
    assert selected.prefill_spans.scale_metadata is None
    assert (selected.retained_key_cache, selected.retained_value_cache) == fixture.retained
    assert selected.retained_append_spans.scale_metadata is fixture.metadata


@pytest.mark.parametrize("selector", ["slot_local", "packed"])
def test_retained_prefix_rejects_key_only_before_selecting_two_int8_planes(monkeypatch, selector):
    fixture = _fixture(monkeypatch, selector=selector, blocks=(1, 3, 2), start=512, mirrored=False)
    fixture.session.int8_kv_value_bf16 = True
    fixture.state.kv_layout.int8_kv_value_bf16 = True
    with pytest.raises(NotImplementedError, match="INT8 K and V"):
        _select(fixture, selector, resumed=True)


def test_retained_prefix_rejects_block16_before_any_bf16_write():
    with pytest.raises(NotImplementedError, match="block16"):
        gguf._check_retained_prefix_format(SimpleNamespace(granularity="block16"), value_bf16=False)


def test_default_direct_prefill_uses_supported_geometry(monkeypatch):
    monkeypatch.delenv("HIPENGINE_GGUF_INT8_PREFILL_KERNEL", raising=False)
    assert gguf._gguf_int8_prefill_kernel(geometry=(16, 2, 256)) == "sequential"
    assert gguf._gguf_int8_prefill_kernel(geometry=(24, 4, 256)) == "flash"


def test_fp16_scales_do_not_select_fp32_only_flash_reader(monkeypatch):
    monkeypatch.delenv("HIPENGINE_GGUF_INT8_PREFILL_KERNEL", raising=False)
    assert gguf._gguf_int8_prefill_kernel(
        geometry=(24, 4, 256), scale_dtype=DType.FP16,
    ) == "sequential"
    monkeypatch.setenv("HIPENGINE_GGUF_INT8_PREFILL_KERNEL", "flash")
    with pytest.raises(ValueError, match="fp32"):
        gguf._gguf_int8_prefill_kernel(geometry=(24, 4, 256), scale_dtype=DType.FP16)
