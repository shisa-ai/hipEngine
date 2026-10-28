"""Host-only slot geometry and lifetime contracts for resident tensor views."""

from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession


def _session(dtype):
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = Device("hip", 0)
    session.max_batch_size = 4
    session.blocks = 3
    session.block_size = 16
    session.config = SimpleNamespace(num_key_value_heads=2, head_dim=8)
    conv = Tensor.from_handle(0x10000, (12, 4), DType.FP32, session.device)
    recurrent = Tensor.from_handle(0x20000, (2, 4, 8), DType.FP32, session.device)
    session.linear_states = {1: (conv, recurrent, DeviceBuffer(conv.ptr, 4 * conv.numel * 4), DeviceBuffer(recurrent.ptr, 4 * recurrent.numel * 4), None, None)}
    shape = (3, 16, 2, 8)
    key = Tensor.from_handle(0x30000, shape, dtype, session.device)
    value = Tensor.from_handle(0x40000, shape, dtype, session.device)
    session.full_caches = {2: (key, value, DeviceBuffer(key.ptr, 4 * key.numel * dtype.itemsize), DeviceBuffer(value.ptr, 4 * value.numel * dtype.itemsize))}
    session._slot_linear_state_cache = {}
    session._slot_full_cache_cache = {}
    session._full_cache_all_slots_cache = {}
    return session


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
@pytest.mark.parametrize("dtype", [DType.BF16, DType.INT8, DType.FP32])
@pytest.mark.parametrize("slot", [0, 1, 3])
def test_resident_views_always_cache_exact_geometry(monkeypatch, retired, dtype, slot):
    if retired is None:
        monkeypatch.delenv("HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE", retired)
    session = _session(dtype)
    # Even partially constructed owners without the retired construction-time
    # selector must use the same cache behavior as ordinary resident sessions.
    linear = session._slot_linear_state(1, slot)
    full = session._slot_full_cache(2, slot)
    all_slots = session._full_cache_all_slots(2)
    assert session._slot_linear_state(1, slot) is linear
    assert session._slot_full_cache(2, slot) is full
    assert session._full_cache_all_slots(2) is all_slots
    for view, source in zip(linear, session.linear_states[1][:2], strict=True):
        assert view.ptr == source.ptr + slot * source.numel * source.dtype.itemsize
        assert view.shape == source.shape and view.dtype == source.dtype and view.device == source.device
    for view, source, combined in zip(full, session.full_caches[2][:2], all_slots, strict=True):
        assert view.ptr == source.ptr + slot * source.numel * source.dtype.itemsize
        assert view.shape == source.shape and view.dtype == source.dtype and view.device == source.device
        assert combined.ptr == source.ptr
        assert combined.shape == (12, 16, 2, 8)
        assert combined.dtype == source.dtype and combined.device == source.device
    assert set(session._slot_linear_state_cache) == {(1, slot)}
    assert set(session._slot_full_cache_cache) == {(2, slot)}
    assert set(session._full_cache_all_slots_cache) == {2}


@pytest.mark.parametrize("method,layer", [("_slot_linear_state", 1), ("_slot_full_cache", 2)])
def test_resident_cached_views_still_check_slot_capacity(method, layer):
    session = _session(DType.BF16)
    call = getattr(session, method)
    call(layer, 3)
    session.max_batch_size = 2
    with pytest.raises(ValueError, match="outside batch capacity"):
        call(layer, 3)
    with pytest.raises(ValueError, match="outside batch capacity"):
        call(layer, -1)


def test_resident_clear_rebuilds_views_for_new_pointers_and_capacity():
    session = _session(DType.BF16)
    before = (session._slot_linear_state(1, 1), session._slot_full_cache(2, 1), session._full_cache_all_slots(2))
    conv, recurrent, *_ = session.linear_states[1]
    session.linear_states[1] = (conv, recurrent, DeviceBuffer(0x50000, 8192), DeviceBuffer(0x60000, 8192), None, None)
    key, value, *_ = session.full_caches[2]
    session.full_caches[2] = (key, value, DeviceBuffer(0x70000, 8192), DeviceBuffer(0x80000, 8192))
    session.max_batch_size = 2
    session.blocks = 5
    session._clear_resident_tensor_view_caches()
    assert not session._slot_linear_state_cache and not session._slot_full_cache_cache and not session._full_cache_all_slots_cache
    after = (session._slot_linear_state(1, 1), session._slot_full_cache(2, 1), session._full_cache_all_slots(2))
    assert all(new is not old for old, new in zip(before, after, strict=True))
    assert after[0][0].ptr == 0x50000 + conv.numel * conv.dtype.itemsize
    assert after[0][1].ptr == 0x60000 + recurrent.numel * recurrent.dtype.itemsize
    assert after[1][0].ptr == 0x70000 + key.numel * key.dtype.itemsize
    assert after[1][1].ptr == 0x80000 + value.numel * value.dtype.itemsize
    assert after[2][0].ptr == 0x70000 and after[2][1].ptr == 0x80000
    assert after[2][0].shape == after[2][1].shape == (10, 16, 2, 8)


def test_resident_views_do_not_cross_slots_or_layers():
    session = _session(DType.BF16)
    session.linear_states[5] = session.linear_states[1]
    session.full_caches[6] = session.full_caches[2]
    for method, layers in ((session._slot_linear_state, (1, 5)), (session._slot_full_cache, (2, 6))):
        first = method(layers[0], 0)
        assert method(layers[1], 0) is not first
        assert method(layers[0], 1) is not first
        assert method(layers[0], 0) is first
    assert session._full_cache_all_slots(2) is not session._full_cache_all_slots(6)


def test_sequence_capacity_change_clears_all_resident_views():
    session = _session(DType.BF16)
    session.config.hidden_size = 8
    session.chunk_size = 16
    session._resolve_prefill_config_for_length = lambda length: None
    session._slot_linear_state(1, 0)
    session._slot_full_cache(2, 0)
    session._full_cache_all_slots(2)
    session._set_sequence_capacity(113)
    assert session.max_sequence_length == 113
    assert not session._slot_linear_state_cache
    assert not session._slot_full_cache_cache
    assert not session._full_cache_all_slots_cache
    assert session._full_cache_all_slots(2)[0].shape[0] == session.max_batch_size * session.blocks
