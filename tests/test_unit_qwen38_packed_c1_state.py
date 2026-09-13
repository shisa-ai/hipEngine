"""CPU tests for pre-acceptance resident-state isolation instrumentation."""
from types import SimpleNamespace as NS

import numpy as np
import pytest


def _session():
    def buf(ptr, nbytes):
        return NS(ptr=ptr, nbytes=nbytes)
    return NS(position=2, runtime=NS(device_synchronize=lambda: None),
              runner=NS(weights=NS(config=NS(head_count_kv=1, key_length=2))),
              scratch=NS(layer_conv_states=[buf(10, 8), None],
                         layer_recurrent_states=[buf(20, 16), None],
                         full_key_caches=[None, buf(30, 32)],
                         full_value_caches=[None, buf(40, 32)],
                         hidden_seed_fp32=buf(50, 16),
                         position_host=np.array([2]), context_host=np.array([3]),
                         position_buf=buf(60, 8), context_buf=buf(70, 8)))


@pytest.fixture(autouse=True)
def mock_kv_hash(monkeypatch):
    from scripts import qwen38_packed_c1_kv as kv
    monkeypatch.setattr(kv, '_device_hash', lambda session, buffer: str((buffer.ptr, buffer.nbytes)))


@pytest.mark.parametrize('layout', ['slot', 'pages'])
def test_snapshot_tracks_physical_live_kv_and_page_ownership(monkeypatch, layout):
    from scripts import qwen38_packed_c1_state as module
    monkeypatch.setattr(module, '_device_hash', lambda *a, **kw: 'hash')
    s = _session()
    for b in (s.scratch.full_key_caches[1], s.scratch.full_value_caches[1]):
        b.nbytes = 4096
    if layout == 'slot':
        s._resident_slot_index = 2
        s._resident_batch_owner = NS(_target_scratch_owner=NS(max_positions=256))
        rows = (512, 513)
    else:
        s.position = 257
        s._device_kv_allocation = NS(block_ids=(12, 10), chunk_start_block_id=10)
        rows = tuple(range(512, 768)) + (0,)
    before = module.snapshot_committed_state(s)
    assert before['buffers']['key:1']['physical_rows'] == rows
    assert before['buffers']['key:1']['checked_nbytes'] == len(rows) * 4
    if layout == 'slot':
        s._resident_slot_index = 1
    else:
        s._device_kv_allocation.block_ids = (10, 12)
    with pytest.raises(ValueError, match='buffers'):
        module.assert_committed_state_unchanged(before, module.snapshot_committed_state(s))


@pytest.mark.parametrize('row,changed', [(512, True), (513, True), (514, False), (0, False)])
def test_snapshot_detects_live_slot_bytes_only(monkeypatch, row, changed):
    from scripts import qwen38_packed_c1_state as module
    from scripts import qwen38_packed_c1_kv as kv
    monkeypatch.setattr(module, '_device_hash', lambda *a, **kw: 'hash')
    s = _session()
    s._resident_slot_index = 2
    s._resident_batch_owner = NS(_target_scratch_owner=NS(max_positions=256))
    for b in (s.scratch.full_key_caches[1], s.scratch.full_value_caches[1]):
        b.nbytes = 4096
    memory = {}
    monkeypatch.setattr(kv, '_device_hash', lambda session, b:
                        tuple(memory.get(i, 0) for i in range(b.ptr, b.ptr + b.nbytes)))
    before = module.snapshot_committed_state(s)
    memory[30 + row * 4] = 1
    after = module.snapshot_committed_state(s)
    if changed:
        with pytest.raises(ValueError, match='buffers'):
            module.assert_committed_state_unchanged(before, after)
    else:
        module.assert_committed_state_unchanged(before, after)


def test_snapshot_checks_all_committed_surfaces_and_only_live_kv(monkeypatch):
    from scripts import qwen38_packed_c1_state as module
    calls = []
    def digest(session, buffer, *, nbytes=None):
        calls.append((buffer.ptr, nbytes))
        return str((buffer.ptr, nbytes))
    monkeypatch.setattr(module, "_device_hash", digest)
    result = module.snapshot_committed_state(_session())
    assert result["position"] == 2
    assert calls == [(10, None), (20, None), (50, None), (60, None), (70, None)]
    assert result['buffers']['key:1']['blake2b_128'] == ('(30, 8)',)
    assert result['buffers']['value:1']['blake2b_128'] == ('(40, 8)',)
    assert len(result["buffers"]) == 7


@pytest.mark.parametrize("fault", ["short_kv", "missing_conv", "missing_kv", "empty", "negative"])
def test_snapshot_rejects_incomplete_or_invalid_surfaces(monkeypatch, fault):
    from scripts import qwen38_packed_c1_state as module
    monkeypatch.setattr(module, "_device_hash", lambda *a, **kw: "hash")
    session = _session()
    if fault == "short_kv":
        session.scratch.full_key_caches[1].nbytes = 4
    elif fault == "missing_conv":
        session.scratch.layer_conv_states[0] = None
    elif fault == "missing_kv":
        session.scratch.full_value_caches[1] = None
    elif fault == "empty":
        session.scratch.layer_recurrent_states[0].nbytes = 0
    else:
        session.position = -1
    with pytest.raises(ValueError):
        module.snapshot_committed_state(session)


@pytest.mark.parametrize("field", ["position", "position_host", "context_host", "buffers"])
def test_guard_rejects_each_changed_surface(field):
    from scripts.qwen38_packed_c1_state import assert_committed_state_unchanged
    before = dict(position=2, position_host=[2], context_host=[3], buffers={"conv": "a"})
    after = dict(before)
    after[field] = "changed"
    with pytest.raises(ValueError, match=field):
        assert_committed_state_unchanged(before, after)
    assert_committed_state_unchanged(before, before.copy())
