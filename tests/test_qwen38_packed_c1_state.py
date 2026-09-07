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


def test_snapshot_checks_all_committed_surfaces_and_only_live_kv(monkeypatch):
    from scripts import qwen38_packed_c1_state as module
    calls = []
    def digest(session, buffer, *, nbytes=None):
        calls.append((buffer.ptr, nbytes))
        return str((buffer.ptr, nbytes))
    monkeypatch.setattr(module, "_device_hash", digest)
    result = module.snapshot_committed_state(_session())
    assert result["position"] == 2
    assert calls == [(10, None), (20, None), (30, 8), (40, 8),
                     (50, None), (60, None), (70, None)]
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
