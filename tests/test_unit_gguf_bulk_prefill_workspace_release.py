"""Bulk prefill workspace must be releasable and lazily re-acquirable.

Memory review target 3 (second half): the bulk prefill scratch, prefill
hidden buffers and prefill token buffer are dead once prefill compute has
completed. ``_finalize_external_dms_prefill`` releases them before the DMS
compact pack so they do not coexist with the dense BF16 pool and the compact
destination, and every bulk-prefill entry re-acquires the workspace lazily.

Contract under test (CPU, fake device):
- ``_allocate_bulk_prefill_workspace`` appends the workspace buffers to
  ``_buffers`` exactly once and is idempotent;
- ``_release_bulk_prefill_workspace`` frees every workspace buffer (scratch,
  head-major, hidden, token), prunes them from ``_buffers``, and is refused
  while resident slot views exist;
- ``_ensure_bulk_prefill_workspace`` re-acquires after release and raises on
  closed sessions; slot views delegate to their batch owner.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

from tests.test_unit_qwen35_gguf_prefill_scratch_liveness import (
    _fake_dense_qwen36_runner,
    _install_fake_device,
)


def _fake_session(monkeypatch, *, capacity: int = 73_728):
    _install_fake_device(monkeypatch)
    freed: list[object] = []
    real_free = gguf_runner.free

    def tracking_free(buffer, *, runtime):
        real_free_fake = freed.append
        real_free_fake(buffer)
        return None

    monkeypatch.setattr(gguf_runner, "free", tracking_free)
    session = object.__new__(Qwen35GGUFResidentSession)
    session.__dict__.update(
        runner=_fake_dense_qwen36_runner(),
        runtime=SimpleNamespace(),
        scratch=SimpleNamespace(max_positions=capacity),
        backend="hip_gfx1100",
        use_expert_sidecar=False,
        _runtime_state_library=None,
        prefill_chunk_tuning={},
        _int8_prefill_lifetime_plan=SimpleNamespace(required_hidden_capacity=capacity),
        _buffers=(),
        _prefill_token_buf=None,
        _prefill_hidden_a=None,
        _prefill_hidden_b=None,
        _bulk_prefill_scratch=None,
    )
    session._prefill_scratch_rows = lambda capacity: 768  # noqa: ARG001
    return session, freed


def _workspace_buffers(session) -> tuple:
    return (
        *session._bulk_prefill_scratch.buffers,
        session._prefill_token_buf,
        session._prefill_hidden_a,
    )


def test_allocate_is_idempotent_and_appends_buffers(monkeypatch) -> None:
    session, _ = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert session._bulk_prefill_scratch is not None
    assert session._prefill_token_buf is not None
    assert session._prefill_hidden_a is not None
    buffers = session._buffers
    assert all(buffer in buffers for buffer in _workspace_buffers(session))
    # Idempotent: a second call does not re-allocate or duplicate buffers.
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert session._buffers == buffers


def test_release_frees_workspace_and_prunes_buffers(monkeypatch) -> None:
    session, freed = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    base_buffers = session._buffers
    workspace = _workspace_buffers(session)
    # The token buffer/hidden planes are the only buffers beyond the base set
    # built before the workspace; simulate the base set without them.
    session._release_bulk_prefill_workspace()
    assert session._bulk_prefill_scratch is None
    assert session._prefill_token_buf is None
    assert session._prefill_hidden_a is None
    assert session._prefill_hidden_b is None
    assert all(buffer in freed for buffer in workspace)
    assert not any(buffer in session._buffers for buffer in workspace)
    assert len(freed) == len({id(buffer) for buffer in freed}), "no double free"
    assert len(base_buffers) > len(session._buffers)
    # Release is a no-op when already released.
    session._release_bulk_prefill_workspace()
    assert len(freed) == len({id(buffer) for buffer in freed})


def test_release_refused_while_slot_views_exist(monkeypatch) -> None:
    session, freed = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    session._resident_slot_views = (object(),)
    session._release_bulk_prefill_workspace()
    assert session._bulk_prefill_scratch is not None
    assert freed == []
    del session._resident_slot_views


def test_ensure_reacquires_and_raises_when_closed(monkeypatch) -> None:
    session, _ = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    first = session._bulk_prefill_scratch
    session._release_bulk_prefill_workspace()
    session._ensure_bulk_prefill_workspace()
    assert session._bulk_prefill_scratch is not None
    assert session._bulk_prefill_scratch is not first
    assert all(
        buffer in session._buffers for buffer in _workspace_buffers(session)
    )
    # Closed sessions fail closed.
    session._release_bulk_prefill_workspace()
    session.runner = None
    with pytest.raises(RuntimeError, match="closed"):
        session._ensure_bulk_prefill_workspace()


def test_slot_view_ensure_delegates_to_owner(monkeypatch) -> None:
    owner, _ = _fake_session(monkeypatch)
    owner._allocate_bulk_prefill_workspace(SimpleNamespace())
    owner._release_bulk_prefill_workspace()
    view, _ = _fake_session(monkeypatch)
    view._resident_batch_owner = owner
    view._ensure_bulk_prefill_workspace()
    assert view._bulk_prefill_scratch is owner._bulk_prefill_scratch
    assert view._prefill_hidden_b is owner._prefill_hidden_b
