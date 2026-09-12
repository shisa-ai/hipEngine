from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def _session(*, has_dms: bool) -> Qwen35GGUFResidentSession:
    """Build a bare session with just the fields the marker refresh touches."""

    session = object.__new__(Qwen35GGUFResidentSession)
    runner = SimpleNamespace()
    session.runner = runner
    session._dms_backend = SimpleNamespace() if has_dms else None
    return session


def test_dms_session_step_claims_decode_owner_marker() -> None:
    session = _session(has_dms=True)
    other = _session(has_dms=True)
    session.runner.__dict__["_dms_decode_owner"] = other
    session._refresh_dms_decode_owner_marker()
    assert session.runner.__dict__["_dms_decode_owner"] is session


def test_dense_session_step_clears_foreign_decode_owner_marker() -> None:
    session = _session(has_dms=False)
    dms_session = _session(has_dms=True)
    session.runner.__dict__["_dms_decode_owner"] = dms_session
    session._refresh_dms_decode_owner_marker()
    assert "_dms_decode_owner" not in session.runner.__dict__


def test_marker_refresh_is_noop_without_prior_marker() -> None:
    session = _session(has_dms=False)
    session._refresh_dms_decode_owner_marker()
    assert "_dms_decode_owner" not in session.runner.__dict__


def test_dms_session_step_overwrites_stale_self_marker() -> None:
    # Two DMS sessions interleaving steps must each claim the marker for
    # their own step; the previous owner (another DMS session) is replaced.
    first = _session(has_dms=True)
    second = _session(has_dms=True)
    first.runner.__dict__["_dms_decode_owner"] = second
    first._refresh_dms_decode_owner_marker()
    assert first.runner.__dict__["_dms_decode_owner"] is first


@pytest.mark.parametrize("has_dms", [True, False])
def test_marker_refresh_tolerates_missing_runner(has_dms: bool) -> None:
    session = _session(has_dms=has_dms)
    session.runner = None
    # Must not raise; a session without a runner has nothing to refresh.
    session._refresh_dms_decode_owner_marker()
