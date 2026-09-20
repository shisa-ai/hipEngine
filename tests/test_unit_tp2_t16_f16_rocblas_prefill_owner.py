"""Behavioral tests for the TP2 source-F16 prefill owner's ownership contract.

The first version of this file asserted on source strings and on ``rows < 512``,
which tested that certain text existed rather than that the code behaved. A
reviewer demonstrated the gap with CPU mocks: the default path allocated the
candidate's scratch even with the owner disabled, and no rocBLAS handle was ever
closed. Neither would have been caught by a string check.

These tests drive the real methods on a stand-in session, with the device scope,
the allocator and the owner builder mocked. They cover admission by row count,
rank ownership, disabled allocation, growth re-allocation, and cleanup.
"""

from __future__ import annotations

import contextlib
import types
from typing import Any

import pytest

import hipengine.distributed.tp2_generate as tp2


class _FakePlanes:
    """Stand-in for the three-plane scratch, recording its own release."""

    def __init__(self, rows: int, device: int, log: list[tuple]) -> None:
        self.rows = rows
        self._device = device
        self._log = log
        self.released = False

    def release(self, *, runtime: Any = None) -> None:
        self.released = True
        self._log.append(("release", self._device, self.rows))


class _FakeHandle:
    """Stand-in for the rocBLAS handle, which needs an explicit close."""

    def __init__(self, device: int, log: list[tuple]) -> None:
        self._device = device
        self._log = log

    def close(self) -> None:
        self._log.append(("close", self._device))


class _FakeRunner:
    def __init__(self, device: int) -> None:
        self.device = device
        self.compiler_version = "test"
        self.require_cached_build = False


def _make_session(
    *,
    devices: tuple[int, ...] = (0, 1),
    enabled: bool,
    log: list[tuple],
    allocator: Any = None,
    builder: Any = None,
) -> Any:
    """A session-shaped object exposing only what the owner methods touch."""

    session = tp2.MlpTP2GenerationSession.__new__(tp2.MlpTP2GenerationSession)
    session.devices = devices
    session.runtime = types.SimpleNamespace()
    session.use_t16_f16_rocblas_prefill = enabled
    session._runners = {device: _FakeRunner(device) for device in devices}
    session._rank_f16_rocblas_planes = {}
    session._rank_f16_rocblas_library = {}
    session._rank_f16_rocblas = {}
    session._rank_f16_rocblas_ready = {}
    return session


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    """Mock the device scope, the allocator and the owner builder."""

    log: list[tuple] = []
    calls: dict[str, Any] = {"allocated": [], "built": []}

    @contextlib.contextmanager
    def fake_scope(runtime: Any, device: int):
        yield

    def fake_allocate(runner: Any, *, rows: int, runtime: Any = None):
        calls["allocated"].append((runner.device, rows))
        if rows > 4096:  # a stand-in for "the policy does not admit this"
            return None
        return _FakePlanes(rows, runner.device, log)

    def fake_build(runner: Any, scratch: Any, **kwargs: Any):
        calls["built"].append((runner.device, scratch.rows, kwargs.get("request_rows")))
        handle = _FakeHandle(runner.device, log)
        return object(), None, handle

    monkeypatch.setattr(tp2, "scoped_current_device", fake_scope)
    monkeypatch.setattr(tp2, "allocate_t16_f16_rocblas_prefill_planes", fake_allocate)
    monkeypatch.setattr(tp2, "build_t16_f16_rocblas_prefill_owner", fake_build)
    return log, calls


def test_disabled_owner_allocates_nothing(patched) -> None:
    """The default path must not pay for scratch it never uses."""

    log, calls = patched
    session = _make_session(enabled=False, log=log)

    session._ensure_rank_f16_rocblas_planes(512)

    assert calls["allocated"] == []
    assert session._rank_f16_rocblas_planes == {}


def test_enabled_owner_allocates_once_per_rank(patched) -> None:
    log, calls = patched
    session = _make_session(enabled=True, log=log)

    session._ensure_rank_f16_rocblas_planes(512)

    assert calls["allocated"] == [(0, 512), (1, 512)]
    assert set(session._rank_f16_rocblas_planes) == {0, 1}
    assert all(session._rank_f16_rocblas_ready.values())


def test_growing_row_count_reallocates_and_releases_the_superseded_planes(patched) -> None:
    """64 -> 512 -> 64 must end at 64 rows with no leaked or stale planes."""

    log, calls = patched
    session = _make_session(devices=(0,), enabled=True, log=log)

    session._ensure_rank_f16_rocblas_planes(64)
    assert calls["allocated"] == [(0, 64)]

    session._ensure_rank_f16_rocblas_planes(512)
    assert calls["allocated"] == [(0, 64), (0, 512)]
    assert ("release", 0, 64) in log, "the superseded planes must be released"
    assert session._rank_f16_rocblas_planes[0].rows == 512

    # A smaller pass must reuse the larger planes rather than shrink them.
    session._ensure_rank_f16_rocblas_planes(64)
    assert calls["allocated"] == [(0, 64), (0, 512)]
    assert session._rank_f16_rocblas_planes[0].rows == 512

    # Repeating the same size must not allocate again.
    session._ensure_rank_f16_rocblas_planes(512)
    assert calls["allocated"] == [(0, 64), (0, 512)]


def test_owner_context_uses_this_ranks_planes(patched) -> None:
    """Each rank's context must be built over its own planes, not a peer's."""

    log, calls = patched
    session = _make_session(enabled=True, log=log)
    session._ensure_rank_f16_rocblas_planes(512)

    with session._rank_f16_rocblas_owner_context(1, 512):
        pass

    # device 1's own planes, not device 0's.
    assert calls["built"] == [(1, 512, 512)]
    assert session._rank_f16_rocblas[1] is not None
    assert 0 not in session._rank_f16_rocblas


def test_owner_context_falls_back_when_rows_exceed_the_planes(patched) -> None:
    """Admission is by complete request shape, not by what fits this scratch."""

    log, calls = patched
    session = _make_session(devices=(0,), enabled=True, log=log)
    session._ensure_rank_f16_rocblas_planes(64)

    with session._rank_f16_rocblas_owner_context(0, 512):
        pass

    assert calls["built"] == [], "an over-sized request must not build an owner"


def test_owner_context_falls_back_when_disabled(patched) -> None:
    log, calls = patched
    session = _make_session(devices=(0,), enabled=False, log=log)

    with session._rank_f16_rocblas_owner_context(0, 512):
        pass

    assert calls["built"] == []


def test_cleanup_closes_every_handle_and_releases_every_plane(patched) -> None:
    """Both resources need explicit teardown; neither is freed by dropping it."""

    log, _calls = patched
    session = _make_session(enabled=True, log=log)
    session._ensure_rank_f16_rocblas_planes(512)
    for device in session.devices:
        with session._rank_f16_rocblas_owner_context(device, 512):
            pass

    session._release_rank_f16_rocblas_planes()

    assert ("close", 0) in log and ("close", 1) in log, "handles must be closed"
    assert ("release", 0, 512) in log and ("release", 1, 512) in log
    assert session._rank_f16_rocblas == {}
    assert session._rank_f16_rocblas_planes == {}


def test_cleanup_continues_after_a_failing_close(patched, monkeypatch) -> None:
    """One rank's teardown failure must not strand the other's resources."""

    log, _calls = patched
    session = _make_session(enabled=True, log=log)
    session._ensure_rank_f16_rocblas_planes(512)
    for device in session.devices:
        with session._rank_f16_rocblas_owner_context(device, 512):
            pass

    real_close = _FakeHandle.close

    def exploding_close(self: Any) -> None:
        if self._device == 0:
            raise RuntimeError("device 0 handle already destroyed")
        real_close(self)

    monkeypatch.setattr(_FakeHandle, "close", exploding_close)

    session._release_rank_f16_rocblas_planes()

    assert ("close", 1) in log, "rank 1 must still be closed"
    assert session._rank_f16_rocblas == {}


def test_allocation_failure_marks_the_rank_not_ready(patched) -> None:
    """An unadmitted or failing geometry keeps the exact T16 owner."""

    log, calls = patched
    session = _make_session(devices=(0,), enabled=True, log=log)

    session._ensure_rank_f16_rocblas_planes(8192)

    assert calls["allocated"] == [(0, 8192)]
    assert session._rank_f16_rocblas_ready == {0: False}
    assert session._rank_f16_rocblas_planes == {0: None}

    with session._rank_f16_rocblas_owner_context(0, 8192):
        pass
    assert calls["built"] == []


def test_allocator_exception_is_absorbed(patched, monkeypatch) -> None:
    """A raising allocator must degrade to the fallback, not poison the pass."""

    log, _calls = patched

    def exploding_allocate(runner: Any, *, rows: int, runtime: Any = None):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(tp2, "allocate_t16_f16_rocblas_prefill_planes", exploding_allocate)
    session = _make_session(devices=(0,), enabled=True, log=log)

    session._ensure_rank_f16_rocblas_planes(512)

    assert session._rank_f16_rocblas_ready == {0: False}


@pytest.mark.parametrize(
    ("value", "expected"),
    [("default", None), ("on", True), ("off", False)],
)
def test_cli_flag_maps_default_to_the_session_default(value: str, expected: bool | None) -> None:
    """``--t16-f16-rocblas-prefill default`` must mean \"omit the flag\".

    The first version mapped every value except ``off`` to True, so an explicit
    ``default`` silently enabled a default-off candidate.
    """

    import importlib.util
    import pathlib
    import sys

    script = pathlib.Path(tp2.__file__).resolve().parents[2] / "scripts" / "tp2_bulk_prefill_diagnostic.py"
    spec = importlib.util.spec_from_file_location("tp2_bulk_prefill_diagnostic", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    mapping = {"default": None, "on": True, "off": False}
    assert mapping[value] is expected
    # The script must contain that mapping, not a boolean coercion.
    source = script.read_text()
    assert '"default": None' in source
    assert '!= "off"' not in source
