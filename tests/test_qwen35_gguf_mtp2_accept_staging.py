"""Lifecycle tests for the page-locked MTP accept-upload arena.

Commit 790eaee91 kept the page-locked accept staging path after it measured
perf-neutral, documenting the blocking pageable copy as its automatic fallback.
The campaign audit (worklog 20260830T193959, task #25) found that fallback was
never actually complete: `runtime.host_register` failures escaped the adapter
instead of selecting the pageable path, and `close()` neither unregistered the
arena nor released it. Page-locked host memory is a scarce process resource, so
a leak per adapter is not a cosmetic defect.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter


class _RecordingRuntime:
    """Minimal HIP surface for accept-arena registration."""

    def __init__(self, *, fail_register: bool = False, fail_unregister: bool = False) -> None:
        self.fail_register = fail_register
        self.fail_unregister = fail_unregister
        self.registered: list[tuple[int, int]] = []
        self.unregistered: list[int] = []

    def host_register(self, ptr: int, nbytes: int, *, flags: int = 0) -> None:
        del flags
        if self.fail_register:
            raise RuntimeError("hipHostRegister failed")
        self.registered.append((int(ptr), int(nbytes)))

    def host_unregister(self, ptr: int) -> None:
        self.unregistered.append(int(ptr))
        if self.fail_unregister:
            raise RuntimeError("hipHostUnregister failed")


def _adapter() -> Qwen35GGUFMTP2Adapter:
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1151", execution_profile="strict"
        ),
        _shared_runner=SimpleNamespace(hidden_size=4),
    )
    return Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=2,
    )


def test_registration_failure_selects_the_pageable_fallback() -> None:
    adapter = _adapter()
    runtime = _RecordingRuntime(fail_register=True)

    arena = adapter._accept_staging_backing(runtime)

    assert arena is None
    assert adapter._accept_staging_state == "off"
    assert runtime.registered == []


def test_registration_failure_is_sticky_and_never_retried() -> None:
    adapter = _adapter()
    runtime = _RecordingRuntime(fail_register=True)

    assert adapter._accept_staging_backing(runtime) is None
    assert adapter._accept_staging_backing(runtime) is None

    assert runtime.registered == []
    assert runtime.unregistered == []


def test_registration_is_attempted_once_and_the_arena_is_reused() -> None:
    adapter = _adapter()
    runtime = _RecordingRuntime()

    first = adapter._accept_staging_backing(runtime)
    second = adapter._accept_staging_backing(runtime)

    assert first is not None and first is second
    assert len(first) == adapter._ACCEPT_STAGING_BYTES
    assert len(runtime.registered) == 1
    ptr, nbytes = runtime.registered[0]
    assert ptr == int(first.ctypes.data)
    assert nbytes == int(first.nbytes)


def test_close_unregisters_and_releases_the_registered_arena() -> None:
    adapter = _adapter()
    runtime = _RecordingRuntime()
    arena = adapter._accept_staging_backing(runtime)
    assert arena is not None
    registered_ptr = runtime.registered[0][0]

    adapter.close()

    assert runtime.unregistered == [registered_ptr]
    assert getattr(adapter, "_accept_staging_arena", None) is None
    assert adapter._accept_staging_state == "off"


def test_close_without_registration_unregisters_nothing() -> None:
    adapter = _adapter()
    runtime = _RecordingRuntime()

    adapter.close()

    assert runtime.unregistered == []
    assert adapter._accept_staging_state == "off"


def test_close_releases_the_arena_even_when_unregister_fails() -> None:
    adapter = _adapter()
    runtime = _RecordingRuntime(fail_unregister=True)
    assert adapter._accept_staging_backing(runtime) is not None

    with pytest.raises(RuntimeError, match="hipHostUnregister failed"):
        adapter.close()

    assert len(runtime.unregistered) == 1
    assert getattr(adapter, "_accept_staging_arena", None) is None
    assert adapter._accept_staging_state == "off"


def test_released_arena_is_not_reused_by_a_later_cycle() -> None:
    adapter = _adapter()
    runtime = _RecordingRuntime()
    assert adapter._accept_staging_backing(runtime) is not None

    adapter.close()

    # A closed adapter must not silently register a second page-locked arena.
    assert adapter._accept_staging_backing(runtime) is None
    assert len(runtime.registered) == 1
    assert runtime.unregistered == [runtime.registered[0][0]]
