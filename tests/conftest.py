"""Shared pytest fixtures for global registry isolation."""

from __future__ import annotations

from typing import Any

import pytest

_BASELINE_KERNELS: dict[Any, Any] | None = None


def pytest_collection_finish(session: pytest.Session) -> None:  # pragma: no cover - pytest hook
    """Snapshot import-time kernel registrations after test collection.

    Several low-level registry plan tests intentionally call
    ``clear_registry_for_tests()`` in xunit setup and then register only the
    family under test. Pytest imports all test modules before running tests, so
    import-time registrations in later modules are otherwise lost once any prior
    test clears the process-global registry. Restoring the collection-time
    baseline after each test keeps those tests order-independent while preserving
    the per-test clear semantics inside the test body.
    """

    del session
    from hipengine.kernels import registry

    global _BASELINE_KERNELS
    _BASELINE_KERNELS = dict(registry._KERNELS)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:  # pragma: no cover - pytest hook
    del item, nextitem
    if _BASELINE_KERNELS is None:
        return
    from hipengine.kernels import registry

    registry._KERNELS.clear()
    registry._KERNELS.update(_BASELINE_KERNELS)
