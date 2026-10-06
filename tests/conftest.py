"""Shared pytest fixtures for global registry isolation."""

from __future__ import annotations

import os
from typing import Any

import pytest

_BASELINE_KERNELS: dict[Any, Any] | None = None
_BASELINE_RUNTIME_PROFILE_PLANS: dict[Any, Any] | None = None
_BASELINE_PROFILE_ENV: dict[str, str | None] | None = None


def _profile_env_names() -> tuple[str, ...]:
    """Environment selectors written process-globally by profile binders."""
    from hipengine.generation import qwen36_gguf_gfx1100_profiles as profiles

    return tuple(dict.fromkeys(getattr(profiles, name) for name in (
        "FP16_RECURRENT_STATE_ENV", "Q4_FUSED_R28_ENV", "Q6_DP4A_GROUPED_ENV",
        "VERIFY_CAPTURE_PREFILL_GDN_ENV", "VERIFY_F32_POST_NORM_ENV",
        "VERIFY_F32_RESIDUAL_ENV",
    )))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--suite",
        choices=("unit", "integration", "gpu", "benchmark", "live", "slow", "all"),
        default="unit",
        help="Discover test_<suite>_*.py; all explicitly includes legacy tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    suite = config.getoption("--suite")
    pattern = "test_*.py" if suite == "all" else f"test_{suite}_*.py"
    config.getini("python_files")[:] = [pattern]


@pytest.fixture(scope="session")
def hip_test_target_arch() -> str:
    """Return the detected device arch for tests that JIT and launch HIP kernels."""

    import ctypes

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime library is unavailable")

    from hipengine.kernels.backends import HIP_TARGET_ARCH_BACKEND, detect_hip_target_arches

    for target_arch in detect_hip_target_arches():
        if target_arch in HIP_TARGET_ARCH_BACKEND:
            return target_arch
    supported = ", ".join(sorted(HIP_TARGET_ARCH_BACKEND))
    pytest.skip(f"no supported HIP target detected (expected one of: {supported})")


@pytest.fixture(autouse=True)
def _clear_mtp_weight_cache():
    """Clear the MTP weight cache before each test to avoid cross-test contamination."""
    try:
        from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import clear_weight_cache
        clear_weight_cache()
    except ImportError:
        pass
    yield
    try:
        from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import clear_weight_cache
        clear_weight_cache()
    except ImportError:
        pass


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
    from hipengine import execution_profiles
    from hipengine.kernels import registry

    global _BASELINE_KERNELS, _BASELINE_RUNTIME_PROFILE_PLANS, _BASELINE_PROFILE_ENV
    _BASELINE_KERNELS = dict(registry._KERNELS)
    _BASELINE_RUNTIME_PROFILE_PLANS = dict(execution_profiles._RUNTIME_PROFILE_PLANS)
    _BASELINE_PROFILE_ENV = {name: os.environ.get(name) for name in _profile_env_names()}


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:  # pragma: no cover - pytest hook
    del item, nextitem
    if _BASELINE_KERNELS is None:
        return
    from hipengine import execution_profiles
    from hipengine.kernels import registry

    registry.restore_registry_for_tests(_BASELINE_KERNELS)
    if _BASELINE_RUNTIME_PROFILE_PLANS is not None:
        execution_profiles.restore_runtime_profile_registry_for_tests(
            _BASELINE_RUNTIME_PROFILE_PLANS
        )
    # Bindings persist for a serving process, but must not select another
    # test's arithmetic or scratch layout. Preserve explicit caller settings.
    if _BASELINE_PROFILE_ENV is not None:
        for name, baseline in _BASELINE_PROFILE_ENV.items():
            if baseline is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = baseline
