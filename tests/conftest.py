"""Shared pytest fixtures for global registry isolation."""

from __future__ import annotations

import os
from typing import Any

import pytest

_BASELINE_KERNELS: dict[Any, Any] | None = None
_BASELINE_RUNTIME_PROFILE_PLANS: dict[Any, Any] | None = None
_BASELINE_PROFILE_ENV: dict[str, str | None] | None = None
# Broader snapshot of every HIPENGINE_* selector. The explicit profile list
# above restores the binders' own flags; this one also clears flags that a test
# invented and never cleaned up, which is what made whole-suite runs disagree
# with single-file runs.
_BASELINE_HIPENGINE_ENV: dict[str, str] | None = None


def _profile_env_names() -> tuple[str, ...]:
    """Environment selectors written process-globally by profile binders."""
    from hipengine.generation import qwen36_gguf_gfx1100_profiles as profiles

    return tuple(dict.fromkeys(getattr(profiles, name) for name in (
        "FP16_RECURRENT_STATE_ENV", "Q4_FUSED_R28_ENV", "Q6_DP4A_GROUPED_ENV",
        "VERIFY_CAPTURE_PREFILL_GDN_ENV", "VERIFY_F32_POST_NORM_ENV",
        "VERIFY_F32_RESIDUAL_ENV",
    )))


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
    from hipengine.kernels.hip_gfx1151 import import_gfx1100_kernel_modules

    # Several gfx1100 kernel modules register their kernels at module import
    # time, and test modules import hipengine lazily inside test bodies. Import
    # them here so the baseline below actually contains every import-time
    # registration; otherwise the first test that triggers the imports consumes
    # the once-per-process side effect and every later test that relies on those
    # keys after the teardown restore sees an incomplete registry.
    import_gfx1100_kernel_modules()
    # That helper imports the modules gfx1151 aliases, not every module that
    # registers on import. A runtime that imports the rest lazily (Gemma 4's
    # selected-expert prefill owners, for example) registers them during its
    # first test; the teardown restore then drops them for every later test.
    import pkgutil
    from importlib import import_module

    from hipengine.kernels import hip_gfx1100

    for module in pkgutil.walk_packages(hip_gfx1100.__path__, f"{hip_gfx1100.__name__}."):
        import_module(module.name)

    global _BASELINE_KERNELS, _BASELINE_RUNTIME_PROFILE_PLANS
    global _BASELINE_PROFILE_ENV, _BASELINE_HIPENGINE_ENV
    _BASELINE_KERNELS = dict(registry._KERNELS)
    _BASELINE_RUNTIME_PROFILE_PLANS = dict(execution_profiles._RUNTIME_PROFILE_PLANS)
    _BASELINE_PROFILE_ENV = {name: os.environ.get(name) for name in _profile_env_names()}
    # Profile binders write HIPENGINE_* variables straight into os.environ and
    # never restore them, so a GPU or live test that binds a profile leaves
    # flags behind for every later test in the same process. Unit tests that
    # assert a flag's default then fail only in a whole-suite run. Restore the
    # collection-time values after each test, like the registry baseline.
    _BASELINE_HIPENGINE_ENV = {
        name: value for name, value in os.environ.items() if name.startswith("HIPENGINE_")
    }


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
    # Then drop any HIPENGINE_* flag this test invented, so a flag one test sets
    # cannot leak into the next one's default-value assertions.
    if _BASELINE_HIPENGINE_ENV is not None:
        for name in [n for n in os.environ if n.startswith("HIPENGINE_")]:
            if name not in _BASELINE_HIPENGINE_ENV:
                del os.environ[name]
        for name, value in _BASELINE_HIPENGINE_ENV.items():
            if os.environ.get(name) != value:
                os.environ[name] = value
