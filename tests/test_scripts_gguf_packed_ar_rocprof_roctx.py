"""ROCTX SDK discovery for the packed-AR profiler wrapper.

Regression cover for the failure mode hit on the W7900 host on 2026-08-30: the image ships
the legacy ``/opt/rocm/lib/libroctx64.so`` but neither the pip ``_rocm_sdk_*`` packages nor
``librocprofiler-sdk-roctx.so.1``, so ``_default_roctx_sdk`` returned a path that does not
exist and the run died in ``_prepare_roctx_override`` reporting "rocprofiler SDK ROCTX library
not found". The fix added the legacy library as a fallback candidate. These tests pin both the
fallback and the unchanged no-such-SDK behaviour, with no GPU and no rocprofv3.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gguf_packed_ar_rocprof.py"


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("gguf_packed_ar_rocprof_roctx", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> object:
    return _load_module()


def _no_profiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the profiler-prefix probe out of tests that target the other candidates."""
    monkeypatch.setattr('shutil.which', lambda _name: None)


def _patch_exists(monkeypatch: pytest.MonkeyPatch, *, allow) -> None:
    def fake_exists(self: pathlib.Path) -> bool:  # noqa: ANN001
        return allow(str(self))

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)


def test_legacy_roctx64_is_used_when_the_sdk_package_is_absent(
    mod: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_profiler(monkeypatch)
    _patch_exists(monkeypatch, allow=lambda path: path.endswith("/opt/rocm/lib/libroctx64.so.4"))
    chosen = mod._default_roctx_sdk()  # noqa: SLF001
    assert chosen == pathlib.Path("/opt/rocm/lib/libroctx64.so.4"), chosen


def test_plain_legacy_library_is_the_next_fallback(
    mod: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_profiler(monkeypatch)
    _patch_exists(monkeypatch, allow=lambda path: path == "/opt/rocm/lib/libroctx64.so")
    chosen = mod._default_roctx_sdk()  # noqa: SLF001
    assert chosen == pathlib.Path("/opt/rocm/lib/libroctx64.so"), chosen


def test_sdk_package_still_wins_over_the_legacy_library(
    mod: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_profiler(monkeypatch)
    _patch_exists(
        monkeypatch,
        allow=lambda path: (
            "librocprofiler-sdk-roctx" in path or path.endswith("libroctx64.so")
        ),
    )
    chosen = mod._default_roctx_sdk()  # noqa: SLF001
    assert "librocprofiler-sdk-roctx" in str(chosen), chosen


def test_missing_sdk_reports_the_preferred_candidate(
    mod: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_profiler(monkeypatch)
    _patch_exists(monkeypatch, allow=lambda path: False)
    chosen = mod._default_roctx_sdk()  # noqa: SLF001
    # Unchanged behaviour: the error path still names the pip SDK location first, which is
    # the message an operator acts on ("install the SDK or pass --roctx-sdk").
    assert "_rocm_sdk_core" in str(chosen), chosen
    assert str(pathlib.Path(sys.prefix)) in str(chosen), chosen


def test_the_profiler_environment_is_preferred_over_system_libraries(
    mod: object, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """docs/KERNELS.md trap 1: the matching SDK usually ships with the profiler, not the venv.

    Uses real files because the profiler prefix is discovered by globbing the filesystem, so
    faking Path.exists would fake away the thing under test.
    """
    env = tmp_path / "envs" / "rocmenv"
    sdk_lib = env / "lib" / "python3.12" / "site-packages" / "_rocm_sdk_core" / "lib"
    sdk_lib.mkdir(parents=True)
    sdk = sdk_lib / "librocprofiler-sdk-roctx.so.1"
    sdk.write_text("")
    (env / "bin").mkdir()
    (env / "bin" / "rocprofv3").write_text("")
    monkeypatch.setattr("shutil.which", lambda name: str(env / "bin" / "rocprofv3"))
    assert mod._default_roctx_sdk() == sdk  # noqa: SLF001


def test_override_accepts_the_discovered_library(tmp_path: pathlib.Path, mod: object) -> None:
    """Whatever the discovery returns must satisfy _prepare_roctx_override without a hand flag."""
    sdk = tmp_path / "librocprofiler-sdk-roctx.so.1"
    sdk.write_text("")
    monkeypatch_bin = tmp_path / "bin"
    monkeypatch_bin.mkdir()
    (monkeypatch_bin / "rocprofv3").write_text("")
    env = tmp_path / "env"
    discovered = tmp_path / "lib" / "python3.12" / "site-packages" / "_rocm_sdk_core" / "lib"
    discovered.mkdir(parents=True)
    (discovered / "librocprofiler-sdk-roctx.so.1").write_text("")
    override, dependencies = mod._prepare_roctx_override(sdk, tmp_path / "raw")  # noqa: SLF001
    assert (override / "libroctx64.so").resolve() == sdk.resolve()
    assert dependencies
