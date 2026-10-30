"""Profiler SDK primitives preserve wrapper-specific search and loader contracts."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from scripts import roctx_discovery as roctx


def test_default_candidates_keep_prefix_base_and_system_order(monkeypatch):
    monkeypatch.setattr(sys, "prefix", "/venv")
    monkeypatch.setattr(sys, "base_prefix", "/base")
    paths = roctx.sdk_candidates()
    assert str(paths[0]).startswith("/venv/lib/python")
    assert str(paths[4]).startswith("/base/lib/python")
    assert paths[-2:] == [Path("/opt/rocm/lib/librocprofiler-sdk-roctx.so.1"),
                          Path("/opt/rocm/lib/librocprofiler-sdk-roctx.so")]
    monkeypatch.setattr(sys, "base_prefix", "/venv")
    assert len(roctx.sdk_candidates()) == 6


def test_profiler_and_legacy_policies_are_explicit(monkeypatch, tmp_path):
    sdk = tmp_path / "lib/python3.12/site-packages/_rocm_sdk_core/lib/librocprofiler-sdk-roctx.so.1.3"
    sdk.parent.mkdir(parents=True)
    sdk.touch()
    profiler = tmp_path / "bin/rocprofv3"
    profiler.parent.mkdir()
    profiler.touch()
    monkeypatch.setattr("shutil.which", lambda name: str(profiler))
    monkeypatch.setattr(roctx, "sdk_candidates", lambda: [tmp_path / "missing.so"])
    assert roctx.default_sdk() == tmp_path / "missing.so"
    assert roctx.default_sdk(profiler_first=True) == sdk
    monkeypatch.setattr("shutil.which", lambda name: None)
    legacy = Path("/opt/rocm/lib/libroctx64.so.4")
    monkeypatch.setattr(Path, "exists", lambda path: path == legacy)
    assert roctx.default_sdk(legacy_fallback=True) == legacy
    assert roctx.default_sdk() == tmp_path / "missing.so"


def test_override_relinks_and_dependency_paths_are_explicit(tmp_path):
    sdk = tmp_path / "sdk/libroctx.so"
    sdk.parent.mkdir()
    override = tmp_path / "override"
    with pytest.raises(FileNotFoundError, match="SDK ROCTX library not found"):
        roctx.prepare_override(sdk, override)
    assert not override.exists()
    sdk.touch()
    assert roctx.prepare_override(sdk, override) == override
    assert (override / "libroctx64.so").readlink() == sdk
    peer = sdk.with_name("peer.so")
    peer.touch()
    roctx.prepare_override(peer, override)
    assert (override / "libroctx64.so").resolve() == peer
    assert roctx.sdk_dependency_paths(sdk) == (sdk.parent,)
    sysdeps = sdk.parent / "rocm_sysdeps/lib"
    sysdeps.mkdir(parents=True)
    assert roctx.sdk_dependency_paths(sdk) == (sdk.parent, sysdeps)
    assert roctx.sdk_dependency_paths(tmp_path / "absent/lib.so") == ()


@pytest.mark.parametrize("name", (
    "gguf_continuous_owner_rocprof", "qwen38_gfx1151_prefill_kernel_profile",
    "gguf_decode_rocprof", "gguf_mtp_verifier_rocprof", "gguf_packed_ar_rocprof",
    "gguf_mtp_draft_rocprof", "gguf_sh_c0_profile", "qwen35_rocprof_audit", "mtp_verifier_rocprof",
))
def test_wrapper_default_uses_shared_discovery(monkeypatch, tmp_path, name):
    module = importlib.import_module(f"scripts.{name}")
    sdk = tmp_path / "selected.so"
    sdk.touch()
    # Cache-owner wrappers use the packaged implementation; standalone wrappers
    # load the same source without binding hipengine before child setup.
    from hipengine.benchmark import roctx_discovery as packaged_roctx
    for discovery in (roctx, packaged_roctx):
        monkeypatch.setattr(discovery, "sdk_candidates", lambda: [sdk])
        monkeypatch.setattr(discovery, "profiler_prefix", lambda: None)
    assert module._default_roctx_sdk() == sdk


def test_decode_profiler_import_leaves_child_package_selection_unbound(tmp_path):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/gguf_decode_rocprof.py"
    source = (
        "import importlib.util,sys; "
        f"spec=importlib.util.spec_from_file_location('decode_probe',{str(script)!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "assert not any(n == 'hipengine' or n.startswith('hipengine.') for n in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", source], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name,folder,deps", (
    ("gguf_continuous_owner_rocprof", "roctx-override", "prefix"),
    ("qwen38_gfx1151_prefill_kernel_profile", "roctx-override", "prefix"),
    ("gguf_packed_ar_rocprof", "roctx-override", "prefix"),
    ("gguf_decode_rocprof", "hipengine-roctx-sdk-override-gguf-eager", "sysdeps"),
    ("gguf_sh_c0_profile", "hipengine-roctx-sdk-override-gguf-eager", "sysdeps"),
    ("gguf_mtp_verifier_rocprof", "hipengine-roctx-sdk-override-gguf-mtp", None),
    ("gguf_mtp_draft_rocprof", "hipengine-roctx-sdk-override-gguf-mtp", None),
    ("mtp_verifier_rocprof", "hipengine-roctx-sdk-override-mtp", None),
    ("qwen35_rocprof_audit", "hipengine-roctx-sdk-override", None),
))
def test_wrappers_keep_override_and_dependency_contracts(monkeypatch, tmp_path, name, folder, deps):
    module = importlib.import_module(f"scripts.{name}")
    sdk = tmp_path / "sdk/libroctx.so"
    sdk.parent.mkdir()
    sdk.touch()
    sysdeps = sdk.parent / "rocm_sysdeps/lib"
    sysdeps.mkdir(parents=True)
    prefix = tmp_path / "python"
    (prefix / "lib").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    seen = []

    def prepare(path, directory, **kwargs):
        seen.append((path, directory, kwargs))
        return directory

    monkeypatch.setattr(roctx, "prepare_override", prepare)
    # Adapters import the function directly; patch its owning module too.
    target = importlib.import_module(module._prepare_roctx_override.__module__)
    monkeypatch.setattr(target, "prepare_override", prepare)
    run = tmp_path / "run"
    if deps == "prefix":
        override, dependencies = module._prepare_roctx_override(sdk, run)
        assert override == run / folder
        assert dependencies == (sdk.parent, prefix / "lib", sysdeps)
    elif deps == "sysdeps":
        override, dependencies = module._prepare_roctx_override(sdk)
        assert override == Path("/tmp") / folder
        assert dependencies == (sdk.parent, sysdeps)
    else:
        override = module._prepare_roctx_override(sdk)
        assert override == Path("/tmp") / folder
    assert len(seen) == 1 and seen[0][0] == sdk
    assert seen[0][1] == override


@pytest.mark.parametrize("name", ("mtp_verifier_rocprof", "qwen35_rocprof_audit"))
def test_home_free_default_and_explicit_override_reach_cli(monkeypatch, tmp_path, name, capsys):
    module = importlib.import_module(f"scripts.{name}")
    chosen = tmp_path / "chosen.so"
    explicit = tmp_path / "explicit.so"
    selected = []
    monkeypatch.setattr(module, "_default_roctx_sdk", lambda: chosen)
    monkeypatch.setattr(module, "_prepare_roctx_override", lambda path: selected.append(path) or tmp_path)
    for extra, expected in (([], chosen), (["--roctx-sdk", str(explicit)], explicit)):
        monkeypatch.setattr(sys, "argv", [name, "--dry-run", "--raw-root", str(tmp_path / "raw"),
                                        "--out", str(tmp_path / "out.json"), *extra])
        assert module.main() == 0
        assert selected[-1] == expected
        assert "rocprof" in capsys.readouterr().out


def test_paro_audit_forwards_discovered_sdk_dependencies(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.qwen35_rocprof_audit")
    sdk = tmp_path / "sdk/libroctx.so"
    sdk.parent.mkdir()
    sdk.touch()
    sysdeps = sdk.parent / "rocm_sysdeps/lib"
    sysdeps.mkdir(parents=True)
    monkeypatch.setattr(module, "_prepare_roctx_override", lambda path: tmp_path / "override")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/original/lib")
    monkeypatch.setattr(sys, "argv", ["audit", "--workloads", "8/1", "--roctx-sdk", str(sdk),
                                    "--raw-root", str(tmp_path / "raw"), "--out", str(tmp_path / "out.json")])

    class Captured(RuntimeError):
        pass

    def run(command, **kwargs):
        assert kwargs["env"]["LD_LIBRARY_PATH"].split(":") == [
            str(tmp_path / "override"), str(sdk.parent), str(sysdeps), "/original/lib",
        ]
        raise Captured

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(Captured):
        module.main()


@pytest.mark.parametrize("name", ("gguf_mtp_verifier_rocprof", "mtp_verifier_rocprof"))
def test_verifier_dependency_paths_use_shared_contract(tmp_path, name):
    module = importlib.import_module(f"scripts.{name}")
    sdk = tmp_path / "sdk/libroctx.so"
    sdk.parent.mkdir()
    sysdeps = sdk.parent / "rocm_sysdeps/lib"
    sysdeps.mkdir(parents=True)
    assert module._roctx_sdk_dep_paths(sdk) == (sdk.parent, sysdeps)
