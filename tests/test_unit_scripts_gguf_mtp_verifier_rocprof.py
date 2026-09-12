"""ROCTX discovery for the GGUF verifier profiler.

The block-verify repro arm died with `rocprofiler SDK ROCTX library not found:
/home/lhl/hipEngine-main/.venv/lib/python3.12/site-packages/_rocm_sdk_core/...` while the library
sat on disk the whole time. The candidate list searched `sys.prefix` only, and the ROCm SDK lives
under `sys.base_prefix` - the conda env the venv is built on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import gguf_mtp_verifier_rocprof as profiler
from scripts.gguf_mtp_verifier_rocprof import (
    NATIVE_SPEC_TARGET_ROWS,
    _default_roctx_sdk,
    _prepare_roctx_override,
    _roctx_candidates,
)


def test_module_puts_its_own_worktree_first_on_sys_path() -> None:
    """Child mode re-runs this file, so sys.path[0] is scripts/ and hipengine
    would otherwise resolve to the editable install's other worktree."""

    import importlib

    saved = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if entry != str(profiler.REPO_ROOT)]
        assert str(profiler.REPO_ROOT) not in sys.path
        importlib.reload(profiler)
        assert sys.path[0] == str(profiler.REPO_ROOT), sys.path[:3]
    finally:
        sys.path[:] = saved
    assert (profiler.REPO_ROOT / "hipengine" / "__init__.py").is_file()


def test_candidates_cover_the_base_prefix_not_just_the_venv() -> None:
    texts = [str(path) for path in _roctx_candidates()]
    assert any(sys.prefix in text for text in texts), texts
    assert any(sys.base_prefix in text for text in texts), (
        "a venv built on a ROCm conda env has no _rocm_sdk_* packages of its own; searching only "
        "sys.prefix can never find the SDK there"
    )


def test_candidates_name_both_packages_and_never_duplicate_prefixes() -> None:
    texts = [str(path) for path in _roctx_candidates()]
    assert any("_rocm_sdk_core" in text for text in texts)
    assert any("_rocm_sdk_devel" in text for text in texts)
    assert len(texts) == len(set(texts)), "prefixes must be deduplicated"
    if sys.prefix == sys.base_prefix:
        assert len(texts) <= 6, texts


def test_default_resolution_prefers_an_existing_library() -> None:
    resolved = _default_roctx_sdk()
    assert isinstance(resolved, Path)
    if any(path.exists() for path in _roctx_candidates()):
        assert resolved.exists(), resolved
    else:  # pragma: no cover - a runner with no ROCm SDK must still get a candidate back
        assert resolved == _roctx_candidates()[0]


def test_a_missing_library_says_what_was_searched(tmp_path) -> None:
    """A one-line FileNotFoundError naming one path is how this cost a whole profiling arm."""
    try:
        _prepare_roctx_override(tmp_path / "absent.so")
    except FileNotFoundError as exc:
        message = str(exc)
        assert "searched" in message
        assert message.count("librocprofiler-sdk-roctx") >= len(_roctx_candidates())
        assert "--roctx-sdk" in message
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError for a non-existent SDK path")


def test_quant_selection_reaches_the_child(monkeypatch) -> None:
    """A plain artifact aborts with a sentinel token when the prefill quant axis
    is left at the generic default, so the flag must survive arg parsing."""

    captured = {}
    monkeypatch.setattr(profiler, "_run_child", lambda args: captured.update(vars(args)) or 0)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gguf_mtp_verifier_rocprof.py", "--child", "--quant", "gguf_q4_k_m"],
    )
    assert profiler.main() == 0
    assert captured["quant"] == "gguf_q4_k_m"


def test_native_device_accept_commit_defaults_to_the_production_bucket(monkeypatch) -> None:
    """The census must replay the same graph bucket production replays, so the
    device-side accept/commit default is on; the opt-out stays available."""

    captured = {}
    monkeypatch.setattr(profiler, "_run_child", lambda args: captured.update(vars(args)) or 0)
    monkeypatch.setattr(sys, "argv", ["gguf_mtp_verifier_rocprof.py", "--child"])
    assert profiler.main() == 0
    assert captured["native_device_accept_commit"] is True
    monkeypatch.setattr(
        sys,
        "argv",
        ["gguf_mtp_verifier_rocprof.py", "--child", "--no-native-device-accept-commit"],
    )
    assert profiler.main() == 0
    assert captured["native_device_accept_commit"] is False


def test_native_cycle_accepts_every_bucket_row_count(monkeypatch) -> None:
    """The production native leaf is B1-B3: four rows is the B3 verifier shape."""

    assert NATIVE_SPEC_TARGET_ROWS == frozenset({2, 3, 4})
    monkeypatch.setattr(profiler, "_run_child", lambda _args: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_mtp_verifier_rocprof.py",
            "--child",
            "--native-spec-target-cycle",
            "--mode",
            "block-verify",
            "--block-rows",
            "4",
        ],
    )
    assert profiler.main() == 0


@pytest.mark.parametrize("rows", ("5", "6", "7", "8"))
def test_native_bucket_stops_at_four_rows(monkeypatch, rows) -> None:
    """The graph builder admits 2-8 rows, but the fused add/RMSNorm and device
    accept/commit kernels raise `rows must be 2, 3, or 4` above four."""

    monkeypatch.setattr(profiler, "_run_child", lambda _args: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_mtp_verifier_rocprof.py",
            "--child",
            "--native-spec-target-cycle",
            "--mode",
            "block-verify",
            "--block-rows",
            rows,
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        profiler.main()
    assert excinfo.value.code == 2


@pytest.mark.parametrize("rows", ("1", "9"))
def test_native_cycle_rejects_rows_outside_the_bucket(monkeypatch, rows) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_mtp_verifier_rocprof.py",
            "--child",
            "--native-spec-target-cycle",
            "--mode",
            "block-verify",
            "--block-rows",
            rows,
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        profiler.main()
    assert excinfo.value.code == 2
