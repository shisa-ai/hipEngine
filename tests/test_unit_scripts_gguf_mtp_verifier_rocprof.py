"""ROCTX discovery for the GGUF verifier profiler.

The block-verify repro arm died with `rocprofiler SDK ROCTX library not found:
/home/lhl/hipEngine-main/.venv/lib/python3.12/site-packages/_rocm_sdk_core/...` while the library
sat on disk the whole time. The candidate list searched `sys.prefix` only, and the ROCm SDK lives
under `sys.base_prefix` - the conda env the venv is built on.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.gguf_mtp_verifier_rocprof import (
    _default_roctx_sdk,
    _prepare_roctx_override,
    _roctx_candidates,
)


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
