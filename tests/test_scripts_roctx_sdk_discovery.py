"""ROCTX discovery is duplicated across the profiling wrappers; pin them all at once.

Four scripts carry their own `_default_roctx_sdk()`. All four built every candidate from
`sys.prefix`, so all four fail identically when run as `.venv/bin/python`: the venv has no
`_rocm_sdk_*` packages, because the ROCm SDK ships in the conda env the venv is built on
(`sys.base_prefix`). The 04:31 worklog entry diagnosed this and left the one-line fix unclaimed;
8c59be6d8 implemented it for one wrapper and this unit covers the rest. The duplication itself is
tracked in docs/REFACTOR.md.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    "gguf_continuous_owner_rocprof",
    "gguf_decode_rocprof",
    "gguf_mtp_verifier_rocprof",
    "gguf_packed_ar_rocprof",
)
LIB = "librocprofiler-sdk-roctx.so"


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"roctx_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sdk_shipped_in_base_prefix() -> bool:
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = Path(sys.base_prefix) / "lib" / python_dir / "site-packages"
    return any(
        (site / pkg / "lib" / f"{LIB}.1").is_file()
        for pkg in ("_rocm_sdk_core", "_rocm_sdk_devel")
    )


@pytest.mark.parametrize("name", WRAPPERS)
def test_wrapper_resolves_the_sdk_that_is_actually_installed(name: str) -> None:
    """The bug was silent: a nonexistent .venv path is a valid-looking Path.

    So the assertion is about the host, not the path shape - when the SDK really is installed under
    base_prefix, a default that does not exist is a defect.
    """
    resolved = _load(name)._default_roctx_sdk()
    if _sdk_shipped_in_base_prefix():
        assert resolved.exists(), (
            f"{name}: ROCTX is installed under {sys.base_prefix} but the default resolved to a "
            f"missing {resolved}; the candidate list needs sys.base_prefix, not just sys.prefix"
        )
    else:  # pragma: no cover - runner without the ROCm SDK packages
        assert str(resolved).endswith(LIB), resolved
