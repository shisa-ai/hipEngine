"""ROCTX SDK discovery and override setup shared by profiling entry points."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def sdk_candidates() -> list[Path]:
    """Probe the active and base Python environments, then system ROCm."""
    names = ("librocprofiler-sdk-roctx.so.1", "librocprofiler-sdk-roctx.so")
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(root) / "lib" / python_dir / "site-packages" / package / "lib" / name
        for root in dict.fromkeys((sys.prefix, sys.base_prefix))
        for package in ("_rocm_sdk_core", "_rocm_sdk_devel")
        for name in names
    ]
    return candidates + [Path("/opt/rocm/lib") / name for name in names]


def candidates_for_prefix(prefix: Path) -> list[Path]:
    """Find SDK sonames under a profiler's prefix, newest Python path first."""
    found: list[Path] = []
    for package in ("_rocm_sdk_core", "_rocm_sdk_devel"):
        found.extend(sorted(
            (prefix / "lib").glob(
                f"python3*/site-packages/{package}/lib/librocprofiler-sdk-roctx.so*"
            ),
            reverse=True,
        ))
    return found


def profiler_prefix() -> Path | None:
    executable = shutil.which("rocprofv3")
    if not executable:
        return None
    resolved = Path(executable).resolve()
    return resolved.parents[1] if resolved.parent.name == "bin" else None


def default_sdk(*, profiler_first: bool = False, legacy_fallback: bool = False) -> Path:
    """Resolve a library without changing a wrapper's declared search policy.

    An absent library returns the preferred SDK path so setup can raise an
    actionable error. Marker tracing must not silently become unscoped tracing.
    """
    preferred = sdk_candidates()
    prefix = profiler_prefix() if profiler_first else None
    candidates = candidates_for_prefix(prefix) if prefix is not None else []
    candidates += preferred
    if legacy_fallback:
        candidates += [Path("/opt/rocm/lib/libroctx64.so.4"), Path("/opt/rocm/lib/libroctx64.so")]
    return next((path for path in candidates if path.exists()), preferred[0])


def prepare_override(sdk_path: Path, override: Path, *, require_file: bool = False) -> Path:
    """Link an explicitly selected library; dependency-path policy stays with callers."""
    available = sdk_path.is_file() if require_file else sdk_path.exists()
    if not available:
        raise FileNotFoundError(f"rocprofiler SDK ROCTX library not found: {sdk_path}")
    override.mkdir(parents=True, exist_ok=True)
    symlink = override / "libroctx64.so"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(sdk_path)
    return override


def sdk_dependency_paths(sdk_path: Path) -> tuple[Path, ...]:
    """Return existing SDK and bundled system-dependency library directories."""
    root = sdk_path.parent
    if not root.is_dir():
        return ()
    sysdeps = root / "rocm_sysdeps" / "lib"
    return (root, sysdeps) if sysdeps.is_dir() else (root,)
