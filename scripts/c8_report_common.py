#!/usr/bin/env python3
"""Shared C8 benchmark report metadata (RR-6 evidence discipline).

Mirrors the C5 driver's clean-source/hash discipline so C8 throughput artifacts
retain: the full transitive implementation manifest (entire ``hipengine``
package + every CUDA source), the clean hipEngine git revision, the measured
CUDA system-library bytes each route loads (cuBLASLt/cuDNN/cuBLAS/cuDART), and
raw per-route timing samples with actual P50/P95.  Import-only; safe to import
without a GPU.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(repo_root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return (
                subprocess.run(
                    ["git", "-C", str(repo_root), *args],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                .stdout.strip()
            )
        except Exception:
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def implementation_files(repo_root: Path) -> dict[str, Path]:
    """Complete transitive runtime manifest (RR-3/RR-6): the whole package."""

    files: dict[str, Path] = {}
    package = repo_root / "hipengine"
    if package.is_dir():
        for suffix in ("*.py", "*.cu", "*.cuh", "*.h"):
            for path in sorted(package.rglob(suffix)):
                files[str(path.relative_to(repo_root))] = path
    return files


def implementation_sha256(repo_root: Path) -> dict[str, str]:
    return {
        str(rel): sha256_file(path)
        for rel, path in sorted(implementation_files(repo_root).items())
        if path.is_file()
    }


def cuda_system_library_bytes() -> dict[str, int]:
    """Byte measure of the exact CUDA system libraries a ctypes load pulls in."""

    def resolve(names: tuple[str, ...]) -> Path | None:
        import ctypes.util
        import glob as _glob

        candidates: list[Path] = []
        for root in ("/opt/cuda", "/usr/local/cuda"):
            for libdir in (
                f"{root}/targets/x86_64-linux/lib",
                f"{root}/lib64",
                f"{root}/lib",
            ):
                for name in names:
                    candidates.extend(Path(p) for p in _glob.glob(f"{libdir}/{name}"))
        for name in names:
            candidates.extend(Path(p) for p in _glob.glob(f"/usr/lib/{name}"))
        for name in names:
            found = ctypes.util.find_library(name)
            if found:
                candidates.append(Path(found))
        for path in candidates:
            if path.is_symlink():
                try:
                    target = Path(os.path.realpath(path))
                    if target.is_file():
                        return target
                except OSError:
                    pass
            elif path.is_file():
                return path
        return None

    def size(names: tuple[str, ...]) -> int:
        path = resolve(names)
        return int(path.stat().st_size) if path is not None else -1

    import glob as _glob

    cudnn_total = 0
    seen: set[str] = set()
    for pattern in (
        "/usr/lib/libcudnn*.so.*",
        "/usr/local/cuda/lib64/libcudnn*.so.*",
        "/opt/cuda/targets/x86_64-linux/lib/libcudnn*.so.*",
    ):
        for path in map(Path, _glob.glob(pattern)):
            if path.is_symlink():
                real = os.path.realpath(path)
                if os.path.isfile(real) and real not in seen:
                    seen.add(real)
                    cudnn_total += os.path.getsize(real)
            elif path.is_file() and str(path) not in seen:
                seen.add(str(path))
                cudnn_total += path.stat().st_size
    return {
        "libcudart_so13": size(("libcudart.so.13", "libcudart.so")),
        "libcublasLt_so13": size(("libcublasLt.so.13", "libcublasLt.so")),
        "libcublas_so13": size(("libcublas.so.13", "libcublas.so")),
        "cudnn_payload_total": cudnn_total,
    }


def build_report(
    *,
    artifact: str,
    scope: dict[str, Any],
    environment: dict[str, Any],
    results: dict[str, Any],
    correctness: dict[str, Any],
    method: dict[str, Any],
    dependency_route: str,
) -> dict[str, Any]:
    """Assemble a RR-6-complete report from benchmark results."""

    repo = _repo_root()
    libs = cuda_system_library_bytes()
    return {
        "schema": 1,
        "artifact": artifact,
        "date": datetime.now(UTC).date().isoformat(),
        "status": "retained_rr6_raw_samples_hashes_dependency_bytes",
        "scope": scope,
        "model": {
            "id": "shisa-ai/shisa-realtime-asr-0.92b",
            "revision": "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
        },
        "environment": environment,
        "hipengine_git": git_state(repo),
        "implementation_sha256": implementation_sha256(repo),
        # RR-6/RR-5: dependency-adjusted bytes for the route actually measured.
        "dependency_adjusted": {
            "route": dependency_route,
            "cuda_system_libraries_bytes": libs,
        },
        "method": method,
        "results": results,
        "correctness": correctness,
    }


def percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    index = min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1))))
    return float(sorted_vals[index])
