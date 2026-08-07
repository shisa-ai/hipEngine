#!/usr/bin/env python3
"""Shared C8 benchmark report metadata and timing schema.

C8 reports retain the benchmark/report sources, the complete hipEngine source
manifest, clean git identity, raw route-wall samples, mechanically derived
P50/P95 values, explicit units, and route-specific CUDA runtime dependency
bytes. Import-only; safe to import without a GPU.
"""

from __future__ import annotations

import hashlib
import os
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

REPORT_SCHEMA = 2
_COMMON_SOURCE = "scripts/c8_report_common.py"
_SUPPORTED_DEPENDENCY_ROUTES = {
    "custom_cuda_runtime_subset",
    "c8_batch_encoder_cublaslt_route",
    "c8_batch_encoder_cudnn_route",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(repo_root: Path) -> dict[str, Any]:
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


def implementation_files(
    repo_root: Path,
    *,
    report_sources: Iterable[str] = (),
) -> dict[str, Path]:
    """Complete package plus report-producing scripts.

    The previous RR-6 manifest covered only ``hipengine/`` and therefore could
    not prove which benchmark/report code produced the retained samples.
    """

    files: dict[str, Path] = {}
    package = repo_root / "hipengine"
    if package.is_dir():
        for suffix in ("*.py", "*.cu", "*.cuh", "*.h"):
            for path in sorted(package.rglob(suffix)):
                files[str(path.relative_to(repo_root))] = path
    for relative in sorted({_COMMON_SOURCE, *map(str, report_sources)}):
        path = (repo_root / relative).resolve()
        try:
            canonical = str(path.relative_to(repo_root.resolve()))
        except ValueError as error:
            raise ValueError(f"report source escapes repository: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"report source is missing: {canonical}")
        files[canonical] = path
    return files


def implementation_sha256(
    repo_root: Path,
    *,
    report_sources: Iterable[str] = (),
) -> dict[str, str]:
    return {
        rel: sha256_file(path)
        for rel, path in sorted(
            implementation_files(repo_root, report_sources=report_sources).items()
        )
    }


def cuda_system_library_bytes() -> dict[str, int]:
    """Measure CUDA runtime/library files available on this host."""

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
            real = os.path.realpath(path) if path.is_symlink() else str(path)
            if os.path.isfile(real) and real not in seen:
                seen.add(real)
                cudnn_total += os.path.getsize(real)
    return {
        "libcudart_so13": size(("libcudart.so.13", "libcudart.so")),
        "libcublasLt_so13": size(("libcublasLt.so.13", "libcublasLt.so")),
        "libcublas_so13": size(("libcublas.so.13", "libcublas.so")),
        "cudnn_payload_total": cudnn_total,
    }


def _sum_known(values: Iterable[int]) -> int:
    values = tuple(int(value) for value in values)
    return sum(values) if all(value >= 0 for value in values) else -1


def dependency_adjusted_bytes(
    route: str,
    *,
    libraries: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return explicit base + route-only + total CUDA runtime bytes.

    This scope intentionally covers system CUDA runtime libraries only. The C5
    deployment report owns Python packages, model, tokenizer, and generated
    kernel binaries. Keeping the scopes separate avoids presenting an inventory
    of unrelated installed libraries as if every C8 route loaded them.
    """

    if route not in _SUPPORTED_DEPENDENCY_ROUTES:
        raise ValueError(f"unsupported C8 dependency route: {route}")
    libs = dict(cuda_system_library_bytes() if libraries is None else libraries)
    base = {"libcudart_so13": int(libs["libcudart_so13"])}
    if route == "c8_batch_encoder_cublaslt_route":
        route_only = {
            "libcublasLt_so13": int(libs["libcublasLt_so13"]),
            "libcublas_so13": int(libs["libcublas_so13"]),
        }
    elif route == "c8_batch_encoder_cudnn_route":
        route_only = {"cudnn_payload_total": int(libs["cudnn_payload_total"])}
    else:
        route_only = {}
    base_total = _sum_known(base.values())
    route_total = _sum_known(route_only.values())
    total = _sum_known((*base.values(), *route_only.values()))
    return {
        "scope": "system_cuda_runtime_libraries_only",
        "route": route,
        "base": base,
        "base_total_bytes": base_total,
        "route_only": route_only,
        "route_only_total_bytes": route_total,
        "total_bytes": total,
        "host_library_inventory_bytes": libs,
    }


def percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank-on-index percentile used by all retained C8 reports."""

    if not sorted_vals:
        return 0.0
    index = min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1))))
    return float(sorted_vals[index])


def route_timing_result(
    times_s: Iterable[float],
    *,
    batch: int,
    c1_seq_total_s: float,
) -> dict[str, Any]:
    """Build one schema-v2 timing row with explicit and consistent units."""

    values = [float(value) for value in times_s]
    if not values:
        raise ValueError("at least one route-wall sample is required")
    if batch <= 0 or c1_seq_total_s <= 0:
        raise ValueError("batch and c1_seq_total_s must be positive")
    median_s = float(statistics.median(values))
    sorted_values = sorted(values)
    p50_ms = percentile(sorted_values, 50) * 1000.0
    p95_ms = percentile(sorted_values, 95) * 1000.0
    req_s = float(batch / median_s)
    c1_req_s = float(batch / c1_seq_total_s)
    return {
        "c1_seq_total_s": float(c1_seq_total_s),
        "route_wall_median_s": median_s,
        "batch_median_ms": median_s * 1000.0,
        "batch_req_per_s": req_s,
        "vs_c1_seq_req_per_s": req_s / c1_req_s,
        "route_wall_s_raw": values,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "sample_count": len(values),
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
    benchmark_source: str,
) -> dict[str, Any]:
    """Assemble a schema-v2 C8 report from benchmark results."""

    repo = _repo_root()
    state = git_state(repo)
    if state["dirty"]:
        raise RuntimeError("refusing to publish a C8 report from a dirty hipEngine tree")
    return {
        "schema": REPORT_SCHEMA,
        "artifact": artifact,
        "date": datetime.now(UTC).date().isoformat(),
        "status": "retained_c8_schema2_raw_samples_source_hashes_dependency_totals",
        "units": {
            "route_wall_s_raw": "seconds",
            "route_wall_median_s": "seconds",
            "batch_median_ms": "milliseconds",
            "p50_ms": "milliseconds",
            "p95_ms": "milliseconds",
        },
        "scope": scope,
        "model": {
            "id": "shisa-ai/shisa-realtime-asr-0.92b",
            "revision": "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
        },
        "environment": environment,
        "hipengine_git": state,
        "implementation_sha256": implementation_sha256(
            repo,
            report_sources=(benchmark_source,),
        ),
        "dependency_adjusted": dependency_adjusted_bytes(dependency_route),
        "method": method,
        "results": results,
        "correctness": correctness,
    }
