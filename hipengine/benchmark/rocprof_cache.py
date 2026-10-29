"""Host-side cache-only checks and process setup for rocprof entry points."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.roctx_discovery import (
    default_sdk as _default_roctx_sdk,
    prepare_override,
    sdk_dependency_paths,
)

_COMPILER_PROCESS_RE = re.compile(r"(?:^|/)(?:hipcc|amdclang\+\+|amdclang|clang\+\+|clang)(?:\s|$)")


class RocprofCacheError(RuntimeError):
    """Raised when isolated-cache or profiler invariants fail."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot_cache_tree(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    files: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(value for value in root.rglob("*") if value.is_file()):
            metadata = path.stat()
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": int(metadata.st_size),
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "mtime_ns": int(metadata.st_mtime_ns),
                    "sha256": _file_sha256(path),
                }
            )
    payload = {"root": str(root), "file_count": len(files), "files": files}
    payload["tree_sha256"] = _payload_sha256(payload)
    return payload


def cache_build_manifest_summary(root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(Path(root).expanduser().resolve().rglob("manifest.txt")):
        fields: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line == "compiler_version<<EOF":
                break
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value
        outputs = [value for value in path.parent.iterdir() if value.suffix == ".so"]
        summaries.append(
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "family": fields.get("family"),
                "profile": fields.get("profile"),
                "cache_key": fields.get("cache_key"),
                "target_arch": fields.get("target_arch"),
                "output_sha256": {
                    value.name: _file_sha256(value) for value in sorted(outputs)
                },
            }
        )
    return summaries


def prepare_compiler_guard(root: Path) -> dict[str, Path]:
    directory = Path(root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "compiler-invoked.log"
    if marker.exists():
        raise ValueError(f"compiler guard marker already exists: {marker}")
    guard = directory / "hipcc"
    guard.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$0 $*\" >> {shlex.quote(str(marker))}\n"
        "exit 97\n",
        encoding="utf-8",
    )
    guard.chmod(0o755)
    return {"directory": directory, "marker": marker, "hipcc": guard}


def validate_cache_only_stage(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    compiler_guard_marker: Path,
    observed_compiler_processes: Sequence[str],
) -> None:
    if Path(compiler_guard_marker).exists():
        raise ValueError(f"compiler guard was invoked: {compiler_guard_marker}")
    if tuple(observed_compiler_processes):
        raise ValueError(
            "compiler subprocess observed during cache-only stage: "
            + "; ".join(str(value) for value in observed_compiler_processes)
        )
    if before.get("tree_sha256") != after.get("tree_sha256"):
        raise ValueError("cache mutated during cache-only stage")


def _children(pid: int) -> tuple[int, ...]:
    path = Path(f"/proc/{int(pid)}/task/{int(pid)}/children")
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ()
    return tuple(int(value) for value in raw.split()) if raw else ()


def _descendants(pid: int) -> tuple[int, ...]:
    pending = list(_children(pid))
    observed: list[int] = []
    while pending:
        child = pending.pop()
        if child in observed:
            continue
        observed.append(child)
        pending.extend(_children(child))
    return tuple(observed)


def _cmdline(pid: int) -> str:
    try:
        data = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""
    return " ".join(part.decode("utf-8", errors="replace") for part in data.split(b"\0") if part)


def _run_monitored(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    observed_compilers: set[str] = set()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + float(timeout_seconds)
        while process.poll() is None:
            for child_pid in _descendants(process.pid):
                command_line = _cmdline(child_pid)
                if command_line and _COMPILER_PROCESS_RE.search(command_line):
                    observed_compilers.add(command_line)
            if time.monotonic() >= deadline:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=30.0)
                raise RocprofCacheError(
                    f"command timed out after {timeout_seconds}s: {shlex.join(command)}"
                )
            time.sleep(0.02)
        return {
            "returncode": int(process.returncode),
            "observed_compiler_processes": sorted(observed_compilers),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }


def _prepare_roctx_override(sdk_path: Path, run_root: Path) -> tuple[Path, tuple[Path, ...]]:
    sdk_path = sdk_path.expanduser().resolve()
    override = prepare_override(sdk_path, run_root / "roctx-override", require_file=True)
    dependencies = tuple(dict.fromkeys((
        *(path for path in (sdk_path.parent, Path(sys.prefix) / "lib") if path.exists()),
        *sdk_dependency_paths(sdk_path),
    )))
    return override, dependencies


def _cache_only_environment(
    base: Mapping[str, str],
    *,
    guard: Mapping[str, Path],
) -> dict[str, str]:
    environment = {str(key): str(value) for key, value in base.items()}
    environment["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    environment["PATH"] = str(guard["directory"]) + os.pathsep + environment.get("PATH", "")
    return environment
