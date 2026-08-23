"""Canonical provenance for benchmark and retained-result artifacts.

The collector is stdlib-only and torch-free so server, PARO, GGUF, and micro
benchmark harnesses can share one identity contract without importing a model
runtime.  It intentionally keeps staged, unstaged, and untracked git state as
separate fields: a single ``git diff`` check is not enough evidence for a clean
performance claim.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hipengine.kernels.backends import (
    HIP_BACKEND_TARGET_ARCH,
    detect_hip_target_arches,
    select_backend,
)


ARTIFACT_PROVENANCE_KIND = "hipengine_artifact_provenance"
ARTIFACT_PROVENANCE_SCHEMA_VERSION = 2
_FULL_HASH_MAX_BYTES = 8 * 1024 * 1024
_SAMPLE_BYTES = 1024 * 1024
_UNSET = object()
_DEFAULT_ENV_KEYS = (
    "HIPENGINE_BACKEND",
    "HIPENGINE_HIP_ARCH",
    "GPU_MAX_HW_QUEUES",
    "HIPENGINE_GPU_MAX_HW_QUEUES_POLICY",
    "HIPENGINE_BUILD_CACHE_ROOT",
    "HIPENGINE_REQUIRE_CACHED_BUILD",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)
_REQUIRED_FIELDS = (
    "kind",
    "schema_version",
    "collected_at",
    "repo_root",
    "hipengine_commit",
    "git_branch",
    "staged_dirty",
    "unstaged_dirty",
    "untracked_dirty",
    "untracked_count",
    "dirty",
    "configured_backend",
    "resolved_backend",
    "target_arch",
    "device_name",
    "model_path",
    "model_revision",
    "model_fingerprint",
    "quant",
    "kv_dtype",
    "command",
    "environment",
    "rocm_version",
    "hipcc_version",
    "build_profile",
    "timing_protocol",
    "warmups",
    "repetitions",
    "profiler",
)


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 5.0,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(part) for part in command],
            cwd=None if cwd is None else str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return subprocess.CompletedProcess([str(part) for part in command], 127, "", "")


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(("git", *args), cwd=repo_root)


def collect_repo_state(repo_root: str | Path) -> dict[str, Any]:
    """Return commit identity plus all three dirty-worktree axes."""

    root = Path(repo_root).expanduser().resolve()
    commit_result = _git(root, "rev-parse", "HEAD")
    if commit_result.returncode != 0 or not commit_result.stdout.strip():
        raise ValueError(f"benchmark repo_root is not a readable git worktree: {root}")
    branch_result = _git(root, "branch", "--show-current")
    unstaged_result = _git(root, "diff", "--quiet", "--no-ext-diff")
    staged_result = _git(root, "diff", "--cached", "--quiet", "--no-ext-diff")
    if unstaged_result.returncode not in {0, 1} or staged_result.returncode not in {0, 1}:
        raise ValueError(f"could not inspect tracked dirty state for benchmark repo: {root}")
    untracked_result = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    if untracked_result.returncode != 0:
        raise ValueError(f"could not inspect untracked files for benchmark repo: {root}")
    untracked_paths = [path for path in untracked_result.stdout.split("\0") if path]
    staged_dirty = staged_result.returncode == 1
    unstaged_dirty = unstaged_result.returncode == 1
    untracked_dirty = bool(untracked_paths)
    return {
        "repo_root": str(root),
        "hipengine_commit": commit_result.stdout.strip(),
        "git_branch": branch_result.stdout.strip() or None,
        "staged_dirty": staged_dirty,
        "unstaged_dirty": unstaged_dirty,
        "untracked_dirty": untracked_dirty,
        "untracked_count": len(untracked_paths),
        "dirty": staged_dirty or unstaged_dirty or untracked_dirty,
    }


def _hash_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= _FULL_HASH_MAX_BYTES:
        return {
            "algorithm": "sha256-full-v1",
            "value": _hash_stream(path),
            "size_bytes": int(size),
            "sampled_bytes": int(size),
        }

    sample_size = min(_SAMPLE_BYTES, size)
    offsets = tuple(
        dict.fromkeys(
            (
                0,
                max(0, (size - sample_size) // 2),
                max(0, size - sample_size),
            )
        )
    )
    digest = hashlib.sha256()
    digest.update(f"size={size}\n".encode("ascii"))
    sampled_bytes = 0
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            sample = handle.read(sample_size)
            digest.update(f"offset={offset};bytes={len(sample)}\n".encode("ascii"))
            digest.update(sample)
            sampled_bytes += len(sample)
    return {
        "algorithm": "sha256-sampled-v1",
        "value": digest.hexdigest(),
        "size_bytes": int(size),
        "sampled_bytes": int(sampled_bytes),
        "sample_offsets": list(offsets),
    }


def _directory_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    sampled_bytes = 0
    for child in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: str(item.relative_to(path))):
        relative = child.relative_to(path).as_posix()
        child_fingerprint = _file_fingerprint(child)
        file_count += 1
        total_size += int(child_fingerprint["size_bytes"])
        sampled_bytes += int(child_fingerprint["sampled_bytes"])
        digest.update(
            json.dumps(
                {
                    "path": relative,
                    "algorithm": child_fingerprint["algorithm"],
                    "value": child_fingerprint["value"],
                    "size_bytes": child_fingerprint["size_bytes"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "algorithm": "sha256-directory-manifest-v1",
        "value": digest.hexdigest(),
        "size_bytes": total_size,
        "sampled_bytes": sampled_bytes,
        "file_count": file_count,
    }


def _snapshot_revision(path: Path) -> str | None:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and index + 1 < len(parts):
            revision = parts[index + 1].strip()
            return revision or None
    return None


def collect_model_identity(
    model_path: str | Path | None,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Return a stable model path/revision/content fingerprint triple."""

    if model_path is None:
        return {"path": None, "revision": revision, "fingerprint": None}
    path = Path(model_path).expanduser().resolve()
    inferred_revision = revision or _snapshot_revision(path)
    if path.is_file():
        fingerprint = _file_fingerprint(path)
        fingerprint["exists"] = True
        fingerprint["path_type"] = "file"
    elif path.is_dir():
        fingerprint = _directory_fingerprint(path)
        fingerprint["exists"] = True
        fingerprint["path_type"] = "directory"
    else:
        fingerprint = {
            "algorithm": "sha256-missing-path-v1",
            "value": hashlib.sha256(str(path).encode("utf-8")).hexdigest(),
            "size_bytes": 0,
            "sampled_bytes": 0,
            "exists": False,
            "path_type": "missing",
        }
    return {
        "path": str(path),
        "revision": inferred_revision,
        "fingerprint": fingerprint,
    }


def detect_device_name() -> str | None:
    """Return the current HIP device marketing name, or ``None`` if unavailable."""

    try:
        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        hip.hipGetDevice.restype = ctypes.c_int
        device = ctypes.c_int()
        if int(hip.hipGetDevice(ctypes.byref(device))) != 0:
            return None
        hip.hipDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        hip.hipDeviceGetName.restype = ctypes.c_int
        name = ctypes.create_string_buffer(256)
        if int(hip.hipDeviceGetName(name, len(name), int(device.value))) != 0:
            return None
        text = name.value.decode("utf-8", errors="replace").strip()
        return text or None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _command_output(command: Sequence[str]) -> str | None:
    result = _run(command, timeout=10.0)
    text = result.stdout.strip()
    return text if result.returncode == 0 and text else None


def _detect_rocm_version(hipcc_version: str | None) -> str | None:
    # The hipcc resolved from PATH identifies the active toolchain. This must
    # win over an unrelated host /opt/rocm tree when benchmarks run inside a
    # hermetic TheRock environment.
    if hipcc_version:
        for line in hipcc_version.splitlines():
            if "HIP version" in line or "ROCm" in line:
                return line.strip()
    for candidate in (
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/.info/version-dev"),
    ):
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return None


def collect_artifact_provenance(
    *,
    repo_root: str | Path,
    configured_backend: str = "auto",
    resolved_backend: str | None = None,
    detected_arches: Sequence[str] | None = None,
    target_arch: str | None = None,
    device_name: str | None = None,
    model_path: str | Path | None = None,
    model_revision: str | None = None,
    quant: str | None = None,
    kv_dtype: str | None = None,
    command: Sequence[str] | None = None,
    environment: Mapping[str, str | None] | None = None,
    build_profile: str | None = None,
    timing_protocol: str | None = None,
    warmups: int | None = None,
    repetitions: int | None = None,
    profiler: Mapping[str, Any] | None = None,
    host_name: str | None = None,
    rocm_version: str | None | object = _UNSET,
    hipcc_version: str | None | object = _UNSET,
) -> dict[str, Any]:
    """Collect and validate the canonical artifact provenance object."""

    repo = collect_repo_state(repo_root)
    requested = str(configured_backend).strip() or "auto"
    selected_target = None if target_arch is None else str(target_arch).strip() or None
    if selected_target is None:
        selected_target = (os.environ.get("HIPENGINE_HIP_ARCH") or "").strip() or None
    arches = tuple(detected_arches) if detected_arches is not None else detect_hip_target_arches()
    resolution_arches = arches or ((selected_target,) if selected_target is not None else ())
    resolved = (
        str(resolved_backend).strip()
        if resolved_backend is not None and str(resolved_backend).strip()
        else select_backend(requested, detected_arches=resolution_arches).backend
    )
    if selected_target is None:
        selected_target = HIP_BACKEND_TARGET_ARCH.get(resolved)
    if selected_target is None and arches:
        selected_target = str(arches[0])

    model = collect_model_identity(model_path, revision=model_revision)
    if hipcc_version is _UNSET:
        detected_hipcc_version = _command_output(("hipcc", "--version"))
    else:
        detected_hipcc_version = None if hipcc_version is None else str(hipcc_version)
    if rocm_version is _UNSET:
        detected_rocm_version = _detect_rocm_version(detected_hipcc_version)
    else:
        detected_rocm_version = None if rocm_version is None else str(rocm_version)
    env_payload = (
        {str(key): None if value is None else str(value) for key, value in environment.items()}
        if environment is not None
        else {key: os.environ.get(key) for key in _DEFAULT_ENV_KEYS}
    )
    payload: dict[str, Any] = {
        "kind": ARTIFACT_PROVENANCE_KIND,
        "schema_version": ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "host_name": str(host_name or socket.gethostname()).strip(),
        **repo,
        "configured_backend": requested,
        "resolved_backend": resolved,
        "target_arch": selected_target,
        "device_name": device_name or detect_device_name(),
        "model_path": model["path"],
        "model_revision": model["revision"],
        "model_fingerprint": model["fingerprint"],
        "quant": None if quant is None else str(quant),
        "kv_dtype": None if kv_dtype is None else str(kv_dtype),
        "command": [str(part) for part in (command if command is not None else sys.argv)],
        "environment": env_payload,
        "rocm_version": detected_rocm_version,
        "hipcc_version": detected_hipcc_version,
        "build_profile": None if build_profile is None else str(build_profile),
        "timing_protocol": None if timing_protocol is None else str(timing_protocol),
        "warmups": None if warmups is None else int(warmups),
        "repetitions": None if repetitions is None else int(repetitions),
        "profiler": None if profiler is None else dict(profiler),
    }
    validate_artifact_provenance(payload)
    return payload


def validate_artifact_provenance(
    payload: Mapping[str, Any],
    *,
    require_model: bool = False,
) -> dict[str, Any]:
    """Validate and return a plain dict for the canonical provenance schema."""

    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("artifact provenance schema_version must be 1 or 2")
    required_fields = _REQUIRED_FIELDS + (("host_name",) if schema_version >= 2 else ())
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"artifact provenance missing fields: {missing}")
    if payload.get("kind") != ARTIFACT_PROVENANCE_KIND:
        raise ValueError(f"artifact provenance kind must be {ARTIFACT_PROVENANCE_KIND!r}")
    if not isinstance(payload.get("collected_at"), str) or not str(payload["collected_at"]).strip():
        raise ValueError("artifact provenance collected_at must be a non-empty string")
    for field in (
        "repo_root",
        "hipengine_commit",
        "configured_backend",
        "resolved_backend",
        *(("host_name",) if schema_version >= 2 else ()),
    ):
        if not isinstance(payload.get(field), str) or not str(payload[field]).strip():
            raise ValueError(f"artifact provenance {field} must be a non-empty string")
    if payload["resolved_backend"] == "auto":
        raise ValueError("artifact provenance resolved_backend must be concrete, not 'auto'")
    for field in ("staged_dirty", "unstaged_dirty", "untracked_dirty", "dirty"):
        if type(payload.get(field)) is not bool:
            raise ValueError(f"artifact provenance {field} must be a bool")
    untracked_count = payload.get("untracked_count")
    if type(untracked_count) is not int or untracked_count < 0:
        raise ValueError("artifact provenance untracked_count must be a non-negative int")
    expected_dirty = bool(
        payload["staged_dirty"] or payload["unstaged_dirty"] or payload["untracked_dirty"]
    )
    if payload["dirty"] is not expected_dirty:
        raise ValueError("artifact provenance dirty does not match the three dirty axes")
    if payload["untracked_dirty"] is not bool(untracked_count):
        raise ValueError("artifact provenance untracked_dirty does not match untracked_count")
    if str(payload["resolved_backend"]).startswith("hip_"):
        if not isinstance(payload.get("target_arch"), str) or not str(payload["target_arch"]).strip():
            raise ValueError("HIP artifact provenance requires target_arch")
    if str(payload["resolved_backend"]).startswith("hip_") or payload["resolved_backend"] == "vulkan":
        if not isinstance(payload.get("device_name"), str) or not str(payload["device_name"]).strip():
            raise ValueError("GPU artifact provenance requires device_name")
    if payload.get("git_branch") is not None and not isinstance(payload.get("git_branch"), str):
        raise ValueError("artifact provenance git_branch must be a string or null")
    for field in (
        "target_arch",
        "device_name",
        "model_path",
        "model_revision",
        "quant",
        "kv_dtype",
        "rocm_version",
        "hipcc_version",
        "build_profile",
        "timing_protocol",
    ):
        if payload.get(field) is not None and not isinstance(payload.get(field), str):
            raise ValueError(f"artifact provenance {field} must be a string or null")
    command = payload.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        raise ValueError("artifact provenance command must be a non-empty list of strings")
    environment = payload.get("environment")
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or (value is not None and not isinstance(value, str))
        for key, value in environment.items()
    ):
        raise ValueError("artifact provenance environment must map strings to strings or null")
    for field in ("warmups", "repetitions"):
        value = payload.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"artifact provenance {field} must be a non-negative int or null")
    if payload.get("profiler") is not None and not isinstance(payload.get("profiler"), Mapping):
        raise ValueError("artifact provenance profiler must be an object or null")

    fingerprint = payload.get("model_fingerprint")
    if fingerprint is not None:
        if not isinstance(fingerprint, Mapping):
            raise ValueError("artifact provenance model_fingerprint must be an object or null")
        for field in ("algorithm", "value"):
            if not isinstance(fingerprint.get(field), str) or not str(fingerprint[field]).strip():
                raise ValueError(f"artifact provenance model_fingerprint.{field} is required")
        for field in ("size_bytes", "sampled_bytes"):
            value = fingerprint.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"artifact provenance model_fingerprint.{field} must be a non-negative int"
                )
        if type(fingerprint.get("exists")) is not bool:
            raise ValueError("artifact provenance model_fingerprint.exists must be a bool")
        if fingerprint.get("path_type") not in {"file", "directory", "missing"}:
            raise ValueError("artifact provenance model_fingerprint.path_type is invalid")
    if require_model:
        if not isinstance(payload.get("model_path"), str) or not str(payload["model_path"]).strip():
            raise ValueError("artifact provenance model_path is required")
        if not isinstance(fingerprint, Mapping) or fingerprint.get("exists") is not True:
            raise ValueError("artifact provenance requires an existing model_fingerprint")
    return dict(payload)
