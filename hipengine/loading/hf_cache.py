"""Hugging Face cache path resolution helpers.

These helpers intentionally use local cache state only. hipEngine model loading
must not surprise users by downloading weights during server startup or tests.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_model_path(model_ref: str | Path) -> Path:
    """Resolve a local path or Hugging Face repo id to a filesystem path.

    Resolution order:
    1. Existing filesystem path, after ``~`` expansion.
    2. Hugging Face Hub local cache snapshot for repo ids such as
       ``shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed``.
    3. Non-existing filesystem path fallback, so existing callers still raise
       their normal config/weights errors.

    No network access is performed. If ``huggingface_hub`` is installed, it is
    asked for ``local_files_only=True``; otherwise we inspect the standard cache
    layout directly.
    """

    path = Path(model_ref).expanduser()
    if path.exists():
        return path

    repo_id = str(model_ref)
    if not _could_be_hf_repo_id(repo_id):
        return path

    hub_resolved = _resolve_with_huggingface_hub(repo_id)
    if hub_resolved is not None:
        return hub_resolved

    cache_resolved = _resolve_hf_cache_snapshot(repo_id)
    if cache_resolved is not None:
        return cache_resolved

    return path


def is_hf_repo_id(model_ref: str | Path) -> bool:
    """Return whether a non-existing reference has Hugging Face repo-id shape."""

    text = str(model_ref)
    if not text or text.startswith(("/", "./", "../", "~")):
        return False
    if "://" in text:
        return False
    if any(part in {"", ".", ".."} for part in text.split("/")):
        return False
    # Hugging Face repo IDs are one or two path components for model repos.
    return 1 <= text.count("/") + 1 <= 2


def _could_be_hf_repo_id(text: str) -> bool:
    return is_hf_repo_id(text)


def _resolve_with_huggingface_hub(repo_id: str) -> Path | None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return Path(snapshot_download(repo_id, local_files_only=True)).expanduser()
    except Exception:
        return None


def _resolve_hf_cache_snapshot(repo_id: str) -> Path | None:
    repo_cache_name = "models--" + repo_id.replace("/", "--")
    for hub_cache in _candidate_hub_caches():
        repo_cache = hub_cache / repo_cache_name
        snapshots = repo_cache / "snapshots"
        if not snapshots.is_dir():
            continue

        refs = repo_cache / "refs"
        for ref_name in ("main", "master"):
            ref_path = refs / ref_name
            if ref_path.is_file():
                commit = ref_path.read_text(encoding="utf-8").strip()
                snapshot = snapshots / commit
                if snapshot.is_dir():
                    return snapshot

        snapshot_dirs = [path for path in snapshots.iterdir() if path.is_dir()]
        if len(snapshot_dirs) == 1:
            return snapshot_dirs[0]
        if snapshot_dirs:
            return max(snapshot_dirs, key=lambda path: path.stat().st_mtime)
    return None


def _candidate_hub_caches() -> tuple[Path, ...]:
    candidates: list[Path] = []

    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        candidates.append(Path(explicit).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        candidates.append(Path(xdg_cache).expanduser() / "huggingface" / "hub")

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return tuple(deduped)


__all__ = ["is_hf_repo_id", "resolve_model_path"]
