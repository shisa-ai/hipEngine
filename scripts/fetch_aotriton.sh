#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$repo_root/hipengine/kernels/hip_gfx1100/attention/aotriton_release.toml"
dest="${HIPENGINE_AOTRITON_HOME:-$HOME/.cache/hipengine/aotriton}"
force=0
prune=1
dry_run=0
verify_sha=1
local_tarball_dir=""

usage() {
  cat <<'EOF'
Usage: scripts/fetch_aotriton.sh [options]

Fetch the pinned standalone AOTriton runtime + GPU-image tarballs into a local
cache.  hipEngine vendors the pruned baseline AOTriton tree; this helper is for
refreshing that pin, populating an external override cache, or offline mirrors.
See docs/PREFILL.md "AOTriton distribution and pinning strategy" for the
rationale.

Output layout under the cache:
  ${HIPENGINE_AOTRITON_HOME:-~/.cache/hipengine/aotriton}/<version>/
    include/aotriton/                       # headers (from runtime tarball)
    lib/libaotriton_v2.so[.<version>]       # shared library
    lib/aotriton.images/<arch>/flash/...    # pretuned kernel images
    MANIFEST.local.json                     # provenance + prune state

Options:
  --manifest PATH            Manifest TOML (default: hipengine/.../aotriton_release.toml)
  --dest PATH                Cache root (default: ${HIPENGINE_AOTRITON_HOME:-~/.cache/hipengine/aotriton})
  --local-tarball-dir PATH   Look for pre-downloaded tarballs in PATH before
                             hitting the network (matched by filename).
  --no-prune                 Keep every flash subdir and every arch image dir
  --no-verify-sha            Skip SHA256 verification (offline mirrors only)
  --force                    Re-extract even if the version directory exists
  --dry-run                  Print the fetch/extract plan without downloading
  -h, --help                 Show this help

Environment:
  HIPENGINE_AOTRITON_HOME    Cache root (overridden by --dest)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="$2"; shift 2 ;;
    --dest) dest="$2"; shift 2 ;;
    --local-tarball-dir) local_tarball_dir="$2"; shift 2 ;;
    --no-prune) prune=0; shift ;;
    --no-verify-sha) verify_sha=0; shift ;;
    --force) force=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

python3 - "$manifest" "$dest" "$force" "$prune" "$dry_run" "$verify_sha" "$local_tarball_dir" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 only
    import tomli as tomllib  # type: ignore

manifest_path = Path(sys.argv[1]).expanduser().resolve()
dest_root = Path(sys.argv[2]).expanduser().resolve()
force = bool(int(sys.argv[3]))
prune = bool(int(sys.argv[4]))
dry_run = bool(int(sys.argv[5]))
verify_sha = bool(int(sys.argv[6]))
local_tarball_dir = Path(sys.argv[7]).expanduser().resolve() if sys.argv[7] else None

cfg = tomllib.loads(manifest_path.read_text())
aot = cfg["aotriton"]
archives = aot.get("archives")
if not archives:
    raise SystemExit(
        "manifest does not declare [[aotriton.archives]]; this fetcher requires the "
        "0.11.x-style runtime+images split"
    )
prune_cfg = aot.get("prune", {})
version = aot["version"]
so_name = aot.get("so_name", "libaotriton_v2.so")
version_dir = dest_root / version
tar_dir = dest_root / "tarballs"


def archive_label(entry: dict) -> str:
    if entry.get("kind") == "images":
        return f"images:{entry.get('arch', '?')}"
    return entry.get("kind", "?")


plan_archives = []
for entry in archives:
    if "url" not in entry or "sha256" not in entry:
        raise SystemExit(f"archive entry missing url/sha256: {entry}")
    plan_archives.append({
        "kind": archive_label(entry),
        "url": entry["url"],
        "tarball": str(tar_dir / Path(entry["url"]).name),
        "sha256": entry["sha256"],
        "size_bytes": entry.get("size_bytes"),
    })
plan = {
    "manifest": str(manifest_path),
    "version": version,
    "dest": str(version_dir),
    "prune": prune,
    "verify_sha": verify_sha,
    "so_name": so_name,
    "local_tarball_dir": str(local_tarball_dir) if local_tarball_dir else None,
    "archives": plan_archives,
}
if dry_run:
    print(json.dumps(plan, indent=2))
    raise SystemExit(0)

if version_dir.exists() and not force:
    print(f"AOTriton {version} already installed at {version_dir}")
    print(f"Set HIPENGINE_AOTRITON_HOME={dest_root}  # or omit; default is the same path")
    raise SystemExit(0)

tar_dir.mkdir(parents=True, exist_ok=True)
dest_root.mkdir(parents=True, exist_ok=True)


def fetch_one(entry: dict) -> Path:
    name = Path(entry["url"]).name
    tar_path = tar_dir / name
    if not tar_path.exists() and local_tarball_dir is not None:
        candidate = local_tarball_dir / name
        if candidate.is_file():
            print(f"Using local tarball {candidate}", flush=True)
            shutil.copy2(candidate, tar_path)
    if not tar_path.exists():
        print(f"Downloading {entry['url']} -> {tar_path}", flush=True)
        with urllib.request.urlopen(entry["url"]) as response, tar_path.open("wb") as out:
            shutil.copyfileobj(response, out)
    if verify_sha:
        h = hashlib.sha256()
        with tar_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != entry["sha256"]:
            raise SystemExit(
                f"SHA256 mismatch for {tar_path}: expected {entry['sha256']}, got {actual}"
            )
    return tar_path


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if dest_resolved != target and dest_resolved not in target.parents:
            raise SystemExit(f"refusing to extract path outside destination: {member.name}")
    # Python 3.12+ supports filter='data' which rejects absolute paths,
    # parent-relative entries, devices, and unusual modes — matches our
    # safe_extract checks above with a stable upstream contract.
    try:
        tar.extractall(dest, filter="data")
    except TypeError:  # pragma: no cover - Python <3.12
        tar.extractall(dest)


if version_dir.exists():
    shutil.rmtree(version_dir)

with tempfile.TemporaryDirectory(prefix="hipengine-aotriton-", dir=str(dest_root)) as td:
    staging = Path(td)
    for entry in plan_archives:
        manifest_entry = next(a for a in archives if a["url"] == entry["url"])
        tar_path = fetch_one(manifest_entry)
        print(f"Extracting {tar_path}", flush=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            safe_extract(tar, staging)
    # Both tarballs ship with a top-level 'aotriton/' directory; merge.
    aot_root = staging / "aotriton"
    if not aot_root.is_dir():
        # Fallback: locate the lib/libaotriton_v2.so* inside the staging dir.
        candidates = [p for p in staging.rglob("lib") if any(p.glob("libaotriton_v2.so*"))]
        if not candidates:
            raise SystemExit("extracted archives do not contain lib/libaotriton_v2.so*")
        aot_root = candidates[0].parent
    shutil.copytree(aot_root, version_dir)

if prune:
    keep_flash = set(prune_cfg.get("keep_flash_subdirs", []))
    keep_archs = set(prune_cfg.get("keep_archs", []))
    images_roots = [version_dir / "lib" / "aotriton.images", version_dir / "aotriton.images"]
    images_root = next((p for p in images_roots if p.is_dir()), None)
    if images_root is not None:
        if keep_archs:
            for child in images_root.iterdir():
                if child.is_dir() and child.name not in keep_archs:
                    shutil.rmtree(child)
        if keep_flash:
            for arch in images_root.iterdir():
                flash = arch / "flash"
                if not flash.is_dir():
                    continue
                for child in flash.iterdir():
                    if child.is_dir() and child.name not in keep_flash:
                        shutil.rmtree(child)

local_manifest = {
    "manifest": str(manifest_path),
    "version": version,
    "so_name": so_name,
    "fetched_at_unix": int(time.time()),
    "pruned": prune,
    "archives": [
        {
            "kind": archive_label(entry),
            "url": entry["url"],
            "sha256": entry["sha256"],
            "size_bytes": entry.get("size_bytes"),
        }
        for entry in archives
    ],
}
(version_dir / "MANIFEST.local.json").write_text(json.dumps(local_manifest, indent=2) + "\n")
print(f"Installed AOTriton {version} at {version_dir}")
print(f"Set HIPENGINE_AOTRITON_HOME={dest_root}  # or rely on the default ~/.cache/hipengine/aotriton")
PY
