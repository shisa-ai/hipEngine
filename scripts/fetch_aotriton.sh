#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$repo_root/hipengine/kernels/hip_gfx1100/attention/aotriton_release.toml"
dest="${HIPENGINE_AOTRITON_HOME:-$HOME/.cache/hipengine/aotriton}"
force=0
prune=1
dry_run=0
verify_sha=1

usage() {
  cat <<'EOF'
Usage: scripts/fetch_aotriton.sh [options]

Fetch the pinned standalone AOTriton runtime tarball into a local cache.  The
source submodule is for audit/header reference only; this script provides the
actual libaotriton_v2.so + aotriton.images runtime used by hipENGINE wrappers.

Options:
  --manifest PATH     Manifest TOML (default: hipengine/.../aotriton_release.toml)
  --dest PATH         Cache root (default: ${HIPENGINE_AOTRITON_HOME:-~/.cache/hipengine/aotriton})
  --no-prune          Keep all architectures and flash image directories
  --no-verify-sha     Skip SHA256 verification (offline mirrors only)
  --force             Re-extract even if the version directory exists
  --dry-run           Print the fetch/extract plan without downloading
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="$2"; shift 2 ;;
    --dest) dest="$2"; shift 2 ;;
    --no-prune) prune=0; shift ;;
    --no-verify-sha) verify_sha=0; shift ;;
    --force) force=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

python3 - "$manifest" "$dest" "$force" "$prune" "$dry_run" "$verify_sha" <<'PY'
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

cfg = tomllib.loads(manifest_path.read_text())
aot = cfg["aotriton"]
archive = aot["archive"]
prune_cfg = aot.get("prune", {})
version = aot["version"]
url = archive["url"]
expected_sha = archive.get("sha256", "")
so_name = aot.get("so_name", "libaotriton_v2.so")
version_dir = dest_root / version
tar_dir = dest_root / "tarballs"
tar_path = tar_dir / Path(url).name

plan = {
    "manifest": str(manifest_path),
    "url": url,
    "tarball": str(tar_path),
    "dest": str(version_dir),
    "prune": prune,
    "verify_sha": verify_sha,
    "so_name": so_name,
}
if dry_run:
    print(json.dumps(plan, indent=2))
    raise SystemExit(0)

if version_dir.exists() and not force:
    print(f"AOTriton {version} already installed at {version_dir}")
    print(f"Set HIPENGINE_AOTRITON_RUNTIME_ROOT={version_dir}")
    raise SystemExit(0)

tar_dir.mkdir(parents=True, exist_ok=True)
if not tar_path.exists():
    print(f"Downloading {url} -> {tar_path}", flush=True)
    with urllib.request.urlopen(url) as response, tar_path.open("wb") as out:
        shutil.copyfileobj(response, out)

if verify_sha and expected_sha:
    h = hashlib.sha256()
    with tar_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha:
        raise SystemExit(f"SHA256 mismatch for {tar_path}: expected {expected_sha}, got {actual}")

if version_dir.exists():
    shutil.rmtree(version_dir)
def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if dest_resolved != target and dest_resolved not in target.parents:
            raise SystemExit(f"refusing to extract path outside destination: {member.name}")
    tar.extractall(dest)


with tempfile.TemporaryDirectory(prefix="hipengine-aotriton-", dir=str(dest_root)) as td:
    tmp = Path(td)
    print(f"Extracting {tar_path}", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        _safe_extract(tar, tmp)
    candidates = [p for p in tmp.rglob("lib") if any(p.glob("libaotriton_v2.so*"))]
    if not candidates:
        raise SystemExit("extracted archive does not contain lib/libaotriton_v2.so*")
    root = candidates[0].parent
    shutil.copytree(root, version_dir)

if prune:
    images_roots = [version_dir / "lib" / "aotriton.images", version_dir / "aotriton.images"]
    images_root = next((p for p in images_roots if p.is_dir()), None)
    if images_root is not None:
        keep_arch = set(prune_cfg.get("architectures", []))
        keep_flash = set(prune_cfg.get("flash_subdirs", []))
        if keep_arch:
            for child in images_root.iterdir():
                if child.is_dir() and child.name not in keep_arch:
                    shutil.rmtree(child)
        for arch in images_root.iterdir():
            flash = arch / "flash"
            if not flash.is_dir() or not keep_flash:
                continue
            for child in flash.iterdir():
                if child.is_dir() and child.name not in keep_flash:
                    shutil.rmtree(child)

local_manifest = {
    "manifest": str(manifest_path),
    "version": version,
    "url": url,
    "tarball": str(tar_path),
    "sha256": expected_sha,
    "fetched_at_unix": int(time.time()),
    "pruned": prune,
}
(version_dir / "MANIFEST.local.json").write_text(json.dumps(local_manifest, indent=2) + "\n")
print(f"Installed AOTriton {version} at {version_dir}")
print(f"Set HIPENGINE_AOTRITON_RUNTIME_ROOT={version_dir}")
PY
