#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$repo_root/hipengine/kernels/hip_gfx1100/attention/aotriton_release.toml"
cache_root="${HIPENGINE_AOTRITON_HOME:-$HOME/.cache/hipengine/aotriton}"
vendor_base="$repo_root/hipengine/kernels/hip_gfx1100/attention/aotriton_runtime"
fetch=1
force=0
verify_sha=1
local_tarball_dir=""
no_prune=0

usage() {
  cat <<'EOF'
Usage: scripts/vendor_aotriton.sh [options]

Fetch the manifest-pinned AOTriton release into a cache, then copy the pruned
runtime/images needed by hipEngine into the repository vendor tree:

  hipengine/kernels/hip_gfx1100/attention/aotriton_runtime/<version>/

The resulting binary/image files are intended to be tracked with Git LFS.  This
script is the reproducible path for refreshing or bumping the vendored AOTriton
baseline after editing aotriton_release.toml.

Options:
  --manifest PATH            Manifest TOML (default: hipengine/.../aotriton_release.toml)
  --cache-root PATH          External fetch cache root (default: ${HIPENGINE_AOTRITON_HOME:-~/.cache/hipengine/aotriton})
  --vendor-base PATH         Vendor output parent (default: hipengine/.../aotriton_runtime)
  --local-tarball-dir PATH   Reuse already-downloaded tarballs by filename
  --skip-fetch               Copy from an already-populated cache; do not run fetch_aotriton.sh
  --no-verify-sha            Pass through to fetch_aotriton.sh
  --no-prune                 Pass through to fetch_aotriton.sh cache fetch; vendor pruning still follows [aotriton.vendor]
  --force                    Re-fetch/re-copy over existing cache/vendor version directories
  -h, --help                 Show this help

Typical use:
  git lfs install
  scripts/vendor_aotriton.sh --force
  git add .gitattributes hipengine/kernels/hip_gfx1100/attention/aotriton_runtime
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="$2"; shift 2 ;;
    --cache-root|--cache) cache_root="$2"; shift 2 ;;
    --vendor-base|--dest) vendor_base="$2"; shift 2 ;;
    --local-tarball-dir) local_tarball_dir="$2"; shift 2 ;;
    --skip-fetch) fetch=0; shift ;;
    --no-verify-sha) verify_sha=0; shift ;;
    --no-prune) no_prune=1; shift ;;
    --force) force=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ $fetch -eq 1 ]]; then
  fetch_args=(--manifest "$manifest" --dest "$cache_root")
  [[ $force -eq 1 ]] && fetch_args+=(--force)
  [[ $verify_sha -eq 0 ]] && fetch_args+=(--no-verify-sha)
  [[ $no_prune -eq 1 ]] && fetch_args+=(--no-prune)
  [[ -n "$local_tarball_dir" ]] && fetch_args+=(--local-tarball-dir "$local_tarball_dir")
  "$repo_root/scripts/fetch_aotriton.sh" "${fetch_args[@]}"
fi

python3 - "$manifest" "$cache_root" "$vendor_base" "$force" <<'PY'
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 only
    import tomli as tomllib  # type: ignore

manifest_path = Path(sys.argv[1]).expanduser().resolve()
cache_root = Path(sys.argv[2]).expanduser().resolve()
vendor_base = Path(sys.argv[3]).expanduser().resolve()
force = bool(int(sys.argv[4]))

cfg = tomllib.loads(manifest_path.read_text())
aot = cfg["aotriton"]
version = str(aot["version"])
so_name = str(aot.get("so_name", "libaotriton_v2.so"))
vendor_cfg = aot.get("vendor", {})
relative_path = str(vendor_cfg.get("relative_path", f"aotriton_runtime/{version}"))
# The manifest path is relative to hipengine/kernels/hip_gfx1100/attention/.
default_vendor_dir = manifest_path.parent / relative_path
vendor_dir = vendor_base / version if vendor_base.name == "aotriton_runtime" else Path(vendor_base)
if relative_path and vendor_base.name == "aotriton_runtime":
    vendor_dir = default_vendor_dir if vendor_base == manifest_path.parent / "aotriton_runtime" else vendor_base / version
image_glob = str(vendor_cfg.get("image_glob", "lib/aotriton.images/amd-gfx11xx/flash/attn_fwd/*.aks2"))
expected_count = int(vendor_cfg.get("image_count", 0) or 0)

source_dir = cache_root / version
if not source_dir.is_dir():
    raise SystemExit(
        f"AOTriton cache {source_dir} is missing. Run scripts/fetch_aotriton.sh or omit --skip-fetch."
    )
if vendor_dir.exists():
    if not force:
        raise SystemExit(f"vendor directory already exists: {vendor_dir} (use --force)")
    shutil.rmtree(vendor_dir)

include_src = source_dir / "include"
lib_src = source_dir / "lib" / so_name
if not include_src.is_dir():
    raise SystemExit(f"missing include tree in cache: {include_src}")
if not lib_src.is_file():
    raise SystemExit(f"missing AOTriton library in cache: {lib_src}")

vendor_dir.mkdir(parents=True)
shutil.copytree(include_src, vendor_dir / "include")
(vendor_dir / "lib").mkdir(parents=True)
shutil.copy2(lib_src, vendor_dir / "lib" / so_name)
link = vendor_dir / "lib" / "libaotriton_v2.so"
try:
    link.symlink_to(so_name)
except OSError:
    shutil.copy2(lib_src, link)

image_root = source_dir
images = [p for p in source_dir.rglob("*.aks2") if fnmatch.fnmatch(str(p.relative_to(source_dir)), image_glob)]
images = sorted(images)
if expected_count and len(images) != expected_count:
    raise SystemExit(
        f"vendor image count mismatch for {image_glob}: expected {expected_count}, found {len(images)}"
    )
if not images:
    raise SystemExit(f"no images matched vendor glob {image_glob!r} under {source_dir}")

image_entries = []
for src in images:
    rel = src.relative_to(source_dir)
    dst = vendor_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    h = hashlib.sha256()
    h.update(dst.read_bytes())
    image_entries.append({"path": str(rel), "bytes": dst.stat().st_size, "sha256": h.hexdigest()})

h = hashlib.sha256()
h.update((vendor_dir / "lib" / so_name).read_bytes())
manifest = {
    "version": version,
    "source_cache_layout": "${HIPENGINE_AOTRITON_HOME:-~/.cache/hipengine/aotriton}/<version>",
    "source_manifest": os.path.relpath(manifest_path, vendor_dir),
    "library": f"lib/{so_name}",
    "library_bytes": (vendor_dir / "lib" / so_name).stat().st_size,
    "library_sha256": h.hexdigest(),
    "images_policy": "manifest [aotriton.vendor].image_glob",
    "image_glob": image_glob,
    "image_count": len(image_entries),
    "images": image_entries,
}
(vendor_dir / "MANIFEST.vendor.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Vendored AOTriton {version} at {vendor_dir}")
print(f"Images: {len(image_entries)}")
print(f"Size: {sum(p.stat().st_size for p in vendor_dir.rglob('*') if p.is_file())} bytes")
PY
