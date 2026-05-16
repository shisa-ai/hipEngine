"""AOTriton source/runtime discovery for gfx1100 attention wrappers.

This module is intentionally torch-free.  It does not import or load AOTriton;
it only centralizes path discovery so the eventual C-ABI shim can build against
the pinned submodule instead of an ad-hoc local download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_AOTRITON_SOURCE_ENV = "HIPENGINE_AOTRITON_SOURCE_ROOT"
_AOTRITON_RUNTIME_ENV = "HIPENGINE_AOTRITON_RUNTIME_ROOT"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SUBMODULE_ROOT = _REPO_ROOT / "third_party" / "aotriton"


@dataclass(frozen=True)
class AotritonSourceTree:
    """Paths needed to compile a torch-free AOTriton shim."""

    root: Path
    include_dir: Path
    flash_header: Path


@dataclass(frozen=True)
class AotritonRuntimeTree:
    """Paths needed to build/load against an extracted AOTriton runtime tarball."""

    root: Path
    include_dir: Path
    flash_header: Path
    library: Path
    images_dir: Path


def aotriton_source_tree(root: str | Path | None = None) -> AotritonSourceTree:
    """Return the pinned AOTriton source tree used for headers.

    Search order:
    1. explicit ``root`` argument,
    2. ``HIPENGINE_AOTRITON_SOURCE_ROOT``,
    3. the repository submodule at ``third_party/aotriton``.
    """

    base = _resolve_root(root, _AOTRITON_SOURCE_ENV, _SUBMODULE_ROOT)
    include_dir = base / "include"
    flash_header = include_dir / "aotriton" / "flash.h"
    if not flash_header.is_file():
        raise FileNotFoundError(f"AOTriton flash header not found: {flash_header}")
    return AotritonSourceTree(root=base, include_dir=include_dir, flash_header=flash_header)


def aotriton_runtime_tree(root: str | Path | None = None) -> AotritonRuntimeTree:
    """Return a built AOTriton runtime tree.

    Unlike the source tree helper, this never falls back to the older local
    ``~/Downloads`` dump.  Set ``HIPENGINE_AOTRITON_RUNTIME_ROOT`` (or pass
    ``root``) to a built tree containing ``lib/libaotriton_v2.so*`` and
    ``aotriton.images``.  If the submodule has been built in-place, it is used
    automatically.
    """

    base = _resolve_root(root, _AOTRITON_RUNTIME_ENV, _SUBMODULE_ROOT)
    lib_dir = base / "lib"
    candidates = sorted(lib_dir.glob("libaotriton_v2.so*"))
    if not candidates:
        raise FileNotFoundError(
            f"AOTriton runtime library not found under {lib_dir}; build the submodule or set {_AOTRITON_RUNTIME_ENV}"
        )
    include_dir = base / "include"
    flash_header = include_dir / "aotriton" / "flash.h"
    if not flash_header.is_file():
        raise FileNotFoundError(f"AOTriton runtime header not found: {flash_header}")
    image_candidates = (base / "lib" / "aotriton.images", base / "aotriton.images")
    images_dir = next((path for path in image_candidates if path.is_dir()), None)
    if images_dir is None:
        raise FileNotFoundError(
            "AOTriton images directory not found under " + " or ".join(str(path) for path in image_candidates)
        )
    return AotritonRuntimeTree(root=base, include_dir=include_dir, flash_header=flash_header, library=candidates[0], images_dir=images_dir)


def _resolve_root(root: str | Path | None, env_name: str, default: Path) -> Path:
    value = root if root is not None else os.environ.get(env_name)
    return (Path(value) if value is not None else default).expanduser().resolve()
