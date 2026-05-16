"""AOTriton runtime discovery for gfx1100 attention wrappers.

hipENGINE does not vendor AOTriton.  The C++ wrapper (``aotriton_wrap.cc``)
builds against a manifest-pinned AOTriton runtime tree that the user installs
once via ``scripts/fetch_aotriton.sh``.  This module is torch-free and only
resolves paths; it never dlopens AOTriton directly.

Lookup chain (first hit wins):

1. ``HIPENGINE_AOTRITON_LIB`` — explicit path to ``libaotriton_v2.so``.
   The matching ``include/`` and ``aotriton.images/`` trees must live at
   ``<parent>/../include`` and ``<parent>/aotriton.images`` (the standard
   release layout).
2. ``${HIPENGINE_AOTRITON_HOME:-~/.cache/hipengine/aotriton}/<version>/`` —
   the fetch-on-install destination.  ``<version>`` is read from
   ``aotriton_release.toml``.
3. ``/opt/rocm/lib/libaotriton_v2.so`` — only when the SONAME matches the
   manifest's pinned ``so_name``.
4. Nothing found → :class:`AotritonNotInstalledError` with a clear hint
   pointing at ``scripts/fetch_aotriton.sh``.

The intentionally minimal env-var surface mirrors ``docs/PREFILL.md``
"AOTriton distribution and pinning strategy".  Earlier env vars
(``HIPENGINE_AOTRITON_SOURCE_ROOT`` / ``HIPENGINE_AOTRITON_RUNTIME_ROOT``)
have been removed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 only
    import tomli as tomllib  # type: ignore


_MANIFEST_PATH = Path(__file__).with_name("aotriton_release.toml")
_ENV_LIB = "HIPENGINE_AOTRITON_LIB"
_ENV_HOME = "HIPENGINE_AOTRITON_HOME"
_DEFAULT_HOME = Path.home() / ".cache" / "hipengine" / "aotriton"
_SYSTEM_LIB_CANDIDATES = (
    Path("/opt/rocm/lib/libaotriton_v2.so"),
)
_LIB_GLOB_NAME = "libaotriton_v2.so"


class AotritonNotInstalledError(RuntimeError):
    """Raised when no AOTriton runtime can be located via the lookup chain."""


@dataclass(frozen=True)
class AotritonManifest:
    """Pin information read from ``aotriton_release.toml``."""

    version: str
    so_name: str


@dataclass(frozen=True)
class AotritonRuntimeTree:
    """A resolved AOTriton runtime tree on disk."""

    root: Path
    include_dir: Path
    flash_header: Path
    library: Path
    images_dir: Path
    source: str
    manifest: AotritonManifest


def load_manifest(manifest_path: str | Path | None = None) -> AotritonManifest:
    """Load the pinned version + soname from the in-tree manifest."""

    path = Path(manifest_path).expanduser().resolve() if manifest_path is not None else _MANIFEST_PATH
    if not path.is_file():
        raise FileNotFoundError(f"AOTriton manifest not found: {path}")
    data = tomllib.loads(path.read_text())
    aot = data.get("aotriton")
    if not isinstance(aot, dict):
        raise ValueError(f"AOTriton manifest missing [aotriton] table: {path}")
    version = aot.get("version")
    so_name = aot.get("so_name")
    if not version or not so_name:
        raise ValueError(f"AOTriton manifest missing version/so_name: {path}")
    return AotritonManifest(version=str(version), so_name=str(so_name))


def aotriton_runtime_tree(
    root: str | Path | None = None,
    *,
    manifest: AotritonManifest | None = None,
) -> AotritonRuntimeTree:
    """Locate the AOTriton runtime tree via the documented lookup chain.

    ``root`` is a developer override equivalent to ``HIPENGINE_AOTRITON_HOME``
    pointing at a cache root that contains ``<version>/lib/libaotriton_v2.so``.
    Tests use this to construct a fake tree without setting env vars.
    """

    pin = manifest or load_manifest()

    # 1. Explicit library override via env var.
    env_lib = os.environ.get(_ENV_LIB)
    if env_lib:
        lib_path = Path(env_lib).expanduser().resolve()
        if not lib_path.is_file():
            raise FileNotFoundError(
                f"{_ENV_LIB}={env_lib!r} does not point at an existing file"
            )
        return _from_library(lib_path, pin, source=_ENV_LIB)

    # 2. Cache layout under HIPENGINE_AOTRITON_HOME (or developer override).
    home_override = root
    if home_override is None:
        env_home = os.environ.get(_ENV_HOME)
        home_override = env_home if env_home else _DEFAULT_HOME
    home_root = Path(home_override).expanduser().resolve()
    cache_lib = home_root / pin.version / "lib" / _LIB_GLOB_NAME
    if cache_lib.is_file() or cache_lib.is_symlink():
        return _from_library(cache_lib.resolve(), pin, source=str(home_root))

    # 3. System ROCm fallback (SONAME-gated).
    for candidate in _SYSTEM_LIB_CANDIDATES:
        if candidate.is_file() or candidate.is_symlink():
            soname = _read_soname(candidate)
            if soname and _soname_matches(soname, pin.so_name):
                return _from_library(candidate.resolve(), pin, source="system")

    raise AotritonNotInstalledError(
        "AOTriton runtime not found.\n"
        f"Looked at:\n"
        f"  ${_ENV_LIB} (env override)\n"
        f"  {home_root}/{pin.version}/lib/{_LIB_GLOB_NAME}\n"
        f"  {_SYSTEM_LIB_CANDIDATES[0]}\n"
        "Install the pinned release with:\n"
        "  scripts/fetch_aotriton.sh\n"
        "Then either accept the default cache path (~/.cache/hipengine/aotriton) "
        f"or set ${_ENV_HOME} to the cache root."
    )


def _from_library(library: Path, pin: AotritonManifest, *, source: str) -> AotritonRuntimeTree:
    lib_dir = library.parent
    root = lib_dir.parent
    include_dir = root / "include"
    flash_header = include_dir / "aotriton" / "flash.h"
    if not flash_header.is_file():
        raise FileNotFoundError(
            f"AOTriton flash header not found alongside {library}: expected {flash_header}"
        )
    image_candidates = (lib_dir / "aotriton.images", root / "aotriton.images")
    images_dir = next((p for p in image_candidates if p.is_dir()), None)
    if images_dir is None:
        raise FileNotFoundError(
            "AOTriton images directory not found under "
            + " or ".join(str(p) for p in image_candidates)
        )
    return AotritonRuntimeTree(
        root=root,
        include_dir=include_dir,
        flash_header=flash_header,
        library=library,
        images_dir=images_dir,
        source=source,
        manifest=pin,
    )


def _read_soname(library: Path) -> str | None:
    """Best-effort SONAME read.  Returns None when SONAME cannot be determined."""

    try:
        with library.open("rb") as fh:
            data = fh.read(4096 * 16)
    except OSError:
        return None
    # Look for the literal SONAME string the dynamic linker would resolve to.
    matches = re.findall(rb"libaotriton_v2\.so(?:\.[0-9]+)*", data)
    if not matches:
        return None
    return max(matches, key=len).decode("ascii", errors="replace")


def _soname_matches(actual: str, expected: str) -> bool:
    """Match SONAME against the manifest's pinned soname.

    The check accepts exact equality plus any sub-version prefix relationship
    so that an unversioned ``libaotriton_v2.so`` symlink resolving to
    ``libaotriton_v2.so.0.11.2`` is accepted when the manifest pins
    ``libaotriton_v2.so.0.11.2`` or ``libaotriton_v2.so.0.11``.
    """

    if actual == expected:
        return True
    return actual.startswith(expected) or expected.startswith(actual)
