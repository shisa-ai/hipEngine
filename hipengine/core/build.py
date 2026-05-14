"""Torch-free HIP/CUDA JIT build cache skeleton.

The build key is a hash of source bytes, normalized flags, and compiler version. Tests use
``dry_run=True`` / ``plan_hip_build`` so no ROCm installation is required for this scaffold.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

CompilerKind = Literal["hip", "cuda"]
ProfileName = Literal["decode", "prefill", "baseline"]

DEFAULT_CACHE_ROOT = Path("~/.cache/hipengine/build").expanduser()


@dataclass(frozen=True)
class BuildProfile:
    name: ProfileName
    flags: tuple[str, ...]
    wavefront: int


@dataclass(frozen=True)
class BuildArtifact:
    family: str
    profile: BuildProfile
    cache_key: str
    cache_dir: Path
    output_path: Path
    command: tuple[str, ...]
    sources: tuple[Path, ...]
    flags: tuple[str, ...]
    compiler: str
    compiler_version: str


PROFILES: dict[ProfileName, BuildProfile] = {
    "decode": BuildProfile(
        name="decode",
        flags=("-mllvm", "-amdgpu-unroll-threshold-local=600", "-mcumode"),
        wavefront=32,
    ),
    "prefill": BuildProfile(
        name="prefill",
        flags=("-mllvm", "-amdgpu-unroll-threshold-local=600"),
        wavefront=32,
    ),
    "baseline": BuildProfile(name="baseline", flags=(), wavefront=32),
}


def plan_hip_build(
    *,
    sources: Sequence[str | Path],
    family: str,
    profile: ProfileName = "baseline",
    cache_root: str | Path | None = None,
    compiler: str = "hipcc",
    compiler_version: str | None = None,
    include_dirs: Sequence[str | Path] = (),
    extra_flags: Sequence[str] = (),
    output_name: str | None = None,
) -> BuildArtifact:
    """Return the deterministic build artifact plan without invoking a compiler."""

    if not family:
        raise ValueError("family must be non-empty")
    build_profile = _profile(profile)
    source_paths = tuple(_resolve_source(path) for path in sources)
    if not source_paths:
        raise ValueError("at least one source is required")
    compiler_version = compiler_version or f"{compiler}:unprobed"
    include_flags = tuple(f"-I{Path(path).expanduser()}" for path in include_dirs)
    flags = (*build_profile.flags, *include_flags, *tuple(extra_flags))
    cache_key = _cache_key(
        sources=source_paths,
        flags=flags,
        compiler=compiler,
        compiler_version=compiler_version,
    )
    root = Path(cache_root).expanduser() if cache_root is not None else DEFAULT_CACHE_ROOT
    cache_dir = root / f"{family}-{cache_key[:16]}"
    output_path = cache_dir / (output_name or f"{family}.so")
    command = (
        compiler,
        "-shared",
        "-fPIC",
        "-O3",
        *flags,
        *(str(path) for path in source_paths),
        "-o",
        str(output_path),
    )
    return BuildArtifact(
        family=family,
        profile=build_profile,
        cache_key=cache_key,
        cache_dir=cache_dir,
        output_path=output_path,
        command=command,
        sources=source_paths,
        flags=flags,
        compiler=compiler,
        compiler_version=compiler_version,
    )


def build_hip(
    *,
    sources: Sequence[str | Path],
    family: str,
    profile: ProfileName = "baseline",
    cache_root: str | Path | None = None,
    compiler: str = "hipcc",
    compiler_version: str | None = None,
    include_dirs: Sequence[str | Path] = (),
    extra_flags: Sequence[str] = (),
    output_name: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    """Build a HIP shared object into the hash cache and load it with ``ctypes``.

    ``dry_run=True`` returns the planned artifact without creating directories or invoking
    ``hipcc``. ``load=False`` builds or reuses the shared object but returns metadata instead
    of calling ``ctypes.CDLL``.

    ``require_cached=True`` refuses to invoke ``hipcc`` when the expected ``.so`` is missing.
    This is useful under ``rocprofv3`` because the profiler preloads into child processes and
    can hang or abort when a profiled Python process spawns ``hipcc``/clang. Pair it with an
    explicit ``compiler_version`` or ``HIPENGINE_COMPILER_VERSION_FILE`` so the cache key can be
    computed without probing ``hipcc --version``.
    """

    version = _resolve_compiler_version(
        compiler=compiler,
        compiler_version=compiler_version,
        dry_run=dry_run,
    )
    artifact = plan_hip_build(
        sources=sources,
        family=family,
        profile=profile,
        cache_root=cache_root,
        compiler=compiler,
        compiler_version=version,
        include_dirs=include_dirs,
        extra_flags=extra_flags,
        output_name=output_name,
    )
    if dry_run:
        return artifact

    if force or not artifact.output_path.exists():
        if require_cached:
            raise FileNotFoundError(
                "cached build artifact missing for require_cached=True: "
                f"{artifact.output_path}. Prebuild outside rocprofv3 or pass the same "
                "compiler_version used by the cached artifact."
            )
        artifact.cache_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(artifact)
        subprocess.run(artifact.command, check=True)
    if not load:
        return artifact
    return ctypes.CDLL(str(artifact.output_path))


def compiler_version_text(compiler: str) -> str:
    result = subprocess.run(
        (compiler, "--version"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def _resolve_compiler_version(
    *,
    compiler: str,
    compiler_version: str | None,
    dry_run: bool,
) -> str:
    if compiler_version is not None:
        return compiler_version.strip()
    env_version = _compiler_version_from_environment(compiler)
    if env_version is not None:
        return env_version
    if dry_run:
        return f"{compiler}:unprobed"
    return compiler_version_text(compiler)


def _compiler_version_from_environment(compiler: str) -> str | None:
    specific = _compiler_env_prefix(compiler)
    for name in (f"{specific}_VERSION_TEXT", "HIPENGINE_COMPILER_VERSION_TEXT"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    for name in (f"{specific}_VERSION_FILE", "HIPENGINE_COMPILER_VERSION_FILE"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().read_text().strip()
    return None


def _compiler_env_prefix(compiler: str) -> str:
    basename = Path(compiler).name or compiler
    safe = "".join(char if char.isalnum() else "_" for char in basename).upper()
    return f"HIPENGINE_{safe}"


def _profile(name: ProfileName) -> BuildProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown build profile {name!r}; expected one of: {valid}") from exc


def _resolve_source(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _cache_key(
    *,
    sources: Sequence[Path],
    flags: Sequence[str],
    compiler: str,
    compiler_version: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"hipengine-build-v1\0")
    digest.update(compiler.encode())
    digest.update(b"\0")
    digest.update(compiler_version.encode())
    digest.update(b"\0")
    for flag in flags:
        digest.update(flag.encode())
        digest.update(b"\0")
    for source in sources:
        digest.update(os.fsencode(source.name))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_manifest(artifact: BuildArtifact) -> None:
    manifest = artifact.cache_dir / "manifest.txt"
    manifest.write_text(
        "\n".join(
            (
                f"family={artifact.family}",
                f"profile={artifact.profile.name}",
                f"wavefront={artifact.profile.wavefront}",
                f"cache_key={artifact.cache_key}",
                f"compiler={artifact.compiler}",
                "compiler_version<<EOF",
                artifact.compiler_version,
                "EOF",
                "command=" + " ".join(artifact.command),
                "sources=" + ",".join(str(path) for path in artifact.sources),
            )
        )
        + "\n"
    )
