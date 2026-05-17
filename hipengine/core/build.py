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

_ENV_HIP_ARCH = "HIPENGINE_HIP_ARCH"
_ENV_HIP_OFFLOAD_ARCH = "HIPENGINE_HIP_OFFLOAD_ARCH"
_ENV_ROCM_DEVICE_LIB_PATH = "HIPENGINE_ROCM_DEVICE_LIB_PATH"
_ENV_HIP_DEVICE_LIB_PATH = "HIP_DEVICE_LIB_PATH"

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
    target_arch: str | None = None


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
    target_arch: str | None = None,
    output_name: str | None = None,
) -> BuildArtifact:
    """Return the deterministic build artifact plan without invoking a compiler.

    ``target_arch`` is the native HIP offload architecture, e.g. ``gfx1100`` or
    ``gfx1151``. When omitted, ``HIPENGINE_HIP_ARCH`` /
    ``HIPENGINE_HIP_OFFLOAD_ARCH`` provide a process-wide default. The resulting
    ``--offload-arch=...`` flag is part of the cache key so gfx1100 and gfx1151
    code objects never share artifacts. When a ROCm device-library path is provided
    through ``HIPENGINE_ROCM_DEVICE_LIB_PATH`` / ``HIP_DEVICE_LIB_PATH``, it is also
    emitted as an explicit compiler flag and included in the cache key.
    """

    if not family:
        raise ValueError("family must be non-empty")
    build_profile = _profile(profile)
    source_paths = tuple(_resolve_source(path) for path in sources)
    if not source_paths:
        raise ValueError("at least one source is required")
    compiler_version = compiler_version or f"{compiler}:unprobed"
    include_flags = tuple(f"-I{Path(path).expanduser()}" for path in include_dirs)
    target_arch = _normalize_target_arch(target_arch or _target_arch_from_environment())
    arch_flags = (f"--offload-arch={target_arch}",) if target_arch is not None else ()
    device_lib_flags = _rocm_device_lib_flags()
    flags = (*build_profile.flags, *arch_flags, *device_lib_flags, *include_flags, *tuple(extra_flags))
    flags = _maybe_enable_prefill_mcumode(flags, build_profile)
    flags = _maybe_disable_unroll600(flags)
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
        target_arch=target_arch,
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
    target_arch: str | None = None,
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
        target_arch=target_arch,
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


def _target_arch_from_environment() -> str | None:
    return os.environ.get(_ENV_HIP_ARCH) or os.environ.get(_ENV_HIP_OFFLOAD_ARCH)


def _rocm_device_lib_flags() -> tuple[str, ...]:
    path = os.environ.get(_ENV_ROCM_DEVICE_LIB_PATH) or os.environ.get(_ENV_HIP_DEVICE_LIB_PATH)
    if not path:
        return ()
    resolved = str(Path(path).expanduser())
    return (f"--rocm-device-lib-path={resolved}",)


def _normalize_target_arch(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if any(char.isspace() for char in stripped):
        raise ValueError(f"HIP target architecture must not contain whitespace: {value!r}")
    return stripped


def _maybe_enable_prefill_mcumode(flags: tuple[str, ...], profile: BuildProfile) -> tuple[str, ...]:
    """Diagnostic P1.6 ablation: add ``-mcumode`` to prefill-profile builds.

    Most dual-use decode/prefill libraries already build with the decode profile,
    and the compact-WMMA prefill library adds ``-mcumode`` explicitly.  This knob
    isolates the remaining prefill-profile surfaces without changing decode
    kernels or duplicating the flag on libraries that already request it.
    """

    if profile.name != "prefill" or not _env_truthy(os.environ.get("HIPENGINE_PREFILL_MCUMODE")):
        return flags
    if "-mcumode" in flags:
        return flags
    return (*flags, "-mcumode")


def _maybe_disable_unroll600(flags: tuple[str, ...]) -> tuple[str, ...]:
    """Diagnostic W.1 ablation: strip only the unroll-600 pair from build flags.

    This keeps other profile flags (notably decode `-mcumode`) intact, so the probe answers
    whether `-mllvm -amdgpu-unroll-threshold-local=600` itself helps the hot kernels.
    """

    if not _env_truthy(os.environ.get("HIPENGINE_DISABLE_UNROLL600")):
        return flags
    out: list[str] = []
    i = 0
    while i < len(flags):
        if (
            i + 1 < len(flags)
            and flags[i] == "-mllvm"
            and flags[i + 1] == "-amdgpu-unroll-threshold-local=600"
        ):
            i += 2
            continue
        out.append(flags[i])
        i += 1
    return tuple(out)


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


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
                f"target_arch={artifact.target_arch or ''}",
                "compiler_version<<EOF",
                artifact.compiler_version,
                "EOF",
                "command=" + " ".join(artifact.command),
                "sources=" + ",".join(str(path) for path in artifact.sources),
            )
        )
        + "\n"
    )
