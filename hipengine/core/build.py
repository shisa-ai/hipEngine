"""Torch-free HIP/CUDA JIT build cache.

Build keys hash source bytes, normalized flags, and compiler versions. HIP and CUDA
planners support dry runs so CPU-only tests do not require either GPU toolchain.
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
_ENV_CUDA_ARCH = "HIPENGINE_CUDA_ARCH"
_ENV_CUDA_TARGET_ARCH = "HIPENGINE_CUDA_TARGET_ARCH"
_ENV_ROCM_DEVICE_LIB_PATH = "HIPENGINE_ROCM_DEVICE_LIB_PATH"
_ENV_HIP_DEVICE_LIB_PATH = "HIP_DEVICE_LIB_PATH"
_ENV_BUILD_CACHE_ROOT = "HIPENGINE_BUILD_CACHE_ROOT"
_ENV_REQUIRE_CACHED_BUILD = "HIPENGINE_REQUIRE_CACHED_BUILD"

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

CUDA_PROFILES: dict[ProfileName, BuildProfile] = {
    name: BuildProfile(name=name, flags=(), wavefront=32)
    for name in ("decode", "prefill", "baseline")
}


def _resolve_cache_root(cache_root: str | Path | None) -> Path:
    if cache_root is not None:
        return Path(cache_root).expanduser()
    environment_root = _env_get(_ENV_BUILD_CACHE_ROOT).strip()
    return Path(environment_root).expanduser() if environment_root else DEFAULT_CACHE_ROOT


# Environment variables that change what a HIP build resolves to. Every knob the
# key derivation (`_resolve_cache_root`, `_target_arch_from_environment`,
# `_resolve_compiler_version`) or the flag builders (`_maybe_prefill_mcumode`,
# `_maybe_disable_unroll600`, `_rocm_device_lib_flags`) reads must appear here,
# because ``_build_fast_key`` treats an unlisted knob as constant for the
# lifetime of the process. `test_unit_build.py` fails if a listed name stops
# changing the key or a new `HIPENGINE_*` knob is read by the build path without
# being listed.
_BUILD_ENV_KEYS = (
    _ENV_BUILD_CACHE_ROOT,
    _ENV_HIP_ARCH,
    _ENV_HIP_OFFLOAD_ARCH,
    _ENV_REQUIRE_CACHED_BUILD,
    _ENV_ROCM_DEVICE_LIB_PATH,
    _ENV_HIP_DEVICE_LIB_PATH,
    "HIPENGINE_PREFILL_MCUMODE",
    "HIPENGINE_DISABLE_UNROLL600",
)
_BUILD_ENV_KEYS_BYTES = tuple(name.encode("ascii") for name in _BUILD_ENV_KEYS)


def _build_env_signature(compiler: str) -> tuple:
    """Raw values of every environment knob the HIP build path reads."""

    if _fast_environ_available():
        values = tuple(_env_value(_ENVIRON_DATA.get(key)) for key in _BUILD_ENV_KEYS_BYTES)
    else:
        values = tuple(os.environ.get(name) or "" for name in _BUILD_ENV_KEYS)
    return values + (_environment_version_identity(compiler),)


def _build_fast_key(
    *,
    family: str,
    profile: ProfileName,
    output_name: str | None,
    sources: Sequence[str | Path],
    cache_root: str | Path | None,
    compiler: str,
    compiler_version: str | None,
    target_arch: str | None,
    include_dirs: Sequence[str | Path],
    extra_flags: Sequence[str],
) -> tuple | None:
    """Cheap identity for a repeated per-launch ``build_X(load=True)`` request.

    The kernel launchers resolve their library on every launch (they do not pass
    ``library=``), so ``build_hip`` re-derives the compiler version, the target
    arch and the cache root once per *kernel launch*: measured at 8.0 us of an
    11.8 us host launch cost, against 1.8 us for the bare ctypes call. This key
    is built from the raw call inputs plus the environment values those
    derivations read, so a repeated request short-circuits before the
    derivations run while a mid-process environment change still selects a
    fresh resolution.

    Returns ``None`` when an input is neither a string nor a path-like: converting
    it is exactly the cost this key exists to avoid, so those calls take the slow
    path.
    """

    try:
        # ``os.fspath`` is a plain attribute read for both ``str`` and ``Path``,
        # unlike ``str(Path(...))``, which re-parses the path (~2 us) and is the
        # second largest term in the derivation this fast path removes.
        source_names = tuple(os.fspath(source) for source in sources)
        include_names = tuple(os.fspath(directory) for directory in include_dirs)
        cache_root_name = None if cache_root is None else os.fspath(cache_root)
    except TypeError:
        return None
    return (
        family,
        profile,
        output_name,
        source_names,
        cache_root_name,
        compiler,
        compiler_version,
        target_arch,
        include_names,
        tuple(extra_flags),
        _build_env_signature(compiler),
    )


# ``os.environ`` is a ``MutableMapping`` wrapper that fsencodes the key on every
# access (~0.46 us on POSIX), and its ``_data`` is the *bytes*-keyed dict it
# shares with ``os.environb``. The per-launch build fast path reads a dozen knobs
# per kernel launch, so it reads that dict directly with precomputed byte keys
# and falls back to the public mapping whenever CPython does not expose it or a
# caller has swapped ``os.environ`` for another object. Both paths observe
# in-place mutations of ``os.environ``, so the environment-response contract is
# unchanged.
_ENVIRON_DATA = getattr(os.environ, "_data", None)
if not isinstance(_ENVIRON_DATA, dict):
    _ENVIRON_DATA = None


def _fast_environ_available() -> bool:
    """True when the raw environment dict may be read directly."""

    return _ENVIRON_DATA is not None and getattr(os.environ, "_data", None) is _ENVIRON_DATA


def _env_value(raw: object) -> str:
    return "" if raw is None else (raw if isinstance(raw, str) else raw.decode("utf-8"))


def _env_get(name: str) -> str:
    """Read one environment variable without ``os.environ``'s per-access cost."""

    if _fast_environ_available():
        return _env_value(_ENVIRON_DATA.get(name.encode("ascii")))
    return os.environ.get(name) or ""


def _environment_requires_cached_build() -> bool:
    raw = _env_get(_ENV_REQUIRE_CACHED_BUILD).strip().lower()
    if not raw:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"invalid {_ENV_REQUIRE_CACHED_BUILD}={raw!r}; expected a boolean value"
    )


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
    root = _resolve_cache_root(cache_root)
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


def plan_cuda_build(
    *,
    sources: Sequence[str | Path],
    family: str,
    profile: ProfileName = "baseline",
    cache_root: str | Path | None = None,
    compiler: str = "nvcc",
    compiler_version: str | None = None,
    include_dirs: Sequence[str | Path] = (),
    extra_flags: Sequence[str] = (),
    target_arch: str | None = None,
    output_name: str | None = None,
) -> BuildArtifact:
    """Return a deterministic CUDA shared-library build plan.

    The default target comes from ``HIPENGINE_CUDA_ARCH`` or
    ``HIPENGINE_CUDA_TARGET_ARCH``. CUDA profiles deliberately start without
    HIP-specific optimization flags; every CUDA flag is explicit and hashed.
    """

    if not family:
        raise ValueError("family must be non-empty")
    build_profile = _profile(profile, CUDA_PROFILES)
    source_paths = tuple(_resolve_source(path) for path in sources)
    if not source_paths:
        raise ValueError("at least one source is required")
    compiler_version = compiler_version or f"{compiler}:unprobed"
    include_flags = tuple(f"-I{Path(path).expanduser()}" for path in include_dirs)
    target_arch = _normalize_cuda_target_arch(
        target_arch or _cuda_target_arch_from_environment()
    )
    arch_flags = (f"-arch={target_arch}",) if target_arch is not None else ()
    flags = (*build_profile.flags, *arch_flags, *include_flags, *tuple(extra_flags))
    cache_key = _cache_key(
        sources=source_paths,
        flags=flags,
        compiler=compiler,
        compiler_version=compiler_version,
    )
    root = _resolve_cache_root(cache_root)
    cache_dir = root / f"{family}-{cache_key[:16]}"
    output_path = cache_dir / (output_name or f"{family}.so")
    command = (
        compiler,
        "-std=c++17",
        "-O3",
        "--shared",
        "-Xcompiler=-fPIC",
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
    computed without probing ``hipcc --version``. ``HIPENGINE_BUILD_CACHE_ROOT``
    scopes implicit cache roots, and ``HIPENGINE_REQUIRE_CACHED_BUILD=1`` makes
    every HIP builder in the process fail closed without per-call plumbing.
    """

    fast_key: tuple | None = None
    if load and not dry_run and not force:
        fast_key = _build_fast_key(
            family=family,
            profile=profile,
            output_name=output_name,
            sources=sources,
            cache_root=cache_root,
            compiler=compiler,
            compiler_version=compiler_version,
            target_arch=target_arch,
            include_dirs=include_dirs,
            extra_flags=extra_flags,
        )
        if fast_key is not None:
            cached_fast = _FAST_PATH_CACHE.get(fast_key)
            if cached_fast is not None:
                return cached_fast

    require_cached = bool(require_cached or _environment_requires_cached_build())

    # Process-level loaded-library cache. Without it, every ``build_hip(load=True)``
    # re-hashes the sources (plan_hip_build) and rebuilds a fresh ``ctypes.CDLL``,
    # which dominates the per-kernel-launch host cost (~90 us; see WORKLOG
    # 2026-06-28 "C-dispatch breakthrough"). On a cache hit this is a dict lookup.
    # The key must capture every build-affecting param (notably extra_flags /
    # include_dirs / target_arch / compiler_version) so different builds of the
    # same family (e.g. WMMA tile variants) do not collide.
    version = _resolve_compiler_version(
        compiler=compiler,
        compiler_version=compiler_version,
        dry_run=dry_run,
    )
    resolved_target_arch = _normalize_target_arch(
        target_arch or _target_arch_from_environment()
    )
    cache_key: tuple | None = None
    if load and not dry_run:
        cache_key = (
            family,
            profile,
            output_name,
            tuple(str(Path(s)) for s in sources),
            str(_resolve_cache_root(cache_root)),
            resolved_target_arch,
            compiler,
            version,
            tuple(str(Path(d)) for d in include_dirs),
            tuple(extra_flags),
        )
        if not force:
            cached_lib = _LOADED_LIB_CACHE.get(cache_key)
            if cached_lib is not None:
                return cached_lib

    artifact = plan_hip_build(
        sources=sources,
        family=family,
        profile=profile,
        cache_root=cache_root,
        compiler=compiler,
        compiler_version=version,
        include_dirs=include_dirs,
        extra_flags=extra_flags,
        target_arch=resolved_target_arch,
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
    lib = ctypes.CDLL(str(artifact.output_path))
    if cache_key is not None:
        _LOADED_LIB_CACHE[cache_key] = lib
    if fast_key is not None:
        _FAST_PATH_CACHE[fast_key] = lib
    return lib


def build_cuda(
    *,
    sources: Sequence[str | Path],
    family: str,
    profile: ProfileName = "baseline",
    cache_root: str | Path | None = None,
    compiler: str = "nvcc",
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
    """Build or reuse a CUDA shared object and optionally load it."""

    require_cached = bool(require_cached or _environment_requires_cached_build())
    version = _resolve_compiler_version(
        compiler=compiler,
        compiler_version=compiler_version,
        dry_run=dry_run,
    )
    resolved_target_arch = _normalize_cuda_target_arch(
        target_arch or _cuda_target_arch_from_environment()
    )
    cache_key: tuple | None = None
    if load and not dry_run:
        cache_key = (
            "cuda",
            family,
            profile,
            output_name,
            tuple(str(Path(source)) for source in sources),
            str(_resolve_cache_root(cache_root)),
            resolved_target_arch,
            compiler,
            version,
            tuple(str(Path(directory)) for directory in include_dirs),
            tuple(extra_flags),
        )
        if not force:
            cached_lib = _LOADED_LIB_CACHE.get(cache_key)
            if cached_lib is not None:
                return cached_lib

    artifact = plan_cuda_build(
        sources=sources,
        family=family,
        profile=profile,
        cache_root=cache_root,
        compiler=compiler,
        compiler_version=version,
        include_dirs=include_dirs,
        extra_flags=extra_flags,
        target_arch=resolved_target_arch,
        output_name=output_name,
    )
    if dry_run:
        return artifact

    if force or not artifact.output_path.exists():
        if require_cached:
            raise FileNotFoundError(
                "cached build artifact missing for require_cached=True: "
                f"{artifact.output_path}. Prebuild outside profilers or pass the same "
                "compiler_version used by the cached artifact."
            )
        artifact.cache_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(artifact)
        subprocess.run(artifact.command, check=True)
    if not load:
        return artifact
    lib = ctypes.CDLL(str(artifact.output_path))
    if cache_key is not None:
        _LOADED_LIB_CACHE[cache_key] = lib
    return lib


def compiler_version_text(compiler: str) -> str:
    result = subprocess.run(
        (compiler, "--version"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


_COMPILER_VERSION_CACHE: dict[str, str] = {}
# Process-level cache of loaded ``ctypes.CDLL`` handles keyed by build identity.
_LOADED_LIB_CACHE: dict[tuple, "ctypes.CDLL"] = {}

# Fast path in front of the loaded-library cache, keyed on the raw request plus
# the build environment signature. See ``_build_fast_key``.
_FAST_PATH_CACHE: dict[tuple, "ctypes.CDLL"] = {}


def _resolve_compiler_version(
    *,
    compiler: str,
    compiler_version: str | None,
    dry_run: bool,
) -> str:
    if compiler_version is not None:
        version = compiler_version.strip()
        # An explicit version becomes the process default so every later
        # per-call ``build_X(load=True)`` (which passes compiler_version=None)
        # resolves to the same version and hits the loaded-library cache. Without
        # this, a pinned session version misses the cache (the launch path
        # resolves a probed version), re-running the full build/load machinery on
        # every launch (~20-30 us/call; see worklog pn5-router-lib-hoist).
        if not dry_run:
            _COMPILER_VERSION_CACHE[compiler] = version
        return version
    env_version = _compiler_version_from_environment(compiler)
    if env_version is not None:
        return env_version
    if dry_run:
        return f"{compiler}:unprobed"
    # Cache the compiler version to avoid 60ms subprocess call per build_hip()
    if compiler in _COMPILER_VERSION_CACHE:
        return _COMPILER_VERSION_CACHE[compiler]
    version = compiler_version_text(compiler)
    _COMPILER_VERSION_CACHE[compiler] = version
    return version


_ENV_VERSION_CACHE: dict[tuple[str, tuple[str, str, str, str]], str] = {}
_VERSION_ENV_KEYS_BYTES: dict[str, tuple[bytes, bytes, bytes, bytes]] = {}


def _version_env_keys_bytes(compiler: str) -> tuple[bytes, bytes, bytes, bytes]:
    """The four version-override variables' byte keys, built once per compiler.

    ``_environment_version_identity`` runs on every per-launch build fast path,
    where building the two per-compiler names with f-strings and re-encoding
    them costs more than the reads themselves.
    """

    keys = _VERSION_ENV_KEYS_BYTES.get(compiler)
    if keys is None:
        specific = _compiler_env_prefix(compiler)
        keys = tuple(
            name.encode("ascii")
            for name in (
                f"{specific}_VERSION_TEXT",
                "HIPENGINE_COMPILER_VERSION_TEXT",
                f"{specific}_VERSION_FILE",
                "HIPENGINE_COMPILER_VERSION_FILE",
            )
        )
        _VERSION_ENV_KEYS_BYTES[compiler] = keys
    return keys


def _environment_version_identity(compiler: str) -> tuple[str, str, str, str]:
    """Raw values of the four version-override environment variables.

    The identity is the cache key component alongside the compiler name:
    changing any override variable (for example switching from
    ``HIPENGINE_COMPILER_VERSION_TEXT`` to ``HIPENGINE_COMPILER_VERSION_FILE``,
    or pointing the file variable at different content) must select a
    different resolution instead of reusing the first cached value.
    """
    if _fast_environ_available():
        return tuple(_env_value(_ENVIRON_DATA.get(key)) for key in _version_env_keys_bytes(compiler))
    specific = _compiler_env_prefix(compiler)
    return (
        _env_get(f"{specific}_VERSION_TEXT"),
        _env_get("HIPENGINE_COMPILER_VERSION_TEXT"),
        _env_get(f"{specific}_VERSION_FILE"),
        _env_get("HIPENGINE_COMPILER_VERSION_FILE"),
    )


def _compiler_version_from_environment(compiler: str) -> str | None:
    # The env/file version is static for a given override identity; cache the
    # resolved value so per-launch ``build_X(load=True)`` calls do not re-read
    # the version file from disk on every kernel launch (measured 919 resolves
    # and ~8.8 ms per Qwen4Exp decode step before this cache; see the R7
    # wrapper-host screen, 2026-09-10). The cache is keyed by the compiler AND
    # the raw override-variable values, so a changed override selects a fresh
    # resolution instead of the stale first one (test_build regression found
    # in review, 2026-09-10). Only non-None results are cached: the
    # env-unset path is the rare cold-start case and stays dynamic.
    identity = _environment_version_identity(compiler)
    cached = _ENV_VERSION_CACHE.get((compiler, identity))
    if cached is not None:
        return cached
    specific_text, generic_text, specific_file, generic_file = identity
    if specific_text or generic_text:
        value = (specific_text or generic_text).strip()
        _ENV_VERSION_CACHE[(compiler, identity)] = value
        return value
    if specific_file or generic_file:
        _ENV_VERSION_CACHE[(compiler, identity)] = (
            Path(specific_file or generic_file).expanduser().read_text().strip()
        )
        return _ENV_VERSION_CACHE[(compiler, identity)]
    return None


_COMPILER_ENV_PREFIX_CACHE: dict[str, str] = {}


def _compiler_env_prefix(compiler: str) -> str:
    cached = _COMPILER_ENV_PREFIX_CACHE.get(compiler)
    if cached is not None:
        return cached
    basename = Path(compiler).name or compiler
    safe = "".join(char if char.isalnum() else "_" for char in basename).upper()
    prefix = f"HIPENGINE_{safe}"
    _COMPILER_ENV_PREFIX_CACHE[compiler] = prefix
    return prefix


def _target_arch_from_environment() -> str | None:
    return _env_get(_ENV_HIP_ARCH) or _env_get(_ENV_HIP_OFFLOAD_ARCH)


def _cuda_target_arch_from_environment() -> str | None:
    return os.environ.get(_ENV_CUDA_ARCH) or os.environ.get(_ENV_CUDA_TARGET_ARCH)


def _rocm_device_lib_flags() -> tuple[str, ...]:
    path = _env_get(_ENV_ROCM_DEVICE_LIB_PATH) or _env_get(_ENV_HIP_DEVICE_LIB_PATH)
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


def _normalize_cuda_target_arch(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip().lower()
    if not stripped:
        return None
    if any(char.isspace() for char in stripped):
        raise ValueError(f"CUDA target architecture must not contain whitespace: {value!r}")
    if stripped.startswith("compute_"):
        stripped = "sm_" + stripped[len("compute_") :]
    if stripped.startswith("sm_"):
        suffix = stripped[len("sm_") :].replace(".", "")
    else:
        suffix = stripped.replace(".", "")
    architecture_qualified = suffix.endswith("a")
    digits = suffix[:-1] if architecture_qualified else suffix
    if not digits.isdigit():
        raise ValueError(f"invalid CUDA target architecture: {value!r}")
    return f"sm_{digits}{'a' if architecture_qualified else ''}"


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


def _profile(
    name: ProfileName,
    profiles: dict[ProfileName, BuildProfile] = PROFILES,
) -> BuildProfile:
    try:
        return profiles[name]
    except KeyError as exc:
        valid = ", ".join(sorted(profiles))
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
