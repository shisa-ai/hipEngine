"""Opt-in torch-free AOT CUTLASS/CuTe FP16 encoder self-attention route (``sm_120a``).

The kernel in ``moonshine_attention_cutlass.cu`` is a single-pass online-softmax
flash-style FP16 encoder self-attention built AOT to an architecture-qualified
``.so`` from a pinned CUTLASS/CuTe source checkout (review §8.3 item 3).  A
deployment host needs only the prebuilt ``.so`` plus the CUDA runtime libraries,
never the CUTLASS headers.

Route selection (review §8.3 item 4/5):
- the custom kernel stays the default, so the deployment path never changes and
  no Torch / ``flash_attn`` is ever added to it;
- the AOT route is armed only when ``HIPENGINE_CUTLASS_ATTENTION`` is truthy and
  a source is configured: either ``HIPENGINE_CUTLASS_ATTENTION_SO`` (prebuilt
  ``.so``, deployment) or ``HIPENGINE_CUTLASS_DIR`` (pinned CUTLASS checkout,
  development, compiled through the hashed build cache);
- ``moonshine_encoder_attention_fp16()`` is the gated dispatcher: it runs the
  AOT route when armed and otherwise calls the custom kernel unchanged (exact
  fallback).
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_attention_cutlass.cu")
_OUTPUT_NAME = "moonshine_attention_cutlass.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_HEADS = 8
_HEAD_DIM = 52
_AOT_THREADS = 512

_ENV_ROUTE = "HIPENGINE_CUTLASS_ATTENTION"       # arming flag (default off)
_ENV_CUTLASS_DIR = "HIPENGINE_CUTLASS_DIR"       # pinned CUTLASS source root (dev)
_ENV_PREBUILT_SO = "HIPENGINE_CUTLASS_ATTENTION_SO"  # prebuilt .so path (deploy)
_HEADER_TREES = ("cute", "cutlass")


@dataclass(frozen=True)
class CutlassSourceIdentity:
    """Content-addressed identity of every header tree consumed by the build."""

    root: str
    git_commit: str | None
    git_revision: str
    consumed_headers_dirty: bool
    headers_sha256: str
    header_file_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


def cutlass_include_dir() -> Path | None:
    """The pinned CUTLASS include root (``<root>/include``), or ``None``.

    Raises ``FileNotFoundError`` when ``HIPENGINE_CUTLASS_DIR`` is set but does
    not actually contain the CuTe headers, so a mistyped pin fails fast instead
    of silently falling back to the custom kernel.
    """

    root = os.environ.get(_ENV_CUTLASS_DIR)
    if not root:
        return None
    include = Path(root).expanduser() / "include"
    if not (include / "cute" / "tensor.hpp").is_file():
        raise FileNotFoundError(
            f"{_ENV_CUTLASS_DIR}={root!r} does not contain include/cute/tensor.hpp; "
            "point it at a pinned CUTLASS checkout"
        )
    return include


def prebuilt_so() -> Path | None:
    """The prebuilt AOT ``.so`` named by ``HIPENGINE_CUTLASS_ATTENTION_SO``."""

    value = os.environ.get(_ENV_PREBUILT_SO)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"prebuilt AOT attention .so missing: {path}")
    return path


def aot_attention_enabled() -> bool:
    """True when the AOT CUTLASS attention route is armed (opt-in only)."""

    if not _env_truthy(os.environ.get(_ENV_ROUTE)):
        return False
    return prebuilt_so() is not None or cutlass_include_dir() is not None


def _headers_digest(include_dir: Path) -> tuple[str, int]:
    """Hash paths and bytes of both transitive CUTLASS/CuTe include trees."""

    digest = hashlib.sha256()
    count = 0
    for tree_name in _HEADER_TREES:
        tree = include_dir / tree_name
        if not tree.is_dir():
            raise FileNotFoundError(f"CUTLASS include tree is missing: {tree}")
        for path in sorted(candidate for candidate in tree.rglob("*") if candidate.is_file()):
            relative = path.relative_to(include_dir).as_posix().encode()
            digest.update(len(relative).to_bytes(4, "little"))
            digest.update(relative)
            data = path.read_bytes()
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
            count += 1
    return digest.hexdigest(), count


def cutlass_source_identity(include_dir: Path | None = None) -> CutlassSourceIdentity:
    """Return a clean revision + content identity for all consumed headers.

    Dirty files outside ``include/{cute,cutlass}`` do not affect compilation;
    any change inside either consumed tree is rejected for source builds and is
    also captured by ``headers_sha256``. Source archives without ``.git`` are
    accepted through their content identity alone.
    """

    include = include_dir or cutlass_include_dir()
    if include is None:
        raise ValueError(f"set {_ENV_CUTLASS_DIR} to identify CUTLASS sources")
    include = include.resolve()
    root = include.parent
    headers_sha256, header_file_count = _headers_digest(include)
    commit: str | None = None
    revision = f"content-{headers_sha256[:16]}"
    dirty = False
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        revision = subprocess.check_output(
            ["git", "-C", str(root), "describe", "--tags", "--always"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--",
                    *[f"include/{name}" for name in _HEADER_TREES],
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return CutlassSourceIdentity(
        root=str(root),
        git_commit=commit,
        git_revision=revision,
        consumed_headers_dirty=dirty,
        headers_sha256=headers_sha256,
        header_file_count=header_file_count,
    )


def _cutlass_flags(include_dir: Path) -> tuple[str, ...]:
    """Fold revision and full consumed-header content into the build key."""

    identity = cutlass_source_identity(include_dir)
    if identity.consumed_headers_dirty:
        raise RuntimeError(
            "refusing to build AOT attention from dirty consumed CUTLASS headers; "
            "use a clean checkout or a content-addressed source archive"
        )
    commit = identity.git_commit or "none"
    return (
        f'-DMOONSHINE_CUTLASS_REVISION="{identity.git_revision}"',
        f'-DMOONSHINE_CUTLASS_COMMIT="{commit}"',
        f'-DMOONSHINE_CUTLASS_HEADERS_SHA256="{identity.headers_sha256}"',
    )


def plan_moonshine_attention_cutlass_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    include_dir = cutlass_include_dir()
    if include_dir is None:
        raise ValueError(
            f"pinned CUTLASS checkout is required to build the AOT attention "
            f"route; set {_ENV_CUTLASS_DIR} (or supply a prebuilt .so via "
            f"{_ENV_PREBUILT_SO} and skip building)"
        )
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_attention_cutlass",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        include_dirs=[include_dir],
        extra_flags=_cutlass_flags(include_dir),
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_attention_cutlass(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
    prebuilt_path: str | Path | None = None,
) -> ctypes.CDLL | BuildArtifact:
    """Load (or build) the AOT CUTLASS attention ``.so``.

    ``prebuilt_path`` (or the ``HIPENGINE_CUTLASS_ATTENTION_SO`` environment
    variable) is loaded directly without any CUTLASS toolchain, which is the
    deployment shape.  Without a prebuilt path the pinned CUTLASS checkout from
    ``HIPENGINE_CUTLASS_DIR`` is used to compile the kernel through the hashed
    build cache.
    """

    path = Path(prebuilt_path).expanduser() if prebuilt_path is not None else prebuilt_so()
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"prebuilt AOT attention .so missing: {path}")
        return ctypes.CDLL(str(path))
    include_dir = cutlass_include_dir()
    if include_dir is None:
        raise ValueError(
            f"AOT attention route is not configured: set {_ENV_PREBUILT_SO} "
            f"(prebuilt .so) or {_ENV_CUTLASS_DIR} (pinned CUTLASS checkout)"
        )
    return build_cuda(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_attention_cutlass",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        include_dirs=[include_dir],
        extra_flags=_cutlass_flags(include_dir),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


# ABI is identical to the custom encoder attention kernel:
# (query, key, value, mask, output, heads, head_dim, length, scale, threads, stream).
_ATTENTION_ARGS = (
    *(ctypes.c_void_p for _ in range(5)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def _launch(
    library: ctypes.CDLL,
    arguments: tuple[object, ...],
    runtime: CudaRuntime,
) -> None:
    function = signed_kernel_fn(
        library,
        "hipengine_cuda_sm120a_moonshine_encoder_attention_cutlass_fp16",
        _ATTENTION_ARGS,
        ctypes.c_int,
    )
    error = function(*arguments)
    if int(error) != CUDA_SUCCESS:
        runtime.check(int(error))


def moonshine_encoder_attention_cutlass_fp16(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    mask_ptr: int,
    output_ptr: int,
    heads: int,
    head_dim: int,
    sequence: int,
    *,
    scale: float | None = None,
    threads: int = _AOT_THREADS,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
    prebuilt_path: str | Path | None = None,
) -> None:
    """Explicit AOT CUTLASS/CuTe encoder self-attention launch (no fallback)."""

    if heads != _HEADS:
        raise ValueError(f"heads must be the Moonshine contract value {_HEADS}")
    if head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim must be the Moonshine logical dimension {_HEAD_DIM}")
    if sequence <= 0:
        raise ValueError("sequence must be positive")
    if threads != _AOT_THREADS:
        raise ValueError(f"threads must be {_AOT_THREADS} for the AOT route")
    scale_value = float(head_dim**-0.5) if scale is None else float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("scale must be positive and finite")
    library = library or build_moonshine_attention_cutlass(load=True, prebuilt_path=prebuilt_path)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        (
            query_ptr, key_ptr, value_ptr, mask_ptr, output_ptr,
            heads, head_dim, sequence, scale_value, threads, stream,
        ),
        runtime,
    )


def moonshine_encoder_attention_fp16(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    mask_ptr: int,
    output_ptr: int,
    heads: int,
    head_dim: int,
    sequence: int,
    *,
    scale: float | None = None,
    threads: int = 32,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Gated encoder self-attention dispatcher (AOT when armed, else custom).

    This is the drop-in replacement used when the runtime opts into the AOT
    route.  When ``HIPENGINE_CUTLASS_ATTENTION`` is not armed it dispatches to
    the unchanged custom kernel, so the deployment path and numerics are
    identical to before.  When armed, it routes to the AOT CUTLASS kernel.
    """

    if aot_attention_enabled():
        moonshine_encoder_attention_cutlass_fp16(
            query_ptr,
            key_ptr,
            value_ptr,
            mask_ptr,
            output_ptr,
            heads,
            head_dim,
            sequence,
            scale=scale,
            threads=_AOT_THREADS,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        return
    from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder import (
        moonshine_encoder_attention_fp16 as _custom_encoder_attention_fp16,
    )

    _custom_encoder_attention_fp16(
        query_ptr,
        key_ptr,
        value_ptr,
        mask_ptr,
        output_ptr,
        heads,
        head_dim,
        sequence,
        scale=scale,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_moonshine_attention_cutlass_kernels(*, replace: bool = True) -> None:
    registrations = (
        (
            KernelKey(_BACKEND, "moonshine_self_attention", "fp16", "aot_cutlass"),
            moonshine_encoder_attention_cutlass_fp16,
        ),
        (
            KernelKey(_BACKEND, "moonshine_self_attention", "fp16", "aot_cutlass_gated"),
            moonshine_encoder_attention_fp16,
        ),
    )
    for key, kernel in registrations:
        register(key, kernel, replace=replace)


register_moonshine_attention_cutlass_kernels()

__all__ = [
    "CutlassSourceIdentity",
    "aot_attention_enabled",
    "build_moonshine_attention_cutlass",
    "cutlass_include_dir",
    "cutlass_source_identity",
    "moonshine_encoder_attention_cutlass_fp16",
    "moonshine_encoder_attention_fp16",
    "plan_moonshine_attention_cutlass_build",
    "prebuilt_so",
    "register_moonshine_attention_cutlass_kernels",
]
