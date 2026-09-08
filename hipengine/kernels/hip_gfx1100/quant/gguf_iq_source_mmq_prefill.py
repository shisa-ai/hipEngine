"""Source-shaped IQ3_XXS/IQ4_XS integer-MMQ wrappers, selected and dense.

The kernel addresses weights as

    qweight + expert*expert_bytes + out_row*weight_row_bytes + block*block_bytes

with ``expert_bytes = out_features * weight_row_bytes``, so at ``expert == 0``
it reads exactly the dense raw GGUF layout. The dense route is therefore the
degenerate single-expert case of the selected route: no repack, no weight
sidecar, only trivial per-``rows`` metadata and a Q8_1 activation plane.
"""

from __future__ import annotations

import contextlib
import ctypes
from dataclasses import dataclass, field
import hashlib
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.hip_gfx1100.quant.gguf_k_mmq_prefill import (
    gguf_q8_1_ds4_quantize_bf16_kmajor,
    q8_1_ds4_kmajor_nbytes,
)
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_iq_source_mmq_prefill.hip")
_PARENT_SOURCE = Path(__file__).with_name("gguf_iq_gemv.hip")
_OUTPUT_NAME = "gguf_iq_source_mmq_prefill.so"
_VARIANT = "selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out"
_D4X2_VARIANT = (
    "selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out"
)
_DENSE_VARIANT = "dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out"
# int64 metadata the kernel reads: expert_start_compact[2], expert_start_mmq[2],
# tile_expert[ceil(rows/128)]. Sized for the largest admitted prefill slab.
_METADATA_MAX_TILES = 1024
_METADATA_NBYTES = (2 + 2 + _METADATA_MAX_TILES) * 8
_D4X2_SYMBOL = (
    "hipengine_gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_"
    "prefill_compact_bf16_bf16_out"
)
_SYMBOL_TEMPLATE = (
    "hipengine_{quant}_selected_mmq_i128_j128_k256_q8_1_ds4_"
    "prefill_compact_bf16_bf16_out"
)
_MMQ_ROWS = 128
_QK_K = 256
_SOURCE_FLAGS = (
    "-mcumode",
    "-funsafe-math-optimizations",
    "-ffast-math",
    "-fno-finite-math-only",
)


@dataclass(frozen=True)
class IQSourceMMQ128Metadata:
    """Expert-local 128-row padding and source-MMQ tile ownership."""

    expert_start_mmq: np.ndarray
    tile_expert: np.ndarray
    mmq_total_rows: int


def build_iq_source_mmq128_metadata(
    counts: Sequence[int],
) -> IQSourceMMQ128Metadata:
    """Build host metadata without materializing padded activation rows."""

    if not counts:
        raise ValueError("counts must be non-empty")
    normalized = np.asarray(counts, dtype=np.int64)
    if np.any(normalized < 0):
        raise ValueError("counts must be non-negative")
    padded = ((normalized + (_MMQ_ROWS - 1)) // _MMQ_ROWS) * _MMQ_ROWS
    starts = np.zeros(len(normalized) + 1, dtype=np.int64)
    starts[1:] = np.cumsum(padded, dtype=np.int64)
    tiles = np.repeat(
        np.arange(len(normalized), dtype=np.int64),
        padded // _MMQ_ROWS,
    )
    return IQSourceMMQ128Metadata(
        expert_start_mmq=starts,
        tile_expert=np.ascontiguousarray(tiles, dtype=np.int64),
        mmq_total_rows=int(starts[-1]),
    )


def _extra_flags() -> tuple[str, ...]:
    parent_tag = int(hashlib.sha256(_PARENT_SOURCE.read_bytes()).hexdigest()[:8], 16)
    return (*_SOURCE_FLAGS, f"-DHIPENGINE_IQ_GEMV_SOURCE_TAG={parent_tag}")


def plan_gguf_iq_source_mmq_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_iq_source_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_iq_source_mmq_prefill(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="gguf_iq_source_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _launch_iq_source_mmq(
    quant: str,
    xq_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq_ptr: int,
    tile_expert_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    mmq_total_rows: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if compact_rows <= 0:
        raise ValueError("compact_rows must be positive")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be positive and divisible by 256")
    if out_features <= 0 or out_features % _MMQ_ROWS != 0:
        raise ValueError("out_features must be a positive multiple of 128")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if mmq_total_rows <= 0 or mmq_total_rows % _MMQ_ROWS != 0:
        raise ValueError("mmq_total_rows must be positive and a multiple of 128")
    library = library or build_gguf_iq_source_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TEMPLATE.format(quant=quant))
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(mmq_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out(
    *args, **kwargs
) -> None:
    _launch_iq_source_mmq("gguf_iq3_xxs", *args, **kwargs)


def gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out(
    *args, **kwargs
) -> None:
    _launch_iq_source_mmq("gguf_iq4_xs", *args, **kwargs)


def gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out(
    xq_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq_ptr: int,
    tile_expert_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    mmq_total_rows: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch IQ3 MMQ with primary and residual D4 activation planes."""

    if compact_rows <= 0:
        raise ValueError("compact_rows must be positive")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be positive and divisible by 256")
    if out_features <= 0 or out_features % _MMQ_ROWS != 0:
        raise ValueError("out_features must be a positive multiple of 128")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if mmq_total_rows <= 0 or mmq_total_rows % _MMQ_ROWS != 0:
        raise ValueError("mmq_total_rows must be positive and a multiple of 128")
    library = library or build_gguf_iq_source_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _D4X2_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(mmq_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


# --- dense route -----------------------------------------------------------
#
# The caller owns one bounded workspace holding the Q8_1 activation plane and
# the small per-``rows`` metadata, exactly as the dense planar-Q6 integer MMQ
# route does. Binding it is what admits the route; without a workspace the
# dispatcher keeps the strict GEMV owner.


@dataclass
class IQDenseMMQWorkspace:
    """Caller-owned bounded workspace for the dense IQ integer-MMQ route."""

    ptr: int
    nbytes: int
    library: object = None
    producer_library: object = None
    # Metadata depends only on ``rows``; upload once per distinct value.
    _uploaded_rows: int = field(default=0, repr=False)

    def metadata_ptr(self) -> int:
        return int(self.ptr) + int(self.nbytes) - _METADATA_NBYTES


_iq_dense_mmq_workspace: ContextVar[IQDenseMMQWorkspace | None] = ContextVar(
    "iq_dense_mmq_workspace", default=None
)


def iq_dense_mmq_activation_nbytes(rows: int, hidden: int) -> int:
    """Return the Q8_1 activation-plane bytes one dense launch consumes."""

    return q8_1_ds4_kmajor_nbytes(int(rows), int(hidden))


def iq_dense_mmq_nbytes(rows: int, hidden: int) -> int:
    """Return total workspace bytes: activation plane plus metadata tail."""

    if (int(rows) + _MMQ_ROWS - 1) // _MMQ_ROWS > _METADATA_MAX_TILES:
        raise ValueError(
            f"dense IQ MMQ supports at most {_METADATA_MAX_TILES * _MMQ_ROWS} rows"
        )
    return iq_dense_mmq_activation_nbytes(rows, hidden) + _METADATA_NBYTES


def iq_dense_mmq_workspace() -> IQDenseMMQWorkspace | None:
    """Return the active caller-owned dense IQ MMQ workspace."""

    return _iq_dense_mmq_workspace.get()


@contextlib.contextmanager
def iq_dense_mmq_session(
    enabled: bool = True,
    *,
    workspace_ptr: int = 0,
    workspace_nbytes: int = 0,
    library: object = None,
    producer_library: object = None,
) -> Iterator[None]:
    """Bind a bounded workspace for the dense IQ integer-MMQ prefill route."""

    workspace = None
    if enabled:
        if int(workspace_ptr) <= 0 or int(workspace_nbytes) <= _METADATA_NBYTES:
            raise ValueError(
                "dense IQ MMQ requires a positive workspace larger than its metadata tail"
            )
        workspace = IQDenseMMQWorkspace(
            ptr=int(workspace_ptr),
            nbytes=int(workspace_nbytes),
            library=library,
            producer_library=producer_library,
        )
    token = _iq_dense_mmq_workspace.set(workspace)
    try:
        yield
    finally:
        _iq_dense_mmq_workspace.reset(token)


def _upload_dense_metadata(
    workspace: IQDenseMMQWorkspace,
    rows: int,
    runtime: HipRuntime | None,
):
    """Write the single-expert metadata for ``rows`` into the workspace tail.

    Returns ``(compact_ptr, mmq_start_ptr, tile_ptr, mmq_total_rows)``. The
    upload is skipped when the workspace already holds this row count, so a
    whole prefill pass pays it once rather than once per projection.
    """

    from hipengine.core.memory import host_array_ptr
    from hipengine.core.runtime import MemcpyKind

    metadata = build_iq_source_mmq128_metadata([int(rows)])
    base = workspace.metadata_ptr()
    compact_ptr = base
    mmq_start_ptr = base + 16
    tile_ptr = base + 32
    if workspace._uploaded_rows != int(rows):
        payload = np.ascontiguousarray(
            np.concatenate(
                (
                    np.array([0, int(rows)], dtype=np.int64),
                    metadata.expert_start_mmq.astype(np.int64),
                    metadata.tile_expert.astype(np.int64),
                )
            )
        )
        nbytes = int(payload.nbytes)
        if nbytes > _METADATA_NBYTES:
            raise ValueError("dense IQ MMQ metadata exceeds its workspace tail")
        (runtime or get_hip_runtime()).memcpy(
            base, host_array_ptr(payload), nbytes, MemcpyKind.HOST_TO_DEVICE
        )
        workspace._uploaded_rows = int(rows)
    return compact_ptr, mmq_start_ptr, tile_ptr, int(metadata.mmq_total_rows)


def _launch_iq_dense_mmq(
    quant: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    **_ignored,
) -> None:
    """Dense IQ integer-MMQ prefill through the caller-owned workspace.

    Signature matches the raw GGUF linear launch ABI so the dispatcher can swap
    this in by variant name alone. ``library`` is the caller's raw-IQ library
    and is deliberately ignored: this route's libraries come from the
    workspace, which is what binds the route.
    """

    workspace = iq_dense_mmq_workspace()
    if workspace is None:
        raise RuntimeError("dense IQ integer MMQ requires an active workspace session")
    required = iq_dense_mmq_nbytes(rows, in_features)
    if required > int(workspace.nbytes):
        raise ValueError(
            f"dense IQ MMQ workspace holds {workspace.nbytes} bytes, needs {required}"
        )
    compact_ptr, mmq_start_ptr, tile_ptr, mmq_total_rows = _upload_dense_metadata(
        workspace, rows, runtime
    )
    gguf_q8_1_ds4_quantize_bf16_kmajor(
        int(x_ptr),
        int(workspace.ptr),
        int(rows),
        int(in_features),
        stream=stream,
        library=workspace.producer_library,
        runtime=runtime,
    )
    _launch_iq_source_mmq(
        quant,
        int(workspace.ptr),
        compact_ptr,
        mmq_start_ptr,
        tile_ptr,
        int(qweight_ptr),
        int(out_ptr),
        compact_rows=int(rows),
        in_features=int(in_features),
        out_features=int(out_features),
        num_experts=1,
        mmq_total_rows=mmq_total_rows,
        stream=stream,
        library=workspace.library,
        runtime=runtime,
    )


def gguf_iq4_xs_dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out(
    *args, **kwargs
) -> None:
    _launch_iq_dense_mmq("gguf_iq4_xs", *args, **kwargs)


def gguf_iq3_xxs_dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out(
    *args, **kwargs
) -> None:
    _launch_iq_dense_mmq("gguf_iq3_xxs", *args, **kwargs)


def register_gguf_iq_source_mmq_prefill_kernels(*, replace: bool = True) -> None:
    for quant, function in (
        (
            "gguf_iq3_xxs",
            gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant, _VARIANT),
            function,
            replace=replace,
        )
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_iq3_xxs", _D4X2_VARIANT),
        gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out,
        replace=replace,
    )
    for quant, function in (
        ("gguf_iq4_xs", gguf_iq4_xs_dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out),
        ("gguf_iq3_xxs", gguf_iq3_xxs_dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out),
    ):
        register(
            KernelKey("hip_gfx1100", "linear", quant, _DENSE_VARIANT),
            function,
            replace=replace,
        )


register_gguf_iq_source_mmq_prefill_kernels()


__all__ = [
    "IQDenseMMQWorkspace",
    "IQSourceMMQ128Metadata",
    "build_gguf_iq_source_mmq_prefill",
    "build_iq_source_mmq128_metadata",
    "gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out",
    "gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out",
    "gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out",
    "gguf_iq3_xxs_dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out",
    "gguf_iq4_xs_dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out",
    "iq_dense_mmq_activation_nbytes",
    "iq_dense_mmq_nbytes",
    "iq_dense_mmq_session",
    "iq_dense_mmq_workspace",
    "plan_gguf_iq_source_mmq_prefill_build",
    "register_gguf_iq_source_mmq_prefill_kernels",
]
