"""GGUF K-quant x prequantized-Q8_1 MMQ wrappers.

Most entry points are selected-MoE diagnostics. The dense planar-Q6 composite
is separately session-owned and registry-selected on qualified backends.
"""

from __future__ import annotations

import contextlib
import ctypes
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_q8_1_selected_prefill.hip")
_OUTPUT_NAME = "gguf_q4_k_q8_1_selected_prefill.so"
_SYMBOL_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_MMQ32_BF16 = (
    "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_X8_DS4_MMQ32_BF16 = (
    "hipengine_gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_T16_DS4_MMQ32_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_T16_DS4X3_MMQ32_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_T16_DS4X3_GUARDED_MMQ32_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4x3_guarded_mmq32_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_T16_SPARSE_EXACT_CORRECT_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_sparse_exact_correct_bf16"
)
_SYMBOL_DS4_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_WMMA_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_WMMA32_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_WMMA64_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_PREVIEW_WMMA32_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_WMMA32_LDSPACK_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_WMMA32_LDS_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out"
_SYMBOL_WMMA_I8_PROBE = "hipengine_gguf_q4_k_q8_1_wmma_i8_probe_16x16"
_SYMBOL_DS4_PACK_BF16 = "hipengine_gguf_q8_1_mmq_ds4_pack_bf16"
_SYMBOL_DS4X3_PACK_BF16 = "hipengine_gguf_q8_1_mmq_ds4_pack_bf16_d4x3"
_SYMBOL_DS4_F32_PACK_BF16 = {
    1: "hipengine_gguf_q8_1_mmq_ds4_f32_pack_bf16_d4",
    2: "hipengine_gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x2",
    3: "hipengine_gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3",
}
_SYMBOL_DS4_F32_PACK_BF16_D4_Q6_HALF_SUMS = (
    "hipengine_gguf_q8_1_mmq_ds4_f32_pack_bf16_d4_q6_half_sums"
)
_SYMBOL_DS4_F32_PACK_DUAL_SILU_BF16 = {
    1: "hipengine_gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4",
    2: "hipengine_gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4x2",
    3: "hipengine_gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4x3",
}
_SYMBOL_DS4_F32_PACK_DUAL_SILU_BF16_D4_Q6_HALF_SUMS = (
    "hipengine_gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4_q6_half_sums"
)
_SYMBOL_DS8_F32_PACK_BF16 = (
    "hipengine_gguf_q8_1_mmq_ds8_f32_pack_bf16"
)
_SYMBOL_DS8_F32_PACK_BF16_Q4_HALF_SUMS = (
    "hipengine_gguf_q8_1_mmq_ds8_f32_pack_bf16_q4_half_sums"
)
_SYMBOL_Q6_T16_DS4_F32_MMQ64X32_BF16 = {
    passes: (
        "hipengine_gguf_q6_k_t16_selected_q8_1_"
        f"ds4{'x' + str(passes) if passes > 1 else ''}_f32_"
        "mmq64x32_prefill_compact32_bf16_bf16_out"
    )
    for passes in (1, 2, 3)
}
_SYMBOL_Q6_T16_DS4_F32_MMQ64X32_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_selected_q8_1_ds4_f32_"
    "mmq64x32_rowvec_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_DS4_F32_MMQ64X32_BF16 = {
    passes: (
        "hipengine_gguf_q6_k_t16_qmicro_selected_q8_1_"
        f"ds4{'x' + str(passes) if passes > 1 else ''}_f32_"
        "mmq64x32_prefill_compact32_bf16_bf16_out"
    )
    for passes in (1, 2, 3)
}
_SYMBOL_Q6_T16_QMICRO_DS4_F32_MMQ64X32_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_selected_q8_1_ds4_f32_"
    "mmq64x32_rowvec_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q6_T16_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_selected_q8_1_ds4_f32_"
    "mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_selected_q8_1_ds4_f32_"
    "mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_COMPACT_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_compact_activation_selected_q8_1_ds4_f32_"
    "mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_HALF_ROW_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_half_row_activation_selected_q8_1_ds4_f32_"
    "mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_skip_padded_activation_selected_q8_1_ds4_f32_"
    "mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_PERMUTE_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_permute_skip_padded_activation_selected_q8_1_"
    "ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_PLANAR_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_skip_padded_activation_selected_q8_1_"
    "ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_PLANAR_INTEGER_WMMA_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_integer_wmma_skip_padded_activation_"
    "selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_PLANAR_INTEGER_WMMA_HOIST_ACTIVATION_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_integer_wmma_hoist_activation_"
    "skip_padded_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_"
    "prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_PLANAR_INTEGER_WMMA_HOIST_ACTIVATION_PREFETCH_WEIGHT_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_integer_wmma_hoist_activation_"
    "prefetch_weight_skip_padded_activation_selected_q8_1_ds4_f32_"
    "mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_PLANAR_INTEGER_WMMA_HOIST_ACTIVATION_PREFETCH_WEIGHT_ACTIVATION_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_integer_wmma_hoist_activation_"
    "prefetch_weight_activation_skip_padded_activation_selected_q8_1_ds4_f32_"
    "mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_PLANAR_INTEGER_WMMA_HOIST_ACTIVATION_PREFETCH_WEIGHT_ACTIVATION_PRECOMPUTED_SUMS_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_integer_wmma_hoist_activation_"
    "prefetch_weight_activation_precomputed_sums_skip_padded_activation_"
    "selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out"
)
_SYMBOL_Q6_T16_QMICRO_PLANAR_DENSE_INTEGER_MMQ64X64_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_integer_mmq64x64_dense_"
    "bf16_bf16_out"
)
_SYMBOL_Q4_T16_DS4_F32_MMQ64X32_BF16 = {
    passes: (
        "hipengine_gguf_q4_k_t16_selected_dual_q8_1_"
        f"ds4{'x' + str(passes) if passes > 1 else ''}_f32_"
        "mmq64x32_prefill_compact32_bf16_bf16_out"
    )
    for passes in (1, 2, 3)
}
_SYMBOL_Q4_T16_SINGLE_DS4_F32_MMQ64X32_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_q8_1_ds4_f32_"
    "mmq64x32_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_SINGLE_DS4_F32_MMQ64X32_ROWVEC_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_q8_1_ds4_f32_"
    "mmq64x32_rowvec_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_SINGLE_DS4_F32_MMQ64X32_WAVECOLS_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_q8_1_ds4_f32_"
    "mmq64x32_wavecols_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_SINGLE_DS4_F32_MMQ64X32_WAVECOLS_DIRECT_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_q8_1_ds4_f32_"
    "mmq64x32_wavecols_direct_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_SINGLE_RAW_PREFETCH_DS4_F32_MMQ64X32_WAVECOLS_DIRECT_BF16 = {
    packs: (
        f"hipengine_gguf_q4_k_t16_raw_prefetch_p{packs}_selected_q8_1_"
        "ds4_f32_mmq64x32_wavecols_direct_"
        "prefill_compact32_bf16_bf16_out"
    )
    for packs in (8,)
}
_SYMBOL_Q4_T16_DS4_F32_MMQ64X32_ROWVEC_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4_f32_"
    "mmq64x32_rowvec_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_DS8_F32_MMQ128X32_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds8_f32_"
    "mmq128x32_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_DS8_F32_MMQ128X32_ROWVEC_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds8_f32_"
    "mmq128x32_rowvec_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_DS8_F32_MMQ128X32_WAVECOLS_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds8_f32_"
    "mmq128x32_wavecols_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_DS8_F32_MMQ128X32_WAVECOLS_DIRECT_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds8_f32_"
    "mmq128x32_wavecols_direct_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_DS8_F32_MMQ128X32_WAVECOLS_DIRECT_DOUBLEBUF_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds8_f32_"
    "mmq128x32_wavecols_direct_doublebuf_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_RAW_PREFETCH_DS8_F32_MMQ128X32_WAVECOLS_DIRECT_DOUBLEBUF_BF16 = {
    packs: (
        f"hipengine_gguf_q4_k_t16_raw_prefetch_p{packs}_selected_dual_q8_1_"
        "ds8_f32_mmq128x32_wavecols_direct_doublebuf_"
        "prefill_compact32_bf16_bf16_out"
    )
    for packs in (8,)
}
_SYMBOL_Q4_T16_RAW_PREFETCH_PRECOMPUTED_SUMS_DS8_F32_MMQ128X32_WAVECOLS_DIRECT_DOUBLEBUF_BF16 = (
    "hipengine_gguf_q4_k_t16_raw_prefetch_p8_precomputed_sums_"
    "selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_direct_"
    "doublebuf_prefill_compact32_bf16_bf16_out"
)
_SYMBOL_Q4_T16_DUAL_INTERLEAVED_RAW_PREFETCH_DS8_F32_MMQ128X32_WAVECOLS_DIRECT_DOUBLEBUF_BF16 = (
    "hipengine_gguf_q4_k_t16_dual_interleaved_raw_prefetch_p8_"
    "selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_direct_"
    "doublebuf_prefill_compact32_bf16_bf16_out"
)
_Q4_K_BLOCK = 256
_Q8_1_MMQ_BLOCK = 128
_Q8_1_DS4_F32_BLOCK_BYTES = 160


@dataclass(frozen=True)
class Q6DenseIntegerMMQWorkspace:
    """Caller-owned Q8 workspace and library for one dense MMQ session."""

    ptr: int
    nbytes: int
    library: object

    def __post_init__(self) -> None:
        if int(self.ptr) <= 0 or int(self.nbytes) <= 0:
            raise ValueError("dense Q6 integer MMQ workspace must be non-empty")
        if self.library is None:
            raise ValueError("dense Q6 integer MMQ library is required")


_q6_dense_integer_mmq_workspace: ContextVar[
    Q6DenseIntegerMMQWorkspace | None
] = ContextVar("q6_dense_integer_mmq_workspace", default=None)


def q6_dense_integer_mmq_nbytes(rows: int, hidden: int) -> int:
    """Return the one-plane D4 Q8_1 bytes required by a dense launch."""

    _check_positive(rows, "rows")
    _check_positive(hidden, "hidden")
    if hidden % _Q8_1_MMQ_BLOCK:
        raise ValueError("hidden must be divisible by 128")
    return (
        int(rows)
        * (int(hidden) // _Q8_1_MMQ_BLOCK)
        * _Q8_1_DS4_F32_BLOCK_BYTES
    )


def q6_dense_integer_mmq_workspace() -> Q6DenseIntegerMMQWorkspace | None:
    """Return the active caller-owned dense MMQ workspace."""

    return _q6_dense_integer_mmq_workspace.get()


@contextlib.contextmanager
def q6_dense_integer_mmq_session(
    enabled: bool = True,
    *,
    workspace_ptr: int = 0,
    workspace_nbytes: int = 0,
    library: object = None,
) -> Iterator[None]:
    """Bind a bounded workspace for dense planar-Q6 integer MMQ."""

    workspace = None
    if enabled:
        workspace = Q6DenseIntegerMMQWorkspace(
            ptr=int(workspace_ptr),
            nbytes=int(workspace_nbytes),
            library=library,
        )
    token = _q6_dense_integer_mmq_workspace.set(workspace)
    try:
        yield
    finally:
        _q6_dense_integer_mmq_workspace.reset(token)


def plan_gguf_q4_k_q8_1_selected_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_q8_1_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_q8_1_selected_prefill(
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
        family="gguf_q4_k_q8_1_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q4_k_q8_1_wmma_i8_probe_16x16(
    a_rows_ptr: int,
    b_cols_ptr: int,
    out_ptr: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch a diagnostic RDNA3 int8/uint8 WMMA 16x16 probe.

    ``a_rows`` is row-major ``int8[16, 16]`` and ``b_cols`` is row-major
    ``uint8[16, 16]`` where each row represents one logical output column over
    K. The kernel writes ``int32[16, 16]`` equal to ``a_rows @ b_cols.T``.
    """

    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WMMA_I8_PROBE)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(a_rows_ptr),
        ctypes.c_void_p(b_cols_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q8_1_mmq_ds4_pack_bf16(
    x_bf16_ptr: int,
    out_q8_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pack BF16 activations to llama.cpp-style DS4 block_q8_1_mmq on GPU."""

    _check_positive(rows, "rows")
    _check_positive(hidden, "hidden")
    if hidden % _Q8_1_MMQ_BLOCK != 0:
        raise ValueError("hidden must be divisible by DS4 Q8_1 MMQ block size 128")
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_PACK_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_bf16_ptr),
        ctypes.c_void_p(out_q8_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q8_1_mmq_ds4_pack_bf16_d4x3(
    x_bf16_ptr: int,
    out_q8_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pack BF16 activations as primary DS4 plus two residual DS4 planes."""

    _check_positive(rows, "rows")
    _check_positive(hidden, "hidden")
    if hidden % _Q8_1_MMQ_BLOCK != 0:
        raise ValueError("hidden must be divisible by DS4 Q8_1 MMQ block size 128")
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4X3_PACK_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_bf16_ptr),
        ctypes.c_void_p(out_q8_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
    x_bf16_ptr: int,
    out_q8_ptr: int,
    rows: int,
    hidden: int,
    *,
    residual_passes: int = 3,
    split16: bool = False,
    q6_half_sums: bool = False,
    q4_half_sums_ptr: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pack BF16 rows as three residual DS4 planes with FP32 metadata."""

    _check_positive(rows, "rows")
    _check_positive(hidden, "hidden")
    if hidden % _Q8_1_MMQ_BLOCK != 0:
        raise ValueError("hidden must be divisible by DS4 Q8_1 MMQ block size 128")
    if residual_passes not in _SYMBOL_DS4_F32_PACK_BF16:
        raise ValueError("residual_passes must be 1, 2, or 3")
    if split16 and residual_passes != 1:
        raise ValueError("split16 requires residual_passes=1")
    if q6_half_sums and (split16 or residual_passes != 1):
        raise ValueError("q6_half_sums requires one-plane D4")
    if q4_half_sums_ptr and (not split16 or residual_passes != 1):
        raise ValueError("q4_half_sums_ptr requires one-plane split16 D8")
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    symbol = (
        _SYMBOL_DS8_F32_PACK_BF16_Q4_HALF_SUMS
        if q4_half_sums_ptr
        else _SYMBOL_DS4_F32_PACK_BF16_D4_Q6_HALF_SUMS
        if q6_half_sums
        else
        _SYMBOL_DS8_F32_PACK_BF16
        if split16
        else _SYMBOL_DS4_F32_PACK_BF16[residual_passes]
    )
    fn = getattr(library, symbol)
    fn.argtypes = (
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        if q4_half_sums_ptr
        else [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
    )
    fn.restype = ctypes.c_int
    args = [
        ctypes.c_void_p(x_bf16_ptr),
        ctypes.c_void_p(out_q8_ptr),
    ]
    if q4_half_sums_ptr:
        args.append(ctypes.c_void_p(q4_half_sums_ptr))
    args.extend(
        [
            ctypes.c_int64(rows),
            ctypes.c_int64(hidden),
            ctypes.c_void_p(stream),
        ]
    )
    err = fn(*args)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4x3(
    gate_up_bf16_ptr: int,
    out_q8_ptr: int,
    rows: int,
    hidden: int,
    *,
    residual_passes: int = 3,
    q6_half_sums: bool = False,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Preserve BF16 SiLU output semantics while directly packing dual rows."""

    _check_positive(rows, "rows")
    _check_positive(hidden, "hidden")
    if hidden % _Q8_1_MMQ_BLOCK != 0:
        raise ValueError("hidden must be divisible by DS4 Q8_1 MMQ block size 128")
    if residual_passes not in _SYMBOL_DS4_F32_PACK_DUAL_SILU_BF16:
        raise ValueError("residual_passes must be 1, 2, or 3")
    if q6_half_sums and residual_passes != 1:
        raise ValueError("q6_half_sums requires one-plane D4")
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        (
            _SYMBOL_DS4_F32_PACK_DUAL_SILU_BF16_D4_Q6_HALF_SUMS
            if q6_half_sums
            else _SYMBOL_DS4_F32_PACK_DUAL_SILU_BF16[residual_passes]
        ),
    )
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(gate_up_bf16_ptr),
        ctypes.c_void_p(out_q8_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq32_ptr: int,
    mmq_tile_expert_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    mmq_total_rows: int,
    *,
    residual_passes: int = 3,
    rowvec: bool = False,
    tile_rows: int = 32,
    qmicro: bool = False,
    compact_activation: bool = False,
    half_row_activation: bool = False,
    skip_padded_activation: bool = False,
    qmicro_permute: bool = False,
    qmicro_planar: bool = False,
    integer_wmma: bool | None = None,
    wmma_hoist_activation: bool | None = None,
    wmma_prefetch_weight: bool = False,
    wmma_prefetch_activation: bool = False,
    precomputed_activation_sums: bool = False,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch range-safe Q6T16 packed-dot selected down."""

    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    _check_positive(num_experts, "num_experts")
    _check_positive(mmq_total_rows, "mmq_total_rows")
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF K block size 256")
    if out_features % 64 != 0:
        raise ValueError("out_features must be a multiple of 64")
    if tile_rows not in (32, 64):
        raise ValueError("tile_rows must be 32 or 64")
    if mmq_total_rows % tile_rows != 0:
        raise ValueError(f"mmq_total_rows must be a multiple of {tile_rows}")
    if residual_passes not in _SYMBOL_Q6_T16_DS4_F32_MMQ64X32_BF16:
        raise ValueError("residual_passes must be 1, 2, or 3")
    if rowvec and residual_passes != 1:
        raise ValueError("rowvec requires residual_passes=1")
    if tile_rows == 64 and not rowvec:
        raise ValueError("tile_rows=64 requires rowvec=True")
    if compact_activation and not (qmicro and rowvec and tile_rows == 64):
        raise ValueError(
            "compact_activation requires qmicro=True, rowvec=True, tile_rows=64"
        )
    if half_row_activation and not compact_activation:
        raise ValueError(
            "half_row_activation requires compact_activation=True"
        )
    if skip_padded_activation and not half_row_activation:
        raise ValueError(
            "skip_padded_activation requires half_row_activation=True"
        )
    if qmicro_permute and not (
        qmicro
        and compact_activation
        and half_row_activation
        and skip_padded_activation
        and rowvec
        and tile_rows == 64
    ):
        raise ValueError(
            "qmicro_permute requires the production Q6 qmicro row64 path"
        )
    if qmicro_planar and not (
        qmicro
        and compact_activation
        and half_row_activation
        and skip_padded_activation
        and rowvec
        and tile_rows == 64
    ):
        raise ValueError(
            "qmicro_planar requires the production Q6 qmicro row64 path"
        )
    if qmicro_planar and qmicro_permute:
        raise ValueError(
            "qmicro_planar and qmicro_permute are mutually exclusive"
        )
    if integer_wmma is None:
        integer_wmma = qmicro_planar
    if integer_wmma and not qmicro_planar:
        raise ValueError("integer_wmma requires qmicro_planar=True")
    if wmma_hoist_activation is None:
        wmma_hoist_activation = integer_wmma
    if wmma_hoist_activation and not integer_wmma:
        raise ValueError(
            "wmma_hoist_activation requires integer_wmma=True"
        )
    if wmma_prefetch_weight and not wmma_hoist_activation:
        raise ValueError(
            "wmma_prefetch_weight requires wmma_hoist_activation=True"
        )
    if wmma_prefetch_activation and not wmma_prefetch_weight:
        raise ValueError(
            "wmma_prefetch_activation requires wmma_prefetch_weight=True"
        )
    if precomputed_activation_sums and not wmma_prefetch_activation:
        raise ValueError(
            "precomputed_activation_sums requires production WMMA prefetch"
        )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        (
            _SYMBOL_Q6_T16_QMICRO_PLANAR_INTEGER_WMMA_HOIST_ACTIVATION_PREFETCH_WEIGHT_ACTIVATION_PRECOMPUTED_SUMS_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16
            if precomputed_activation_sums
            else
            _SYMBOL_Q6_T16_QMICRO_PLANAR_INTEGER_WMMA_HOIST_ACTIVATION_PREFETCH_WEIGHT_ACTIVATION_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16
            if wmma_prefetch_activation
            else _SYMBOL_Q6_T16_QMICRO_PLANAR_INTEGER_WMMA_HOIST_ACTIVATION_PREFETCH_WEIGHT_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16
            if wmma_prefetch_weight
            else _SYMBOL_Q6_T16_QMICRO_PLANAR_INTEGER_WMMA_HOIST_ACTIVATION_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16
            if wmma_hoist_activation
            else _SYMBOL_Q6_T16_QMICRO_PLANAR_INTEGER_WMMA_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16
            if integer_wmma
            else _SYMBOL_Q6_T16_QMICRO_PLANAR_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16
            if qmicro_planar
            else _SYMBOL_Q6_T16_QMICRO_PERMUTE_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16
            if qmicro_permute
            else _SYMBOL_Q6_T16_QMICRO_SKIP_PADDED_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16
            if skip_padded_activation
            else _SYMBOL_Q6_T16_QMICRO_HALF_ROW_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16
            if half_row_activation
            else _SYMBOL_Q6_T16_QMICRO_COMPACT_ACTIVATION_DS4_F32_MMQ64X64_ROWVEC_BF16
            if compact_activation
            else _SYMBOL_Q6_T16_QMICRO_DS4_F32_MMQ64X64_ROWVEC_BF16
            if qmicro and tile_rows == 64
            else _SYMBOL_Q6_T16_QMICRO_DS4_F32_MMQ64X32_ROWVEC_BF16
            if qmicro and rowvec
            else _SYMBOL_Q6_T16_QMICRO_DS4_F32_MMQ64X32_BF16[residual_passes]
            if qmicro
            else _SYMBOL_Q6_T16_DS4_F32_MMQ64X64_ROWVEC_BF16
            if tile_rows == 64
            else _SYMBOL_Q6_T16_DS4_F32_MMQ64X32_ROWVEC_BF16
            if rowvec
            else _SYMBOL_Q6_T16_DS4_F32_MMQ64X32_BF16[residual_passes]
        ),
    )
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
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq32_ptr),
        ctypes.c_void_p(mmq_tile_expert_ptr),
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


def _launch_q6_k_t16_qmicro_planar_dense_q8_1_mmq64x64(
    x_q8_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the one-expert dense specialization over packed Q8_1 rows."""

    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    if rows > 64:
        raise ValueError("dense Q6 integer MMQ supports at most 64 rows")
    if in_features % _Q4_K_BLOCK:
        raise ValueError("in_features must be divisible by 256")
    if out_features % 64:
        raise ValueError("out_features must be divisible by 64")
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_Q6_T16_QMICRO_PLANAR_DENSE_INTEGER_MMQ64X64_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_t16_qmicro_planar_dense_q8_1_mmq64x64_bf16_bf16_out(
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
) -> None:
    """Pack BF16 rows and launch the session-owned dense planar-Q6 MMQ."""

    workspace = q6_dense_integer_mmq_workspace()
    required_nbytes = q6_dense_integer_mmq_nbytes(rows, in_features)
    if workspace is None:
        raise RuntimeError("dense Q6 integer MMQ escaped its owner session")
    if int(workspace.nbytes) < required_nbytes:
        raise ValueError(
            "dense Q6 integer MMQ workspace is too small: "
            f"required={required_nbytes}, available={workspace.nbytes}"
        )
    runtime = runtime or get_hip_runtime()
    candidate_library = workspace.library
    gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
        x_ptr,
        workspace.ptr,
        rows,
        in_features,
        residual_passes=1,
        q6_half_sums=True,
        stream=stream,
        library=candidate_library,
        runtime=runtime,
    )
    _launch_q6_k_t16_qmicro_planar_dense_q8_1_mmq64x64(
        workspace.ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=candidate_library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    compact_to_source_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq32_ptr: int,
    mmq_tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    source_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    mmq_total_rows: int,
    *,
    residual_passes: int = 3,
    split16: bool = False,
    rowvec: bool = False,
    wave_cols: bool = False,
    direct_wave_decode: bool = False,
    double_buffer_activation: bool = False,
    raw_weight_prefetch_packs: int = 0,
    precomputed_activation_sums_ptr: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP32-metadata Q4T16 64x32 packed-dot selected gate/up."""

    _check_positive(source_rows, "source_rows")
    _check_mmq32_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows,
    )
    if out_features_a % 64 != 0 or out_features_b % 64 != 0:
        raise ValueError("out_features must be multiples of 64")
    if residual_passes not in _SYMBOL_Q4_T16_DS4_F32_MMQ64X32_BF16:
        raise ValueError("residual_passes must be 1, 2, or 3")
    if split16 and residual_passes != 1:
        raise ValueError("split16 requires residual_passes=1")
    if rowvec and residual_passes != 1:
        raise ValueError("rowvec requires residual_passes=1")
    if wave_cols and not rowvec:
        raise ValueError("wave_cols requires rowvec")
    if direct_wave_decode and not wave_cols:
        raise ValueError("direct_wave_decode requires wave_cols")
    if double_buffer_activation and not direct_wave_decode:
        raise ValueError(
            "double_buffer_activation requires direct_wave_decode"
        )
    if raw_weight_prefetch_packs not in (0, 8):
        raise ValueError("raw_weight_prefetch_packs must be 0 or 8")
    if raw_weight_prefetch_packs and not (
        residual_passes == 1
        and split16
        and rowvec
        and wave_cols
        and direct_wave_decode
        and double_buffer_activation
    ):
        raise ValueError(
            "raw weight prefetch requires production Q4 geometry"
        )
    if precomputed_activation_sums_ptr and not (
        raw_weight_prefetch_packs == 8
        and residual_passes == 1
        and split16
        and rowvec
        and wave_cols
        and direct_wave_decode
        and double_buffer_activation
    ):
        raise ValueError(
            "precomputed activation sums require production Q4 P8 geometry"
        )
    if wave_cols and not split16:
        raise ValueError("wave_cols requires split16 D8")
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    symbol = (
        _SYMBOL_Q4_T16_RAW_PREFETCH_PRECOMPUTED_SUMS_DS8_F32_MMQ128X32_WAVECOLS_DIRECT_DOUBLEBUF_BF16
        if precomputed_activation_sums_ptr
        else _SYMBOL_Q4_T16_RAW_PREFETCH_DS8_F32_MMQ128X32_WAVECOLS_DIRECT_DOUBLEBUF_BF16[
            raw_weight_prefetch_packs
        ]
        if raw_weight_prefetch_packs
        else _SYMBOL_Q4_T16_DS8_F32_MMQ128X32_WAVECOLS_DIRECT_DOUBLEBUF_BF16
        if double_buffer_activation
        else _SYMBOL_Q4_T16_DS8_F32_MMQ128X32_WAVECOLS_DIRECT_BF16
        if direct_wave_decode
        else _SYMBOL_Q4_T16_DS8_F32_MMQ128X32_WAVECOLS_BF16
        if wave_cols
        else (
            _SYMBOL_Q4_T16_DS8_F32_MMQ128X32_ROWVEC_BF16
            if split16
            else _SYMBOL_Q4_T16_DS4_F32_MMQ64X32_ROWVEC_BF16
        )
        if rowvec
        else _SYMBOL_Q4_T16_DS8_F32_MMQ128X32_BF16
        if split16
        else _SYMBOL_Q4_T16_DS4_F32_MMQ64X32_BF16[residual_passes]
    )
    fn = getattr(library, symbol)
    pointer_args = 9 if precomputed_activation_sums_ptr else 8
    fn.argtypes = (
        [ctypes.c_void_p] * pointer_args
        + [ctypes.c_int64] * 7
        + [ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    args = [ctypes.c_void_p(x_q8_ptr)]
    if precomputed_activation_sums_ptr:
        args.append(ctypes.c_void_p(precomputed_activation_sums_ptr))
    args.extend(
        [
            ctypes.c_void_p(compact_to_source_ptr),
            ctypes.c_void_p(expert_start_compact_ptr),
            ctypes.c_void_p(expert_start_mmq32_ptr),
            ctypes.c_void_p(mmq_tile_expert_ptr),
            ctypes.c_void_p(qweight_a_ptr),
            ctypes.c_void_p(qweight_b_ptr),
            ctypes.c_void_p(out_ptr),
            ctypes.c_int64(compact_rows),
            ctypes.c_int64(source_rows),
            ctypes.c_int64(in_features),
            ctypes.c_int64(out_features_a),
            ctypes.c_int64(out_features_b),
            ctypes.c_int64(num_experts),
            ctypes.c_int64(mmq_total_rows),
            ctypes.c_void_p(stream),
        ]
    )
    err = fn(*args)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_t16_dual_interleaved_selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_direct_doublebuf_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    compact_to_source_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq32_ptr: int,
    mmq_tile_expert_ptr: int,
    qweight_dual_ptr: int,
    out_ptr: int,
    compact_rows: int,
    source_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    mmq_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch production D8 prefill from one byte-neutral dual T16 layout."""

    _check_positive(source_rows, "source_rows")
    _check_mmq32_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows,
    )
    if out_features_a != out_features_b:
        raise ValueError("dual-interleaved gate/up features must match")
    if out_features_a % 128 != 0:
        raise ValueError("out_features must be multiples of 128")
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_Q4_T16_DUAL_INTERLEAVED_RAW_PREFETCH_DS8_F32_MMQ128X32_WAVECOLS_DIRECT_DOUBLEBUF_BF16,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 7
        + [ctypes.c_int64] * 7
        + [ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(compact_to_source_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq32_ptr),
        ctypes.c_void_p(mmq_tile_expert_ptr),
        ctypes.c_void_p(qweight_dual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(source_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(mmq_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq32_ptr: int,
    mmq_tile_expert_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    mmq_total_rows: int,
    *,
    rowvec: bool = False,
    wave_cols: bool = False,
    direct_wave_decode: bool = False,
    raw_weight_prefetch_packs: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch one-plane Q4T16 64x32 packed-dot selected down."""

    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    _check_positive(num_experts, "num_experts")
    _check_positive(mmq_total_rows, "mmq_total_rows")
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF K block size 256")
    if out_features % 64 != 0:
        raise ValueError("out_features must be a multiple of 64")
    if mmq_total_rows % 32 != 0:
        raise ValueError("mmq_total_rows must be a multiple of 32")
    if wave_cols and not rowvec:
        raise ValueError("wave_cols requires rowvec")
    if direct_wave_decode and not wave_cols:
        raise ValueError("direct_wave_decode requires wave_cols")
    if raw_weight_prefetch_packs not in (0, 8):
        raise ValueError("raw_weight_prefetch_packs must be 0 or 8")
    if raw_weight_prefetch_packs and not direct_wave_decode:
        raise ValueError(
            "raw weight prefetch requires direct wave-column decode"
        )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    if raw_weight_prefetch_packs:
        symbol = (
            _SYMBOL_Q4_T16_SINGLE_RAW_PREFETCH_DS4_F32_MMQ64X32_WAVECOLS_DIRECT_BF16[
                raw_weight_prefetch_packs
            ]
        )
    elif direct_wave_decode:
        symbol = _SYMBOL_Q4_T16_SINGLE_DS4_F32_MMQ64X32_WAVECOLS_DIRECT_BF16
    elif wave_cols:
        symbol = _SYMBOL_Q4_T16_SINGLE_DS4_F32_MMQ64X32_WAVECOLS_BF16
    elif rowvec:
        symbol = _SYMBOL_Q4_T16_SINGLE_DS4_F32_MMQ64X32_ROWVEC_BF16
    else:
        symbol = _SYMBOL_Q4_T16_SINGLE_DS4_F32_MMQ64X32_BF16
    fn = getattr(
        library,
        symbol,
    )
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
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq32_ptr),
        ctypes.c_void_p(mmq_tile_expert_ptr),
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


def gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
    x_qs_ptr: int,
    x_d_ptr: int,
    x_sum_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x prequantized-Q8_1 selected prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_qs_ptr),
        ctypes.c_void_p(x_d_ptr),
        ctypes.c_void_p(x_sum_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x DS4 Q8_1 integer-WMMA prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_WMMA_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x DS4 Q8_1 32-column WMMA prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_WMMA32_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 DS4 Q8_1 x raw Q4_K four-wave WMMA prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_WMMA64_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    q4_a_ptr: int,
    scale_a_ptr: int,
    min_a_ptr: int,
    q4_b_ptr: int,
    scale_b_ptr: int,
    min_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 DS4 Q8_1 x pre-unpacked Q4_K preview WMMA prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_PREVIEW_WMMA32_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(q4_a_ptr),
        ctypes.c_void_p(scale_a_ptr),
        ctypes.c_void_p(min_a_ptr),
        ctypes.c_void_p(q4_b_ptr),
        ctypes.c_void_p(scale_b_ptr),
        ctypes.c_void_p(min_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x DS4 Q8_1 32-column WMMA+packed-LDS prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_WMMA32_LDSPACK_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x DS4 Q8_1 32-column WMMA+LDS prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_WMMA32_LDS_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    compact_to_source_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq32_ptr: int,
    mmq_tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    mmq_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    _symbol: str = _SYMBOL_DS4_MMQ32_BF16,
) -> None:
    """Launch the source-faithful 32x32 Q4_K x DS4-Q8_1 packed-dot MMQ leaf."""

    _check_mmq32_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(compact_to_source_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq32_ptr),
        ctypes.c_void_p(mmq_tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(mmq_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    compact_to_source_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq32_ptr: int,
    mmq_tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    mmq_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch MMQ32 against byte-exact Q4_K X8 replacement weights."""

    gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
        x_q8_ptr,
        compact_to_source_ptr,
        expert_start_compact_ptr,
        expert_start_mmq32_ptr,
        mmq_tile_expert_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows,
        stream=stream,
        library=library,
        runtime=runtime,
        _symbol=_SYMBOL_X8_DS4_MMQ32_BF16,
    )


def gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    compact_to_source_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq32_ptr: int,
    mmq_tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    mmq_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch MMQ32 directly against resident Q4_K T16 weights."""

    gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
        x_q8_ptr,
        compact_to_source_ptr,
        expert_start_compact_ptr,
        expert_start_mmq32_ptr,
        mmq_tile_expert_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows,
        stream=stream,
        library=library,
        runtime=runtime,
        _symbol=_SYMBOL_T16_DS4_MMQ32_BF16,
    )


def gguf_q4_k_t16_selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    compact_to_source_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq32_ptr: int,
    mmq_tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    source_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    mmq_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch direct-T16 MMQ32 with three residual DS4 activation planes."""

    _check_positive(source_rows, "source_rows")
    _check_mmq32_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_T16_DS4X3_MMQ32_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(compact_to_source_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq32_ptr),
        ctypes.c_void_p(mmq_tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(source_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(mmq_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_t16_selected_dual_q8_1_ds4x3_guarded_mmq32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    compact_to_source_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq32_ptr: int,
    mmq_tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_tiles_ptr: int,
    max_risks: int,
    risk_threshold: float,
    compact_rows: int,
    source_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    mmq_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch residual T16 MMQ32 and queue uncertain BF16 output tiles."""

    _check_positive(source_rows, "source_rows")
    _check_positive(max_risks, "max_risks")
    if risk_threshold < 0:
        raise ValueError("risk_threshold must be non-negative")
    _check_mmq32_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_T16_DS4X3_GUARDED_MMQ32_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(compact_to_source_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq32_ptr),
        ctypes.c_void_p(mmq_tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(risk_count_ptr),
        ctypes.c_void_p(risk_tiles_ptr),
        ctypes.c_int64(max_risks),
        ctypes.c_float(risk_threshold),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(source_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(mmq_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_t16_selected_dual_sparse_exact_correct_bf16(
    source_x_ptr: int,
    compact_to_source_ptr: int,
    expert_start_compact_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_tiles_ptr: int,
    max_risks: int,
    compact_rows: int,
    source_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Exactly recompute queued T16 gate/up tiles; overflow recomputes all."""

    _check_positive(max_risks, "max_risks")
    _check_positive(compact_rows, "compact_rows")
    _check_positive(source_rows, "source_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features_a, "out_features_a")
    _check_positive(out_features_b, "out_features_b")
    _check_positive(num_experts, "num_experts")
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_T16_SPARSE_EXACT_CORRECT_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(source_x_ptr),
        ctypes.c_void_p(compact_to_source_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(risk_count_ptr),
        ctypes.c_void_p(risk_tiles_ptr),
        ctypes.c_int64(max_risks),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(source_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x DS4 block_q8_1_mmq selected prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_common(
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
) -> None:
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features_a, "out_features_a")
    _check_positive(out_features_b, "out_features_b")
    _check_positive(num_experts, "num_experts")
    _check_positive(wmma_total_rows, "wmma_total_rows")
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q4_K block size 256")
    if out_features_a % 16 != 0:
        raise ValueError("out_features_a must be a multiple of 16")
    if out_features_b % 16 != 0:
        raise ValueError("out_features_b must be a multiple of 16")
    if wmma_total_rows % 16 != 0:
        raise ValueError("wmma_total_rows must be a multiple of 16")


def _check_mmq32_common(
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    mmq_total_rows: int,
) -> None:
    if mmq_total_rows % 32 != 0:
        raise ValueError("mmq_total_rows must be a multiple of 32")
    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows,
    )
    if out_features_a % 32 != 0:
        raise ValueError("out_features_a must be a multiple of 32")
    if out_features_b % 32 != 0:
        raise ValueError("out_features_b must be a multiple of 32")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def register_gguf_q4_k_q8_1_selected_prefill_kernels(*, replace: bool = True) -> None:
    """Register the standalone diagnostic Q8_1 selected-prefill prototype."""

    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_dual_interleaved_v1",
            variant=(
                "selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_"
                "direct_doublebuf_prefill_compact32_bf16_bf16_out"
            ),
        ),
        gguf_q4_k_t16_dual_interleaved_selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_direct_doublebuf_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="activation_quant",
            quant="q8_1_ds4x3",
            variant="bf16",
        ),
        gguf_q8_1_mmq_ds4_pack_bf16_d4x3,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="activation_quant",
            quant="q8_1_ds4x3_f32",
            variant="bf16",
        ),
        gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="silu_mul_dual+activation_quant",
            quant="q8_1_ds4x3_f32",
            variant="bf16",
        ),
        gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4x3,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_prefill_compact_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_x8_v1",
            variant="selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_t16_selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant=(
                "selected_dual_q8_1_ds4x3_f32_mmq64x32_"
                "prefill_compact32_bf16_bf16_out"
            ),
        ),
        gguf_q4_k_t16_selected_dual_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_q8_1_ds4x3_guarded_mmq32_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_t16_selected_dual_q8_1_ds4x3_guarded_mmq32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear_repair",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_sparse_exact_bf16",
        ),
        gguf_q4_k_t16_selected_dual_sparse_exact_correct_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q6_k_t16_v1",
            variant=(
                "selected_q8_1_ds4x3_f32_mmq64x32_"
                "prefill_compact32_bf16_bf16_out"
            ),
        ),
        gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant=(
                "selected_q8_1_ds4x3_f32_mmq64x32_"
                "prefill_compact32_bf16_bf16_out"
            ),
        ),
        gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq64x32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_prefill_compact_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )


register_gguf_q4_k_q8_1_selected_prefill_kernels()
