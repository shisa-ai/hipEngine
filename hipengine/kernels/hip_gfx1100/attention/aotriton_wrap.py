"""Python bindings for hipEngine's stable AOTriton C-ABI shim.

The module is torch-free.  It builds/loads only the small hipEngine-owned wrapper
shared object; the wrapper links against the manifest-pinned AOTriton runtime
found by :mod:`hipengine.kernels.hip_gfx1100.attention.aotriton`.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Sequence

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.hip_gfx1100.attention.aotriton import AotritonRuntimeTree, aotriton_runtime_tree
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("aotriton_wrap.cc")
_OUTPUT_NAME = "hipengine_aotriton_wrap.so"
_SYMBOL_CHECK_GPU = "hipengine_aotriton_check_gpu"
_SYMBOL_GATE_MUL_FP16_INPLACE = "hipengine_aotriton_gate_mul_fp16_inplace"
_SYMBOL_GATE_MUL_BF16_TO_FP16 = "hipengine_aotriton_gate_mul_bf16_to_fp16"
_SYMBOL_ATTN_FWD_COMPACT_VARLEN = "hipengine_aotriton_attn_fwd_compact_varlen"
_SYMBOL_ATTN_FWD_V3_COMPACT_VARLEN = "hipengine_aotriton_attn_fwd_v3_compact_varlen"
_SYMBOL_ATTN_FWD_COMPACT_VARLEN_GQA_PER_Q_HEAD = "hipengine_aotriton_attn_fwd_compact_varlen_gqa_per_q_head"

AOTRITON_DTYPE_FP32 = 1
AOTRITON_DTYPE_FP16 = 2
AOTRITON_DTYPE_BF16 = 3
AOTRITON_DTYPE_INT32 = 12
AOTRITON_DTYPE_INT64 = 13

_DTYPE_TO_AOTRITON = {
    DType.FP32: AOTRITON_DTYPE_FP32,
    DType.FP16: AOTRITON_DTYPE_FP16,
    DType.BF16: AOTRITON_DTYPE_BF16,
    DType.INT32: AOTRITON_DTYPE_INT32,
    DType.INT64: AOTRITON_DTYPE_INT64,
}


class AotritonTensor1(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("sizes", ctypes.c_int64 * 1),
        ("strides", ctypes.c_int64 * 1),
        ("dtype", ctypes.c_int32),
    ]


class AotritonTensor2(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
        ("dtype", ctypes.c_int32),
    ]


class AotritonTensor4(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("sizes", ctypes.c_int64 * 4),
        ("strides", ctypes.c_int64 * 4),
        ("dtype", ctypes.c_int32),
    ]


def plan_aotriton_wrap_build(
    *,
    home_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    runtime = aotriton_runtime_tree(home_root)
    return plan_hip_build(
        sources=[_SOURCE],
        family="aotriton_wrap",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        include_dirs=[runtime.include_dir],
        extra_flags=_link_flags(runtime),
        output_name=_OUTPUT_NAME,
    )


def build_aotriton_wrap(
    *,
    home_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    runtime = aotriton_runtime_tree(home_root)
    return build_hip(
        sources=[_SOURCE],
        family="aotriton_wrap",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        include_dirs=[runtime.include_dir],
        extra_flags=_link_flags(runtime),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def aotriton_attn_fwd_compact_varlen(
    q: AotritonTensor4,
    k: AotritonTensor4,
    v: AotritonTensor4,
    cu_seqlens_q: AotritonTensor1,
    cu_seqlens_k: AotritonTensor1,
    softmax_lse: AotritonTensor2,
    out: AotritonTensor4,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    is_causal: bool = True,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    home_root: str | Path | None = None,
) -> None:
    """Launch AOTriton compact-varlen forward attention through the C shim."""

    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("max_seqlen_q and max_seqlen_k must be positive")
    library = library or build_aotriton_wrap(home_root=home_root, load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ATTN_FWD_COMPACT_VARLEN)
    fn.argtypes = [
        ctypes.POINTER(AotritonTensor4),
        ctypes.POINTER(AotritonTensor4),
        ctypes.POINTER(AotritonTensor4),
        ctypes.POINTER(AotritonTensor1),
        ctypes.POINTER(AotritonTensor1),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(AotritonTensor2),
        ctypes.POINTER(AotritonTensor4),
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.byref(q),
        ctypes.byref(k),
        ctypes.byref(v),
        ctypes.byref(cu_seqlens_q),
        ctypes.byref(cu_seqlens_k),
        ctypes.c_int32(max_seqlen_q),
        ctypes.c_int32(max_seqlen_k),
        ctypes.byref(softmax_lse),
        ctypes.byref(out),
        ctypes.c_float(sm_scale),
        ctypes.c_int32(1 if is_causal else 0),
        ctypes.c_void_p(stream),
    )
    _check_hip(runtime, err)


def aotriton_attn_fwd_v3_compact_varlen(
    q: AotritonTensor4,
    k: AotritonTensor4,
    v: AotritonTensor4,
    cu_seqlens_q: AotritonTensor1,
    cu_seqlens_k: AotritonTensor1,
    softmax_lse: AotritonTensor2,
    out: AotritonTensor4,
    *,
    persistent_atomic_counter_ptr: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    is_causal: bool = True,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    home_root: str | Path | None = None,
) -> None:
    """Launch AOTriton V3 compact-varlen forward attention through the C shim."""

    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("max_seqlen_q and max_seqlen_k must be positive")
    if is_causal and int(persistent_atomic_counter_ptr) == 0:
        raise ValueError("causal AOTriton V3 prefill requires a persistent atomic counter")
    library = library or build_aotriton_wrap(home_root=home_root, load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ATTN_FWD_V3_COMPACT_VARLEN)
    fn.argtypes = [
        ctypes.POINTER(AotritonTensor4),
        ctypes.POINTER(AotritonTensor4),
        ctypes.POINTER(AotritonTensor4),
        ctypes.POINTER(AotritonTensor1),
        ctypes.POINTER(AotritonTensor1),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(AotritonTensor2),
        ctypes.POINTER(AotritonTensor4),
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.byref(q),
        ctypes.byref(k),
        ctypes.byref(v),
        ctypes.byref(cu_seqlens_q),
        ctypes.byref(cu_seqlens_k),
        ctypes.c_int32(max_seqlen_q),
        ctypes.c_int32(max_seqlen_k),
        ctypes.byref(softmax_lse),
        ctypes.byref(out),
        ctypes.c_void_p(persistent_atomic_counter_ptr),
        ctypes.c_float(sm_scale),
        ctypes.c_int32(1 if is_causal else 0),
        ctypes.c_void_p(stream),
    )
    _check_hip(runtime, err)


def aotriton_attn_fwd_compact_varlen_gqa_per_q_head(
    q: AotritonTensor4,
    k: AotritonTensor4,
    v: AotritonTensor4,
    cu_seqlens_q: AotritonTensor1,
    cu_seqlens_k: AotritonTensor1,
    softmax_lse: AotritonTensor2,
    out: AotritonTensor4,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    is_causal: bool = True,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    home_root: str | Path | None = None,
) -> None:
    """Launch GQA by issuing one H=1 AOTriton compact-varlen call per Q head."""

    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("max_seqlen_q and max_seqlen_k must be positive")
    library = library or build_aotriton_wrap(home_root=home_root, load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ATTN_FWD_COMPACT_VARLEN_GQA_PER_Q_HEAD)
    fn.argtypes = [
        ctypes.POINTER(AotritonTensor4),
        ctypes.POINTER(AotritonTensor4),
        ctypes.POINTER(AotritonTensor4),
        ctypes.POINTER(AotritonTensor1),
        ctypes.POINTER(AotritonTensor1),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(AotritonTensor2),
        ctypes.POINTER(AotritonTensor4),
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.byref(q),
        ctypes.byref(k),
        ctypes.byref(v),
        ctypes.byref(cu_seqlens_q),
        ctypes.byref(cu_seqlens_k),
        ctypes.c_int32(max_seqlen_q),
        ctypes.c_int32(max_seqlen_k),
        ctypes.byref(softmax_lse),
        ctypes.byref(out),
        ctypes.c_float(sm_scale),
        ctypes.c_int32(1 if is_causal else 0),
        ctypes.c_void_p(stream),
    )
    _check_hip(runtime, err)


def aotriton_gate_mul_fp16_inplace(
    attn_out_ptr: int,
    gate_ptr: int,
    total: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    home_root: str | Path | None = None,
) -> None:
    """Apply the Qwen3.5 sigmoid gate in-place to FP16 AOTriton output."""

    if total <= 0:
        raise ValueError("total must be positive")
    library = library or build_aotriton_wrap(home_root=home_root, load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GATE_MUL_FP16_INPLACE)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    _check_hip(
        runtime,
        fn(ctypes.c_void_p(attn_out_ptr), ctypes.c_void_p(gate_ptr), ctypes.c_int64(total), ctypes.c_void_p(stream)),
    )


def aotriton_gate_mul_bf16_to_fp16(
    attn_out_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    total: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    home_root: str | Path | None = None,
) -> None:
    """Apply Qwen3.5 sigmoid gate from BF16 attention output into FP16 output."""

    if total <= 0:
        raise ValueError("total must be positive")
    library = library or build_aotriton_wrap(home_root=home_root, load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GATE_MUL_BF16_TO_FP16)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    _check_hip(
        runtime,
        fn(
            ctypes.c_void_p(attn_out_ptr),
            ctypes.c_void_p(gate_ptr),
            ctypes.c_void_p(out_ptr),
            ctypes.c_int64(total),
            ctypes.c_void_p(stream),
        ),
    )


def aotriton_check_gpu(
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    home_root: str | Path | None = None,
) -> None:
    """Run AOTriton's stream/GPU compatibility check through the shim."""

    library = library or build_aotriton_wrap(home_root=home_root, load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_CHECK_GPU)
    fn.argtypes = [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    _check_hip(runtime, fn(ctypes.c_void_p(stream)))


def register_aotriton_wrap_kernels(*, replace: bool = True) -> None:
    """Register the AOTriton prefill-attention variant without selecting it by default."""

    register(
        KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "aotriton_attn_fwd"),
        aotriton_attn_fwd_compact_varlen,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "aotriton_attn_fwd_v3"),
        aotriton_attn_fwd_v3_compact_varlen,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "full_attn_prefill", "gguf_qwen35", "aotriton_attn_fwd_v3"),
        aotriton_attn_fwd_v3_compact_varlen,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "aotriton_attn_fwd_gqa_per_q_head"),
        aotriton_attn_fwd_compact_varlen_gqa_per_q_head,
        replace=replace,
    )


def tensor1(ptr: int, sizes: Sequence[int], strides: Sequence[int], dtype: str | DType) -> AotritonTensor1:
    return _tensor(AotritonTensor1, ptr, sizes, strides, dtype, rank=1)


def tensor2(ptr: int, sizes: Sequence[int], strides: Sequence[int], dtype: str | DType) -> AotritonTensor2:
    return _tensor(AotritonTensor2, ptr, sizes, strides, dtype, rank=2)


def tensor4(ptr: int, sizes: Sequence[int], strides: Sequence[int], dtype: str | DType) -> AotritonTensor4:
    return _tensor(AotritonTensor4, ptr, sizes, strides, dtype, rank=4)


def aotriton_dtype(dtype: str | DType) -> int:
    parsed = DType.parse(dtype)
    try:
        return _DTYPE_TO_AOTRITON[parsed]
    except KeyError as exc:
        raise ValueError(f"dtype {parsed.value!r} is not supported by the AOTriton wrapper") from exc


def _tensor(cls, ptr: int, sizes: Sequence[int], strides: Sequence[int], dtype: str | DType, *, rank: int):
    if ptr == 0:
        raise ValueError("AOTriton tensor pointer must be non-null")
    if len(sizes) != rank or len(strides) != rank:
        raise ValueError(f"expected rank-{rank} sizes/strides")
    if any(int(value) < 0 for value in (*sizes, *strides)):
        raise ValueError("AOTriton tensor sizes/strides must be non-negative")
    return cls(
        ctypes.c_void_p(int(ptr)),
        (ctypes.c_int64 * rank)(*(int(value) for value in sizes)),
        (ctypes.c_int64 * rank)(*(int(value) for value in strides)),
        ctypes.c_int32(aotriton_dtype(dtype)),
    )


def _link_flags(runtime: AotritonRuntimeTree) -> tuple[str, ...]:
    lib_dir = runtime.library.parent
    return (
        "-Wl,--no-as-needed",
        f"-L{lib_dir}",
        "-laotriton_v2",
        "-Wl,--as-needed",
        f"-Wl,-rpath,{lib_dir}",
    )


def _check_hip(runtime: HipRuntime, err: int) -> None:
    if err != 0:
        raise RuntimeError(f"AOTriton HIP call failed: {runtime.error_string(err)} ({err})")


register_aotriton_wrap_kernels()
