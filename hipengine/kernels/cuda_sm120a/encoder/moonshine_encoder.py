"""Raw-pointer Moonshine FP16 encoder primitives for CUDA ``sm_120a``."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_encoder.cu")
_OUTPUT_NAME = "moonshine_encoder.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_CONV1_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_CONV_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_GROUPNORM_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_GELU_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ROPE_ARGS = (
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
)
_ATTENTION_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_TRANSPOSE_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_moonshine_encoder_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_encoder",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_encoder(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_cuda(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_encoder",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _launch(
    library: ctypes.CDLL,
    symbol: str,
    arg_types: tuple[object, ...],
    args: tuple[object, ...],
    runtime: CudaRuntime,
) -> None:
    function = signed_kernel_fn(library, symbol, arg_types, ctypes.c_int)
    error = function(*args)
    if int(error) != CUDA_SUCCESS:
        runtime.check(int(error))


def moonshine_conv1_tanh_fp16(
    input_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    length: int,
    out_length: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if length <= 0:
        raise ValueError("length must be positive")
    if out_length <= 0:
        raise ValueError("out_length must be positive")
    if threads != 256:
        raise ValueError("threads must be 256")
    library = library or build_moonshine_encoder(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_conv1_tanh_fp16",
        _CONV1_ARGS,
        (input_ptr, weight_ptr, output_ptr, length, out_length, threads, stream),
        runtime,
    )


def _conv_gelu(
    library: ctypes.CDLL,
    runtime: CudaRuntime,
    symbol: str,
    input_ptr: int,
    weight_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    in_length: int,
    out_length: int,
    threads: int,
    stream: int,
) -> None:
    if in_length <= 0:
        raise ValueError("in_length must be positive")
    if out_length <= 0:
        raise ValueError("out_length must be positive")
    _launch(
        library,
        symbol,
        _CONV_ARGS,
        (input_ptr, weight_ptr, bias_ptr, output_ptr, in_length, out_length, threads, stream),
        runtime,
    )


def moonshine_conv2_gelu_fp16(
    input_ptr: int,
    weight_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    in_length: int,
    out_length: int,
    *,
    threads: int = 832,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if threads != 832:
        raise ValueError("threads must be 832 for conv2")
    library = library or build_moonshine_encoder(load=True)
    runtime = runtime or get_cuda_runtime()
    _conv_gelu(
        library, runtime, "hipengine_cuda_sm120a_moonshine_conv2_gelu_fp16",
        input_ptr, weight_ptr, bias_ptr, output_ptr, in_length, out_length,
        threads, stream,
    )


def moonshine_conv3_gelu_fp16(
    input_ptr: int,
    weight_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    in_length: int,
    out_length: int,
    *,
    threads: int = 416,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if threads != 416:
        raise ValueError("threads must be 416 for conv3")
    library = library or build_moonshine_encoder(load=True)
    runtime = runtime or get_cuda_runtime()
    _conv_gelu(
        library, runtime, "hipengine_cuda_sm120a_moonshine_conv3_gelu_fp16",
        input_ptr, weight_ptr, bias_ptr, output_ptr, in_length, out_length,
        threads, stream,
    )


def moonshine_groupnorm_fp16(
    input_ptr: int,
    weight_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    partial_ptr: int,
    mean_rstd_ptr: int,
    channels: int,
    length: int,
    *,
    eps: float = 1.0e-5,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if channels <= 0:
        raise ValueError("channels must be positive")
    if length <= 0:
        raise ValueError("length must be positive")
    if threads != 256:
        raise ValueError("threads must be 256")
    eps_value = float(eps)
    if not eps_value > 0:
        raise ValueError("eps must be positive")
    library = library or build_moonshine_encoder(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_groupnorm_fp16",
        _GROUPNORM_ARGS,
        (
            input_ptr, weight_ptr, bias_ptr, output_ptr, partial_ptr, mean_rstd_ptr,
            channels, length, eps_value, threads, stream,
        ),
        runtime,
    )


def moonshine_gelu_fp16(
    input_ptr: int,
    output_ptr: int,
    elements: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    if threads != 256:
        raise ValueError("threads must be 256")
    library = library or build_moonshine_encoder(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_gelu_fp16",
        _GELU_ARGS,
        (input_ptr, output_ptr, elements, threads, stream),
        runtime,
    )


def moonshine_encoder_rope_fp16(
    query_ptr: int,
    key_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_output_ptr: int,
    key_output_ptr: int,
    heads: int,
    sequence: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if heads <= 0 or sequence <= 0 or head_dim <= 0:
        raise ValueError("heads, sequence, and head_dim must be positive")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    # A table with ``max_positions`` rows serves positions 0..max_positions-1,
    # so a sequence of exactly ``max_positions`` is valid; only longer
    # sequences are rejected.
    if sequence > max_positions:
        raise ValueError("sequence must not exceed the RoPE table size")
    if threads != 256:
        raise ValueError("threads must be 256")
    library = library or build_moonshine_encoder(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_encoder_rope_fp16",
        _ROPE_ARGS,
        (
            query_ptr, key_ptr, cos_ptr, sin_ptr, query_output_ptr, key_output_ptr,
            heads, sequence, head_dim, rotary_dim, max_positions, threads, stream,
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
    if heads != 8:
        raise ValueError("heads must be 8")
    if head_dim != 52:
        raise ValueError("head_dim must be 52")
    if sequence <= 0:
        raise ValueError("sequence must be positive")
    if threads != 32:
        raise ValueError("threads must be 32")
    scale_value = float(head_dim**-0.5) if scale is None else float(scale)
    if not scale_value > 0:
        raise ValueError("scale must be positive")
    library = library or build_moonshine_encoder(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_encoder_attention_fp16",
        _ATTENTION_ARGS,
        (
            query_ptr, key_ptr, value_ptr, mask_ptr, output_ptr, heads, head_dim,
            sequence, scale_value, threads, stream,
        ),
        runtime,
    )


def moonshine_encoder_transpose_head_major_fp16(
    input_ptr: int,
    output_ptr: int,
    sequence: int,
    heads: int,
    head_dim: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Transpose row-major ``[sequence, heads*head_dim]`` to head-major ``[heads, seq, dim]``.

    Bridges the row-major projection layout to the head-major layout expected
    by the full-sequence RoPE and encoder self-attention kernels.
    """

    if sequence <= 0:
        raise ValueError("sequence must be positive")
    if heads <= 0:
        raise ValueError("heads must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if threads != 256:
        raise ValueError("threads must be 256")
    library = library or build_moonshine_encoder(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_encoder_transpose_head_major_fp16",
        _TRANSPOSE_ARGS,
        (input_ptr, output_ptr, sequence, heads, head_dim, threads, stream),
        runtime,
    )


def register_moonshine_encoder_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(_BACKEND, "moonshine_conv1_tanh", "fp16", "strided_valid"),
        moonshine_conv1_tanh_fp16,
        replace=replace,
    )
    register(
        KernelKey(_BACKEND, "moonshine_conv2_gelu", "fp16", "strided_valid"),
        moonshine_conv2_gelu_fp16,
        replace=replace,
    )
    register(
        KernelKey(_BACKEND, "moonshine_conv3_gelu", "fp16", "strided_valid"),
        moonshine_conv3_gelu_fp16,
        replace=replace,
    )
    register(
        KernelKey(_BACKEND, "moonshine_groupnorm", "fp16", "single_group_plane"),
        moonshine_groupnorm_fp16,
        replace=replace,
    )
    register(
        KernelKey(_BACKEND, "moonshine_gelu", "fp16", "exact_erf"),
        moonshine_gelu_fp16,
        replace=replace,
    )
    register(
        KernelKey(_BACKEND, "moonshine_encoder_rope", "fp16", "full_sequence"),
        moonshine_encoder_rope_fp16,
        replace=replace,
    )
    register(
        KernelKey(
            _BACKEND,
            "moonshine_encoder_attention",
            "fp16",
            "full_sequence_non_causal",
        ),
        moonshine_encoder_attention_fp16,
        replace=replace,
    )
    register(
        KernelKey(
            _BACKEND,
            "moonshine_encoder_transpose_head_major",
            "fp16",
            "row_to_head_major",
        ),
        moonshine_encoder_transpose_head_major_fp16,
        replace=replace,
    )


register_moonshine_encoder_kernels()

__all__ = [
    "build_moonshine_encoder",
    "moonshine_conv1_tanh_fp16",
    "moonshine_conv2_gelu_fp16",
    "moonshine_conv3_gelu_fp16",
    "moonshine_encoder_attention_fp16",
    "moonshine_encoder_rope_fp16",
    "moonshine_encoder_transpose_head_major_fp16",
    "moonshine_gelu_fp16",
    "moonshine_groupnorm_fp16",
    "plan_moonshine_encoder_build",
    "register_moonshine_encoder_kernels",
]
