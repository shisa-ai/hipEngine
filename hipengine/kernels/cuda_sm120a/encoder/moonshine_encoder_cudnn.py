"""cuDNN long-bucket conv-route epilogue kernels for the CUDA ``sm_120a`` batch encoder.

Three element-wise FP16-rounding epilogues for the cuDNN conv front end (see
``moonshine_encoder_cudnn.cu``):

- ``moonshine_tanh_apply_fp16``: ``out[i] = fp16(tanhf(fp32(in[i])))`` (conv1);
- ``moonshine_bias_gelu_apply_fp16``:
  ``out[i] = fp16(gelu_f32(fp32(in[i]) + fp32(bias[ch])))``, NCHW
  channel-major layout (conv2, in place);
- ``moonshine_bias_gelu_apply_rowmajor_fp16``: same math but reads NCHW
  channel-major and writes row-major ``[plane, position, channel]`` (the
  encoder ``hidden`` layout, conv3).

These preserve the retained FP16-rounding contract (bias added in FP32,
rounded once; gelu computed on the rounded fp16 value, exactly like the custom
``moonshine_conv_gelu_batch_fp16`` kernels).  The only divergence from the
exact custom route is that cuDNN rounds the conv accumulator to fp16 before
the epilogue and sums in a different order -- the accepted C8 re-derived
numerical gate, not a boundary-rounding change.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_encoder_cudnn.cu")
_OUTPUT_NAME = "moonshine_encoder_cudnn.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)

_TANH_ARGS = (
    ctypes.c_void_p,  # input (fp16)
    ctypes.c_void_p,  # output (fp16)
    ctypes.c_int64,   # elements
    ctypes.c_void_p,  # stream
)
_BIAS_GELU_ARGS = (
    ctypes.c_void_p,  # input (fp16)
    ctypes.c_void_p,  # bias (fp16)
    ctypes.c_void_p,  # output (fp16)
    ctypes.c_int64,   # elements
    ctypes.c_int64,   # channels
    ctypes.c_int64,   # length
    ctypes.c_void_p,  # stream
)
_ROWMAJOR_ARGS = (
    ctypes.c_void_p,  # input (fp16)
    ctypes.c_void_p,  # bias (fp16)
    ctypes.c_void_p,  # output (fp16)
    ctypes.c_int64,   # planes
    ctypes.c_int64,   # channels
    ctypes.c_int64,   # length
    ctypes.c_void_p,  # stream
)


def plan_moonshine_encoder_cudnn_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_encoder_cudnn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_encoder_cudnn(
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
        family="cuda_sm120a_moonshine_encoder_cudnn",
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


def moonshine_tanh_apply_fp16(
    input_ptr: int,
    output_ptr: int,
    elements: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    library = library or build_moonshine_encoder_cudnn(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_tanh_apply_fp16",
        _TANH_ARGS,
        (input_ptr, output_ptr, elements, stream),
        runtime,
    )


def moonshine_bias_gelu_apply_fp16(
    input_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    elements: int,
    channels: int,
    length: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if elements <= 0 or channels <= 0 or length <= 0:
        raise ValueError("elements, channels, and length must be positive")
    library = library or build_moonshine_encoder_cudnn(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_bias_gelu_apply_fp16",
        _BIAS_GELU_ARGS,
        (input_ptr, bias_ptr, output_ptr, elements, channels, length, stream),
        runtime,
    )


def moonshine_bias_gelu_apply_rowmajor_fp16(
    input_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    planes: int,
    channels: int,
    length: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if planes <= 0 or channels <= 0 or length <= 0:
        raise ValueError("planes, channels, and length must be positive")
    library = library or build_moonshine_encoder_cudnn(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_bias_gelu_apply_rowmajor_fp16",
        _ROWMAJOR_ARGS,
        (input_ptr, bias_ptr, output_ptr, planes, channels, length, stream),
        runtime,
    )
