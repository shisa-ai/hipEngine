"""Minimal torch-free cuDNN FP16 convolution surface (CUDA ``sm_120a``).

This is the CUDA counterpart of :mod:`hipengine.core.hipblaslt` /
:mod:`hipengine.core.cublaslt`, and mirrors the call sequence of the read-only
C6 long-bucket conv screen (``scripts/screen_cuda_long_bucket_conv.cu``) that
measured ``cudnnConvolutionForward`` fp16 conv vs the custom Moonshine conv
kernels at 40 / 207 / 1,248 frames.

Only the three production Moonshine front-end convolutions are supported:

- conv1: ``1 -> 416`` channels, kernel 127, stride 64, ``+ tanh`` (no bias);
- conv2: ``416 -> 832`` channels, kernel 7,  stride 3, ``+ bias + gelu``;
- conv3: ``832 -> 416`` channels, kernel 3,  stride 2, ``+ bias + gelu``.

Each problem runs ``cudnnConvolutionForward`` with fp16 input/filter/output
(NCHW, ``CUDNN_DEFAULT_MATH`` = FP32 accumulate) and the fastest algorithm
measured by the C6 screen (``CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM``, which
needs no workspace).  The element-wise activation (tanh or bias+gelu) is *not*
inside the cuDNN call; the caller applies the matching retained FP16-rounding
epilogue via the ``moonshine_encoder_cudnn`` kernels, exactly as the C6 screen
did.  Like the cuBLASLt route, the output diverges from the exact custom
kernel at the FP32-reassociation (ULP) level, so this surface belongs to the
C8 batch path behind a re-derived numerical gate with a byte-exact custom
fallback below the long bucket.

Importing this module does not load ``libcudnn``; the shared library is loaded
only when :class:`Cudnn` is constructed.  Descriptors are created once per
shape outside any timed region; the timed :meth:`CudnnConv.forward` performs
no allocation and is stream-ordered (``cudnnSetStream`` is called once at
construction and rebound by :meth:`CudnnConv.forward` if the caller changes
the stream).
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Final

from hipengine.core.cuda import CudaRuntime, get_cuda_runtime

CUDNN_STATUS_SUCCESS: Final[int] = 0

CUDNN_DATA_FLOAT: Final[int] = 0
CUDNN_DATA_HALF: Final[int] = 2

CUDNN_TENSOR_NCHW: Final[int] = 0

CUDNN_DEFAULT_MATH: Final[int] = 0

CUDNN_CROSS_CORRELATION: Final[int] = 1

CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: Final[int] = 0

DEFAULT_CUDNN_LIBRARY: Final[str] = "libcudnn.so.9"


def _default_cudnn_library() -> str:
    found = ctypes.util.find_library("cudnn")
    if found:
        return found
    return DEFAULT_CUDNN_LIBRARY


class CudnnError(RuntimeError):
    """Raised when a cuDNN call returns a non-success status."""

    def __init__(self, code: int, label: str):
        self.code = int(code)
        super().__init__(f"{label} failed with cuDNN status {int(code)}")


def _c_fp32(value: float) -> ctypes.c_float:
    return ctypes.c_float(value)


class Cudnn:
    """Own one cuDNN handle.

    ``runtime`` (a :class:`CudaRuntime`) is used for the per-problem workspace
    allocation (IMPLICIT_GEMM needs none, but a small buffer is reserved so
    the descriptor/workspace contract holds), which happens at problem
    creation outside any timed encode/decode region.  If omitted, the
    process-default CUDA runtime is used.
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        runtime: CudaRuntime | None = None,
    ) -> None:
        self.path = str(path or _default_cudnn_library())
        self.library = ctypes.CDLL(self.path)
        self.runtime = runtime if runtime is not None else get_cuda_runtime()
        self._configure()
        handle = ctypes.c_void_p()
        self._check(self.library.cudnnCreate(ctypes.byref(handle)), "cudnnCreate")
        self.handle = int(handle.value or 0)
        self._convs: list[CudnnConv] = []

    def _configure(self) -> None:
        library = self.library
        library.cudnnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.cudnnCreate.restype = ctypes.c_int
        library.cudnnDestroy.argtypes = [ctypes.c_void_p]
        library.cudnnDestroy.restype = ctypes.c_int
        library.cudnnSetStream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.cudnnSetStream.restype = ctypes.c_int
        library.cudnnCreateTensorDescriptor.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        library.cudnnCreateTensorDescriptor.restype = ctypes.c_int
        library.cudnnDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]
        library.cudnnDestroyTensorDescriptor.restype = ctypes.c_int
        library.cudnnSetTensor4dDescriptor.argtypes = [
            ctypes.c_void_p,  # descriptor
            ctypes.c_int,     # format
            ctypes.c_int,     # data type
            ctypes.c_int,     # n
            ctypes.c_int,     # c
            ctypes.c_int,     # h
            ctypes.c_int,     # w
        ]
        library.cudnnSetTensor4dDescriptor.restype = ctypes.c_int
        library.cudnnCreateFilterDescriptor.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        library.cudnnCreateFilterDescriptor.restype = ctypes.c_int
        library.cudnnDestroyFilterDescriptor.argtypes = [ctypes.c_void_p]
        library.cudnnDestroyFilterDescriptor.restype = ctypes.c_int
        library.cudnnSetFilter4dDescriptor.argtypes = [
            ctypes.c_void_p,  # descriptor
            ctypes.c_int,     # data type
            ctypes.c_int,     # format
            ctypes.c_int,     # k
            ctypes.c_int,     # c
            ctypes.c_int,     # h
            ctypes.c_int,     # w
        ]
        library.cudnnSetFilter4dDescriptor.restype = ctypes.c_int
        library.cudnnCreateConvolutionDescriptor.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        library.cudnnCreateConvolutionDescriptor.restype = ctypes.c_int
        library.cudnnDestroyConvolutionDescriptor.argtypes = [ctypes.c_void_p]
        library.cudnnDestroyConvolutionDescriptor.restype = ctypes.c_int
        library.cudnnSetConvolution2dDescriptor.argtypes = [
            ctypes.c_void_p,  # descriptor
            ctypes.c_int,     # pad_h
            ctypes.c_int,     # pad_w
            ctypes.c_int,     # u (vertical stride)
            ctypes.c_int,     # v (horizontal stride)
            ctypes.c_int,     # dilation_h
            ctypes.c_int,     # dilation_w
            ctypes.c_int,     # mode
            ctypes.c_int,     # compute data type
        ]
        library.cudnnSetConvolution2dDescriptor.restype = ctypes.c_int
        library.cudnnSetConvolutionMathType.argtypes = [
            ctypes.c_void_p,  # descriptor
            ctypes.c_int,     # math type
        ]
        library.cudnnSetConvolutionMathType.restype = ctypes.c_int
        library.cudnnGetConvolutionForwardWorkspaceSize.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # xDesc
            ctypes.c_void_p,  # wDesc
            ctypes.c_void_p,  # convDesc
            ctypes.c_void_p,  # yDesc
            ctypes.c_int,     # algo
            ctypes.POINTER(ctypes.c_size_t),  # size in bytes
        ]
        library.cudnnGetConvolutionForwardWorkspaceSize.restype = ctypes.c_int
        library.cudnnConvolutionForward.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # alpha
            ctypes.c_void_p,  # xDesc
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # wDesc
            ctypes.c_void_p,  # w
            ctypes.c_void_p,  # convDesc
            ctypes.c_int,     # algo
            ctypes.c_void_p,  # workspace
            ctypes.c_size_t,  # workspace bytes
            ctypes.c_void_p,  # beta
            ctypes.c_void_p,  # yDesc
            ctypes.c_void_p,  # y
        ]
        library.cudnnConvolutionForward.restype = ctypes.c_int

    def _check(self, status: int, label: str) -> None:
        if int(status) != CUDNN_STATUS_SUCCESS:
            raise CudnnError(int(status), label)

    def conv(
        self,
        batch: int,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        in_length: int,
        *,
        stream: int = 0,
        workspace_bytes: int = 0,
        output_dtype: int = CUDNN_DATA_HALF,
    ) -> "CudnnConv":
        """Create one fixed-shape cuDNN conv problem (outside any timed region)."""

        out_length = (in_length - kernel) // stride + 1
        if out_length <= 0:
            raise ValueError(
                f"invalid conv geometry (in_length={in_length}, kernel={kernel}, "
                f"stride={stride})"
            )
        conv = CudnnConv(
            owner=self,
            batch=batch,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            in_length=in_length,
            out_length=out_length,
            stream=stream,
            workspace_bytes=workspace_bytes,
            output_dtype=output_dtype,
        )
        self._convs.append(conv)
        return conv

    def set_stream(self, stream: int) -> None:
        self._check(
            self.library.cudnnSetStream(self.handle, ctypes.c_void_p(stream)),
            "cudnnSetStream",
        )

    def close(self) -> None:
        for conv in self._convs:
            conv._destroy()
        self._convs.clear()
        if self.handle:
            self._check(self.library.cudnnDestroy(self.handle), "cudnnDestroy")
            self.handle = 0


class CudnnConv:
    """One fixed-shape ``cudnnConvolutionForward`` problem with descriptors.

    The descriptors are created once at construction; :meth:`forward` binds
    the handle to the requested stream and launches a single
    ``cudnnConvolutionForward`` (IMPLICIT_GEMM, no workspace) with
    ``alpha = 1`` / ``beta = 0``.  fp16 in/out, FP32 accumulate
    (``CUDNN_DEFAULT_MATH``), NCHW layouts.  The caller supplies the x/w/y
    device pointers and the (optional) workspace buffer.
    """

    def __init__(
        self,
        *,
        owner: Cudnn,
        batch: int,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        in_length: int,
        out_length: int,
        stream: int,
        workspace_bytes: int,
        output_dtype: int = CUDNN_DATA_HALF,
    ) -> None:
        self.owner = owner
        self.batch = int(batch)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel = int(kernel)
        self.stride = int(stride)
        self.in_length = int(in_length)
        self.out_length = int(out_length)
        self.output_dtype = int(output_dtype)
        self.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        self.workspace_bytes = int(workspace_bytes)
        library = owner.library
        self._x_desc = ctypes.c_void_p()
        self._y_desc = ctypes.c_void_p()
        self._w_desc = ctypes.c_void_p()
        self._conv_desc = ctypes.c_void_p()
        self._destroyed = False
        try:
            owner._check(
                library.cudnnCreateTensorDescriptor(ctypes.byref(self._x_desc)),
                "cudnnCreateTensorDescriptor(x)",
            )
            owner._check(
                library.cudnnCreateTensorDescriptor(ctypes.byref(self._y_desc)),
                "cudnnCreateTensorDescriptor(y)",
            )
            owner._check(
                library.cudnnCreateFilterDescriptor(ctypes.byref(self._w_desc)),
                "cudnnCreateFilterDescriptor",
            )
            owner._check(
                library.cudnnCreateConvolutionDescriptor(
                    ctypes.byref(self._conv_desc)
                ),
                "cudnnCreateConvolutionDescriptor",
            )
            owner._check(
                library.cudnnSetTensor4dDescriptor(
                    self._x_desc,
                    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_HALF,
                    self.batch,
                    self.in_channels,
                    1,
                    self.in_length,
                ),
                "cudnnSetTensor4dDescriptor(x)",
            )
            owner._check(
                library.cudnnSetTensor4dDescriptor(
                    self._y_desc,
                    CUDNN_TENSOR_NCHW,
                    self.output_dtype,
                    self.batch,
                    self.out_channels,
                    1,
                    self.out_length,
                ),
                "cudnnSetTensor4dDescriptor(y)",
            )
            owner._check(
                library.cudnnSetFilter4dDescriptor(
                    self._w_desc,
                    CUDNN_DATA_HALF,
                    CUDNN_TENSOR_NCHW,
                    self.out_channels,
                    self.in_channels,
                    1,
                    self.kernel,
                ),
                "cudnnSetFilter4dDescriptor",
            )
            owner._check(
                library.cudnnSetConvolution2dDescriptor(
                    self._conv_desc,
                    0,
                    0,
                    1,
                    self.stride,
                    1,
                    1,
                    CUDNN_CROSS_CORRELATION,
                    CUDNN_DATA_HALF,
                ),
                "cudnnSetConvolution2dDescriptor",
            )
            owner._check(
                library.cudnnSetConvolutionMathType(
                    self._conv_desc,
                    CUDNN_DEFAULT_MATH,
                ),
                "cudnnSetConvolutionMathType",
            )
            required = ctypes.c_size_t()
            owner._check(
                library.cudnnGetConvolutionForwardWorkspaceSize(
                    owner.handle,
                    self._x_desc,
                    self._w_desc,
                    self._conv_desc,
                    self._y_desc,
                    self.algo,
                    ctypes.byref(required),
                ),
                "cudnnGetConvolutionForwardWorkspaceSize",
            )
            self.required_workspace_bytes = int(required.value)
            if self.workspace_bytes < self.required_workspace_bytes:
                self.workspace_bytes = self.required_workspace_bytes
        except Exception:
            self._destroy()
            raise

    def forward(
        self,
        x_ptr: int,
        w_ptr: int,
        y_ptr: int,
        *,
        workspace_ptr: int = 0,
        stream: int = 0,
    ) -> None:
        """Run one conv forward on ``stream`` with the given device pointers."""

        if self._destroyed:
            raise RuntimeError("CudnnConv is closed")
        owner = self.owner
        owner.set_stream(stream)
        alpha = _c_fp32(1.0)
        beta = _c_fp32(0.0)
        status = owner.library.cudnnConvolutionForward(
            owner.handle,
            ctypes.byref(alpha),
            self._x_desc,
            ctypes.c_void_p(x_ptr),
            self._w_desc,
            ctypes.c_void_p(w_ptr),
            self._conv_desc,
            self.algo,
            ctypes.c_void_p(workspace_ptr),
            ctypes.c_size_t(self.workspace_bytes),
            ctypes.byref(beta),
            self._y_desc,
            ctypes.c_void_p(y_ptr),
        )
        owner._check(status, "cudnnConvolutionForward")

    def _destroy(self) -> None:
        if self._destroyed:
            return
        library = self.owner.library
        if self._conv_desc.value:
            self.owner._check(
                library.cudnnDestroyConvolutionDescriptor(self._conv_desc),
                "cudnnDestroyConvolutionDescriptor",
            )
            self._conv_desc = ctypes.c_void_p()
        if self._w_desc.value:
            self.owner._check(
                library.cudnnDestroyFilterDescriptor(self._w_desc),
                "cudnnDestroyFilterDescriptor",
            )
            self._w_desc = ctypes.c_void_p()
        if self._y_desc.value:
            self.owner._check(
                library.cudnnDestroyTensorDescriptor(self._y_desc),
                "cudnnDestroyTensorDescriptor",
            )
            self._y_desc = ctypes.c_void_p()
        if self._x_desc.value:
            self.owner._check(
                library.cudnnDestroyTensorDescriptor(self._x_desc),
                "cudnnDestroyTensorDescriptor",
            )
            self._x_desc = ctypes.c_void_p()
        self._destroyed = True
