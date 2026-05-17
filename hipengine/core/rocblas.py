"""Lazy ctypes wrappers for the small rocBLAS surface hipEngine uses.

The module is intentionally torch-free and does not load ``librocblas`` at import
time. Callers opt in to these helpers for diagnostic/prototype bulk GEMM paths.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

ROCBLAS_SUCCESS = 0
ROCBLAS_OPERATION_NONE = 111
ROCBLAS_OPERATION_TRANSPOSE = 112
ROCBLAS_DATATYPE_F16_R = 150
ROCBLAS_DATATYPE_F32_R = 151
ROCBLAS_GEMM_ALGO_STANDARD = 0
DEFAULT_ROCBLAS_LIBRARY = "librocblas.so"


class RocblasError(RuntimeError):
    """Raised when a rocBLAS API call returns a non-success status."""

    def __init__(self, code: int, message: str = "rocBLAS call failed") -> None:
        self.code = int(code)
        super().__init__(f"rocBLAS error {self.code}: {message}")


@dataclass
class Rocblas:
    library: ctypes.CDLL
    handle: int

    @classmethod
    def load(cls, path: str = DEFAULT_ROCBLAS_LIBRARY) -> "Rocblas":
        library = ctypes.CDLL(path)
        _configure(library)
        handle = ctypes.c_void_p()
        _check(library.rocblas_create_handle(ctypes.byref(handle)), "rocblas_create_handle")
        return cls(library=library, handle=0 if handle.value is None else int(handle.value))

    def close(self) -> None:
        if self.handle:
            _check(self.library.rocblas_destroy_handle(ctypes.c_void_p(self.handle)), "rocblas_destroy_handle")
            self.handle = 0

    def set_stream(self, stream: int) -> None:
        _check(
            self.library.rocblas_set_stream(ctypes.c_void_p(self.handle), ctypes.c_void_p(stream)),
            "rocblas_set_stream",
        )

    def hgemm_rowmajor_nt(
        self,
        x_ptr: int,
        weight_ptr: int,
        out_ptr: int,
        *,
        rows: int,
        in_features: int,
        out_features: int,
        stream: int = 0,
    ) -> None:
        """Compute ``out[rows,out_features] = x[rows,in] @ weight[out,in].T``.

        All tensors are contiguous row-major FP16 device buffers. rocBLAS is
        column-major, so the call computes the transposed view:
        ``C_col[out_features, rows] = weight @ x.T``.
        """

        _check_shape(rows=rows, in_features=in_features, out_features=out_features)
        self.set_stream(stream)
        alpha = ctypes.c_uint16(0x3C00)  # IEEE FP16 1.0
        beta = ctypes.c_uint16(0x0000)  # IEEE FP16 0.0
        _check(
            self.library.rocblas_hgemm(
                ctypes.c_void_p(self.handle),
                ctypes.c_int(ROCBLAS_OPERATION_TRANSPOSE),
                ctypes.c_int(ROCBLAS_OPERATION_NONE),
                ctypes.c_int(out_features),
                ctypes.c_int(rows),
                ctypes.c_int(in_features),
                ctypes.byref(alpha),
                ctypes.c_void_p(weight_ptr),
                ctypes.c_int(in_features),
                ctypes.c_void_p(x_ptr),
                ctypes.c_int(in_features),
                ctypes.byref(beta),
                ctypes.c_void_p(out_ptr),
                ctypes.c_int(out_features),
            ),
            "rocblas_hgemm",
        )

    def gemm_ex_rowmajor_nt_fp16_compute_f32(
        self,
        x_ptr: int,
        weight_ptr: int,
        out_ptr: int,
        *,
        rows: int,
        in_features: int,
        out_features: int,
        stream: int = 0,
    ) -> None:
        """FP16 row-major NT GEMM with FP32 accumulation and FP16 output."""

        _check_shape(rows=rows, in_features=in_features, out_features=out_features)
        self.set_stream(stream)
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        _check(
            self.library.rocblas_gemm_ex(
                ctypes.c_void_p(self.handle),
                ctypes.c_int(ROCBLAS_OPERATION_TRANSPOSE),
                ctypes.c_int(ROCBLAS_OPERATION_NONE),
                ctypes.c_int(out_features),
                ctypes.c_int(rows),
                ctypes.c_int(in_features),
                ctypes.byref(alpha),
                ctypes.c_void_p(weight_ptr),
                ctypes.c_int(ROCBLAS_DATATYPE_F16_R),
                ctypes.c_int(in_features),
                ctypes.c_void_p(x_ptr),
                ctypes.c_int(ROCBLAS_DATATYPE_F16_R),
                ctypes.c_int(in_features),
                ctypes.byref(beta),
                ctypes.c_void_p(out_ptr),
                ctypes.c_int(ROCBLAS_DATATYPE_F16_R),
                ctypes.c_int(out_features),
                ctypes.c_void_p(out_ptr),
                ctypes.c_int(ROCBLAS_DATATYPE_F16_R),
                ctypes.c_int(out_features),
                ctypes.c_int(ROCBLAS_DATATYPE_F32_R),
                ctypes.c_int(ROCBLAS_GEMM_ALGO_STANDARD),
                ctypes.c_int32(0),
                ctypes.c_uint32(0),
            ),
            "rocblas_gemm_ex",
        )


_DEFAULT_ROCBLAS: Rocblas | None = None


def get_rocblas(path: str = DEFAULT_ROCBLAS_LIBRARY) -> Rocblas:
    global _DEFAULT_ROCBLAS
    if _DEFAULT_ROCBLAS is None:
        _DEFAULT_ROCBLAS = Rocblas.load(path)
    return _DEFAULT_ROCBLAS


def reset_default_rocblas_for_tests() -> None:
    global _DEFAULT_ROCBLAS
    if _DEFAULT_ROCBLAS is not None:
        _DEFAULT_ROCBLAS.close()
    _DEFAULT_ROCBLAS = None


def rocblas_hgemm_rowmajor_nt_fp16(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    stream: int = 0,
    handle: Rocblas | None = None,
) -> None:
    blas = handle or get_rocblas()
    blas.hgemm_rowmajor_nt(
        x_ptr,
        weight_ptr,
        out_ptr,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        stream=stream,
    )


def rocblas_gemm_ex_rowmajor_nt_fp16_compute_f32(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    stream: int = 0,
    handle: Rocblas | None = None,
) -> None:
    blas = handle or get_rocblas()
    blas.gemm_ex_rowmajor_nt_fp16_compute_f32(
        x_ptr,
        weight_ptr,
        out_ptr,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        stream=stream,
    )


def _configure(library: ctypes.CDLL) -> None:
    library.rocblas_create_handle.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.rocblas_create_handle.restype = ctypes.c_int
    library.rocblas_destroy_handle.argtypes = [ctypes.c_void_p]
    library.rocblas_destroy_handle.restype = ctypes.c_int
    library.rocblas_set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.rocblas_set_stream.restype = ctypes.c_int
    library.rocblas_hgemm.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    library.rocblas_hgemm.restype = ctypes.c_int
    library.rocblas_gemm_ex.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_uint32,
    ]
    library.rocblas_gemm_ex.restype = ctypes.c_int


def _check(code: int, message: str) -> None:
    if int(code) != ROCBLAS_SUCCESS:
        raise RocblasError(int(code), message)


def _check_shape(*, rows: int, in_features: int, out_features: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
