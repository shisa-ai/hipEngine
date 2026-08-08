"""Minimal torch-free cuBLASLt FP16-input/FP32-accumulate GEMM surface.

This is the CUDA ``sm_120a`` counterpart of :mod:`hipengine.core.hipblaslt`,
and mirrors the call sequence of the read-only C6 long-bucket screen
(``scripts/screen_cuda_long_bucket_lt.cu``) that measured cuBLASLt fp16 GEMM
vs the custom Moonshine row-projection kernel at M = 40 / 207 / 1248.

Contract: every problem is ``C = alpha * A @ W.T + beta * C`` where ``A`` is a
row-major ``[M, K]`` FP16 activation tensor and ``W`` is the row-major
``[N, K]`` FP16 weight tensor (so ``W.T`` is the logical ``[K, N]`` B operand,
matching the Moonshine ``X @ weight.T`` storage).  Compute is
``CUBLAS_COMPUTE_32F`` (FP32 accumulation); the output ``C``/``D`` layout is
either ``CUDA_R_16F`` (plain projections, matching the custom kernel's single
FP16 output rounding) or ``CUDA_R_32F`` (when the caller needs an FP32 boundary,
e.g. an FP32 bias/residual epilogue kernel that preserves the retained FP16
rounding contract).

Importing this module does not load ``libcublasLt``; the shared library is
loaded only when :class:`CublasLt` is constructed.  Workspace and descriptors
are created once per shape outside any timed region; the timed :meth:`launch`
performs no allocation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Final

from hipengine.core.cuda import CudaRuntime, get_cuda_runtime

CUBLAS_STATUS_SUCCESS: Final[int] = 0

CUBLAS_OP_N: Final[int] = 0
CUBLAS_OP_T: Final[int] = 1

CUBLAS_COMPUTE_32F: Final[int] = 68

CUDA_R_16F: Final[int] = 2
CUDA_R_32F: Final[int] = 0

CUBLASLT_ORDER_COL: Final[int] = 0
CUBLASLT_ORDER_ROW: Final[int] = 1

CUBLASLT_MATRIX_LAYOUT_TYPE: Final[int] = 0
CUBLASLT_MATRIX_LAYOUT_ORDER: Final[int] = 1
CUBLASLT_MATRIX_LAYOUT_LD: Final[int] = 4

CUBLASLT_MATMUL_DESC_TRANSA: Final[int] = 3
CUBLASLT_MATMUL_DESC_TRANSB: Final[int] = 4

CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: Final[int] = 1

DEFAULT_CUBLASLT_LIBRARY: Final[str] = "libcublasLt.so.13"


class CublasLtError(RuntimeError):
    """Raised when a cuBLASLt call returns a non-success status."""

    def __init__(self, code: int, label: str):
        self.code = int(code)
        super().__init__(f"{label} failed with cuBLASLt status {int(code)}")


class CublasLtAlgo(ctypes.Structure):
    """Semi-opaque 64-byte ``cublasLtMatmulAlgo_t`` (``uint64_t data[8]``)."""

    _fields_ = (("data", ctypes.c_uint64 * 8),)


class CublasLtHeuristicResult(ctypes.Structure):
    """``cublasLtMatmulHeuristicResult_t`` (96 bytes on LP64)."""

    _fields_ = (
        ("algo", CublasLtAlgo),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    )


def _default_cublaslt_library() -> str:
    return ctypes.util.find_library("cublasLt") or DEFAULT_CUBLASLT_LIBRARY


class CublasLt:
    """Own one cuBLASLt handle and shape-specific descriptor children.

    ``runtime`` (a :class:`CudaRuntime`) is used only for the per-problem
    workspace allocation, which happens at problem creation (outside any timed
    encode/decode region).  If omitted, the process-default CUDA runtime is
    used.
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        runtime: CudaRuntime | None = None,
    ) -> None:
        self.path = str(path or _default_cublaslt_library())
        self.library = ctypes.CDLL(self.path)
        self.runtime = runtime if runtime is not None else get_cuda_runtime()
        self._configure()
        handle = ctypes.c_void_p()
        self._check(
            self.library.cublasLtCreate(ctypes.byref(handle)),
            "cublasLtCreate",
        )
        self.handle = int(handle.value or 0)
        self._problems: list[CublasLtProblem] = []

    def problem(
        self,
        rows: int,
        in_features: int,
        out_features: int,
        *,
        output_dtype: int = CUDA_R_16F,
        workspace_nbytes: int = 0,
    ) -> "CublasLtProblem":
        """Create a ``rows x in_features x out_features`` GEMM problem.

        ``output_dtype`` is ``CUDA_R_16F`` (plain FP16 output, single rounding)
        or ``CUDA_R_32F`` (FP32 output for a caller-owned epilogue kernel).
        """
        problem = CublasLtProblem(
            self,
            rows,
            in_features,
            out_features,
            output_dtype=output_dtype,
            workspace_nbytes=workspace_nbytes,
        )
        self._problems.append(problem)
        return problem

    def close(self) -> None:
        for problem in reversed(self._problems):
            problem.close()
        self._problems.clear()
        if self.handle:
            self._check(
                self.library.cublasLtDestroy(ctypes.c_void_p(self.handle)),
                "cublasLtDestroy",
            )
            self.handle = 0

    @staticmethod
    def _check(code: int, label: str) -> None:
        if int(code) != CUBLAS_STATUS_SUCCESS:
            raise CublasLtError(int(code), label)

    def _configure(self) -> None:
        specs = {
            "cublasLtCreate": ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "cublasLtDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "cublasLtMatmulDescCreate": (
                [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                    ctypes.c_int,
                ],
                ctypes.c_int,
            ),
            "cublasLtMatmulDescDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "cublasLtMatmulDescSetAttribute": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
                ctypes.c_int,
            ),
            "cublasLtMatrixLayoutCreate": (
                [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_int64,
                ],
                ctypes.c_int,
            ),
            "cublasLtMatrixLayoutDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "cublasLtMatrixLayoutSetAttribute": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
                ctypes.c_int,
            ),
            "cublasLtMatmulPreferenceCreate": (
                [ctypes.POINTER(ctypes.c_void_p)],
                ctypes.c_int,
            ),
            "cublasLtMatmulPreferenceDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "cublasLtMatmulPreferenceSetAttribute": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
                ctypes.c_int,
            ),
            "cublasLtMatmulAlgoGetHeuristic": (
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.POINTER(CublasLtHeuristicResult),
                    ctypes.POINTER(ctypes.c_int),
                ],
                ctypes.c_int,
            ),
            "cublasLtMatmul": (
                [
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
                    ctypes.c_void_p,
                    ctypes.POINTER(CublasLtAlgo),
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_void_p,
                ],
                ctypes.c_int,
            ),
        }
        for name, (argtypes, restype) in specs.items():
            function = getattr(self.library, name)
            function.argtypes = argtypes
            function.restype = restype


class CublasLtProblem:
    """Row-major descriptor view over ``X @ weight.T`` FP16/FP32 GEMM storage.

    All descriptors are created in the constructor (outside any timed region).
    ``launch`` enqueues one GEMM on the caller's stream with zero allocation.
    """

    def __init__(
        self,
        owner: CublasLt,
        rows: int,
        in_features: int,
        out_features: int,
        *,
        output_dtype: int = CUDA_R_16F,
        workspace_nbytes: int = 0,
    ) -> None:
        if min(rows, in_features, out_features) <= 0:
            raise ValueError("cuBLASLt dimensions must be positive")
        if workspace_nbytes < 0:
            raise ValueError("cuBLASLt workspace_nbytes must be non-negative")
        if output_dtype not in (CUDA_R_16F, CUDA_R_32F):
            raise ValueError("cuBLASLt output_dtype must be CUDA_R_16F or CUDA_R_32F")
        self.owner = owner
        self.rows = int(rows)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.output_dtype = int(output_dtype)
        self.requested_workspace = int(workspace_nbytes)
        self.workspace_size = 0
        self.workspace_ptr = 0
        self.matmul_desc = ctypes.c_void_p()
        self.preference = ctypes.c_void_p()
        self.layouts: list[ctypes.c_void_p] = []
        self._closed = False
        library = owner.library
        self.a = self._layout(CUDA_R_16F, self.rows, self.in_features, self.in_features)
        # Stored weight W is N x K (row-major, ld=K); with TRANSB=OP_T the
        # logical B is K x N, so the physical layout is rows=N, cols=K, ld=K.
        self.b = self._layout(CUDA_R_16F, self.out_features, self.in_features, self.in_features)
        self.c = self._layout(self.output_dtype, self.rows, self.out_features, self.out_features)
        self.d = self._layout(self.output_dtype, self.rows, self.out_features, self.out_features)
        owner._check(
            library.cublasLtMatmulDescCreate(
                ctypes.byref(self.matmul_desc),
                CUBLAS_COMPUTE_32F,
                CUDA_R_32F,
            ),
            "cublasLtMatmulDescCreate",
        )
        trans_a = ctypes.c_int(CUBLAS_OP_N)
        owner._check(
            library.cublasLtMatmulDescSetAttribute(
                self.matmul_desc,
                CUBLASLT_MATMUL_DESC_TRANSA,
                ctypes.byref(trans_a),
                ctypes.sizeof(trans_a),
            ),
            "cublasLtMatmulDescSetAttribute(TRANSA)",
        )
        trans_b = ctypes.c_int(CUBLAS_OP_T)
        owner._check(
            library.cublasLtMatmulDescSetAttribute(
                self.matmul_desc,
                CUBLASLT_MATMUL_DESC_TRANSB,
                ctypes.byref(trans_b),
                ctypes.sizeof(trans_b),
            ),
            "cublasLtMatmulDescSetAttribute(TRANSB)",
        )
        owner._check(
            library.cublasLtMatmulPreferenceCreate(ctypes.byref(self.preference)),
            "cublasLtMatmulPreferenceCreate",
        )
        maximum = ctypes.c_uint64(self.requested_workspace)
        owner._check(
            library.cublasLtMatmulPreferenceSetAttribute(
                self.preference,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ctypes.byref(maximum),
                ctypes.sizeof(maximum),
            ),
            "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE)",
        )
        self._select_algorithm()

    def _select_algorithm(self) -> None:
        """Pick the first success heuristic and allocate its workspace once."""
        library = self.owner.library
        results = (CublasLtHeuristicResult * 16)()
        count = ctypes.c_int()
        self.owner._check(
            library.cublasLtMatmulAlgoGetHeuristic(
                ctypes.c_void_p(self.owner.handle),
                self.matmul_desc,
                self.a,
                self.b,
                self.c,
                self.d,
                self.preference,
                16,
                results,
                ctypes.byref(count),
            ),
            "cublasLtMatmulAlgoGetHeuristic",
        )
        chosen = None
        for index in range(int(count.value)):
            result = results[index]
            if int(result.state) == CUBLAS_STATUS_SUCCESS:
                chosen = result
                break
        if chosen is None:
            raise RuntimeError(
                f"cuBLASLt returned no successful algorithm for M={self.rows} "
                f"K={self.in_features} N={self.out_features}"
            )
        self.algo = chosen.algo
        self.workspace_size = int(chosen.workspace_size)
        if self.workspace_size > 0:
            self.workspace_ptr = self.owner.runtime.malloc(self.workspace_size)

    def launch(
        self,
        x_ptr: int,
        weight_ptr: int,
        out_ptr: int,
        *,
        stream: int = 0,
    ) -> None:
        """Enqueue one FP16 GEMM ``out = x @ weight.T`` on ``stream``."""
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        code = self.owner.library.cublasLtMatmul(
            ctypes.c_void_p(self.owner.handle),
            self.matmul_desc,
            ctypes.byref(alpha),
            ctypes.c_void_p(x_ptr),
            self.a,
            ctypes.c_void_p(weight_ptr),
            self.b,
            ctypes.byref(beta),
            ctypes.c_void_p(out_ptr),
            self.c,
            ctypes.c_void_p(out_ptr),
            self.d,
            ctypes.byref(self.algo),
            ctypes.c_void_p(self.workspace_ptr),
            self.workspace_size,
            ctypes.c_void_p(stream),
        )
        self.owner._check(code, "cublasLtMatmul")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        library = self.owner.library
        if self.workspace_ptr:
            self.owner.runtime.free(self.workspace_ptr)
            self.workspace_ptr = 0
        for layout in reversed(self.layouts):
            self.owner._check(
                library.cublasLtMatrixLayoutDestroy(layout),
                "cublasLtMatrixLayoutDestroy",
            )
        self.layouts.clear()
        if self.preference.value:
            self.owner._check(
                library.cublasLtMatmulPreferenceDestroy(self.preference),
                "cublasLtMatmulPreferenceDestroy",
            )
            self.preference = ctypes.c_void_p()
        if self.matmul_desc.value:
            self.owner._check(
                library.cublasLtMatmulDescDestroy(self.matmul_desc),
                "cublasLtMatmulDescDestroy",
            )
            self.matmul_desc = ctypes.c_void_p()

    def _layout(
        self,
        dtype: int,
        rows: int,
        cols: int,
        leading: int,
    ) -> ctypes.c_void_p:
        layout = ctypes.c_void_p()
        library = self.owner.library
        self.owner._check(
            library.cublasLtMatrixLayoutCreate(
                ctypes.byref(layout),
                dtype,
                rows,
                cols,
                leading,
            ),
            "cublasLtMatrixLayoutCreate",
        )
        order = ctypes.c_int(CUBLASLT_ORDER_ROW)
        self.owner._check(
            library.cublasLtMatrixLayoutSetAttribute(
                layout,
                CUBLASLT_MATRIX_LAYOUT_ORDER,
                ctypes.byref(order),
                ctypes.sizeof(order),
            ),
            "cublasLtMatrixLayoutSetAttribute(ORDER)",
        )
        self.layouts.append(layout)
        return layout


__all__ = [
    "CublasLt",
    "CublasLtAlgo",
    "CublasLtError",
    "CublasLtHeuristicResult",
    "CublasLtProblem",
    "CUBLAS_COMPUTE_32F",
    "CUBLAS_OP_N",
    "CUBLAS_OP_T",
    "CUDA_R_16F",
    "CUDA_R_32F",
]
