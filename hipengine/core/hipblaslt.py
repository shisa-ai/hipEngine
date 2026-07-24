"""Minimal torch-free hipBLASLt FP16-input/weight, FP32-output surface."""

from __future__ import annotations

import ctypes

HIP_R_32F = 0
HIP_R_16F = 2
HIPBLAS_COMPUTE_32F = 2
HIPBLAS_OP_T = 112
HIPBLASLT_ORDER_COL = 0
HIPBLASLT_MATRIX_LAYOUT_ORDER = 3
HIPBLASLT_MATMUL_DESC_TRANSA = 0
HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
HIPBLAS_STATUS_SUCCESS = 0


class HipblasLtAlgo(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.c_uint8 * 16),
        ("max_workspace_bytes", ctypes.c_size_t),
    )


class HipblasLtHeuristicResult(ctypes.Structure):
    _fields_ = (
        ("algo", HipblasLtAlgo),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    )


class HipblasLt:
    """Own one hipBLASLt handle and shape-specific descriptor children."""

    def __init__(self, path: str = "libhipblaslt.so") -> None:
        self.path = str(path)
        self.library = ctypes.CDLL(self.path)
        self._configure()
        handle = ctypes.c_void_p()
        self._check(
            self.library.hipblasLtCreate(ctypes.byref(handle)),
            "hipblasLtCreate",
        )
        self.handle = int(handle.value or 0)
        self._problems: list[HipblasLtProblem] = []

    def problem(
        self,
        rows: int,
        in_features: int,
        out_features: int,
        workspace_nbytes: int = 0,
    ) -> "HipblasLtProblem":
        problem = HipblasLtProblem(
            self,
            rows,
            in_features,
            out_features,
            workspace_nbytes,
        )
        self._problems.append(problem)
        return problem

    def close(self) -> None:
        for problem in reversed(self._problems):
            problem.close()
        self._problems.clear()
        if self.handle:
            self._check(
                self.library.hipblasLtDestroy(ctypes.c_void_p(self.handle)),
                "hipblasLtDestroy",
            )
            self.handle = 0

    @staticmethod
    def _check(code: int, label: str) -> None:
        if int(code) != HIPBLAS_STATUS_SUCCESS:
            raise RuntimeError(f"{label} failed with hipBLAS status {int(code)}")

    def _configure(self) -> None:
        specs = {
            "hipblasLtCreate": ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "hipblasLtDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "hipblasLtMatrixLayoutCreate": (
                [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_int64,
                ],
                ctypes.c_int,
            ),
            "hipblasLtMatrixLayoutDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "hipblasLtMatrixLayoutSetAttribute": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
                ctypes.c_int,
            ),
            "hipblasLtMatmulDescCreate": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int],
                ctypes.c_int,
            ),
            "hipblasLtMatmulDescDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "hipblasLtMatmulDescSetAttribute": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
                ctypes.c_int,
            ),
            "hipblasLtMatmulPreferenceCreate": (
                [ctypes.POINTER(ctypes.c_void_p)],
                ctypes.c_int,
            ),
            "hipblasLtMatmulPreferenceDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "hipblasLtMatmulPreferenceSetAttribute": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
                ctypes.c_int,
            ),
            "hipblasLtMatmulAlgoGetHeuristic": (
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.POINTER(HipblasLtHeuristicResult),
                    ctypes.POINTER(ctypes.c_int),
                ],
                ctypes.c_int,
            ),
            "hipblasLtMatmul": (
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
                    ctypes.POINTER(HipblasLtAlgo),
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


class HipblasLtProblem:
    """Column-major descriptor view over row-major ``X @ weight.T`` storage."""

    def __init__(
        self,
        owner: HipblasLt,
        rows: int,
        in_features: int,
        out_features: int,
        workspace_nbytes: int,
    ) -> None:
        if min(rows, in_features, out_features) <= 0:
            raise ValueError("hipBLASLt dimensions must be positive")
        if workspace_nbytes < 0:
            raise ValueError("hipBLASLt workspace_nbytes must be non-negative")
        self.owner = owner
        self.rows = int(rows)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.workspace_nbytes = int(workspace_nbytes)
        self.matmul_desc = ctypes.c_void_p()
        self.preference = ctypes.c_void_p()
        self.layouts: list[ctypes.c_void_p] = []
        self._closed = False
        library = owner.library
        owner._check(
            library.hipblasLtMatmulDescCreate(
                ctypes.byref(self.matmul_desc),
                HIPBLAS_COMPUTE_32F,
                HIP_R_32F,
            ),
            "hipblasLtMatmulDescCreate",
        )
        transpose = ctypes.c_int(HIPBLAS_OP_T)
        owner._check(
            library.hipblasLtMatmulDescSetAttribute(
                self.matmul_desc,
                HIPBLASLT_MATMUL_DESC_TRANSA,
                ctypes.byref(transpose),
                ctypes.sizeof(transpose),
            ),
            "hipblasLtMatmulDescSetAttribute(TRANSA)",
        )
        self.a = self._layout(
            HIP_R_16F,
            self.in_features,
            self.out_features,
            self.in_features,
        )
        self.b = self._layout(
            HIP_R_16F,
            self.in_features,
            self.rows,
            self.in_features,
        )
        self.c = self._layout(
            HIP_R_32F,
            self.out_features,
            self.rows,
            self.out_features,
        )
        self.d = self._layout(
            HIP_R_32F,
            self.out_features,
            self.rows,
            self.out_features,
        )
        owner._check(
            library.hipblasLtMatmulPreferenceCreate(ctypes.byref(self.preference)),
            "hipblasLtMatmulPreferenceCreate",
        )
        maximum = ctypes.c_uint64(self.workspace_nbytes)
        owner._check(
            library.hipblasLtMatmulPreferenceSetAttribute(
                self.preference,
                HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ctypes.byref(maximum),
                ctypes.sizeof(maximum),
            ),
            "hipblasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE)",
        )

    def algorithms(self, maximum: int = 16) -> tuple[HipblasLtHeuristicResult, ...]:
        if maximum <= 0:
            raise ValueError("maximum hipBLASLt algorithms must be positive")
        results = (HipblasLtHeuristicResult * int(maximum))()
        count = ctypes.c_int()
        library = self.owner.library
        self.owner._check(
            library.hipblasLtMatmulAlgoGetHeuristic(
                ctypes.c_void_p(self.owner.handle),
                self.matmul_desc,
                self.a,
                self.b,
                self.c,
                self.d,
                self.preference,
                int(maximum),
                results,
                ctypes.byref(count),
            ),
            "hipblasLtMatmulAlgoGetHeuristic",
        )
        usable = tuple(results[index] for index in range(int(count.value)))
        if not usable:
            raise RuntimeError(
                f"hipBLASLt returned no algorithms for M={self.rows} "
                f"K={self.in_features} N={self.out_features}"
            )
        return usable

    def algorithm(self, preferred_index: int = 4) -> HipblasLtHeuristicResult:
        """Select the measured gfx1151 fast heuristic, clamped to availability."""

        algorithms = self.algorithms()
        return algorithms[min(max(int(preferred_index), 0), len(algorithms) - 1)]

    def launch(
        self,
        algorithm: HipblasLtHeuristicResult,
        x_ptr: int,
        weight_ptr: int,
        out_ptr: int,
        workspace_ptr: int = 0,
        *,
        stream: int = 0,
    ) -> None:
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        code = self.owner.library.hipblasLtMatmul(
            ctypes.c_void_p(self.owner.handle),
            self.matmul_desc,
            ctypes.byref(alpha),
            ctypes.c_void_p(weight_ptr),
            self.a,
            ctypes.c_void_p(x_ptr),
            self.b,
            ctypes.byref(beta),
            ctypes.c_void_p(out_ptr),
            self.c,
            ctypes.c_void_p(out_ptr),
            self.d,
            ctypes.byref(algorithm.algo),
            ctypes.c_void_p(workspace_ptr),
            int(algorithm.workspace_size),
            ctypes.c_void_p(stream),
        )
        self.owner._check(code, "hipblasLtMatmul")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        library = self.owner.library
        for layout in reversed(self.layouts):
            self.owner._check(
                library.hipblasLtMatrixLayoutDestroy(layout),
                "hipblasLtMatrixLayoutDestroy",
            )
        self.layouts.clear()
        if self.preference.value:
            self.owner._check(
                library.hipblasLtMatmulPreferenceDestroy(self.preference),
                "hipblasLtMatmulPreferenceDestroy",
            )
            self.preference = ctypes.c_void_p()
        if self.matmul_desc.value:
            self.owner._check(
                library.hipblasLtMatmulDescDestroy(self.matmul_desc),
                "hipblasLtMatmulDescDestroy",
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
            library.hipblasLtMatrixLayoutCreate(
                ctypes.byref(layout),
                dtype,
                rows,
                cols,
                leading,
            ),
            "hipblasLtMatrixLayoutCreate",
        )
        order = ctypes.c_int(HIPBLASLT_ORDER_COL)
        self.owner._check(
            library.hipblasLtMatrixLayoutSetAttribute(
                layout,
                HIPBLASLT_MATRIX_LAYOUT_ORDER,
                ctypes.byref(order),
                ctypes.sizeof(order),
            ),
            "hipblasLtMatrixLayoutSetAttribute(ORDER)",
        )
        self.layouts.append(layout)
        return layout


__all__ = [
    "HipblasLt",
    "HipblasLtAlgo",
    "HipblasLtHeuristicResult",
    "HipblasLtProblem",
]
