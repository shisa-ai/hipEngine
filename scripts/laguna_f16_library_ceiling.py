#!/usr/bin/env python3
"""Measure rocBLAS/hipBLASLt ceilings for Laguna source-F16 projections."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.hip_gfx1100.convert import (
    bf16_to_f32,
    build_cast,
    f32_to_bf16,
    f32_to_fp16,
)
from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
    build_laguna_f16_projection_prefill,
    laguna_f16w_tiled_bf16_bf16_out,
    laguna_f16w_tiled_bf16_f32_out,
)
from scripts.laguna_target_ar_bench import _compiler_version, _repo_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = (16, 32, 64, 128, 256, 512)
DEFAULT_OUTPUT = (
    ROOT / "benchmarks/results/2026-07-23-gfx1151-laguna-f16-library-ceiling.json"
)
_MODE_ORDER = ("exact", "rocblas_core", "rocblas_inclusive", "hipblaslt_core", "hipblaslt_inclusive")
_FAMILIES = ("full", "swa")
_SHAPES = {
    "full_q": (3072, 6144),
    "swa_q": (3072, 9216),
    "kv": (3072, 1024),
    "full_gate": (3072, 48),
    "swa_gate": (3072, 72),
    "full_o": (6144, 3072),
    "swa_o": (9216, 3072),
}

# hipBLASLt ABI constants from ROCm 7.15 headers.
_HIP_R_32F = 0
_HIP_R_16F = 2
_HIPBLAS_COMPUTE_32F = 2
_HIPBLAS_OP_T = 112
_HIPBLASLT_ORDER_COL = 0
_HIPBLASLT_MATRIX_LAYOUT_ORDER = 3
_HIPBLASLT_MATMUL_DESC_TRANSA = 0
_HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
_HIPBLAS_STATUS_SUCCESS = 0


class _HipblasLtAlgo(ctypes.Structure):
    _fields_ = (("data", ctypes.c_uint8 * 16), ("max_workspace_bytes", ctypes.c_size_t))


class _HipblasLtHeuristicResult(ctypes.Structure):
    _fields_ = (
        ("algo", _HipblasLtAlgo),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    )


@dataclass(frozen=True)
class _TimedAlgorithm:
    index: int
    median_ms: float
    samples_ms: tuple[float, ...]
    workspace_nbytes: int
    waves_count: float
    algorithm_id: tuple[int, ...]


class _HipblasLt:
    """Lazy, benchmark-only hipBLASLt ctypes surface."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.library = ctypes.CDLL(path)
        self._configure()
        handle = ctypes.c_void_p()
        self._check(self.library.hipblasLtCreate(ctypes.byref(handle)), "hipblasLtCreate")
        self.handle = int(handle.value or 0)

    def close(self) -> None:
        if self.handle:
            self._check(
                self.library.hipblasLtDestroy(ctypes.c_void_p(self.handle)),
                "hipblasLtDestroy",
            )
            self.handle = 0

    def problem(self, rows: int, in_features: int, out_features: int, workspace_nbytes: int):
        return _HipblasLtProblem(self, rows, in_features, out_features, workspace_nbytes)

    @staticmethod
    def _check(code: int, label: str) -> None:
        if int(code) != _HIPBLAS_STATUS_SUCCESS:
            raise RuntimeError(f"{label} failed with hipBLAS status {int(code)}")

    def _configure(self) -> None:
        specs = {
            "hipblasLtCreate": ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "hipblasLtDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "hipblasLtMatrixLayoutCreate": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64],
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
            "hipblasLtMatmulPreferenceCreate": ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
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
                    ctypes.POINTER(_HipblasLtHeuristicResult),
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
                    ctypes.POINTER(_HipblasLtAlgo),
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_void_p,
                ],
                ctypes.c_int,
            ),
        }
        for name, (argtypes, restype) in specs.items():
            fn = getattr(self.library, name)
            fn.argtypes = argtypes
            fn.restype = restype


class _HipblasLtProblem:
    """Column-major descriptor view over row-major X and W storage."""

    def __init__(
        self,
        owner: _HipblasLt,
        rows: int,
        in_features: int,
        out_features: int,
        workspace_nbytes: int,
    ) -> None:
        if min(rows, in_features, out_features) <= 0:
            raise ValueError("hipBLASLt dimensions must be positive")
        self.owner = owner
        self.rows = rows
        self.in_features = in_features
        self.out_features = out_features
        self.workspace_nbytes = workspace_nbytes
        self.matmul_desc = ctypes.c_void_p()
        self.preference = ctypes.c_void_p()
        self.layouts: list[ctypes.c_void_p] = []
        lib = owner.library
        owner._check(
            lib.hipblasLtMatmulDescCreate(
                ctypes.byref(self.matmul_desc), _HIPBLAS_COMPUTE_32F, _HIP_R_32F
            ),
            "hipblasLtMatmulDescCreate",
        )
        transpose = ctypes.c_int(_HIPBLAS_OP_T)
        owner._check(
            lib.hipblasLtMatmulDescSetAttribute(
                self.matmul_desc,
                _HIPBLASLT_MATMUL_DESC_TRANSA,
                ctypes.byref(transpose),
                ctypes.sizeof(transpose),
            ),
            "hipblasLtMatmulDescSetAttribute(TRANSA)",
        )
        # Weight row-major [N,K] is column-major [K,N] and is transposed.
        # X row-major [M,K] is column-major [K,M]. D column-major [N,M]
        # aliases row-major [M,N].
        self.a = self._layout(_HIP_R_16F, in_features, out_features, in_features)
        self.b = self._layout(_HIP_R_16F, in_features, rows, in_features)
        self.c = self._layout(_HIP_R_32F, out_features, rows, out_features)
        self.d = self._layout(_HIP_R_32F, out_features, rows, out_features)
        owner._check(
            lib.hipblasLtMatmulPreferenceCreate(ctypes.byref(self.preference)),
            "hipblasLtMatmulPreferenceCreate",
        )
        maximum = ctypes.c_uint64(workspace_nbytes)
        owner._check(
            lib.hipblasLtMatmulPreferenceSetAttribute(
                self.preference,
                _HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ctypes.byref(maximum),
                ctypes.sizeof(maximum),
            ),
            "hipblasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE)",
        )

    def algorithms(self, maximum: int) -> tuple[_HipblasLtHeuristicResult, ...]:
        if maximum <= 0:
            raise ValueError("maximum hipBLASLt algorithms must be positive")
        results = (_HipblasLtHeuristicResult * maximum)()
        count = ctypes.c_int()
        lib = self.owner.library
        self.owner._check(
            lib.hipblasLtMatmulAlgoGetHeuristic(
                ctypes.c_void_p(self.owner.handle),
                self.matmul_desc,
                self.a,
                self.b,
                self.c,
                self.d,
                self.preference,
                maximum,
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

    def launch(
        self,
        result: _HipblasLtHeuristicResult,
        x_ptr: int,
        weight_ptr: int,
        out_ptr: int,
        workspace_ptr: int,
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
            ctypes.byref(result.algo),
            ctypes.c_void_p(workspace_ptr),
            int(result.workspace_size),
            ctypes.c_void_p(stream),
        )
        self.owner._check(code, "hipblasLtMatmul")

    def close(self) -> None:
        lib = self.owner.library
        for layout in reversed(self.layouts):
            self.owner._check(
                lib.hipblasLtMatrixLayoutDestroy(layout),
                "hipblasLtMatrixLayoutDestroy",
            )
        self.layouts.clear()
        if self.preference.value:
            self.owner._check(
                lib.hipblasLtMatmulPreferenceDestroy(self.preference),
                "hipblasLtMatmulPreferenceDestroy",
            )
            self.preference = ctypes.c_void_p()
        if self.matmul_desc.value:
            self.owner._check(
                lib.hipblasLtMatmulDescDestroy(self.matmul_desc),
                "hipblasLtMatmulDescDestroy",
            )
            self.matmul_desc = ctypes.c_void_p()

    def _layout(self, dtype: int, rows: int, cols: int, leading: int) -> ctypes.c_void_p:
        layout = ctypes.c_void_p()
        lib = self.owner.library
        self.owner._check(
            lib.hipblasLtMatrixLayoutCreate(
                ctypes.byref(layout), dtype, rows, cols, leading
            ),
            "hipblasLtMatrixLayoutCreate",
        )
        order = ctypes.c_int(_HIPBLASLT_ORDER_COL)
        self.owner._check(
            lib.hipblasLtMatrixLayoutSetAttribute(
                layout,
                _HIPBLASLT_MATRIX_LAYOUT_ORDER,
                ctypes.byref(order),
                ctypes.sizeof(order),
            ),
            "hipblasLtMatrixLayoutSetAttribute(ORDER)",
        )
        self.layouts.append(layout)
        return layout


@dataclass
class _Buffers:
    x_bf16: DeviceBuffer
    x_f32: DeviceBuffer
    x_fp16: DeviceBuffer
    weight_fp16: DeviceBuffer
    out_f32: DeviceBuffer
    out_bf16: DeviceBuffer
    workspace: DeviceBuffer

    def all(self) -> tuple[DeviceBuffer, ...]:
        return (
            self.x_bf16,
            self.x_f32,
            self.x_fp16,
            self.weight_fp16,
            self.out_f32,
            self.out_bf16,
            self.workspace,
        )


def _parse_rows(value: str) -> tuple[int, ...]:
    try:
        rows = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rows must be comma-separated integers") from exc
    if not rows or any(row <= 0 for row in rows):
        raise argparse.ArgumentTypeError("rows must be distinct positive integers")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--rows", type=_parse_rows, default=DEFAULT_ROWS)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--algorithm-repetitions", type=int, default=5)
    parser.add_argument("--algorithm-warmups", type=int, default=2)
    parser.add_argument("--max-algorithms", type=int, default=16)
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--rocblas-library", default="librocblas.so")
    parser.add_argument("--hipblaslt-library", default="libhipblaslt.so")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _event_sample(runtime: HipRuntime, launch: Callable[[], None], iterations: int) -> tuple[float, float]:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        wall_start = time.perf_counter()
        for _ in range(iterations):
            launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        wall_ms = (time.perf_counter() - wall_start) * 1.0e3 / iterations
        gpu_ms = runtime.event_elapsed_time_ms(start, stop) / iterations
        return gpu_ms, wall_ms
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def _time_algorithm(
    runtime: HipRuntime,
    launch: Callable[[], None],
    *,
    warmups: int,
    repetitions: int,
) -> tuple[float, ...]:
    for _ in range(warmups):
        launch()
    runtime.device_synchronize()
    return tuple(_event_sample(runtime, launch, 1)[0] for _ in range(repetitions))


def _select_algorithm(
    problem: _HipblasLtProblem,
    runtime: HipRuntime,
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    workspace_ptr: int,
    *,
    maximum: int,
    warmups: int,
    repetitions: int,
) -> tuple[_HipblasLtHeuristicResult, tuple[_TimedAlgorithm, ...]]:
    timed: list[_TimedAlgorithm] = []
    results = problem.algorithms(maximum)
    for index, result in enumerate(results):
        samples = _time_algorithm(
            runtime,
            lambda result=result: problem.launch(
                result, x_ptr, weight_ptr, out_ptr, workspace_ptr
            ),
            warmups=warmups,
            repetitions=repetitions,
        )
        timed.append(
            _TimedAlgorithm(
                index=index,
                median_ms=statistics.median(samples),
                samples_ms=samples,
                workspace_nbytes=int(result.workspace_size),
                waves_count=float(result.waves_count),
                algorithm_id=tuple(int(value) for value in result.algo.data),
            )
        )
    best = min(timed, key=lambda item: item.median_ms)
    return results[best.index], tuple(timed)


def _offset(base: int, rows: int, columns: int) -> int:
    return base + rows * columns * np.dtype(np.float32).itemsize


def _family_launches(
    family: str,
    rows: int,
    runtime: HipRuntime,
    buffers: _Buffers,
    exact_library,
    cast_library,
    rocblas: Rocblas,
    problems: Mapping[str, _HipblasLtProblem],
    selected_algorithms: Mapping[str, _HipblasLtHeuristicResult],
) -> dict[str, Callable[[], None]]:
    q_name = f"{family}_q"
    gate_name = f"{family}_gate"
    o_name = f"{family}_o"
    q_width = _SHAPES[q_name][1]
    o_k = _SHAPES[o_name][0]
    k_width = _SHAPES["kv"][1]
    q_out = buffers.out_f32.ptr
    k_out = _offset(q_out, rows, q_width)
    v_out = _offset(k_out, rows, k_width)
    gate_out = _offset(v_out, rows, k_width)

    def exact() -> None:
        for name, out_ptr in (
            (q_name, q_out),
            ("kv", k_out),
            ("kv", v_out),
            (gate_name, gate_out),
        ):
            k, n = _SHAPES[name]
            laguna_f16w_tiled_bf16_f32_out(
                buffers.x_bf16.ptr,
                buffers.weight_fp16.ptr,
                out_ptr,
                rows,
                k,
                n,
                library=exact_library,
                runtime=runtime,
            )
        laguna_f16w_tiled_bf16_bf16_out(
            buffers.x_bf16.ptr,
            buffers.weight_fp16.ptr,
            buffers.out_bf16.ptr,
            rows,
            o_k,
            3072,
            library=exact_library,
            runtime=runtime,
        )

    def cast_input(width: int) -> None:
        count = rows * width
        bf16_to_f32(
            buffers.x_bf16.ptr,
            buffers.x_f32.ptr,
            count,
            library=cast_library,
            runtime=runtime,
        )
        f32_to_fp16(
            buffers.x_f32.ptr,
            buffers.x_fp16.ptr,
            count,
            library=cast_library,
            runtime=runtime,
        )

    def rocblas_core() -> None:
        for name, out_ptr in (
            (q_name, q_out),
            ("kv", k_out),
            ("kv", v_out),
            (gate_name, gate_out),
        ):
            k, n = _SHAPES[name]
            rocblas.gemm_ex_rowmajor_nt_fp16_f32_out(
                buffers.x_fp16.ptr,
                buffers.weight_fp16.ptr,
                out_ptr,
                rows=rows,
                in_features=k,
                out_features=n,
            )
        rocblas.gemm_ex_rowmajor_nt_fp16_f32_out(
            buffers.x_fp16.ptr,
            buffers.weight_fp16.ptr,
            q_out,
            rows=rows,
            in_features=o_k,
            out_features=3072,
        )
        f32_to_bf16(
            q_out,
            buffers.out_bf16.ptr,
            rows * 3072,
            library=cast_library,
            runtime=runtime,
        )

    def rocblas_inclusive() -> None:
        cast_input(3072)
        for name, out_ptr in (
            (q_name, q_out),
            ("kv", k_out),
            ("kv", v_out),
            (gate_name, gate_out),
        ):
            k, n = _SHAPES[name]
            rocblas.gemm_ex_rowmajor_nt_fp16_f32_out(
                buffers.x_fp16.ptr,
                buffers.weight_fp16.ptr,
                out_ptr,
                rows=rows,
                in_features=k,
                out_features=n,
            )
        cast_input(o_k)
        rocblas.gemm_ex_rowmajor_nt_fp16_f32_out(
            buffers.x_fp16.ptr,
            buffers.weight_fp16.ptr,
            q_out,
            rows=rows,
            in_features=o_k,
            out_features=3072,
        )
        f32_to_bf16(
            q_out,
            buffers.out_bf16.ptr,
            rows * 3072,
            library=cast_library,
            runtime=runtime,
        )

    def hipblaslt_core() -> None:
        for name, out_ptr in (
            (q_name, q_out),
            ("kv", k_out),
            ("kv", v_out),
            (gate_name, gate_out),
            (o_name, q_out),
        ):
            problems[name].launch(
                selected_algorithms[name],
                buffers.x_fp16.ptr,
                buffers.weight_fp16.ptr,
                out_ptr,
                buffers.workspace.ptr,
            )
        f32_to_bf16(
            q_out,
            buffers.out_bf16.ptr,
            rows * 3072,
            library=cast_library,
            runtime=runtime,
        )

    def hipblaslt_inclusive() -> None:
        cast_input(3072)
        for name, out_ptr in (
            (q_name, q_out),
            ("kv", k_out),
            ("kv", v_out),
            (gate_name, gate_out),
        ):
            problems[name].launch(
                selected_algorithms[name],
                buffers.x_fp16.ptr,
                buffers.weight_fp16.ptr,
                out_ptr,
                buffers.workspace.ptr,
            )
        cast_input(o_k)
        problems[o_name].launch(
            selected_algorithms[o_name],
            buffers.x_fp16.ptr,
            buffers.weight_fp16.ptr,
            q_out,
            buffers.workspace.ptr,
        )
        f32_to_bf16(
            q_out,
            buffers.out_bf16.ptr,
            rows * 3072,
            library=cast_library,
            runtime=runtime,
        )

    return {
        "exact": exact,
        "rocblas_core": rocblas_core,
        "rocblas_inclusive": rocblas_inclusive,
        "hipblaslt_core": hipblaslt_core,
        "hipblaslt_inclusive": hipblaslt_inclusive,
    }


def _summarize(
    rows: Sequence[int],
    samples: Mapping[int, Mapping[str, Mapping[str, Sequence[float]]]],
    wall_samples: Mapping[int, Mapping[str, Mapping[str, Sequence[float]]]],
) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    failed: list[str] = []
    for row in rows:
        families: dict[str, Any] = {}
        for family in _FAMILIES:
            exact = statistics.median(samples[row][family]["exact"])
            modes: dict[str, Any] = {}
            for mode in _MODE_ORDER:
                gpu = tuple(float(value) for value in samples[row][family][mode])
                wall = tuple(float(value) for value in wall_samples[row][family][mode])
                if not gpu or len(gpu) != len(wall):
                    raise ValueError(f"rows={row}/{family}/{mode} requires equal non-empty samples")
                median_gpu = statistics.median(gpu)
                modes[mode] = {
                    "gpu_ms_samples": list(gpu),
                    "gpu_ms_median": median_gpu,
                    "wall_ms_samples": list(wall),
                    "wall_ms_median": statistics.median(wall),
                    "speedup_vs_exact": exact / median_gpu,
                }
            if modes["hipblaslt_inclusive"]["speedup_vs_exact"] <= 1.0:
                failed.append(f"rows_{row}_{family}_hipblaslt_inclusive_not_faster")
            families[family] = modes
        shapes[str(row)] = {"rows": row, "families": families}
    return {
        "pass": not failed,
        "failed_checks": failed,
        "shapes": shapes,
        "policy": "diagnostic is complete and inclusive hipBLASLt is faster at every retained row/family shape",
    }


def _math_smoke(
    runtime: HipRuntime,
    rocblas: Rocblas,
    hipblaslt: _HipblasLt,
    buffers: _Buffers,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.1, size=(2, 16)).astype(np.float16)
    weight = rng.normal(0.0, 0.1, size=(3, 16)).astype(np.float16)
    expected = x.astype(np.float32) @ weight.astype(np.float32).T
    copy_host_to_device(buffers.x_fp16, host_array_ptr(x), x.nbytes, runtime=runtime)
    copy_host_to_device(
        buffers.weight_fp16, host_array_ptr(weight), weight.nbytes, runtime=runtime
    )
    problem = hipblaslt.problem(2, 16, 3, buffers.workspace.nbytes)
    try:
        algorithm = problem.algorithms(16)[0]
        problem.launch(
            algorithm,
            buffers.x_fp16.ptr,
            buffers.weight_fp16.ptr,
            buffers.out_f32.ptr,
            buffers.workspace.ptr,
        )
        runtime.device_synchronize()
        lt = np.empty((2, 3), dtype=np.float32)
        copy_device_to_host(host_array_ptr(lt), buffers.out_f32, lt.nbytes, runtime=runtime)
        rocblas.gemm_ex_rowmajor_nt_fp16_f32_out(
            buffers.x_fp16.ptr,
            buffers.weight_fp16.ptr,
            buffers.out_f32.ptr,
            rows=2,
            in_features=16,
            out_features=3,
        )
        runtime.device_synchronize()
        rb = np.empty((2, 3), dtype=np.float32)
        copy_device_to_host(host_array_ptr(rb), buffers.out_f32, rb.nbytes, runtime=runtime)
    finally:
        problem.close()
    lt_error = float(np.max(np.abs(lt - expected)))
    rb_error = float(np.max(np.abs(rb - expected)))
    passed = bool(
        np.allclose(lt, expected, rtol=1.0e-4, atol=1.0e-5)
        and np.allclose(rb, expected, rtol=1.0e-4, atol=1.0e-5)
    )
    return {
        "pass": passed,
        "shape": [2, 16, 3],
        "hipblaslt_max_abs_error": lt_error,
        "rocblas_max_abs_error": rb_error,
    }


def _allocate_buffers(runtime: HipRuntime, rows: Sequence[int], workspace_nbytes: int) -> _Buffers:
    maximum_rows = max(rows)
    maximum_input = maximum_rows * max(k for k, _ in _SHAPES.values())
    maximum_weight = max(k * n for k, n in _SHAPES.values())
    maximum_outputs = maximum_rows * (9216 + 1024 + 1024 + 72)

    def allocate(nbytes: int) -> DeviceBuffer:
        buffer = malloc(nbytes, runtime=runtime)
        runtime.memset(buffer.ptr, 0, nbytes)
        return buffer

    return _Buffers(
        x_bf16=allocate(maximum_input * 2),
        x_f32=allocate(maximum_input * 4),
        x_fp16=allocate(maximum_input * 2),
        weight_fp16=allocate(maximum_weight * 2),
        out_f32=allocate(maximum_outputs * 4),
        out_bf16=allocate(maximum_rows * 3072 * 2),
        workspace=allocate(workspace_nbytes),
    )


def _algorithm_payload(timed: Sequence[_TimedAlgorithm]) -> list[dict[str, Any]]:
    return [
        {
            "heuristic_index": item.index,
            "median_ms": item.median_ms,
            "samples_ms": list(item.samples_ms),
            "workspace_nbytes": item.workspace_nbytes,
            "waves_count": item.waves_count,
            "algorithm_id": list(item.algorithm_id),
        }
        for item in timed
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = tuple(int(value) for value in args.rows)
    if rows != DEFAULT_ROWS:
        raise ValueError(f"retained Laguna F16 ceiling requires rows {DEFAULT_ROWS}")
    if args.backend != "hip_gfx1151":
        raise ValueError("retained Laguna F16 ceiling is qualified only for hip_gfx1151")
    if args.iterations <= 0 or args.repetitions < 3 or args.warmups < 0:
        raise ValueError("iterations must be positive, repetitions >=3, and warmups non-negative")
    if args.algorithm_repetitions < 3 or args.algorithm_warmups < 0:
        raise ValueError("algorithm repetitions must be >=3 and warmups non-negative")
    if args.max_algorithms <= 0 or args.workspace_mib <= 0:
        raise ValueError("max algorithms and workspace MiB must be positive")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna F16 ceiling requires a clean tracked worktree")

    compiler_version = _compiler_version(args.compiler_version_file)
    exact_library = build_laguna_f16_projection_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    cast_library = build_cast(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    workspace_nbytes = args.workspace_mib << 20
    rocblas = Rocblas.load(args.rocblas_library)
    hipblaslt = _HipblasLt(args.hipblaslt_library)
    buffers = _allocate_buffers(runtime, rows, workspace_nbytes)
    samples = {
        row: {family: {mode: [] for mode in _MODE_ORDER} for family in _FAMILIES}
        for row in rows
    }
    wall_samples = {
        row: {family: {mode: [] for mode in _MODE_ORDER} for family in _FAMILIES}
        for row in rows
    }
    algorithm_screens: dict[str, Any] = {}
    smoke: dict[str, Any] = {"pass": False}
    try:
        smoke = _math_smoke(runtime, rocblas, hipblaslt, buffers, args.seed)
        if not smoke["pass"]:
            raise RuntimeError("rocBLAS/hipBLASLt math smoke failed")
        for buffer in buffers.all():
            runtime.memset(buffer.ptr, 0, buffer.nbytes)

        for row in rows:
            problems: dict[str, _HipblasLtProblem] = {}
            selected: dict[str, _HipblasLtHeuristicResult] = {}
            algorithm_screens[str(row)] = {}
            try:
                for name, (in_features, out_features) in _SHAPES.items():
                    problem = hipblaslt.problem(
                        row, in_features, out_features, workspace_nbytes
                    )
                    problems[name] = problem
                    best, timed = _select_algorithm(
                        problem,
                        runtime,
                        buffers.x_fp16.ptr,
                        buffers.weight_fp16.ptr,
                        buffers.out_f32.ptr,
                        buffers.workspace.ptr,
                        maximum=args.max_algorithms,
                        warmups=args.algorithm_warmups,
                        repetitions=args.algorithm_repetitions,
                    )
                    selected[name] = best
                    payload = _algorithm_payload(timed)
                    algorithm_screens[str(row)][name] = {
                        "shape_mkn": [row, in_features, out_features],
                        "heuristic_count": len(payload),
                        "selected_heuristic_index": min(
                            payload, key=lambda item: item["median_ms"]
                        )["heuristic_index"],
                        "algorithms": payload,
                    }

                for family in _FAMILIES:
                    launches = _family_launches(
                        family,
                        row,
                        runtime,
                        buffers,
                        exact_library,
                        cast_library,
                        rocblas,
                        problems,
                        selected,
                    )
                    for _ in range(args.warmups):
                        for mode in _MODE_ORDER:
                            launches[mode]()
                    runtime.device_synchronize()
                    for repetition in range(args.repetitions):
                        shift = repetition % len(_MODE_ORDER)
                        order = (*_MODE_ORDER[shift:], *_MODE_ORDER[:shift])
                        if (repetition // len(_MODE_ORDER)) % 2:
                            order = tuple(reversed(order))
                        for mode in order:
                            gpu_ms, wall_ms = _event_sample(
                                runtime, launches[mode], args.iterations
                            )
                            samples[row][family][mode].append(gpu_ms)
                            wall_samples[row][family][mode].append(wall_ms)
            finally:
                for problem in reversed(tuple(problems.values())):
                    problem.close()
    finally:
        runtime.device_synchronize()
        for buffer in reversed(buffers.all()):
            free(buffer, runtime=runtime)
        hipblaslt.close()
        rocblas.close()

    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during Laguna F16 ceiling")
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    summary = _summarize(rows, samples, wall_samples)
    if not smoke["pass"]:
        summary["failed_checks"].append("library_math_smoke_failed")
    if not recovered:
        summary["failed_checks"].append("tracked_ownership_not_recovered")
    summary["pass"] = not summary["failed_checks"]
    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch="gfx1151",
        quant="source_f16_projection_geometry",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_f16_library_ceiling",
        timing_protocol="hip_event_counterbalanced_family_sequences_and_algorithm_sweep",
        warmups=args.warmups,
        repetitions=args.repetitions,
        hipcc_version=compiler_version,
    )
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_f16_library_ceiling",
        "status": "diagnostic_passed" if summary["pass"] else "diagnostic_failed",
        "pass": bool(summary["pass"]),
        "performance_claim": False,
        "performance_claim_scope": (
            "synthetic zero-data production geometry; establishes a library ceiling only, "
            "not a runtime candidate or model throughput claim"
        ),
        "provenance": provenance,
        "repo": repo,
        "platform": {
            "backend": args.backend,
            "target_arch": "gfx1151",
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
            "rocblas_library": args.rocblas_library,
            "hipblaslt_library": args.hipblaslt_library,
        },
        "protocol": {
            "rows": list(rows),
            "projection_shapes_kn": {
                name: list(shape) for name, shape in _SHAPES.items()
            },
            "families": {
                "full": ["full_q", "kv", "kv", "full_gate", "full_o"],
                "swa": ["swa_q", "kv", "kv", "swa_gate", "swa_o"],
            },
            "modes": {
                "exact": "retained BF16-input/F16-weight reduction-order-preserving tiled kernels",
                "rocblas_core": "FP16-input/F16-weight/FP32-output rocblas_gemm_ex plus O BF16 cast",
                "rocblas_inclusive": "rocblas_core plus conservative BF16->F32->FP16 input casts before QKV/gate and O",
                "hipblaslt_core": "best screened FP16-input/F16-weight/FP32-output hipBLASLt algorithms plus O BF16 cast",
                "hipblaslt_inclusive": "hipblaslt_core plus conservative BF16->F32->FP16 input casts before QKV/gate and O",
            },
            "iterations_per_sample": args.iterations,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "algorithm_repetitions": args.algorithm_repetitions,
            "algorithm_warmups": args.algorithm_warmups,
            "max_algorithms": args.max_algorithms,
            "workspace_nbytes": workspace_nbytes,
            "timed_order": "rotated and reversed mode order by repetition",
            "data": "zero BF16/FP16 buffers; GEMM control mapping independently checked with seeded nonzero 2x16x3 math smoke",
        },
        "summary": summary,
        "hipblaslt_algorithm_screens": algorithm_screens,
        "correctness": {
            "pass": smoke["pass"] and recovered,
            "library_math_smoke": smoke,
            "tracked_returned_to_baseline": recovered,
            "model_quality_gate_required_for_runtime_candidate": True,
        },
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
            "workspace_nbytes": workspace_nbytes,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "Core rows start from preconverted FP16 and are the matrix-library ceiling.",
            "Inclusive rows conservatively use two existing cast kernels per input because the runtime has no direct BF16->FP16 primitive; a custom WMMA body may convert operands in registers.",
            "All library outputs accumulate/store FP32; O then uses the existing FP32->BF16 boundary. Q/K/V/gate retain FP32 output.",
            "hipBLASLt descriptors are benchmark-local and lazy; neither library becomes a hipEngine runtime dependency.",
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
