"""Bounded dense-initial hipBLASLt attention candidate for Laguna gfx1151."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.hipblaslt import (
    HIPBLASLT_MATRIX_LAYOUT_ORDER,
    HIPBLASLT_MATMUL_DESC_TRANSA,
    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    HIPBLASLT_ORDER_COL,
    HIPBLAS_COMPUTE_32F,
    HIPBLAS_OP_T,
    HIP_R_32F,
    HipblasLt,
    HipblasLtHeuristicResult,
)
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
    laguna_dense_initial_cache_bf16_to_f32_spans,
    laguna_dense_initial_causal_softmax_f32_spans,
    laguna_dense_initial_query_head_transpose_f32,
)
from hipengine.kvcache import KVLiveSpans

_HIPBLAS_OP_N = 111
_MATRIX_LAYOUT_BATCH_COUNT = 0
_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 1
_ROWS = 128
_MAX_CONTEXT = 512
_MAX_Q_HEADS = 72
_KV_HEADS = 8
_HEAD_DIM = 128

# Best zero-workspace heuristic indices from the bounded gfx1151 F32
# contraction screen. Keys are (query heads, context, QK/PV).
_ALGORITHM_INDEX = {
    (48, 128, "qk"): 0,
    (48, 128, "pv"): 0,
    (48, 256, "qk"): 23,
    (48, 256, "pv"): 26,
    (48, 384, "qk"): 28,
    (48, 384, "pv"): 26,
    (48, 512, "qk"): 18,
    (48, 512, "pv"): 26,
    (72, 128, "qk"): 0,
    (72, 128, "pv"): 25,
    (72, 256, "qk"): 0,
    (72, 256, "pv"): 21,
    (72, 384, "qk"): 26,
    (72, 384, "pv"): 9,
    (72, 512, "qk"): 14,
    (72, 512, "pv"): 21,
}

_PACKED_ALGORITHM_INDEX = {
    (48, 128, "qk"): 0,
    (48, 128, "pv"): 5,
    (48, 256, "qk"): 0,
    (48, 256, "pv"): 0,
    (48, 384, "qk"): 31,
    (48, 384, "pv"): 0,
    (48, 512, "qk"): 1,
    (48, 512, "pv"): 22,
    (72, 128, "qk"): 0,
    (72, 128, "pv"): 3,
    (72, 256, "qk"): 0,
    (72, 256, "pv"): 3,
    (72, 384, "qk"): 30,
    (72, 384, "pv"): 18,
    (72, 512, "qk"): 1,
    (72, 512, "pv"): 3,
}


@dataclass(frozen=True)
class _Layout:
    rows: int
    cols: int
    leading: int
    batch_count: int
    batch_stride: int


class _BatchedProblem:
    """One zero-workspace strided-batch descriptor with noncompact strides."""

    def __init__(
        self,
        owner: HipblasLt,
        *,
        transa: int,
        layouts: tuple[_Layout, _Layout, _Layout, _Layout],
        preferred_index: int,
    ) -> None:
        self.owner = owner
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
        transpose = ctypes.c_int(int(transa))
        owner._check(
            library.hipblasLtMatmulDescSetAttribute(
                self.matmul_desc,
                HIPBLASLT_MATMUL_DESC_TRANSA,
                ctypes.byref(transpose),
                ctypes.sizeof(transpose),
            ),
            "hipblasLtMatmulDescSetAttribute(TRANSA)",
        )
        for spec in layouts:
            self.layouts.append(self._make_layout(spec))
        owner._check(
            library.hipblasLtMatmulPreferenceCreate(
                ctypes.byref(self.preference)
            ),
            "hipblasLtMatmulPreferenceCreate",
        )
        maximum = ctypes.c_uint64(0)
        owner._check(
            library.hipblasLtMatmulPreferenceSetAttribute(
                self.preference,
                HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ctypes.byref(maximum),
                ctypes.sizeof(maximum),
            ),
            "hipblasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE)",
        )
        algorithms = self._algorithms()
        index = min(max(int(preferred_index), 0), len(algorithms) - 1)
        self.algorithm = algorithms[index]
        if int(self.algorithm.workspace_size) != 0:
            raise RuntimeError(
                "Laguna attention hipBLASLt requires zero workspace"
            )

    def _make_layout(self, spec: _Layout) -> ctypes.c_void_p:
        layout = ctypes.c_void_p()
        library = self.owner.library
        self.owner._check(
            library.hipblasLtMatrixLayoutCreate(
                ctypes.byref(layout),
                HIP_R_32F,
                int(spec.rows),
                int(spec.cols),
                int(spec.leading),
            ),
            "hipblasLtMatrixLayoutCreate",
        )
        order = ctypes.c_int(HIPBLASLT_ORDER_COL)
        batch_count = ctypes.c_int(int(spec.batch_count))
        batch_stride = ctypes.c_int64(int(spec.batch_stride))
        for attribute, value in (
            (HIPBLASLT_MATRIX_LAYOUT_ORDER, order),
            (_MATRIX_LAYOUT_BATCH_COUNT, batch_count),
            (_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, batch_stride),
        ):
            self.owner._check(
                library.hipblasLtMatrixLayoutSetAttribute(
                    layout,
                    attribute,
                    ctypes.byref(value),
                    ctypes.sizeof(value),
                ),
                "hipblasLtMatrixLayoutSetAttribute",
            )
        return layout

    def _algorithms(self) -> tuple[HipblasLtHeuristicResult, ...]:
        results = (HipblasLtHeuristicResult * 32)()
        count = ctypes.c_int()
        library = self.owner.library
        self.owner._check(
            library.hipblasLtMatmulAlgoGetHeuristic(
                ctypes.c_void_p(self.owner.handle),
                self.matmul_desc,
                *self.layouts,
                self.preference,
                32,
                results,
                ctypes.byref(count),
            ),
            "hipblasLtMatmulAlgoGetHeuristic",
        )
        algorithms = tuple(results[index] for index in range(count.value))
        if not algorithms:
            raise RuntimeError(
                "hipBLASLt returned no Laguna attention algorithms"
            )
        return algorithms

    def launch(
        self,
        a_ptr: int,
        b_ptr: int,
        out_ptr: int,
        *,
        stream: int,
    ) -> None:
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        library = self.owner.library
        self.owner._check(
            library.hipblasLtMatmul(
                ctypes.c_void_p(self.owner.handle),
                self.matmul_desc,
                ctypes.byref(alpha),
                ctypes.c_void_p(a_ptr),
                self.layouts[0],
                ctypes.c_void_p(b_ptr),
                self.layouts[1],
                ctypes.byref(beta),
                ctypes.c_void_p(out_ptr),
                self.layouts[2],
                ctypes.c_void_p(out_ptr),
                self.layouts[3],
                ctypes.byref(self.algorithm.algo),
                ctypes.c_void_p(),
                0,
                ctypes.c_void_p(stream),
            ),
            "hipblasLtMatmul",
        )

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


@dataclass(frozen=True)
class _ProblemPair:
    qk: _BatchedProblem
    pv: _BatchedProblem


class LagunaAttentionHipblasLt:
    """Own F32 staging and cached descriptors for initial M128 attention."""

    def __init__(
        self,
        *,
        library_path: str = "libhipblaslt.so",
        runtime: HipRuntime | None = None,
        packed_queries: bool = False,
    ) -> None:
        self.runtime = runtime or get_hip_runtime()
        self.owner = HipblasLt(library_path)
        self.packed_queries = bool(packed_queries)
        self._problems: dict[tuple[int, int], _ProblemPair] = {}
        self._buffers: list[DeviceBuffer] = []
        self._closed = False
        try:
            self.key_f32 = self._allocate(
                _MAX_CONTEXT * _KV_HEADS * _HEAD_DIM * 4
            )
            self.value_f32 = self._allocate(
                _MAX_CONTEXT * _KV_HEADS * _HEAD_DIM * 4
            )
            self.scores_f32 = self._allocate(
                _MAX_Q_HEADS * _ROWS * _MAX_CONTEXT * 4
            )
            self.head_major_f32 = (
                self._allocate(_MAX_Q_HEADS * _ROWS * _HEAD_DIM * 4)
                if self.packed_queries
                else None
            )
        except Exception:
            self.close()
            raise

    def _allocate(self, nbytes: int) -> DeviceBuffer:
        buffer = malloc(int(nbytes), runtime=self.runtime)
        self._buffers.append(buffer)
        return buffer

    @staticmethod
    def supports(
        *,
        rows: int,
        start_position: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> bool:
        context = int(start_position) + int(rows)
        return (
            int(rows) == _ROWS
            and context in {128, 256, 384, 512}
            and int(num_q_heads) in {48, 72}
            and int(num_kv_heads) == _KV_HEADS
            and int(head_dim) == _HEAD_DIM
        )

    def _problem(self, num_q_heads: int, context: int) -> _ProblemPair:
        key = (int(num_q_heads), int(context))
        cached = self._problems.get(key)
        if cached is not None:
            return cached
        q_heads, parsed_context = key
        q_group = q_heads // _KV_HEADS
        batch_count = _KV_HEADS if self.packed_queries else q_group
        wide_rows = _ROWS * q_group if self.packed_queries else _ROWS
        query_leading = _HEAD_DIM if self.packed_queries else q_heads * _HEAD_DIM
        query_batch_stride = (
            wide_rows * _HEAD_DIM if self.packed_queries else _HEAD_DIM
        )
        score_batch_stride = parsed_context * wide_rows
        qk = _BatchedProblem(
            self.owner,
            transa=HIPBLAS_OP_T,
            layouts=(
                _Layout(
                    _HEAD_DIM,
                    parsed_context,
                    _KV_HEADS * _HEAD_DIM,
                    batch_count,
                    _HEAD_DIM if self.packed_queries else 0,
                ),
                _Layout(
                    _HEAD_DIM,
                    wide_rows,
                    query_leading,
                    batch_count,
                    query_batch_stride,
                ),
                _Layout(
                    parsed_context,
                    wide_rows,
                    parsed_context,
                    batch_count,
                    score_batch_stride,
                ),
                _Layout(
                    parsed_context,
                    wide_rows,
                    parsed_context,
                    batch_count,
                    score_batch_stride,
                ),
            ),
            preferred_index=(
                _PACKED_ALGORITHM_INDEX[(q_heads, parsed_context, "qk")]
                if self.packed_queries
                else _ALGORITHM_INDEX[(q_heads, parsed_context, "qk")]
            ),
        )
        try:
            pv = _BatchedProblem(
                self.owner,
                transa=_HIPBLAS_OP_N,
                layouts=(
                    _Layout(
                        _HEAD_DIM,
                        parsed_context,
                        _KV_HEADS * _HEAD_DIM,
                        batch_count,
                        _HEAD_DIM if self.packed_queries else 0,
                    ),
                    _Layout(
                        parsed_context,
                        wide_rows,
                        parsed_context,
                        batch_count,
                        score_batch_stride,
                    ),
                    _Layout(
                        _HEAD_DIM,
                        wide_rows,
                        query_leading,
                        batch_count,
                        query_batch_stride,
                    ),
                    _Layout(
                        _HEAD_DIM,
                        wide_rows,
                        query_leading,
                        batch_count,
                        query_batch_stride,
                    ),
                ),
                preferred_index=(
                    _PACKED_ALGORITHM_INDEX[(q_heads, parsed_context, "pv")]
                    if self.packed_queries
                    else _ALGORITHM_INDEX[(q_heads, parsed_context, "pv")]
                ),
            )
        except Exception:
            qk.close()
            raise
        cached = _ProblemPair(qk=qk, pv=pv)
        self._problems[key] = cached
        return cached

    def launch(
        self,
        query_ptr: int,
        key_cache_ptr: int,
        value_cache_ptr: int,
        out_ptr: int,
        spans: KVLiveSpans,
        *,
        rows: int,
        start_position: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        stream: int = 0,
        kv_library=None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Laguna attention hipBLASLt route is closed")
        if not self.supports(
            rows=rows,
            start_position=start_position,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        ):
            raise ValueError("unsupported Laguna attention hipBLASLt shape")
        q_heads = int(num_q_heads)
        context = int(start_position) + int(rows)
        q_group = q_heads // _KV_HEADS
        problems = self._problem(q_heads, context)
        laguna_dense_initial_cache_bf16_to_f32_spans(
            key_cache_ptr,
            value_cache_ptr,
            self.key_f32.ptr,
            self.value_f32.ptr,
            spans,
            context,
            num_kv_heads,
            head_dim,
            stream=stream,
            library=kv_library,
            runtime=self.runtime,
        )
        if self.packed_queries:
            assert self.head_major_f32 is not None
            laguna_dense_initial_query_head_transpose_f32(
                query_ptr,
                self.head_major_f32.ptr,
                rows,
                q_heads,
                head_dim,
                to_head_major=True,
                stream=stream,
                library=kv_library,
                runtime=self.runtime,
            )
            problems.qk.launch(
                self.key_f32.ptr,
                self.head_major_f32.ptr,
                self.scores_f32.ptr,
                stream=stream,
            )
        else:
            for kv_head in range(_KV_HEADS):
                problems.qk.launch(
                    self.key_f32.ptr + kv_head * _HEAD_DIM * 4,
                    int(query_ptr)
                    + kv_head * q_group * _HEAD_DIM * 4,
                    self.scores_f32.ptr
                    + kv_head * q_group * _ROWS * context * 4,
                    stream=stream,
                )
        laguna_dense_initial_causal_softmax_f32_spans(
            self.scores_f32.ptr,
            spans,
            rows,
            context,
            q_heads,
            start_position,
            scale,
            stream=stream,
            library=kv_library,
            runtime=self.runtime,
        )
        if self.packed_queries:
            assert self.head_major_f32 is not None
            problems.pv.launch(
                self.value_f32.ptr,
                self.scores_f32.ptr,
                self.head_major_f32.ptr,
                stream=stream,
            )
            laguna_dense_initial_query_head_transpose_f32(
                self.head_major_f32.ptr,
                out_ptr,
                rows,
                q_heads,
                head_dim,
                to_head_major=False,
                stream=stream,
                library=kv_library,
                runtime=self.runtime,
            )
        else:
            for kv_head in range(_KV_HEADS):
                problems.pv.launch(
                    self.value_f32.ptr + kv_head * _HEAD_DIM * 4,
                    self.scores_f32.ptr
                    + kv_head * q_group * _ROWS * context * 4,
                    int(out_ptr)
                    + kv_head * q_group * _HEAD_DIM * 4,
                    stream=stream,
                )

    @property
    def scratch_nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self._buffers)

    @property
    def cached_shape_count(self) -> int:
        return len(self._problems)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for pair in reversed(tuple(self._problems.values())):
            pair.pv.close()
            pair.qk.close()
        self._problems.clear()
        if self.owner.handle:
            self.owner.close()
        for buffer in reversed(self._buffers):
            free(buffer, runtime=self.runtime)
        self._buffers.clear()


__all__ = ["LagunaAttentionHipblasLt"]
