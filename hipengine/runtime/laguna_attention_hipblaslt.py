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
    laguna_dense_initial_attention_tile_merge_f32,
    laguna_dense_initial_cache_block_bf16_to_f32_spans,
    laguna_dense_initial_cache_bf16_to_f32_spans,
    laguna_dense_initial_contiguous_cache_block_bf16_to_f32_spans,
    laguna_dense_initial_causal_softmax_f32_spans,
    laguna_dense_initial_causal_softmax_tile_wave_rows_f32_spans,
    laguna_dense_initial_causal_softmax_wave_rows_f32_spans,
    laguna_dense_initial_query_head_transpose_f32,
    laguna_swa_union_bf16_to_f32_spans,
    laguna_swa_union_softmax_wave_rows_f32,
)
from hipengine.kvcache import KVLiveSpans

_HIPBLAS_OP_N = 111
_MATRIX_LAYOUT_BATCH_COUNT = 0
_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 1
_ROWS = 128
_PRODUCTION_MAX_CONTEXT = 512
_MAX_Q_HEADS = 72
_KV_HEADS = 8
_HEAD_DIM = 128
_WIDE_QUERY_ROWS = (128, 256, 512, 1024, 2048)

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
    (72, 639, "qk"): 25,
    (72, 639, "pv"): 18,
    (48, 4096, "qk"): 20,
    (48, 4096, "pv"): 25,
}

_WIDE_PACKED_ALGORITHM_INDEX = {
    (2_048, 48, 2_048, "qk"): 15,
    (2_048, 48, 2_048, "pv"): 1,
    (2_048, 48, 4_096, "qk"): 15,
    (2_048, 48, 4_096, "pv"): 2,
}

# Robust context bands from the gfx1151 LC-1 packed-F32 screens. The production
# loop visits every M128 context, so use algorithms that stayed near the winner
# across adjacent anchor shapes rather than overfitting exact benchmark points.
_LONG_PACKED_ALGORITHM_BANDS = (
    (1024, 30, 0),
    (2048, 20, 19),
    (8192, 22, 25),
    (16384, 28, 1),
    (131072, 28, 3),
)


def _preferred_algorithm_index(
    *,
    query_rows: int,
    query_heads: int,
    context: int,
    operation: str,
    packed_queries: bool,
) -> int:
    key = (int(query_heads), int(context), str(operation))
    wide_key = (
        int(query_rows),
        int(query_heads),
        int(context),
        str(operation),
    )
    if packed_queries and wide_key in _WIDE_PACKED_ALGORITHM_INDEX:
        return int(_WIDE_PACKED_ALGORITHM_INDEX[wide_key])
    table = _PACKED_ALGORITHM_INDEX if packed_queries else _ALGORITHM_INDEX
    selected = table.get(key)
    if selected is not None:
        return int(selected)
    if packed_queries and int(query_heads) == 48 and int(context) > 512:
        for upper_context, qk_index, pv_index in _LONG_PACKED_ALGORITHM_BANDS:
            if int(context) <= upper_context:
                return qk_index if operation == "qk" else pv_index
    return 0


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
        self.algorithms = self._algorithms()
        self.algorithm_index = min(
            max(int(preferred_index), 0),
            len(self.algorithms) - 1,
        )
        self.algorithm = self.algorithms[self.algorithm_index]
        if any(int(algorithm.workspace_size) != 0 for algorithm in self.algorithms):
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
        algorithm_index: int | None = None,
    ) -> None:
        selected = self.algorithm_index
        if algorithm_index is not None:
            selected = int(algorithm_index)
            if selected < 0 or selected >= len(self.algorithms):
                raise ValueError("hipBLASLt algorithm index is out of range")
        algorithm = self.algorithms[selected]
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
                ctypes.byref(algorithm.algo),
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
        wave_rows_softmax: bool = False,
        max_context: int = _PRODUCTION_MAX_CONTEXT,
        max_q_heads: int = _MAX_Q_HEADS,
        block_context: int | None = None,
        query_rows: int = _ROWS,
    ) -> None:
        self.runtime = runtime or get_hip_runtime()
        self.packed_queries = bool(packed_queries)
        self.wave_rows_softmax = bool(wave_rows_softmax)
        self.max_context = int(max_context)
        self.max_q_heads = int(max_q_heads)
        self.query_rows = int(query_rows)
        self.block_context = (
            None if block_context is None else int(block_context)
        )
        if self.query_rows not in _WIDE_QUERY_ROWS:
            raise ValueError(
                "Laguna attention query_rows must be one of "
                "128/256/512/1024/2048"
            )
        if (
            self.max_context < _PRODUCTION_MAX_CONTEXT
            or self.max_context > 131072
            or self.max_context % _ROWS != 0
        ):
            raise ValueError(
                "Laguna attention max_context must be an M128 multiple "
                "within [512, 131072]"
            )
        if self.max_q_heads not in {48, 72}:
            raise ValueError("Laguna attention max_q_heads must be 48 or 72")
        if self.block_context is not None and (
            not self.packed_queries
            or self.block_context < _ROWS
            or self.block_context > self.max_context
            or self.block_context % _ROWS != 0
        ):
            raise ValueError(
                "blocked Laguna attention requires packed queries and an "
                "M128 block_context within [128, max_context]"
            )
        scratch_context = self.block_context or self.max_context
        self.owner = HipblasLt(library_path)
        self._problems: dict[tuple[int, int], _ProblemPair] = {}
        self._buffers: list[DeviceBuffer] = []
        self._closed = False
        try:
            self.key_f32 = self._allocate(
                scratch_context * _KV_HEADS * _HEAD_DIM * 4
            )
            self.value_f32 = self._allocate(
                scratch_context * _KV_HEADS * _HEAD_DIM * 4
            )
            self.scores_f32 = self._allocate(
                self.max_q_heads * self.query_rows * scratch_context * 4
            )
            self.head_major_f32 = (
                self._allocate(
                    self.max_q_heads * self.query_rows * _HEAD_DIM * 4
                )
                if self.packed_queries
                else None
            )
            blocked_rows = self.max_q_heads * self.query_rows
            self.tile_output_f32 = (
                self._allocate(blocked_rows * _HEAD_DIM * 4)
                if self.block_context is not None
                else None
            )
            self.accum_head_major_f32 = (
                self._allocate(blocked_rows * _HEAD_DIM * 4)
                if self.block_context is not None
                else None
            )
            self.row_max_f32 = (
                self._allocate(blocked_rows * 4)
                if self.block_context is not None
                else None
            )
            self.row_sum_f32 = (
                self._allocate(blocked_rows * 4)
                if self.block_context is not None
                else None
            )
            self.merge_scales_f32 = (
                self._allocate(blocked_rows * 2 * 4)
                if self.block_context is not None
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

    def _supports_shape(
        self,
        *,
        rows: int,
        start_position: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> bool:
        if self.max_context == _PRODUCTION_MAX_CONTEXT:
            return self.supports(
                rows=rows,
                start_position=start_position,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        context = int(start_position) + int(rows)
        return (
            int(rows) == self.query_rows
            and _ROWS <= context <= self.max_context
            and context % _ROWS == 0
            and int(num_q_heads) in {48, 72}
            and int(num_q_heads) <= self.max_q_heads
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
        wide_rows = (
            self.query_rows * q_group
            if self.packed_queries
            else self.query_rows
        )
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
                _preferred_algorithm_index(
                    query_rows=self.query_rows,
                    query_heads=q_heads,
                    context=parsed_context,
                    operation="qk",
                    packed_queries=self.packed_queries,
                )
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
                    _preferred_algorithm_index(
                        query_rows=self.query_rows,
                        query_heads=q_heads,
                        context=parsed_context,
                        operation="pv",
                        packed_queries=self.packed_queries,
                    )
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
        query_is_packed: bool = False,
        unpack_output: bool = True,
        qk_algorithm_index: int | None = None,
        pv_algorithm_index: int | None = None,
        dense_contiguous_cache: bool = False,
    ) -> None:
        if self._closed:
            raise RuntimeError("Laguna attention hipBLASLt route is closed")
        if not self._supports_shape(
            rows=rows,
            start_position=start_position,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        ):
            raise ValueError("unsupported Laguna attention hipBLASLt shape")
        if query_is_packed and not self.packed_queries:
            raise ValueError("packed query input requires the packed-query route")
        if self.block_context is not None:
            self._launch_blocked(
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                out_ptr,
                spans,
                rows=rows,
                start_position=start_position,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                scale=scale,
                stream=stream,
                kv_library=kv_library,
                query_is_packed=query_is_packed,
                unpack_output=unpack_output,
                qk_algorithm_index=qk_algorithm_index,
                pv_algorithm_index=pv_algorithm_index,
                dense_contiguous_cache=dense_contiguous_cache,
            )
            return
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
            query_head_major_ptr = int(query_ptr)
            if not query_is_packed:
                query_head_major_ptr = self.head_major_f32.ptr
                laguna_dense_initial_query_head_transpose_f32(
                    query_ptr,
                    query_head_major_ptr,
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
                query_head_major_ptr,
                self.scores_f32.ptr,
                stream=stream,
                algorithm_index=qk_algorithm_index,
            )
        else:
            for kv_head in range(_KV_HEADS):
                problems.qk.launch(
                    self.key_f32.ptr + kv_head * _HEAD_DIM * 4,
                    int(query_ptr)
                    + kv_head * q_group * _HEAD_DIM * 4,
                    self.scores_f32.ptr
                    + kv_head * q_group * self.query_rows * context * 4,
                    stream=stream,
                    algorithm_index=qk_algorithm_index,
                )
        if not self.wave_rows_softmax:
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
        else:
            laguna_dense_initial_causal_softmax_wave_rows_f32_spans(
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
                self.head_major_f32.ptr if unpack_output else out_ptr,
                stream=stream,
                algorithm_index=pv_algorithm_index,
            )
            if unpack_output:
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
                    + kv_head * q_group * self.query_rows * context * 4,
                    int(out_ptr)
                    + kv_head * q_group * _HEAD_DIM * 4,
                    stream=stream,
                    algorithm_index=pv_algorithm_index,
                )

    def _launch_blocked(
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
        stream: int,
        kv_library,
        query_is_packed: bool,
        unpack_output: bool,
        qk_algorithm_index: int | None,
        pv_algorithm_index: int | None,
        dense_contiguous_cache: bool,
    ) -> None:
        assert self.block_context is not None
        assert self.head_major_f32 is not None
        assert self.tile_output_f32 is not None
        assert self.accum_head_major_f32 is not None
        assert self.row_max_f32 is not None
        assert self.row_sum_f32 is not None
        assert self.merge_scales_f32 is not None
        q_heads = int(num_q_heads)
        context = int(start_position) + int(rows)
        query_head_major_ptr = int(query_ptr)
        if not query_is_packed:
            query_head_major_ptr = self.head_major_f32.ptr
            laguna_dense_initial_query_head_transpose_f32(
                query_ptr,
                query_head_major_ptr,
                rows,
                q_heads,
                head_dim,
                to_head_major=True,
                stream=stream,
                library=kv_library,
                runtime=self.runtime,
            )
        for tile_start in range(0, context, self.block_context):
            tile_count = min(self.block_context, context - tile_start)
            first_tile = tile_start == 0
            final_tile = tile_start + tile_count == context
            cache_widen = (
                laguna_dense_initial_contiguous_cache_block_bf16_to_f32_spans
                if dense_contiguous_cache
                else laguna_dense_initial_cache_block_bf16_to_f32_spans
            )
            cache_widen(
                key_cache_ptr,
                value_cache_ptr,
                self.key_f32.ptr,
                self.value_f32.ptr,
                spans,
                tile_start,
                tile_count,
                context,
                num_kv_heads,
                head_dim,
                stream=stream,
                library=kv_library,
                runtime=self.runtime,
            )
            problems = self._problem(q_heads, tile_count)
            problems.qk.launch(
                self.key_f32.ptr,
                query_head_major_ptr,
                self.scores_f32.ptr,
                stream=stream,
                algorithm_index=qk_algorithm_index,
            )
            laguna_dense_initial_causal_softmax_tile_wave_rows_f32_spans(
                self.scores_f32.ptr,
                self.row_max_f32.ptr,
                self.row_sum_f32.ptr,
                self.merge_scales_f32.ptr,
                spans,
                rows,
                tile_start,
                tile_count,
                context,
                q_heads,
                start_position,
                scale,
                stream=stream,
                library=kv_library,
                runtime=self.runtime,
            )
            problems.pv.launch(
                self.value_f32.ptr,
                self.scores_f32.ptr,
                self.tile_output_f32.ptr,
                stream=stream,
                algorithm_index=pv_algorithm_index,
            )
            final_output_ptr = (
                self.accum_head_major_f32.ptr
                if unpack_output
                else int(out_ptr)
            )
            laguna_dense_initial_attention_tile_merge_f32(
                self.accum_head_major_f32.ptr,
                self.tile_output_f32.ptr,
                final_output_ptr,
                self.row_sum_f32.ptr,
                self.merge_scales_f32.ptr,
                rows,
                q_heads,
                head_dim,
                first_tile=first_tile,
                final_tile=final_tile,
                stream=stream,
                library=kv_library,
                runtime=self.runtime,
            )
        if unpack_output:
            laguna_dense_initial_query_head_transpose_f32(
                self.accum_head_major_f32.ptr,
                out_ptr,
                rows,
                q_heads,
                head_dim,
                to_head_major=False,
                stream=stream,
                library=kv_library,
                runtime=self.runtime,
            )

    def algorithm_counts(
        self,
        *,
        num_q_heads: int,
        context: int,
    ) -> tuple[int, int]:
        """Return zero-workspace QK/PV heuristic counts for one admitted shape."""

        pair = self._problem(int(num_q_heads), int(context))
        return len(pair.qk.algorithms), len(pair.pv.algorithms)

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


class LagunaSwaAttentionHipblasLt(LagunaAttentionHipblasLt):
    """Tensorized rolling M128 attention over one 512-token SWA window."""

    _WINDOW = 512
    _UNION_CONTEXT = _WINDOW + _ROWS - 1

    def __init__(
        self,
        *,
        library_path: str = "libhipblaslt.so",
        runtime: HipRuntime | None = None,
    ) -> None:
        super().__init__(
            library_path=library_path,
            runtime=runtime,
            packed_queries=True,
            wave_rows_softmax=True,
            max_context=640,
            max_q_heads=72,
        )

    @staticmethod
    def supports(
        *,
        rows: int,
        start_position: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int,
    ) -> bool:
        return (
            int(rows) == _ROWS
            and int(start_position) >= 512
            and int(num_q_heads) == 72
            and int(num_kv_heads) == _KV_HEADS
            and int(head_dim) == _HEAD_DIM
            and int(sliding_window) == 512
        )

    def launch(
        self,
        query_ptr: int,
        current_key_ptr: int,
        current_value_ptr: int,
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
        sliding_window: int,
        scale: float,
        stream: int = 0,
        kv_library=None,
        query_is_packed: bool = False,
        unpack_output: bool = True,
        qk_algorithm_index: int | None = None,
        pv_algorithm_index: int | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Laguna SWA hipBLASLt route is closed")
        if not self.supports(
            rows=rows,
            start_position=start_position,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window=sliding_window,
        ):
            raise ValueError("unsupported Laguna rolling-SWA hipBLASLt shape")
        assert self.head_major_f32 is not None
        laguna_swa_union_bf16_to_f32_spans(
            current_key_ptr,
            current_value_ptr,
            key_cache_ptr,
            value_cache_ptr,
            self.key_f32.ptr,
            self.value_f32.ptr,
            spans,
            rows,
            num_kv_heads,
            head_dim,
            start_position,
            sliding_window=sliding_window,
            stream=stream,
            library=kv_library,
            runtime=self.runtime,
        )
        query_head_major_ptr = int(query_ptr)
        if not query_is_packed:
            query_head_major_ptr = self.head_major_f32.ptr
            laguna_dense_initial_query_head_transpose_f32(
                query_ptr,
                query_head_major_ptr,
                rows,
                num_q_heads,
                head_dim,
                to_head_major=True,
                stream=stream,
                library=kv_library,
                runtime=self.runtime,
            )
        problems = self._problem(num_q_heads, self._UNION_CONTEXT)
        problems.qk.launch(
            self.key_f32.ptr,
            query_head_major_ptr,
            self.scores_f32.ptr,
            stream=stream,
            algorithm_index=qk_algorithm_index,
        )
        laguna_swa_union_softmax_wave_rows_f32(
            self.scores_f32.ptr,
            rows,
            num_q_heads,
            scale,
            union_context=self._UNION_CONTEXT,
            sliding_window=sliding_window,
            stream=stream,
            library=kv_library,
            runtime=self.runtime,
        )
        output_head_major_ptr = (
            self.head_major_f32.ptr if unpack_output else int(out_ptr)
        )
        problems.pv.launch(
            self.value_f32.ptr,
            self.scores_f32.ptr,
            output_head_major_ptr,
            stream=stream,
            algorithm_index=pv_algorithm_index,
        )
        if unpack_output:
            laguna_dense_initial_query_head_transpose_f32(
                output_head_major_ptr,
                out_ptr,
                rows,
                num_q_heads,
                head_dim,
                to_head_major=False,
                stream=stream,
                library=kv_library,
                runtime=self.runtime,
            )


__all__ = [
    "LagunaAttentionHipblasLt",
    "LagunaSwaAttentionHipblasLt",
]
