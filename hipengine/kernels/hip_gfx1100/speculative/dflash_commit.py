"""Raw-pointer GPU DFlash verified-state commit wrappers."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.dtype import DType, dtype_itemsize
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register
from hipengine.speculative.interfaces import TargetStateCommitBuffers

_SOURCE = Path(__file__).with_name("dflash_commit.hip")
_OUTPUT_NAME = "dflash_commit.so"
_SYMBOL_CHAIN_I32 = "hipengine_dflash_commit_chain_i32"
_SYMBOL_LINEAR_PAIR_COMMIT_I32 = "hipengine_linear_state_pair_commit_i32"
_SYMBOL_LINEAR_PAIR_COMMIT_CHUNKED_I32 = "hipengine_linear_state_pair_commit_chunked_i32"


def plan_dflash_commit_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="dflash_commit",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dflash_commit(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="dflash_commit",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def dflash_commit_chain_i32(
    buffers: TargetStateCommitBuffers,
    *,
    target_rows: int,
    accepted_rows: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Commit accepted DFlash verifier rows on device.

    ``buffers`` must contain int32 summary tensors produced by the GPU accept
    path. Optional row-major state tensors are copied as follows:

    - ``linear_state_src[commit_row] -> linear_state_dst[request_index]``;
    - accepted full-attention ``kv_rows_src`` path rows, reconstructed from
      ``parent_rows`` and ``commit_rows``, compact into ``kv_rows_dst``;
    - tap-major ``hidden_taps_src[:, commit_row, :]`` rows copy to drafter
      context ``hidden_taps_dst[:, request_index, :]``;
    - compact committed output ids/lengths copy to the output ring while
      clearing suffix slots to ``-1``;
    - position metadata writes ``last_positions_dst=commit_position`` and
      ``context_lengths_dst=commit_position+1``.

    Full logits never participate in this API; the fast path consumes only the
    compact accept-summary buffers.
    """

    shape = _CommitShape.from_buffers(buffers, target_rows=target_rows, accepted_rows=accepted_rows)
    library = library or build_dflash_commit(load=True)
    runtime = runtime or get_hip_runtime()
    if buffers.kv_rows_dst is not None and shape.kv_dst_nbytes > 0:
        runtime.memset_async(buffers.kv_rows_dst.ptr, 0, shape.kv_dst_nbytes, stream)
    fn = getattr(library, _SYMBOL_CHAIN_I32)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(buffers.accepted_counts.ptr),
        ctypes.c_void_p(buffers.commit_rows.ptr),
        ctypes.c_void_p(buffers.commit_positions.ptr),
        _voidp(buffers.parent_rows),
        _voidp(buffers.linear_state_src),
        _voidp(buffers.linear_state_dst),
        ctypes.c_int64(shape.linear_row_bytes),
        _voidp(buffers.kv_rows_src),
        _voidp(buffers.kv_rows_dst),
        ctypes.c_int64(shape.kv_row_bytes),
        ctypes.c_int64(shape.kv_dst_rows),
        _voidp(buffers.hidden_taps_src),
        _voidp(buffers.hidden_taps_dst),
        ctypes.c_int64(shape.hidden_tap_count),
        ctypes.c_int64(shape.hidden_tap_src_row_stride_bytes),
        ctypes.c_int64(shape.hidden_tap_dst_row_stride_bytes),
        ctypes.c_int64(shape.hidden_tap_src_plane_stride_bytes),
        ctypes.c_int64(shape.hidden_tap_dst_plane_stride_bytes),
        _voidp(buffers.next_tokens_src),
        _voidp(buffers.committed_output_ids_src),
        _voidp(buffers.committed_output_lengths_src),
        _voidp(buffers.output_ids_dst),
        _voidp(buffers.output_lengths_dst),
        ctypes.c_int64(shape.output_src_cols),
        ctypes.c_int64(shape.output_src_row_stride),
        ctypes.c_int64(shape.output_dst_cols),
        ctypes.c_int64(shape.output_dst_row_stride),
        _voidp(buffers.last_positions_dst),
        _voidp(buffers.context_lengths_dst),
        ctypes.c_int64(shape.target_rows),
        ctypes.c_int64(shape.request_count),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def linear_state_pair_commit_i32(
    src_conv_table: int,
    dst_conv_table: int,
    conv_row_nbytes: int,
    src_recurrent_table: int,
    dst_recurrent_table: int,
    recurrent_row_nbytes: int,
    commit_row: int,
    n_layers: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """M12.4: commit selected linear-attention state rows for all layers in one launch.

    Replaces the ``2*n_layers`` ``hipMemcpyAsync`` calls in
    ``Qwen35ParoResidentSession._commit_bulk_linear_states``.  Both source and
    destination per-layer pointer tables live on the device (each is a
    contiguous ``uint64[n_layers]`` array of base addresses).  Each linear-
    attention layer owns its own workspace, so source bases are not uniform
    across layers; the host refreshes the source table before each launch.
    ``commit_row`` is the device pointer to ``verify_commit_rows[0]`` written
    by ``dflash_accept_chain_i32``.
    """

    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if conv_row_nbytes <= 0 and recurrent_row_nbytes <= 0:
        raise ValueError("at least one of conv_row_nbytes / recurrent_row_nbytes must be positive")
    library = library or build_dflash_commit(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_LINEAR_PAIR_COMMIT_I32)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(src_conv_table),
        ctypes.c_void_p(dst_conv_table),
        ctypes.c_int64(conv_row_nbytes),
        ctypes.c_void_p(src_recurrent_table),
        ctypes.c_void_p(dst_recurrent_table),
        ctypes.c_int64(recurrent_row_nbytes),
        ctypes.c_void_p(commit_row),
        ctypes.c_int64(n_layers),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def linear_state_pair_commit_chunked_i32(
    src_conv_table: int,
    dst_conv_table: int,
    conv_row_nbytes: int,
    src_recurrent_table: int,
    dst_recurrent_table: int,
    recurrent_row_nbytes: int,
    commit_row: int,
    n_layers: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Commit selected linear-attention state rows with multiple CTAs per row.

    The M12.4 fused commit uses one CTA per ``(layer, state-family)`` row.  The
    recurrent rows are large enough on 35B-A3B MTP that a chunked grid can expose
    more copy parallelism while keeping the same pointer-table ABI.
    """

    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if conv_row_nbytes <= 0 and recurrent_row_nbytes <= 0:
        raise ValueError("at least one of conv_row_nbytes / recurrent_row_nbytes must be positive")
    library = library or build_dflash_commit(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_LINEAR_PAIR_COMMIT_CHUNKED_I32)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(src_conv_table),
        ctypes.c_void_p(dst_conv_table),
        ctypes.c_int64(conv_row_nbytes),
        ctypes.c_void_p(src_recurrent_table),
        ctypes.c_void_p(dst_recurrent_table),
        ctypes.c_int64(recurrent_row_nbytes),
        ctypes.c_void_p(commit_row),
        ctypes.c_int64(n_layers),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_dflash_commit_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "dflash_commit_chain", "w4_paro", "i32"),
        dflash_commit_chain_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_state_pair_commit", "w4_paro", "i32"),
        linear_state_pair_commit_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_state_pair_commit", "w4_paro", "chunked_i32"),
        linear_state_pair_commit_chunked_i32,
        replace=replace,
    )


class _CommitShape:
    def __init__(
        self,
        *,
        target_rows: int,
        request_count: int,
        linear_row_bytes: int = 0,
        kv_row_bytes: int = 0,
        kv_dst_rows: int = 0,
        kv_dst_nbytes: int = 0,
        hidden_tap_count: int = 0,
        hidden_tap_src_row_stride_bytes: int = 0,
        hidden_tap_dst_row_stride_bytes: int = 0,
        hidden_tap_src_plane_stride_bytes: int = 0,
        hidden_tap_dst_plane_stride_bytes: int = 0,
        output_src_cols: int = 0,
        output_src_row_stride: int = 0,
        output_dst_cols: int = 0,
        output_dst_row_stride: int = 0,
    ) -> None:
        self.target_rows = int(target_rows)
        self.request_count = int(request_count)
        self.linear_row_bytes = int(linear_row_bytes)
        self.kv_row_bytes = int(kv_row_bytes)
        self.kv_dst_rows = int(kv_dst_rows)
        self.kv_dst_nbytes = int(kv_dst_nbytes)
        self.hidden_tap_count = int(hidden_tap_count)
        self.hidden_tap_src_row_stride_bytes = int(hidden_tap_src_row_stride_bytes)
        self.hidden_tap_dst_row_stride_bytes = int(hidden_tap_dst_row_stride_bytes)
        self.hidden_tap_src_plane_stride_bytes = int(hidden_tap_src_plane_stride_bytes)
        self.hidden_tap_dst_plane_stride_bytes = int(hidden_tap_dst_plane_stride_bytes)
        self.output_src_cols = int(output_src_cols)
        self.output_src_row_stride = int(output_src_row_stride)
        self.output_dst_cols = int(output_dst_cols)
        self.output_dst_row_stride = int(output_dst_row_stride)

    @classmethod
    def from_buffers(
        cls,
        buffers: TargetStateCommitBuffers,
        *,
        target_rows: int,
        accepted_rows: int | None,
    ) -> "_CommitShape":
        rows = int(target_rows)
        request_count = buffers.request_count
        if rows <= 0:
            raise ValueError("target_rows must be positive")
        if request_count <= 0 or request_count > rows:
            raise ValueError("request_count must be positive and no larger than target_rows")
        _require_int32("accepted_counts", buffers.accepted_counts)
        _require_int32("commit_rows", buffers.commit_rows)
        _require_int32("commit_positions", buffers.commit_positions)
        if buffers.parent_rows is not None:
            _require_int32("parent_rows", buffers.parent_rows)
            if buffers.parent_rows.shape != (rows,):
                raise ValueError("parent_rows must have shape (target_rows,)")
        linear_row_bytes = 0
        if buffers.linear_state_src is not None:
            linear_row_bytes = _validate_row_copy_pair(
                "linear_state",
                buffers.linear_state_src,
                buffers.linear_state_dst,
                rows,
                request_count,
            )
        kv_row_bytes = 0
        kv_dst_rows = 0
        kv_dst_nbytes = 0
        if buffers.kv_rows_src is not None:
            if buffers.parent_rows is None:
                raise ValueError("parent_rows are required when committing KV rows")
            assert buffers.kv_rows_dst is not None
            kv_row_bytes = _validate_row_copy_pair(
                "kv_rows",
                buffers.kv_rows_src,
                buffers.kv_rows_dst,
                rows,
                0,
            )
            kv_dst_rows = buffers.kv_rows_dst.shape[0]
            kv_dst_nbytes = _tensor_nbytes(buffers.kv_rows_dst.shape, buffers.kv_rows_dst.dtype)
            if accepted_rows is not None and kv_dst_rows < int(accepted_rows):
                raise ValueError("KV destination rows must cover accepted token rows")
        hidden_tap_count = 0
        hidden_src_stride = 0
        hidden_dst_stride = 0
        hidden_src_plane_stride = 0
        hidden_dst_plane_stride = 0
        if buffers.hidden_taps_src is not None:
            assert buffers.hidden_taps_dst is not None
            _require_fixed_itemsize("hidden_taps_src", buffers.hidden_taps_src.dtype)
            if buffers.hidden_taps_src.shape[1] < rows:
                raise ValueError("hidden_taps source rows must cover target_rows")
            if buffers.hidden_taps_dst.shape[1] < request_count:
                raise ValueError("hidden_taps destination rows must cover request_count")
            hidden_tap_count = buffers.hidden_taps_src.shape[0]
            hidden_src_strides = buffers.hidden_taps_src.strides or (
                buffers.hidden_taps_src.shape[1] * buffers.hidden_taps_src.shape[2],
                buffers.hidden_taps_src.shape[2],
                1,
            )
            hidden_dst_strides = buffers.hidden_taps_dst.strides or (
                buffers.hidden_taps_dst.shape[1] * buffers.hidden_taps_dst.shape[2],
                buffers.hidden_taps_dst.shape[2],
                1,
            )
            if hidden_src_strides[2] != 1 or hidden_dst_strides[2] != 1:
                raise ValueError("hidden_taps buffers must have contiguous hidden elements")
            if hidden_src_strides[1] != buffers.hidden_taps_src.shape[2] or hidden_dst_strides[1] != buffers.hidden_taps_dst.shape[2]:
                raise ValueError("hidden_taps row strides must be compact")
            src_itemsize = dtype_itemsize(buffers.hidden_taps_src.dtype)
            dst_itemsize = dtype_itemsize(buffers.hidden_taps_dst.dtype)
            hidden_src_stride = hidden_src_strides[1] * src_itemsize
            hidden_dst_stride = hidden_dst_strides[1] * dst_itemsize
            hidden_src_plane_stride = hidden_src_strides[0] * src_itemsize
            hidden_dst_plane_stride = hidden_dst_strides[0] * dst_itemsize
        output_src_cols = 0
        output_src_row_stride = 0
        output_dst_cols = 0
        output_dst_row_stride = 0
        if buffers.committed_output_ids_src is not None:
            assert buffers.output_ids_dst is not None
            assert buffers.committed_output_lengths_src is not None
            assert buffers.output_lengths_dst is not None
            for name, tensor in (
                ("committed_output_ids_src", buffers.committed_output_ids_src),
                ("committed_output_lengths_src", buffers.committed_output_lengths_src),
                ("output_ids_dst", buffers.output_ids_dst),
                ("output_lengths_dst", buffers.output_lengths_dst),
                ("next_tokens_src", buffers.next_tokens_src),
            ):
                if tensor is not None:
                    _require_int32(name, tensor)
            output_src_cols = buffers.committed_output_ids_src.shape[1]
            output_dst_cols = buffers.output_ids_dst.shape[1]
            src_strides = buffers.committed_output_ids_src.strides or (output_src_cols, 1)
            dst_strides = buffers.output_ids_dst.strides or (output_dst_cols, 1)
            if src_strides[1] != 1 or dst_strides[1] != 1:
                raise ValueError("output id buffers must have contiguous row elements")
            output_src_row_stride = src_strides[0]
            output_dst_row_stride = dst_strides[0]
        for name, tensor in (
            ("last_positions_dst", buffers.last_positions_dst),
            ("context_lengths_dst", buffers.context_lengths_dst),
        ):
            if tensor is not None:
                _require_int32(name, tensor)
        return cls(
            target_rows=rows,
            request_count=request_count,
            linear_row_bytes=linear_row_bytes,
            kv_row_bytes=kv_row_bytes,
            kv_dst_rows=kv_dst_rows,
            kv_dst_nbytes=kv_dst_nbytes,
            hidden_tap_count=hidden_tap_count,
            hidden_tap_src_row_stride_bytes=hidden_src_stride,
            hidden_tap_dst_row_stride_bytes=hidden_dst_stride,
            hidden_tap_src_plane_stride_bytes=hidden_src_plane_stride,
            hidden_tap_dst_plane_stride_bytes=hidden_dst_plane_stride,
            output_src_cols=output_src_cols,
            output_src_row_stride=output_src_row_stride,
            output_dst_cols=output_dst_cols,
            output_dst_row_stride=output_dst_row_stride,
        )


def _validate_row_copy_pair(name: str, src, dst, target_rows: int, min_dst_rows: int) -> int:
    assert dst is not None
    if src.shape[0] < target_rows:
        raise ValueError(f"{name} source rows must cover target_rows")
    if dst.shape[0] < min_dst_rows:
        raise ValueError(f"{name} destination rows must cover required rows")
    _require_fixed_itemsize(f"{name}_src", src.dtype)
    return _row_tail_numel(src.shape) * dtype_itemsize(src.dtype)


def _row_tail_numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape[1:]:
        out *= int(dim)
    return out


def _tensor_nbytes(shape: tuple[int, ...], dtype: DType) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * dtype_itemsize(dtype)


def _require_int32(name: str, tensor) -> None:
    if tensor.dtype != DType.INT32:
        raise ValueError(f"{name} must be int32 for dflash_commit_chain_i32")


def _require_fixed_itemsize(name: str, dtype: DType) -> None:
    try:
        dtype_itemsize(dtype)
    except ValueError as exc:
        raise ValueError(f"{name} dtype must have a fixed item size") from exc


def _voidp(tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p() if tensor is None else ctypes.c_void_p(tensor.ptr)


register_dflash_commit_kernels()
