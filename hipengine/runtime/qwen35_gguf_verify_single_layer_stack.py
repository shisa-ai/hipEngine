"""Ordered eager layer walk for the single-session GGUF target verifier.

``Qwen35GGUFResidentSession.verify_target_block`` used to inline its
``for layer_id, layer_type in enumerate(layer_types)`` walk. This module owns
that walk: the per-layer F32 residual-limit transition, the optional pre-timer
synchronize, the per-layer expert-sidecar load/free lifetime, chunk-scratch
selection, the four selected operation call sites, the aggregate wall-clock
stage marks and the layer boundary/output captures.

The resident session still owns preflight, token staging and embedding, the
wmma/gemv/native/IQ policy context managers, output norm, LM head, host
readback, cursor/result/transaction ownership and graph/enqueue controls.

Every per-layer lookup that the inline loop performed lazily - the chunk
scratch, the next norm weight, the linear state rows, the initial state
snapshot and the full-attention cache - stays a narrow typed operation invoked
at the same point in the same order. Nothing here precomputes per-layer state
or validates the whole layer list before execution, so a failure leaves the
same partial mutation behind.

The executor does not read owner attributes, environment policy or a HIP
runtime. Borrowed buffers, scratch views and sidecars stay alive only for the
synchronous walk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol

from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION

LINEAR_ATTENTION_STAGE = "target_block_linear_attn_layers"
FULL_ATTENTION_STAGE = "target_block_full_attn_layers"
OTHER_LAYER_STAGE = "target_block_other_layers"
LAYER_TOTAL_STAGE = "target_block_layer_total"

LINEAR_LAYER_STAGE_PREFIX = "target_block_linear_attn"
FULL_ATTENTION_LAYER_STAGE_PREFIX = "target_block_full_attn"


# ---------------------------------------------------------------------------
# narrow typed operations
# ---------------------------------------------------------------------------
class NativeAttentionBulkFfnRowsOperation(Protocol):
    """The native bulk-attention + FFN row owner this stack calls."""

    def __call__(
        self,
        layer_id: int,
        layer_type: str,
        hidden_ptr: int,
        out_ptr: int,
        scratch: object,
        *,
        rows: int,
        decode_scratch: object,
        start_position: int = 0,
        linear_state_rows: tuple[object, object] | None = None,
        initial_state_snapshot: tuple[object, object] | None = None,
        commit_final_linear_state: bool = True,
        hidden_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
        input_norm_ptr: int | None = None,
        next_norm_weight_ptr: int | None = None,
        next_norm_out_ptr: int | None = None,
        decode_row_scratches: tuple[object, ...] | None = None,
        attention_context_limit: int | None = None,
        row_views_keep_scalar_caps: bool = False,
        stream: int = 0,
        verify_f32_post_norm: bool | None = None,
    ) -> None: ...


class LinearAttentionPrefillRowsOperation(Protocol):
    """The ordinary linear-attention row owner this stack calls."""

    def __call__(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch: object,
        *,
        rows: int,
        decode_scratch: object,
        stream: int = 0,
        expert_sidecar: object | None = None,
        linear_state_rows: tuple[object, object] | None = None,
        prefill_gdn_capture: bool | None = None,
        commit_final_linear_state: bool = True,
        hidden_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = LINEAR_LAYER_STAGE_PREFIX,
        gpu_stage_recorder: object | None = None,
        verify_f32_post_norm: bool | None = None,
    ) -> None: ...


class FullAttentionDecodeBatchRowsOperation(Protocol):
    """The full-attention decode-batch row owner this stack calls."""

    def __call__(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch: object,
        *,
        stream: int = 0,
        expert_sidecar: object | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = FULL_ATTENTION_LAYER_STAGE_PREFIX,
        hidden_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
        verify_f32_post_norm: bool | None = None,
    ) -> object: ...


class FullAttentionPrefillAotritonOperation(Protocol):
    """The full-attention prefill row owner this stack calls."""

    def __call__(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch: object,
        *,
        stream: int = 0,
        expert_sidecar: object | None = None,
        verify_f32_post_norm: bool | None = None,
    ) -> object: ...


class ChunkScratchResolver(Protocol):
    """Resolve one invocation-local chunk scratch view at its old point."""

    def __call__(self, start: int, rows: int, total_tokens: int) -> object: ...


class NextNormWeightResolver(Protocol):
    """Look up the next layer's attention-norm weight at its old point."""

    def __call__(self, layer_id: int) -> int: ...


class LinearStateRowsResolver(Protocol):
    """Look up one layer's captured linear state rows at its old point."""

    def __call__(self, layer_id: int) -> tuple[object, object] | None: ...


class InitialStateSnapshotResolver(Protocol):
    """Look up one layer's rollback snapshot at its old point."""

    def __call__(self, layer_id: int) -> tuple[object, object] | None: ...


class FullAttentionCacheResolver(Protocol):
    """Look up one layer's full-attention cache at its old point."""

    def __call__(self, layer_id: int) -> tuple[object, object]: ...


class SidecarLoader(Protocol):
    """Load one layer's expert sidecar after the timer, before the try."""

    def __call__(self, layer_id: int) -> object: ...


class SidecarFreer(Protocol):
    """Release one layer's expert sidecar first in the finally block."""

    def __call__(self, sidecar: object) -> None: ...


class Synchronizer(Protocol):
    """Synchronize the stream before the layer timer when timing is on."""

    def __call__(self) -> None: ...


class BoundaryCapture(Protocol):
    """Capture one layer's boundary tensors after a successful operation."""

    def __call__(
        self,
        layer_id: int,
        layer_type: str,
        *,
        hidden_in_ptr: int,
        hidden_in_f32_ptr: int | None,
        layer_out_ptr: int,
        layer_out_f32_ptr: int | None,
        scratch: object,
        prefill_gdn_capture: bool | None,
        verify_f32_post_norm: bool | None,
    ) -> dict[str, object]: ...


class OutputCapture(Protocol):
    """Capture one layer's output rows after the ping-pong swap."""

    def __call__(
        self,
        layer_id: int,
        *,
        hidden: object,
        hidden_f32: object | None,
        use_f32_residual: bool,
    ) -> object: ...


# ---------------------------------------------------------------------------
# explicit records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerifySingleLayerGeometry:
    """Layer order, span, row layout and resolved per-call execution inputs."""

    layer_types: tuple[str, ...]
    rows: int
    start: int
    end: int
    scratch_row_start: int
    row_nbytes: int
    stream: int
    f32_residual_layer_limit: int
    chain_dense_next_rms: bool
    bulk_attention_mode: str
    capture_linear_state_rows: bool
    defer_linear_state_commit: bool
    capture_prefill_gdn: bool | None
    verify_f32_post_norm: bool | None
    use_f32_post_norm: bool
    sync_stages: bool
    sync_stage_timings: bool
    decode_row_scratches: tuple[object, ...] | None
    attention_context_limit: int | None
    row_views_keep_scalar_caps: bool
    capture_layer_ids: frozenset[int]
    capture_layer_boundary_ids: frozenset[int]


@dataclass(frozen=True)
class VerifySingleLayerHidden:
    """The borrowed BF16/F32 hidden ping-pong pair and its F32 switch."""

    first: object
    first_ptr: int
    second: object
    second_ptr: int
    first_f32: object | None
    first_f32_ptr: int | None
    second_f32: object | None
    second_f32_ptr: int | None
    use_f32_residual: bool


@dataclass(frozen=True)
class VerifySingleLayerScratch:
    """The borrowed chunk-scratch source, decode state and cache tables."""

    prebuilt: object | None
    resolve_chunk: ChunkScratchResolver
    native_conv_out: object | None
    decode_scratch: object
    cos_table: object
    sin_table: object


@dataclass(frozen=True)
class VerifySingleLayerSidecar:
    """The per-layer expert-sidecar load/free pair."""

    load: SidecarLoader | None
    free: SidecarFreer


@dataclass(frozen=True)
class VerifySingleLayerCapture:
    """The optional boundary and output capture operations."""

    boundary: BoundaryCapture
    output: OutputCapture


@dataclass(frozen=True)
class VerifySingleLayerResult:
    """Final borrowed endpoints and the captured per-layer host maps."""

    hidden: object
    hidden_f32: object | None
    use_f32_residual: bool
    layer_output_hidden: dict[int, object]
    layer_boundary_hidden: dict[int, dict[str, object]]


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _add_verify_stage(
    stage_timings: dict[str, float],
    name: str,
    ms: float,
) -> None:
    if ms < 0.0:
        raise RuntimeError(f"negative target block stage timing for {name}: {ms}")
    stage_timings[name] = stage_timings.get(name, 0.0) + float(ms)


def _row_ptr(base_ptr: int, scratch_row_start: int, row_nbytes: int) -> int:
    return int(base_ptr) + int(scratch_row_start) * int(row_nbytes)


def run_verify_single_layer_stack(
    geometry: VerifySingleLayerGeometry,
    hidden: VerifySingleLayerHidden,
    scratch: VerifySingleLayerScratch,
    sidecar: VerifySingleLayerSidecar,
    capture: VerifySingleLayerCapture,
    run_native_attention_bulk_ffn_rows: NativeAttentionBulkFfnRowsOperation,
    run_linear_attention_prefill_rows: LinearAttentionPrefillRowsOperation,
    run_full_attention_decode_batch_rows: FullAttentionDecodeBatchRowsOperation,
    run_full_attention_prefill_aotriton: FullAttentionPrefillAotritonOperation,
    *,
    resolve_next_norm_weight: NextNormWeightResolver,
    resolve_linear_state_rows: LinearStateRowsResolver,
    resolve_initial_state_snapshot: InitialStateSnapshotResolver,
    resolve_full_attention_cache: FullAttentionCacheResolver,
    synchronize: Synchronizer,
    stage_timings: dict[str, float] | None,
) -> VerifySingleLayerResult:
    """Run every layer in order and return the final endpoints and captures.

    The walk deliberately performs no whole-plan validation: each lookup and
    operation runs at the same point as the inline loop, so a failure exposes
    the same partially mutated state.
    """

    layer_types = geometry.layer_types
    rows = int(geometry.rows)
    start = int(geometry.start)
    end = int(geometry.end)
    scratch_row_start = int(geometry.scratch_row_start)
    row_nbytes = int(geometry.row_nbytes)
    f32_residual_layer_limit = int(geometry.f32_residual_layer_limit)
    bulk_attention_mode = geometry.bulk_attention_mode
    capture_linear_state_rows = bool(geometry.capture_linear_state_rows)
    defer_linear_state_commit = bool(geometry.defer_linear_state_commit)
    chain_dense_next_rms = bool(geometry.chain_dense_next_rms)
    capture_layer_ids = geometry.capture_layer_ids
    capture_layer_boundary_ids = geometry.capture_layer_boundary_ids

    src = hidden.first
    src_ptr = int(hidden.first_ptr)
    dst = hidden.second
    dst_ptr = int(hidden.second_ptr)
    src_f32 = hidden.first_f32
    src_f32_ptr = hidden.first_f32_ptr
    dst_f32 = hidden.second_f32
    dst_f32_ptr = hidden.second_f32_ptr
    use_f32_residual = bool(hidden.use_f32_residual)
    if not use_f32_residual:
        src_f32 = None
        src_f32_ptr = None
        dst_f32 = None
        dst_f32_ptr = None
    input_norm_ptr: int | None = None
    layer_output_hidden: dict[int, object] = {}
    layer_boundary_hidden: dict[int, dict[str, object]] = {}

    for layer_id, layer_type in enumerate(layer_types):
        # The F32 layer-limit transition happens before the pre-timer sync and
        # before the sidecar load, exactly as the inline loop did.
        if use_f32_residual and layer_id >= f32_residual_layer_limit:
            use_f32_residual = False
            src_f32 = None
            src_f32_ptr = None
            dst_f32 = None
            dst_f32_ptr = None
        if geometry.sync_stages:
            synchronize()
        t_layer0 = perf_counter() if stage_timings is not None else 0.0

        expert_sidecar = None
        if sidecar.load is not None:
            expert_sidecar = sidecar.load(layer_id)
        try:
            bulk_scratch = (
                scratch.prebuilt
                if scratch.prebuilt is not None
                else scratch.resolve_chunk(start, rows, end)
            )
            if bulk_attention_mode == "native" and scratch.native_conv_out is not None:
                bulk_scratch = replace(
                    bulk_scratch, conv_out=scratch.native_conv_out
                )
            src_chunk_ptr = _row_ptr(src_ptr, scratch_row_start, row_nbytes)
            dst_chunk_ptr = _row_ptr(dst_ptr, scratch_row_start, row_nbytes)
            src_f32_chunk_ptr = (
                None
                if not use_f32_residual or src_f32_ptr is None
                else int(src_f32_ptr)
            )
            dst_f32_chunk_ptr = (
                None
                if not use_f32_residual or dst_f32_ptr is None
                else int(dst_f32_ptr)
            )
            next_norm_weight_ptr: int | None = None
            next_norm_out_ptr: int | None = None
            if chain_dense_next_rms and layer_id + 1 < len(layer_types):
                next_norm_weight_ptr = int(
                    resolve_next_norm_weight(layer_id + 1)
                )
                next_norm_out_ptr = int(bulk_scratch.norm.ptr)

            if bulk_attention_mode == "native":
                run_native_attention_bulk_ffn_rows(
                    layer_id,
                    layer_type,
                    src_chunk_ptr,
                    dst_chunk_ptr,
                    bulk_scratch,
                    rows=rows,
                    start_position=start,
                    stream=geometry.stream,
                    decode_scratch=scratch.decode_scratch,
                    linear_state_rows=(
                        resolve_linear_state_rows(layer_id)
                        if capture_linear_state_rows
                        else None
                    ),
                    initial_state_snapshot=(
                        resolve_initial_state_snapshot(layer_id)
                        if capture_linear_state_rows
                        and defer_linear_state_commit
                        else None
                    ),
                    commit_final_linear_state=not defer_linear_state_commit,
                    hidden_f32_ptr=src_f32_chunk_ptr,
                    out_f32_ptr=dst_f32_chunk_ptr,
                    input_norm_ptr=input_norm_ptr,
                    next_norm_weight_ptr=next_norm_weight_ptr,
                    next_norm_out_ptr=next_norm_out_ptr,
                    decode_row_scratches=geometry.decode_row_scratches,
                    attention_context_limit=geometry.attention_context_limit,
                    row_views_keep_scalar_caps=geometry.row_views_keep_scalar_caps,
                    verify_f32_post_norm=geometry.verify_f32_post_norm,
                )
            elif layer_type == LINEAR_ATTENTION:
                run_linear_attention_prefill_rows(
                    layer_id,
                    src_chunk_ptr,
                    dst_chunk_ptr,
                    bulk_scratch,
                    rows=rows,
                    stream=geometry.stream,
                    decode_scratch=scratch.decode_scratch,
                    expert_sidecar=expert_sidecar,
                    linear_state_rows=(
                        resolve_linear_state_rows(layer_id)
                        if capture_linear_state_rows
                        else None
                    ),
                    prefill_gdn_capture=geometry.capture_prefill_gdn,
                    commit_final_linear_state=not defer_linear_state_commit,
                    hidden_f32_ptr=src_f32_chunk_ptr,
                    out_f32_ptr=dst_f32_chunk_ptr,
                    stage_timings=stage_timings,
                    sync_stage_timings=bool(geometry.sync_stage_timings),
                    stage_prefix=LINEAR_LAYER_STAGE_PREFIX,
                    verify_f32_post_norm=geometry.verify_f32_post_norm,
                )
            elif layer_type == FULL_ATTENTION:
                key_cache, value_cache = resolve_full_attention_cache(layer_id)
                layer_scratch = replace(
                    bulk_scratch,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    cos_table=scratch.cos_table,
                    sin_table=scratch.sin_table,
                )
                if end < 1024:
                    run_full_attention_decode_batch_rows(
                        layer_id,
                        src_chunk_ptr,
                        dst_chunk_ptr,
                        layer_scratch,
                        stream=geometry.stream,
                        expert_sidecar=expert_sidecar,
                        stage_timings=stage_timings,
                        sync_stage_timings=bool(geometry.sync_stage_timings),
                        stage_prefix=FULL_ATTENTION_LAYER_STAGE_PREFIX,
                        hidden_f32_ptr=src_f32_chunk_ptr,
                        out_f32_ptr=dst_f32_chunk_ptr,
                        verify_f32_post_norm=geometry.verify_f32_post_norm,
                    )
                else:
                    if use_f32_residual:
                        raise RuntimeError(
                            "HIPENGINE_GGUF_VERIFY_F32_RESIDUAL currently supports "
                            "the full-attention decode-batch verifier path only "
                            "(context < 1024)"
                        )
                    run_full_attention_prefill_aotriton(
                        layer_id,
                        src_chunk_ptr,
                        dst_chunk_ptr,
                        layer_scratch,
                        stream=geometry.stream,
                        expert_sidecar=expert_sidecar,
                        verify_f32_post_norm=geometry.verify_f32_post_norm,
                    )
            else:
                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
        finally:
            if expert_sidecar is not None:
                sidecar.free(expert_sidecar)
            if stage_timings is not None:
                layer_ms = (perf_counter() - t_layer0) * 1000
                if layer_type == LINEAR_ATTENTION:
                    _add_verify_stage(
                        stage_timings, LINEAR_ATTENTION_STAGE, layer_ms
                    )
                elif layer_type == FULL_ATTENTION:
                    _add_verify_stage(
                        stage_timings, FULL_ATTENTION_STAGE, layer_ms
                    )
                else:
                    _add_verify_stage(stage_timings, OTHER_LAYER_STAGE, layer_ms)
                _add_verify_stage(stage_timings, LAYER_TOTAL_STAGE, layer_ms)

        if layer_id in capture_layer_boundary_ids:
            layer_boundary_hidden[int(layer_id)] = capture.boundary(
                int(layer_id),
                str(layer_type),
                hidden_in_ptr=src_chunk_ptr,
                hidden_in_f32_ptr=src_f32_chunk_ptr,
                layer_out_ptr=dst_chunk_ptr,
                layer_out_f32_ptr=dst_f32_chunk_ptr,
                scratch=bulk_scratch,
                prefill_gdn_capture=geometry.capture_prefill_gdn,
                verify_f32_post_norm=geometry.use_f32_post_norm,
            )
        input_norm_ptr = next_norm_out_ptr if chain_dense_next_rms else None
        src, dst = dst, src
        src_ptr, dst_ptr = dst_ptr, src_ptr
        if use_f32_residual:
            src_f32, dst_f32 = dst_f32, src_f32
            src_f32_ptr, dst_f32_ptr = dst_f32_ptr, src_f32_ptr
        if layer_id in capture_layer_ids:
            layer_output_hidden[int(layer_id)] = capture.output(
                int(layer_id),
                hidden=src,
                hidden_f32=src_f32,
                use_f32_residual=use_f32_residual,
            )

    return VerifySingleLayerResult(
        hidden=src,
        hidden_f32=src_f32,
        use_f32_residual=use_f32_residual,
        layer_output_hidden=layer_output_hidden,
        layer_boundary_hidden=layer_boundary_hidden,
    )
