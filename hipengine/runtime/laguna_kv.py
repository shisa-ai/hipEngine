"""Torch-free BF16 KV ownership for Laguna global and sliding layers."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import (
    backend_package_capability,
    load_backend_kernel_package,
)
from hipengine.kernels.registry import resolve
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION

_GLOBAL_BLOCK_SIZE = 256
_LAGUNA_KV_HEADS = 8
_LAGUNA_HEAD_DIM = 128
_BASELINE_GLOBAL_PREFILL_VARIANT = "global_context_rows_spans"
_CACHED_GLOBAL_PREFILL_VARIANT = "global_context_rows_qrow4_cached_online_spans"
_CACHED_META_GLOBAL_PREFILL_VARIANT = (
    "global_context_rows_qrow4_cached_meta_online_spans"
)
_CACHED_META_GLOBAL_QROW6_PREFILL_VARIANT = (
    "global_context_rows_qrow6_cached_meta_online_spans"
)
_DENSE_INITIAL_GLOBAL_PREFILL_VARIANT = (
    "global_context_rows_qrow4_dense_initial_online_spans"
)
_DENSE_INITIAL_GLOBAL_QROW6_PREFILL_VARIANT = (
    "global_context_rows_qrow6_dense_initial_online_spans"
)
_GLOBAL_PREFILL_VARIANTS = frozenset(
    {
        _BASELINE_GLOBAL_PREFILL_VARIANT,
        "global_context_rows_qrow2_online_spans",
        "global_context_rows_qrow4_online_spans",
        "global_context_rows_qrow4_m128_online_spans",
    }
)
_BASELINE_SWA_DECODE_VARIANT = "swa_context_spans"
_SWA_DECODE_VARIANTS = frozenset(
    {
        _BASELINE_SWA_DECODE_VARIANT,
        "swa_context_token4_exact_spans",
    }
)
_BASELINE_SWA_PREFILL_VARIANT = "swa_context_rows_spans"
_CACHED_SWA_PREFILL_VARIANT = "swa_context_rows_qrow4_cached_online_spans"
_CACHED_META_SWA_PREFILL_VARIANT = "swa_context_rows_qrow4_cached_meta_online_spans"
_DENSE_INITIAL_SWA_PREFILL_VARIANT = (
    "swa_context_rows_qrow4_dense_initial_online_spans"
)
_SWA_PREFILL_VARIANTS = frozenset(
    {
        _BASELINE_SWA_PREFILL_VARIANT,
        "swa_context_rows_wave32_exact_spans",
        "swa_context_rows_qrow2_exact_spans",
        "swa_context_rows_qrow2_m128_c128_exact_spans",
        "swa_context_rows_qrow2_online_spans",
        "swa_context_rows_qrow4_online_spans",
        "swa_context_rows_qrow4_sourcequal_online_spans",
        "swa_context_rows_qrow4_m128_online_spans",
    }
)


class _LagunaKVConfig(Protocol):
    block_count: int
    layer_types: tuple[str, ...]
    head_counts: tuple[int, ...]
    head_count_kv: int
    key_length: int
    value_length: int
    sliding_window: int


@dataclass(frozen=True)
class LagunaKVLayerState:
    """Owned payload and spans for one Laguna attention layer."""

    layer_id: int
    attention_type: str
    q_heads: int
    capacity: int
    physical_capacity: int
    key_cache: DeviceBuffer
    value_cache: DeviceBuffer
    append_spans: KVLiveSpans
    spans: KVLiveSpans
    write_variant: str
    write_rows_variant: str
    attention_variant: str
    attention_prefill_variant: str

    @property
    def payload_nbytes(self) -> int:
        return self.key_cache.nbytes + self.value_cache.nbytes


class LagunaKVCache:
    """One request's 12-global/36-SWA token-serial BF16 KV owner."""

    def __init__(
        self,
        *,
        layers: tuple[LagunaKVLayerState, ...],
        buffers: tuple[DeviceBuffer, ...],
        context_length: int,
        sliding_window: int,
        backend: str,
        prefill_cached_meta: bool,
        prefill_global_qrow6: bool,
        prefill_dense_initial: bool,
        row_position: DeviceBuffer,
        runtime: HipRuntime,
    ) -> None:
        self.layers = layers
        self._buffers = buffers
        self.context_length = int(context_length)
        self.sliding_window = int(sliding_window)
        self.backend = str(backend)
        self.prefill_cached_meta = bool(prefill_cached_meta)
        self.prefill_global_qrow6 = bool(prefill_global_qrow6)
        self.prefill_dense_initial = bool(prefill_dense_initial)
        self._row_position = row_position
        self.runtime = runtime
        self.position = -1
        self._pending_positions: tuple[int, ...] = ()
        self._dense_initial_metadata_valid = True
        self._closed = False

    @property
    def payload_nbytes(self) -> int:
        return sum(layer.payload_nbytes for layer in self.layers)

    @property
    def resident_nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self._buffers)

    @property
    def allocation_count(self) -> int:
        return len(self._buffers)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_positions(self) -> tuple[int, ...]:
        return self._pending_positions

    def layer(self, layer_id: int) -> LagunaKVLayerState:
        self._check_open()
        layer = int(layer_id)
        if layer < 0 or layer >= len(self.layers):
            raise IndexError(f"layer_id {layer} outside [0, {len(self.layers)})")
        return self.layers[layer]

    def prepare_position(self, position: int) -> None:
        """Publish one consecutive absolute token position to every layer."""

        self._check_open()
        if self._pending_positions:
            raise RuntimeError("cannot prepare one token while bulk positions are pending")
        parsed = int(position)
        if parsed < 0 or parsed >= self.context_length:
            raise ValueError("position must be within the admitted context")
        if parsed != self.position + 1:
            raise ValueError(
                f"Laguna KV owner is token-serial: expected {self.position + 1}, got {parsed}"
            )
        _copy_i64(self._row_position, parsed, self.runtime)
        self.position = parsed

    def prepare_rows(self, positions: Sequence[int]) -> None:
        """Publish one consecutive matrix chunk without committing its final position.

        A matrix transaction may span more rows than the SWA ring, provided
        every attention/write operation consumes a resident-position-backed
        slice no larger than the ring. This keeps projection/MoE batching
        independent from the physical KV window.
        """

        self._check_open()
        if self._pending_positions:
            raise RuntimeError("Laguna KV bulk positions are already pending")
        parsed = tuple(int(position) for position in positions)
        if not parsed:
            raise ValueError("Laguna KV bulk positions must be non-empty")
        expected_start = self.position + 1
        if parsed != tuple(range(expected_start, expected_start + len(parsed))):
            raise ValueError(
                f"Laguna KV bulk positions must be consecutive from {expected_start}"
            )
        if parsed[-1] >= self.context_length:
            raise ValueError("Laguna KV bulk positions exceed the admitted context")
        _copy_i64(self._row_position, parsed[0], self.runtime)
        self._pending_positions = parsed

    def commit_rows(self) -> None:
        """Commit the currently prepared chunk after every layer has appended it."""

        self._check_open()
        if not self._pending_positions:
            raise RuntimeError("no Laguna KV bulk positions are pending")
        self.position = self._pending_positions[-1]
        self._pending_positions = ()

    def discard_rows(self) -> None:
        """Discard prepared transient query rows without advancing committed state.

        DFlash noise blocks use ``attend_prefill`` against current K/V rows but
        must not append those speculative query rows to the projected-context
        cache. Clearing only the pending position transaction preserves every
        committed payload/span and allows the same positions to be retried.
        """

        self._check_open()
        if not self._pending_positions:
            raise RuntimeError("no Laguna KV bulk positions are pending")
        self._pending_positions = ()

    def reset(self) -> None:
        """Reset request metadata while retaining payload allocations and addresses."""

        self._check_open()
        if self._pending_positions:
            raise RuntimeError("cannot reset Laguna KV while bulk rows are pending")
        _copy_i64(self._row_position, -1, self.runtime)
        seen: set[tuple[int, int, int]] = set()
        for state in self.layers:
            spans = state.spans
            signature = (
                spans.live_counts.ptr,
                spans.token_positions.ptr,
                spans.evict_mask.ptr,
            )
            if signature in seen:
                continue
            seen.add(signature)
            self.runtime.memset(
                spans.live_counts.ptr,
                0,
                spans.live_counts.numel * spans.live_counts.dtype.itemsize,
            )
            self.runtime.memset(
                spans.token_positions.ptr,
                0xFF,
                spans.token_positions.numel * spans.token_positions.dtype.itemsize,
            )
            self.runtime.memset(
                spans.evict_mask.ptr,
                1,
                spans.evict_mask.numel * spans.evict_mask.dtype.itemsize,
            )
        self.position = -1
        self._dense_initial_metadata_valid = True

    def append(
        self,
        layer_id: int,
        key_ptr: int,
        value_ptr: int,
        *,
        stream: int = 0,
        library=None,
    ) -> None:
        """Append one layer's current F32 K/V row into its BF16 cache."""

        state = self.layer(layer_id)
        self._check_prepared()
        fn = self._resolve("laguna_kv_write", state.write_variant)
        fn(
            key_ptr,
            value_ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            state.append_spans,
            _LAGUNA_KV_HEADS,
            _LAGUNA_HEAD_DIM,
            stream=stream,
            library=library,
            runtime=self.runtime,
        )

    def append_rows(
        self,
        layer_id: int,
        key_ptr: int,
        value_ptr: int,
        rows: int,
        *,
        row_offset: int = 0,
        row_positions_ptr: int | None = None,
        stream: int = 0,
        library=None,
    ) -> None:
        """Append all or one resident-position-backed slice of current F32 K/V rows."""

        state = self.layer(layer_id)
        spans = self._bulk_slice_spans(
            state.append_spans,
            row_offset=row_offset,
            rows=rows,
            row_positions_ptr=row_positions_ptr,
        )
        fn = self._resolve("laguna_kv_write", state.write_rows_variant)
        fn(
            key_ptr,
            value_ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            spans,
            int(rows),
            _LAGUNA_KV_HEADS,
            _LAGUNA_HEAD_DIM,
            stream=stream,
            library=library,
            runtime=self.runtime,
        )

    def attend(
        self,
        layer_id: int,
        query_ptr: int,
        out_ptr: int,
        *,
        scale: float = _LAGUNA_HEAD_DIM**-0.5,
        stream: int = 0,
        library=None,
    ) -> None:
        """Run ungated global or SWA context attention for the current row."""

        state = self.layer(layer_id)
        self._check_prepared()
        fn = self._resolve("laguna_attention_decode", state.attention_variant)
        common = (
            query_ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            out_ptr,
            state.spans,
        )
        if state.attention_type == FULL_ATTENTION:
            fn(
                *common,
                self.context_length,
                state.q_heads,
                _LAGUNA_KV_HEADS,
                _LAGUNA_HEAD_DIM,
                scale,
                stream=stream,
                library=library,
                runtime=self.runtime,
            )
        else:
            fn(
                *common,
                state.q_heads,
                _LAGUNA_KV_HEADS,
                _LAGUNA_HEAD_DIM,
                scale,
                sliding_window=self.sliding_window,
                stream=stream,
                library=library,
                runtime=self.runtime,
            )

    def attend_prefill(
        self,
        layer_id: int,
        query_ptr: int,
        current_key_ptr: int,
        current_value_ptr: int,
        out_ptr: int,
        rows: int,
        *,
        row_offset: int = 0,
        row_positions_ptr: int | None = None,
        scale: float = _LAGUNA_HEAD_DIM**-0.5,
        stream: int = 0,
        library=None,
    ) -> None:
        """Run causal attention over all or one resident-position-backed row slice."""

        state = self.layer(layer_id)
        spans = self._bulk_slice_spans(
            state.spans,
            row_offset=row_offset,
            rows=rows,
            row_positions_ptr=row_positions_ptr,
        )
        fn = self._resolve("laguna_attention_prefill", state.attention_prefill_variant)
        common = (
            query_ptr,
            current_key_ptr,
            current_value_ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            out_ptr,
            spans,
            int(rows),
        )
        if state.attention_type == FULL_ATTENTION:
            fn(
                *common,
                self.context_length,
                state.q_heads,
                _LAGUNA_KV_HEADS,
                _LAGUNA_HEAD_DIM,
                scale,
                stream=stream,
                library=library,
                runtime=self.runtime,
            )
        else:
            fn(
                *common,
                state.q_heads,
                _LAGUNA_KV_HEADS,
                _LAGUNA_HEAD_DIM,
                scale,
                sliding_window=self.sliding_window,
                start_position=int(self._pending_positions[int(row_offset)]),
                stream=stream,
                library=library,
                runtime=self.runtime,
            )

    def can_preappend_prefill(
        self,
        layer_id: int,
        rows: int,
        *,
        row_offset: int = 0,
    ) -> bool:
        """Return whether one M128 tile can be cached before causal attention."""

        state = self.layer(layer_id)
        offset = int(row_offset)
        count = int(rows)
        if (
            count != 128
            or not self._pending_positions
            or offset < 0
            or offset + count > len(self._pending_positions)
        ):
            return False
        start_position = self._pending_positions[offset]
        return start_position + count <= state.capacity

    def can_dense_initial_prefill(
        self,
        layer_id: int,
        rows: int,
        *,
        row_offset: int = 0,
    ) -> bool:
        """Return whether cached metadata still has its initial dense identity."""

        if (
            not self._dense_initial_metadata_valid
            or not self.can_preappend_prefill(
                layer_id,
                rows,
                row_offset=row_offset,
            )
        ):
            return False
        offset = int(row_offset)
        count = int(rows)
        start_position = self._pending_positions[offset]
        return (
            start_position >= 0
            and self._pending_positions[offset : offset + count]
            == tuple(range(start_position, start_position + count))
        )

    def dense_initial_prefill_view(
        self,
        layer_id: int,
        rows: int,
        *,
        row_offset: int = 0,
        row_positions_ptr: int | None = None,
    ) -> tuple[LagunaKVLayerState, KVLiveSpans, int]:
        """Return the qualified state/span/start tuple for a composite route."""

        if not self.can_dense_initial_prefill(
            layer_id,
            rows,
            row_offset=row_offset,
        ):
            raise ValueError("Laguna dense-initial prefill view is not qualified")
        state = self.layer(layer_id)
        spans = self._bulk_slice_spans(
            state.spans,
            row_offset=row_offset,
            rows=rows,
            row_positions_ptr=row_positions_ptr,
        )
        return state, spans, int(self._pending_positions[int(row_offset)])

    def can_rolling_swa_prefill(
        self,
        layer_id: int,
        rows: int,
        *,
        row_offset: int = 0,
    ) -> bool:
        """Return whether one M128 tile has an intact rolling 512-token window."""

        state = self.layer(layer_id)
        offset = int(row_offset)
        count = int(rows)
        if (
            state.attention_type != SLIDING_ATTENTION
            or not self._dense_initial_metadata_valid
            or count != 128
            or state.capacity != self.sliding_window
            or self.sliding_window != 512
            or not self._pending_positions
            or offset < 0
            or offset + count > len(self._pending_positions)
        ):
            return False
        start_position = self._pending_positions[offset]
        return (
            start_position >= self.sliding_window
            and self._pending_positions[offset : offset + count]
            == tuple(range(start_position, start_position + count))
        )

    def rolling_swa_prefill_view(
        self,
        layer_id: int,
        rows: int,
        *,
        row_offset: int = 0,
        row_positions_ptr: int | None = None,
    ) -> tuple[LagunaKVLayerState, KVLiveSpans, int]:
        """Return the preappend rolling-SWA state/span/start tuple."""

        if not self.can_rolling_swa_prefill(
            layer_id,
            rows,
            row_offset=row_offset,
        ):
            raise ValueError("Laguna rolling-SWA prefill view is not qualified")
        state = self.layer(layer_id)
        spans = self._bulk_slice_spans(
            state.spans,
            row_offset=row_offset,
            rows=rows,
            row_positions_ptr=row_positions_ptr,
        )
        return state, spans, int(self._pending_positions[int(row_offset)])

    def attend_prefill_cached(
        self,
        layer_id: int,
        query_ptr: int,
        current_key_ptr: int,
        current_value_ptr: int,
        out_ptr: int,
        rows: int,
        *,
        row_offset: int = 0,
        row_positions_ptr: int | None = None,
        scale: float = _LAGUNA_HEAD_DIM**-0.5,
        stream: int = 0,
        library=None,
    ) -> None:
        """Run qrow4 causal attention after a safe M128 tile was appended."""

        state = self.layer(layer_id)
        if not self.can_preappend_prefill(
            layer_id,
            rows,
            row_offset=row_offset,
        ):
            raise ValueError("cached Laguna prefill requires one safe M128 tile")
        spans = self._bulk_slice_spans(
            state.spans,
            row_offset=row_offset,
            rows=rows,
            row_positions_ptr=row_positions_ptr,
        )
        start_position = int(self._pending_positions[int(row_offset)])
        dense_initial = (
            self.prefill_dense_initial
            and self.can_dense_initial_prefill(
                layer_id,
                rows,
                row_offset=row_offset,
            )
        )
        if state.attention_type == FULL_ATTENTION:
            if dense_initial:
                variant = (
                    _DENSE_INITIAL_GLOBAL_QROW6_PREFILL_VARIANT
                    if self.prefill_global_qrow6 and start_position >= 128
                    else _DENSE_INITIAL_GLOBAL_PREFILL_VARIANT
                )
            elif self.prefill_cached_meta and start_position >= 128:
                variant = (
                    _CACHED_META_GLOBAL_QROW6_PREFILL_VARIANT
                    if self.prefill_global_qrow6
                    else _CACHED_META_GLOBAL_PREFILL_VARIANT
                )
            else:
                variant = _CACHED_GLOBAL_PREFILL_VARIANT
        else:
            variant = (
                _DENSE_INITIAL_SWA_PREFILL_VARIANT
                if dense_initial
                else _CACHED_META_SWA_PREFILL_VARIANT
                if self.prefill_cached_meta
                else _CACHED_SWA_PREFILL_VARIANT
            )
        fn = self._resolve("laguna_attention_prefill", variant)
        common = (
            query_ptr,
            current_key_ptr,
            current_value_ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            out_ptr,
            spans,
            int(rows),
        )
        if state.attention_type == FULL_ATTENTION:
            extra = {"start_position": start_position} if dense_initial else {}
            fn(
                *common,
                self.context_length,
                state.q_heads,
                _LAGUNA_KV_HEADS,
                _LAGUNA_HEAD_DIM,
                scale,
                **extra,
                stream=stream,
                library=library,
                runtime=self.runtime,
            )
        else:
            fn(
                *common,
                state.q_heads,
                _LAGUNA_KV_HEADS,
                _LAGUNA_HEAD_DIM,
                scale,
                sliding_window=self.sliding_window,
                start_position=start_position,
                stream=stream,
                library=library,
                runtime=self.runtime,
            )

    def evict_position(self, layer_id: int, position: int) -> None:
        """Mark one currently addressable absolute position invisible."""

        state = self.layer(layer_id)
        parsed = int(position)
        if parsed < 0 or parsed > self.position:
            raise ValueError("evicted position must already have been appended")
        if state.attention_type == SLIDING_ATTENTION:
            if parsed <= self.position - state.capacity:
                raise ValueError("evicted position is no longer resident in the SWA ring")
            slot = parsed % state.capacity
        else:
            if parsed >= state.capacity:
                raise ValueError("evicted position is outside the global cache")
            slot = parsed
        marker = ctypes.c_uint8(1)
        self.runtime.memcpy(
            state.spans.evict_mask.ptr + slot,
            ctypes.addressof(marker),
            1,
            HipMemcpyKind.HOST_TO_DEVICE,
        )
        self._dense_initial_metadata_valid = False

    def evict_swa_position(self, layer_id: int, position: int) -> None:
        """Mark one currently addressable absolute SWA position invisible."""

        if self.layer(layer_id).attention_type != SLIDING_ATTENTION:
            raise ValueError("explicit ring eviction is valid only for a sliding layer")
        self.evict_position(layer_id, position)

    def free(self) -> None:
        if self._closed:
            return
        self._closed = True
        for buffer in reversed(self._buffers):
            free(buffer, runtime=self.runtime)

    def __enter__(self) -> "LagunaKVCache":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.free()

    def _resolve(self, layer: str, variant: str):
        fn = resolve(
            backend=self.backend,
            layer=layer,
            quant="bf16",
            variant=variant,
            missing="none",
        )
        if fn is None:
            from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
                register_laguna_kv_attention_kernels,
            )

            register_laguna_kv_attention_kernels()
            load_backend_kernel_package(self.backend)
            fn = resolve(
                backend=self.backend,
                layer=layer,
                quant="bf16",
                variant=variant,
            )
        return fn

    def _check_prepared(self) -> None:
        if self.position < 0:
            raise RuntimeError("prepare_position must run before KV append/attention")
        if self._pending_positions:
            raise RuntimeError("token-serial KV operations cannot run while bulk rows are pending")

    def _bulk_slice_spans(
        self,
        spans: KVLiveSpans,
        *,
        row_offset: int,
        rows: int,
        row_positions_ptr: int | None,
    ) -> KVLiveSpans:
        if not self._pending_positions:
            raise RuntimeError("prepare_rows must run before bulk KV operations")
        offset = int(row_offset)
        count = int(rows)
        if offset < 0 or count <= 0 or offset + count > len(self._pending_positions):
            raise ValueError("bulk KV slice must fit the prepared position count")
        if count > self.sliding_window:
            raise ValueError("bulk KV operations must fit the SWA ring capacity")
        if (
            len(self._pending_positions) > self.sliding_window
            and row_positions_ptr is None
        ):
            raise ValueError(
                "wide bulk KV operations require a resident row_positions_ptr"
            )
        if offset == 0 and count == len(self._pending_positions) and row_positions_ptr is None:
            return spans
        if row_positions_ptr is None:
            raise ValueError("bulk KV slice requires a resident row_positions_ptr")
        assert spans.row_positions is not None
        row_positions = Tensor.from_handle(
            int(row_positions_ptr),
            (1,),
            DType.INT64,
            spans.row_positions.device,
        )
        return replace(spans, row_positions=row_positions)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Laguna KV cache is closed")


def resolve_laguna_global_prefill_variant(
    backend: str,
    requested: str | None = None,
) -> str:
    """Resolve an explicit rollback or the architecture-qualified global default."""

    selected = (
        backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_PREFILL_VARIANT",
            _BASELINE_GLOBAL_PREFILL_VARIANT,
        )
        if requested is None
        else str(requested)
    )
    parsed = str(selected)
    if parsed not in _GLOBAL_PREFILL_VARIANTS:
        raise ValueError("unsupported Laguna global prefill variant")
    return parsed


def resolve_laguna_swa_decode_variant(
    backend: str,
    requested: str | None = None,
) -> str:
    """Resolve an explicit rollback or the architecture-qualified SWA default."""

    selected = (
        backend_package_capability(
            backend,
            "LAGUNA_SWA_DECODE_VARIANT",
            _BASELINE_SWA_DECODE_VARIANT,
        )
        if requested is None
        else str(requested)
    )
    parsed = str(selected)
    if parsed not in _SWA_DECODE_VARIANTS:
        raise ValueError("unsupported Laguna SWA decode variant")
    return parsed


def resolve_laguna_swa_prefill_variant(
    backend: str,
    requested: str | None = None,
) -> str:
    """Resolve an explicit rollback or the architecture-qualified SWA default."""

    selected = (
        backend_package_capability(
            backend,
            "LAGUNA_SWA_PREFILL_VARIANT",
            _BASELINE_SWA_PREFILL_VARIANT,
        )
        if requested is None
        else str(requested)
    )
    parsed = str(selected)
    if parsed not in _SWA_PREFILL_VARIANTS:
        raise ValueError("unsupported Laguna SWA prefill variant")
    return parsed


def allocate_laguna_kv_cache(
    config: _LagunaKVConfig,
    *,
    context_length: int,
    backend: str = "hip_gfx1100",
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    global_prefill_variant: str | None = None,
    swa_decode_variant: str | None = None,
    swa_prefill_variant: str | None = None,
    prefill_cached_meta: bool = False,
    prefill_global_qrow6: bool = False,
    prefill_dense_initial: bool = False,
) -> LagunaKVCache:
    """Allocate per-layer BF16 payloads and complete device span metadata."""

    context = int(context_length)
    if context <= 0:
        raise ValueError("context_length must be positive")
    parsed_global_prefill_variant = resolve_laguna_global_prefill_variant(
        backend,
        global_prefill_variant,
    )
    parsed_swa_decode_variant = resolve_laguna_swa_decode_variant(
        backend,
        swa_decode_variant,
    )
    parsed_swa_prefill_variant = resolve_laguna_swa_prefill_variant(
        backend,
        swa_prefill_variant,
    )
    runtime = runtime or get_hip_runtime()
    device = device or Device("hip", 0)
    layer_types, head_counts, sliding_window = _validate_config(config, context)
    has_global = FULL_ATTENTION in layer_types
    has_sliding = SLIDING_ATTENTION in layer_types
    buffers: list[DeviceBuffer] = []

    def allocate_raw(nbytes: int) -> DeviceBuffer:
        buffer = malloc(int(nbytes), runtime=runtime)
        buffers.append(buffer)
        return buffer

    def metadata(values: ctypes.Array, shape: tuple[int, ...], dtype: DType) -> Tensor:
        buffer = allocate_raw(ctypes.sizeof(values))
        runtime.memcpy(
            buffer.ptr,
            ctypes.addressof(values),
            buffer.nbytes,
            HipMemcpyKind.HOST_TO_DEVICE,
        )
        return Tensor.from_handle(buffer.ptr, shape, dtype, device)

    try:
        global_blocks = (context + _GLOBAL_BLOCK_SIZE - 1) // _GLOBAL_BLOCK_SIZE
        global_physical_capacity = global_blocks * _GLOBAL_BLOCK_SIZE
        global_offsets = (
            metadata(
                (ctypes.c_int32 * global_blocks)(*range(global_blocks)),
                (global_blocks,),
                DType.INT32,
            )
            if has_global
            else None
        )
        sliding_offsets = (
            metadata(
                (ctypes.c_int32 * sliding_window)(*range(sliding_window)),
                (sliding_window,),
                DType.INT32,
            )
            if has_sliding
            else None
        )
        row_position = metadata((ctypes.c_int64 * 1)(-1), (1,), DType.INT64)

        states: list[LagunaKVLayerState] = []
        element_bytes = DType.BF16.itemsize
        for layer_id, (attention_type, q_heads) in enumerate(
            zip(layer_types, head_counts, strict=True)
        ):
            if attention_type == FULL_ATTENTION:
                capacity = context
                physical_capacity = global_physical_capacity
                payload_elements = physical_capacity * _LAGUNA_KV_HEADS * _LAGUNA_HEAD_DIM
                key_cache = allocate_raw(payload_elements * element_bytes)
                value_cache = allocate_raw(payload_elements * element_bytes)
                live_counts = metadata(
                    (ctypes.c_int64 * 1)(0),
                    (1,),
                    DType.INT64,
                )
                token_positions = metadata(
                    (ctypes.c_int64 * capacity)(*([-1] * capacity)),
                    (capacity,),
                    DType.INT64,
                )
                evict_mask = metadata(
                    (ctypes.c_uint8 * capacity)(*([1] * capacity)),
                    (capacity,),
                    DType.BOOL,
                )
                decode_spans = KVLiveSpans.paged_dense(
                    block_table=global_offsets,
                    live_counts=live_counts,
                    token_positions=token_positions,
                    evict_mask=evict_mask,
                    row_positions=row_position,
                    capacity=capacity,
                    block_size=_GLOBAL_BLOCK_SIZE,
                    storage_dtype=DType.BF16,
                )
                append_spans = decode_spans
                write_variant = "global_f32_spans"
                write_rows_variant = "global_f32_rows_spans"
                attention_variant = "global_context_spans"
                attention_prefill_variant = parsed_global_prefill_variant
            else:
                capacity = sliding_window
                physical_capacity = sliding_window
                payload_elements = capacity * _LAGUNA_KV_HEADS * _LAGUNA_HEAD_DIM
                key_cache = allocate_raw(payload_elements * element_bytes)
                value_cache = allocate_raw(payload_elements * element_bytes)
                live_counts = metadata(
                    (ctypes.c_int64 * 1)(0),
                    (1,),
                    DType.INT64,
                )
                token_positions = metadata(
                    (ctypes.c_int64 * capacity)(*([-1] * capacity)),
                    (capacity,),
                    DType.INT64,
                )
                evict_mask = metadata(
                    (ctypes.c_uint8 * capacity)(*([1] * capacity)),
                    (capacity,),
                    DType.BOOL,
                )
                decode_spans = KVLiveSpans.sliding_ring(
                    base_offsets=sliding_offsets,
                    live_counts=live_counts,
                    token_positions=token_positions,
                    evict_mask=evict_mask,
                    row_positions=row_position,
                    capacity=capacity,
                    storage_dtype=DType.BF16,
                )
                append_spans = decode_spans
                write_variant = "swa_f32_spans"
                write_rows_variant = "swa_f32_rows_spans"
                attention_variant = parsed_swa_decode_variant
                attention_prefill_variant = parsed_swa_prefill_variant
            states.append(
                LagunaKVLayerState(
                    layer_id=layer_id,
                    attention_type=attention_type,
                    q_heads=q_heads,
                    capacity=capacity,
                    physical_capacity=physical_capacity,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    append_spans=append_spans,
                    spans=decode_spans,
                    write_variant=write_variant,
                    write_rows_variant=write_rows_variant,
                    attention_variant=attention_variant,
                    attention_prefill_variant=attention_prefill_variant,
                )
            )

        return LagunaKVCache(
            layers=tuple(states),
            buffers=tuple(buffers),
            context_length=context,
            sliding_window=sliding_window,
            backend=backend,
            prefill_cached_meta=prefill_cached_meta,
            prefill_global_qrow6=prefill_global_qrow6,
            prefill_dense_initial=prefill_dense_initial,
            row_position=_buffer_for_tensor(row_position, buffers),
            runtime=runtime,
        )
    except Exception:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        raise


def _validate_config(
    config: _LagunaKVConfig,
    context_length: int,
) -> tuple[tuple[str, ...], tuple[int, ...], int]:
    layer_types = tuple(str(value) for value in config.layer_types)
    head_counts = tuple(int(value) for value in config.head_counts)
    if len(layer_types) != int(config.block_count) or len(head_counts) != len(layer_types):
        raise ValueError("Laguna KV config layer arrays must match block_count")
    if not layer_types:
        raise ValueError("Laguna KV config must contain at least one layer")
    if int(config.head_count_kv) != _LAGUNA_KV_HEADS:
        raise ValueError("Laguna KV config requires eight KV heads")
    if int(config.key_length) != _LAGUNA_HEAD_DIM or int(config.value_length) != _LAGUNA_HEAD_DIM:
        raise ValueError("Laguna KV config requires 128-wide K/V heads")
    sliding_window = int(config.sliding_window)
    if sliding_window != 512:
        raise ValueError("Laguna S 2.1 KV config requires a 512-token sliding window")
    if context_length < 1:
        raise ValueError("context_length must be positive")
    for layer_id, (attention_type, q_heads) in enumerate(
        zip(layer_types, head_counts, strict=True)
    ):
        if attention_type not in {FULL_ATTENTION, SLIDING_ATTENTION}:
            raise ValueError(f"unsupported Laguna attention type at layer {layer_id}")
        expected_heads = 48 if attention_type == FULL_ATTENTION else 72
        if q_heads != expected_heads:
            raise ValueError(f"Laguna layer {layer_id} requires {expected_heads} query heads")
    return layer_types, head_counts, sliding_window


def _copy_i64(buffer: DeviceBuffer, value: int, runtime: HipRuntime) -> None:
    host = ctypes.c_int64(int(value))
    runtime.memcpy(
        buffer.ptr,
        ctypes.addressof(host),
        ctypes.sizeof(host),
        HipMemcpyKind.HOST_TO_DEVICE,
    )


def _buffer_for_tensor(
    tensor: Tensor,
    buffers: list[DeviceBuffer],
) -> DeviceBuffer:
    for buffer in buffers:
        if buffer.ptr == tensor.ptr:
            return buffer
    raise RuntimeError("Laguna KV metadata tensor has no owned buffer")


__all__ = [
    "LagunaKVCache",
    "LagunaKVLayerState",
    "allocate_laguna_kv_cache",
    "resolve_laguna_global_prefill_variant",
    "resolve_laguna_swa_decode_variant",
    "resolve_laguna_swa_prefill_variant",
]
