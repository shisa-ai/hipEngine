"""Torch-free BF16 KV ownership for Laguna global and sliding layers."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
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
_BASELINE_SWA_DECODE_VARIANT = "swa_context_spans"
_SWA_DECODE_VARIANTS = frozenset(
    {
        _BASELINE_SWA_DECODE_VARIANT,
        "swa_context_token4_exact_spans",
    }
)
_BASELINE_SWA_PREFILL_VARIANT = "swa_context_rows_spans"
_SWA_PREFILL_VARIANTS = frozenset(
    {
        _BASELINE_SWA_PREFILL_VARIANT,
        "swa_context_rows_wave32_exact_spans",
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
        row_position: DeviceBuffer,
        split_score_scratch: DeviceBuffer | None,
        split_physical_scratch: DeviceBuffer | None,
        global_split_min_live: int | None,
        swa_split_min_live: int | None,
        swa_split_tile16_min_live: int | None,
        split_gate_fusion: bool,
        runtime: HipRuntime,
    ) -> None:
        self.layers = layers
        self._buffers = buffers
        self.context_length = int(context_length)
        self.sliding_window = int(sliding_window)
        self.backend = str(backend)
        self._row_position = row_position
        self._split_score_scratch = split_score_scratch
        self._split_physical_scratch = split_physical_scratch
        self.global_split_min_live = global_split_min_live
        self.swa_split_min_live = swa_split_min_live
        self.swa_split_tile16_min_live = swa_split_tile16_min_live
        self.split_gate_fusion = bool(split_gate_fusion)
        self.runtime = runtime
        self.position = -1
        self._pending_positions: tuple[int, ...] = ()
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
        """Publish one bounded consecutive chunk without committing its final position."""

        self._check_open()
        if self._pending_positions:
            raise RuntimeError("Laguna KV bulk positions are already pending")
        parsed = tuple(int(position) for position in positions)
        if not parsed:
            raise ValueError("Laguna KV bulk positions must be non-empty")
        if len(parsed) > self.sliding_window:
            raise ValueError("Laguna KV bulk rows cannot exceed the SWA ring capacity")
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
        stream: int = 0,
        library=None,
    ) -> None:
        """Append all current F32 K/V rows after bulk attention has consumed them."""

        state = self.layer(layer_id)
        self._check_bulk_rows(rows)
        fn = self._resolve("laguna_kv_write", state.write_rows_variant)
        fn(
            key_ptr,
            value_ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            state.append_spans,
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
        gate_ptr: int | None = None,
        gated_out_ptr: int | None = None,
        scale: float = _LAGUNA_HEAD_DIM**-0.5,
        stream: int = 0,
        library=None,
    ) -> bool:
        """Run context attention, optionally fusing the split-path BF16 gate."""

        state = self.layer(layer_id)
        self._check_prepared()
        if (gate_ptr is None) != (gated_out_ptr is None):
            raise ValueError("gate_ptr and gated_out_ptr must be provided together")
        use_gated = gate_ptr is not None and self.split_gate_fusion
        live_count = min(self.position + 1, state.capacity)
        split_threshold = (
            self.global_split_min_live
            if state.attention_type == FULL_ATTENTION
            else self.swa_split_min_live
        )
        use_split = split_threshold is not None and live_count >= split_threshold
        use_tile16 = (
            state.attention_type == SLIDING_ATTENTION
            and self.swa_split_tile16_min_live is not None
            and live_count >= self.swa_split_tile16_min_live
        )
        common = (
            query_ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            out_ptr,
        )
        if use_split:
            if self._split_score_scratch is None or self._split_physical_scratch is None:
                raise RuntimeError("Laguna split attention scratch is unavailable")
            variant = (
                (
                    "global_context_split_exact_gated_spans"
                    if use_gated
                    else "global_context_split_exact_spans"
                )
                if state.attention_type == FULL_ATTENTION
                else (
                    (
                        "swa_context_split_tile16_exact_gated_spans"
                        if use_gated
                        else "swa_context_split_tile16_exact_spans"
                    )
                    if use_tile16
                    else (
                        "swa_context_split_exact_gated_spans"
                        if use_gated
                        else "swa_context_split_exact_spans"
                    )
                )
            )
            fn = self._resolve("laguna_attention_decode", variant)
            gated_args = (
                (int(gate_ptr), int(gated_out_ptr)) if use_gated else ()
            )
            split_common = (
                *common,
                *gated_args,
                self._split_score_scratch.ptr,
                self._split_physical_scratch.ptr,
                state.spans,
                live_count,
            )
            if state.attention_type == FULL_ATTENTION:
                fn(
                    *split_common,
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
                    *split_common,
                    state.q_heads,
                    _LAGUNA_KV_HEADS,
                    _LAGUNA_HEAD_DIM,
                    scale,
                    sliding_window=self.sliding_window,
                    stream=stream,
                    library=library,
                    runtime=self.runtime,
                )
            return use_gated

        fn = self._resolve("laguna_attention_decode", state.attention_variant)
        fallback_common = (*common, state.spans)
        if state.attention_type == FULL_ATTENTION:
            fn(
                *fallback_common,
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
                *fallback_common,
                state.q_heads,
                _LAGUNA_KV_HEADS,
                _LAGUNA_HEAD_DIM,
                scale,
                sliding_window=self.sliding_window,
                stream=stream,
                library=library,
                runtime=self.runtime,
            )
        return False

    def attend_prefill(
        self,
        layer_id: int,
        query_ptr: int,
        current_key_ptr: int,
        current_value_ptr: int,
        out_ptr: int,
        rows: int,
        *,
        scale: float = _LAGUNA_HEAD_DIM**-0.5,
        stream: int = 0,
        library=None,
    ) -> None:
        """Run causal bulk attention over prior state plus uncommitted current rows."""

        state = self.layer(layer_id)
        self._check_bulk_rows(rows)
        fn = self._resolve("laguna_attention_prefill", state.attention_prefill_variant)
        common = (
            query_ptr,
            current_key_ptr,
            current_value_ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            out_ptr,
            state.spans,
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

    def _check_bulk_rows(self, rows: int) -> None:
        if not self._pending_positions:
            raise RuntimeError("prepare_rows must run before bulk KV operations")
        if int(rows) != len(self._pending_positions):
            raise ValueError("bulk KV rows must match the prepared position count")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Laguna KV cache is closed")


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


def resolve_laguna_split_thresholds(
    backend: str,
    *,
    context_length: int,
    sliding_window: int,
    global_split_min_live: int | None = None,
    swa_split_min_live: int | None = None,
    use_split_attention: bool | None = None,
) -> tuple[int | None, int | None]:
    """Resolve explicit split crossovers or architecture-qualified defaults."""

    context = int(context_length)
    window = min(int(sliding_window), context)
    if context <= 0 or window <= 0:
        raise ValueError("Laguna split threshold capacities must be positive")
    has_explicit_threshold = (
        global_split_min_live is not None or swa_split_min_live is not None
    )
    if use_split_attention is False:
        if has_explicit_threshold:
            raise ValueError(
                "use_split_attention=False cannot be combined with split thresholds"
            )
        return None, None
    if has_explicit_threshold:
        return global_split_min_live, swa_split_min_live

    global_default = backend_package_capability(
        backend,
        "LAGUNA_GLOBAL_SPLIT_MIN_LIVE",
        None,
    )
    swa_default = backend_package_capability(
        backend,
        "LAGUNA_SWA_SPLIT_MIN_LIVE",
        None,
    )
    parsed_global = None if global_default is None else int(global_default)
    parsed_swa = None if swa_default is None else int(swa_default)
    return (
        parsed_global if parsed_global is not None and parsed_global <= context else None,
        parsed_swa if parsed_swa is not None and parsed_swa <= window else None,
    )


def resolve_laguna_swa_split_tile16_threshold(
    backend: str,
    *,
    sliding_window: int,
    swa_split_tile16_min_live: int | None = None,
    use_swa_split_tile16: bool | None = None,
) -> int | None:
    """Resolve the explicit SWA tile16 crossover or backend-qualified default."""

    window = int(sliding_window)
    if window <= 0:
        raise ValueError("Laguna SWA tile16 threshold capacity must be positive")
    if use_swa_split_tile16 is False:
        if swa_split_tile16_min_live is not None:
            raise ValueError(
                "use_swa_split_tile16=False cannot be combined with an explicit threshold"
            )
        return None
    if swa_split_tile16_min_live is not None:
        return int(swa_split_tile16_min_live)
    default = backend_package_capability(
        backend,
        "LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE",
        None,
    )
    parsed = None if default is None else int(default)
    return parsed if parsed is not None and parsed <= window else None


def allocate_laguna_kv_cache(
    config: _LagunaKVConfig,
    *,
    context_length: int,
    backend: str = "hip_gfx1100",
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    swa_decode_variant: str | None = None,
    swa_prefill_variant: str | None = None,
    global_split_min_live: int | None = None,
    swa_split_min_live: int | None = None,
    swa_split_tile16_min_live: int | None = None,
    use_swa_split_tile16: bool | None = None,
    use_split_attention: bool | None = None,
    use_split_gate_fusion: bool | None = None,
) -> LagunaKVCache:
    """Allocate per-layer BF16 payloads and complete device span metadata."""

    context = int(context_length)
    if context <= 0:
        raise ValueError("context_length must be positive")
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
    if use_split_attention is False and (
        swa_split_tile16_min_live is not None or use_swa_split_tile16 is True
    ):
        raise ValueError(
            "use_split_attention=False cannot be combined with split thresholds"
        )
    selected_global_split, selected_swa_split = resolve_laguna_split_thresholds(
        backend,
        context_length=context,
        sliding_window=sliding_window,
        global_split_min_live=global_split_min_live,
        swa_split_min_live=swa_split_min_live,
        use_split_attention=use_split_attention,
    )
    parsed_global_split = _validate_split_threshold(
        selected_global_split,
        context,
        "global_split_min_live",
    )
    parsed_swa_split = _validate_split_threshold(
        selected_swa_split,
        sliding_window,
        "swa_split_min_live",
    )
    selected_swa_tile16 = (
        None
        if use_split_attention is False
        else resolve_laguna_swa_split_tile16_threshold(
            backend,
            sliding_window=sliding_window,
            swa_split_tile16_min_live=swa_split_tile16_min_live,
            use_swa_split_tile16=use_swa_split_tile16,
        )
    )
    parsed_swa_tile16 = _validate_split_threshold(
        selected_swa_tile16,
        sliding_window,
        "swa_split_tile16_min_live",
    )
    selected_split_gate_fusion = bool(
        backend_package_capability(
            backend,
            "LAGUNA_SPLIT_GATE_FUSION",
            False,
        )
        if use_split_gate_fusion is None
        else use_split_gate_fusion
    )
    _validate_split_backend(
        backend,
        parsed_global_split,
        parsed_swa_split,
        parsed_swa_tile16,
        split_gate_fusion=selected_split_gate_fusion,
    )
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
        split_enabled = any(
            threshold is not None
            for threshold in (
                parsed_global_split,
                parsed_swa_split,
                parsed_swa_tile16,
            )
        )
        split_elements = max(
            (
                q_heads * (context if attention_type == FULL_ATTENTION else sliding_window)
                for attention_type, q_heads in zip(layer_types, head_counts, strict=True)
            ),
            default=0,
        )
        split_score_scratch = (
            allocate_raw(split_elements * DType.FP32.itemsize)
            if split_enabled
            else None
        )
        split_physical_scratch = (
            allocate_raw(split_elements * DType.INT32.itemsize)
            if split_enabled
            else None
        )

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
                attention_prefill_variant = "global_context_rows_spans"
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
            row_position=_buffer_for_tensor(row_position, buffers),
            split_score_scratch=split_score_scratch,
            split_physical_scratch=split_physical_scratch,
            global_split_min_live=parsed_global_split,
            swa_split_min_live=parsed_swa_split,
            swa_split_tile16_min_live=parsed_swa_tile16,
            split_gate_fusion=(selected_split_gate_fusion and split_enabled),
            runtime=runtime,
        )
    except Exception:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        raise


def _validate_split_backend(
    backend: str,
    global_threshold: int | None,
    swa_threshold: int | None,
    swa_tile16_threshold: int | None,
    *,
    split_gate_fusion: bool,
) -> None:
    if all(
        threshold is None
        for threshold in (global_threshold, swa_threshold, swa_tile16_threshold)
    ):
        return
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        register_laguna_kv_attention_kernels,
    )

    register_laguna_kv_attention_kernels()
    load_backend_kernel_package(backend)
    requested = [
        (global_threshold, "global_context_split_exact_spans"),
        (swa_threshold, "swa_context_split_exact_spans"),
        (swa_tile16_threshold, "swa_context_split_tile16_exact_spans"),
    ]
    if split_gate_fusion:
        requested.extend(
            (
                (global_threshold, "global_context_split_exact_gated_spans"),
                (swa_threshold, "swa_context_split_exact_gated_spans"),
                (
                    swa_tile16_threshold,
                    "swa_context_split_tile16_exact_gated_spans",
                ),
            )
        )
    for threshold, variant in requested:
        if threshold is None:
            continue
        if (
            resolve(
                backend=backend,
                layer="laguna_attention_decode",
                quant="bf16",
                variant=variant,
                missing="none",
            )
            is None
        ):
            raise ValueError(
                f"Laguna split attention variant {variant!r} is unavailable "
                f"for backend {backend!r}"
            )


def _validate_split_threshold(
    value: int | None,
    capacity: int,
    name: str,
) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0 or parsed > int(capacity):
        raise ValueError(f"{name} must be within [1, {capacity}]")
    return parsed


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
    "resolve_laguna_split_thresholds",
    "resolve_laguna_swa_decode_variant",
    "resolve_laguna_swa_split_tile16_threshold",
    "resolve_laguna_swa_prefill_variant",
]
