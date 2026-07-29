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
        split_score_scratch: DeviceBuffer | None,
        split_physical_scratch: DeviceBuffer | None,
        global_split_min_live: int | None,
        swa_split_min_live: int | None,
        swa_split_tile16_min_live: int | None,
        split_gate_fusion: bool,
        global_split_fixedshape_reduce: bool,
        global_fused_fixedshape: bool,
        global_gqa2_vstage64_fixedshape: bool,
        global_gqa2_vstage64_vec16_fixedshape: bool,
        global_gqa2_vstage64_vec16_direct_fixedshape: bool,
        global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape: bool,
        global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape: bool,
        global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape: bool,
        global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape: bool,
        global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape: bool,
        global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape: bool,
        swa_split_wave_local: bool,
        swa_split_gqa3_scores: bool,
        swa_split_fixed512_reduce: bool,
        swa_fused_fixed512: bool,
        swa_gqa3_local384_fixed512: bool,
        swa_gqa3_vstage64_fixed512: bool,
        swa_gqa3_vstage64_vec16_fixed512: bool,
        swa_gqa3_vstage64_vec16_direct_fixed512: bool,
        swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512: bool,
        swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512: bool,
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
        self._split_score_scratch = split_score_scratch
        self._split_physical_scratch = split_physical_scratch
        self.global_split_min_live = global_split_min_live
        self.swa_split_min_live = swa_split_min_live
        self.swa_split_tile16_min_live = swa_split_tile16_min_live
        self.split_gate_fusion = bool(split_gate_fusion)
        self.global_split_fixedshape_reduce = bool(
            global_split_fixedshape_reduce
        )
        self.global_fused_fixedshape = bool(global_fused_fixedshape)
        self.global_gqa2_vstage64_fixedshape = bool(
            global_gqa2_vstage64_fixedshape
        )
        self.global_gqa2_vstage64_vec16_fixedshape = bool(
            global_gqa2_vstage64_vec16_fixedshape
        )
        self.global_gqa2_vstage64_vec16_direct_fixedshape = bool(
            global_gqa2_vstage64_vec16_direct_fixedshape
        )
        self.global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape = bool(
            global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape
        )
        self.global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape = bool(
            global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape
        )
        self.global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape = bool(
            global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape
        )
        self.global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape = bool(
            global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape
        )
        self.global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape = bool(
            global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape
        )
        self.global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape = bool(
            global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape
        )
        self.swa_split_wave_local = bool(swa_split_wave_local)
        self.swa_split_gqa3_scores = bool(swa_split_gqa3_scores)
        self.swa_split_fixed512_reduce = bool(swa_split_fixed512_reduce)
        self.swa_fused_fixed512 = bool(swa_fused_fixed512)
        self.swa_gqa3_local384_fixed512 = bool(
            swa_gqa3_local384_fixed512
        )
        self.swa_gqa3_vstage64_fixed512 = bool(
            swa_gqa3_vstage64_fixed512
        )
        self.swa_gqa3_vstage64_vec16_fixed512 = bool(
            swa_gqa3_vstage64_vec16_fixed512
        )
        self.swa_gqa3_vstage64_vec16_direct_fixed512 = bool(
            swa_gqa3_vstage64_vec16_direct_fixed512
        )
        self.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = bool(
            swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
        )
        self.swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
            bool(
                swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
            )
        )
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
                    (
                        "global_context_fused_exact_gated_mixed32_exp32_"
                        "producer_max_dpp_qk_probability_vec4_vstage64_vec16_"
                        "direct_assume_exp_fixedshape_spans"
                    )
                    if (
                        use_gated
                        and self.global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape
                        and self.global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape
                        and live_count <= 4000
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else (
                        "global_context_fused_exact_gated_mixed32_exp32_"
                        "producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_"
                        "fixedshape_spans"
                    )
                    if (
                        use_gated
                        and self.global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape
                        and self.global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape
                        and live_count <= 4000
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else (
                        "global_context_fused_exact_gated_mixed32_exp32_"
                        "producer_max_vstage64_vec16_direct_assume_exp_"
                        "fixedshape_spans"
                    )
                    if (
                        use_gated
                        and self.global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape
                        and self.global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape
                        and live_count <= 4000
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else (
                        "global_context_fused_exact_gated_mixed32_exp32_"
                        "vstage64_vec16_direct_assume_exp_fixedshape_spans"
                    )
                    if (
                        use_gated
                        and self.global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape
                        and self.global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape
                        and live_count <= 4000
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else (
                        "global_context_fused_exact_gated_gqa2_exp32_"
                        "vstage64_vec16_direct_assume_exp_fixedshape_spans"
                    )
                    if (
                        use_gated
                        and self.global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape
                        and self.global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape
                        and live_count <= 4000
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else (
                        "global_context_fused_exact_gated_gqa2_vstage64_"
                        "vec16_direct_assume_exp_fixedshape_spans"
                    )
                    if (
                        use_gated
                        and self.global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape
                        and self.global_gqa2_vstage64_vec16_direct_fixedshape
                        and live_count <= 4000
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else (
                        "global_context_fused_exact_gated_"
                        "gqa2_vstage64_vec16_direct_fixedshape_spans"
                    )
                    if (
                        use_gated
                        and self.global_gqa2_vstage64_vec16_direct_fixedshape
                        and live_count <= 4000
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else (
                        "global_context_fused_exact_gated_"
                        "gqa2_vstage64_vec16_fixedshape_spans"
                    )
                    if (
                        use_gated
                        and self.global_gqa2_vstage64_vec16_fixedshape
                        and live_count <= 4000
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else (
                        "global_context_fused_exact_gated_"
                        "gqa2_vstage64_fixedshape_spans"
                    )
                    if (
                        use_gated
                        and self.global_gqa2_vstage64_fixedshape
                        and live_count <= 4000
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else "global_context_fused_exact_gated_gqa1_fixedshape_spans"
                    if (
                        use_gated
                        and self.global_fused_fixedshape
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else "global_context_split_exact_gated_fixedshape_spans"
                    if (
                        use_gated
                        and self.global_split_fixedshape_reduce
                        and state.capacity == 4096
                        and state.q_heads == 48
                    )
                    else "global_context_split_exact_gated_spans"
                    if use_gated
                    else "global_context_split_exact_spans"
                )
                if state.attention_type == FULL_ATTENTION
                else (
                    (
                        (
                            (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    "swa_context_fused_exact_gated_"
                                                                    "mixed32_exp32_producer_max_"
                                                                    "gate_stage_pcache_idle_vec4_"
                                                                    "denom_probability_vstage64_"
                                                                    "vec16_direct_assume_exp_"
                                                                    "fixed512_spans"
                                                                )
                                                                if (
                                                                    self.swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
                                                                    and self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
                                                                    and self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512
                                                                    and self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512
                                                                    and self.swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
                                                                    and self.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
                                                                    and self.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
                                                                )
                                                                else (
                                                                    "swa_context_fused_exact_gated_"
                                                                    "mixed32_exp32_producer_max_"
                                                                    "gate_stage_pcache_vec4_denom_"
                                                                    "probability_vstage64_vec16_"
                                                                    "direct_assume_exp_fixed512_spans"
                                                                )
                                                            )
                                                            if (
                                                                self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
                                                                and self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512
                                                                and self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512
                                                                and self.swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
                                                                and self.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
                                                                and self.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
                                                            )
                                                            else (
                                                                "swa_context_fused_exact_gated_"
                                                                "mixed32_exp32_producer_max_"
                                                                "gate_stage_pcache_vec4_denom_"
                                                                "vstage64_vec16_direct_assume_"
                                                                "exp_fixed512_spans"
                                                            )
                                                            if (
                                                                self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512
                                                                and self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512
                                                                and self.swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
                                                                and self.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
                                                                and self.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
                                                            )
                                                            else (
                                                            "swa_context_fused_exact_gated_"
                                                            "mixed32_exp32_producer_max_"
                                                            "gate_stage_pcache_vstage64_"
                                                            "vec16_direct_assume_exp_"
                                                            "fixed512_spans"
                                                        )
                                                        if (
                                                            self.swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512
                                                            and self.swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
                                                            and self.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
                                                            and self.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
                                                        )
                                                        else (
                                                            "swa_context_fused_exact_gated_"
                                                            "mixed32_exp32_producer_max_"
                                                            "gate_vstage64_vec16_direct_"
                                                            "assume_exp_fixed512_spans"
                                                        )
                                                        if (
                                                            self.swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
                                                            and self.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
                                                            and self.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
                                                        )
                                                        else (
                                                            "swa_context_fused_exact_gated_"
                                                            "mixed32_exp32_producer_max_"
                                                            "vstage64_vec16_direct_assume_"
                                                            "exp_fixed512_spans"
                                                        )
                                                            )
                                                        if (
                                                            self.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
                                                            and self.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
                                                        )
                                                        else (
                                                            "swa_context_fused_exact_gated_"
                                                            "mixed32_exp32_vstage64_vec16_"
                                                            "direct_assume_exp_fixed512_spans"
                                                        )
                                                        if self.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
                                                        else (
                                                            "swa_context_fused_exact_gated_"
                                                            "mixed32_exp16_vstage64_vec16_"
                                                            "direct_assume_exp_fixed512_spans"
                                                        )
                                                    )
                                                )
                                                if self.swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512
                                                else (
                                                    "swa_context_fused_exact_gated_"
                                                    "mixed32_exp8_vstage64_vec16_"
                                                    "direct_assume_exp_fixed512_spans"
                                                )
                                            )
                                            if (
                                                self.swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512
                                                and self.swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512
                                                and self.swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512
                                                and self.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
                                                and live_count == 512
                                                and state.capacity == 512
                                            )
                                            else
                                            (
                                                "swa_context_fused_exact_gated_"
                                                "mixed32_exp4_vstage64_vec16_direct_"
                                            "assume_exp_fixed512_spans"
                                        )
                                        if (
                                            self.swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512
                                            and self.swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512
                                            and self.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
                                            and live_count == 512
                                            and state.capacity == 512
                                        )
                                        else
                                        (
                                            "swa_context_fused_exact_gated_"
                                            "mixed32_vstage64_vec16_direct_"
                                            "assume_exp_fixed512_spans"
                                        )
                                        if (
                                            self.swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512
                                            and self.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
                                            and live_count == 512
                                            and state.capacity == 512
                                        )
                                        else
                                        "swa_context_fused_exact_gated_"
                                        "gqa3_vstage64_vec16_direct_assume_"
                                        "exp_fixed512_spans"
                                    )
                                    if (
                                        self.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
                                        and self.swa_gqa3_vstage64_vec16_direct_fixed512
                                        and live_count == 512
                                        and state.capacity == 512
                                    )
                                    else (
                                        "swa_context_fused_exact_gated_"
                                        "gqa3_vstage64_vec16_direct_"
                                        "fixed512_spans"
                                    )
                                    if (
                                        self.swa_gqa3_vstage64_vec16_direct_fixed512
                                        and live_count == 512
                                        and state.capacity == 512
                                    )
                                    else (
                                        "swa_context_fused_exact_gated_"
                                        "gqa3_vstage64_vec16_fixed512_spans"
                                    )
                                    if (
                                        self.swa_gqa3_vstage64_vec16_fixed512
                                        and live_count == 512
                                        and state.capacity == 512
                                    )
                                    else (
                                        "swa_context_fused_exact_gated_"
                                        "gqa3_vstage64_fixed512_spans"
                                    )
                                )
                                if (
                                    self.swa_gqa3_vstage64_fixed512
                                    and live_count == 512
                                    and state.capacity == 512
                                )
                                else (
                                    "swa_context_fused_exact_gated_"
                                    "gqa3_local384_fixed512_spans"
                                )
                                if (
                                    self.swa_gqa3_local384_fixed512
                                    and live_count == 512
                                    and state.capacity == 512
                                )
                                else "swa_context_fused_exact_gated_gqa2_fixed512_spans"
                                if (
                                    self.swa_fused_fixed512
                                    and live_count == 512
                                    and state.capacity == 512
                                )
                                else (
                                    "swa_context_split_tile16_exact_gated_"
                                    "gqa3_scores_fixed512_spans"
                                )
                                if (
                                    self.swa_split_fixed512_reduce
                                    and self.swa_split_gqa3_scores
                                    and live_count == 512
                                    and state.capacity == 512
                                )
                                else "swa_context_split_tile16_exact_gated_gqa3_scores_spans"
                                if self.swa_split_gqa3_scores
                                else "swa_context_split_tile16_exact_gated_wave_local_spans"
                            )
                            if self.swa_split_wave_local and use_gated
                            else "swa_context_split_tile16_exact_gated_spans"
                        )
                        if use_gated
                        else "swa_context_split_tile16_exact_spans"
                    )
                    if use_tile16
                    else (
                        (
                            (
                                "swa_context_split_exact_gated_gqa3_scores_spans"
                                if self.swa_split_gqa3_scores
                                else "swa_context_split_exact_gated_wave_local_spans"
                            )
                            if self.swa_split_wave_local and use_gated
                            else "swa_context_split_exact_gated_spans"
                        )
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
        """Return whether one bounded tile can be cached before causal attention."""

        state = self.layer(layer_id)
        offset = int(row_offset)
        count = int(rows)
        if (
            (
                count != 128
                and not (
                    state.attention_type == FULL_ATTENTION
                    and count in {256, 512, 1_024, 2_048}
                )
            )
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
        if (
            spans.spans_mode == "sliding_ring"
            and count > self.sliding_window
        ):
            raise ValueError("bulk KV operations must fit the SWA ring capacity")
        if (
            offset == 0
            and count == len(self._pending_positions)
            and row_positions_ptr is None
        ):
            return spans
        if (
            len(self._pending_positions) > self.sliding_window
            and row_positions_ptr is None
        ):
            raise ValueError(
                "wide bulk KV operations require a resident row_positions_ptr"
            )
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
    global_prefill_variant: str | None = None,
    swa_decode_variant: str | None = None,
    swa_prefill_variant: str | None = None,
    prefill_cached_meta: bool = False,
    prefill_global_qrow6: bool = False,
    prefill_dense_initial: bool = False,
    global_split_min_live: int | None = None,
    swa_split_min_live: int | None = None,
    swa_split_tile16_min_live: int | None = None,
    use_swa_split_tile16: bool | None = None,
    use_split_attention: bool | None = None,
    use_split_gate_fusion: bool | None = None,
    use_swa_split_wave_local: bool | None = None,
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
    if use_split_attention is False and (
        swa_split_tile16_min_live is not None
        or use_swa_split_tile16 is True
        or use_swa_split_wave_local is True
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
    selected_swa_split_wave_local = (
        False
        if use_split_attention is False
        else bool(
            backend_package_capability(
                backend,
                "LAGUNA_SWA_SPLIT_WAVE_LOCAL",
                False,
            )
            if use_swa_split_wave_local is None
            else use_swa_split_wave_local
        )
    )
    if selected_swa_split_wave_local and (
        parsed_swa_split is None or not selected_split_gate_fusion
    ):
        raise ValueError(
            "SWA wave-local reduction requires exact split attention and "
            "split-gate fusion"
        )
    selected_swa_split_gqa3_scores = bool(
        selected_swa_split_wave_local
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_SPLIT_GQA3_SCORES",
            False,
        )
    )
    selected_swa_split_fixed512_reduce = bool(
        selected_swa_split_gqa3_scores
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_SPLIT_FIXED512_REDUCE",
            False,
        )
    )
    selected_swa_fused_fixed512 = bool(
        selected_swa_split_fixed512_reduce
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_FUSED_FIXED512",
            False,
        )
    )
    selected_swa_gqa3_local384_fixed512 = bool(
        selected_swa_fused_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_GQA3_LOCAL384_FIXED512",
            False,
        )
    )
    selected_swa_gqa3_vstage64_fixed512 = bool(
        selected_swa_gqa3_local384_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_GQA3_VSTAGE64_FIXED512",
            False,
        )
    )
    selected_swa_gqa3_vstage64_vec16_fixed512 = bool(
        selected_swa_gqa3_vstage64_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_GQA3_VSTAGE64_VEC16_FIXED512",
            False,
        )
    )
    selected_swa_gqa3_vstage64_vec16_direct_fixed512 = bool(
        selected_swa_gqa3_vstage64_vec16_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_GQA3_VSTAGE64_VEC16_DIRECT_FIXED512",
            False,
        )
    )
    selected_swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_gqa3_vstage64_vec16_direct_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_GQA3_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_EXP4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_EXP8_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_EXP16_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = bool(
        selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
        and backend_package_capability(
            backend,
            "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
            False,
        )
    )
    selected_global_split_fixedshape_reduce = bool(
        selected_split_gate_fusion
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_SPLIT_FIXEDSHAPE_REDUCE",
            False,
        )
    )
    selected_global_fused_fixedshape = bool(
        selected_global_split_fixedshape_reduce
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_FUSED_FIXEDSHAPE",
            False,
        )
    )
    selected_global_gqa2_vstage64_fixedshape = bool(
        selected_global_fused_fixedshape
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_GQA2_VSTAGE64_FIXEDSHAPE",
            False,
        )
    )
    selected_global_gqa2_vstage64_vec16_fixedshape = bool(
        selected_global_gqa2_vstage64_fixedshape
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_FIXEDSHAPE",
            False,
        )
    )
    selected_global_gqa2_vstage64_vec16_direct_fixedshape = bool(
        selected_global_gqa2_vstage64_vec16_fixedshape
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_DIRECT_FIXEDSHAPE",
            False,
        )
    )
    selected_global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape = bool(
        selected_global_gqa2_vstage64_vec16_direct_fixedshape
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
            False,
        )
    )
    selected_global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape = bool(
        selected_global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_GQA2_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
            False,
        )
    )
    selected_global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape = bool(
        selected_global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
            False,
        )
    )
    selected_global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape = bool(
        selected_global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
            False,
        )
    )
    selected_global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape = bool(
        selected_global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
            False,
        )
    )
    selected_global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape = bool(
        selected_global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape
        and backend_package_capability(
            backend,
            "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
            False,
        )
    )
    _validate_split_backend(
        backend,
        parsed_global_split,
        parsed_swa_split,
        parsed_swa_tile16,
        split_gate_fusion=selected_split_gate_fusion,
        global_split_fixedshape_reduce=(
            selected_global_split_fixedshape_reduce
        ),
        global_fused_fixedshape=selected_global_fused_fixedshape,
        global_gqa2_vstage64_fixedshape=(
            selected_global_gqa2_vstage64_fixedshape
        ),
        global_gqa2_vstage64_vec16_fixedshape=(
            selected_global_gqa2_vstage64_vec16_fixedshape
        ),
        global_gqa2_vstage64_vec16_direct_fixedshape=(
            selected_global_gqa2_vstage64_vec16_direct_fixedshape
        ),
        global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape=(
            selected_global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape
        ),
        global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape=(
            selected_global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape
        ),
        global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape=(
            selected_global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape
        ),
        global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape=(
            selected_global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape
        ),
        global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape=(
            selected_global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape
        ),
        global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape=(
            selected_global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape
        ),
        swa_split_wave_local=selected_swa_split_wave_local,
        swa_split_gqa3_scores=selected_swa_split_gqa3_scores,
        swa_split_fixed512_reduce=selected_swa_split_fixed512_reduce,
        swa_fused_fixed512=selected_swa_fused_fixed512,
        swa_gqa3_local384_fixed512=(
            selected_swa_gqa3_local384_fixed512
        ),
        swa_gqa3_vstage64_fixed512=(
            selected_swa_gqa3_vstage64_fixed512
        ),
        swa_gqa3_vstage64_vec16_fixed512=(
            selected_swa_gqa3_vstage64_vec16_fixed512
        ),
        swa_gqa3_vstage64_vec16_direct_fixed512=(
            selected_swa_gqa3_vstage64_vec16_direct_fixed512
        ),
        swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
        ),
        swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512=(
            selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
        ),
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
            split_score_scratch=split_score_scratch,
            split_physical_scratch=split_physical_scratch,
            global_split_min_live=parsed_global_split,
            swa_split_min_live=parsed_swa_split,
            swa_split_tile16_min_live=parsed_swa_tile16,
            split_gate_fusion=(selected_split_gate_fusion and split_enabled),
            global_split_fixedshape_reduce=(
                selected_global_split_fixedshape_reduce and split_enabled
            ),
            global_fused_fixedshape=(
                selected_global_fused_fixedshape and split_enabled
            ),
            global_gqa2_vstage64_fixedshape=(
                selected_global_gqa2_vstage64_fixedshape and split_enabled
            ),
            global_gqa2_vstage64_vec16_fixedshape=(
                selected_global_gqa2_vstage64_vec16_fixedshape
                and split_enabled
            ),
            global_gqa2_vstage64_vec16_direct_fixedshape=(
                selected_global_gqa2_vstage64_vec16_direct_fixedshape
                and split_enabled
            ),
            global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape=(
                selected_global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape
                and split_enabled
            ),
            global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape=(
                selected_global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape
                and split_enabled
            ),
            global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape=(
                selected_global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape
                and split_enabled
            ),
            global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape=(
                selected_global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape
                and split_enabled
            ),
            global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape=(
                selected_global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape
                and split_enabled
            ),
            global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape=(
                selected_global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape
                and split_enabled
            ),
            swa_split_wave_local=(selected_swa_split_wave_local and split_enabled),
            swa_split_gqa3_scores=(
                selected_swa_split_gqa3_scores and split_enabled
            ),
            swa_split_fixed512_reduce=(
                selected_swa_split_fixed512_reduce and split_enabled
            ),
            swa_fused_fixed512=(
                selected_swa_fused_fixed512 and split_enabled
            ),
            swa_gqa3_local384_fixed512=(
                selected_swa_gqa3_local384_fixed512 and split_enabled
            ),
            swa_gqa3_vstage64_fixed512=(
                selected_swa_gqa3_vstage64_fixed512 and split_enabled
            ),
            swa_gqa3_vstage64_vec16_fixed512=(
                selected_swa_gqa3_vstage64_vec16_fixed512 and split_enabled
            ),
            swa_gqa3_vstage64_vec16_direct_fixed512=(
                selected_swa_gqa3_vstage64_vec16_direct_fixed512
                and split_enabled
            ),
            swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
            swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512=(
                selected_swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
                and split_enabled
            ),
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
    global_split_fixedshape_reduce: bool,
    global_fused_fixedshape: bool,
    global_gqa2_vstage64_fixedshape: bool,
    global_gqa2_vstage64_vec16_fixedshape: bool,
    global_gqa2_vstage64_vec16_direct_fixedshape: bool,
    global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape: bool,
    global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape: bool,
    global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape: bool,
    global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape: bool,
    global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape: bool,
    global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape: bool,
    swa_split_wave_local: bool,
    swa_split_gqa3_scores: bool,
    swa_split_fixed512_reduce: bool,
    swa_fused_fixed512: bool,
    swa_gqa3_local384_fixed512: bool,
    swa_gqa3_vstage64_fixed512: bool,
    swa_gqa3_vstage64_vec16_fixed512: bool,
    swa_gqa3_vstage64_vec16_direct_fixed512: bool,
    swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512: bool,
    swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512: bool,
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
    if global_split_fixedshape_reduce:
        requested.append(
            (
                global_threshold,
                "global_context_split_exact_gated_fixedshape_spans",
            )
        )
    if global_fused_fixedshape:
        requested.append(
            (
                global_threshold,
                "global_context_fused_exact_gated_gqa1_fixedshape_spans",
            )
        )
    if global_gqa2_vstage64_fixedshape:
        requested.append(
            (
                global_threshold,
                (
                    "global_context_fused_exact_gated_"
                    "gqa2_vstage64_fixedshape_spans"
                ),
            )
        )
    if global_gqa2_vstage64_vec16_fixedshape:
        requested.append(
            (
                global_threshold,
                (
                    "global_context_fused_exact_gated_"
                    "gqa2_vstage64_vec16_fixedshape_spans"
                ),
            )
        )
    if global_gqa2_vstage64_vec16_direct_fixedshape:
        requested.append(
            (
                global_threshold,
                (
                    "global_context_fused_exact_gated_"
                    "gqa2_vstage64_vec16_direct_fixedshape_spans"
                ),
            )
        )
    if global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape:
        requested.append(
            (
                global_threshold,
                (
                    "global_context_fused_exact_gated_gqa2_vstage64_"
                    "vec16_direct_assume_exp_fixedshape_spans"
                ),
            )
        )
    if global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape:
        requested.append(
            (
                global_threshold,
                (
                    "global_context_fused_exact_gated_gqa2_exp32_vstage64_"
                    "vec16_direct_assume_exp_fixedshape_spans"
                ),
            )
        )
    if global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape:
        requested.append(
            (
                global_threshold,
                (
                    "global_context_fused_exact_gated_mixed32_exp32_vstage64_"
                    "vec16_direct_assume_exp_fixedshape_spans"
                ),
            )
        )
    if global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape:
        requested.append(
            (
                global_threshold,
                (
                    "global_context_fused_exact_gated_mixed32_exp32_"
                    "producer_max_vstage64_vec16_direct_assume_exp_"
                    "fixedshape_spans"
                ),
            )
        )
    if global_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape:
        requested.append(
            (
                global_threshold,
                (
                    "global_context_fused_exact_gated_mixed32_exp32_"
                    "producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_"
                    "fixedshape_spans"
                ),
            )
        )
    if global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape:
        requested.append(
            (
                global_threshold,
                (
                    "global_context_fused_exact_gated_mixed32_exp32_"
                    "producer_max_dpp_qk_probability_vec4_vstage64_vec16_"
                    "direct_assume_exp_fixedshape_spans"
                ),
            )
        )
    if swa_split_wave_local:
        requested.extend(
            (
                (
                    swa_threshold,
                    "swa_context_split_exact_gated_wave_local_spans",
                ),
                (
                    swa_tile16_threshold,
                    "swa_context_split_tile16_exact_gated_wave_local_spans",
                ),
            )
        )
    if swa_split_gqa3_scores:
        requested.extend(
            (
                (
                    swa_threshold,
                    "swa_context_split_exact_gated_gqa3_scores_spans",
                ),
                (
                    swa_tile16_threshold,
                    "swa_context_split_tile16_exact_gated_gqa3_scores_spans",
                ),
            )
        )
    if swa_split_fixed512_reduce:
        requested.append(
            (
                swa_tile16_threshold,
                "swa_context_split_tile16_exact_gated_gqa3_scores_fixed512_spans",
            )
        )
    if swa_fused_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                "swa_context_fused_exact_gated_gqa2_fixed512_spans",
            )
        )
    if swa_gqa3_local384_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                "swa_context_fused_exact_gated_gqa3_local384_fixed512_spans",
            )
        )
    if swa_gqa3_vstage64_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                "swa_context_fused_exact_gated_gqa3_vstage64_fixed512_spans",
            )
        )
    if swa_gqa3_vstage64_vec16_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_"
                    "gqa3_vstage64_vec16_fixed512_spans"
                ),
            )
        )
    if swa_gqa3_vstage64_vec16_direct_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_"
                    "gqa3_vstage64_vec16_direct_fixed512_spans"
                ),
            )
        )
    if swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_gqa3_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            )
        )
    if swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            )
        )
    if swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_exp4_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            )
        )
    if swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_exp8_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            )
        )
    if swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_exp16_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            )
        )
    if swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_exp32_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            )
        )
    if swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_exp32_producer_max_"
                    "vstage64_vec16_direct_assume_exp_fixed512_spans"
                ),
            )
        )
    if swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_exp32_producer_max_"
                    "gate_vstage64_vec16_direct_assume_exp_fixed512_spans"
                ),
            )
        )
    if swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_exp32_producer_max_"
                    "gate_stage_pcache_vstage64_vec16_direct_assume_exp_"
                    "fixed512_spans"
                ),
            )
        )
    if swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_exp32_producer_max_"
                    "gate_stage_pcache_vec4_denom_vstage64_vec16_direct_"
                    "assume_exp_fixed512_spans"
                ),
            )
        )
    if swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_exp32_producer_max_"
                    "gate_stage_pcache_vec4_denom_probability_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            )
        )
    if swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512:
        requested.append(
            (
                swa_tile16_threshold,
                (
                    "swa_context_fused_exact_gated_mixed32_exp32_producer_max_"
                    "gate_stage_pcache_idle_vec4_denom_probability_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
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
    "resolve_laguna_global_prefill_variant",
    "resolve_laguna_split_thresholds",
    "resolve_laguna_swa_decode_variant",
    "resolve_laguna_swa_split_tile16_threshold",
    "resolve_laguna_swa_prefill_variant",
]
