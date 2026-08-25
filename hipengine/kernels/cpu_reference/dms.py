"""CPU-reference kernels for compact DMS retention and attention."""

from __future__ import annotations

import numpy as np

from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache.dms import (
    DMSRetrofitConfig,
    build_dms_live_mask,
    compact_attention_reference,
    decode_dms_payload,
    encode_dms_payload,
    extract_dms_eviction_decisions,
)


def external_dms_linear_decision_reference(
    hidden: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    alpha_scale: float,
    alpha_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project normalized hidden rows without borrowing or mutating Q channels."""

    values = np.asarray(hidden, dtype=np.float32)
    weights = np.asarray(weight, dtype=np.float32)
    biases = np.asarray(bias, dtype=np.float32)
    if values.ndim != 2 or weights.ndim != 2:
        raise ValueError("external DMS hidden/weight tensors must be rank-2")
    if values.shape[1] != weights.shape[1] or biases.shape != (weights.shape[0],):
        raise ValueError("external DMS hidden/weight/bias shapes do not align")
    if not np.isfinite(float(alpha_scale)) or float(alpha_scale) == 0.0:
        raise ValueError("external DMS alpha_scale must be finite and non-zero")
    if not np.isfinite(float(alpha_offset)):
        raise ValueError("external DMS alpha_offset must be finite")
    logits = np.ascontiguousarray(values @ weights.T + biases, dtype=np.float32)
    decisions = logits * float(alpha_scale) - float(alpha_offset) > 0.0
    return logits, np.asarray(decisions, dtype=np.bool_)


def dms_streaming_pack_reference(
    k: np.ndarray,
    v: np.ndarray,
    eviction: np.ndarray,
    *,
    current_position: int,
    window_size: int,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Count/rank/scatter oracle with no retained dense sidecar."""

    key = np.asarray(k, dtype=np.float32)
    value = np.asarray(v, dtype=np.float32)
    evict = np.asarray(eviction, dtype=np.bool_)
    if key.ndim != 4 or value.shape != key.shape or evict.shape != key.shape[:3]:
        raise ValueError("DMS pack expects K/V[T,L,H,D] and eviction[T,L,H]")
    positions = np.arange(key.shape[0], dtype=np.int32)
    packed_k: list[list[np.ndarray]] = []
    packed_v: list[list[np.ndarray]] = []
    packed_positions: list[list[np.ndarray]] = []
    for layer in range(key.shape[1]):
        layer_k: list[np.ndarray] = []
        layer_v: list[np.ndarray] = []
        layer_positions: list[np.ndarray] = []
        for head in range(key.shape[2]):
            live = build_dms_live_mask(
                evict[:, layer, head][None, :],
                current_position=int(current_position),
                window_size=int(window_size),
                positions=positions,
            )[0]
            layer_k.append(np.ascontiguousarray(key[live, layer, head]))
            layer_v.append(np.ascontiguousarray(value[live, layer, head]))
            layer_positions.append(np.ascontiguousarray(positions[live]))
        packed_k.append(layer_k)
        packed_v.append(layer_v)
        packed_positions.append(layer_positions)
    return packed_k, packed_v, packed_positions


def register_dms_cpu_reference_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "cpu_reference",
            "dms_decision_source",
            "bf16",
            "external_linear_sidecar_v1",
        ),
        external_dms_linear_decision_reference,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "dms_extract_decision", "bf16", "corrected_mask"),
        extract_dms_eviction_decisions,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "dms_streaming_pack", "bf16", "count_rank_scatter"),
        dms_streaming_pack_reference,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "dms_compact_attn_decode", "bf16", "grouped_gqa"),
        compact_attention_reference,
        replace=replace,
    )
    register(
        KernelKey(
            "cpu_reference",
            "dms_compact_attn_decode",
            "int8_per_token_head",
            "grouped_gqa",
        ),
        compact_attention_reference,
        replace=replace,
    )
    register(
        KernelKey(
            "cpu_reference",
            "dms_payload_encode",
            "int8_per_token_head",
            "symmetric",
        ),
        encode_dms_payload,
        replace=replace,
    )
    register(
        KernelKey(
            "cpu_reference",
            "dms_payload_decode",
            "int8_per_token_head",
            "symmetric",
        ),
        decode_dms_payload,
        replace=replace,
    )


__all__ = [
    "DMSRetrofitConfig",
    "dms_streaming_pack_reference",
    "external_dms_linear_decision_reference",
    "register_dms_cpu_reference_kernels",
]
