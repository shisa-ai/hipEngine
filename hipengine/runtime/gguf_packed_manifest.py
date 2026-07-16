"""Observable execution contract for GGUF packed autoregressive decode.

The packed Q4_K_M route remains an exact-hybrid bring-up path. Full attention,
MoE/FFN, LM head, and greedy argmax consume all live rows together. Recurrent
linear attention either uses the registered indexed Conv/GDN closure or falls
back to the c1-exact row subgraph when that capability is absent. This module
keeps the selected boundary explicit without expensive hot-path tracing.

The route-dependent structural launch counts are checked against a
backend-specific real rocprof trace by ``scripts/gguf_packed_ar_rocprof.py``. A
dispatch change therefore fails the census instead of silently making this
manifest stale.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

# One c1-exact recurrent attention row currently launches:
#   projection: qkv+gate pair, alpha, beta, and ssm_out (4)
#   recurrent: Conv and GDN (2)
#   normalization: attention RMSNorm (1)
_ROW_LOCAL_PROJECTION_LAUNCHES = 4
_ROW_LOCAL_CONV_GDN_LAUNCHES = 2
_ROW_LOCAL_NORMALIZATION_LAUNCHES = 1


def _positive_int(value: int, *, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def build_packed_decode_execution_manifest(
    *,
    rows: int,
    layer_types: Sequence[str],
    imported_slot_indices: Sequence[int],
    import_positions: Sequence[int],
    scatter_state: bool,
    blocks_per_slot: int,
    full_attention_decode_path: str,
    moe_decode_path: str,
    moe_top_k: int,
    lm_head_decode_path: str,
    sampler_decode_path: str,
    metadata_prepare_path: str,
    capture_layer_count: int = 0,
    linear_attention_decode_path: str = "exact_row_local",
) -> dict[str, Any]:
    """Build the auditable host/runtime contract for one packed decode step.

    ``imported_slot_indices`` is computed immediately before packed-state
    import. It is empty for steady ``scatter_state=False`` and non-empty after
    prefill, membership changes, or an explicit flush. Copy counts are host
    call counts, not inferred kernel dispatches; the paired profiler census
    supplies measured GPU launch counts and durations.
    """

    row_count = _positive_int(rows, name="rows")
    block_count = _positive_int(blocks_per_slot, name="blocks_per_slot")
    layer_tuple = tuple(str(layer_type) for layer_type in layer_types)
    if not layer_tuple:
        raise ValueError("layer_types must be non-empty")
    unsupported = sorted(set(layer_tuple) - {LINEAR_ATTENTION, FULL_ATTENTION})
    if unsupported:
        raise ValueError(f"unsupported GGUF packed layer types: {unsupported}")

    positions = tuple(int(position) for position in import_positions)
    if len(positions) != row_count:
        raise ValueError("import_positions must contain one entry per row")
    if any(position < 0 for position in positions):
        raise ValueError("import_positions must be non-negative")
    imported_indices = tuple(int(index) for index in imported_slot_indices)
    if len(set(imported_indices)) != len(imported_indices):
        raise ValueError("imported_slot_indices must be unique")
    if any(index < 0 or index >= row_count for index in imported_indices):
        raise ValueError("imported_slot_indices are outside the packed rows")
    captures = int(capture_layer_count)
    if captures < 0 or captures > len(layer_tuple):
        raise ValueError("capture_layer_count is outside the model layer range")

    linear_layers = layer_tuple.count(LINEAR_ATTENTION)
    full_layers = layer_tuple.count(FULL_ATTENTION)
    linear_path = str(linear_attention_decode_path)
    supported_linear_paths = {"exact_row_local", "indexed_batch", "not_applicable"}
    if linear_path not in supported_linear_paths:
        raise ValueError(
            "unsupported linear_attention_decode_path "
            f"{linear_path!r}; expected one of {sorted(supported_linear_paths)!r}"
        )
    if linear_layers and linear_path == "not_applicable":
        raise ValueError(
            "linear_attention_decode_path cannot be not_applicable when linear layers are present"
        )
    if not linear_layers and linear_path != "not_applicable":
        raise ValueError(
            "linear_attention_decode_path must be not_applicable without linear layers"
        )
    indexed_linear_decode = linear_path == "indexed_batch"

    full_attention_path = str(full_attention_decode_path)
    supported_full_attention_paths = {"kv_live_spans_batch", "not_applicable"}
    if full_attention_path not in supported_full_attention_paths:
        raise ValueError(
            "unsupported full_attention_decode_path "
            f"{full_attention_path!r}; expected one of {sorted(supported_full_attention_paths)!r}"
        )
    if full_layers and full_attention_path != "kv_live_spans_batch":
        raise ValueError(
            "full_attention_decode_path must be kv_live_spans_batch when full-attention layers are present"
        )
    if not full_layers and full_attention_path != "not_applicable":
        raise ValueError(
            "full_attention_decode_path must be not_applicable without full-attention layers"
        )

    moe_path = str(moe_decode_path)
    supported_moe_paths = {"dense_ffn_rows", "selected_rows_batch"}
    if moe_path not in supported_moe_paths:
        raise ValueError(
            f"unsupported moe_decode_path {moe_path!r}; "
            f"expected one of {sorted(supported_moe_paths)!r}"
        )
    top_k = int(moe_top_k)
    if moe_path == "selected_rows_batch":
        top_k = _positive_int(top_k, name="moe_top_k")
    elif top_k != 0:
        raise ValueError("dense_ffn_rows requires moe_top_k=0")
    lm_head_path = str(lm_head_decode_path)
    supported_lm_head_paths = {
        "direct_top1_rows",
        "q6_rowtile_f32_logits",
        "row_linear_f32_logits",
    }
    if lm_head_path not in supported_lm_head_paths:
        raise ValueError(
            f"unsupported lm_head_decode_path {lm_head_path!r}; "
            f"expected one of {sorted(supported_lm_head_paths)!r}"
        )
    sampler_path = str(sampler_decode_path)
    supported_sampler_paths = {"argmax_i32_rows", "fused_top1_i32_rows"}
    if sampler_path not in supported_sampler_paths:
        raise ValueError(
            f"unsupported sampler_decode_path {sampler_path!r}; "
            f"expected one of {sorted(supported_sampler_paths)!r}"
        )
    if lm_head_path == "direct_top1_rows" and sampler_path != "fused_top1_i32_rows":
        raise ValueError("direct_top1_rows requires fused_top1_i32_rows sampling")
    if lm_head_path != "direct_top1_rows" and sampler_path != "argmax_i32_rows":
        raise ValueError("logit-producing lm-head paths require argmax_i32_rows sampling")
    metadata_path = str(metadata_prepare_path)
    supported_metadata_paths = {"device_prepare_persistent", "host_upload"}
    if metadata_path not in supported_metadata_paths:
        raise ValueError(
            f"unsupported metadata_prepare_path {metadata_path!r}; "
            f"expected one of {sorted(supported_metadata_paths)!r}"
        )

    row_loop_iterations = 0 if indexed_linear_decode else linear_layers * row_count
    projection_row_launches = row_loop_iterations * _ROW_LOCAL_PROJECTION_LAUNCHES
    conv_gdn_row_launches = row_loop_iterations * _ROW_LOCAL_CONV_GDN_LAUNCHES
    normalization_row_launches = row_loop_iterations * _ROW_LOCAL_NORMALIZATION_LAUNCHES
    exact_row_launches = (
        projection_row_launches
        + conv_gdn_row_launches
        + normalization_row_launches
    )

    # ``for_packed_verify_layout`` uploads eight metadata arrays every step.
    metadata_bytes = (
        row_count * block_count * 4  # block table, int32
        + row_count * 8  # positions, int64
        + row_count * 8  # context/live counts, int64
        + 2 * 4  # cu_q, int32[2]
        + 2 * 4  # cu_k, int32[2]
        + 4  # atomic counter, int32[1]
        + (row_count + 1) * 4  # GDN cu_seqlens, int32[C+1]
        + row_count * 8  # GDN state indices, int64[C]
    )
    input_bytes = row_count * 8  # packed token ids, int64[C]
    metadata_host_copies = 8 if metadata_path == "host_upload" else 0
    metadata_host_bytes = metadata_bytes if metadata_host_copies else 0
    metadata_device_prepare_launches = (
        1 if metadata_path == "device_prepare_persistent" else 0
    )

    imported_positions = tuple(positions[index] for index in imported_indices)
    state_import_copies = sum(
        2 * linear_layers + (2 * full_layers if position > 0 else 0)
        for position in imported_positions
    )
    state_scatter_copies = (
        row_count * (2 * linear_layers + 2 * full_layers)
        if bool(scatter_state)
        else 0
    )

    layer_families: dict[str, dict[str, Any]] = {
        "projection": {
            "execution": "packed_native" if indexed_linear_decode else "hybrid",
            "packed_native_work": [
                "embedding",
                "full_attention_qkv_o",
                "moe_selected_and_shared_projections",
                *(
                    [
                        "linear_attention_qkv_gate",
                        "linear_attention_alpha",
                        "linear_attention_beta",
                        "linear_attention_ssm_out_fp32_input",
                    ]
                    if indexed_linear_decode
                    else []
                ),
            ],
            "exact_row_local_work": (
                []
                if indexed_linear_decode
                else [
                    "linear_attention_qkv_gate",
                    "linear_attention_alpha",
                    "linear_attention_beta",
                    "linear_attention_ssm_out",
                ]
            ),
            "host_row_loop_sites": 0 if indexed_linear_decode else linear_layers,
            "host_row_iterations": row_loop_iterations,
            "exact_row_local_kernel_launches": projection_row_launches,
        },
        "conv_gdn": {
            "execution": "packed_native" if indexed_linear_decode else "exact_row_local",
            "packed_native_work": (
                ["conv_decode_indexed", "gdn_recurrent_decode_segments_fp32_out"]
                if indexed_linear_decode
                else []
            ),
            "exact_row_local_work": (
                []
                if indexed_linear_decode
                else ["conv_decode", "gdn_recurrent_decode"]
            ),
            "host_row_loop_sites": 0 if indexed_linear_decode else linear_layers,
            "host_row_iterations": row_loop_iterations,
            "exact_row_local_kernel_launches": conv_gdn_row_launches,
        },
        "normalization": {
            "execution": "packed_native" if indexed_linear_decode else "hybrid",
            "packed_native_work": [
                "packed_layer_and_output_norms",
                *(["linear_attention_attn_norm"] if indexed_linear_decode else []),
            ],
            "exact_row_local_work": (
                [] if indexed_linear_decode else ["linear_attention_attn_norm"]
            ),
            "host_row_loop_sites": 0 if indexed_linear_decode else linear_layers,
            "host_row_iterations": row_loop_iterations,
            "exact_row_local_kernel_launches": normalization_row_launches,
        },
        "full_attention": {
            "execution": "packed_native",
            "decode_path": full_attention_path,
            "kv_abi": "KVLiveSpans" if full_layers else "not_applicable",
            "layer_invocations": full_layers,
            "row_positions": row_count,
            "live_counts": [position + 1 for position in positions],
            "host_row_loop_sites": 0,
            "host_row_iterations": 0,
            "exact_row_local_kernel_launches": 0,
        },
        "moe_ffn": {
            "execution": "packed_native",
            "decode_path": moe_path,
            "layer_invocations": len(layer_tuple),
            "router_rows": row_count,
            "top_k": top_k,
            "selected_lanes": row_count * top_k,
            "lane_to_row": "selected_lane // top_k",
            "host_row_loop_sites": 0,
            "host_row_iterations": 0,
            "exact_row_local_kernel_launches": 0,
        },
        "lm_head": {
            "execution": "packed_native",
            "decode_path": lm_head_path,
            "layer_invocations": 1,
            "output_rows": row_count,
            "full_vocab_host_readback": False,
            "host_row_loop_sites": 0,
            "host_row_iterations": 0,
            "exact_row_local_kernel_launches": 0,
        },
        "sampler": {
            "execution": "packed_native",
            "decode_path": sampler_path,
            "layer_invocations": 1,
            "host_row_loop_sites": 0,
            "host_row_iterations": 0,
            "exact_row_local_kernel_launches": 0,
            "device_result": sampler_path,
            "host_readback": "one_i32_vector",
        },
    }

    return {
        "schema": 1,
        "kind": "gguf_packed_ar_execution_manifest",
        "mode": "decode",
        "claim_level": "exact_hybrid",
        "rows": row_count,
        "linear_attention_decode_path": linear_path,
        "full_attention_decode_path": full_attention_path,
        "moe_decode_path": moe_path,
        "lm_head_decode_path": lm_head_path,
        "sampler_decode_path": sampler_path,
        "metadata_prepare_path": metadata_path,
        "layers": {
            "total": len(layer_tuple),
            "linear_attention": linear_layers,
            "full_attention": full_layers,
        },
        "model_step": {
            "complete_c1_session_replays": 0,
            "complete_c1_layer_replays": 0,
            "host_model_row_loop_sites": 0 if indexed_linear_decode else linear_layers,
            "host_model_row_iterations": row_loop_iterations,
            "per_row_model_subgraph_invocations": row_loop_iterations,
            "expected_exact_row_local_kernel_launches": exact_row_launches,
        },
        "layer_families": layer_families,
        "host_device_movement": {
            "host_to_device_metadata_copies": metadata_host_copies,
            "host_to_device_metadata_bytes": metadata_host_bytes,
            "device_metadata_prepare_launches": metadata_device_prepare_launches,
            "host_to_device_input_copies": 1,
            "host_to_device_input_bytes": input_bytes,
            "host_to_device_total_copies": metadata_host_copies + 1,
            "host_to_device_total_bytes": metadata_host_bytes + input_bytes,
            "device_to_device_state_import_copies": state_import_copies,
            "device_to_device_state_scatter_copies": state_scatter_copies,
            "diagnostic_layer_capture_device_to_host_copies": captures,
            "device_to_host_vector_copies": 1,
            "device_to_host_vector_values": row_count,
            "device_to_host_vector_bytes": row_count * 4,
            "device_to_host_scalar_copies": 0,
        },
        "synchronizations": 2,
        "scalar_fallbacks": 0,
        "state_import_slots": len(imported_indices),
        "state_import_slot_indices": list(imported_indices),
        "scatter_state": bool(scatter_state),
        "steady_packed_state_reused": not imported_indices and not bool(scatter_state),
        "profiler_contract": {
            "expected_execution_buckets": (
                ["packed_native"]
                if exact_row_launches == 0
                else ["exact_row_local", "packed_native"]
            ),
            "expected_exact_row_local_kernel_launches": exact_row_launches,
            "require_cached_build": True,
        },
    }


__all__ = ["build_packed_decode_execution_manifest"]
