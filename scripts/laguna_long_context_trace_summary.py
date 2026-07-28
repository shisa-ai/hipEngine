#!/usr/bin/env python3
"""Attach a compact all-family prefill summary to a Laguna rocprofv3 trace."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

_RESOURCE_FIELDS = (
    "Kernel_Name",
    "Workgroup_Size_X",
    "Grid_Size_X",
    "Grid_Size_Y",
    "VGPR_Count",
    "SGPR_Count",
    "LDS_Block_Size",
    "Scratch_Size",
)

_FAMILY_ORDER = (
    "embedding",
    "source_f16_projection",
    "selected_q4_gate_up",
    "selected_q4_q6_down",
    "selected_iq_gate_up",
    "selected_iq_down",
    "dense_shared_quant_projection",
    "router",
    "prefill_kv_write",
    "global_attention",
    "swa_attention",
    "norm_rope_gate",
    "activation_reduce_residual",
    "lm_head_argmax",
    "other",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _duration_ns(row: Mapping[str, Any]) -> int:
    return int(row["End_Timestamp"]) - int(row["Start_Timestamp"])


def _is_embedding(name: str) -> bool:
    return (
        "gguf_" in name
        and "_embedding_bf16_out_kernel" in name
    )


def _is_argmax_end(name: str) -> bool:
    return "argmax_stage2_kernel" in name


def _kernel_family(name: str) -> str:
    """Map one demangled Laguna prefill symbol to an attribution family."""

    lowered = str(name).lower()
    if _is_embedding(lowered):
        return "embedding"
    if "argmax_stage" in lowered:
        return "lm_head_argmax"
    if (
        "laguna_f16w_" in lowered
        and any(
            marker in lowered for marker in ("gemv_kernel", "tiled_exact_kernel", "wmma")
        )
    ) or any(
        marker in lowered
        for marker in (
            "cijk_alik_bljk_hss_",
            "bf16_to_fp16_scaled_rows_kernel",
            "f32_scale_rows_kernel",
            "f32_scale_rows_to_bf16_kernel",
        )
    ):
        return "source_f16_projection"
    if "gguf_q8_1_mmq_ds8_f32_pack_bf16_kernel" in lowered:
        return "selected_q4_gate_up"
    if "gguf_q8_1_mmq_ds4_f32_pack_bf16_kernel" in lowered:
        return "selected_q4_q6_down"
    if "q4_k_t16_selected_dual" in lowered and any(
        marker in lowered
        for marker in ("gemv_kernel", "grouped_smallm_kernel", "mmq64x32")
    ):
        if (
            "mmq64x32" in lowered
            and "kernel<1, true, false, 64" in lowered
        ):
            return "selected_q4_q6_down"
        return "selected_q4_gate_up"
    if any(
        marker in lowered
        for marker in (
            "qk_t16_selected",
            "q4_k_t16_selected",
            "q5_k_t16_selected",
            "q6_k_t16_selected",
        )
    ) and any(
        marker in lowered
        for marker in ("gemv_kernel", "grouped_smallm_kernel", "mmq64x32")
    ):
        return "selected_q4_q6_down"
    if any(
        marker in lowered
        for marker in (
            "gguf_iq2_xs_selected_dual_",
            "gguf_iq3_xxs_selected_dual_",
        )
    ):
        return "selected_iq_gate_up"
    if any(
        marker in lowered
        for marker in (
            "gguf_iq3_xxs_selected_gemv",
            "gguf_iq4_xs_selected_gemv",
        )
    ):
        return "selected_iq_down"
    if "laguna_global_attention_prefill" in lowered:
        return "global_attention"
    if "laguna_swa_attention_prefill" in lowered:
        return "swa_attention"
    if "write_kv_rows" in lowered and "laguna_" in lowered:
        return "prefill_kv_write"
    if "router" in lowered or "sigmoid_correction_topk" in lowered:
        return "router"
    if any(
        marker in lowered
        for marker in (
            "rmsnorm",
            "rotary",
            "rope",
            "softplus_head_gate",
            "attention_gate",
        )
    ):
        return "norm_rope_gate"
    if "selected" not in lowered and any(
        marker in lowered
        for marker in (
            "q4_k_pack8",
            "q5_k_pack8",
            "q6_k_pack8",
            "q4_k_gemv",
            "q5_k_gemv",
            "q6_k_gemv",
            "q4_k_t16_gemv",
            "q5_k_t16_gemv",
            "q6_k_t16_gemv",
            "gguf_k_prefill_out_kernel",
            "_k_prefill_wmma_kernel",
        )
    ):
        return "dense_shared_quant_projection"
    if any(
        marker in lowered
        for marker in (
            "silu",
            "weighted_sum",
            "weighted_lanes",
            "bf16_add",
            "elementwise",
            "residual",
            "gate_mul",
        )
    ):
        return "activation_reduce_residual"
    return "other"


def _attention_family(name: str) -> str | None:
    family = _kernel_family(name)
    return family if family in {"global_attention", "swa_attention"} else None


def _is_dense_initial_causal_softmax(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            "laguna_dense_initial_causal_softmax_f32_kernel",
            "laguna_dense_initial_causal_softmax_wave_rows_f32_kernel",
        )
    )


def _dense_initial_blas_attention_families(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, str]:
    """Identify the bounded dense-initial BLAS attention composites."""

    families: dict[int, str] = {}
    for index, row in enumerate(rows):
        name = str(row["Kernel_Name"])
        if "laguna_dense_initial_cache_bf16_to_f32_kernel<" not in name:
            continue
        family = (
            "global_attention"
            if "kernel<true>" in name
            else "swa_attention"
        )
        packed_query_transpose = (
            index + 1 < len(rows)
            and "laguna_dense_initial_query_head_transpose_f32_kernel<true>"
            in str(rows[index + 1]["Kernel_Name"])
        )
        direct_packed_query = (
            not packed_query_transpose
            and index + 3 < len(rows)
            and str(rows[index + 1]["Kernel_Name"]).startswith("Cijk_")
            and _is_dense_initial_causal_softmax(
                str(rows[index + 2]["Kernel_Name"])
            )
            and str(rows[index + 3]["Kernel_Name"]).startswith("Cijk_")
        )
        if packed_query_transpose or direct_packed_query:
            qk_index = index + (2 if packed_query_transpose else 1)
            softmax_index = qk_index + 1
            pv_index = softmax_index + 1
            if pv_index >= len(rows):
                raise ValueError(
                    "packed-query dense-initial BLAS attention trace is truncated"
                )
            qk_name = str(rows[qk_index]["Kernel_Name"])
            softmax_name = str(rows[softmax_index]["Kernel_Name"])
            pv_name = str(rows[pv_index]["Kernel_Name"])
            if (
                not qk_name.startswith("Cijk_")
                or not _is_dense_initial_causal_softmax(softmax_name)
                or not pv_name.startswith("Cijk_")
            ):
                raise ValueError(
                    "packed-query dense-initial BLAS attention trace does not "
                    "match widen + optional pack + QK + softmax + PV"
                )
            final_index = pv_index
            if (
                pv_index + 1 < len(rows)
                and "laguna_dense_initial_query_head_transpose_f32_kernel<false>"
                in str(rows[pv_index + 1]["Kernel_Name"])
            ):
                final_index = pv_index + 1
            for target in range(index, final_index + 1):
                if target in families:
                    raise ValueError(
                        "dense-initial BLAS attention composites overlap"
                    )
                families[target] = family
            continue

        softmax_index = index + 9
        final_index = index + 17
        if final_index >= len(rows):
            raise ValueError("dense-initial BLAS attention trace is truncated")
        qk_names = [
            str(rows[target]["Kernel_Name"])
            for target in range(index + 1, softmax_index)
        ]
        softmax_name = str(rows[softmax_index]["Kernel_Name"])
        pv_names = [
            str(rows[target]["Kernel_Name"])
            for target in range(softmax_index + 1, final_index + 1)
        ]
        if (
            len(qk_names) != 8
            or len(pv_names) != 8
            or not all(name.startswith("Cijk_") for name in (*qk_names, *pv_names))
            or not _is_dense_initial_causal_softmax(softmax_name)
        ):
            raise ValueError(
                "dense-initial BLAS attention trace does not match "
                "widen + 8 QK + softmax + 8 PV"
            )
        for target in range(index, final_index + 1):
            if target in families:
                raise ValueError("dense-initial BLAS attention composites overlap")
            families[target] = family
    return families


def _trace_row_family(
    rows: Sequence[Mapping[str, Any]],
    index: int,
    *,
    attention_families: Mapping[int, str] | None = None,
) -> str:
    """Classify one row, including the otherwise ambiguous final Q6 LM head."""

    if index < 0 or index >= len(rows):
        raise IndexError("trace row index out of range")
    if attention_families is not None and index in attention_families:
        return str(attention_families[index])
    if index + 2 < len(rows):
        next_name = str(rows[index + 1]["Kernel_Name"])
        final_name = str(rows[index + 2]["Kernel_Name"])
        if "argmax_stage1" in next_name and "argmax_stage2" in final_name:
            return "lm_head_argmax"
    return _kernel_family(str(rows[index]["Kernel_Name"]))


def _read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Grid_Size_Y",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("rocprof trace is empty or missing required kernel columns")
    rows.sort(key=lambda row: (int(row["Start_Timestamp"]), int(row["Dispatch_Id"])))
    return rows


def _segment_requests(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    started = False
    for row in rows:
        name = str(row["Kernel_Name"])
        if _is_embedding(name):
            if not started:
                current = []
                started = True
        if not started:
            continue
        current.append(row)
        if not _is_argmax_end(name):
            continue
        embedding_rows = [
            int(item["Grid_Size_Y"])
            for item in current
            if _is_embedding(str(item["Kernel_Name"]))
        ]
        if not embedding_rows:
            raise ValueError("profile segment ended without an embedding launch")
        segments.append(
            {
                "length": sum(embedding_rows),
                "chunks": len(embedding_rows),
                "rows": list(current),
            }
        )
        current = []
        started = False
    if started:
        raise ValueError("rocprof trace ended inside a Laguna prefill request")
    return segments


def _summarize_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(segment["rows"])
    attention_families = _dense_initial_blas_attention_families(rows)
    kernel_sum = sum(_duration_ns(row) for row in rows)
    if kernel_sum <= 0:
        raise ValueError("profile segment kernel sum must be positive")
    start = min(int(row["Start_Timestamp"]) for row in rows)
    end = max(int(row["End_Timestamp"]) for row in rows)
    family_calls: dict[str, int] = defaultdict(int)
    family_duration: dict[str, int] = defaultdict(int)
    # The resident prompt path ends with one LM-head launch and two argmax
    # stages. Classify that three-launch chain together because the Q6 head
    # symbol is also used by ordinary quantized projections. Synthetic/partial
    # traces without both argmax stages retain normal symbol classification.
    for index, row in enumerate(rows):
        family = _trace_row_family(
            rows,
            index,
            attention_families=attention_families,
        )
        family_calls[family] += 1
        family_duration[family] += _duration_ns(row)
    if sum(family_duration.values()) != kernel_sum:
        raise ValueError("all-family attribution does not cover the complete kernel sum")
    attention_ns = sum(
        family_duration[family] for family in ("global_attention", "swa_attention")
    )
    return {
        "length": int(segment["length"]),
        "chunks": int(segment["chunks"]),
        "dispatches": len(rows),
        "kernel_sum_ns": kernel_sum,
        "kernel_span_ns": end - start,
        "attention_duration_ns": attention_ns,
        "attention_share_of_kernel_sum": attention_ns / kernel_sum,
        "families": {
            family: {
                "calls": family_calls[family],
                "duration_ns": family_duration[family],
                "share_of_kernel_sum": family_duration[family] / kernel_sum,
            }
            for family in _FAMILY_ORDER
        },
    }


def _aggregate_segments(segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for segment in segments:
        grouped[int(segment["length"])].append(segment)
    result = {}
    for length, selected in sorted(grouped.items()):
        result[str(length)] = {
            "passes": len(selected),
            "chunks_per_pass": [int(item["chunks"]) for item in selected],
            "dispatches_per_pass": [int(item["dispatches"]) for item in selected],
            "median_kernel_sum_ns": statistics.median(
                int(item["kernel_sum_ns"]) for item in selected
            ),
            "median_kernel_span_ns": statistics.median(
                int(item["kernel_span_ns"]) for item in selected
            ),
            "median_attention_duration_ns": statistics.median(
                int(item["attention_duration_ns"]) for item in selected
            ),
            "median_attention_share_of_kernel_sum": statistics.median(
                float(item["attention_share_of_kernel_sum"]) for item in selected
            ),
            "families": {
                family: {
                    "calls_per_pass": [
                        int(item["families"][family]["calls"]) for item in selected
                    ],
                    "median_duration_ns": statistics.median(
                        int(item["families"][family]["duration_ns"])
                        for item in selected
                    ),
                    "median_share_of_kernel_sum": statistics.median(
                        float(item["families"][family]["share_of_kernel_sum"])
                        for item in selected
                    ),
                }
                for family in _FAMILY_ORDER
            },
        }
    return result


def _trace_resources(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    attention_families = _dense_initial_blas_attention_families(rows)
    resources = {}
    for index, row in enumerate(rows):
        family = attention_families.get(index)
        if family is None:
            family = _attention_family(str(row["Kernel_Name"]))
        if family is None:
            continue
        resources.setdefault(family, {field: str(row[field]) for field in _RESOURCE_FIELDS})
    return [resources[key] for key in sorted(resources)]


def _family_resources(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    attention_families = _dense_initial_blas_attention_families(rows)
    resources: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        name = str(row["Kernel_Name"])
        family = _trace_row_family(
            rows,
            index,
            attention_families=attention_families,
        )
        key = (family, name)
        item = resources.setdefault(
            key,
            {
                "family": family,
                **{field: str(row[field]) for field in _RESOURCE_FIELDS},
                "calls": 0,
                "duration_ns": 0,
                "observed_grid_sizes": set(),
            },
        )
        item["calls"] += 1
        item["duration_ns"] += _duration_ns(row)
        item["observed_grid_sizes"].add(
            (str(row["Grid_Size_X"]), str(row["Grid_Size_Y"]))
        )
    output = []
    for item in resources.values():
        item["observed_grid_sizes"] = [
            {"x": x, "y": y} for x, y in sorted(item["observed_grid_sizes"])
        ]
        output.append(item)
    return sorted(
        output,
        key=lambda item: (str(item["family"]), -int(item["duration_ns"])),
    )


def attach_summary(
    child: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    trace_path: Path,
    trace_sha256: str,
) -> dict[str, Any]:
    raw_segments = _segment_requests(rows)
    summarized = [_summarize_segment(segment) for segment in raw_segments]
    expected = [int(child["protocol"]["warmup_rows"])] + [
        int(row["length"]) for row in child["rows"]
    ]
    actual = [int(segment["length"]) for segment in summarized]
    if actual != expected:
        raise ValueError(f"profile segment lengths {actual} do not match child order {expected}")
    warmup = summarized[0]
    timed = summarized[1:]
    required = {int(value) for value in child["protocol"]["lengths"]}
    if {int(segment["length"]) for segment in timed} != required:
        raise ValueError("profile trace does not cover every required LPF-5 length")
    output = dict(child)
    output["profiler"] = {
        "kind": "rocprofv3_kernel_trace_laguna_lpf5_long_context",
        "attached_at": datetime.now(timezone.utc).isoformat(),
        "raw_csv_committed": False,
        "raw_csv_path": str(trace_path),
        "raw_csv_sha256": trace_sha256,
        "segmentation": (
            "request starts at its first Q4 embedding, includes every chunk, and ends at "
            "argmax stage 2; summed embedding Grid_Size_Y equals logical context length"
        ),
        "warmup": warmup,
        "lengths": _aggregate_segments(timed),
        "attention_resources": _trace_resources(rows),
        "family_resources": _family_resources(rows),
    }
    output["derived_artifact_repairs"] = [
        {
            "kind": "attach_rocprof_summary",
            "source_child_sha256": hashlib.sha256(
                (json.dumps(child, indent=2, sort_keys=True) + "\n").encode("utf-8")
            ).hexdigest(),
            "source_trace_sha256": trace_sha256,
            "measurement_values_changed": False,
        }
    ]
    return output


def main() -> int:
    args = _parse_args()
    child = json.loads(args.child.read_text(encoding="utf-8"))
    if not child.get("pass") or child.get("performance_claim"):
        raise ValueError("LPF-5 child must be a passing attribution-only artifact")
    rows = _read_trace(args.trace)
    trace_bytes = args.trace.read_bytes()
    result = attach_summary(
        child,
        rows,
        trace_path=args.trace,
        trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
