#!/usr/bin/env python3
"""Audit layer-15 full-attention Q/K head-RMSNorm, RoPE, and KV-write."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    Qwen35GGUFResidentSession,
    _copy_bf16_ptr_to_host_f32,
    _copy_f32_ptr_to_host,
    _rope_tables,
)
from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    bf16_roundtrip_array,
    delta_summary,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    summarize_capture,
)
from scripts.llamacpp_mtp_audit_layer15_full_attention_qkv_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER15_QKV,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter439-layer15-qk-norm-rotary-kv-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

BoundaryCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]
WeightConfigLoader = Callable[[Path, int], dict[str, Any]]

PRE_SPLIT_FIELDS = ("full_q_f32", "full_k_f32", "full_v_f32")
SPLIT_FIELDS = ("full_query_raw_f32", "full_gate_f32", "full_key_raw_f32")
ROTARY_FIELDS = ("full_query_f32", "full_key_f32")
CACHE_FIELDS = ("key_cache_position_f32", "value_cache_position_f32")


# Keep this tolerance small: it only covers FP32 reduction/rsqrt/FMA ordering in
# the head-RMSNorm+RoPE kernel. BF16 split and KV-write boundaries are still
# required to match exactly.
DEFAULT_ROTARY_NEAR_ATOL = 1.0e-5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv-artifact", type=Path, default=DEFAULT_LAYER15_QKV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=15)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument(
        "--rotary-near-atol",
        type=float,
        default=DEFAULT_ROTARY_NEAR_ATOL,
    )
    parser.add_argument("--iteration", type=int, default=439)
    args = parser.parse_args()

    artifact = audit_layer15_qk_norm_rotary_kv_oracle(
        qkv_artifact_path=args.qkv_artifact,
        model_path=args.model,
        layer_id=args.layer_id,
        max_sequence_length=args.max_sequence_length,
        rotary_near_atol=args.rotary_near_atol,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "classification": artifact["classification"],
                "layer_id": artifact["layer_id"],
                "preflight_classification": artifact.get("preflight_classification"),
                "field_classifications": field_classifications(artifact),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer15_qk_norm_rotary_kv_oracle(
    *,
    qkv_artifact_path: Path,
    model_path: Path | None = None,
    layer_id: int = 15,
    max_sequence_length: int | None = None,
    rotary_near_atol: float = DEFAULT_ROTARY_NEAR_ATOL,
    iteration: int = 439,
    boundary_capture_fn: BoundaryCaptureFn | None = None,
    weight_config_loader: WeightConfigLoader | None = None,
) -> dict[str, Any]:
    qkv_artifact = json.loads(qkv_artifact_path.read_text())
    validate_layer15_qkv_artifact(qkv_artifact, expected_layer_id=int(layer_id))
    resolved_model = Path(model_path or qkv_artifact["model"])
    prompt_tokens = tuple(int(token) for token in qkv_artifact["prompt_tokens"])
    position = int(qkv_artifact["position"])
    token_id = int(prompt_tokens[position])
    capture_fn = boundary_capture_fn or capture_layer15_qk_norm_rotary_kv_boundary
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(layer_id),
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            qkv_artifact_path=qkv_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    preflight = compare_qkv_preflight(capture=capture, artifact=qkv_artifact)
    preflight_classification = classify_qkv_preflight(
        preflight,
        capture=capture,
        layer_id=int(layer_id),
    )
    if preflight_classification != "layer15_qk_norm_rotary_preflight_matches_qkv_artifact":
        return blocked_artifact(
            qkv_artifact_path=qkv_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            preflight=preflight,
            preflight_classification=preflight_classification,
            iteration=iteration,
        )
    loader = weight_config_loader or load_qk_norm_weights_and_config
    weights_config = loader(resolved_model, int(layer_id))
    split_results = build_split_results(capture=capture)
    rotary_results = build_rotary_results(
        capture=capture,
        weights_config=weights_config,
        position=position,
        rotary_near_atol=float(rotary_near_atol),
    )
    cache_results = build_cache_results(capture=capture, rotary_results=rotary_results)
    classification = classify_layer15_qk_norm_rotary_kv(
        preflight_classification,
        split_results,
        rotary_results,
        cache_results,
    )
    return {
        "schema": 1,
        "kind": "layer15_qk_norm_rotary_kv_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "qkv_artifact_path": str(qkv_artifact_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "layer15 ready full_q/full_k/full_v BF16 QKV projection artifact",
            "qkv_classification": qkv_artifact.get("classification"),
            "split_formula": "split full_q BF16 rows into F32 query_raw and BF16 gate",
            "key_raw_formula": "BF16 full_k projection converted to F32",
            "rotary_formula": (
                "per-head RMSNorm(src, attn_{q,k}_norm.weight, eps_model) "
                "then llama.cpp/Qwen half-rotation RoPE"
            ),
            "kv_write_formula": (
                "key_cache[position] = BF16(full_key_f32); "
                "value_cache[position] = BF16(full_v_bf16)"
            ),
            "exact_required_for": ["split", "key_raw", "gate", "kv_write"],
            "fp32_tolerance_only_for": "head_rmsnorm_partial_rotary",
        },
        "preflight": preflight,
        "preflight_classification": preflight_classification,
        "weights_config": summarize_weights_config(weights_config),
        "hipengine_capture": summarize_capture(capture),
        "split_results": split_results,
        "rotary_results": rotary_results,
        "cache_results": cache_results,
        "rotary_near_atol": float(rotary_near_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer15_qkv_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if artifact.get("status") != "ready":
        raise ValueError("layer15 QKV artifact must be ready")
    acceptable_classifications = {
        "layer15_full_attention_qkv_matches_bf16_oracle_exactly",
        "layer15_full_attention_qkv_matches_bf16_oracle_within_rounding",
    }
    if artifact.get("classification") not in acceptable_classifications:
        raise ValueError("layer15 QKV artifact must be ready under the BF16 contract")
    if int(artifact.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer15 QKV layer_id does not match requested layer")
    summary = (artifact.get("hipengine_capture") or {}).get("summary") or {}
    if str(summary.get("layer_type")) != "full_attention":
        raise ValueError("layer15 QKV artifact must be a full_attention capture")
    if int(summary.get("preceding_layer_count", -1)) != int(expected_layer_id):
        raise ValueError("layer15 QKV artifact must include exact preceding layers")
    field_summaries = (
        (artifact.get("hipengine_capture") or {}).get("field_summaries") or {}
    )
    missing = [field for field in PRE_SPLIT_FIELDS if field not in field_summaries]
    if missing:
        raise ValueError(f"layer15 QKV artifact missing field summaries: {missing}")
    if artifact.get("next_action") != (
        "audit_layer15_qk_norm_rotary_or_kv_write_under_bf16_contract"
    ):
        raise ValueError("layer15 QKV artifact must point to QK norm/rotary audit")


def compare_qkv_preflight(
    *,
    capture: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    field_summaries = artifact["hipengine_capture"]["field_summaries"]
    fields = capture["fields"]
    results: dict[str, Any] = {}
    for field in PRE_SPLIT_FIELDS:
        values = np.asarray(fields[field], dtype=np.float32)
        actual_sha = sha256_float32(values)
        expected_sha = str(field_summaries[field]["sha256"])
        results[field] = {
            "field": field,
            "reference_source": "layer15_full_attention_qkv_oracle",
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "exact_hash_match": actual_sha == expected_sha,
            "summary": summarize_array(values),
        }
    return results


def classify_qkv_preflight(
    preflight: Mapping[str, Any],
    *,
    capture: Mapping[str, Any],
    layer_id: int,
) -> str:
    summary = capture.get("summary") or {}
    if int(summary.get("layer_id", -1)) != int(layer_id):
        return "layer15_qk_norm_rotary_wrong_layer_capture"
    if str(summary.get("layer_type")) != "full_attention":
        return "layer15_qk_norm_rotary_wrong_layer_type"
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        return "layer15_qk_norm_rotary_wrong_preceding_layer_count"
    if not all(result.get("exact_hash_match") for result in preflight.values()):
        return "layer15_qk_norm_rotary_blocked_qkv_input_mismatch"
    return "layer15_qk_norm_rotary_preflight_matches_qkv_artifact"


def build_split_results(*, capture: Mapping[str, Any]) -> dict[str, Any]:
    fields = capture["fields"]
    dims = capture_dimensions(capture)
    full_q = np.asarray(fields["full_q_f32"], dtype=np.float32)
    full_k = np.asarray(fields["full_k_f32"], dtype=np.float32)
    expected_query_raw, expected_gate = split_full_q_projection(
        full_q,
        num_q_heads=dims["num_q_heads"],
        head_dim=dims["head_dim"],
    )
    specs = {
        "full_query_raw_f32": {
            "expected": expected_query_raw,
            "actual": np.asarray(fields["full_query_raw_f32"], dtype=np.float32),
            "oracle": "split first head_dim values from each full_q head pair",
        },
        "full_gate_f32": {
            "expected": expected_gate,
            "actual": np.asarray(fields["full_gate_f32"], dtype=np.float32),
            "oracle": "split second head_dim BF16 values from each full_q head pair",
        },
        "full_key_raw_f32": {
            "expected": full_k,
            "actual": np.asarray(fields["full_key_raw_f32"], dtype=np.float32),
            "oracle": "BF16 full_k projection converted to F32",
        },
    }
    return {
        name: field_result(
            field=name,
            expected=spec["expected"],
            actual=spec["actual"],
            oracle=str(spec["oracle"]),
            classifier=classify_exact_delta,
        )
        for name, spec in specs.items()
    }


def build_rotary_results(
    *,
    capture: Mapping[str, Any],
    weights_config: Mapping[str, Any],
    position: int,
    rotary_near_atol: float,
) -> dict[str, Any]:
    fields = capture["fields"]
    dims = capture_dimensions(capture)
    cfg = weights_config["config"]
    cos_table, sin_table = _rope_tables(
        max_positions=int(cfg["max_positions"]),
        rotary_dim=dims["rotary_dim"],
        base=float(cfg["rope_freq_base"]),
    )
    if int(position) >= int(cos_table.shape[0]):
        raise ValueError("position exceeds generated RoPE table")
    q_weight = np.asarray(weights_config["q_norm"][0], dtype=np.float32)
    k_weight = np.asarray(weights_config["k_norm"][0], dtype=np.float32)
    query_expected = head_rmsnorm_partial_rotary(
        np.asarray(fields["full_query_raw_f32"], dtype=np.float32),
        q_weight,
        cos_table[int(position)],
        sin_table[int(position)],
        num_heads=dims["num_q_heads"],
        head_dim=dims["head_dim"],
        rotary_dim=dims["rotary_dim"],
        eps=float(cfg["rms_norm_eps"]),
    )
    key_expected = head_rmsnorm_partial_rotary(
        np.asarray(fields["full_key_raw_f32"], dtype=np.float32),
        k_weight,
        cos_table[int(position)],
        sin_table[int(position)],
        num_heads=dims["num_kv_heads"],
        head_dim=dims["head_dim"],
        rotary_dim=dims["rotary_dim"],
        eps=float(cfg["rms_norm_eps"]),
    )
    return {
        "full_query_f32": field_result(
            field="full_query_f32",
            expected=query_expected,
            actual=np.asarray(fields["full_query_f32"], dtype=np.float32),
            oracle="CPU mirror of gguf_head_rmsnorm_partial_rotary_f32_weight_row for Q",
            classifier=lambda delta: classify_near_delta(delta, near_atol=rotary_near_atol),
        ),
        "full_key_f32": field_result(
            field="full_key_f32",
            expected=key_expected,
            actual=np.asarray(fields["full_key_f32"], dtype=np.float32),
            oracle="CPU mirror of gguf_head_rmsnorm_partial_rotary_f32_weight_row for K",
            classifier=lambda delta: classify_near_delta(delta, near_atol=rotary_near_atol),
        ),
    }


def build_cache_results(
    *,
    capture: Mapping[str, Any],
    rotary_results: Mapping[str, Any],
) -> dict[str, Any]:
    _ = rotary_results
    fields = capture["fields"]
    key_from_hip = bf16_roundtrip_array(np.asarray(fields["full_key_f32"], dtype=np.float32))
    value_from_hip = np.asarray(fields["full_v_f32"], dtype=np.float32)
    return {
        "key_cache_position_f32": field_result(
            field="key_cache_position_f32",
            expected=key_from_hip,
            actual=np.asarray(fields["key_cache_position_f32"], dtype=np.float32),
            oracle="BF16(full_key_f32 HIP rotary output) written through KVLiveSpans",
            classifier=classify_exact_delta,
        ),
        "value_cache_position_f32": field_result(
            field="value_cache_position_f32",
            expected=value_from_hip,
            actual=np.asarray(fields["value_cache_position_f32"], dtype=np.float32),
            oracle="BF16(full_v BF16 projection value) written through KVLiveSpans",
            classifier=classify_exact_delta,
        ),
    }


def field_result(
    *,
    field: str,
    expected: np.ndarray,
    actual: np.ndarray,
    oracle: str,
    classifier: Callable[[Mapping[str, Any]], str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected32 = np.asarray(expected, dtype=np.float32)
    actual32 = np.asarray(actual, dtype=np.float32)
    delta = delta_summary(expected32, actual32)
    result = {
        "field": field,
        "oracle": oracle,
        "expected_summary": summarize_array(expected32),
        "hipengine_summary": summarize_array(actual32),
        "delta": delta,
        "classification": classifier(delta),
    }
    if extra:
        result.update(extra)
    return result


def classify_exact_delta(delta: Mapping[str, Any]) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "qk_norm_rotary_kv_oracle_unavailable"
    if delta.get("exact_match"):
        return "qk_norm_rotary_kv_matches_exactly"
    return "qk_norm_rotary_kv_mismatch"


def classify_near_delta(delta: Mapping[str, Any], *, near_atol: float) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "qk_norm_rotary_kv_oracle_unavailable"
    if delta.get("exact_match"):
        return "qk_norm_rotary_kv_matches_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "qk_norm_rotary_matches_within_fp32_kernel_tolerance"
    return "qk_norm_rotary_kv_mismatch"


def classify_layer15_qk_norm_rotary_kv(
    preflight_classification: str,
    split_results: Mapping[str, Any],
    rotary_results: Mapping[str, Any],
    cache_results: Mapping[str, Any],
) -> str:
    if preflight_classification != "layer15_qk_norm_rotary_preflight_matches_qkv_artifact":
        return preflight_classification
    split_classes = [result["classification"] for result in split_results.values()]
    rotary_classes = [result["classification"] for result in rotary_results.values()]
    cache_classes = [result["classification"] for result in cache_results.values()]
    exact = "qk_norm_rotary_kv_matches_exactly"
    near = "qk_norm_rotary_matches_within_fp32_kernel_tolerance"
    if any("mismatch" in item for item in (*split_classes, *rotary_classes, *cache_classes)):
        return "layer15_qk_norm_rotary_kv_mismatch"
    if any(
        item.endswith("unavailable")
        for item in (*split_classes, *rotary_classes, *cache_classes)
    ):
        return "layer15_qk_norm_rotary_kv_oracle_unavailable"
    if all(item == exact for item in (*split_classes, *rotary_classes, *cache_classes)):
        return "layer15_qk_norm_rotary_kv_matches_cpu_oracle_exactly"
    if all(item == exact for item in (*split_classes, *cache_classes)) and all(
        item in {exact, near} for item in rotary_classes
    ):
        return "layer15_qk_norm_rotary_matches_cpu_oracle_within_fp32_tolerance"
    return "layer15_qk_norm_rotary_kv_mismatch"


def status_from_classification(classification: str) -> str:
    if classification.endswith("unavailable"):
        return "unavailable"
    if "blocked" in classification or "wrong" in classification:
        return "blocked"
    if "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer15_qk_norm_rotary_kv_matches_cpu_oracle_exactly",
        "layer15_qk_norm_rotary_matches_cpu_oracle_within_fp32_tolerance",
    }:
        return "audit_layer15_full_attention_scores_or_attn_output_under_bf16_contract"
    if classification == "layer15_qk_norm_rotary_blocked_qkv_input_mismatch":
        return "reconcile_layer15_qkv_projection_before_qk_norm_rotary"
    if "wrong" in classification:
        return "inspect_layer15_qk_norm_rotary_capture_metadata"
    if classification == "layer15_qk_norm_rotary_kv_mismatch":
        return "inspect_layer15_qk_norm_rotary_or_kv_write_kernel"
    return "rerun_layer15_qk_norm_rotary_kv_oracle_on_rocm_host"


def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for section in ("split_results", "rotary_results", "cache_results"):
        for name, result in artifact.get(section, {}).items():
            out[name] = result.get("classification")
    return out


def split_full_q_projection(
    full_q: np.ndarray,
    *,
    num_q_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(full_q, dtype=np.float32)
    expected = int(num_q_heads) * 2 * int(head_dim)
    if values.size != expected:
        raise ValueError(f"full_q size {values.size} does not match expected {expected}")
    rows = values.reshape(int(num_q_heads), 2 * int(head_dim))
    query = np.ascontiguousarray(rows[:, : int(head_dim)].reshape(-1))
    gate = np.ascontiguousarray(rows[:, int(head_dim) :].reshape(-1))
    return query.astype(np.float32, copy=False), gate.astype(np.float32, copy=False)


def head_rmsnorm_partial_rotary(
    values: np.ndarray,
    weight: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    threads: int = 256,
) -> np.ndarray:
    src = np.asarray(values, dtype=np.float32).reshape(int(num_heads), int(head_dim))
    w = np.asarray(weight, dtype=np.float32)
    if w.size != int(head_dim):
        raise ValueError("head RMSNorm weight length must match head_dim")
    cos_row = np.asarray(cos, dtype=np.float32)
    sin_row = np.asarray(sin, dtype=np.float32)
    if cos_row.size < int(rotary_dim) or sin_row.size < int(rotary_dim):
        raise ValueError("RoPE cos/sin row shorter than rotary_dim")
    out = np.empty_like(src, dtype=np.float32)
    for head in range(int(num_heads)):
        out[head] = head_rmsnorm_partial_rotary_row(
            src[head],
            w,
            cos_row,
            sin_row,
            head_dim=int(head_dim),
            rotary_dim=int(rotary_dim),
            eps=float(eps),
            threads=int(threads),
        )
    return np.ascontiguousarray(out.reshape(-1))


def head_rmsnorm_partial_rotary_row(
    src: np.ndarray,
    weight: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    threads: int = 256,
) -> np.ndarray:
    partial = np.zeros((int(threads),), dtype=np.float32)
    src32 = np.asarray(src, dtype=np.float32)
    for tid in range(int(threads)):
        acc = np.float32(0.0)
        for dim in range(tid, int(head_dim), int(threads)):
            value = np.float32(src32[dim])
            acc = np.float32(acc + np.float32(value * value))
        partial[tid] = acc
    stride = int(threads) >> 1
    while stride > 0:
        for tid in range(stride):
            partial[tid] = np.float32(partial[tid] + partial[tid + stride])
        stride >>= 1
    variance = np.float32(partial[0] / np.float32(head_dim) + np.float32(eps))
    inv_rms = np.float32(1.0 / math.sqrt(float(variance)))
    half_rotary = int(rotary_dim) // 2
    out = np.empty((int(head_dim),), dtype=np.float32)
    weight32 = np.asarray(weight, dtype=np.float32)
    cos32 = np.asarray(cos, dtype=np.float32)
    sin32 = np.asarray(sin, dtype=np.float32)
    for dim in range(int(head_dim)):
        value = np.float32(np.float32(src32[dim] * inv_rms) * weight32[dim])
        out_value = value
        if dim < int(rotary_dim):
            pair_dim = dim + half_rotary if dim < half_rotary else dim - half_rotary
            paired = np.float32(np.float32(src32[pair_dim] * inv_rms) * weight32[pair_dim])
            rotated = np.float32(-paired if dim < half_rotary else paired)
            out_value = np.float32(
                np.float32(value * cos32[dim]) + np.float32(rotated * sin32[dim])
            )
        out[dim] = out_value
    return out


def capture_dimensions(capture: Mapping[str, Any]) -> dict[str, int]:
    summary = capture.get("summary") or {}
    num_q_heads = int(summary["num_q_heads"])
    num_kv_heads = int(summary["num_kv_heads"])
    head_dim = int(summary["head_dim"])
    rotary_dim = int(summary["rotary_dim"])
    q_width = int(summary.get("q_width", num_q_heads * head_dim))
    kv_width = int(summary.get("kv_width", num_kv_heads * head_dim))
    if q_width != num_q_heads * head_dim:
        raise ValueError("q_width does not match num_q_heads * head_dim")
    if kv_width != num_kv_heads * head_dim:
        raise ValueError("kv_width does not match num_kv_heads * head_dim")
    return {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "q_width": q_width,
        "kv_width": kv_width,
    }


def load_qk_norm_weights_and_config(model_path: Path, layer_id: int) -> dict[str, Any]:
    reader = GGUFReader(model_path)
    config = qwen35_gguf_config_from_metadata(reader.info)
    weights: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for slot, suffix in (
        ("q_norm", "attn_q_norm.weight"),
        ("k_norm", "attn_k_norm.weight"),
    ):
        tensor_name = f"blk.{layer_id}.{suffix}"
        tensor = reader.tensor_info(tensor_name)
        values = np.asarray(reader.dequantize_tensor(tensor_name), dtype=np.float32)
        weights[slot] = (
            values,
            {
                "tensor_name": tensor_name,
                "ggml_type": tensor.ggml_type_name,
                "shape": list(values.shape),
                "summary": summarize_array(values),
                "sha256": sha256_float32(values),
            },
        )
    weights["config"] = {
        "num_q_heads": int(config.head_count),
        "num_kv_heads": int(config.head_count_kv),
        "head_dim": int(config.key_length),
        "value_head_dim": int(config.value_length),
        "rotary_dim": int(config.rope_dimension_count),
        "rms_norm_eps": float(config.rms_norm_eps),
        "rope_freq_base": float(config.rope_freq_base),
        "max_positions": int(config.context_length),
        "layer_type": str(config.layer_types[int(layer_id)]),
    }
    return weights


def summarize_weights_config(weights_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "q_norm": weights_config["q_norm"][1],
        "k_norm": weights_config["k_norm"][1],
        "config": dict(weights_config["config"]),
    }


def capture_layer15_qk_norm_rotary_kv_boundary(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    layer_id: int,
    max_sequence_length: int | None,
) -> dict[str, Any]:
    max_seq = int(max_sequence_length or max(len(prompt_tokens) + 8, 32))
    with Qwen35GGUFResidentSession(model_path, max_sequence_length=max_seq) as session:
        for index, token_id in enumerate(prompt_tokens[:position]):
            session.step(int(token_id), position=index, return_logits=False)
        layer_capture = session.capture_attention_layer(
            int(prompt_tokens[position]),
            position=int(position),
            layer_id=int(layer_id),
            run_preceding_layers=True,
        )
        runtime = session.runtime
        runner = session.runner
        scratch = session.scratch
        if runtime is None or runner is None or scratch is None:
            raise RuntimeError("resident session did not expose runtime/scratch")
        cfg = runner.weights.config
        key_cache, value_cache = scratch.full_cache(int(layer_id))
        kv_width = int(runner.kv_width)
        cache_offset_bytes = int(position) * kv_width * 2
        return {
            "status": "captured",
            "summary": layer_capture.as_summary_dict()
            | {
                "q_width": int(runner.q_width),
                "kv_width": kv_width,
                "num_q_heads": int(cfg.head_count),
                "num_kv_heads": int(cfg.head_count_kv),
                "head_dim": int(cfg.key_length),
                "value_head_dim": int(cfg.value_length),
                "rotary_dim": int(cfg.rope_dimension_count),
                "rms_norm_eps": float(cfg.rms_norm_eps),
                "rope_freq_base": float(cfg.rope_freq_base),
                "block_size": int(scratch.block_size),
                "max_positions": int(scratch.max_positions),
                "cache_offset_elements": int(position) * kv_width,
            },
            "fields": {
                "full_q_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.full_q.ptr), int(2 * runner.q_width), runtime=runtime
                ),
                "full_k_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.full_k.ptr), kv_width, runtime=runtime
                ),
                "full_v_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.full_v.ptr), kv_width, runtime=runtime
                ),
                "full_query_raw_f32": _copy_f32_ptr_to_host(
                    int(scratch.full_query_raw.ptr), int(runner.q_width), runtime=runtime
                ),
                "full_gate_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.full_gate.ptr), int(runner.q_width), runtime=runtime
                ),
                "full_key_raw_f32": _copy_f32_ptr_to_host(
                    int(scratch.full_key_raw.ptr), kv_width, runtime=runtime
                ),
                "full_query_f32": _copy_f32_ptr_to_host(
                    int(scratch.full_query.ptr), int(runner.q_width), runtime=runtime
                ),
                "full_key_f32": _copy_f32_ptr_to_host(
                    int(scratch.full_key.ptr), kv_width, runtime=runtime
                ),
                "key_cache_position_f32": _copy_bf16_ptr_to_host_f32(
                    int(key_cache.ptr) + cache_offset_bytes, kv_width, runtime=runtime
                ),
                "value_cache_position_f32": _copy_bf16_ptr_to_host_f32(
                    int(value_cache.ptr) + cache_offset_bytes, kv_width, runtime=runtime
                ),
            },
        }


def unavailable_artifact(
    *,
    qkv_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    classification = "layer15_qk_norm_rotary_kv_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer15_qk_norm_rotary_kv_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": classification,
        "qkv_artifact_path": str(qkv_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def blocked_artifact(
    *,
    qkv_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    preflight: Mapping[str, Any],
    preflight_classification: str,
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "layer15_qk_norm_rotary_kv_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(preflight_classification),
        "classification": preflight_classification,
        "qkv_artifact_path": str(qkv_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "preflight": dict(preflight),
        "preflight_classification": str(preflight_classification),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": next_action(preflight_classification),
    }


if __name__ == "__main__":
    main()
