#!/usr/bin/env python3
"""Audit layer-7 full-attention context and gate-multiply boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    Qwen35GGUFResidentSession,
    _copy_bf16_ptr_to_host_f32,
    _copy_f32_ptr_to_host,
    _use_gguf_full_attention_split_decode,
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
from scripts.llamacpp_mtp_audit_layer7_qk_norm_rotary_kv_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER7_QK_NORM_ROTARY_KV,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter380-layer7-full-attention-context-gate-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_CONTEXT_NEAR_ATOL = 1.0e-5

BoundaryCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]

PREFLIGHT_FIELDS = (
    "full_query_f32",
    "full_key_f32",
    "full_gate_f32",
    "key_cache_position_f32",
    "value_cache_position_f32",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qk-norm-rotary-kv-artifact", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=7)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument(
        "--context-near-atol",
        type=float,
        default=DEFAULT_CONTEXT_NEAR_ATOL,
    )
    parser.add_argument("--iteration", type=int, default=380)
    args = parser.parse_args()

    source = args.qk_norm_rotary_kv_artifact or DEFAULT_LAYER7_QK_NORM_ROTARY_KV
    artifact = audit_layer7_full_attention_context_gate_oracle(
        qk_norm_rotary_kv_artifact_path=source,
        model_path=args.model,
        layer_id=args.layer_id,
        max_sequence_length=args.max_sequence_length,
        context_near_atol=args.context_near_atol,
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


def audit_layer7_full_attention_context_gate_oracle(
    *,
    qk_norm_rotary_kv_artifact_path: Path,
    model_path: Path | None = None,
    layer_id: int = 7,
    max_sequence_length: int | None = None,
    context_near_atol: float = DEFAULT_CONTEXT_NEAR_ATOL,
    iteration: int = 380,
    boundary_capture_fn: BoundaryCaptureFn | None = None,
) -> dict[str, Any]:
    source_artifact = json.loads(qk_norm_rotary_kv_artifact_path.read_text())
    validate_qk_norm_rotary_kv_artifact(
        source_artifact,
        expected_layer_id=int(layer_id),
    )
    resolved_model = Path(model_path or source_artifact["model"])
    prompt_tokens = tuple(int(token) for token in source_artifact["prompt_tokens"])
    position = int(source_artifact["position"])
    token_id = int(prompt_tokens[position])
    capture_fn = boundary_capture_fn or capture_layer7_full_attention_context_gate_boundary
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(layer_id),
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            source_path=qk_norm_rotary_kv_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    preflight = compare_preflight(capture=capture, artifact=source_artifact)
    preflight_classification = classify_preflight(
        preflight,
        capture=capture,
        layer_id=int(layer_id),
    )
    if preflight_classification != "layer7_context_gate_preflight_matches_qk_artifact":
        return blocked_artifact(
            source_path=qk_norm_rotary_kv_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            preflight=preflight,
            classification=preflight_classification,
            iteration=iteration,
        )
    context_result = build_context_result(
        capture=capture,
        near_atol=float(context_near_atol),
    )
    gate_result = build_gate_result(capture=capture)
    classification = classify_context_gate(
        preflight_classification,
        context_result,
        gate_result,
    )
    return {
        "schema": 1,
        "kind": "layer7_full_attention_context_gate_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "qk_norm_rotary_kv_artifact_path": str(qk_norm_rotary_kv_artifact_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "layer7 validated Q/K norm+RoPE and KV cache artifact",
            "context_formula": (
                "paged GQA attention over KVLiveSpans live_count=position+1, "
                "BF16 K/V caches, FP32 query, scale=head_dim^-0.5"
            ),
            "gate_formula": "BF16(full_attn_context_f32 * sigmoid(full_gate_bf16))",
            "exact_required_for": ["gate_multiply"],
            "fp32_tolerance_only_for": "attention_context_softmax",
        },
        "preflight": preflight,
        "preflight_classification": preflight_classification,
        "hipengine_capture": summarize_capture(capture),
        "context_result": context_result,
        "gate_result": gate_result,
        "context_near_atol": float(context_near_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_qk_norm_rotary_kv_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if artifact.get("status") != "ready":
        raise ValueError("layer7 QK norm/rotary/KV artifact must be ready")
    if artifact.get("classification") not in {
        "layer7_qk_norm_rotary_kv_matches_cpu_oracle_exactly",
        "layer7_qk_norm_rotary_matches_cpu_oracle_within_fp32_tolerance",
    }:
        raise ValueError("layer7 QK norm/rotary/KV artifact must be validated")
    if int(artifact.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer7 QK norm/rotary/KV artifact layer_id mismatch")
    if artifact.get("next_action") != (
        "audit_layer7_full_attention_scores_or_attn_output_under_bf16_contract"
    ):
        raise ValueError("source artifact must point to full-attention context audit")
    fields = (artifact.get("hipengine_capture") or {}).get("field_summaries") or {}
    missing = [field for field in PREFLIGHT_FIELDS if field not in fields]
    if missing:
        raise ValueError(f"source artifact missing field summaries: {missing}")


def compare_preflight(
    *,
    capture: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    source_fields = artifact["hipengine_capture"]["field_summaries"]
    fields = capture["fields"]
    result: dict[str, Any] = {}
    for field in PREFLIGHT_FIELDS:
        values = np.asarray(fields[field], dtype=np.float32)
        actual_sha = sha256_float32(values)
        expected_sha = str(source_fields[field]["sha256"])
        result[field] = {
            "field": field,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "exact_hash_match": actual_sha == expected_sha,
            "summary": summarize_array(values),
        }
    return result


def classify_preflight(
    preflight: Mapping[str, Any],
    *,
    capture: Mapping[str, Any],
    layer_id: int,
) -> str:
    summary = capture.get("summary") or {}
    if int(summary.get("layer_id", -1)) != int(layer_id):
        return "layer7_context_gate_wrong_layer_capture"
    if str(summary.get("layer_type")) != "full_attention":
        return "layer7_context_gate_wrong_layer_type"
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        return "layer7_context_gate_wrong_preceding_layer_count"
    if bool(summary.get("used_split_decode")):
        return "layer7_context_gate_blocked_split_decode_path"
    if not all(item.get("exact_hash_match") for item in preflight.values()):
        return "layer7_context_gate_blocked_qk_input_mismatch"
    return "layer7_context_gate_preflight_matches_qk_artifact"


def build_context_result(
    *,
    capture: Mapping[str, Any],
    near_atol: float,
) -> dict[str, Any]:
    fields = capture["fields"]
    dims = capture_dimensions(capture)
    context = full_attention_context_cpu(
        np.asarray(fields["full_query_f32"], dtype=np.float32),
        np.asarray(fields["key_cache_context_f32"], dtype=np.float32),
        np.asarray(fields["value_cache_context_f32"], dtype=np.float32),
        context_len=dims["active_context"],
        num_q_heads=dims["num_q_heads"],
        num_kv_heads=dims["num_kv_heads"],
        head_dim=dims["head_dim"],
        scale=np.float32(dims["head_dim"] ** -0.5),
    )
    actual = np.asarray(fields["full_attn_context_f32"], dtype=np.float32)
    delta = delta_summary(context, actual)
    return {
        "field": "full_attn_context_f32",
        "oracle": "CPU paged GQA softmax context over BF16 K/V cache",
        "expected_summary": summarize_array(context),
        "hipengine_summary": summarize_array(actual),
        "delta": delta,
        "classification": classify_near_delta(delta, near_atol=near_atol),
    }


def build_gate_result(*, capture: Mapping[str, Any]) -> dict[str, Any]:
    fields = capture["fields"]
    context = np.asarray(fields["full_attn_context_f32"], dtype=np.float32)
    gate = np.asarray(fields["full_gate_f32"], dtype=np.float32)
    expected = bf16_roundtrip_array(
        np.asarray(context * sigmoid_f32(gate), dtype=np.float32)
    )
    actual = np.asarray(fields["full_gated_f32"], dtype=np.float32)
    delta = delta_summary(expected, actual)
    return {
        "field": "full_gated_f32",
        "oracle": "BF16(full_attn_context_f32 * sigmoid(full_gate_bf16))",
        "expected_summary": summarize_array(expected),
        "hipengine_summary": summarize_array(actual),
        "delta": delta,
        "classification": classify_exact_delta(delta),
    }


def full_attention_context_cpu(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    *,
    context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: np.float32,
    warp_size: int = 64,
) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(int(num_q_heads), int(head_dim))
    k = np.asarray(key_cache, dtype=np.float32).reshape(
        int(context_len), int(num_kv_heads), int(head_dim)
    )
    v = np.asarray(value_cache, dtype=np.float32).reshape(
        int(context_len), int(num_kv_heads), int(head_dim)
    )
    out = np.empty_like(q, dtype=np.float32)
    kv_group = int(num_q_heads) // int(num_kv_heads)
    for q_head in range(int(num_q_heads)):
        kv_head = q_head // kv_group
        scores = np.empty((int(context_len),), dtype=np.float32)
        for token in range(int(context_len)):
            dot = warp_style_dot(q[q_head], k[token, kv_head], warp_size=int(warp_size))
            scores[token] = np.float32(dot * np.float32(scale))
        probs = softmax_f32(scores)
        for dim in range(int(head_dim)):
            acc = np.float32(0.0)
            for token in range(int(context_len)):
                acc = np.float32(acc + np.float32(probs[token] * v[token, kv_head, dim]))
            out[q_head, dim] = acc
    return np.ascontiguousarray(out.reshape(-1))


def warp_style_dot(a: np.ndarray, b: np.ndarray, *, warp_size: int = 64) -> np.float32:
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    partial = np.zeros((int(warp_size),), dtype=np.float32)
    for lane in range(int(warp_size)):
        acc = np.float32(0.0)
        for dim in range(lane * 4, int(left.size), int(warp_size) * 4):
            for offset in range(4):
                idx = dim + offset
                if idx < int(left.size):
                    acc = np.float32(acc + np.float32(left[idx] * right[idx]))
        partial[lane] = acc
    step = int(warp_size) // 2
    while step > 0:
        for lane in range(step):
            partial[lane] = np.float32(partial[lane] + partial[lane + step])
        step //= 2
    return np.float32(partial[0])


def softmax_f32(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    max_score = np.max(values).astype(np.float32)
    exp_values = np.exp(np.asarray(values - max_score, dtype=np.float32), dtype=np.float32)
    denom = np.float32(max(float(np.sum(exp_values, dtype=np.float32)), 1.0e-20))
    return np.asarray(exp_values / denom, dtype=np.float32)


def sigmoid_f32(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    return np.asarray(1.0 / (1.0 + np.exp(-vals, dtype=np.float32)), dtype=np.float32)


def classify_exact_delta(delta: Mapping[str, Any]) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "full_attention_context_gate_oracle_unavailable"
    if delta.get("exact_match"):
        return "full_attention_context_gate_matches_exactly"
    return "full_attention_context_gate_mismatch"


def classify_near_delta(delta: Mapping[str, Any], *, near_atol: float) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "full_attention_context_gate_oracle_unavailable"
    if delta.get("exact_match"):
        return "full_attention_context_gate_matches_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "full_attention_context_matches_within_fp32_kernel_tolerance"
    return "full_attention_context_gate_mismatch"


def classify_context_gate(
    preflight_classification: str,
    context_result: Mapping[str, Any],
    gate_result: Mapping[str, Any],
) -> str:
    if preflight_classification != "layer7_context_gate_preflight_matches_qk_artifact":
        return preflight_classification
    context_class = str(context_result.get("classification"))
    gate_class = str(gate_result.get("classification"))
    exact = "full_attention_context_gate_matches_exactly"
    near = "full_attention_context_matches_within_fp32_kernel_tolerance"
    if "mismatch" in context_class or "mismatch" in gate_class:
        return "layer7_full_attention_context_gate_mismatch"
    if context_class.endswith("unavailable") or gate_class.endswith("unavailable"):
        return "layer7_full_attention_context_gate_oracle_unavailable"
    if context_class == exact and gate_class == exact:
        return "layer7_full_attention_context_gate_matches_cpu_oracle_exactly"
    if context_class in {exact, near} and gate_class == exact:
        return "layer7_full_attention_context_matches_cpu_oracle_within_fp32_tolerance"
    return "layer7_full_attention_context_gate_mismatch"


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
        "layer7_full_attention_context_gate_matches_cpu_oracle_exactly",
        "layer7_full_attention_context_matches_cpu_oracle_within_fp32_tolerance",
    }:
        return "audit_layer7_attn_output_projection_under_bf16_contract"
    if classification == "layer7_context_gate_blocked_qk_input_mismatch":
        return "reconcile_layer7_qk_norm_rotary_kv_before_attention_context"
    if "wrong" in classification or "split_decode" in classification:
        return "inspect_layer7_full_attention_context_gate_capture_metadata"
    if classification == "layer7_full_attention_context_gate_mismatch":
        return "inspect_layer7_attention_context_or_gate_kernel"
    return "rerun_layer7_full_attention_context_gate_oracle_on_rocm_host"


def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    if "context_result" in artifact:
        out["full_attn_context_f32"] = artifact["context_result"].get("classification")
    if "gate_result" in artifact:
        out["full_gated_f32"] = artifact["gate_result"].get("classification")
    return out


def capture_dimensions(capture: Mapping[str, Any]) -> dict[str, int]:
    summary = capture.get("summary") or {}
    return {
        "active_context": int(summary["active_context"]),
        "num_q_heads": int(summary["num_q_heads"]),
        "num_kv_heads": int(summary["num_kv_heads"]),
        "head_dim": int(summary["head_dim"]),
        "q_width": int(summary["q_width"]),
        "kv_width": int(summary["kv_width"]),
    }


def capture_layer7_full_attention_context_gate_boundary(
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
        active_context = int(position) + 1
        kv_width = int(runner.kv_width)
        context_elements = active_context * kv_width
        used_split_decode = _use_gguf_full_attention_split_decode(active_context)
        return {
            "status": "captured",
            "summary": layer_capture.as_summary_dict()
            | {
                "q_width": int(runner.q_width),
                "kv_width": kv_width,
                "num_q_heads": int(cfg.head_count),
                "num_kv_heads": int(cfg.head_count_kv),
                "head_dim": int(cfg.key_length),
                "active_context": active_context,
                "block_size": int(scratch.block_size),
                "used_split_decode": bool(used_split_decode),
                "scale": float(cfg.key_length ** -0.5),
            },
            "fields": {
                "full_query_f32": _copy_f32_ptr_to_host(
                    int(scratch.full_query.ptr),
                    int(runner.q_width),
                    runtime=runtime,
                ),
                "full_key_f32": _copy_f32_ptr_to_host(
                    int(scratch.full_key.ptr),
                    kv_width,
                    runtime=runtime,
                ),
                "full_gate_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.full_gate.ptr),
                    int(runner.q_width),
                    runtime=runtime,
                ),
                "key_cache_position_f32": _copy_bf16_ptr_to_host_f32(
                    int(key_cache.ptr) + int(position) * kv_width * 2,
                    kv_width,
                    runtime=runtime,
                ),
                "value_cache_position_f32": _copy_bf16_ptr_to_host_f32(
                    int(value_cache.ptr) + int(position) * kv_width * 2,
                    kv_width,
                    runtime=runtime,
                ),
                "key_cache_context_f32": _copy_bf16_ptr_to_host_f32(
                    int(key_cache.ptr), context_elements, runtime=runtime
                ),
                "value_cache_context_f32": _copy_bf16_ptr_to_host_f32(
                    int(value_cache.ptr), context_elements, runtime=runtime
                ),
                "full_attn_context_f32": _copy_f32_ptr_to_host(
                    int(scratch.full_attn_context.ptr),
                    int(runner.q_width),
                    runtime=runtime,
                ),
                "full_gated_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.full_gated.ptr),
                    int(runner.q_width),
                    runtime=runtime,
                ),
                "attn_out_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.attn_out.ptr),
                    int(runner.hidden_size),
                    runtime=runtime,
                ),
            },
        }


def unavailable_artifact(
    *,
    source_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    classification = "layer7_full_attention_context_gate_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer7_full_attention_context_gate_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": classification,
        "qk_norm_rotary_kv_artifact_path": str(source_path),
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
    source_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    preflight: Mapping[str, Any],
    classification: str,
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "layer7_full_attention_context_gate_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "qk_norm_rotary_kv_artifact_path": str(source_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "preflight": dict(preflight),
        "preflight_classification": str(classification),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


if __name__ == "__main__":
    main()
