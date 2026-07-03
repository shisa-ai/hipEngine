#!/usr/bin/env python3
"""Audit layer-3 full-attention output projection boundary."""

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

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    Qwen35GGUFResidentSession,
    _copy_bf16_ptr_to_host_f32,
)
from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    bf16_roundtrip_array,
    delta_summary,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    project_f32,
    summarize_capture,
)
from scripts.llamacpp_mtp_audit_layer3_full_attention_context_gate_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_CONTEXT_GATE,
)

DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter350-layer3-attn-output-oracle.json")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_NEAR_ATOL = 2.5e-4

BoundaryCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
WeightLoader = Callable[[Path, int], tuple[np.ndarray, dict[str, Any]]]

PREFLIGHT_FIELDS = ("full_gated_f32",)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-gate-artifact", type=Path, default=DEFAULT_CONTEXT_GATE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--near-atol", type=float, default=DEFAULT_NEAR_ATOL)
    parser.add_argument("--iteration", type=int, default=350)
    args = parser.parse_args()

    artifact = audit_layer3_attn_output_oracle(
        context_gate_artifact_path=args.context_gate_artifact,
        model_path=args.model,
        layer_id=args.layer_id,
        max_sequence_length=args.max_sequence_length,
        near_atol=args.near_atol,
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



def audit_layer3_attn_output_oracle(
    *,
    context_gate_artifact_path: Path,
    model_path: Path | None = None,
    layer_id: int = 3,
    max_sequence_length: int | None = None,
    near_atol: float = DEFAULT_NEAR_ATOL,
    iteration: int = 350,
    boundary_capture_fn: BoundaryCaptureFn | None = None,
    weight_loader: WeightLoader | None = None,
) -> dict[str, Any]:
    source_artifact = json.loads(context_gate_artifact_path.read_text())
    validate_context_gate_artifact(source_artifact, expected_layer_id=int(layer_id))
    resolved_model = Path(model_path or source_artifact["model"])
    prompt_tokens = tuple(int(token) for token in source_artifact["prompt_tokens"])
    position = int(source_artifact["position"])
    token_id = int(prompt_tokens[position])
    capture_fn = boundary_capture_fn or capture_layer3_attn_output_boundary
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(layer_id),
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            source_path=context_gate_artifact_path,
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
    if preflight_classification != "layer3_attn_output_preflight_matches_context_gate":
        return blocked_artifact(
            source_path=context_gate_artifact_path,
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
    loader = weight_loader or load_attn_output_weight
    weight, weight_metadata = loader(resolved_model, int(layer_id))
    projection_result = build_attn_output_result(
        capture=capture,
        weight=weight,
        weight_metadata=weight_metadata,
        near_atol=float(near_atol),
    )
    classification = classify_attn_output(
        preflight_classification,
        projection_result,
    )
    return {
        "schema": 1,
        "kind": "layer3_attn_output_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "context_gate_artifact_path": str(context_gate_artifact_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "validated layer3 full-attention context/gate artifact",
            "context_gate_classification": source_artifact.get("classification"),
            "formula": "BF16(project_f32(full_gated_f32, GGUF attn_output.weight))",
            "input_contract": "full_gated_f32 is BF16(full_attn_context * sigmoid(gate))",
            "output_contract": "attn_out_f32 is BF16 output projection input to post-attn residual",
        },
        "preflight": preflight,
        "preflight_classification": preflight_classification,
        "attn_output_weight": weight_metadata,
        "hipengine_capture": summarize_capture(capture),
        "projection_result": projection_result,
        "near_atol": float(near_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }



def validate_context_gate_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if artifact.get("status") != "ready":
        raise ValueError("layer3 context/gate artifact must be ready")
    if artifact.get("classification") not in {
        "layer3_full_attention_context_gate_matches_cpu_oracle_exactly",
        "layer3_full_attention_context_matches_cpu_oracle_within_fp32_tolerance",
    }:
        raise ValueError("layer3 context/gate artifact must be validated")
    if int(artifact.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer3 context/gate artifact layer_id mismatch")
    if artifact.get("next_action") != "audit_layer3_attn_output_projection_under_bf16_contract":
        raise ValueError("source artifact must point to layer3 attn-output audit")
    gate_result = artifact.get("gate_result") or {}
    if gate_result.get("classification") != "full_attention_context_gate_matches_exactly":
        raise ValueError("source artifact must establish exact full_gated input")
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
        return "layer3_attn_output_wrong_layer_capture"
    if str(summary.get("layer_type")) != "full_attention":
        return "layer3_attn_output_wrong_layer_type"
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        return "layer3_attn_output_wrong_preceding_layer_count"
    if not all(item.get("exact_hash_match") for item in preflight.values()):
        return "layer3_attn_output_blocked_context_gate_input_mismatch"
    return "layer3_attn_output_preflight_matches_context_gate"



def build_attn_output_result(
    *,
    capture: Mapping[str, Any],
    weight: np.ndarray,
    weight_metadata: Mapping[str, Any],
    near_atol: float,
) -> dict[str, Any]:
    fields = capture["fields"]
    input_values = np.asarray(fields["full_gated_f32"], dtype=np.float32)
    expected_f32 = project_f32(input_values, np.asarray(weight, dtype=np.float32))
    expected_bf16 = bf16_roundtrip_array(expected_f32)
    actual = np.asarray(fields["attn_out_f32"], dtype=np.float32)
    f32_delta = delta_summary(expected_f32, actual)
    bf16_delta = delta_summary(expected_bf16, actual)
    return {
        "field": "attn_out_f32",
        "oracle": "matmul(full_gated_bf16, dequantized GGUF attn_output) -> BF16",
        "weight": dict(weight_metadata),
        "f32_oracle_summary": summarize_array(expected_f32),
        "bf16_oracle_summary": summarize_array(expected_bf16),
        "hipengine_summary": summarize_array(actual),
        "delta_f32_oracle_vs_hip": f32_delta,
        "delta_bf16_oracle_vs_hip": bf16_delta,
        "classification": classify_attn_output_delta(bf16_delta, near_atol=near_atol),
    }



def classify_attn_output_delta(delta: Mapping[str, Any], *, near_atol: float) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "full_attention_attn_output_oracle_unavailable"
    if delta.get("exact_match"):
        return "full_attention_attn_output_matches_bf16_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "full_attention_attn_output_matches_bf16_oracle_within_rounding"
    return "full_attention_attn_output_mismatch_after_bf16_oracle"



def classify_attn_output(
    preflight_classification: str,
    projection_result: Mapping[str, Any],
) -> str:
    if preflight_classification != "layer3_attn_output_preflight_matches_context_gate":
        return preflight_classification
    projection_class = str(projection_result.get("classification"))
    if projection_class == "full_attention_attn_output_matches_bf16_oracle_exactly":
        return "layer3_attn_output_projection_matches_bf16_oracle_exactly"
    if projection_class == "full_attention_attn_output_matches_bf16_oracle_within_rounding":
        return "layer3_attn_output_projection_matches_bf16_oracle_within_rounding"
    if "mismatch" in projection_class:
        return "layer3_attn_output_projection_mismatch_after_bf16_oracle"
    return "layer3_attn_output_projection_oracle_unavailable"



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
        "layer3_attn_output_projection_matches_bf16_oracle_exactly",
        "layer3_attn_output_projection_matches_bf16_oracle_within_rounding",
    }:
        return "audit_layer3_post_attn_residual_or_moe_boundary"
    if classification == "layer3_attn_output_blocked_context_gate_input_mismatch":
        return "reconcile_layer3_context_gate_before_attn_output"
    if classification in {
        "layer3_attn_output_wrong_layer_capture",
        "layer3_attn_output_wrong_layer_type",
        "layer3_attn_output_wrong_preceding_layer_count",
    }:
        return "inspect_layer3_attn_output_capture_metadata"
    if classification == "layer3_attn_output_projection_mismatch_after_bf16_oracle":
        return "inspect_layer3_attn_output_weight_or_projection_kernel"
    return "rerun_layer3_attn_output_oracle_on_rocm_host"



def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if "projection_result" not in artifact:
        return {}
    return {"attn_out_f32": artifact["projection_result"].get("classification")}



def load_attn_output_weight(
    model_path: Path,
    layer_id: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    reader = GGUFReader(model_path)
    tensor_name = f"blk.{layer_id}.attn_output.weight"
    tensor = reader.tensor_info(tensor_name)
    values = np.asarray(reader.dequantize_tensor(tensor_name), dtype=np.float32)
    return values, {
        "tensor_name": tensor_name,
        "ggml_type": tensor.ggml_type_name,
        "shape": list(values.shape),
        "summary": summarize_array(values.reshape(-1)),
        "sha256": sha256_float32(values.reshape(-1)),
    }



def capture_layer3_attn_output_boundary(
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
        return {
            "status": "captured",
            "summary": layer_capture.as_summary_dict()
            | {
                "q_width": int(runner.q_width),
                "attn_output_shape": [int(runner.hidden_size), int(runner.q_width)],
            },
            "fields": {
                "full_gated_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.full_gated.ptr), int(runner.q_width), runtime=runtime
                ),
                "attn_out_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.attn_out.ptr), int(runner.hidden_size), runtime=runtime
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
    classification = "layer3_attn_output_projection_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer3_attn_output_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": classification,
        "context_gate_artifact_path": str(source_path),
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
        "kind": "layer3_attn_output_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "context_gate_artifact_path": str(source_path),
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
