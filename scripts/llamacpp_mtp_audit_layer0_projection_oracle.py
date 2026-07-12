#!/usr/bin/env python3
"""Audit layer-0 linear-attention projections under the BF16-contracted oracle."""

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
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    bf16_roundtrip_array,
    delta_summary,
    load_capture_values,
    rmsnorm_f32,
    sha256_float32,
    summarize_array,
)

DEFAULT_INPUT_COMPARE = Path("benchmarks/results/mtp-gguf-iter318-input-embed-compare.json")
DEFAULT_POLICY = Path("benchmarks/results/mtp-gguf-iter322-layer0-dtype-oracle-policy.json")
DEFAULT_FORMULA_AUDIT = Path(
    "benchmarks/results/mtp-gguf-iter321-layer0-attn-norm-formula-audit.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter323-layer0-bf16-projection-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

ProjectionWeights = Mapping[str, tuple[np.ndarray, dict[str, Any]]]
BoundaryCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
ProjectionWeightLoader = Callable[[Path, int], ProjectionWeights]
NormWeightLoader = Callable[[Path, int], tuple[np.ndarray, float, dict[str, Any]]]

PROJECTION_SPECS = {
    "linear_qkv_f32": {
        "tensor_name": "attn_qkv.weight",
        "weight_slot": "attn_qkv",
        "hip_field": "linear_qkv_f32",
        "semantic_stage": "attn_qkv projection output",
    },
    "linear_z_f32": {
        "tensor_name": "attn_gate.weight",
        "weight_slot": "attn_gate",
        "hip_field": "linear_z_f32",
        "semantic_stage": "attn_gate projection output",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-compare", type=Path, default=DEFAULT_INPUT_COMPARE)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--formula-audit", type=Path, default=DEFAULT_FORMULA_AUDIT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--near-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=323)
    args = parser.parse_args()

    artifact = audit_layer0_projection_oracle(
        input_compare_path=args.input_compare,
        policy_path=args.policy,
        formula_audit_path=args.formula_audit,
        model_path=args.model,
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
                "attn_norm_exact": artifact["attn_norm_oracle"]["delta_vs_hip"][
                    "exact_match"
                ],
                "linear_qkv_classification": artifact["projection_results"][
                    "linear_qkv_f32"
                ]["classification"],
                "linear_z_classification": artifact["projection_results"][
                    "linear_z_f32"
                ]["classification"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer0_projection_oracle(
    *,
    input_compare_path: Path,
    policy_path: Path,
    formula_audit_path: Path,
    model_path: Path | None = None,
    max_sequence_length: int | None = None,
    near_atol: float = 2.5e-4,
    iteration: int = 323,
    boundary_capture_fn: BoundaryCaptureFn | None = None,
    projection_weight_loader: ProjectionWeightLoader | None = None,
    norm_weight_loader: NormWeightLoader | None = None,
) -> dict[str, Any]:
    input_compare = json.loads(input_compare_path.read_text())
    policy = json.loads(policy_path.read_text())
    formula_audit = json.loads(formula_audit_path.read_text())
    validate_policy(policy)
    resolved_model = Path(model_path or policy["model"])
    layer_id = int(policy["layer_id"])
    position = int(policy["position"])
    prompt_tokens = tuple(int(token) for token in formula_audit["prompt_tokens"])
    input_f32 = load_capture_values(input_compare["llamacpp_capture"])
    selected_norm_loader = norm_weight_loader or load_attn_norm_weight
    norm_weight, eps, norm_metadata = selected_norm_loader(resolved_model, layer_id)
    attn_norm_oracle = bf16_roundtrip_array(
        rmsnorm_f32(bf16_roundtrip_array(input_f32), norm_weight, eps)
    )
    selected_boundary_capture_fn = boundary_capture_fn or capture_hipengine_linear_boundary
    capture = selected_boundary_capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        layer_id,
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            input_compare_path=input_compare_path,
            policy_path=policy_path,
            formula_audit_path=formula_audit_path,
            model_path=resolved_model,
            layer_id=layer_id,
            position=position,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    hip_attn_norm = np.asarray(capture["fields"]["attn_norm_f32"], dtype=np.float32)
    attn_norm_delta = delta_summary(attn_norm_oracle, hip_attn_norm)
    selected_projection_loader = projection_weight_loader or load_projection_weights
    projection_weights = selected_projection_loader(resolved_model, layer_id)
    projection_results = build_projection_results(
        attn_norm_oracle=attn_norm_oracle,
        capture=capture,
        projection_weights=projection_weights,
        near_atol=near_atol,
    )
    classification = classify_projection_audit(attn_norm_delta, projection_results)
    return {
        "schema": 1,
        "kind": "layer0_bf16_contracted_projection_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "input_compare_path": str(input_compare_path),
        "policy_path": str(policy_path),
        "formula_audit_path": str(formula_audit_path),
        "model": str(resolved_model),
        "layer_id": layer_id,
        "position": position,
        "token_id": int(prompt_tokens[position]),
        "prompt_tokens": list(prompt_tokens),
        "policy": summarize_policy(policy),
        "attn_norm_oracle": {
            "source": "BF16(input_embed_f32) + attn_norm.weight_f32 + model_eps -> BF16 output",
            "summary": summarize_array(attn_norm_oracle),
            "delta_vs_hip": attn_norm_delta,
            "weight": norm_metadata,
            "eps": float(eps),
        },
        "projection_results": projection_results,
        "hipengine_capture": summarize_capture(capture),
        "near_atol": float(near_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("status") != "ready":
        raise ValueError("dtype oracle policy must be ready")
    decision = policy.get("decision") or {}
    if decision.get("selected_policy") != "bf16_contracted_llamacpp_or_cpu_oracle":
        raise ValueError("dtype oracle policy must select the BF16-contracted oracle")


def build_projection_results(
    *,
    attn_norm_oracle: np.ndarray,
    capture: Mapping[str, Any],
    projection_weights: ProjectionWeights,
    near_atol: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    fields = capture["fields"]
    for field, spec in PROJECTION_SPECS.items():
        weight, metadata = projection_weights[spec["weight_slot"]]
        f32 = project_f32(attn_norm_oracle, weight)
        bf16 = bf16_roundtrip_array(f32)
        hip = np.asarray(fields[spec["hip_field"]], dtype=np.float32)
        f32_delta = delta_summary(f32, hip)
        bf16_delta = delta_summary(bf16, hip)
        classification = classify_projection_delta(bf16_delta, near_atol=near_atol)
        results[field] = {
            "field": field,
            "weight_slot": spec["weight_slot"],
            "semantic_stage": spec["semantic_stage"],
            "oracle": "matmul(attn_norm_bf16, dequantized_gguf_weight) -> BF16 output",
            "weight": metadata,
            "f32_oracle_summary": summarize_array(f32),
            "bf16_oracle_summary": summarize_array(bf16),
            "hipengine_summary": summarize_array(hip),
            "delta_f32_oracle_vs_hip": f32_delta,
            "delta_bf16_oracle_vs_hip": bf16_delta,
            "classification": classification,
        }
    return results


def project_f32(input_values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x = np.asarray(input_values, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] != x.shape[0]:
        raise ValueError(f"projection shape mismatch: weight={w.shape}, input={x.shape}")
    return np.asarray(w @ x, dtype=np.float32)


def classify_projection_delta(delta: Mapping[str, Any], *, near_atol: float) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "projection_oracle_unavailable"
    if delta.get("exact_match"):
        return "projection_matches_bf16_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "projection_matches_bf16_oracle_within_one_bf16_step"
    return "projection_mismatch_after_bf16_contracted_oracle"


def classify_projection_audit(
    attn_norm_delta: Mapping[str, Any],
    projection_results: Mapping[str, Any],
) -> str:
    if not attn_norm_delta.get("exact_match"):
        return "projection_audit_blocked_attn_norm_oracle_mismatch"
    classes = [result["classification"] for result in projection_results.values()]
    if all(item == "projection_matches_bf16_oracle_exactly" for item in classes):
        return "layer0_projections_match_bf16_oracle_exactly"
    if all(item.startswith("projection_matches_bf16_oracle") for item in classes):
        return "layer0_projections_match_bf16_oracle_within_rounding"
    return "layer0_projection_mismatch_after_bf16_oracle"


def status_from_classification(classification: str) -> str:
    if classification.startswith("projection_audit_blocked"):
        return "blocked"
    if "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer0_projections_match_bf16_oracle_exactly",
        "layer0_projections_match_bf16_oracle_within_rounding",
    }:
        return "continue_layer0_bf16_bisection_at_conv_or_gdn_state"
    if classification == "projection_audit_blocked_attn_norm_oracle_mismatch":
        return "rerun_layer0_attn_norm_formula_audit"
    return "audit_layer0_projection_weight_or_kernel_quantization"


def load_attn_norm_weight(
    model_path: Path,
    layer_id: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    reader = GGUFReader(model_path)
    tensor_name = f"blk.{layer_id}.attn_norm.weight"
    weight = np.asarray(reader.dequantize_tensor(tensor_name), dtype=np.float32)
    eps_key = "qwen35moe.attention.layer_norm_rms_epsilon"
    eps = float(reader.info.metadata[eps_key])
    return weight, eps, {
        "tensor_name": tensor_name,
        "ggml_type": reader.tensor_info(tensor_name).ggml_type_name,
        "shape": list(weight.shape),
        "summary": summarize_array(weight),
        "metadata_eps_key": eps_key,
        "metadata_eps": eps,
    }


def load_projection_weights(model_path: Path, layer_id: int) -> ProjectionWeights:
    reader = GGUFReader(model_path)
    weights: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for slot, tensor_suffix in (
        ("attn_qkv", "attn_qkv.weight"),
        ("attn_gate", "attn_gate.weight"),
    ):
        tensor_name = f"blk.{layer_id}.{tensor_suffix}"
        tensor = reader.tensor_info(tensor_name)
        values = np.asarray(reader.dequantize_tensor(tensor_name), dtype=np.float32)
        weights[slot] = (
            values,
            {
                "tensor_name": tensor_name,
                "ggml_type": tensor.ggml_type_name,
                "shape": list(values.shape),
                "summary": summarize_array(values.reshape(-1)),
                "sha256": sha256_float32(values.reshape(-1)),
            },
        )
    return weights


def capture_hipengine_linear_boundary(
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
        capture = session.capture_linear_attention_boundary(
            int(prompt_tokens[position]),
            position=int(position),
            layer_id=int(layer_id),
        )
        return {
            "status": "captured",
            "summary": capture.as_summary_dict(),
            "fields": {
                "attn_norm_f32": np.asarray(capture.attn_norm_f32, dtype=np.float32),
                "linear_qkv_f32": np.asarray(capture.linear_qkv_f32, dtype=np.float32),
                "linear_z_f32": np.asarray(capture.linear_z_f32, dtype=np.float32),
                "ssm_alpha_f32": np.asarray(capture.ssm_alpha_f32, dtype=np.float32),
                "ssm_beta_f32": np.asarray(capture.ssm_beta_f32, dtype=np.float32),
                "conv_out_f32": np.asarray(capture.conv_out_f32, dtype=np.float32),
                "recurrent_out_f32": np.asarray(capture.recurrent_out_f32, dtype=np.float32),
                "recurrent_bf16_f32": np.asarray(capture.recurrent_bf16_f32, dtype=np.float32),
                "attn_out_f32": np.asarray(capture.attn_out_f32, dtype=np.float32),
            },
        }


def summarize_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    if capture.get("status") != "captured":
        return {"status": capture.get("status"), "reason": capture.get("reason")}
    return {
        "status": "captured",
        "summary": capture.get("summary"),
        "field_summaries": {
            name: summarize_array(np.asarray(values, dtype=np.float32))
            for name, values in capture.get("fields", {}).items()
        },
    }


def summarize_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selected_policy": (policy.get("decision") or {}).get("selected_policy"),
        "scope": (policy.get("decision") or {}).get("scope"),
        "next_action": policy.get("next_action"),
        "first_probe": (policy.get("next_probe_plan") or {})
        .get("ordered_probes", [{}])[0]
        .get("field"),
    }


def unavailable_artifact(
    *,
    input_compare_path: Path,
    policy_path: Path,
    formula_audit_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "layer0_bf16_contracted_projection_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": "layer0_projection_oracle_unavailable",
        "input_compare_path": str(input_compare_path),
        "policy_path": str(policy_path),
        "formula_audit_path": str(formula_audit_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(prompt_tokens[position]),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": "rerun_layer0_projection_oracle_on_rocm_host",
    }


if __name__ == "__main__":
    main()
