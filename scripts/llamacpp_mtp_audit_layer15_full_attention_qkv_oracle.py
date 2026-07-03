#!/usr/bin/env python3
"""Audit layer-15 full-attention Q/K/V projections under the BF16 contract."""

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
from scripts.llamacpp_mtp_audit_layer1_projection_oracle import (  # noqa: E402
    bf16_step_summary,
)
from scripts.llamacpp_mtp_audit_layer15_attn_norm_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER15_ATTN_NORM,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter438-layer15-full-attention-qkv-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

BoundaryCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]
ProjectionWeightLoader = Callable[
    [Path, int],
    Mapping[str, tuple[np.ndarray, dict[str, Any]]],
]

QKV_SPECS = {
    "full_q_f32": {
        "weight_slot": "attn_q",
        "tensor_suffix": "attn_q.weight",
        "hip_field": "full_q_f32",
        "semantic_stage": "full-attention Q+gate BF16 projection output",
    },
    "full_k_f32": {
        "weight_slot": "attn_k",
        "tensor_suffix": "attn_k.weight",
        "hip_field": "full_k_f32",
        "semantic_stage": "full-attention K BF16 projection output before key RMSNorm/rotary",
    },
    "full_v_f32": {
        "weight_slot": "attn_v",
        "tensor_suffix": "attn_v.weight",
        "hip_field": "full_v_f32",
        "semantic_stage": "full-attention V BF16 projection output before KV write",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attn-norm-artifact",
        type=Path,
        default=DEFAULT_LAYER15_ATTN_NORM,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=15)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--near-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=438)
    args = parser.parse_args()

    artifact = audit_layer15_full_attention_qkv_oracle(
        attn_norm_artifact_path=args.attn_norm_artifact,
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
                "input_classification": artifact.get("input_classification"),
                "field_classifications": field_classifications(artifact),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer15_full_attention_qkv_oracle(
    *,
    attn_norm_artifact_path: Path,
    model_path: Path | None = None,
    layer_id: int = 15,
    max_sequence_length: int | None = None,
    near_atol: float = 2.5e-4,
    iteration: int = 438,
    boundary_capture_fn: BoundaryCaptureFn | None = None,
    projection_weight_loader: ProjectionWeightLoader | None = None,
) -> dict[str, Any]:
    attn_norm_artifact = json.loads(attn_norm_artifact_path.read_text())
    validate_layer15_attn_norm_artifact(
        attn_norm_artifact,
        expected_layer_id=int(layer_id),
    )
    resolved_model = Path(model_path or attn_norm_artifact["model"])
    prompt_tokens = tuple(int(token) for token in attn_norm_artifact["prompt_tokens"])
    position = int(attn_norm_artifact["position"])
    token_id = int(prompt_tokens[position])
    capture_fn = boundary_capture_fn or capture_layer15_full_attention_qkv_boundary
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(layer_id),
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            attn_norm_artifact_path=attn_norm_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    input_result = compare_attn_norm_input(capture=capture, artifact=attn_norm_artifact)
    input_classification = classify_input(input_result)
    preflight = classify_qkv_preflight(
        input_classification,
        capture=capture,
        layer_id=int(layer_id),
    )
    if preflight is not None:
        return blocked_artifact(
            attn_norm_artifact_path=attn_norm_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            input_result=input_result,
            input_classification=input_classification,
            classification=preflight,
            iteration=iteration,
        )
    loader = projection_weight_loader or load_full_attention_qkv_weights
    weights = loader(resolved_model, int(layer_id))
    projection_results = build_qkv_results(
        attn_norm_values=np.asarray(
            capture["fields"]["attn_norm_f32"],
            dtype=np.float32,
        ),
        capture=capture,
        projection_weights=weights,
        near_atol=float(near_atol),
    )
    classification = classify_layer15_qkv(
        input_classification,
        projection_results,
        capture=capture,
        layer_id=int(layer_id),
    )
    return {
        "schema": 1,
        "kind": "layer15_full_attention_qkv_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "attn_norm_artifact_path": str(attn_norm_artifact_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "layer15 attn_norm_f32 from exact resident-BF16 RMSNorm oracle",
            "attn_norm_classification": attn_norm_artifact.get("classification"),
            "layer_type": "full_attention",
            "formula": "BF16(project_f32(attn_norm_f32, GGUF attn_q/k/v weight))",
            "q_output": "attn_q produces concatenated query+gate with 2*q_width rows",
            "k_output": "attn_k produces raw BF16 K before head RMSNorm/rotary",
            "v_output": "attn_v produces BF16 V before paged KV write",
        },
        "input_result": input_result,
        "input_classification": input_classification,
        "weights": {name: metadata for name, (_value, metadata) in weights.items()},
        "hipengine_capture": summarize_capture(capture),
        "projection_results": projection_results,
        "near_atol": float(near_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer15_attn_norm_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if artifact.get("status") != "ready":
        raise ValueError("layer15 attn_norm artifact must be ready")
    if artifact.get("classification") != (
        "layer15_attn_norm_matches_bf16_oracle_exactly"
    ):
        raise ValueError("layer15 attn_norm artifact must be exact")
    if int(artifact.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer15 attn_norm layer_id does not match requested layer")
    summary = (artifact.get("hipengine_capture") or {}).get("summary") or {}
    if str(summary.get("layer_type")) != "full_attention":
        raise ValueError("layer15 attn_norm artifact must be a full_attention capture")
    delta = artifact.get("attn_norm_delta") or {}
    if (
        delta.get("exact_match") is not True
        or float(delta.get("max_abs_diff", 1.0)) != 0.0
    ):
        raise ValueError("layer15 attn_norm delta must be exact")
    if artifact.get("next_action") != (
        "audit_layer15_attention_projection_under_bf16_contract"
    ):
        raise ValueError("layer15 attn_norm artifact must point to QKV audit")


def compare_attn_norm_input(
    *,
    capture: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    attn_norm = np.asarray(capture["fields"]["attn_norm_f32"], dtype=np.float32)
    actual_sha = sha256_float32(attn_norm)
    expected_sha = str(
        artifact["hipengine_capture"]["fields"]["attn_norm_f32"]["sha256"]
    )
    exact = actual_sha == expected_sha
    return {
        "field": "attn_norm_f32",
        "reference_source": "layer15_attn_norm_oracle",
        "reference_classification": artifact.get("classification"),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "exact_hash_match": exact,
        "summary": summarize_array(attn_norm),
        "classification": (
            "layer15_qkv_input_matches_attn_norm_artifact"
            if exact
            else "layer15_qkv_input_mismatch_before_projection"
        ),
    }


def classify_input(input_result: Mapping[str, Any]) -> str:
    if input_result.get("exact_hash_match"):
        return "layer15_qkv_input_matches_attn_norm_artifact"
    return "layer15_qkv_input_mismatch_before_projection"


def build_qkv_results(
    *,
    attn_norm_values: np.ndarray,
    capture: Mapping[str, Any],
    projection_weights: Mapping[str, tuple[np.ndarray, dict[str, Any]]],
    near_atol: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    fields = capture["fields"]
    for field, spec in QKV_SPECS.items():
        weight, metadata = projection_weights[spec["weight_slot"]]
        f32 = project_f32(attn_norm_values, weight)
        bf16 = bf16_roundtrip_array(f32)
        hip = np.asarray(fields[spec["hip_field"]], dtype=np.float32)
        f32_delta = delta_summary(f32, hip)
        bf16_delta = delta_summary(bf16, hip)
        bf16_step = bf16_step_summary(bf16, hip)
        classification = classify_qkv_delta(
            bf16_delta,
            near_atol=near_atol,
            bf16_step=bf16_step,
        )
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
            "bf16_step_oracle_vs_hip": bf16_step,
            "classification": classification,
        }
    return results


def classify_qkv_delta(
    delta: Mapping[str, Any],
    *,
    near_atol: float,
    bf16_step: Mapping[str, Any] | None = None,
) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "full_attention_qkv_oracle_unavailable"
    if delta.get("exact_match"):
        return "full_attention_qkv_matches_bf16_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "full_attention_qkv_matches_bf16_oracle_within_one_bf16_step"
    if bf16_step is not None and bf16_step.get("within_one_bf16_step"):
        return "full_attention_qkv_matches_bf16_oracle_within_one_bf16_step"
    return "full_attention_qkv_mismatch_after_bf16_oracle"


def classify_qkv_preflight(
    input_classification: str,
    *,
    capture: Mapping[str, Any],
    layer_id: int,
) -> str | None:
    summary = capture.get("summary") or {}
    if input_classification != "layer15_qkv_input_matches_attn_norm_artifact":
        return "layer15_full_attention_qkv_blocked_attn_norm_input_mismatch"
    if int(summary.get("layer_id", -1)) != int(layer_id):
        return "layer15_full_attention_qkv_wrong_layer_capture"
    if str(summary.get("layer_type")) != "full_attention":
        return "layer15_full_attention_qkv_wrong_layer_type"
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        return "layer15_full_attention_qkv_wrong_preceding_layer_count"
    return None


def classify_layer15_qkv(
    input_classification: str,
    projection_results: Mapping[str, Any],
    *,
    capture: Mapping[str, Any],
    layer_id: int,
) -> str:
    preflight = classify_qkv_preflight(
        input_classification,
        capture=capture,
        layer_id=layer_id,
    )
    if preflight is not None:
        return preflight
    classes = [projection_results[name]["classification"] for name in QKV_SPECS]
    if all(
        item == "full_attention_qkv_matches_bf16_oracle_exactly"
        for item in classes
    ):
        return "layer15_full_attention_qkv_matches_bf16_oracle_exactly"
    if all(
        item.startswith("full_attention_qkv_matches_bf16_oracle")
        for item in classes
    ):
        return "layer15_full_attention_qkv_matches_bf16_oracle_within_rounding"
    if any("mismatch" in item for item in classes):
        return "layer15_full_attention_qkv_mismatch_after_bf16_oracle"
    return "layer15_full_attention_qkv_oracle_unavailable"


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
        "layer15_full_attention_qkv_matches_bf16_oracle_exactly",
        "layer15_full_attention_qkv_matches_bf16_oracle_within_rounding",
    }:
        return "audit_layer15_qk_norm_rotary_or_kv_write_under_bf16_contract"
    if classification == "layer15_full_attention_qkv_blocked_attn_norm_input_mismatch":
        return "reconcile_layer15_attn_norm_before_qkv_projection"
    if classification in {
        "layer15_full_attention_qkv_wrong_layer_capture",
        "layer15_full_attention_qkv_wrong_layer_type",
        "layer15_full_attention_qkv_wrong_preceding_layer_count",
    }:
        return "inspect_layer15_full_attention_qkv_capture_metadata"
    if classification == "layer15_full_attention_qkv_mismatch_after_bf16_oracle":
        return "inspect_layer15_attn_qkv_weight_or_projection_kernel"
    return "rerun_layer15_full_attention_qkv_oracle_on_rocm_host"


def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: artifact["projection_results"][name]["classification"]
        for name in QKV_SPECS
        if name in artifact.get("projection_results", {})
    }


def load_full_attention_qkv_weights(
    model_path: Path,
    layer_id: int,
) -> Mapping[str, tuple[np.ndarray, dict[str, Any]]]:
    reader = GGUFReader(model_path)
    weights: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for slot, tensor_suffix in (
        ("attn_q", "attn_q.weight"),
        ("attn_k", "attn_k.weight"),
        ("attn_v", "attn_v.weight"),
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


def capture_layer15_full_attention_qkv_boundary(
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
                "kv_width": int(runner.kv_width),
                "full_q_shape": [int(2 * runner.q_width)],
                "full_k_shape": [int(runner.kv_width)],
                "full_v_shape": [int(runner.kv_width)],
            },
            "fields": {
                "attn_norm_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.norm.ptr),
                    int(runner.hidden_size),
                    runtime=runtime,
                ),
                "full_q_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.full_q.ptr),
                    int(2 * runner.q_width),
                    runtime=runtime,
                ),
                "full_k_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.full_k.ptr),
                    int(runner.kv_width),
                    runtime=runtime,
                ),
                "full_v_f32": _copy_bf16_ptr_to_host_f32(
                    int(scratch.full_v.ptr),
                    int(runner.kv_width),
                    runtime=runtime,
                ),
            },
        }


def unavailable_artifact(
    *,
    attn_norm_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    classification = "layer15_full_attention_qkv_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer15_full_attention_qkv_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": classification,
        "attn_norm_artifact_path": str(attn_norm_artifact_path),
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
    attn_norm_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    input_result: Mapping[str, Any],
    input_classification: str,
    classification: str,
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "layer15_full_attention_qkv_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "attn_norm_artifact_path": str(attn_norm_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "input_result": dict(input_result),
        "input_classification": str(input_classification),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


if __name__ == "__main__":
    main()
