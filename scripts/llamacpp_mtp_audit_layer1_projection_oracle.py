#!/usr/bin/env python3
"""Audit layer-1 linear-attention projection inputs under the BF16 contract."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    bf16_roundtrip_array,
    delta_summary,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    load_projection_weights,
    project_f32,
)
from scripts.llamacpp_mtp_audit_layer0_warm_conv_gdn_oracle import (  # noqa: E402
    load_alpha_beta_weights,
)
from scripts.llamacpp_mtp_audit_layer1_attn_norm_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER1_ATTN_NORM,
)

DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter333-layer1-projection-oracle.json")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

BoundaryCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
ProjectionWeights = Mapping[str, tuple[np.ndarray, dict[str, Any]]]
ProjectionWeightLoader = Callable[[Path, int], ProjectionWeights]

PROJECTION_SPECS = {
    "linear_qkv_f32": {
        "weight_slot": "attn_qkv",
        "hip_field": "linear_qkv_f32",
        "semantic_stage": "combined Q/K/V projection input to conv/GDN",
    },
    "linear_z_f32": {
        "weight_slot": "attn_gate",
        "hip_field": "linear_z_f32",
        "semantic_stage": "linear attention gate projection input to conv/GDN",
    },
    "ssm_alpha_f32": {
        "weight_slot": "alpha",
        "hip_field": "ssm_alpha_f32",
        "semantic_stage": "SSM alpha recurrence projection",
    },
    "ssm_beta_f32": {
        "weight_slot": "beta",
        "hip_field": "ssm_beta_f32",
        "semantic_stage": "SSM beta recurrence projection",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer1-attn-norm", type=Path, default=DEFAULT_LAYER1_ATTN_NORM)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--near-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=333)
    args = parser.parse_args()

    artifact = audit_layer1_projection_oracle(
        layer1_attn_norm_path=args.layer1_attn_norm,
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
                "field_classifications": {
                    name: artifact["projection_results"][name]["classification"]
                    for name in PROJECTION_SPECS
                    if name in artifact.get("projection_results", {})
                },
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer1_projection_oracle(
    *,
    layer1_attn_norm_path: Path,
    model_path: Path | None = None,
    layer_id: int = 1,
    max_sequence_length: int | None = None,
    near_atol: float = 2.5e-4,
    iteration: int = 333,
    boundary_capture_fn: BoundaryCaptureFn | None = None,
    projection_weight_loader: ProjectionWeightLoader | None = None,
) -> dict[str, Any]:
    attn_norm = json.loads(layer1_attn_norm_path.read_text())
    validate_layer1_attn_norm(attn_norm, expected_layer_id=int(layer_id))
    resolved_model = Path(model_path or attn_norm["model"])
    prompt_tokens = tuple(int(token) for token in attn_norm["prompt_tokens"])
    position = int(attn_norm["position"])
    token_id = int(prompt_tokens[position])
    capture_fn = boundary_capture_fn or capture_layer1_projection_boundary
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(layer_id),
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            layer1_attn_norm_path=layer1_attn_norm_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    if int((capture.get("summary") or {}).get("preceding_layer_count", -1)) != int(layer_id):
        raise ValueError("layer-1 projection capture must run preceding layers")
    selected_loader = projection_weight_loader or load_layer1_projection_weights
    projection_weights = selected_loader(resolved_model, int(layer_id))
    projection_results = build_projection_results(
        attn_norm_values=np.asarray(capture["fields"]["attn_norm_f32"], dtype=np.float32),
        capture=capture,
        projection_weights=projection_weights,
        near_atol=float(near_atol),
    )
    classification = classify_projection_audit(projection_results)
    return {
        "schema": 1,
        "kind": "layer1_bf16_projection_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "layer1_attn_norm_path": str(layer1_attn_norm_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "layer1 attn_norm_f32 from exact resident-BF16 RMSNorm oracle",
            "formula": "BF16(project_f32(attn_norm_f32, GGUF projection weight))",
            "attn_norm_classification": attn_norm.get("classification"),
            "captured_with_run_preceding_layers": True,
        },
        "weights": {
            name: metadata for name, (_value, metadata) in projection_weights.items()
        },
        "hipengine_capture": summarize_capture(capture),
        "projection_results": projection_results,
        "near_atol": float(near_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer1_attn_norm(
    artifact: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if artifact.get("status") != "ready":
        raise ValueError("layer1 attn_norm artifact must be ready")
    if artifact.get("classification") != "layer1_attn_norm_matches_bf16_oracle_exactly":
        raise ValueError("layer1 attn_norm artifact must be exact")
    if int(artifact.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer1 attn_norm layer_id does not match requested layer")
    delta = artifact.get("attn_norm_delta") or {}
    if delta.get("exact_match") is not True or float(delta.get("max_abs_diff", 1.0)) != 0.0:
        raise ValueError("layer1 attn_norm delta must be exact")
    if artifact.get("next_action") != "audit_layer1_projection_or_conv_gdn_under_bf16_contract":
        raise ValueError("layer1 attn_norm artifact must point to projection audit")


def load_layer1_projection_weights(model_path: Path, layer_id: int) -> ProjectionWeights:
    weights = dict(load_projection_weights(model_path, int(layer_id)))
    weights.update(load_alpha_beta_weights(model_path, int(layer_id)))
    return weights


def build_projection_results(
    *,
    attn_norm_values: np.ndarray,
    capture: Mapping[str, Any],
    projection_weights: ProjectionWeights,
    near_atol: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    fields = capture["fields"]
    for field, spec in PROJECTION_SPECS.items():
        weight, metadata = projection_weights[spec["weight_slot"]]
        f32 = project_f32(attn_norm_values, weight)
        bf16 = bf16_roundtrip_array(f32)
        hip = np.asarray(fields[spec["hip_field"]], dtype=np.float32)
        f32_delta = delta_summary(f32, hip)
        bf16_delta = delta_summary(bf16, hip)
        bf16_step = bf16_step_summary(bf16, hip)
        classification = classify_projection_delta(
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


def classify_projection_delta(
    delta: Mapping[str, Any],
    *,
    near_atol: float,
    bf16_step: Mapping[str, Any] | None = None,
) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "projection_oracle_unavailable"
    if delta.get("exact_match"):
        return "projection_matches_bf16_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "projection_matches_bf16_oracle_within_one_bf16_step"
    if bf16_step is not None and bf16_step.get("within_one_bf16_step"):
        return "projection_matches_bf16_oracle_within_one_bf16_step"
    return "projection_mismatch_after_bf16_oracle"


def bf16_step_summary(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    act = np.asarray(actual, dtype=np.float32).reshape(-1)
    if ref.shape != act.shape:
        return {"available": True, "shape_match": False}
    spacing = np.maximum(bf16_spacing(ref), bf16_spacing(act))
    diff = np.abs(act - ref)
    finite = np.isfinite(ref) & np.isfinite(act) & np.isfinite(spacing)
    within = np.zeros_like(diff, dtype=bool)
    within[finite] = diff[finite] <= (spacing[finite] + np.float32(1.0e-12))
    bad = finite & ~within
    return {
        "available": True,
        "shape_match": True,
        "finite_count": int(np.count_nonzero(finite)),
        "within_one_bf16_step": bool(np.all(within[finite])) if np.any(finite) else False,
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "max_allowed_step": float(np.max(spacing[finite])) if np.any(finite) else 0.0,
        "bad_count": int(np.count_nonzero(bad)),
    }


def bf16_spacing(values: np.ndarray) -> np.ndarray:
    arr = bf16_roundtrip_array(np.asarray(values, dtype=np.float32).reshape(-1))
    bits = (arr.view(np.uint32) >> 16).astype(np.uint32)
    sign = (bits & 0x8000) != 0
    up = bits.copy()
    down = bits.copy()
    pos = ~sign
    up[pos] = np.minimum(bits[pos] + 1, np.uint32(0x7F80))
    down[pos] = np.where(bits[pos] > 0, bits[pos] - 1, 1)
    up[sign] = np.where(bits[sign] > 0x8000, bits[sign] - 1, 0)
    down[sign] = np.minimum(bits[sign] + 1, np.uint32(0xFF80))
    up_f32 = (up.astype(np.uint32) << 16).view(np.float32)
    down_f32 = (down.astype(np.uint32) << 16).view(np.float32)
    return np.maximum(np.abs(up_f32 - arr), np.abs(arr - down_f32)).astype(np.float32)


def classify_projection_audit(projection_results: Mapping[str, Any]) -> str:
    classes = [projection_results[name]["classification"] for name in PROJECTION_SPECS]
    if all(item == "projection_matches_bf16_oracle_exactly" for item in classes):
        return "layer1_projections_match_bf16_oracle_exactly"
    if all(item.startswith("projection_matches_bf16_oracle") for item in classes):
        return "layer1_projections_match_bf16_oracle_within_rounding"
    if any("mismatch" in item for item in classes):
        return "layer1_projection_mismatch_after_bf16_oracle"
    return "layer1_projection_oracle_unavailable"


def status_from_classification(classification: str) -> str:
    if classification.endswith("unavailable"):
        return "unavailable"
    if "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer1_projections_match_bf16_oracle_exactly",
        "layer1_projections_match_bf16_oracle_within_rounding",
    }:
        return "audit_layer1_conv_gdn_under_bf16_contract"
    if classification == "layer1_projection_mismatch_after_bf16_oracle":
        return "inspect_layer1_projection_weight_or_kernel_quantization"
    return "rerun_layer1_projection_oracle_on_rocm_host"


def capture_layer1_projection_boundary(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    layer_id: int,
    max_sequence_length: int | None,
) -> dict[str, Any]:
    if not hip_available():
        return {"status": "skipped_no_hip_runtime", "fields": {}}
    from hipengine.runtime.qwen35_gguf_runner import (
        DType,
        Qwen35GGUFResidentSession,
        _copy_bf16_ptr_to_host_f32,
    )

    max_seq = int(max_sequence_length or max(len(prompt_tokens) + 8, 32))
    with Qwen35GGUFResidentSession(model_path, max_sequence_length=max_seq) as session:
        for index, token in enumerate(prompt_tokens[:position]):
            session.step(int(token), position=index, return_logits=False)
        layer_capture = session.capture_attention_layer(
            int(prompt_tokens[position]),
            position=int(position),
            layer_id=int(layer_id),
            run_preceding_layers=True,
        )
        runtime = session.runtime
        cfg = session.runner.weights.config
        hidden_size = int(layer_capture.hidden_size)
        rank = int(cfg.ssm_time_step_rank)
        linear_qkv_width = int(session.runner.linear_qkv_width)
        ssm_inner_size = int(cfg.ssm_inner_size)
        alpha_ptr = int(session.scratch.linear_alpha.ptr)
        beta_ptr = int(session.scratch.linear_beta.ptr)
        if cfg.is_moe:
            alpha_ptr = int(session.scratch.linear_alpha_beta.ptr)
            beta_ptr = alpha_ptr + rank * DType.BF16.itemsize
        fields = {
            "attn_norm_f32": _copy_bf16_ptr_to_host_f32(
                int(session.scratch.norm.ptr),
                hidden_size,
                runtime=runtime,
            ),
            "linear_qkv_f32": _copy_bf16_ptr_to_host_f32(
                int(session.scratch.linear_qkv.ptr),
                linear_qkv_width,
                runtime=runtime,
            ),
            "linear_z_f32": _copy_bf16_ptr_to_host_f32(
                int(session.scratch.linear_z.ptr),
                ssm_inner_size,
                runtime=runtime,
            ),
            "ssm_alpha_f32": _copy_bf16_ptr_to_host_f32(
                alpha_ptr,
                rank,
                runtime=runtime,
            ),
            "ssm_beta_f32": _copy_bf16_ptr_to_host_f32(
                beta_ptr,
                rank,
                runtime=runtime,
            ),
        }
    return {
        "status": "captured",
        "summary": {
            **layer_capture.as_summary_dict(),
            "attn_norm_shape": list(fields["attn_norm_f32"].shape),
            "linear_qkv_shape": list(fields["linear_qkv_f32"].shape),
            "linear_z_shape": list(fields["linear_z_f32"].shape),
            "ssm_alpha_shape": list(fields["ssm_alpha_f32"].shape),
            "ssm_beta_shape": list(fields["ssm_beta_f32"].shape),
        },
        "fields": fields,
    }


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def summarize_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(capture.get("summary") or {})
    fields = capture.get("fields") or {}
    field_summaries: dict[str, Any] = {}
    for name, values in fields.items():
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        field_summaries[name] = {
            "summary": summarize_array(arr),
            "sha256": sha256_float32(arr),
        }
    return {
        "status": capture.get("status"),
        "summary": summary,
        "fields": field_summaries,
    }


def unavailable_artifact(
    *,
    layer1_attn_norm_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    classification = "layer1_projection_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer1_bf16_projection_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": classification,
        "layer1_attn_norm_path": str(layer1_attn_norm_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


if __name__ == "__main__":
    main()
