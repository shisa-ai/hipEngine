from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import bf16_roundtrip_array
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import project_f32
from scripts.llamacpp_mtp_audit_layer1_projection_oracle import (
    audit_layer1_projection_oracle,
    bf16_step_summary,
    classify_projection_audit,
    classify_projection_delta,
    validate_layer1_attn_norm,
)


def test_validate_layer1_attn_norm_accepts_exact_artifact() -> None:
    validate_layer1_attn_norm(_attn_norm_artifact(), expected_layer_id=1)


def test_validate_layer1_attn_norm_rejects_nonexact_delta() -> None:
    artifact = _attn_norm_artifact()
    artifact["attn_norm_delta"]["max_abs_diff"] = 0.5

    with pytest.raises(ValueError, match="attn_norm delta"):
        validate_layer1_attn_norm(artifact, expected_layer_id=1)


def test_projection_delta_classification_exact_near_and_mismatch() -> None:
    assert classify_projection_delta(
        {"available": True, "shape_match": True, "exact_match": True},
        near_atol=2.5e-4,
    ) == "projection_matches_bf16_oracle_exactly"
    assert classify_projection_delta(
        {
            "available": True,
            "shape_match": True,
            "exact_match": False,
            "max_abs_diff": 2.0e-4,
        },
        near_atol=2.5e-4,
    ) == "projection_matches_bf16_oracle_within_one_bf16_step"
    assert classify_projection_delta(
        {
            "available": True,
            "shape_match": True,
            "exact_match": False,
            "max_abs_diff": 1.0e-2,
        },
        near_atol=2.5e-4,
    ) == "projection_mismatch_after_bf16_oracle"


def test_projection_delta_accepts_one_bf16_adjacent_step() -> None:
    reference = np.array([-0.52734375], dtype=np.float32)
    actual = np.array([-0.53125], dtype=np.float32)
    bf16_step = bf16_step_summary(reference, actual)

    assert bf16_step["within_one_bf16_step"] is True
    assert classify_projection_delta(
        {
            "available": True,
            "shape_match": True,
            "exact_match": False,
            "max_abs_diff": 0.00390625,
        },
        near_atol=2.5e-4,
        bf16_step=bf16_step,
    ) == "projection_matches_bf16_oracle_within_one_bf16_step"


def test_classify_projection_audit_combines_field_classes() -> None:
    exact = {
        field: {"classification": "projection_matches_bf16_oracle_exactly"}
        for field in ["linear_qkv_f32", "linear_z_f32", "ssm_alpha_f32", "ssm_beta_f32"]
    }
    near = json.loads(json.dumps(exact))
    near["ssm_alpha_f32"]["classification"] = (
        "projection_matches_bf16_oracle_within_one_bf16_step"
    )
    mismatch = json.loads(json.dumps(exact))
    mismatch["ssm_beta_f32"]["classification"] = "projection_mismatch_after_bf16_oracle"

    assert classify_projection_audit(exact) == "layer1_projections_match_bf16_oracle_exactly"
    assert classify_projection_audit(near) == (
        "layer1_projections_match_bf16_oracle_within_rounding"
    )
    assert classify_projection_audit(mismatch) == "layer1_projection_mismatch_after_bf16_oracle"


def test_audit_layer1_projection_oracle_with_injected_exact_capture(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    artifact = audit_layer1_projection_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer1_projections_match_bf16_oracle_exactly"
    for field in ["linear_qkv_f32", "linear_z_f32", "ssm_alpha_f32", "ssm_beta_f32"]:
        assert artifact["projection_results"][field]["classification"] == (
            "projection_matches_bf16_oracle_exactly"
        )
    assert artifact["next_action"] == "audit_layer1_conv_gdn_under_bf16_contract"
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer1_projection_oracle_with_near_rounding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, qkv_offset=2.0e-4)

    artifact = audit_layer1_projection_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer1_projections_match_bf16_oracle_within_rounding"
    assert artifact["projection_results"]["linear_qkv_f32"]["classification"] == (
        "projection_matches_bf16_oracle_within_one_bf16_step"
    )


def test_audit_layer1_projection_oracle_classifies_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, beta_offset=0.25)

    artifact = audit_layer1_projection_oracle(**fixture)

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer1_projection_mismatch_after_bf16_oracle"
    assert artifact["projection_results"]["ssm_beta_f32"]["classification"] == (
        "projection_mismatch_after_bf16_oracle"
    )
    assert artifact["next_action"] == "inspect_layer1_projection_weight_or_kernel_quantization"


def test_audit_layer1_projection_oracle_reports_unavailable_capture(tmp_path: Path) -> None:
    attn_path = tmp_path / "attn_norm.json"
    attn_path.write_text(json.dumps(_attn_norm_artifact()))

    artifact = audit_layer1_projection_oracle(
        layer1_attn_norm_path=attn_path,
        model_path=Path("/tmp/model.gguf"),
        boundary_capture_fn=lambda *_args: {"status": "skipped_no_hip_runtime", "fields": {}},
        projection_weight_loader=lambda *_args: _weights(),
        iteration=333,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == "layer1_projection_oracle_unavailable"
    assert artifact["next_action"] == "rerun_layer1_projection_oracle_on_rocm_host"


def _fixture(
    tmp_path: Path,
    *,
    qkv_offset: float = 0.0,
    beta_offset: float = 0.0,
) -> dict[str, object]:
    attn_path = tmp_path / "attn_norm.json"
    attn_path.write_text(json.dumps(_attn_norm_artifact()))
    attn_norm = bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))
    weights = _weights()
    qkv = bf16_roundtrip_array(project_f32(attn_norm, weights["attn_qkv"][0]))
    gate = bf16_roundtrip_array(project_f32(attn_norm, weights["attn_gate"][0]))
    alpha = bf16_roundtrip_array(project_f32(attn_norm, weights["alpha"][0]))
    beta = bf16_roundtrip_array(project_f32(attn_norm, weights["beta"][0]))
    qkv = qkv.copy()
    qkv[0] += np.float32(qkv_offset)
    beta = beta.copy()
    beta[0] += np.float32(beta_offset)
    capture = {
        "status": "captured",
        "summary": {
            "layer_id": 1,
            "position": 2,
            "token_id": 12,
            "hidden_size": 4,
            "preceding_layer_count": 1,
            "finite": True,
        },
        "fields": {
            "attn_norm_f32": attn_norm,
            "linear_qkv_f32": qkv,
            "linear_z_f32": gate,
            "ssm_alpha_f32": alpha,
            "ssm_beta_f32": beta,
        },
    }
    return {
        "layer1_attn_norm_path": attn_path,
        "model_path": Path("/tmp/model.gguf"),
        "boundary_capture_fn": lambda *_args: capture,
        "projection_weight_loader": lambda *_args: weights,
        "near_atol": 2.5e-4,
        "iteration": 333,
    }


def _attn_norm_artifact() -> dict:
    return {
        "status": "ready",
        "classification": "layer1_attn_norm_matches_bf16_oracle_exactly",
        "model": "/tmp/model.gguf",
        "layer_id": 1,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "attn_norm_delta": {"exact_match": True, "max_abs_diff": 0.0},
        "next_action": "audit_layer1_projection_or_conv_gdn_under_bf16_contract",
    }


def _weights() -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    return {
        "attn_qkv": (
            np.array([[1.0, -0.5, 0.25, 2.0], [-0.25, 0.75, 1.5, -1.0]], dtype=np.float32),
            {"tensor_name": "blk.1.attn_qkv.weight"},
        ),
        "attn_gate": (
            np.array([[0.1, -0.2, 0.3, -0.4], [1.25, -1.0, 0.5, 0.25]], dtype=np.float32),
            {"tensor_name": "blk.1.attn_gate.weight"},
        ),
        "alpha": (
            np.array([[0.25, 0.0, -0.5, 1.0]], dtype=np.float32),
            {"tensor_name": "blk.1.ssm_alpha.weight"},
        ),
        "beta": (
            np.array([[-1.0, 0.5, 0.25, 0.0]], dtype=np.float32),
            {"tensor_name": "blk.1.ssm_beta.weight"},
        ),
    }
