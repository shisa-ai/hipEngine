from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (
    bf16_roundtrip_array,
    sha256_float32,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import project_f32
from scripts.llamacpp_mtp_audit_layer13_projection_oracle import (
    audit_layer13_projection_oracle,
    classify_input,
    classify_layer13_projection_audit,
    compare_attn_norm_input,
    validate_layer13_attn_norm,
)

FIELDS = ("linear_qkv_f32", "linear_z_f32", "ssm_alpha_f32", "ssm_beta_f32")


def test_validate_layer13_attn_norm_accepts_exact_artifact() -> None:
    validate_layer13_attn_norm(
        _attn_norm_artifact(_attn_norm()),
        expected_layer_id=13,
    )


def test_validate_layer13_attn_norm_rejects_nonexact_delta() -> None:
    artifact = _attn_norm_artifact(_attn_norm())
    artifact["attn_norm_delta"]["max_abs_diff"] = 0.5

    with pytest.raises(ValueError, match="attn_norm delta"):
        validate_layer13_attn_norm(artifact, expected_layer_id=13)


def test_validate_layer13_attn_norm_rejects_wrong_next_action() -> None:
    artifact = _attn_norm_artifact(_attn_norm())
    artifact["next_action"] = "audit_something_else"

    with pytest.raises(ValueError, match="projection audit"):
        validate_layer13_attn_norm(artifact, expected_layer_id=13)


def test_compare_attn_norm_input_accepts_expected_hash() -> None:
    attn_norm = _attn_norm()
    capture = _capture(attn_norm, _projected_fields(attn_norm))
    artifact = _attn_norm_artifact(attn_norm)

    result = compare_attn_norm_input(capture=capture, attn_norm=artifact)

    assert result["exact_hash_match"] is True
    assert result["classification"] == (
        "layer13_projection_input_matches_attn_norm_artifact"
    )
    assert classify_input(result) == (
        "layer13_projection_input_matches_attn_norm_artifact"
    )


def test_compare_attn_norm_input_classifies_mismatch() -> None:
    attn_norm = _attn_norm()
    capture = _capture(attn_norm + np.float32(0.25), _projected_fields(attn_norm))
    artifact = _attn_norm_artifact(attn_norm)

    result = compare_attn_norm_input(capture=capture, attn_norm=artifact)

    assert result["exact_hash_match"] is False
    assert result["classification"] == (
        "layer13_projection_input_mismatch_before_projection"
    )
    assert classify_input(result) == (
        "layer13_projection_input_mismatch_before_projection"
    )


def test_classify_layer13_projection_audit_combines_field_classes() -> None:
    capture = {
        "summary": {
            "layer_id": 13,
            "layer_type": "linear_attention",
            "preceding_layer_count": 13,
        }
    }
    exact = {
        field: {"classification": "projection_matches_bf16_oracle_exactly"}
        for field in FIELDS
    }
    near = json.loads(json.dumps(exact))
    near["ssm_alpha_f32"]["classification"] = (
        "projection_matches_bf16_oracle_within_one_bf16_step"
    )
    mismatch = json.loads(json.dumps(exact))
    mismatch["ssm_beta_f32"]["classification"] = (
        "projection_mismatch_after_bf16_oracle"
    )

    assert classify_layer13_projection_audit(
        "layer13_projection_input_matches_attn_norm_artifact",
        exact,
        capture=capture,
        layer_id=13,
    ) == "layer13_projections_match_bf16_oracle_exactly"
    assert classify_layer13_projection_audit(
        "layer13_projection_input_matches_attn_norm_artifact",
        near,
        capture=capture,
        layer_id=13,
    ) == "layer13_projections_match_bf16_oracle_within_rounding"
    assert classify_layer13_projection_audit(
        "layer13_projection_input_matches_attn_norm_artifact",
        mismatch,
        capture=capture,
        layer_id=13,
    ) == "layer13_projection_mismatch_after_bf16_oracle"


def test_audit_layer13_projection_oracle_with_injected_exact_capture(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    artifact = audit_layer13_projection_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer13_projections_match_bf16_oracle_exactly"
    )
    assert artifact["input_classification"] == (
        "layer13_projection_input_matches_attn_norm_artifact"
    )
    for field in FIELDS:
        assert artifact["projection_results"][field]["classification"] == (
            "projection_matches_bf16_oracle_exactly"
        )
    assert artifact["next_action"] == (
        "audit_layer13_conv_gdn_under_bf16_contract"
    )
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer13_projection_oracle_with_near_rounding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, qkv_offset=2.0e-4)

    artifact = audit_layer13_projection_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer13_projections_match_bf16_oracle_within_rounding"
    )
    assert artifact["projection_results"]["linear_qkv_f32"]["classification"] == (
        "projection_matches_bf16_oracle_within_one_bf16_step"
    )


def test_audit_layer13_projection_oracle_classifies_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, beta_offset=0.25)

    artifact = audit_layer13_projection_oracle(**fixture)

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == (
        "layer13_projection_mismatch_after_bf16_oracle"
    )
    assert artifact["projection_results"]["ssm_beta_f32"]["classification"] == (
        "projection_mismatch_after_bf16_oracle"
    )
    assert artifact["next_action"] == (
        "inspect_layer13_projection_weight_or_kernel_quantization"
    )


def test_audit_layer13_projection_oracle_blocks_wrong_layer_type(
    tmp_path: Path,
) -> None:
    attn_norm = _attn_norm()
    fixture = _fixture(tmp_path)
    fixture["boundary_capture_fn"] = lambda *_args: _capture(
        attn_norm,
        _projected_fields(attn_norm),
        layer_type="full_attention",
    )

    artifact = audit_layer13_projection_oracle(**fixture)

    assert artifact["status"] == "blocked"
    assert artifact["classification"] == "layer13_projection_wrong_layer_type"
    assert artifact["next_action"] == (
        "inspect_layer13_projection_capture_layer_metadata"
    )


def test_audit_layer13_projection_oracle_reports_unavailable_capture(
    tmp_path: Path,
) -> None:
    attn_path = tmp_path / "attn_norm.json"
    attn_path.write_text(json.dumps(_attn_norm_artifact(_attn_norm())))

    artifact = audit_layer13_projection_oracle(
        layer13_attn_norm_path=attn_path,
        model_path=Path("/tmp/model.gguf"),
        boundary_capture_fn=lambda *_args: {
            "status": "skipped_no_hip_runtime",
            "fields": {},
        },
        projection_weight_loader=lambda *_args: _weights(),
        iteration=424,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == "layer13_projection_oracle_unavailable"
    assert artifact["next_action"] == (
        "rerun_layer13_projection_oracle_on_rocm_host"
    )


def _fixture(
    tmp_path: Path,
    *,
    qkv_offset: float = 0.0,
    beta_offset: float = 0.0,
) -> dict[str, object]:
    attn_norm = _attn_norm()
    fields = _projected_fields(attn_norm)
    fields["linear_qkv_f32"] = fields["linear_qkv_f32"].copy()
    fields["linear_qkv_f32"][0] += np.float32(qkv_offset)
    fields["ssm_beta_f32"] = fields["ssm_beta_f32"].copy()
    fields["ssm_beta_f32"][0] += np.float32(beta_offset)
    attn_path = tmp_path / "attn_norm.json"
    attn_path.write_text(json.dumps(_attn_norm_artifact(attn_norm)))
    capture = _capture(attn_norm, fields)
    return {
        "layer13_attn_norm_path": attn_path,
        "model_path": Path("/tmp/model.gguf"),
        "boundary_capture_fn": lambda *_args: capture,
        "projection_weight_loader": lambda *_args: _weights(),
        "near_atol": 2.5e-4,
        "iteration": 424,
    }


def _attn_norm() -> np.ndarray:
    return bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))


def _attn_norm_artifact(attn_norm: np.ndarray) -> dict:
    return {
        "status": "ready",
        "classification": "layer13_attn_norm_matches_bf16_oracle_exactly",
        "model": "/tmp/model.gguf",
        "layer_id": 13,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "hipengine_capture": {
            "fields": {"attn_norm_f32": {"sha256": sha256_float32(attn_norm)}}
        },
        "attn_norm_delta": {"exact_match": True, "max_abs_diff": 0.0},
        "next_action": "audit_layer13_projection_or_conv_gdn_under_bf16_contract",
    }


def _projected_fields(attn_norm: np.ndarray) -> dict[str, np.ndarray]:
    weights = _weights()
    return {
        "linear_qkv_f32": bf16_roundtrip_array(
            project_f32(attn_norm, weights["attn_qkv"][0])
        ),
        "linear_z_f32": bf16_roundtrip_array(
            project_f32(attn_norm, weights["attn_gate"][0])
        ),
        "ssm_alpha_f32": bf16_roundtrip_array(
            project_f32(attn_norm, weights["alpha"][0])
        ),
        "ssm_beta_f32": bf16_roundtrip_array(
            project_f32(attn_norm, weights["beta"][0])
        ),
    }


def _capture(
    attn_norm: np.ndarray,
    projected: dict[str, np.ndarray],
    *,
    preceding_layer_count: int = 13,
    layer_type: str = "linear_attention",
) -> dict:
    return {
        "status": "captured",
        "summary": {
            "layer_id": 13,
            "position": 2,
            "token_id": 12,
            "hidden_size": int(attn_norm.size),
            "layer_type": layer_type,
            "preceding_layer_count": int(preceding_layer_count),
            "finite": True,
        },
        "fields": {
            "attn_norm_f32": np.asarray(attn_norm, dtype=np.float32),
            **{
                name: np.asarray(value, dtype=np.float32)
                for name, value in projected.items()
            },
        },
    }


def _weights() -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    return {
        "attn_qkv": (
            np.array(
                [[1.0, -0.5, 0.25, 2.0], [-0.25, 0.75, 1.5, -1.0]],
                dtype=np.float32,
            ),
            {"tensor_name": "blk.13.attn_qkv.weight"},
        ),
        "attn_gate": (
            np.array(
                [[0.1, -0.2, 0.3, -0.4], [1.25, -1.0, 0.5, 0.25]],
                dtype=np.float32,
            ),
            {"tensor_name": "blk.13.attn_gate.weight"},
        ),
        "alpha": (
            np.array([[0.25, 0.0, -0.5, 1.0]], dtype=np.float32),
            {"tensor_name": "blk.13.ssm_alpha.weight"},
        ),
        "beta": (
            np.array([[-1.0, 0.5, 0.25, 0.0]], dtype=np.float32),
            {"tensor_name": "blk.13.ssm_beta.weight"},
        ),
    }
