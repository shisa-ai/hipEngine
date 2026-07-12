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
from scripts.llamacpp_mtp_audit_layer11_full_attention_qkv_oracle import (
    QKV_SPECS,
    audit_layer11_full_attention_qkv_oracle,
    classify_input,
    classify_layer11_qkv,
    compare_attn_norm_input,
    validate_layer11_attn_norm_artifact,
)

FIELDS = tuple(QKV_SPECS)


def test_validate_layer11_attn_norm_artifact_accepts_exact_full_attention() -> None:
    validate_layer11_attn_norm_artifact(
        _attn_norm_artifact(_attn_norm()),
        expected_layer_id=11,
    )


def test_validate_layer11_attn_norm_artifact_rejects_wrong_layer_type() -> None:
    artifact = _attn_norm_artifact(_attn_norm())
    artifact["hipengine_capture"]["summary"]["layer_type"] = "linear_attention"

    with pytest.raises(ValueError, match="full_attention"):
        validate_layer11_attn_norm_artifact(artifact, expected_layer_id=11)


def test_compare_attn_norm_input_accepts_expected_hash() -> None:
    attn_norm = _attn_norm()
    capture = _capture(attn_norm, _projected_fields(attn_norm))
    artifact = _attn_norm_artifact(attn_norm)

    result = compare_attn_norm_input(capture=capture, artifact=artifact)

    assert result["exact_hash_match"] is True
    assert result["classification"] == "layer11_qkv_input_matches_attn_norm_artifact"
    assert classify_input(result) == "layer11_qkv_input_matches_attn_norm_artifact"


def test_compare_attn_norm_input_classifies_mismatch() -> None:
    attn_norm = _attn_norm()
    capture = _capture(attn_norm + np.float32(0.25), _projected_fields(attn_norm))
    artifact = _attn_norm_artifact(attn_norm)

    result = compare_attn_norm_input(capture=capture, artifact=artifact)

    assert result["exact_hash_match"] is False
    assert result["classification"] == "layer11_qkv_input_mismatch_before_projection"
    assert classify_input(result) == "layer11_qkv_input_mismatch_before_projection"


def test_classify_layer11_qkv_combines_field_classes() -> None:
    capture = {
        "summary": {
            "layer_id": 11,
            "layer_type": "full_attention",
            "preceding_layer_count": 11,
        }
    }
    exact = {
        field: {"classification": "full_attention_qkv_matches_bf16_oracle_exactly"}
        for field in FIELDS
    }
    near = json.loads(json.dumps(exact))
    near["full_k_f32"]["classification"] = (
        "full_attention_qkv_matches_bf16_oracle_within_one_bf16_step"
    )
    mismatch = json.loads(json.dumps(exact))
    mismatch["full_v_f32"]["classification"] = (
        "full_attention_qkv_mismatch_after_bf16_oracle"
    )

    assert classify_layer11_qkv(
        "layer11_qkv_input_matches_attn_norm_artifact",
        exact,
        capture=capture,
        layer_id=11,
    ) == "layer11_full_attention_qkv_matches_bf16_oracle_exactly"
    assert classify_layer11_qkv(
        "layer11_qkv_input_matches_attn_norm_artifact",
        near,
        capture=capture,
        layer_id=11,
    ) == "layer11_full_attention_qkv_matches_bf16_oracle_within_rounding"
    assert classify_layer11_qkv(
        "layer11_qkv_input_matches_attn_norm_artifact",
        mismatch,
        capture=capture,
        layer_id=11,
    ) == "layer11_full_attention_qkv_mismatch_after_bf16_oracle"


def test_audit_layer11_full_attention_qkv_oracle_with_injected_exact_capture(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    artifact = audit_layer11_full_attention_qkv_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer11_full_attention_qkv_matches_bf16_oracle_exactly"
    )
    assert artifact["input_classification"] == (
        "layer11_qkv_input_matches_attn_norm_artifact"
    )
    for field in FIELDS:
        assert artifact["projection_results"][field]["classification"] == (
            "full_attention_qkv_matches_bf16_oracle_exactly"
        )
    assert artifact["next_action"] == (
        "audit_layer11_qk_norm_rotary_or_kv_write_under_bf16_contract"
    )
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer11_full_attention_qkv_oracle_with_bf16_step_rounding(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, q_offset=2.0e-3)

    artifact = audit_layer11_full_attention_qkv_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer11_full_attention_qkv_matches_bf16_oracle_within_rounding"
    )
    q_result = artifact["projection_results"]["full_q_f32"]
    assert q_result["delta_bf16_oracle_vs_hip"]["max_abs_diff"] > 2.5e-4
    assert q_result["bf16_step_oracle_vs_hip"]["within_one_bf16_step"] is True
    assert q_result["classification"] == (
        "full_attention_qkv_matches_bf16_oracle_within_one_bf16_step"
    )


def test_audit_layer11_full_attention_qkv_oracle_classifies_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, v_offset=0.25)

    artifact = audit_layer11_full_attention_qkv_oracle(**fixture)

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == (
        "layer11_full_attention_qkv_mismatch_after_bf16_oracle"
    )
    assert artifact["projection_results"]["full_v_f32"]["classification"] == (
        "full_attention_qkv_mismatch_after_bf16_oracle"
    )
    assert artifact["next_action"] == "inspect_layer11_attn_qkv_weight_or_projection_kernel"


def test_audit_layer11_full_attention_qkv_oracle_blocks_wrong_layer_type(
    tmp_path: Path,
) -> None:
    attn_norm = _attn_norm()
    fixture = _fixture(tmp_path)
    fixture["boundary_capture_fn"] = lambda *_args: _capture(
        attn_norm,
        _projected_fields(attn_norm),
        layer_type="linear_attention",
    )

    artifact = audit_layer11_full_attention_qkv_oracle(**fixture)

    assert artifact["status"] == "blocked"
    assert artifact["classification"] == "layer11_full_attention_qkv_wrong_layer_type"
    assert artifact["next_action"] == "inspect_layer11_full_attention_qkv_capture_metadata"


def test_audit_layer11_full_attention_qkv_oracle_reports_unavailable_capture(
    tmp_path: Path,
) -> None:
    attn_path = tmp_path / "attn_norm.json"
    attn_path.write_text(json.dumps(_attn_norm_artifact(_attn_norm())))

    artifact = audit_layer11_full_attention_qkv_oracle(
        attn_norm_artifact_path=attn_path,
        model_path=Path("/tmp/model.gguf"),
        boundary_capture_fn=lambda *_args: {
            "status": "skipped_no_hip_runtime",
            "fields": {},
        },
        projection_weight_loader=lambda *_args: _weights(),
        iteration=408,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == "layer11_full_attention_qkv_oracle_unavailable"
    assert artifact["next_action"] == "rerun_layer11_full_attention_qkv_oracle_on_rocm_host"


def _fixture(
    tmp_path: Path,
    *,
    q_offset: float = 0.0,
    v_offset: float = 0.0,
) -> dict[str, object]:
    attn_norm = _attn_norm()
    fields = _projected_fields(attn_norm)
    fields["full_q_f32"] = fields["full_q_f32"].copy()
    fields["full_q_f32"][0] += np.float32(q_offset)
    fields["full_v_f32"] = fields["full_v_f32"].copy()
    fields["full_v_f32"][0] += np.float32(v_offset)
    attn_path = tmp_path / "attn_norm.json"
    attn_path.write_text(json.dumps(_attn_norm_artifact(attn_norm)))
    capture = _capture(attn_norm, fields)
    return {
        "attn_norm_artifact_path": attn_path,
        "model_path": Path("/tmp/model.gguf"),
        "boundary_capture_fn": lambda *_args: capture,
        "projection_weight_loader": lambda *_args: _weights(),
        "near_atol": 2.5e-4,
        "iteration": 408,
    }


def _attn_norm() -> np.ndarray:
    return bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))


def _attn_norm_artifact(attn_norm: np.ndarray) -> dict:
    return {
        "status": "ready",
        "classification": "layer11_attn_norm_matches_bf16_oracle_exactly",
        "model": "/tmp/model.gguf",
        "layer_id": 11,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "hipengine_capture": {
            "summary": {
                "layer_type": "full_attention",
                "preceding_layer_count": 11,
            },
            "fields": {"attn_norm_f32": {"sha256": sha256_float32(attn_norm)}},
        },
        "attn_norm_delta": {"exact_match": True, "max_abs_diff": 0.0},
        "next_action": "audit_layer11_attention_projection_under_bf16_contract",
    }


def _projected_fields(attn_norm: np.ndarray) -> dict[str, np.ndarray]:
    weights = _weights()
    return {
        "full_q_f32": bf16_roundtrip_array(
            project_f32(attn_norm, weights["attn_q"][0])
        ),
        "full_k_f32": bf16_roundtrip_array(
            project_f32(attn_norm, weights["attn_k"][0])
        ),
        "full_v_f32": bf16_roundtrip_array(
            project_f32(attn_norm, weights["attn_v"][0])
        ),
    }


def _capture(
    attn_norm: np.ndarray,
    projected: dict[str, np.ndarray],
    *,
    layer_type: str = "full_attention",
    preceding_layer_count: int = 11,
) -> dict:
    return {
        "status": "captured",
        "summary": {
            "layer_id": 11,
            "position": 2,
            "token_id": 12,
            "hidden_size": int(attn_norm.size),
            "layer_type": layer_type,
            "preceding_layer_count": int(preceding_layer_count),
            "q_width": 2,
            "kv_width": 1,
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
        "attn_q": (
            np.array(
                [[1.0, -0.5, 0.25, 2.0], [-0.25, 0.75, 1.5, -1.0]],
                dtype=np.float32,
            ),
            {"tensor_name": "blk.11.attn_q.weight"},
        ),
        "attn_k": (
            np.array([[0.1, -0.2, 0.3, -0.4]], dtype=np.float32),
            {"tensor_name": "blk.11.attn_k.weight"},
        ),
        "attn_v": (
            np.array([[1.25, -1.0, 0.5, 0.25]], dtype=np.float32),
            {"tensor_name": "blk.11.attn_v.weight"},
        ),
    }
