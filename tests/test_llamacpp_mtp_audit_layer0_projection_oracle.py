from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (
    bf16_roundtrip_array,
    pack_float32,
    rmsnorm_f32,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (
    audit_layer0_projection_oracle,
    classify_projection_delta,
    project_f32,
)


def test_project_f32_uses_weight_rows_as_output_features() -> None:
    x = np.asarray([1.0, 2.0], dtype=np.float32)
    weight = np.asarray([[3.0, 4.0], [-1.0, 0.5]], dtype=np.float32)

    actual = project_f32(x, weight)

    np.testing.assert_allclose(actual, np.asarray([11.0, 0.0], dtype=np.float32))


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
            "max_abs_diff": 2.4e-4,
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
    ) == "projection_mismatch_after_bf16_contracted_oracle"


def test_audit_layer0_projection_oracle_exact_synthetic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    artifact = audit_layer0_projection_oracle(**fixture.kwargs())

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer0_projections_match_bf16_oracle_exactly"
    assert artifact["attn_norm_oracle"]["delta_vs_hip"]["exact_match"] is True
    assert artifact["projection_results"]["linear_qkv_f32"]["classification"] == (
        "projection_matches_bf16_oracle_exactly"
    )
    assert artifact["projection_results"]["linear_z_f32"]["classification"] == (
        "projection_matches_bf16_oracle_exactly"
    )
    assert artifact["projection_results"]["linear_qkv_f32"]["weight"]["tensor_name"] == (
        "blk.0.attn_qkv.weight"
    )
    assert artifact["next_action"] == "continue_layer0_bf16_bisection_at_conv_or_gdn_state"
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer0_projection_oracle_near_rounding_synthetic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, qkv_offset=2.0e-4)

    artifact = audit_layer0_projection_oracle(**fixture.kwargs())

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer0_projections_match_bf16_oracle_within_rounding"
    assert artifact["projection_results"]["linear_qkv_f32"]["classification"] == (
        "projection_matches_bf16_oracle_within_one_bf16_step"
    )
    assert artifact["projection_results"]["linear_z_f32"]["classification"] == (
        "projection_matches_bf16_oracle_exactly"
    )


def test_audit_layer0_projection_oracle_mismatch_synthetic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, qkv_offset=0.01)

    artifact = audit_layer0_projection_oracle(**fixture.kwargs())

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer0_projection_mismatch_after_bf16_oracle"
    assert artifact["projection_results"]["linear_qkv_f32"]["classification"] == (
        "projection_mismatch_after_bf16_contracted_oracle"
    )
    assert artifact["next_action"] == "audit_layer0_projection_weight_or_kernel_quantization"


def test_audit_layer0_projection_oracle_requires_ready_policy(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    policy = json.loads(fixture.policy_path.read_text())
    policy["status"] = "blocked"
    fixture.policy_path.write_text(json.dumps(policy))

    try:
        audit_layer0_projection_oracle(**fixture.kwargs())
    except ValueError as exc:
        assert "policy must be ready" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


class _Fixture:
    def __init__(
        self,
        *,
        input_path: Path,
        policy_path: Path,
        formula_path: Path,
        model_path: Path,
        norm_weight: np.ndarray,
        eps: float,
        projection_weights: dict[str, np.ndarray],
        capture: dict[str, object],
    ) -> None:
        self.input_path = input_path
        self.policy_path = policy_path
        self.formula_path = formula_path
        self.model_path = model_path
        self.norm_weight = norm_weight
        self.eps = eps
        self.projection_weights = projection_weights
        self.capture = capture

    def kwargs(self) -> dict[str, object]:
        return {
            "input_compare_path": self.input_path,
            "policy_path": self.policy_path,
            "formula_audit_path": self.formula_path,
            "model_path": self.model_path,
            "boundary_capture_fn": lambda *_args: self.capture,
            "projection_weight_loader": lambda *_args: _projection_loader(
                self.projection_weights
            ),
            "norm_weight_loader": lambda *_args: (
                self.norm_weight,
                self.eps,
                {"tensor_name": "blk.0.attn_norm.weight"},
            ),
            "near_atol": 2.5e-4,
        }


def _fixture(tmp_path: Path, *, qkv_offset: float = 0.0) -> _Fixture:
    input_f32 = np.asarray([0.1253, -0.0627, 0.0311, -0.0189], dtype=np.float32)
    norm_weight = np.asarray([1.1, 0.9, 1.25, 0.75], dtype=np.float32)
    eps = 1.0e-6
    attn_norm = bf16_roundtrip_array(
        rmsnorm_f32(bf16_roundtrip_array(input_f32), norm_weight, eps)
    )
    qkv_weight = np.asarray(
        [[1.0, -0.5, 0.25, 2.0], [-0.25, 0.75, 1.5, -1.0]],
        dtype=np.float32,
    )
    z_weight = np.asarray(
        [[0.1, -0.2, 0.3, -0.4], [1.25, -1.0, 0.5, 0.25], [-0.5, 0.5, -0.5, 0.5]],
        dtype=np.float32,
    )
    qkv = bf16_roundtrip_array(project_f32(attn_norm, qkv_weight))
    z = bf16_roundtrip_array(project_f32(attn_norm, z_weight))
    qkv_with_offset = qkv.copy()
    qkv_with_offset[0] += np.float32(qkv_offset)
    input_path = _write_input_compare(tmp_path, input_f32)
    policy_path = _write_policy(tmp_path)
    formula_path = _write_formula(tmp_path)
    capture = {
        "status": "captured",
        "summary": {"finite": True},
        "fields": {
            "attn_norm_f32": attn_norm,
            "linear_qkv_f32": qkv_with_offset,
            "linear_z_f32": z,
            "ssm_alpha_f32": np.zeros((1,), dtype=np.float32),
            "ssm_beta_f32": np.zeros((1,), dtype=np.float32),
            "conv_out_f32": np.zeros((2,), dtype=np.float32),
            "recurrent_out_f32": np.zeros((1,), dtype=np.float32),
            "recurrent_bf16_f32": np.zeros((1,), dtype=np.float32),
            "attn_out_f32": np.zeros((4,), dtype=np.float32),
        },
    }
    return _Fixture(
        input_path=input_path,
        policy_path=policy_path,
        formula_path=formula_path,
        model_path=tmp_path / "model.gguf",
        norm_weight=norm_weight,
        eps=eps,
        projection_weights={"attn_qkv": qkv_weight, "attn_gate": z_weight},
        capture=capture,
    )


def _write_input_compare(tmp_path: Path, values: np.ndarray) -> Path:
    binary = tmp_path / "input.f32"
    binary.write_bytes(pack_float32(values))
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"llamacpp_capture": {"binary_path": str(binary)}}))
    return path


def _write_policy(tmp_path: Path) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "status": "ready",
                "model": str(tmp_path / "model.gguf"),
                "layer_id": 0,
                "position": 2,
                "decision": {"selected_policy": "bf16_contracted_llamacpp_or_cpu_oracle"},
            }
        )
    )
    return path


def _write_formula(tmp_path: Path) -> Path:
    path = tmp_path / "formula.json"
    path.write_text(json.dumps({"prompt_tokens": [3, 5, 9]}))
    return path


def _projection_loader(weights: dict[str, np.ndarray]):
    return {
        slot: (
            value,
            {
                "tensor_name": f"blk.0.{slot.replace('attn_gate', 'attn_gate')}.weight",
                "shape": list(value.shape),
            },
        )
        for slot, value in weights.items()
    }
