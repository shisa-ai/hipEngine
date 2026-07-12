from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_warm_conv_gdn_oracle import (
    replay_conv_gdn_sequence,
)
from scripts.llamacpp_mtp_audit_layer10_conv_gdn_oracle import (
    audit_layer10_conv_gdn_oracle,
    classify_layer10_conv_gdn,
    validate_layer10_projection,
)

FIELDS = ("conv_out_f32", "recurrent_out_f32", "recurrent_bf16_f32", "attn_out_f32")


def test_validate_layer10_projection_accepts_ready_artifact() -> None:
    validate_layer10_projection(_projection_artifact(), expected_layer_id=10)


def test_validate_layer10_projection_rejects_unmatched_field() -> None:
    artifact = _projection_artifact()
    artifact["projection_results"]["linear_qkv_f32"]["classification"] = (
        "projection_mismatch_after_bf16_oracle"
    )

    with pytest.raises(ValueError, match="linear_qkv_f32"):
        validate_layer10_projection(artifact, expected_layer_id=10)


def test_validate_layer10_projection_requires_input_hash_match() -> None:
    artifact = _projection_artifact()
    artifact["input_result"]["exact_hash_match"] = False

    with pytest.raises(ValueError, match="input hash"):
        validate_layer10_projection(artifact, expected_layer_id=10)


def test_validate_layer10_projection_rejects_wrong_next_action() -> None:
    artifact = _projection_artifact()
    artifact["next_action"] = "audit_something_else"

    with pytest.raises(ValueError, match="conv/GDN audit"):
        validate_layer10_projection(artifact, expected_layer_id=10)


def test_classify_layer10_conv_gdn_maps_exact_with_layer10_prefix() -> None:
    results = {
        field: {"classification": "warm_field_matches_oracle_exactly"}
        for field in FIELDS
    }

    assert classify_layer10_conv_gdn("target_inputs_match_replay_exactly", results) == (
        "layer10_warm_conv_gdn_matches_oracle_exactly"
    )


def test_classify_layer10_conv_gdn_blocks_on_input_mismatch() -> None:
    results = {
        field: {"classification": "warm_field_matches_oracle_exactly"}
        for field in FIELDS
    }

    assert classify_layer10_conv_gdn(
        "target_inputs_mismatch_before_conv_gdn_replay",
        results,
    ) == "layer10_warm_conv_gdn_blocked_target_input_mismatch"


def test_audit_layer10_conv_gdn_oracle_with_injected_exact_sequence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    artifact = audit_layer10_conv_gdn_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer10_warm_conv_gdn_matches_oracle_exactly"
    assert artifact["target_position"] == 2
    assert artifact["target_input_classification"] == (
        "target_inputs_match_replay_exactly"
    )
    assert artifact["replay_contract"]["starts_from_zero_state"] is True
    assert artifact["replay_contract"]["projection_source"].startswith(
        "layer10_projections_match_bf16_oracle"
    )
    assert artifact["next_action"] == (
        "audit_layer10_post_attn_residual_or_moe_boundary"
    )
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer10_conv_gdn_oracle_classifies_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, conv_offset=1.0)

    artifact = audit_layer10_conv_gdn_oracle(**fixture)

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == (
        "layer10_warm_conv_gdn_mismatch_after_replay_oracle"
    )
    assert artifact["oracle_results"]["conv_out_f32"]["classification"] == (
        "warm_field_mismatch_after_replay_oracle"
    )
    assert artifact["next_action"] == (
        "inspect_layer10_first_warm_conv_gdn_replay_mismatch"
    )


def test_audit_layer10_conv_gdn_oracle_blocks_target_input_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, input_offset=1.0)

    artifact = audit_layer10_conv_gdn_oracle(**fixture)

    assert artifact["status"] == "blocked"
    assert artifact["classification"] == (
        "layer10_warm_conv_gdn_blocked_target_input_mismatch"
    )
    assert artifact["target_input_classification"] == (
        "target_inputs_mismatch_before_conv_gdn_replay"
    )
    assert artifact["next_action"] == (
        "inspect_layer10_projection_input_capture_sequence"
    )


def test_audit_layer10_conv_gdn_oracle_reports_unavailable_sequence(
    tmp_path: Path,
) -> None:
    projection_path = tmp_path / "projection.json"
    projection_path.write_text(json.dumps(_projection_artifact()))

    artifact = audit_layer10_conv_gdn_oracle(
        layer10_projection_path=projection_path,
        model_path=Path("/tmp/model.gguf"),
        sequence_capture_fn=lambda *_args: {"status": "skipped_no_hip_runtime"},
        conv_gdn_weight_loader=lambda *_args: _weights(),
        iteration=402,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == "layer10_warm_conv_gdn_oracle_unavailable"
    assert artifact["next_action"] == "rerun_layer10_conv_gdn_oracle_on_rocm_host"


def _fixture(
    tmp_path: Path,
    *,
    conv_offset: float = 0.0,
    input_offset: float = 0.0,
) -> dict[str, object]:
    projection_path = tmp_path / "projection.json"
    projection_path.write_text(json.dumps(_projection_artifact()))
    replay_inputs, weights, dims = _synthetic_replay_inputs_weights_dims()
    records, _summary = replay_conv_gdn_sequence(
        replay_inputs=replay_inputs,
        weights=weights,
        dimensions=dims,
        eps=1.0e-6,
    )
    target_capture = _target_capture_from_record(records[2], replay_inputs[2])
    if conv_offset:
        target_capture["fields"]["conv_out_f32"] = target_capture["fields"][
            "conv_out_f32"
        ].copy()
        target_capture["fields"]["conv_out_f32"][0] += np.float32(conv_offset)
    if input_offset:
        target_capture["fields"]["linear_qkv_f32"] = target_capture["fields"][
            "linear_qkv_f32"
        ].copy()
        target_capture["fields"]["linear_qkv_f32"][0] += np.float32(input_offset)
    sequence = {
        "status": "captured",
        "replay_inputs": replay_inputs,
        "target_capture": target_capture,
        "dimensions": dims | {"conv_state_floats": 18, "recurrent_state_floats": 4},
        "rms_norm_eps": 1.0e-6,
        "metadata": {"source": "synthetic", "position_count": 3},
    }
    return {
        "layer10_projection_path": projection_path,
        "model_path": Path("/tmp/model.gguf"),
        "sequence_capture_fn": lambda *_args: sequence,
        "conv_gdn_weight_loader": lambda *_args: weights,
        "iteration": 402,
    }


def _projection_artifact() -> dict:
    matched = {"classification": "projection_matches_bf16_oracle_exactly"}
    return {
        "status": "ready",
        "classification": "layer10_projections_match_bf16_oracle_within_rounding",
        "model": "/tmp/model.gguf",
        "layer_id": 10,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "input_result": {"exact_hash_match": True},
        "projection_results": {
            "linear_qkv_f32": dict(matched),
            "linear_z_f32": dict(matched),
            "ssm_alpha_f32": dict(matched),
            "ssm_beta_f32": dict(matched),
        },
        "next_action": "audit_layer10_conv_gdn_under_bf16_contract",
    }


def _dims() -> dict[str, int]:
    return {
        "ssm_group_count": 1,
        "ssm_time_step_rank": 1,
        "ssm_state_size": 2,
        "ssm_value_dim": 2,
        "ssm_inner_size": 2,
        "linear_qkv_width": 6,
        "ssm_conv_kernel": 3,
    }


def _synthetic_replay_inputs_weights_dims():
    dims = _dims()
    replay_inputs = []
    for scale in (1.0, 1.5, -0.5):
        replay_inputs.append(
            {
                "attn_norm_f32": np.full((4,), scale, dtype=np.float32),
                "linear_qkv_f32": np.asarray(
                    [0.25, -0.5, 0.75, -1.0, 0.5, -0.25],
                    dtype=np.float32,
                )
                * np.float32(scale),
                "linear_z_f32": np.asarray([0.5, -0.25], dtype=np.float32)
                * np.float32(scale),
                "ssm_alpha_f32": np.zeros((1,), dtype=np.float32),
                "ssm_beta_f32": np.asarray([0.25], dtype=np.float32),
            }
        )
    return replay_inputs, _weights(), dims


def _weights() -> dict[str, tuple[np.ndarray, dict[str, str]]]:
    return {
        "ssm_conv1d": (np.ones((6, 3), dtype=np.float32), {"tensor_name": "conv"}),
        "ssm_dt_bias": (np.zeros((1,), dtype=np.float32), {"tensor_name": "dt"}),
        "ssm_a_log": (np.zeros((1,), dtype=np.float32), {"tensor_name": "a"}),
        "ssm_norm": (np.ones((2,), dtype=np.float32), {"tensor_name": "norm"}),
        "ssm_out": (np.eye(2, dtype=np.float32), {"tensor_name": "out"}),
    }


def _target_capture_from_record(record, inputs):
    fields = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in inputs.items()
    }
    for name in FIELDS:
        fields[name] = np.asarray(record[name], dtype=np.float32)
    return {"status": "captured", "summary": {"position": 2}, "fields": fields}
