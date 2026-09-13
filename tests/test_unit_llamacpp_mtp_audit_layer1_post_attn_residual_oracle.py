from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import bf16_roundtrip_array
from scripts.llamacpp_mtp_audit_layer0_post_attn_residual_oracle import (
    add_rmsnorm_bf16_oracle,
)
from scripts.llamacpp_mtp_audit_layer1_post_attn_residual_oracle import (
    audit_layer1_post_attn_residual_oracle,
    classify_inputs,
    classify_layer1_post_attn,
    compare_input_hashes,
    load_input_references,
    validate_layer1_conv_gdn,
)


def test_validate_layer1_conv_gdn_accepts_ready_artifact() -> None:
    validate_layer1_conv_gdn(_conv_artifact(), expected_layer_id=1)


def test_validate_layer1_conv_gdn_rejects_unmatched_attn_out() -> None:
    artifact = _conv_artifact()
    artifact["oracle_results"]["attn_out_f32"]["classification"] = (
        "warm_field_mismatch_after_replay_oracle"
    )

    with pytest.raises(ValueError, match="attn_out"):
        validate_layer1_conv_gdn(artifact, expected_layer_id=1)


def test_compare_input_hashes_accepts_expected_hashes() -> None:
    capture = _capture()
    refs = _input_refs(capture)

    results = compare_input_hashes(capture=capture, input_references=refs)

    assert results["hidden_in_f32"]["classification"] == (
        "post_attn_input_hash_matches_reference"
    )
    assert results["attn_out_f32"]["classification"] == (
        "post_attn_input_hash_matches_reference"
    )
    assert classify_inputs(results) == "layer1_post_attn_inputs_match_prior_artifacts"


def test_compare_input_hashes_classifies_hash_mismatch() -> None:
    capture = _capture()
    refs = _input_refs(capture)
    refs["attn_out_f32"]["expected_sha256"] = "bad-sha"

    results = compare_input_hashes(capture=capture, input_references=refs)

    assert results["attn_out_f32"]["classification"] == (
        "post_attn_input_hash_mismatch_before_residual"
    )
    assert classify_inputs(results) == "layer1_post_attn_inputs_mismatch_before_residual"


def test_classify_layer1_post_attn_maps_exact_and_blocked() -> None:
    exact = {
        "residual_f32": {"classification": "post_attn_field_matches_oracle_exactly"},
        "post_norm_f32": {"classification": "post_attn_field_matches_oracle_exactly"},
    }

    assert classify_layer1_post_attn(
        "layer1_post_attn_inputs_match_prior_artifacts",
        exact,
    ) == "layer1_post_attn_residual_matches_oracle_exactly"
    assert classify_layer1_post_attn("layer1_post_attn_inputs_mismatch_before_residual", exact) == (
        "layer1_post_attn_residual_blocked_input_mismatch"
    )


def test_audit_layer1_post_attn_residual_oracle_with_injected_exact_inputs(
    tmp_path: Path,
) -> None:
    conv_path = tmp_path / "conv.json"
    conv_path.write_text(json.dumps(_conv_artifact()))
    capture = _capture(position=2, token_id=12)

    artifact = audit_layer1_post_attn_residual_oracle(
        conv_gdn_artifact_path=conv_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        norm_weight_loader=lambda *_args: (
            np.ones((4,), dtype=np.float32),
            1.0e-6,
            {"tensor_name": "blk.1.post_attention_norm.weight"},
        ),
        input_reference_loader=lambda *_args: _input_refs(capture),
        iteration=335,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer1_post_attn_residual_matches_oracle_exactly"
    assert artifact["target_position"] == 2
    assert artifact["input_classification"] == "layer1_post_attn_inputs_match_prior_artifacts"
    assert artifact["next_action"] == "audit_layer1_moe_router_from_post_norm"
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer1_post_attn_residual_oracle_classifies_mismatch(tmp_path: Path) -> None:
    conv_path = tmp_path / "conv.json"
    conv_path.write_text(json.dumps(_conv_artifact()))
    capture = _capture(position=2, token_id=12)
    capture["fields"]["post_norm_f32"] = capture["fields"]["post_norm_f32"].copy()
    capture["fields"]["post_norm_f32"][0] += np.float32(0.5)

    artifact = audit_layer1_post_attn_residual_oracle(
        conv_gdn_artifact_path=conv_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        norm_weight_loader=lambda *_args: (
            np.ones((4,), dtype=np.float32),
            1.0e-6,
            {"tensor_name": "blk.1.post_attention_norm.weight"},
        ),
        input_reference_loader=lambda *_args: _input_refs(capture),
        iteration=335,
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer1_post_attn_residual_mismatch_after_oracle"
    assert artifact["oracle_results"]["post_norm_f32"]["classification"] == (
        "post_attn_field_mismatch_after_oracle"
    )
    assert artifact["next_action"] == "inspect_layer1_post_attn_residual_or_norm_kernel"


def test_load_input_references_follows_nested_artifacts(tmp_path: Path) -> None:
    hidden_sha = "hidden-sha"
    attn_sha = "attn-sha"
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "kind": "layer1_bf16_handoff_audit",
                "classification": "layer1_hidden_in_matches_layer0_layer_out_exactly",
                "target_capture": {"fields": {"hidden_in_f32": {"sha256": hidden_sha}}},
            }
        )
    )
    attn_path = tmp_path / "attn.json"
    attn_path.write_text(json.dumps({"layer1_handoff_path": str(handoff_path)}))
    projection_path = tmp_path / "projection.json"
    projection_path.write_text(json.dumps({"layer1_attn_norm_path": str(attn_path)}))
    conv = _conv_artifact()
    conv["layer1_projection_path"] = str(projection_path)
    conv["oracle_results"]["attn_out_f32"]["hipengine_summary"]["sha256"] = attn_sha
    conv_path = tmp_path / "conv.json"
    conv_path.write_text(json.dumps(conv))

    refs = load_input_references(conv_path, conv)

    assert refs["hidden_in_f32"]["expected_sha256"] == hidden_sha
    assert refs["attn_out_f32"]["expected_sha256"] == attn_sha


def _conv_artifact() -> dict:
    return {
        "status": "ready",
        "classification": "layer1_warm_conv_gdn_matches_oracle_within_tolerance",
        "model": "/tmp/model.gguf",
        "layer_id": 1,
        "target_position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "layer1_projection_path": "/tmp/projection.json",
        "oracle_results": {
            "attn_out_f32": {
                "classification": "warm_field_matches_oracle_within_tolerance",
                "hipengine_summary": {"sha256": "placeholder"},
            }
        },
        "next_action": "audit_layer1_post_attn_residual_or_moe_boundary",
    }


def _capture(*, position: int = 0, token_id: int = 1):
    hidden = bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, -0.25], dtype=np.float32))
    attn = bf16_roundtrip_array(np.array([0.25, 0.5, -0.75, 2.0], dtype=np.float32))
    residual, post_norm = add_rmsnorm_bf16_oracle(
        hidden,
        attn,
        np.ones((4,), dtype=np.float32),
        eps=1.0e-6,
        threads=4,
    )
    return {
        "status": "captured",
        "summary": {
            "layer_id": 1,
            "position": int(position),
            "token_id": int(token_id),
            "is_moe": True,
            "preceding_layer_count": 1,
        },
        "fields": {
            "hidden_in_f32": hidden,
            "attn_out_f32": attn,
            "residual_f32": residual,
            "post_norm_f32": post_norm,
            "ffn_or_moe_down_f32": np.zeros((8,), dtype=np.float32),
            "layer_out_f32": np.zeros((4,), dtype=np.float32),
            "moe_routing_weights_f32": np.array([1.0, 0.0], dtype=np.float32),
            "moe_shared_gate_f32": np.array([0.5], dtype=np.float32),
            "moe_selected_experts_i64": np.array([0, 1], dtype=np.int64),
        },
    }


def _input_refs(capture: dict) -> dict[str, dict[str, object]]:
    return {
        "hidden_in_f32": {
            "source": "/tmp/handoff.json",
            "source_classification": "layer1_hidden_in_matches_layer0_layer_out_exactly",
            "expected_sha256": sha256(capture["fields"]["hidden_in_f32"]),
        },
        "attn_out_f32": {
            "source": "/tmp/conv.json",
            "source_classification": "layer1_warm_conv_gdn_matches_oracle_within_tolerance",
            "expected_sha256": sha256(capture["fields"]["attn_out_f32"]),
        },
    }


def sha256(values: np.ndarray) -> str:
    from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import sha256_float32

    return sha256_float32(np.asarray(values, dtype=np.float32))
