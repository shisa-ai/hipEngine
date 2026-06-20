from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (
    bf16_roundtrip_array,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_post_attn_residual_oracle import (
    add_rmsnorm_bf16_oracle,
)
from scripts.llamacpp_mtp_audit_layer1_post_attn_residual_oracle import (
    compare_input_hashes,
)
from scripts.llamacpp_mtp_audit_layer5_post_attn_residual_oracle import (
    audit_layer5_post_attn_residual_oracle,
    classify_capture_metadata,
    classify_layer5_inputs,
    load_layer5_input_references,
    validate_layer5_conv_gdn,
    validate_layer5_handoff,
    validate_source_alignment,
)


def test_validate_source_artifacts_accept_ready_inputs() -> None:
    validate_layer5_conv_gdn(_conv_gdn_artifact(), expected_layer_id=5)
    validate_layer5_handoff(_handoff_artifact(), expected_layer_id=5)
    validate_source_alignment(conv_gdn=_conv_gdn_artifact(), handoff=_handoff_artifact())


def test_validate_layer5_conv_gdn_rejects_wrong_next_action() -> None:
    artifact = _conv_gdn_artifact()
    artifact["next_action"] = "audit_something_else"

    with pytest.raises(ValueError, match="post-attn audit"):
        validate_layer5_conv_gdn(artifact, expected_layer_id=5)


def test_validate_layer5_conv_gdn_rejects_nonmatching_inputs() -> None:
    artifact = _conv_gdn_artifact()
    artifact["target_input_classification"] = "target_inputs_mismatch_before_replay"

    with pytest.raises(ValueError, match="target inputs"):
        validate_layer5_conv_gdn(artifact, expected_layer_id=5)


def test_validate_layer5_handoff_rejects_non_exact_classification() -> None:
    artifact = _handoff_artifact()
    artifact["classification"] = "layer5_hidden_in_mismatch"

    with pytest.raises(ValueError, match="must be exact"):
        validate_layer5_handoff(artifact, expected_layer_id=5)


def test_validate_source_alignment_rejects_position_mismatch() -> None:
    handoff = _handoff_artifact()
    handoff["position"] = 1

    with pytest.raises(ValueError, match="position mismatch"):
        validate_source_alignment(conv_gdn=_conv_gdn_artifact(), handoff=handoff)


def test_compare_input_hashes_accepts_prior_artifacts(tmp_path: Path) -> None:
    refs = _input_references(tmp_path)
    result = compare_input_hashes(capture=_capture(), input_references=refs)

    assert classify_layer5_inputs(result) == "layer5_post_attn_inputs_match_prior_artifacts"
    assert result["hidden_in_f32"]["exact_hash_match"] is True
    assert result["attn_out_f32"]["exact_hash_match"] is True


def test_compare_input_hashes_blocks_attn_out_mismatch(tmp_path: Path) -> None:
    refs = _input_references(tmp_path)
    capture = _capture()
    capture["fields"]["attn_out_f32"] = capture["fields"]["attn_out_f32"].copy()
    capture["fields"]["attn_out_f32"][0] += np.float32(0.25)

    result = compare_input_hashes(capture=capture, input_references=refs)

    assert classify_layer5_inputs(result) == "layer5_post_attn_inputs_mismatch_before_residual"


def test_classify_capture_metadata_blocks_wrong_layer_type() -> None:
    assert classify_capture_metadata(_capture(layer_type="full_attention"), layer_id=5) == (
        "layer5_post_attn_wrong_layer_type"
    )


def test_audit_layer5_post_attn_residual_oracle_exact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    artifact = audit_layer5_post_attn_residual_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer5_post_attn_residual_matches_oracle_exactly"
    assert artifact["input_classification"] == "layer5_post_attn_inputs_match_prior_artifacts"
    assert artifact["oracle_results"]["residual_f32"]["classification"] == (
        "post_attn_field_matches_oracle_exactly"
    )
    assert artifact["oracle_results"]["post_norm_f32"]["classification"] == (
        "post_attn_field_matches_oracle_exactly"
    )
    assert artifact["next_action"] == "audit_layer5_moe_router_from_post_norm"
    json.dumps(artifact)


def test_audit_layer5_post_attn_residual_oracle_allows_near_post_norm(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, post_norm_offset=5.0e-5, post_norm_atol=1.0e-4)

    artifact = audit_layer5_post_attn_residual_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer5_post_attn_residual_matches_oracle_within_tolerance"
    )
    assert artifact["oracle_results"]["post_norm_f32"]["classification"] == (
        "post_attn_field_matches_oracle_within_tolerance"
    )


def test_audit_layer5_post_attn_residual_oracle_classifies_residual_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, residual_offset=0.5)

    artifact = audit_layer5_post_attn_residual_oracle(**fixture)

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer5_post_attn_residual_mismatch_after_oracle"
    assert artifact["next_action"] == "inspect_layer5_post_attn_residual_or_norm_kernel"


def test_audit_layer5_post_attn_residual_oracle_blocks_bad_metadata(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["layer_capture_fn"] = lambda *_args: _capture(layer_type="full_attention")

    artifact = audit_layer5_post_attn_residual_oracle(**fixture)

    assert artifact["status"] == "blocked"
    assert artifact["classification"] == "layer5_post_attn_wrong_layer_type"
    assert artifact["next_action"] == "inspect_layer5_post_attn_capture_metadata"


def test_audit_layer5_post_attn_residual_oracle_reports_unavailable(tmp_path: Path) -> None:
    conv_path, handoff_path = _write_sources(tmp_path)

    artifact = audit_layer5_post_attn_residual_oracle(
        conv_gdn_artifact_path=conv_path,
        handoff_artifact_path=handoff_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: {"status": "skipped_no_hip_runtime"},
        iteration=365,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == "layer5_post_attn_residual_oracle_unavailable"
    assert artifact["next_action"] == "rerun_layer5_post_attn_residual_oracle_on_rocm_host"


def _fixture(
    tmp_path: Path,
    *,
    residual_offset: float = 0.0,
    post_norm_offset: float = 0.0,
    post_norm_atol: float = 2.5e-4,
) -> dict[str, object]:
    conv_path, handoff_path = _write_sources(tmp_path)
    capture = _capture(residual_offset=residual_offset, post_norm_offset=post_norm_offset)
    return {
        "conv_gdn_artifact_path": conv_path,
        "handoff_artifact_path": handoff_path,
        "model_path": Path("/tmp/model.gguf"),
        "layer_capture_fn": lambda *_args: capture,
        "norm_weight_loader": lambda *_args: (_weight(), 1.0e-6, _weight_metadata()),
        "residual_atol": 0.0,
        "post_norm_atol": post_norm_atol,
        "iteration": 365,
    }


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    conv_path = tmp_path / "conv_gdn.json"
    handoff_path = tmp_path / "handoff.json"
    conv_path.write_text(json.dumps(_conv_gdn_artifact()))
    handoff_path.write_text(json.dumps(_handoff_artifact()))
    return conv_path, handoff_path


def _input_references(tmp_path: Path) -> dict[str, object]:
    conv_path, handoff_path = _write_sources(tmp_path)
    return load_layer5_input_references(
        conv_path,
        _conv_gdn_artifact(),
        handoff_path,
        _handoff_artifact(),
    )


def _conv_gdn_artifact() -> dict:
    fields = _base_fields()
    return {
        "status": "ready",
        "classification": "layer5_warm_conv_gdn_matches_oracle_within_tolerance",
        "kind": "layer5_warm_bf16_contracted_conv_gdn_oracle",
        "model": "/tmp/model.gguf",
        "layer_id": 5,
        "target_position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "target_input_classification": "target_inputs_match_replay_exactly",
        "oracle_results": {
            "attn_out_f32": {
                "classification": "warm_field_matches_oracle_within_tolerance",
                "hipengine_summary": summarize_array(fields["attn_out_f32"]),
            }
        },
        "next_action": "audit_layer5_post_attn_residual_or_moe_boundary",
    }


def _handoff_artifact() -> dict:
    fields = _base_fields()
    return {
        "status": "ready",
        "classification": "layer5_hidden_in_matches_layer4_layer_out_exactly",
        "kind": "layer5_bf16_handoff_audit",
        "model": "/tmp/model.gguf",
        "source_layer": 4,
        "target_layer": 5,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "target_capture": {
            "fields": {
                "hidden_in_f32": summarize_array(fields["hidden_in_f32"]),
            }
        },
    }


def _capture(
    *,
    layer_type: str = "linear_attention",
    residual_offset: float = 0.0,
    post_norm_offset: float = 0.0,
) -> dict:
    fields = _base_fields()
    fields["residual_f32"] = fields["residual_f32"].copy()
    fields["residual_f32"][0] += np.float32(residual_offset)
    fields["post_norm_f32"] = fields["post_norm_f32"].copy()
    fields["post_norm_f32"][0] += np.float32(post_norm_offset)
    return {
        "status": "captured",
        "summary": {
            "layer_id": 5,
            "position": 2,
            "token_id": 12,
            "hidden_size": 4,
            "layer_type": layer_type,
            "preceding_layer_count": 5,
            "finite": True,
        },
        "fields": fields,
    }


def _base_fields() -> dict[str, np.ndarray]:
    hidden = bf16_roundtrip_array(np.array([0.5, -0.25, 1.0, -0.75], dtype=np.float32))
    attn_out = bf16_roundtrip_array(np.array([0.125, 0.5, -0.25, 0.75], dtype=np.float32))
    residual, post_norm = add_rmsnorm_bf16_oracle(hidden, attn_out, _weight(), eps=1.0e-6)
    return {
        "hidden_in_f32": hidden,
        "attn_out_f32": attn_out,
        "residual_f32": residual,
        "post_norm_f32": post_norm,
        "ffn_or_moe_down_f32": bf16_roundtrip_array(np.zeros((4,), dtype=np.float32)),
        "layer_out_f32": bf16_roundtrip_array(np.zeros((4,), dtype=np.float32)),
    }


def _weight() -> np.ndarray:
    return np.array([1.0, 0.5, -0.25, 0.75], dtype=np.float32)


def _weight_metadata() -> dict[str, object]:
    values = _weight()
    return {
        "tensor_name": "blk.5.post_attention_norm.weight",
        "ggml_type": "F32",
        "shape": list(values.shape),
        "summary": summarize_array(values),
    }
