from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (
    bf16_roundtrip_array,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer11_attn_output_oracle import (
    audit_layer11_attn_output_oracle,
    build_attn_output_result,
    classify_attn_output,
    classify_preflight,
    compare_preflight,
    validate_context_gate_artifact,
)



def test_validate_context_gate_artifact_accepts_ready_source() -> None:
    validate_context_gate_artifact(_source_artifact(), expected_layer_id=11)



def test_validate_context_gate_artifact_rejects_wrong_next_action() -> None:
    artifact = _source_artifact()
    artifact["next_action"] = "audit_something_else"

    with pytest.raises(ValueError, match="attn-output audit"):
        validate_context_gate_artifact(artifact, expected_layer_id=11)



def test_validate_context_gate_artifact_rejects_non_exact_gate() -> None:
    artifact = _source_artifact()
    artifact["gate_result"]["classification"] = "full_attention_context_gate_mismatch"

    with pytest.raises(ValueError, match="exact full_gated input"):
        validate_context_gate_artifact(artifact, expected_layer_id=11)



def test_compare_preflight_accepts_full_gated_hash() -> None:
    capture = _capture()
    result = compare_preflight(capture=capture, artifact=_source_artifact())

    assert result["full_gated_f32"]["exact_hash_match"] is True
    assert classify_preflight(result, capture=capture, layer_id=11) == (
        "layer11_attn_output_preflight_matches_context_gate"
    )



def test_compare_preflight_blocks_full_gated_mismatch() -> None:
    capture = _capture()
    capture["fields"]["full_gated_f32"] = capture["fields"]["full_gated_f32"].copy()
    capture["fields"]["full_gated_f32"][0] += np.float32(0.25)

    result = compare_preflight(capture=capture, artifact=_source_artifact())

    assert result["full_gated_f32"]["exact_hash_match"] is False
    assert classify_preflight(result, capture=capture, layer_id=11) == (
        "layer11_attn_output_blocked_context_gate_input_mismatch"
    )



def test_build_attn_output_result_exact() -> None:
    result = build_attn_output_result(
        capture=_capture(),
        weight=_weight(),
        weight_metadata=_weight_metadata(),
        near_atol=1.0e-6,
    )

    assert result["classification"] == (
        "full_attention_attn_output_matches_bf16_oracle_exactly"
    )
    assert result["delta_bf16_oracle_vs_hip"]["max_abs_diff"] == 0.0



def test_classify_attn_output_combines_exact_and_near() -> None:
    assert classify_attn_output(
        "layer11_attn_output_preflight_matches_context_gate",
        {"classification": "full_attention_attn_output_matches_bf16_oracle_exactly"},
    ) == "layer11_attn_output_projection_matches_bf16_oracle_exactly"
    assert classify_attn_output(
        "layer11_attn_output_preflight_matches_context_gate",
        {
            "classification": (
                "full_attention_attn_output_matches_bf16_oracle_within_rounding"
            )
        },
    ) == "layer11_attn_output_projection_matches_bf16_oracle_within_rounding"



def test_audit_layer11_attn_output_oracle_exact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    artifact = audit_layer11_attn_output_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer11_attn_output_projection_matches_bf16_oracle_exactly"
    )
    assert artifact["preflight_classification"] == (
        "layer11_attn_output_preflight_matches_context_gate"
    )
    assert artifact["projection_result"]["classification"] == (
        "full_attention_attn_output_matches_bf16_oracle_exactly"
    )
    assert artifact["next_action"] == (
        "audit_layer11_post_attn_residual_or_moe_boundary"
    )
    json.dumps(artifact)



def test_audit_layer11_attn_output_oracle_allows_near_projection(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, attn_out_offset=5.0e-5, near_atol=1.0e-4)

    artifact = audit_layer11_attn_output_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer11_attn_output_projection_matches_bf16_oracle_within_rounding"
    )
    assert artifact["projection_result"]["classification"] == (
        "full_attention_attn_output_matches_bf16_oracle_within_rounding"
    )



def test_audit_layer11_attn_output_oracle_classifies_projection_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, attn_out_offset=0.5)

    artifact = audit_layer11_attn_output_oracle(**fixture)

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == (
        "layer11_attn_output_projection_mismatch_after_bf16_oracle"
    )
    assert artifact["next_action"] == (
        "inspect_layer11_attn_output_weight_or_projection_kernel"
    )



def test_audit_layer11_attn_output_oracle_blocks_wrong_layer_type(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["boundary_capture_fn"] = lambda *_args: _capture(
        layer_type="linear_attention"
    )

    artifact = audit_layer11_attn_output_oracle(**fixture)

    assert artifact["status"] == "blocked"
    assert artifact["classification"] == "layer11_attn_output_wrong_layer_type"
    assert artifact["next_action"] == "inspect_layer11_attn_output_capture_metadata"



def test_audit_layer11_attn_output_oracle_reports_unavailable(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(_source_artifact()))

    artifact = audit_layer11_attn_output_oracle(
        context_gate_artifact_path=source_path,
        model_path=Path("/tmp/model.gguf"),
        boundary_capture_fn=lambda *_args: {"status": "skipped_no_hip_runtime"},
        iteration=411,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == (
        "layer11_attn_output_projection_oracle_unavailable"
    )
    assert artifact["next_action"] == "rerun_layer11_attn_output_oracle_on_rocm_host"



def _fixture(
    tmp_path: Path,
    *,
    attn_out_offset: float = 0.0,
    near_atol: float = 2.5e-4,
) -> dict[str, object]:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(_source_artifact()))
    capture = _capture(attn_out_offset=attn_out_offset)
    return {
        "context_gate_artifact_path": source_path,
        "model_path": Path("/tmp/model.gguf"),
        "boundary_capture_fn": lambda *_args: capture,
        "weight_loader": lambda *_args: (_weight(), _weight_metadata()),
        "near_atol": near_atol,
        "iteration": 411,
    }



def _source_artifact() -> dict:
    fields = _base_fields()
    return {
        "status": "ready",
        "classification": (
            "layer11_full_attention_context_matches_cpu_oracle_within_fp32_tolerance"
        ),
        "model": "/tmp/model.gguf",
        "layer_id": 11,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "gate_result": {
            "classification": "full_attention_context_gate_matches_exactly"
        },
        "hipengine_capture": {
            "field_summaries": {
                "full_gated_f32": summarize_array(fields["full_gated_f32"]),
            }
        },
        "next_action": "audit_layer11_attn_output_projection_under_bf16_contract",
    }



def _capture(
    *,
    layer_type: str = "full_attention",
    attn_out_offset: float = 0.0,
) -> dict:
    fields = _base_fields()
    fields["attn_out_f32"] = fields["attn_out_f32"].copy()
    fields["attn_out_f32"][0] += np.float32(attn_out_offset)
    return {
        "status": "captured",
        "summary": {
            "layer_id": 11,
            "position": 2,
            "token_id": 12,
            "hidden_size": 2,
            "layer_type": layer_type,
            "preceding_layer_count": 11,
            "q_width": 3,
            "finite": True,
        },
        "fields": fields,
    }



def _base_fields() -> dict[str, np.ndarray]:
    full_gated = bf16_roundtrip_array(np.array([0.5, -0.25, 1.0], dtype=np.float32))
    attn_out = bf16_roundtrip_array(
        np.asarray(_weight() @ full_gated, dtype=np.float32)
    )
    return {
        "full_gated_f32": full_gated,
        "attn_out_f32": attn_out,
    }



def _weight() -> np.ndarray:
    return np.array(
        [
            [0.25, -0.5, 0.75],
            [-1.0, 0.125, 0.5],
        ],
        dtype=np.float32,
    )



def _weight_metadata() -> dict[str, object]:
    values = _weight()
    return {
        "tensor_name": "blk.11.attn_output.weight",
        "ggml_type": "F32",
        "shape": list(values.shape),
        "summary": summarize_array(values.reshape(-1)),
    }
