from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (
    bf16_roundtrip_array,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer3_full_attention_context_gate_oracle import (
    audit_layer3_full_attention_context_gate_oracle,
    build_context_result,
    build_gate_result,
    classify_context_gate,
    classify_preflight,
    compare_preflight,
    full_attention_context_cpu,
    sigmoid_f32,
    validate_qk_norm_rotary_kv_artifact,
)


def test_validate_qk_norm_rotary_kv_artifact_accepts_ready_source() -> None:
    validate_qk_norm_rotary_kv_artifact(_source_artifact(), expected_layer_id=3)


def test_validate_qk_norm_rotary_kv_artifact_rejects_wrong_next_action() -> None:
    artifact = _source_artifact()
    artifact["next_action"] = "audit_something_else"

    with pytest.raises(ValueError, match="context audit"):
        validate_qk_norm_rotary_kv_artifact(artifact, expected_layer_id=3)


def test_compare_preflight_accepts_expected_hashes() -> None:
    capture = _capture()
    result = compare_preflight(capture=capture, artifact=_source_artifact())

    assert all(item["exact_hash_match"] for item in result.values())
    assert classify_preflight(result, capture=capture, layer_id=3) == (
        "layer3_context_gate_preflight_matches_qk_artifact"
    )


def test_compare_preflight_blocks_query_hash_mismatch() -> None:
    capture = _capture()
    capture["fields"]["full_query_f32"] = capture["fields"]["full_query_f32"].copy()
    capture["fields"]["full_query_f32"][0] += np.float32(0.25)

    result = compare_preflight(capture=capture, artifact=_source_artifact())

    assert result["full_query_f32"]["exact_hash_match"] is False
    assert classify_preflight(result, capture=capture, layer_id=3) == (
        "layer3_context_gate_blocked_qk_input_mismatch"
    )


def test_full_attention_context_cpu_returns_expected_shape() -> None:
    fields = _base_fields()
    context = full_attention_context_cpu(
        fields["full_query_f32"],
        fields["key_cache_context_f32"],
        fields["value_cache_context_f32"],
        context_len=3,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        scale=np.float32(4 ** -0.5),
    )

    assert context.shape == (8,)
    assert np.all(np.isfinite(context))


def test_build_context_and_gate_results_exact() -> None:
    capture = _capture()

    context = build_context_result(capture=capture, near_atol=1.0e-6)
    gate = build_gate_result(capture=capture)

    assert context["classification"] == "full_attention_context_gate_matches_exactly"
    assert gate["classification"] == "full_attention_context_gate_matches_exactly"


def test_classify_context_gate_combines_exact_and_near() -> None:
    exact = "full_attention_context_gate_matches_exactly"
    near = "full_attention_context_matches_within_fp32_kernel_tolerance"

    assert classify_context_gate(
        "layer3_context_gate_preflight_matches_qk_artifact",
        {"classification": exact},
        {"classification": exact},
    ) == "layer3_full_attention_context_gate_matches_cpu_oracle_exactly"
    assert classify_context_gate(
        "layer3_context_gate_preflight_matches_qk_artifact",
        {"classification": near},
        {"classification": exact},
    ) == "layer3_full_attention_context_matches_cpu_oracle_within_fp32_tolerance"


def test_audit_layer3_full_attention_context_gate_oracle_exact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    artifact = audit_layer3_full_attention_context_gate_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer3_full_attention_context_gate_matches_cpu_oracle_exactly"
    )
    assert artifact["preflight_classification"] == (
        "layer3_context_gate_preflight_matches_qk_artifact"
    )
    assert artifact["context_result"]["classification"] == (
        "full_attention_context_gate_matches_exactly"
    )
    assert artifact["gate_result"]["classification"] == (
        "full_attention_context_gate_matches_exactly"
    )
    assert artifact["next_action"] == "audit_layer3_attn_output_projection_under_bf16_contract"
    json.dumps(artifact)


def test_audit_layer3_full_attention_context_gate_oracle_allows_near_context(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, context_offset=5.0e-6, context_near_atol=1.0e-4)

    artifact = audit_layer3_full_attention_context_gate_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer3_full_attention_context_matches_cpu_oracle_within_fp32_tolerance"
    )
    assert artifact["context_result"]["classification"] == (
        "full_attention_context_matches_within_fp32_kernel_tolerance"
    )


def test_audit_layer3_full_attention_context_gate_oracle_classifies_gate_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, gate_offset=0.25)

    artifact = audit_layer3_full_attention_context_gate_oracle(**fixture)

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer3_full_attention_context_gate_mismatch"
    assert artifact["gate_result"]["classification"] == "full_attention_context_gate_mismatch"


def test_audit_layer3_full_attention_context_gate_oracle_blocks_split_decode(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["boundary_capture_fn"] = lambda *_args: _capture(used_split_decode=True)

    artifact = audit_layer3_full_attention_context_gate_oracle(**fixture)

    assert artifact["status"] == "blocked"
    assert artifact["classification"] == "layer3_context_gate_blocked_split_decode_path"
    assert artifact["next_action"] == "inspect_layer3_full_attention_context_gate_capture_metadata"


def test_audit_layer3_full_attention_context_gate_oracle_reports_unavailable(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(_source_artifact()))

    artifact = audit_layer3_full_attention_context_gate_oracle(
        qk_norm_rotary_kv_artifact_path=source_path,
        model_path=Path("/tmp/model.gguf"),
        boundary_capture_fn=lambda *_args: {"status": "skipped_no_hip_runtime", "fields": {}},
        iteration=349,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == "layer3_full_attention_context_gate_oracle_unavailable"
    assert artifact["next_action"] == (
        "rerun_layer3_full_attention_context_gate_oracle_on_rocm_host"
    )


def _fixture(
    tmp_path: Path,
    *,
    context_offset: float = 0.0,
    gate_offset: float = 0.0,
    context_near_atol: float = 1.0e-5,
) -> dict[str, object]:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(_source_artifact()))
    capture = _capture(context_offset=context_offset, gate_offset=gate_offset)
    return {
        "qk_norm_rotary_kv_artifact_path": source_path,
        "model_path": Path("/tmp/model.gguf"),
        "boundary_capture_fn": lambda *_args: capture,
        "context_near_atol": context_near_atol,
        "iteration": 349,
    }


def _source_artifact() -> dict:
    fields = _base_fields()
    return {
        "status": "ready",
        "classification": "layer3_qk_norm_rotary_matches_cpu_oracle_within_fp32_tolerance",
        "model": "/tmp/model.gguf",
        "layer_id": 3,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "hipengine_capture": {
            "field_summaries": {
                name: summarize_array(fields[name])
                for name in (
                    "full_query_f32",
                    "full_key_f32",
                    "full_gate_f32",
                    "key_cache_position_f32",
                    "value_cache_position_f32",
                )
            }
        },
        "next_action": "audit_layer3_full_attention_scores_or_attn_output_under_bf16_contract",
    }


def _capture(
    *,
    layer_type: str = "full_attention",
    used_split_decode: bool = False,
    context_offset: float = 0.0,
    gate_offset: float = 0.0,
) -> dict:
    fields = _base_fields()
    fields["full_attn_context_f32"] = fields["full_attn_context_f32"].copy()
    fields["full_attn_context_f32"][0] += np.float32(context_offset)
    fields["full_gated_f32"] = fields["full_gated_f32"].copy()
    fields["full_gated_f32"][0] += np.float32(gate_offset)
    return {
        "status": "captured",
        "summary": {
            "layer_id": 3,
            "position": 2,
            "token_id": 12,
            "hidden_size": 4,
            "layer_type": layer_type,
            "preceding_layer_count": 3,
            "q_width": 8,
            "kv_width": 4,
            "num_q_heads": 2,
            "num_kv_heads": 1,
            "head_dim": 4,
            "active_context": 3,
            "block_size": 256,
            "used_split_decode": used_split_decode,
            "finite": True,
        },
        "fields": fields,
    }


def _base_fields() -> dict[str, np.ndarray]:
    query = np.array([0.5, -0.25, 0.75, 1.0, -0.5, 0.25, 1.25, -0.75], dtype=np.float32)
    gate = bf16_roundtrip_array(
        np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], dtype=np.float32)
    )
    key_cache = bf16_roundtrip_array(
        np.array(
            [
                0.25,
                -0.5,
                0.75,
                1.0,
                -0.75,
                0.25,
                0.5,
                -1.0,
                1.25,
                -0.25,
                -0.5,
                0.75,
            ],
            dtype=np.float32,
        )
    )
    value_cache = bf16_roundtrip_array(
        np.array(
            [
                -0.25,
                0.5,
                -0.75,
                1.0,
                0.75,
                -0.5,
                0.25,
                -1.25,
                -1.0,
                0.25,
                0.5,
                -0.75,
            ],
            dtype=np.float32,
        )
    )
    context = full_attention_context_cpu(
        query,
        key_cache,
        value_cache,
        context_len=3,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        scale=np.float32(4 ** -0.5),
    )
    gated = bf16_roundtrip_array(context * sigmoid_f32(gate))
    return {
        "full_query_f32": query,
        "full_key_f32": key_cache[-4:].copy(),
        "full_gate_f32": gate,
        "key_cache_position_f32": key_cache[-4:].copy(),
        "value_cache_position_f32": value_cache[-4:].copy(),
        "key_cache_context_f32": key_cache,
        "value_cache_context_f32": value_cache,
        "full_attn_context_f32": context,
        "full_gated_f32": gated,
        "attn_out_f32": bf16_roundtrip_array(np.zeros((4,), dtype=np.float32)),
    }
