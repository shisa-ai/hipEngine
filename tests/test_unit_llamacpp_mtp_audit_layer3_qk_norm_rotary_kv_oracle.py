from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (
    bf16_roundtrip_array,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer3_qk_norm_rotary_kv_oracle import (
    audit_layer3_qk_norm_rotary_kv_oracle,
    build_cache_results,
    build_split_results,
    classify_layer3_qk_norm_rotary_kv,
    classify_qkv_preflight,
    compare_qkv_preflight,
    head_rmsnorm_partial_rotary,
    split_full_q_projection,
    validate_layer3_qkv_artifact,
    _rope_tables,
)


def test_validate_layer3_qkv_artifact_accepts_exact_full_attention() -> None:
    validate_layer3_qkv_artifact(_qkv_artifact(), expected_layer_id=3)


def test_validate_layer3_qkv_artifact_rejects_non_exact_classification() -> None:
    artifact = _qkv_artifact()
    artifact["classification"] = "layer3_full_attention_qkv_mismatch_after_bf16_oracle"

    with pytest.raises(ValueError, match="exact"):
        validate_layer3_qkv_artifact(artifact, expected_layer_id=3)


def test_compare_qkv_preflight_accepts_expected_hashes() -> None:
    capture = _capture()
    artifact = _qkv_artifact()

    result = compare_qkv_preflight(capture=capture, artifact=artifact)

    assert all(item["exact_hash_match"] for item in result.values())
    assert classify_qkv_preflight(result, capture=capture, layer_id=3) == (
        "layer3_qk_norm_rotary_preflight_matches_qkv_artifact"
    )


def test_compare_qkv_preflight_blocks_input_hash_mismatch() -> None:
    capture = _capture()
    capture["fields"]["full_q_f32"] = capture["fields"]["full_q_f32"].copy()
    capture["fields"]["full_q_f32"][0] += np.float32(0.25)

    result = compare_qkv_preflight(capture=capture, artifact=_qkv_artifact())

    assert result["full_q_f32"]["exact_hash_match"] is False
    assert classify_qkv_preflight(result, capture=capture, layer_id=3) == (
        "layer3_qk_norm_rotary_blocked_qkv_input_mismatch"
    )


def test_split_full_q_projection_matches_qwen_layout() -> None:
    query, gate = split_full_q_projection(_full_q(), num_q_heads=2, head_dim=4)

    np.testing.assert_array_equal(
        query,
        np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 4.0, -0.5], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        gate,
        np.array([0.25, -0.75, 1.5, -1.25, 0.5, 0.75, -0.25, 1.25], dtype=np.float32),
    )


def test_build_split_results_requires_exact_split_and_key_raw() -> None:
    results = build_split_results(capture=_capture())

    assert set(results) == {"full_query_raw_f32", "full_gate_f32", "full_key_raw_f32"}
    assert all(
        item["classification"] == "qk_norm_rotary_kv_matches_exactly"
        for item in results.values()
    )


def test_head_rmsnorm_partial_rotary_returns_expected_no_rotation_for_tail() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    weight = np.ones((4,), dtype=np.float32)
    cos = np.ones((4,), dtype=np.float32)
    sin = np.zeros((4,), dtype=np.float32)

    out = head_rmsnorm_partial_rotary(
        values,
        weight,
        cos,
        sin,
        num_heads=1,
        head_dim=4,
        rotary_dim=2,
        eps=0.0,
    )

    scale = np.float32(1.0 / np.sqrt(np.mean(values * values, dtype=np.float32)))
    np.testing.assert_allclose(out, values * scale, rtol=0.0, atol=1.0e-7)


def test_build_cache_results_requires_exact_kv_write_from_hip_inputs() -> None:
    capture = _capture()
    results = build_cache_results(capture=capture, rotary_results={})

    assert results["key_cache_position_f32"]["classification"] == (
        "qk_norm_rotary_kv_matches_exactly"
    )
    assert results["value_cache_position_f32"]["classification"] == (
        "qk_norm_rotary_kv_matches_exactly"
    )


def test_classify_combines_exact_and_near_rotary_results() -> None:
    exact = "qk_norm_rotary_kv_matches_exactly"
    near = "qk_norm_rotary_matches_within_fp32_kernel_tolerance"
    split = {name: {"classification": exact} for name in ("a", "b")}
    cache = {"cache": {"classification": exact}}
    rotary_exact = {"q": {"classification": exact}, "k": {"classification": exact}}
    rotary_near = {"q": {"classification": exact}, "k": {"classification": near}}

    assert classify_layer3_qk_norm_rotary_kv(
        "layer3_qk_norm_rotary_preflight_matches_qkv_artifact",
        split,
        rotary_exact,
        cache,
    ) == "layer3_qk_norm_rotary_kv_matches_cpu_oracle_exactly"
    assert classify_layer3_qk_norm_rotary_kv(
        "layer3_qk_norm_rotary_preflight_matches_qkv_artifact",
        split,
        rotary_near,
        cache,
    ) == "layer3_qk_norm_rotary_matches_cpu_oracle_within_fp32_tolerance"


def test_audit_layer3_qk_norm_rotary_kv_oracle_with_injected_exact_capture(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    artifact = audit_layer3_qk_norm_rotary_kv_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer3_qk_norm_rotary_kv_matches_cpu_oracle_exactly"
    assert artifact["preflight_classification"] == (
        "layer3_qk_norm_rotary_preflight_matches_qkv_artifact"
    )
    for section in ("split_results", "rotary_results", "cache_results"):
        assert all(
            item["classification"] == "qk_norm_rotary_kv_matches_exactly"
            for item in artifact[section].values()
        )
    assert artifact["next_action"] == (
        "audit_layer3_full_attention_scores_or_attn_output_under_bf16_contract"
    )
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer3_qk_norm_rotary_kv_oracle_allows_tiny_rotary_delta(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, query_rotary_offset=5.0e-6, rotary_near_atol=1.0e-4)

    artifact = audit_layer3_qk_norm_rotary_kv_oracle(**fixture)

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer3_qk_norm_rotary_matches_cpu_oracle_within_fp32_tolerance"
    )
    assert artifact["rotary_results"]["full_query_f32"]["classification"] == (
        "qk_norm_rotary_matches_within_fp32_kernel_tolerance"
    )


def test_audit_layer3_qk_norm_rotary_kv_oracle_classifies_cache_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, value_cache_offset=0.25)

    artifact = audit_layer3_qk_norm_rotary_kv_oracle(**fixture)

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer3_qk_norm_rotary_kv_mismatch"
    assert artifact["cache_results"]["value_cache_position_f32"]["classification"] == (
        "qk_norm_rotary_kv_mismatch"
    )


def test_audit_layer3_qk_norm_rotary_kv_oracle_blocks_wrong_layer_type(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["boundary_capture_fn"] = lambda *_args: _capture(layer_type="linear_attention")

    artifact = audit_layer3_qk_norm_rotary_kv_oracle(**fixture)

    assert artifact["status"] == "blocked"
    assert artifact["classification"] == "layer3_qk_norm_rotary_wrong_layer_type"
    assert artifact["next_action"] == "inspect_layer3_qk_norm_rotary_capture_metadata"


def test_audit_layer3_qk_norm_rotary_kv_oracle_reports_unavailable_capture(
    tmp_path: Path,
) -> None:
    qkv_path = tmp_path / "qkv.json"
    qkv_path.write_text(json.dumps(_qkv_artifact()))

    artifact = audit_layer3_qk_norm_rotary_kv_oracle(
        qkv_artifact_path=qkv_path,
        model_path=Path("/tmp/model.gguf"),
        boundary_capture_fn=lambda *_args: {"status": "skipped_no_hip_runtime", "fields": {}},
        weight_config_loader=lambda *_args: _weights_config(),
        iteration=348,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == "layer3_qk_norm_rotary_kv_oracle_unavailable"
    assert artifact["next_action"] == "rerun_layer3_qk_norm_rotary_kv_oracle_on_rocm_host"


def _fixture(
    tmp_path: Path,
    *,
    query_rotary_offset: float = 0.0,
    value_cache_offset: float = 0.0,
    rotary_near_atol: float = 1.0e-5,
) -> dict[str, object]:
    qkv_path = tmp_path / "qkv.json"
    qkv_path.write_text(json.dumps(_qkv_artifact()))
    capture = _capture(
        query_rotary_offset=query_rotary_offset,
        value_cache_offset=value_cache_offset,
    )
    return {
        "qkv_artifact_path": qkv_path,
        "model_path": Path("/tmp/model.gguf"),
        "boundary_capture_fn": lambda *_args: capture,
        "weight_config_loader": lambda *_args: _weights_config(),
        "rotary_near_atol": rotary_near_atol,
        "iteration": 348,
    }


def _qkv_artifact() -> dict:
    fields = _base_fields()
    return {
        "status": "ready",
        "classification": "layer3_full_attention_qkv_matches_bf16_oracle_exactly",
        "model": "/tmp/model.gguf",
        "layer_id": 3,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "hipengine_capture": {
            "summary": {
                "layer_id": 3,
                "layer_type": "full_attention",
                "preceding_layer_count": 3,
            },
            "field_summaries": {
                name: summarize_array(values)
                for name, values in fields.items()
                if name in {"full_q_f32", "full_k_f32", "full_v_f32"}
            },
        },
        "projection_results": {
            name: {"classification": "full_attention_qkv_matches_bf16_oracle_exactly"}
            for name in ("full_q_f32", "full_k_f32", "full_v_f32")
        },
        "next_action": "audit_layer3_qk_norm_rotary_or_kv_write_under_bf16_contract",
    }


def _capture(
    *,
    layer_type: str = "full_attention",
    query_rotary_offset: float = 0.0,
    value_cache_offset: float = 0.0,
) -> dict:
    fields = _base_fields()
    fields["full_query_f32"] = fields["full_query_f32"].copy()
    fields["full_query_f32"][0] += np.float32(query_rotary_offset)
    fields["value_cache_position_f32"] = fields["value_cache_position_f32"].copy()
    fields["value_cache_position_f32"][0] += np.float32(value_cache_offset)
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
            "value_head_dim": 4,
            "rotary_dim": 4,
            "rms_norm_eps": 1.0e-6,
            "rope_freq_base": 10000.0,
            "block_size": 256,
            "max_positions": 32,
            "finite": True,
        },
        "fields": fields,
    }


def _base_fields() -> dict[str, np.ndarray]:
    full_q = _full_q()
    full_k = _full_k()
    full_v = _full_v()
    query_raw, gate = split_full_q_projection(full_q, num_q_heads=2, head_dim=4)
    cos, sin = _rope_tables(max_positions=32, rotary_dim=4, base=10000.0)
    weights = _weights_config()
    full_query = head_rmsnorm_partial_rotary(
        query_raw,
        weights["q_norm"][0],
        cos[2],
        sin[2],
        num_heads=2,
        head_dim=4,
        rotary_dim=4,
        eps=1.0e-6,
    )
    full_key = head_rmsnorm_partial_rotary(
        full_k,
        weights["k_norm"][0],
        cos[2],
        sin[2],
        num_heads=1,
        head_dim=4,
        rotary_dim=4,
        eps=1.0e-6,
    )
    return {
        "full_q_f32": full_q,
        "full_k_f32": full_k,
        "full_v_f32": full_v,
        "full_query_raw_f32": query_raw,
        "full_gate_f32": gate,
        "full_key_raw_f32": full_k,
        "full_query_f32": full_query,
        "full_key_f32": full_key,
        "key_cache_position_f32": bf16_roundtrip_array(full_key),
        "value_cache_position_f32": full_v.copy(),
    }


def _full_q() -> np.ndarray:
    return bf16_roundtrip_array(
        np.array(
            [
                1.0,
                -2.0,
                0.5,
                3.0,
                0.25,
                -0.75,
                1.5,
                -1.25,
                -1.0,
                2.0,
                4.0,
                -0.5,
                0.5,
                0.75,
                -0.25,
                1.25,
            ],
            dtype=np.float32,
        )
    )


def _full_k() -> np.ndarray:
    return bf16_roundtrip_array(np.array([0.25, -1.0, 2.0, -0.5], dtype=np.float32))


def _full_v() -> np.ndarray:
    return bf16_roundtrip_array(np.array([-0.5, 0.75, -1.25, 1.5], dtype=np.float32))


def _weights_config() -> dict[str, object]:
    q_norm = np.array([1.0, 0.75, 1.25, 0.5], dtype=np.float32)
    k_norm = np.array([0.5, 1.5, 0.25, 1.0], dtype=np.float32)
    return {
        "q_norm": (q_norm, {"tensor_name": "blk.3.attn_q_norm.weight"}),
        "k_norm": (k_norm, {"tensor_name": "blk.3.attn_k_norm.weight"}),
        "config": {
            "num_q_heads": 2,
            "num_kv_heads": 1,
            "head_dim": 4,
            "value_head_dim": 4,
            "rotary_dim": 4,
            "rms_norm_eps": 1.0e-6,
            "rope_freq_base": 10000.0,
            "max_positions": 32,
            "layer_type": "full_attention",
        },
    }
