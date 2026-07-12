from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (
    bf16_roundtrip_array,
    rmsnorm_f32,
    sha256_float32,
)
from scripts.llamacpp_mtp_audit_layer17_attn_norm_oracle import (
    audit_layer17_attn_norm_oracle,
    classify_input,
    classify_layer17_attn_norm,
    compare_hidden_input,
    validate_layer17_handoff,
)


def test_validate_layer17_handoff_accepts_exact_linear_attention_artifact() -> None:
    validate_layer17_handoff(_handoff(_hidden()), expected_layer_id=17)


def test_validate_layer17_handoff_rejects_wrong_target_type() -> None:
    handoff = _handoff(_hidden())
    handoff["target_capture"]["summary"]["layer_type"] = "full_attention"

    with pytest.raises(ValueError, match="linear_attention"):
        validate_layer17_handoff(handoff, expected_layer_id=17)


def test_validate_layer17_handoff_rejects_nonexact_source_reference() -> None:
    handoff = _handoff(_hidden())
    handoff["source_reference"]["exact_hash_match"] = False

    with pytest.raises(ValueError, match="source reference"):
        validate_layer17_handoff(handoff, expected_layer_id=17)


def test_validate_layer17_handoff_rejects_wrong_next_action() -> None:
    handoff = _handoff(_hidden())
    handoff["next_action"] = "audit_something_else"

    with pytest.raises(ValueError, match="attn_norm audit"):
        validate_layer17_handoff(handoff, expected_layer_id=17)


def test_compare_hidden_input_accepts_expected_reference() -> None:
    hidden = _hidden()
    capture = _capture(hidden, hidden)
    handoff = _handoff(hidden)

    result = compare_hidden_input(capture=capture, handoff=handoff)

    assert result["exact_hash_match"] is True
    assert result["classification"] == (
        "layer17_attn_norm_input_matches_handoff_artifact"
    )
    assert classify_input(result) == (
        "layer17_attn_norm_input_matches_handoff_artifact"
    )


def test_compare_hidden_input_classifies_mismatch() -> None:
    hidden = _hidden()
    capture = _capture(hidden + np.float32(0.25), hidden)
    handoff = _handoff(hidden)

    result = compare_hidden_input(capture=capture, handoff=handoff)

    assert result["exact_hash_match"] is False
    assert result["classification"] == "layer17_attn_norm_input_mismatch_before_norm"
    assert classify_input(result) == "layer17_attn_norm_input_mismatch_before_norm"


def test_classify_layer17_attn_norm_accepts_exact_linear_attention_match() -> None:
    delta = {
        "available": True,
        "shape_match": True,
        "exact_match": True,
        "max_abs_diff": 0.0,
    }
    capture = {
        "summary": {
            "layer_id": 17,
            "layer_type": "linear_attention",
            "preceding_layer_count": 17,
        }
    }

    assert classify_layer17_attn_norm(
        "layer17_attn_norm_input_matches_handoff_artifact",
        delta,
        capture=capture,
        layer_id=17,
        attn_norm_atol=0.0,
    ) == "layer17_attn_norm_matches_bf16_oracle_exactly"


def test_classify_layer17_attn_norm_rejects_wrong_layer_type() -> None:
    delta = {
        "available": True,
        "shape_match": True,
        "exact_match": True,
        "max_abs_diff": 0.0,
    }
    capture = {
        "summary": {
            "layer_id": 17,
            "layer_type": "full_attention",
            "preceding_layer_count": 17,
        }
    }

    assert classify_layer17_attn_norm(
        "layer17_attn_norm_input_matches_handoff_artifact",
        delta,
        capture=capture,
        layer_id=17,
        attn_norm_atol=0.0,
    ) == "layer17_attn_norm_wrong_layer_type"


def test_audit_layer17_attn_norm_oracle_with_injected_exact_capture(
    tmp_path: Path,
) -> None:
    hidden = _hidden()
    weight = np.array([1.0, 0.75, 1.25, 0.5], dtype=np.float32)
    eps = 1.0e-6
    expected = bf16_roundtrip_array(rmsnorm_f32(hidden, weight, eps))
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(_handoff(hidden)))

    artifact = audit_layer17_attn_norm_oracle(
        layer17_handoff_path=handoff_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: _capture(hidden, expected),
        weight_loader=_weight_loader(weight, eps),
        iteration=453,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer17_attn_norm_matches_bf16_oracle_exactly"
    )
    assert artifact["input_classification"] == (
        "layer17_attn_norm_input_matches_handoff_artifact"
    )
    assert artifact["attn_norm_delta"]["exact_match"] is True
    assert artifact["attn_norm_delta"]["max_abs_diff"] == 0.0
    assert artifact["next_action"] == (
        "audit_layer17_projection_or_conv_gdn_under_bf16_contract"
    )
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer17_attn_norm_oracle_classifies_mismatch(tmp_path: Path) -> None:
    hidden = _hidden()
    weight = np.array([1.0, 0.75, 1.25, 0.5], dtype=np.float32)
    actual = bf16_roundtrip_array(rmsnorm_f32(hidden, weight, 1.0e-6))
    actual = actual.copy()
    actual[0] += np.float32(0.25)
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(_handoff(hidden)))

    artifact = audit_layer17_attn_norm_oracle(
        layer17_handoff_path=handoff_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: _capture(hidden, actual),
        weight_loader=_weight_loader(weight, 1.0e-6),
        iteration=453,
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == (
        "layer17_attn_norm_mismatch_after_bf16_oracle"
    )
    assert artifact["attn_norm_delta"]["max_abs_diff"] > 0.0
    assert artifact["next_action"] == (
        "inspect_layer17_attn_norm_weight_or_rmsnorm_semantics"
    )


def test_audit_layer17_attn_norm_oracle_reports_unavailable_capture(
    tmp_path: Path,
) -> None:
    hidden = _hidden()
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(_handoff(hidden)))

    artifact = audit_layer17_attn_norm_oracle(
        layer17_handoff_path=handoff_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: {
            "status": "skipped_no_hip_runtime",
            "fields": {},
        },
        weight_loader=_weight_loader(np.ones((4,), dtype=np.float32), 1.0e-6),
        iteration=453,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == "layer17_attn_norm_oracle_unavailable"
    assert artifact["next_action"] == "rerun_layer17_attn_norm_oracle_on_rocm_host"


def _hidden() -> np.ndarray:
    return bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))


def _handoff(hidden: np.ndarray) -> dict:
    return {
        "status": "ready",
        "classification": "layer17_hidden_in_matches_layer16_layer_out_exactly",
        "model": "/tmp/model.gguf",
        "target_layer": 17,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "source_reference": {"exact_hash_match": True},
        "target_capture": {
            "summary": {
                "layer_type": "linear_attention",
                "preceding_layer_count": 17,
            },
            "fields": {"hidden_in_f32": {"sha256": sha256_float32(hidden)}},
        },
        "handoff_delta": {"exact_match": True, "max_abs_diff": 0.0},
        "next_action": "audit_layer17_attn_norm_under_bf16_contract_or_mtp_boundary",
    }


def _capture(hidden: np.ndarray, attn_norm: np.ndarray):
    return {
        "status": "captured",
        "summary": {
            "layer_id": 17,
            "position": 2,
            "token_id": 12,
            "hidden_size": int(hidden.size),
            "layer_type": "linear_attention",
            "preceding_layer_count": 17,
            "finite": True,
        },
        "fields": {
            "hidden_in_f32": np.asarray(hidden, dtype=np.float32),
            "attn_norm_f32": np.asarray(attn_norm, dtype=np.float32),
        },
    }


def _weight_loader(weight: np.ndarray, eps: float):
    def load(_model: Path, layer_id: int):
        return np.asarray(weight, dtype=np.float32), float(eps), {
            "tensor_name": f"blk.{layer_id}.attn_norm.weight",
            "ggml_type": "F32",
            "shape": list(weight.shape),
            "summary": {"count": int(weight.size)},
            "config_eps": float(eps),
            "metadata_eps": float(eps),
            "materialization_layout": "dense_f32",
            "materialization_quant_key": "f32",
        }

    return load
