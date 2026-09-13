from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (
    bf16_roundtrip_array,
    rmsnorm_f32,
)
from scripts.llamacpp_mtp_audit_layer1_attn_norm_oracle import (
    audit_layer1_attn_norm_oracle,
    classify_attn_norm,
    validate_layer1_handoff,
)


def test_validate_layer1_handoff_accepts_exact_artifact() -> None:
    validate_layer1_handoff(_handoff(), expected_layer_id=1)


def test_validate_layer1_handoff_rejects_nonexact_delta() -> None:
    handoff = _handoff()
    handoff["handoff_delta"]["max_abs_diff"] = 0.25

    with pytest.raises(ValueError, match="handoff delta"):
        validate_layer1_handoff(handoff, expected_layer_id=1)


def test_classify_attn_norm_accepts_exact_match() -> None:
    delta = {
        "available": True,
        "shape_match": True,
        "exact_match": True,
        "max_abs_diff": 0.0,
    }
    capture = {"summary": {"layer_id": 1, "preceding_layer_count": 1}}

    assert classify_attn_norm(
        delta,
        capture=capture,
        layer_id=1,
        attn_norm_atol=0.0,
    ) == "layer1_attn_norm_matches_bf16_oracle_exactly"


def test_classify_attn_norm_rejects_wrong_preceding_layer_count() -> None:
    delta = {
        "available": True,
        "shape_match": True,
        "exact_match": True,
        "max_abs_diff": 0.0,
    }
    capture = {"summary": {"layer_id": 1, "preceding_layer_count": 0}}

    assert classify_attn_norm(
        delta,
        capture=capture,
        layer_id=1,
        attn_norm_atol=0.0,
    ) == "layer1_attn_norm_wrong_preceding_layer_count"


def test_audit_layer1_attn_norm_oracle_with_injected_exact_capture(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(_handoff()))
    hidden = bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))
    weight = np.array([1.0, 0.75, 1.25, 0.5], dtype=np.float32)
    eps = 1.0e-6
    expected = bf16_roundtrip_array(rmsnorm_f32(hidden, weight, eps))

    artifact = audit_layer1_attn_norm_oracle(
        layer1_handoff_path=handoff_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=_capture(hidden, expected),
        weight_loader=_weight_loader(weight, eps),
        iteration=332,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer1_attn_norm_matches_bf16_oracle_exactly"
    assert artifact["attn_norm_delta"]["exact_match"] is True
    assert artifact["attn_norm_delta"]["max_abs_diff"] == 0.0
    assert artifact["next_action"] == "audit_layer1_projection_or_conv_gdn_under_bf16_contract"
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer1_attn_norm_oracle_classifies_mismatch(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(_handoff()))
    hidden = bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))
    weight = np.array([1.0, 0.75, 1.25, 0.5], dtype=np.float32)
    actual = bf16_roundtrip_array(rmsnorm_f32(hidden, weight, 1.0e-6))
    actual = actual.copy()
    actual[0] += np.float32(0.25)

    artifact = audit_layer1_attn_norm_oracle(
        layer1_handoff_path=handoff_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=_capture(hidden, actual),
        weight_loader=_weight_loader(weight, 1.0e-6),
        iteration=332,
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer1_attn_norm_mismatch_after_bf16_oracle"
    assert artifact["attn_norm_delta"]["max_abs_diff"] > 0.0
    assert artifact["next_action"] == "inspect_layer1_attn_norm_weight_or_rmsnorm_semantics"


def test_audit_layer1_attn_norm_oracle_reports_unavailable_capture(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(_handoff()))

    artifact = audit_layer1_attn_norm_oracle(
        layer1_handoff_path=handoff_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: {"status": "skipped_no_hip_runtime", "fields": {}},
        weight_loader=_weight_loader(np.ones((4,), dtype=np.float32), 1.0e-6),
        iteration=332,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == "layer1_attn_norm_oracle_unavailable"
    assert artifact["next_action"] == "rerun_layer1_attn_norm_oracle_on_rocm_host"


def _handoff() -> dict:
    return {
        "status": "ready",
        "classification": "layer1_hidden_in_matches_layer0_layer_out_exactly",
        "model": "/tmp/model.gguf",
        "target_layer": 1,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "handoff_delta": {"exact_match": True, "max_abs_diff": 0.0},
        "next_action": "audit_layer1_attn_norm_under_bf16_contract",
    }


def _capture(hidden: np.ndarray, attn_norm: np.ndarray):
    def capture(
        _model_path: Path,
        prompt_tokens: tuple[int, ...],
        position: int,
        layer_id: int,
        _max_sequence_length: int | None,
    ):
        return {
            "status": "captured",
            "summary": {
                "layer_id": int(layer_id),
                "position": int(position),
                "token_id": int(prompt_tokens[position]),
                "hidden_size": int(hidden.size),
                "layer_type": "linear_attention",
                "preceding_layer_count": int(layer_id),
                "finite": True,
            },
            "fields": {
                "hidden_in_f32": np.asarray(hidden, dtype=np.float32),
                "attn_norm_f32": np.asarray(attn_norm, dtype=np.float32),
            },
        }

    return capture


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
