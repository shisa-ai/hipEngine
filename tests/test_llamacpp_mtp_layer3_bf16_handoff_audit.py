from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import sha256_float32
from scripts.llamacpp_mtp_layer3_bf16_handoff_audit import (
    audit_layer3_bf16_handoff,
    classify_handoff,
    compare_source_layer_out,
    validate_layer2_expert_outputs,
)


def test_validate_layer2_expert_outputs_accepts_ready_exact_artifact() -> None:
    validate_layer2_expert_outputs(_layer2_artifact(), expected_layer_id=2)


def test_validate_layer2_expert_outputs_rejects_nonexact_layer_out() -> None:
    artifact = _layer2_artifact()
    artifact["oracle_results"]["layer_out_f32"]["delta_oracle_vs_hip"][
        "exact_match"
    ] = False

    with pytest.raises(ValueError, match="layer_out"):
        validate_layer2_expert_outputs(artifact, expected_layer_id=2)


def test_validate_layer2_expert_outputs_rejects_wrong_next_action() -> None:
    artifact = _layer2_artifact()
    artifact["next_action"] = "audit_something_else"

    with pytest.raises(ValueError, match="layer3 handoff"):
        validate_layer2_expert_outputs(artifact, expected_layer_id=2)


def test_compare_source_layer_out_accepts_matching_hash() -> None:
    source = _capture(layer_id=2, preceding=2, layer_out=_layer2_out())
    layer2 = _layer2_artifact()

    result = compare_source_layer_out(source_capture=source, layer2=layer2)

    assert result["exact_hash_match"] is True
    assert result["classification"] == "layer3_handoff_source_matches_layer2_artifact"


def test_compare_source_layer_out_detects_source_mismatch() -> None:
    source = _capture(layer_id=2, preceding=2, layer_out=_layer2_out() + np.float32(0.25))
    layer2 = _layer2_artifact()

    result = compare_source_layer_out(source_capture=source, layer2=layer2)

    assert result["exact_hash_match"] is False
    assert result["classification"] == "layer3_handoff_source_mismatch_before_handoff"


def test_classify_handoff_accepts_exact_match_with_expected_preceding_counts() -> None:
    delta = {"available": True, "shape_match": True, "exact_match": True, "max_abs_diff": 0.0}
    source = {"summary": {"preceding_layer_count": 2}}
    target = {"summary": {"preceding_layer_count": 3}}
    source_reference = {"exact_hash_match": True}

    assert classify_handoff(
        delta,
        source_capture=source,
        target_capture=target,
        source_layer=2,
        target_layer=3,
        source_reference=source_reference,
        handoff_atol=0.0,
    ) == "layer3_hidden_in_matches_layer2_layer_out_exactly"


def test_classify_handoff_rejects_wrong_target_preceding_count() -> None:
    delta = {"available": True, "shape_match": True, "exact_match": True, "max_abs_diff": 0.0}
    source = {"summary": {"preceding_layer_count": 2}}
    target = {"summary": {"preceding_layer_count": 2}}
    source_reference = {"exact_hash_match": True}

    assert classify_handoff(
        delta,
        source_capture=source,
        target_capture=target,
        source_layer=2,
        target_layer=3,
        source_reference=source_reference,
        handoff_atol=0.0,
    ) == "layer3_bf16_handoff_wrong_target_preceding_layer_count"


def test_audit_layer3_bf16_handoff_with_injected_exact_captures(tmp_path: Path) -> None:
    layer2_path = tmp_path / "layer2.json"
    layer2_path.write_text(json.dumps(_layer2_artifact()))

    artifact = audit_layer3_bf16_handoff(
        layer2_experts_path=layer2_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=_capture_fn(exact=True),
        iteration=345,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer3_hidden_in_matches_layer2_layer_out_exactly"
    assert artifact["source_layer"] == 2
    assert artifact["target_layer"] == 3
    assert artifact["source_reference"]["exact_hash_match"] is True
    assert artifact["handoff_delta"]["exact_match"] is True
    assert artifact["handoff_delta"]["max_abs_diff"] == 0.0
    assert artifact["next_action"] == "audit_layer3_attn_norm_under_bf16_contract"
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer3_bf16_handoff_classifies_mismatch(tmp_path: Path) -> None:
    layer2_path = tmp_path / "layer2.json"
    layer2_path.write_text(json.dumps(_layer2_artifact()))

    artifact = audit_layer3_bf16_handoff(
        layer2_experts_path=layer2_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=_capture_fn(exact=False),
        iteration=345,
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer3_hidden_in_mismatch_after_layer2_bf16_handoff"
    assert artifact["handoff_delta"]["max_abs_diff"] > 0.0
    assert artifact["next_action"] == "inspect_layer2_to_layer3_hidden_buffer_handoff"


def test_audit_layer3_bf16_handoff_reports_unavailable_capture(tmp_path: Path) -> None:
    layer2_path = tmp_path / "layer2.json"
    layer2_path.write_text(json.dumps(_layer2_artifact()))

    artifact = audit_layer3_bf16_handoff(
        layer2_experts_path=layer2_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: {"status": "skipped_no_hip_runtime", "fields": {}},
        iteration=345,
    )

    assert artifact["status"] == "unavailable"
    assert artifact["classification"] == "layer3_bf16_handoff_capture_unavailable"
    assert artifact["next_action"] == "rerun_layer3_bf16_handoff_on_rocm_host"


def _layer2_out() -> np.ndarray:
    return np.array([0.25, -1.5, 2.0, 4.0], dtype=np.float32)


def _layer2_artifact() -> dict:
    values = _layer2_out()
    return {
        "status": "ready",
        "classification": "layer2_moe_expert_outputs_match_oracle_within_tolerance",
        "model": "/tmp/model.gguf",
        "layer_id": 2,
        "target_position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "hipengine_capture": {
            "field_summaries": {
                "layer_out_f32": {"sha256": sha256_float32(values)},
            }
        },
        "oracle_results": {
            "layer_out_f32": {
                "delta_oracle_vs_hip": {"exact_match": True, "max_abs_diff": 0.0},
            }
        },
        "next_action": "advance_bisection_to_layer3_or_next_layer_boundary_under_bf16_contract",
    }


def _capture(
    *,
    layer_id: int,
    preceding: int,
    hidden: np.ndarray | None = None,
    layer_out: np.ndarray | None = None,
):
    values = _layer2_out()
    return {
        "status": "captured",
        "summary": {
            "layer_id": int(layer_id),
            "position": 2,
            "token_id": 12,
            "hidden_size": 4,
            "layer_type": "linear_attention",
            "preceding_layer_count": int(preceding),
            "finite": True,
        },
        "fields": {
            "hidden_in_f32": np.asarray(hidden if hidden is not None else values, dtype=np.float32),
            "layer_out_f32": np.asarray(
                layer_out if layer_out is not None else values,
                dtype=np.float32,
            ),
        },
    }


def _capture_fn(*, exact: bool):
    layer2_out = _layer2_out()
    layer3_hidden = layer2_out.copy()
    if not exact:
        layer3_hidden[1] += np.float32(0.25)

    def capture(
        _model_path: Path,
        _prompt_tokens: tuple[int, ...],
        _position: int,
        layer_id: int,
        run_preceding_layers: bool,
        _max_sequence_length: int | None,
    ):
        if layer_id == 2:
            return _capture(layer_id=2, preceding=2 if run_preceding_layers else 0)
        return _capture(
            layer_id=3,
            preceding=3 if run_preceding_layers else 0,
            hidden=layer3_hidden,
            layer_out=np.zeros_like(layer2_out),
        )

    return capture
