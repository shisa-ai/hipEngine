from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import sha256_float32
from scripts.llamacpp_mtp_layer9_bf16_handoff_audit import (
    audit_layer9_bf16_handoff,
    classify_handoff,
    compare_source_layer_out,
    validate_layer8_expert_outputs,
)


def test_validate_layer8_expert_outputs_accepts_ready_exact_artifact() -> None:
    validate_layer8_expert_outputs(_layer8_artifact(), expected_layer_id=8)


def test_validate_layer8_expert_outputs_rejects_nonexact_layer_out() -> None:
    artifact = _layer8_artifact()
    artifact["oracle_results"]["layer_out_f32"]["delta_oracle_vs_hip"][
        "exact_match"
    ] = False

    with pytest.raises(ValueError, match="layer_out"):
        validate_layer8_expert_outputs(artifact, expected_layer_id=8)


def test_validate_layer8_expert_outputs_rejects_wrong_next_action() -> None:
    artifact = _layer8_artifact()
    artifact["next_action"] = "audit_something_else"

    with pytest.raises(ValueError, match="layer9 handoff"):
        validate_layer8_expert_outputs(artifact, expected_layer_id=8)


def test_compare_source_layer_out_accepts_matching_hash() -> None:
    source = _capture(layer_id=8, preceding=8, layer_out=_layer8_out())
    layer8 = _layer8_artifact()

    result = compare_source_layer_out(source_capture=source, layer8=layer8)

    assert result["exact_hash_match"] is True
    assert result["classification"] == (
        "layer9_handoff_source_matches_layer8_artifact"
    )


def test_compare_source_layer_out_detects_source_mismatch() -> None:
    source = _capture(
        layer_id=8,
        preceding=8,
        layer_out=_layer8_out() + np.float32(0.25),
    )
    layer8 = _layer8_artifact()

    result = compare_source_layer_out(source_capture=source, layer8=layer8)

    assert result["exact_hash_match"] is False
    assert result["classification"] == (
        "layer9_handoff_source_mismatch_before_handoff"
    )


def test_classify_handoff_accepts_exact_match_with_expected_preceding_counts() -> None:
    delta = {
        "available": True,
        "shape_match": True,
        "exact_match": True,
        "max_abs_diff": 0.0,
    }
    source = {"summary": {"preceding_layer_count": 8}}
    target = {"summary": {"preceding_layer_count": 9}}
    source_reference = {"exact_hash_match": True}

    assert classify_handoff(
        delta,
        source_capture=source,
        target_capture=target,
        source_layer=8,
        target_layer=9,
        source_reference=source_reference,
        handoff_atol=0.0,
    ) == "layer9_hidden_in_matches_layer8_layer_out_exactly"


def test_classify_handoff_rejects_wrong_target_preceding_count() -> None:
    delta = {
        "available": True,
        "shape_match": True,
        "exact_match": True,
        "max_abs_diff": 0.0,
    }
    source = {"summary": {"preceding_layer_count": 8}}
    target = {"summary": {"preceding_layer_count": 8}}
    source_reference = {"exact_hash_match": True}

    assert classify_handoff(
        delta,
        source_capture=source,
        target_capture=target,
        source_layer=8,
        target_layer=9,
        source_reference=source_reference,
        handoff_atol=0.0,
    ) == "layer9_bf16_handoff_wrong_target_preceding_layer_count"


def test_audit_layer9_bf16_handoff_with_injected_exact_captures(tmp_path: Path) -> None:
    layer8_path = tmp_path / "layer8.json"
    layer8_path.write_text(json.dumps(_layer8_artifact()))

    artifact = audit_layer9_bf16_handoff(
        layer8_experts_path=layer8_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=_capture_fn(exact=True),
        iteration=392,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer9_hidden_in_matches_layer8_layer_out_exactly"
    )
    assert artifact["source_layer"] == 8
    assert artifact["target_layer"] == 9
    assert artifact["source_reference"]["exact_hash_match"] is True
    assert artifact["handoff_delta"]["exact_match"] is True
    assert artifact["handoff_delta"]["max_abs_diff"] == 0.0
    assert artifact["next_action"] == (
        "audit_layer9_attn_norm_under_bf16_contract_or_mtp_boundary"
    )
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer9_bf16_handoff_classifies_mismatch(tmp_path: Path) -> None:
    layer8_path = tmp_path / "layer8.json"
    layer8_path.write_text(json.dumps(_layer8_artifact()))

    artifact = audit_layer9_bf16_handoff(
        layer8_experts_path=layer8_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=_capture_fn(exact=False),
        iteration=392,
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == (
        "layer9_hidden_in_mismatch_after_layer8_bf16_handoff"
    )
    assert artifact["handoff_delta"]["max_abs_diff"] > 0.0
    assert artifact["next_action"] == "inspect_layer8_to_layer9_hidden_buffer_handoff"


def test_audit_layer9_bf16_handoff_reports_unavailable_capture(tmp_path: Path) -> None:
    layer8_path = tmp_path / "layer8.json"
    layer8_path.write_text(json.dumps(_layer8_artifact()))

    artifact = audit_layer9_bf16_handoff(
        layer8_experts_path=layer8_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: {
            "status": "skipped_no_hip_runtime",
            "fields": {},
        },
        iteration=392,
    )

    assert artifact["status"] == "unavailable"
    assert artifact["classification"] == "layer9_bf16_handoff_capture_unavailable"
    assert artifact["next_action"] == "rerun_layer9_bf16_handoff_on_rocm_host"


def _layer8_out() -> np.ndarray:
    return np.array([0.25, -1.5, 2.0, 4.0], dtype=np.float32)


def _layer8_artifact() -> dict:
    values = _layer8_out()
    return {
        "status": "ready",
        "classification": "layer8_moe_expert_outputs_match_oracle_exactly",
        "model": "/tmp/model.gguf",
        "layer_id": 8,
        "position": 2,
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
        "next_action": "audit_layer9_bf16_handoff_or_mtp_next_boundary",
    }


def _capture(
    *,
    layer_id: int,
    preceding: int,
    hidden: np.ndarray | None = None,
    layer_out: np.ndarray | None = None,
):
    values = _layer8_out()
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
            "hidden_in_f32": np.asarray(
                hidden if hidden is not None else values,
                dtype=np.float32,
            ),
            "layer_out_f32": np.asarray(
                layer_out if layer_out is not None else values,
                dtype=np.float32,
            ),
        },
    }


def _capture_fn(*, exact: bool):
    layer8_out = _layer8_out()
    layer9_hidden = layer8_out.copy()
    if not exact:
        layer9_hidden[1] += np.float32(0.25)

    def capture(
        _model_path: Path,
        _prompt_tokens: tuple[int, ...],
        _position: int,
        layer_id: int,
        run_preceding_layers: bool,
        _max_sequence_length: int | None,
    ):
        if layer_id == 8:
            return _capture(
                layer_id=8,
                preceding=8 if run_preceding_layers else 0,
                layer_out=layer8_out,
            )
        return _capture(
            layer_id=9,
            preceding=9 if run_preceding_layers else 0,
            hidden=layer9_hidden,
            layer_out=np.zeros_like(layer8_out),
        )

    return capture
