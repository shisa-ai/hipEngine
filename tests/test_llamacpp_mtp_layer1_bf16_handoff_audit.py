from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_layer1_bf16_handoff_audit import (
    audit_layer1_bf16_handoff,
    classify_handoff,
    validate_layer0_conclusion,
)


def test_validate_layer0_conclusion_accepts_ready_exact_chain() -> None:
    validate_layer0_conclusion(_layer0_conclusion())


def test_validate_layer0_conclusion_rejects_nonexact_internal_chain() -> None:
    conclusion = _layer0_conclusion()
    conclusion["internal_bf16_oracle_chain"]["final_layer_out_delta"]["exact_match"] = False

    with pytest.raises(ValueError, match="final layer_out oracle"):
        validate_layer0_conclusion(conclusion)


def test_classify_handoff_requires_exact_preceding_layer_count() -> None:
    delta = {
        "available": True,
        "shape_match": True,
        "exact_match": True,
        "max_abs_diff": 0.0,
    }
    target_capture = {"summary": {"preceding_layer_count": 0}}

    assert classify_handoff(
        delta,
        target_capture=target_capture,
        target_layer=1,
        handoff_atol=0.0,
    ) == "layer1_bf16_handoff_wrong_preceding_layer_count"


def test_classify_handoff_accepts_exact_match() -> None:
    delta = {
        "available": True,
        "shape_match": True,
        "exact_match": True,
        "max_abs_diff": 0.0,
    }
    target_capture = {"summary": {"preceding_layer_count": 1}}

    assert classify_handoff(
        delta,
        target_capture=target_capture,
        target_layer=1,
        handoff_atol=0.0,
    ) == "layer1_hidden_in_matches_layer0_layer_out_exactly"


def test_audit_layer1_bf16_handoff_with_injected_exact_captures(tmp_path: Path) -> None:
    conclusion_path = tmp_path / "layer0.json"
    conclusion_path.write_text(json.dumps(_layer0_conclusion()))

    artifact = audit_layer1_bf16_handoff(
        layer0_conclusion_path=conclusion_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=_capture_fn(exact=True),
        iteration=331,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer1_hidden_in_matches_layer0_layer_out_exactly"
    assert artifact["source_layer"] == 0
    assert artifact["target_layer"] == 1
    assert artifact["handoff_delta"]["exact_match"] is True
    assert artifact["handoff_delta"]["max_abs_diff"] == 0.0
    assert artifact["next_action"] == "audit_layer1_attn_norm_under_bf16_contract"
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer1_bf16_handoff_classifies_mismatch(tmp_path: Path) -> None:
    conclusion_path = tmp_path / "layer0.json"
    conclusion_path.write_text(json.dumps(_layer0_conclusion()))

    artifact = audit_layer1_bf16_handoff(
        layer0_conclusion_path=conclusion_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=_capture_fn(exact=False),
        iteration=331,
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == (
        "layer1_hidden_in_mismatch_after_layer0_bf16_handoff"
    )
    assert artifact["handoff_delta"]["max_abs_diff"] > 0.0
    assert artifact["next_action"] == "inspect_layer0_to_layer1_hidden_buffer_handoff"


def test_audit_layer1_bf16_handoff_reports_unavailable_capture(tmp_path: Path) -> None:
    conclusion_path = tmp_path / "layer0.json"
    conclusion_path.write_text(json.dumps(_layer0_conclusion()))

    artifact = audit_layer1_bf16_handoff(
        layer0_conclusion_path=conclusion_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: {"status": "skipped_no_hip_runtime", "fields": {}},
        iteration=331,
    )

    assert artifact["status"] == "unavailable"
    assert artifact["classification"] == "layer1_bf16_handoff_capture_unavailable"
    assert artifact["next_action"] == "rerun_layer1_bf16_handoff_on_rocm_host"


def _layer0_conclusion() -> dict:
    return {
        "status": "ready",
        "classification": "layer0_runtime_matches_bf16_oracle_chain_after_llamacpp_f32_split",
        "model": "/tmp/model.gguf",
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "internal_bf16_oracle_chain": {
            "final_layer_out_delta": {"exact_match": True, "max_abs_diff": 0.0},
        },
        "next_action": "advance_bisection_to_layer1_or_next_layer_boundary_under_bf16_contract",
    }


def _capture_fn(*, exact: bool):
    layer0_out = np.array([1.0, -2.0, 0.5, 4.0], dtype=np.float32)
    layer1_hidden = layer0_out.copy()
    if not exact:
        layer1_hidden[2] += np.float32(0.25)

    def capture(
        _model_path: Path,
        _prompt_tokens: tuple[int, ...],
        position: int,
        layer_id: int,
        run_preceding_layers: bool,
        _max_sequence_length: int | None,
    ):
        if layer_id == 0:
            values = layer0_out
            hidden = np.zeros_like(layer0_out)
            preceding = 0
        else:
            values = np.zeros_like(layer0_out)
            hidden = layer1_hidden
            preceding = 1 if run_preceding_layers else 0
        return {
            "status": "captured",
            "summary": {
                "layer_id": int(layer_id),
                "position": int(position),
                "token_id": 12,
                "hidden_size": 4,
                "layer_type": "linear_attention",
                "preceding_layer_count": preceding,
                "finite": True,
            },
            "fields": {
                "hidden_in_f32": hidden.astype(np.float32),
                "layer_out_f32": values.astype(np.float32),
            },
        }

    return capture
