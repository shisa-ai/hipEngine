from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import bf16_roundtrip_array
from scripts.llamacpp_mtp_audit_layer0_moe_router_oracle import (
    bf16_roundtrip_like,
    softmax_f32,
)
from scripts.llamacpp_mtp_audit_layer9_moe_router_oracle import (
    audit_layer9_moe_router_oracle,
    classify_layer9_moe_router,
    classify_post_norm_input,
    compare_post_norm_hash,
    validate_layer9_post_attn_artifact,
)


def test_validate_layer9_post_attn_artifact_accepts_exact_artifact() -> None:
    validate_layer9_post_attn_artifact(
        _post_attn_artifact(_capture()),
        expected_layer_id=9,
    )


def test_validate_layer9_post_attn_artifact_rejects_nonexact_post_norm() -> None:
    artifact = _post_attn_artifact(_capture())
    artifact["oracle_results"]["post_norm_f32"]["delta_oracle_vs_hip"][
        "exact_match"
    ] = False

    with pytest.raises(ValueError, match="post_norm"):
        validate_layer9_post_attn_artifact(artifact, expected_layer_id=9)


def test_compare_post_norm_hash_accepts_expected_reference() -> None:
    capture = _capture()
    post_attn = _post_attn_artifact(capture)

    result = compare_post_norm_hash(capture=capture, post_attn=post_attn)

    assert result["exact_hash_match"] is True
    assert result["classification"] == "router_input_post_norm_matches_reference"
    assert classify_post_norm_input(result) == (
        "layer9_moe_router_input_matches_post_attn_artifact"
    )


def test_compare_post_norm_hash_classifies_mismatch() -> None:
    capture = _capture()
    post_attn = _post_attn_artifact(capture)
    post_attn["hipengine_capture"]["field_summaries"]["post_norm_f32"][
        "sha256"
    ] = "bad-sha"

    result = compare_post_norm_hash(capture=capture, post_attn=post_attn)

    assert result["exact_hash_match"] is False
    assert result["classification"] == "router_input_post_norm_mismatch_before_router"
    assert classify_post_norm_input(result) == (
        "layer9_moe_router_input_mismatch_before_router"
    )


def test_classify_layer9_moe_router_maps_exact_and_blocked() -> None:
    exact = {
        "selected_experts_i64": {
            "classification": "router_field_matches_oracle_exactly"
        },
        "routing_weights_f32": {
            "classification": "router_field_matches_oracle_exactly"
        },
        "shared_gate_logit_f32": {
            "classification": "router_field_matches_oracle_exactly"
        },
    }

    assert classify_layer9_moe_router(
        "layer9_moe_router_input_matches_post_attn_artifact",
        exact,
    ) == "layer9_moe_router_matches_oracle_exactly"
    assert classify_layer9_moe_router(
        "layer9_moe_router_input_mismatch_before_router",
        exact,
    ) == "layer9_moe_router_blocked_post_norm_input_mismatch"


def test_audit_layer9_moe_router_oracle_with_injected_exact_inputs(
    tmp_path: Path,
) -> None:
    capture = _capture()
    post_attn_path = tmp_path / "post_attn.json"
    post_attn_path.write_text(json.dumps(_post_attn_artifact(capture)))

    artifact = audit_layer9_moe_router_oracle(
        post_attn_artifact_path=post_attn_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        router_weight_loader=lambda *_args: (_weights(), {"source": "synthetic"}),
        iteration=397,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer9_moe_router_matches_oracle_exactly"
    assert artifact["position"] == 2
    assert artifact["post_norm_input_classification"] == (
        "layer9_moe_router_input_matches_post_attn_artifact"
    )
    assert artifact["next_action"] == (
        "audit_layer9_moe_selected_and_shared_expert_outputs"
    )
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer9_moe_router_oracle_classifies_router_mismatch(
    tmp_path: Path,
) -> None:
    capture = _capture()
    capture["fields"]["moe_selected_experts_i64"] = np.array([0, 1], dtype=np.int64)
    post_attn_path = tmp_path / "post_attn.json"
    post_attn_path.write_text(json.dumps(_post_attn_artifact(_capture())))

    artifact = audit_layer9_moe_router_oracle(
        post_attn_artifact_path=post_attn_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        router_weight_loader=lambda *_args: (_weights(), {"source": "synthetic"}),
        iteration=397,
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer9_moe_router_mismatch_after_oracle"
    assert artifact["oracle_results"]["selected_experts_i64"]["classification"] == (
        "router_field_mismatch_after_oracle"
    )
    assert artifact["next_action"] == (
        "inspect_layer9_moe_router_dtype_or_topk_semantics"
    )


def test_audit_layer9_moe_router_oracle_blocks_wrong_preceding_count(
    tmp_path: Path,
) -> None:
    capture = _capture()
    capture["summary"]["preceding_layer_count"] = 1
    post_attn_path = tmp_path / "post_attn.json"
    post_attn_path.write_text(json.dumps(_post_attn_artifact(_capture())))

    with pytest.raises(ValueError, match="preceding layers"):
        audit_layer9_moe_router_oracle(
            post_attn_artifact_path=post_attn_path,
            model_path=Path("/tmp/model.gguf"),
            layer_capture_fn=lambda *_args: capture,
            router_weight_loader=lambda *_args: (_weights(), {"source": "synthetic"}),
            iteration=397,
        )


def test_audit_layer9_moe_router_oracle_reports_unavailable_capture(
    tmp_path: Path,
) -> None:
    capture = _capture()
    post_attn_path = tmp_path / "post_attn.json"
    post_attn_path.write_text(json.dumps(_post_attn_artifact(capture)))

    artifact = audit_layer9_moe_router_oracle(
        post_attn_artifact_path=post_attn_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: {
            "status": "skipped_no_hip_runtime",
            "fields": {},
        },
        router_weight_loader=lambda *_args: (_weights(), {"source": "synthetic"}),
        iteration=397,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == "layer9_moe_router_oracle_unavailable"
    assert artifact["next_action"] == "rerun_layer9_moe_router_oracle_on_rocm_host"


def _capture() -> dict:
    post_norm = bf16_roundtrip_array(
        np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    )
    routing = softmax_f32(np.array([3.0, 2.0], dtype=np.float32))
    return {
        "status": "captured",
        "summary": {
            "layer_id": 9,
            "top_k": 2,
            "is_moe": True,
            "position": 2,
            "preceding_layer_count": 9,
        },
        "fields": {
            "post_norm_f32": post_norm,
            "moe_selected_experts_i64": np.array([2, 1], dtype=np.int64),
            "moe_routing_weights_f32": routing,
            "moe_shared_gate_f32": np.array([-3.75], dtype=np.float32),
        },
    }


def _weights() -> dict[str, np.ndarray]:
    return {
        "router_weight_bf16_f32": bf16_roundtrip_like(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        ),
        "shared_gate_weight_bf16_f32": bf16_roundtrip_like(
            np.array([0.25, 0.5, 0.0, -1.0], dtype=np.float32)
        ),
    }


def _post_attn_artifact(capture: dict) -> dict:
    from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import sha256_float32

    return {
        "status": "ready",
        "classification": "layer9_post_attn_residual_matches_oracle_exactly",
        "model": "/tmp/model.gguf",
        "layer_id": 9,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "hipengine_capture": {
            "field_summaries": {
                "post_norm_f32": {
                    "sha256": sha256_float32(capture["fields"]["post_norm_f32"]),
                }
            }
        },
        "oracle_results": {
            "post_norm_f32": {
                "delta_oracle_vs_hip": {"exact_match": True, "max_abs_diff": 0.0},
            }
        },
        "next_action": "audit_layer9_moe_router_from_post_norm",
    }
