from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.llamacpp_mtp_audit_layer0_moe_expert_outputs_oracle import (
    compute_moe_expert_outputs_oracle,
)
from scripts.llamacpp_mtp_audit_layer0_moe_router_oracle import bf16_roundtrip_like
from scripts.llamacpp_mtp_audit_layer17_moe_expert_outputs_oracle import (
    audit_layer17_moe_expert_outputs_oracle,
    classify_layer17_expert_outputs,
    classify_router_inputs,
    compare_router_inputs,
    validate_layer17_router_artifact,
)


def test_validate_layer17_router_artifact_accepts_ready_artifact() -> None:
    capture = _capture()
    validate_layer17_router_artifact(_router_artifact(capture), expected_layer_id=17)


def test_validate_layer17_router_artifact_rejects_bad_post_norm_input() -> None:
    capture = _capture()
    artifact = _router_artifact(capture)
    artifact["post_norm_input"]["exact_hash_match"] = False

    with pytest.raises(ValueError, match="post_norm input"):
        validate_layer17_router_artifact(artifact, expected_layer_id=17)


def test_validate_layer17_router_artifact_rejects_bad_next_action() -> None:
    capture = _capture()
    artifact = _router_artifact(capture)
    artifact["next_action"] = "audit_something_else"

    with pytest.raises(ValueError, match="expert-output audit"):
        validate_layer17_router_artifact(artifact, expected_layer_id=17)


def test_compare_router_inputs_accepts_expected_post_norm_and_selection() -> None:
    capture = _capture()
    router = _router_artifact(capture)

    results = compare_router_inputs(capture=capture, router_artifact=router)

    assert results["post_norm_f32"]["exact_hash_match"] is True
    assert results["selected_experts_i64"]["exact_match"] is True
    assert classify_router_inputs(results) == (
        "layer17_moe_expert_inputs_match_router_artifact"
    )


def test_compare_router_inputs_classifies_selection_mismatch() -> None:
    capture = _capture()
    router = _router_artifact(capture)
    router["oracle_results"]["selected_experts_i64"]["oracle_values"] = [0, 1]

    results = compare_router_inputs(capture=capture, router_artifact=router)

    assert results["selected_experts_i64"]["classification"] == (
        "expert_input_selected_experts_mismatch_before_experts"
    )
    assert classify_router_inputs(results) == (
        "layer17_moe_expert_inputs_mismatch_before_experts"
    )


def test_classify_layer17_expert_outputs_maps_exact_near_and_blocked() -> None:
    exact = {
        "ffn_or_moe_down_f32": {
            "classification": "expert_output_matches_oracle_exactly"
        },
        "moe_shared_out_f32": {
            "classification": "expert_output_matches_oracle_exactly"
        },
        "layer_out_f32": {"classification": "expert_output_matches_oracle_exactly"},
    }
    near = json.loads(json.dumps(exact))
    near["ffn_or_moe_down_f32"]["classification"] = (
        "expert_output_matches_oracle_within_tolerance"
    )

    assert classify_layer17_expert_outputs(
        "layer17_moe_expert_inputs_match_router_artifact",
        exact,
    ) == "layer17_moe_expert_outputs_match_oracle_exactly"
    assert classify_layer17_expert_outputs(
        "layer17_moe_expert_inputs_match_router_artifact",
        near,
    ) == "layer17_moe_expert_outputs_match_oracle_within_tolerance"
    assert classify_layer17_expert_outputs(
        "layer17_moe_expert_inputs_mismatch_before_experts",
        exact,
    ) == "layer17_moe_expert_outputs_blocked_router_input_mismatch"


def test_audit_layer17_moe_expert_outputs_oracle_with_injected_exact_inputs(
    tmp_path: Path,
) -> None:
    capture = _capture()
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(_router_artifact(capture)))

    artifact = audit_layer17_moe_expert_outputs_oracle(
        router_artifact_path=router_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        expert_weight_loader=lambda *_args: (_weights(), {"source": "synthetic"}),
        iteration=459,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer17_moe_expert_outputs_match_oracle_exactly"
    )
    assert artifact["position"] == 2
    assert artifact["router_input_classification"] == (
        "layer17_moe_expert_inputs_match_router_artifact"
    )
    assert artifact["next_action"] == (
        "audit_layer18_bf16_handoff_or_mtp_next_boundary"
    )
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_audit_layer17_moe_expert_outputs_oracle_accepts_selected_within_tolerance(
    tmp_path: Path,
) -> None:
    capture = _capture()
    capture["fields"]["ffn_or_moe_down_f32"] = capture["fields"][
        "ffn_or_moe_down_f32"
    ].copy()
    capture["fields"]["ffn_or_moe_down_f32"][0] += np.float32(2.44e-4)
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(_router_artifact(_capture())))

    artifact = audit_layer17_moe_expert_outputs_oracle(
        router_artifact_path=router_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        expert_weight_loader=lambda *_args: (_weights(), {"source": "synthetic"}),
        iteration=459,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer17_moe_expert_outputs_match_oracle_within_tolerance"
    )
    assert artifact["oracle_results"]["ffn_or_moe_down_f32"]["classification"] == (
        "expert_output_matches_oracle_within_tolerance"
    )


def test_audit_layer17_moe_expert_outputs_oracle_accepts_shared_within_tolerance(
    tmp_path: Path,
) -> None:
    capture = _capture()
    capture["fields"]["moe_shared_out_f32"] = capture["fields"][
        "moe_shared_out_f32"
    ].copy()
    capture["fields"]["moe_shared_out_f32"][0] += np.float32(2.0e-7)
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(_router_artifact(_capture())))

    artifact = audit_layer17_moe_expert_outputs_oracle(
        router_artifact_path=router_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        expert_weight_loader=lambda *_args: (_weights(), {"source": "synthetic"}),
        shared_atol=5.0e-7,
        iteration=459,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer17_moe_expert_outputs_match_oracle_within_tolerance"
    )
    assert artifact["oracle_results"]["moe_shared_out_f32"]["classification"] == (
        "expert_output_matches_oracle_within_tolerance"
    )


def test_audit_layer17_moe_expert_outputs_oracle_classifies_mismatch(
    tmp_path: Path,
) -> None:
    capture = _capture()
    capture["fields"]["moe_shared_out_f32"] = capture["fields"][
        "moe_shared_out_f32"
    ].copy()
    capture["fields"]["moe_shared_out_f32"][0] += np.float32(1.0)
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(_router_artifact(_capture())))

    artifact = audit_layer17_moe_expert_outputs_oracle(
        router_artifact_path=router_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        expert_weight_loader=lambda *_args: (_weights(), {"source": "synthetic"}),
        iteration=459,
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == (
        "layer17_moe_expert_outputs_mismatch_after_oracle"
    )
    assert artifact["oracle_results"]["moe_shared_out_f32"]["classification"] == (
        "expert_output_mismatch_after_oracle"
    )
    assert artifact["next_action"] == (
        "inspect_layer17_moe_expert_output_dtype_or_dequant_semantics"
    )


def test_audit_layer17_moe_expert_outputs_oracle_rejects_wrong_preceding_count(
    tmp_path: Path,
) -> None:
    capture = _capture()
    capture["summary"]["preceding_layer_count"] = 1
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(_router_artifact(_capture())))

    with pytest.raises(ValueError, match="preceding layers"):
        audit_layer17_moe_expert_outputs_oracle(
            router_artifact_path=router_path,
            model_path=Path("/tmp/model.gguf"),
            layer_capture_fn=lambda *_args: capture,
            expert_weight_loader=lambda *_args: (_weights(), {"source": "synthetic"}),
            iteration=459,
        )


def test_audit_layer17_moe_expert_outputs_oracle_reports_unavailable_capture(
    tmp_path: Path,
) -> None:
    capture = _capture()
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(_router_artifact(capture)))

    artifact = audit_layer17_moe_expert_outputs_oracle(
        router_artifact_path=router_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: {
            "status": "skipped_no_hip_runtime",
            "fields": {},
        },
        expert_weight_loader=lambda *_args: (_weights(), {"source": "synthetic"}),
        iteration=459,
    )

    assert artifact["status"] == "skipped_no_hip_runtime"
    assert artifact["classification"] == (
        "layer17_moe_expert_outputs_oracle_unavailable"
    )
    assert artifact["next_action"] == (
        "rerun_layer17_moe_expert_outputs_oracle_on_rocm_host"
    )


def _capture() -> dict:
    post_norm = bf16_roundtrip_like(
        np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    )
    residual = bf16_roundtrip_like(
        np.array([0.5, 0.25, -0.5, 1.0], dtype=np.float32)
    )
    oracle = compute_moe_expert_outputs_oracle(
        post_norm=post_norm,
        residual=residual,
        selected_experts=np.array([7, 3], dtype=np.int64),
        routing_weights=np.array([0.25, 0.75], dtype=np.float32),
        shared_gate_logit=np.array([0.0], dtype=np.float32),
        weights=_weights(),
        threads=32,
    )
    return {
        "status": "captured",
        "summary": {
            "layer_id": 17,
            "layer_type": "linear_attention",
            "hidden_size": 4,
            "top_k": 2,
            "is_moe": True,
            "position": 2,
            "preceding_layer_count": 17,
        },
        "fields": {
            "post_norm_f32": post_norm,
            "residual_f32": residual,
            "moe_selected_experts_i64": np.array([7, 3], dtype=np.int64),
            "moe_routing_weights_f32": np.array([0.25, 0.75], dtype=np.float32),
            "moe_shared_gate_f32": np.array([0.0], dtype=np.float32),
            "ffn_or_moe_down_f32": oracle["selected_down_f32"].reshape(-1).copy(),
            "moe_shared_out_f32": oracle["shared_out_f32"].copy(),
            "layer_out_f32": oracle["layer_out_f32"].copy(),
        },
    }


def _weights() -> dict[str, np.ndarray]:
    selected_gate = np.zeros((2, 2, 4), dtype=np.float32)
    selected_up = np.zeros((2, 2, 4), dtype=np.float32)
    selected_down = np.zeros((2, 4, 2), dtype=np.float32)
    selected_gate[0, 0, 0] = 1.0
    selected_gate[0, 1, 1] = -1.0
    selected_gate[1, 0, 2] = 2.0
    selected_gate[1, 1, 3] = 0.5
    selected_up[:, :, :] = selected_gate + np.float32(0.25)
    selected_down[0, :, :] = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [0.5, 0.5]],
        dtype=np.float32,
    )
    selected_down[1, :, :] = np.array(
        [[-1.0, 1.0], [0.5, 0.0], [0.0, 0.5], [1.0, 1.0]],
        dtype=np.float32,
    )
    shared_gate = np.eye(2, 4, dtype=np.float32)
    shared_up = np.flip(shared_gate, axis=1).astype(np.float32)
    shared_down = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 1.0]],
        dtype=np.float32,
    )
    return {
        "selected_gate": selected_gate,
        "selected_up": selected_up,
        "selected_down": selected_down,
        "shared_gate": shared_gate,
        "shared_up": shared_up,
        "shared_down": shared_down,
    }


def _router_artifact(capture: dict) -> dict:
    from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import sha256_float32

    selected = capture["fields"]["moe_selected_experts_i64"].tolist()
    post_norm_sha = sha256_float32(capture["fields"]["post_norm_f32"])
    return {
        "status": "ready",
        "classification": "layer17_moe_router_matches_oracle_within_tolerance",
        "model": "/tmp/model.gguf",
        "layer_id": 17,
        "position": 2,
        "token_id": 12,
        "prompt_tokens": [10, 11, 12],
        "post_norm_input": {
            "expected_sha256": post_norm_sha,
            "exact_hash_match": True,
        },
        "oracle_results": {
            "selected_experts_i64": {
                "oracle_values": [int(value) for value in selected]
            },
        },
        "next_action": "audit_layer17_moe_selected_and_shared_expert_outputs",
    }
