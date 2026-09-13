from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_layer0_dtype_oracle_policy import (
    BF16_CONTRACTION_CONCLUSION,
    EXPECTED_HIP_CANDIDATE,
    EXPECTED_LLAMA_BF16_CANDIDATE,
    EXPECTED_LLAMA_CANDIDATE,
    audit_readiness,
    build_layer0_dtype_oracle_policy,
    extract_boundary_capture_fields,
    extract_dataclass_fields,
)


def test_audit_readiness_accepts_exact_bf16_contraction_evidence() -> None:
    readiness = audit_readiness(_audit())

    assert readiness["ready"] is True
    assert readiness["facts"]["bf16_contraction_conclusion"] is True
    assert readiness["facts"]["llama_f32_candidate_exact"] is True
    assert readiness["facts"]["llama_bf16_candidate_exact"] is True
    assert readiness["facts"]["hip_bf16_candidate_exact"] is True
    assert readiness["evidence"]["llamacpp_input_bf16_sha256"] == "input-bf16"
    assert readiness["evidence"]["hipengine_attn_norm_sha256"] == "hip-attn"


def test_audit_readiness_rejects_missing_exact_candidate() -> None:
    audit = _audit()
    audit["best_candidates"]["vs_hipengine_attn_norm"]["delta"]["exact_match"] = False

    readiness = audit_readiness(audit)

    assert readiness["ready"] is False
    assert readiness["facts"]["hip_bf16_candidate_exact"] is False


def test_extract_boundary_capture_fields_from_runner_text() -> None:
    fields = extract_boundary_capture_fields(_runner_text(), _audit())

    assert fields["ready"] is True
    assert fields["shape_context"]["hidden_size"] == 2048
    assert fields["fields"][0]["field"] == "attn_norm_f32"
    assert fields["fields"][1]["field"] == "linear_qkv_f32"
    assert fields["fields"][1]["host_copy"] == "BF16_to_F32"
    assert fields["fields"][5]["field"] == "conv_out_f32"
    assert fields["fields"][5]["host_copy"] == "F32"


def test_extract_dataclass_fields_returns_empty_when_missing() -> None:
    assert extract_dataclass_fields("class Other:\n    pass\n") == set()


def test_build_layer0_dtype_oracle_policy_from_synthetic_inputs(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    compare_path = tmp_path / "compare.json"
    runner_path = tmp_path / "runner.py"
    audit_path.write_text(json.dumps(_audit()))
    compare_path.write_text(json.dumps(_compare()))
    runner_path.write_text(_runner_text())

    artifact = build_layer0_dtype_oracle_policy(
        audit_path=audit_path,
        attn_compare_path=compare_path,
        runner_path=runner_path,
        iteration=322,
    )

    assert artifact["status"] == "ready"
    assert artifact["decision"]["selected_policy"] == (
        "bf16_contracted_llamacpp_or_cpu_oracle"
    )
    assert artifact["decision"]["scope"] == "layer0_ar_boundary_bisection_only"
    assert artifact["decision"]["direct_llamacpp_f32_classification"] == (
        "layer0_attn_norm_mismatch_after_bf16_roundtrip"
    )
    assert artifact["constraints"]["must_not_change_hot_path"] is True
    assert artifact["constraints"]["mtp_seed_contract_unchanged"] == (
        "post_output_norm fp32 hidden seed"
    )
    assert artifact["next_probe_plan"]["ordered_probes"][0]["field"] == (
        "linear_qkv_f32"
    )
    assert artifact["next_probe_plan"]["ordered_probes"][1]["field"] == "linear_z_f32"
    assert artifact["next_action"] == "build_layer0_bf16_contracted_projection_oracle"
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def _audit() -> dict[str, object]:
    return {
        "status": "ready",
        "conclusion": BF16_CONTRACTION_CONCLUSION,
        "model": "/models/model.gguf",
        "layer_id": 0,
        "position": 16,
        "token_id": 271,
        "input_capture": {"sha256": "input-f32"},
        "input_bf16_roundtrip": {"sha256": "input-bf16"},
        "llamacpp_attn_norm": {"sha256": "llama-attn"},
        "hipengine_attn_norm": {
            "summary": {"sha256": "hip-attn"},
            "capture_summary": {
                "hidden_size": 2048,
                "linear_qkv_width": 8192,
                "ssm_inner_size": 4096,
                "ssm_time_step_rank": 32,
                "attn_norm_shape": [2048],
                "linear_qkv_shape": [8192],
                "linear_z_shape": [4096],
                "ssm_alpha_shape": [32],
                "ssm_beta_shape": [32],
                "conv_out_shape": [8192],
                "recurrent_out_shape": [4096],
                "recurrent_bf16_shape": [4096],
                "attn_out_shape": [2048],
                "finite": True,
            },
        },
        "weight": {"materialization_layout": "dense_f32", "materialization_quant_key": "f32"},
        "formula": {"model_eps": 1.0e-6},
        "best_candidates": {
            "vs_llamacpp_attn_norm": _candidate(EXPECTED_LLAMA_CANDIDATE),
            "vs_llamacpp_attn_norm_bf16": _candidate(EXPECTED_LLAMA_BF16_CANDIDATE),
            "vs_hipengine_attn_norm": _candidate(EXPECTED_HIP_CANDIDATE),
        },
    }


def _candidate(name: str) -> dict[str, object]:
    return {"name": name, "delta": {"exact_match": True, "rmse": 0.0}}


def _compare() -> dict[str, object]:
    return {
        "status": "mismatched",
        "classification": "layer0_attn_norm_mismatch_after_bf16_roundtrip",
    }


def _runner_text() -> str:
    return '''@dataclass(frozen=True)
class Qwen35GGUFLinearAttentionBoundaryCapture:
    layer_id: int
    token_id: int
    position: int
    hidden_size: int
    ssm_time_step_rank: int
    linear_qkv_width: int
    ssm_inner_size: int
    attn_norm_f32: np.ndarray
    linear_qkv_f32: np.ndarray
    linear_z_f32: np.ndarray
    ssm_alpha_f32: np.ndarray
    ssm_beta_f32: np.ndarray
    conv_out_f32: np.ndarray
    recurrent_out_f32: np.ndarray
    recurrent_bf16_f32: np.ndarray
    attn_out_f32: np.ndarray

@dataclass(frozen=True)
class Other:
    pass
'''
