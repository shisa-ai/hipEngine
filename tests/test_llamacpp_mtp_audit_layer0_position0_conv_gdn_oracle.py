from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.llamacpp_mtp_audit_layer0_position0_conv_gdn_oracle import (
    audit_layer0_position0_conv_gdn_oracle,
    build_position0_oracle_results,
    classify_position0_oracle,
    conv_decode_zero_state,
    gdn_recurrent_zero_state,
    load_conv_gdn_weights,
    silu_f32,
    tree_reduce_sum,
)
from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import bf16_roundtrip_array
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import project_f32


def test_conv_decode_zero_state_uses_last_kernel_tap_only() -> None:
    linear_qkv = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    conv_weight = np.array(
        [[10.0, 20.0, 30.0, 0.5], [1.0, 2.0, 3.0, -0.25], [0.0, 0.0, 0.0, 2.0]],
        dtype=np.float32,
    )

    actual = conv_decode_zero_state(linear_qkv, conv_weight)

    expected = np.array(
        [silu_f32(0.5), silu_f32(0.5), silu_f32(6.0)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_tree_reduce_sum_matches_kernel_thread_grouping() -> None:
    values = np.arange(1, 17, dtype=np.float32)

    actual = tree_reduce_sum(values, threads=8, group=2)

    assert actual == np.float32(np.sum(values, dtype=np.float32))


def test_gdn_recurrent_zero_state_matches_small_manual_case() -> None:
    # num_k_heads=1, num_v_heads=1, head_k_dim=2, head_v_dim=2.
    # q=[3,4] => q_scale=(1/5)/sqrt(2); k=[6,8] => k_scale=1/10.
    # beta=sigmoid(0)=0.5; values=[10,20]; gate=[1,1]; norm=[1,1].
    conv = np.array([3.0, 4.0, 6.0, 8.0, 10.0, 20.0], dtype=np.float32)
    gate = np.ones((2,), dtype=np.float32)
    beta = np.zeros((1,), dtype=np.float32)

    actual = gdn_recurrent_zero_state(
        conv_out=conv,
        gate=gate,
        alpha=np.zeros((1,), dtype=np.float32),
        beta=beta,
        dt_bias=np.zeros((1,), dtype=np.float32),
        a_log=np.zeros((1,), dtype=np.float32),
        norm_weight=np.ones((2,), dtype=np.float32),
        eps=0.0,
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=2,
        head_v_dim=2,
    )

    q_scale = np.float32((1.0 / 5.0) / np.sqrt(np.float32(2.0)))
    k_scale = np.float32(1.0 / 10.0)
    beta_value = np.float32(0.5)
    raw = []
    for value in (np.float32(10.0), np.float32(20.0)):
        delta = np.float32(value * beta_value)
        out_acc = np.float32(0.0)
        for q, k in zip((3.0, 4.0), (6.0, 8.0), strict=True):
            out_acc = np.float32(
                out_acc
                + np.float32(np.float32(q) * q_scale)
                * np.float32(np.float32(k) * k_scale)
                * delta
            )
        raw.append(out_acc)
    raw_arr = np.asarray(raw, dtype=np.float32)
    inv_rms = np.float32(1.0 / np.sqrt(np.mean(raw_arr * raw_arr, dtype=np.float32)))
    expected = np.asarray([x * inv_rms * silu_f32(1.0) for x in raw_arr], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)


def test_build_position0_oracle_results_classifies_exact_synthetic_capture() -> None:
    capture, weights, dims = _synthetic_capture_weights_and_dims()

    results = build_position0_oracle_results(
        capture=capture,
        weights=weights,
        dimensions=dims,
        eps=1.0e-6,
        tolerances={
            "conv_out_f32": 0.0,
            "recurrent_out_f32": 0.0,
            "recurrent_bf16_f32": 0.0,
            "attn_out_f32": 0.0,
        },
    )

    assert results["conv_out_f32"]["classification"] == "field_matches_oracle_exactly"
    assert results["recurrent_out_f32"]["classification"] == "field_matches_oracle_exactly"
    assert results["recurrent_bf16_f32"]["classification"] == "field_matches_oracle_exactly"
    assert results["attn_out_f32"]["classification"] == "field_matches_oracle_exactly"
    assert classify_position0_oracle(results) == "layer0_position0_conv_gdn_matches_oracle_exactly"


def test_build_position0_oracle_results_classifies_mismatch() -> None:
    capture, weights, dims = _synthetic_capture_weights_and_dims()
    capture["fields"]["conv_out_f32"] = capture["fields"]["conv_out_f32"].copy()
    capture["fields"]["conv_out_f32"][0] += np.float32(0.5)

    results = build_position0_oracle_results(
        capture=capture,
        weights=weights,
        dimensions=dims,
        eps=1.0e-6,
        tolerances={
            "conv_out_f32": 1.0e-6,
            "recurrent_out_f32": 1.0e-6,
            "recurrent_bf16_f32": 1.0e-6,
            "attn_out_f32": 1.0e-6,
        },
    )

    assert results["conv_out_f32"]["classification"] == (
        "field_mismatch_after_position0_oracle"
    )
    assert classify_position0_oracle(results) == "layer0_position0_conv_gdn_mismatch_after_oracle"


def test_audit_layer0_position0_conv_gdn_oracle_with_injected_inputs(tmp_path: Path) -> None:
    capture, weights, dims = _synthetic_capture_weights_and_dims()
    projection_path = tmp_path / "projection.json"
    plan_path = tmp_path / "plan.json"
    projection_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "classification": "layer0_projections_match_bf16_oracle_within_rounding",
                "model": "/tmp/model.gguf",
                "layer_id": 0,
                "prompt_tokens": [42],
            }
        )
    )
    plan_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "decision": {"selected_strategy": "position0_stateless_conv_gdn_oracle_first"},
                "model_metadata": {
                    "dimensions": dims | {"conv_state_floats": 32, "recurrent_state_floats": 16},
                    "rms_norm_eps": 1.0e-6,
                },
            }
        )
    )

    artifact = audit_layer0_position0_conv_gdn_oracle(
        projection_artifact_path=projection_path,
        plan_path=plan_path,
        model_path=Path("/tmp/model.gguf"),
        boundary_capture_fn=lambda *_args: capture,
        weight_loader=lambda *_args: weights,
        iteration=325,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer0_position0_conv_gdn_matches_oracle_exactly"
    assert artifact["position"] == 0
    assert artifact["token_id"] == 42
    assert artifact["input_contract"]["zero_state"] is True
    assert artifact["next_action"] == (
        "extend_conv_gdn_oracle_to_warm_position16_replay_or_state_capture"
    )
    json.dumps(artifact)


def test_load_conv_gdn_weights_converts_ssm_a_to_kernel_a_log(monkeypatch, tmp_path: Path) -> None:
    class FakeReader:
        def __init__(self, path: Path) -> None:
            self.path = path

        def tensor_info(self, name: str):
            return SimpleNamespace(ggml_type_name="F32")

        def dequantize_tensor(self, name: str):
            values = {
                "blk.0.ssm_conv1d.weight": np.ones((6, 4), dtype=np.float32),
                "blk.0.ssm_dt.bias": np.zeros((1,), dtype=np.float32),
                "blk.0.ssm_norm.weight": np.ones((2,), dtype=np.float32),
                "blk.0.ssm_out.weight": np.eye(2, dtype=np.float32),
                "blk.0.ssm_a": np.array([-1.0, -2.0], dtype=np.float32),
            }
            return values[name]

    monkeypatch.setattr(
        "scripts.llamacpp_mtp_audit_layer0_position0_conv_gdn_oracle.GGUFReader",
        FakeReader,
    )

    weights = load_conv_gdn_weights(tmp_path / "model.gguf", 0)

    np.testing.assert_allclose(weights["ssm_a_log"][0], np.log([1.0, 2.0]).astype(np.float32))
    assert weights["ssm_a_log"][1]["source_transform"] == "log(-GGUF blk.*.ssm_a)"


def _synthetic_capture_weights_and_dims():
    dims = {
        "ssm_group_count": 1,
        "ssm_time_step_rank": 1,
        "ssm_state_size": 2,
        "ssm_value_dim": 2,
        "ssm_inner_size": 2,
        "linear_qkv_width": 6,
    }
    linear_qkv = bf16_roundtrip_array(
        np.array([0.25, -0.5, 0.75, -1.0, 0.5, -0.25], dtype=np.float32)
    )
    linear_z = bf16_roundtrip_array(np.array([0.5, -0.25], dtype=np.float32))
    alpha = bf16_roundtrip_array(np.array([0.0], dtype=np.float32))
    beta = bf16_roundtrip_array(np.array([0.25], dtype=np.float32))
    conv_weight = np.ones((6, 4), dtype=np.float32)
    conv_out = conv_decode_zero_state(linear_qkv, conv_weight)
    recurrent_out = gdn_recurrent_zero_state(
        conv_out=conv_out,
        gate=linear_z,
        alpha=alpha,
        beta=beta,
        dt_bias=np.zeros((1,), dtype=np.float32),
        a_log=np.zeros((1,), dtype=np.float32),
        norm_weight=np.ones((2,), dtype=np.float32),
        eps=1.0e-6,
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=2,
        head_v_dim=2,
    )
    recurrent_bf16 = bf16_roundtrip_array(recurrent_out)
    ssm_out = np.eye(2, dtype=np.float32)
    attn_out = bf16_roundtrip_array(project_f32(recurrent_bf16, ssm_out))
    capture = {
        "status": "captured",
        "summary": {"position": 0},
        "fields": {
            "linear_qkv_f32": linear_qkv,
            "linear_z_f32": linear_z,
            "ssm_alpha_f32": alpha,
            "ssm_beta_f32": beta,
            "conv_out_f32": conv_out,
            "recurrent_out_f32": recurrent_out,
            "recurrent_bf16_f32": recurrent_bf16,
            "attn_out_f32": attn_out,
        },
    }
    weights = {
        "ssm_conv1d": (conv_weight, {"tensor_name": "conv"}),
        "ssm_dt_bias": (np.zeros((1,), dtype=np.float32), {"tensor_name": "dt"}),
        "ssm_a_log": (np.zeros((1,), dtype=np.float32), {"tensor_name": "a"}),
        "ssm_norm": (np.ones((2,), dtype=np.float32), {"tensor_name": "norm"}),
        "ssm_out": (ssm_out, {"tensor_name": "out"}),
    }
    return capture, weights, dims
