from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import bf16_roundtrip_array
from scripts.llamacpp_mtp_audit_layer0_post_attn_residual_oracle import (
    add_rmsnorm_bf16_oracle,
    audit_layer0_post_attn_residual_oracle,
    build_post_attn_oracle_results,
    classify_inputs,
    classify_post_attn_oracle,
    compare_inputs,
    kernel_strided_sum,
    load_post_attention_norm_weight,
)


def test_kernel_strided_sum_matches_manual_thread_reduction() -> None:
    values = np.arange(1, 17, dtype=np.float32)

    actual = kernel_strided_sum(values, threads=4)

    partial = np.asarray(
        [
            values[0] + values[4] + values[8] + values[12],
            values[1] + values[5] + values[9] + values[13],
            values[2] + values[6] + values[10] + values[14],
            values[3] + values[7] + values[11] + values[15],
        ],
        dtype=np.float32,
    )
    partial[0] = np.float32(partial[0] + partial[2])
    partial[1] = np.float32(partial[1] + partial[3])
    partial[0] = np.float32(partial[0] + partial[1])
    assert actual == partial[0]


def test_add_rmsnorm_bf16_oracle_uses_unrounded_sum_for_norm() -> None:
    hidden = bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, -0.25], dtype=np.float32))
    add = bf16_roundtrip_array(np.array([0.25, 0.5, -0.75, 2.0], dtype=np.float32))
    weight = np.ones((4,), dtype=np.float32)

    residual, post_norm = add_rmsnorm_bf16_oracle(hidden, add, weight, eps=0.0, threads=4)

    summed = np.asarray(hidden + add, dtype=np.float32)
    expected_residual = bf16_roundtrip_array(summed)
    inv_rms = np.float32(1.0 / np.sqrt(np.mean(summed * summed, dtype=np.float32)))
    expected_post_norm = bf16_roundtrip_array(summed * inv_rms)
    np.testing.assert_allclose(residual, expected_residual, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(post_norm, expected_post_norm, rtol=0.0, atol=0.0)


def test_compare_inputs_accepts_exact_hidden_and_warm_attn_out() -> None:
    capture = _capture()
    token_hidden = capture["fields"]["hidden_in_f32"].copy()

    results = compare_inputs(capture=capture, token_hidden=token_hidden, near_atol=0.0)

    assert results["hidden_in_f32"]["classification"] == (
        "post_attn_field_matches_oracle_exactly"
    )
    assert results["attn_out_f32"]["classification"] == "input_covered_by_warm_conv_gdn_oracle"
    assert classify_inputs(results) == "post_attn_inputs_match_oracle"


def test_build_post_attn_oracle_results_classifies_exact_capture() -> None:
    capture = _capture()
    weight = np.ones((4,), dtype=np.float32)

    results = build_post_attn_oracle_results(
        capture=capture,
        post_norm_weight=weight,
        eps=1.0e-6,
        residual_atol=0.0,
        post_norm_atol=0.0,
    )

    assert results["residual_f32"]["classification"] == "post_attn_field_matches_oracle_exactly"
    assert results["post_norm_f32"]["classification"] == "post_attn_field_matches_oracle_exactly"
    assert classify_post_attn_oracle("post_attn_inputs_match_oracle", results) == (
        "layer0_post_attn_residual_matches_oracle_exactly"
    )


def test_build_post_attn_oracle_results_classifies_mismatch() -> None:
    capture = _capture()
    capture["fields"]["residual_f32"] = capture["fields"]["residual_f32"].copy()
    capture["fields"]["residual_f32"][0] += np.float32(0.5)

    results = build_post_attn_oracle_results(
        capture=capture,
        post_norm_weight=np.ones((4,), dtype=np.float32),
        eps=1.0e-6,
        residual_atol=1.0e-6,
        post_norm_atol=1.0e-6,
    )

    assert results["residual_f32"]["classification"] == "post_attn_field_mismatch_after_oracle"
    assert classify_post_attn_oracle("post_attn_inputs_match_oracle", results) == (
        "layer0_post_attn_residual_mismatch_after_oracle"
    )


def test_audit_layer0_post_attn_residual_oracle_with_injected_inputs(tmp_path: Path) -> None:
    warm_path = tmp_path / "warm.json"
    warm_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "classification": "layer0_warm_conv_gdn_matches_oracle_within_tolerance",
                "next_action": "continue_layer0_bisection_after_attn_out_or_residual",
                "model": "/tmp/model.gguf",
                "layer_id": 0,
                "target_position": 2,
                "prompt_tokens": [10, 11, 12],
            }
        )
    )
    capture = _capture(position=2, token_id=12)

    artifact = audit_layer0_post_attn_residual_oracle(
        warm_artifact_path=warm_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        norm_weight_loader=lambda *_args: (
            np.ones((4,), dtype=np.float32),
            1.0e-6,
            {"tensor_name": "blk.0.post_attention_norm.weight"},
        ),
        token_hidden_loader=lambda *_args: (
            capture["fields"]["hidden_in_f32"].copy(),
            {"source": "synthetic"},
        ),
        iteration=327,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer0_post_attn_residual_matches_oracle_exactly"
    assert artifact["target_position"] == 2
    assert artifact["input_classification"] == "post_attn_inputs_match_oracle"
    assert artifact["next_action"] == "audit_layer0_moe_router_from_post_norm"
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_load_post_attention_norm_weight_reads_expected_tensor(monkeypatch, tmp_path: Path) -> None:
    class FakeReader:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.info = SimpleNamespace(
                metadata={"qwen35moe.attention.layer_norm_rms_epsilon": 1.0e-6}
            )

        def tensor_info(self, name: str):
            assert name == "blk.3.post_attention_norm.weight"
            return SimpleNamespace(ggml_type_name="F32")

        def dequantize_tensor(self, name: str):
            assert name == "blk.3.post_attention_norm.weight"
            return np.array([1.0, 2.0], dtype=np.float32)

    monkeypatch.setattr(
        "scripts.llamacpp_mtp_audit_layer0_post_attn_residual_oracle.GGUFReader",
        FakeReader,
    )

    values, eps, metadata = load_post_attention_norm_weight(tmp_path / "model.gguf", 3)

    np.testing.assert_array_equal(values, np.array([1.0, 2.0], dtype=np.float32))
    assert eps == 1.0e-6
    assert metadata["tensor_name"] == "blk.3.post_attention_norm.weight"


def _capture(*, position: int = 0, token_id: int = 1):
    hidden = bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, -0.25], dtype=np.float32))
    attn = bf16_roundtrip_array(np.array([0.25, 0.5, -0.75, 2.0], dtype=np.float32))
    residual, post_norm = add_rmsnorm_bf16_oracle(
        hidden,
        attn,
        np.ones((4,), dtype=np.float32),
        eps=1.0e-6,
        threads=4,
    )
    return {
        "status": "captured",
        "summary": {
            "layer_id": 0,
            "position": int(position),
            "token_id": int(token_id),
            "is_moe": True,
        },
        "fields": {
            "hidden_in_f32": hidden,
            "attn_out_f32": attn,
            "residual_f32": residual,
            "post_norm_f32": post_norm,
            "ffn_or_moe_down_f32": np.zeros((8,), dtype=np.float32),
            "layer_out_f32": np.zeros((4,), dtype=np.float32),
            "moe_routing_weights_f32": np.array([1.0, 0.0], dtype=np.float32),
            "moe_shared_gate_f32": np.array([0.5], dtype=np.float32),
            "moe_selected_experts_i64": np.array([0, 1], dtype=np.int64),
        },
    }
