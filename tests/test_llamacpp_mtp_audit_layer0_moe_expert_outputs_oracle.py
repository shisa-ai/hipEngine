from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.llamacpp_mtp_audit_layer0_moe_expert_outputs_oracle import (
    audit_layer0_moe_expert_outputs_oracle,
    classify_expert_output_oracle,
    compare_expert_output_oracle,
    compute_moe_expert_outputs_oracle,
    kernel_order_dot_matrix,
    load_expert_output_weights,
    silu_mul_bf16,
    weighted_sum_bf16,
)
from scripts.llamacpp_mtp_audit_layer0_moe_router_oracle import bf16_roundtrip_like


def test_kernel_order_dot_matrix_matches_manual_wave_reduction() -> None:
    x = np.linspace(-0.25, 0.5, 64, dtype=np.float32)
    weight = np.vstack([np.ones_like(x), np.arange(64, dtype=np.float32) / 17.0])

    actual = kernel_order_dot_matrix(weight, x, threads=32)

    partial = np.zeros((32, 2), dtype=np.float32)
    for tid in range(32):
        acc = np.zeros((2,), dtype=np.float32)
        for k in range(tid, 64, 32):
            acc = np.float32(acc + np.float32(weight[:, k] * x[k]))
        partial[tid] = acc
    lanes = partial.copy()
    for offset in (16, 8, 4, 2, 1):
        lanes[: 32 - offset] = np.float32(lanes[: 32 - offset] + lanes[offset:32])
    np.testing.assert_array_equal(actual, lanes[0])


def test_silu_mul_and_weighted_sum_round_to_bf16() -> None:
    gate = bf16_roundtrip_like(np.array([1.0, -2.0], dtype=np.float32))
    up = bf16_roundtrip_like(np.array([0.5, 4.0], dtype=np.float32))

    inter = silu_mul_bf16(gate, up)
    weighted = weighted_sum_bf16(
        np.vstack([inter, -inter]).astype(np.float32),
        np.array([0.25, 0.75], dtype=np.float32),
    )

    expected_inter = bf16_roundtrip_like(
        np.float32(
            np.float32(gate * np.asarray(1.0 / (1.0 + np.exp(-gate)), dtype=np.float32))
            * up
        )
    )
    expected_weighted = bf16_roundtrip_like(
        np.float32(np.float32(inter * 0.25) + np.float32(-inter * 0.75))
    )
    np.testing.assert_array_equal(inter, expected_inter)
    np.testing.assert_array_equal(weighted, expected_weighted)


def test_compute_moe_expert_outputs_oracle_synthetic_identity_paths() -> None:
    post_norm = bf16_roundtrip_like(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))
    residual = bf16_roundtrip_like(np.array([0.5, 0.25, -0.5, 1.0], dtype=np.float32))
    weights = _synthetic_weights()

    oracle = compute_moe_expert_outputs_oracle(
        post_norm=post_norm,
        residual=residual,
        selected_experts=np.array([7, 3], dtype=np.int64),
        routing_weights=np.array([0.25, 0.75], dtype=np.float32),
        shared_gate_logit=np.array([0.0], dtype=np.float32),
        weights=weights,
        threads=32,
    )

    assert oracle["selected_down_f32"].shape == (2, 4)
    assert oracle["shared_out_f32"].shape == (4,)
    assert oracle["layer_out_f32"].shape == (4,)
    assert np.all(np.isfinite(oracle["layer_out_f32"]))
    np.testing.assert_array_equal(
        oracle["selected_experts_i64"],
        np.array([7, 3], dtype=np.int64),
    )


def test_compare_expert_output_oracle_classifies_exact_capture() -> None:
    oracle, capture = _oracle_and_capture()

    results = compare_expert_output_oracle(
        oracle=oracle,
        capture=capture,
        selected_atol=0.0,
        shared_atol=0.0,
        layer_out_atol=0.0,
    )

    assert results["ffn_or_moe_down_f32"]["classification"] == (
        "expert_output_matches_oracle_exactly"
    )
    assert results["moe_shared_out_f32"]["classification"] == (
        "expert_output_matches_oracle_exactly"
    )
    assert results["layer_out_f32"]["classification"] == (
        "expert_output_matches_oracle_exactly"
    )
    assert classify_expert_output_oracle(results) == (
        "layer0_moe_expert_outputs_match_oracle_exactly"
    )


def test_compare_expert_output_oracle_classifies_branch_mismatch() -> None:
    oracle, capture = _oracle_and_capture()
    capture["fields"]["moe_shared_out_f32"] = capture["fields"]["moe_shared_out_f32"].copy()
    capture["fields"]["moe_shared_out_f32"][0] += np.float32(1.0)

    results = compare_expert_output_oracle(
        oracle=oracle,
        capture=capture,
        selected_atol=0.0,
        shared_atol=0.0,
        layer_out_atol=0.0,
    )

    assert results["moe_shared_out_f32"]["classification"] == (
        "expert_output_mismatch_after_oracle"
    )
    assert classify_expert_output_oracle(results) == (
        "layer0_moe_expert_outputs_mismatch_after_oracle"
    )


def test_audit_layer0_moe_expert_outputs_oracle_with_injected_inputs(tmp_path: Path) -> None:
    oracle, capture = _oracle_and_capture()
    router_path = tmp_path / "router.json"
    router_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "classification": "layer0_moe_router_matches_oracle_within_tolerance",
                "next_action": "audit_layer0_moe_selected_and_shared_expert_outputs",
                "model": "/tmp/model.gguf",
                "layer_id": 0,
                "target_position": 2,
                "prompt_tokens": [10, 11, 12],
                "oracle_results": {
                    "selected_experts_i64": {"oracle_values": [7, 3]},
                },
            }
        )
    )

    artifact = audit_layer0_moe_expert_outputs_oracle(
        router_artifact_path=router_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        expert_weight_loader=lambda *_args: (_synthetic_weights(), {"source": "synthetic"}),
        iteration=329,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer0_moe_expert_outputs_match_oracle_exactly"
    assert artifact["next_action"] == "continue_layer0_bisection_after_moe_ffn_or_compare_layer_out"
    assert artifact["external_checkout_modified"] is False
    assert artifact["oracle_contract"]["selected_experts"] == [7, 3]
    json.dumps(artifact)


def test_load_expert_output_weights_uses_selected_rows(monkeypatch, tmp_path: Path) -> None:
    raw_tensors = {
        "blk.1.ffn_gate_exps.weight": np.arange(3 * 2 * 4, dtype=np.uint8).reshape(3, 2, 4),
        "blk.1.ffn_up_exps.weight": np.arange(24, 48, dtype=np.uint8).reshape(3, 2, 4),
        "blk.1.ffn_down_exps.weight": np.arange(48, 72, dtype=np.uint8).reshape(3, 2, 4),
        "blk.1.ffn_gate_shexp.weight": np.arange(8, dtype=np.uint8).reshape(2, 4),
        "blk.1.ffn_up_shexp.weight": np.arange(8, 16, dtype=np.uint8).reshape(2, 4),
        "blk.1.ffn_down_shexp.weight": np.arange(16, 24, dtype=np.uint8).reshape(2, 4),
    }

    class FakeReader:
        def __init__(self, path: Path) -> None:
            self.path = path

        def tensor_info(self, name: str):
            return SimpleNamespace(
                name=name,
                shape=raw_tensors[name].shape,
                ggml_type=0,
                ggml_type_name="F32",
            )

        def tensor_data(self, name: str):
            return raw_tensors[name]

    def fake_dequant(data, _qtype):
        return np.asarray(data, dtype=np.float32)

    monkeypatch.setattr(
        "scripts.llamacpp_mtp_audit_layer0_moe_expert_outputs_oracle.GGUFReader",
        FakeReader,
    )
    monkeypatch.setattr(
        "scripts.llamacpp_mtp_audit_layer0_moe_expert_outputs_oracle.dequantize_gguf_data",
        fake_dequant,
    )

    weights, metadata = load_expert_output_weights(
        tmp_path / "model.gguf",
        1,
        np.array([2, 0], dtype=np.int64),
    )

    np.testing.assert_array_equal(weights["selected_gate"], raw_tensors[
        "blk.1.ffn_gate_exps.weight"
    ][[2, 0]].astype(np.float32))
    assert weights["shared_down"].shape == (2, 4)
    assert metadata["selected_gate"]["selection"] == [2, 0]
    assert metadata["shared_gate"]["selection"] == "all_rows"


def _synthetic_weights() -> dict[str, np.ndarray]:
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


def _oracle_and_capture():
    weights = _synthetic_weights()
    oracle = compute_moe_expert_outputs_oracle(
        post_norm=bf16_roundtrip_like(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)),
        residual=bf16_roundtrip_like(np.array([0.5, 0.25, -0.5, 1.0], dtype=np.float32)),
        selected_experts=np.array([7, 3], dtype=np.int64),
        routing_weights=np.array([0.25, 0.75], dtype=np.float32),
        shared_gate_logit=np.array([0.0], dtype=np.float32),
        weights=weights,
        threads=32,
    )
    capture = {
        "status": "captured",
        "summary": {"hidden_size": 4, "top_k": 2, "is_moe": True, "position": 2},
        "fields": {
            "post_norm_f32": bf16_roundtrip_like(
                np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
            ),
            "residual_f32": bf16_roundtrip_like(
                np.array([0.5, 0.25, -0.5, 1.0], dtype=np.float32)
            ),
            "moe_selected_experts_i64": np.array([7, 3], dtype=np.int64),
            "moe_routing_weights_f32": np.array([0.25, 0.75], dtype=np.float32),
            "moe_shared_gate_f32": np.array([0.0], dtype=np.float32),
            "ffn_or_moe_down_f32": oracle["selected_down_f32"].reshape(-1).copy(),
            "moe_shared_out_f32": oracle["shared_out_f32"].copy(),
            "layer_out_f32": oracle["layer_out_f32"].copy(),
        },
    }
    return oracle, capture
