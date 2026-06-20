from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import bf16_roundtrip_array
from scripts.llamacpp_mtp_audit_layer0_moe_router_oracle import (
    audit_layer0_moe_router_oracle,
    bf16_roundtrip_like,
    classify_moe_router_oracle,
    compare_router_oracle,
    compute_router_oracle,
    load_router_weights,
    router_dot_kernel_order,
    select_topk_descending,
    softmax_f32,
)


def test_router_dot_kernel_order_matches_manual_thread_reduction() -> None:
    hidden = np.arange(1, 9, dtype=np.float32)
    weight = np.linspace(-0.5, 0.25, 8, dtype=np.float32)

    actual = router_dot_kernel_order(hidden, weight, threads=4)

    partial = np.zeros((4,), dtype=np.float32)
    acc = np.float32(0.0)
    for k in range(8):
        acc = np.float32(acc + np.float32(hidden[k] * weight[k]))
    partial[0] = acc
    partial[0] = np.float32(partial[0] + partial[2])
    partial[1] = np.float32(partial[1] + partial[3])
    partial[0] = np.float32(partial[0] + partial[1])
    assert actual == partial[0]


def test_select_topk_descending_and_softmax_are_stable() -> None:
    logits = np.array([0.5, 2.0, 2.0, -1.0, 1.0], dtype=np.float32)

    selected = select_topk_descending(logits, top_k=3)
    routing = softmax_f32(logits[selected])

    np.testing.assert_array_equal(selected, np.array([1, 2, 4], dtype=np.int64))
    assert np.isclose(float(np.sum(routing)), 1.0, atol=1.0e-6)
    assert routing[0] == routing[1]
    assert routing[0] > routing[2]


def test_compute_router_oracle_matches_expected_synthetic_values() -> None:
    hidden = bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))
    router_weight = bf16_roundtrip_like(
        np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    )
    shared_weight = bf16_roundtrip_like(np.array([0.25, 0.5, 0.0, -1.0], dtype=np.float32))

    router = compute_router_oracle(hidden, router_weight, shared_weight, top_k=2, threads=4)

    np.testing.assert_array_equal(router["selected_experts_i64"], np.array([2, 1], dtype=np.int64))
    np.testing.assert_allclose(router["expert_logits_f32"], np.array([1.0, 2.0, 3.0]))
    expected_shared = np.float32(1.0 * 0.25 + -2.0 * 0.5 + 3.0 * -1.0)
    assert router["shared_gate_logit_f32"][0] == expected_shared


def test_compare_router_oracle_classifies_exact_synthetic_capture() -> None:
    capture, router = _capture_and_router()

    results = compare_router_oracle(
        router=router,
        capture=capture,
        routing_atol=0.0,
        shared_gate_atol=0.0,
    )

    assert results["selected_experts_i64"]["classification"] == (
        "router_field_matches_oracle_exactly"
    )
    assert results["routing_weights_f32"]["classification"] == (
        "router_field_matches_oracle_exactly"
    )
    assert results["shared_gate_logit_f32"]["classification"] == (
        "router_field_matches_oracle_exactly"
    )
    assert classify_moe_router_oracle(results) == "layer0_moe_router_matches_oracle_exactly"


def test_compare_router_oracle_classifies_selected_expert_mismatch() -> None:
    capture, router = _capture_and_router()
    capture["fields"]["moe_selected_experts_i64"] = np.array([0, 1], dtype=np.int64)

    results = compare_router_oracle(
        router=router,
        capture=capture,
        routing_atol=0.0,
        shared_gate_atol=0.0,
    )

    assert results["selected_experts_i64"]["classification"] == (
        "router_field_mismatch_after_oracle"
    )
    assert classify_moe_router_oracle(results) == "layer0_moe_router_mismatch_after_oracle"


def test_audit_layer0_moe_router_oracle_with_injected_inputs(tmp_path: Path) -> None:
    capture, router = _capture_and_router()
    post_attn_path = tmp_path / "post_attn.json"
    post_attn_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "classification": "layer0_post_attn_residual_matches_oracle_exactly",
                "next_action": "audit_layer0_moe_router_from_post_norm",
                "model": "/tmp/model.gguf",
                "layer_id": 0,
                "target_position": 2,
                "prompt_tokens": [10, 11, 12],
            }
        )
    )
    weights = {
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
    weight_meta = {"source": "synthetic"}

    artifact = audit_layer0_moe_router_oracle(
        post_attn_artifact_path=post_attn_path,
        model_path=Path("/tmp/model.gguf"),
        layer_capture_fn=lambda *_args: capture,
        router_weight_loader=lambda *_args: (weights, weight_meta),
        iteration=328,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer0_moe_router_matches_oracle_exactly"
    assert artifact["target_position"] == 2
    assert artifact["next_action"] == "audit_layer0_moe_selected_and_shared_expert_outputs"
    assert artifact["external_checkout_modified"] is False
    assert artifact["oracle_contract"]["shared_gate_capture"].startswith("raw shared-gate")
    json.dumps(artifact)


def test_load_router_weights_rounds_f32_to_bf16(monkeypatch, tmp_path: Path) -> None:
    class FakeReader:
        def __init__(self, path: Path) -> None:
            self.path = path

        def tensor_info(self, name: str):
            return SimpleNamespace(ggml_type_name="F32")

        def dequantize_tensor(self, name: str):
            values = {
                "blk.1.ffn_gate_inp.weight": np.array([[1.1, -2.2]], dtype=np.float32),
                "blk.1.ffn_gate_inp_shexp.weight": np.array([0.3, -0.4], dtype=np.float32),
            }
            return values[name]

    monkeypatch.setattr(
        "scripts.llamacpp_mtp_audit_layer0_moe_router_oracle.GGUFReader",
        FakeReader,
    )

    weights, metadata = load_router_weights(tmp_path / "model.gguf", 1)

    np.testing.assert_array_equal(
        weights["router_weight_bf16_f32"],
        bf16_roundtrip_like(np.array([[1.1, -2.2]], dtype=np.float32)),
    )
    np.testing.assert_array_equal(
        weights["shared_gate_weight_bf16_f32"],
        bf16_roundtrip_like(np.array([0.3, -0.4], dtype=np.float32)),
    )
    assert metadata["router_weight_bf16_f32"]["contract"].startswith("source F32")


def _capture_and_router():
    post_norm = bf16_roundtrip_array(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))
    router = {
        "expert_logits_f32": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "shared_gate_logit_f32": np.array([-3.75], dtype=np.float32),
        "selected_experts_i64": np.array([2, 1], dtype=np.int64),
        "routing_weights_f32": softmax_f32(np.array([3.0, 2.0], dtype=np.float32)),
        "shared_gate_sigmoid_f32": np.array([1.0 / (1.0 + np.exp(3.75))], dtype=np.float32),
    }
    capture = {
        "status": "captured",
        "summary": {"top_k": 2, "is_moe": True, "position": 2},
        "fields": {
            "post_norm_f32": post_norm,
            "moe_selected_experts_i64": router["selected_experts_i64"].copy(),
            "moe_routing_weights_f32": router["routing_weights_f32"].copy(),
            "moe_shared_gate_f32": router["shared_gate_logit_f32"].copy(),
        },
    }
    return capture, router
