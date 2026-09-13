from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hipengine.quant.gguf import GGMLQuantizationType
from scripts.llamacpp_mtp_audit_layer0_warm_conv_gdn_oracle import (
    audit_layer0_warm_conv_gdn_oracle,
    build_warm_oracle_results,
    classify_target_inputs,
    classify_warm_oracle,
    compare_target_inputs,
    conv_decode_step,
    gdn_recurrent_step,
    load_token_embedding_rows,
    replay_conv_gdn_sequence,
)


def test_conv_decode_step_shifts_state_and_uses_history() -> None:
    state = np.zeros((2, 3), dtype=np.float32)
    weight = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=np.float32)

    first = conv_decode_step(np.array([2.0, -1.0], dtype=np.float32), weight, state)
    second = conv_decode_step(np.array([4.0, 3.0], dtype=np.float32), weight, state)

    np.testing.assert_allclose(state, np.array([[0.0, 2.0, 4.0], [0.0, -1.0, 3.0]]))
    np.testing.assert_allclose(first, _silu(np.array([6.0, -2.0], dtype=np.float32)))
    np.testing.assert_allclose(second, _silu(np.array([16.0, 5.5], dtype=np.float32)))


def test_gdn_recurrent_step_state_changes_second_step() -> None:
    dims = _dims()
    recurrent_state = np.zeros((1, 2, 2), dtype=np.float32)
    conv = np.array([3.0, 4.0, 6.0, 8.0, 10.0, 20.0], dtype=np.float32)
    kwargs = dict(
        conv_out=conv,
        gate=np.ones((2,), dtype=np.float32),
        alpha=np.zeros((1,), dtype=np.float32),
        beta=np.zeros((1,), dtype=np.float32),
        dt_bias=np.zeros((1,), dtype=np.float32),
        a_log=np.zeros((1,), dtype=np.float32),
        norm_weight=np.ones((2,), dtype=np.float32),
        eps=1.0e-6,
        num_k_heads=dims["ssm_group_count"],
        num_v_heads=dims["ssm_time_step_rank"],
        head_k_dim=dims["ssm_state_size"],
        head_v_dim=dims["ssm_value_dim"],
    )

    first = gdn_recurrent_step(recurrent_state=recurrent_state, **kwargs)
    state_after_first = recurrent_state.copy()
    second = gdn_recurrent_step(recurrent_state=recurrent_state, **kwargs)

    assert np.any(state_after_first != 0.0)
    assert np.any(recurrent_state != state_after_first)
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))


def test_replay_conv_gdn_sequence_matches_synthetic_target() -> None:
    replay_inputs, weights, dims = _synthetic_replay_inputs_weights_dims()
    records, summary = replay_conv_gdn_sequence(
        replay_inputs=replay_inputs,
        weights=weights,
        dimensions=dims,
        eps=1.0e-6,
    )

    assert len(records) == 3
    assert summary["replayed_positions"] == 3
    assert records[2]["conv_out_f32"].shape == (6,)
    assert records[2]["recurrent_out_f32"].shape == (2,)
    assert records[2]["attn_out_f32"].shape == (2,)


def test_build_warm_oracle_results_classifies_exact_synthetic_target() -> None:
    replay_inputs, weights, dims = _synthetic_replay_inputs_weights_dims()
    records, _summary = replay_conv_gdn_sequence(
        replay_inputs=replay_inputs,
        weights=weights,
        dimensions=dims,
        eps=1.0e-6,
    )
    target_capture = _target_capture_from_record(records[2], replay_inputs[2])

    results, replay_summary = build_warm_oracle_results(
        target_capture=target_capture,
        replay_inputs=replay_inputs,
        weights=weights,
        dimensions=dims,
        eps=1.0e-6,
        target_position=2,
        tolerances={
            "conv_out_f32": 0.0,
            "recurrent_out_f32": 0.0,
            "recurrent_bf16_f32": 0.0,
            "attn_out_f32": 0.0,
        },
    )

    assert results["conv_out_f32"]["classification"] == "warm_field_matches_oracle_exactly"
    assert results["recurrent_out_f32"]["classification"] == "warm_field_matches_oracle_exactly"
    assert replay_summary["replayed_positions"] == 3
    assert classify_warm_oracle("target_inputs_match_replay_exactly", results) == (
        "layer0_warm_conv_gdn_matches_oracle_exactly"
    )


def test_compare_target_inputs_blocks_when_projection_inputs_mismatch() -> None:
    replay_inputs, weights, dims = _synthetic_replay_inputs_weights_dims()
    records, _summary = replay_conv_gdn_sequence(
        replay_inputs=replay_inputs,
        weights=weights,
        dimensions=dims,
        eps=1.0e-6,
    )
    target_capture = _target_capture_from_record(records[1], replay_inputs[1])
    target_capture["fields"]["linear_qkv_f32"] = target_capture["fields"]["linear_qkv_f32"].copy()
    target_capture["fields"]["linear_qkv_f32"][0] += np.float32(1.0)

    target_inputs = compare_target_inputs(replay_inputs[1], target_capture, near_atol=1.0e-6)

    assert target_inputs["linear_qkv_f32"]["classification"] == (
        "warm_field_mismatch_after_replay_oracle"
    )
    assert classify_target_inputs(target_inputs) == "target_inputs_mismatch_before_conv_gdn_replay"


def test_load_token_embedding_rows_preserves_requested_order() -> None:
    class FakeReader:
        def tensor_info(self, name: str):
            assert name == "token_embd.weight"
            return SimpleNamespace(
                shape=(5, 3),
                ggml_type=int(GGMLQuantizationType.F32),
                ggml_type_name="F32",
            )

        def tensor_data(self, name: str):
            assert name == "token_embd.weight"
            return np.arange(15, dtype=np.float32).reshape(5, 3)

    rows, metadata = load_token_embedding_rows(FakeReader(), (4, 1, 4))

    np.testing.assert_array_equal(rows[0], np.array([12.0, 13.0, 14.0], dtype=np.float32))
    np.testing.assert_array_equal(rows[1], np.array([3.0, 4.0, 5.0], dtype=np.float32))
    np.testing.assert_array_equal(rows[2], rows[0])
    assert metadata["selected_token_ids"] == [4, 1, 4]


def test_audit_layer0_warm_conv_gdn_oracle_with_injected_inputs(tmp_path: Path) -> None:
    replay_inputs, weights, dims = _synthetic_replay_inputs_weights_dims()
    records, _summary = replay_conv_gdn_sequence(
        replay_inputs=replay_inputs,
        weights=weights,
        dimensions=dims,
        eps=1.0e-6,
    )
    target_capture = _target_capture_from_record(records[2], replay_inputs[2])
    projection_path = tmp_path / "projection.json"
    plan_path = tmp_path / "plan.json"
    position0_path = tmp_path / "position0.json"
    projection_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "classification": "layer0_projections_match_bf16_oracle_within_rounding",
                "model": "/tmp/model.gguf",
                "layer_id": 0,
                "position": 2,
                "prompt_tokens": [10, 11, 12],
            }
        )
    )
    plan_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "model_metadata": {
                    "dimensions": dims | {"conv_state_floats": 18, "recurrent_state_floats": 4},
                    "rms_norm_eps": 1.0e-6,
                },
            }
        )
    )
    position0_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "classification": "layer0_position0_conv_gdn_matches_oracle_exactly",
            }
        )
    )

    artifact = audit_layer0_warm_conv_gdn_oracle(
        projection_artifact_path=projection_path,
        plan_path=plan_path,
        position0_artifact_path=position0_path,
        model_path=Path("/tmp/model.gguf"),
        boundary_capture_fn=lambda *_args: target_capture,
        replay_input_builder=lambda *_args: (replay_inputs, {"source": "synthetic"}),
        conv_gdn_weight_loader=lambda *_args: weights,
        iteration=326,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == "layer0_warm_conv_gdn_matches_oracle_exactly"
    assert artifact["target_position"] == 2
    assert artifact["target_input_classification"] == "target_inputs_match_replay_exactly"
    assert artifact["replay_contract"]["starts_from_zero_state"] is True
    assert artifact["next_action"] == "continue_layer0_bisection_after_attn_out_or_residual"
    json.dumps(artifact)


def _dims() -> dict[str, int]:
    return {
        "ssm_group_count": 1,
        "ssm_time_step_rank": 1,
        "ssm_state_size": 2,
        "ssm_value_dim": 2,
        "ssm_inner_size": 2,
        "linear_qkv_width": 6,
        "ssm_conv_kernel": 3,
    }


def _synthetic_replay_inputs_weights_dims():
    dims = _dims()
    replay_inputs = []
    for scale in (1.0, 1.5, -0.5):
        replay_inputs.append(
            {
                "attn_norm_f32": np.full((4,), scale, dtype=np.float32),
                "linear_qkv_f32": np.asarray(
                    [0.25, -0.5, 0.75, -1.0, 0.5, -0.25],
                    dtype=np.float32,
                )
                * np.float32(scale),
                "linear_z_f32": np.asarray([0.5, -0.25], dtype=np.float32) * np.float32(scale),
                "ssm_alpha_f32": np.zeros((1,), dtype=np.float32),
                "ssm_beta_f32": np.asarray([0.25], dtype=np.float32),
            }
        )
    weights = {
        "ssm_conv1d": (np.ones((6, 3), dtype=np.float32), {"tensor_name": "conv"}),
        "ssm_dt_bias": (np.zeros((1,), dtype=np.float32), {"tensor_name": "dt"}),
        "ssm_a_log": (np.zeros((1,), dtype=np.float32), {"tensor_name": "a"}),
        "ssm_norm": (np.ones((2,), dtype=np.float32), {"tensor_name": "norm"}),
        "ssm_out": (np.eye(2, dtype=np.float32), {"tensor_name": "out"}),
    }
    return replay_inputs, weights, dims


def _target_capture_from_record(record, inputs):
    fields = {name: np.asarray(value, dtype=np.float32) for name, value in inputs.items()}
    for name in ("conv_out_f32", "recurrent_out_f32", "recurrent_bf16_f32", "attn_out_f32"):
        fields[name] = np.asarray(record[name], dtype=np.float32)
    return {"status": "captured", "summary": {"position": 2}, "fields": fields}


def _silu(values: np.ndarray) -> np.ndarray:
    return values / (np.float32(1.0) + np.exp(-values, dtype=np.float32))
