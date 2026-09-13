from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_layer0_conv_gdn_plan import (
    audit_kernel_state_dependencies,
    audit_llamacpp_linear_attention_anchors,
    audit_projection_prerequisite,
    audit_runtime_conv_gdn_sequence,
    build_layer0_conv_gdn_plan,
    decide_conv_gdn_strategy,
)


def test_projection_prerequisite_accepts_iter323_success_shape() -> None:
    prereq = audit_projection_prerequisite(_projection())

    assert prereq["ready"] is True
    assert prereq["facts"]["attn_norm_exact"] is True
    assert prereq["facts"]["linear_qkv_usable"] is True
    assert prereq["facts"]["linear_z_usable"] is True
    assert prereq["evidence"]["qkv_bf16_max_abs_diff"] == 0.000244140625
    assert prereq["evidence"]["linear_z_bf16_max_abs_diff"] == 0.0


def test_projection_prerequisite_rejects_unexplained_mismatch() -> None:
    projection = _projection()
    projection["projection_results"]["linear_qkv_f32"]["classification"] = (
        "projection_mismatch_after_bf16_contracted_oracle"
    )

    prereq = audit_projection_prerequisite(projection)

    assert prereq["ready"] is False
    assert prereq["facts"]["linear_qkv_usable"] is False


def test_runtime_conv_gdn_sequence_detects_order_and_capture_fields() -> None:
    audit = audit_runtime_conv_gdn_sequence(_runner_text())

    assert audit["ready"] is True
    assert audit["facts"]["conv_before_gdn"] is True
    assert audit["facts"]["gdn_before_recurrent_cast"] is True
    assert audit["facts"]["recurrent_cast_before_ssm_out"] is True
    assert audit["fields_to_compare"] == [
        "conv_out_f32",
        "recurrent_out_f32",
        "recurrent_bf16_f32",
        "attn_out_f32",
    ]
    assert "scratch.layer_conv_states[layer_id]" in audit["stateful_inputs"]


def test_kernel_state_dependencies_require_stateful_conv_and_gdn() -> None:
    audit = audit_kernel_state_dependencies(_conv_kernel_text(), _gdn_kernel_text())

    assert audit["ready"] is True
    assert audit["facts"]["conv_decode_lowp_reads_bf16_input"] is True
    assert audit["facts"]["conv_decode_outputs_silu"] is True
    assert audit["facts"]["gdn_updates_recurrent_state"] is True
    assert audit["position0_zero_state_feasible"] is True
    assert audit["warm_position_requires_state_replay_or_capture"] is True


def test_llamacpp_linear_attention_anchors_are_present() -> None:
    audit = audit_llamacpp_linear_attention_anchors(_llama_text())

    assert audit["ready"] is True
    assert "conv_output_silu" in audit["candidate_taps"]
    assert "linear_attn_out" in audit["candidate_taps"]


def test_decide_conv_gdn_strategy_selects_position_zero_when_ready() -> None:
    decision = decide_conv_gdn_strategy(
        projection_ready=True,
        runtime_ready=True,
        kernels_ready=True,
        llama_ready=True,
        model_ready=True,
    )

    assert decision["ready"] is True
    assert decision["selected_strategy"] == "position0_stateless_conv_gdn_oracle_first"
    assert decision["next_action"] == "build_position0_layer0_conv_gdn_oracle"
    assert "position-0 oracle" in decision["reason"]


def test_build_layer0_conv_gdn_plan_from_synthetic_inputs(tmp_path: Path) -> None:
    projection_path = tmp_path / "projection.json"
    runner_path = tmp_path / "runner.py"
    conv_path = tmp_path / "conv.hip"
    gdn_path = tmp_path / "gdn.hip"
    llama_path = tmp_path / "qwen35moe.cpp"
    projection_path.write_text(json.dumps(_projection()))
    runner_path.write_text(_runner_text())
    conv_path.write_text(_conv_kernel_text())
    gdn_path.write_text(_gdn_kernel_text())
    llama_path.write_text(_llama_text())

    artifact = build_layer0_conv_gdn_plan(
        projection_artifact_path=projection_path,
        runner_path=runner_path,
        conv_kernel_path=conv_path,
        gdn_kernel_path=gdn_path,
        llamacpp_qwen35moe_path=llama_path,
        model_path=tmp_path / "model.gguf",
        metadata_loader=lambda _path: _model_metadata(),
        iteration=324,
    )

    assert artifact["status"] == "ready"
    assert artifact["conclusion"] == "layer0_conv_gdn_plan_ready"
    assert artifact["decision"]["selected_strategy"] == (
        "position0_stateless_conv_gdn_oracle_first"
    )
    assert artifact["next_probe_plan"]["first_probe"] == (
        "position0_conv_out_recurrent_out_attn_out"
    )
    assert artifact["model_metadata"]["dimensions"]["conv_state_floats"] == 32768
    assert artifact["model_metadata"]["dimensions"]["recurrent_state_floats"] == 524288
    assert artifact["constraints"]["must_not_change_hot_path"] is True
    assert artifact["external_checkout_modified"] is False
    assert artifact["next_action"] == "build_position0_layer0_conv_gdn_oracle"
    json.dumps(artifact)


def _projection() -> dict[str, object]:
    return {
        "status": "ready",
        "classification": "layer0_projections_match_bf16_oracle_within_rounding",
        "model": "/models/model.gguf",
        "layer_id": 0,
        "position": 16,
        "token_id": 271,
        "attn_norm_oracle": {"delta_vs_hip": {"exact_match": True}},
        "projection_results": {
            "linear_qkv_f32": {
                "classification": "projection_matches_bf16_oracle_within_one_bf16_step",
                "delta_bf16_oracle_vs_hip": {"max_abs_diff": 0.000244140625},
            },
            "linear_z_f32": {
                "classification": "projection_matches_bf16_oracle_exactly",
                "delta_bf16_oracle_vs_hip": {"max_abs_diff": 0.0},
            },
        },
        "next_action": "continue_layer0_bf16_bisection_at_conv_or_gdn_state",
    }


def _runner_text() -> str:
    return '''class Runner:
    def _run_linear_attention_attn_only(self):
        gguf_rmsnorm_bf16_f32_weight()
        launch_gguf_linear_pair(layer.weight("attn_qkv"), layer.weight("attn_gate"))
        layer.weight("ssm_alpha")
        layer.weight("ssm_beta")
        qwen35_linear_attn_conv_decode_bf16()
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16()
        f32_to_bf16()
        layer.weight("ssm_out")

    def capture_linear_attention_boundary(self):
        return Qwen35GGUFLinearAttentionBoundaryCapture(
            conv_out_f32=conv,
            recurrent_out_f32=rec,
            recurrent_bf16_f32=rec_bf16,
            attn_out_f32=attn,
        )
'''


def _conv_kernel_text() -> str:
    return '''template <typename scalar_t>
__global__ void qwen35_linear_attn_conv_decode_lowp_kernel(
    const scalar_t* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out) {
  for (int64_t idx = 0; idx < kernel_size - 1; ++idx) {
    const float value = conv_state[offset + idx + 1];
    conv_state[offset + idx] = value;
  }
  const float newest = scalar_to_float_qwen35(hidden_states[channel]);
  conv_state[offset + kernel_size - 1] = newest;
  out[channel] = silu_f32(acc);
}
'''


def _gdn_kernel_text() -> str:
    return '''__global__ void qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* out) {
  const float beta = sigmoid_f32(scalar_to_float_qwen35(b[v_head]));
  const float decay = expf(-expf(a_log[v_head]) *
      softplus_f32(scalar_to_float_qwen35(a[v_head]) + dt_bias[v_head]));
  float x = recurrent_state[state_col + idx * head_v_dim] * decay;
  const float new_state = recurrent_state[state_offset] * decay + k_norm * delta;
  recurrent_state[state_offset] = new_state;
  out[pos] = out_shared[value_idx] * inv_rms *
      norm_weight[value_idx] * silu_f32(scalar_to_float_qwen35(gate[pos]));
}
'''


def _llama_text() -> str:
    return '''void f() {
    ggml_tensor * conv_input = build_conv_state(
        inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);
    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);
    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, a, b, c, d, e, f, g, h);
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, a, b, c, d, e, f, g, h);
    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);
    ggml_tensor * output = build_recurrent_attn(
        inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);
    cb(final_output, "final_output", il);
    cb(cur, "linear_attn_out", il);
}
'''


def _model_metadata() -> dict[str, object]:
    return {
        "ready": True,
        "dimensions": {
            "ssm_conv_kernel": 4,
            "ssm_group_count": 16,
            "ssm_inner_size": 4096,
            "ssm_state_size": 128,
            "ssm_time_step_rank": 32,
            "linear_qkv_width": 8192,
            "ssm_value_dim": 128,
            "conv_state_floats": 32768,
            "recurrent_state_floats": 524288,
        },
        "tensors": {},
        "facts": {"qkv_width_matches_conv_channels": True},
    }
