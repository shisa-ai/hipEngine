from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.gguf_mtp_forced_target_probe import (
    _bf16_roundtrip_f32,
    _boundary_array_summaries,
    _capture_moe_component_arrays,
    _scored_boundary_capture_record,
    build_arg_parser,
    diagnostic_env_payload,
)


def test_parser_exposes_direct_partial_replay_mode() -> None:
    args = build_arg_parser().parse_args(
        [
            "--trace",
            "trace.json",
            "--cycle",
            "12",
            "--state-lifecycle-compare",
            "--target-block-direct-partial-replay-mode",
            "serial-state-only",
        ]
    )

    assert args.state_lifecycle_compare is True
    assert args.target_block_direct_partial_replay_mode == "serial-state-only"


def test_parser_exposes_llama_direct_partial_commit_mode() -> None:
    args = build_arg_parser().parse_args(
        [
            "--trace",
            "trace.json",
            "--cycle",
            "12",
            "--state-lifecycle-compare",
            "--target-block-direct-partial-replay-mode",
            "direct-commit",
        ]
    )

    assert args.state_lifecycle_compare is True
    assert args.target_block_direct_partial_replay_mode == "direct-commit"


def test_parser_exposes_scored_layer_boundary_rows() -> None:
    args = build_arg_parser().parse_args(
        [
            "--trace",
            "trace.json",
            "--cycle",
            "3",
            "--scored-layer-boundary-row",
            "14:2",
            "--raw-scored-layer-boundary-row",
            "13:1",
        ]
    )

    assert args.scored_layer_boundary_row == [(14, 2)]
    assert args.raw_scored_layer_boundary_row == [(13, 1)]


def test_diagnostic_env_payload_records_f32_flags(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", "1")
    monkeypatch.setenv("UNRELATED_FLAG", "1")

    payload = diagnostic_env_payload()

    assert payload == {
        "HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE": "1",
        "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
    }


def test_capture_moe_component_arrays_emits_weighted_and_shared_gated_terms() -> None:
    capture = _fake_moe_capture()

    arrays = _capture_moe_component_arrays(capture)

    expected_weighted_rows = np.asarray(
        [[0.25, 0.5, 0.75, 1.0], [-1.5, 1.5, 0.0, 0.75]], dtype=np.float32
    )
    expected_selected_sum = np.asarray([-1.25, 2.0, 0.75, 1.75], dtype=np.float32)
    expected_selected_bf16 = _bf16_roundtrip_f32(expected_selected_sum)
    expected_gate = np.float32(1.0 / (1.0 + np.exp(np.float32(-0.25))))
    expected_shared_gated = np.asarray([0.5, -0.25, 0.0, 1.0], dtype=np.float32) * expected_gate

    np.testing.assert_allclose(arrays["moe_selected_down_weighted"], expected_weighted_rows)
    np.testing.assert_allclose(arrays["moe_selected_weighted_sum_f32"], expected_selected_sum)
    np.testing.assert_allclose(arrays["moe_selected_weighted_bf16"], expected_selected_bf16)
    np.testing.assert_allclose(arrays["moe_shared_gated"], expected_shared_gated)
    np.testing.assert_allclose(
        arrays["ffn_out_combined_from_components"],
        expected_selected_bf16 + expected_shared_gated,
    )


def test_boundary_array_summaries_include_fine_grained_moe_taps() -> None:
    capture = _fake_moe_capture()

    summaries, values = _boundary_array_summaries(
        capture,
        row_index=1,
        input_token=15495,
        position=123,
    )

    for name in (
        "linear_qkv",
        "linear_z",
        "ssm_alpha",
        "ssm_beta",
        "conv_out",
        "recurrent_out",
        "recurrent_bf16",
        "attn_post_norm",
        "attn_post_norm_bf16",
        "moe_router_logits",
        "moe_selected_swiglu",
        "moe_selected_down_weighted",
        "moe_selected_weighted_sum_f32",
        "moe_shared_intermediate",
        "moe_shared_gated",
        "ffn_out_combined_from_components",
        "post_moe_rounded_from_components",
    ):
        assert name in summaries
        assert name in values
        assert summaries[name]["depth"] == 1
        assert summaries[name]["token_id"] == 15495
        assert summaries[name]["position"] == 123


def test_scored_boundary_capture_record_formats_bulk_rows() -> None:
    arrays = {
        "layer_type_id": np.asarray([[0], [0]], dtype=np.int64),
        "hidden_in": np.asarray([[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]], dtype=np.float32),
        "attn_norm": np.asarray([[0.4, 0.5, 0.6, 0.7], [1.4, 1.5, 1.6, 1.7]], dtype=np.float32),
        "linear_qkv": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "linear_z": np.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
        "ssm_alpha": np.asarray([[0.01], [0.02]], dtype=np.float32),
        "ssm_beta": np.asarray([[0.03], [0.04]], dtype=np.float32),
        "conv_out": np.asarray([[0.9, 1.0], [1.1, 1.2]], dtype=np.float32),
        "recurrent_out": np.asarray([[1.3, 1.4], [1.5, 1.6]], dtype=np.float32),
        "recurrent_bf16": np.asarray([[1.25, 1.375], [1.5, 1.625]], dtype=np.float32),
        "attn_out": np.asarray([[0.2, 0.3, 0.4, 0.5], [1.2, 1.3, 1.4, 1.5]], dtype=np.float32),
        "attn_residual": np.asarray([[0.6, 0.7, 0.8, 0.9], [1.6, 1.7, 1.8, 1.9]], dtype=np.float32),
        "attn_post_norm_bf16": np.asarray([[0.8, 0.9, 1.0, 1.1], [1.8, 1.9, 2.0, 2.1]], dtype=np.float32),
        "moe_router_logits": np.asarray([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
        "moe_selected_experts": np.asarray([[1, 2], [7, 8]], dtype=np.int64),
        "moe_routing_weights": np.asarray([[0.25, 0.75], [0.4, 0.6]], dtype=np.float32),
        "moe_selected_swiglu": np.asarray(
            [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
            dtype=np.float32,
        ),
        "ffn_or_moe_down": np.asarray(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [[1.0, 1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 1.7]],
            ],
            dtype=np.float32,
        ),
        "moe_shared_intermediate": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "moe_shared_out": np.asarray([[0.2, 0.3, 0.4, 0.5], [1.2, 1.3, 1.4, 1.5]], dtype=np.float32),
        "moe_shared_gate": np.asarray([[0.1], [0.25]], dtype=np.float32),
        "layer_out": np.asarray([[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]], dtype=np.float32),
    }

    record = _scored_boundary_capture_record(
        layer_id=14,
        row_index=1,
        arrays=arrays,
        input_token=567,
        position=75,
        target_tokens=[11, 668],
        include_raw=True,
    )

    assert record["capture_source"] == "scored_target_block"
    assert record["capture"]["array_shapes"]["moe_router_logits"] == [2, 4]
    assert record["selected_experts"] == [7, 8]
    assert record["trace_target_token"] == 668
    assert "moe_router_logits" in record["summaries"]
    assert "ffn_out_combined_from_components" in record["summaries"]
    assert "moe_router_logits" in record["values"]


def _fake_moe_capture() -> SimpleNamespace:
    hidden = np.asarray([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    return SimpleNamespace(
        layer_id=31,
        hidden_size=4,
        is_moe=True,
        top_k=2,
        hidden_in_f32=hidden,
        attn_norm_f32=hidden + 1.0,
        linear_qkv_f32=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        linear_z_f32=np.asarray([0.6, 0.7, 0.8, 0.9], dtype=np.float32),
        ssm_alpha_f32=np.asarray([0.01, 0.02], dtype=np.float32),
        ssm_beta_f32=np.asarray([0.03, 0.04], dtype=np.float32),
        conv_out_f32=np.asarray([1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32),
        recurrent_out_f32=np.asarray([1.6, 1.7, 1.8, 1.9], dtype=np.float32),
        recurrent_bf16_f32=np.asarray([1.6015625, 1.703125, 1.796875, 1.8984375], dtype=np.float32),
        attn_out_f32=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        residual_f32=hidden + 0.25,
        post_norm_f32=hidden - 0.25,
        moe_router_logits_f32=np.asarray([0.5, 0.25, -0.5, 0.0], dtype=np.float32),
        moe_selected_intermediate_f32=np.asarray(
            [[0.1, 0.2], [0.3, 0.4]], dtype=np.float32
        ),
        ffn_or_moe_down_f32=np.asarray(
            [[1.0, 2.0, 3.0, 4.0], [-2.0, 2.0, 0.0, 1.0]], dtype=np.float32
        ),
        moe_routing_weights_f32=np.asarray([0.25, 0.75], dtype=np.float32),
        moe_shared_intermediate_f32=np.asarray([0.6, -0.4], dtype=np.float32),
        moe_shared_out_f32=np.asarray([0.5, -0.25, 0.0, 1.0], dtype=np.float32),
        moe_shared_gate_f32=np.asarray([0.25], dtype=np.float32),
        layer_out_f32=hidden + 0.5,
    )
