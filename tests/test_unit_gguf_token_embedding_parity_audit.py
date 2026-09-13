from __future__ import annotations

from pathlib import Path

from scripts.gguf_token_embedding_parity_audit import (
    bf16_round_to_f32,
    build_token_embedding_parity_artifact,
    conclude,
    next_action,
    parse_prompt_tokens,
)


def test_build_artifact_explains_layer0_drift_as_bf16_output(tmp_path: Path) -> None:
    raw = [0.004868030548095703, -0.0033245086669921875, 1.0]
    rounded = bf16_round_to_f32(raw)

    artifact = build_token_embedding_parity_artifact(
        model_path=tmp_path / "missing.gguf",
        layer_sweep_path=tmp_path / "missing-layer-sweep.json",
        layer=0,
        position=2,
        token_id=271,
        llamacpp_values=raw,
        raw_dequant_values=raw,
        hipengine_values=rounded,
        tensor_info_override={
            "shape": [1000, len(raw)],
            "ggml_type_name": "Q8_0",
        },
    )

    assert artifact["status"] == "explained"
    assert artifact["conclusion"] == "layer0_drift_is_bf16_embedding_output"
    assert artifact["comparisons"]["llamacpp_vs_raw_dequant"]["exact_match"] is True
    assert artifact["comparisons"]["hipengine_vs_bf16_round"]["exact_match"] is True
    assert artifact["comparisons"]["llamacpp_vs_hipengine"]["exact_match"] is False
    assert artifact["next_action"] == "decide_embedding_hidden_precision_for_llamacpp_exact_parity"


def test_build_artifact_flags_raw_embedding_mismatch(tmp_path: Path) -> None:
    artifact = build_token_embedding_parity_artifact(
        model_path=tmp_path / "missing.gguf",
        layer_sweep_path=tmp_path / "missing-layer-sweep.json",
        layer=0,
        position=1,
        token_id=7,
        llamacpp_values=[1.0, 2.0],
        raw_dequant_values=[1.25, 2.0],
        hipengine_values=[1.25, 2.0],
        tensor_info_override={"shape": [8, 2]},
    )

    assert artifact["status"] == "mismatched"
    assert artifact["conclusion"] == "llamacpp_layer0_does_not_match_raw_token_embedding"
    assert artifact["next_action"] == "inspect_llamacpp_layer0_capture_row_selection"


def test_build_artifact_records_hip_skipped_partial_result(tmp_path: Path) -> None:
    raw = [0.004868030548095703, -0.0033245086669921875]

    artifact = build_token_embedding_parity_artifact(
        model_path=tmp_path / "missing.gguf",
        layer_sweep_path=tmp_path / "missing-layer-sweep.json",
        layer=0,
        position=1,
        token_id=7,
        llamacpp_values=raw,
        raw_dequant_values=raw,
        tensor_info_override={"shape": [8, 2]},
        skip_hip=True,
    )

    assert artifact["status"] == "partial"
    assert artifact["hipengine_capture_status"] == "skipped_by_flag"
    assert artifact["conclusion"] == "raw_match_bf16_rounding_suspect_hip_skipped"
    assert artifact["next_action"] == "rerun_token_embedding_audit_with_hip_capture"


def test_conclude_detects_embedding_kernel_not_bf16_round() -> None:
    result = conclude(
        comparisons={
            "llamacpp_vs_raw_dequant": {"exact_match": True},
            "llamacpp_vs_bf16_round": {"exact_match": False},
            "bf16_round_vs_raw_dequant": {"exact_match": False},
            "hipengine_vs_bf16_round": {"exact_match": False},
        },
        hip_status="captured",
    )

    assert result == "hip_embedding_kernel_differs_from_bf16_round"
    assert next_action(result) == "debug_gguf_embedding_kernel_dequantization"


def test_parse_prompt_tokens_rejects_empty() -> None:
    assert parse_prompt_tokens("1, 2,3") == (1, 2, 3)
    try:
        parse_prompt_tokens(" , ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")
