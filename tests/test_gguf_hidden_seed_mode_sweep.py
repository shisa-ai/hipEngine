from __future__ import annotations

import json
from pathlib import Path

from scripts.gguf_hidden_seed_mode_sweep import (
    build_mode_sweep_artifact,
    compare_named_vectors,
    parse_modes,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import pack_float32, sha256_bytes


def test_mode_sweep_reports_matching_mode(tmp_path: Path) -> None:
    prior = _write_prior_llamacpp_artifact(tmp_path, [1.0, 2.0])

    artifact = build_mode_sweep_artifact(
        llamacpp_artifact_path=prior,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(10, 11),
        position=1,
        modes=("prefill-bulk", "prefill-serial"),
        mode_capture_fn=_mode_capture({
            "prefill-bulk": [1.5, 2.0],
            "prefill-serial": [1.0, 2.0],
        }),
    )

    assert artifact["status"] == "matched"
    assert artifact["conclusion"] == "some_hipengine_seed_mode_matches_llamacpp"
    assert artifact["ranking"]["best_mode"]["mode"] == "prefill-serial"
    assert artifact["next_action"] == "route_mtp_seed_capture_through_matching_mode_or_fix_default"
    json.dumps(artifact)


def test_mode_sweep_reports_hip_modes_agree_but_mismatch(tmp_path: Path) -> None:
    prior = _write_prior_llamacpp_artifact(tmp_path, [1.0, 2.0])

    artifact = build_mode_sweep_artifact(
        llamacpp_artifact_path=prior,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(10, 11),
        position=1,
        modes=("prefill-bulk", "prefill-native", "step-serial"),
        mode_capture_fn=_mode_capture({
            "prefill-bulk": [1.25, 2.0],
            "prefill-native": [1.25, 2.0],
            "step-serial": [1.25, 2.0],
        }),
    )

    assert artifact["status"] == "mismatched"
    assert artifact["conclusion"] == "hipengine_seed_modes_agree_but_mismatch_llamacpp"
    assert artifact["ranking"]["hip_modes_match"] is True
    assert artifact["hipengine_pairwise"]["all_exact"] is True
    assert artifact["next_action"] == "audit_shared_bf16_activation_or_output_norm_precision"


def test_mode_sweep_reports_modes_diverge_and_mismatch(tmp_path: Path) -> None:
    prior = _write_prior_llamacpp_artifact(tmp_path, [1.0, 2.0])

    artifact = build_mode_sweep_artifact(
        llamacpp_artifact_path=prior,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(10, 11),
        position=1,
        modes=("prefill-bulk", "prefill-native"),
        mode_capture_fn=_mode_capture({
            "prefill-bulk": [1.25, 2.0],
            "prefill-native": [1.5, 2.0],
        }),
    )

    assert artifact["conclusion"] == "hipengine_seed_modes_diverge_and_mismatch_llamacpp"
    assert artifact["ranking"]["hip_modes_match"] is False
    assert artifact["next_action"] == "bisect_prefill_bulk_native_serial_seed_path"


def test_compare_named_vectors_handles_skipped_and_shape_mismatch() -> None:
    skipped = compare_named_vectors(
        "left",
        {"status": "skipped", "values": []},
        "right",
        {"status": "captured", "values": [1.0]},
        exact_atol=0.0,
    )
    mismatch = compare_named_vectors(
        "left",
        {"status": "captured", "values": [1.0, 2.0]},
        "right",
        {"status": "captured", "values": [1.0]},
        exact_atol=0.0,
    )

    assert skipped["available"] is False
    assert mismatch["shape_match"] is False
    assert mismatch["left_count"] == 2
    assert mismatch["right_count"] == 1


def test_parse_modes_and_final_position_guard(tmp_path: Path) -> None:
    prior = _write_prior_llamacpp_artifact(tmp_path, [1.0])

    assert parse_modes("prefill-bulk, step-serial") == ("prefill-bulk", "step-serial")
    try:
        build_mode_sweep_artifact(
            llamacpp_artifact_path=prior,
            model_path=tmp_path / "model.gguf",
            prompt_tokens=(10, 11, 12),
            position=1,
            modes=("prefill-bulk",),
            mode_capture_fn=_mode_capture({"prefill-bulk": [1.0]}),
        )
    except ValueError as exc:
        assert "final prompt token" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


def _write_prior_llamacpp_artifact(tmp_path: Path, values: list[float]) -> Path:
    binary = tmp_path / "llama.f32"
    meta = tmp_path / "llama.json"
    binary.write_bytes(pack_float32(values))
    meta.write_text(json.dumps({"kind": "llamacpp_hidden_seed_capture"}))
    artifact = {
        "llamacpp_capture": {
            "binary_path": str(binary),
            "metadata_path": str(meta),
            "sha256": sha256_bytes(pack_float32(values)),
        }
    }
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(artifact))
    return path


def _mode_capture(values_by_mode: dict[str, list[float]]):
    def capture(
        _model: Path,
        prompt_tokens: tuple[int, ...],
        position: int,
        _max_seq: int | None,
        mode: str,
    ):
        values = values_by_mode[mode]
        return {
            "status": "captured",
            "mode": mode,
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "next_token_id": 0,
            "contract": {"dtype": "FP32", "ready_for_mtp": True},
            "dtype": "FP32",
            "values": [float(value) for value in values],
        }

    return capture
