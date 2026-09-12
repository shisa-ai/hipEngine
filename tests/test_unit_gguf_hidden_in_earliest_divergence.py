from __future__ import annotations

import json
from pathlib import Path

from scripts.gguf_hidden_in_earliest_divergence import (
    build_earliest_divergence_artifact,
    compare_vectors,
    layer_from_slot,
    parse_prompt_tokens,
    precision_records_before_layer,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import pack_float32, sha256_bytes


def test_compare_vectors_detects_exact_and_mismatch() -> None:
    exact = compare_vectors([1.0, 2.0], [1.0, 2.0], exact_atol=0.0)
    mismatch = compare_vectors([1.0, 2.5], [1.0, 2.0], exact_atol=0.0)

    assert exact["exact_match"] is True
    assert exact["max_abs_diff"] == 0.0
    assert mismatch["exact_match"] is False
    assert mismatch["max_abs_diff"] == 0.5
    assert mismatch["rmse"] > 0.0


def test_build_artifact_finds_first_divergence_after_layer0(tmp_path: Path) -> None:
    sweep_path = _write_layer_sweep(
        tmp_path,
        {
            0: [1.0, 2.0],
            1: [3.0, 4.0],
            2: [5.0, 6.0],
        },
    )
    precision = {
        "available": True,
        "count": 2,
        "records": [
            {"slot_path": "layers.0.ffn_gate_inp"},
            {"slot_path": "layers.1.ssm_alpha"},
        ],
    }

    artifact = build_earliest_divergence_artifact(
        model_path=tmp_path / "missing.gguf",
        layer_sweep_path=sweep_path,
        layers=[0, 1, 2],
        prompt_tokens=(10, 11, 12),
        position=2,
        hipengine_captures={
            0: [1.0, 2.0],
            1: [3.25, 4.0],
            2: [5.5, 6.0],
        },
        precision_audit=precision,
    )

    assert artifact["status"] == "mismatched"
    assert artifact["ranking"]["first_mismatch_layer"] == 1
    assert artifact["ranking"]["last_exact_prefix_layer"] == 0
    assert artifact["conclusion"] == "first_hidden_in_divergence_after_layer_0"
    assert artifact["next_action"] == "compare_layer_0_internal_taps_and_precision_contractors"
    layer1 = artifact["layer_results"][1]
    assert layer1["preceding_precision_contractions"]["slot_paths"] == [
        "layers.0.ffn_gate_inp"
    ]
    assert layer1["current_layer_precision_contractions"]["slot_paths"] == [
        "layers.1.ssm_alpha"
    ]
    json.dumps(artifact)


def test_build_artifact_flags_layer0_embedding_mismatch(tmp_path: Path) -> None:
    sweep_path = _write_layer_sweep(tmp_path, {0: [1.0, 2.0], 1: [3.0, 4.0]})

    artifact = build_earliest_divergence_artifact(
        model_path=tmp_path / "missing.gguf",
        layer_sweep_path=sweep_path,
        layers=[0, 1],
        prompt_tokens=(10, 11),
        position=1,
        hipengine_captures={0: [1.125, 2.0], 1: [3.0, 4.0]},
        precision_audit={"available": True, "count": 0, "records": []},
    )

    assert artifact["ranking"]["first_mismatch_layer"] == 0
    assert artifact["conclusion"] == "layer0_hidden_in_mismatch_embedding_or_capture"
    assert artifact["next_action"] == (
        "compare_token_embedding_weight_materialization_and_embedding_kernel"
    )


def test_build_artifact_reports_all_requested_layers_match(tmp_path: Path) -> None:
    sweep_path = _write_layer_sweep(tmp_path, {0: [1.0, 2.0], 1: [3.0, 4.0]})

    artifact = build_earliest_divergence_artifact(
        model_path=tmp_path / "missing.gguf",
        layer_sweep_path=sweep_path,
        layers=[0, 1],
        prompt_tokens=(10, 11),
        position=1,
        hipengine_captures={0: [1.0, 2.0], 1: [3.0, 4.0]},
        precision_audit={"available": True, "count": 0, "records": []},
    )

    assert artifact["status"] == "matched"
    assert artifact["ranking"]["first_mismatch_layer"] is None
    assert artifact["conclusion"] == "hidden_in_matches_through_requested_layers"


def test_precision_helpers_and_prompt_parser() -> None:
    records = [
        {"slot_path": "layers.0.ffn_gate_inp"},
        {"slot_path": "layers.2.ssm_alpha"},
        {"slot_path": "other"},
    ]

    assert layer_from_slot("layers.12.foo") == 12
    assert layer_from_slot("bad") == -1
    assert precision_records_before_layer(records, target_layer=2)["slot_paths"] == [
        "layers.0.ffn_gate_inp"
    ]
    assert parse_prompt_tokens("1, 2,3") == (1, 2, 3)


def _write_layer_sweep(tmp_path: Path, vectors: dict[int, list[float]]) -> Path:
    results = []
    for layer, values in vectors.items():
        binary = tmp_path / f"layer{layer}.f32"
        binary.write_bytes(pack_float32(values))
        results.append(
            {
                "layer": layer,
                "status": "mismatched",
                "binary_path": str(binary),
                "capture_sha256": sha256_bytes(binary.read_bytes()),
                "selected_row": 2,
            }
        )
    path = tmp_path / "layer-sweep.json"
    path.write_text(json.dumps({"layer_results": results}))
    return path
