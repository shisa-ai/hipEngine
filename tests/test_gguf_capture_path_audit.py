from __future__ import annotations

import json
from pathlib import Path

from scripts.gguf_capture_path_audit import (
    audit_runner_source,
    build_capture_path_audit,
    conclude,
    extract_function_body,
    next_action,
    parse_layers,
)


def test_audit_runner_source_detects_embedding_and_preceding_layer_path() -> None:
    audit = audit_runner_source(_runner_source())
    facts = audit["facts"]

    assert facts["capture_calls_set_token_id_device"] is True
    assert facts["set_token_id_launches_embedding"] is True
    assert facts["embedding_launches_gguf_embedding"] is True
    assert facts["capture_sets_embedding"] is True
    assert facts["capture_replays_preceding_layers"] is True
    assert facts["capture_hidden_tap_copies_target_src_ptr"] is True
    assert audit["anchors"]["capture_attention_layer"] == 2


def test_extract_function_body_stops_at_next_method() -> None:
    text = "class X:\n    def a(self):\n        one()\n    def b(self):\n        two()\n"

    body = extract_function_body(text, "a")

    assert "one()" in body
    assert "def b" not in body


def test_conclusion_prefers_precision_contractors_after_other_causes_ruled_out() -> None:
    source = {"facts": {"capture_sets_embedding": True}}
    precision = {"count": 2}
    evidence = {
        "tap_compare_best_key": "hidden_in_f32",
        "layer_sweep_best_layer": 3,
        "capture_layer_id": 3,
    }

    result = conclude(source_audit=source, precision=precision, evidence=evidence)

    assert result == "precision_contractions_or_preceding_layer_math_suspect"
    assert next_action(result) == (
        "run_earliest_layer_hidden_in_sweep_and_audit_precision_contractors"
    )


def test_conclusion_flags_embedding_setup_first() -> None:
    result = conclude(
        source_audit={"facts": {"capture_sets_embedding": False}},
        precision={"count": 0},
        evidence={
            "tap_compare_best_key": "hidden_in_f32",
            "layer_sweep_best_layer": 3,
            "capture_layer_id": 3,
        },
    )

    assert result == "capture_embedding_setup_suspect"


def test_build_capture_path_audit_with_missing_model(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    capture = tmp_path / "capture.json"
    tap = tmp_path / "tap.json"
    sweep = tmp_path / "sweep.json"
    runner.write_text(_runner_source())
    capture.write_text(
        json.dumps(
            {
                "layer_id": 3,
                "position": 16,
                "run_preceding_layers": True,
                "capture_summary": {"preceding_layer_count": 3},
            }
        )
    )
    tap.write_text(
        json.dumps(
            {
                "conclusion": "target_hidden_in_is_closest_but_mismatched",
                "ranking": {"best_same_width": {"key": "hidden_in_f32", "rmse": 0.1}},
            }
        )
    )
    sweep.write_text(
        json.dumps(
            {
                "conclusion": "target_layer_best_but_mismatched",
                "ranking": {"best_selected": {"layer": 3, "rmse": 0.1}},
            }
        )
    )

    artifact = build_capture_path_audit(
        runner_path=runner,
        model_path=tmp_path / "missing.gguf",
        capture_path=capture,
        tap_compare_path=tap,
        layer_sweep_path=sweep,
        layers=(0, 1, 2, 3),
    )

    assert artifact["status"] == "audited"
    assert artifact["precision_audit"]["available"] is False
    assert artifact["conclusion"] == "preceding_layer_math_or_materialization_suspect"
    assert artifact["next_action"] == (
        "run_earliest_layer_hidden_in_sweep_between_hipengine_and_llamacpp"
    )


def test_parse_layers_supports_ranges() -> None:
    assert parse_layers("0-3,7,9-8") == (0, 1, 2, 3, 7, 9, 8)


def _runner_source() -> str:
    return '''class Session:
    def capture_attention_layer(self, token_id, *, position, layer_id, run_preceding_layers=False):
        self._set_token_id_device(int(token_id), stream=0)
        src = self._hidden_a
        dst = self._hidden_b
        if run_preceding_layers:
            for prev_layer_id, prev_layer_type in enumerate(layer_types[:layer_id]):
                self.runner._run_full_attention_layer(
                    prev_layer_id, src.ptr, dst.ptr, self.scratch, position=position)
        hidden_in_f32=_copy_bf16_ptr_to_host_f32(target_src_ptr, hidden_size)

    def _set_token_id_device(self, token_id, *, stream=0):
        self._set_token_embedding_from_ptr(self._token_buf.ptr, stream=stream)

    def _set_token_embedding_from_ptr(self, token_id_ptr, *, stream=0):
        launch_gguf_embedding(self.runner.weights.root("token_embedding"), token_id_ptr)

    def _run_current_hidden_to_final_hidden(self, *, position, stream=0):
        src = self._hidden_a
'''
