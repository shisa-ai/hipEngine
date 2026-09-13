"""CPU-only regression coverage for rowtile gate result handling."""

import json
import sys

import pytest

from scripts import qwen38_gfx1151_q4_dense_rowtile_gate as gate


@pytest.mark.parametrize(
    "arms,passed", [("admitted", False), ("excluded", False),
                    ("excluded,admitted", True), ("admitted,excluded", True)]
)
def test_arm_selection_writes_diagnostic_without_false_qualification(
    monkeypatch, tmp_path, arms, passed
):
    output = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys, "argv",
        ["gate", "--arms", arms, "--rows", "2", "--output", str(output)],
    )
    monkeypatch.setattr(gate, "register_gfx1151_kernels", lambda: None)
    monkeypatch.setattr(gate, "_configure", lambda admitted: {})
    monkeypatch.setattr(gate, "_restore_counters", lambda: None)
    monkeypatch.setattr(gate, "_calls_by_rows", lambda: {})
    monkeypatch.setattr(gate, "_mem_available_bytes", lambda: None)
    monkeypatch.setattr(gate, "_arm_args", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gate, "run", lambda args: {
            "status": "eq_ok", "passed": True,
            "prompt_rows": [{"id": "a", "category": "code"},
                            {"id": "b", "category": "general_en"}],
            "independent_c1_token_ids": [[1], [2]],
            "runs": [{"generated_token_ids": [[1], [2]],
                      "row_equal": [True, True], "all_rows_equal": True,
                      "native_caware_decode": True,
                      "serial_decode_fallback": False}],
        },
    )
    assert gate.main() == (0 if passed else 1)
    payload = json.loads(output.read_text())
    assert payload["gate_passed"] is passed
    assert len(payload["runs"]) == len(arms.split(","))
    assert len(payload["verdicts"]) == int(passed)
