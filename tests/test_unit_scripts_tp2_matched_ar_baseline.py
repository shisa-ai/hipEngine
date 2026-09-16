"""CPU tests for the matched TP1-vs-TP2 AR baseline protocol (cell C1).

No ROCm: the accounting/ratio logic is pure and is exercised directly; the TP2
worker and the TP1 subprocess are not launched here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def matched():
    return _load("tp2_matched_ar_baseline", SCRIPTS / "tp2_matched_ar_baseline.py")


def _good_row(arm: str, **overrides):
    cell = {"output_tokens": 128, "warmup_decode_tokens": 0, "logits_per_decode_step": True}
    row = {
        "id": "p0",
        "category": "code",
        "prompt_tokens": 10,
        "output_tokens": 128,
        "timed_decode_transitions": 128,
        "context_position_at_timing_start": 10,
        "generated_count": 129 if arm in ("tp1-d0", "tp1-d1") else 128,
        "logits_per_decode_step": True,
        "eos_policy": "none",
        "finite_final_logits": True,
        "decode_ms": 100.0,
        "graph_capture_ms_included": 20.0 if arm in ("tp1-d0", "tp1-d1") else 0.0,
    }
    row.update(overrides)
    return row, cell


@pytest.mark.parametrize("arm", ["tp1-d0", "tp1-d1", "tp2"])
def test_validate_accounting_accepts_a_matching_row(matched, arm):
    row, cell = _good_row(arm)
    assert matched.validate_accounting(arm, row, cell=cell) == []


def test_validate_accounting_rejects_timed_transition_mismatch(matched):
    row, cell = _good_row("tp2", timed_decode_transitions=127)
    failures = matched.validate_accounting("tp2", row, cell=cell)
    assert any("timed_decode_transitions" in f for f in failures)


def test_validate_accounting_rejects_context_position_mismatch(matched):
    # A nonzero warmup would start timing one position later on TP1; the
    # declared cell forbids it and the validator must catch it.
    row, cell = _good_row("tp1-d0", context_position_at_timing_start=11)
    failures = matched.validate_accounting("tp1-d0", row, cell=cell)
    assert any("context_position_at_timing_start" in f for f in failures)


def test_validate_accounting_rejects_generated_count_convention(matched):
    # The prefill sample is in TP1's generated list and not TP2's.
    row, cell = _good_row("tp2", generated_count=129)
    failures = matched.validate_accounting("tp2", row, cell=cell)
    assert any("generated_count" in f for f in failures)
    row, cell = _good_row("tp1-d0", generated_count=128)
    failures = matched.validate_accounting("tp1-d0", row, cell=cell)
    assert any("generated_count" in f for f in failures)


def test_validate_accounting_rejects_nonfinite_and_eos_and_logits(matched):
    row, cell = _good_row("tp2", finite_final_logits=False)
    assert any("not finite" in f for f in matched.validate_accounting("tp2", row, cell=cell))
    row, cell = _good_row("tp2", eos_policy="stop")
    assert any("eos_policy" in f for f in matched.validate_accounting("tp2", row, cell=cell))
    row, cell = _good_row("tp2", logits_per_decode_step=False)
    assert any("logits_per_decode_step" in f for f in matched.validate_accounting("tp2", row, cell=cell))


def test_capture_is_excluded_consistently(matched):
    tp1, _ = _good_row("tp1-d0")
    tp2, _ = _good_row("tp2")
    assert matched.decode_ms_excluding_capture(tp1, arm="tp1-d0") == pytest.approx(80.0)
    assert matched.decode_ms_excluding_capture(tp2, arm="tp2") == pytest.approx(100.0)


def test_compute_ratios_uses_the_faster_tp1_arm(matched):
    def arm_agg(tok_s):
        return {
            "decode_tok_s_excluding_capture": tok_s,
            "total_output_tokens": 128,
            "decode_ms_excluding_capture": 100.0,
            "graph_capture_ms": 0.0,
        }

    reps = [
        {
            "rep": 0,
            "arms": {
                "tp1-d0": arm_agg(30.0),
                "tp1-d1": arm_agg(40.0),
                "tp2": arm_agg(50.0),
            },
        },
        {
            "rep": 1,
            "arms": {
                "tp1-d0": arm_agg(32.0),
                "tp1-d1": arm_agg(38.0),
                "tp2": arm_agg(48.0),
            },
        },
        {
            "rep": 2,
            "arms": {
                "tp1-d0": arm_agg(31.0),
                "tp1-d1": arm_agg(39.0),
                "tp2": arm_agg(52.0),
            },
        },
    ]
    ratios = matched.compute_ratios(reps)
    assert ratios["reps"] == 3
    assert ratios["per_rep"][0]["faster_tp1_arm"] == "tp1-d1"
    assert ratios["per_rep"][0]["tp2_vs_faster_tp1"] == pytest.approx(50.0 / 40.0)
    assert ratios["median_tp2_vs_faster_tp1"] == pytest.approx(48.0 / 38.0)
    assert ratios["min_tp2_vs_faster_tp1"] == pytest.approx(50.0 / 40.0)
    assert ratios["max_tp2_vs_faster_tp1"] == pytest.approx(52.0 / 39.0)


def test_main_requires_three_reps_and_a_mode(matched, capsys):
    with pytest.raises(SystemExit):
        matched.main(["--run", "--reps", "2"])
    with pytest.raises(SystemExit):
        matched.main([])


def test_cell_c1_is_predeclared(matched):
    cell = matched.CELL_C1
    assert cell["context_tokens"] == 128
    assert cell["output_tokens"] == 128
    assert cell["warmup_decode_tokens"] == 0
    assert cell["eos"] is None
    assert cell["logits_per_decode_step"] is True
