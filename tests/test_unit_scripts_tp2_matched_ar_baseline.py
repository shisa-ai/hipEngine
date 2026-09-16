"""CPU tests for the matched TP1-vs-TP2 AR diagnostic protocol (cell C1-natural).

No ROCm: accounting/ratio/suite logic is pure and is exercised directly, and
``main`` is driven with monkeypatched arm runners so no subprocess is launched.
"""

from __future__ import annotations

import importlib.util
import json
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


def matched_cell() -> dict:
    return {
        "name": "C1-natural",
        "context": "natural",
        "output_tokens": 128,
        "warmup_decode_tokens": 0,
        "sampling": "greedy",
        "eos": None,
        "logits_per_decode_step": True,
        "graph_replay_decode": True,
        "tp1_product_baseline": False,
    }


def _good_row(matched, arm: str, *, prompt_id: str = "p0", category: str = "code", **overrides):
    cell = matched_cell()
    sampled = list(range(100, 100 + cell["output_tokens"]))
    row = {
        "id": prompt_id,
        "category": category,
        "prompt_tokens": 10,
        "prompt_token_sha256": matched.token_tuple_sha256([1, 2, 3]),
        "context_tokens": 10,
        "output_tokens": cell["output_tokens"],
        "timed_decode_transitions": cell["output_tokens"],
        "context_position_at_timing_start": 10,
        "graph_effective": True,
        "logits_per_decode_step": True,
        "eos_policy": "none",
        "finite_final_logits": True,
        "finite_all_decode_logits": True,
        "prefill_sample_id": 7,
        "sampled_output_ids": sampled,
        "sampled_output_sha256": matched._ids_hash(sampled),
        "total_generation_ms": 120.0,
        "prefill_ms": 20.0,
        "capture_ms": 20.0 if arm in matched.TP1_ARMS else 0.0,
        "destroy_ms": 0.0,
    }
    row.update(overrides)
    return row, cell


@pytest.mark.parametrize("arm", ["tp1-d0", "tp1-d1", "tp2"])
def test_validate_accounting_accepts_a_matching_row(matched, arm):
    row, cell = _good_row(matched, arm)
    assert matched.validate_accounting(arm, row, cell=cell) == []


def test_validate_accounting_rejects_timed_transition_mismatch(matched):
    row, cell = _good_row(matched, "tp2", timed_decode_transitions=127)
    assert any("timed_decode_transitions" in f for f in matched.validate_accounting("tp2", row, cell=cell))


def test_validate_accounting_rejects_context_position_mismatch(matched):
    row, cell = _good_row(matched, "tp1-d0", context_position_at_timing_start=11)
    assert any(
        "context_position_at_timing_start" in f
        for f in matched.validate_accounting("tp1-d0", row, cell=cell)
    )


def test_validate_accounting_rejects_natural_context_mismatch(matched):
    row, cell = _good_row(matched, "tp2", context_tokens=9)
    assert any("natural-length" in f for f in matched.validate_accounting("tp2", row, cell=cell))


def test_validate_accounting_rejects_fixed_context_cell_as_unimplemented(matched):
    row, cell = _good_row(matched, "tp2")
    cell = dict(cell, context=128)
    failures = matched.validate_accounting("tp2", row, cell=cell)
    assert any("not implemented" in f for f in failures)


def test_validate_accounting_requires_graph_effective(matched):
    row, cell = _good_row(matched, "tp1-d0", graph_effective=False)
    assert any("graph replay not effective" in f for f in matched.validate_accounting("tp1-d0", row, cell=cell))


def test_validate_accounting_requires_finite_all_decode_logits(matched):
    row, cell = _good_row(matched, "tp2", finite_all_decode_logits=False)
    assert any("all decode-step logits" in f for f in matched.validate_accounting("tp2", row, cell=cell))


def test_validate_accounting_rejects_sampled_count_and_hash(matched):
    row, cell = _good_row(matched, "tp2")
    bad = dict(row, sampled_output_ids=row["sampled_output_ids"][:-1])
    assert any("sampled_output_ids" in f for f in matched.validate_accounting("tp2", bad, cell=cell))
    bad = dict(row, sampled_output_sha256="0" * 64)
    assert any("sampled_output_sha256" in f for f in matched.validate_accounting("tp2", bad, cell=cell))


def test_validate_accounting_requires_positive_adjusted_window(matched):
    row, cell = _good_row(matched, "tp1-d0", total_generation_ms=10.0, capture_ms=10.0)
    assert any("adjusted window" in f for f in matched.validate_accounting("tp1-d0", row, cell=cell))
    row, cell = _good_row(matched, "tp2", total_generation_ms=float("nan"))
    assert any("total_generation_ms" in f for f in matched.validate_accounting("tp2", row, cell=cell))


def test_validate_accounting_rejects_nonfinite_and_eos_and_logits(matched):
    row, cell = _good_row(matched, "tp2", finite_final_logits=False)
    assert any("not finite" in f for f in matched.validate_accounting("tp2", row, cell=cell))
    row, cell = _good_row(matched, "tp2", eos_policy="stop")
    assert any("eos_policy" in f for f in matched.validate_accounting("tp2", row, cell=cell))
    row, cell = _good_row(matched, "tp2", logits_per_decode_step=False)
    assert any("logits_per_decode_step" in f for f in matched.validate_accounting("tp2", row, cell=cell))


def test_validate_accounting_rejects_missing_and_nonfinite_fields_without_raising(matched):
    row, cell = _good_row(matched, "tp2")
    del row["total_generation_ms"]
    assert any("total_generation_ms" in f for f in matched.validate_accounting("tp2", row, cell=cell))
    row, cell = _good_row(matched, "tp2", total_generation_ms=None)
    assert any("total_generation_ms" in f for f in matched.validate_accounting("tp2", row, cell=cell))
    row, cell = _good_row(matched, "tp2", prompt_tokens=None)
    assert any("prompt_tokens" in f for f in matched.validate_accounting("tp2", row, cell=cell))
    row, cell = _good_row(matched, "tp2")
    del row["sampled_output_ids"]
    assert any("sampled_output_ids" in f for f in matched.validate_accounting("tp2", row, cell=cell))
    row, cell = _good_row(matched, "tp2")
    del row["prompt_token_sha256"]
    assert any("prompt_token_sha256" in f for f in matched.validate_accounting("tp2", row, cell=cell))


def test_validate_arm_rows_rejects_zero_duplicate_missing_extra_and_reorder(matched):
    ids = ["a", "b"]
    cats = {"a": "code", "b": "general_en"}
    good = [_good_row(matched, "tp2", prompt_id="a", category="code")[0],
            _good_row(matched, "tp2", prompt_id="b", category="general_en")[0]]
    assert matched.validate_arm_rows("tp2", good, expected_ids=ids, expected_categories=cats) == []
    assert any("zero prompt rows" in f for f in matched.validate_arm_rows("tp2", [], expected_ids=ids, expected_categories=cats))
    dup = [good[0], good[0]]
    failures = matched.validate_arm_rows("tp2", dup, expected_ids=ids, expected_categories=cats)
    assert any("duplicate" in f for f in failures)
    missing = [good[0]]
    failures = matched.validate_arm_rows("tp2", missing, expected_ids=ids, expected_categories=cats)
    assert any("coverage/order" in f for f in failures)
    extra = good + [_good_row(matched, "tp2", prompt_id="c", category="code")[0]]
    failures = matched.validate_arm_rows("tp2", extra, expected_ids=ids, expected_categories=cats)
    assert any("coverage/order" in f for f in failures)
    reordered = [good[1], good[0]]
    failures = matched.validate_arm_rows("tp2", reordered, expected_ids=ids, expected_categories=cats)
    assert any("coverage/order" in f for f in failures)
    changed = [_good_row(matched, "tp2", prompt_id="a", category="mixed_ja_en")[0], good[1]]
    failures = matched.validate_arm_rows("tp2", changed, expected_ids=ids, expected_categories=cats)
    assert any("category" in f for f in failures)


def test_cross_arm_token_hashes_requires_identical_tokenization(matched):
    rows_a = [_good_row(matched, "tp1-d0", prompt_id="a")[0]]
    rows_b = [_good_row(matched, "tp2", prompt_id="a")[0]]
    assert matched.cross_arm_token_hashes({"tp1-d0": rows_a, "tp2": rows_b}, expected_ids=["a"]) == []
    rows_b[0]["prompt_token_sha256"] = "different"
    failures = matched.cross_arm_token_hashes({"tp1-d0": rows_a, "tp2": rows_b}, expected_ids=["a"])
    assert any("differs across arms" in f for f in failures)


def test_prompt_ids_sha256_is_deterministic_and_not_python_hash(matched):
    assert matched.prompt_ids_sha256(["a", "b"]) == matched.prompt_ids_sha256(["a", "b"])
    assert matched.prompt_ids_sha256(["a", "b"]) != matched.prompt_ids_sha256(["b", "a"])
    # It must not depend on PYTHONHASHSEED; a second interpreter must agree.
    import subprocess

    snippet = (
        "import importlib.util,sys;"
        f"sys.path.insert(0,{str(SCRIPTS)!r});"
        "spec=importlib.util.spec_from_file_location('m',"
        f"{str(SCRIPTS / 'tp2_matched_ar_baseline.py')!r});"
        "m=importlib.util.module_from_spec(spec);sys.modules['m']=m;spec.loader.exec_module(m);"
        "print(m.prompt_ids_sha256(['a','b']))"
    )
    import os

    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == matched.prompt_ids_sha256(["a", "b"])


def test_adjusted_ms_removes_capture_and_destroy(matched):
    row, _ = _good_row(matched, "tp1-d0")
    assert matched.adjusted_ms(row) == pytest.approx(100.0)
    row, _ = _good_row(matched, "tp2")
    assert matched.adjusted_ms(row) == pytest.approx(120.0)


def _rep(matched, index: int, *, tok_s=(30.0, 40.0, 50.0), run_prefix="run", complete=True, independent=True):
    arms = {
        "tp1-d0": {"adjusted_tok_s": tok_s[0]},
        "tp1-d1": {"adjusted_tok_s": tok_s[1]},
        "tp2": {"adjusted_tok_s": tok_s[2]},
    }
    return {
        "rep": index,
        "independent": independent,
        "complete": complete,
        "run_ids": {arm: f"{run_prefix}-{index}-{arm}" for arm in matched.ARMS},
        "arms": arms,
    }


def test_compute_ratios_uses_the_faster_tp1_arm(matched):
    reps = [_rep(matched, 0), _rep(matched, 1, tok_s=(32.0, 38.0, 48.0)), _rep(matched, 2, tok_s=(31.0, 39.0, 52.0))]
    ratios = matched.compute_ratios(reps)
    assert ratios["reps"] == 3
    assert ratios["per_rep"][0]["faster_tp1_arm"] == "tp1-d1"
    assert ratios["per_rep"][0]["tp2_vs_faster_tp1"] == pytest.approx(50.0 / 40.0)
    assert ratios["median_tp2_vs_faster_tp1"] == pytest.approx(48.0 / 38.0)
    assert ratios["min_tp2_vs_faster_tp1"] == pytest.approx(50.0 / 40.0)
    assert ratios["max_tp2_vs_faster_tp1"] == pytest.approx(52.0 / 39.0)


def test_compute_ratios_is_none_for_incomplete_or_reused_reps(matched):
    assert matched.compute_ratios([_rep(matched, 0, complete=False)]) is None
    incomplete = _rep(matched, 0)
    del incomplete["arms"]["tp1-d1"]
    assert matched.compute_ratios([incomplete]) is None
    # Two reps carrying the same execution id is reuse, not independence.
    reused = [_rep(matched, 0), _rep(matched, 1)]
    reused[1]["run_ids"] = dict(reused[0]["run_ids"])
    assert matched.compute_ratios(reused) is None


def test_validate_rep_detects_reuse_and_missing_arms(matched):
    seen: dict[str, str] = {}
    first = _rep(matched, 0)
    assert matched.validate_rep(first, seen_run_ids=seen) == []
    second = _rep(matched, 1)
    second["run_ids"] = dict(first["run_ids"])
    failures = matched.validate_rep(second, seen_run_ids=seen)
    assert any("reused execution run id" in f for f in failures)
    missing = _rep(matched, 2)
    del missing["arms"]["tp2"]
    assert any("missing arms" in f for f in matched.validate_rep(missing))


def test_cell_is_declared_natural_not_context_128(matched):
    cell = matched.CELL_C1_NATURAL
    assert cell["context"] == "natural"
    assert cell["output_tokens"] == 128
    assert cell["warmup_decode_tokens"] == 0
    assert cell["eos"] is None
    assert cell["logits_per_decode_step"] is True
    assert cell["tp1_product_baseline"] is False


def test_load_cell_prompts_requires_heldout(matched, tmp_path):
    canonical = tmp_path / "canon.jsonl"
    canonical.write_text(json.dumps({"id": "a", "category": "code", "messages": []}) + "\n")
    rows, failures = matched.load_cell_prompts(canonical, tmp_path / "missing.jsonl")
    assert rows == [] and any("heldout suite missing" in f for f in failures)


def test_load_cell_prompts_dedups_and_requires_heldout_only_rows(matched, tmp_path):
    canonical = tmp_path / "canon.jsonl"
    canonical.write_text(
        "".join(json.dumps({"id": i, "category": "code", "prompt": "p"}) + "\n" for i in ("a", "b"))
    )
    heldout = tmp_path / "held.jsonl"
    heldout.write_text(
        "".join(json.dumps({"id": i, "category": "code", "prompt": "p"}) + "\n" for i in ("b", "c"))
    )
    rows, failures = matched.load_cell_prompts(canonical, heldout)
    assert [r["id"] for r in rows] == ["a", "b", "c"] and failures == []
    # A heldout file whose rows are all already canonical adds nothing.
    only_dup = tmp_path / "dup.jsonl"
    only_dup.write_text(json.dumps({"id": "a", "category": "code", "prompt": "p"}) + "\n")
    _, failures = matched.load_cell_prompts(canonical, only_dup)
    assert any("no heldout-only rows" in f for f in failures)


# -- main() negative tests --------------------------------------------------

def _fake_suite():
    return [
        {"id": "a", "category": "code", "prompt": "a"},
        {"id": "b", "category": "general_en", "prompt": "b"},
    ]


def _patch_main(
    matched,
    monkeypatch,
    *,
    fail_arms=(),
    fail_first_only=(),
    bad_row_arm=None,
    reuse_ids=False,
    zero_rows_arm=None,
    duplicate_arm=None,
    wrong_hash_arm=None,
    missing_field_arm=None,
    no_tp1_provenance=False,
):
    monkeypatch.setattr(matched, "load_cell_prompts", lambda canonical, heldout: (_fake_suite(), []))
    monkeypatch.setattr(matched, "_model_sha256", lambda path: "deadbeef")
    monkeypatch.setattr(matched, "_git_revision", lambda: "testrev")
    monkeypatch.setattr(matched, "_git_dirty", lambda: False)
    monkeypatch.setattr(matched, "_host_identity", lambda: {"node": "test"})
    counter = {"n": 0}
    seen: dict[str, int] = {}

    def fake_run_arm(arm, *, model, prompts, cell, workdir, reps=1, timeout_s=2400.0):
        seen[arm] = seen.get(arm, 0) + 1
        if arm in fail_arms:
            raise matched.MatchedBaselineError(f"{arm} boom")
        if arm in fail_first_only and seen[arm] == 1:
            raise matched.MatchedBaselineError(f"{arm} first-call boom")
        counter["n"] += 1
        run_id = "shared-run-id" if reuse_ids else f"{arm}-{counter['n']}"
        rows = [
            _good_row(matched, arm, prompt_id=p, category=c)[0]
            for p, c in (("a", "code"), ("b", "general_en"))
        ]
        if arm == zero_rows_arm:
            rows = []
        if arm == duplicate_arm and rows:
            rows = [rows[0], rows[0]]
        if arm == wrong_hash_arm and rows:
            rows[0]["prompt_token_sha256"] = "wrong"
        if arm == missing_field_arm and rows:
            del rows[0]["total_generation_ms"]
        if arm == bad_row_arm and rows:
            rows[0]["graph_effective"] = False
        if arm == "tp2":
            return {
                "arm": "tp2",
                "run_id": run_id,
                "command": "fake",
                "log": "fake",
                "device": None,
                "session_capture_ms": 5.0,
                "session_destroy_ms": 3.0,
                "sweeps": [rows],
                "devices": {},
                "route": {"schedule": "graphed"},
            }
        return {
            "arm": arm,
            "run_id": run_id,
            "command": "fake",
            "log": "fake",
            "device": 0 if arm == "tp1-d0" else 1,
            "prompt_metrics": rows,
            "tp1_provenance": None if no_tp1_provenance else {"device": "fake"},
            "tp1_timing_protocol": {},
        }

    monkeypatch.setattr(matched, "run_arm", fake_run_arm)


def _run_main(matched, tmp_path):
    out = tmp_path / "artifact.json"
    code = matched.main(
        ["--run", "--reps", "3", "--json", str(out), "--workdir", str(tmp_path / "wd")]
    )
    return code, json.loads(out.read_text())


def test_main_produces_a_diagnostic_ratio_but_never_qualifies(matched, monkeypatch, tmp_path, capsys):
    _patch_main(matched, monkeypatch)
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 0
    assert artifact["status"] == "diagnostic"
    assert artifact["accounting_failures"] == []
    assert artifact["matching_failures"] == []
    assert artifact["provenance_gaps"] == []
    assert artifact["ratios"] is not None and artifact["ratios"]["reps"] == 3
    # Unmeasured qualification gates preclude qualification and the headline gate.
    assert artifact["qualified"] is False
    assert artifact["all_gates_passed"] is False
    assert artifact["performance_claim"] is False
    assert set(artifact["qualification"].values()) == {False}
    assert any("qualification gate not measured" in f for f in artifact["gate_failures"])
    assert artifact["cell"]["context"] == "natural"
    # Per-prompt rows and run/teardown details are persisted for later review.
    arm_entry = artifact["reps"][0]["arms"]["tp2"]
    assert len(arm_entry["prompt_metrics"]) == 2
    assert arm_entry["session_capture_ms"] == 5.0 and arm_entry["session_destroy_ms"] == 3.0
    assert artifact["reps"][0]["arms"]["tp1-d0"]["tp1_provenance"] is not None
    assert len(artifact["provenance"]["runs"]) == 9


def test_main_uses_a_latin_square_rotation(matched, monkeypatch, tmp_path, capsys):
    _patch_main(matched, monkeypatch)
    _, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    orders = [rep["declared_order"] for rep in artifact["reps"]]
    assert orders[0][0] != orders[1][0] != orders[2][0]
    for arm in matched.ARMS:
        assert sum(1 for order in orders if order[0] == arm) == 1


def test_main_suppresses_ratio_when_an_arm_is_blocked(matched, monkeypatch, tmp_path, capsys):
    _patch_main(matched, monkeypatch, fail_arms=("tp1-d1",))
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1
    assert artifact["status"] == "blocked"
    assert artifact["ratios"] is None
    assert "tp1-d1" in artifact["missing_arms"]
    assert artifact["all_gates_passed"] is False
    assert any("missing arms" in f for f in artifact["accounting_failures"])


def test_main_suppresses_ratio_on_accounting_failure(matched, monkeypatch, tmp_path, capsys):
    _patch_main(matched, monkeypatch, bad_row_arm="tp2")
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1
    assert artifact["ratios"] is None
    assert any("graph replay not effective" in f for f in artifact["accounting_failures"])


def test_main_detects_reused_execution_ids(matched, monkeypatch, tmp_path, capsys):
    _patch_main(matched, monkeypatch, reuse_ids=True)
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1
    assert artifact["ratios"] is None
    assert any("reused execution run id" in f for f in artifact["accounting_failures"])


def test_main_records_suite_failure_and_suppresses_ratio(matched, monkeypatch, tmp_path, capsys):
    _patch_main(matched, monkeypatch)
    monkeypatch.setattr(
        matched, "load_cell_prompts", lambda canonical, heldout: (_fake_suite(), ["heldout suite missing: x"])
    )
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1
    assert artifact["ratios"] is None
    assert any("heldout suite missing" in f for f in artifact["accounting_failures"])


def test_main_emits_a_blocked_artifact_when_the_suite_is_empty(matched, monkeypatch, tmp_path, capsys):
    _patch_main(matched, monkeypatch)
    monkeypatch.setattr(
        matched, "load_cell_prompts", lambda canonical, heldout: ([], ["heldout suite missing: x"])
    )
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1
    assert artifact["status"] == "blocked"
    assert artifact["ratios"] is None
    assert artifact["reps"][0]["arms"] == {}


def test_main_detects_zero_and_duplicate_rows(matched, monkeypatch, tmp_path, capsys):
    _patch_main(matched, monkeypatch, zero_rows_arm="tp2")
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1 and artifact["ratios"] is None
    assert any("zero prompt rows" in f for f in artifact["accounting_failures"])

    _patch_main(matched, monkeypatch, duplicate_arm="tp2")
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1 and artifact["ratios"] is None
    assert any("duplicate prompt ids" in f for f in artifact["accounting_failures"])


def test_main_detects_wrong_token_hash_and_missing_field(matched, monkeypatch, tmp_path, capsys):
    _patch_main(matched, monkeypatch, wrong_hash_arm="tp1-d0")
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1 and artifact["ratios"] is None
    assert any("differs across arms" in f for f in artifact["matching_failures"])

    _patch_main(matched, monkeypatch, missing_field_arm="tp2")
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1 and artifact["ratios"] is None
    assert any("total_generation_ms" in f for f in artifact["accounting_failures"])


def test_main_records_missing_tp1_provenance_as_a_gap(matched, monkeypatch, tmp_path, capsys):
    _patch_main(matched, monkeypatch, no_tp1_provenance=True)
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1 and artifact["ratios"] is None
    assert artifact["provenance_gaps"]


def test_main_keeps_reps_keyed_when_an_early_arm_fails(matched, monkeypatch, tmp_path, capsys):
    # rep0's first call fails, later reps succeed. The old success-compressing
    # list would have shifted rep1's tp1-d0 result into rep0.
    _patch_main(matched, monkeypatch, fail_first_only=("tp1-d0",))
    code, artifact = _run_main(matched, tmp_path)
    capsys.readouterr()
    assert code == 1 and artifact["ratios"] is None
    assert "tp1-d0" not in artifact["reps"][0]["run_ids"]
    assert "tp1-d0" in artifact["reps"][1]["run_ids"]
    assert artifact["reps"][1]["run_ids"]["tp1-d0"] != artifact["reps"][0]["run_ids"].get("tp1-d0")
    # The successful later runs are still the ones recorded, not rep0's slot.
    assert artifact["reps"][1]["arms"]["tp1-d0"]["prompt_metrics"]


def test_main_requires_three_reps_and_a_mode(matched):
    with pytest.raises(SystemExit):
        matched.main(["--run", "--reps", "2"])
    with pytest.raises(SystemExit):
        matched.main([])
