"""CPU tests for the TP2 teacher-control harnesses.

These cover the host-checkable pieces the parent reviews flagged: the parent's
timeout/reap supervision retains partial flushed output; its per-case verdict is
fail-closed (no false PASS from a stray CHILD_OK); prefix checks are
length-aware; case parsing handles comma-bearing device lists and rejects
malformed input; the coverage suite is the canonical + heldout-only
deduplicated set; full trajectories are scored against expected lengths and
vocab width; every gate is enforced (non-finite metrics, incomplete
trajectories, missing controls/categories, suite completeness, >=3-sweep
determinism, state boundaries); and `main()` integration wiring fails closed.
The end-to-end GPU sweeps are separate, expensive, and run manually.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
def parent():
    return _load("tp2_teacher_bisect_parent", SCRIPTS / "tp2_teacher_bisect_parent.py")


@pytest.fixture(scope="module")
def coverage():
    return _load("tp2_teacher_coverage_broad", SCRIPTS / "tp2_teacher_coverage_broad.py")


@pytest.fixture(scope="module")
def child():
    return _load("tp2_teacher_child", SCRIPTS / "tp2_teacher_child.py")


# -- parent supervision / parsing -------------------------------------------


def _good_parsed(**overrides):
    parsed = {
        "tokens": [1, 2, 3],
        "build_s": 90.5,
        "exec_started": True,
        "calls": [
            {"call": 0, "positions": 3, "wall_s": 0.4, "finite": True, "argmax_last": 7},
        ],
        "ok": {"calls": 1, "positions": 3, "total_wall_s": 0.4},
        "fail": None,
        "cleanup_ok": True,
        "cleanup_fail": None,
    }
    parsed.update(overrides)
    return parsed


def test_parse_recovers_phases_calls_and_cleanup(parent):
    lines = [
        "CHILD_TOKENS [1, 2, 3]",
        "CHILD_BUILD_OK elapsed_s=90.500 mode=tp1 schedule=eager devices=[1]",
        "CHILD_EXEC_START calls=1 positions=3 reset_between=False",
        "CHILD_CALL call=0 positions=3 wall_s=0.4200 finite=True argmax_last=42",
        "CHILD_OK calls=1 positions=3 total_wall_s=0.4200",
        "CHILD_CLEANUP_OK",
    ]
    parsed = parent._parse(lines)
    assert parsed["tokens"] == [1, 2, 3]
    assert parsed["build_s"] == 90.5
    assert parsed["exec_started"] is True
    assert parsed["calls"][0]["positions"] == 3
    assert parsed["ok"] == {"calls": 1, "positions": 3, "total_wall_s": 0.42}
    assert parsed["cleanup_ok"] is True
    assert parsed["fail"] is None


def test_parse_timeout_partial_marks_build_phase(parent):
    lines = [
        "CHILD_TOKENS [1, 2]",
        "CHILD_BUILD_OK elapsed_s=12.0 mode=tp1 schedule=eager devices=[1]",
    ]
    parsed = parent._parse(lines)
    assert parsed["build_s"] == 12.0
    assert parsed["exec_started"] is False
    assert parsed["ok"] is None


def test_supervise_reaps_and_retains_flushed_output(parent):
    cmd = [
        sys.executable,
        "-u",
        "-c",
        "import time; print('EARLY', flush=True); time.sleep(120)",
    ]
    started = time.perf_counter()
    lines, timed_out, returncode, wall = parent.supervise(cmd, timeout=1.0, grace=1.0)
    assert timed_out is True
    assert "EARLY" in lines
    assert returncode is not None
    assert time.perf_counter() - started < 20.0


def test_evaluate_case_pass(parent):
    verdict, reasons = parent.evaluate_case(
        _good_parsed(), timed_out=False, returncode=0, expected_calls=1, positions=3
    )
    assert verdict == "PASS" and reasons == []


def test_evaluate_case_no_false_pass_from_ok(parent):
    # CHILD_OK present but the process timed out afterwards: must not PASS.
    verdict, reasons = parent.evaluate_case(
        _good_parsed(), timed_out=True, returncode=None, expected_calls=1, positions=3
    )
    assert verdict.startswith("TIMEOUT") and reasons


def test_evaluate_case_nonzero_exit_fails(parent):
    verdict, reasons = parent.evaluate_case(
        _good_parsed(), timed_out=False, returncode=1, expected_calls=1, positions=3
    )
    assert verdict.startswith("FAIL") and any("returncode" in r for r in reasons)


def test_evaluate_case_requires_cleanup(parent):
    verdict, reasons = parent.evaluate_case(
        _good_parsed(cleanup_ok=False),
        timed_out=False,
        returncode=0,
        expected_calls=1,
        positions=3,
    )
    assert verdict.startswith("FAIL") and any("CLEANUP" in r for r in reasons)


def test_evaluate_case_requires_exact_calls(parent):
    parsed = _good_parsed(
        calls=[],
        ok={"calls": 1, "positions": 3, "total_wall_s": 0.4},
    )
    verdict, reasons = parent.evaluate_case(
        parsed, timed_out=False, returncode=0, expected_calls=2, positions=3
    )
    assert verdict.startswith("FAIL")
    assert any("call count" in r for r in reasons)
    assert any("ok.calls" in r for r in reasons)


def test_evaluate_case_requires_call_indices_and_positions(parent):
    parsed = _good_parsed(
        calls=[
            {"call": 1, "positions": 4, "wall_s": 0.4, "finite": False, "argmax_last": 7},
        ],
        ok={"calls": 1, "positions": 3, "total_wall_s": 0.4},
    )
    verdict, reasons = parent.evaluate_case(
        parsed, timed_out=False, returncode=0, expected_calls=1, positions=3
    )
    assert verdict.startswith("FAIL")
    assert any("index" in r for r in reasons)
    assert any("positions" in r for r in reasons)
    assert any("non-finite" in r for r in reasons)


def test_evaluate_case_child_failure_fails(parent):
    verdict, reasons = parent.evaluate_case(
        _good_parsed(fail="invalid-logits detail=non-finite logits", ok=None, cleanup_ok=True),
        timed_out=False,
        returncode=1,
        expected_calls=1,
        positions=3,
    )
    assert verdict.startswith("FAIL")
    assert any("invalid-logits" in r for r in reasons)


def test_prefix_status_length_and_consistency(parent):
    ok, detail = parent.prefix_status([1, 2, 3], positions=3, reference=[1, 2, 3, 4])
    assert ok and detail == "consistent"
    # A 16 -> 24 progression over a shared prefix is legitimate.
    ok, detail = parent.prefix_status([1, 2], positions=2, reference=[1, 2, 3, 4])
    assert ok and detail == "consistent"
    ok, detail = parent.prefix_status([1, 9], positions=2, reference=[1, 2, 3, 4])
    assert not ok and "mismatch" in detail
    ok, detail = parent.prefix_status([1, 2], positions=3, reference=[1, 2, 3])
    assert not ok and "token count" in detail
    ok, detail = parent.prefix_status(None, positions=3, reference=None)
    assert not ok


def test_parse_case_keeps_comma_device_lists(parent):
    case = parent.parse_case("positions=16,devices=0,1,mode=tp2,calls=2,reset_between=true")
    assert case == {
        "positions": 16,
        "devices": "0,1",
        "mode": "tp2",
        "calls": 2,
        "reset_between": True,
    }


def test_parse_case_rejects_unknown_and_malformed(parent):
    with pytest.raises(ValueError):
        parent.parse_case("positions=16,bogus=1")
    with pytest.raises(ValueError):
        parent.parse_case("positions=16,devices")
    with pytest.raises(ValueError):
        parent.parse_case("positions=abc")


def test_parse_gpu_csv(parent):
    assert parent._parse_gpu_csv("device,GPU use (%)\ncard0,54\ncard1,0\n") == [54, 0]
    assert parent._parse_gpu_csv("") is None


def test_wait_for_idle_missing_telemetry_is_not_idle(parent, monkeypatch):
    monkeypatch.setattr(
        parent,
        "host_snapshot",
        lambda: {
            "nice": 16,
            "loadavg": [0.0, 0.0, 0.0],
            "gpu_busy_percent": None,
            "gpu_telemetry_available": False,
        },
    )
    idle, snap = parent.wait_for_idle(max_load=1.0, max_gpu=5, timeout_s=0.01, poll_s=0.001)
    assert idle is False
    assert snap["gpu_telemetry_available"] is False


def test_main_rejects_bad_repeat_and_timeout(parent, capsys):
    with pytest.raises(SystemExit) as exc:
        parent.main(["--repeat-per-case", "0"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        parent.main(["--timeout", "0"])
    assert exc.value.code == 2


def test_main_require_idle_implies_gate(parent, monkeypatch, capsys):
    captured = {}

    def fake_run_case(**kwargs):
        captured.update(kwargs)
        return {"verdict": "CONTENDED", "prefix_consistent": False, "tokens": None}

    monkeypatch.setattr(parent, "run_case", fake_run_case)
    parent.main(["--case", "positions=16,devices=1,mode=tp1", "--require-idle"])
    assert captured["idle_gate"] is True
    assert captured["require_idle"] is True


# -- coverage suite and metrics ---------------------------------------------


def test_suite_is_canonical_plus_heldout_only(coverage):
    suite = coverage.load_prompt_suite()
    ids = [row["id"] for row in suite]
    assert len(ids) == len(set(ids)), "suite must be deduplicated by id"
    canonical = {
        json.loads(line)["id"]
        for line in coverage.CANONICAL_SUITE.read_text().splitlines()
        if line.strip()
    }
    heldout = {
        json.loads(line)["id"]
        for line in coverage.HELDOUT_SUITE.read_text().splitlines()
        if line.strip()
    }
    assert set(ids) == canonical | heldout
    assert set(ids) == coverage.full_suite_ids()
    assert {row["category"] for row in suite} == set(coverage.CATEGORIES)
    assert all(row["heldout"] is False for row in suite if row["id"] in canonical)
    assert all(row["heldout"] is True for row in suite if row["id"] in heldout - canonical)


def test_kl_rows_is_per_position_and_zero_for_identical(coverage):
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(7, 64)).astype(np.float32)
    kl, top1 = coverage._kl_rows(logits, logits.copy())
    assert kl.shape == (7,)
    assert top1.shape == (7,)
    assert np.allclose(kl, 0.0, atol=1e-12)
    assert top1.all()


def test_score_arm_counts_full_trajectory(coverage):
    teacher = [np.zeros((3, 8), dtype=np.float32), np.zeros((2, 8), dtype=np.float32)]
    student = [t.copy() for t in teacher]
    scored = coverage.score_arm(
        teacher, student, ["code", "general_ja"], [False, False],
        expected_positions=[3, 2], vocab_size=8,
    )
    assert scored["global"]["rows"] == 5
    assert scored["categories"]["code"]["rows"] == 3
    assert scored["categories"]["general_ja"]["rows"] == 2
    assert scored["shape_mismatches"] == []


def test_score_arm_rejects_identically_truncated_trajectories(coverage):
    # Both arms agree with each other but are shorter than the prompt: must be
    # recorded as incomplete, not silently scored.
    teacher = [np.zeros((2, 8), dtype=np.float32)]
    student = [np.zeros((2, 8), dtype=np.float32)]
    scored = coverage.score_arm(
        teacher, student, ["code"], [False], expected_positions=[5], vocab_size=8
    )
    assert scored["scored_rows"] == 0
    assert any(m["kind"] == "positions" for m in scored["shape_mismatches"])


def test_score_arm_rejects_wrong_vocab_width(coverage):
    teacher = [np.zeros((3, 8), dtype=np.float32)]
    student = [np.zeros((3, 8), dtype=np.float32)]
    scored = coverage.score_arm(
        teacher, student, ["code"], [False], expected_positions=[3], vocab_size=16
    )
    assert scored["scored_rows"] == 0
    assert any(m["kind"] == "vocab" for m in scored["shape_mismatches"])


def test_score_arm_records_nonfinite(coverage):
    teacher = [np.full((3, 8), np.nan, dtype=np.float32)]
    student = [np.zeros((3, 8), dtype=np.float32)]
    scored = coverage.score_arm(
        teacher, student, ["code"], [False], expected_positions=[3], vocab_size=8
    )
    assert scored["scored_rows"] == 0
    assert scored["nonfinite_rows"] == [0]


def test_score_arm_separates_canonical_heldout_and_category_scope(coverage):
    teacher = [
        np.zeros((2, 8), dtype=np.float32),
        np.zeros((3, 8), dtype=np.float32),
    ]
    student = [t.copy() for t in teacher]
    scored = coverage.score_arm(
        teacher, student, ["code", "code"], [False, True],
        expected_positions=[2, 3], vocab_size=8,
    )
    assert scored["scopes"]["canonical"]["rows"] == 2
    assert scored["scopes"]["heldout"]["rows"] == 3
    assert scored["category_scopes"]["code"]["canonical"]["rows"] == 2
    assert scored["category_scopes"]["code"]["heldout"]["rows"] == 3


def test_envelope_gate_rejects_nan_metrics(coverage):
    summary = {
        "rows": 100,
        "mean_kl": float("nan"),
        "p95_kl": 1e-4,
        "p99_kl": 1e-4,
        "max_kl": 1e-4,
        "top1_agreement": 1.0,
        "flipped_rows": 0,
    }
    gate = coverage._envelope_gate(summary, top1_bar=0.99)
    assert gate["passed"] is False
    assert any("non-finite" in f for f in gate["failures"])


def test_envelope_gate_binds_on_top1_and_kl(coverage):
    passing = {
        "rows": 100,
        "mean_kl": 1e-4,
        "p95_kl": 1e-4,
        "p99_kl": 1e-4,
        "max_kl": 1e-4,
        "top1_agreement": 0.98,
        "flipped_rows": 2,
    }
    assert coverage._envelope_gate(passing, top1_bar=0.97)["passed"] is True
    assert coverage._envelope_gate(
        dict(passing, top1_agreement=0.96), top1_bar=0.97
    )["passed"] is False
    assert coverage._envelope_gate(dict(passing, p99_kl=3e-2), top1_bar=0.97)[
        "passed"
    ] is False


def test_qualification_label_is_probe_scale_only(coverage):
    assert coverage.qualification_label(60) == "short_probe"
    assert coverage.qualification_label(coverage.QUALIFICATION_ROWS) == "extended_probe"
    assert "full" not in coverage.qualification_label(10_000)


def test_row_hash_is_content_stable(coverage):
    rng = np.random.default_rng(11)
    row = rng.normal(size=(4, 16)).astype(np.float32)
    assert coverage._row_hash(row) == coverage._row_hash(row.copy())
    other = row.copy()
    other[0, 0] += 1.0
    assert coverage._row_hash(row) != coverage._row_hash(other)


# -- fail-closed gate evaluation --------------------------------------------


def _good_summary() -> dict[str, object]:
    good = {
        "rows": 100,
        "mean_kl": 1e-4,
        "p95_kl": 1e-4,
        "p99_kl": 1e-4,
        "max_kl": 1e-4,
        "top1_agreement": 1.0,
        "flipped_rows": 0,
    }
    categories = ("code", "general_en", "general_ja", "mixed_ja_en")
    return {
        "global": dict(good),
        "categories": {c: dict(good) for c in categories},
        "scopes": {"canonical": dict(good), "heldout": dict(good)},
        "category_scopes": {
            c: {"canonical": dict(good), "heldout": dict(good)} for c in categories
        },
        "shape_mismatches": [],
        "nonfinite_rows": [],
        "scored_rows": 100,
    }


def _good_comparisons() -> dict[str, object]:
    return {c: _good_summary() for c in ("tp1-d0", "tp1-d1")}


def _good_state_boundaries() -> dict[str, object]:
    return {
        "reset_reuse_bit_exact": True,
        "intervening_prompt_reuse_bit_exact": True,
        "reset_after_generation_bit_exact": True,
    }


def _evaluate(coverage, **overrides):
    kwargs = dict(
        comparisons=_good_comparisons(),
        categories_present=set(coverage.CATEGORIES),
        suite_has_heldout=True,
        suite_complete=True,
        nonfinite_rows=[],
        determinism={"sweeps": coverage.MIN_DETERMINISM_SWEEPS, "per_row_match": True},
        state_boundaries=_good_state_boundaries(),
    )
    kwargs.update(overrides)
    return coverage.evaluate_gates(**kwargs)


def test_evaluate_gates_passes_only_when_all_required(coverage):
    ok, failures = _evaluate(coverage)
    assert ok is True and failures == []


def test_evaluate_gates_rejects_nonfinite(coverage):
    ok, failures = _evaluate(coverage, nonfinite_rows=["tp2:0"])
    assert ok is False and any("non-finite" in f for f in failures)


def test_evaluate_gates_requires_three_sweeps(coverage):
    ok, failures = _evaluate(coverage, determinism={"sweeps": 2, "per_row_match": True})
    assert ok is False and any(">=3" in f for f in failures)


def test_evaluate_gates_rejects_determinism_mismatch(coverage):
    ok, failures = _evaluate(
        coverage, determinism={"sweeps": 3, "per_row_match": False}
    )
    assert ok is False and any("determinism" in f for f in failures)


def test_evaluate_gates_rejects_state_boundary_failures(coverage):
    for key in _good_state_boundaries():
        boundaries = _good_state_boundaries()
        boundaries[key] = False
        ok, failures = _evaluate(coverage, state_boundaries=boundaries)
        assert ok is False, key
        assert any("boundary" in f for f in failures), key


def test_evaluate_gates_requires_both_controls(coverage):
    comparisons = _good_comparisons()
    del comparisons["tp1-d1"]
    ok, failures = _evaluate(coverage, comparisons=comparisons)
    assert ok is False and any("missing control tp1-d1" in f for f in failures)


def test_evaluate_gates_requires_all_categories(coverage):
    present = set(coverage.CATEGORIES) - {"mixed_ja_en"}
    ok, failures = _evaluate(coverage, categories_present=present)
    assert ok is False and any("missing categories" in f for f in failures)


def test_evaluate_gates_requires_heldout(coverage):
    ok, failures = _evaluate(coverage, suite_has_heldout=False)
    assert ok is False and any("heldout" in f for f in failures)


def test_evaluate_gates_requires_complete_suite_by_default(coverage):
    ok, failures = _evaluate(coverage, suite_complete=False)
    assert ok is False and any("incomplete" in f for f in failures)
    # Explicit diagnostic subset: the completeness gate is waived.
    ok, failures = _evaluate(coverage, suite_complete=False, require_suite_complete=False)
    assert ok is True


def test_evaluate_gates_rejects_shape_mismatch(coverage):
    comparisons = _good_comparisons()
    comparisons["tp1-d0"]["shape_mismatches"] = [{"kind": "row", "index": 0}]
    ok, failures = _evaluate(coverage, comparisons=comparisons)
    assert ok is False and any("incomplete trajectories" in f for f in failures)


def test_evaluate_gates_rejects_arm_nonfinite(coverage):
    comparisons = _good_comparisons()
    comparisons["tp1-d1"]["nonfinite_rows"] = [2]
    ok, failures = _evaluate(coverage, comparisons=comparisons)
    assert ok is False and any("non-finite trajectories" in f for f in failures)


def test_evaluate_gates_rejects_category_kl(coverage):
    comparisons = _good_comparisons()
    comparisons["tp1-d0"]["categories"]["code"]["max_kl"] = 0.2
    ok, failures = _evaluate(coverage, comparisons=comparisons)
    assert ok is False and any("category code" in f for f in failures)


def test_evaluate_gates_rejects_category_scope_intersection(coverage):
    comparisons = _good_comparisons()
    comparisons["tp1-d1"]["category_scopes"]["general_ja"]["heldout"]["top1_agreement"] = 0.5
    ok, failures = _evaluate(coverage, comparisons=comparisons)
    assert ok is False and any("general_ja/heldout" in f for f in failures)


def test_evaluate_gates_rejects_missing_measurement(coverage):
    ok, failures = _evaluate(coverage, determinism=None, state_boundaries=None)
    assert ok is False
    assert any("determinism not measured" in f for f in failures)
    assert any("state boundaries not measured" in f for f in failures)


# -- child fail-closed behavior (mocked, no GPU) ----------------------------


class _FakeSession:
    def __init__(self, *, positions, vocab_size=8, bad_shape=False, close_exc=None):
        self.vocab_size = vocab_size
        self.schedule = "eager"
        self._positions = positions
        self._bad_shape = bad_shape
        self._close_exc = close_exc
        self.closed = False
        self.reset_calls = 0
        self.generate_calls = 0

    def teacher_forced_logits(self, tokens):
        cols = 4 if self._bad_shape else self.vocab_size
        return np.zeros((self._positions, cols), dtype=np.float32)

    def reset(self):
        self.reset_calls += 1

    def generate(self, tokens, max_new_tokens=2):
        self.generate_calls += 1
        return object()

    def close(self):
        self.closed = True
        if self._close_exc is not None:
            raise self._close_exc


def _patch_child(child, monkeypatch, session):
    monkeypatch.setattr(child, "canonical_tokens", lambda: tuple(range(1, 40)))
    monkeypatch.setattr(
        child, "_session_factory", lambda model, *, devices, mode, schedule: session
    )


def test_child_rejects_zero_or_negative_calls(child, monkeypatch, capsys):
    for calls in ("0", "-1"):
        code = child.main(
            ["--positions", "3", "--devices", "0", "--mode", "tp1", "--calls", calls]
        )
        out = capsys.readouterr().out
        assert code == 2
        assert "bad-calls" in out
        assert "CHILD_EXEC_START" not in out
        assert "CHILD_OK" not in out


def test_child_rejects_invalid_logits_shape(child, monkeypatch, capsys):
    session = _FakeSession(positions=3, vocab_size=8, bad_shape=True)
    _patch_child(child, monkeypatch, session)
    code = child.main(
        ["--positions", "3", "--devices", "0", "--mode", "tp1", "--calls", "1"]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "invalid-logits" in out
    assert "CHILD_OK" not in out
    assert session.closed is True


def test_child_rejects_wrong_row_count(child, monkeypatch, capsys):
    session = _FakeSession(positions=2, vocab_size=8)
    _patch_child(child, monkeypatch, session)
    code = child.main(
        ["--positions", "3", "--devices", "0", "--mode", "tp1", "--calls", "1"]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "rows=2 expected 3" in out


def test_child_fails_closed_on_cleanup_error(child, monkeypatch, capsys):
    session = _FakeSession(positions=3, vocab_size=8, close_exc=RuntimeError("boom"))
    _patch_child(child, monkeypatch, session)
    code = child.main(
        ["--positions", "3", "--devices", "0", "--mode", "tp1", "--calls", "1"]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "CHILD_OK" in out
    assert "CHILD_CLEANUP_FAIL" in out
    assert session.closed is True


def test_child_success_emits_ok_and_cleanup(child, monkeypatch, capsys):
    session = _FakeSession(positions=3, vocab_size=8)
    _patch_child(child, monkeypatch, session)
    code = child.main(
        ["--positions", "3", "--devices", "0", "--mode", "tp1", "--calls", "2", "--reset-between"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("CHILD_CALL call=") == 2
    assert "CHILD_OK" in out
    assert "CHILD_CLEANUP_OK" in out
    assert session.reset_calls == 1
    assert session.closed is True


def test_validate_logits_direct(child):
    ok, detail = child.validate_logits(
        np.zeros((3, 8), dtype=np.float32), positions=3, vocab_size=8
    )
    assert ok and detail == ""
    ok, detail = child.validate_logits(
        np.zeros((3, 8), dtype=np.float32), positions=4, vocab_size=8
    )
    assert not ok and "rows" in detail
    ok, detail = child.validate_logits(
        np.full((3, 8), np.nan, dtype=np.float32), positions=3, vocab_size=8
    )
    assert not ok and "non-finite" in detail
    ok, detail = child.validate_logits(np.zeros(8, dtype=np.float32), positions=3, vocab_size=8)
    assert not ok and "ndim" in detail


# -- coverage main() integration (mocked sessions) --------------------------


class _FakeRuntime:
    def device_info(self, device):
        return SimpleNamespace(
            name=f"fake-gpu-{device}", uuid=f"uuid{device}", pci_bus_id=f"0000:0{device}:00.0"
        )


class _FakeCoverageSession:
    def __init__(self, devices, mode, *, nonfinite=False):
        self.devices = tuple(devices)
        self.mode = mode
        self.schedule = "eager"
        self.driver = "compiled"
        self.reduce_mode = "host"
        self.head_shard = False
        self.max_sequence_length = 2048
        self.vocab_size = 8
        self.runtime = _FakeRuntime()
        self._nonfinite = nonfinite
        self.closed = False

    def teacher_forced_logits(self, tokens):
        value = np.nan if self._nonfinite else 1.0
        return np.full((len(tokens), self.vocab_size), value, dtype=np.float32)

    def generate(self, tokens, max_new_tokens=2):
        return object()

    def close(self):
        self.closed = True


class _FakeTokenizer:
    def encode(self, text):
        return list(range(4))


def _fake_suite():
    rows = []
    for category, heldout in (
        ("code", False),
        ("general_en", False),
        ("general_ja", True),
        ("mixed_ja_en", True),
    ):
        rows.append(
            {
                "id": f"row_{category}",
                "category": category,
                "messages": [{"role": "user", "content": category}],
                "heldout": heldout,
            }
        )
    return rows


def _patch_coverage(coverage, monkeypatch, *, nonfinite=False, complete=True):
    suite = _fake_suite()
    monkeypatch.setattr(coverage, "load_prompt_suite", lambda: suite)
    monkeypatch.setattr(
        coverage, "full_suite_ids", lambda: {r["id"] for r in suite} if complete else set()
    )
    monkeypatch.setattr(coverage, "_load_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(
        coverage,
        "_session_factory",
        lambda model, *, devices, mode: _FakeCoverageSession(devices, mode, nonfinite=nonfinite),
    )
    monkeypatch.setattr(coverage, "_git_dirty", lambda: False)


def test_coverage_main_passes_with_consistent_fake_arms(coverage, monkeypatch, tmp_path, capsys):
    _patch_coverage(coverage, monkeypatch)
    out = tmp_path / "artifact.json"
    code = coverage.main(["--json", str(out), "--model-hash", "none", "--repeat-tp2", "3"])
    capsys.readouterr()
    assert code == 0
    artifact = json.loads(out.read_text())
    assert artifact["all_gates_passed"] is True
    assert artifact["gate_failures"] == []
    assert artifact["determinism"]["per_row_match"] is True
    assert artifact["state_boundaries"]["reset_reuse_bit_exact"] is True
    assert artifact["suite"]["complete"] is True
    assert artifact["certification"].startswith("none")


def test_coverage_main_fails_on_nonfinite_arm(coverage, monkeypatch, tmp_path, capsys):
    _patch_coverage(coverage, monkeypatch, nonfinite=True)
    out = tmp_path / "artifact.json"
    code = coverage.main(["--json", str(out), "--model-hash", "none", "--repeat-tp2", "3"])
    capsys.readouterr()
    assert code == 1
    artifact = json.loads(out.read_text())
    assert artifact["all_gates_passed"] is False
    assert any("non-finite" in f for f in artifact["gate_failures"])


def test_coverage_main_flags_incomplete_suite(coverage, monkeypatch, tmp_path, capsys):
    _patch_coverage(coverage, monkeypatch, complete=False)
    out = tmp_path / "artifact.json"
    code = coverage.main(
        ["--json", str(out), "--model-hash", "none", "--repeat-tp2", "3"]
    )
    capsys.readouterr()
    assert code == 1
    artifact = json.loads(out.read_text())
    assert artifact["suite"]["diagnostic_scope"] is True
    assert any("incomplete" in f for f in artifact["gate_failures"])


def test_logits_stats_reports_nan_and_finite_ranges(child):
    stats = child.logits_stats(np.array([[1.0, np.nan, 3.0]], dtype=np.float32))
    assert "nan=1" in stats and "inf=0" in stats and "finite=2/3" in stats
    stats = child.logits_stats(np.full((2, 2), np.inf, dtype=np.float32))
    assert "inf=4" in stats and "finite=0/4" in stats
