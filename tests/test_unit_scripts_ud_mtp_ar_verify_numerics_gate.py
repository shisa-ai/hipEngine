"""CPU-only checks for the AR-versus-verification numerics gate.

The gate itself needs a GPU; these cover the metric and screen arithmetic.
"""
from __future__ import annotations

from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ud_mtp_ar_verify_numerics_gate import (
    ENVELOPE,
    _pool,
    _row_kl,
    _screen,
)


def test_identical_logits_have_zero_kl_and_full_top1() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(4, 64)).astype(np.float32)
    kl = _row_kl(logits, logits)
    assert np.allclose(kl, 0.0, atol=1e-12)


def test_kl_is_non_negative_and_grows_with_the_shift() -> None:
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(6, 128)).astype(np.float32)
    near = reference + 1e-3
    far = reference + 1.0
    small = _row_kl(reference, near)
    large = _row_kl(reference, far)
    assert np.all(small >= 0.0) and np.all(large >= 0.0)
    assert float(large.mean()) > float(small.mean())


def test_row_kl_matches_a_hand_computed_two_class_case() -> None:
    reference = np.array([[0.0, 0.0]], dtype=np.float64)
    # shift the second class up: p = [0.5, 0.5], q = sigmoid-shifted
    candidate = np.array([[0.0, np.log(3.0)]], dtype=np.float64)
    p = np.array([0.5, 0.5])
    q = np.array([0.25, 0.75])
    expected = float(np.sum(p * (np.log(p) - np.log(q))))
    assert _row_kl(reference, candidate)[0] == pytest.approx(expected, rel=1e-12)


def test_row_kl_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        _row_kl(np.zeros((2, 4)), np.zeros((3, 4)))


def _run(rows, *, budget=1, category="code", kl=None, top1=None):
    kl = kl if kl is not None else [0.0] * len(rows)
    top1 = top1 if top1 is not None else [True] * len(rows)
    return {
        "id": "p",
        "category": category,
        "prompt_tokens": 64,
        "ar_tokens": [],
        "cases": [{
            "budget": budget,
            "rows": len(rows),
            "kl": list(kl),
            "top1": list(top1),
            "root_position": 64,
            "native_graph": True,
            "fallback_reason": None,
            "target_top1": [],
            "ar_top1": [],
            "target_top1_matches_ar": True,
            "max_abs_diff": 0.0,
            "finite": True,
        }],
    }


def test_pool_aggregates_scope_and_budget_and_flags_p99_outliers() -> None:
    results = [
        _run([0, 0], budget=1, category="code", kl=[1e-9, 1e-9]),
        _run([0, 0, 0], budget=2, category="general_ja",
             kl=[1e-9, 1e-9, 1.0], top1=[True, True, False]),
    ]
    pooled = _pool(results)
    assert pooled["rows"] == 5
    assert pooled["kl_max"] == pytest.approx(1.0)
    assert pooled["top1_agreement"] == pytest.approx(4 / 5)
    assert pooled["top1_by_scope"]["code"]["agreement"] == 1.0
    assert pooled["top1_by_scope"]["general_ja"]["agreement"] == pytest.approx(2 / 3)
    assert set(pooled["top1_by_budget"]) == {"1", "2"}
    assert pooled["rows_above_p99"] and pooled["rows_above_p99"][0]["kl"] == 1.0


def test_screen_binds_every_section_61_threshold() -> None:
    clean = _pool([_run([0, 0], kl=[0.0, 0.0])])
    assert _screen(clean)["passed"] is True

    for name, value, key in (
        ("mean", ENVELOPE["mean"] * 2, "kl_mean"),
        ("p95", ENVELOPE["p95"] * 2, "kl_p95"),
        ("p99", ENVELOPE["p99"] * 2, "kl_p99"),
        ("max", ENVELOPE["max"] * 2, "kl_max"),
    ):
        broken = _pool([_run([0, 0], kl=[value, value])])
        assert broken[key] > ENVELOPE[name]
        assert _screen(broken)["passed"] is False

    bad_top1 = _pool([_run([0, 0, 0, 0], top1=[False, False, True, True])])
    assert _screen(bad_top1)["passed"] is False


def _invoke_gate(monkeypatch, tmp_path, results, extra_args=()):
    from scripts import ud_mtp_ar_verify_numerics_gate as gate
    import hipengine.loading.gguf as loading
    import hipengine.runtime.qwen35_gguf_runner as runtime
    import hipengine.tokenization.gguf as tokenization

    monkeypatch.setattr(loading, "scan_gguf", lambda path: None)
    monkeypatch.setattr(
        tokenization, "Qwen35GGUFTokenizer",
        SimpleNamespace(from_gguf_info=lambda info: None),
    )
    closed = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

    monkeypatch.setattr(runtime, "Qwen35GGUFResidentSession", lambda *a, **kw: Session())
    monkeypatch.setattr(gate, "_seed_ids", lambda *a: [1])
    monkeypatch.setattr(gate, "_extend_to_window", lambda *a: [1] * 64)
    outcomes = iter(results)
    monkeypatch.setattr(gate, "_run_prompt", lambda *a, **kw: deepcopy(next(outcomes)))
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"p","category":"code"}\n')
    output = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", [
        "gate", "--json", str(output), "--prompts", str(prompts),
        "--repeat-runs", str(len(results)), "--budgets", "1", *extra_args,
    ])
    status = gate.main()
    assert closed == [True]
    return status, json.loads(output.read_text())


def test_cli_returns_success_only_for_passing_repeated_gate(monkeypatch, tmp_path, capsys):
    clean = _run([0, 0])
    status, payload = _invoke_gate(monkeypatch, tmp_path, [clean, clean])
    assert status == 0
    assert payload["passed"] is True
    assert "passed=True" in capsys.readouterr().out


def test_cli_fails_and_writes_report_for_numerical_failure(monkeypatch, tmp_path, capsys):
    failed = _run([0, 0], kl=[0.1, 0.1])
    status, payload = _invoke_gate(monkeypatch, tmp_path, [failed, failed])
    assert status == 1
    assert payload["passed"] is False
    assert payload["checks"]["numerical_envelope"] is False
    assert "passed=False" in capsys.readouterr().out


def test_cli_checks_later_repeats_not_just_the_first(monkeypatch, tmp_path):
    status, payload = _invoke_gate(
        monkeypatch, tmp_path, [_run([0, 0]), _run([0, 0], top1=[True, False])],
    )
    assert status == 1
    assert payload["runs"][0]["screen"]["passed"] is True
    assert payload["checks"]["numerical_envelope"] is False


def test_cli_fails_nondeterminism_even_with_passing_numerics(monkeypatch, tmp_path):
    status, payload = _invoke_gate(
        monkeypatch, tmp_path, [_run([0, 0]), _run([0, 0], kl=[1e-5, 1e-5])],
    )
    assert all(run["screen"]["passed"] for run in payload["runs"])
    assert status == 1
    assert payload["deterministic"] is False


@pytest.mark.parametrize("require_native,expected", [(False, 0), (True, 1)])
def test_cli_enforces_requested_native_route(monkeypatch, tmp_path, require_native, expected):
    fallback = _run([0, 0])
    fallback["cases"][0].update(native_graph=False, fallback_reason="eager")
    args = ("--require-native-graph",) if require_native else ("--allow-eager-fallback",)
    status, payload = _invoke_gate(monkeypatch, tmp_path, [fallback, fallback], args)
    assert status == expected
    assert payload["checks"]["native_graph"] is (not require_native)


def test_cli_fails_nonfinite_case(monkeypatch, tmp_path):
    failed = _run([0, 0])
    failed["cases"][0]["finite"] = False
    status, payload = _invoke_gate(monkeypatch, tmp_path, [failed, failed])
    assert status == 1
    assert payload["checks"]["finite_logits"] is False


def test_cli_rejects_empty_results(monkeypatch, tmp_path):
    empty = _run([])
    empty["cases"] = []
    status, payload = _invoke_gate(monkeypatch, tmp_path, [empty, empty])
    assert status == 1
    assert payload["checks"]["nonempty_results"] is False


@pytest.mark.parametrize("flag", ["--repeat-runs", "--prompt-tokens", "--limit"])
def test_cli_rejects_nonpositive_counts_before_loading(monkeypatch, tmp_path, flag):
    from scripts import ud_mtp_ar_verify_numerics_gate as gate

    monkeypatch.setattr(sys, "argv", ["gate", "--json", str(tmp_path / "out.json"), flag, "0"])
    with pytest.raises(SystemExit) as exc:
        gate.main()
    assert exc.value.code == 2


def test_cli_rejects_empty_prompt_suite_before_loading(monkeypatch, tmp_path):
    from scripts import ud_mtp_ar_verify_numerics_gate as gate

    monkeypatch.setattr(sys, "argv", ["gate", "--json", str(tmp_path / "out.json"), "--prompts"])
    with pytest.raises(SystemExit) as exc:
        gate.main()
    assert exc.value.code == 2
