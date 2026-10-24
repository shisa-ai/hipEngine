"""Unit tests for the crossed-state PARO diagnostic (CPU only, no GPU or HIP runtime)."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mtp_paro_crossed_state as crossed
from scripts.mtp_paro_crossed_state import (
    DiagnosticError,
    FocusWindow,
    ambient_env,
    annotate_alignment,
    build_token_batch,
    capture_line,
    compare_logits,
    compare_states,
    finish_report,
    first_mismatch,
    history_alignment,
    numerical_rows,
    parse_focus,
    report_problems,
    restore,
    snapshot,
    source_hashes,
    state_digest,
    state_spans,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- focus window and argument parsing -------------------------------------


def test_parse_focus_accepts_an_inclusive_window() -> None:
    assert parse_focus("85:92") == FocusWindow(low=85, high=92)
    assert parse_focus(" 3 : 4 ").low == 3


@pytest.mark.parametrize("text", ["85", "85:92:1", "a:b", "-1:3", "9:3", ""])
def test_parse_focus_rejects_malformed_windows(text: str) -> None:
    with pytest.raises(DiagnosticError):
        parse_focus(text)


def test_focus_window_selects_ar_states_and_verifier_cycles() -> None:
    window = FocusWindow(low=85, high=92)

    # The first generated token always keeps its pre-generation state, and the
    # margin reaches AR_CONTEXT_MARGIN tokens past each end of the window.
    assert window.wants_ar_state(0)
    assert window.wants_ar_state(82)
    assert not window.wants_ar_state(81)
    assert window.wants_ar_state(95)
    assert not window.wants_ar_state(96)

    # A verifier cycle is captured when its rows overlap the window.
    assert window.wants_capture(0, 4)
    assert window.wants_capture(82, 4)
    assert window.wants_capture(84, 2)
    assert not window.wants_capture(93, 2)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", ()), ("2,3", (2, 3)), ("3,2,4", (2, 3, 4)), ("strict", ("strict",))],
)
def test_parse_choice_list_accepts_known_values(text: str, expected: tuple) -> None:
    assert crossed.parse_choice_list(text, (2, 3, 4, "strict"), "--flag") == expected


# --- state comparison ------------------------------------------------------


def test_state_digest_is_order_independent_and_content_sensitive() -> None:
    left = {"a": np.array([1, 2], dtype=np.uint8), "b": np.array([3], dtype=np.uint8)}
    right = {"b": np.array([3], dtype=np.uint8), "a": np.array([1, 2], dtype=np.uint8)}

    assert state_digest(left) == state_digest(right)
    assert state_digest({**left, "a": np.array([1, 9], dtype=np.uint8)}) != state_digest(left)
    with pytest.raises(DiagnosticError, match="missing"):
        compare_states({"a": np.zeros(1, np.uint8)}, {})


def test_compare_states_reports_unequal_views() -> None:
    expected = {"4.conv": np.array([1, 2], dtype=np.uint8), "7.k": np.array([3], dtype=np.uint8)}

    equal = compare_states(expected, {name: value.copy() for name, value in expected.items()})
    assert equal["equal"] is True
    assert equal["unequal"] == []
    assert equal["sha256_expected"] == equal["sha256_observed"]

    drifted = compare_states(expected, {**expected, "7.k": np.array([4], dtype=np.uint8)})
    assert drifted["equal"] is False
    assert drifted["unequal"] == ["7.k"]
    assert drifted["sha256_expected"] != drifted["sha256_observed"]


def test_compare_states_rejects_mismatched_view_sets() -> None:
    with pytest.raises(DiagnosticError, match="extra"):
        compare_states({}, {"a": np.zeros(1, np.uint8)})


# --- logit comparison ------------------------------------------------------


def _logits(rows: int = 4, vocab: int = 64) -> np.ndarray:
    return np.random.default_rng(7).normal(size=(rows, vocab)).astype(np.float32)


def test_compare_logits_marks_identical_rows_bit_exact_and_catches_drift() -> None:
    reference = _logits()

    rows = compare_logits(reference, reference.copy())

    assert len(rows) == 4
    assert all(row["kl"] == 0.0 for row in rows)
    assert all(row["bit_exact"] and row["top1_equal"] for row in rows)
    assert all(row["max_abs_logit_delta"] == 0.0 for row in rows)

    candidate = reference.copy()
    candidate[1, [0, 1]] = candidate[1, [1, 0]]
    drifted = compare_logits(reference, candidate)

    assert drifted[0]["bit_exact"] is True
    assert drifted[1]["bit_exact"] is False
    assert drifted[1]["kl"] > 0.0
    assert drifted[1]["max_abs_logit_delta"] > 0.0


def test_compare_logits_rejects_non_finite_and_unaligned_input() -> None:
    reference = _logits()
    broken = reference.copy()
    broken[0, 0] = np.nan

    with pytest.raises(DiagnosticError, match="finite"):
        compare_logits(reference, broken)
    with pytest.raises(DiagnosticError, match="aligned rank-2"):
        compare_logits(reference, reference[:2])


# --- verifier row construction --------------------------------------------


def test_build_token_batch_matches_the_public_row_layout() -> None:
    batch = build_token_batch((271, 71093, 12305, 198), 28)

    assert batch.mode == "verify_chain"
    assert batch.rows == 4
    assert batch.tokens == (271, 71093, 12305, 198)
    assert batch.positions == (28, 29, 30, 31)
    assert batch.root_rows == (0,)
    assert batch.candidate_rows == (1, 2, 3)
    assert batch.parent_rows == (-1, 0, 1, 2)


@pytest.mark.parametrize("tokens", [(), (271,), (271, -1)])
def test_build_token_batch_rejects_degenerate_rows(tokens: tuple[int, ...]) -> None:
    with pytest.raises(DiagnosticError):
        build_token_batch(tokens, 28)


# --- state views, snapshot, restore (fake device, no HIP) ------------------


class _FakeDType:
    def __init__(self, itemsize: int) -> None:
        self.itemsize = itemsize


class _FakeTensor:
    def __init__(self, ptr: int, shape: tuple[int, ...], itemsize: int = 2) -> None:
        self.ptr = ptr
        self.shape = shape
        self.dtype = _FakeDType(itemsize)


class _FakeRuntime:
    def __init__(self) -> None:
        self.synchronizations = 0

    def device_synchronize(self) -> None:
        self.synchronizations += 1


class _FakeSession:
    """Minimal stand-in for the resident session's carried-state surface."""

    def __init__(self) -> None:
        self.runtime = _FakeRuntime()
        self.positions: list[tuple[int, int]] = []
        self.calls: list[dict] = []
        self.linear_states = {4: ()}
        self.full_caches = {7: ()}
        self._linear = {(4, 0): (_FakeTensor(0x1000, (2, 3)), _FakeTensor(0x2000, (2, 3)))}
        self._full = {(7, 0): (_FakeTensor(0x3000, (2, 8, 3, 4)), _FakeTensor(0x4000, (2, 8, 3, 4)))}

    def _slot_linear_state(self, layer: int, slot: int) -> tuple[_FakeTensor, ...]:
        return self._linear[(layer, slot)]

    def _slot_full_cache(self, layer: int, slot: int) -> tuple[_FakeTensor, ...]:
        return self._full[(layer, slot)]

    def _set_slot_position(self, position: int, *, slot: int) -> None:
        self.positions.append((int(position), int(slot)))


@pytest.fixture
def fake_device(monkeypatch: pytest.MonkeyPatch) -> dict[int, bytearray]:
    """Back the module's device copies with a byte-addressable fake device."""

    memory: dict[int, bytearray] = {
        0x1000: bytearray(range(12)),
        0x2000: bytearray(range(12, 24)),
        0x3000: bytearray(4096),
        0x4000: bytearray(4096),
        # Verifier-shaped scratch used by the hook-glue test.
        0x5000: bytearray(4096),
        0x6000: bytearray(4096),
        0x7000: bytearray(4096),
        0x8000: bytearray(4096),
    }

    def copy_d2h(host_ptr, buffer, nbytes=None, *, runtime=None) -> None:
        count = buffer.nbytes if nbytes is None else nbytes
        source = bytes(memory[buffer.ptr][:count])
        assert len(source) == count
        ctypes.memmove(host_ptr, (ctypes.c_uint8 * count).from_buffer_copy(source), count)

    def copy_h2d(buffer, host_ptr, nbytes=None, *, runtime=None) -> None:
        count = buffer.nbytes if nbytes is None else nbytes
        memory[buffer.ptr][:count] = bytes((ctypes.c_uint8 * count).from_address(host_ptr))

    monkeypatch.setattr(crossed, "copy_device_to_host", copy_d2h)
    monkeypatch.setattr(crossed, "copy_host_to_device", copy_h2d)
    return memory


def test_state_spans_scales_full_cache_spans_with_the_context() -> None:
    session = _FakeSession()

    spans = state_spans(session, 5)

    assert spans["4.conv"] == (0x1000, 12)
    assert spans["4.recurrent"] == (0x2000, 12)
    assert spans["7.k"] == (0x3000, 5 * 3 * 4 * 2)
    assert spans["7.v"] == (0x4000, 5 * 3 * 4 * 2)
    assert state_spans(session, 1)["7.k"][1] == 3 * 4 * 2
    with pytest.raises(DiagnosticError, match="positive context"):
        state_spans(session, 0)


def test_snapshot_and_restore_roundtrip_raw_device_bytes(
    fake_device: dict[int, bytearray],
) -> None:
    session = _FakeSession()
    state = snapshot(session, 5)

    assert set(state) == {"4.conv", "4.recurrent", "7.k", "7.v"}
    assert all(value.dtype == np.uint8 for value in state.values())
    assert state["4.conv"].nbytes == 12
    assert state["7.k"].nbytes == 5 * 3 * 4 * 2
    assert fake_device[0x1000][:12] == bytes(state["4.conv"])

    # Move the fake device away, then prove restore puts the snapshot back.
    fake_device[0x1000][:12] = bytes(range(12, 24))
    fake_device[0x3000][:120] = b"\x00" * 120
    restore(session, state, 5)

    assert fake_device[0x1000][:12] == bytes(range(12))
    assert fake_device[0x3000][:120] == bytes(state["7.k"])
    assert session.positions == [(4, 0)]
    assert session.runtime.synchronizations >= 2


def test_restore_fails_when_the_copy_did_not_land(
    fake_device: dict[int, bytearray], monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _FakeSession()
    state = snapshot(session, 5)
    fake_device[0x1000][:12] = bytes(range(12, 24))
    monkeypatch.setattr(crossed, "copy_host_to_device", lambda *args, **kwargs: None)

    with pytest.raises(DiagnosticError, match="did not land"):
        restore(session, state, 5)


def test_restore_rejects_a_snapshot_that_does_not_fit_the_view(
    fake_device: dict[int, bytearray],
) -> None:
    session = _FakeSession()
    state = snapshot(session, 5)

    with pytest.raises(DiagnosticError, match="needs 120"):
        restore(session, {**state, "7.k": np.zeros(64, dtype=np.uint8)}, 5)
    with pytest.raises(DiagnosticError, match="raw uint8"):
        restore(session, {**state, "4.conv": state["4.conv"].view(np.int8)}, 5)


# --- environment, provenance, history alignment ---------------------------


def test_route_env_applies_and_restores_the_full_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = crossed.route_env_keys()
    kept, cleared = keys[0], keys[-1]
    monkeypatch.setenv(kept, "keep")
    monkeypatch.setenv(cleared, "stale")
    snapshot = {key: "1" for key in keys}

    with crossed.route_env(snapshot, keys):
        assert crossed.route_env_snapshot(keys) == snapshot

    assert os.environ[kept] == "keep"
    assert os.environ[cleared] == "stale"

    # A key the snapshot leaves out is cleared, never inherited from the ambient env.
    with crossed.route_env({**snapshot, cleared: None}, keys):
        assert cleared not in os.environ

    assert os.environ[cleared] == "stale"


def test_verifier_flags_bind_profiles_over_the_route_names() -> None:
    names = crossed.verifier_flags()
    strict = crossed.verifier_flags("strict")
    production = crossed.verifier_flags("production")

    assert set(strict) <= set(names) and set(production) <= set(names)
    assert strict != production
    with pytest.raises(DiagnosticError, match="unknown verifier profile"):
        crossed.verifier_flags("eager")


def test_source_hashes_records_paths_and_digests(tmp_path: Path) -> None:
    source = tmp_path / "tool.py"
    source.write_text("print('a')\n", encoding="utf-8")

    hashes = source_hashes({"tool": source, "module": crossed})

    assert hashes["tool"] == {"path": str(source),
                              "sha256": hashlib.sha256(b"print('a')\n").hexdigest()}
    assert Path(hashes["module"]["path"]).is_file()
    with pytest.raises(DiagnosticError, match="not a file"):
        source_hashes({"missing": tmp_path / "absent.py"})


def test_ambient_env_keeps_only_hipengine_selectors() -> None:
    ambient = ambient_env(
        {
            "HIPENGINE_GDN_TLOOP_C1_EXACT": "1",
            "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS": "3",
            "OTHER_SETTING": "2",
        }
    )

    assert ambient == {
        "HIPENGINE_GDN_TLOOP_C1_EXACT": "1",
        "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS": "3",
    }


def test_first_mismatch_reports_the_first_disagreement() -> None:
    assert first_mismatch([1, 2, 3], [1, 9, 3]) == 1
    assert first_mismatch([1, 2, 3], [1, 2, 3]) is None
    assert first_mismatch([1, 2, 3], [1, 2]) is None


def test_history_alignment_needs_the_mismatch_and_the_root_index() -> None:
    assert history_alignment(83, 89)["aligned"] is True
    assert history_alignment(89, 89)["aligned"] is True
    diverged = history_alignment(95, 89)
    assert diverged["aligned"] is False
    assert diverged["shared_history_tokens"] == 89
    assert diverged["reason"] == "free_running_divergence"
    assert history_alignment(95, None)["aligned"] is True
    unknown = history_alignment(95, 89, known=False)
    assert unknown["aligned"] is None
    assert unknown["reason"] == "first_mismatch_not_recorded"


# --- report completeness ---------------------------------------------------


def _item(tmp_path: Path, label: str, kind: str = "carried", **overrides: object) -> dict:
    raw = tmp_path / f"raw-{label}.npz"
    np.savez_compressed(raw, logits=np.zeros((2, 8), dtype=np.float32))
    item: dict = {
        "kind": kind,
        "label": label,
        "rows": {"batch": 4, "shared": 4},
        "logits": [{"kl": 0.0, "top1_equal": True, "max_abs_logit_delta": 0.0, "bit_exact": True}],
        "selected_state": {"comparable": True, "row": 2, "equal": True, "unequal": []},
        "result": {"accepted_count": 3},
        "extra": {},
        "raw": str(raw),
    }
    if kind == "carried":
        item["extra"] = {
            "serial_vs_public_logits": [],
            "serial_vs_public_hidden_exact": [True],
            "same_state_hidden_exact": [True],
            "commit_row_agrees_with_public": True,
        }
    item.update(overrides)
    return item


def _capture(tmp_path: Path, **overrides: object) -> dict:
    capture: dict = {
        "context": 111,
        "root_index": 83,
        "tokens": [198, 256, 471, 2972],
        "rows": 4,
        "public_result": {"accepted_count": 3},
        "public_hidden_sha256": "0" * 64,
        "ar_state_available": True,
        "carried_ar_vs_verifier_state": {"equal": False, "unequal": ["4.conv"]},
        "public_repeat_exact": True,
        "post_state_repeat": {"equal": True, "unequal": []},
        "env_restored": True,
        "comparisons": [_item(tmp_path, "verifier-carried"), _item(tmp_path, "ar-carried")],
    }
    capture.update(overrides)
    return capture


def _report(tmp_path: Path, **overrides: object) -> dict:
    report: dict = {
        "inputs": {"focus": [85, 92], "outputs": 129, "cross_widths": [], "cross_profiles": []},
        "prompt_ids": [1, 2, 3],
        "ar_ids": list(range(129)),
        "mtp_ids": list(range(129)),
        "first_mismatch": 89,
        "schedule": [{"context": 111, "root_index": 83, "rows": 4}],
        "captures": [_capture(tmp_path)],
        "hook_failures": [],
        "route": {"specdec2_mtp2_used": True, "serial_decode_fallback": False},
        "mtp_accounting": {"cycles": 1, "recoverable_failures": 0, "failure_reasons": []},
    }
    report.update(overrides)
    return annotate_alignment(report)


def test_report_problems_accepts_a_complete_capture(tmp_path: Path) -> None:
    report = _report(tmp_path)
    seen: list[str] = []

    assert report_problems(report) == []
    assert report_problems(report, exists=lambda path: seen.append(path) or True) == []
    assert seen




def test_report_problems_allows_a_context_without_an_ar_state(tmp_path: Path) -> None:
    report = _report(tmp_path)
    capture = report["captures"][0]
    capture.update(ar_state_available=False, comparisons=capture["comparisons"][:1])
    capture.pop("carried_ar_vs_verifier_state")

    assert report_problems(report) == []


def test_report_problems_flags_requested_widths_and_profiles(tmp_path: Path) -> None:
    report = _report(tmp_path)
    report["inputs"].update(cross_widths=[3], cross_profiles=["strict"])

    problems = report_problems(report)

    assert any("requested width(s) [3]" in problem for problem in problems), problems
    assert any("requested profile(s) ['strict']" in problem for problem in problems), problems


def test_report_problems_flags_a_requested_profile_with_no_profile_entry(
    tmp_path: Path,
) -> None:
    # A profile entry that exists but omits a requested arm is caught by the
    # per-item check; this is the harder case, where no profile entry ran at all.
    report = _report(tmp_path)
    report["inputs"]["cross_profiles"] = ["strict", "production"]
    assert any("requested profile(s)" in problem for problem in report_problems(report))

    report["captures"][0]["comparisons"].append(
        _item(tmp_path, "profile:strict", kind="profile", profiles=["strict"])
    )

    problems = report_problems(report)

    assert any("requested profile(s) ['production']" in problem for problem in problems), problems


_DELETE = object()


def _set(report: dict, path: tuple, value: object) -> None:
    target = report
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        target.pop(path[-1], None)
    else:
        target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "fragment"),
    [
        (("captures",), [], "no crossed-state capture"),
        (("schedule",), [], "verifier hook never ran"),
        (("prompt_ids",), [], "no prompt token ids"),
        (("ar_ids",), [], "ar_ids is empty"),
        (("mtp_ids",), list(range(9)), "holds 9 tokens"),
        (("first_mismatch",), _DELETE, "first mismatch was not recorded"),
        (("hook_failures",), [{"hook": "verify_chain_bulk_and_commit",
                               "error": "DiagnosticError: boom"}], "raised DiagnosticError"),
        (("captures", 0, "public_repeat_exact"), False, "perturbed the public call"),
        (("captures", 0, "post_state_repeat"), {"equal": False, "unequal": ["7.k"]},
         "did not reproduce the committed state"),
        (("captures", 0, "env_restored"), False, "was not restored"),
        (("captures", 0, "tokens"), _DELETE, "is missing tokens"),
        (("captures", 0, "history_alignment"), None, "no token-history alignment label"),
        (("captures", 0, "comparisons", 0, "raw"), "/nonexistent/x.npz", "is missing"),
        (("captures", 0, "comparisons", 0, "logits"), [], "the logit comparison is missing"),
        (("captures", 0, "comparisons", 0, "selected_state"), _DELETE,
         "selected-state comparison is missing"),
        (("captures", 0, "comparisons", 0, "extra", "same_state_hidden_exact"), _DELETE,
         "same_state_hidden_exact is missing"),
        (("route", "serial_decode_fallback"), True, "fell back to serial decode"),
        (("route", "specdec2_mtp2_used"), False, "did not run speculative MTP"),
        (("mtp_accounting", "recoverable_failures"), 2, "recovered from 2 failed cycle"),
        (("mtp_accounting", "failure_reasons"), [{"category": "hook", "detail": "boom"}],
         "recoverable cycle failure hook"),
        (("mtp_accounting", "cycles"), 42, "42 cycles but 1 verifier"),
        (("inputs", "focus"), [200, 210], "beyond the 129-token budget"),
    ],
)
def test_report_problems_flags_incomplete_runs(
    tmp_path: Path, path: tuple, value: object, fragment: str
) -> None:
    report = _report(tmp_path)
    _set(report, path, value)

    problems = report_problems(report)

    assert any(fragment in problem for problem in problems), problems


def test_report_problems_flags_an_appended_skip_and_a_missing_ar_replay(tmp_path: Path) -> None:
    skipped = _report(tmp_path)
    skipped["captures"][0]["comparisons"].append(
        {"kind": "width", "label": "width3", "width": 3, "skipped": "row capacity"}
    )
    assert any("not run (row capacity)" in p for p in report_problems(skipped))

    dropped = _report(tmp_path)
    dropped["captures"][0]["comparisons"].pop(1)
    assert any("no AR-carried replay" in p for p in report_problems(dropped))




def test_finish_report_stamps_status_and_completeness(tmp_path: Path) -> None:
    complete = finish_report(_report(tmp_path))
    assert (complete["status"], complete["complete"], complete["problems"]) == ("complete", True, [])

    incomplete = _report(tmp_path)
    incomplete["captures"] = []
    finish_report(incomplete)
    assert (incomplete["status"], incomplete["complete"]) == ("incomplete", False)
    assert incomplete["problems"]

    failed = finish_report(_report(tmp_path), failed=True)
    assert (failed["status"], failed["complete"]) == ("failed", False)

    line = capture_line(_capture(tmp_path))
    assert "context=111" in line
    assert "ar_state_equal=False" in line
    assert "same_state_kl=[0.0]" in line






def test_annotate_alignment_pools_only_the_verifier_checkpoint(tmp_path: Path) -> None:
    report = _report(tmp_path)
    capture = report["captures"][0]

    # The AR checkpoint is a cross-check on the canonical rows, not a second sample.
    assert [item["scope"] for item in capture["comparisons"]] == [
        "same_history_carried_state",
        "same_history_ar_checkpoint",
    ]
    assert [item["numerical_pool_eligible"] for item in capture["comparisons"]] == [True, False]
    assert report["alignment"]["canonical_comparisons"] == 1
    assert report["alignment"]["ar_checkpoint_comparisons"] == 1
    assert report["alignment"]["cross_history_comparisons"] == 0
    assert [row["label"] for row in numerical_rows(report)] == ["verifier-carried"]

    diverged = _report(tmp_path, first_mismatch=89)
    diverged["captures"][0]["root_index"] = 95
    annotate_alignment(diverged)
    ar_item = diverged["captures"][0]["comparisons"][1]

    assert ar_item["scope"] == "cross_history_carried_state"
    assert ar_item["numerical_pool_eligible"] is False
    assert "not a kernel arithmetic result" in ar_item["alignment_note"]
    assert diverged["alignment"]["cross_history_comparisons"] == 1
    assert [row["label"] for row in numerical_rows(diverged)] == ["verifier-carried"]


def test_annotate_alignment_marks_the_carried_state_as_expected_to_differ(
    tmp_path: Path,
) -> None:
    state = _report(tmp_path)["captures"][0]["carried_ar_vs_verifier_state"]

    assert state["scope"] == "state_only_carried_state"
    assert state["numerical_pool_eligible"] is False
    assert state["expected_to_differ"] is False


def test_a_skipped_width_lands_in_no_row_list(tmp_path: Path) -> None:
    report = _report(tmp_path)
    report["captures"][0]["comparisons"].append(
        {"kind": "width", "label": "width4", "width": 4, "skipped": "row capacity"}
    )
    annotate_alignment(report)

    assert all(row["label"] != "width4" for row in numerical_rows(report))
    assert all(row["label"] != "width4" for row in crossed.diagnostic_observations(report))


def test_annotate_alignment_keeps_width_and_profile_rows_out_of_the_pool(tmp_path: Path) -> None:
    report = _report(tmp_path)
    capture = report["captures"][0]
    capture["comparisons"] += [
        _item(tmp_path, "width3", kind="width", width=3, scope="same_state_shared_prefix"),
        _item(tmp_path, "strict-vs-production", kind="profile", profiles=["strict", "production"],
              scope="same_state_same_rows_env_flags"),
    ]

    annotate_alignment(report)

    assert [item["scope"] for item in capture["comparisons"][2:]] == [
        "same_state_shared_prefix",
        "same_state_same_rows_env_flags",
    ]
    # Width 3 and width 4 share the same prefix rows, and those rows also appear
    # in the carried comparison, so they cannot be counted as samples.
    assert [item["numerical_pool_eligible"] for item in capture["comparisons"]] == [
        True, False, False, False,
    ]
    assert {row["scope"] for row in numerical_rows(report)} == {"same_history_carried_state"}
    assert {row["scope"] for row in crossed.diagnostic_observations(report)} == {
        "same_history_ar_checkpoint",
        "same_state_shared_prefix",
        "same_state_same_rows_env_flags",
    }
    assert report["alignment"]["poolable_scopes"] == ["same_history_carried_state"]


# --- hook installation and failure recording ------------------------------


def _stub_session_class() -> type:
    class _StubSession:
        def _run_layers(self, **kwargs):  # pragma: no cover - never called here
            raise AssertionError("stub")

        def _sample_verify_rows_from_hidden(self, hidden, rows, **kwargs):
            raise AssertionError("stub")

        def verify_chain_bulk_and_commit(self, batch, **kwargs):
            raise AssertionError("stub")

    return _StubSession


def _capture_object(tmp_path: Path, session_cls: type, **kwargs: object) -> crossed.CrossedStateCapture:
    return crossed.CrossedStateCapture(
        session_cls=session_cls,
        focus=FocusWindow(low=85, high=92),
        out=tmp_path / "report.json",
        report={"hook_failures": [], "captures": [], "schedule": []},
        declared_env={},
        **kwargs,
    )


def test_install_and_uninstall_restore_the_session_methods(tmp_path: Path) -> None:
    session_cls = _stub_session_class()
    originals = {name: getattr(session_cls, name) for name in crossed.HOOKS}
    capture = _capture_object(tmp_path, session_cls)

    capture.install()
    assert all(getattr(session_cls, name) is not originals[name] for name in originals)

    capture.uninstall()
    assert all(getattr(session_cls, name) is originals[name] for name in originals)


def test_install_rejects_a_session_without_the_expected_hooks(tmp_path: Path) -> None:
    class _Bare:
        pass

    with pytest.raises(DiagnosticError, match="_run_layers"):
        _capture_object(tmp_path, _Bare).install()


def test_hook_failure_is_recorded_and_reraised(tmp_path: Path) -> None:
    capture = _capture_object(tmp_path, _stub_session_class())
    capture._report["kl"] = np.float32(1.5)

    with pytest.raises(DiagnosticError, match="boom"):
        with capture._guard("verify_chain_bulk_and_commit"):
            raise DiagnosticError("boom")

    failure = capture._report["hook_failures"][0]
    assert failure["hook"] == "verify_chain_bulk_and_commit"
    assert "DiagnosticError: boom" in failure["error"]
    assert "Traceback" in failure["traceback"]
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["hook_failures"][0]["detail"] == "boom"
    assert written["kl"] == 1.5  # numpy scalars stay serialisable






def test_stale_hidden_state_is_rejected(tmp_path: Path) -> None:
    capture = _capture_object(tmp_path, _stub_session_class())
    hidden = np.zeros((2, 4), dtype=np.uint16)
    capture.verify_hidden = hidden

    # A hook that did not run leaves the previous row in place; comparing
    # against it would report a stale row as fresh.
    with pytest.raises(DiagnosticError, match="did not record"):
        capture._fresh(hidden, 3, 3, "the cycle at context 111", "_sample_verify_rows_from_hidden")

    capture.verify_epoch = 4
    assert capture._fresh(hidden, 4, 3, "where", "hook") is hidden
    with pytest.raises(DiagnosticError, match="did not record"):
        capture._fresh(None, 5, 4, "where", "hook")


# --- hook glue: one full capture against a fake resident session ----------

class _FakeVerifyResult:
    def __init__(self, *, context: int, rows: int, commit_row: int) -> None:
        self.gpu_accept_match_cpu = True
        self.accepted_count = commit_row
        self.commit_row = commit_row
        self.commit_position = context + commit_row
        self.next_token = 7
        self.target_top1 = tuple(range(rows))

    def to_json_dict(self) -> dict:
        return {"accepted_count": self.accepted_count, "commit_row": self.commit_row,
                "commit_position": self.commit_position}


def _row_logits(position: int, vocab: int = 8) -> np.ndarray:
    return (np.arange(vocab, dtype=np.float32) * 0.125 + position).astype(np.float32)


def _row_hidden(position: int, hidden_size: int = 4) -> np.ndarray:
    return (np.arange(hidden_size, dtype=np.uint16) + position).astype(np.uint16)


class _FakeCaptureSession(_FakeSession):
    """Resident-session stand-in with deterministic, position-indexed outputs.

    ``verify_chain_bulk_and_commit`` mirrors the resident signature exactly: the
    four capture keywords are required, so a caller that omits one fails here the
    same way it fails on the real session.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.vocab_size = 8
        self.max_batch_size = 4
        self.lm_logits = SimpleNamespace(ptr=0x5000)
        self.verify_lm_logits = SimpleNamespace(ptr=0x6000)
        self.memory: dict[int, bytearray] = {}
        self.fail_on_call: int | None = None
        self.prepared_rows = 0
        self.scratch_calls: list[dict] = []

    def prepare_specdec2_verify_scratch(self, *, rows: int, chain_attn_mode: str,
                                        max_context_tokens: int) -> None:
        """Mirror the resident scratch reservation, including its row bounds."""
        if not 1 < int(rows) <= self.max_batch_size:
            raise AssertionError(f"scratch rows {rows} outside (1, {self.max_batch_size}]")
        if str(chain_attn_mode) not in {"c1_loop", "decode_batched"}:
            raise AssertionError(f"unsupported scratch chain_attn_mode {chain_attn_mode!r}")
        self.scratch_calls.append({"rows": int(rows), "chain_attn_mode": str(chain_attn_mode),
                                   "max_context_tokens": int(max_context_tokens)})
        self.prepared_rows = int(rows)

    def _write(self, ptr: int, array: np.ndarray) -> None:
        self.memory.setdefault(ptr, bytearray(4096))[: array.nbytes] = array.tobytes()

    def _run_layers(self, *, position: int, **kwargs: object) -> _FakeTensor:
        self._write(0x7000, _row_hidden(position))
        return _FakeTensor(0x7000, (4,))

    def step(self, token_id: int, *, position: int, sample: bool = True) -> None:
        self._write(0x5000, _row_logits(position))
        self._run_layers(position=position)

    def _sample_verify_rows_from_hidden(self, hidden: _FakeTensor, rows: int, **kwargs: object):
        return None

    def verify_chain_bulk_and_commit(
        self,
        batch,
        *,
        base_slot: int,
        capture_layer_ids,
        capture_hidden_concat,
        capture_row_start: int,
        stream: int = 0,
        graph_mode: str = "off",
        chain_attn_mode: str = "c1_loop",
        canonicalize_after: bool = True,
        synchronize_after_commit: bool = True,
        candidate_token_ids_i32=None,
    ) -> _FakeVerifyResult:
        context = int(batch.positions[0])
        rows = int(batch.rows)
        if graph_mode != "off":
            raise AssertionError(f"the public route runs graph_mode='off', got {graph_mode!r}")
        if chain_attn_mode not in {"c1_loop", "batched", "decode_batched"}:
            raise AssertionError(f"unsupported chain_attn_mode {chain_attn_mode!r}")
        if int(base_slot) != 0:
            raise AssertionError(f"the public route verifies slot 0, got {base_slot}")
        captured = tuple(int(layer) for layer in capture_layer_ids)
        if capture_hidden_concat.shape[1] != len(captured) * self.config.hidden_size:
            raise AssertionError("capture width must be captured layers * hidden size")
        if capture_row_start < 0 or capture_row_start + rows > capture_hidden_concat.shape[0]:
            raise AssertionError(
                f"capture rows {capture_row_start}+{rows} exceed the "
                f"{capture_hidden_concat.shape[0]}-row capture tensor")
        if rows > self.prepared_rows:
            raise AssertionError(
                f"verifier rows {rows} exceed the {self.prepared_rows}-row prepared scratch")
        self.calls.append({
            "rows": rows, "context": context, "base_slot": int(base_slot),
            "capture_layer_ids": captured, "capture_row_start": int(capture_row_start),
            "chain_attn_mode": str(chain_attn_mode), "graph_mode": str(graph_mode),
            "canonicalize_after": bool(canonicalize_after),
            "synchronize_after_commit": bool(synchronize_after_commit),
            "candidate_token_ids": candidate_token_ids_i32 is not None,
            "env": crossed.route_env_snapshot(crossed.route_env_keys()),
        })
        if self.fail_on_call is not None and len(self.calls) - 1 == self.fail_on_call:
            raise DiagnosticError(f"fake verifier failure on call {self.fail_on_call}")
        self._write(0x6000, np.stack([_row_logits(context + row) for row in range(rows)]))
        self._write(0x8000, np.stack([_row_hidden(context + row) for row in range(rows)]))
        self._sample_verify_rows_from_hidden(_FakeTensor(0x8000, (rows, 4)), rows)
        # The fast verifier variant can commit a different row from the strict one.
        fast = os.environ.get("HIPENGINE_GDN_TLOOP_C1_EXACT") == "0"
        return _FakeVerifyResult(context=context, rows=rows,
                                 commit_row=1 if fast else min(2, rows - 1))


def _public_kwargs(rows: int, *, chain_attn_mode: str = "c1_loop") -> dict:
    """The keyword set the public MTP2 adapter passes to the verifier."""
    return {
        "base_slot": 0,
        "capture_layer_ids": (),
        "capture_hidden_concat": crossed.Tensor.from_handle(
            0, (int(rows), 0), crossed.DType.BF16, crossed.Device("hip", 0)),
        "capture_row_start": 0,
        "chain_attn_mode": chain_attn_mode,
        "graph_mode": "off",
        "canonicalize_after": False,
        "synchronize_after_commit": False,
        "candidate_token_ids_i32": None,
    }


def _run_hook(
    tmp_path: Path,
    *,
    tokens: tuple[int, ...] = (271, 71093, 12305),
    widths: tuple[int, ...] = (2, 3, 4),
    profiles: tuple[str, ...] = ("strict", "production"),
    chain_attn_mode: str = "c1_loop",
    declared_chain_attn_mode: str | None = None,
    declared_env: dict | None = None,
    fail_on_call: int | None = None,
) -> tuple[_FakeCaptureSession, crossed.CrossedStateCapture, dict]:
    session = _FakeCaptureSession()
    session.fail_on_call = fail_on_call
    # The public adapter reserves verifier scratch for its own row count during
    # prompt priming, before any verifier cycle runs.
    session.prepared_rows = len(tokens)
    report: dict = {
        "inputs": {"focus": [0, 0], "outputs": 129, "cross_widths": list(widths),
                   "cross_profiles": list(profiles)},
        "prompt_ids": list(range(28)), "ar_ids": list(range(129)),
        "mtp_ids": list(range(129)), "first_mismatch": None,
        "schedule": [], "captures": [], "hook_failures": [],
        "route": {"specdec2_mtp2_used": True, "serial_decode_fallback": False},
        "mtp_accounting": {"cycles": 1, "recoverable_failures": 0, "failure_reasons": []},
    }
    capture = crossed.CrossedStateCapture(
        session_cls=_FakeCaptureSession, focus=FocusWindow(low=0, high=0),
        out=tmp_path / "report.json", report=report, declared_env=declared_env or {},
        widths=widths, profiles=profiles,
        chain_attn_mode=declared_chain_attn_mode or chain_attn_mode,
        outputs=129, max_sequence_length=8192)
    capture.prompt_length = 28
    capture.ar_ids = list(report["ar_ids"])
    # The AR arm always snapshots the pre-generation state at the prompt end.
    capture.ar_states[28] = snapshot(session, 28)
    batch = build_token_batch(tokens, 28)
    capture.install()
    try:
        capture._verify_hook(session, batch, **_public_kwargs(int(batch.rows),
                                                             chain_attn_mode=chain_attn_mode))
    finally:
        capture.uninstall()
    return session, capture, report


def test_one_capture_through_the_hooks_reports_a_complete_run(tmp_path, fake_device) -> None:
    session, capture, report = _run_hook(tmp_path)
    captured = report["captures"][0]

    assert len(report["schedule"]) == 1
    assert report["schedule"][0]["context"] == 28
    assert report["schedule"][0]["root_index"] == 0
    assert report["schedule"][0]["tokens"] == [271, 71093, 12305]
    assert captured["ar_state_available"] is True
    assert captured["carried_ar_vs_verifier_state"]["equal"] is True
    assert captured["public_repeat_exact"] is True
    assert captured["post_state_repeat"]["equal"] is True
    assert captured["env_restored"] is True
    assert [item["label"] for item in captured["comparisons"]] == [
        "verifier-carried", "ar-carried", "width2", "width3", "width4",
        "strict-vs-production",
    ]

    carried = captured["comparisons"][0]
    assert [row["bit_exact"] for row in carried["logits"]] == [True] * 3
    assert carried["extra"]["same_state_hidden_exact"] == [True] * 3
    assert carried["extra"]["commit_row_agrees_with_public"] is True
    assert carried["selected_state"]["equal"] is True
    assert Path(carried["raw"]).is_file()

    # A width inside the public row set reuses the prefix; width 4 borrows one
    # fixed AR-teacher token, and that extra row stays diagnostic.
    assert captured["comparisons"][3]["suffix"] is None
    assert captured["comparisons"][4]["suffix"]["tokens"] == [3]
    assert captured["comparisons"][4]["suffix"]["diagnostic_only"] is True
    assert captured["comparisons"][4]["rows"] == {"batch": 4, "shared": 3}
    assert [row["bit_exact"] for row in captured["comparisons"][4]["logits"]] == [True] * 3

    profile = captured["comparisons"][5]
    assert [row["bit_exact"] for row in profile["logits"]] == [True] * 3
    assert set(profile["extra"]["arms"]) == {"strict", "production"}

    annotate_alignment(report)
    finish_report(report)
    capture.write()

    assert {row["scope"] for row in report["numerical_rows"]} == {"same_history_carried_state"}
    assert len(report["numerical_rows"]) == 3
    assert {row["label"] for row in report["diagnostic_observations"]} == {
        "ar-carried", "width2", "width3", "width4", "strict-vs-production"}
    assert report["status"] == "complete", report["problems"]
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["complete"] is True


def test_crossed_calls_match_the_resident_verifier_api(tmp_path, fake_device) -> None:
    session, capture, report = _run_hook(tmp_path, chain_attn_mode="decode_batched")
    item = report["captures"][0]["comparisons"][0]

    # The fake mirrors the resident signature, so a missing required keyword
    # fails there exactly as it would on the real session.
    assert all(call["base_slot"] == 0 for call in session.calls)
    assert all(call["capture_layer_ids"] == () for call in session.calls)
    assert all(call["capture_row_start"] == 0 for call in session.calls)
    assert all(call["graph_mode"] == "off" for call in session.calls)
    assert item["invocation"] == {
        "base_slot": 0, "capture_layer_ids": [], "capture_row_start": 0,
        "chain_attn_mode": "decode_batched", "graph_mode": "off",
        "canonicalize_after": False, "synchronize_after_commit": True,
        "candidate_token_ids_i32": False, "env": item["invocation"]["env"],
        # The public row count fits the scratch the public route already reserved.
        "reserved": None,
    }
    assert item["invocation"]["env"] == crossed.route_env_snapshot(crossed.route_env_keys())


def test_route_chain_attn_mode_matches_the_adapter_rule() -> None:
    assert crossed.route_chain_attn_mode("strict") == "c1_loop"
    assert crossed.route_chain_attn_mode("production") == "decode_batched"
    assert crossed.route_chain_attn_mode("anything-else") == "decode_batched"


def test_actual_public_mode_overrides_the_predeclared_profile_guess(tmp_path, fake_device) -> None:
    session, _capture, report = _run_hook(
        tmp_path, chain_attn_mode="decode_batched", declared_chain_attn_mode="c1_loop",
    )
    assert {call["chain_attn_mode"] for call in session.calls} == {"decode_batched"}
    assert report["captures"][0]["public_invocation"]["chain_attn_mode"] == "decode_batched"


def test_crossed_calls_preserve_the_incoming_chain_attn_mode(tmp_path, fake_device) -> None:
    session, capture, report = _run_hook(tmp_path, chain_attn_mode="decode_batched")

    modes = {call["chain_attn_mode"] for call in session.calls}
    assert modes == {"decode_batched"}
    assert all(item["invocation"]["chain_attn_mode"] == "decode_batched"
               for item in report["captures"][0]["comparisons"])


def test_route_env_snapshot_is_taken_at_the_hook_not_before_the_binder(
    tmp_path, fake_device, monkeypatch
) -> None:
    keys = crossed.route_env_keys()
    declared = {key: "declared" for key in keys}
    bound = {key: "bound" for key in keys}
    for key, value in bound.items():
        monkeypatch.setenv(key, value)

    session, capture, report = _run_hook(tmp_path, widths=(), profiles=(),
                                         declared_env=declared)
    captured = report["captures"][0]

    assert captured["env_incoming"] == bound
    assert report["env"]["route_flags_at_public_call"] == bound
    assert report["env"]["route_flags_declared_before_binder"] == declared
    assert captured["env_restored"] is True
    assert all(call["env"] == bound for call in session.calls)


def test_route_env_is_restored_when_a_crossed_call_fails(
    tmp_path, fake_device, monkeypatch
) -> None:
    keys = crossed.route_env_keys()
    bound = {key: "bound" for key in keys}
    for key, value in bound.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(DiagnosticError, match="fake verifier failure"):
        _run_hook(tmp_path, widths=(), profiles=(), fail_on_call=2)

    assert crossed.route_env_snapshot(keys) == bound
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["hook_failures"][0]["hook"] == "verify_chain_bulk_and_commit"


def test_profile_selected_state_is_incomparable_when_commit_rows_differ(
    tmp_path, fake_device, monkeypatch
) -> None:
    monkeypatch.setenv("HIPENGINE_GDN_TLOOP_C1_EXACT", "1")
    session, capture, report = _run_hook(tmp_path, widths=(), profiles=("strict", "production"))
    item = report["captures"][0]["comparisons"][-1]

    assert item["kind"] == "profile"
    assert item["logits"]  # logits stay comparable across a different commit row
    assert item["selected_state"]["comparable"] is False
    assert item["selected_state"]["row"] == 1
    assert item["selected_state"]["reference_row"] == 2
    assert item["extra"]["arms"]["production"]["selected_state"]["comparable"] is False
    assert item["extra"]["arms"]["strict"]["selected_state"]["comparable"] is True

    annotate_alignment(report)
    finish_report(report)
    assert report["status"] == "complete", report["problems"]


def test_numerical_rows_use_the_verifier_checkpoint_only(tmp_path, fake_device) -> None:
    session, capture, report = _run_hook(tmp_path)
    annotate_alignment(report)
    captured = report["captures"][0]
    ar_item = next(item for item in captured["comparisons"] if item["label"] == "ar-carried")

    assert captured["carried_ar_vs_verifier_state"]["equal"] is True
    assert ar_item["scope"] == "same_history_ar_checkpoint"
    assert ar_item["numerical_pool_eligible"] is False
    assert ar_item["duplicate_of_canonical"] is True
    assert "repeats the canonical rows" in ar_item["note"]
    assert {row["label"] for row in report["numerical_rows"]} == {"verifier-carried"}
    assert "ar-carried" in {row["label"] for row in report["diagnostic_observations"]}


def test_wider_cross_reserves_verifier_scratch_and_restores_it(tmp_path, fake_device) -> None:
    # Two public rows: width 4 needs two teacher tokens and its own scratch.
    session, capture, report = _run_hook(tmp_path, tokens=(271, 71093), widths=(4,),
                                         profiles=())
    width = report["captures"][0]["comparisons"][-1]

    assert width["kind"] == "width"
    assert width["rows"] == {"batch": 4, "shared": 2}
    assert width["suffix"]["tokens"] == [2, 3]
    assert width["invocation"]["reserved"] == {
        "rows": 4, "chain_attn_mode": "c1_loop", "max_context_tokens": 159}
    # Reserve for the wide call, then hand the scratch back to the public row count.
    assert [call["rows"] for call in session.scratch_calls] == [4, 2]
    assert session.prepared_rows == 2
    annotate_alignment(report)
    finish_report(report)
    assert report["status"] == "complete", report["problems"]


def test_a_width_above_resident_capacity_is_reported_not_skipped_silently(
    tmp_path, fake_device
) -> None:
    session = _FakeCaptureSession()
    # Force the mismatch the guard exists for: the scratch is reserved for the
    # public cycle, but the resident capacity is narrower than the requested width.
    session.max_batch_size = 2
    session.prepared_rows = 4
    report: dict = {
        "inputs": {"focus": [0, 0], "outputs": 4, "cross_widths": [3], "cross_profiles": []},
        "prompt_ids": list(range(28)), "ar_ids": list(range(4)), "mtp_ids": list(range(4)),
        "first_mismatch": None, "schedule": [], "captures": [], "hook_failures": [],
    }
    capture = crossed.CrossedStateCapture(
        session_cls=_FakeCaptureSession, focus=FocusWindow(low=0, high=0),
        out=tmp_path / "report.json", report=report, declared_env={}, widths=(3,))
    capture.prompt_length = 28
    capture.ar_ids = list(report["ar_ids"])
    capture.max_rows = 2
    batch = build_token_batch((271, 71093, 12305, 198), 28)

    capture.install()
    try:
        capture._verify_hook(session, batch, **_public_kwargs(4))
    finally:
        capture.uninstall()

    skipped = report["captures"][0]["comparisons"][-1]
    assert skipped["skipped"].startswith("width 3 exceeds the resident verifier row capacity")
    annotate_alignment(report)
    finish_report(report)
    assert report["status"] == "incomplete"
    assert any("not run (width 3 exceeds" in problem for problem in report["problems"])


# --- provenance and reservation -------------------------------------------


def test_provenance_hashes_cover_the_adapter_and_binder() -> None:
    hashes = crossed.provenance_hashes()

    for relative in ("hipengine/generation/qwen35_paro_mtp2.py",
                     "hipengine/speculative/paro_mtp_profiles.py",
                     "hipengine/runtime/qwen35_paro_runner.py",
                     "scripts/mtp_paro_crossed_state.py"):
        assert relative in hashes, sorted(hashes)
        assert hashes[relative]["path"] == relative
        assert len(hashes[relative]["sha256"]) == 64


def test_reserved_candidate_budget_covers_the_widest_cross() -> None:
    assert crossed.reserved_candidate_budget(3, (2, 3, 4)) == 3
    assert crossed.reserved_candidate_budget(1, (2, 3, 4)) == 3
    assert crossed.reserved_candidate_budget(6, (2, 3, 4)) == 6
    assert crossed.reserved_candidate_budget(3, ()) == 3
    for widths in ((), (2,), (2, 3, 4)):
        reserved = crossed.reserved_candidate_budget(1, widths)
        assert reserved + 1 >= max(widths, default=1)
        assert reserved <= max(3, max(widths, default=1))


def test_width_crosses_reserve_teacher_tokens_beyond_public_output_tail() -> None:
    assert crossed.teacher_output_budget(129, ()) == 129
    assert crossed.teacher_output_budget(129, (2, 3, 4)) == 132
    assert crossed.teacher_output_budget(25, (2,)) == 26


# --- command line surface --------------------------------------------------


def test_parse_args_records_the_focus_window_and_crosses() -> None:
    args = crossed.parse_args(
        ["--out", "report.json", "--focus", "85:92", "--cross-widths", "2,3",
         "--cross-profiles", "strict,production"]
    )

    assert args.focus_window == FocusWindow(low=85, high=92)
    assert (args.budget, args.outputs) == (3, 129)
    assert args.cross_widths == (2, 3)
    assert args.cross_profiles == ("production", "strict")


def test_run_capture_rejects_bad_paths_before_loading_a_model(tmp_path: Path) -> None:
    imported_before = set(sys.modules)
    missing_suite = crossed.parse_args(
        ["--out", str(tmp_path / "report.json"), "--prompts-file", str(tmp_path / "absent.jsonl")]
    )
    with pytest.raises(DiagnosticError, match="prompt suite"):
        crossed.run_capture(missing_suite)

    missing_model = crossed.parse_args(
        ["--out", str(tmp_path / "report.json"), "--model", str(tmp_path / "absent-model")]
    )
    with pytest.raises(DiagnosticError, match="model directory"):
        crossed.run_capture(missing_model)

    # Neither rejection may have imported the serving stack.
    assert "hipengine.llm" not in set(sys.modules) - imported_before


@pytest.mark.parametrize(
    "argv",
    [
        ["--out", "report.json", "--focus", "9:3"],
        ["--out", "report.json", "--focus", "85"],
        ["--out", "report.json", "--budget", "0"],
        ["--out", "report.json", "--outputs", "0"],
        ["--out", "report.json", "--pad", "-1"],
        ["--out", "report.json", "--cross-widths", "5"],
        ["--out", "report.json", "--cross-widths", "3,3"],
        ["--out", "report.json", "--cross-profiles", "eager"],
    ],
)
def test_parse_args_rejects_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        crossed.parse_args(argv)

    assert excinfo.value.code == 2


def _run_module(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.mtp_paro_crossed_state", *argv],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_help_succeeds_without_a_gpu() -> None:
    result = _run_module("--help")

    assert result.returncode == 0, result.stderr
    assert "--focus" in result.stdout
    assert "Diagnostic only" in result.stdout
