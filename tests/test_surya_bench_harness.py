"""Unit tests for the Surya benchmark harness itself.

These are deliberately GPU-free and fast: they test the measurement and
validation machinery, not the model. A benchmark harness that reports a wrong
number is worse than no harness, so the timing/isolation logic gets its own
coverage.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "surya_perf_compare", _ROOT / "scripts" / "surya_perf_compare.py"
)
assert _SPEC is not None and _SPEC.loader is not None
H = importlib.util.module_from_spec(_SPEC)
# dataclasses resolves cls.__module__ through sys.modules, so register first
sys.modules[_SPEC.name] = H
_SPEC.loader.exec_module(H)


# --- workload suite ---------------------------------------------------------


def test_suite_splits_are_disjoint_and_non_empty() -> None:
    """A tuning run must never be reportable on held-out cases."""

    tuning = {c.name for c in H.SUITE if c.split == "tuning"}
    heldout = {c.name for c in H.SUITE if c.split == "heldout"}
    assert tuning and heldout, "both splits must be populated"
    assert not (tuning & heldout), "a case cannot be in both splits"
    assert len({c.name for c in H.SUITE}) == len(H.SUITE), "case names must be unique"
    # both halves must be big enough to be meaningful
    assert len(tuning) >= 4 and len(heldout) >= 4


def test_suite_covers_the_document_types_the_gate_requires() -> None:
    """The suite must span scripts, density, layout, degradation and length.

    A suite that only held single-column English prose would let a tuning
    change pass while regressing Japanese or dense small text.
    """

    names = {c.name for c in H.SUITE}
    for required in ("ja", "mixed", "dense", "table", "blank", "scan", "long"):
        assert required in names, f"suite is missing the {required} case"


def test_suite_covers_both_output_formats() -> None:
    formats = {c.format for c in H.SUITE}
    assert {"markup", "json_nonempty"} <= formats
    assert formats <= {"json", "json_nonempty", "markup", "any"}


def test_blank_case_accepts_its_actual_markup_output() -> None:
    """A blank page emits ``<div><img/></div>``, which is markup, not ``[]``."""

    case = H.CASES_BY_NAME["blank"]
    assert case.format == "markup"
    ref = H._load_oracle(case)
    assert ref is not None
    run = _run(generated=list(ref), e2e_generated=list(ref),
               termination="eos", e2e_termination="eos",
               text="<div><img/></div>")
    assert H._validate(run, case, ref)["correctness"] == "PASS"
    # and a JSON case must reject that same text
    other = H.CASES_BY_NAME["full"]
    other_ref = H._load_oracle(other)
    if other_ref is not None:
        bad = _run(generated=list(other_ref), e2e_generated=list(other_ref),
                   termination="eos", e2e_termination="eos",
                   text="<div><img/></div>")
        assert H._validate(bad, other, other_ref)["correctness"] == "FAIL"


def test_declared_format_matches_what_the_oracle_actually_emits() -> None:
    """The task check must be derived from the oracle, not guessed.

    Surya returns layout JSON for some pages and HTML-ish markup for others, so
    a hardcoded expectation would silently fail a correct lane (or pass a wrong
    one). This is the gate that keeps ``format`` honest.
    """

    checked = 0
    for case in H.SUITE:
        record = H._load_oracle_record(case)
        if record is None or "text" not in record:
            continue
        text = record["text"]
        checked += 1
        if case.format == "markup":
            assert text.lstrip().startswith("<"), (
                f"case {case.name} declares markup but the oracle starts "
                f"{text[:40]!r}"
            )
        elif case.format in ("json", "json_nonempty"):
            parsed = json.loads(text)
            assert isinstance(parsed, list), case.name
            if case.format == "json_nonempty":
                assert parsed, f"case {case.name} declares non-empty JSON"
        elif case.format == "any":
            continue
        else:  # pragma: no cover - guards against a typo in the suite
            raise AssertionError(f"unknown format {case.format!r}")
    assert checked >= 8, "expected most cases to carry an oracle with text"


def test_oracle_capture_metadata_is_recorded_where_available() -> None:
    """Bench oracles record whether they stopped on EOS or hit their budget."""

    record = H._load_oracle_record(H.CASES_BY_NAME["ja"])
    assert record is not None
    assert record["reached_limit"] is False, "ja must reach EOS inside its budget"
    scan = H._load_oracle_record(H.CASES_BY_NAME["scan"])
    assert scan is not None and scan["reached_limit"] is False


def test_every_case_fixture_exists() -> None:
    for case in H.SUITE:
        assert (H.FIXTURES / case.page).exists(), f"missing page for {case.name}"


def test_case_budget_can_reproduce_its_oracle() -> None:
    """``max_tokens`` must be large enough to reach the captured oracle length.

    If it were smaller the case could never match, and if the oracle stopped on
    EOS while the budget is tighter than the capture the comparison would be
    meaningless.
    """

    for case in H.SUITE:
        ref = H._load_oracle(case)
        if ref is None:
            continue
        assert len(ref) <= case.max_tokens, (
            f"case {case.name} budget {case.max_tokens} cannot reach its "
            f"{len(ref)}-token oracle"
        )
        assert len(ref) > 0


def test_small_case_oracle_is_the_documented_length_capped_capture() -> None:
    """``small`` is the only case whose oracle hit its budget rather than EOS.

    Pinning this keeps the termination assertion below honest: the case is
    expected to end on the length limit, so a change that makes it stop early
    is a behaviour change, not a harness detail.
    """

    case = H.CASES_BY_NAME["small"]
    ref = H._load_oracle(case)
    assert ref is not None and len(ref) == case.max_tokens


# --- timing primitives ------------------------------------------------------


def test_series_with_setup_restores_state_before_every_sample() -> None:
    """The isolation guarantee, tested without a GPU.

    ``fn`` mutates shared state and its result depends on that state, which is
    exactly the shape of a decode loop. Without the setup the repeats would
    observe each other; with it every sample must be identical.
    """

    state = {"n": 0}

    def setup() -> None:
        state["n"] = 0

    def fn() -> int:
        state["n"] += 1
        return state["n"]

    samples, outputs = H._series_with_setup(setup, fn, 3, None)
    assert outputs == [1, 1, 1], "repeats observed each other's state"
    assert len(samples) == 3
    assert H._repeatable(outputs)

    # the un-isolated variant is what the bug looked like: outputs drift
    _, leaky = H._series(fn, 3, None)
    assert leaky != [1, 1, 1], "the un-isolated series should drift"
    assert not H._repeatable(leaky)


def test_series_discards_a_warmup_sample() -> None:
    calls: list[int] = []

    def fn() -> int:
        calls.append(len(calls))
        return len(calls)

    samples, outputs = H._series(fn, 2, None)
    assert len(samples) == 2, "only the requested runs are timed"
    assert len(calls) == 3, "one warmup call must be discarded"
    assert len(outputs) == 2


def test_repeatable_handles_tuples_and_single_samples() -> None:
    assert H._repeatable([( [1, 2], "eos")])
    assert H._repeatable([([1, 2], "eos"), ([1, 2], "eos")])
    assert not H._repeatable([([1, 2], "eos"), ([1, 2], "length")])


def test_distribution_reports_the_full_shape() -> None:
    dist = H._dist([1.0, 2.0, 3.0])
    assert dist["n"] == 3
    assert dist["min_s"] == 1.0 and dist["max_s"] == 3.0
    assert dist["median_s"] == 2.0 and dist["mean_s"] == 2.0
    assert dist["stdev_s"] > 0
    assert H._dist([])["n"] == 0
    assert H._dist([5.0])["stdev_s"] == 0.0


# --- validation -------------------------------------------------------------


def _run(**kwargs) -> "H.LaneRun":
    base = {
        "lane": "test",
        "backend": "test",
        "init_s": 0.0,
        "stages": {"decode": 1.0},
    }
    base.update(kwargs)
    return H.LaneRun(**base)


def test_validation_rejects_a_matching_prefix_with_a_divergent_tail() -> None:
    """The gate is full-sequence equality, not a prefix match."""

    case = H.CASES_BY_NAME["small"]
    ref = H._load_oracle(case)
    assert ref is not None

    good = _run(generated=list(ref), e2e_generated=list(ref),
                termination="length", e2e_termination="length",
                text="<div>x</div>")
    assert H._validate(good, case, ref)["correctness"] == "PASS"

    truncated = _run(
        generated=list(ref[:10]), e2e_generated=list(ref[:10]),
        termination="length", e2e_termination="length", text="<div>x</div>",
    )
    checks = H._validate(truncated, case, ref)
    assert checks["ids_match"] is False
    assert checks["first_divergence"] == 10
    assert checks["correctness"] == "FAIL"


def test_validation_flags_a_decode_stage_that_disagrees_with_e2e() -> None:
    """tok/s comes from the decode stage, so e2e must produce the same ids."""

    case = H.CASES_BY_NAME["small"]
    ref = H._load_oracle(case)
    assert ref is not None
    run = _run(
        generated=list(ref), e2e_generated=list(ref[:-1]),
        termination="length", e2e_termination="length", text="<div>x</div>",
    )
    checks = H._validate(run, case, ref)
    assert checks["e2e_agrees"] is False
    assert checks["correctness"] == "FAIL"
    assert "e2e_agrees" in checks["gate_failures"]


def test_validation_records_termination_and_format_basis() -> None:
    case = H.CASES_BY_NAME["list"]
    ref = H._load_oracle(case)
    assert ref is not None
    run = _run(
        generated=list(ref),
        e2e_generated=list(ref),
        termination="eos",
        e2e_termination="eos",
        text='[{"label": "Text", "bbox": "1 2 3 4", "count": 3}]',
    )
    checks = H._validate(run, case, ref, reference_source="torch_fixture")
    assert checks["termination"] == "eos"
    assert checks["correctness_basis"] == "torch_fixture"
    assert checks["layout_json"] is True
    assert checks["task_format"] is True
    assert checks["correctness"] == "PASS"


def test_basis_is_not_guessed_when_the_source_is_unnamed() -> None:
    """A record must not claim a basis the caller did not name.

    Defaulting to ``torch_fixture`` would let a CPU reference be published as a
    captured fixture, which is the same mislabelling that let the rectangular
    rows claim ``cpu_reference`` while comparing against nothing.
    """

    case = H.CASES_BY_NAME["list"]
    ref = H._load_oracle(case)
    run = _run(generated=list(ref), e2e_generated=list(ref),
               termination="eos", e2e_termination="eos",
               text='[{"label": "Text", "bbox": "1 2 3 4", "count": 3}]')
    assert H._validate(run, case, ref)["correctness_basis"] == "unspecified"


# --- fail-closed gates ------------------------------------------------------
#
# Each test below pins one condition that a PASS must require. Before these
# gates existed a lane could PASS with no reference at all (``ids_match``
# null), with a decode stage that did not repeat, and with a termination that
# disagreed with the reference or with the end-to-end run.


def test_validation_fails_closed_when_no_reference_is_available() -> None:
    """A missing reference must not read as a pass.

    ``rect`` declares no captured fixture, so its basis is the CPU reference
    the main loop computes in-process. Passing ``None`` here is the harness
    defect this guards: the record used to claim ``cpu_reference`` as its basis
    while comparing against nothing, which is how the committed rectangular
    rows came to carry ``ids_match: null`` and ``correctness: PASS``.
    """

    case = H.CASES_BY_NAME["rect"]
    assert H._load_oracle(case) is None
    run = _run(generated=[1, 2, 3], e2e_generated=[1, 2, 3], text="anything",
               termination="eos", e2e_termination="eos")
    checks = H._validate(run, case, None)
    assert checks["reference_present"] is False
    assert checks["ids_match"] is None
    assert checks["correctness"] == "FAIL"
    assert "reference_present" in checks["gate_failures"]
    assert checks["correctness_basis"] == "none"


def test_pass_requires_decode_repeatable() -> None:
    """An isolated decode is the premise of the decode timing.

    If repeat 2 started where repeat 1 stopped, the decode rate describes a
    different computation than the one reported, so it cannot be a pass.
    """

    case = H.CASES_BY_NAME["small"]
    ref = H._load_oracle(case)
    run = _run(generated=list(ref), e2e_generated=list(ref),
               termination="length", e2e_termination="length",
               text="<div>x</div>", decode_repeatable=False)
    checks = H._validate(run, case, ref)
    assert checks["decode_repeatable"] is False
    assert checks["correctness"] == "FAIL"
    assert "decode_repeatable" in checks["gate_failures"]


def test_pass_requires_termination_to_agree_with_the_reference() -> None:
    """A budget-capped capture and an EOS capture must not be interchangeable.

    ``small`` fills its whole 64-token budget, so a lane reporting EOS produced
    a different sequence than the reference even if every shared id matches.
    """

    case = H.CASES_BY_NAME["small"]
    ref = H._load_oracle(case)
    assert len(ref) == case.max_tokens
    run = _run(generated=list(ref), e2e_generated=list(ref), termination="eos",
               e2e_termination="eos", text="<div>x</div>")
    checks = H._validate(run, case, ref)
    assert checks["reference_termination"] == "length"
    assert checks["termination_match"] is False
    assert checks["correctness"] == "FAIL"
    assert "termination_match" in checks["gate_failures"]

    # and the converse: a case that stops on EOS must not claim the budget
    other = H.CASES_BY_NAME["ja"]
    other_ref = H._load_oracle(other)
    assert len(other_ref) < other.max_tokens
    bad = _run(generated=list(other_ref), e2e_generated=list(other_ref),
               termination="length", e2e_termination="length",
               text=H._load_oracle_record(other)["text"])
    checks = H._validate(bad, other, other_ref)
    assert checks["reference_termination"] == "eos"
    assert checks["termination_match"] is False
    assert checks["correctness"] == "FAIL"


def test_pass_requires_e2e_termination_to_agree_with_the_decode_stage() -> None:
    """The timed decode loop and the end-to-end run must stop the same way."""

    case = H.CASES_BY_NAME["small"]
    ref = H._load_oracle(case)
    run = _run(generated=list(ref), e2e_generated=list(ref),
               termination="length", e2e_termination="eos",
               text="<div>x</div>")
    checks = H._validate(run, case, ref)
    assert checks["e2e_termination_agrees"] is False
    assert checks["correctness"] == "FAIL"
    assert "e2e_termination_agrees" in checks["gate_failures"]


def test_gate_failures_is_empty_only_for_a_clean_pass() -> None:
    """A FAIL must name the condition that failed, not just say FAIL."""

    case = H.CASES_BY_NAME["blank"]
    ref = H._load_oracle(case)
    run = _run(generated=list(ref), e2e_generated=list(ref), termination="eos",
               e2e_termination="eos", text="<div><img/></div>")
    checks = H._validate(run, case, ref)
    assert checks["gate_failures"] == []
    assert checks["correctness"] == "PASS"

    broken = _run(generated=list(ref), e2e_generated=list(ref),
                  termination="eos", e2e_termination="length",
                  text="<div><img/></div>", decode_repeatable=False)
    failures = H._validate(broken, case, ref)["gate_failures"]
    assert set(failures) == {"decode_repeatable", "e2e_termination_agrees"}


def test_reference_termination_comes_from_the_budget() -> None:
    """A capture that filled its budget ended on length, not on EOS."""

    assert H._reference_termination([1, 2, 3], 64) == "eos"
    assert H._reference_termination(list(range(64)), 64) == "length"
    # a reference longer than the budget cannot have stopped on EOS either
    assert H._reference_termination(list(range(65)), 64) == "length"


def test_a_declared_but_missing_oracle_file_is_an_error() -> None:
    """A declared fixture that is absent must not degrade to "no reference".

    Returning ``None`` for a missing file made a case silently fall back to an
    unverified basis; only ``oracle=None`` may mean "compute the CPU
    reference".
    """

    case = H.Case("ghost", "page_small.png", H.PROMPT, 8, "tuning", "any",
                  "oracle_does_not_exist.json")
    with pytest.raises(FileNotFoundError):
        H._load_oracle_record(case)
    with pytest.raises(FileNotFoundError):
        H._load_oracle(case)


def test_only_rect_declares_no_fixture() -> None:
    """Pin the blast radius of the CPU-reference fallback."""

    without = [c.name for c in H.SUITE if c.oracle is None]
    assert without == ["rect"], without


def test_validation_rejects_non_json_output_for_a_json_case() -> None:
    case = H.CASES_BY_NAME["full"]
    ref = H._load_oracle(case)
    assert ref is not None
    run = _run(generated=list(ref), e2e_generated=list(ref), text="not json at all")
    checks = H._validate(run, case, ref)
    assert checks["ids_match"] is True, "ids still match; only the task check fails"
    assert checks["correctness"] == "FAIL"


def test_summary_line_survives_a_lane_with_no_samples() -> None:
    """One failed lane must not abort a long multi-case run."""

    line = H._summary_line({
        "e2e_median_s": None,
        "stages_s": {},
        "decode_tok_per_s": None,
        "correctness": "FAIL",
        "termination": "unknown",
    })
    assert "n/a" in line
    assert "FAIL" in line


# --- structure parity -------------------------------------------------------

# The shape the degraded scan actually produces under this suite's ad-hoc
# prompt: a rigid template both lanes free-run into. Only the bbox digits move.
_TEMPLATE_REF = json.dumps([
    {"label": "Section-Header", "bbox": "44 26 361 63", "count": 30},
    {"label": "Text", "bbox": "44 83 566 114", "count": 70},
    {"label": "Text", "bbox": "44 122 526 153", "count": 70},
])


def _template_text(body_x1: int, header_x1: int = 362, y_step: int = 40) -> str:
    """A lane's realization of the template: body x1 flat, header x1 correct."""

    return json.dumps([
        {"label": "Section-Header", "bbox": f"44 26 {header_x1} 62", "count": 30},
        {"label": "Text", "bbox": f"44 83 {body_x1} 114", "count": 70},
        {"label": "Text", "bbox": f"44 {83 + y_step} {body_x1} {114 + y_step}",
         "count": 70},
    ])


def _structure_run(text: str, ref_len: int) -> "H.LaneRun":
    ids = [1] * ref_len
    return _run(generated=list(ids), e2e_generated=list(ids),
                termination="eos", e2e_termination="eos", text=text)


def test_scan_declares_structure_parity_and_everything_else_exact() -> None:
    """One case relaxes the id gate; the rest must keep it."""

    relaxed = {c.name for c in H.SUITE if c.parity != "exact"}
    assert relaxed == {"scan"}, relaxed
    for case in H.SUITE:
        assert case.parity in {"exact", "structure"}
        if case.parity == "structure":
            # a structure case without a captured fixture cannot check its
            # skeleton, so it would fail closed forever
            assert case.oracle is not None


def test_structure_parity_passes_on_a_different_bbox_template() -> None:
    """The point of the mode: coordinate digits are not gated."""

    case = H.CASES_BY_NAME["scan"]
    ref = H._load_oracle(case)
    assert ref is not None
    # a flat x1 where the reference varies, and a 38-of-1000 drift
    checks = H._validate(_structure_run(_template_text(564), len(ref)),
                         case, ref, reference_source="torch_fixture",
                         ref_text=_TEMPLATE_REF)
    assert checks["ids_match"] is False, "the digits really did move"
    assert checks["structure_match"] is True
    assert checks["bbox_max_delta"] == 38
    assert checks["correctness"] == "PASS", checks["gate_failures"]
    assert "ids_match" not in checks["gate_failures"]


def test_structure_parity_rejects_an_omitted_box() -> None:
    """A dropped region must still fail, even though ids are not gated."""

    case = H.CASES_BY_NAME["scan"]
    ref = H._load_oracle(case)
    short = json.dumps(json.loads(_template_text(564))[:2])
    checks = H._validate(_structure_run(short, len(ref)), case, ref,
                         ref_text=_TEMPLATE_REF)
    assert checks["structure_match"] is False
    assert checks["correctness"] == "FAIL"
    assert "structure_match" in checks["gate_failures"]


def test_structure_parity_rejects_a_relabelled_box() -> None:
    """Same count, same coordinates, wrong label is still a failure."""

    case = H.CASES_BY_NAME["scan"]
    ref = H._load_oracle(case)
    boxes = json.loads(_template_text(564))
    boxes[1]["label"] = "Table"
    checks = H._validate(_structure_run(json.dumps(boxes), len(ref)), case, ref,
                         ref_text=_TEMPLATE_REF)
    assert checks["structure_match"] is False
    assert checks["correctness"] == "FAIL"


def test_structure_parity_fails_closed_without_reference_text() -> None:
    """No skeleton to compare against is a failure, not a free pass."""

    case = H.CASES_BY_NAME["scan"]
    ref = H._load_oracle(case)
    checks = H._validate(_structure_run(_template_text(564), len(ref)),
                         case, ref, ref_text=None)
    assert checks["structure_match"] is False
    assert checks["correctness"] == "FAIL"


def test_structure_parity_rejects_non_layout_output() -> None:
    """Markup cannot compare equal to an empty skeleton."""

    case = H.CASES_BY_NAME["scan"]
    ref = H._load_oracle(case)
    checks = H._validate(_structure_run("<div><p>x</p></div>", len(ref)),
                         case, ref, ref_text=_TEMPLATE_REF)
    assert checks["structure_match"] is False
    assert checks["generated_boxes"] is None


def test_bbox_bound_is_a_gate_only_when_the_case_declares_one() -> None:
    """``max_bbox_delta`` turns the diagnostic into a bound on demand."""

    from dataclasses import replace

    case = replace(H.CASES_BY_NAME["scan"], max_bbox_delta=8.0)
    ref = H._load_oracle(case)
    checks = H._validate(_structure_run(_template_text(564), len(ref)), case, ref,
                         ref_text=_TEMPLATE_REF)
    assert checks["bbox_within_bound"] is False
    assert "bbox_within_bound" in checks["gate_failures"]
    # and the bound passes when the drift is inside it
    tight = H._validate(_structure_run(_TEMPLATE_REF, len(ref)), case, ref,
                        ref_text=_TEMPLATE_REF)
    assert tight["bbox_max_delta"] == 0
    assert tight["correctness"] == "PASS"


def test_structure_case_still_gates_termination_and_format() -> None:
    """Relaxing the id gate must not relax the other gates."""

    case = H.CASES_BY_NAME["scan"]
    ref = H._load_oracle(case)
    ids = [1] * len(ref)
    run = _run(generated=list(ids), e2e_generated=list(ids),
               termination="length", e2e_termination="length",
               text=_template_text(564))
    checks = H._validate(run, case, ref, ref_text=_TEMPLATE_REF)
    assert checks["structure_match"] is True
    assert "termination_match" in checks["gate_failures"]
    assert checks["correctness"] == "FAIL"


def test_structure_parity_requires_a_reference_to_be_present() -> None:
    case = H.CASES_BY_NAME["scan"]
    checks = H._validate(_structure_run(_template_text(564), 3), case, None,
                         ref_text=_TEMPLATE_REF)
    assert "reference_present" in checks["gate_failures"]
    assert checks["correctness"] == "FAIL"


def test_artifact_records_the_parity_mode_per_case(tmp_path, monkeypatch) -> None:
    """A reader must be able to tell which gate a retained row used."""

    out = tmp_path / "artifact.json"
    ref = H._load_oracle(H.CASES_BY_NAME["blank"])
    monkeypatch.setattr(H, "LANES", {"fake": lambda case, runs: _run(
        lane="fake", generated=list(ref), e2e_generated=list(ref),
        termination="eos", e2e_termination="eos", text="<div><img/></div>",
        stages={"decode": 0.5}, e2e_samples=[1.0])})
    monkeypatch.setattr(sys, "argv", [
        "surya_perf_compare.py", "--cases", "blank", "--lanes", "fake",
        "--out", str(out)])
    H.main()
    suite = json.loads(out.read_text())["suite"]
    assert suite[0]["parity"] == "exact"
    assert H.CASES_BY_NAME["scan"].parity == "structure"


def test_case_selection_honours_the_frozen_split() -> None:
    class Args:
        cases = None
        split = "tuning"

    tuning = H._select_cases(Args())
    assert {c.name for c in tuning} == {
        c.name for c in H.SUITE if c.split == "tuning"
    }
    Args.split = "all"
    assert len(H._select_cases(Args())) == len(H.SUITE)


def test_lane_record_derives_tok_per_s_from_the_timed_stage() -> None:
    case = H.CASES_BY_NAME["rect"]
    run = _run(
        generated=[1, 2, 3, 4],
        e2e_generated=[1, 2, 3, 4],
        stages={"decode": 0.5},
        e2e_samples=[1.0, 1.2],
    )
    record = H._lane_record(run, case, None)
    assert record["decode_tokens"] == 4
    assert record["decode_tok_per_s"] == pytest.approx(8.0)
    assert record["e2e_median_s"] == pytest.approx(1.1)
    assert record["e2e_distribution_s"]["n"] == 2


def test_artifact_records_the_protocol_and_suite(tmp_path, monkeypatch) -> None:
    """The artifact must carry provenance and the frozen suite definition."""

    out = tmp_path / "artifact.json"
    monkeypatch.setattr(H, "_cpu_reference", lambda case: [1, 2, 3])
    monkeypatch.setattr(
        H,
        "LANES",
        {"fake": lambda case, runs: _run(
            lane="fake",
            generated=[1, 2, 3],
            e2e_generated=[1, 2, 3],
            termination="eos",
            e2e_termination="eos",
            text="x",
            stages={"decode": 0.5, "vision_prefill": 0.25},
            e2e_samples=[1.0, 1.1],
        )},
    )
    # drive main() with explicit argv
    monkeypatch.setattr(
        sys, "argv",
        ["surya_perf_compare.py", "--cases", "rect",
         "--lanes", "fake", "--out", str(out)],
    )
    H.main()
    artifact = json.loads(out.read_text())
    assert artifact["protocol"]["sync"].startswith("sync()")
    assert artifact["protocol"]["decode_derivation"].startswith("decode timed")
    # the gate the PASS is measured against must be stated, not implied
    assert "ids_match" in artifact["protocol"]["gate"]
    assert "decode_repeatable" in artifact["protocol"]["gate"]
    assert "termination" in artifact["protocol"]["gate"]
    assert artifact["provenance"]["command"].endswith(str(out))
    assert [c["name"] for c in artifact["suite"]] == ["rect"]
    assert artifact["results"][0]["lane"] == "fake"


def test_main_computes_the_cpu_reference_when_a_case_has_no_fixture(
        tmp_path, monkeypatch) -> None:
    """``rect`` declares no fixture, so the main loop must supply a basis.

    Before this, ``main`` called ``_load_oracle`` and passed whatever came
    back: ``None`` for a fixture-less case. The record then reported
    ``correctness_basis: cpu_reference`` without ever computing one and
    ``ids_match: null`` while still reading PASS.
    """

    out = tmp_path / "artifact.json"
    calls: list[str] = []

    def fake_reference(case):
        calls.append(case.name)
        return [7, 8, 9]

    monkeypatch.setattr(H, "_cpu_reference", fake_reference)
    monkeypatch.setattr(H, "LANES", {"fake": lambda case, runs: _run(
        lane="fake", generated=[7, 8, 9], e2e_generated=[7, 8, 9],
        termination="eos", e2e_termination="eos", text="x",
        stages={"decode": 0.5}, e2e_samples=[1.0])})
    monkeypatch.setattr(sys, "argv", [
        "surya_perf_compare.py", "--cases", "rect", "--lanes", "fake",
        "--out", str(out)])
    H.main()
    record = json.loads(out.read_text())["results"][0]
    assert calls == ["rect"], "the reference must be computed for the case"
    assert record["reference_present"] is True
    assert record["correctness_basis"] == "cpu_reference"
    assert record["ids_match"] is True
    assert record["gate_failures"] == []
    assert record["correctness"] == "PASS"


def test_main_does_not_compute_a_cpu_reference_when_a_fixture_exists(
        tmp_path, monkeypatch) -> None:
    """The captured fixture is the basis; the fallback must not shadow it."""

    out = tmp_path / "artifact.json"

    def boom(case):
        raise AssertionError(f"computed a CPU reference for {case.name}")

    ref = H._load_oracle(H.CASES_BY_NAME["blank"])
    assert ref is not None
    monkeypatch.setattr(H, "_cpu_reference", boom)
    monkeypatch.setattr(H, "LANES", {"fake": lambda case, runs: _run(
        lane="fake", generated=list(ref), e2e_generated=list(ref),
        termination="eos", e2e_termination="eos", text="<div><img/></div>",
        stages={"decode": 0.5}, e2e_samples=[1.0])})
    monkeypatch.setattr(sys, "argv", [
        "surya_perf_compare.py", "--cases", "blank", "--lanes", "fake",
        "--out", str(out)])
    H.main()
    record = json.loads(out.read_text())["results"][0]
    assert record["correctness_basis"] == "torch_fixture"
    assert record["ids_match"] is True
    assert record["correctness"] == "PASS"


def test_main_reports_a_mismatching_lane_as_failed(tmp_path, monkeypatch) -> None:
    """The end-to-end path must not launder a mismatch into a PASS."""

    out = tmp_path / "artifact.json"
    ref = H._load_oracle(H.CASES_BY_NAME["blank"])
    monkeypatch.setattr(H, "LANES", {"fake": lambda case, runs: _run(
        lane="fake", generated=list(ref)[:-1], e2e_generated=list(ref)[:-1],
        termination="eos", e2e_termination="eos", text="<div><img/></div>",
        stages={"decode": 0.5}, e2e_samples=[1.0])})
    monkeypatch.setattr(sys, "argv", [
        "surya_perf_compare.py", "--cases", "blank", "--lanes", "fake",
        "--out", str(out)])
    H.main()
    record = json.loads(out.read_text())["results"][0]
    assert record["correctness"] == "FAIL"
    assert "ids_match" in record["gate_failures"]


# --- lane-isolated merge ----------------------------------------------------


def _artifact(path, rows, *, protocol=None, suite=None) -> Path:
    """A minimal artifact; the suite defaults to the cases the rows name."""

    if suite is None:
        suite = [{"name": name, "page": f"page_{name}.png"}
                 for name in dict.fromkeys(row["case"] for row in rows)]
    path.write_text(json.dumps({
        "provenance": {"command": f"run for {path.name}"},
        "protocol": protocol if protocol is not None else {"runs": 3},
        "suite": suite,
        "results": rows,
    }))
    return path


def test_merge_interleaves_lanes_in_suite_order(tmp_path) -> None:
    """Each lane is measured alone; the merged table reads case-major."""

    one = _artifact(tmp_path / "one.json", [
        {"case": "a", "lane": "hip", "correctness": "PASS"},
        {"case": "b", "lane": "hip", "correctness": "PASS"},
    ])
    two = _artifact(tmp_path / "two.json", [
        {"case": "a", "lane": "torch", "correctness": "PASS"},
        {"case": "b", "lane": "torch", "correctness": "PASS"},
    ])
    merged = H._merge_artifacts([one, two])
    assert [(r["case"], r["lane"]) for r in merged["results"]] == [
        ("a", "hip"), ("a", "torch"), ("b", "hip"), ("b", "torch")]
    assert merged["provenance"]["merged"] is True
    assert len(merged["provenance"]["sources"]) == 2
    assert merged["protocol"]["lanes"] == ["hip", "torch"]
    assert "one process per lane" in merged["protocol"]["lane_isolation"]


def test_merge_rejects_a_mismatched_suite(tmp_path) -> None:
    """Mixing two different suites would silently produce a wrong table."""

    one = _artifact(tmp_path / "one.json", [{"case": "a", "lane": "hip"}])
    other = _artifact(tmp_path / "two.json", [{"case": "a", "lane": "torch"}],
                      suite=[{"name": "a", "page": "page_other.png"}])
    with pytest.raises(ValueError, match="different suite"):
        H._merge_artifacts([one, other])


def test_merge_rejects_a_mismatched_protocol(tmp_path) -> None:
    one = _artifact(tmp_path / "one.json", [{"case": "a", "lane": "hip"}])
    other = _artifact(tmp_path / "two.json", [{"case": "a", "lane": "torch"}],
                      protocol={"runs": 5})
    with pytest.raises(ValueError, match="different protocol"):
        H._merge_artifacts([one, other])


def test_merge_rejects_a_duplicate_row(tmp_path) -> None:
    """The same lane measured twice is a mistake, not something to average."""

    one = _artifact(tmp_path / "one.json", [{"case": "a", "lane": "hip"}])
    other = _artifact(tmp_path / "two.json", [{"case": "a", "lane": "hip"}])
    with pytest.raises(ValueError, match="both measured"):
        H._merge_artifacts([one, other])


def test_merge_rejects_a_missing_row(tmp_path) -> None:
    """A lane that skipped a case must not vanish from the table."""

    suite = [{"name": "a", "page": "page_a.png"},
             {"name": "b", "page": "page_b.png"}]
    one = _artifact(tmp_path / "one.json", [
        {"case": "a", "lane": "hip"}, {"case": "b", "lane": "hip"}], suite=suite)
    two = _artifact(tmp_path / "two.json", [{"case": "a", "lane": "torch"}],
                    suite=suite)
    with pytest.raises(ValueError, match="missing rows"):
        H._merge_artifacts([one, two])


def test_merge_through_main_writes_the_artifact(tmp_path, monkeypatch) -> None:
    one = _artifact(tmp_path / "one.json", [{"case": "a", "lane": "hip"}])
    two = _artifact(tmp_path / "two.json", [{"case": "a", "lane": "torch"}])
    out = tmp_path / "merged.json"
    monkeypatch.setattr(sys, "argv", [
        "surya_perf_compare.py", "--merge", str(one), str(two), "--out", str(out)])
    H.main()
    assert [r["lane"] for r in json.loads(out.read_text())["results"]] == \
        ["hip", "torch"]


def test_merge_without_out_is_an_error(tmp_path, monkeypatch) -> None:
    one = _artifact(tmp_path / "one.json", [{"case": "a", "lane": "hip"}])
    monkeypatch.setattr(sys, "argv", [
        "surya_perf_compare.py", "--merge", str(one)])
    with pytest.raises(SystemExit, match="needs --out"):
        H.main()


# --- attention-memory shape sweep -------------------------------------------


_M_SPEC = importlib.util.spec_from_file_location(
    "surya_attention_memory", _ROOT / "scripts" / "surya_attention_memory.py"
)
assert _M_SPEC is not None and _M_SPEC.loader is not None
M = importlib.util.module_from_spec(_M_SPEC)
sys.modules[_M_SPEC.name] = M
_M_SPEC.loader.exec_module(M)


def test_text_shape_sweep_lifts_and_restores_the_planner_envelope() -> None:
    """A sweep row must measure the width it names, and must not leak.

    ``--text-blocks`` exists to measure shapes the production planner would not
    choose, so it lifts the shape envelope and the wavefront rounding. The lift
    has to reach the planner in both regimes: where the envelope caps the
    budget's width (small grids, one or two tiles) and where it only rounds it
    (larger grids, where the budget's own shape wins). If that lift silently
    failed, every sweep row would be the production shape wearing a different
    label, which is exactly the failure mode the sweep rules out. The lift is
    process-global planner state, so it must also be restored, or the next
    measurement in the process inherits a shape rule it did not ask for.
    """

    from hipengine.runtime import surya

    shipped = M._production_envelope()
    assert M._ENVELOPE == shipped, "the shipped constants are captured at import"
    assert shipped == (
        int(surya.SHAPE_TILE_ROWS),
        int(surya.SHAPE_TILE_DIVISOR),
        int(surya.SHAPE_TILE_MULTIPLE),
    )
    heads, tokens = 8, 8580
    try:
        # production: 8580 tokens leaves 5 tiles, so the budget's own 1955 rows
        # survive, rounded down to the 32-lane wavefront multiple
        assert surya.plan_score_tiles(tokens, heads, 512 * 1024**2)[0] == 1952
        # production, capped regime: 1024 tokens admits the whole dense matrix
        # in one tile, so the envelope splits it into 8
        assert surya.plan_score_tiles(1024, heads, 512 * 1024**2)[0] == 128
        # lifted: the same backwards-solved budget returns the named width,
        # including a non-multiple of 32 the production planner would round
        M._shape_envelope(False)
        assert surya.SHAPE_TILE_MULTIPLE == 1
        assert surya.plan_score_tiles(
            tokens, heads, heads * tokens * 1955 * 4
        )[0] == 1955
        # lifted, capped regime: the dense width the envelope splits is reachable
        assert surya.plan_score_tiles(1024, heads, 512 * 1024**2)[0] == 1024
        # the width the budget admits at the full context
        assert surya.plan_score_tiles(
            16384, heads, heads * 16384 * 1024 * 4
        )[0] == 1024
    finally:
        M._shape_envelope(True)
    assert M._production_envelope() == shipped
    assert surya.plan_score_tiles(tokens, heads, 512 * 1024**2)[0] == 1952
