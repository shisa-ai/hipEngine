"""Category-heldout coverage must certify engagement, and only engagement.

X-7 is the ledger's declared prerequisite ("Close before extending the admitted
scope") and lists category-heldout numerical evidence as its first missing
certificate. Before this, the gate hardcoded one canonical fixture hash, so a
heldout run could not be expressed at all and the blocker was unfalsifiable from
the artifact.

The heldout suite is **short**: its 18 prompts run 39-71 tokens. Whether that
engages a route depends on the route, and the floors differ by more than an order
of magnitude:

    ``_wmma_prefill_dispatch``       ``if not use_wmma or rows <= 1``   -> 2
    ``_q8_dense_wide_dispatch``      ``or rows <= 256``                 -> 257
    ``_raw_k_staged_f32_dispatch``   no row guard at all                -> 1

So the floor is a per-route property and never a shared constant. **Both**
directions of error are silent: too low certifies a route that never ran (the
false pass recorded in the sibling packet), too high refuses a certificate that
is actually valid. The first version of this check made the second mistake - it
took a since-removed route's 256-row floor, applied it to the f16 WMMA route
whose floor is 2, and refused a heldout run that had genuinely engaged and
passed every numerical gate.

These tests pin both directions, and pin that the check discriminates between
routes rather than returning one answer for all of them.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "execution_profile_q8_wmma_prefill_layers_gate.py"
ADMISSION = REPO / "benchmarks" / "fixtures" / "qwen4exp_admission_suite_18prompt.json"

HELDOUT_CASES = (
    ("code_merge_intervals", "code"),
    ("code_topological_sort", "code"),
    ("code_lru_cache", "code"),
    ("code_markdown_table", "code"),
    ("general_en_plan", "general_en"),
    ("general_en_explain", "general_en"),
    ("general_ja_plan", "general_ja"),
    ("general_ja_explain", "general_ja"),
    ("mixed_ja_en_translate", "mixed_ja_en"),
    ("mixed_ja_en_review", "mixed_ja_en"),
    ("heldout_code_rate_limiter", "code"),
    ("heldout_code_interval_schedule", "code"),
    ("heldout_general_en_incident", "general_en"),
    ("heldout_general_en_quant", "general_en"),
    ("heldout_general_ja_migration", "general_ja"),
    ("heldout_general_ja_speculative", "general_ja"),
    ("heldout_mixed_summary", "mixed_ja_en"),
    ("heldout_mixed_review", "mixed_ja_en"),
)

# The admission suite declares 24 transitions, not the canonical 128.
HELDOUT_DECODE_STEPS = 24


def _gate():
    if not GATE.exists():
        pytest.skip(f"gate script not present: {GATE}")
    spec = importlib.util.spec_from_file_location("_wmma_gate_heldout", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Capture:
    def __init__(self, prompt_id: str, category: str):
        self.prompt_id = prompt_id
        self.category = category
        self.strict = tuple({} for _ in range(HELDOUT_DECODE_STEPS + 1))
        self.candidate_runs = {
            "candidate": tuple(
                tuple({} for _ in range(HELDOUT_DECODE_STEPS + 1))
                for _ in range(3)
            )
        }


def _captures():
    return [_Capture(i, c) for i, c in HELDOUT_CASES]


def _rows():
    return [int(c["prompt_tokens"]) for c in json.loads(ADMISSION.read_text())["cases"]]


def _coverage(gate, rows, floor):
    return gate.qualification_coverage(
        captures=_captures(),
        fixture_sha256=gate.ADMISSION_FIXTURE_SHA256,
        prefill_chunk_size=1024,
        fixture_rows=list(rows),
        heldout_decode_steps=HELDOUT_DECODE_STEPS,
        min_engaging_rows=floor,
    )


def _verdict(gate, rows, floor, *, quality_passed=True):
    return gate.qualification_verdict(
        _coverage(gate, rows, floor),
        quality_passed=quality_passed,
        deterministic=True,
        clean=True,
    )


def test_route_row_floors_are_per_route_and_match_the_dispatches():
    """A shared constant is the bug; these values are read from the guards."""

    gate = _gate()
    assert gate.ROUTE_ROW_FLOORS["wmma"] == 2, (
        "_wmma_prefill_dispatch declines at rows <= 1, so it engages at 2"
    )
    assert gate.ROUTE_ROW_FLOORS["dense_wide"] == 257, (
        "_q8_dense_wide_dispatch declines at rows <= 256, so it engages at 257"
    )
    assert gate.ROUTE_ROW_FLOORS["staged_f32"] == 1, (
        "_raw_k_staged_f32_dispatch has no row guard, so it engages at 1"
    )
    # The floors must actually be wired onto the routes, not just tabulated.
    for name, route in gate.SELECTORS.items():
        assert route.min_engaging_rows == gate.ROUTE_ROW_FLOORS[name], (
            f"{name} does not carry its own floor; a route gated with another "
            "route's floor is silently wrong in one direction or the other"
        )


def test_fixture_kind_separates_the_two_protocols():
    gate = _gate()
    assert gate._fixture_kind(gate.CANONICAL_FIXTURE_SHA256) == "canonical"
    assert gate._fixture_kind(gate.ADMISSION_FIXTURE_SHA256) == "category_heldout"
    assert gate._fixture_kind("0" * 64) == "unknown"


def test_heldout_fixture_hash_matches_the_file_on_disk():
    gate = _gate()
    if not ADMISSION.exists():
        pytest.skip(f"heldout fixture not present: {ADMISSION}")
    observed = hashlib.sha256(ADMISSION.read_bytes()).hexdigest()
    assert observed == gate.ADMISSION_FIXTURE_SHA256, (
        "the heldout fixture changed on disk; the gate would classify it as "
        "'unknown' and silently stop producing heldout coverage"
    )


def test_heldout_run_does_not_claim_canonical_coverage():
    """The two protocols are not interchangeable.

    A heldout run readable as canonical coverage would let one certificate stand
    in for the other, which is the accounting defect X-6 records rather than a
    missing measurement.
    """

    gate = _gate()
    coverage = _coverage(gate, _rows(), gate.ROUTE_ROW_FLOORS["wmma"])
    assert coverage["fixture_kind"] == "category_heldout"
    assert coverage["complete"] is False, (
        "a heldout run must never set the canonical protocol's complete flag"
    )
    assert coverage["checks"]["canonical_fixture_identity"] is False


def test_heldout_uses_the_fixture_declared_transition_count():
    """The heldout protocol is not the canonical 128-step one.

    Assuming 128 made the trajectory checks fail, which would have left the
    blocker open for the wrong reason - blaming trajectories instead of the row
    shortfall.
    """

    gate = _gate()
    heldout = _coverage(gate, _rows(), gate.ROUTE_ROW_FLOORS["wmma"])["category_heldout"]
    assert heldout["decode_steps"] == HELDOUT_DECODE_STEPS
    assert heldout["observed_rows_per_trajectory"] == HELDOUT_DECODE_STEPS + 1
    assert heldout["checks"]["heldout_teacher_rows"] is True
    assert heldout["checks"]["heldout_candidate_three_repeats"] is True


def test_the_real_heldout_suite_qualifies_the_wmma_route():
    """The finding this file exists for.

    The suite's 39-71 tokens clear the WMMA route's 2-row floor, so a complete
    heldout run against it is a genuine certificate and the blocker must drop.
    The first version of this check refused it by applying the wrong route's
    floor.
    """

    gate = _gate()
    rows = _rows()
    floor = gate.ROUTE_ROW_FLOORS["wmma"]
    assert max(rows) >= floor, "premise stale: the suite no longer clears the floor"

    heldout = _coverage(gate, rows, floor)["category_heldout"]
    assert heldout["checks"]["heldout_prompts_exceed_route_row_floor"] is True
    assert heldout["complete"] is True
    assert heldout["blocked_reason"] is None

    verdict = _verdict(gate, rows, floor)
    assert verdict["category_heldout_qualification_claim"] is True
    assert "category-heldout numerical evidence" not in verdict["qualification_blockers"], (
        "producing the evidence must drop the blocker, or the accounting can "
        "never be closed by measurement"
    )


def test_the_same_suite_does_not_qualify_the_dense_wide_route():
    """The check must discriminate, or it is a rubber stamp.

    Same fixture, same captures, different route: 71 rows clears 2 but not 257.
    A check that returned the same answer for every route would be worthless in
    both directions at once.
    """

    gate = _gate()
    rows = _rows()
    floor = gate.ROUTE_ROW_FLOORS["dense_wide"]
    assert max(rows) < floor, "premise stale: the suite now clears the wide floor"

    heldout = _coverage(gate, rows, floor)["category_heldout"]
    assert heldout["checks"]["heldout_prompts_exceed_route_row_floor"] is False
    assert heldout["complete"] is False
    assert "below the" in heldout["blocked_reason"]

    verdict = _verdict(gate, rows, floor)
    assert verdict["category_heldout_qualification_claim"] is False
    assert "category-heldout numerical evidence" in verdict["qualification_blockers"]


def test_qualification_claim_requires_the_evidence_to_pass():
    """Collected is not passed, and the field name promises the second.

    The first version asserted ``category_heldout_qualification_claim`` on
    complete-but-failing coverage. The real heldout run is exactly that case: 18
    cases, four categories, real engagement - and ``hard_gates_passed: False``
    with a per-scope KL failure at the prefill->c1 transition. A green claim on
    that run is the same shape of defect as the two false passes this packet
    already records.
    """

    gate = _gate()
    rows = _rows()
    floor = gate.ROUTE_ROW_FLOORS["wmma"]

    passing = _verdict(gate, rows, floor, quality_passed=True)
    assert passing["category_heldout_evidence_complete"] is True
    assert passing["category_heldout_qualification_claim"] is True

    failing = _verdict(gate, rows, floor, quality_passed=False)
    assert failing["category_heldout_evidence_complete"] is True, (
        "the evidence is still collected; only the claim should change"
    )
    assert failing["category_heldout_qualification_claim"] is False, (
        "a heldout run that fails a quality gate must not claim qualification"
    )
    assert "calibrated quality thresholds" in failing["qualification_blockers"]


def test_row_floor_boundary_is_inclusive():
    """Engagement is ``longest >= floor``, because the guards are ``<=``."""

    gate = _gate()
    floor = gate.ROUTE_ROW_FLOORS["wmma"]
    at_floor = _coverage(gate, [floor] * len(HELDOUT_CASES), floor)["category_heldout"]
    assert at_floor["checks"]["heldout_prompts_exceed_route_row_floor"] is True, (
        "a prompt exactly at the engaging row count does run the route"
    )
    below = _coverage(gate, [floor - 1] * len(HELDOUT_CASES), floor)["category_heldout"]
    assert below["checks"]["heldout_prompts_exceed_route_row_floor"] is False
