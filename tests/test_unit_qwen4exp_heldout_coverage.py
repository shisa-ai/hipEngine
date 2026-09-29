"""Category-heldout coverage must refuse to certify a route that never ran.

X-7 is the ledger's declared prerequisite ("Close before extending the admitted
scope") and lists category-heldout numerical evidence as its first missing
certificate. Before this, the gate hardcoded one canonical fixture hash, so a
heldout run could not be expressed at all and the blocker was unfalsifiable from
the artifact.

The trap this file guards: the heldout suite is **short**. Its 18 prompts run
39-71 tokens, and both layer-scoped routes decline at ``rows <= 256``. Pointing
the gate at it produces a flawless-looking comparison between two runs of the
*same* kernel - bit-identical candidate/teacher logits, a KL of exactly zero,
``measurement_valid: true``. The ledger's X-11 says the same thing from the other
side: "39-71-token task prompts never enter >256-row candidates. Inactive
short-task passes are controls, not qualification."

So the row floor is a check, not a note, and these tests pin both directions: it
refuses a suite that cannot engage the route, and it certifies one that can.
"""

import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "execution_profile_q8_wmma_prefill_layers_gate.py"
ADMISSION = REPO / "benchmarks" / "fixtures" / "qwen4exp_admission_suite_18prompt.json"

# The 18 heldout case ids and their categories, in the fixture's own order.
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


def _gate():
    if not GATE.exists():
        pytest.skip(f"gate script not present: {GATE}")
    spec = importlib.util.spec_from_file_location("_wmma_gate_heldout", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Capture:
    """The minimum shape ``qualification_coverage`` reads."""

    def __init__(self, prompt_id: str, category: str):
        self.prompt_id = prompt_id
        self.category = category
        self.strict = tuple({} for _ in range(129))
        self.candidate_runs = {
            "candidate": tuple(
                tuple({} for _ in range(129)) for _ in range(3)
            )
        }


def _captures():
    return [_Capture(i, c) for i, c in HELDOUT_CASES]


def _coverage(gate, rows):
    return gate.qualification_coverage(
        captures=_captures(),
        fixture_sha256=gate.ADMISSION_FIXTURE_SHA256,
        prefill_chunk_size=1024,
        fixture_rows=list(rows),
    )


def test_fixture_kind_separates_the_two_protocols():
    gate = _gate()
    assert gate._fixture_kind(gate.CANONICAL_FIXTURE_SHA256) == "canonical"
    assert gate._fixture_kind(gate.ADMISSION_FIXTURE_SHA256) == "category_heldout"
    assert gate._fixture_kind("0" * 64) == "unknown"


def test_heldout_fixture_hash_matches_the_file_on_disk():
    """A rotated fixture must not silently keep certifying under the old hash."""

    gate = _gate()
    if not ADMISSION.exists():
        pytest.skip(f"heldout fixture not present: {ADMISSION}")
    import hashlib

    observed = hashlib.sha256(ADMISSION.read_bytes()).hexdigest()
    assert observed == gate.ADMISSION_FIXTURE_SHA256, (
        "the heldout fixture changed on disk; the gate would classify it as "
        "'unknown' and silently stop producing heldout coverage"
    )


def test_heldout_run_does_not_claim_canonical_coverage():
    """The two protocols are not interchangeable.

    A heldout run that could be read as canonical coverage would let one
    certificate stand in for the other, which is the accounting defect X-6
    records rather than a missing measurement.
    """

    gate = _gate()
    rows = [int(c["prompt_tokens"]) for c in json.loads(ADMISSION.read_text())["cases"]]
    coverage = _coverage(gate, rows)
    assert coverage["fixture_kind"] == "category_heldout"
    assert coverage["complete"] is False, (
        "a heldout run must never set the canonical protocol's complete flag"
    )
    assert coverage["checks"]["canonical_fixture_identity"] is False


def test_short_heldout_suite_is_refused_as_a_control_not_a_qualification():
    """The false-pass guard, against the real suite on disk.

    This is the state the ledger is actually in: the suite exists and is
    complete, and it still cannot qualify the route, because 71 rows does not
    reach a 256-row floor.
    """

    gate = _gate()
    rows = [int(c["prompt_tokens"]) for c in json.loads(ADMISSION.read_text())["cases"]]
    assert max(rows) <= gate.ROUTE_ROW_FLOOR, (
        "the heldout suite now exceeds the row floor; this test's premise is "
        "stale and the blocker may genuinely be closable"
    )
    heldout = _coverage(gate, rows)["category_heldout"]
    assert heldout["checks"]["heldout_prompts_exceed_route_row_floor"] is False
    assert heldout["complete"] is False
    assert "control rather than qualification" in heldout["blocked_reason"]

    verdict = gate.qualification_verdict(
        _coverage(gate, rows), quality_passed=True, deterministic=True, clean=True
    )
    assert verdict["category_heldout_qualification_claim"] is False
    assert "category-heldout numerical evidence" in verdict["qualification_blockers"]


def test_row_floor_check_is_not_vacuous():
    """It must certify when the evidence exists, or it proves nothing.

    A check that is always ``False`` would keep the blocker open forever while
    looking like diligence. Both directions are pinned here, including the
    boundary: exactly at the floor is refused, because the routes decline at
    ``rows <= 256`` rather than ``rows < 256``.
    """

    gate = _gate()
    floor = gate.ROUTE_ROW_FLOOR

    long_form = _coverage(gate, [floor + 256] * len(HELDOUT_CASES))
    heldout = long_form["category_heldout"]
    assert heldout["checks"]["heldout_prompts_exceed_route_row_floor"] is True
    assert heldout["complete"] is True
    assert heldout["blocked_reason"] is None

    verdict = gate.qualification_verdict(
        long_form, quality_passed=True, deterministic=True, clean=True
    )
    assert verdict["category_heldout_qualification_claim"] is True
    assert "category-heldout numerical evidence" not in verdict["qualification_blockers"], (
        "producing the evidence must drop the blocker, or the accounting can "
        "never be closed by measurement"
    )

    at_floor = _coverage(gate, [floor] * len(HELDOUT_CASES))["category_heldout"]
    assert at_floor["checks"]["heldout_prompts_exceed_route_row_floor"] is False, (
        "the routes decline at rows <= 256, so exactly-at-floor is not "
        "engagement and must not be certified"
    )
