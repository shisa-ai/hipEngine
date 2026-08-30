"""The provenance gate must stay wired into the publish checklist."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISH = ROOT / "docs" / "PUBLISH.md"
GATE = ROOT / "scripts" / "check_artifact_provenance.py"


def test_gate_script_exists():
    assert GATE.is_file(), "the provenance gate script must exist to be referenced"


def test_publish_checklist_runs_the_provenance_gate():
    text = PUBLISH.read_text()
    assert "scripts/check_artifact_provenance.py --export" in text, (
        "docs/PUBLISH.md must run the artifact provenance gate: reuse of retained artifacts is "
        "mandated by the next bullet, so the artifact-to-row mapping keeps published hardware "
        "provenance honest"
    )
    # The gate is only useful if the reason is stated; a bare command line gets skipped.
    assert "hardware rows cite one artifact" in text


def test_provenance_gate_sits_next_to_the_rollup_check():
    """Ordering matters: provenance must be verified before rows are exported/consumed."""
    lines = PUBLISH.read_text().splitlines()
    rollup = next(i for i, l in enumerate(lines) if "sync_benchmark_readme.py --check" in l)
    provenance = next(i for i, l in enumerate(lines) if "check_artifact_provenance.py" in l)
    needle = "Reuse current retained benchmark artifacts"
    reuse = next(i for i, l in enumerate(lines) if needle in l)
    assert rollup < provenance < reuse, (
        f"expected rollup({rollup}) < provenance({provenance}) < reuse({reuse}); a gate listed "
        "after the reuse instruction is read too late to prevent the mismatch it detects"
    )
