"""The provenance gate must stay wired into the publish checklist."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISH = ROOT / "docs" / "PUBLISH.md"
GATE = ROOT / "scripts" / "check_artifact_provenance.py"


def test_gate_script_exists():
    assert GATE.is_file(), "the provenance gate script must exist to be referenced"


def test_publish_checklist_runs_the_provenance_gate():
    text = PUBLISH.read_text()
    assert "scripts/check_artifact_provenance.py" in text, (
        "docs/PUBLISH.md must run the artifact provenance gate: reuse of retained artifacts is "
        "mandated by the next bullet, so the artifact-to-row mapping keeps published hardware "
        "provenance honest"
    )
    # The gate is only useful if the reason is stated; a bare command line gets skipped.
    assert "exist in the artifact each row cites" in text


def test_every_documented_flag_is_accepted_by_the_tool():
    """Docs may not cite flags the tool does not have.

    This checklist told the reader to run `check_artifact_provenance.py --export` and
    `--list-unreferenced`; neither exists, and the previous version of this test asserted the doc
    contained that exact string, so the test was certifying a command that could not run. Same class
    as `check_published_command_drift.py`, applied to prose instead of artifacts.
    """
    import re

    flat = " ".join(PUBLISH.read_text().split())
    documented: set[str] = set()
    for match in re.finditer(r"check_artifact_provenance\.py([^`]*)", flat):
        # `[^`]*` stops at the closing backtick, so only flags inside the code span count.
        documented |= set(re.findall(r"--[a-z][a-z0-9-]*", match.group(1)))
    assert documented, "the checklist should name the gate with its flags"
    help_text = subprocess.run(
        [sys.executable, str(GATE), "--help"], capture_output=True, text=True, timeout=120
    ).stdout
    assert "--show-warnings" in help_text
    for flag in sorted(documented):
        assert flag in help_text, f"docs/PUBLISH.md documents {flag}, which the gate rejects"


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
