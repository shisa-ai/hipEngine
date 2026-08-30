"""RED test for the published-artifact command drift gate.

Two bugs in one session had the same shape: a benchmark artifact records the command that
produced it, then a later refactor renames or deletes a flag and the published row silently
loses its repro recipe. Once, the flag my own rollup rewrite deleted (`--prior-config-changed`)
broke a published artifact's `source_command`; later, `--require-mtp` was gone from
`gguf_mtp_c1c8_server_bench.py` while the headline grouped-prefill artifact still recorded it.

The gate cannot police every historical artifact - 1579 recorded commands in `scripts/*.py`
carry 125 distinct historical drifts, mostly flags removed months after the run - so it polices
the set that matters: artifacts cited by `benchmarks/README.md`, i.e. the published rows. Pre-
existing exceptions are listed with dates rather than rewritten, because a published row's
provenance belongs to its author."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_published_command_drift.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_published_command_drift_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load()


def _make_repo(root: pathlib.Path, *, command: str, artifact: str = "a.json") -> pathlib.Path:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "bench.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        'p.add_argument("--widths")\n'
        'p.add_argument("--model")\n'
        "args = p.parse_args()\n"
    )
    results = root / "benchmarks" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / artifact).write_text(json.dumps({"command": command, "value": 1}) + "\n")
    (root / "benchmarks" / "README.md").write_text(
        f"Row produced by `{artifact}`.\n"
    )
    return root


GOOD = ".venv/bin/python scripts/bench.py --widths 1,2 --model m.gguf"


def test_a_valid_command_passes(tool, tmp_path: pathlib.Path) -> None:
    repo = _make_repo(tmp_path, command=GOOD)
    report = tool.check_repo(repo)
    assert report["violations"] == [], report
    assert report["artifacts_checked"] == 1


def test_a_flag_the_script_no_longer_declares_is_a_violation(tool, tmp_path: pathlib.Path) -> None:
    repo = _make_repo(tmp_path, command=GOOD + " --require-mtp")
    report = tool.check_repo(repo)
    assert [v["problem"] for v in report["violations"]] == ["UNKNOWN-FLAG"], report
    assert report["violations"][0]["detail"] == "--require-mtp"


def test_flag_with_inline_value_is_matched_on_the_flag_name(tool, tmp_path: pathlib.Path) -> None:
    repo = _make_repo(tmp_path, command=GOOD + " --widths=3")
    assert tool.check_repo(repo)["violations"] == []


def test_commands_pointing_outside_the_repo_cannot_be_reproduced(
    tool, tmp_path: pathlib.Path
) -> None:
    repo = _make_repo(tmp_path, command=".venv/bin/python /tmp/scratch/bench.py --widths 1")
    report = tool.check_repo(repo)
    assert [v["problem"] for v in report["violations"]] == ["SCRIPT-NOT-IN-REPO"], report


def test_missing_script_is_a_violation(tool, tmp_path: pathlib.Path) -> None:
    repo = _make_repo(tmp_path, command=".venv/bin/python scripts/gone.py --widths 1")
    report = tool.check_repo(repo)
    assert [v["problem"] for v in report["violations"]] == ["SCRIPT-MISSING"], report


def test_allowlisted_problems_are_reported_but_not_failing(
    tool, tmp_path: pathlib.Path
) -> None:
    repo = _make_repo(tmp_path, command=GOOD + " --gone-flag")
    key = tool.exception_key("a.json", "UNKNOWN-FLAG", "--gone-flag")
    report = tool.check_repo(repo, exceptions={key: "2026-01-01 pre-existing"})
    assert report["violations"] == [], report
    assert report["exceptions_matched"] == [key]


def test_unmatched_exception_entries_are_theirselves_reported(
    tool, tmp_path: pathlib.Path
) -> None:
    repo = _make_repo(tmp_path, command=GOOD)
    report = tool.check_repo(repo, exceptions={"a.json::NOPE::--x": "stale note"})
    assert report["exceptions_unmatched"] == ["a.json::NOPE::--x"], report


def test_readme_citation_discovery_finds_every_cited_artifact(
    tool, tmp_path: pathlib.Path
) -> None:
    repo = _make_repo(tmp_path, command=GOOD)
    (repo / "benchmarks" / "results" / "b.json").write_text(json.dumps({"command": GOOD}))
    (repo / "benchmarks" / "README.md").write_text("see `a.json` and `b.json`")
    assert tool.check_repo(repo)["artifacts_checked"] == 2


def test_the_real_repository_passes_the_gate_with_recorded_exceptions(tool) -> None:
    """The gate must be green on HEAD, with pre-existing drift named rather than hidden."""
    report = tool.check_repo(REPO)
    assert not [
        v for v in report["violations"]
    ], f"published commands drifted: {report['violations'][:4]}"
    assert report["artifacts_checked"] > 20
    assert report["exceptions_unmatched"] == []
