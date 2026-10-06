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


def test_prompt_fixture_link_is_not_a_result_artifact(tool, tmp_path):
    repo = _make_repo(tmp_path, command=GOOD)
    readme = repo / "benchmarks" / "README.md"
    readme.write_text(readme.read_text() + "\n[fixture](prompts/session.json)\n")
    report = tool.check_repo(repo)
    assert report["violations"] == []
    assert report["artifacts_checked"] == 1


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


@pytest.mark.parametrize("extra,expected", [("", []), (" --removed", ["UNKNOWN-FLAG"])])
def test_registered_peer_worktree_commands_use_current_script(
    tool, tmp_path: pathlib.Path, monkeypatch, extra, expected,
) -> None:
    repo = tmp_path / "current"
    peer = tmp_path / "peer"
    _make_repo(repo, command=f"python {peer}/scripts/bench.py --widths 1{extra}")
    monkeypatch.setattr(tool, "_worktree_roots", lambda root: (repo, peer), raising=False)
    report = tool.check_repo(repo)
    assert [v["problem"] for v in report["violations"]] == expected


def test_unregistered_same_basename_is_not_a_worktree_alias(tool, tmp_path, monkeypatch):
    repo = tmp_path / "current"
    _make_repo(repo, command=f"python {tmp_path}/outside/scripts/bench.py --widths 1")
    monkeypatch.setattr(tool, "_worktree_roots", lambda root: (repo,), raising=False)
    assert tool.check_repo(repo)["violations"][0]["problem"] == "SCRIPT-NOT-IN-REPO"


def test_peer_worktree_path_cannot_escape_current_repo(tool, tmp_path, monkeypatch):
    repo = tmp_path / "current"
    peer = tmp_path / "peer"
    _make_repo(repo, command=f"python {peer}/../outside/scripts/bench.py --widths 1")
    monkeypatch.setattr(tool, "_worktree_roots", lambda root: (repo, peer), raising=False)
    assert tool.check_repo(repo)["violations"][0]["problem"] == "SCRIPT-NOT-IN-REPO"


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


def _write_script(root: pathlib.Path, name: str, body: str) -> None:
    (root / "scripts" / name).write_text(body)


def test_flags_from_a_composed_parser_are_accepted(tool, tmp_path: pathlib.Path) -> None:
    """`suite.build_parser()` flags belong to the script that composes them."""
    repo = _make_repo(tmp_path, command=".venv/bin/python scripts/gate.py --shared-flag")
    _write_script(
        repo,
        "suite.py",
        "import argparse\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        '    parser.add_argument("--shared-flag")\n'
        "    return parser\n",
    )
    _write_script(
        repo,
        "gate.py",
        "from scripts import suite\n"
        "parser = suite.build_parser()\n"
        "args = parser.parse_args()\n",
    )
    assert tool.check_repo(repo)["violations"] == []


def test_flags_from_a_shared_helper_and_its_literal_spread_are_accepted(
    tool, tmp_path: pathlib.Path
) -> None:
    """`add_kv_policy_args` style helpers declare flags the named script never mentions."""
    repo = _make_repo(
        tmp_path,
        command=(
            ".venv/bin/python scripts/gate.py --kv-storage int8 "
            "--kv-storage-dtype int8"
        ),
    )
    _write_script(
        repo,
        "kv_args.py",
        "def add_kv_args(parser, *, legacy_flags=()):\n"
        '    parser.add_argument("--kv-storage", *legacy_flags)\n',
    )
    _write_script(
        repo,
        "gate.py",
        "import argparse\n"
        "from scripts.kv_args import add_kv_args\n"
        "parser = argparse.ArgumentParser()\n"
        'add_kv_args(parser, legacy_flags=("--kv-storage-dtype",))\n'
        "args = parser.parse_args()\n",
    )
    assert tool.check_repo(repo)["violations"] == []


def test_boolean_optional_action_also_accepts_the_negation(tool, tmp_path: pathlib.Path) -> None:
    repo = _make_repo(tmp_path, command=".venv/bin/python scripts/bench.py --no-warmup")
    (repo / "scripts" / "bench.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        'p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)\n'
        "args = p.parse_args()\n"
    )
    assert tool.check_repo(repo)["violations"] == []


def test_an_unreadable_helper_skips_the_script_instead_of_guessing(
    tool, tmp_path: pathlib.Path
) -> None:
    """A helper whose option names are computed disables the gate for that script, visibly."""
    repo = _make_repo(tmp_path, command=".venv/bin/python scripts/gate.py --maybe-removed")
    _write_script(
        repo,
        "kv_args.py",
        "def add_kv_args(parser, names):\n"
        "    parser.add_argument(*names)\n",
    )
    _write_script(
        repo,
        "gate.py",
        "from scripts.kv_args import add_kv_args\n"
        "add_kv_args(parser, flags)\n",
    )
    report = tool.check_repo(repo)
    assert report["violations"] == []
    assert report["scripts_skipped"] == ["scripts/gate.py"]


def _record_rename(root: pathlib.Path, old: str, new: str) -> None:
    directory = root / "docs" / "testing"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test-tier-migration-2026-09-12.json").write_text(
        json.dumps({"modules": [{"old": old, "new": new}]}) + "\n"
    )


@pytest.mark.parametrize("record", [
    "docs/testing/test-tier-migration-2026-09-12.json",
    "docs/testing/test-tier-migration-ud-2026-09-13.json",
])
def test_a_renamed_test_target_resolves_through_the_migration_record(
    tool, tmp_path: pathlib.Path, record: str
) -> None:
    """A historical test path stays checkable without rewriting the published row."""
    repo = _make_repo(
        tmp_path, command=".venv/bin/python -m pytest -q tests/test_old_name.py"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_unit_new_name.py").write_text("")
    _record_rename(repo, "tests/test_old_name.py", "tests/test_unit_new_name.py")
    original = repo / tool.TIER_RENAME_RECORD
    target = repo / record
    if target != original:
        original.rename(target)
    report = tool.check_repo(repo)
    assert report["violations"] == []
    assert report["renamed_targets"] == [
        "a.json::tests/test_old_name.py->tests/test_unit_new_name.py"
    ]


def test_a_test_target_that_is_neither_present_nor_recorded_is_a_violation(
    tool, tmp_path: pathlib.Path
) -> None:
    repo = _make_repo(tmp_path, command=".venv/bin/python -m pytest -q tests/test_gone.py")
    report = tool.check_repo(repo)
    assert [v["problem"] for v in report["violations"]] == ["SCRIPT-MISSING"]
    assert report["violations"][0]["detail"] == "tests/test_gone.py"
    assert report["renamed_targets"] == []


def test_every_python_target_in_a_pytest_command_is_checked(
    tool, tmp_path: pathlib.Path
) -> None:
    """Only checking the first `.py` token hid four renamed targets in one published row."""
    repo = _make_repo(
        tmp_path,
        command=".venv/bin/python -m pytest -q tests/test_here.py tests/test_gone.py",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_here.py").write_text("")
    report = tool.check_repo(repo)
    assert [v["detail"] for v in report["violations"]] == ["tests/test_gone.py"]


def test_pytest_invocations_are_not_checked_against_a_script_parser(
    tool, tmp_path: pathlib.Path
) -> None:
    """A test module declares no argparse flags; pytest's own flags are not drift."""
    repo = _make_repo(
        tmp_path,
        command=".venv/bin/python -m pytest -q tests/test_here.py --maxfail=1",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_here.py").write_text("")
    assert tool.check_repo(repo)["violations"] == []


def test_a_glob_test_target_expands_to_the_files_it_matches(
    tool, tmp_path: pathlib.Path
) -> None:
    """A recorded `pytest tests/test_unit_yue2_*.py` is reproducible; it is just not one path."""
    repo = _make_repo(
        tmp_path, command=".venv/bin/python -m pytest -q tests/test_unit_yue2_*.py"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_unit_yue2_ar_runtime.py").write_text("")
    (repo / "tests" / "test_unit_yue2_vae.py").write_text("")
    report = tool.check_repo(repo)
    assert report["violations"] == [], report
    assert report["glob_targets"] == ["a.json::tests/test_unit_yue2_*.py->2"]


def test_a_glob_that_matches_nothing_is_a_violation(tool, tmp_path: pathlib.Path) -> None:
    """Expansion must fail closed, or a glob would stand in for targets that no longer exist."""
    repo = _make_repo(
        tmp_path, command=".venv/bin/python -m pytest -q tests/test_unit_yue2_*.py"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_unit_other.py").write_text("")
    report = tool.check_repo(repo)
    assert [v["problem"] for v in report["violations"]] == ["SCRIPT-MISSING"], report
    assert report["violations"][0]["detail"] == "tests/test_unit_yue2_*.py"
    assert report["glob_targets"] == []


def test_the_real_repository_passes_the_gate_with_recorded_exceptions(tool) -> None:
    """The gate must be green on HEAD, with pre-existing drift named rather than hidden."""
    report = tool.check_repo(REPO)
    assert not [
        v for v in report["violations"]
    ], f"published commands drifted: {report['violations'][:4]}"
    assert report["artifacts_checked"] > 20
    assert report["exceptions_unmatched"] == []
    # The tier migration renamed the DMS targets this published row recorded. Resolved, not
    # hidden: the artifact keeps the path that existed when the row was measured.
    assert [
        name for name in report["renamed_targets"]
        if name.startswith("2026-09-07-rx7900xtx-dms-int8-postfix-audit.json::")
    ] == [
        "2026-09-07-rx7900xtx-dms-int8-postfix-audit.json::tests/"
        "test_dms_int8_backend_integration.py->tests/test_gpu_dms_int8_backend_integration.py",
        "2026-09-07-rx7900xtx-dms-int8-postfix-audit.json::tests/"
        "test_dms_int8_device_payloads.py->tests/test_gpu_dms_int8_device_payloads.py",
        "2026-09-07-rx7900xtx-dms-int8-postfix-audit.json::tests/"
        "test_dms_streaming_pack_hip.py->tests/test_gpu_dms_streaming_pack_hip.py",
        "2026-09-07-rx7900xtx-dms-int8-postfix-audit.json::tests/"
        "test_kvcache_dms.py->tests/test_unit_kvcache_dms.py",
        "2026-09-07-rx7900xtx-dms-int8-postfix-audit.json::tests/"
        "test_kvcache_dms_device_hip.py->tests/test_gpu_kvcache_dms_device_hip.py",
    ]
    # A script whose CLI cannot be read statically is skipped, never guessed at. Pinning the
    # list means a newly un-inspectable published script fails here instead of quietly
    # dropping out of the gate, which is how `--require-mtp` drift went unnoticed.
    assert report["scripts_skipped"] == ["scripts/qwen35_batch_retained_bench.py"]
