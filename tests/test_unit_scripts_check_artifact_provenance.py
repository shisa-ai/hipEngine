"""Tests for scripts/check_artifact_provenance.py (parse-only, tmp repos, no GPU).

The fixtures that matter are the ones shaped like the real mistake: a model line naming a quant the
run never used, and a supersession citing a file that does not exist.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_artifact_provenance.py"

BASE = {
    "model": "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
    "quant": "Q4_K_M",
    "hardware": "AMD Radeon Pro W7900 (gfx1100, RDNA3), device 0 of 2, 48.3 GB VRAM",
    "host": "Linux 7.1.3-2-cachyos x86_64, 64192 MB RAM",
}


def _load():
    spec = importlib.util.spec_from_file_location("cap_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path: pathlib.Path, artifacts: dict[str, dict], cited: list[str] | None = None):
    results = tmp_path / "benchmarks" / "results"
    results.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        (results / name).write_text(json.dumps(payload))
    names = list(artifacts) if cited is None else cited
    readme = tmp_path / "benchmarks" / "README.md"
    readme.write_text("# rollup\n\n" + "\n".join(f"see {n}" for n in names) + "\n")
    return tmp_path


def _pair(tmp_path: pathlib.Path, b_extra: dict | None = None):
    return _repo(
        tmp_path,
        {
            "2026-01-01-a.json": dict(BASE),
            "2026-01-02-b.json": {**BASE, **(b_extra or {})},
        },
    )


def test_clean_published_set_passes(tmp_path):
    report = _load().check_repo(_pair(tmp_path))
    assert report["violations"] == []
    assert report["artifacts_checked"] == 2
    assert report["exceptions_unmatched"] == []


def test_dangling_supersession_fails(tmp_path):
    """Today's actual failure: `supersedes` named a file that has never existed."""
    repo = _pair(
        tmp_path,
        {"supersedes": "w7900-gguf-q4ks-ar-amortization-prefill-split-2026-08-30.json"},
    )
    problems = {(v["problem"], v["artifact"]) for v in _load().check_repo(repo)["violations"]}
    assert ("DANGLING-SUPERSEDES", "2026-01-02-b.json") in problems


def test_dangling_link_inside_a_list_is_caught(tmp_path):
    repo = _pair(tmp_path, {"links": ["2026-01-01-a.json", "2025-12-31-gone.json"]})
    details = " ".join(v["detail"] for v in _load().check_repo(repo)["violations"])
    assert "links cites 2025-12-31-gone.json" in details


def test_valid_supersession_target_is_accepted(tmp_path):
    repo = _pair(tmp_path, {"supersedes": "2026-01-01-a.json (its route reading)"})
    assert _load().check_repo(repo)["violations"] == []


def test_quant_model_conflict_fails(tmp_path):
    """A Q4_K_S label on a Q4_K_M run is the second half of the same incident."""
    repo = _pair(tmp_path, {"model": "Qwen3.5-9B GGUF Q4_K_S, 152001 KiB"})
    problems = {v["problem"] for v in _load().check_repo(repo)["violations"]}
    assert "QUANT-MODEL-CONFLICT" in problems


def test_internal_quant_key_spelling_is_not_a_conflict(tmp_path):
    """gguf_q4_k_m and Q4_K_M are one fact; failing on that cries wolf on real rows."""
    repo = _repo(
        tmp_path,
        {
            "2026-01-01-a.json": {**BASE, "quant": "gguf_q4_k_m"},
            "2026-01-02-b.json": {**BASE, "quant": "Q4_K_M"},
        },
    )
    assert _load().check_repo(repo)["violations"] == []


def test_model_without_a_quant_token_is_not_a_conflict(tmp_path):
    repo = _pair(tmp_path, {"model": "Qwen3.6-35B-A3B", "quant": "Q8_K_XL"})
    assert _load().check_repo(repo)["violations"] == []


def test_missing_artifact_cited_by_readme_fails(tmp_path):
    repo = _repo(
        tmp_path,
        {"2026-01-01-a.json": dict(BASE)},
        cited=["2026-01-01-a.json", "gone.json"],
    )
    problems = {v["problem"] for v in _load().check_repo(repo)["violations"]}
    assert "MISSING-ARTIFACT" in problems


def test_unparseable_artifact_is_reported(tmp_path):
    repo = _repo(tmp_path, {"2026-01-01-a.json": dict(BASE)})
    (repo / "benchmarks" / "results" / "2026-01-01-a.json").write_text("{not json")
    problems = {v["problem"] for v in _load().check_repo(repo)["violations"]}
    assert "BAD-JSON" in problems


def test_jsonl_prose_is_not_mistaken_for_an_artifact_citation(tmp_path):
    repo = _repo(tmp_path, {"2026-01-01-a.json": dict(BASE)})
    (repo / "benchmarks" / "README.md").write_text(
        "prompts from prompts.jsonl and see 2026-01-01-a.json\n"
    )
    report = _load().check_repo(repo)
    assert report["artifacts_cited"] == 1


def test_unique_host_is_a_warning_not_a_failure(tmp_path):
    repo = _pair(tmp_path, {"host": "Linux 1.2.3-unique x86_64, 123 MB RAM"})
    report = _load().check_repo(repo)
    assert report["violations"] == []
    assert {w["problem"] for w in report["warnings"]} == {"UNIQUE-HOST"}


def test_structured_hardware_is_not_flagged_unique(tmp_path):
    repo = _pair(tmp_path, {"hardware": {"gpu": "AMD Radeon Pro W7900", "arch": "gfx1100"}})
    report = _load().check_repo(repo)
    flagged = {w["artifact"] for w in report["warnings"] if w["problem"] == "UNIQUE-HARDWARE"}
    # the other artifact's plain string may legitimately be unique; the dict one must not be flagged
    assert "2026-01-02-b.json" not in flagged


def test_exception_suppresses_and_stale_exception_fails(tmp_path):
    module = _load()
    repo = _pair(tmp_path, {"supersedes": "gone.json"})
    wrong = module.exception_key(
        "2026-01-02-b.json", "DANGLING-SUPERSEDES", "links cites gone.json"
    )
    unmatched = module.check_repo(repo, exceptions={wrong: "2026-01-01 test reason"})
    assert unmatched["exceptions_unmatched"] == [wrong]

    good = module.exception_key(
        "2026-01-02-b.json", "DANGLING-SUPERSEDES", "supersedes cites gone.json"
    )
    clean = module.check_repo(repo, exceptions={good: "2026-01-01 test reason"})
    assert clean["violations"] == []
    assert clean["exceptions_matched"] == [good]


def test_main_exit_codes(tmp_path):
    module = _load()
    assert module.main(["--repo", str(_pair(tmp_path))]) == 0
    bad = _repo(tmp_path, {"2026-01-01-a.json": {**BASE, "supersedes": "gone.json"}})
    assert module.main(["--repo", str(bad)]) == 1


def test_published_set_in_this_repo_is_clean():
    """The real tree must pass today, or the gate ships broken."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    report = _load().check_repo(repo)
    # The compact scoreboard contract permits fewer than 40 direct result links;
    # keep enough breadth to catch an accidentally empty or single-lane publication.
    assert report["artifacts_checked"] >= 30
    assert report["violations"] == []


# --- machine-identity rules (added after I fabricated host blocks in two artifacts) ---------------

def _named(tmp_path, entries):
    """Two or more artifacts that all name the machine and share every other field."""
    return _repo(
        tmp_path,
        {
            f"2026-01-0{i}-x.json": {
                **BASE,
                "provenance": {"host_name": host, "cpu_model": cpu},
            }
            for i, (host, cpu) in enumerate(entries, start=1)
        },
    )


def _problems(report, problem):
    return [w for w in report["warnings"] if w["problem"] == problem]


def test_one_host_with_two_cpu_families_warns(tmp_path):
    repo = _named(
        tmp_path,
        [
            ("epyc", "AMD Ryzen 9 5950X 16-Core Processor"),
            ("epyc", "AMD Ryzen Threadripper 3970X 32-Core Processor"),
        ],
    )
    report = _load().check_repo(repo)
    conflicts = _problems(report, "HOST-CPU-CONFLICT")
    assert len(conflicts) == 1, conflicts
    assert conflicts[0]["artifact"] == "2026-01-02-x.json"
    assert "epyc" in conflicts[0]["detail"]
    assert report["violations"] == [], "identity mismatches stay in the warning tier"


def test_suffix_variants_of_one_cpu_are_not_a_conflict(tmp_path):
    repo = _named(
        tmp_path,
        [
            ("epyc", "AMD Ryzen 9 5950X"),
            ("epyc", "AMD Ryzen 9 5950X 16-Core Processor"),
            ("epyc", "AMD Ryzen 9 5950X (16 cores)"),
        ],
    )
    assert _problems(_load().check_repo(repo), "HOST-CPU-CONFLICT") == []


def test_prose_naming_another_cpu_does_not_trigger_the_rule(tmp_path):
    """Key-scoped on purpose: quoting the wrong CPU is not the same as asserting it."""
    repo = _repo(
        tmp_path,
        {
            "2026-01-01-x.json": {
                **BASE,
                "provenance": {"host_name": "epyc", "cpu_model": "AMD Ryzen 9 5950X"},
                "host_correction_note": "previously miswritten as Threadripper 3970X",
            }
        },
    )
    assert _problems(_load().check_repo(repo), "HOST-CPU-CONFLICT") == []


def test_host_asserted_by_too_few_artifacts_warns(tmp_path):
    """The signature of an invented hostname: a name hardly any published artifact cites."""
    repo = _repo(
        tmp_path,
        {
            f"2026-01-0{i}-x.json": {
                **BASE,
                "provenance": {
                    "host_name": "epyc" if i < 4 else "gputm-3087-00104",
                    "cpu_model": "AMD Ryzen 9 5950X",
                },
            }
            for i in (1, 2, 3, 4)
        },
    )
    report = _load().check_repo(repo)
    rare = _problems(report, "RARE-HOST")
    assert [w["artifact"] for w in rare] == ["2026-01-04-x.json"], rare
    assert "gputm-3087-00104" in rare[0]["detail"]
    assert report["violations"] == []


def test_identity_coverage_is_reported(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "2026-01-01-x.json": {
                **BASE,
                "provenance": {"host_name": "epyc", "cpu_model": "AMD Ryzen 9 5950X"},
            },
            "2026-01-02-x.json": dict(BASE),
        },
    )
    coverage = _load().check_repo(repo)["identity_coverage"]
    assert coverage == {"cited": 2, "naming_machine": 1}


# --- placeholder invocations (after I shipped a command no process ever ran) -------------------

def _with_command(tmp_path, command, extra: dict | None = None):
    return _repo(tmp_path, {"2026-01-01-x.json": {**BASE, "command": command, **(extra or {})}})


def test_a_placeholder_in_a_recorded_command_warns(tmp_path):
    repo = _with_command(tmp_path, ".venv/bin/python script.py --model <path> --out <file>")
    warnings = _problems(_load().check_repo(repo), "PLACEHOLDER-COMMAND")
    assert len(warnings) == 1
    assert "<file>" in warnings[0]["detail"] or "<path>" in warnings[0]["detail"]


def test_null_command_is_honest_and_does_not_warn(tmp_path):
    """The remedy for an unrecordable invocation is null plus an explanation."""
    repo = _with_command(
        tmp_path,
        None,
        {"command_note": "assembled by an inline builder with no argv"},
    )
    assert _problems(_load().check_repo(repo), "PLACEHOLDER-COMMAND") == []


def test_a_real_command_does_not_warn(tmp_path):
    repo = _with_command(tmp_path, "scripts/probe.py --model /models/a.gguf --widths 1,2,3")
    assert _problems(_load().check_repo(repo), "PLACEHOLDER-COMMAND") == []


def test_placeholder_scan_is_scoped_to_invocation_keys(tmp_path):
    """Templates quoted in prose (findings, notes) must not be mistaken for recorded commands."""
    repo = _repo(
        tmp_path,
        {
            "2026-01-01-x.json": {
                **BASE,
                "command": "scripts/probe.py --rows 8",
                "findings": ["the launch form is scripts/probe.py --rows <n> per lane"],
            }
        },
    )
    assert _problems(_load().check_repo(repo), "PLACEHOLDER-COMMAND") == []
