"""Unit tests for the legacy worklog port helper and its Worklog2 schema fixes."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import worklog  # noqa: E402
import worklog_port_legacy as port  # noqa: E402


# ---------------------------------------------------------------------------
# worklog.py schema-authority fixes
# ---------------------------------------------------------------------------

def test_slugify_truncation_never_leaves_trailing_dash() -> None:
    long_title = "Port " + "qwen3 " * 12 + "tail"
    topic = worklog.slugify(long_title, fallback="entry")
    assert not topic.endswith("-")
    assert worklog.slugify(topic, fallback="entry") == topic  # self-consistent slug


def test_conflict_markers_only_flag_column_zero_outside_fences() -> None:
    documented = "The union driver rewrites `<<<<<<< HEAD` blocks mid-line.\n```\n>>>>>>> side\n=======\n```\n"
    assert worklog._line_conflict_marker(documented) is None
    assert worklog._line_conflict_marker("<<<<<<< HEAD\nresolved\n") is not None
    assert worklog._line_conflict_marker("=======\n") is not None
    assert worklog._line_conflict_marker(">>>>>>>\n") is not None


def test_indented_fence_lines_do_not_toggle_fence_state() -> None:
    body = "## Summary\n\n- text\n\n    ```\n    # not a title\n    ```\n\n## Next\n"
    positions, sections = worklog._markdown_heading_positions(body)
    assert positions == []  # only entries carry a `# ` title; body has none
    assert sections["## Summary"] == [0]
    assert sections["## Next"] == [8]  # indented fences above never toggled state


# ---------------------------------------------------------------------------
# Legacy journal splitting
# ---------------------------------------------------------------------------

LEGACY_SAMPLE = (
    "# hipEngine Work Log\n"
    "\n"
    "Preamble line.\n"
    "\n"
    "## 2026-05-12 — First entry\n"
    "\n"
    "Body of the first entry.\n"
    "\n"
    "## 2026-05-13 — Second entry\n"
    "\n"
    "```python\n"
    "## not a heading inside a fence\n"
    "# 13 passed\n"
    "```\n"
    "\n"
    "## 2026-05-13 — Third entry\n"
    "\n"
    "Final body.\n"
)


@pytest.fixture()
def split_sample() -> tuple[list[str], list[int], list[str], set[int]]:
    return port.split_legacy(LEGACY_SAMPLE)


def test_split_legacy_counts_entries_and_fenced_headings(split_sample) -> None:
    lines, heading_indices, headings, inside = split_sample
    assert headings == [
        "2026-05-12 — First entry",
        "2026-05-13 — Second entry",
        "2026-05-13 — Third entry",
    ]
    assert heading_indices[0] == 4  # six-line preamble: title, blank, text, blank, ---, blank
    assert "## not a heading inside a fence" not in headings
    assert any("# 13 passed" == lines[index] and index in inside for index in range(len(lines)))


def test_entry_body_is_verbatim_span(split_sample) -> None:
    lines, heading_indices, headings, inside = split_sample
    body = port.entry_body_lines(lines, heading_indices, 1, len(lines))
    assert body == ["", "```python", "## not a heading inside a fence", "# 13 passed", "```", ""]
    span = port.entry_span(heading_indices, 1, len(lines))
    assert span == (heading_indices[1] + 1, heading_indices[2])


def test_body_needs_wrap_rules() -> None:
    assert port.body_needs_wrap(["", "# 13 passed", ""], set(), 10)
    assert port.body_needs_wrap(["", "## Summary", ""], set(), 10)
    assert port.body_needs_wrap(["", "   ", ""], set(), 10)  # empty body
    assert not port.body_needs_wrap(["", "Plain text.", ""], set(), 10)
    # body line i sits at heading_index+1+i; fence-interior lines come from the map
    assert not port.body_needs_wrap(["", "```", "# inside fence", "```", ""], {14}, 11)


def test_wrapper_fence_run_outranks_body_fences() -> None:
    assert port.wrapper_fence_run(["", "```python", "code", "```", ""]) == 4
    assert port.wrapper_fence_run(["", "no fences", ""]) == 4


# ---------------------------------------------------------------------------
# Commit mapping
# ---------------------------------------------------------------------------

def _patch_record(sha: str, adate: str, subject: str, adds: list[str], dels: int = 0) -> str:
    patch = f"\x01{sha}\x02{adate}\x02{subject}\n"
    for line in adds:
        patch += f"+## {line}\n"
    for _ in range(dels):
        patch += "-removed line\n"
    return patch


def test_build_commit_map_attributes_first_appends(monkeypatch) -> None:
    stream = (
        _patch_record("a" * 40, "2026-05-12T10:00:00+09:00", "docs: seed", ["2026-05-12 — First entry"])
        + _patch_record("b" * 40, "2026-05-12T11:00:00+09:00", "feat: second", ["2026-05-13 — Second entry"])
        # union-merge replay re-adds the first two headings plus one new one
        + _patch_record("c" * 40, "2026-05-12T12:00:00+09:00", "merge: integrate", [
            "2026-05-12 — First entry",
            "2026-05-13 — Second entry",
            "2026-05-13 — Third entry",
        ], dels=2)
    )
    monkeypatch.setattr(port, "run_git", lambda *args: stream)
    from collections import Counter

    mapping = port.build_commit_map(Counter(["2026-05-12 — First entry", "2026-05-13 — Second entry", "2026-05-13 — Third entry"]))
    assert mapping["missing"] == {}
    assert mapping["anomalies"]["readd_skipped"] == 2
    assert mapping["anomalies"]["deletion_commits"][0]["commit"] == "c" * 40
    first = mapping["events"]["2026-05-12 — First entry"]
    assert len(first) == 1 and first[0]["commit"] == "a" * 40
    third = mapping["events"]["2026-05-13 — Third entry"]
    assert len(third) == 1 and third[0]["commit"] == "c" * 40


def test_assign_occurrences_binds_repeats_by_date(monkeypatch) -> None:
    from collections import Counter

    monkeypatch.setattr(port, "run_git", lambda *args: (
        _patch_record("d" * 40, "2026-06-17T09:00:00+09:00", "docs: one", ["2026-06-17"])
        + _patch_record("e" * 40, "2026-06-17T18:00:00+09:00", "docs: two", ["2026-06-17"])
        + _patch_record("f" * 40, "2026-06-18T09:00:00+09:00", "docs: three", ["2026-06-17"])
    ))
    mapping = port.build_commit_map(Counter({"2026-06-17": 3}))
    assignments, duplicates = port.assign_occurrences(["2026-06-17"] * 3, mapping, "0" * 40)
    assert len(duplicates) == 1 and duplicates[0]["occurrences"] == 3
    commits = [item["event"]["commit"] for item in assignments]
    assert commits == ["d" * 40, "e" * 40, "f" * 40]  # bound in author-date order
    assert all(item["timestamp_source"] == "commit" for item in assignments)


def test_assign_occurrences_window_fallback(monkeypatch) -> None:
    monkeypatch.setattr(port, "run_git", lambda *args: "")
    mapping = port.build_commit_map(Counter({"2026-06-17": 1}))
    assignments, duplicates = port.assign_occurrences(["2026-06-17"], mapping, "9" * 40)
    assert duplicates == []
    assert assignments[0]["timestamp_source"] == "window"
    assert assignments[0]["event"]["commit"] == "9" * 40
    _moment, stamp_text = port.entry_timestamp(assignments[0], "2026-06-17")
    assert stamp_text == "2026-06-17T12:00:00.000000Z"


# ---------------------------------------------------------------------------
# Rendering and schema validation
# ---------------------------------------------------------------------------

def test_render_entry_passes_worklog_schema() -> None:
    filename, text = port.render_entry(
        heading="2026-05-12 — First entry",
        body_lines=["", "Body of the first entry.", ""],
        span=(5, 8),
        timestamp_text="2026-05-12T01:00:00.000000Z",
        commit="a" * 40,
        subject="docs: seed the journal",
        topic="2026-05-12-first-entry",
        wrapped=False,
    )
    assert filename.startswith("20260512T010000.000000Z-legacy-2026-05-12-first-entry-")
    port.validate_in_memory(filename, text)
    fields, _body = worklog.parse_entry_text(text)
    assert fields["worker"] == "legacy"
    assert fields["base_commit"] == "a" * 40
    assert fields["timestamp"] == "2026-05-12T01:00:00.000000Z"


def test_render_wrapped_entry_with_inner_fences_stays_scannable() -> None:
    body = ["", "# 13 passed", "", "```python", "## fenced", "```", ""]
    filename, text = port.render_entry(
        heading="2026-05-13 — Second entry",
        body_lines=body,
        span=(9, 16),
        timestamp_text="2026-05-13T02:00:00.000001Z",
        commit="b" * 40,
        subject="feat: second",
        topic="2026-05-13-second-entry",
        wrapped=True,
    )
    port.validate_in_memory(filename, text)
    assert "````markdown" in text  # wrapper outranks the 3-backtick body fence


def test_entry_timestamp_uses_intra_commit_order() -> None:
    assignment = {
        "timestamp_source": "commit",
        "event": {"author_date": "2026-05-13T18:30:00+09:00", "order": 2},
    }
    _moment, stamp = port.entry_timestamp(assignment, "2026-05-13 — anything")
    assert stamp == "2026-05-13T09:30:00.000002Z"  # author date converted to UTC


def test_stamp_of_matches_filename_regex() -> None:
    stamp = port.stamp_of("2026-05-13T09:30:00.000002Z")
    assert stamp == "20260513T093000.000002Z"
    assert worklog.FILENAME_RE.fullmatch(f"{stamp}-legacy-topic-abcdef.md")


def test_validate_in_memory_names_offending_file() -> None:
    with pytest.raises(port.PortError, match=r"worklog/entries/20260513T093000"):
        port.validate_in_memory("20260513T093000.000002Z-legacy-topic-abcdef.md", "not an entry")


# ---------------------------------------------------------------------------
# Retired-journal provenance
# ---------------------------------------------------------------------------

def _retired_provenance(tmp_path: Path, payload: bytes, recorded_sha: str) -> Path:
    manifest_path = tmp_path / "legacy-port-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "source_sha256": recorded_sha,
                "cutoff_commit": "7" * 40,
                "source_history": {"commit": "a" * 40, "path": "WORKLOG-LEGACY.md"},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_load_legacy_reads_retired_journal_from_recorded_history(monkeypatch, tmp_path) -> None:
    payload = b"# Legacy\n\n## 2026-08-10 - Frozen\n\n- Evidence.\n"
    manifest_path = _retired_provenance(
        tmp_path, payload, hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(port, "LEGACY_PATH", tmp_path / "missing-legacy.md")
    monkeypatch.setattr(port, "LEGACY_MANIFEST_PATH", tmp_path / "missing-manifest.json")
    monkeypatch.setattr(port, "PORT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(port, "run_git_bytes", lambda *args: payload)

    text, meta = port.load_legacy()

    assert text.startswith("# Legacy")
    assert meta["sha256"] == hashlib.sha256(payload).hexdigest()
    assert meta["cutoff_commit"] == "7" * 40
    assert meta["source"]["commit"] == "a" * 40


def test_load_legacy_rejects_history_bytes_that_drift(monkeypatch, tmp_path) -> None:
    payload = b"# Legacy\n"
    manifest_path = _retired_provenance(
        tmp_path, payload, hashlib.sha256(b"different bytes\n").hexdigest()
    )
    monkeypatch.setattr(port, "LEGACY_PATH", tmp_path / "missing-legacy.md")
    monkeypatch.setattr(port, "LEGACY_MANIFEST_PATH", tmp_path / "missing-manifest.json")
    monkeypatch.setattr(port, "PORT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(port, "run_git_bytes", lambda *args: payload)

    with pytest.raises(port.PortError, match="sha256"):
        port.load_legacy()
