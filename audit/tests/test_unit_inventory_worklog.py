"""Contract tests for the worklog extractor.

The property that matters here is precision: the journal holds 10,000+ entries
and almost all of them are history. A row must mean the entry *declared
unfinished business*, so the tests pin both sides — the markers that make a row
and the ordinary shapes that must not.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hipaudit.inventory import worklog  # noqa: E402

BASE = "0" * 40


def entry(name, *, status="completed", worker="pi", topic="demo-topic",
          timestamp="2026-08-01T00:00:00.000000Z", changes="did the work",
          next_text="None."):
    return f"""---
schema: 1
timestamp: {timestamp}
worker: {worker}
branch: main
worktree: hipEngine-main
base_commit: {BASE}
status: {status}
topic: {topic}
---

# {name}

## Summary

one line

## Changes

{changes}

## Validation

ran it

## Next

{next_text}
"""


class WorklogExtractor(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._dir.name)
        self._patch = mock.patch.object(worklog, "ENTRY_DIR", self.dir)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._dir.cleanup()

    def write(self, name, text):
        (self.dir / f"{name}.md").write_text(text, encoding="utf-8")

    def extract(self):
        return worklog.extract()

    def rows_for(self, name):
        rows, _ = self.extract()
        return [r for r in rows if r.key == name]

    def test_blocked_handoff_and_checkpoint_are_rows(self):
        for status in ("blocked", "handoff", "checkpoint"):
            with self.subTest(status=status):
                self.write("a", entry("A", status=status, next_text="Wait for the ROCm fix."))
                rows = self.rows_for("a")
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].evidence["open_by"], "status")
                self.assertTrue(any(f"recorded as {status}" in s for s in rows[0].signals))

    def test_completed_forward_reference_is_not_a_row(self):
        """A completed entry's Next is the following unit, not a claim of debt."""
        self.write("a", entry("A", next_text="- Run the B1-B5 matrix, then freeze the table."))
        self.assertEqual(self.rows_for("a"), [])

    def test_completed_entry_whose_next_names_a_dependency_is_a_row(self):
        self.write("a", entry("A", next_text="- Recovery stays blocked on gfx1100 hardware access."))
        rows = self.rows_for("a")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].evidence["open_by"], "dependency")
        self.assertIn("dependency", rows[0].signals[0])

    def test_a_bare_mention_of_a_blocker_is_not_a_dependency(self):
        """Entries preserve blocker records; only the entry saying it stalled counts."""
        self.write("a", entry("A", next_text="- Preserve the current XTX OOM blocker record."))
        self.assertEqual(self.rows_for("a"), [])

    def test_legacy_entry_with_a_next_subsection_is_a_row(self):
        changes = "### Result\n\nlanded the port\n\n### Next\n\n- Port the smoke kernel.\n"
        self.write("a", entry("A", worker="legacy", topic="2026-05-12-port",
                              changes=changes, next_text="None. Historical record."))
        rows = self.rows_for("a")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].evidence["open_by"], "legacy-next")

    def test_legacy_entry_with_the_port_boilerplate_is_not_a_row(self):
        self.write("a", entry("A", worker="legacy", topic="2026-05-12-port",
                              changes="### Result\n\nlanded\n",
                              next_text="- None. Historical record; the frozen legacy journal remains."))
        self.assertEqual(self.rows_for("a"), [])

    def test_a_heading_inside_a_fence_is_not_a_section(self):
        """Ported bodies wrap their own headings in a longer fence."""
        changes = "````markdown\n## Next\n\n- this is quoted content, not the entry's own\n````\n"
        self.write("a", entry("A", changes=changes, next_text="None."))
        self.assertEqual(self.rows_for("a"), [])

    def test_the_location_points_at_the_line_carrying_the_evidence(self):
        self.write("a", entry("A", status="blocked", next_text="Wait."))
        line = int(self.rows_for("a")[0].location.rsplit(":", 1)[1])
        text = (self.dir / "a.md").read_text(encoding="utf-8").splitlines()
        self.assertEqual(text[line - 1], "status: blocked")

    def test_a_legacy_location_points_at_its_next_heading(self):
        changes = "### Result\n\nlanded\n\n### Next\n\n- Port the smoke kernel.\n"
        self.write("a", entry("A", worker="legacy", topic="2026-05-12-port",
                              changes=changes, next_text="None."))
        line = int(self.rows_for("a")[0].location.rsplit(":", 1)[1])
        text = (self.dir / "a.md").read_text(encoding="utf-8").splitlines()
        self.assertEqual(text[line - 1], "### Next")

    def test_row_identity_is_stable_across_extractions(self):
        self.write("a", entry("A", status="checkpoint", next_text="Finish it."))
        first, second = self.rows_for("a")[0], self.rows_for("a")[0]
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.evidence_hash, second.evidence_hash)

    def test_every_row_carries_a_rebinding_anchor(self):
        self.write("a", entry("A", status="handoff", next_text="Pick it up."))
        anchor = self.rows_for("a")[0].hints["anchor"]
        self.assertTrue(anchor.endswith("a.md"), anchor)
        in_repo = worklog.REPO_ROOT / "worklog" / "entries" / "a.md"
        self.assertEqual(worklog._relative(in_repo), "worklog/entries/a.md",
                         "an anchor in the checkout must be repo-relative")

    def test_a_later_entry_on_the_same_topic_is_reported_as_continuation(self):
        self.write("early", entry("Early", status="checkpoint", topic="demo-topic",
                                  timestamp="2026-08-01T00:00:00.000000Z", next_text="Continue."))
        self.write("late", entry("Late", status="checkpoint", topic="demo-topic-followup",
                                 timestamp="2026-08-02T00:00:00.000000Z", next_text="Continue."))
        early, late = self.rows_for("early")[0], self.rows_for("late")[0]
        self.assertEqual(early.evidence["continuation_entries"], 1)
        self.assertEqual(late.evidence["continuation_entries"], 0)
        self.assertTrue(any("1 later entry" in s for s in early.signals))
        self.assertTrue(any("no later entry" in s for s in late.signals))

    def test_a_different_worker_does_not_count_as_continuation(self):
        self.write("early", entry("Early", status="checkpoint", worker="pi", next_text="Continue."))
        self.write("late", entry("Late", status="checkpoint", worker="lhl", topic="demo-topic-two",
                                 timestamp="2026-08-02T00:00:00.000000Z", next_text="Continue."))
        self.assertEqual(self.rows_for("early")[0].evidence["continuation_entries"], 0)

    def test_a_named_flag_nothing_reads_is_reported(self):
        self.write("a", entry("A", status="blocked", next_text="Unblock HIPENGINE_NOT_A_REAL_FLAG."))
        self.assertEqual(self.rows_for("a")[0].evidence["flags_unread"], ["HIPENGINE_NOT_A_REAL_FLAG"])

    def test_a_named_path_that_no_longer_exists_is_reported(self):
        self.write("a", entry("A", status="blocked",
                              next_text="Unblock `hipengine/gone/forever.py` once it lands."))
        self.assertEqual(self.rows_for("a")[0].evidence["paths_missing"], ["hipengine/gone/forever.py"])

    def test_a_bare_filename_is_not_read_as_a_repo_path(self):
        """`rmsnorm.hip` in prose names a kernel, not a location in this tree."""
        self.write("a", entry("A", status="blocked", next_text="Port `rmsnorm.hip` next."))
        self.assertEqual(self.rows_for("a")[0].evidence["paths_missing"], [])

    def test_an_entry_that_does_not_parse_is_counted_not_crashed(self):
        (self.dir / "broken.md").write_text("no front matter here\n", encoding="utf-8")
        self.write("a", entry("A", status="blocked", next_text="Wait."))
        rows, meta = self.extract()
        self.assertEqual(len(rows), 1)
        self.assertEqual(meta["unparsable"], 1)

    def test_meta_counts_what_the_rule_set_did(self):
        self.write("a", entry("A", status="blocked", next_text="Wait."))
        self.write("b", entry("B", next_text="- nothing open"))
        _, meta = self.extract()
        self.assertEqual(meta["entries"], 2)
        self.assertEqual(meta["rows"], 1)
        self.assertEqual(meta["by_rule"], {"status": 1})


if __name__ == "__main__":
    unittest.main()
