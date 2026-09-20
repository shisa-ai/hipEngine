"""Contract tests for the audit core.

The property that makes the tool usable is durability: re-extracting an
inventory must never destroy triage, and a decision made against evidence that
has since changed must come back for review rather than silently standing.
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hipaudit import core  # noqa: E402


def row(key="X", **evidence):
    return core.Row(kind="flag", key=key, title=key, location="a.py:1",
                    evidence=evidence or {"default": "off"}, signals=["s"])


def decide(target, **kw):
    fields = dict(id=target.id, tag="DEAD-FLAG", disposition="remove",
                  severity="low", note="because", evidence_hash=target.evidence_hash)
    fields.update(kw)
    return core.Triage(**fields)


class RowIdentity(unittest.TestCase):
    def test_id_is_stable_across_extractions(self):
        self.assertEqual(row().id, row().id)

    def test_evidence_hash_tracks_evidence_not_order(self):
        a = core.Row(kind="flag", key="X", title="X", evidence={"a": 1, "b": 2}, signals=["p", "q"])
        b = core.Row(kind="flag", key="X", title="X", evidence={"b": 2, "a": 1}, signals=["q", "p"])
        self.assertEqual(a.evidence_hash, b.evidence_hash)

    def test_evidence_hash_changes_when_evidence_changes(self):
        self.assertNotEqual(row(default="off").evidence_hash, row(default="on").evidence_hash)


class Reconciliation(unittest.TestCase):
    def test_untriaged_row_is_open(self):
        state = core.reconcile([row()], {})
        self.assertEqual(len(state["open"]), 1)

    def test_triage_survives_reextraction_when_evidence_is_unchanged(self):
        first = row()
        decisions = {first.id: decide(first)}
        again = row()                                  # a later extraction of the same thing
        state = core.reconcile([again], decisions)
        self.assertEqual(len(state["triaged"]), 1)
        self.assertEqual(state["stale"], [])

    def test_changed_evidence_makes_a_decision_stale(self):
        before = row(default="off")
        decisions = {before.id: decide(before)}
        after = row(default="on")                      # the flag was flipped since triage
        state = core.reconcile([after], decisions)
        self.assertEqual(len(state["stale"]), 1,
                         "a decision made against different facts must come back for review")
        self.assertEqual(state["triaged"], [])

    def test_decision_whose_row_vanished_is_reported(self):
        gone = row("GONE")
        lost = core.orphaned([row("STILL_HERE")], {gone.id: decide(gone)})
        self.assertEqual([d.id for d in lost], [gone.id])

    def test_resolved_decisions_are_not_reported_as_orphaned(self):
        gone = row("GONE")
        lost = core.orphaned([], {gone.id: decide(gone, resolved=True)})
        self.assertEqual(lost, [])


class TriageValidation(unittest.TestCase):
    def test_unknown_tag_and_disposition_are_rejected(self):
        problems = core.Triage(id="flag/X", tag="NONSENSE", disposition="ponder",
                               severity="critical", note="x").problems()
        self.assertEqual(len(problems), 3, problems)

    def test_empty_note_is_rejected(self):
        problems = core.Triage(id="flag/X", tag="DEAD-FLAG", disposition="remove", note="  ").problems()
        self.assertTrue(any("note" in p for p in problems), problems)

    def test_a_complete_decision_validates(self):
        self.assertEqual(decide(row()).problems(), [])


class Store(unittest.TestCase):
    def test_round_trip_preserves_every_field(self):
        with mock.patch.object(core, "TRIAGE_DIR", pathlib.Path(self.tmp)):
            original = decide(row(), note="keep this text", by="tester")
            core.save_triage({original.id: original})
            loaded = core.load_triage()[original.id]
        self.assertEqual(loaded, original)

    def setUp(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = self._dir.name

    def tearDown(self):
        self._dir.cleanup()


if __name__ == "__main__":
    unittest.main()
