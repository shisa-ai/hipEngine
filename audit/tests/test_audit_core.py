"""Contract tests for the audit core.

The property that makes the tool usable is durability: re-extracting an
inventory must never destroy triage, and a decision made against evidence that
has since changed must come back for review rather than silently standing.
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hipaudit import core  # noqa: E402


def row(key="X", **evidence):
    return core.Row(kind="flag", key=key, title=key, location="a.py:1",
                    evidence=evidence or {"default": "off"}, signals=["s"])


def ledger_row(key, title, refs, anchor="docs/REFACTOR.md"):
    from hipaudit.inventory import tokens
    return core.Row(kind="ledger", key=key, title=title, location=f"{anchor}:1",
                    evidence={}, signals=[],
                    hints={"anchor": anchor, "refs": sorted(refs), "tokens": tokens(title)})


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


class Rebinding(unittest.TestCase):
    """A rescan must land on the same audit item after ordinary editing."""

    def test_reworded_heading_keeps_its_decision(self):
        before = ledger_row("a-1111", "Packed workspace lease reserves a full session context",
                            ["hipengine/runtime/pool.py", "HIPENGINE_POOL_LEASE"])
        decisions = {before.id: decide(before, tag="STALE-LEDGER", disposition="remove")}
        decisions[before.id].hints = before.hints
        #  Same entry, reworded and re-hashed by the extractor.
        after = ledger_row("a-2222", "The packed workspace lease still reserves one full session context",
                           ["hipengine/runtime/pool.py", "HIPENGINE_POOL_LEASE"])
        moved = core.rebind([after], decisions)
        self.assertEqual([(m[0], m[1]) for m in moved], [(before.id, after.id)])
        self.assertIn(after.id, decisions)
        self.assertEqual(decisions[after.id].rebound_from, before.id)
        self.assertEqual(decisions[after.id].note, "because")

    def test_unrelated_row_is_not_claimed(self):
        before = ledger_row("a-1111", "Packed workspace lease reserves a session context",
                            ["hipengine/runtime/pool.py"])
        decisions = {before.id: decide(before)}
        decisions[before.id].hints = before.hints
        other = ledger_row("b-3333", "Prefix snapshot eviction destroys the retained entry",
                           ["hipengine/runtime/prefix.py"])
        self.assertEqual(core.rebind([other], decisions), [])
        self.assertNotIn(other.id, decisions)

    def test_a_row_that_already_has_a_decision_is_never_stolen(self):
        old = ledger_row("a-1111", "Workspace lease reserves a full session context", ["p.py"])
        twin = ledger_row("a-2222", "Workspace lease reserves a full session context", ["p.py"])
        decisions = {old.id: decide(old, note="first"), twin.id: decide(twin, note="second")}
        decisions[old.id].hints, decisions[twin.id].hints = old.hints, twin.hints
        core.rebind([twin], decisions)
        self.assertEqual(decisions[twin.id].note, "second")

    def test_a_decision_without_hints_is_not_rebound(self):
        before = ledger_row("a-1111", "Some entry", ["p.py"])
        decisions = {before.id: decide(before)}          # hints left empty
        after = ledger_row("a-2222", "Some entry", ["p.py"])
        self.assertEqual(core.rebind([after], decisions), [])

    def test_a_finding_that_moved_down_the_file_keeps_its_decision(self):
        """A ref is ``path:line``, and inserting a line above moves it.

        The referent is the same code, so a line shift must not un-triage every
        finding beneath the insertion and fail the budget gate.
        """

        def finding(line, message):
            return core.Row(
                kind="stub",
                key=f"api.py:{line}",
                title=message,
                location=f"hipengine/server/api.py:{line}",
                evidence={"path": "hipengine/server/api.py", "line": line},
                signals=["NotImplementedError in runtime code"],
                hints={
                    "anchor": "hipengine/server/api.py",
                    "refs": [f"hipengine/server/api.py:{line}"],
                    "tokens": sorted(message.lower().split()),
                },
            )

        before = finding(100, "unimplemented: raise NotImplementedError(")
        decisions = {before.id: decide(before)}
        decisions[before.id].hints = before.hints
        after = finding(128, "unimplemented: raise NotImplementedError(")

        moved = core.rebind([after], decisions)

        self.assertEqual([(m[0], m[1]) for m in moved], [(before.id, after.id)])
        self.assertEqual(decisions[after.id].rebound_from, before.id)

    def test_a_different_finding_in_the_same_file_is_not_claimed(self):
        def finding(line, message):
            return core.Row(
                kind="stub",
                key=f"api.py:{line}",
                title=message,
                location=f"hipengine/server/api.py:{line}",
                evidence={},
                signals=[],
                hints={
                    "anchor": "hipengine/server/api.py",
                    "refs": [f"hipengine/server/api.py:{line}"],
                    "tokens": sorted(message.lower().split()),
                },
            )

        before = finding(
            100, 'unimplemented: raise NotImplementedError("speculative MTP serving")'
        )
        decisions = {before.id: decide(before)}
        decisions[before.id].hints = before.hints
        other = finding(128, "unimplemented: raise NotImplementedError(\"vision tower missing\")")

        self.assertEqual(core.rebind([other], decisions), [])


class Expiry(unittest.TestCase):
    def test_a_decision_past_its_date_is_reported(self):
        target = row()
        decisions = {target.id: decide(target, disposition="defer", expires="2020-01-01")}
        self.assertEqual([d.id for d in core.expired(decisions, today="2026-09-21")], [target.id])

    def test_an_unexpired_decision_is_not_reported(self):
        target = row()
        decisions = {target.id: decide(target, disposition="defer", expires="2099-01-01")}
        self.assertEqual(core.expired(decisions, today="2026-09-21"), [])

    def test_reconcile_separates_expired_from_triaged(self):
        target = row()
        decisions = {target.id: decide(target, expires="2020-01-01")}
        state = core.reconcile([target], decisions)
        self.assertEqual(len(state["expired"]), 1)
        self.assertEqual(state["triaged"], [])

    def test_wontfix_is_a_valid_disposition(self):
        self.assertEqual(decide(row(), disposition="wontfix").problems(), [])


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


class Checks(unittest.TestCase):
    """A finding must name the edit that closes it, or it is not actionable."""

    def test_every_check_is_registered_with_a_callable(self):
        from hipaudit.checks import CHECKS
        self.assertTrue(CHECKS)
        for name, fn in CHECKS.items():
            self.assertTrue(callable(fn), name)

    def test_a_finding_carries_a_fix_and_a_reason(self):
        from hipaudit.checks import finding
        made = finding("demo", "k", "title", "a.py:1", fix="do this", why="because")
        self.assertEqual(made.evidence["fix"], "do this")
        self.assertEqual(made.signals, ["because"])
        self.assertTrue(made.hints["anchor"], "a finding must be re-matchable across rescans")

    def test_findings_and_inventory_share_one_triage_store(self):
        from hipaudit.checks import finding
        row = finding("demo", "k", "title", "a.py:1", fix="f", why="w")
        decisions = {row.id: core.Triage(id=row.id, tag="NOT-DEBT", disposition="wontfix",
                                         note="deliberate", evidence_hash=row.evidence_hash)}
        state = core.reconcile([row], decisions)
        self.assertEqual(len(state["triaged"]), 1,
                         "a wontfix recorded on a finding must suppress it like any other row")


class Budget(unittest.TestCase):
    """The budget ratchets down. An automatic refresh must never absorb new debt."""

    def setUp(self):
        import tempfile
        from hipaudit import report
        self._dir = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(report, "BUDGET",
                                        pathlib.Path(self._dir.name) / "budget.json")
        self._patch.start()
        self.report = report

    def tearDown(self):
        self._patch.stop()
        self._dir.cleanup()

    def test_lower_only_refuses_to_raise_a_ceiling(self):
        one = row("A")
        self.report.save_budget([one], {})                       # ceiling: 1 open flag
        self.assertEqual(self.report.load_budget()["open"]["flag"], 1)
        grown = [one, row("B"), row("C")]                        # two new untriaged rows
        self.report.save_budget(grown, {}, lower_only=True)
        self.assertEqual(self.report.load_budget()["open"]["flag"], 1,
                         "a refresh must not absorb new debt into the budget")

    def test_lower_only_still_ratchets_down_as_rows_are_triaged(self):
        rows = [row("A"), row("B")]
        self.report.save_budget(rows, {})
        self.assertEqual(self.report.load_budget()["open"]["flag"], 2)
        decided = {rows[0].id: decide(rows[0])}
        self.report.save_budget(rows, decided, lower_only=True)
        self.assertEqual(self.report.load_budget()["open"]["flag"], 1,
                         "cleanup must show up as the ceiling dropping")

    def test_an_explicit_budget_call_may_raise(self):
        self.report.save_budget([row("A")], {})
        self.report.save_budget([row("A"), row("B")], {})        # deliberate, not lower_only
        self.assertEqual(self.report.load_budget()["open"]["flag"], 2)

    def test_a_fully_triaged_kind_keeps_its_zero_ceiling(self):
        rows = [row("A"), row("B")]
        self.report.save_budget(rows, {})
        decided = {r.id: decide(r) for r in rows}
        self.report.save_budget(rows, decided, lower_only=True)
        self.assertEqual(
            self.report.load_budget()["open"].get("flag"), 0,
            "a fully triaged kind must keep a zero ceiling, or its next row is ungated")
        self.report.save_budget([row("C")], {}, lower_only=True)
        self.assertEqual(self.report.load_budget()["open"]["flag"], 0,
                         "a new untriaged row must not slip in under a dropped ceiling")


class GateSelect(unittest.TestCase):
    """A kind whose population grows with normal work is gated on the rows that
    need action. The rest stay untriaged and visible, but do not set the ceiling."""

    def setUp(self):
        import tempfile
        from hipaudit import report
        self._dir = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(report, "BUDGET",
                                        pathlib.Path(self._dir.name) / "budget.json")
        self._patch.start()
        self.report = report

    def tearDown(self):
        self._patch.stop()
        self._dir.cleanup()

    def entry(self, status, key=None):
        key = key or f"w-{status}"
        return core.Row(kind="worklog", key=key, title=key,
                        location=f"worklog/entries/{key}.md:3",
                        evidence={"status": status, "open_by": "status"}, signals=[])

    def policy(self, select):
        self.report.BUDGET.write_text(json.dumps({"open": {}, "select": select}),
                                      encoding="utf-8")

    def test_a_kind_gate_counts_only_the_rows_that_need_action(self):
        rows = [self.entry("blocked"), self.entry("checkpoint"), self.entry("handoff")]
        self.policy({"worklog": {"evidence.status": ["blocked", "handoff"]}})
        gated, ungated = self.report.open_counts(rows, {})
        self.assertEqual(gated["worklog"], 2)
        self.assertEqual(ungated["worklog"], 1)

    def test_an_ungated_kind_counts_every_open_row(self):
        rows = [row("A"), row("B")]
        self.policy({"worklog": {"evidence.status": ["blocked"]}})
        gated, ungated = self.report.open_counts(rows, {})
        self.assertEqual(gated["flag"], 2, "a kind with no selector keeps the whole-population gate")
        self.assertEqual(sum(ungated.values()), 0)

    def test_rows_outside_the_gate_stay_open_and_visible(self):
        rows = [self.entry("blocked"), self.entry("checkpoint")]
        self.policy({"worklog": {"evidence.status": ["blocked", "handoff"]}})
        gated, _ = self.report.open_counts(rows, {})
        self.assertEqual(gated["worklog"], 1)
        self.assertIn("| 2 |", self.report.state_table(rows, {}),
                      "an ungated row is still untriaged, and the state table must say so")

    def test_a_list_of_specs_is_ored(self):
        rows = [self.entry("blocked", "a"), self.entry("checkpoint", "b"),
                core.Row(kind="worklog", key="c", title="c", location="w.md:1",
                         evidence={"status": "completed", "open_by": "dependency"}, signals=[])]
        self.policy({"worklog": [{"evidence.status": ["blocked", "handoff"]},
                                  {"evidence.open_by": "dependency"}]})
        gated, ungated = self.report.open_counts(rows, {})
        self.assertEqual(gated["worklog"], 2)
        self.assertEqual(ungated["worklog"], 1)

    def test_lower_only_records_the_gated_count(self):
        rows = [self.entry("blocked"), self.entry("checkpoint"), self.entry("handoff")]
        self.policy({"worklog": {"evidence.status": ["blocked", "handoff"]}})
        self.report.BUDGET.write_text(
            json.dumps({"open": {"worklog": 40},
                        "select": {"worklog": {"evidence.status": ["blocked", "handoff"]}}}),
            encoding="utf-8")
        self.report.save_budget(rows, {}, lower_only=True)
        self.assertEqual(self.report.load_budget()["open"]["worklog"], 2,
                         "the ceiling is the gated population, ratcheted down")

    def test_save_budget_keeps_the_select_policy(self):
        rows = [self.entry("blocked"), self.entry("checkpoint")]
        self.policy({"worklog": {"evidence.status": ["blocked", "handoff"]}})
        self.report.save_budget(rows, {})
        self.assertEqual(self.report.load_budget()["select"],
                         {"worklog": {"evidence.status": ["blocked", "handoff"]}},
                         "a policy is configuration: recording a ceiling must not erase it")
