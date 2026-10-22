"""Audit maintenance commands distinguish fresh data from reviewed decisions."""
import argparse
import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hipaudit import core, report

spec = importlib.util.spec_from_file_location("audit_cli", pathlib.Path(__file__).resolve().parents[1] / "audit.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class Maintenance(unittest.TestCase):
    def test_expiry_has_its_own_table_column(self):
        row = core.Row("flag", "X", "X")
        d = core.Triage(row.id, "NOT-DEBT", "keep", note="verified", expires="2020-01-01")
        table = report.state_table([row], {row.id: d})
        self.assertIn("| Expired |", table)
        self.assertIn("| flag | 1 | 0 | 0 | 0 | 1 | 0 |", table)

    def test_review_gate_fails_without_changing_regular_budget(self):
        row = core.Row("flag", "X", "X")
        d = core.Triage(row.id, "NOT-DEBT", "keep", note="verified", expires="2020-01-01")
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(core, "INVENTORY_DIR", pathlib.Path(tmp)), \
             mock.patch.object(core, "FINDINGS_DIR", pathlib.Path(tmp)), \
             mock.patch.object(core, "load_all", return_value=[row]), \
             mock.patch.object(core, "load_triage", return_value={row.id: d}), \
             mock.patch.object(cli, "run_inventory"), mock.patch.object(cli, "run_scan"), \
             mock.patch.object(report, "load_budget", return_value={"open": {"flag": 0}}), \
             mock.patch("builtins.print"):
            self.assertEqual(cli.cmd_check(argparse.Namespace(refresh=False, review=False)), 0)
            self.assertEqual(cli.cmd_check(argparse.Namespace(refresh=False, review=True)), 1)

    def test_archive_listing_does_not_scan_or_mutate(self):
        with mock.patch.object(core, "load_archive", return_value=[]), \
             mock.patch.object(cli, "run_inventory") as inventory, \
             mock.patch.object(core, "archive_orphans") as archive, mock.patch("builtins.print"):
            self.assertEqual(cli.cmd_archive(argparse.Namespace(apply=False, number=10)), 0)
            inventory.assert_not_called()
            archive.assert_not_called()

    def test_archive_apply_scans_both_stores_before_archiving(self):
        calls = []
        with mock.patch.object(cli, "run_inventory", side_effect=lambda _: calls.append("inventory")), \
             mock.patch.object(cli, "run_scan", side_effect=lambda _: calls.append("scan")), \
             mock.patch.object(core, "load_all", return_value=[]), \
             mock.patch.object(core, "load_triage", return_value={}), \
             mock.patch.object(core, "archive_orphans", side_effect=lambda *_: calls.append("archive") or []), \
             mock.patch.object(core, "save_triage") as save, \
             mock.patch.object(core, "load_archive", return_value=[]), mock.patch("builtins.print"):
            cli.cmd_archive(argparse.Namespace(apply=True, number=10))
            self.assertEqual(calls, ["inventory", "scan", "archive"])
            save.assert_called_once_with({})

    def confirm(self, decision, row, **kw):
        args = argparse.Namespace(ids=[row.id], note="still holds", expires=None)
        vars(args).update(kw)
        with mock.patch.object(core, "load_all", return_value=[row]), \
             mock.patch.object(core, "load_triage", return_value={row.id: decision}), \
             mock.patch.object(core, "save_triage") as save, \
             mock.patch("builtins.print"), mock.patch("sys.stderr"):
            return cli.cmd_confirm(args), save

    def test_confirm_rebases_a_stale_decision_and_keeps_the_verdict(self):
        row = core.Row(kind="flag", key="X", title="X", evidence={"default": "on"}, signals=[])
        decision = core.Triage(id=row.id, tag="NOT-DEBT", disposition="keep", note="lever",
                               evidence_hash="old", basis={"title": "X"})
        status, save = self.confirm(decision, row)
        self.assertEqual(status, 0)
        save.assert_called_once()
        self.assertEqual(decision.evidence_hash, row.evidence_hash)
        self.assertEqual(decision.basis, core.compact_basis(row.basis()))
        self.assertEqual((decision.tag, decision.disposition), ("NOT-DEBT", "keep"))
        self.assertTrue(decision.note.startswith("lever [") and decision.note.endswith("still holds]"))

    def test_confirm_refuses_to_keep_a_past_review_date(self):
        row = core.Row(kind="flag", key="X", title="X", evidence={}, signals=[])
        decision = core.Triage(id=row.id, tag="NOT-DEBT", disposition="defer", note="later",
                               expires="2000-01-01")
        status, save = self.confirm(decision, row)
        self.assertEqual(status, 1)
        save.assert_not_called()
        status, save = self.confirm(decision, row, expires="2999-01-01")
        self.assertEqual(status, 0)
        self.assertEqual(decision.expires, "2999-01-01")


if __name__ == "__main__":
    unittest.main()
