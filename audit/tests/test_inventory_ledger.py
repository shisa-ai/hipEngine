"""Contract tests for explicit cleanup conditions in ledger entries."""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hipaudit.inventory import ledger


class LedgerConditions(unittest.TestCase):
    def extract_signals(self, condition):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "REFACTOR.md"
            path.write_text(
                "## Example cleanup (2026-09-29)\n\n"
                "The wrapper has no callers. Its declaration and export are the "
                "remaining references and should be cleaned up as one unit.\n\n"
                + condition + "\n", encoding="utf-8",
            )
            with mock.patch.object(ledger, "LEDGER", path), mock.patch.object(
                ledger, "corpus", return_value={}
            ):
                rows, _ = ledger.extract()
        self.assertEqual(len(rows), 1)
        return rows[0].signals

    def test_explicit_condition_labels_are_recognized(self):
        for condition in (
            "Clearing condition: no callers remain and the unit tests pass.",
            "Removal condition: the next edit to the dense branch.",
            "CLEARING CONDITION: validate the replacement route.",
        ):
            with self.subTest(condition=condition):
                self.assertNotIn("states no removal condition", self.extract_signals(condition))

    def test_missing_or_empty_condition_is_still_reported(self):
        for condition in ("", "Removal condition:", "Clearing condition:   ",
                          "A removal condition should be added later."):
            with self.subTest(condition=condition):
                self.assertIn("states no removal condition", self.extract_signals(condition))

    def test_existing_condition_wording_remains_recognized(self):
        self.assertNotIn("states no removal condition", self.extract_signals(
            "Remove it when the replacement has no fallback callers."
        ))
