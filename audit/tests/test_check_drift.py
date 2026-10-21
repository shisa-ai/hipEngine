"""Contract tests for the doc-path-drift check."""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hipaudit.checks import drift


DOC = """# Catalog

Paths in each section are relative to its backend directory.

## HIP

Device sources: `pkg/kernels/hip/`.

### Attention

| `attention/decode.hip` | decode |

## Inventory

| `attention/prefill.hip` | listed without a root |
"""


class SectionRoots(unittest.TestCase):
    def scan(self, text, doc="docs/CATALOG.md"):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "pkg/kernels/hip/attention").mkdir(parents=True)
            for name in ("decode.hip", "prefill.hip"):
                (root / "pkg/kernels/hip/attention" / name).write_text("")
            drift.tree_index.cache_clear()
            try:
                with mock.patch.object(drift, "REPO_ROOT", root), \
                        mock.patch.object(drift, "doc_corpus", lambda: {doc: text}):
                    rows, _ = drift.doc_path_drift()
            finally:
                drift.tree_index.cache_clear()
        return [row.location for row in rows]

    def test_a_declared_root_covers_its_subsections(self):
        self.assertEqual(drift.declared_roots(DOC.splitlines())[10], ("pkg/kernels/hip/",))

    def test_a_section_relative_path_under_its_root_is_not_drift(self):
        located = self.scan(DOC)
        self.assertNotIn("docs/CATALOG.md:11", located)

    def test_a_table_header_and_source_line_declare_roots(self):
        lines = ["| Target file (`pkg/kernels/hip/...`) | Kernels |",
                 "Source: `pkg/kernels/cpu/`. NumPy references.",
                 "A note (`pkg/`) in passing is not a declaration."]
        self.assertEqual(drift.declared_roots(lines)[-1], ("pkg/kernels/hip/", "pkg/kernels/cpu/"))

    def test_a_path_relative_to_its_own_document_is_not_drift(self):
        self.assertEqual(self.scan("See `hip/attention/decode.hip`.\n", doc="pkg/kernels/INDEX.md"), [])

    def test_a_closed_record_is_exempt(self):
        record = "---\nstatus: closed\nowns: x\n---\n" + DOC
        self.assertEqual(self.scan(record), [])

    def test_the_root_ends_at_the_next_section(self):
        self.assertEqual(self.scan(DOC), ["docs/CATALOG.md:15"])


if __name__ == "__main__":
    unittest.main()
