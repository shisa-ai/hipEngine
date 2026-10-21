"""Contract tests for finding identity: parsed statements, never a line number.

A finding's key is what carries its triage decision across rescans. Keying by
`path:line` detached every decision as soon as anything above it moved. These
tests pin what replaces it: the statement as Python parses it, the `def`/`class`
it lives in, and - where two identical statements share a scope - the blocks
around them and the group they form, so that adding, removing or re-contexting a
duplicate reopens the group instead of sliding a decision onto a neighbouring
site.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hipaudit import core  # noqa: E402
from hipaudit.checks import (  # noqa: E402
    SourceIndex, drift, invariants, skeletons,
)

STUB = 'raise NotImplementedError("the fused path needs a compiled kernel")\n'
MARKED = "def f():\n    # TODO: remove\n    return 1\n"
MARKED_MOVED = "import os\n\n\n" + MARKED
BRANCH = 'def pick(backend):\n    if backend == "hip":\n        return 1\n'
DUPLICATE = ('class A:\n'
             '    def first(self):\n'
             '        raise NotImplementedError("not supported here")\n'
             '\n\nclass B:\n'
             '    def second(self):\n'
             '        raise NotImplementedError("not supported here")\n')
ZERO_THEN_DUPLICATE = ('class Zero:\n'
                       '    def zeroth(self):\n'
                       '        raise NotImplementedError("not supported here")\n'
                       '\n\n' + DUPLICATE)
FIRST_TRIMMED = ('class A:\n'
                 '    def first(self):\n'
                 '        return 1\n'
                 '\n\nclass B:\n'
                 '    def second(self):\n'
                 '        raise NotImplementedError("not supported here")\n')
SAME = 'raise NotImplementedError("same")\n'


def long_message(tail: str, reasons: int = 30) -> str:
    """A stub whose message wraps far past any line cap, ending in `tail`."""
    body = "\n".join(f'        "reason {index}; "' for index in range(1, reasons))
    return ('def build(self, spec):\n'
            '    raise NotImplementedError(\n'
            f'{body}\n'
            f'        "{tail}")\n')


def stubs(files):
    with mock.patch.object(skeletons, "corpus", lambda: dict(files)):
        rows, _ = skeletons.stub()
    return rows


def markers(files):
    with mock.patch.object(skeletons, "corpus", lambda: dict(files)):
        rows, _ = skeletons.marker()
    return rows


def axis_branches(files):
    with mock.patch.object(invariants, "corpus", lambda: dict(files)):
        rows, _ = invariants.axis_branch()
    return rows


def decide(row, **overrides):
    fields = dict(id=row.id, tag="SKELETON", disposition="defer", severity="low",
                  note="planned work", evidence_hash=row.evidence_hash, hints=row.hints)
    fields.update(overrides)
    return core.Triage(**fields)


def ids(rows):
    return sorted(row.id for row in rows)


def keys(rows):
    return sorted(row.key for row in rows)


def by_scope(rows, needle):
    """The one finding whose key names `needle`, or None."""
    found = [row for row in rows if f":{needle}-" in row.key]
    return found[0] if len(found) == 1 else None


def decisions_for(rows):
    return {row.id: decide(row) for row in rows}


class StatementIdentity(unittest.TestCase):
    """A statement is the whole statement as Python parses it."""

    def test_a_message_past_the_old_line_cap_is_read_whole(self):
        lines = long_message("kernel lands").splitlines()
        statement = SourceIndex("\n".join(lines)).statement(2)
        self.assertIn("reason 1;", statement)
        self.assertIn("reason 29;", statement)
        self.assertIn("kernel lands", statement)

    def test_a_message_changed_past_the_old_line_cap_changes_the_key(self):
        before = stubs({"hipengine/x.py": long_message("kernel lands")})
        after = stubs({"hipengine/x.py": long_message("kernel lands, see the ledger")})
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)
        self.assertNotEqual(before[0].key, after[0].key)

    def test_messages_differing_only_past_the_old_line_cap_do_not_collide(self):
        rows = stubs({"hipengine/x.py":
                      long_message("kernel lands") + "\n\n" + long_message("kernel lands, later")})
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(set(keys(rows))), 2)
        self.assertNotIn("@", "".join(keys(rows)))     # two sites, not one group

    def test_reflowing_a_message_does_not_change_its_key(self):
        wrapped = ('def build(self):\n'
                   '    raise NotImplementedError(\n'
                   '        "one reason; "\n'
                   '        "another reason")\n')
        single = ('def build(self):\n'
                  '    raise NotImplementedError("one reason; another reason")\n')
        self.assertEqual(keys(stubs({"hipengine/x.py": wrapped})),
                         keys(stubs({"hipengine/x.py": single})))

    def test_a_bracket_inside_a_message_does_not_continue_the_statement(self):
        lines = ('def f(self):\n'
                 '    raise NotImplementedError("bad ( value")\n'
                 '    raise NotImplementedError("second")\n').splitlines()
        self.assertEqual(SourceIndex("\n".join(lines)).statement(2),
                         "raise NotImplementedError('bad ( value')")

    def test_a_match_inside_a_string_falls_back_to_its_own_line(self):
        source = ('def f(self):\n'
                  '    template = """\n'
                  '    raise NotImplementedError("generated")\n'
                  '"""\n')
        rows = stubs({"hipengine/x.py": source})
        self.assertEqual(len(rows), 1)
        self.assertEqual(SourceIndex(source).statement(3), "")
        self.assertEqual(rows[0].key, stubs({"hipengine/x.py": source})[0].key)

    def test_a_file_that_does_not_parse_still_yields_keys(self):
        broken = 'def f(self):\n    raise NotImplementedError("m"\n'   # unclosed paren
        rows = stubs({"hipengine/x.py": broken})
        self.assertEqual(len(rows), 1)
        self.assertEqual(keys(rows), keys(stubs({"hipengine/x.py": broken})))


class Scopes(unittest.TestCase):
    """The def/class chain a line sits in, from the parse tree."""

    SOURCE = '''"""Docstring quoting def fake(): and an ( bracket."""

import json


@register("thing")
class KV:
    def __init__(self, a):
        self.a = a

    async def load(
        self,
        path,
    ):
        # a comment at column zero does not end the body
        return path


def loose():
    return 1
'''

    def index(self):
        return SourceIndex(self.SOURCE)

    def test_nested_defs_and_classes_chain(self):
        index = self.index()
        self.assertEqual(index.scope(8), "KV")
        self.assertEqual(index.scope(9), "KV.__init__")
        self.assertEqual(index.scope(14), "KV.load")
        self.assertEqual(index.scope(19), "")
        self.assertEqual(index.scope(20), "loose")

    def test_a_docstring_quoting_a_def_is_not_an_anchor(self):
        self.assertEqual(self.index().scope(3), "")

    def test_a_multiline_signature_still_opens_its_body(self):
        self.assertEqual(self.index().scope(13), "KV.load")

    def test_a_decorator_belongs_to_the_enclosing_scope(self):
        self.assertEqual(self.index().scope(6), "")
        self.assertEqual(self.index().scope(7), "")

    def test_a_sibling_def_closes_the_previous_one(self):
        lines = ("class A:\n"
                 "    def one(self):\n"
                 "        return 1\n"
                 "\n"
                 "    def two(self):\n"
                 "        return 2\n").splitlines()
        self.assertEqual(SourceIndex("\n".join(lines)).scope(6), "A.two")


class Blocks(unittest.TestCase):
    """The block chain inside a scope, which tells identical statements apart."""

    TWO_BLOCKS = ('def f(self):\n'
                  '    if self.flag:\n'
                  '        raise NotImplementedError("same")\n'
                  '    if self.other:\n'
                  '        raise NotImplementedError("same")\n')

    def test_indistinguishable_blocks_reopen_on_semantic_scope_changes(self):
        source = ('def f(value):\n'
                  '    if value:\n'
                  '        raise NotImplementedError("same")\n'
                  '    else:\n'
                  '        prepare()\n'
                  '        raise NotImplementedError("same")\n')
        before = stubs({"hipengine/x.py": source})
        after = stubs({"hipengine/x.py": source.replace("prepare()", "other()")})
        self.assertTrue({r.id for r in before}.isdisjoint(r.id for r in after))
        unchanged = stubs({"hipengine/x.py": source.replace("prepare()", "# comment\n        prepare()")})
        self.assertEqual([r.id for r in before], [r.id for r in unchanged])

    def test_match_cases_have_locations_and_guard_identity(self):
        source = ('def f(value):\n'
                  '    match value:\n'
                  '        case 1 if ready:\n'
                  '            raise NotImplementedError("same")\n'
                  '        case _:\n'
                  '            raise NotImplementedError("same")\n')
        before = stubs({"hipengine/x.py": source})
        after = stubs({"hipengine/x.py": source.replace("if ready", "if enabled")})
        self.assertEqual(len(before), 2)
        self.assertTrue({r.id for r in before}.isdisjoint(r.id for r in after))

    def test_the_block_chain_names_the_headers(self):
        index = SourceIndex(self.TWO_BLOCKS)
        self.assertEqual(index.block(3), "If self.flag")
        self.assertEqual(index.block(5), "If self.other")

    def test_duplicates_in_different_blocks_are_distinct_sites(self):
        rows = stubs({"hipengine/x.py": self.TWO_BLOCKS})
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(set(keys(rows))), 2)
        self.assertEqual(len({row.key.split("@")[-1] for row in rows}), 2)

    def test_blank_lines_and_comments_do_not_change_a_group(self):
        noisy = ('def f(self):\n'
                 '\n'
                 '    # a note nobody needs\n'
                 '    if self.flag:\n'
                 '\n'
                 '        # another\n'
                 '        raise NotImplementedError("same")\n'
                 '\n'
                 '    if self.other:  # trailing\n'
                 '        raise NotImplementedError("same")\n')
        self.assertEqual(keys(stubs({"hipengine/x.py": self.TWO_BLOCKS})),
                         keys(stubs({"hipengine/x.py": noisy})))

    def test_editing_a_block_header_reopens_the_group(self):
        edited = self.TWO_BLOCKS.replace("if self.flag:", "if self.mode == 'x':")
        before, after = stubs({"hipengine/x.py": self.TWO_BLOCKS}), \
            stubs({"hipengine/x.py": edited})
        self.assertFalse(set(ids(before)) & set(ids(after)))
        decisions = decisions_for(before)
        self.assertEqual(core.rebind(after, decisions), [])
        self.assertEqual(core.reconcile(after, decisions)["open"], after)

    def test_swapping_two_duplicates_in_different_blocks_keeps_their_ids(self):
        swapped = ('def f(self):\n'
                   '    if self.other:\n'
                   '        raise NotImplementedError("same")\n'
                   '    if self.flag:\n'
                   '        raise NotImplementedError("same")\n')
        self.assertEqual(ids(stubs({"hipengine/x.py": self.TWO_BLOCKS})),
                         ids(stubs({"hipengine/x.py": swapped})))


class DuplicateSafety(unittest.TestCase):
    """Adding or removing a duplicate must never move a decision to another site."""

    ONE = 'def f(self):\n    raise NotImplementedError("same")\n'
    TWO_SAME_BLOCK = ('def f(self):\n'
                      '    raise NotImplementedError("same")\n'
                      '    raise NotImplementedError("same")\n')

    def test_inserting_a_duplicate_in_the_same_scope_reopens_the_group(self):
        before = stubs({"hipengine/x.py": self.ONE})
        after = stubs({"hipengine/x.py": self.TWO_SAME_BLOCK})
        decisions = decisions_for(before)
        self.assertFalse(set(ids(before)) & set(ids(after)))
        self.assertEqual(core.rebind(after, decisions), [])
        self.assertEqual(core.reconcile(after, decisions)["open"], after)

    def test_inserting_a_duplicate_above_the_original_does_not_hand_it_the_decision(self):
        below = ('def f(self):\n'
                 '    if self.flag:\n'
                 '        raise NotImplementedError("same")\n')
        above = ('def f(self):\n'
                 '    raise NotImplementedError("same")\n'
                 '    if self.flag:\n'
                 '        raise NotImplementedError("same")\n')
        before = stubs({"hipengine/x.py": below})
        after = stubs({"hipengine/x.py": above})
        decisions = decisions_for(before)
        self.assertFalse(set(ids(before)) & set(ids(after)))
        self.assertEqual(core.rebind(after, decisions), [])
        for row in after:
            self.assertNotIn(row.id, decisions)

    def test_removing_the_first_duplicate_reopens_the_group(self):
        before = stubs({"hipengine/x.py": self.TWO_SAME_BLOCK})
        after = stubs({"hipengine/x.py":
                       'def f(self):\n    raise NotImplementedError("same")\n'})
        decisions = decisions_for(before)
        self.assertFalse(set(ids(before)) & set(ids(after)))
        self.assertEqual(core.rebind(after, decisions), [])
        self.assertEqual(core.reconcile(after, decisions)["open"], after)

    def test_removing_a_member_of_a_group_of_three_reopens_the_rest(self):
        three = ('def f(self):\n'
                 '    if self.a:\n'
                 '        raise NotImplementedError("same")\n'
                 '    if self.b:\n'
                 '        raise NotImplementedError("same")\n'
                 '    if self.c:\n'
                 '        raise NotImplementedError("same")\n')
        two = three.replace('    if self.c:\n        raise NotImplementedError("same")\n', "")
        before, after = stubs({"hipengine/x.py": three}), stubs({"hipengine/x.py": two})
        self.assertEqual(len(before), 3)
        self.assertFalse(set(ids(before)) & set(ids(after)))

    def test_a_duplicate_in_another_function_does_not_reopen_this_one(self):
        before = stubs({"hipengine/x.py": DUPLICATE})
        after = stubs({"hipengine/x.py": ZERO_THEN_DUPLICATE})
        decisions = {by_scope(before, "b-second").id: decide(by_scope(before, "b-second"))}
        self.assertEqual(core.rebind(after, decisions), [])
        self.assertEqual(set(decisions), {by_scope(after, "b-second").id})
        self.assertEqual(core.reconcile(after, decisions)["triaged"],
                         [by_scope(after, "b-second")])

    def test_duplicates_in_different_functions_get_distinct_keys(self):
        rows = stubs({"hipengine/x.py": DUPLICATE})
        self.assertEqual(len(set(keys(rows))), 2)
        self.assertIsNotNone(by_scope(rows, "a-first"))
        self.assertIsNotNone(by_scope(rows, "b-second"))
        self.assertNotIn("@", "".join(keys(rows)))     # separate scopes, no group

    def test_removing_one_function_leaves_the_other_alone(self):
        rows = stubs({"hipengine/x.py": DUPLICATE})
        later = by_scope(rows, "b-second").id
        self.assertEqual(ids(stubs({"hipengine/x.py": FIRST_TRIMMED})), [later])


class LineMovement(unittest.TestCase):
    """Inserting lines above a site must not change its identity."""

    BEFORE = '''"""Runtime."""


def export(self):
    raise NotImplementedError("dlpack export lands with the tensor runtime")
'''

    AFTER = '''"""Runtime, with a longer docstring than before.

It also gained a second paragraph, and an import.
"""

import json

MARKER = 1


def export(self):
    raise NotImplementedError("dlpack export lands with the tensor runtime")
'''

    def test_the_key_survives_lines_moving_above_it(self):
        self.assertEqual(keys(stubs({"hipengine/core/tensor.py": self.BEFORE})),
                         keys(stubs({"hipengine/core/tensor.py": self.AFTER})))

    def test_the_location_still_moves(self):
        before = stubs({"hipengine/core/tensor.py": self.BEFORE})
        after = stubs({"hipengine/core/tensor.py": self.AFTER})
        self.assertNotEqual(before[0].location, after[0].location)
        self.assertTrue(after[0].location.endswith(":12"))

    def test_a_decision_stays_attached_and_is_not_stale(self):
        before = stubs({"hipengine/core/tensor.py": self.BEFORE})
        after = stubs({"hipengine/core/tensor.py": self.AFTER})
        decisions = decisions_for(before)
        self.assertEqual(core.rebind(after, decisions), [])
        state = core.reconcile(after, decisions)
        self.assertEqual((state["open"], state["stale"], state["triaged"]), ([], [], after))

    def test_markers_move_the_same_way(self):
        self.assertEqual(keys(markers({"hipengine/x.py": MARKED})),
                         keys(markers({"hipengine/x.py": MARKED_MOVED})))

    def test_axis_branches_move_the_same_way(self):
        before = axis_branches({"hipengine/x.py": BRANCH})
        after = axis_branches({"hipengine/x.py": "# header\n\n\n" + BRANCH})
        self.assertEqual(keys(before), keys(after))


class ChangedContent(unittest.TestCase):
    """Changed text is a new site, not a re-used id."""

    def test_a_changed_message_changes_the_key(self):
        before = stubs({"hipengine/x.py": 'def f(self):\n    raise NotImplementedError("one")\n'})
        after = stubs({"hipengine/x.py": 'def f(self):\n    raise NotImplementedError("two")\n'})
        self.assertNotEqual(keys(before), keys(after))

    def test_a_changed_message_does_not_inherit_the_old_decision(self):
        before = stubs({"hipengine/x.py": 'def f(self):\n    raise NotImplementedError("one")\n'})
        after = stubs({"hipengine/x.py": 'def f(self):\n    raise NotImplementedError("two")\n'})
        decisions = decisions_for(before)
        self.assertEqual(core.rebind(after, decisions), [])
        self.assertEqual(core.reconcile(after, decisions)["open"], after)

    def test_a_renamed_scope_is_a_new_site(self):
        """Pinned trade-off: scope is part of identity, so renaming it re-opens the row."""
        before = stubs({"hipengine/x.py": "def f(self):\n    " + STUB})
        after = stubs({"hipengine/x.py": "def renamed(self):\n    " + STUB})
        self.assertNotEqual(keys(before), keys(after))


class FlagBranches(unittest.TestCase):
    """The flag check anchors on scope and name, and keeps both readable."""

    NAMES = ("HIPENGINE_ONE", "HIPENGINE_TWO")

    def rows(self, source):
        inventory = [core.Row(kind="flag", key=name, title=name,
                              evidence={"default": "off", "in_refactor_ledger": False,
                                        "sites_by_root": {"hipengine": 1}})
                     for name in self.NAMES]
        with mock.patch.object(skeletons, "load_inventory", lambda: inventory), \
                mock.patch.object(skeletons, "corpus", lambda: {"hipengine/x.py": source}):
            found, _ = skeletons.ungoverned_flag_branch()
        return found

    def test_two_identical_branches_in_different_functions_are_distinct_sites(self):
        rows = self.rows('def a():\n    return environ.get("HIPENGINE_ONE")\n'
                         '\n\ndef b():\n    return environ.get("HIPENGINE_ONE")\n')
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(set(keys(rows))), 2)
        self.assertTrue(all(row.key.endswith(":HIPENGINE_ONE") for row in rows))

    def test_two_flags_on_one_line_are_distinct_sites(self):
        rows = self.rows('def a():\n'
                         '    return environ.get("HIPENGINE_ONE") or '
                         'environ.get("HIPENGINE_TWO")\n')
        self.assertEqual(sorted(row.key.rsplit(":", 1)[-1] for row in rows),
                         ["HIPENGINE_ONE", "HIPENGINE_TWO"])
        self.assertEqual(len(set(keys(rows))), 2)
        self.assertNotIn("@", "".join(keys(rows)))

    def test_a_branch_keeps_its_key_when_lines_move_above_it(self):
        source = 'def a():\n    return environ.get("HIPENGINE_ONE")\n'
        self.assertEqual(keys(self.rows(source)), keys(self.rows("import os\n\n\n" + source)))


class DocTargets(unittest.TestCase):
    """`doc-path-drift` ids are document plus target, and never duplicated."""

    DOC = """# Catalog

`kernels/cpu_reference` is not at the repo root.

Elsewhere the same path is named again: `kernels/cpu_reference`.

And once more in a table:

| `kernels/cpu_reference` | x |
"""

    def scan(self, docs):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "pkg/kernels/cpu_reference").mkdir(parents=True)
            (root / "pkg/kernels/cpu_reference/__init__.py").write_text("")
            drift.tree_index.cache_clear()
            try:
                with mock.patch.object(drift, "REPO_ROOT", root), \
                        mock.patch.object(drift, "doc_corpus", lambda: dict(docs)):
                    rows, _ = drift.doc_path_drift()
            finally:
                drift.tree_index.cache_clear()
        return rows

    def test_a_repeated_target_yields_one_finding(self):
        rows = self.scan({"docs/A.md": self.DOC})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].key, "docs/A.md:kernels/cpu_reference")

    def test_no_two_rows_share_an_id(self):
        rows = self.scan({"docs/A.md": self.DOC, "docs/B.md": self.DOC})
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(set(ids(rows))), 2)

    def test_the_key_carries_no_line_number(self):
        self.assertNotIn(":1", self.scan({"docs/A.md": self.DOC})[0].key.rsplit("/", 1)[-1])

    def test_a_target_that_moves_down_the_page_keeps_its_decision(self):
        moved = "\n".join(["", "", ""] + self.DOC.splitlines()) + "\n"
        before = self.scan({"docs/A.md": self.DOC})
        after = self.scan({"docs/A.md": moved})
        self.assertEqual(ids(before), ids(after))
        decisions = decisions_for(before)
        self.assertEqual(core.reconcile(after, decisions)["triaged"], after)


if __name__ == "__main__":
    unittest.main()
