"""Contract tests for the shadowed-definition check.

A second `def name` in one scope rebinds the first before anything can read it.
The check reports that pair and nothing else: every case where the repeat can be
intentional (decorators, property setters, overloads, class bodies, conditional
alternatives) and every case where the earlier binding is still reachable (an
alias, a registration, a default, an annotation, an intervening call) is skipped.

Identity is content-based, so moving a definition keeps its row and changing
either body makes a new one.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hipaudit.checks import CHECKS, shadowed  # noqa: E402
from hipaudit.inventory import corpus  # noqa: E402

#  The pair the check was calibrated on: `Qwen35GGUFBringupGenerator.configure_dms`
#  in hipengine/generation/qwen35_gguf.py at 2d8c2ea72, with the ~350 lines of
#  other methods between the two definitions abbreviated to one. The later
#  definition adds a preparation-state guard; the earlier one is unreachable.
REAL_PAIR = '''class Qwen35GGUFBringupGenerator:
    def configure_dms(self, config) -> None:
        """Bind either a compact DMS config or a sidecar metadata path."""
        if isinstance(config, str):
            self.dms_metadata_path = config
            self.dms_config = None
        else:
            self.dms_config = config
    def _prepare_kv_policy(self, params: Any | None) -> None:
        ...
    def configure_dms(self, config) -> None:
        """Bind either a compact DMS config or a sidecar metadata path."""
        if getattr(self, "_prepared_kv_signature", None) is not None:
            raise RuntimeError("DMS must be configured before resident preparation")
        if isinstance(config, str):
            self.dms_metadata_path = config
            self.dms_config = None
        else:
            self.dms_config = config
'''

PLAIN = ('def f(self):\n'
         '    return 1\n'
         '\n\n'
         'def f(self):\n'
         '    return 2\n')

IDENTICAL = ('def f(self):\n'
             '    return 1\n'
             '\n\n'
             'def f(self):\n'
             '    return 1\n')

GUARDED = ('def configure(self, config):\n'
           '    self.config = config\n'
           '\n\n'
           'def configure(self, config):\n'
           '    if self.prepared:\n'
           '        raise RuntimeError("configure first")\n'
           '    self.config = config\n')

THREE = ('def f(self):\n'
         '    return 1\n'
         '\n\n'
         'def f(self):\n'
         '    return 2\n'
         '\n\n'
         'def f(self):\n'
         '    return 3\n')


def scan(files):
    with mock.patch.object(shadowed, "corpus", lambda: dict(files)):
        rows, _ = shadowed.shadowed_definition()
    return rows


def one(source, path="hipengine/x.py"):
    """The single finding `source` yields, so a failure reads as a count."""
    rows = scan({path: source})
    if len(rows) != 1:
        raise AssertionError(f"expected one finding, got {len(rows)}: "
                             f"{[(row.location, row.title) for row in rows]}")
    return rows[0]


class RealPair(unittest.TestCase):
    """The duplicate the check exists for, as it appears in the tree."""

    def test_the_configure_dms_pair_is_reported(self):
        row = one(REAL_PAIR, path="hipengine/generation/qwen35_gguf.py")
        self.assertEqual(row.evidence["scope"], "Qwen35GGUFBringupGenerator")
        self.assertEqual(row.evidence["name"], "configure_dms")
        self.assertEqual(row.location, "hipengine/generation/qwen35_gguf.py:2")
        self.assertEqual(row.evidence["lines"], {"earlier": 2, "later": 11})

    def test_the_two_bodies_are_offered_for_comparison(self):
        row = one(REAL_PAIR)
        self.assertFalse(row.evidence["bodies_identical"])
        self.assertEqual(row.evidence["earlier_starts_with"], 'if isinstance(config, str): '
                         "self.dms_metadata_path = config self.dms_config = None else: "
                         "self.dms_config = config")
        self.assertTrue(row.evidence["later_starts_with"].startswith(
            "if getattr(self, '_prepared_kv_signature', None) is not None:"))
        self.assertTrue(row.evidence["later_difference"].startswith(
            "if getattr(self, '_prepared_kv_signature', None) is not None:"))
        self.assertTrue(row.evidence["earlier_difference"].startswith(
            "if isinstance(config, str):"))

    def test_another_method_between_the_two_definitions_does_not_hide_the_pair(self):
        """The real pair has hundreds of lines between it; adjacency is not required."""
        self.assertIn("_prepare_kv_policy", REAL_PAIR)
        self.assertEqual(len(scan({"hipengine/x.py": REAL_PAIR})), 1)


class Duplicates(unittest.TestCase):
    """A repeat the check should report."""

    def test_a_plain_repeat_is_one_row_naming_both_locations(self):
        row = one(PLAIN)
        self.assertEqual(row.kind, "shadowed-definition")
        self.assertEqual(row.location, "hipengine/x.py:1")
        self.assertEqual(row.evidence["lines"], {"earlier": 1, "later": 5})

    def test_the_row_states_the_edit_and_the_reason(self):
        row = one(PLAIN)
        self.assertIn("Compare the two definitions", row.evidence["fix"])
        self.assertIn("before deleting", row.evidence["fix"])
        self.assertIn("rebinds the first", row.evidence["why"])
        self.assertEqual(row.hints["anchor"], "hipengine/x.py")

    def test_the_row_does_not_claim_the_earlier_body_is_dead(self):
        """A dynamic reference can still reach it, so the fix asks for a check first."""
        row = one(PLAIN)
        self.assertIn("removal candidate", row.evidence["why"])
        self.assertIn("not a proven dead path", row.evidence["why"])
        self.assertNotIn("Delete the earlier definition", row.evidence["fix"])
        self.assertNotIn("unreachable", row.evidence["fix"] + row.evidence["why"])

    def test_a_signature_change_with_the_same_body_is_a_signature_difference(self):
        """The metadata names which half moved; a body digest is not a definition digest."""
        wider = ('def f(self):\n'
                 '    return 1\n'
                 '\n\n'
                 'def f(self, extra=1):\n'
                 '    return 1\n')
        row = one(wider)
        self.assertFalse(row.evidence["signature_identical"])
        self.assertTrue(row.evidence["bodies_identical"])
        self.assertIn("the two signatures differ", row.signals)
        self.assertIn("the two bodies are identical", row.signals)
        self.assertNotIn("the two bodies differ only in their docstring", row.signals)
        self.assertNotEqual(row.evidence["earlier_definition"], row.evidence["later_definition"])

    def test_identical_bodies_are_reported_as_identical(self):
        row = one(IDENTICAL)
        self.assertTrue(row.evidence["bodies_identical"])
        self.assertEqual((row.evidence["earlier_difference"], row.evidence["later_difference"]),
                         ("", ""))

    def test_bodies_that_start_alike_report_where_they_diverge(self):
        """The shape of `yue2.distribution`: same opening, different tail."""
        diverging = ('def f(x):\n'
                     '    row = load(x)\n'
                     '    return full(row)\n'
                     '\n\n'
                     'def f(x):\n'
                     '    row = load(x)\n'
                     '    return window(row)\n')
        row = one(diverging)
        self.assertEqual(row.evidence["earlier_starts_with"],
                         row.evidence["later_starts_with"])
        self.assertEqual(row.evidence["earlier_difference"], "return full(row)")
        self.assertEqual(row.evidence["later_difference"], "return window(row)")

    def test_a_docstring_only_change_reports_no_code_difference(self):
        docstring = ('def f(self):\n'
                     '    """Old wording."""\n'
                     '    return 1\n'
                     '\n\n'
                     'def f(self):\n'
                     '    """New wording."""\n'
                     '    return 1\n')
        row = one(docstring)
        self.assertFalse(row.evidence["bodies_identical"])
        self.assertEqual((row.evidence["earlier_difference"],
                          row.evidence["later_difference"]), ("", ""))
        self.assertIn("the two bodies differ only in their docstring", row.signals)

    def test_a_body_that_only_grew_reports_the_extra_statement(self):
        grown = ('def f(x):\n'
                 '    return 1\n'
                 '\n\n'
                 'def f(x):\n'
                 '    return 1\n'
                 '    log(x)\n')
        row = one(grown)
        self.assertEqual((row.evidence["earlier_difference"],
                          row.evidence["later_difference"]), ("", "log(x)"))

    def test_a_body_differing_only_by_a_guard_is_reported_as_different(self):
        row = one(GUARDED)
        self.assertFalse(row.evidence["bodies_identical"])
        self.assertEqual(row.evidence["earlier_starts_with"], "self.config = config")
        self.assertEqual(row.evidence["later_starts_with"],
                         "if self.prepared: raise RuntimeError('configure first')")
        self.assertIn("the two bodies differ", row.signals)

    def test_three_definitions_are_outside_the_pair_only_check(self):
        self.assertEqual(scan({"hipengine/x.py": THREE}), [])
        retained = THREE.replace("def f(self):\n    return 2", "def f(self, saved=f):\n    return saved")
        self.assertEqual(scan({"hipengine/x.py": retained}), [])

    def test_async_definitions_are_covered(self):
        repeated = ('async def load(self):\n'
                    '    return 1\n'
                    '\n\n'
                    'async def load(self):\n'
                    '    return 2\n')
        self.assertEqual(one(repeated).evidence["name"], "load")
        mixed = ('async def load(self):\n'
                 '    return 1\n'
                 '\n\n'
                 'def load(self):\n'
                 '    return 2\n')
        self.assertEqual(one(mixed).evidence["lines"], {"earlier": 1, "later": 5})

    def test_a_repeat_inside_a_function_body_is_reported(self):
        nested = ('def outer():\n'
                  '    def inner():\n'
                  '        return 1\n'
                  '    def inner():\n'
                  '        return 2\n'
                  '    return inner\n')
        row = one(nested)
        self.assertEqual(row.evidence["scope"], "outer")
        self.assertEqual(row.evidence["lines"], {"earlier": 2, "later": 4})

    def test_a_read_after_the_pair_does_not_keep_the_earlier_one(self):
        self.assertEqual(len(scan({"hipengine/x.py": PLAIN + "\nf()\n"})), 1)

    def test_same_name_in_distinct_scopes_is_not_a_repeat(self):
        separate = ('class A:\n'
                    '    def f(self):\n'
                    '        return 1\n'
                    '\n\n'
                    'class B:\n'
                    '    def f(self):\n'
                    '        return 2\n'
                    '\n\n'
                    'def f():\n'
                    '    return 3\n')
        self.assertEqual(scan({"hipengine/x.py": separate}), [])


class IntentionalRebindings(unittest.TestCase):
    """Patterns where a second `def` is deliberate, so nothing is queued."""

    def test_decorated_definitions_are_skipped(self):
        decorated = ('import functools\n'
                     '\n\n'
                     '@functools.cache\n'
                     'def f(self):\n'
                     '    return 1\n'
                     '\n\n'
                     '@functools.cache\n'
                     'def f(self):\n'
                     '    return 2\n')
        self.assertEqual(scan({"hipengine/x.py": decorated}), [])

    def test_a_property_setter_is_not_a_repeat(self):
        prop = ('class A:\n'
                '    @property\n'
                '    def value(self):\n'
                '        return self._value\n'
                '\n'
                '    @value.setter\n'
                '    def value(self, new):\n'
                '        self._value = new\n')
        self.assertEqual(scan({"hipengine/x.py": prop}), [])

    def test_overloads_are_not_a_repeat(self):
        overloads = ('from typing import overload\n'
                     '\n\n'
                     '@overload\n'
                     'def f(x: int) -> int: ...\n'
                     '\n\n'
                     '@overload\n'
                     'def f(x: str) -> str: ...\n'
                     '\n\n'
                     'def f(x):\n'
                     '    return x\n')
        self.assertEqual(scan({"hipengine/x.py": overloads}), [])

    def test_conditional_alternatives_are_not_a_repeat(self):
        platform = ('import sys\n'
                    '\n\n'
                    'if sys.platform == "win32":\n'
                    '    def path():\n'
                    '        return 1\n'
                    'else:\n'
                    '    def path():\n'
                    '        return 2\n')
        fallback = ('try:\n'
                    '    def load():\n'
                    '        return 1\n'
                    'except ImportError:\n'
                    '    def load():\n'
                    '        return 2\n')
        one_guarded = ('def f():\n'
                       '    return 1\n'
                       '\n\n'
                       'if flag:\n'
                       '    def f():\n'
                       '        return 2\n')
        for source in (platform, fallback, one_guarded):
            self.assertEqual(scan({"hipengine/x.py": source}), [])

    def test_a_decorated_definition_between_two_plain_ones_rejects_the_group(self):
        """The name then has a binding this check cannot read as a plain replacement."""
        mixed = ('def f(self):\n'
                 '    return 1\n'
                 '\n\n'
                 '@functools.cache\n'
                 'def f(self):\n'
                 '    return 2\n'
                 '\n\n'
                 'def f(self):\n'
                 '    return 3\n')
        self.assertEqual(scan({"hipengine/x.py": mixed}), [])

    def test_a_class_between_two_plain_definitions_rejects_the_group(self):
        mixed = ('def f(self):\n'
                 '    return 1\n'
                 '\n\n'
                 'class f:\n'
                 '    pass\n'
                 '\n\n'
                 'def f(self):\n'
                 '    return 3\n')
        self.assertEqual(scan({"hipengine/x.py": mixed}), [])

    def test_a_conditional_definition_between_the_pair_rejects_it(self):
        conditional = ('def f(self):\n'
                       '    return 1\n'
                       '\n\n'
                       'if flag:\n'
                       '    def f(self):\n'
                       '        return 2\n'
                       '\n\n'
                       'def f(self):\n'
                       '    return 3\n')
        self.assertEqual(scan({"hipengine/x.py": conditional}), [])

    def test_a_conditional_rebinding_between_the_pair_rejects_it(self):
        rebound = ('def f(self):\n'
                   '    return 1\n'
                   '\n\n'
                   'if flag:\n'
                   '    f = other\n'
                   '\n\n'
                   'def f(self):\n'
                   '    return 3\n')
        imported = ('def f(self):\n'
                    '    return 1\n'
                    '\n\n'
                    'import f\n'
                    '\n\n'
                    'def f(self):\n'
                    '    return 3\n')
        for source in (rebound, imported):
            self.assertEqual(scan({"hipengine/x.py": source}), [])

    def test_class_definitions_are_not_reported(self):
        """A class body runs at definition time and may register itself."""
        registered = ('class Thing:\n'
                      '    REGISTRY.append("thing")\n'
                      '\n\n'
                      'class Thing:\n'
                      '    pass\n')
        self.assertEqual(scan({"hipengine/x.py": registered}), [])
        mixed = ('def Thing():\n'
                 '    return 1\n'
                 '\n\n'
                 'class Thing:\n'
                 '    pass\n')
        self.assertEqual(scan({"hipengine/x.py": mixed}), [])


class EarlierBindingRetained(unittest.TestCase):
    """The earlier definition is still reachable, so it is not shadowed."""

    def test_an_alias_keeps_the_earlier_definition(self):
        alias = ('def f(self):\n'
                 '    return 1\n'
                 '\n\n'
                 'handler = f\n'
                 '\n\n'
                 'def f(self):\n'
                 '    return 2\n')
        self.assertEqual(scan({"hipengine/x.py": alias}), [])

    def test_a_registration_keeps_the_earlier_definition(self):
        registered = ('def f(self):\n'
                      '    return 1\n'
                      '\n\n'
                      'HANDLERS.append(f)\n'
                      '\n\n'
                      'def f(self):\n'
                      '    return 2\n')
        self.assertEqual(scan({"hipengine/x.py": registered}), [])

    def test_a_default_argument_using_the_earlier_definition_is_definition_time_use(self):
        later = ('def f(self):\n'
                 '    return 1\n'
                 '\n\n'
                 'def g(cb=f):\n'
                 '    return cb\n'
                 '\n\n'
                 'def f(self):\n'
                 '    return 2\n')
        own = ('def f(self):\n'
               '    return 1\n'
               '\n\n'
               'def f(self, cb=f):\n'
               '    return cb\n')
        self.assertEqual(scan({"hipengine/x.py": later}), [])
        self.assertEqual(scan({"hipengine/x.py": own}), [])

    def test_an_annotation_using_the_earlier_definition_is_definition_time_use(self):
        annotated = ('def f(self):\n'
                     '    return 1\n'
                     '\n\n'
                     'def g() -> f:\n'
                     '    ...\n'
                     '\n\n'
                     'def f(self):\n'
                     '    return 2\n')
        self.assertEqual(scan({"hipengine/x.py": annotated}), [])

    def test_a_one_line_later_default_is_definition_time_use(self):
        one_line = ('def f(self):\n'
                    '    return 1\n'
                    '\n\n'
                    'def f(self, cb=f): return cb\n')
        self.assertEqual(scan({"hipengine/x.py": one_line}), [])

    def test_a_wrapped_header_ending_on_the_first_body_line_is_definition_time_use(self):
        """The default shares a line with the body, which a line window misread as a read."""
        wrapped = ('def f(self):\n'
                   '    return 1\n'
                   '\n\n'
                   'def f(\n'
                   '    self,\n'
                   '    cb=f): return cb\n')
        self.assertEqual(scan({"hipengine/x.py": wrapped}), [])

    def test_a_wrapped_return_annotation_on_the_first_body_line_is_definition_time_use(self):
        wrapped = ('def f(self):\n'
                   '    return 1\n'
                   '\n\n'
                   'def f(\n'
                   '    self,\n'
                   ') -> f: return self\n')
        self.assertEqual(scan({"hipengine/x.py": wrapped}), [])

    def test_intervening_execution_that_reads_the_name_keeps_the_earlier_definition(self):
        called = ('def f(self):\n'
                  '    return 1\n'
                  '\n\n'
                  'prepare(f)\n'
                  '\n\n'
                  'def f(self):\n'
                  '    return 2\n')
        deferred = ('def f(self):\n'
                    '    return 1\n'
                    '\n\n'
                    'def helper():\n'
                    '    return f()\n'
                    '\n\n'
                    'def f(self):\n'
                    '    return 2\n')
        deleted = ('def f(self):\n'
                   '    return 1\n'
                   '\n\n'
                   'del f\n'
                   '\n\n'
                   'def f(self):\n'
                   '    return 2\n')
        for source in (called, deferred, deleted):
            self.assertEqual(scan({"hipengine/x.py": source}), [])


class Identity(unittest.TestCase):
    """The key is content, so positions can move and edits cannot be inherited."""

    def test_blank_lines_above_and_inside_keep_the_key(self):
        moved = "# a header\n\n\nimport os\n\n\n" + PLAIN.replace("\n\n\n", "\n\n\n# note\n")
        before, after = one(PLAIN), one(moved)
        self.assertEqual(before.key, after.key)
        self.assertNotEqual(before.location, after.location)

    def test_a_move_above_the_pair_does_not_age_the_evidence(self):
        """Positions are volatile, so a shift above the definitions is not new evidence."""
        moved = "import os\n\n\nMARKER = 1\n\n\n" + PLAIN
        before, after = one(PLAIN), one(moved)
        self.assertNotEqual(before.evidence["lines"], after.evidence["lines"])
        self.assertEqual(before.evidence_hash, after.evidence_hash)

    def test_reformatting_a_body_keeps_the_key(self):
        reflowed = ('def f(self):\n'
                    '    return 1\n'
                    '\n\n'
                    'def f(\n'
                    '    self,\n'
                    '):\n'
                    '    return 2\n')
        self.assertEqual(one(PLAIN).key, one(reflowed).key)

    def test_a_semantic_edit_to_either_body_changes_the_key(self):
        edited = PLAIN.replace("return 2", "return 3")
        self.assertNotEqual(one(PLAIN).key, one(edited).key)
        self.assertNotEqual(one(PLAIN).key, one(PLAIN.replace("return 1", "return 0")).key)

    def test_a_new_scope_changes_the_key(self):
        renamed = PLAIN.replace("def f(", "def g(")
        self.assertNotEqual(one(PLAIN).key, one(renamed).key)

    def test_no_two_rows_share_an_id(self):
        rows = scan({"hipengine/x.py": PLAIN + "\n" + GUARDED, "hipengine/y.py": PLAIN})
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row.id for row in rows}), 3)


class UnreadableSources(unittest.TestCase):
    """A tree that is momentarily broken yields nothing rather than noise."""

    def test_a_file_that_does_not_parse_yields_nothing(self):
        broken = ('def f(self):\n'
                  '    return 1\n'
                  '\n\n'
                  'def f(self):\n'
                  '    return (\n')
        self.assertEqual(scan({"hipengine/x.py": broken}), [])

    def test_a_null_byte_yields_nothing(self):
        self.assertEqual(scan({"hipengine/x.py": PLAIN + "\x00"}), [])

    def test_non_python_files_are_ignored(self):
        self.assertEqual(scan({"hipengine/kernel.hip": PLAIN, "scripts/run.sh": PLAIN}), [])


class LiveTree(unittest.TestCase):
    """On the real corpus every row names two definitions that exist there."""

    def test_the_check_runs_on_the_corpus_and_each_row_is_verifiable(self):
        rows, meta = shadowed.shadowed_definition()
        self.assertEqual(meta["scanned"], "**/*.py")
        self.assertGreater(meta["parsed"], 0)
        texts = corpus()
        for row in rows:
            path = row.location.split(":")[0]
            definitions = [node for node in ast.walk(ast.parse(texts[path]))
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and node.name == row.evidence["name"]]
            located = {node.lineno for node in definitions}
            self.assertIn(row.evidence["lines"]["earlier"], located)
            self.assertIn(row.evidence["lines"]["later"], located)
            self.assertLess(row.evidence["lines"]["earlier"], row.evidence["lines"]["later"])
            pair_lines = set(row.evidence["lines"].values())
            self.assertTrue(all(not node.decorator_list for node in definitions
                                if node.lineno in pair_lines))


class Registration(unittest.TestCase):
    def test_the_check_is_registered(self):
        self.assertIn("shadowed-definition", CHECKS)


if __name__ == "__main__":
    unittest.main()
