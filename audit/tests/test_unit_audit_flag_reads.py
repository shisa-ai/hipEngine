"""How the runtime uses a flag name: the facts flag triage rests on."""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hipaudit.inventory import flags  # noqa: E402

PATH = "hipengine/x.py"


def reads(source: str, name: str = "HIPENGINE_X") -> list[tuple[str, int, str]]:
    flags.module_reads.cache_clear()
    return flags.module_reads(PATH, source).get(name, [])


class FlagReads(unittest.TestCase):
    def test_a_direct_read_and_its_default(self):
        self.assertEqual(reads('import os\nv = os.environ.get("HIPENGINE_X", "1")\n'),
                         [("env", 2, "on")])
        self.assertEqual(reads('import os\nv = os.environ.get("HIPENGINE_X")\n'), [("env", 2, "off")])

    def test_a_read_through_a_module_constant_and_a_helper(self):
        source = ('_X_ENV = "HIPENGINE_X"\n_LIMIT = 65_792\n'
                  'def f():\n    return _env_int(_X_ENV, _LIMIT)\n')
        self.assertEqual(reads(source), [("env", 4, "value")])
        self.assertEqual(reads('_X_ENV = (\n    "HIPENGINE_X"\n)\nv = _env_flag(_X_ENV, True)\n'),
                         [("env", 4, "on")])

    def test_a_name_handed_to_another_function_is_indirect(self):
        self.assertEqual(reads('v = layers_allowed("HIPENGINE_X", 3)\n'), [("indirect", 1, "unknown")])
        self.assertEqual(reads('TABLE = {"budget": "HIPENGINE_X"}\n'), [("indirect", 1, "unknown")])

    def test_writes_are_bindings_not_reads(self):
        self.assertEqual(reads('import os\nos.environ["HIPENGINE_X"] = "1"\n'), [("bound", 2, "unknown")])
        self.assertEqual(reads('import os\nos.environ.pop("HIPENGINE_X", None)\n'),
                         [("bound", 2, "unknown")])
        self.assertEqual(reads('_bind_profile_env("HIPENGINE_X", "0")\n'), [("bound", 1, "unknown")])

    def test_a_docstring_or_comment_is_not_a_read(self):
        self.assertEqual(reads('"""Set HIPENGINE_X to enable."""\n# HIPENGINE_X\n'), [])

    def test_an_exported_python_name_is_not_a_flag(self):
        source = 'HIPENGINE_X_ENV = "HIPENGINE_X"\n__all__ = ("HIPENGINE_X_ENV",)\n'
        self.assertEqual(reads(source, "HIPENGINE_X_ENV"), [])

    def test_shell_and_hip_sources_use_getenv_and_expansions(self):
        flags.module_reads.cache_clear()
        found = flags.module_reads("hipengine/k.hip", 'const char* v = getenv("HIPENGINE_X");\n')
        self.assertEqual(found["HIPENGINE_X"], [("env", 1, "unknown")])


if __name__ == "__main__":
    unittest.main()
