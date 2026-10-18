#!/usr/bin/env python3
"""Check that docs/ENVS.md covers every environment variable read in the tree.

Scans ``hipengine/``, ``scripts/``, ``tests/``, and ``benchmarks/`` (.py and
.sh) for environment-variable names:

* literal and identifier arguments of ``os.environ`` / ``getenv`` /
  ``setdefault`` / ``pop`` / index accesses (resolving module-level ``CONST =
  "NAME"`` indirection),
* every ``HIPENGINE_*`` string literal (covers tuple-of-names loops such as
  ``for flag in (...): os.environ[flag] = "1"``),
* ``${VAR}`` / ``$VAR`` reads in shell scripts.

Then verifies each name appears in ``docs/ENVS.md``. Names explained in the
doc's "Names that are not environment variables" appendix count as covered,
because the checker treats any backticked token in the doc as documented.

Usage::

    python3 scripts/check_envs_docs.py            # report and exit nonzero on gaps
    python3 scripts/check_envs_docs.py --list    # print uncovered names only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "ENVS.md"
SCAN_ROOTS = ("hipengine", "scripts", "tests", "benchmarks")

# Standard process/shell variables that are documented as a group in the doc's
# third-party section and would otherwise need per-name backticks.
IMPLICIT_GROUPS = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG",
}

CONST_DEF = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=\s*["\']([A-Z][A-Z0-9_]+)["\']',
    re.M,
)
ENV_ACCESS = re.compile(
    r"(?:os\.environ(?:\.get)?|os\.getenv|environ\.get|getenv|setdefault"
    r"|os\.environ\.pop)\s*\(\s*(?:\"([A-Z0-9_]+)\"|'([A-Z0-9_]+)'"
    r"|([A-Za-z_][A-Za-z0-9_]*))"
)
ENV_INDEX = re.compile(
    r"os\.environ\[\s*(?:\"([A-Z0-9_]+)\"|'([A-Z0-9_]+)'|([A-Za-z_][A-Za-z0-9_]*))\s*\]"
)
HIPENGINE_LITERAL = re.compile(r"HIPENGINE_[A-Z0-9_]+")
SHELL_READ = re.compile(r"\$\{?([A-Z][A-Z0-9_]{3,})\}?")


def scan_names() -> set[str]:
    py_files: list[Path] = []
    sh_files: list[Path] = []
    self_path = Path(__file__).resolve()
    for root in SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        py_files.extend(p for p in base.rglob("*.py") if p.resolve() != self_path)
        sh_files.extend(base.rglob("*.sh"))

    const_map: dict[str, set[str]] = {}
    texts: dict[Path, str] = {}
    for path in py_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        texts[path] = text
        for match in CONST_DEF.finditer(text):
            const_map.setdefault(match.group(1), set()).add(match.group(2))

    read_names: set[str] = set()
    for path, text in texts.items():
        for pattern in (ENV_ACCESS, ENV_INDEX):
            for match in pattern.finditer(text):
                literal = match.group(1) or match.group(2)
                identifier = match.group(3)
                if literal:
                    read_names.add(literal)
                elif identifier:
                    read_names.update(const_map.get(identifier, ()))
        # Every HIPENGINE_* literal counts: tuple-of-names loops, harness env
        # dicts, and provenance key lists all reach the runtime this way.
        read_names.update(HIPENGINE_LITERAL.findall(text))

    for path in sh_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in SHELL_READ.finditer(text):
            read_names.add(match.group(1))

    read_names.difference_update(IMPLICIT_GROUPS)
    return read_names


def documented_names() -> set[str]:
    doc = DOC_PATH.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"`([A-Z][A-Z0-9_]+)`", doc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print only the uncovered variable names",
    )
    args = parser.parse_args()

    read_names = scan_names()
    covered = documented_names()
    missing = sorted(name for name in read_names if name not in covered)

    if args.list:
        for name in missing:
            print(name)
        return 1 if missing else 0

    print(f"env names read in code: {len(read_names)}")
    print(f"names documented in docs/ENVS.md: {len(covered & read_names)}")
    if missing:
        print(f"MISSING from docs/ENVS.md ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
        print(
            "Add each name to the matching section of docs/ENVS.md, or to its "
            "'Names that are not environment variables' appendix if it is a "
            "macro, prefix fragment, constant identifier, or placeholder."
        )
        return 1
    print("docs/ENVS.md coverage is complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
