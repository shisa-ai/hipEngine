"""Extractors.

Each returns `(rows, meta)`. An extractor reports **what it observed**, in
`Row.signals`, and never what it concluded. "no reader outside tests" is an
observation; "dead flag" is a judgement, and judgements belong in triage.
"""

from __future__ import annotations

import functools
import pathlib
import re
from typing import Callable

from ..core import REPO_ROOT, Row

CODE_ROOTS = ("hipengine", "scripts", "tests", "benchmarks")
CODE_SUFFIXES = (".py", ".hip", ".h", ".sh", ".toml")


@functools.lru_cache(maxsize=1)
def corpus() -> dict[str, str]:
    """Every code file's text, keyed by repo-relative path. Read once per process."""
    out: dict[str, str] = {}
    for root in CODE_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix not in CODE_SUFFIXES or not path.is_file():
                continue
            if "__pycache__" in path.parts or ".venv" in path.parts:
                continue
            try:
                out[path.relative_to(REPO_ROOT).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return out


@functools.lru_cache(maxsize=1)
def doc_corpus() -> dict[str, str]:
    """Every doc, keyed by repo-relative path. Excludes immutable worklog entries."""
    out: dict[str, str] = {}
    for path in (REPO_ROOT / "docs").rglob("*.md"):
        out[path.relative_to(REPO_ROOT).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    for name in ("AGENTS.md", "README.md", "TODO.md"):
        candidate = REPO_ROOT / name
        if candidate.exists():
            out[name] = candidate.read_text(encoding="utf-8", errors="replace")
    return out


def grep(needle: str, texts: dict[str, str]) -> list[str]:
    """`path:line` for every occurrence. Plain substring, no regex surprises."""
    hits: list[str] = []
    for path, text in texts.items():
        if needle not in text:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if needle in line:
                hits.append(f"{path}:{number}")
    return hits


EXTRACTORS: dict[str, Callable[[], tuple[list[Row], dict]]] = {}


def register(name: str):
    def wrap(fn):
        EXTRACTORS[name] = fn
        return fn
    return wrap


from . import ledger, flags, kernels, candidates  # noqa: E402,F401  (registers them)
