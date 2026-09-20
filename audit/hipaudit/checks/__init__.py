"""Code checks: scan the tree and queue work that can actually be done.

The inventory catalogues debt somebody already wrote down. These checks look at
the **code** and find things with a concrete fix, so each finding names the edit
that closes it.

Findings share the triage store with the inventory, so a `wontfix` recorded here
is honoured everywhere, and a finding survives rescans the same way an inventory
row does.
"""

from __future__ import annotations

import re
from typing import Callable

from ..core import REPO_ROOT, Row
from ..inventory import corpus, doc_corpus, tokens

CHECKS: dict[str, Callable[[], tuple[list[Row], dict]]] = {}


def register(name: str):
    def wrap(fn):
        CHECKS[name] = fn
        return fn
    return wrap


def finding(kind: str, key: str, title: str, location: str, *,
            fix: str, why: str, evidence: dict | None = None) -> Row:
    """A finding always carries the edit that closes it."""
    return Row(
        kind=kind, key=key, title=title, location=location,
        evidence={"fix": fix, "why": why, **(evidence or {})},
        signals=[why],
        hints={"anchor": location.split(":")[0], "refs": [key], "tokens": tokens(title)},
    )


from . import invariants, skeletons, drift  # noqa: E402,F401
