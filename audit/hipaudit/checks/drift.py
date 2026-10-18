"""Documentation that disagrees with the tree.

A path named in a document that does not exist sends every reader, human and
agent, to the wrong place. This is the check that catches `kernels/` being
documented as a repo-root directory when it lives at `hipengine/kernels/`.

Precision matters more than recall here: documents are full of placeholders,
MIME types, and illustrative paths. A finding is only raised when the named
thing is **findable somewhere else in the tree**, which makes it both a real
error and a concrete edit.
"""

from __future__ import annotations

import functools
import re

from ..core import REPO_ROOT, Row
from ..inventory import doc_corpus
from . import finding, register

PATHISH = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*(?:/[A-Za-z0-9_./-]*|\.(?:py|hip|md|json|toml)))`")
#  Records of what something said at the time; drift there is history, not error.
EXEMPT = ("docs/campaigns/", "docs/archive/", "docs/testing/", "docs/examples/")
MIME = re.compile(r"^(?:image|application|text|audio|video|multipart)/")
#  OUTDIR/TAG-NAME.log, <path>, FOO/BAR — illustrative, not real.
PLACEHOLDER = re.compile(r"[<>{}*]|(?:^|/)[A-Z][A-Z0-9_-]{2,}(?:/|$|\.)")
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".ruff_cache"}


@functools.lru_cache(maxsize=1)
def tree_index() -> dict[str, list[str]]:
    """Every repo path, indexed by basename, so a moved file can be located."""
    index: dict[str, list[str]] = {}
    for path in REPO_ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        index.setdefault(path.name, []).append(path.relative_to(REPO_ROOT).as_posix())
    return index


@register("doc-path-drift")
def doc_path_drift() -> tuple[list[Row], dict]:
    index = tree_index()
    rows, checked, skipped = [], 0, 0
    for path, text in sorted(doc_corpus().items()):
        if any(path.startswith(prefix) for prefix in EXEMPT):
            continue
        seen: set[str] = set()
        for number, line in enumerate(text.splitlines(), 1):
            for target in PATHISH.findall(line):
                target = target.rstrip("/")
                if target in seen:
                    continue
                seen.add(target)
                #  A single bare word is a prose reference, not a path claim:
                #  `reference/` inside a sentence about the docs layout is fine.
                multi_segment = target.count("/") >= 1
                has_suffix = target.endswith((".py", ".hip", ".md", ".json", ".toml", ".sh"))
                if not (multi_segment and (has_suffix or target.count("/") >= 1)):
                    skipped += 1
                    continue
                if MIME.match(target) or PLACEHOLDER.search(target) or target.startswith(("~", "http")):
                    skipped += 1
                    continue
                checked += 1
                if (REPO_ROOT / target).exists():
                    continue
                #  Only report it when we can say where it actually is. An
                #  unfindable name is far more likely to be illustrative.
                #  Require the whole documented path to be a suffix of a real one,
                #  so `a/b.py` matches `pkg/a/b.py` but not any stray `b.py`.
                elsewhere = [p for p in index.get(target.rsplit("/", 1)[-1], [])
                             if p == target or p.endswith("/" + target)]
                if not elsewhere:
                    skipped += 1
                    continue
                rows.append(finding(
                    "doc-path-drift", f"{path}:{number}:{target}",
                    f"names `{target}`, which is actually at `{elsewhere[0]}`", f"{path}:{number}",
                    fix=f"Change `{target}` to `{elsewhere[0]}`"
                        + (f" (or one of {len(elsewhere)} matches)." if len(elsewhere) > 1 else "."),
                    why="documented path does not exist, but the file is present elsewhere",
                    evidence={"target": target, "doc": path, "actual": elsewhere[:4]},
                ))
    return rows, {"paths_checked": checked, "skipped_as_illustrative": skipped, "exempt": list(EXEMPT)}
