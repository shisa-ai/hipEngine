"""Half-done work left in the tree: markers, stubs, and flag-guarded branches.

The flag check is the one that cross-references the inventory: a branch guarded
by a flag the inventory shows is harness-only or unread is a branch nothing in
production can reach, and deleting it is a concrete edit.
"""

from __future__ import annotations

import re

from ..core import REPO_ROOT, Row, load_inventory
from ..inventory import corpus
from . import SourceIndex, finding, register, site_keys

MARKER = re.compile(r"(?:^|\s)#\s*(TODO|FIXME|XXX|HACK)\b[: ]?(.{0,110})")
STUB = re.compile(r"^\s*raise NotImplementedError\b(.*)$")
FLAG_IN_LINE = re.compile(r"\bHIPENGINE_[A-Z0-9_]+\b")


def _hits(text: str, pattern: re.Pattern) -> list[tuple[int, str, re.Match]]:
    """`(line number, line, match)` for every line of `text` the pattern matches."""
    return [(number, line, hit)
            for number, line in enumerate(text.splitlines(), 1)
            if (hit := pattern.search(line))]


@register("marker")
def marker() -> tuple[list[Row], dict]:
    """TODO/FIXME/XXX/HACK in shipped runtime code."""
    rows, counts = [], {}
    for path, text in sorted(corpus().items()):
        if not path.startswith("hipengine/") or not path.endswith(".py"):
            continue
        hits = _hits(text, MARKER)
        if not hits:
            continue
        keys = site_keys(path, SourceIndex(text),
                         [(number, line.strip()) for number, line, _ in hits])
        for (number, line, hit), key in zip(hits, keys):
            kind, note = hit.group(1), hit.group(2).strip()
            counts[kind] = counts.get(kind, 0) + 1
            rows.append(finding(
                "marker", key,
                f"{kind}: {note or '(no text)'}"[:120], f"{path}:{number}",
                fix="Do it, or move it to docs/REFACTOR.md with a removal condition and delete "
                    "the marker. A marker in runtime code is a note to nobody.",
                why=f"{kind} marker in shipped runtime code",
                evidence={"marker": kind, "text": note[:200]},
            ))
    return rows, {"scanned": "hipengine/**/*.py", "by_marker": counts}


@register("stub")
def stub() -> tuple[list[Row], dict]:
    """`raise NotImplementedError` reachable from the runtime."""
    rows = []
    for path, text in sorted(corpus().items()):
        if not path.startswith("hipengine/") or not path.endswith(".py"):
            continue
        hits = _hits(text, STUB)
        if not hits:
            continue
        index = SourceIndex(text)
        #  The whole statement is the identity, so a message that wraps past any
        #  number of lines is read in full. A match the parser does not see as a
        #  raise - one inside a string, say - falls back to its own line.
        sites = [(number, index.statement(number) or line.strip()) for number, line, _ in hits]
        keys = site_keys(path, index, sites, drop="raise NotImplementedError")
        for (number, line, _), key in zip(hits, keys):
            rows.append(finding(
                "stub", key,
                f"unimplemented: {line.strip()[:100]}", f"{path}:{number}",
                fix="Implement it, or make the caller unable to reach it and say so at the call "
                    "site. A NotImplementedError a user can hit is a crash with extra steps.",
                why="NotImplementedError in runtime code",
            ))
    return rows, {"scanned": "hipengine/**/*.py"}


@register("ungoverned-flag-branch")
def ungoverned_flag_branch() -> tuple[list[Row], dict]:
    """Runtime branches on a flag that is default-off with no recorded way to retire it.

    Cross-references `audit/inventory/flags.json`. `AGENTS.md` "Flags are a cost,
    not a feature" says a warranted flag defaults to the production behaviour and
    carries a removal condition in `docs/REFACTOR.md`. A default-off flag with
    neither is a branch nobody will ever turn on and nobody has agreed to delete
    — the shape that leaves working code switched off indefinitely.
    """
    inventory = {row.key: row for row in load_inventory() if row.kind == "flag"}
    suspect = {
        name: row for name, row in inventory.items()
        if row.evidence.get("default") in ("off", "split")
        and not row.evidence.get("in_refactor_ledger")
        and row.evidence.get("sites_by_root", {}).get("hipengine")
        #  A name the runtime only mentions (a docstring, a constant nothing
        #  reads) guards no branch.
        and row.evidence.get("runtime_read", "env") in ("env", "indirect")
    }
    rows = []
    for path, text in sorted(corpus().items()):
        if not path.startswith("hipengine/") or not path.endswith(".py"):
            continue
        sites = [(number, line.strip(), name)
                 for number, line in enumerate(text.splitlines(), 1)
                 for name in FLAG_IN_LINE.findall(line) if name in suspect]
        if not sites:
            continue
        keys = site_keys(path, SourceIndex(text), sites)
        for (number, _line, name), key in zip(sites, keys):
            rows.append(finding(
                "ungoverned-flag-branch", key,
                f"runtime branch on {name}: default-{suspect[name].evidence.get('default')} "
                f"with no removal condition recorded", f"{path}:{number}",
                fix=f"Decide the flag's fate: promote {name} to the default and keep it only "
                    f"as a rollback lever, or delete the branch. Either way add a "
                    f"docs/REFACTOR.md entry naming the removal condition. "
                    f"See `audit.py show flag/{name}`.",
                why="AGENTS.md 'Flags are a cost, not a feature': default-off with no "
                    "recorded removal condition",
                evidence={"flag": name, "default": suspect[name].evidence.get("default"),
                          "flag_signals": suspect[name].signals},
            ))
    return rows, {"suspect_flags": len(suspect), "cross_reference": "inventory/flags.json"}
