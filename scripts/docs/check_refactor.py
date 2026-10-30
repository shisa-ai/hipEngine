#!/usr/bin/env python3
"""Check the active refactor backlog's IDs, ownership and completion contracts.

Each level-three entry requires Owner (a repository file/directory), Scope and
Done when fields. Combined and derived sweep IDs remain distinct; RF entries may
refer to their shared R0187 origin without creating duplicate active ownership.
History belongs in worklogs, not closed entries in this ledger.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
ID = re.compile(r"RF-\d+|R\d{4}(?:-[A-Z][A-Z0-9-]*)?")
RANGE = re.compile(r"(R\d{4})[–-](R\d{4})")
CLOSED = {"closed", "completed", "complete", "resolved", "superseded", "retired", "fixed"}
PLACEHOLDER = re.compile(r"^(?:TODO|TBD|<[^>]+>|none|n/a)[.!]?$", re.I)


def _without_fences(text: str) -> str:
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            lines.append("")
        else:
            lines.append("" if fence else line)
    return "\n".join(lines)


def _identifiers(label: str) -> list[str]:
    def expand(match):
        start, stop = (int(value[1:]) for value in match.groups())
        if stop < start or stop - start > 1000:
            raise ValueError("invalid identifier range")
        return "/".join(f"R{value:04d}" for value in range(start, stop + 1))

    expanded = RANGE.sub(expand, label)
    values = ID.findall(expanded)
    if not values or re.sub(r"[\s/,]+", "", ID.sub("", expanded)):
        raise ValueError("invalid identifier")
    # R0187 is an original sweep referral, not each RF task's active ID.
    if values[0].startswith("RF-"):
        values = [value for value in values if value != "R0187"]
    return values


def _field(body: str, name: str) -> str:
    match = re.search(
        rf"(?m)^{re.escape(name)}:[ \t]*(.*(?:\n[ \t]+[^\n]+)*)", body
    )
    return " ".join(match[1].split()).strip("` .") if match else ""


def _anchor(title: str) -> str:
    title = re.sub(r"[^\w\- ]", "", title.lower().replace("`", ""))
    return title.replace(" ", "-")


def _link_errors(text: str, root: Path, ledger: Path) -> list[str]:
    errors = []
    for match in re.finditer(r"\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text):
        target = match[1].strip("<>")
        parts = urlsplit(target)
        if parts.scheme or parts.netloc:
            continue
        path = ((root / ledger).parent / unquote(parts.path)).resolve() if parts.path else (root / ledger).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f"dangling local link: {target}")
            continue
        if parts.fragment:
            source = _without_fences(path.read_text(encoding="utf-8"))
            anchors = set()
            counts: dict[str, int] = {}
            for heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", source):
                slug = _anchor(heading)
                count = counts.get(slug, 0)
                counts[slug] = count + 1
                anchors.add(slug if count == 0 else f"{slug}-{count}")
            anchors.update(re.findall(r'(?:id|name)=[\"\']([^\"\']+)[\"\']', source))
            if unquote(parts.fragment) not in anchors:
                errors.append(f"dangling local link anchor: {target}")
    return errors


def validate_ledger(text: str, *, root: Path, ledger: Path) -> list[str]:
    """Return actionable errors without changing the ledger or reading GPU state."""
    source = _without_fences(text)
    entries = list(re.finditer(r"(?m)^### (.+)\n([\s\S]*?)(?=^### |\Z)", source))
    errors = []
    if not entries:
        errors.append("no active ledger entries")
    seen: dict[str, str] = {}
    for entry in entries:
        heading, body = entry.groups()
        label, separator, title = heading.partition(" — ")
        if not separator:
            errors.append(f"{heading}: identifier must precede an em dash and title")
        try:
            identifiers = _identifiers(label)
        except ValueError:
            identifiers = []
            errors.append(f"{heading}: invalid identifier")
        for identifier in identifiers:
            if identifier in seen:
                errors.append(f"{heading}: duplicate active ID {identifier} (also {seen[identifier]})")
            seen[identifier] = heading
        status = re.search(r"(?im)^\s*(?:\*\*)?Status:(?:\*\*)?\s*([\w-]+)", body)
        if (status and status[1].lower() in CLOSED) or re.search(
            r"\((?:completed|closed|resolved|fixed|retired)\)\s*$", title, re.I
        ):
            errors.append(f"{heading}: closed entry belongs in worklog history")
        for name in ("Owner", "Scope", "Done when"):
            value = _field(body, name)
            if not value or PLACEHOLDER.fullmatch(value):
                errors.append(f"{heading}: missing concrete {name} field")
        owner = _field(body, "Owner")
        if owner and not PLACEHOLDER.fullmatch(owner):
            path = (root / owner).resolve()
            if not path.is_relative_to(root.resolve()) or not path.exists():
                errors.append(f"{heading}: owner path does not exist in repository: {owner}")
    errors.extend(_link_errors(source, root, ledger))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("docs/REFACTOR.md"))
    args = parser.parse_args(argv)
    path = args.ledger if args.ledger.is_absolute() else ROOT / args.ledger
    try:
        errors = validate_ledger(path.read_text(encoding="utf-8"), root=ROOT, ledger=path)
    except OSError as exc:
        errors = [str(exc)]
    for error in errors:
        print(f"refactor: {error}")
    if not errors:
        print("refactor ledger: IDs, open status, ownership, scope, completion and links pass")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
