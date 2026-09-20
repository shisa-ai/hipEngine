#!/usr/bin/env python3
"""Validate docs/ front-matter and regenerate the docs indexes.

Every Markdown file under docs/ carries a front-matter block:

    ---
    status: normative | current | closed | superseded
    owns: one line naming what this document is the source of truth for
    superseded_by: docs/...      # required when status is `superseded`
    ---

`status` makes "is this binding?" greppable without opening the file, and
`owns` makes the generated index useful. Top-level and reference documents must
carry a real `owns` line; campaign, model-card, and archive documents may leave
it as TODO, which is reported as a warning so it can be filled in later.

    python3 scripts/docs/check_docs.py            # validate + report index drift
    python3 scripts/docs/check_docs.py --write    # regenerate the index tables

Stdlib only, no third-party YAML dependency.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"

STATUSES = ("normative", "current", "closed", "superseded")
#  Directories that carry their own generated index page.
SECTIONS = {
    "": ("Top level", "Rules and live references every agent should know exist."),
    "reference": ("Reference", "Current subsystem contracts. Read the one your task touches."),
    "campaigns": ("Campaigns", "Bounded model/hardware campaigns. Mostly closed; kept as evidence."),
    "model-cards": ("Model cards", "Per-model support status, quality, and integration notes."),
    "archive": ("Archive", "Superseded documents, closed proposals, and frozen history."),
}
#  Not part of the indexed doc set.
EXCLUDE_DIRS = ("examples", "testing")
INDEX_NAME = "README.md"

BEGIN = "<!-- BEGIN GENERATED: {} -->"
END = "<!-- END GENERATED: {} -->"


def parse_front_matter(text: str) -> tuple[dict[str, str], bool]:
    """Return (fields, found).  Hand-rolled: only `key: value` lines are used."""
    if not text.startswith("---\n"):
        return {}, False
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, False
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields, True


#  Hugging Face model cards ship with the weights and carry their own
#  front-matter schema. They are published artifacts, not internal docs, so
#  they are not indexed and must not be given a `status:` key.
HF_CARD_KEYS = ("pipeline_tag", "base_model", "license")


def is_published_model_card(fields: dict[str, str]) -> bool:
    return sum(key in fields for key in HF_CARD_KEYS) >= 2


def indexed_docs() -> list[pathlib.Path]:
    out = []
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(DOCS)
        if rel.parts and rel.parts[0] in EXCLUDE_DIRS:
            continue
        if path.name == INDEX_NAME:
            continue
        fields, found = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
        if found and is_published_model_card(fields):
            continue
        out.append(path)
    return out


def section_of(path: pathlib.Path) -> str:
    rel = path.relative_to(DOCS)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def validate() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for path in indexed_docs():
        rel = path.relative_to(REPO_ROOT)
        fields, found = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
        if not found:
            errors.append(
                f"{rel}: missing front-matter. Add a `status:` and `owns:` block "
                f"(see scripts/docs/check_docs.py docstring)."
            )
            continue
        status = fields.get("status", "")
        if status not in STATUSES:
            errors.append(f"{rel}: status must be one of {'/'.join(STATUSES)}, got {status!r}.")
        owns = fields.get("owns", "").strip()
        section = section_of(path)
        strict = section in ("", "reference")
        if not owns or owns == "TODO":
            message = f"{rel}: `owns:` is not filled in — say what this document is the source of truth for."
            (errors if strict else warnings).append(message)
        superseded_by = fields.get("superseded_by", "").strip()
        if status == "superseded":
            if not superseded_by:
                errors.append(f"{rel}: status is `superseded` but no `superseded_by:` is set.")
            elif not (REPO_ROOT / superseded_by).exists():
                errors.append(f"{rel}: superseded_by points at a missing file: {superseded_by}")
        elif superseded_by:
            errors.append(f"{rel}: superseded_by is set but status is {status!r}, not `superseded`.")
    return errors, warnings


def render_rows(paths: list[pathlib.Path], base: pathlib.Path) -> str:
    badge = {"normative": "**normative**", "current": "current", "closed": "closed", "superseded": "superseded"}
    lines = ["| Document | Status | Owns |", "| --- | --- | --- |"]
    for path in paths:
        fields, _ = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
        link = path.relative_to(base).as_posix()
        owns = fields.get("owns", "").strip() or "—"
        if owns == "TODO":
            owns = "_TODO_"
        extra = fields.get("superseded_by", "").strip()
        if extra:
            owns = f"{owns} Superseded by `{extra}`."
        lines.append(f"| [`{path.name}`]({link}) | {badge.get(fields.get('status',''), '—')} | {owns} |")
    return "\n".join(lines)


def splice(text: str, marker: str, body: str) -> str:
    begin, end = BEGIN.format(marker), END.format(marker)
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    block = f"{begin}\n{body}\n{end}"
    if not pattern.search(text):
        raise SystemExit(f"index is missing the {marker!r} generated block")
    return pattern.sub(lambda _: block, text)


def build_indexes() -> dict[pathlib.Path, str]:
    docs = indexed_docs()
    grouped: dict[str, list[pathlib.Path]] = {key: [] for key in SECTIONS}
    for path in docs:
        grouped[section_of(path)].append(path)

    out: dict[pathlib.Path, str] = {}

    root_index = DOCS / INDEX_NAME
    text = root_index.read_text(encoding="utf-8")
    text = splice(text, "top", render_rows(grouped[""], DOCS))
    counts = "\n".join(
        f"| [`{name}/`]({name}/) | {SECTIONS[name][1]} | {len(grouped[name])} |"
        for name in ("reference", "campaigns", "model-cards", "archive")
    )
    text = splice(text, "sections", "| Directory | Contents | Files |\n| --- | --- | --- |\n" + counts)
    out[root_index] = text

    for name in ("reference", "campaigns", "model-cards", "archive"):
        index = DOCS / name / INDEX_NAME
        if not index.exists():
            continue
        body = render_rows(grouped[name], DOCS / name)
        out[index] = splice(index.read_text(encoding="utf-8"), name, body)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="regenerate the index tables in place")
    args = parser.parse_args()

    errors, warnings = validate()
    for message in warnings:
        print(f"warning: {message}")

    rendered = build_indexes()
    drifted = [path for path, text in rendered.items() if path.read_text(encoding="utf-8") != text]

    if args.write:
        for path, text in rendered.items():
            path.write_text(text, encoding="utf-8")
        if drifted:
            print(f"regenerated {len(drifted)} index page(s): " + ", ".join(p.name for p in drifted))
        else:
            print("indexes already current")
    elif drifted:
        errors.append(
            "docs indexes are stale: "
            + ", ".join(str(p.relative_to(REPO_ROOT)) for p in drifted)
            + " — run `python3 scripts/docs/check_docs.py --write`"
        )

    for message in errors:
        print(f"error: {message}", file=sys.stderr)
    if errors:
        print(f"\ndocs: {len(errors)} error(s), {len(warnings)} warning(s)", file=sys.stderr)
        return 1
    print(f"docs: {len(indexed_docs())} documents valid, {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
