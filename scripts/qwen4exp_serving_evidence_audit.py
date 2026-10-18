#!/usr/bin/env python3
"""Audit that every cited campaign artifact exists and contains the quoted numbers.

The production-serving checklist quotes specific numbers next to each evidence
link. A link that resolves to nothing, or a number that appears nowhere in the
artifacts it cites, is a documentation defect: the reader cannot check the claim.

This is a mechanical check, not a review. It proves existence and numeric
presence; it cannot judge whether a measurement means what the row says.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "QWEN4EXP-PRODUCTION-SERVING.md"
LINK = re.compile(r"\]\(\.\./(benchmarks/results/[^)\s]+)\)")
DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")
# A slash pair inside a KL chain ("6.3e-4/3.6e-3/8.0e-3") is not a count.
RATIO = re.compile(r"(?<![\w/.-])(\d+/\d+)(?![\w/])")
PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)%")


def _artifact_text(path: Path, cache: dict[Path, str]) -> str:
    if path in cache:
        return cache[path]
    text = ""
    if path.is_file():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
    elif path.is_dir():
        parts = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix in {".json", ".md"}:
                try:
                    parts.append(child.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
        text = "\n".join(parts)
    cache[path] = text
    return text


def _rows(doc: str) -> list[tuple[str, str]]:
    rows = []
    for line in doc.splitlines():
        if not line.startswith("| PS"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        rows.append((cells[0], line))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    doc_text = args.doc.read_text(encoding="utf-8")
    cache: dict[Path, str] = {}
    findings: list[dict[str, Any]] = []
    checked_links = 0

    for row_name, row_text in _rows(doc_text):
        links = LINK.findall(row_text)
        missing = []
        artifact_text_parts = []
        for link in links:
            path = (REPO_ROOT / link).resolve()
            checked_links += 1
            if not path.exists():
                missing.append(link)
                continue
            artifact_text_parts.append(_artifact_text(path, cache))
        blob = "\n".join(artifact_text_parts)

        if missing:
            findings.append(
                {"row": row_name, "kind": "missing_artifact", "detail": missing}
            )

        absent_numbers = []
        if links:
            for pattern in (DECIMAL, RATIO):
                for value in sorted(set(pattern.findall(row_text))):
                    if value in blob:
                        continue
                    if pattern is RATIO:
                        # A count pair is present when both integers appear.
                        numerator, denominator = value.split("/")
                        if numerator in blob and denominator in blob:
                            continue
                    # A quoted value may be a rounded form of the artifact's.
                    if pattern is DECIMAL:
                        try:
                            target = float(value)
                        except ValueError:
                            target = None
                        if target is not None:
                            candidates = [
                                float(found)
                                for found in DECIMAL.findall(blob)
                                if abs(float(found) - target) <= max(5e-4, abs(target) * 0.005)
                            ]
                            if candidates:
                                continue
                    absent_numbers.append(value)
            for value in sorted(set(PERCENT.findall(row_text))):
                if f"{value}%" in blob:
                    continue
                try:
                    fraction = float(value) / 100.0
                except ValueError:
                    continue
                if any(
                    abs(float(found) - fraction) <= 5e-4
                    for found in DECIMAL.findall(blob)
                ):
                    continue
                absent_numbers.append(f"{value}%")
        if absent_numbers:
            findings.append(
                {
                    "row": row_name,
                    "kind": "number_not_in_cited_artifacts",
                    "detail": absent_numbers,
                }
            )

    # Every JSON artifact the rows cite must parse.
    bad_json = []
    for path, text in cache.items():
        if path.suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                bad_json.append({"path": str(path.relative_to(REPO_ROOT)), "error": str(exc)})
    if bad_json:
        findings.append({"row": "*", "kind": "unparseable_artifact", "detail": bad_json})

    summary = {
        "schema": "hipengine.qwen4exp.serving_evidence_audit.v1",
        "doc": str(args.doc.relative_to(REPO_ROOT)),
        "rows": len(_rows(doc_text)),
        "links_checked": checked_links,
        "findings": findings,
        "finding_count": len(findings),
        "status": "passed" if not findings else "failed",
    }
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for finding in findings:
        print(f"{finding['kind']}: {finding['row']} -> {finding['detail']}")
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "links": summary["links_checked"],
                "findings": summary["finding_count"],
                "status": summary["status"],
            }
        )
    )
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
