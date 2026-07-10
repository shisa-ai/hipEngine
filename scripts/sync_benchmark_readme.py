#!/usr/bin/env python3
"""Copy canonical benchmark table blocks into the repository README."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


BEGIN_RE = re.compile(r"^<!-- BEGIN TOPLINE:([A-Z0-9_]+) -->$")
END_RE = re.compile(r"^<!-- END TOPLINE:([A-Z0-9_]+) -->$")


@dataclass(frozen=True)
class Block:
    name: str
    begin: int
    end: int
    body: str


def _blocks(text: str, path: Path) -> dict[str, Block]:
    lines = text.splitlines(keepends=True)
    found: dict[str, Block] = {}
    active_name: str | None = None
    active_begin = -1

    for index, line in enumerate(lines):
        marker = line.rstrip("\r\n")
        begin_match = BEGIN_RE.fullmatch(marker)
        end_match = END_RE.fullmatch(marker)

        if begin_match:
            if active_name is not None:
                raise ValueError(
                    f"{path}:{index + 1}: nested TOPLINE block inside {active_name}"
                )
            active_name = begin_match.group(1)
            if active_name in found:
                raise ValueError(f"{path}:{index + 1}: duplicate block {active_name}")
            active_begin = index
            continue

        if end_match:
            end_name = end_match.group(1)
            if active_name is None:
                raise ValueError(f"{path}:{index + 1}: unmatched end block {end_name}")
            if end_name != active_name:
                raise ValueError(
                    f"{path}:{index + 1}: end block {end_name} does not match "
                    f"{active_name}"
                )
            body = "".join(lines[active_begin + 1 : index])
            found[active_name] = Block(active_name, active_begin, index, body)
            active_name = None
            active_begin = -1

    if active_name is not None:
        raise ValueError(f"{path}: unterminated block {active_name}")
    if not found:
        raise ValueError(f"{path}: no TOPLINE blocks found")
    return found


def _synchronized(source_text: str, target_text: str, source: Path, target: Path) -> str:
    source_blocks = _blocks(source_text, source)
    target_blocks = _blocks(target_text, target)
    missing = sorted(set(source_blocks) - set(target_blocks))
    extra = sorted(set(target_blocks) - set(source_blocks))
    if missing or extra:
        raise ValueError(
            f"TOPLINE block mismatch: missing in target={missing}, extra in target={extra}"
        )

    lines = target_text.splitlines(keepends=True)
    for block in sorted(target_blocks.values(), key=lambda item: item.begin, reverse=True):
        replacement = source_blocks[block.name].body.splitlines(keepends=True)
        lines[block.begin + 1 : block.end] = replacement
    return "".join(lines)


def _parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="fail if README.md is stale")
    action.add_argument("--write", action="store_true", help="update README.md in place")
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "benchmarks" / "README.md",
        help="canonical markdown source",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=repo_root / "README.md",
        help="markdown file receiving the exported blocks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = _parser(repo_root).parse_args(argv)
    source = args.source.resolve()
    target = args.target.resolve()

    try:
        source_text = source.read_text(encoding="utf-8")
        target_text = target.read_text(encoding="utf-8")
        synchronized = _synchronized(source_text, target_text, source, target)
    except (OSError, ValueError) as error:
        print(f"benchmark README sync error: {error}", file=sys.stderr)
        return 2

    if args.check:
        if synchronized != target_text:
            stale = [
                name
                for name, block in _blocks(source_text, source).items()
                if block.body != _blocks(target_text, target)[name].body
            ]
            print(f"README.md has stale TOPLINE blocks: {', '.join(stale)}", file=sys.stderr)
            print("run: python3 scripts/sync_benchmark_readme.py --write", file=sys.stderr)
            return 1
        print("README benchmark blocks are synchronized")
        return 0

    if synchronized == target_text:
        print("README benchmark blocks already synchronized")
        return 0
    target.write_text(synchronized, encoding="utf-8")
    print(f"updated {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
