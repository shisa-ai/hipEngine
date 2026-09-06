#!/usr/bin/env python3
"""Copy the compact canonical benchmark summary into the repository README."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


BEGIN_RE = re.compile(r"^<!-- BEGIN TOPLINE:([A-Z0-9_]+) -->$")
END_RE = re.compile(r"^<!-- END TOPLINE:([A-Z0-9_]+) -->$")
DEFAULT_BLOCKS = ("README_HIGHLIGHTS",)
PUBLIC_README_MAX_LINES = 400


@dataclass(frozen=True)
class PublicExportBudget:
    max_lines: int
    max_prose_paragraphs: int
    max_prose_chars: int


PUBLIC_EXPORT_BUDGETS = {
    "README_HIGHLIGHTS": PublicExportBudget(
        max_lines=130,
        max_prose_paragraphs=8,
        max_prose_chars=1500,
    ),
}


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


def _prose_paragraphs(body: str) -> list[str]:
    paragraphs: list[str] = []
    active: list[str] = []

    def finish() -> None:
        if active:
            paragraphs.append(" ".join(active))
            active.clear()

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "|", "<!--")):
            finish()
        else:
            active.append(stripped)
    finish()
    return paragraphs


def _validate_public_export(block: Block, source: Path) -> None:
    budget = PUBLIC_EXPORT_BUDGETS.get(block.name)
    if budget is None:
        return

    line_count = len(block.body.splitlines())
    if line_count > budget.max_lines:
        raise ValueError(
            f"{source}: TOPLINE block {block.name} exceeds the public README "
            f"line budget ({line_count}/{budget.max_lines}); move detail to the "
            "benchmark artifacts or worklog"
        )

    paragraphs = _prose_paragraphs(block.body)
    prose_chars = sum(len(paragraph) for paragraph in paragraphs)
    if (
        len(paragraphs) > budget.max_prose_paragraphs
        or prose_chars > budget.max_prose_chars
    ):
        raise ValueError(
            f"{source}: TOPLINE block {block.name} exceeds the public README "
            f"prose budget ({len(paragraphs)}/{budget.max_prose_paragraphs} "
            f"paragraphs, {prose_chars}/{budget.max_prose_chars} characters); "
            "move implementation and evidence detail to the benchmark artifacts "
            "or worklog"
        )


def _validate_public_readme(text: str, target: Path) -> None:
    line_count = len(text.splitlines())
    if line_count > PUBLIC_README_MAX_LINES:
        raise ValueError(
            f"{target}: exceeds the public README document budget "
            f"({line_count}/{PUBLIC_README_MAX_LINES} lines); move detailed "
            "history or evidence to the benchmark artifacts or worklog"
        )


def _synchronized(
    source_text: str,
    target_text: str,
    source: Path,
    target: Path,
    *,
    block_names: tuple[str, ...] = DEFAULT_BLOCKS,
) -> str:
    source_blocks = _blocks(source_text, source)
    target_blocks = _blocks(target_text, target)
    selected = set(block_names)
    missing_source = sorted(selected - set(source_blocks))
    missing_target = sorted(selected - set(target_blocks))
    extra_target = sorted(set(target_blocks) - selected)
    if missing_source or missing_target or extra_target:
        raise ValueError(
            "TOPLINE block mismatch: "
            f"missing in source={missing_source}, missing in target={missing_target}, "
            f"unselected in target={extra_target}"
        )
    for name in block_names:
        _validate_public_export(source_blocks[name], source)

    lines = target_text.splitlines(keepends=True)
    selected_blocks = (target_blocks[name] for name in block_names)
    for block in sorted(selected_blocks, key=lambda item: item.begin, reverse=True):
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
        _validate_public_readme(synchronized, target)
    except (OSError, ValueError) as error:
        print(f"benchmark README sync error: {error}", file=sys.stderr)
        return 2

    if args.check:
        if synchronized != target_text:
            source_blocks = _blocks(source_text, source)
            target_blocks = _blocks(target_text, target)
            stale = [
                name
                for name in DEFAULT_BLOCKS
                if source_blocks[name].body != target_blocks[name].body
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
