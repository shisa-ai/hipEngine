"""Reject an invalid distribution batch before any release file is uploaded."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Conservative decimal MB ceiling, also below PyPI's default 100 MiB limit.
DEFAULT_LIMIT_BYTES = 100_000_000


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-bytes", type=_positive_int, default=DEFAULT_LIMIT_BYTES)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = False
    for path in args.files:
        try:
            if not path.is_file():
                raise ValueError("not a regular file")
            size = path.stat().st_size
            if size == 0:
                raise ValueError("empty distribution")
            if size > args.limit_bytes:
                raise ValueError(f"{size:,} bytes exceeds the {args.limit_bytes:,}-byte limit")
        except (OSError, ValueError) as exc:
            print(f"Distribution size check failed: {path}: {exc}", file=sys.stderr)
            failed = True
        else:
            print(f"{path.name}: {size:,} / {args.limit_bytes:,} bytes")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
