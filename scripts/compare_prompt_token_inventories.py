#!/usr/bin/env python3
"""Compare prompt token inventory JSON files.

This is a parity gate for MTP-GGUF work.  It deliberately compares exact token
ID arrays by prompt name; equal text hashes are not enough for cross-engine
accepted/output comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _rows_by_name(inventory: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    rows = inventory.get("prompts")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} inventory has no non-empty 'prompts' list")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} inventory contains a non-object prompt row: {row!r}")
        name = row.get("name")
        token_ids = row.get("token_ids")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} prompt row is missing a non-empty name: {row!r}")
        if name in by_name:
            raise ValueError(f"{label} inventory contains duplicate prompt name: {name}")
        if not isinstance(token_ids, list) or not all(isinstance(token_id, int) for token_id in token_ids):
            raise ValueError(f"{label} prompt {name!r} is missing integer token_ids")
        by_name[name] = row
    return by_name


def _first_mismatch(left: Sequence[int], right: Sequence[int]) -> tuple[int | None, int | None, int | None]:
    for idx, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return idx, int(left_token), int(right_token)
    if len(left) != len(right):
        idx = min(len(left), len(right))
        left_token = int(left[idx]) if idx < len(left) else None
        right_token = int(right[idx]) if idx < len(right) else None
        return idx, left_token, right_token
    return None, None, None


def compare_prompt_token_inventories(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_label: str = "left",
    right_label: str = "right",
    context_tokens: int = 8,
) -> dict[str, Any]:
    """Compare prompt token IDs by prompt name."""

    left_rows = _rows_by_name(left, label=left_label)
    right_rows = _rows_by_name(right, label=right_label)
    left_names = set(left_rows)
    right_names = set(right_rows)
    common_names = sorted(left_names & right_names)
    missing_in_left = sorted(right_names - left_names)
    missing_in_right = sorted(left_names - right_names)

    matches: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for name in common_names:
        left_row = left_rows[name]
        right_row = right_rows[name]
        left_tokens = [int(token_id) for token_id in left_row["token_ids"]]
        right_tokens = [int(token_id) for token_id in right_row["token_ids"]]
        if left_tokens == right_tokens:
            matches.append(name)
            continue
        idx, left_token, right_token = _first_mismatch(left_tokens, right_tokens)
        start = max(0, (idx or 0) - context_tokens)
        end = min(max(len(left_tokens), len(right_tokens)), (idx or 0) + context_tokens + 1)
        mismatches.append(
            {
                "name": name,
                "left_token_count": len(left_tokens),
                "right_token_count": len(right_tokens),
                "left_token_ids_sha256": left_row.get("token_ids_sha256"),
                "right_token_ids_sha256": right_row.get("token_ids_sha256"),
                "rendered_sha256_match": left_row.get("rendered_sha256") == right_row.get("rendered_sha256"),
                "first_mismatch_index": idx,
                "left_token_id": left_token,
                "right_token_id": right_token,
                "left_window": left_tokens[start:end],
                "right_window": right_tokens[start:end],
            }
        )

    all_match = not missing_in_left and not missing_in_right and not mismatches
    return {
        "schema": 1,
        "kind": "prompt_token_inventory_comparison",
        "left_label": left_label,
        "right_label": right_label,
        "left_kind": left.get("kind"),
        "right_kind": right.get("kind"),
        "all_match": all_match,
        "compared_prompts": len(common_names),
        "matched_prompts": matches,
        "missing_in_left": missing_in_left,
        "missing_in_right": missing_in_right,
        "mismatches": mismatches,
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path, help="left inventory JSON")
    parser.add_argument("--right", required=True, type=Path, help="right inventory JSON")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--context-tokens", type=int, default=8)
    parser.add_argument("--out", type=Path, help="write comparison JSON to this path")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    comparison = compare_prompt_token_inventories(
        load_json(args.left),
        load_json(args.right),
        left_label=args.left_label,
        right_label=args.right_label,
        context_tokens=args.context_tokens,
    )
    payload = json.dumps(comparison, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.fail_on_mismatch and not comparison["all_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
