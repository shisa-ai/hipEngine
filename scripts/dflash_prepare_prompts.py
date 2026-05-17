#!/usr/bin/env python3
"""Build or validate the stable DFlash speculative prompt fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.prompts import (  # noqa: E402
    DEFAULT_STABLE_PROMPT_FIXTURE,
    DEFAULT_SYNTHETIC_LENGTHS,
    DEFAULT_SYNTHETIC_SEED,
    build_prompt_records,
    load_prompt_records,
    load_tokenizer,
    validate_prompt_records,
    write_prompt_jsonl,
)

DEFAULT_TOKENIZER_MODEL = (
    "/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/"
    "snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
)


def _parse_lengths(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, int] = {}
    categories: dict[str, int] = {}
    for record in records:
        groups[str(record.get("benchmark_group"))] = groups.get(str(record.get("benchmark_group")), 0) + 1
        categories[str(record.get("category"))] = categories.get(str(record.get("category")), 0) + 1
    return {
        "rows": len(records),
        "benchmark_groups": dict(sorted(groups.items())),
        "categories": dict(sorted(categories.items())),
        "prompt_tokens_min": min(int(r["prompt_tokens"]) for r in records) if records else 0,
        "prompt_tokens_max": max(int(r["prompt_tokens"]) for r in records) if records else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_STABLE_PROMPT_FIXTURE)
    parser.add_argument("--synthetic-lengths", default=",".join(str(x) for x in DEFAULT_SYNTHETIC_LENGTHS))
    parser.add_argument("--synthetic-seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    parser.add_argument("--no-token-ids", action="store_true", help="Only write hashes/counts, not prompt_ids")
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing fixture instead of rebuilding it")
    args = parser.parse_args()

    if args.validate_only:
        records = load_prompt_records(args.output)
    else:
        tokenizer = load_tokenizer(args.tokenizer_model)
        records = build_prompt_records(
            tokenizer,
            tokenizer_path=args.tokenizer_model,
            synthetic_lengths=_parse_lengths(args.synthetic_lengths),
            synthetic_seed=args.synthetic_seed,
            include_token_ids=not args.no_token_ids,
        )
        write_prompt_jsonl(records, args.output)
    errors = validate_prompt_records(records)
    output = {"fixture": str(args.output), "passed": not errors, "errors": list(errors), **_summary(records)}
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
