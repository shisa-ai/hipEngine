#!/usr/bin/env python3
"""Collect free-run task answers from the served route for semantic review.

Quality is not a numeric gate: the answers have to do what the prompt asked.
This script runs the full mtp-bench category suite and the category heldouts
through the live server and writes one artifact with every prompt, answer,
finish reason, and token count. A reviewer then judges each answer against its
prompt; the artifact is the evidence the review is recorded against.

Never judge quality from a single prompt: the suite covers code, general_en,
general_ja, and mixed_ja_en, and the heldouts cover the same categories with
different tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SUITE = REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"
DEFAULT_HELDOUTS = REPO_ROOT / "benchmarks" / "prompts" / "gdn-prefill-category-heldouts.jsonl"


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18400")
    parser.add_argument("--model", default="qwen38fn-capon")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--heldouts", type=Path, default=DEFAULT_HELDOUTS)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated case ids to run instead of the whole suite.",
    )
    args = parser.parse_args(argv)

    import httpx

    cases = [("suite", row) for row in _load(args.suite)] + [
        ("heldout", row) for row in _load(args.heldouts)
    ]
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        cases = [item for item in cases if item[1]["id"] in wanted]
        missing = wanted - {row["id"] for _, row in cases}
        if missing:
            parser.error(f"unknown case ids: {sorted(missing)}")
    results: list[dict[str, Any]] = []
    failures = 0
    with httpx.Client(timeout=args.timeout) as client:
        for origin, row in cases:
            started = time.perf_counter()
            response = client.post(
                f"{args.url.rstrip('/')}/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": row["messages"],
                    "temperature": 0.0,
                    "max_tokens": args.max_tokens,
                },
            )
            elapsed = time.perf_counter() - started
            body = response.json()
            if response.status_code != 200:
                failures += 1
                results.append(
                    {
                        "id": row["id"],
                        "category": row["category"],
                        "origin": origin,
                        "status": response.status_code,
                        "error": json.dumps(body)[:600],
                        "prompt": row["messages"][-1]["content"],
                    }
                )
                print(f"{row['id']}: ERROR {response.status_code}")
                continue
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            answer = message.get("content") or ""
            results.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "origin": origin,
                    "status": 200,
                    "prompt": row["messages"][-1]["content"],
                    "answer": answer,
                    "reasoning_content": message.get("reasoning_content"),
                    "finish_reason": choice.get("finish_reason"),
                    "finish_details": choice.get("finish_details"),
                    "usage": body.get("usage"),
                    "elapsed_s": round(elapsed, 3),
                    "answer_chars": len(answer),
                }
            )
            print(
                f"{row['id']}: {choice.get('finish_reason')} "
                f"{len(answer)} chars in {elapsed:.1f}s"
            )

    summary = {
        "schema": "hipengine.qwen4exp.http_task_quality_gate.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "suite": str(args.suite),
        "heldouts": str(args.heldouts),
        "cases": results,
        "case_count": len(results),
        "error_count": failures,
        "review": {"status": "pending", "reviewer": None, "verdicts": []},
    }
    args.output.write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": len(results), "errors": failures, "artifact": str(args.output)}))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
