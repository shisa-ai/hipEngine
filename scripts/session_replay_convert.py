#!/usr/bin/env python3
"""Convert a Jouzu agent session transcript into a replay benchmark fixture.

The fixture feeds the ``session_replay`` lane of
``scripts/prefix_cache_multiturn_bench.py``: a real agentic client resend
pattern in which every request re-sends the whole transcript and the
transcript grows by one assistant round plus its tool results, or by one
user message, between consecutive requests.

The session log does not record the client system prompt or tool schemas,
so a representative coding-agent policy and stub tool schemas are
substituted; assistant thinking blocks are not replayed.  See
``hipengine/benchmark/session_replay.py`` for the ancestry and resend
contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.session_replay import (  # noqa: E402
    SessionReplayError,
    convert_session_jsonl,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Jouzu session .jsonl transcript to convert",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output fixture JSON path",
    )
    parser.add_argument(
        "--workload-id",
        default="session-tail",
        help="workload id inside the fixture (default: session-tail)",
    )
    parser.add_argument(
        "--tip",
        default=None,
        help=(
            "session record id of the conversation tip. Default: the last "
            "content-bearing assistant message of the live pre-compaction chain."
        ),
    )
    parser.add_argument(
        "--tail-rounds",
        type=int,
        default=None,
        help=(
            "keep only the last N request rounds of the conversation "
            "(the slice may start mid-conversation)"
        ),
    )
    parser.add_argument(
        "--max-tool-result-chars",
        type=int,
        default=None,
        help="cap each tool-result message to this many characters",
    )
    parser.add_argument("--suite", default=None, help="fixture suite label")
    parser.add_argument("--description", default=None, help="fixture description")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        fixture = convert_session_jsonl(
            args.session,
            workload_id=args.workload_id,
            tip=args.tip,
            tail_rounds=args.tail_rounds,
            max_tool_result_chars=args.max_tool_result_chars,
            suite=args.suite,
            description=args.description,
        )
    except SessionReplayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    workload = fixture["workloads"][0]
    slice_info = workload["slice"]
    ends = workload["request_ends"]
    print(f"wrote {args.out}")
    print(f"  suite: {fixture['suite']}")
    print(f"  session tip: {slice_info['tip_record']}")
    print(
        f"  chain messages: {slice_info['chain_messages']} "
        f"(excluded off-chain: {slice_info['excluded_off_chain_messages']})"
    )
    print(
        f"  entries: {len(workload['entries'])} | requests: {len(ends)} "
        f"| tools: {', '.join(tool['name'] for tool in fixture['tools'])}"
    )
    if slice_info.get("skipped_empty_messages"):
        print(f"  skipped empty messages: {slice_info['skipped_empty_messages']}")
    if slice_info.get("synthesized_interrupt_results"):
        print(
            "  synthesized interrupt results: "
            f"{slice_info['synthesized_interrupt_results']}"
        )
    if slice_info.get("truncated_tool_results"):
        print(f"  truncated tool results: {slice_info['truncated_tool_results']}")
    prompt_chars = [
        sum(len(str(entry.get("content", ""))) for entry in workload["entries"][:end])
        for end in ends
    ]
    print(
        f"  request prompt chars: first={prompt_chars[0]} "
        f"last={prompt_chars[-1]} max={max(prompt_chars)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
