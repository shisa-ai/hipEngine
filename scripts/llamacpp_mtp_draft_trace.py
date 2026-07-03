#!/usr/bin/env python3
"""Parse llama.cpp verbose draft-mtp candidate logs into a compact trace.

llama.cpp emits draft candidate top-k lines at LOG_DBG verbosity from
``common_speculative_impl_draft_mtp::draft``.  This parser turns those log lines
into a stable JSON artifact for hipEngine MTP-GGUF parity debugging.  It is a
log parser only; it does not run llama.cpp, load weights, or execute hipEngine.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_CANDIDATE_PREFIX_RE = re.compile(
    r"^\s*- seq_id\s+(?P<seq_id>\d+),\s+draft candidate\s+"
    r"(?P<rank>\d+),\s+pos\s+(?P<pos>\d+):\s+"
    r"(?P<token_id>-?\d+)\s+\(\s*(?P<prob>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\)\s+'(?P<piece>.*)$"
)
_CALL_RE = re.compile(
    r"common_speculative_draft: called impl draft-mtp, hist size = (?P<hist_size>\d+), "
    r"call_count = (?P<call_count>\d+), gen = (?P<generated>\d+)"
)
_ACCEPT_RE = re.compile(r"accepted (?P<accepted>\d+)/(?P<generated>\d+) draft tokens")
_TIMING_RE = re.compile(
    r"draft acceptance = (?P<acceptance>[+-]?(?:\d+(?:\.\d*)?|\.\d+)) "
    r"\(\s*(?P<accepted>\d+) accepted /\s*(?P<generated>\d+) generated\)"
)
_REQUEST_RE = re.compile(r"new prompt, .*task\.n_tokens = (?P<prompt_tokens>\d+)")
_OUTPUT_RE = re.compile(r"tokens_predicted\":\s*(?P<tokens_predicted>\d+)")


def parse_llamacpp_mtp_draft_trace(
    text: str,
    *,
    top_k: int | None = None,
    source_log: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pending_candidates: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    pending_accepts: list[dict[str, int]] = []
    prompt_tokens: int | None = None
    timing_summary: dict[str, Any] | None = None

    lines = text.splitlines()
    line_index = 0
    while line_index < len(lines):
        line_no = line_index + 1
        line = lines[line_index]
        request_match = _REQUEST_RE.search(line)
        if request_match:
            prompt_tokens = int(request_match.group("prompt_tokens"))

        candidate_match = _CANDIDATE_PREFIX_RE.match(line)
        if candidate_match:
            piece = candidate_match.group("piece")
            while not piece.endswith("'") and line_index + 1 < len(lines):
                line_index += 1
                piece += "\n" + lines[line_index]
            if piece.endswith("'"):
                piece = piece[:-1]
            pending_candidates.append(
                {
                    "line": line_no,
                    "seq_id": int(candidate_match.group("seq_id")),
                    "rank": int(candidate_match.group("rank")),
                    "pos": int(candidate_match.group("pos")),
                    "token_id": int(candidate_match.group("token_id")),
                    "prob": float(candidate_match.group("prob")),
                    "piece": piece,
                }
            )
            line_index += 1
            continue

        call_match = _CALL_RE.search(line)
        if call_match:
            candidates = pending_candidates
            if top_k is not None:
                candidates = [item for item in candidates if item["rank"] < top_k]
            calls.append(
                {
                    "line": line_no,
                    "hist_size": int(call_match.group("hist_size")),
                    "call_count": int(call_match.group("call_count")),
                    "generated": int(call_match.group("generated")),
                    "candidates": candidates,
                }
            )
            pending_candidates = []
            line_index += 1
            continue

        accept_match = _ACCEPT_RE.search(line)
        if accept_match:
            pending_accepts.append(
                {
                    "accepted": int(accept_match.group("accepted")),
                    "generated": int(accept_match.group("generated")),
                }
            )
            line_index += 1
            continue

        timing_match = _TIMING_RE.search(line)
        if timing_match:
            timing_summary = {
                "draft_acceptance": float(timing_match.group("acceptance")),
                "draft_n_accepted": int(timing_match.group("accepted")),
                "draft_n": int(timing_match.group("generated")),
            }
        line_index += 1

    for call, accept in zip(calls, pending_accepts, strict=False):
        call["accepted"] = accept["accepted"]
        call["accept_generated"] = accept["generated"]

    all_candidates = [candidate for call in calls for candidate in call["candidates"]]
    observed_top_k = max((candidate["rank"] for candidate in all_candidates), default=-1) + 1
    summary = {
        "draft_call_count": len(calls),
        "candidate_count": len(all_candidates),
        "observed_top_k": observed_top_k,
        "draft_n": timing_summary["draft_n"] if timing_summary else sum(call.get("generated", 0) for call in calls),
        "draft_n_accepted": (
            timing_summary["draft_n_accepted"]
            if timing_summary
            else sum(call.get("accepted", 0) for call in calls)
        ),
    }
    summary["draft_acceptance"] = (
        summary["draft_n_accepted"] / summary["draft_n"] if summary["draft_n"] else None
    )

    result: dict[str, Any] = {
        "schema": 1,
        "kind": "llamacpp_mtp_draft_candidate_trace",
        "source_log": source_log,
        "metadata": metadata or {},
        "prompt_tokens": prompt_tokens,
        "summary": summary,
        "calls": calls,
    }
    if timing_summary is not None:
        result["llamacpp_timing_summary"] = timing_summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="llama.cpp verbose log to parse")
    parser.add_argument("--top-k", type=int, default=None, help="keep only ranks < top-k")
    parser.add_argument("--metadata", type=Path, help="optional JSON object merged into metadata")
    parser.add_argument("--out", type=Path, help="write JSON trace to this path")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
    trace = parse_llamacpp_mtp_draft_trace(
        args.log.read_text(errors="replace"),
        top_k=args.top_k,
        source_log=str(args.log),
        metadata=metadata,
    )
    payload = json.dumps(trace, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
