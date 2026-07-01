#!/usr/bin/env python3
"""Compare hipEngine and llama.cpp MTP proposal token traces.

This is an offline diagnostic for the llama-compat parity lane. hipEngine
``gguf_mtp_bench.py`` artifacts already carry per-cycle draft and output tokens;
llama.cpp artifacts carry matching fields when collected with
``LLAMA_MTP_TOKEN_TRACE=1`` / ``--stage-token-trace`` / ``--token-trace``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProposalRow:
    source: str
    index: int
    cycle: int | None
    generated_draft_tokens: int
    accepted_draft_tokens: int
    visible_output_tokens: int
    draft_token_ids: list[int]
    accepted_token_ids: list[int]
    output_token_ids: list[int]
    bonus_token_id: int | None
    rejected_draft_token_id: int | None


def _ints(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"expected list, got {type(value).__name__}")
    return [int(item) for item in value]


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _hip_row(row: dict[str, Any], *, index: int) -> ProposalRow:
    draft = _ints(row.get("draft_tokens"))
    accepted_n = int(row.get("accepted_draft_tokens") or 0)
    output = _ints(row.get("output_tokens"))
    return ProposalRow(
        source="hipengine",
        index=index,
        cycle=_int_or_none(row.get("cycle")),
        generated_draft_tokens=int(row.get("generated_draft_tokens") or len(draft)),
        accepted_draft_tokens=accepted_n,
        visible_output_tokens=int(row.get("visible_output_tokens") or len(output)),
        draft_token_ids=draft,
        accepted_token_ids=draft[:accepted_n],
        output_token_ids=output,
        bonus_token_id=output[-1] if output else None,
        rejected_draft_token_id=draft[accepted_n] if accepted_n < len(draft) else None,
    )


def _llama_row(row: dict[str, Any], *, index: int) -> ProposalRow:
    draft = _ints(row.get("draft_token_ids"))
    accepted = _ints(row.get("accepted_token_ids"))
    output = _ints(row.get("output_token_ids"))
    return ProposalRow(
        source="llamacpp",
        index=index,
        cycle=_int_or_none(row.get("cycle")),
        generated_draft_tokens=int(row.get("generated_draft_tokens") or len(draft)),
        accepted_draft_tokens=int(row.get("accepted_draft_tokens") or len(accepted)),
        visible_output_tokens=int(row.get("visible_output_tokens") or len(output)),
        draft_token_ids=draft,
        accepted_token_ids=accepted,
        output_token_ids=output,
        bonus_token_id=_int_or_none(row.get("bonus_token_id")),
        rejected_draft_token_id=_int_or_none(row.get("rejected_draft_token_id")),
    )


def load_hipengine_rows(path: Path) -> list[ProposalRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cycles")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not look like a hipEngine gguf_mtp_bench artifact: missing cycles[]")
    return [_hip_row(row, index=index) for index, row in enumerate(rows) if isinstance(row, dict)]


def load_llamacpp_rows(path: Path) -> list[ProposalRow]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        raw_rows = _llamacpp_rows_from_artifact(payload)
    token_rows = [
        row
        for row in raw_rows
        if isinstance(row, dict)
        and "draft_token_ids" in row
        and int(row.get("visible_output_tokens") or 0) > 0
    ]
    return [_llama_row(row, index=index) for index, row in enumerate(token_rows)]


def _llamacpp_rows_from_artifact(payload: dict[str, Any]) -> list[dict[str, Any]]:
    stage = payload.get("stage_timing_summary")
    if isinstance(stage, dict):
        for key in ("measured_excluding_first_task", "all"):
            section = stage.get(key)
            if isinstance(section, dict) and isinstance(section.get("proposal_trace_sample"), list):
                return section["proposal_trace_sample"]
    if isinstance(payload.get("proposal_trace_sample"), list):
        return payload["proposal_trace_sample"]
    raise ValueError("llama.cpp artifact has no proposal_trace_sample; rerun with token tracing enabled")


def compare_rows(hip_rows: list[ProposalRow], llama_rows: list[ProposalRow], *, limit: int | None = None) -> dict[str, Any]:
    n = min(len(hip_rows), len(llama_rows))
    if limit is not None:
        n = min(n, int(limit))
    pairs = list(zip(hip_rows[:n], llama_rows[:n], strict=False))

    exact_draft = 0
    exact_accepted = 0
    exact_output = 0
    accepted_count_match = 0
    first_divergence: dict[str, Any] | None = None

    for index, (hip, llama) in enumerate(pairs):
        draft_match = hip.draft_token_ids == llama.draft_token_ids
        accepted_match = hip.accepted_token_ids == llama.accepted_token_ids
        output_match = hip.output_token_ids == llama.output_token_ids
        count_match = hip.accepted_draft_tokens == llama.accepted_draft_tokens
        exact_draft += int(draft_match)
        exact_accepted += int(accepted_match)
        exact_output += int(output_match)
        accepted_count_match += int(count_match)
        if first_divergence is None and not (draft_match and accepted_match and output_match and count_match):
            first_divergence = {
                "pair_index": index,
                "draft_match": draft_match,
                "accepted_match": accepted_match,
                "output_match": output_match,
                "accepted_count_match": count_match,
                "hipengine": asdict(hip),
                "llamacpp": asdict(llama),
            }

    return {
        "schema": "hipengine.mtp_proposal_trace_compare.v1",
        "compared_rows": n,
        "hipengine_rows": len(hip_rows),
        "llamacpp_rows": len(llama_rows),
        "exact_draft_rows": exact_draft,
        "exact_accepted_rows": exact_accepted,
        "exact_output_rows": exact_output,
        "accepted_count_match_rows": accepted_count_match,
        "exact_draft_rate": (exact_draft / n) if n else None,
        "exact_accepted_rate": (exact_accepted / n) if n else None,
        "exact_output_rate": (exact_output / n) if n else None,
        "accepted_count_match_rate": (accepted_count_match / n) if n else None,
        "hipengine_totals": _totals(hip_rows[:n]),
        "llamacpp_totals": _totals(llama_rows[:n]),
        "first_divergence": first_divergence,
    }


def _totals(rows: list[ProposalRow]) -> dict[str, Any]:
    output = sum(row.visible_output_tokens for row in rows)
    accepted = sum(row.accepted_draft_tokens for row in rows)
    draft = sum(row.generated_draft_tokens for row in rows)
    return {
        "cycles": len(rows),
        "visible_output_tokens": output,
        "accepted_draft_tokens": accepted,
        "generated_draft_tokens": draft,
        "accepted_per_output": (accepted / output) if output else None,
        "draft_acceptance": (accepted / draft) if draft else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipengine", type=Path, required=True, help="hipEngine gguf_mtp_bench JSON artifact")
    parser.add_argument("--llamacpp", type=Path, required=True, help="llama.cpp stage JSONL or wrapper JSON artifact")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = compare_rows(
        load_hipengine_rows(args.hipengine),
        load_llamacpp_rows(args.llamacpp),
        limit=args.limit,
    )
    text = json.dumps(summary, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
