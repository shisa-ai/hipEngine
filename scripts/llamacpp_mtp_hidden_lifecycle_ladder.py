#!/usr/bin/env python3
"""Summarize multiple llama.cpp-vs-hipEngine hidden lifecycle comparisons."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "llamacpp_mtp_hidden_lifecycle_ladder.v1"


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit("--comparison must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise SystemExit("--comparison must be LABEL=PATH")
    return label, Path(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _margin_key(margin: dict[str, Any]) -> str:
    for key, value in margin.items():
        if key.endswith("_minus_") or "_minus_" in key:
            if isinstance(value, int | float):
                return key
    for key in margin:
        if "_minus_" in key:
            return key
    raise SystemExit("token margin has no *_minus_* field")


def _abs_margin_error(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    key = _margin_key(reference)
    if key not in candidate:
        raise SystemExit(f"candidate margin missing {key}")
    return abs(float(reference[key]) - float(candidate[key]))


def _comparison_row(label: str, artifact: dict[str, Any]) -> dict[str, Any]:
    llama_cycle = dict(artifact.get("llamacpp_cycle") or {})
    llama_margin = dict(artifact.get("llamacpp_token_margin") or {})
    margin_key = _margin_key(llama_margin)
    row_index = int((artifact.get("inputs") or {}).get("row", 0))
    hip_rows = []
    for hip in artifact.get("hipengine_comparisons") or []:
        verify_rows = list(hip.get("verify_h_row_comparisons") or [])
        if row_index >= len(verify_rows):
            raise SystemExit(f"{label} hip comparison lacks row {row_index}")
        margin = dict(hip.get("token_margin") or {})
        hip_rows.append(
            {
                "label": hip.get("label"),
                "sampled_tokens": (hip.get("hipengine_cycle") or {}).get("sampled_tokens"),
                "accepted_draft_tokens": (hip.get("hipengine_cycle") or {}).get("accepted_draft_tokens"),
                "prefix_mae": float(hip["prefix_vs_llama_draft_seed_input"]["mean_abs_diff"]),
                "decision_row_mae": float(verify_rows[row_index]["delta"]["mean_abs_diff"]),
                "row_mae": [
                    float(row["delta"]["mean_abs_diff"])
                    for row in verify_rows
                ],
                "token_margin": margin,
                "margin_abs_error": _abs_margin_error(llama_margin, margin),
            }
        )
    nearest_prefix = min(hip_rows, key=lambda row: row["prefix_mae"])
    nearest_decision_row = min(hip_rows, key=lambda row: row["decision_row_mae"])
    nearest_margin = min(hip_rows, key=lambda row: row["margin_abs_error"])
    return {
        "label": label,
        "hip_cycle": label,
        "llamacpp_task_id": llama_cycle.get("task_id"),
        "llamacpp_cycle": llama_cycle.get("cycle"),
        "seed_position": llama_cycle.get("seed_position"),
        "draft_token_ids": llama_cycle.get("draft_token_ids"),
        "llamacpp_accepted_draft_tokens": llama_cycle.get("accepted_draft_tokens"),
        "llamacpp_output_token_ids": llama_cycle.get("output_token_ids"),
        "decision_row": row_index,
        "margin_key": margin_key,
        "llamacpp_margin": float(llama_margin[margin_key]),
        "hipengine": hip_rows,
        "nearest_prefix_label": nearest_prefix["label"],
        "nearest_decision_row_label": nearest_decision_row["label"],
        "nearest_margin_label": nearest_margin["label"],
        "nearest_margin_abs_error": nearest_margin["margin_abs_error"],
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [_parse_named_path(value) for value in args.comparison]
    rows = [_comparison_row(label, _load_json(path)) for label, path in inputs]
    return {
        "schema": SCHEMA,
        "kind": "diagnostic",
        "status": "completed",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": [{"label": label, "path": str(path)} for label, path in inputs],
        "rows": rows,
        "summary": {
            "cycles": len(rows),
            "nearest_prefix_counts": _count_labels(row["nearest_prefix_label"] for row in rows),
            "nearest_decision_row_counts": _count_labels(row["nearest_decision_row_label"] for row in rows),
            "nearest_margin_counts": _count_labels(row["nearest_margin_label"] for row in rows),
        },
    }


def _count_labels(labels: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        help="LABEL=PATH for one hidden lifecycle comparison artifact; repeat in ladder order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifact = build_artifact(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
