#!/usr/bin/env python3
"""Roll up paired same-server AR/MTP bench arms into one compact artifact.

``scripts/gguf_mtp_server_pair_bench.py`` measures one arm per server process:
the automatic MTP route, or the true no-MTP autoregressive control selected by
the request's own ``speculative_mtp`` field. This assembler joins the arms of
one or more server processes (for example a candidate verifier route and its
strict control) into the single compact row set a benchmark rollup cites:

* the paired rate and its ratio per shape or prompt,
* the routing block each arm reported about itself, so an arm that never
  engaged speculation cannot be read as a speculative rate,
* text identity and exact generated-id identity between the arms,
* cross-arm differences, which is how a candidate arm proves it changed the
  route and not only its configuration.

usage:
    python3 scripts/gguf_mtp_pair_rollup.py \
      --pair staged=/tmp/staged-mtp.json,/tmp/staged-ar.json \
      --pair rowwise=/tmp/rowwise-mtp.json,/tmp/rowwise-ar.json \
      --label "gfx1151 Qwen3.8-27B Q4_K_M long-context verifier route" \
      --out benchmarks/results/<artifact>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


def _rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Shape-keyed or prompt-keyed rows, whichever the arm measured."""
    if payload.get("shapes"):
        return {str(key): value for key, value in payload["shapes"].items()}
    return {str(key): value for key, value in (payload.get("prompts") or {}).items()}


def _median(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def _identity(first: Mapping[str, Any] | None, second: Mapping[str, Any] | None) -> dict:
    """Cross-arm greedy identity for one row pair."""
    first = first or {}
    second = second or {}
    first_ids = list(first.get("generated_ids") or [])
    second_ids = list(second.get("generated_ids") or [])
    divergence = None
    for index, (left, right) in enumerate(zip(first_ids, second_ids)):
        if left != right:
            divergence = index
            break
    if divergence is None and len(first_ids) != len(second_ids):
        divergence = min(len(first_ids), len(second_ids))
    return {
        "ids_exact": bool(first_ids) and first_ids == second_ids,
        "ids_compared": min(len(first_ids), len(second_ids)),
        "first_divergent_index": divergence,
        "text_exact": bool(first.get("text")) and first.get("text") == second.get("text"),
        "chars_compared": min(len(first.get("text") or ""), len(second.get("text") or "")),
    }


def _arm_route(row: Mapping[str, Any]) -> dict:
    """The routing/accounting block an arm reported for a row."""
    route = dict(row.get("route") or {})
    if row.get("identity"):
        route["identity_route"] = row["identity"].get("effective_route")
        route["identity_mtp_used"] = row["identity"].get("mtp_used")
    return route


def build(pairs: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]]) -> dict:
    """Join ``(label, mtp_payload, ar_payload)`` triples into one row set."""
    arms: dict[str, Any] = {}
    shapes: list[str] = []
    for label, mtp, ar in pairs:
        mtp_rows = _rows(mtp)
        ar_rows = _rows(ar)
        joined: dict[str, Any] = {}
        for key in mtp_rows:
            if key not in ar_rows:
                raise SystemExit(f"{label}: shape {key} missing from the AR arm")
            if key not in shapes:
                shapes.append(key)
            mtp_row = mtp_rows[key]
            ar_row = ar_rows[key]
            mtp_rate = _median([run.get("decode_tok_s") for run in mtp_row.get("runs") or []])
            ar_rate = _median([run.get("decode_tok_s") for run in ar_row.get("runs") or []])
            joined[key] = {
                "prompt_tokens": mtp_row.get("prompt_tokens"),
                "ar_prompt_tokens": ar_row.get("prompt_tokens"),
                "completion_tokens": mtp_row.get("completion_tokens"),
                "ar_completion_tokens": ar_row.get("completion_tokens"),
                "mtp_decode_tok_s": mtp_rate,
                "ar_decode_tok_s": ar_rate,
                "ratio": (mtp_rate / ar_rate) if mtp_rate and ar_rate else None,
                "mtp_decode_tok_s_cv": mtp_row.get("decode_tok_s_cv"),
                "ar_decode_tok_s_cv": ar_row.get("decode_tok_s_cv"),
                "mtp_ttft_ms": mtp_row.get("ttft_ms_median"),
                "ar_ttft_ms": ar_row.get("ttft_ms_median"),
                "mtp_prefill_tok_s": mtp_row.get("prefill_tok_s_median"),
                "ar_prefill_tok_s": ar_row.get("prefill_tok_s_median"),
                "mtp_route": _arm_route(mtp_row),
                "ar_route": _arm_route(ar_row),
                "identity": _identity(mtp_row.get("identity"), ar_row.get("identity")),
            }
        arms[label] = {
            "mtp_tag": mtp.get("tag"),
            "ar_tag": ar.get("tag"),
            "request_extras": mtp.get("request_extras"),
            "served_models": mtp.get("served_models"),
            "rows": joined,
        }
    return {"arms": arms, "row_order": shapes}


def compare_candidate(artifact: Mapping[str, Any], candidate: str, control: str) -> dict:
    """Per-row candidate-versus-control deltas, including route engagement."""
    left = (artifact["arms"].get(candidate) or {}).get("rows") or {}
    right = (artifact["arms"].get(control) or {}).get("rows") or {}
    compared: dict[str, Any] = {}
    for key, row in left.items():
        other = right.get(key)
        if other is None:
            continue
        compared[key] = {
            "ratio_candidate": row.get("ratio"),
            "ratio_control": other.get("ratio"),
            "candidate_mtp_used": (row.get("mtp_route") or {}).get("mtp_used"),
            "control_mtp_used": (other.get("mtp_route") or {}).get("mtp_used"),
            "candidate_mtp_coverage": (row.get("mtp_route") or {}).get("mtp_coverage"),
            "control_mtp_coverage": (other.get("mtp_route") or {}).get("mtp_coverage"),
            "candidate_speculative_cycles": (row.get("mtp_route") or {}).get(
                "speculative_cycles"
            ),
            "control_speculative_cycles": (other.get("mtp_route") or {}).get(
                "speculative_cycles"
            ),
            "candidate_ar_rate": row.get("ar_decode_tok_s"),
            "control_ar_rate": other.get("ar_decode_tok_s"),
            "ar_rate_delta_pct": (
                (row["ar_decode_tok_s"] - other["ar_decode_tok_s"])
                / other["ar_decode_tok_s"]
                * 100.0
                if row.get("ar_decode_tok_s") and other.get("ar_decode_tok_s")
                else None
            ),
        }
    return compared


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LABEL=MTP_JSON,AR_JSON",
        help="One server process's two arms: the MTP arm artifact and the true-AR arm.",
    )
    parser.add_argument("--label", required=True, help="Row-set description for the artifact.")
    parser.add_argument("--note", default=None, help="Free-form provenance note.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--commit", default=None)
    parser.add_argument(
        "--candidate",
        default=None,
        help="Arm label to compare against --control (cross-arm route check).",
    )
    parser.add_argument("--control", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    pairs = []
    for spec in args.pair:
        if "=" not in spec:
            raise SystemExit(f"--pair requires LABEL=MTP,AR: {spec}")
        label, paths = spec.split("=", 1)
        mtp_path, _, ar_path = paths.partition(",")
        if not mtp_path or not ar_path:
            raise SystemExit(f"--pair requires both artifacts: {spec}")
        pairs.append(
            (
                label,
                json.loads(Path(mtp_path).read_text(encoding="utf-8")),
                json.loads(Path(ar_path).read_text(encoding="utf-8")),
            )
        )
    if not pairs:
        raise SystemExit("at least one --pair is required")

    artifact = build(pairs)
    artifact.update(
        {
            "label": args.label,
            "note": args.note,
            "host": args.host,
            "gpu": args.gpu,
            "model": args.model,
            "commit": args.commit or _git_commit(),
            "command": ["python3", "scripts/gguf_mtp_pair_rollup.py", *argv] if argv else None,
        }
    )
    if args.candidate and args.control:
        artifact["candidate_vs_control"] = compare_candidate(
            artifact, args.candidate, args.control
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for label, arm in artifact["arms"].items():
        for key, row in arm["rows"].items():
            print(
                f"  [{label}] {key}: AR {row['ar_decode_tok_s']:.2f} -> "
                f"MTP {row['mtp_decode_tok_s']:.2f} tok/s ({row['ratio']:.3f}x) "
                f"coverage {(row['mtp_route'] or {}).get('mtp_coverage')} "
                f"ids_exact {row['identity']['ids_exact']}"
            )
    return 0


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
