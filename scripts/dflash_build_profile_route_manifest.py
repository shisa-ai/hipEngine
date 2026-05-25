#!/usr/bin/env python3
"""Build a zero-probe DFlash profile-route manifest from prior bench rows.

The online adaptive probe is too expensive for short DFlash decode horizons.  A
profile/history route keeps startup cost at zero: prompts whose previous exact
chain row beat same-session AR are routed to ``chain``; everything else falls
back to plain AR.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _rows_from_artifact(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    measurements = artifact.get("measurements")
    if isinstance(measurements, dict) and isinstance(measurements.get("rows"), list):
        return [row for row in measurements["rows"] if isinstance(row, dict)]
    rows = artifact.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    raise ValueError("input artifact does not contain measurements.rows")


def _prompt_id(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, dict):
        return ""
    return str(prompt.get("id") or "").strip()


def _prompt_group(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, dict):
        return ""
    return str(prompt.get("benchmark_group") or prompt.get("category") or "").strip()


def _row_speedup(row: dict[str, Any]) -> float:
    spec = row.get("spec")
    if isinstance(spec, dict):
        value = spec.get("speedup_vs_ar")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        ar = row.get("ar")
        if isinstance(ar, dict):
            spec_tok_s = spec.get("decode_tok_s")
            ar_tok_s = ar.get("decode_tok_s")
            if isinstance(spec_tok_s, (int, float)) and isinstance(ar_tok_s, (int, float)) and float(ar_tok_s) > 0:
                return float(spec_tok_s) / float(ar_tok_s)
    value = row.get("speedup_vs_ar")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    raise ValueError(f"row for prompt {_prompt_id(row)!r} has no finite speedup_vs_ar")


def _row_exact(row: dict[str, Any]) -> bool:
    correctness = row.get("correctness")
    if isinstance(correctness, dict):
        return bool(correctness.get("passed") and correctness.get("exact_match_ar"))
    spec = row.get("spec")
    if isinstance(spec, dict):
        return bool(spec.get("exact_match_ar", False))
    return False


def _row_is_chain(row: dict[str, Any]) -> bool:
    config = row.get("config")
    if not isinstance(config, dict):
        return True
    profile_route = config.get("profile_route")
    proposal = str(config.get("proposal_mode") or "").lower()
    verify = str(config.get("verify_mode") or "").lower()
    return (profile_route in {None, "", "chain"}) and proposal == "chain" and verify in {"verify_chain", "chain"}


def build_manifest(
    artifact: dict[str, Any],
    *,
    source: str,
    min_chain_speedup: float,
    default_route: str = "ar",
) -> dict[str, Any]:
    if min_chain_speedup <= 0.0:
        raise ValueError("min_chain_speedup must be positive")
    if default_route not in {"ar", "spec", "chain", "tree"}:
        raise ValueError("default_route must be ar, spec, chain, or tree")

    routes: dict[str, str] = {}
    row_evidence: list[dict[str, Any]] = []
    skipped = 0
    for row in _rows_from_artifact(artifact):
        prompt_id = _prompt_id(row)
        if not prompt_id:
            skipped += 1
            continue
        exact = _row_exact(row)
        chain_shape = _row_is_chain(row)
        speedup = _row_speedup(row)
        route = "chain" if exact and chain_shape and speedup >= min_chain_speedup else default_route
        if route != default_route:
            routes[prompt_id] = route
        row_evidence.append(
            {
                "prompt_id": prompt_id,
                "benchmark_group": _prompt_group(row) or None,
                "route": route,
                "speedup_vs_ar": speedup,
                "exact_match_ar": exact,
                "chain_shape": chain_shape,
            }
        )

    return {
        "default": default_route,
        "routes": routes,
        "description": "Zero-probe DFlash profile/history route generated from exact same-session bench rows.",
        "source_artifact": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_chain_speedup": min_chain_speedup,
        "summary": {
            "rows_seen": len(row_evidence) + skipped,
            "rows_with_prompt_id": len(row_evidence),
            "skipped_rows": skipped,
            "chain_routes": sum(1 for value in routes.values() if value == "chain"),
            "default_route": default_route,
        },
        "row_evidence": row_evidence,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="DFlash bench artifact JSON to mine for exact row speedups")
    parser.add_argument("--output", type=Path, required=True, help="Route manifest JSON to write")
    parser.add_argument("--min-chain-speedup", type=float, default=1.02, help="Route a prompt to chain only when prior exact chain row is at least this fast vs AR")
    parser.add_argument("--default-route", choices=("ar", "spec", "chain", "tree"), default="ar", help="Fallback route for prompts not selected for chain")
    args = parser.parse_args(argv)

    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("input artifact must be a JSON object")
    manifest = build_manifest(
        artifact,
        source=str(args.input),
        min_chain_speedup=float(args.min_chain_speedup),
        default_route=args.default_route,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
