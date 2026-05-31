#!/usr/bin/env python3
"""Compare compact fields from Qwen3.5/PARO hidden-bisect artifacts.

The hidden-bisect artifacts can be very large.  This helper extracts the
projection bit-drift rollups from two or more artifacts and emits a small JSON
comparison so C2.3 handoffs can compare complementary diagnostic routes without
manually diffing full per-step traces.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"artifact {path} must contain a JSON object")
    return payload


def _parse_artifact_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("artifact label must not be empty")
        path = Path(raw_path.strip())
    else:
        path = Path(value)
        label = path.stem
    if not str(path):
        raise ValueError("artifact path must not be empty")
    return label, path


def _first_over_atol_layer_limit_value(rollup: dict[str, Any]) -> int | None:
    entry = rollup.get("first_over_atol_layer_limit")
    if not isinstance(entry, dict):
        return None
    try:
        return int(entry["layer_limit"])
    except (KeyError, TypeError, ValueError):
        return None


def _stage_list(entry: dict[str, Any] | None, key: str) -> list[str]:
    if not isinstance(entry, dict):
        return []
    stages = entry.get(key, [])
    if not isinstance(stages, list):
        return []
    return [str(stage) for stage in stages]


def _small_drift_record(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    compact: dict[str, Any] = {}
    for key in (
        "layer_limit",
        "decode_step",
        "generated_index",
        "layer_index",
        "stage",
        "row",
        "comparison_kind",
        "passed_under_atol",
        "bit_mismatch",
        "max_abs",
        "max_abs_flat_index",
        "max_abs_index",
        "elements_over_atol",
    ):
        if key in record:
            compact[key] = record[key]
    return compact


def _projection_rollup(payload: dict[str, Any]) -> dict[str, Any]:
    correctness = payload.get("correctness", {})
    if not isinstance(correctness, dict):
        return {}
    rollup = correctness.get("decode_linear_projection_bit_drift_summary", {})
    return rollup if isinstance(rollup, dict) else {}


def _layer_limit_entries(rollup: dict[str, Any]) -> dict[int, dict[str, Any]]:
    entries: dict[int, dict[str, Any]] = {}
    raw_entries = rollup.get("layer_limits", [])
    if not isinstance(raw_entries, list):
        return entries
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        try:
            layer_limit = int(raw_entry["layer_limit"])
        except (KeyError, TypeError, ValueError):
            continue
        entries[layer_limit] = raw_entry
    return entries


def _artifact_projection_summary(label: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    correctness = payload.get("correctness", {})
    correctness = correctness if isinstance(correctness, dict) else {}
    rollup = _projection_rollup(payload)
    limit_entries = _layer_limit_entries(rollup)
    return {
        "label": label,
        "artifact_path": str(path),
        "status": payload.get("status"),
        "hidden_passed": correctness.get("hidden_passed"),
        "token_passed": correctness.get("token_passed"),
        "projection": {
            "bit_exact": rollup.get("bit_exact"),
            "passed_under_atol": rollup.get("passed_under_atol"),
            "drift_stages": _stage_list(rollup, "drift_stages"),
            "under_atol_drift_stages": _stage_list(rollup, "under_atol_drift_stages"),
            "over_atol_drift_stages": _stage_list(rollup, "over_atol_drift_stages"),
            "first_over_atol_layer_limit": _first_over_atol_layer_limit_value(rollup),
            "first_over_atol_drift": _small_drift_record(rollup.get("first_over_atol_drift")),
            "layer_limits": [
                {
                    "layer_limit": layer_limit,
                    "drift_stages": _stage_list(entry, "drift_stages"),
                    "drift_stage_count": int(entry.get("drift_stage_count", len(_stage_list(entry, "drift_stages")))),
                    "under_atol_drift_stages": _stage_list(entry, "under_atol_drift_stages"),
                    "under_atol_drift_stage_count": int(
                        entry.get("under_atol_drift_stage_count", len(_stage_list(entry, "under_atol_drift_stages")))
                    ),
                    "over_atol_drift_stages": _stage_list(entry, "over_atol_drift_stages"),
                    "over_atol_drift_stage_count": int(
                        entry.get("over_atol_drift_stage_count", len(_stage_list(entry, "over_atol_drift_stages")))
                    ),
                    "first_over_atol_drift": _small_drift_record(entry.get("first_over_atol_drift")),
                }
                for layer_limit, entry in sorted(limit_entries.items())
            ],
        },
    }


def _limit_comparison(labels: Sequence[str], summaries: dict[str, dict[str, Any]], layer_limit: int) -> dict[str, Any]:
    per_artifact: dict[str, dict[str, Any]] = {}
    drift_signatures: dict[str, tuple[str, ...]] = {}
    over_atol_signatures: dict[str, tuple[str, ...]] = {}
    for label in labels:
        projection = summaries[label]["projection"]
        entries = {
            int(entry["layer_limit"]): entry
            for entry in projection.get("layer_limits", [])
            if isinstance(entry, dict) and "layer_limit" in entry
        }
        entry = entries.get(layer_limit, {})
        drift_stages = _stage_list(entry, "drift_stages")
        under_atol_stages = _stage_list(entry, "under_atol_drift_stages")
        over_atol_stages = _stage_list(entry, "over_atol_drift_stages")
        first_over = _small_drift_record(entry.get("first_over_atol_drift")) if isinstance(entry, dict) else None
        per_artifact[label] = {
            "drift_stages": drift_stages,
            "under_atol_drift_stages": under_atol_stages,
            "over_atol_drift_stages": over_atol_stages,
            "first_over_atol_drift": first_over,
        }
        drift_signatures[label] = tuple(drift_stages)
        over_atol_signatures[label] = tuple(over_atol_stages)
    drift_agrees = len(set(drift_signatures.values())) <= 1
    over_atol_agrees = len(set(over_atol_signatures.values())) <= 1
    return {
        "layer_limit": layer_limit,
        "per_artifact": per_artifact,
        "drift_agrees": drift_agrees,
        "over_atol_agrees": over_atol_agrees,
    }


def compare_artifacts(artifacts: Sequence[tuple[str, Path, dict[str, Any]]]) -> dict[str, Any]:
    if len(artifacts) < 2:
        raise ValueError("at least two artifacts are required for comparison")
    labels = [label for label, _, _ in artifacts]
    if len(set(labels)) != len(labels):
        raise ValueError("artifact labels must be unique")

    summaries = {
        label: _artifact_projection_summary(label, path, payload)
        for label, path, payload in artifacts
    }
    limit_sets: list[set[int]] = []
    for summary in summaries.values():
        limit_sets.append(
            {
                int(entry["layer_limit"])
                for entry in summary["projection"].get("layer_limits", [])
                if isinstance(entry, dict) and "layer_limit" in entry
            }
        )
    common_limits = sorted(set.intersection(*limit_sets)) if limit_sets else []
    all_limits = sorted(set.union(*limit_sets)) if limit_sets else []
    per_limit = [_limit_comparison(labels, summaries, layer_limit) for layer_limit in all_limits]
    first_diverging = next(
        (entry["layer_limit"] for entry in per_limit if not entry["drift_agrees"] or not entry["over_atol_agrees"]),
        None,
    )
    first_over_by_label = {
        label: summaries[label]["projection"].get("first_over_atol_layer_limit") for label in labels
    }
    first_over_values = tuple(first_over_by_label.values())
    projection_drift_agreement = all(entry["drift_agrees"] for entry in per_limit)
    projection_over_atol_agreement = all(entry["over_atol_agrees"] for entry in per_limit)
    return {
        "schema": 1,
        "mode": "qwen35_batch_hidden_artifact_compare",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "artifacts": [summaries[label] for label in labels],
        "comparison": {
            "labels": labels,
            "common_layer_limits": common_limits,
            "all_layer_limits": all_limits,
            "first_over_atol_layer_limits_by_label": first_over_by_label,
            "labels_agree_on_first_over_atol_layer_limit": len(set(first_over_values)) <= 1,
            "projection_drift_agreement": projection_drift_agreement,
            "projection_over_atol_agreement": projection_over_atol_agreement,
            "first_diverging_layer_limit": first_diverging,
            "hidden_passed_all": all(summary.get("hidden_passed") is True for summary in summaries.values()),
            "token_passed_all": all(summary.get("token_passed") is True for summary in summaries.values()),
            "all_statuses_eq_ok": all(summary.get("status") == "eq_ok" for summary in summaries.values()),
            "layer_limits": per_limit,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_artifacts = getattr(args, "artifact", None) or getattr(args, "artifacts", None)
    if not raw_artifacts:
        raise ValueError("at least two --artifact entries are required")
    parsed: list[tuple[str, Path, dict[str, Any]]] = []
    for raw in raw_artifacts:
        label, path = _parse_artifact_arg(str(raw))
        parsed.append((label, path, _load_json(path)))
    payload = compare_artifacts(parsed)
    json_path = getattr(args, "json", None)
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Artifact to compare, as LABEL=PATH or PATH. Repeat at least twice.",
    )
    parser.add_argument("--json", type=Path, help="Optional output JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run(args)
    if args.json is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
