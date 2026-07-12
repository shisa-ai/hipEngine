#!/usr/bin/env python3
"""Compare two hipEngine forced-target verifier path artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "gguf_mtp_forced_target_path_compare.v1"


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def _selected_artifact(data: Any, *, cycle: int | None, label: str) -> dict[str, Any]:
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        return data
    if cycle is None:
        raise SystemExit(f"{label} artifact has no top-level result; pass --cycle")
    if isinstance(data, list):
        try:
            selected = data[int(cycle)]
        except IndexError as exc:
            raise SystemExit(f"{label} artifact list has no cycle index {cycle}") from exc
    elif isinstance(data, dict):
        key = str(int(cycle))
        if key not in data:
            raise SystemExit(f"{label} artifact object has no cycle key {key}")
        selected = data[key]
    else:
        raise SystemExit(f"{label} artifact must be a JSON object, list, or cycle-key object")
    if not isinstance(selected, dict) or not isinstance(selected.get("result"), dict):
        raise SystemExit(f"{label} selected cycle is missing a result object")
    return selected


def _result(artifact: dict[str, Any], *, label: str) -> dict[str, Any]:
    result = artifact.get("result")
    if not isinstance(result, dict):
        raise SystemExit(f"{label} artifact is missing result object")
    return result


def _row(result: dict[str, Any], *, row_index: int, label: str) -> dict[str, Any]:
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise SystemExit(f"{label} result is missing rows list")
    for row in rows:
        if int(row.get("row", -1)) == int(row_index):
            return row
    raise SystemExit(f"{label} result has no row {row_index}")


def _as_f32(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise SystemExit(f"{label} has no values")
    return np.ascontiguousarray(array)


def _sha256_16(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()[:16]


def _diff_metrics(reference_values: Any, candidate_values: Any) -> dict[str, Any]:
    reference = _as_f32(reference_values, label="reference")
    candidate = _as_f32(candidate_values, label="candidate")
    if reference.shape != candidate.shape:
        raise SystemExit(f"shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    reference_norm = float(np.linalg.norm(reference64))
    candidate_norm = float(np.linalg.norm(candidate64))
    return {
        "size": int(reference.size),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "cosine": float(np.dot(reference64, candidate64) / (reference_norm * candidate_norm))
        if reference_norm and candidate_norm
        else None,
        "reference_sha256_16": _sha256_16(reference),
        "candidate_sha256_16": _sha256_16(candidate),
        "candidate_minus_reference_first8": [float(value) for value in diff[:8]],
    }


def _candidate_map(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    scores = row.get("candidate_scores")
    if not isinstance(scores, list):
        raise SystemExit("row is missing candidate_scores list")
    return {int(score["token_id"]): score for score in scores}


def _margin(row: dict[str, Any], *, token_a: int, token_b: int) -> dict[str, Any]:
    candidates = _candidate_map(row)
    if token_a not in candidates or token_b not in candidates:
        raise SystemExit(f"candidate scores must include {token_a} and {token_b}")
    logit_a = float(candidates[token_a]["logit"])
    logit_b = float(candidates[token_b]["logit"])
    return {
        "sampled_token": row.get("sampled_token"),
        "logits": {str(token_a): logit_a, str(token_b): logit_b},
        f"{token_a}_minus_{token_b}": float(logit_a - logit_b),
        "ranks": {str(token_a): candidates[token_a].get("rank"), str(token_b): candidates[token_b].get("rank")},
    }


def _common_layers(reference_row: dict[str, Any], candidate_row: dict[str, Any]) -> list[int]:
    reference_layers = reference_row.get("layer_output_hidden_values")
    candidate_layers = candidate_row.get("layer_output_hidden_values")
    if not isinstance(reference_layers, dict) or not isinstance(candidate_layers, dict):
        return []
    common = set(reference_layers) & set(candidate_layers)
    return sorted(int(layer) for layer in common)


def _layer_comparisons(
    reference_row: dict[str, Any],
    candidate_row: dict[str, Any],
    *,
    layers: list[int],
) -> list[dict[str, Any]]:
    reference_layers = reference_row.get("layer_output_hidden_values")
    candidate_layers = candidate_row.get("layer_output_hidden_values")
    if not isinstance(reference_layers, dict):
        raise SystemExit("reference row is missing layer_output_hidden_values")
    if not isinstance(candidate_layers, dict):
        raise SystemExit("candidate row is missing layer_output_hidden_values")

    comparisons: list[dict[str, Any]] = []
    for layer in layers:
        key = str(int(layer))
        if key not in reference_layers:
            raise SystemExit(f"reference row is missing layer_output_hidden_values.{key}")
        if key not in candidate_layers:
            raise SystemExit(f"candidate row is missing layer_output_hidden_values.{key}")
        comparisons.append(
            {
                "name": f"layer_output_hidden_{layer}",
                "layer": int(layer),
                "delta": _diff_metrics(reference_layers[key], candidate_layers[key]),
            }
        )
    return comparisons


def _captures_by_layer_row(
    result: dict[str, Any],
    *,
    source: str,
    label: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    captures = result.get(source)
    if captures is None:
        return {}
    if not isinstance(captures, list):
        raise SystemExit(f"{label} result {source} must be a list")
    mapped: dict[tuple[int, int], dict[str, Any]] = {}
    for capture in captures:
        if not isinstance(capture, dict):
            continue
        if "layer" not in capture or "row" not in capture:
            continue
        mapped[(int(capture["layer"]), int(capture["row"]))] = capture
    return mapped


def _capture_values(capture: dict[str, Any], *, label: str) -> dict[str, Any]:
    values = capture.get("values")
    if not isinstance(values, dict):
        raise SystemExit(f"{label} boundary capture is missing values")
    return values


def _boundary_layers(
    reference_captures: dict[tuple[int, int], dict[str, Any]],
    candidate_captures: dict[tuple[int, int], dict[str, Any]],
    *,
    row_index: int,
    requested_layers: list[int],
) -> list[int]:
    if requested_layers:
        return requested_layers
    common = {
        layer
        for layer, row in set(reference_captures) & set(candidate_captures)
        if int(row) == int(row_index)
    }
    return sorted(common)


def _boundary_comparisons(
    reference_result: dict[str, Any],
    candidate_result: dict[str, Any],
    *,
    source: str,
    row_index: int,
    layers: list[int],
    ignored_values: set[str],
) -> list[dict[str, Any]]:
    reference_captures = _captures_by_layer_row(reference_result, source=source, label="reference")
    candidate_captures = _captures_by_layer_row(candidate_result, source=source, label="candidate")
    selected_layers = _boundary_layers(
        reference_captures,
        candidate_captures,
        row_index=row_index,
        requested_layers=layers,
    )
    comparisons: list[dict[str, Any]] = []
    for layer in selected_layers:
        key = (int(layer), int(row_index))
        if key not in reference_captures:
            raise SystemExit(f"reference {source} has no layer {layer} row {row_index}")
        if key not in candidate_captures:
            raise SystemExit(f"candidate {source} has no layer {layer} row {row_index}")
        reference_values = _capture_values(reference_captures[key], label="reference")
        candidate_values = _capture_values(candidate_captures[key], label="candidate")
        value_rows: list[dict[str, Any]] = []
        for name in sorted(set(reference_values) & set(candidate_values)):
            if str(name) in ignored_values:
                continue
            try:
                delta = _diff_metrics(reference_values[name], candidate_values[name])
            except SystemExit:
                continue
            value_rows.append({"name": str(name), "delta": delta})
        comparisons.append(
            {
                "source": source,
                "layer": int(layer),
                "row": int(row_index),
                "values": value_rows,
            }
        )
    return comparisons


def _optional_vector_comparisons(reference_row: dict[str, Any], candidate_row: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        ("hidden_seed_values", "hidden_seed"),
        ("pre_output_norm_hidden_values", "pre_output_norm_hidden"),
    )
    comparisons: list[dict[str, Any]] = []
    for key, name in keys:
        if key in reference_row and key in candidate_row:
            comparisons.append({"name": name, "delta": _diff_metrics(reference_row[key], candidate_row[key])})
    return comparisons


def _path_summary(artifact: dict[str, Any], result: dict[str, Any], row: dict[str, Any], *, path: Path, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "cycle": (artifact.get("probe") or {}).get("cycle"),
        "capture_linear_state_rows": result.get("capture_linear_state_rows"),
        "target_block_verify_mode": result.get("target_block_verify_mode"),
        "replay_target_block_verify_mode": result.get("replay_target_block_verify_mode"),
        "sampled_tokens": result.get("sampled_tokens"),
        "accepted_draft_tokens": result.get("accepted_draft_tokens"),
        "row": {
            "row": row.get("row"),
            "position": row.get("position"),
            "input_token": row.get("input_token"),
            "sampled_token": row.get("sampled_token"),
        },
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    token_ids = _parse_int_list(args.candidate_tokens)
    if len(token_ids) != 2:
        raise SystemExit("--candidate-tokens must contain exactly two comma-separated token IDs")

    reference_artifact = _selected_artifact(
        _json_load(args.reference),
        cycle=args.cycle,
        label=args.reference_label,
    )
    candidate_artifact = _selected_artifact(
        _json_load(args.candidate),
        cycle=args.cycle,
        label=args.candidate_label,
    )
    reference_result = _result(reference_artifact, label=args.reference_label)
    candidate_result = _result(candidate_artifact, label=args.candidate_label)
    reference_row = _row(reference_result, row_index=int(args.row), label=args.reference_label)
    candidate_row = _row(candidate_result, row_index=int(args.row), label=args.candidate_label)

    layers = _parse_int_list(args.layers)
    if not layers:
        layers = _common_layers(reference_row, candidate_row)
    if not layers:
        raise SystemExit("no layer-output comparisons available; pass --layers or capture layer_output_hidden_values")

    reference_margin = _margin(reference_row, token_a=token_ids[0], token_b=token_ids[1])
    candidate_margin = _margin(candidate_row, token_a=token_ids[0], token_b=token_ids[1])
    margin_key = f"{token_ids[0]}_minus_{token_ids[1]}"
    layer_comparisons = _layer_comparisons(reference_row, candidate_row, layers=layers)
    vector_comparisons = _optional_vector_comparisons(reference_row, candidate_row)
    boundary_layers = _parse_int_list(args.boundary_layers)
    ignored_boundary_values = _parse_str_list(args.ignore_boundary_values)
    boundary_comparisons = (
        _boundary_comparisons(
            reference_result,
            candidate_result,
            source=args.boundary_source,
            row_index=int(args.row),
            layers=boundary_layers,
            ignored_values=ignored_boundary_values,
        )
        if args.boundary_layers is not None
        else []
    )

    first_layer_over_threshold = next(
        (
            {
                "layer": row["layer"],
                "mean_abs_diff": row["delta"]["mean_abs_diff"],
            }
            for row in layer_comparisons
            if float(row["delta"]["mean_abs_diff"]) >= float(args.threshold)
        ),
        None,
    )
    largest_layer = max(
        (
            {"layer": row["layer"], "mean_abs_diff": row["delta"]["mean_abs_diff"]}
            for row in layer_comparisons
        ),
        key=lambda item: float(item["mean_abs_diff"]),
    )
    boundary_value_rows = [
        {
            "layer": int(comparison["layer"]),
            "name": str(value["name"]),
            "mean_abs_diff": float(value["delta"]["mean_abs_diff"]),
        }
        for comparison in boundary_comparisons
        for value in comparison["values"]
    ]
    largest_boundary_value = (
        max(boundary_value_rows, key=lambda item: float(item["mean_abs_diff"]))
        if boundary_value_rows
        else None
    )
    boundary_layer_out = [
        {
            "layer": int(comparison["layer"]),
            "mean_abs_diff": float(value["delta"]["mean_abs_diff"]),
        }
        for comparison in boundary_comparisons
        for value in comparison["values"]
        if value["name"] == "layer_out"
    ]

    return {
        "schema": SCHEMA,
        "kind": "diagnostic",
        "status": "completed",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "reference": str(args.reference),
            "candidate": str(args.candidate),
            "reference_label": args.reference_label,
            "candidate_label": args.candidate_label,
            "cycle": args.cycle,
            "row": int(args.row),
            "candidate_tokens": token_ids,
            "layers": layers,
            "boundary_source": args.boundary_source,
            "boundary_layers": boundary_layers,
            "ignored_boundary_values": sorted(ignored_boundary_values),
            "threshold": float(args.threshold),
        },
        "paths": {
            "reference": _path_summary(
                reference_artifact,
                reference_result,
                reference_row,
                path=args.reference,
                label=args.reference_label,
            ),
            "candidate": _path_summary(
                candidate_artifact,
                candidate_result,
                candidate_row,
                path=args.candidate,
                label=args.candidate_label,
            ),
        },
        "token_margin": {
            "token_a": token_ids[0],
            "token_b": token_ids[1],
            "margin_key": margin_key,
            "reference": reference_margin,
            "candidate": candidate_margin,
            "candidate_minus_reference": float(candidate_margin[margin_key] - reference_margin[margin_key]),
        },
        "sampled_token_changed": reference_row.get("sampled_token") != candidate_row.get("sampled_token"),
        "accepted_draft_tokens_delta": (
            int(candidate_result.get("accepted_draft_tokens", 0))
            - int(reference_result.get("accepted_draft_tokens", 0))
        ),
        "comparisons": layer_comparisons + vector_comparisons,
        "boundary_comparisons": boundary_comparisons,
        "summary": {
            "first_layer_mean_abs_diff_ge_threshold": first_layer_over_threshold,
            "largest_layer_mean_abs_diff": largest_layer,
            "largest_boundary_value_mean_abs_diff": largest_boundary_value,
            "boundary_layer_out_mean_abs_diff": boundary_layer_out,
            "pre_output_norm_mean_abs_diff": next(
                (
                    row["delta"]["mean_abs_diff"]
                    for row in vector_comparisons
                    if row["name"] == "pre_output_norm_hidden"
                ),
                None,
            ),
            "hidden_seed_mean_abs_diff": next(
                (row["delta"]["mean_abs_diff"] for row in vector_comparisons if row["name"] == "hidden_seed"),
                None,
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument(
        "--candidate-tokens",
        required=True,
        help="Two comma-separated token IDs. The artifact reports token_a_minus_token_b logits.",
    )
    parser.add_argument("--layers", help="Comma-separated layer IDs. Defaults to common captured layers.")
    parser.add_argument(
        "--boundary-layers",
        help=(
            "Comma-separated scored/isolated boundary layer IDs to compare. "
            "When omitted, boundary captures are not compared."
        ),
    )
    parser.add_argument(
        "--boundary-source",
        choices=("scored_layer_boundary_captures", "layer_boundary_captures"),
        default="scored_layer_boundary_captures",
    )
    parser.add_argument(
        "--ignore-boundary-values",
        help="Comma-separated boundary value names to exclude from comparisons.",
    )
    parser.add_argument("--threshold", type=float, default=1.0e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = build_artifact(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
