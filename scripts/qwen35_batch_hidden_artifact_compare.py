#!/usr/bin/env python3
"""Compare compact fields from Qwen3.5/PARO hidden-bisect artifacts.

The hidden-bisect artifacts can be very large.  This helper extracts the
projection bit-drift rollups from two or more artifacts and emits a small JSON
comparison so C2.3 handoffs can compare complementary diagnostic routes without
manually diffing full per-step traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scripts.qwen35_batch_artifact_schema import _load_payload


def _payload_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def _load_json(path: Path) -> dict[str, Any]:
    return dict(_load_payload(path))


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


def _drift_location(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        key: record[key]
        for key in (
            "layer_limit",
            "decode_step",
            "generated_index",
            "layer_index",
            "stage",
            "row",
            "comparison_kind",
            "max_abs_flat_index",
            "max_abs_index",
            "elements_over_atol",
        )
        if key in record
    }


def _drift_bit_mismatch(record: dict[str, Any] | None) -> int | None:
    if record is None or "bit_mismatch" not in record:
        return None
    try:
        return int(record["bit_mismatch"])
    except (TypeError, ValueError):
        return None


def _record_key(record: dict[str, Any] | None) -> str:
    return json.dumps(record, sort_keys=True, allow_nan=False)


def _records_agree(records: Sequence[dict[str, Any] | None]) -> bool:
    return len({_record_key(record) for record in records}) <= 1


def _int_delta(values: Sequence[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present) - min(present)


def _projection_difference_kinds(
    *,
    drift_agrees: bool,
    over_atol_agrees: bool,
    first_over_layer_limit_agrees: bool | None = None,
    first_over_location_agrees: bool | None = None,
    first_over_record_agrees: bool | None = None,
    first_over_bit_mismatch_delta: int | None = None,
) -> list[str]:
    kinds: list[str] = []
    if not drift_agrees:
        kinds.append("drift_stages")
    if not over_atol_agrees:
        kinds.append("over_atol_stages")
    if first_over_layer_limit_agrees is False:
        kinds.append("first_over_atol_layer_limit")
    if first_over_location_agrees is False:
        kinds.append("first_over_atol_location")
    if first_over_record_agrees is False:
        kinds.append("first_over_atol_record")
    if first_over_bit_mismatch_delta not in (None, 0):
        kinds.append("first_over_atol_bit_mismatch")
    return kinds


def _projection_route_classification(
    difference_kinds: Sequence[str],
    *,
    has_first_over_atol_drift: bool,
    over_atol_agrees: bool,
    first_over_layer_limit_agrees: bool | None = None,
    first_over_location_agrees: bool | None = None,
    first_over_record_agrees: bool | None = None,
) -> str:
    if not difference_kinds:
        return "projection_rollups_match"
    if not over_atol_agrees:
        return "over_atol_stage_delta"
    if not has_first_over_atol_drift:
        if "drift_stages" in difference_kinds:
            return "no_over_atol_with_drift_delta"
        return "no_over_atol_with_metadata_delta"
    if first_over_layer_limit_agrees is False:
        return "first_over_atol_layer_limit_delta"
    if first_over_location_agrees is False:
        return "first_over_atol_location_delta"
    if first_over_record_agrees is False and "drift_stages" in difference_kinds:
        return "same_first_over_atol_location_with_record_and_drift_delta"
    if first_over_record_agrees is False:
        return "same_first_over_atol_location_with_record_delta"
    if "drift_stages" in difference_kinds:
        return "same_first_over_atol_location_with_drift_delta"
    return "same_first_over_atol_location_with_metadata_delta"


def _artifact_source_metadata(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


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
        "source_artifact": _artifact_source_metadata(path),
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
    first_over_records: dict[str, dict[str, Any] | None] = {}
    first_over_locations: dict[str, dict[str, Any] | None] = {}
    first_over_bit_mismatches: dict[str, int | None] = {}
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
        first_over_records[label] = first_over
        first_over_locations[label] = _drift_location(first_over)
        first_over_bit_mismatches[label] = _drift_bit_mismatch(first_over)
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
    first_over_locations_agree = _records_agree(tuple(first_over_locations.values()))
    first_over_records_agree = _records_agree(tuple(first_over_records.values()))
    first_over_bit_mismatch_delta = _int_delta(tuple(first_over_bit_mismatches.values()))
    difference_kinds = _projection_difference_kinds(
        drift_agrees=drift_agrees,
        over_atol_agrees=over_atol_agrees,
        first_over_location_agrees=first_over_locations_agree,
        first_over_record_agrees=first_over_records_agree,
        first_over_bit_mismatch_delta=first_over_bit_mismatch_delta,
    )
    return {
        "layer_limit": layer_limit,
        "per_artifact": per_artifact,
        "drift_agrees": drift_agrees,
        "over_atol_agrees": over_atol_agrees,
        "route_difference_kinds": difference_kinds,
        "route_classification": _projection_route_classification(
            difference_kinds,
            has_first_over_atol_drift=any(record is not None for record in first_over_records.values()),
            over_atol_agrees=over_atol_agrees,
            first_over_location_agrees=first_over_locations_agree,
            first_over_record_agrees=first_over_records_agree,
        ),
        "first_over_atol_drift_by_label": first_over_records,
        "first_over_atol_drift_location_by_label": first_over_locations,
        "first_over_atol_drift_locations_agree": first_over_locations_agree,
        "first_over_atol_drift_records_agree": first_over_records_agree,
        "first_over_atol_bit_mismatch_by_label": first_over_bit_mismatches,
        "first_over_atol_bit_mismatch_delta": first_over_bit_mismatch_delta,
    }


def _parse_expected_kinds(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_expected_scalar(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _parse_expected_location(values: Sequence[str] | None) -> dict[str, int | str]:
    location: dict[str, int | str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"location expectation {value!r} must use KEY=VALUE")
        key, raw_expected = value.split("=", 1)
        key = key.strip()
        raw_expected = raw_expected.strip()
        if not key or not raw_expected:
            raise ValueError(f"location expectation {value!r} must use non-empty KEY=VALUE")
        location[key] = _parse_expected_scalar(raw_expected)
    return location


def _parse_layer_expectation(value: str) -> tuple[int, str]:
    if "=" not in value:
        raise ValueError(f"layer expectation {value!r} must use LAYER=VALUE")
    raw_layer, expected = value.split("=", 1)
    raw_layer = raw_layer.strip()
    expected = expected.strip()
    if not raw_layer or not expected:
        raise ValueError(f"layer expectation {value!r} must use non-empty LAYER=VALUE")
    try:
        layer_limit = int(raw_layer)
    except ValueError as exc:
        raise ValueError(f"layer expectation {value!r} has non-integer layer limit") from exc
    return layer_limit, expected


def _parse_layer_int_expectation(value: str) -> tuple[int, int]:
    layer_limit, raw_expected = _parse_layer_expectation(value)
    try:
        return layer_limit, int(raw_expected)
    except ValueError as exc:
        raise ValueError(f"layer expectation {value!r} has non-integer expected value") from exc


def _parse_label_expectation(value: str, description: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{description} expectation {value!r} must use LABEL=VALUE")
    label, expected = value.split("=", 1)
    label = label.strip()
    expected = expected.strip()
    if not label or not expected:
        raise ValueError(f"{description} expectation {value!r} must use non-empty LABEL=VALUE")
    return label, expected


def _parse_label_int_expectation(value: str, description: str) -> tuple[str, int]:
    label, raw_expected = _parse_label_expectation(value, description)
    try:
        return label, int(raw_expected)
    except ValueError as exc:
        raise ValueError(f"{description} expectation {value!r} has non-integer expected value") from exc


def _artifact_by_label(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = payload.get("artifacts", [])
    if not isinstance(entries, list):
        return {}
    artifacts: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        if isinstance(label, str) and label:
            artifacts[label] = entry
    return artifacts


def _layer_limit_by_value(comparison: dict[str, Any], layer_limit: int) -> dict[str, Any] | None:
    entries = comparison.get("layer_limits", [])
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            if int(entry.get("layer_limit")) == layer_limit:
                return entry
        except (TypeError, ValueError):
            continue
    return None


def _expectation_metadata(args: argparse.Namespace) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    expected_route_classification = getattr(args, "expect_route_classification", None)
    if expected_route_classification is not None:
        metadata["route_classification"] = str(expected_route_classification)
    expected_kinds = _parse_expected_kinds(getattr(args, "expect_route_difference_kinds", None))
    if expected_kinds is not None:
        metadata["route_difference_kinds"] = expected_kinds
    expected_first_diverging = getattr(args, "expect_first_diverging_layer_limit", None)
    if expected_first_diverging is not None:
        metadata["first_diverging_layer_limit"] = int(expected_first_diverging)
    expected_bit_mismatch_delta = getattr(args, "expect_first_over_atol_bit_mismatch_delta", None)
    if expected_bit_mismatch_delta is not None:
        metadata["first_over_atol_bit_mismatch_delta"] = int(expected_bit_mismatch_delta)
    expected_location = _parse_expected_location(getattr(args, "expect_first_over_atol_location", None))
    if expected_location:
        metadata["first_over_atol_location"] = expected_location
    layer_classifications: dict[str, str] = {}
    for raw_expectation in getattr(args, "expect_layer_route_classification", None) or []:
        layer_limit, expected = _parse_layer_expectation(str(raw_expectation))
        layer_classifications[str(layer_limit)] = expected
    if layer_classifications:
        metadata["layer_route_classifications"] = layer_classifications
    layer_difference_kinds: dict[str, list[str]] = {}
    for raw_expectation in getattr(args, "expect_layer_route_difference_kinds", None) or []:
        layer_limit, raw_expected = _parse_layer_expectation(str(raw_expectation))
        layer_difference_kinds[str(layer_limit)] = _parse_expected_kinds(raw_expected) or []
    if layer_difference_kinds:
        metadata["layer_route_difference_kinds"] = layer_difference_kinds
    layer_bit_mismatch_deltas: dict[str, int] = {}
    for raw_expectation in getattr(args, "expect_layer_first_over_atol_bit_mismatch_delta", None) or []:
        layer_limit, expected = _parse_layer_int_expectation(str(raw_expectation))
        layer_bit_mismatch_deltas[str(layer_limit)] = expected
    if layer_bit_mismatch_deltas:
        metadata["layer_first_over_atol_bit_mismatch_deltas"] = layer_bit_mismatch_deltas
    source_sha256: dict[str, str] = {}
    for raw_expectation in getattr(args, "expect_artifact_sha256", None) or []:
        label, expected = _parse_label_expectation(str(raw_expectation), "artifact sha256")
        source_sha256[label] = expected
    if source_sha256:
        metadata["source_artifact_sha256"] = source_sha256
    source_sizes: dict[str, int] = {}
    for raw_expectation in getattr(args, "expect_artifact_size_bytes", None) or []:
        label, expected = _parse_label_int_expectation(str(raw_expectation), "artifact size")
        source_sizes[label] = expected
    if source_sizes:
        metadata["source_artifact_size_bytes"] = source_sizes
    required_booleans: dict[str, bool] = {}
    if getattr(args, "expect_hidden_passed_all", False):
        required_booleans["hidden_passed_all"] = True
    if getattr(args, "expect_token_passed_all", False):
        required_booleans["token_passed_all"] = True
    if getattr(args, "expect_all_statuses_eq_ok", False):
        required_booleans["all_statuses_eq_ok"] = True
    if required_booleans:
        metadata["required_booleans"] = required_booleans
    return metadata or None


def _validate_expectations(payload: dict[str, Any], args: argparse.Namespace) -> None:
    comparison = payload.get("comparison", {})
    if not isinstance(comparison, dict):
        raise ValueError("comparison payload is missing or malformed")
    errors: list[str] = []
    expected_route_classification = getattr(args, "expect_route_classification", None)
    if expected_route_classification is not None and comparison.get("route_classification") != expected_route_classification:
        errors.append(
            "route_classification expected "
            f"{expected_route_classification!r} but found {comparison.get('route_classification')!r}"
        )
    expected_kinds = _parse_expected_kinds(getattr(args, "expect_route_difference_kinds", None))
    if expected_kinds is not None and comparison.get("route_difference_kinds") != expected_kinds:
        errors.append(
            "route_difference_kinds expected "
            f"{expected_kinds!r} but found {comparison.get('route_difference_kinds')!r}"
        )
    expected_first_diverging = getattr(args, "expect_first_diverging_layer_limit", None)
    if expected_first_diverging is not None and comparison.get("first_diverging_layer_limit") != expected_first_diverging:
        errors.append(
            "first_diverging_layer_limit expected "
            f"{expected_first_diverging!r} but found {comparison.get('first_diverging_layer_limit')!r}"
        )
    expected_bit_mismatch_delta = getattr(args, "expect_first_over_atol_bit_mismatch_delta", None)
    if (
        expected_bit_mismatch_delta is not None
        and comparison.get("first_over_atol_bit_mismatch_delta") != expected_bit_mismatch_delta
    ):
        errors.append(
            "first_over_atol_bit_mismatch_delta expected "
            f"{expected_bit_mismatch_delta!r} but found {comparison.get('first_over_atol_bit_mismatch_delta')!r}"
        )
    expected_location = _parse_expected_location(getattr(args, "expect_first_over_atol_location", None))
    if expected_location:
        locations = comparison.get("first_over_atol_drift_location_by_label", {})
        if not isinstance(locations, dict) or not locations:
            errors.append("first_over_atol_location expected but no first-over-atol locations were found")
        else:
            for label, location in locations.items():
                if not isinstance(location, dict):
                    errors.append(f"first_over_atol_location for {label} expected {expected_location!r} but found {location!r}")
                    continue
                for key, expected in expected_location.items():
                    found = location.get(key)
                    if found != expected:
                        errors.append(
                            f"first_over_atol_location[{label}].{key} expected {expected!r} but found {found!r}"
                        )
    for raw_expectation in getattr(args, "expect_layer_route_classification", None) or []:
        layer_limit, expected = _parse_layer_expectation(str(raw_expectation))
        entry = _layer_limit_by_value(comparison, layer_limit)
        found = None if entry is None else entry.get("route_classification")
        if found != expected:
            errors.append(
                f"layer {layer_limit} route_classification expected {expected!r} but found {found!r}"
            )
    for raw_expectation in getattr(args, "expect_layer_route_difference_kinds", None) or []:
        layer_limit, raw_expected = _parse_layer_expectation(str(raw_expectation))
        expected = _parse_expected_kinds(raw_expected) or []
        entry = _layer_limit_by_value(comparison, layer_limit)
        found = None if entry is None else entry.get("route_difference_kinds")
        if found != expected:
            errors.append(
                f"layer {layer_limit} route_difference_kinds expected {expected!r} but found {found!r}"
            )
    for raw_expectation in getattr(args, "expect_layer_first_over_atol_bit_mismatch_delta", None) or []:
        layer_limit, expected = _parse_layer_int_expectation(str(raw_expectation))
        entry = _layer_limit_by_value(comparison, layer_limit)
        found = None if entry is None else entry.get("first_over_atol_bit_mismatch_delta")
        if found != expected:
            errors.append(
                f"layer {layer_limit} first_over_atol_bit_mismatch_delta expected {expected!r} but found {found!r}"
            )
    artifacts_by_label = _artifact_by_label(payload)
    for raw_expectation in getattr(args, "expect_artifact_sha256", None) or []:
        label, expected = _parse_label_expectation(str(raw_expectation), "artifact sha256")
        artifact = artifacts_by_label.get(label)
        source = artifact.get("source_artifact") if isinstance(artifact, dict) else None
        found = source.get("sha256") if isinstance(source, dict) else None
        if found != expected:
            errors.append(f"artifact {label} source_artifact.sha256 expected {expected!r} but found {found!r}")
    for raw_expectation in getattr(args, "expect_artifact_size_bytes", None) or []:
        label, expected = _parse_label_int_expectation(str(raw_expectation), "artifact size")
        artifact = artifacts_by_label.get(label)
        source = artifact.get("source_artifact") if isinstance(artifact, dict) else None
        found = source.get("size_bytes") if isinstance(source, dict) else None
        if found != expected:
            errors.append(f"artifact {label} source_artifact.size_bytes expected {expected!r} but found {found!r}")
    if getattr(args, "expect_hidden_passed_all", False) and comparison.get("hidden_passed_all") is not True:
        errors.append(f"hidden_passed_all expected True but found {comparison.get('hidden_passed_all')!r}")
    if getattr(args, "expect_token_passed_all", False) and comparison.get("token_passed_all") is not True:
        errors.append(f"token_passed_all expected True but found {comparison.get('token_passed_all')!r}")
    if getattr(args, "expect_all_statuses_eq_ok", False) and comparison.get("all_statuses_eq_ok") is not True:
        errors.append(f"all_statuses_eq_ok expected True but found {comparison.get('all_statuses_eq_ok')!r}")
    if errors:
        raise ValueError("; ".join(errors))


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
    first_over_records = {
        label: summaries[label]["projection"].get("first_over_atol_drift") for label in labels
    }
    first_over_locations = {label: _drift_location(record) for label, record in first_over_records.items()}
    first_over_bit_mismatches = {label: _drift_bit_mismatch(record) for label, record in first_over_records.items()}
    first_over_location_agreement = _records_agree(tuple(first_over_locations.values()))
    first_over_record_agreement = _records_agree(tuple(first_over_records.values()))
    first_over_bit_mismatch_delta = _int_delta(tuple(first_over_bit_mismatches.values()))
    first_over_layer_limit_agreement = len(set(first_over_values)) <= 1
    projection_drift_agreement = all(entry["drift_agrees"] for entry in per_limit)
    projection_over_atol_agreement = all(entry["over_atol_agrees"] for entry in per_limit)
    route_difference_kinds = _projection_difference_kinds(
        drift_agrees=projection_drift_agreement,
        over_atol_agrees=projection_over_atol_agreement,
        first_over_layer_limit_agrees=first_over_layer_limit_agreement,
        first_over_location_agrees=first_over_location_agreement,
        first_over_record_agrees=first_over_record_agreement,
        first_over_bit_mismatch_delta=first_over_bit_mismatch_delta,
    )
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
            "route_difference_kinds": route_difference_kinds,
            "route_classification": _projection_route_classification(
                route_difference_kinds,
                has_first_over_atol_drift=any(record is not None for record in first_over_records.values()),
                over_atol_agrees=projection_over_atol_agreement,
                first_over_layer_limit_agrees=first_over_layer_limit_agreement,
                first_over_location_agrees=first_over_location_agreement,
                first_over_record_agrees=first_over_record_agreement,
            ),
            "first_over_atol_layer_limits_by_label": first_over_by_label,
            "labels_agree_on_first_over_atol_layer_limit": first_over_layer_limit_agreement,
            "first_over_atol_drift_by_label": first_over_records,
            "first_over_atol_drift_location_by_label": first_over_locations,
            "labels_agree_on_first_over_atol_drift_location": first_over_location_agreement,
            "labels_agree_on_first_over_atol_drift_record": first_over_record_agreement,
            "first_over_atol_bit_mismatch_by_label": first_over_bit_mismatches,
            "first_over_atol_bit_mismatch_delta": first_over_bit_mismatch_delta,
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
    expectation_metadata = _expectation_metadata(args)
    if expectation_metadata is not None:
        payload["expectations"] = expectation_metadata
    _validate_expectations(payload, args)
    if expectation_metadata is not None:
        payload["expectations"]["passed"] = True
    json_path = getattr(args, "json", None)
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_payload_json(payload) + "\n")
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
    parser.add_argument("--expect-route-classification", help="Fail unless comparison.route_classification matches")
    parser.add_argument(
        "--expect-route-difference-kinds",
        help="Comma-separated exact comparison.route_difference_kinds expectation",
    )
    parser.add_argument(
        "--expect-layer-route-classification",
        action="append",
        help="Fail unless a layer limit has the expected route classification, as LAYER=CLASSIFICATION",
    )
    parser.add_argument(
        "--expect-layer-route-difference-kinds",
        action="append",
        help="Fail unless a layer limit has exact comma-separated route difference kinds, as LAYER=KIND[,KIND...]",
    )
    parser.add_argument(
        "--expect-first-over-atol-bit-mismatch-delta",
        type=int,
        help="Fail unless comparison.first_over_atol_bit_mismatch_delta matches",
    )
    parser.add_argument(
        "--expect-first-over-atol-location",
        action="append",
        help="Fail unless every label's first-over-atol location has KEY=VALUE (repeatable)",
    )
    parser.add_argument(
        "--expect-layer-first-over-atol-bit-mismatch-delta",
        action="append",
        help="Fail unless a layer limit has the expected first-over-atol bit-mismatch delta, as LAYER=INT",
    )
    parser.add_argument(
        "--expect-artifact-sha256",
        action="append",
        help="Fail unless an artifact's source_artifact.sha256 matches, as LABEL=SHA256",
    )
    parser.add_argument(
        "--expect-artifact-size-bytes",
        action="append",
        help="Fail unless an artifact's source_artifact.size_bytes matches, as LABEL=INT",
    )
    parser.add_argument("--expect-first-diverging-layer-limit", type=int, help="Fail unless first_diverging_layer_limit matches")
    parser.add_argument("--expect-hidden-passed-all", action="store_true", help="Fail unless hidden_passed_all is true")
    parser.add_argument("--expect-token-passed-all", action="store_true", help="Fail unless token_passed_all is true")
    parser.add_argument("--expect-all-statuses-eq-ok", action="store_true", help="Fail unless all_statuses_eq_ok is true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json is None:
        print(_payload_json(payload))


if __name__ == "__main__":
    main()
