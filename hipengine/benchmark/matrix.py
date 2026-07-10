"""Canonical exact-token benchmark matrix normalization and reporting."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.benchmark.exact_tokens import (
    EXACT_TOKEN_ORACLE_KIND,
    EXACT_TOKEN_ORACLE_SCHEMA_VERSION,
    ExactTokenOracle,
)
from hipengine.benchmark.prompts import token_ids_sha256
from hipengine.benchmark.provenance import validate_artifact_provenance


BENCHMARK_MATRIX_MANIFEST_KIND = "hipengine_benchmark_matrix_manifest"
BENCHMARK_MATRIX_KIND = "hipengine_benchmark_matrix"
BENCHMARK_MATRIX_SCHEMA_VERSION = 1

_ENGINES = {"paro", "gguf"}
_SURFACES = {"direct", "server"}
_TIMING_SCOPES = {"choice", "batch", "request", "client"}
_REQUIREMENT_KEYS = {
    "performance_claim",
    "clean_provenance",
    "scoped_timing",
    "memory",
    "profiler",
    "server_generation_shape",
}


class MatrixError(ValueError):
    """Raised when matrix inputs violate an identity or accounting contract."""


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatrixError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MatrixError(f"{label} must be a non-empty string")
    return value.strip()


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MatrixError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _finite_number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise MatrixError(f"{label} must be finite and >= {minimum}")
    return result


def _json_path(base_dir: Path, raw: Any, *, label: str) -> tuple[Path, str]:
    text = _string(raw, label=label)
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise MatrixError(f"{label} does not exist: {path}")
    return path, text


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError(f"cannot read {label} {path}: {exc}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pointer(payload: Any, raw_pointer: Any, *, label: str) -> Any:
    pointer = "" if raw_pointer is None else str(raw_pointer)
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise MatrixError(f"{label}.pointer must be empty or an RFC 6901 JSON pointer")
    current = payload
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if part not in current:
                raise MatrixError(f"{label}.pointer does not exist: {pointer}")
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise MatrixError(f"{label}.pointer list index is invalid: {part!r}") from exc
            if index < 0 or index >= len(current):
                raise MatrixError(f"{label}.pointer list index is out of range: {index}")
            current = current[index]
        else:
            raise MatrixError(f"{label}.pointer traverses a scalar at {part!r}")
    return current


def _token_rows(value: Any, *, label: str, allow_empty_rows: bool) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or not value:
        raise MatrixError(f"{label} must be a non-empty list of token-ID rows")
    rows: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(value):
        if not isinstance(raw_row, list) or (not raw_row and not allow_empty_rows):
            raise MatrixError(f"{label}[{row_index}] must be a token-ID list")
        row: list[int] = []
        for token_index, raw_token in enumerate(raw_row):
            if isinstance(raw_token, bool) or not isinstance(raw_token, int) or raw_token < 0:
                raise MatrixError(
                    f"{label}[{row_index}][{token_index}] must be a non-negative integer"
                )
            row.append(int(raw_token))
        rows.append(tuple(row))
    return tuple(rows)


def _validate_hashes(
    rows: Sequence[Sequence[int]],
    raw_hashes: Any,
    *,
    label: str,
) -> list[str]:
    expected = [token_ids_sha256(row) for row in rows]
    if raw_hashes != expected:
        raise MatrixError(f"{label} do not match token IDs")
    return expected


def _normalize_exact_artifact(payload: Any, *, label: str) -> dict[str, Any]:
    artifact = _mapping(payload, label=label)
    if artifact.get("kind") != EXACT_TOKEN_ORACLE_KIND:
        raise MatrixError(f"{label}.kind must be {EXACT_TOKEN_ORACLE_KIND!r}")
    if artifact.get("schema_version") != EXACT_TOKEN_ORACLE_SCHEMA_VERSION:
        raise MatrixError(
            f"{label}.schema_version must be {EXACT_TOKEN_ORACLE_SCHEMA_VERSION}"
        )
    mode = _string(artifact.get("mode"), label=f"{label}.mode").lower()
    if mode not in {"direct", "http"}:
        raise MatrixError(f"{label}.mode must be direct or http")
    shape = _mapping(artifact.get("shape"), label=f"{label}.shape")
    prompt_count = _strict_int(
        shape.get("prompt_count"), label=f"{label}.shape.prompt_count", minimum=1
    )
    prompt_length = _strict_int(
        shape.get("prompt_length"), label=f"{label}.shape.prompt_length", minimum=1
    )
    max_tokens = _strict_int(
        shape.get("max_tokens"), label=f"{label}.shape.max_tokens", minimum=0
    )
    prompts = _token_rows(
        artifact.get("prompt_token_ids"),
        label=f"{label}.prompt_token_ids",
        allow_empty_rows=False,
    )
    generated = _token_rows(
        artifact.get("generated_token_ids"),
        label=f"{label}.generated_token_ids",
        allow_empty_rows=max_tokens == 0,
    )
    try:
        ExactTokenOracle.from_rows(
            mode=mode,
            prompt_rows=prompts,
            generated_rows=generated,
            max_tokens=max_tokens,
        )
    except ValueError as exc:
        raise MatrixError(f"{label}: {exc}") from exc
    if len(prompts) != prompt_count or any(len(row) != prompt_length for row in prompts):
        raise MatrixError(f"{label}.shape does not match prompt token rows")
    prompt_hashes = _validate_hashes(
        prompts,
        artifact.get("prompt_token_ids_sha256"),
        label=f"{label}.prompt_token_ids_sha256",
    )
    generated_hashes = _validate_hashes(
        generated,
        artifact.get("generated_token_ids_sha256"),
        label=f"{label}.generated_token_ids_sha256",
    )
    measurement = _mapping(artifact.get("measurement"), label=f"{label}.measurement")
    wall_s = _finite_number(
        measurement.get("wall_s"), label=f"{label}.measurement.wall_s", minimum=0.0
    )
    if wall_s <= 0.0:
        raise MatrixError(f"{label}.measurement.wall_s must be positive")
    timing_scope = _string(
        measurement.get("timing_scope"), label=f"{label}.measurement.timing_scope"
    )
    provenance = _mapping(artifact.get("provenance"), label=f"{label}.provenance")
    try:
        provenance = validate_artifact_provenance(provenance, require_model=True)
    except ValueError as exc:
        raise MatrixError(f"{label}.provenance: {exc}") from exc
    request = _mapping(artifact.get("request"), label=f"{label}.request")
    parity_raw = artifact.get("exact_token_parity")
    parity = {} if parity_raw is None else _mapping(parity_raw, label=f"{label}.exact_token_parity")
    return {
        "payload": artifact,
        "mode": mode,
        "shape": {
            "prompt_count": prompt_count,
            "prompt_length": prompt_length,
            "max_tokens": max_tokens,
        },
        "prompt_rows": prompts,
        "generated_rows": generated,
        "prompt_hashes": prompt_hashes,
        "generated_hashes": generated_hashes,
        "measurement": {
            "wall_s": wall_s,
            "timing_scope": timing_scope,
            "eligible": measurement.get("eligible"),
            "reason": measurement.get("reason"),
        },
        "request": request,
        "parity": parity,
        "performance_claim": artifact.get("performance_claim") is True,
        "provenance": provenance,
    }


def _generation_telemetry(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = artifact.get("generation_telemetry")
    if isinstance(raw, list):
        return [_mapping(row, label=f"generation_telemetry[{index}]") for index, row in enumerate(raw)]
    response = artifact.get("response_metadata")
    if not isinstance(response, Mapping):
        return []
    raw = response.get("choice_telemetry")
    if isinstance(raw, list):
        return [_mapping(row, label=f"choice_telemetry[{index}]") for index, row in enumerate(raw)]
    choices = response.get("choices")
    if isinstance(choices, list):
        rows: list[dict[str, Any]] = []
        for choice in choices:
            if isinstance(choice, Mapping) and isinstance(choice.get("hipengine"), Mapping):
                rows.append(dict(choice["hipengine"]))
        return rows
    return []


def _normalize_timing(telemetry: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    legacy_scope_defaulted = False
    for choice_index, payload in enumerate(telemetry):
        raw_timing = payload.get("timing")
        if raw_timing is None:
            continue
        timing_obj = _mapping(raw_timing, label=f"generation_telemetry[{choice_index}].timing")
        timing: dict[str, float] = {}
        for key, value in timing_obj.items():
            timing[key] = _finite_number(
                value,
                label=f"generation_telemetry[{choice_index}].timing.{key}",
            )
        if not timing:
            continue
        scope_explicit = "timing_scope" in payload
        source_was_legacy = payload.get("legacy_scope_defaulted") is True
        raw_scope = payload.get("timing_scope", "choice")
        scope = _string(raw_scope, label=f"generation_telemetry[{choice_index}].timing_scope")
        if scope not in _TIMING_SCOPES:
            raise MatrixError(
                f"generation_telemetry[{choice_index}].timing_scope is invalid: {scope!r}"
            )
        group_rows = _strict_int(
            payload.get("group_rows", 1),
            label=f"generation_telemetry[{choice_index}].group_rows",
            minimum=1,
        )
        owner = payload.get("timing_owner", True)
        if not isinstance(owner, bool):
            raise MatrixError(
                f"generation_telemetry[{choice_index}].timing_owner must be a bool"
            )
        batch_id_raw = payload.get("batch_id")
        batch_id = None if batch_id_raw is None else _string(
            batch_id_raw, label=f"generation_telemetry[{choice_index}].batch_id"
        )
        if scope == "batch":
            if batch_id is None:
                raise MatrixError(
                    f"generation_telemetry[{choice_index}] has batch timing without batch_id"
                )
            if not scope_explicit or "group_rows" not in payload or "timing_owner" not in payload:
                raise MatrixError(
                    f"generation_telemetry[{choice_index}] batch timing ownership is incomplete"
                )
        record: dict[str, Any] = {
            "choice_index": choice_index,
            "timing": timing,
            "timing_scope": scope,
            "group_rows": group_rows,
            "timing_owner": owner,
        }
        if batch_id is not None:
            record["batch_id"] = batch_id
        if not scope_explicit or source_was_legacy:
            record["legacy_scope_defaulted"] = True
            legacy_scope_defaulted = True
        records.append(record)

    if not records:
        return {
            "status": "unavailable",
            "records": [],
            "owned_totals": {},
            "dedup": {
                "batch_ids": [],
                "batch_payloads_counted": 0,
                "choice_payloads_counted": 0,
                "non_owner_copies_ignored": 0,
            },
            "legacy_scope_defaulted": False,
        }

    selected: list[dict[str, Any]] = []
    batches: dict[str, list[dict[str, Any]]] = {}
    choice_count = 0
    for record in records:
        if record["timing_scope"] != "batch":
            selected.append(record)
            choice_count += 1
        else:
            batches.setdefault(str(record["batch_id"]), []).append(record)
    ignored = 0
    for batch_id, batch_records in sorted(batches.items()):
        group_rows = {int(record["group_rows"]) for record in batch_records}
        if len(group_rows) != 1:
            raise MatrixError(f"inconsistent group_rows for batch_id {batch_id!r}")
        expected_payloads = next(iter(group_rows))
        if len(batch_records) != expected_payloads:
            raise MatrixError(
                f"batch_id {batch_id!r} expected {expected_payloads} timing payloads; "
                f"found {len(batch_records)}"
            )
        owners = [record for record in batch_records if record["timing_owner"] is True]
        if len(owners) != 1:
            raise MatrixError(
                f"batch_id {batch_id!r} requires exactly one timing owner; found {len(owners)}"
            )
        selected.append(owners[0])
        ignored += len(batch_records) - 1

    totals: dict[str, float] = {}
    for record in selected:
        for key, value in record["timing"].items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {
        "status": "available",
        "records": records,
        "owned_totals": {key: totals[key] for key in sorted(totals)},
        "dedup": {
            "batch_ids": sorted(batches),
            "batch_payloads_counted": len(batches),
            "choice_payloads_counted": choice_count,
            "non_owner_copies_ignored": ignored,
        },
        "legacy_scope_defaulted": legacy_scope_defaulted,
    }


def _normalize_generation_shape(artifact: Mapping[str, Any]) -> dict[str, Any]:
    response = artifact.get("response_metadata")
    if not isinstance(response, Mapping):
        return {"status": "unavailable", "execution_paths": []}
    hipengine = response.get("hipengine")
    if not isinstance(hipengine, Mapping):
        return {"status": "unavailable", "execution_paths": []}
    raw = hipengine.get("generation_shape")
    if not isinstance(raw, Mapping):
        return {"status": "unavailable", "execution_paths": []}
    shape = _mapping(raw, label="response_metadata.hipengine.generation_shape")
    if shape.get("schema_version") != 1:
        raise MatrixError("generation_shape.schema_version must be 1")
    route = _string(shape.get("route"), label="generation_shape.route")
    cap = _mapping(shape.get("route_cap"), label="generation_shape.route_cap")
    if cap.get("scope") != "queue_requests" or not isinstance(cap.get("applied"), bool):
        raise MatrixError("generation_shape.route_cap contract is invalid")
    cap_value = cap.get("value")
    if cap_value is not None:
        cap_value = _strict_int(cap_value, label="generation_shape.route_cap.value", minimum=1)
    queue = _mapping(shape.get("queue_group"), label="generation_shape.queue_group")
    queue_summary = {
        "id": _string(queue.get("id"), label="generation_shape.queue_group.id"),
        "request_count": _strict_int(
            queue.get("request_count"), label="generation_shape.queue_group.request_count", minimum=1
        ),
        "prompt_rows": _strict_int(
            queue.get("prompt_rows"), label="generation_shape.queue_group.prompt_rows", minimum=1
        ),
        "item_index": _strict_int(
            queue.get("item_index"), label="generation_shape.queue_group.item_index"
        ),
        "item_prompt_offset": _strict_int(
            queue.get("item_prompt_offset"),
            label="generation_shape.queue_group.item_prompt_offset",
        ),
        "item_prompt_rows": _strict_int(
            queue.get("item_prompt_rows"),
            label="generation_shape.queue_group.item_prompt_rows",
            minimum=1,
        ),
    }
    raw_groups = shape.get("backend_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise MatrixError("generation_shape.backend_groups must be non-empty")
    groups: list[dict[str, Any]] = []
    backend_rows: list[int] = []
    for index, raw_group in enumerate(raw_groups):
        group = _mapping(raw_group, label=f"generation_shape.backend_groups[{index}]")
        actual = group.get("actual_group_rows")
        if not isinstance(actual, list) or not actual:
            raise MatrixError(
                f"generation_shape.backend_groups[{index}].actual_group_rows must be non-empty"
            )
        actual_rows = [
            _strict_int(value, label=f"backend_groups[{index}].actual_group_rows", minimum=1)
            for value in actual
        ]
        input_rows = _strict_int(
            group.get("input_rows"), label=f"backend_groups[{index}].input_rows", minimum=1
        )
        if sum(actual_rows) != input_rows:
            raise MatrixError(f"generation_shape.backend_groups[{index}] row sum is invalid")
        max_rows = _strict_int(
            group.get("max_actual_group_rows"),
            label=f"backend_groups[{index}].max_actual_group_rows",
            minimum=1,
        )
        if max_rows != max(actual_rows):
            raise MatrixError(
                f"generation_shape.backend_groups[{index}].max_actual_group_rows is invalid"
            )
        verifier_rows = _strict_int(
            group.get("verifier_rows"), label=f"backend_groups[{index}].verifier_rows"
        )
        groups.append(
            {
                "id": _string(group.get("id"), label=f"backend_groups[{index}].id"),
                "call_index": _strict_int(
                    group.get("call_index"), label=f"backend_groups[{index}].call_index"
                ),
                "prompt_offset": _strict_int(
                    group.get("prompt_offset"), label=f"backend_groups[{index}].prompt_offset"
                ),
                "input_rows": input_rows,
                "actual_group_rows": actual_rows,
                "max_actual_group_rows": max_rows,
                "verifier_rows": verifier_rows,
            }
        )
        backend_rows.extend(actual_rows)
    verifier_total = _strict_int(shape.get("verifier_rows"), label="generation_shape.verifier_rows")
    if verifier_total != sum(group["verifier_rows"] for group in groups):
        raise MatrixError("generation_shape.verifier_rows does not match backend groups")
    return {
        "status": "available",
        "schema_version": 1,
        "route": route,
        "route_cap": {
            "scope": "queue_requests",
            "value": cap_value,
            "applied": cap["applied"],
        },
        "queue_group": queue_summary,
        "backend_groups": groups,
        "backend_group_rows": backend_rows,
        "max_backend_group_rows": max(backend_rows),
        "verifier_rows": verifier_total,
        "execution_paths": [],
    }


def _execution_paths(telemetry: Sequence[Mapping[str, Any]]) -> list[str]:
    paths: set[str] = set()
    for row in telemetry:
        state = row.get("decode_state")
        if isinstance(state, Mapping):
            value = state.get("execution_path")
            if isinstance(value, str) and value.strip():
                paths.add(value.strip())
    return sorted(paths)


def _attachment(
    spec: Any,
    *,
    base_dir: Path,
    label: str,
) -> tuple[Any, dict[str, Any]]:
    attachment = _mapping(spec, label=label)
    unknown = sorted(set(attachment) - {"artifact", "pointer"})
    if unknown:
        raise MatrixError(f"{label} contains unknown fields: {unknown}")
    path, raw_path = _json_path(base_dir, attachment.get("artifact"), label=f"{label}.artifact")
    pointer = "" if attachment.get("pointer") is None else str(attachment.get("pointer"))
    payload = _pointer(_load_json(path, label=label), pointer, label=label)
    return payload, {
        "artifact_path": raw_path,
        "artifact_sha256": _file_sha256(path),
        "pointer": pointer,
    }


def _normalize_memory(value: Any, *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "unavailable"}
    memory = {str(key): item for key, item in value.items()}
    aliases = {
        "current_allocated_bytes": ("current_allocated_bytes", "allocated_after_load_bytes", "after_load_bytes"),
        "peak_allocated_bytes": (
            "peak_allocated_bytes",
            "tracked_peak_allocated_bytes",
            "allocator_reserved_peak_bytes",
        ),
        "peak_reserved_bytes": ("peak_reserved_bytes", "reserved_peak_bytes"),
        "hip_used_peak_sampled_bytes": ("hip_used_peak_sampled_bytes", "hip_peak_used_bytes"),
        "total_allocated_bytes": ("total_allocated_bytes",),
        "total_freed_bytes": ("total_freed_bytes",),
        "active_allocations": ("active_allocations",),
        "peak_allocations": ("peak_allocations",),
    }
    result: dict[str, Any] = {
        "status": "available",
        "scope": str(memory.get("scope") or "unspecified"),
    }
    found = False
    for output_key, keys in aliases.items():
        for key in keys:
            if key not in memory or memory[key] is None:
                continue
            result[output_key] = _strict_int(memory[key], label=f"memory.{key}")
            found = True
            break
    if not found:
        return {"status": "unavailable"}
    if source is not None:
        result["source"] = dict(source)
    return result


def _normalize_profiler(value: Any, *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "unavailable"}
    summary = {str(key): item for key, item in value.items()}
    if summary.get("enabled") is False and not any(
        key in summary for key in ("kernel_calls", "total_gpu_ms", "families", "top_kernels")
    ):
        result: dict[str, Any] = {"status": "disabled", "summary": summary}
    elif not summary:
        return {"status": "unavailable"}
    else:
        result = {"status": "available", "summary": summary}
    if source is not None:
        result["source"] = dict(source)
    return result


def _row_requirements(
    *,
    source: Mapping[str, Any],
    timing: Mapping[str, Any],
    path: Mapping[str, Any],
    memory: Mapping[str, Any],
    profiler: Mapping[str, Any],
    surface: str,
    requirements: Mapping[str, bool],
) -> list[str]:
    blockers: list[str] = []
    provenance = source["provenance"]
    if requirements["performance_claim"] and not source["performance_claim"]:
        blockers.append("source artifact performance_claim is not true")
    if requirements["performance_claim"] and source["measurement_eligible"] is not True:
        blockers.append("source artifact measurement eligibility is not true")
    if requirements["clean_provenance"] and provenance["dirty"]:
        blockers.append("source artifact provenance is dirty")
    if requirements["scoped_timing"]:
        if timing["status"] != "available":
            blockers.append("scoped generation timing is unavailable")
        elif timing["legacy_scope_defaulted"]:
            blockers.append("timing scope was defaulted from a legacy payload")
    if requirements["memory"] and memory["status"] != "available":
        blockers.append("memory summary is unavailable")
    if requirements["profiler"] and profiler["status"] != "available":
        blockers.append("profiler summary is unavailable")
    if requirements["server_generation_shape"] and surface == "server" and path["status"] != "available":
        blockers.append("server generation shape is unavailable")
    if surface == "server" and source["parity"].get("passed") is not True:
        blockers.append("server exact-token parity is not passed")
    return blockers


def _requirements(value: Any) -> dict[str, bool]:
    raw = {} if value is None else _mapping(value, label="manifest.requirements")
    unknown = sorted(set(raw) - _REQUIREMENT_KEYS)
    if unknown:
        raise MatrixError(f"manifest.requirements contains unknown keys: {unknown}")
    defaults = {
        "performance_claim": False,
        "clean_provenance": False,
        "scoped_timing": True,
        "memory": True,
        "profiler": True,
        "server_generation_shape": True,
    }
    for key, value in raw.items():
        if not isinstance(value, bool):
            raise MatrixError(f"manifest.requirements.{key} must be a bool")
        defaults[key] = value
    return defaults


def _required_axis(value: Any, *, label: str, allowed: set[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        raise MatrixError(f"{label} must be a non-empty list")
    items = [_string(item, label=label).lower() for item in value]
    if len(items) != len(set(items)):
        raise MatrixError(f"{label} must not contain duplicates")
    unknown = sorted(set(items) - allowed)
    if unknown:
        raise MatrixError(f"{label} contains unsupported values: {unknown}")
    return items


def build_benchmark_matrix(
    manifest_payload: Mapping[str, Any],
    *,
    base_dir: str | Path,
    report_provenance: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build one strict, denominator-safe matrix artifact from exact-token rows."""

    manifest = _mapping(manifest_payload, label="manifest")
    manifest_fields = {
        "kind",
        "schema_version",
        "name",
        "workdir",
        "required_engines",
        "required_surfaces",
        "requirements",
        "rows",
    }
    unknown_manifest_fields = sorted(set(manifest) - manifest_fields)
    if unknown_manifest_fields:
        raise MatrixError(f"manifest contains unknown fields: {unknown_manifest_fields}")
    if manifest.get("kind") != BENCHMARK_MATRIX_MANIFEST_KIND:
        raise MatrixError(f"manifest.kind must be {BENCHMARK_MATRIX_MANIFEST_KIND!r}")
    if manifest.get("schema_version") != BENCHMARK_MATRIX_SCHEMA_VERSION:
        raise MatrixError(f"manifest.schema_version must be {BENCHMARK_MATRIX_SCHEMA_VERSION}")
    name = _string(manifest.get("name"), label="manifest.name")
    root = Path(base_dir).resolve()
    required_engines = _required_axis(
        manifest.get("required_engines"), label="manifest.required_engines", allowed=_ENGINES
    )
    required_surfaces = _required_axis(
        manifest.get("required_surfaces"), label="manifest.required_surfaces", allowed=_SURFACES
    )
    if "requirements" not in manifest:
        raise MatrixError("manifest.requirements is required")
    requirements = _requirements(manifest.get("requirements"))
    try:
        report_provenance_dict = validate_artifact_provenance(report_provenance)
    except ValueError as exc:
        raise MatrixError(f"report_provenance: {exc}") from exc

    raw_rows = manifest.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise MatrixError("manifest.rows must be a non-empty list")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_case_surfaces: set[tuple[str, str]] = set()
    seen_engine_surface_variants: set[tuple[str, str, str]] = set()
    exact_by_row: dict[str, dict[str, Any]] = {}
    for index, raw_spec in enumerate(raw_rows):
        spec = _mapping(raw_spec, label=f"manifest.rows[{index}]")
        row_spec_fields = {
            "id",
            "case_id",
            "engine",
            "surface",
            "path_variant",
            "artifact",
            "memory",
            "profiler",
            "command",
            "environment",
        }
        unknown_row_fields = sorted(set(spec) - row_spec_fields)
        if unknown_row_fields:
            raise MatrixError(
                f"manifest.rows[{index}] contains unknown fields: {unknown_row_fields}"
            )
        row_id = _string(spec.get("id"), label=f"manifest.rows[{index}].id")
        if row_id in seen_ids:
            raise MatrixError(f"duplicate matrix row id: {row_id!r}")
        seen_ids.add(row_id)
        case_id = _string(spec.get("case_id"), label=f"manifest.rows[{index}].case_id")
        engine = _string(spec.get("engine"), label=f"manifest.rows[{index}].engine").lower()
        surface = _string(spec.get("surface"), label=f"manifest.rows[{index}].surface").lower()
        variant = _string(
            spec.get("path_variant"), label=f"manifest.rows[{index}].path_variant"
        )
        if engine not in _ENGINES:
            raise MatrixError(f"manifest row {row_id!r} has unsupported engine {engine!r}")
        if surface not in _SURFACES:
            raise MatrixError(f"manifest row {row_id!r} has unsupported surface {surface!r}")
        key = (case_id, surface)
        if key in seen_case_surfaces:
            raise MatrixError(f"case {case_id!r} has duplicate {surface!r} rows")
        seen_case_surfaces.add(key)
        engine_surface_variant = (engine, surface, variant)
        if engine_surface_variant in seen_engine_surface_variants:
            raise MatrixError(
                f"duplicate engine/surface/path variant row: {engine_surface_variant!r}"
            )
        seen_engine_surface_variants.add(engine_surface_variant)
        artifact_path, artifact_raw_path = _json_path(
            root, spec.get("artifact"), label=f"manifest.rows[{index}].artifact"
        )
        exact = _normalize_exact_artifact(
            _load_json(artifact_path, label=f"row {row_id!r} artifact"),
            label=f"row {row_id!r} artifact",
        )
        expected_mode = "direct" if surface == "direct" else "http"
        if exact["mode"] != expected_mode:
            raise MatrixError(
                f"row {row_id!r} surface {surface!r} requires artifact mode {expected_mode!r}"
            )
        telemetry = _generation_telemetry(exact["payload"])
        timing = _normalize_timing(telemetry)
        path = _normalize_generation_shape(exact["payload"])
        path["execution_paths"] = _execution_paths(telemetry)

        memory_value: Any = exact["payload"].get("memory")
        memory_source = None
        if spec.get("memory") is not None:
            memory_value, memory_source = _attachment(
                spec["memory"], base_dir=root, label=f"manifest.rows[{index}].memory"
            )
        memory = _normalize_memory(memory_value, source=memory_source)

        profiler_value: Any = exact["payload"].get("profiler")
        if profiler_value is None:
            profiler_value = exact["provenance"].get("profiler")
        profiler_source = None
        if spec.get("profiler") is not None:
            profiler_value, profiler_source = _attachment(
                spec["profiler"], base_dir=root, label=f"manifest.rows[{index}].profiler"
            )
        profiler = _normalize_profiler(profiler_value, source=profiler_source)

        total_prompt_tokens = sum(len(row) for row in exact["prompt_rows"])
        total_generated_tokens = sum(len(row) for row in exact["generated_rows"])
        if total_generated_tokens <= 0:
            raise MatrixError(f"row {row_id!r} has no exact generated tokens")
        wall_s = exact["measurement"]["wall_s"]
        latency = {
            "timing_scope": exact["measurement"]["timing_scope"],
            "wall_s": wall_s,
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "generated_tokens_per_second": total_generated_tokens / wall_s,
            "wall_ms_per_generated_token": 1000.0 * wall_s / total_generated_tokens,
        }
        source = {
            "artifact_path": artifact_raw_path,
            "artifact_sha256": _file_sha256(artifact_path),
            "performance_claim": exact["performance_claim"],
            "measurement_eligible": exact["measurement"].get("eligible"),
            "measurement_reason": exact["measurement"].get("reason"),
            "provenance": exact["provenance"],
            "parity": exact["parity"],
        }
        blockers = _row_requirements(
            source=source,
            timing=timing,
            path=path,
            memory=memory,
            profiler=profiler,
            surface=surface,
            requirements=requirements,
        )
        row = {
            "id": row_id,
            "case_id": case_id,
            "engine": engine,
            "surface": surface,
            "path_variant": variant,
            "source": source,
            "exact_tokens": {
                "shape": exact["shape"],
                "prompt_token_ids_sha256": exact["prompt_hashes"],
                "generated_token_ids_sha256": exact["generated_hashes"],
                "total_prompt_tokens": total_prompt_tokens,
                "total_generated_tokens": total_generated_tokens,
            },
            "latency": latency,
            "timing": timing,
            "path": path,
            "memory": memory,
            "profiler": profiler,
            "eligibility": {"passed": not blockers, "blockers": blockers},
        }
        rows.append(row)
        exact_by_row[row_id] = exact

    first_exact = exact_by_row[rows[0]["id"]]
    protocol_shape = first_exact["shape"]
    protocol_prompts = first_exact["prompt_rows"]
    sampling_keys = ("temperature", "top_p", "ignore_eos")
    protocol_sampling = {key: first_exact["request"].get(key) for key in sampling_keys}
    for row in rows[1:]:
        exact = exact_by_row[row["id"]]
        if exact["shape"] != protocol_shape or exact["prompt_rows"] != protocol_prompts:
            raise MatrixError(f"row {row['id']!r} uses a different exact-token protocol")
        sampling = {key: exact["request"].get(key) for key in sampling_keys}
        if sampling != protocol_sampling:
            raise MatrixError(f"row {row['id']!r} uses different sampling parameters")

    blockers: list[str] = []
    present = {(row["engine"], row["surface"]) for row in rows}
    for engine in sorted(required_engines):
        for surface in sorted(required_surfaces):
            if (engine, surface) not in present:
                blockers.append(
                    f"required matrix row is missing: engine={engine} surface={surface}"
                )
    direct_server: list[dict[str, Any]] = []
    case_ids = sorted({row["case_id"] for row in rows})
    for case_id in case_ids:
        case_rows = {row["surface"]: row for row in rows if row["case_id"] == case_id}
        if set(case_rows) != {"direct", "server"}:
            blockers.append(f"case {case_id!r} does not contain one direct and one server row")
            continue
        direct = case_rows["direct"]
        server = case_rows["server"]
        if direct["engine"] != server["engine"] or direct["path_variant"] != server["path_variant"]:
            raise MatrixError(f"case {case_id!r} mixes engines or path variants")
        direct_exact = exact_by_row[direct["id"]]
        server_exact = exact_by_row[server["id"]]
        if direct_exact["generated_rows"] != server_exact["generated_rows"]:
            raise MatrixError(f"case {case_id!r} generated token IDs differ between direct and server")
        direct_scope = direct["latency"]["timing_scope"]
        server_scope = server["latency"]["timing_scope"]
        ratio = None
        reason = None
        if direct_scope == server_scope:
            ratio = (
                server["latency"]["generated_tokens_per_second"]
                / direct["latency"]["generated_tokens_per_second"]
            )
        else:
            reason = f"timing scopes differ: {direct_scope} vs {server_scope}"
        direct_server.append(
            {
                "case_id": case_id,
                "engine": direct["engine"],
                "path_variant": direct["path_variant"],
                "direct_row": direct["id"],
                "server_row": server["id"],
                "exact_generated_ids_equal": True,
                "rates": {
                    "direct": direct["latency"]["generated_tokens_per_second"],
                    "server": server["latency"]["generated_tokens_per_second"],
                },
                "rate_ratio": ratio,
                "ratio_reason": reason,
            }
        )

    cross_engine: list[dict[str, Any]] = []
    for surface in sorted({row["surface"] for row in rows}):
        for variant in sorted({row["path_variant"] for row in rows if row["surface"] == surface}):
            matching = [
                row for row in rows if row["surface"] == surface and row["path_variant"] == variant
            ]
            cross_engine.append(
                {
                    "surface": surface,
                    "path_variant": variant,
                    "rows": {row["engine"]: row["id"] for row in sorted(matching, key=lambda item: item["engine"])},
                    "rates": {
                        row["engine"]: row["latency"]["generated_tokens_per_second"]
                        for row in sorted(matching, key=lambda item: item["engine"])
                    },
                    "rate_ratio": None,
                    "ratio_reason": "cross-engine ratios require an explicitly identical model/quant/timing protocol",
                }
            )

    for row in rows:
        blockers.extend(f"row {row['id']}: {blocker}" for blocker in row["eligibility"]["blockers"])
    if requirements["clean_provenance"] and report_provenance_dict["dirty"]:
        blockers.append("matrix report provenance is dirty")
    blockers = list(dict.fromkeys(blockers))
    coverage = {
        "engines": sorted({row["engine"] for row in rows}),
        "surfaces": sorted({row["surface"] for row in rows}),
        "path_variants": sorted({row["path_variant"] for row in rows}),
        "required_engines": sorted(required_engines),
        "required_surfaces": sorted(required_surfaces),
        "required_grid_complete": not any(
            blocker.startswith("required matrix row is missing:") for blocker in blockers
        ),
    }
    passed = not blockers
    result = {
        "kind": BENCHMARK_MATRIX_KIND,
        "schema_version": BENCHMARK_MATRIX_SCHEMA_VERSION,
        "name": name,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "performance_claim": (
            passed
            and all(requirements.values())
            and all(row["source"]["performance_claim"] for row in rows)
        ),
        "requirements": requirements,
        "protocol": {
            "shape": protocol_shape,
            "prompt_token_ids_sha256": first_exact["prompt_hashes"],
            "sampling": protocol_sampling,
        },
        "coverage": coverage,
        "rows": rows,
        "comparisons": {
            "direct_server": direct_server,
            "cross_engine": cross_engine,
        },
        "eligibility": {"passed": passed, "blockers": blockers},
        "report_provenance": report_provenance_dict,
    }
    validate_benchmark_matrix(result)
    return result


def validate_benchmark_matrix(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed top-level matrix schema used by committed artifacts."""

    matrix = _mapping(payload, label="matrix")
    required = {
        "kind",
        "schema_version",
        "name",
        "created_at",
        "performance_claim",
        "requirements",
        "protocol",
        "coverage",
        "rows",
        "comparisons",
        "eligibility",
        "report_provenance",
    }
    if set(matrix) != required:
        raise MatrixError(
            f"matrix top-level fields differ: missing={sorted(required - set(matrix))}, "
            f"extra={sorted(set(matrix) - required)}"
        )
    if matrix["kind"] != BENCHMARK_MATRIX_KIND:
        raise MatrixError(f"matrix.kind must be {BENCHMARK_MATRIX_KIND!r}")
    if matrix["schema_version"] != BENCHMARK_MATRIX_SCHEMA_VERSION:
        raise MatrixError(f"matrix.schema_version must be {BENCHMARK_MATRIX_SCHEMA_VERSION}")
    _string(matrix["name"], label="matrix.name")
    _string(matrix["created_at"], label="matrix.created_at")
    if not isinstance(matrix["performance_claim"], bool):
        raise MatrixError("matrix.performance_claim must be a bool")
    if not isinstance(matrix["rows"], list) or not matrix["rows"]:
        raise MatrixError("matrix.rows must be non-empty")
    protocol = _mapping(matrix["protocol"], label="matrix.protocol")
    protocol_shape = _mapping(protocol.get("shape"), label="matrix.protocol.shape")
    protocol_prompt_hashes = protocol.get("prompt_token_ids_sha256")
    if not isinstance(protocol_prompt_hashes, list) or not protocol_prompt_hashes:
        raise MatrixError("matrix.protocol.prompt_token_ids_sha256 must be non-empty")
    row_fields = {
        "id",
        "case_id",
        "engine",
        "surface",
        "path_variant",
        "source",
        "exact_tokens",
        "latency",
        "timing",
        "path",
        "memory",
        "profiler",
        "eligibility",
    }
    seen_ids: set[str] = set()
    rows_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    any_source_without_claim = False
    for index, raw_row in enumerate(matrix["rows"]):
        row = _mapping(raw_row, label=f"matrix.rows[{index}]")
        if set(row) != row_fields:
            raise MatrixError(f"matrix.rows[{index}] fields differ from the row contract")
        row_id = _string(row.get("id"), label=f"matrix.rows[{index}].id")
        if row_id in seen_ids:
            raise MatrixError(f"duplicate matrix row id: {row_id!r}")
        seen_ids.add(row_id)
        case_id = _string(row.get("case_id"), label=f"matrix.rows[{index}].case_id")
        surface = _string(row.get("surface"), label=f"matrix.rows[{index}].surface")
        if surface not in _SURFACES:
            raise MatrixError(f"matrix.rows[{index}].surface is invalid")
        if row.get("engine") not in _ENGINES:
            raise MatrixError(f"matrix.rows[{index}].engine is invalid")
        rows_by_case.setdefault(case_id, {})[surface] = row

        source = _mapping(row.get("source"), label=f"matrix.rows[{index}].source")
        try:
            validate_artifact_provenance(
                _mapping(source.get("provenance"), label=f"matrix.rows[{index}].source.provenance"),
                require_model=True,
            )
        except ValueError as exc:
            raise MatrixError(f"matrix.rows[{index}].source.provenance: {exc}") from exc
        if not isinstance(source.get("performance_claim"), bool):
            raise MatrixError(f"matrix.rows[{index}].source.performance_claim must be a bool")
        any_source_without_claim = any_source_without_claim or not source["performance_claim"]

        exact = _mapping(row.get("exact_tokens"), label=f"matrix.rows[{index}].exact_tokens")
        if exact.get("shape") != protocol_shape:
            raise MatrixError(f"matrix.rows[{index}] exact-token shape differs from protocol")
        if exact.get("prompt_token_ids_sha256") != protocol_prompt_hashes:
            raise MatrixError(f"matrix.rows[{index}] prompt hashes differ from protocol")
        total_prompt = _strict_int(
            exact.get("total_prompt_tokens"),
            label=f"matrix.rows[{index}].exact_tokens.total_prompt_tokens",
            minimum=1,
        )
        total_generated = _strict_int(
            exact.get("total_generated_tokens"),
            label=f"matrix.rows[{index}].exact_tokens.total_generated_tokens",
            minimum=1,
        )
        generated_hashes = exact.get("generated_token_ids_sha256")
        if not isinstance(generated_hashes, list) or not generated_hashes:
            raise MatrixError(f"matrix.rows[{index}] generated hashes must be non-empty")

        latency = _mapping(row.get("latency"), label=f"matrix.rows[{index}].latency")
        wall_s = _finite_number(
            latency.get("wall_s"), label=f"matrix.rows[{index}].latency.wall_s"
        )
        if wall_s <= 0.0:
            raise MatrixError(f"matrix.rows[{index}].latency.wall_s must be positive")
        if latency.get("total_prompt_tokens") != total_prompt:
            raise MatrixError(f"matrix.rows[{index}] prompt denominator is inconsistent")
        if latency.get("total_generated_tokens") != total_generated:
            raise MatrixError(f"matrix.rows[{index}] generated denominator is inconsistent")
        rate = _finite_number(
            latency.get("generated_tokens_per_second"),
            label=f"matrix.rows[{index}].latency.generated_tokens_per_second",
        )
        per_token_ms = _finite_number(
            latency.get("wall_ms_per_generated_token"),
            label=f"matrix.rows[{index}].latency.wall_ms_per_generated_token",
        )
        if not math.isclose(rate, total_generated / wall_s, rel_tol=1e-12, abs_tol=1e-12):
            raise MatrixError(f"matrix.rows[{index}] generated-token rate denominator is forged")
        if not math.isclose(
            per_token_ms,
            1000.0 * wall_s / total_generated,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise MatrixError(f"matrix.rows[{index}] per-token latency denominator is forged")

        timing = _mapping(row.get("timing"), label=f"matrix.rows[{index}].timing")
        records = timing.get("records")
        if not isinstance(records, list):
            raise MatrixError(f"matrix.rows[{index}].timing.records must be a list")
        recomputed = _normalize_timing(
            [_mapping(item, label=f"matrix.rows[{index}].timing.records") for item in records]
        )
        for key in ("status", "owned_totals", "dedup", "legacy_scope_defaulted"):
            if timing.get(key) != recomputed.get(key):
                raise MatrixError(f"matrix.rows[{index}].timing.{key} is inconsistent")

        row_eligibility = _mapping(
            row.get("eligibility"), label=f"matrix.rows[{index}].eligibility"
        )
        row_blockers = row_eligibility.get("blockers")
        if not isinstance(row_blockers, list) or not all(
            isinstance(blocker, str) for blocker in row_blockers
        ):
            raise MatrixError(f"matrix.rows[{index}].eligibility.blockers is invalid")
        if row_eligibility.get("passed") is not (not row_blockers):
            raise MatrixError(f"matrix.rows[{index}].eligibility.passed is inconsistent")

    for case_id, case_rows in rows_by_case.items():
        if set(case_rows) == {"direct", "server"}:
            if (
                case_rows["direct"]["exact_tokens"]["generated_token_ids_sha256"]
                != case_rows["server"]["exact_tokens"]["generated_token_ids_sha256"]
            ):
                raise MatrixError(f"case {case_id!r} generated token hashes differ")
    eligibility = _mapping(matrix["eligibility"], label="matrix.eligibility")
    if not isinstance(eligibility.get("passed"), bool) or not isinstance(
        eligibility.get("blockers"), list
    ):
        raise MatrixError("matrix.eligibility is invalid")
    if not all(isinstance(blocker, str) for blocker in eligibility["blockers"]):
        raise MatrixError("matrix.eligibility.blockers must contain strings")
    if eligibility["passed"] is not (not eligibility["blockers"]):
        raise MatrixError("matrix.eligibility.passed is inconsistent")
    if matrix["performance_claim"] and not eligibility["passed"]:
        raise MatrixError("matrix.performance_claim cannot be true when eligibility fails")
    if matrix["performance_claim"] and any_source_without_claim:
        raise MatrixError("matrix.performance_claim cannot exceed its source artifacts")
    requirements = _mapping(matrix["requirements"], label="matrix.requirements")
    if matrix["performance_claim"] and not all(requirements.get(key) is True for key in _REQUIREMENT_KEYS):
        raise MatrixError("matrix.performance_claim requires every promotion gate")
    expected_row_blockers = {
        f"row {row['id']}: {blocker}"
        for row in matrix["rows"]
        for blocker in row["eligibility"]["blockers"]
    }
    if not expected_row_blockers.issubset(set(eligibility["blockers"])):
        raise MatrixError("matrix.eligibility omits row blockers")
    coverage = _mapping(matrix["coverage"], label="matrix.coverage")
    actual_engines = sorted({str(row["engine"]) for row in matrix["rows"]})
    actual_surfaces = sorted({str(row["surface"]) for row in matrix["rows"]})
    actual_variants = sorted({str(row["path_variant"]) for row in matrix["rows"]})
    if coverage.get("engines") != actual_engines:
        raise MatrixError("matrix.coverage.engines is inconsistent")
    if coverage.get("surfaces") != actual_surfaces:
        raise MatrixError("matrix.coverage.surfaces is inconsistent")
    if coverage.get("path_variants") != actual_variants:
        raise MatrixError("matrix.coverage.path_variants is inconsistent")
    required_engines = coverage.get("required_engines")
    required_surfaces = coverage.get("required_surfaces")
    if not isinstance(required_engines, list) or not isinstance(required_surfaces, list):
        raise MatrixError("matrix.coverage required axes are missing")
    present_grid = {(str(row["engine"]), str(row["surface"])) for row in matrix["rows"]}
    grid_complete = all(
        (str(engine), str(surface)) in present_grid
        for engine in required_engines
        for surface in required_surfaces
    )
    if coverage.get("required_grid_complete") is not grid_complete:
        raise MatrixError("matrix.coverage.required_grid_complete is inconsistent")
    try:
        validate_artifact_provenance(
            _mapping(matrix["report_provenance"], label="matrix.report_provenance")
        )
    except ValueError as exc:
        raise MatrixError(f"matrix.report_provenance: {exc}") from exc
    return matrix
