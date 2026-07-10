"""Evidence loading for Qwen3.5/PARO native batch-width dispatch."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hipengine.dispatch import NativeBatchWidthProfile


DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE = (
    "benchmarks/results/2026-07-10-gfx1151-paro-cn-current-diagnostic-summary.json"
)
QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_ENV = "HIPENGINE_QWEN35_NATIVE_BATCH_WIDTH_PROFILE"


def _is_positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _retained_results_path(value: str) -> Path:
    relative = Path(value)
    if (
        relative.is_absolute()
        or relative.suffix != ".json"
        or len(relative.parts) < 3
        or relative.parts[:2] != ("benchmarks", "results")
        or ".." in relative.parts
    ):
        raise ValueError("native batch width profile must be a JSON path under benchmarks/results")
    results_root = (Path.cwd() / "benchmarks" / "results").resolve()
    path = Path.cwd() / relative
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise ValueError(f"native batch width profile path cannot be resolved: {exc}") from exc
    if not resolved.is_relative_to(results_root):
        raise ValueError("native batch width profile must resolve under benchmarks/results")
    current = path
    while current != Path.cwd() and current != current.parent:
        if current.is_symlink():
            raise ValueError("native batch width profile and its parents must not be symlinks")
        current = current.parent
    return path


def _blocked_profile(
    artifact_path: str,
    blockers: list[str],
    *,
    serial_row_step_ms: float = 1.0,
) -> NativeBatchWidthProfile:
    return NativeBatchWidthProfile(
        source_artifact=artifact_path,
        native_step_ms=(),
        serial_row_step_ms=serial_row_step_ms,
        blockers=tuple(blockers),
    )


def load_qwen35_paro_native_batch_width_profile(
    artifact_path: str,
    *,
    backend: str,
    target_arch: str,
    model_path: str | Path,
    kv_dtype: str,
    quant: str = "w4_paro",
) -> NativeBatchWidthProfile:
    """Load a full-identity-matched PARO native width profile.

    Invalid paths raise because they indicate a configuration error. Missing,
    malformed, mismatched, or correctness-red evidence returns a blocked profile
    so callers retain the exact serial route and can report the reason.
    """

    path = _retained_results_path(artifact_path)
    if not path.is_file():
        return _blocked_profile(artifact_path, ["native batch width profile does not exist"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _blocked_profile(artifact_path, [f"native batch width profile is not readable JSON: {exc}"])
    if not isinstance(payload, Mapping):
        return _blocked_profile(artifact_path, ["native batch width profile must contain a JSON object"])

    blockers: list[str] = []
    embedded = payload.get("native_batch_width_profile")
    if not isinstance(embedded, Mapping):
        blockers.append("native batch width profile must contain an embedded profile object")
        embedded = {}
    elif embedded.get("schema") != 1:
        blockers.append("native batch width profile embedded schema must equal 1")
    if embedded.get("backend") != str(backend):
        blockers.append("native batch width profile embedded backend does not match the resolved backend")
    if embedded.get("target_arch") != str(target_arch):
        blockers.append("native batch width profile embedded target architecture does not match the session")
    if embedded.get("model_snapshot") != Path(model_path).name:
        blockers.append("native batch width profile embedded model snapshot does not match the session")
    if embedded.get("quant") != str(quant):
        blockers.append("native batch width profile embedded quant does not match the session")
    if embedded.get("kv_storage") != str(kv_dtype):
        blockers.append("native batch width profile embedded KV dtype does not match the session")

    hardware = payload.get("hardware")
    if not isinstance(hardware, Mapping):
        blockers.append("native batch width profile hardware must be an object")
        hardware = {}
    if hardware.get("backend") != str(backend):
        blockers.append("native batch width profile backend does not match the resolved backend")
    if hardware.get("target_arch") != str(target_arch):
        blockers.append("native batch width profile target architecture does not match the session")

    model = payload.get("model")
    if not isinstance(model, Mapping):
        blockers.append("native batch width profile model must be an object")
        model = {}
    if model.get("snapshot") != Path(model_path).name:
        blockers.append("native batch width profile model snapshot does not match the session")
    if model.get("quant") != str(quant):
        blockers.append("native batch width profile quant does not match the session")
    if model.get("kv_storage") != str(kv_dtype):
        blockers.append("native batch width profile KV dtype does not match the session")

    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        blockers.append("native batch width profile protocol must be an object")
        protocol = {}
    widths_value = embedded.get("native_widths")
    widths: tuple[int, ...] = ()
    if not isinstance(widths_value, list) or not widths_value:
        blockers.append("native batch width profile native_partition_widths must be a non-empty list")
    elif any(isinstance(width, bool) or not isinstance(width, int) or width <= 1 for width in widths_value):
        blockers.append("native batch width profile widths must be integers greater than one")
    elif len(set(widths_value)) != len(widths_value):
        blockers.append("native batch width profile widths must be unique")
    else:
        widths = tuple(int(width) for width in widths_value)

    min_value = embedded.get("min_position")
    max_value = embedded.get("max_position")
    min_position: int | None = None
    max_position: int | None = None
    if (
        isinstance(min_value, bool)
        or not isinstance(min_value, int)
        or min_value < 0
        or isinstance(max_value, bool)
        or not isinstance(max_value, int)
        or max_value < min_value
    ):
        blockers.append("native batch width profile decode position range is invalid")
    else:
        min_position = int(min_value)
        max_position = int(max_value)
    if protocol.get("native_partition_widths") != list(widths):
        blockers.append("native batch width profile protocol widths do not match the embedded profile")
    if protocol.get("evidenced_decode_position_range") != {
        "min": min_position,
        "max": max_position,
    }:
        blockers.append("native batch width profile protocol position range does not match the embedded profile")

    rows_payload = payload.get("rows")
    if not isinstance(rows_payload, Mapping):
        blockers.append("native batch width profile rows must be an object")
        rows_payload = {}
    serial_row_step_ms = 1.0
    c1 = rows_payload.get("1")
    if not isinstance(c1, Mapping) or not _is_positive_finite(c1.get("decode_step_ms_median_of_run_medians")):
        blockers.append("native batch width profile c1 serial step cost is missing or invalid")
    else:
        serial_row_step_ms = float(c1["decode_step_ms_median_of_run_medians"])

    native_step_ms: list[tuple[int, float]] = []
    for width in widths:
        row = rows_payload.get(str(width))
        if not isinstance(row, Mapping):
            blockers.append(f"native batch width profile c{width} row is missing")
            continue
        if row.get("status") != "diagnostic_exact":
            blockers.append(f"native batch width profile c{width} is not diagnostic_exact")
        if row.get("generated_token_equality") is not True:
            blockers.append(f"native batch width profile c{width} generated-token equality is not green")
        if row.get("primitive_correctness") is not True:
            blockers.append(f"native batch width profile c{width} primitive correctness is not green")
        cost_ms = row.get("decode_step_ms_median_of_run_medians")
        if not _is_positive_finite(cost_ms):
            blockers.append(f"native batch width profile c{width} step cost is missing or invalid")
        else:
            native_step_ms.append((width, float(cost_ms)))

    if blockers:
        return _blocked_profile(
            artifact_path,
            blockers,
            serial_row_step_ms=serial_row_step_ms,
        )
    return NativeBatchWidthProfile(
        source_artifact=artifact_path,
        native_step_ms=tuple(native_step_ms),
        serial_row_step_ms=serial_row_step_ms,
        min_position=min_position,
        max_position=max_position,
    )


__all__ = [
    "DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE",
    "QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_ENV",
    "load_qwen35_paro_native_batch_width_profile",
]
