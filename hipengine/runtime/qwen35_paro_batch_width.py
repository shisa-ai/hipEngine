"""Optional measured-cost hints for Qwen3.5/PARO native batch-width dispatch."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from hipengine.dispatch import NativeBatchWidthProfile


DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE = (
    "benchmarks/results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json"
)
_DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_PACKAGE = (
    Path(__file__).with_name("profiles") / "gfx1151_paro_native_c248.json"
)
QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_ENV = "HIPENGINE_QWEN35_NATIVE_BATCH_WIDTH_PROFILE"
QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_SCHEMA = 2
QWEN35_PARO_NATIVE_BATCH_WIDTH_ORACLE = "independent_single_request_prefill_decode"
_NATIVE_BATCH_WIDTH_KV_DTYPE = "bf16"
_NATIVE_BATCH_WIDTH_QUANT = "w4_paro"


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


def _normalize_widths(
    values: Sequence[int],
    *,
    label: str,
    max_width: int | None = None,
) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
            raise ValueError(f"{label} must contain integers greater than one")
        if max_width is not None and value > int(max_width):
            continue
        if value in seen:
            raise ValueError(f"{label} contains duplicate width {value}")
        seen.add(value)
        normalized.append(int(value))
    return tuple(sorted(normalized))


def native_batch_width_domain(
    max_batch_size: int,
    *,
    row_chunk_widths: Sequence[int] = (),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(executable_widths, whole_row_exact_widths)`` for a session.

    The kernel contract accepts two through ``max_batch_size`` live rows. The
    exact subset is the complement of the implementation's full-attention
    row-chunk widths, so it is derived per backend rather than from a model
    allowlist. Width one stays on the scalar serial path.
    """

    max_width = int(max_batch_size)
    if max_width < 2:
        return (), ()
    widths = tuple(range(2, max_width + 1))
    row_chunk = frozenset(int(width) for width in row_chunk_widths)
    exact = tuple(width for width in widths if width not in row_chunk)
    return widths, exact


def _normalize_geometry_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def load_qwen35_paro_native_batch_width_profile(
    artifact_path: str,
    *,
    backend: str,
    target_arch: str,
    model_path: str | Path,
    kv_dtype: str,
    quant: str = "w4_paro",
    native_widths: Sequence[int] | None = None,
    exact_widths: Sequence[int] | None = None,
    max_native_width: int | None = None,
    artifact_is_explicit: bool = False,
    config_signature: Mapping[str, object] | None = None,
) -> NativeBatchWidthProfile:
    """Load an optional measured cost hint for the executable native widths.

    ``native_widths`` is the caller-supplied executable domain derived from the
    backend/implementation contract. When omitted, the artifact's declared
    widths are used for direct callers. The artifact can annotate those widths
    with measured costs and record concrete failures; model identity, measured
    positions, a missing performance claim, and a different correctness oracle
    are cost notes, never native blockers. A missing default artifact uses
    nominal costs. An explicitly selected missing or malformed artifact raises
    ``ValueError`` with its configuration error.

    Negative findings apply only when the artifact's capability fields and its
    optional ``geometry`` object match ``config_signature``. Model directory
    identity never decides a failure. A legacy artifact without ``geometry``
    keeps its findings as provenance and cost notes rather than widening them
    to every model.
    """

    path = _retained_results_path(artifact_path)
    profile_path = path
    packaged_default = False
    if (
        not profile_path.is_file()
        and artifact_path == DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE
    ):
        profile_path = _DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_PACKAGE
        packaged_default = True

    supplied = (
        None
        if native_widths is None
        else _normalize_widths(native_widths, label="native_widths", max_width=max_native_width)
    )
    exact = (
        ()
        if exact_widths is None
        else _normalize_widths(exact_widths, label="exact_widths", max_width=max_native_width)
    )

    capability_blockers: list[str] = []
    if str(kv_dtype) != _NATIVE_BATCH_WIDTH_KV_DTYPE:
        capability_blockers.append("native c>N decode requires BF16 KV")
    if str(quant) != _NATIVE_BATCH_WIDTH_QUANT:
        capability_blockers.append("native c>N decode requires the PARO w4 quant")

    def _blocked(
        reason: str,
        *,
        widths: Sequence[int] = (),
        exact_widths_value: Sequence[int] = (),
        notes: Sequence[str] = (),
    ) -> NativeBatchWidthProfile:
        return NativeBatchWidthProfile(
            source_artifact=artifact_path,
            native_step_ms=(),
            serial_row_step_ms=1.0,
            native_widths=tuple(widths),
            exact_widths=tuple(exact_widths_value),
            blockers=tuple((*capability_blockers, reason)),
            cost_notes=tuple(notes),
        )

    def _nominal(
        reason: str,
        *,
        widths: Sequence[int],
        exact_widths_value: Sequence[int],
        notes: Sequence[str] = (),
    ) -> NativeBatchWidthProfile:
        return NativeBatchWidthProfile(
            source_artifact=artifact_path,
            native_step_ms=(),
            serial_row_step_ms=1.0,
            native_widths=tuple(widths),
            exact_widths=tuple(exact_widths_value),
            blockers=tuple(capability_blockers),
            cost_notes=tuple((*notes, f"{reason}; using nominal native costs")),
        )

    def _malformed(reason: str) -> NativeBatchWidthProfile:
        if artifact_is_explicit:
            raise ValueError(reason)
        if supplied is None:
            return _blocked(reason, notes=(reason,))
        return _nominal(reason, widths=supplied, exact_widths_value=exact)

    if supplied is not None and not supplied:
        return _blocked("no native batch width fits the session max_batch_size")

    if not profile_path.is_file():
        reason = "native batch width profile does not exist"
        if artifact_is_explicit:
            raise ValueError(reason)
        if supplied is None:
            return _blocked(reason, notes=(reason,))
        return _nominal(reason, widths=supplied, exact_widths_value=exact)

    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        reason = f"native batch width profile is not readable JSON: {exc}"
        if artifact_is_explicit:
            raise ValueError(reason) from exc
        if supplied is None:
            return _blocked(reason, notes=(reason,))
        return _nominal(reason, widths=supplied, exact_widths_value=exact)

    if packaged_default and (
        not isinstance(payload, Mapping) or payload.get("source_artifact") != artifact_path
    ):
        return _malformed(
            "packaged native batch width profile source artifact does not match the default"
        )
    if not isinstance(payload, Mapping):
        return _malformed("native batch width profile must contain a JSON object")

    embedded = payload.get("native_batch_width_profile")
    if not isinstance(embedded, Mapping):
        return _malformed("native batch width profile must contain an embedded profile object")
    if embedded.get("schema") != QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_SCHEMA:
        return _malformed(
            "native batch width profile embedded schema must equal "
            f"{QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_SCHEMA}"
        )

    widths_value = embedded.get("native_widths")
    if not isinstance(widths_value, list) or not widths_value:
        return _malformed("native batch width profile native_widths must be a non-empty list")
    try:
        artifact_widths = _normalize_widths(widths_value, label="native_widths")
    except ValueError as exc:
        return _malformed(str(exc))

    min_value = embedded.get("min_position")
    max_value = embedded.get("max_position")
    if (
        isinstance(min_value, bool)
        or not isinstance(min_value, int)
        or min_value < 0
        or isinstance(max_value, bool)
        or not isinstance(max_value, int)
        or max_value < min_value
    ):
        return _malformed("native batch width profile decode position range is invalid")
    min_position = int(min_value)
    max_position = int(max_value)

    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        return _malformed("native batch width profile protocol must be an object")
    if protocol.get("native_partition_widths") != list(artifact_widths):
        return _malformed(
            "native batch width profile protocol widths do not match the embedded profile"
        )
    if protocol.get("evidenced_decode_position_range") != {
        "min": min_position,
        "max": max_position,
    }:
        return _malformed(
            "native batch width profile protocol position range does not match the embedded profile"
        )

    hardware = payload.get("hardware")
    if not isinstance(hardware, Mapping):
        return _malformed("native batch width profile hardware must be an object")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        return _malformed("native batch width profile model must be an object")

    artifact_geometry = payload.get("geometry")
    geometry_declared = artifact_geometry is not None
    geometry_mismatches: list[str] = []
    if geometry_declared:
        if not isinstance(artifact_geometry, Mapping):
            return _malformed("native batch width profile geometry must be an object")
        if not artifact_geometry:
            geometry_declared = False
        elif config_signature is None:
            geometry_mismatches = ["<no session config signature>"]
        else:
            for key, artifact_value in artifact_geometry.items():
                if key not in config_signature:
                    geometry_mismatches.append(str(key))
                    continue
                if _normalize_geometry_value(artifact_value) != _normalize_geometry_value(
                    config_signature[key]
                ):
                    geometry_mismatches.append(str(key))
    geometry_matches = geometry_declared and not geometry_mismatches

    notes: list[str] = []
    capability_mismatches: list[str] = []
    for label, artifact_value, session_value in (
        ("embedded backend", embedded.get("backend"), str(backend)),
        ("embedded target architecture", embedded.get("target_arch"), str(target_arch)),
        ("embedded quant", embedded.get("quant"), str(quant)),
        ("embedded KV dtype", embedded.get("kv_storage"), str(kv_dtype)),
        ("hardware backend", hardware.get("backend"), str(backend)),
        ("hardware target architecture", hardware.get("target_arch"), str(target_arch)),
        ("model quant", model.get("quant"), str(quant)),
        ("model KV dtype", model.get("kv_storage"), str(kv_dtype)),
    ):
        if artifact_value != session_value:
            capability_mismatches.append(label)
    model_matches = (
        embedded.get("model_snapshot") == Path(model_path).name
        and model.get("snapshot") == Path(model_path).name
    )
    failure_applicable = not capability_mismatches and geometry_matches
    cost_applicable = (
        not capability_mismatches
        and model_matches
        and (geometry_matches if geometry_declared else True)
    )
    if capability_mismatches:
        notes.append(
            "native batch width profile capability fields do not match the session: "
            + ", ".join(capability_mismatches)
        )
    if not model_matches:
        notes.append("native batch width profile model snapshot does not match the session")
    if geometry_declared and geometry_mismatches:
        notes.append(
            "native batch width profile geometry does not match the session: "
            + ", ".join(geometry_mismatches)
        )

    def _failure_note(label: str) -> str:
        if not geometry_declared:
            return (
                f"native batch width profile records a failed {label} check, "
                "but the artifact declares no geometry; provenance only"
            )
        return (
            f"native batch width profile records a failed {label} check, "
            "but its capability/geometry is not applicable; provenance only"
        )

    def _width_failure_note(width: int) -> str:
        if not geometry_declared:
            return (
                f"native batch width profile c{width} records a failure, "
                "but the artifact declares no geometry; provenance only"
            )
        return (
            f"native batch width profile c{width} records a failure, "
            "but its capability/geometry is not applicable; provenance only"
        )

    performance_claim = payload.get("performance_claim") is True
    if not performance_claim:
        notes.append("native batch width profile is not an accepted performance claim")
    oracle_ok = protocol.get("correctness_oracle") == QWEN35_PARO_NATIVE_BATCH_WIDTH_ORACLE
    if not oracle_ok:
        notes.append(
            "native batch width profile does not use the independent single-request oracle"
        )

    for field, label in (
        ("packed_prefill_generated_token_equality", "packed prefill"),
        ("sparse_slot_generated_token_equality", "sparse-slot decode"),
        ("shrinking_batch_generated_token_equality", "shrinking-batch decode"),
    ):
        value = protocol.get(field)
        if failure_applicable:
            if value is None:
                notes.append(
                    f"native batch width profile lacks {label} evidence; not treated as a failure"
                )
            elif value is not True:
                return NativeBatchWidthProfile(
                    source_artifact=artifact_path,
                    native_step_ms=(),
                    serial_row_step_ms=1.0,
                    native_widths=tuple(supplied if supplied is not None else artifact_widths),
                    exact_widths=tuple(exact),
                    min_position=min_position,
                    max_position=max_position,
                    blockers=tuple(
                        (
                            *capability_blockers,
                            f"native batch width profile recorded a failed {label} check",
                        )
                    ),
                    cost_notes=tuple(notes),
                )
        elif value is False:
            notes.append(_failure_note(label))

    domain = list(supplied if supplied is not None else artifact_widths)
    if max_native_width is not None:
        domain = [width for width in domain if width <= int(max_native_width)]

    rows_payload = payload.get("rows")
    if not isinstance(rows_payload, Mapping):
        return _malformed("native batch width profile rows must be an object")

    measured_costs: list[tuple[int, float]] = []
    measured_available = cost_applicable and performance_claim and oracle_ok
    serial_row_step_ms = 1.0
    if measured_available:
        c1 = rows_payload.get("1")
        if not isinstance(c1, Mapping) or not _is_positive_finite(
            c1.get("decode_step_ms_median_of_run_medians")
        ):
            notes.append("native batch width profile c1 serial step cost is missing or invalid")
            measured_available = False
        elif c1.get("status") != "accepted_reference":
            notes.append("native batch width profile c1 is not an accepted independent reference")
            measured_available = False
        else:
            serial_row_step_ms = float(c1["decode_step_ms_median_of_run_medians"])

    excluded: list[str] = []
    if cost_applicable or failure_applicable or geometry_declared:
        for width in tuple(domain):
            row = rows_payload.get(str(width))
            if not isinstance(row, Mapping):
                if measured_available:
                    notes.append(
                        f"native batch width profile c{width} has no measured row; "
                        "using nominal cost"
                    )
                continue
            green = (
                row.get("status") == "accepted_exact"
                and row.get("generated_token_equality") is True
                and row.get("primitive_correctness") is True
            )
            if (
                row.get("status") in {"rejected", "failed"}
                or row.get("generated_token_equality") is False
                or row.get("primitive_correctness") is False
            ):
                if failure_applicable:
                    domain.remove(width)
                    excluded.append(f"c{width}")
                else:
                    notes.append(_width_failure_note(width))
                continue
            if not green:
                if measured_available:
                    notes.append(
                        f"native batch width profile c{width} lacks accepted exact evidence; "
                        "using nominal cost"
                    )
                continue
            if measured_available:
                cost_ms = row.get("decode_step_ms_median_of_run_medians")
                if not _is_positive_finite(cost_ms):
                    notes.append(
                        f"native batch width profile c{width} step cost is missing or invalid"
                    )
                    continue
                measured_costs.append((width, float(cost_ms)))
    if excluded:
        notes.append(
            "native batch width profile excluded recorded failures: " + ", ".join(excluded)
        )

    if not domain:
        reason = (
            "all native widths were excluded by recorded failures"
            if excluded
            else "no executable native width remains"
        )
        return _blocked(reason, notes=tuple(notes))

    if supplied is not None:
        exact_domain = tuple(width for width in exact if width in domain)
    else:
        exact_domain = ()

    if measured_available and not measured_costs:
        measured_available = False

    if not measured_available:
        notes.append(
            "native batch width profile provides no applicable measured costs; "
            "using nominal native costs"
        )
        return NativeBatchWidthProfile(
            source_artifact=artifact_path,
            native_step_ms=(),
            serial_row_step_ms=1.0,
            native_widths=tuple(sorted(domain)),
            exact_widths=tuple(sorted(exact_domain)),
            min_position=min_position,
            max_position=max_position,
            blockers=tuple(capability_blockers),
            cost_notes=tuple(dict.fromkeys(notes)),
        )

    return NativeBatchWidthProfile(
        source_artifact=artifact_path,
        native_step_ms=tuple(sorted(measured_costs)),
        serial_row_step_ms=serial_row_step_ms,
        native_widths=tuple(sorted(domain)),
        exact_widths=tuple(sorted(exact_domain)),
        min_position=min_position,
        max_position=max_position,
        blockers=tuple(capability_blockers),
        cost_notes=tuple(dict.fromkeys(notes)),
    )


__all__ = [
    "DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE",
    "QWEN35_PARO_NATIVE_BATCH_WIDTH_ORACLE",
    "QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_SCHEMA",
    "QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_ENV",
    "load_qwen35_paro_native_batch_width_profile",
    "native_batch_width_domain",
]
