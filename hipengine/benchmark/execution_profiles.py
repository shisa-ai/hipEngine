"""Scenario-driven execution-profile evaluation.

The evaluator is model-agnostic: model adapters capture aligned strict-teacher
logits plus exact control records, while this module owns stable distribution
metrics, dynamic-scenario checks, repeat/isolation comparisons, and compact
artifact construction.  It is torch-free and keeps large logits out of retained
JSON artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.execution_profiles import (
    EXECUTION_PROFILE_SCHEMA_VERSION,
    ExecutionProfile,
    manifest_sha256,
    validate_variant_manifest,
)
from scripts.quant_quality.metrics import per_row_metrics


EXECUTION_PROFILE_EVALUATION_KIND = "hipengine_execution_profile_evaluation"
EXECUTION_PROFILE_EVALUATION_SCHEMA_VERSION = 1
EXECUTION_PROFILE_CAPTURE_KIND = "hipengine_execution_profile_capture"
EXECUTION_PROFILE_CAPTURE_SCHEMA_VERSION = 1
EXECUTION_PROFILE_CONTROL_FIXTURE_KIND = "hipengine_execution_profile_control_fixture"
EXECUTION_PROFILE_CONTROL_FIXTURE_SCHEMA_VERSION = 1
EXECUTION_PROFILE_CONTROL_CAPTURE_KIND = "hipengine_execution_profile_control_capture"
EXECUTION_PROFILE_CONTROL_CAPTURE_SCHEMA_VERSION = 1
_DEFAULT_PERCENTILES = (50.0, 95.0, 99.0)


@dataclass(frozen=True, slots=True)
class EvaluationThresholds:
    mean_kl_max: float = 1.0e-3
    p95_kl_max: float = 5.0e-3
    p99_kl_max: float = 2.0e-2
    max_kl_max: float = 5.0e-2
    top1_min: float = 0.99
    per_scope_top1_min: float = 0.97
    review_kl: float = 2.0e-2

    def __post_init__(self) -> None:
        for field in (
            "mean_kl_max",
            "p95_kl_max",
            "p99_kl_max",
            "max_kl_max",
            "review_kl",
        ):
            value = float(getattr(self, field))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
        for field in ("top1_min", "per_scope_top1_min"):
            value = float(getattr(self, field))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
        if self.review_kl > self.max_kl_max:
            raise ValueError("review_kl cannot exceed max_kl_max")

    def to_dict(self) -> dict[str, float]:
        return {field: float(value) for field, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class Bf16NoninferiorityThresholds:
    mean_kl_delta_max: float = 1.0e-3
    top1_drop_max: float = 0.01
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 1234

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean_kl_delta_max) or self.mean_kl_delta_max < 0.0:
            raise ValueError("mean_kl_delta_max must be finite and non-negative")
        if not np.isfinite(self.top1_drop_max) or not 0.0 <= self.top1_drop_max <= 1.0:
            raise ValueError("top1_drop_max must be in [0, 1]")
        if self.bootstrap_samples <= 0 or self.bootstrap_seed < 0:
            raise ValueError("bootstrap_samples must be positive and bootstrap_seed non-negative")


@dataclass(frozen=True, slots=True)
class RowDescriptor:
    scenario_id: str
    scenario_step: int
    request_id: str
    teacher_step: int
    category: str
    shape: str
    transition: str
    teacher_token_id: int = 0

    def __post_init__(self) -> None:
        for field in ("scenario_id", "request_id", "category", "shape", "transition"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"row {field} must be a non-empty string")
        for field in ("scenario_step", "teacher_step", "teacher_token_id"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"row {field} must be a non-negative integer")

    @property
    def logical_key(self) -> tuple[str, int]:
        return (self.request_id, self.teacher_step)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ControlRecord:
    scenario_id: str
    scenario_step: int
    work_class: str
    request_id: str
    physical_slot: int
    execution_row: int
    physical_width: int
    input_token_id: int
    position: int
    context_length: int
    active: bool
    active_mask_hash: str
    mask_manifest_hash: str
    publication_ordinal: int
    transaction_id: str
    transaction_phase: str
    accepted_token_count: int
    route_decision_hash: str
    route_scatter_owner_hash: str
    route_owner_request_id: str
    route_top_k: int
    kv_base_offset: int
    kv_live_count: int
    kv_token_position: int
    kv_evict: bool
    kv_values_finite: bool
    kv_append_ordinal: int
    state_owner_request_id: str
    state_update_ordinal: int
    state_values_finite: bool
    rng_owner_request_id: str
    rng_seed: int
    rng_counter: int
    route_values_finite: bool
    graph_bucket: str

    def __post_init__(self) -> None:
        for field in (
            "scenario_id",
            "work_class",
            "request_id",
            "active_mask_hash",
            "mask_manifest_hash",
            "transaction_id",
            "transaction_phase",
            "route_decision_hash",
            "route_scatter_owner_hash",
            "route_owner_request_id",
            "state_owner_request_id",
            "rng_owner_request_id",
            "graph_bucket",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"control {field} must be a non-empty string")
        for field in (
            "scenario_step",
            "physical_slot",
            "execution_row",
            "physical_width",
            "input_token_id",
            "publication_ordinal",
            "accepted_token_count",
            "route_top_k",
            "position",
            "context_length",
            "kv_base_offset",
            "kv_live_count",
            "kv_token_position",
            "kv_append_ordinal",
            "state_update_ordinal",
            "rng_seed",
            "rng_counter",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"control {field} must be a non-negative integer")
        for field in (
            "active",
            "kv_evict",
            "kv_values_finite",
            "state_values_finite",
            "route_values_finite",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"control {field} must be boolean")
        if self.physical_width <= 0:
            raise ValueError("control physical_width must be positive")
        if self.execution_row >= self.physical_width:
            raise ValueError("control execution_row must be below physical_width")

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.scenario_id, self.scenario_step, self.work_class, self.request_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunCapture:
    execution_profile: str
    scenario_id: str
    run_id: str
    variant_manifest_sha256: str
    repeat_index: int
    rows: tuple[RowDescriptor, ...]
    logits: np.ndarray
    selected_token_ids: tuple[int, ...]
    controls: tuple[ControlRecord, ...]
    _sha256_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            profile = (
                self.execution_profile.value
                if isinstance(self.execution_profile, ExecutionProfile)
                else ExecutionProfile(str(self.execution_profile)).value
            )
        except ValueError as exc:
            raise ValueError(f"unknown execution profile: {self.execution_profile!r}") from exc
        if not self.scenario_id:
            raise ValueError("capture scenario_id must be non-empty")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("capture run_id must be a non-empty string")
        try:
            valid_manifest_hash = (
                len(self.variant_manifest_sha256) == 64
                and int(self.variant_manifest_sha256, 16) >= 0
            )
        except (TypeError, ValueError):
            valid_manifest_hash = False
        if not valid_manifest_hash:
            raise ValueError("capture variant_manifest_sha256 must be a SHA-256 hex digest")
        if type(self.repeat_index) is not int or self.repeat_index < 0:
            raise ValueError("capture repeat_index must be a non-negative integer")
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("capture needs at least one logical row")
        logical_keys = [row.logical_key for row in rows]
        if len(set(logical_keys)) != len(logical_keys):
            raise ValueError("capture has duplicate logical row")
        if any(row.scenario_id != self.scenario_id for row in rows):
            raise ValueError("capture row scenario_id differs from capture scenario_id")
        logits = self.logits if isinstance(self.logits, np.ndarray) else np.asarray(self.logits)
        if logits.ndim != 2 or logits.shape[0] != len(rows) or logits.shape[1] < 1:
            raise ValueError(
                "capture logits must have shape [len(rows), non-empty vocab], got "
                f"{logits.shape!r} for {len(rows)} rows"
            )
        # Keep large read-only mmap captures out of RAM. Direct writable arrays
        # are copied once so the frozen capture cannot be mutated by its caller.
        if not logits.flags.c_contiguous or logits.flags.writeable:
            logits = np.ascontiguousarray(logits).copy()
        logits.setflags(write=False)
        selected = tuple(self.selected_token_ids)
        if (
            len(selected) != len(rows)
            or any(type(token) is not int or token < 0 for token in selected)
        ):
            raise ValueError(
                "selected_token_ids must be non-negative integers aligned with rows"
            )
        controls = tuple(self.controls)
        if any(control.scenario_id != self.scenario_id for control in controls):
            raise ValueError("capture control scenario_id differs from capture scenario_id")
        _control_map(controls)
        object.__setattr__(self, "execution_profile", profile)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "selected_token_ids", selected)
        object.__setattr__(self, "controls", controls)

    def sha256(self) -> str:
        if self._sha256_cache is not None:
            return self._sha256_cache
        digest = hashlib.sha256()
        metadata = {
            "execution_profile": self.execution_profile,
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "variant_manifest_sha256": self.variant_manifest_sha256,
            "repeat_index": self.repeat_index,
            "rows": [row.to_dict() for row in self.rows],
            "selected_token_ids": list(self.selected_token_ids),
            "controls": [control.to_dict() for control in self.controls],
            "logits_dtype": self.logits.dtype.str,
            "logits_shape": list(self.logits.shape),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        byte_view = self.logits.view(np.uint8).reshape(-1)
        for offset in range(0, byte_view.size, 8 * 1024 * 1024):
            digest.update(memoryview(byte_view[offset : offset + 8 * 1024 * 1024]))
        value = digest.hexdigest()
        object.__setattr__(self, "_sha256_cache", value)
        return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run_capture_manifest(
    payload: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> RunCapture:
    """Load a capture from a small JSON manifest plus an external ``.npy``."""

    required = {
        "kind",
        "schema_version",
        "execution_profile",
        "scenario_id",
        "run_id",
        "variant_manifest_sha256",
        "repeat_index",
        "logits_path",
        "rows",
        "selected_token_ids",
        "controls",
    }
    optional = {"logits_sha256"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"capture manifest missing fields: {sorted(missing)}")
    unknown = set(payload) - required - optional
    if unknown:
        raise ValueError(f"capture manifest has unknown fields: {sorted(unknown)}")
    if payload.get("kind") != EXECUTION_PROFILE_CAPTURE_KIND:
        raise ValueError(f"capture manifest kind must be {EXECUTION_PROFILE_CAPTURE_KIND!r}")
    if payload.get("schema_version") != EXECUTION_PROFILE_CAPTURE_SCHEMA_VERSION:
        raise ValueError(
            f"capture schema_version must be {EXECUTION_PROFILE_CAPTURE_SCHEMA_VERSION}"
        )
    logits_path_value = payload.get("logits_path")
    if not isinstance(logits_path_value, str) or not logits_path_value:
        raise ValueError("capture logits_path must be a non-empty string")
    logits_path = Path(logits_path_value)
    if not logits_path.is_absolute():
        logits_path = Path(base_dir) / logits_path
    expected_sha = payload.get("logits_sha256")
    if expected_sha is not None:
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError("capture logits_sha256 must be a 64-character string")
        actual_sha = _file_sha256(logits_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"capture logits_sha256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
    logits = np.load(logits_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(logits, np.ndarray):
        raise ValueError("capture logits_path must refer to one .npy array, not an archive")
    try:
        rows = tuple(RowDescriptor(**dict(row)) for row in payload["rows"])
        controls = tuple(ControlRecord(**dict(record)) for record in payload["controls"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid capture row/control record: {exc}") from exc
    return RunCapture(
        execution_profile=str(payload["execution_profile"]),
        scenario_id=str(payload["scenario_id"]),
        run_id=payload["run_id"],
        variant_manifest_sha256=payload["variant_manifest_sha256"],
        repeat_index=payload["repeat_index"],
        rows=rows,
        logits=logits,
        selected_token_ids=tuple(payload["selected_token_ids"]),
        controls=controls,
    )


def load_run_capture_manifest(path: str | Path) -> RunCapture:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("capture manifest root must be an object")
    return parse_run_capture_manifest(payload, base_dir=manifest_path.parent)


def load_control_capture(path: str | Path) -> tuple[str, tuple[ControlRecord, ...]]:
    """Load actual runtime control telemetry and its required run identity."""

    capture_path = Path(path)
    with capture_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("control capture root must be an object")
    required = {"kind", "schema_version", "scenario_id", "run_id", "controls"}
    if set(payload) != required:
        raise ValueError(
            "control capture fields must be exactly " f"{sorted(required)}, got {sorted(payload)}"
        )
    if payload.get("kind") != EXECUTION_PROFILE_CONTROL_CAPTURE_KIND:
        raise ValueError(
            f"control capture kind must be {EXECUTION_PROFILE_CONTROL_CAPTURE_KIND!r}"
        )
    if payload.get("schema_version") != EXECUTION_PROFILE_CONTROL_CAPTURE_SCHEMA_VERSION:
        raise ValueError(
            "control capture schema_version must be "
            f"{EXECUTION_PROFILE_CONTROL_CAPTURE_SCHEMA_VERSION}"
        )
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("control capture run_id must be a non-empty string")
    try:
        controls = tuple(ControlRecord(**dict(record)) for record in payload["controls"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid captured control record: {exc}") from exc
    if not controls or any(record.scenario_id != payload["scenario_id"] for record in controls):
        raise ValueError("control capture needs records matching its scenario_id")
    return run_id, controls


def load_control_fixture(path: str | Path) -> tuple[ControlRecord, ...]:
    fixture_path = Path(path)
    with fixture_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("control fixture root must be an object")
    required = {"kind", "schema_version", "scenario_id", "controls"}
    if set(payload) != required:
        raise ValueError(
            "control fixture fields must be exactly " f"{sorted(required)}, got {sorted(payload)}"
        )
    if payload.get("kind") != EXECUTION_PROFILE_CONTROL_FIXTURE_KIND:
        raise ValueError(
            f"control fixture kind must be {EXECUTION_PROFILE_CONTROL_FIXTURE_KIND!r}"
        )
    if payload.get("schema_version") != EXECUTION_PROFILE_CONTROL_FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            "control fixture schema_version must be "
            f"{EXECUTION_PROFILE_CONTROL_FIXTURE_SCHEMA_VERSION}"
        )
    try:
        controls = tuple(ControlRecord(**dict(record)) for record in payload["controls"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid fixture control record: {exc}") from exc
    if not controls or any(record.scenario_id != payload["scenario_id"] for record in controls):
        raise ValueError("fixture needs controls matching its scenario_id")
    return controls


def qwen36_rows_from_teacher_fixture(
    fixture: Mapping[str, Any],
    *,
    scenario_id: str,
    shape: str = "c1",
) -> tuple[RowDescriptor, ...]:
    """Adapt the existing Qwen3.6 quant-quality teacher fixture to profile rows."""

    if fixture.get("kind") != "quant_quality_teacher_fixture":
        raise ValueError("Qwen3.6 fixture kind must be quant_quality_teacher_fixture")
    prompts = fixture.get("prompts")
    teacher_steps = fixture.get("teacher_steps")
    if type(teacher_steps) is not int or teacher_steps <= 0:
        raise ValueError("Qwen3.6 fixture teacher_steps must be positive")
    if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)) or not prompts:
        raise ValueError("Qwen3.6 fixture prompts must be a non-empty sequence")
    rows: list[RowDescriptor] = []
    global_step = 0
    for prompt in prompts:
        if not isinstance(prompt, Mapping):
            raise ValueError("Qwen3.6 fixture prompt must be an object")
        teacher = prompt.get("teacher_token_ids")
        if not isinstance(teacher, Sequence) or isinstance(teacher, (str, bytes)):
            raise ValueError("Qwen3.6 fixture teacher_token_ids must be a sequence")
        if len(teacher) != teacher_steps:
            raise ValueError("Qwen3.6 fixture prompt has the wrong teacher-step count")
        request_id = str(prompt.get("id", ""))
        category = str(prompt.get("category", ""))
        for teacher_step, teacher_token_id in enumerate(teacher):
            rows.append(
                RowDescriptor(
                    scenario_id=scenario_id,
                    scenario_step=global_step,
                    request_id=request_id,
                    teacher_step=teacher_step,
                    category=category,
                    shape="prefill_last" if teacher_step == 0 else shape,
                    transition="prefill_to_c1" if teacher_step == 0 else "steady",
                    teacher_token_id=int(teacher_token_id),
                )
            )
            global_step += 1
    return tuple(rows)


def _control_map(
    records: Sequence[ControlRecord],
) -> dict[tuple[str, int, str, str], ControlRecord]:
    result: dict[tuple[str, int, str, str], ControlRecord] = {}
    for record in records:
        if record.key in result:
            raise ValueError(f"duplicate control record: {record.key!r}")
        result[record.key] = record
    return result


def summarize_scenario(records: Sequence[ControlRecord]) -> dict[str, Any]:
    """Summarize dynamic widths, raggedness, and slot movement."""

    _control_map(records)
    by_step: dict[int, list[ControlRecord]] = {}
    for record in records:
        by_step.setdefault(record.scenario_step, []).append(record)
    width_sequence: list[int] = []
    ragged_steps: list[int] = []
    active_counts: list[int] = []
    for step in sorted(by_step):
        step_records = by_step[step]
        widths = {record.physical_width for record in step_records}
        if len(widths) != 1:
            raise ValueError(f"scenario step {step} has inconsistent physical widths")
        width = next(iter(widths))
        active = [record for record in step_records if record.active]
        slots = [record.physical_slot for record in active]
        rows = [record.execution_row for record in active]
        if len(set(slots)) != len(slots):
            raise ValueError(f"scenario step {step} has an active physical-slot collision")
        if len(set(rows)) != len(rows):
            raise ValueError(f"scenario step {step} has an active execution-row collision")
        if len(active) > width:
            raise ValueError(f"scenario step {step} has more active records than width")
        width_sequence.append(width)
        active_counts.append(len(active))
        if len({record.context_length for record in active}) > 1:
            ragged_steps.append(step)

    previous_slot: dict[str, int] = {}
    previous_progress: dict[str, tuple[int, int, int, int]] = {}
    previous_base: dict[str, int] = {}
    compactions = 0
    page_boundaries = 0
    for record in sorted(records, key=lambda item: (item.scenario_step, item.request_id)):
        prior_slot = previous_slot.get(record.request_id)
        if prior_slot is not None and prior_slot != record.physical_slot:
            compactions += 1
        previous_slot[record.request_id] = record.physical_slot
        progress = (
            record.position,
            record.publication_ordinal,
            record.kv_append_ordinal,
            record.state_update_ordinal,
        )
        prior_progress = previous_progress.get(record.request_id)
        if prior_progress is not None and any(
            current < previous
            for previous, current in zip(prior_progress, progress, strict=True)
        ):
            raise ValueError(
                f"request {record.request_id!r} regressed position/update ordinals"
            )
        previous_progress[record.request_id] = progress
        prior_base = previous_base.get(record.request_id)
        if prior_base is not None and prior_base != record.kv_base_offset:
            page_boundaries += 1
        previous_base[record.request_id] = record.kv_base_offset
    width_transitions = sum(
        int(previous != current)
        for previous, current in zip(width_sequence, width_sequence[1:], strict=False)
    )
    work_classes = sorted({record.work_class for record in records})
    admission_steps = sorted(
        {record.scenario_step for record in records if record.work_class == "ADMIT"}
    )
    cancellation_steps = sorted(
        {record.scenario_step for record in records if record.work_class == "CANCEL"}
    )
    retirement_steps = sorted(
        {
            record.scenario_step
            for record in records
            if record.work_class in {"RETIRE", "COMPLETE"}
        }
    )
    reclaim_steps = sorted(
        {record.scenario_step for record in records if record.work_class == "RECLAIM"}
    )
    sparse_retirement_steps = [
        step
        for step, previous, current in zip(
            sorted(by_step)[1:], active_counts[:-1], active_counts[1:], strict=True
        )
        if current < previous
    ]
    first_step = min(by_step) if by_step else 0
    return {
        "records": len(records),
        "steps": len(by_step),
        "width_sequence": width_sequence,
        "active_counts": active_counts,
        "width_transition_count": width_transitions,
        "compaction_count": compactions,
        "page_boundary_count": page_boundaries,
        "ragged_steps": ragged_steps,
        "sparse_retirement_steps": sparse_retirement_steps,
        "admission_steps": admission_steps,
        "delayed_admission_steps": [step for step in admission_steps if step > first_step],
        "cancellation_steps": cancellation_steps,
        "retirement_steps": retirement_steps,
        "reclaim_steps": reclaim_steps,
        "work_classes": work_classes,
        "request_count": len({record.request_id for record in records}),
    }


def compare_control_records(
    expected: Sequence[ControlRecord],
    actual: Sequence[ControlRecord],
    *,
    diagnostic_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """Compare controls and separate binding from declared diagnostic fields."""

    diagnostic = set(diagnostic_fields)
    valid_fields = set(ControlRecord.__dataclass_fields__)
    if not diagnostic <= valid_fields:
        raise ValueError(f"unknown diagnostic control fields: {sorted(diagnostic - valid_fields)}")
    expected_map = _control_map(expected)
    actual_map = _control_map(actual)
    mismatches: list[dict[str, Any]] = []
    diagnostic_mismatches: list[dict[str, Any]] = []
    for key in sorted(set(expected_map) | set(actual_map)):
        expected_record = expected_map.get(key)
        actual_record = actual_map.get(key)
        key_payload = list(key)
        if expected_record is None or actual_record is None:
            mismatches.append(
                {
                    "key": key_payload,
                    "field": "__record__",
                    "expected": None if expected_record is None else expected_record.to_dict(),
                    "actual": None if actual_record is None else actual_record.to_dict(),
                }
            )
            continue
        expected_payload = expected_record.to_dict()
        actual_payload = actual_record.to_dict()
        for field in expected_payload:
            if expected_payload[field] != actual_payload[field]:
                mismatch = {
                    "key": key_payload,
                    "field": field,
                    "expected": expected_payload[field],
                    "actual": actual_payload[field],
                }
                if field in diagnostic:
                    diagnostic_mismatches.append(mismatch)
                else:
                    mismatches.append(mismatch)
    try:
        actual_summary = summarize_scenario(actual)
        structural_error = None
    except ValueError as exc:
        actual_summary = None
        structural_error = str(exc)
        mismatches.append(
            {
                "key": [],
                "field": "__scenario_structure__",
                "expected": "valid",
                "actual": structural_error,
            }
        )
    return {
        "passed": not mismatches,
        "expected_records": len(expected),
        "actual_records": len(actual),
        "mismatches": mismatches,
        "diagnostic_fields": sorted(diagnostic),
        "diagnostic_mismatches": diagnostic_mismatches,
        "expected_scenario": summarize_scenario(expected),
        "actual_scenario": actual_summary,
        "structural_error": structural_error,
    }


def _metric_vectors(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    labels: np.ndarray,
    top_k: int,
) -> dict[str, np.ndarray]:
    reference = np.asarray(reference_logits)
    candidate = np.asarray(candidate_logits)
    if reference.ndim != 2 or reference.shape != candidate.shape or reference.shape[1] < 1:
        raise ValueError(
            f"strict/candidate logits must share [rows, vocab], got "
            f"{reference.shape!r} and {candidate.shape!r}"
        )
    if top_k <= 0 or top_k > reference.shape[1]:
        raise ValueError(f"top_k must be in [1, {reference.shape[1]}]")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("strict/candidate logits must be finite")

    row_count = reference.shape[0]
    # Reuse the canonical quant-quality row core so profile and BF16 packets use
    # identical KL, teacher-NLL, delta-p, top-1, and top-k definitions.
    canonical = per_row_metrics(
        reference,
        candidate,
        np.asarray(labels, dtype=np.int64),
        top_k=top_k,
    )
    result = {
        "kl": canonical["kl_nats"],
        "top1_equal": canonical["top1_equal"],
        "topk_overlap": canonical["topk_set_overlap"],
        "reference_teacher_nll": canonical["reference_teacher_nll_nats"],
        "candidate_teacher_nll": canonical["teacher_nll_nats"],
        "teacher_delta_p": canonical["delta_p"],
        "strict_margin": np.empty(row_count, dtype=np.float64),
        "strict_top1_candidate_rank": np.empty(row_count, dtype=np.int64),
        "strict_top1_token_id": np.argmax(reference, axis=1).astype(np.int64),
        "candidate_top1_token_id": np.argmax(candidate, axis=1).astype(np.int64),
        "max_abs_logit_delta": canonical["max_abs_logit_delta"],
    }
    for index in range(row_count):
        strict_row = np.asarray(reference[index], dtype=np.float64)
        candidate_row = np.asarray(candidate[index], dtype=np.float64)
        strict_top1 = int(np.argmax(strict_row))
        if strict_row.size == 1:
            # A one-token vocabulary has no runner-up; keep retained JSON finite.
            result["strict_margin"][index] = 0.0
        else:
            top_two = np.partition(strict_row, -2)[-2:]
            result["strict_margin"][index] = float(top_two.max() - top_two.min())
        result["strict_top1_candidate_rank"][index] = int(
            1 + np.count_nonzero(candidate_row > candidate_row[strict_top1])
        )
    return result


def _summary(vectors: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, Any]:
    kl = vectors["kl"][indices]
    top1 = vectors["top1_equal"][indices]
    overlap = vectors["topk_overlap"][indices]
    margins = vectors["strict_margin"][indices]
    max_abs = vectors["max_abs_logit_delta"][indices]
    reference_nll = vectors["reference_teacher_nll"][indices]
    candidate_nll = vectors["candidate_teacher_nll"][indices]
    teacher_delta_p = vectors["teacher_delta_p"][indices]
    percentiles = np.percentile(kl, _DEFAULT_PERCENTILES)
    return {
        "rows": int(indices.size),
        "kl_mean": float(kl.mean()),
        "kl_p50": float(percentiles[0]),
        "kl_p95": float(percentiles[1]),
        "kl_p99": float(percentiles[2]),
        "kl_max": float(kl.max()),
        "top1_agreement": float(top1.mean()),
        "top1_matches": int(top1.sum()),
        "topk_overlap_mean": float(overlap.mean()),
        "strict_teacher_nll_mean": float(reference_nll.mean()),
        "candidate_teacher_nll_mean": float(candidate_nll.mean()),
        "teacher_nll_delta": float(candidate_nll.mean() - reference_nll.mean()),
        "teacher_delta_p_mean": float(teacher_delta_p.mean()),
        "strict_margin_min": float(margins.min()),
        "max_abs_logit_delta": float(max_abs.max()),
    }


def _summary_passes(
    summary: Mapping[str, Any],
    thresholds: EvaluationThresholds,
    *,
    top1_min: float,
) -> bool:
    return bool(
        summary["kl_mean"] <= thresholds.mean_kl_max
        and summary["kl_p95"] <= thresholds.p95_kl_max
        and summary["kl_p99"] <= thresholds.p99_kl_max
        and summary["kl_max"] <= thresholds.max_kl_max
        and summary["top1_agreement"] >= top1_min
    )


def _row_diagnostic(
    row: RowDescriptor,
    index: int,
    vectors: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        **row.to_dict(),
        "row_index": int(index),
        "kl": float(vectors["kl"][index]),
        "top1_equal": bool(vectors["top1_equal"][index]),
        "strict_top1_token_id": int(vectors["strict_top1_token_id"][index]),
        "candidate_top1_token_id": int(vectors["candidate_top1_token_id"][index]),
        "topk_overlap": float(vectors["topk_overlap"][index]),
        "strict_margin": float(vectors["strict_margin"][index]),
        "strict_top1_candidate_rank": int(
            vectors["strict_top1_candidate_rank"][index]
        ),
        "strict_teacher_nll": float(vectors["reference_teacher_nll"][index]),
        "candidate_teacher_nll": float(vectors["candidate_teacher_nll"][index]),
        "teacher_delta_p": float(vectors["teacher_delta_p"][index]),
        "max_abs_logit_delta": float(vectors["max_abs_logit_delta"][index]),
    }


def compare_profile_logits(
    strict_logits: np.ndarray,
    candidate_logits: np.ndarray,
    rows: Sequence[RowDescriptor],
    *,
    thresholds: EvaluationThresholds | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare aligned strict-teacher logits and apply profile tail gates."""

    threshold = thresholds or EvaluationThresholds()
    row_tuple = tuple(rows)
    strict_array = np.asarray(strict_logits)
    candidate_array = np.asarray(candidate_logits)
    if strict_array.ndim != 2 or candidate_array.ndim != 2:
        raise ValueError("strict/candidate logits must be rank-2")
    if strict_array.shape != candidate_array.shape or strict_array.shape[1] < 1:
        raise ValueError("strict/candidate logits must share a non-empty vocabulary shape")
    if strict_array.shape[0] != len(row_tuple):
        raise ValueError("row descriptors must align with strict/candidate logits")
    if not row_tuple:
        raise ValueError("profile comparison needs at least one row")
    if not np.isfinite(strict_array).all() or not np.isfinite(candidate_array).all():
        return {
            "finite": False,
            "top_k": min(top_k, int(strict_array.shape[1])),
            "thresholds": threshold.to_dict(),
            "summary": {"rows": len(row_tuple), "vocab_size": int(strict_array.shape[1])},
            "by_scope": {},
            "scope_failures": [
                {"dimension": "global", "value": "non_finite_logits"}
            ],
            "rows_over_review_boundary": [],
            "top1_mismatch_rows": [],
            "hard_gates_passed": False,
            "requires_outlier_review": False,
            "eligible_for_automatic_admission": False,
        }
    effective_top_k = min(top_k, int(strict_array.shape[1]))
    labels = np.asarray([row.teacher_token_id for row in row_tuple], dtype=np.int64)
    vectors = _metric_vectors(
        strict_logits,
        candidate_logits,
        labels=labels,
        top_k=effective_top_k,
    )
    all_indices = np.arange(len(row_tuple), dtype=np.int64)
    overall = _summary(vectors, all_indices)
    by_scope: dict[str, dict[str, Any]] = {}
    scope_failures: list[dict[str, str]] = []
    for dimension in ("category", "shape", "transition"):
        groups: dict[str, Any] = {}
        values = sorted({str(getattr(row, dimension)) for row in row_tuple})
        for value in values:
            indices = np.asarray(
                [index for index, row in enumerate(row_tuple) if getattr(row, dimension) == value],
                dtype=np.int64,
            )
            summary = _summary(vectors, indices)
            summary["passed"] = _summary_passes(
                summary, threshold, top1_min=threshold.per_scope_top1_min
            )
            if not summary["passed"]:
                scope_failures.append({"dimension": dimension, "value": value})
            groups[value] = summary
        by_scope[dimension] = groups

    over_review = np.flatnonzero(vectors["kl"] > threshold.review_kl)
    review_rows = [
        _row_diagnostic(row_tuple[int(index)], int(index), vectors)
        for index in over_review
    ]
    top1_mismatches = np.flatnonzero(~vectors["top1_equal"])
    top1_mismatch_rows = [
        _row_diagnostic(row_tuple[int(index)], int(index), vectors)
        for index in top1_mismatches
    ]
    hard_pass = (
        _summary_passes(overall, threshold, top1_min=threshold.top1_min)
        and not scope_failures
    )
    requires_review = bool(review_rows)
    return {
        "finite": True,
        "top_k": effective_top_k,
        "thresholds": threshold.to_dict(),
        "summary": overall,
        "by_scope": by_scope,
        "scope_failures": scope_failures,
        "rows_over_review_boundary": review_rows,
        "top1_mismatch_rows": top1_mismatch_rows,
        "hard_gates_passed": bool(hard_pass),
        "requires_outlier_review": requires_review,
        "eligible_for_automatic_admission": bool(hard_pass and not requires_review),
    }


def _rows_compatible(first: RunCapture, second: RunCapture) -> bool:
    return first.rows == second.rows


def _control_fixture_sha256(records: Sequence[ControlRecord]) -> str:
    encoded = json.dumps(
        [record.to_dict() for record in sorted(records, key=lambda item: item.key)],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _attach_comparison_control_checks(
    result: dict[str, Any],
    captures: Sequence[RunCapture],
    expectations: Mapping[str, tuple[ControlRecord, ...]],
    *,
    diagnostic_fields: Sequence[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for capture in captures:
        expected = expectations.get(capture.scenario_id)
        if expected is None:
            checks.append(
                {
                    "scenario_id": capture.scenario_id,
                    "status": "missing",
                    "passed": False,
                    "reason": "no exact control fixture for comparison scenario",
                }
            )
            continue
        comparison = compare_control_records(
            expected,
            capture.controls,
            diagnostic_fields=diagnostic_fields,
        )
        checks.append(
            {
                "scenario_id": capture.scenario_id,
                "status": "evaluated",
                **comparison,
            }
        )
    controls_passed = bool(checks and all(check["passed"] for check in checks))
    result["control_semantics"] = {
        "status": "evaluated" if checks else "missing",
        "passed": controls_passed,
        "checks": checks,
    }
    result["passed"] = bool(result["passed"] and controls_passed)
    return result


def compare_repeat_captures(
    baseline: RunCapture,
    repeats: Sequence[RunCapture],
) -> dict[str, Any]:
    """Require bit-stable logits, IDs, and controls for an identical schedule."""

    if not repeats:
        return {"status": "missing", "passed": False, "repeats": []}
    rows: list[dict[str, Any]] = []
    seen_run_ids = {baseline.run_id}
    for repeat in repeats:
        row_match = _rows_compatible(baseline, repeat)
        independent_run = repeat.run_id not in seen_run_ids
        seen_run_ids.add(repeat.run_id)
        controls = compare_control_records(baseline.controls, repeat.controls)
        result = {
            "repeat_index": repeat.repeat_index,
            "profile_equal": baseline.execution_profile == repeat.execution_profile,
            "scenario_equal": baseline.scenario_id == repeat.scenario_id,
            "variant_manifest_equal": (
                baseline.variant_manifest_sha256 == repeat.variant_manifest_sha256
            ),
            "independent_run": independent_run,
            "rows_equal": row_match,
            "logits_exact": bool(row_match and np.array_equal(baseline.logits, repeat.logits)),
            "selected_token_ids_exact": baseline.selected_token_ids
            == repeat.selected_token_ids,
            "controls_exact": bool(controls["passed"]),
            "capture_sha256": repeat.sha256(),
        }
        result["passed"] = all(
            bool(result[field])
            for field in (
                "profile_equal",
                "scenario_equal",
                "variant_manifest_equal",
                "independent_run",
                "rows_equal",
                "logits_exact",
                "selected_token_ids_exact",
                "controls_exact",
            )
        )
        rows.append(result)
    return {"status": "evaluated", "passed": all(row["passed"] for row in rows), "repeats": rows}


def _logical_rows(capture: RunCapture, request_ids: set[str]) -> dict[tuple[str, int], int]:
    return {
        row.logical_key: index
        for index, row in enumerate(capture.rows)
        if row.request_id in request_ids
    }


def compare_request_results(
    baseline: RunCapture,
    comparisons: Sequence[RunCapture],
    *,
    request_ids: Sequence[str],
) -> dict[str, Any]:
    """Compare logical request results across neighbors, slots, or widths."""

    requested = {str(request_id) for request_id in request_ids}
    if not requested:
        raise ValueError("request invariance needs at least one request_id")
    baseline_map = _logical_rows(baseline, requested)
    if not baseline_map:
        raise ValueError("requested invariant rows are absent from baseline capture")
    if not comparisons:
        return {"status": "missing", "passed": False, "comparisons": []}
    results: list[dict[str, Any]] = []
    seen_run_ids = {baseline.run_id}
    for capture in comparisons:
        independent_run = capture.run_id not in seen_run_ids
        seen_run_ids.add(capture.run_id)
        candidate_map = _logical_rows(capture, requested)
        same_keys = set(baseline_map) == set(candidate_map)
        logits_exact = same_keys
        ids_exact = same_keys
        if same_keys:
            for key in sorted(baseline_map):
                baseline_index = baseline_map[key]
                candidate_index = candidate_map[key]
                logits_exact &= bool(
                    np.array_equal(
                        baseline.logits[baseline_index], capture.logits[candidate_index]
                    )
                )
                ids_exact &= (
                    baseline.selected_token_ids[baseline_index]
                    == capture.selected_token_ids[candidate_index]
                )
        result = {
            "scenario_id": capture.scenario_id,
            "run_id": capture.run_id,
            "capture_sha256": capture.sha256(),
            "independent_run": independent_run,
            "profile_equal": capture.execution_profile == baseline.execution_profile,
            "distinct_scenario": capture.scenario_id != baseline.scenario_id,
            "variant_manifest_equal": (
                baseline.variant_manifest_sha256 == capture.variant_manifest_sha256
            ),
            "logical_rows_equal": same_keys,
            "rows_compared": len(baseline_map) if same_keys else 0,
            "logits_exact": bool(logits_exact),
            "selected_token_ids_exact": bool(ids_exact),
        }
        result["passed"] = bool(
            result["independent_run"]
            and result["profile_equal"]
            and result["distinct_scenario"]
            and result["variant_manifest_equal"]
            and same_keys
            and logits_exact
            and ids_exact
        )
        results.append(result)
    return {
        "status": "evaluated",
        "passed": all(result["passed"] for result in results),
        "request_ids": sorted(requested),
        "comparisons": results,
    }


def _all_request_ids(capture: RunCapture) -> tuple[str, ...]:
    return tuple(sorted({row.request_id for row in capture.rows}))


def compare_bf16_noninferiority(
    bf16_logits: np.ndarray,
    strict_logits: np.ndarray,
    candidate_logits: np.ndarray,
    rows: Sequence[RowDescriptor],
    *,
    thresholds: Bf16NoninferiorityThresholds | None = None,
) -> dict[str, Any]:
    """Report candidate-vs-strict incremental drift relative to BF16."""

    threshold = thresholds or Bf16NoninferiorityThresholds()
    row_tuple = tuple(rows)
    labels = np.asarray([row.teacher_token_id for row in row_tuple], dtype=np.int64)
    effective_top_k = min(5, np.asarray(bf16_logits).shape[1])
    strict_vectors = _metric_vectors(
        bf16_logits,
        strict_logits,
        labels=labels,
        top_k=effective_top_k,
    )
    candidate_vectors = _metric_vectors(
        bf16_logits,
        candidate_logits,
        labels=labels,
        top_k=effective_top_k,
    )
    indices = np.arange(len(row_tuple), dtype=np.int64)
    strict_summary = _summary(strict_vectors, indices)
    candidate_summary = _summary(candidate_vectors, indices)
    mean_delta = candidate_summary["kl_mean"] - strict_summary["kl_mean"]
    top1_delta = candidate_summary["top1_agreement"] - strict_summary["top1_agreement"]
    def scoped(dimension: str) -> tuple[dict[str, Any], bool]:
        groups: dict[str, Any] = {}
        all_passed = True
        for value in sorted({str(getattr(row, dimension)) for row in row_tuple}):
            selected = np.asarray(
                [
                    index
                    for index, row in enumerate(row_tuple)
                    if str(getattr(row, dimension)) == value
                ],
                dtype=np.int64,
            )
            strict_group = _summary(strict_vectors, selected)
            candidate_group = _summary(candidate_vectors, selected)
            group_mean_delta = candidate_group["kl_mean"] - strict_group["kl_mean"]
            group_top1_delta = (
                candidate_group["top1_agreement"] - strict_group["top1_agreement"]
            )
            group_passed = bool(
                group_mean_delta <= threshold.mean_kl_delta_max
                and group_top1_delta >= -threshold.top1_drop_max
            )
            all_passed &= group_passed
            groups[value] = {
                "strict": strict_group,
                "candidate": candidate_group,
                "mean_kl_delta": float(group_mean_delta),
                "top1_delta": float(group_top1_delta),
                "passed": group_passed,
            }
        return groups, all_passed

    by_category, category_pass = scoped("category")
    by_prompt, prompt_pass = scoped("request_id")
    prompt_ids = sorted(by_prompt)
    prompt_kl_delta = np.asarray(
        [by_prompt[prompt_id]["mean_kl_delta"] for prompt_id in prompt_ids],
        dtype=np.float64,
    )
    prompt_top1_delta = np.asarray(
        [by_prompt[prompt_id]["top1_delta"] for prompt_id in prompt_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(threshold.bootstrap_seed)
    samples = rng.integers(
        0,
        len(prompt_ids),
        size=(threshold.bootstrap_samples, len(prompt_ids)),
    )
    bootstrap_kl = prompt_kl_delta[samples].mean(axis=1)
    bootstrap_top1 = prompt_top1_delta[samples].mean(axis=1)
    kl_interval = np.percentile(bootstrap_kl, (2.5, 97.5))
    top1_interval = np.percentile(bootstrap_top1, (2.5, 97.5))
    confidence_pass = bool(
        kl_interval[1] <= threshold.mean_kl_delta_max
        and top1_interval[0] >= -threshold.top1_drop_max
    )
    passed = bool(
        mean_delta <= threshold.mean_kl_delta_max
        and top1_delta >= -threshold.top1_drop_max
        and category_pass
        and prompt_pass
        and confidence_pass
    )
    return {
        "status": "evaluated",
        "thresholds": {
            "mean_kl_delta_max": threshold.mean_kl_delta_max,
            "top1_drop_max": threshold.top1_drop_max,
            "bootstrap_samples": threshold.bootstrap_samples,
            "bootstrap_seed": threshold.bootstrap_seed,
        },
        "strict": strict_summary,
        "candidate": candidate_summary,
        "mean_kl_delta": float(mean_delta),
        "top1_delta": float(top1_delta),
        "by_category": by_category,
        "by_prompt": by_prompt,
        "paired_prompt_bootstrap_95pct": {
            "prompt_count": len(prompt_ids),
            "mean_kl_delta": {
                "low": float(kl_interval[0]),
                "high": float(kl_interval[1]),
            },
            "top1_delta": {
                "low": float(top1_interval[0]),
                "high": float(top1_interval[1]),
            },
            "passed": confidence_pass,
        },
        "passed": passed,
    }


def _task_quality(task_results: Mapping[str, Any]) -> dict[str, Any]:
    if not task_results:
        return {"status": "missing", "passed": False, "results": {}}
    normalized: dict[str, dict[str, Any]] = {}
    for name, value in sorted(task_results.items()):
        if isinstance(value, Mapping):
            if "passed" not in value or type(value["passed"]) is not bool:
                raise ValueError(f"task result {name!r} must contain boolean passed")
            normalized[str(name)] = dict(value)
        elif type(value) is bool:
            normalized[str(name)] = {"passed": value}
        else:
            raise ValueError(f"task result {name!r} must be bool or mapping")
    return {
        "status": "evaluated",
        "passed": all(result["passed"] for result in normalized.values()),
        "results": normalized,
    }


def _generated_id_comparison(
    strict_capture: RunCapture,
    candidate_capture: RunCapture,
    profile: ExecutionProfile,
) -> dict[str, Any]:
    if strict_capture.rows != candidate_capture.rows:
        raise ValueError("strict and candidate rows must align for ID comparison")
    matches = sum(
        first == second
        for first, second in zip(
            strict_capture.selected_token_ids,
            candidate_capture.selected_token_ids,
            strict=True,
        )
    )
    total = len(strict_capture.selected_token_ids)
    return {
        "binding": profile in {ExecutionProfile.STRICT, ExecutionProfile.BATCH_INVARIANT},
        "matches": matches,
        "total": total,
        "agreement": 1.0 if total == 0 else matches / total,
        "all_equal": matches == total,
    }


def build_execution_profile_artifact(
    *,
    variant_manifest: Mapping[str, Any],
    strict_manifest: Mapping[str, Any],
    arithmetic_class: str,
    strict_capture: RunCapture,
    candidate_capture: RunCapture,
    expected_controls: Sequence[ControlRecord],
    strict_expected_controls: Sequence[ControlRecord],
    comparison_expected_controls: Mapping[str, Sequence[ControlRecord]] | None = None,
    repeat_captures: Sequence[RunCapture] = (),
    isolation_captures: Sequence[RunCapture] = (),
    batch_invariant_captures: Sequence[RunCapture] = (),
    task_results: Mapping[str, Any],
    thresholds: EvaluationThresholds | None = None,
    bf16_logits: np.ndarray | None = None,
    bf16_thresholds: Bf16NoninferiorityThresholds | None = None,
) -> dict[str, Any]:
    """Build one compact verdict from aligned profile captures."""

    candidate_manifest = validate_variant_manifest(variant_manifest)
    strict_manifest_payload = validate_variant_manifest(strict_manifest)
    if not expected_controls or not strict_expected_controls:
        raise ValueError(
            "candidate and strict expected controls must contain explicit fixtures"
        )
    primary_expected = tuple(expected_controls)
    strict_expected = tuple(strict_expected_controls)
    primary_scenarios = {record.scenario_id for record in primary_expected}
    if len(primary_scenarios) != 1:
        raise ValueError("expected_controls must describe exactly one primary scenario")
    strict_scenarios = {record.scenario_id for record in strict_expected}
    if len(strict_scenarios) != 1:
        raise ValueError(
            "strict_expected_controls must describe exactly one strict scenario"
        )
    expectations: dict[str, tuple[ControlRecord, ...]] = {
        next(iter(primary_scenarios)): primary_expected
    }
    for scenario_id, records in (comparison_expected_controls or {}).items():
        normalized = tuple(records)
        if not normalized or any(record.scenario_id != scenario_id for record in normalized):
            raise ValueError(
                f"comparison control fixture {scenario_id!r} is empty or misaligned"
            )
        if scenario_id in expectations:
            raise ValueError(f"duplicate control fixture scenario: {scenario_id!r}")
        expectations[str(scenario_id)] = normalized
    profile = ExecutionProfile(candidate_manifest["execution_profile"])
    if strict_manifest_payload["execution_profile"] != ExecutionProfile.STRICT.value:
        raise ValueError("strict_manifest must declare strict execution")
    candidate_manifest_hash = manifest_sha256(candidate_manifest)
    strict_manifest_hash = manifest_sha256(strict_manifest_payload)
    if candidate_capture.execution_profile != profile.value:
        raise ValueError("candidate capture profile differs from variant manifest")
    if candidate_capture.variant_manifest_sha256 != candidate_manifest_hash:
        raise ValueError("candidate capture variant manifest hash differs from evaluator")
    if strict_capture.execution_profile != ExecutionProfile.STRICT.value:
        raise ValueError("strict capture must declare strict execution")
    if strict_capture.variant_manifest_sha256 != strict_manifest_hash:
        raise ValueError("strict capture variant manifest hash differs from evaluator")
    if candidate_capture.scenario_id not in primary_scenarios:
        raise ValueError("candidate expected controls differ from candidate scenario")
    if strict_capture.scenario_id not in strict_scenarios:
        raise ValueError("strict expected controls differ from strict scenario")
    if strict_capture.rows != candidate_capture.rows:
        raise ValueError("strict and candidate capture rows must align")
    if arithmetic_class not in {"T0", "T1", "T2", "T3"}:
        raise ValueError("arithmetic_class must be T0, T1, T2, or T3")

    quality = compare_profile_logits(
        strict_capture.logits,
        candidate_capture.logits,
        candidate_capture.rows,
        thresholds=thresholds,
    )
    strict_controls = compare_control_records(strict_expected, strict_capture.controls)
    candidate_controls = compare_control_records(
        expected_controls,
        candidate_capture.controls,
        diagnostic_fields=("route_decision_hash",)
        if profile is not ExecutionProfile.STRICT
        else (),
    )
    control_semantics = {
        "passed": bool(strict_controls["passed"] and candidate_controls["passed"]),
        "strict": strict_controls,
        "candidate": candidate_controls,
    }
    determinism = compare_repeat_captures(candidate_capture, repeat_captures)
    request_ids = _all_request_ids(candidate_capture)
    diagnostic_control_fields = (
        ("route_decision_hash",) if profile is not ExecutionProfile.STRICT else ()
    )
    isolation = _attach_comparison_control_checks(
        compare_request_results(
            candidate_capture, isolation_captures, request_ids=request_ids
        ),
        isolation_captures,
        expectations,
        diagnostic_fields=diagnostic_control_fields,
    )
    if batch_invariant_captures:
        batch_invariance = _attach_comparison_control_checks(
            compare_request_results(
                candidate_capture, batch_invariant_captures, request_ids=request_ids
            ),
            batch_invariant_captures,
            expectations,
            diagnostic_fields=diagnostic_control_fields,
        )
    else:
        batch_invariance = {"status": "missing", "passed": False, "comparisons": []}
    tasks = _task_quality(task_results)
    generated = _generated_id_comparison(strict_capture, candidate_capture, profile)
    if bf16_logits is None:
        bf16 = {"status": "unavailable", "passed": None}
    else:
        bf16 = compare_bf16_noninferiority(
            bf16_logits,
            strict_capture.logits,
            candidate_capture.logits,
            candidate_capture.rows,
            thresholds=bf16_thresholds,
        )

    strict_exact = bool(
        np.array_equal(strict_capture.logits, candidate_capture.logits)
        and generated["all_equal"]
    )
    quality["strict_exact_logits_and_ids"] = strict_exact
    quality_binding = bool(quality["hard_gates_passed"])
    if profile is ExecutionProfile.STRICT:
        quality_binding &= strict_exact
    if arithmetic_class == "T3" and profile is ExecutionProfile.PRODUCTION:
        quality_binding = False
    required = [
        control_semantics["passed"],
        quality_binding,
        determinism["passed"],
        isolation["passed"],
        tasks["passed"],
    ]
    if generated["binding"]:
        required.append(generated["all_equal"])
    if profile is ExecutionProfile.BATCH_INVARIANT:
        required.append(batch_invariance["passed"])
    if bf16["status"] == "evaluated":
        required.append(bool(bf16["passed"]))
    binding_pass = all(required)
    automatic = bool(binding_pass and quality["eligible_for_automatic_admission"])
    if not binding_pass:
        status = "failed"
    elif quality["requires_outlier_review"]:
        status = "requires_review"
    else:
        status = "passed"

    artifact: dict[str, Any] = {
        "kind": EXECUTION_PROFILE_EVALUATION_KIND,
        "schema_version": EXECUTION_PROFILE_EVALUATION_SCHEMA_VERSION,
        "execution_profile": profile.value,
        "execution_profile_schema": EXECUTION_PROFILE_SCHEMA_VERSION,
        "variant_manifest_sha256": candidate_manifest_hash,
        "strict_manifest_sha256": strict_manifest_hash,
        "arithmetic_class": arithmetic_class,
        "teacher_source": "strict",
        "manifests": {
            "candidate": candidate_manifest,
            "strict": strict_manifest_payload,
        },
        "captures": {
            "strict_sha256": strict_capture.sha256(),
            "candidate_sha256": candidate_capture.sha256(),
            "repeat_sha256": [capture.sha256() for capture in repeat_captures],
            "isolation_sha256": [capture.sha256() for capture in isolation_captures],
            "batch_invariant_sha256": [
                capture.sha256() for capture in batch_invariant_captures
            ],
            "strict_control_fixture_sha256": _control_fixture_sha256(strict_expected),
            "candidate_control_fixture_sha256": _control_fixture_sha256(primary_expected),
            "comparison_control_fixture_sha256": {
                scenario_id: _control_fixture_sha256(records)
                for scenario_id, records in sorted(expectations.items())
                if scenario_id not in primary_scenarios
            },
        },
        "scenario": summarize_scenario(expected_controls),
        "quality": quality,
        "control_semantics": control_semantics,
        "determinism": determinism,
        "isolation": isolation,
        "batch_invariance": batch_invariance,
        "bf16_noninferiority": bf16,
        "task_quality": tasks,
        "generated_id_equality": generated,
        "decision": {
            "status": status,
            "eligible_for_automatic_admission": automatic,
            "binding_gates_passed": bool(binding_pass),
        },
    }
    return validate_execution_profile_artifact(artifact)


def validate_execution_profile_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the compact execution-profile evaluation contract."""

    required = {
        "kind",
        "schema_version",
        "execution_profile",
        "execution_profile_schema",
        "variant_manifest_sha256",
        "strict_manifest_sha256",
        "arithmetic_class",
        "teacher_source",
        "manifests",
        "captures",
        "scenario",
        "quality",
        "control_semantics",
        "determinism",
        "isolation",
        "batch_invariance",
        "bf16_noninferiority",
        "task_quality",
        "generated_id_equality",
        "decision",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"execution-profile artifact missing fields: {sorted(missing)}")
    unknown = set(payload) - required
    if unknown:
        raise ValueError(f"execution-profile artifact has unknown fields: {sorted(unknown)}")
    if payload.get("kind") != EXECUTION_PROFILE_EVALUATION_KIND:
        raise ValueError(f"artifact kind must be {EXECUTION_PROFILE_EVALUATION_KIND!r}")
    if payload.get("schema_version") != EXECUTION_PROFILE_EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            "execution-profile artifact schema_version must be "
            f"{EXECUTION_PROFILE_EVALUATION_SCHEMA_VERSION}"
        )
    try:
        profile = ExecutionProfile(str(payload.get("execution_profile")))
    except ValueError as exc:
        raise ValueError("artifact execution_profile is invalid") from exc
    if payload.get("execution_profile_schema") != EXECUTION_PROFILE_SCHEMA_VERSION:
        raise ValueError("artifact execution_profile_schema is invalid")
    manifests = payload.get("manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != {"candidate", "strict"}:
        raise ValueError("artifact manifests must contain candidate and strict")
    candidate = validate_variant_manifest(manifests["candidate"])
    strict = validate_variant_manifest(manifests["strict"])
    if candidate["execution_profile"] != profile.value:
        raise ValueError("candidate manifest profile differs from artifact")
    if strict["execution_profile"] != ExecutionProfile.STRICT.value:
        raise ValueError("artifact strict manifest is not strict")
    if payload.get("variant_manifest_sha256") != manifest_sha256(candidate):
        raise ValueError("artifact variant_manifest_sha256 mismatch")
    if payload.get("strict_manifest_sha256") != manifest_sha256(strict):
        raise ValueError("artifact strict_manifest_sha256 mismatch")
    if payload.get("arithmetic_class") not in {"T0", "T1", "T2", "T3"}:
        raise ValueError("artifact arithmetic_class is invalid")
    if payload.get("teacher_source") != "strict":
        raise ValueError("artifact teacher_source must be strict")
    decision = payload.get("decision")
    if not isinstance(decision, Mapping) or decision.get("status") not in {
        "passed",
        "requires_review",
        "failed",
    }:
        raise ValueError("artifact decision status is invalid")
    if type(decision.get("eligible_for_automatic_admission")) is not bool:
        raise ValueError("artifact decision eligibility must be boolean")
    if type(decision.get("binding_gates_passed")) is not bool:
        raise ValueError("artifact decision binding_gates_passed must be boolean")
    quality = payload.get("quality")
    generated = payload.get("generated_id_equality")
    bf16 = payload.get("bf16_noninferiority")
    if not all(
        isinstance(payload.get(field), Mapping)
        for field in (
            "control_semantics",
            "determinism",
            "isolation",
            "batch_invariance",
            "bf16_noninferiority",
            "task_quality",
            "generated_id_equality",
            "quality",
        )
    ):
        raise ValueError("artifact gate sections must be objects")
    quality_binding = quality.get("hard_gates_passed") is True
    if profile is ExecutionProfile.STRICT:
        quality_binding &= quality.get("strict_exact_logits_and_ids") is True
    if payload.get("arithmetic_class") == "T3" and profile is ExecutionProfile.PRODUCTION:
        quality_binding = False
    required_checks = [
        payload["control_semantics"].get("passed") is True,
        quality_binding,
        payload["determinism"].get("passed") is True,
        payload["isolation"].get("passed") is True,
        payload["task_quality"].get("passed") is True,
    ]
    if generated.get("binding") is True:
        required_checks.append(generated.get("all_equal") is True)
    if profile is ExecutionProfile.BATCH_INVARIANT:
        required_checks.append(payload["batch_invariance"].get("passed") is True)
    if bf16.get("status") == "evaluated":
        required_checks.append(bf16.get("passed") is True)
    expected_binding = all(required_checks)
    expected_status = (
        "failed"
        if not expected_binding
        else "requires_review"
        if quality.get("requires_outlier_review") is True
        else "passed"
    )
    expected_eligible = expected_status == "passed"
    if (
        decision["status"] != expected_status
        or decision["eligible_for_automatic_admission"] != expected_eligible
        or decision["binding_gates_passed"] != expected_binding
    ):
        raise ValueError("artifact decision fields are inconsistent with gate sections")
    normalized = dict(payload)
    # Fail early on accidental NaN/Infinity before JSON artifact publication.
    json.dumps(normalized, allow_nan=False, sort_keys=True)
    return normalized


__all__ = [
    "EXECUTION_PROFILE_CAPTURE_KIND",
    "EXECUTION_PROFILE_CAPTURE_SCHEMA_VERSION",
    "EXECUTION_PROFILE_CONTROL_CAPTURE_KIND",
    "EXECUTION_PROFILE_CONTROL_CAPTURE_SCHEMA_VERSION",
    "EXECUTION_PROFILE_CONTROL_FIXTURE_KIND",
    "EXECUTION_PROFILE_CONTROL_FIXTURE_SCHEMA_VERSION",
    "EXECUTION_PROFILE_EVALUATION_KIND",
    "EXECUTION_PROFILE_EVALUATION_SCHEMA_VERSION",
    "Bf16NoninferiorityThresholds",
    "ControlRecord",
    "EvaluationThresholds",
    "RowDescriptor",
    "RunCapture",
    "build_execution_profile_artifact",
    "compare_bf16_noninferiority",
    "compare_control_records",
    "compare_profile_logits",
    "compare_repeat_captures",
    "compare_request_results",
    "load_control_capture",
    "load_control_fixture",
    "load_run_capture_manifest",
    "parse_run_capture_manifest",
    "qwen36_rows_from_teacher_fixture",
    "summarize_scenario",
    "validate_execution_profile_artifact",
]
