"""Batch sampler / LM-head dispatch safety policy.

The Qwen/PARO native c>N path currently keeps the row sampler conservative:
per-row serial LM-head sampling is correctness-safe, while a row-aware batched
LM-head/argmax launch must not become a retained path until c>N generated-token
equality is green.  This module centralizes that decision so runtime code can
record explicit blockers instead of relying on ad-hoc env checks.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class BatchSamplerMode(str, Enum):
    """Supported c>N sampler execution modes."""

    SERIAL_LM_HEAD = "serial_lm_head"
    BATCHED_LM_HEAD = "batched_lm_head"


@dataclass(frozen=True, slots=True)
class BatchSamplerDispatchDecision:
    """Resolved sampler dispatch mode for one batch decode step."""

    rows: int
    requested_mode: BatchSamplerMode
    mode: BatchSamplerMode
    native_row_aware_lm_head: bool
    c2_equality_green: bool
    equality_artifact: str | None
    equality_rows: int | None
    blockers: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "requested_mode": self.requested_mode.value,
            "mode": self.mode.value,
            "native_row_aware_lm_head": self.native_row_aware_lm_head,
            "c2_equality_green": self.c2_equality_green,
            "equality_artifact": self.equality_artifact,
            "equality_rows": self.equality_rows,
            "blockers": list(self.blockers),
        }


def _sampler_mode(value: BatchSamplerMode | str) -> BatchSamplerMode:
    try:
        return value if isinstance(value, BatchSamplerMode) else BatchSamplerMode(str(value))
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in BatchSamplerMode)
        raise ValueError(f"unknown batch sampler mode {value!r}; expected one of: {valid}") from exc


def _is_retained_artifact_path(value: str) -> bool:
    path = Path(value)
    if (
        path.is_absolute()
        or len(path.parts) < 3
        or path.parts[:2] != ("benchmarks", "results")
        or ".." in path.parts
    ):
        return False
    results_root = (Path.cwd() / "benchmarks" / "results").resolve()
    try:
        return (Path.cwd() / path).resolve().is_relative_to(results_root)
    except OSError:
        return False


def _optional_positive_int(value: int | str | None) -> tuple[int | None, bool]:
    if value is None:
        return None, True
    if isinstance(value, bool):
        return None, False
    if isinstance(value, int):
        return (value, True) if value > 0 else (None, False)
    text = str(value).strip()
    if not text:
        return None, True
    try:
        parsed = int(text, 10)
    except ValueError:
        return None, False
    if parsed <= 0:
        return None, False
    return parsed, True


def _artifact_row_count(payload: Mapping[str, Any]) -> Any:
    rows = payload.get("rows")
    if rows is not None:
        return rows
    workload = payload.get("workload")
    if isinstance(workload, Mapping):
        return workload.get("concurrency")
    return None


def _generated_token_equality(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    correctness = payload.get("correctness")
    if isinstance(correctness, Mapping):
        equality = correctness.get("generated_token_equality")
        if isinstance(equality, Mapping):
            return equality
    equality = payload.get("generated_token_equality")
    return equality if isinstance(equality, Mapping) else None


def _sampler_execution(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    execution = payload.get("execution")
    batch_execution = execution.get("batch_execution") if isinstance(execution, Mapping) else None
    decode_execution = batch_execution.get("decode_execution") if isinstance(batch_execution, Mapping) else None
    sampler_execution = decode_execution.get("sampler_execution") if isinstance(decode_execution, Mapping) else None
    return sampler_execution if isinstance(sampler_execution, Mapping) else None


def _token_sequence_rows_are_nonempty_int_lists(value: list[Any]) -> bool:
    return all(
        isinstance(row, list)
        and bool(row)
        and all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in row)
        for row in value
    )


def batch_sampler_equality_payload_blockers(
    payload: Mapping[str, Any],
    *,
    rows: int,
    label: str = "batched LM-head equality artifact",
    expected_artifact_path: str | None = None,
) -> tuple[str, ...]:
    """Return blockers for native batched-LM-head equality evidence.

    The sampler gate must be backed by generated-token equality vs independent
    c=1, not merely by a primitive-kernel or ad-hoc ``passed=true`` JSON file.
    Accept both full retained artifacts (``correctness.generated_token_equality``)
    and standalone equality artifacts (top-level ``generated_token_equality``).
    """

    blockers: list[str] = []
    equality = _generated_token_equality(payload)
    correctness = payload.get("correctness")
    passed = payload.get("passed") is True
    if isinstance(correctness, Mapping):
        passed = passed or correctness.get("passed") is True
    if isinstance(equality, Mapping):
        passed = passed or equality.get("passed") is True
    if not passed:
        blockers.append(f"{label} must report passed=true")
    if expected_artifact_path is not None:
        payload_artifact_path = payload.get("artifact_path")
        if not isinstance(payload_artifact_path, str) or not payload_artifact_path.strip():
            blockers.append(f"{label} artifact_path must be a non-empty string")
        elif payload_artifact_path != expected_artifact_path:
            blockers.append(f"{label} artifact_path must match sampler_execution.equality_artifact")
        payload_source_artifact_path = payload.get("source_artifact_path")
        if not isinstance(payload_source_artifact_path, str) or not payload_source_artifact_path.strip():
            blockers.append(f"{label} source_artifact_path must be a non-empty string")
        elif payload_source_artifact_path != expected_artifact_path:
            blockers.append(f"{label} source_artifact_path must match sampler_execution.equality_artifact")
    artifact_rows = _artifact_row_count(payload)
    if isinstance(artifact_rows, bool) or not isinstance(artifact_rows, int):
        blockers.append(f"{label} rows must be an integer")
    elif artifact_rows != rows:
        blockers.append(f"{label} rows must match batch rows")
    if not isinstance(equality, Mapping):
        blockers.append(f"{label} must include generated-token equality details")
        return tuple(blockers)
    if equality.get("passed") is not True:
        blockers.append(f"{label} generated_token_equality.passed must be true")
    if equality.get("skipped") is not False:
        blockers.append(f"{label} generated_token_equality.skipped must be false")
    sampler_execution = _sampler_execution(payload)
    if sampler_execution is not None:
        if sampler_execution.get("requested_mode") != BatchSamplerMode.BATCHED_LM_HEAD.value:
            blockers.append(f"{label} sampler_execution.requested_mode must be batched_lm_head")
        if sampler_execution.get("mode") != BatchSamplerMode.BATCHED_LM_HEAD.value:
            blockers.append(f"{label} sampler_execution.mode must be batched_lm_head")
        if sampler_execution.get("native_row_aware_lm_head") is not True:
            blockers.append(f"{label} sampler_execution.native_row_aware_lm_head must be true")
        if sampler_execution.get("blockers") != []:
            blockers.append(f"{label} sampler_execution.blockers must be empty")
    batch_sequences = equality.get("batch_sequences")
    c1_sequences = equality.get("c1_sequences")
    if not isinstance(batch_sequences, list) or not isinstance(c1_sequences, list):
        blockers.append(f"{label} generated_token_equality batch_sequences and c1_sequences must be lists")
    else:
        if len(batch_sequences) != rows or len(c1_sequences) != rows:
            blockers.append(f"{label} generated_token_equality sequence row counts must match batch rows")
        if not _token_sequence_rows_are_nonempty_int_lists(batch_sequences):
            blockers.append(f"{label} generated_token_equality batch_sequences rows must be non-empty integer lists")
        if not _token_sequence_rows_are_nonempty_int_lists(c1_sequences):
            blockers.append(f"{label} generated_token_equality c1_sequences rows must be non-empty integer lists")
        if batch_sequences != c1_sequences:
            blockers.append(f"{label} generated_token_equality batch_sequences must equal c1_sequences")
    mismatches = equality.get("mismatches")
    if not isinstance(mismatches, list):
        blockers.append(f"{label} generated_token_equality.mismatches must be a list")
    elif mismatches:
        blockers.append(f"{label} generated_token_equality.mismatches must be empty")
    return tuple(blockers)


def _path_has_retained_results_symlink_parent(path: Path) -> bool:
    results_root = (Path.cwd() / "benchmarks" / "results").resolve()
    current = path.parent
    while True:
        try:
            current_resolved = current.resolve()
        except OSError:
            return False
        if not current_resolved.is_relative_to(results_root):
            return False
        if current.is_symlink():
            return True
        if current == current.parent:
            return False
        current = current.parent


def _equality_artifact_blockers(value: str, *, rows: int) -> tuple[str, ...]:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.suffix.lower() != ".json":
        return ("batched LM-head equality artifact must point to a .json artifact",)
    if path.is_symlink():
        return ("batched LM-head equality artifact must point to a regular JSON artifact, not a symlink",)
    if _path_has_retained_results_symlink_parent(path):
        return ("batched LM-head equality artifact parent directories must not be symlinks",)
    if not path.exists():
        return ("batched LM-head equality artifact must point to an existing JSON artifact",)
    if not path.is_file():
        return ("batched LM-head equality artifact must point to a regular JSON artifact",)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return (f"batched LM-head equality artifact must be readable JSON: {exc}",)
    except json.JSONDecodeError as exc:
        return (f"batched LM-head equality artifact must be valid JSON: {exc}",)
    if not isinstance(payload, Mapping):
        return ("batched LM-head equality artifact must be a JSON object",)
    return batch_sampler_equality_payload_blockers(payload, rows=rows, expected_artifact_path=value)


def plan_batch_sampler_dispatch(
    *,
    rows: int,
    requested_mode: BatchSamplerMode | str = BatchSamplerMode.SERIAL_LM_HEAD,
    c2_equality_green: bool = False,
    equality_artifact: str | None = None,
    equality_rows: int | str | None = None,
) -> BatchSamplerDispatchDecision:
    """Plan row sampling for a native batch decode result.

    ``serial_lm_head`` always selects the current per-row c=1 LM-head loop.  A
    requested ``batched_lm_head`` is honored for c>N only when generated-token
    equality evidence is explicitly marked green and an artifact path plus row
    count matching the current batch are supplied.  Otherwise the decision falls
    back to ``serial_lm_head`` with blockers, preserving correctness and
    preventing premature throughput claims.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    requested = _sampler_mode(requested_mode)
    parsed_equality_rows, equality_rows_valid = _optional_positive_int(equality_rows)
    recorded_equality_rows = parsed_equality_rows if equality_rows_valid else None
    if requested is BatchSamplerMode.SERIAL_LM_HEAD:
        return BatchSamplerDispatchDecision(
            rows=rows,
            requested_mode=requested,
            mode=BatchSamplerMode.SERIAL_LM_HEAD,
            native_row_aware_lm_head=False,
            c2_equality_green=bool(c2_equality_green),
            equality_artifact=equality_artifact,
            equality_rows=recorded_equality_rows,
            blockers=(),
        )
    if rows == 1:
        return BatchSamplerDispatchDecision(
            rows=rows,
            requested_mode=requested,
            mode=BatchSamplerMode.BATCHED_LM_HEAD,
            native_row_aware_lm_head=True,
            c2_equality_green=bool(c2_equality_green),
            equality_artifact=equality_artifact,
            equality_rows=recorded_equality_rows,
            blockers=(),
        )
    blockers: list[str] = []
    artifact = str(equality_artifact).strip() if equality_artifact else None
    if not c2_equality_green:
        blockers.append("batched LM-head requires green c>N generated-token equality evidence")
    if not artifact:
        blockers.append("batched LM-head requires an equality artifact path")
    elif not _is_retained_artifact_path(artifact):
        blockers.append("batched LM-head equality artifact path must be under benchmarks/results")
    else:
        blockers.extend(_equality_artifact_blockers(artifact, rows=rows))
    if not equality_rows_valid:
        blockers.append("batched LM-head equality rows must be a positive integer")
    elif parsed_equality_rows is None:
        blockers.append("batched LM-head requires equality rows matching batch rows")
    elif parsed_equality_rows != rows:
        blockers.append("batched LM-head equality rows must match batch rows")
    if blockers:
        return BatchSamplerDispatchDecision(
            rows=rows,
            requested_mode=requested,
            mode=BatchSamplerMode.SERIAL_LM_HEAD,
            native_row_aware_lm_head=False,
            c2_equality_green=bool(c2_equality_green),
            equality_artifact=artifact,
            equality_rows=recorded_equality_rows,
            blockers=tuple(blockers),
        )
    return BatchSamplerDispatchDecision(
        rows=rows,
        requested_mode=requested,
        mode=BatchSamplerMode.BATCHED_LM_HEAD,
        native_row_aware_lm_head=True,
        c2_equality_green=True,
        equality_artifact=artifact,
        equality_rows=parsed_equality_rows,
        blockers=(),
    )


__all__ = [
    "BatchSamplerDispatchDecision",
    "BatchSamplerMode",
    "batch_sampler_equality_payload_blockers",
    "plan_batch_sampler_dispatch",
]
