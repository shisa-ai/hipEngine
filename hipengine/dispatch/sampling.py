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
from pathlib import Path, PurePosixPath
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
    path = PurePosixPath(value)
    return not path.is_absolute() and len(path.parts) >= 3 and path.parts[:2] == ("benchmarks", "results") and ".." not in path.parts


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


def _equality_artifact_blockers(value: str, *, rows: int) -> tuple[str, ...]:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ("batched LM-head equality artifact must point to an existing JSON artifact",)
    except json.JSONDecodeError as exc:
        return (f"batched LM-head equality artifact must be valid JSON: {exc}",)
    if not isinstance(payload, Mapping):
        return ("batched LM-head equality artifact must be a JSON object",)
    blockers: list[str] = []
    if payload.get("passed") is not True:
        blockers.append("batched LM-head equality artifact must report passed=true")
    artifact_rows = _artifact_row_count(payload)
    if isinstance(artifact_rows, bool) or not isinstance(artifact_rows, int):
        blockers.append("batched LM-head equality artifact rows must be an integer")
    elif artifact_rows != rows:
        blockers.append("batched LM-head equality artifact rows must match batch rows")
    return tuple(blockers)


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
    "plan_batch_sampler_dispatch",
]
