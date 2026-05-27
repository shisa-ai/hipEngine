"""c-aware projection dispatch planning helpers.

Projection kernels are registered elsewhere under the standard
``(backend, layer, quant, variant)`` kernel key.  This module only decides when
a c>N call site is allowed to move away from the correctness-safe row-GEMV /
Marlin-K path.  It deliberately takes candidate keys and benchmark evidence as
inputs so runtime code does not grow backend- or quant-specific branches.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from hipengine.dispatch.fusion import KernelPlanStep
from hipengine.kernels.registry import KernelKey


@dataclass(frozen=True, slots=True)
class ProjectionKernelSelection:
    """Layer/quant/variant tuple for a projection kernel candidate."""

    layer: str
    quant: str
    variant: str

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ProjectionKernelSelection":
        errors: list[str] = []
        layer = payload.get("layer")
        if not isinstance(layer, str) or not layer:
            errors.append("selection.layer must be a non-empty string")
        quant = payload.get("quant")
        if not isinstance(quant, str) or not quant:
            errors.append("selection.quant must be a non-empty string")
        variant = payload.get("variant")
        if not isinstance(variant, str) or not variant:
            errors.append("selection.variant must be a non-empty string")
        if errors:
            raise ValueError("invalid projection kernel selection: " + "; ".join(errors))
        return cls(layer=str(layer), quant=str(quant), variant=str(variant))

    def to_json_dict(self) -> dict[str, str]:
        return {"layer": self.layer, "quant": self.quant, "variant": self.variant}

    def key(self, backend: str) -> KernelKey:
        return KernelKey(backend, self.layer, self.quant, self.variant)

    def step(self, backend: str) -> KernelPlanStep:
        return KernelPlanStep(backend=backend, layer=self.layer, quant=self.quant, variant=self.variant)


@dataclass(frozen=True, slots=True)
class ProjectionDispatchEvidence:
    """Benchmark evidence required before a c>N projection fast path is used."""

    artifact_path: str
    aggregate_vs_row_gemv: float
    per_request_vs_row_gemv: float
    accepted: bool = True

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ProjectionDispatchEvidence":
        """Load schema-checked projection speedup evidence from an artifact block."""

        errors: list[str] = []
        artifact_path = payload.get("artifact_path")
        if not isinstance(artifact_path, str) or not artifact_path:
            errors.append("artifact_path must be a non-empty string")
        aggregate_vs_row_gemv = payload.get("aggregate_vs_row_gemv")
        if not _is_positive_number(aggregate_vs_row_gemv):
            errors.append("aggregate_vs_row_gemv must be positive numeric")
        per_request_vs_row_gemv = payload.get("per_request_vs_row_gemv")
        if not _is_positive_number(per_request_vs_row_gemv):
            errors.append("per_request_vs_row_gemv must be positive numeric")
        accepted = payload.get("accepted", True)
        if not isinstance(accepted, bool):
            errors.append("accepted must be a bool")
        if errors:
            raise ValueError("invalid projection dispatch evidence: " + "; ".join(errors))
        return cls(
            artifact_path=str(artifact_path),
            aggregate_vs_row_gemv=float(aggregate_vs_row_gemv),
            per_request_vs_row_gemv=float(per_request_vs_row_gemv),
            accepted=bool(accepted),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "aggregate_vs_row_gemv": float(self.aggregate_vs_row_gemv),
            "per_request_vs_row_gemv": float(self.per_request_vs_row_gemv),
            "accepted": bool(self.accepted),
        }


@dataclass(frozen=True, slots=True)
class ProjectionDispatchCandidate:
    """Potential c>N projection kernel plus the evidence that makes it usable."""

    name: str
    selection: ProjectionKernelSelection
    min_rows: int = 2
    max_rows: int | None = None
    evidence: ProjectionDispatchEvidence | None = None

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ProjectionDispatchCandidate":
        """Load a c-aware projection candidate from retained artifact metadata."""

        errors: list[str] = []
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            errors.append("name must be a non-empty string")
        selection_payload = payload.get("selection")
        selection: ProjectionKernelSelection | None = None
        if not isinstance(selection_payload, Mapping):
            errors.append("selection must be an object")
        else:
            try:
                selection = ProjectionKernelSelection.from_json_dict(selection_payload)
            except ValueError as exc:
                errors.append(str(exc))
        min_rows = payload.get("min_rows", 2)
        if not isinstance(min_rows, int) or isinstance(min_rows, bool) or min_rows <= 0:
            errors.append("min_rows must be a positive int")
        max_rows = payload.get("max_rows")
        if max_rows is not None and (not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0):
            errors.append("max_rows must be a positive int or null")
        if isinstance(min_rows, int) and isinstance(max_rows, int) and not isinstance(min_rows, bool) and not isinstance(max_rows, bool) and max_rows < min_rows:
            errors.append("max_rows must be >= min_rows")
        evidence_payload = payload.get("evidence")
        evidence: ProjectionDispatchEvidence | None = None
        if evidence_payload is not None:
            if not isinstance(evidence_payload, Mapping):
                errors.append("evidence must be an object or null")
            else:
                try:
                    evidence = ProjectionDispatchEvidence.from_json_dict(evidence_payload)
                except ValueError as exc:
                    errors.append(str(exc))
        if errors:
            raise ValueError("invalid projection dispatch candidate: " + "; ".join(errors))
        assert isinstance(name, str)
        assert isinstance(min_rows, int) and not isinstance(min_rows, bool)
        assert selection is not None
        return cls(
            name=name,
            selection=selection,
            min_rows=min_rows,
            max_rows=None if max_rows is None else int(max_rows),
            evidence=evidence,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "selection": self.selection.to_json_dict(),
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "evidence": None if self.evidence is None else self.evidence.to_json_dict(),
        }

    def applies_to(self, rows: int) -> bool:
        return rows >= self.min_rows and (self.max_rows is None or rows <= self.max_rows)


@dataclass(frozen=True, slots=True)
class ProjectionDispatchDecision:
    """Resolved projection dispatch choice for a row count."""

    rows: int
    selection: ProjectionKernelSelection
    selected_candidate: str
    path: str
    throughput_claim_eligible: bool
    blockers: tuple[str, ...]
    evidence: ProjectionDispatchEvidence | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "selected_candidate": self.selected_candidate,
            "path": self.path,
            "selection": self.selection.to_json_dict(),
            "throughput_claim_eligible": self.throughput_claim_eligible,
            "blockers": list(self.blockers),
            "evidence": None if self.evidence is None else self.evidence.to_json_dict(),
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive_number(value: Any) -> bool:
    return _is_number(value) and float(value) > 0.0


def projection_dispatch_candidates_from_json(payload: Any) -> tuple[ProjectionDispatchCandidate, ...]:
    """Load an ordered candidate list from retained projection metadata."""

    if not isinstance(payload, list):
        raise ValueError("projection dispatch candidates must be a list")
    candidates: list[ProjectionDispatchCandidate] = []
    errors: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            errors.append(f"candidates[{index}] must be an object")
            continue
        try:
            candidates.append(ProjectionDispatchCandidate.from_json_dict(item))
        except ValueError as exc:
            errors.append(f"candidates[{index}]: {exc}")
    if errors:
        raise ValueError("invalid projection dispatch candidates: " + "; ".join(errors))
    return tuple(candidates)


def projection_dispatch_candidates_from_artifact(
    payload: Mapping[str, Any],
    *,
    field: str = "projection_dispatch_candidates",
) -> tuple[ProjectionDispatchCandidate, ...]:
    """Extract c-aware projection candidates from a retained benchmark artifact."""

    if not isinstance(field, str) or not field:
        raise ValueError("projection dispatch candidate field must be a non-empty string")
    candidates_payload = payload.get(field)
    if candidates_payload is None:
        return ()
    try:
        return projection_dispatch_candidates_from_json(candidates_payload)
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {exc}") from exc


def plan_projection_dispatch(
    *,
    rows: int,
    row_gemv: ProjectionKernelSelection,
    candidates: Iterable[ProjectionDispatchCandidate] = (),
    min_aggregate_speedup: float = 1.0,
    min_per_request_speedup: float = 1.0,
) -> ProjectionDispatchDecision:
    """Select the projection path for ``rows`` active requests.

    ``rows == 1`` is intentionally pinned to ``row_gemv`` even when a candidate
    advertises broader applicability.  For c>N, a candidate is eligible only if
    it has accepted benchmark evidence and beats row-GEMV on both aggregate and
    per-request decode throughput ratios.  Missing or insufficient evidence
    returns the row-GEMV fallback with explicit blockers instead of silently
    making a retained throughput claim.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    if min_aggregate_speedup <= 0.0 or min_per_request_speedup <= 0.0:
        raise ValueError("speedup thresholds must be positive")
    if rows == 1:
        return ProjectionDispatchDecision(
            rows=rows,
            selection=row_gemv,
            selected_candidate="row_gemv",
            path="row_gemv_c1",
            throughput_claim_eligible=False,
            blockers=(),
        )

    applicable: list[ProjectionDispatchCandidate] = []
    eligible: list[tuple[ProjectionDispatchCandidate, ProjectionDispatchEvidence]] = []
    blockers: list[str] = []
    for candidate in candidates:
        if not candidate.applies_to(rows):
            continue
        applicable.append(candidate)
        evidence = candidate.evidence
        if evidence is None:
            blockers.append(f"{candidate.name}: missing benchmark evidence")
            continue
        if not evidence.accepted:
            blockers.append(f"{candidate.name}: benchmark artifact was not accepted")
            continue
        if evidence.aggregate_vs_row_gemv <= min_aggregate_speedup:
            blockers.append(
                f"{candidate.name}: aggregate_vs_row_gemv={evidence.aggregate_vs_row_gemv:.6g} "
                f"does not beat threshold {min_aggregate_speedup:.6g}"
            )
            continue
        if evidence.per_request_vs_row_gemv <= min_per_request_speedup:
            blockers.append(
                f"{candidate.name}: per_request_vs_row_gemv={evidence.per_request_vs_row_gemv:.6g} "
                f"does not beat threshold {min_per_request_speedup:.6g}"
            )
            continue
        eligible.append((candidate, evidence))

    if eligible:
        best, best_evidence = max(
            eligible,
            key=lambda item: (item[1].aggregate_vs_row_gemv, item[1].per_request_vs_row_gemv),
        )
        return ProjectionDispatchDecision(
            rows=rows,
            selection=best.selection,
            selected_candidate=best.name,
            path="benchmark_accepted_caware_projection",
            throughput_claim_eligible=True,
            blockers=(),
            evidence=best_evidence,
        )

    if not applicable:
        blockers.append("no c-aware projection candidate applies to this row count")
    return ProjectionDispatchDecision(
        rows=rows,
        selection=row_gemv,
        selected_candidate="row_gemv",
        path="row_gemv_until_caware_benchmark",
        throughput_claim_eligible=False,
        blockers=tuple(blockers),
    )


def plan_projection_dispatch_from_artifact(
    *,
    payload: Mapping[str, Any],
    rows: int,
    row_gemv: ProjectionKernelSelection,
    field: str = "projection_dispatch_candidates",
    min_aggregate_speedup: float = 1.0,
    min_per_request_speedup: float = 1.0,
) -> ProjectionDispatchDecision:
    """Plan c-aware projection dispatch directly from retained artifact metadata."""

    candidates = projection_dispatch_candidates_from_artifact(payload, field=field)
    return plan_projection_dispatch(
        rows=rows,
        row_gemv=row_gemv,
        candidates=candidates,
        min_aggregate_speedup=min_aggregate_speedup,
        min_per_request_speedup=min_per_request_speedup,
    )


__all__ = [
    "ProjectionDispatchCandidate",
    "ProjectionDispatchDecision",
    "ProjectionDispatchEvidence",
    "ProjectionKernelSelection",
    "plan_projection_dispatch",
    "plan_projection_dispatch_from_artifact",
    "projection_dispatch_candidates_from_artifact",
    "projection_dispatch_candidates_from_json",
]
