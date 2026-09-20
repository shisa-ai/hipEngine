"""Artifact-scoped speculative-MTP serving admission.

This module owns only immutable pre-mutation decisions.  Model plugins supply
retained evidence; server/model code supplies mechanical request identity.  The
key deliberately has no prompt text, token IDs, benchmark category, heldout, or
oracle fields.

Admission is a physical and ownership question: artifact execution identity,
backend, arch, quant, KV storage, realized group width, resident capacity,
candidate depth, sampling mode, and memory fit.  The artifact axis is layout
coverage, not byte identity: a revision that routes through the same layouts
inherits the retained qualification, while an artifact introducing an
unmeasured layout still fails closed.  Request shape and runtime profile are not
admission axes.  Session length, prompt context, output horizon, and the
resolved variant-manifest hash describe the envelope a benchmark measured; they
change with normal serving traffic and with any kernel or variant selection, so
gating on them silently disables an already-qualified path.  The provider keeps
its own profile authorization (for example FP16 recurrent-state spec-dec2 needs
a complete production manifest).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import json
from typing import Mapping, Sequence


_DEFAULT_STRICT_FALLBACK = "gguf_target_ar"

# Failed axes that are correctness or resource boundaries rather than merely
# unmeasured physical cells: sampling semantics change what the verifier is
# allowed to do, and a memory-fit failure means the cell does not fit at all.
# Everything else records only that no retained row measured the cell, which
# never withholds a path the kernels implement.
STRUCTURAL_REJECTION_AXES = frozenset(
    {
        "sampling_mode_unmeasured",
        "insufficient_memory",
    }
)


def _required_text(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _sha256(value: str | None, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    text = _required_text(value, name).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return text


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SpeculativeMTPServingKey:
    """Complete content and physical-runtime identity for one plan.

    ``artifact_sha256`` and ``artifact_size_bytes`` are provenance: they report
    which file is resident.  Admission reads ``artifact_execution_fingerprint``,
    which is the only artifact axis that participates in matching.
    """

    artifact_sha256: str | None
    artifact_size_bytes: int | None
    content_verified: bool
    backend: str
    target_arch: str
    weight_quant: str
    kv_storage: str
    kv_layout: str
    realized_group_rows: int
    resident_capacity: int
    candidate_budget: int
    sampling_mode: str
    memory_fit: bool
    kv_scale_dtype: str | None = None
    kv_scale_granularity: str | None = None
    artifact_execution_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(self.artifact_sha256, "artifact_sha256", optional=True),
        )
        object.__setattr__(
            self,
            "artifact_execution_fingerprint",
            _sha256(
                self.artifact_execution_fingerprint,
                "artifact_execution_fingerprint",
                optional=True,
            ),
        )
        for name in (
            "backend",
            "target_arch",
            "weight_quant",
            "kv_storage",
            "kv_layout",
            "sampling_mode",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        for name in (
            "realized_group_rows",
            "resident_capacity",
            "candidate_budget",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.artifact_size_bytes is not None:
            size = int(self.artifact_size_bytes)
            if size <= 0:
                raise ValueError("artifact_size_bytes must be positive when present")
            object.__setattr__(self, "artifact_size_bytes", size)
        object.__setattr__(self, "content_verified", bool(self.content_verified))
        object.__setattr__(self, "memory_fit", bool(self.memory_fit))

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_execution_fingerprint": self.artifact_execution_fingerprint,
            "content_verified": self.content_verified,
            "backend": self.backend,
            "target_arch": self.target_arch,
            "weight_quant": self.weight_quant,
            "kv_storage": self.kv_storage,
            "kv_layout": self.kv_layout,
            "realized_group_rows": self.realized_group_rows,
            "resident_capacity": self.resident_capacity,
            "candidate_budget": self.candidate_budget,
            "sampling_mode": self.sampling_mode,
            "memory_fit": self.memory_fit,
            **({"kv_scale_dtype": self.kv_scale_dtype} if self.kv_scale_dtype else {}),
            **({"kv_scale_granularity": self.kv_scale_granularity} if self.kv_scale_granularity else {}),
        }


@dataclass(frozen=True, slots=True)
class SpeculativeMTPServingEvidence:
    """One model-plugin-owned retained serving scope.

    ``artifact_sha256`` and ``artifact_size_bytes`` record the artifact the
    evidence was measured on.  ``artifact_execution_fingerprint`` is the binding
    that admits: a row that declares no identity never admits, and the recorded
    bytes are provenance rather than a gate.
    """

    evidence_key: str
    artifact_sha256: str
    artifact_size_bytes: int
    backend: str
    target_arch: str
    weight_quant: str
    kv_storage: str
    kv_layout: str
    realized_group_rows: int
    resident_capacity: int
    # Inclusive maximum qualified speculative depth. A request at or below
    # this depth admits; a deeper one does not. Depth is a tuning axis, not
    # a correctness axis: the verifier keeps output exact at any depth, and
    # a shallower chain is strictly less speculative work on the same path.
    candidate_budget: int
    sampling_modes: tuple[str, ...]
    reason: str
    evidence_artifacts: tuple[str, ...]
    max_realized_group_rows: int | None = None
    strict_fallback_key: str = _DEFAULT_STRICT_FALLBACK
    automatic_eligible: bool = False
    # Target ownership is part of qualification, not inferred from N or backend.
    packed_c1_target: bool = False
    # Layout-coverage binding: this row admits any artifact whose execution
    # identity matches, which is what kernel routing and verification behaviour
    # depend on.  ``None`` means the identity is unresolved, and an unresolved
    # row never admits; record one with
    # ``python3 scripts/gguf_execution_identity.py <artifact.gguf>``.
    artifact_execution_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "evidence_key",
            "backend",
            "target_arch",
            "weight_quant",
            "kv_storage",
            "kv_layout",
            "reason",
            "strict_fallback_key",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(self.artifact_sha256, "artifact_sha256"),
        )
        object.__setattr__(
            self,
            "artifact_execution_fingerprint",
            _sha256(
                self.artifact_execution_fingerprint,
                "artifact_execution_fingerprint",
                optional=True,
            ),
        )
        for name in (
            "artifact_size_bytes",
            "realized_group_rows",
            "resident_capacity",
            "candidate_budget",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        max_group_rows = (
            self.realized_group_rows
            if self.max_realized_group_rows is None
            else int(self.max_realized_group_rows)
        )
        if not self.realized_group_rows <= max_group_rows <= self.resident_capacity:
            raise ValueError(
                "max_realized_group_rows must cover the evidence row and fit capacity"
            )
        object.__setattr__(self, "max_realized_group_rows", max_group_rows)
        if self.packed_c1_target and max_group_rows != 1:
            raise ValueError("packed_c1_target requires a one-row evidence scope")
        modes = tuple(_required_text(value, "sampling_mode") for value in self.sampling_modes)
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("sampling_modes must be non-empty and unique")
        object.__setattr__(self, "sampling_modes", modes)
        artifacts = tuple(
            _required_text(value, "evidence_artifact")
            for value in self.evidence_artifacts
        )
        if not artifacts:
            raise ValueError("evidence_artifacts must be non-empty")
        object.__setattr__(self, "evidence_artifacts", artifacts)
        object.__setattr__(self, "automatic_eligible", bool(self.automatic_eligible))

    def as_dict(self) -> dict[str, object]:
        return {
            "evidence_key": self.evidence_key,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_execution_fingerprint": self.artifact_execution_fingerprint,
            "backend": self.backend,
            "target_arch": self.target_arch,
            "weight_quant": self.weight_quant,
            "kv_storage": self.kv_storage,
            "kv_layout": self.kv_layout,
            "realized_group_rows": self.realized_group_rows,
            "max_realized_group_rows": self.max_realized_group_rows,
            "resident_capacity": self.resident_capacity,
            "candidate_budget": self.candidate_budget,
            "sampling_modes": list(self.sampling_modes),
            "reason": self.reason,
            "evidence_artifacts": list(self.evidence_artifacts),
            "strict_fallback_key": self.strict_fallback_key,
            "automatic_eligible": self.automatic_eligible,
            "packed_c1_target": self.packed_c1_target,
        }


class SpeculativeMTPStaticState(StrEnum):
    """Request-local provider intent decided before resident scheduling."""

    PERMANENT_AR = "permanent_ar"
    SPECULATIVE_CAPABLE = "speculative_capable"


@dataclass(frozen=True, slots=True)
class SpeculativeMTPStaticEligibility:
    """Typed static eligibility with no selected future C_due or K.

    ``max_realized_group_rows`` is an evidence bound, not a prediction of the
    resident due group. The Generation-2 cycle planner remains the only owner of
    actual C_due and candidate counts.
    """

    state: SpeculativeMTPStaticState
    reason: str
    max_candidate_count: int
    max_realized_group_rows: int
    automatic_eligible: bool
    strict_fallback_key: str
    evidence_key: str | None = None
    evidence_fingerprint: str | None = None
    evidence_artifacts: tuple[str, ...] = ()
    packed_c1_target: bool = False
    implementation_key: str | None = None

    def __post_init__(self) -> None:
        state = SpeculativeMTPStaticState(self.state)
        reason = _required_text(self.reason, "static eligibility reason")
        fallback = _required_text(self.strict_fallback_key, "strict_fallback_key")
        candidates = int(self.max_candidate_count)
        rows = int(self.max_realized_group_rows)
        if self.packed_c1_target and (
            state is not SpeculativeMTPStaticState.SPECULATIVE_CAPABLE or rows < 1
        ):
            raise ValueError("packed_c1_target requires speculative eligibility with a positive row bound")
        if min(candidates, rows) < 0:
            raise ValueError("static eligibility bounds must be non-negative")
        implementation = (
            None if self.implementation_key is None
            else _required_text(self.implementation_key, "implementation_key")
        )
        if state is SpeculativeMTPStaticState.SPECULATIVE_CAPABLE:
            if candidates <= 0 or rows <= 0:
                raise ValueError("speculative-capable eligibility requires positive bounds")
            evidence_key = (
                None if implementation is not None and self.evidence_key is None
                else _required_text(self.evidence_key, "evidence_key")
            )
            evidence_fingerprint = (
                None if implementation is not None and self.evidence_fingerprint is None
                else _required_text(self.evidence_fingerprint, "evidence_fingerprint")
            )
        else:
            if candidates or rows or self.automatic_eligible:
                raise ValueError("permanent-AR eligibility cannot retain speculative bounds")
            evidence_key = None if self.evidence_key is None else _required_text(
                self.evidence_key,
                "evidence_key",
            )
            evidence_fingerprint = (
                None
                if self.evidence_fingerprint is None
                else _required_text(self.evidence_fingerprint, "evidence_fingerprint")
            )
        artifacts = tuple(
            _required_text(value, "evidence_artifact")
            for value in self.evidence_artifacts
        )
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "max_candidate_count", candidates)
        object.__setattr__(self, "max_realized_group_rows", rows)
        object.__setattr__(self, "automatic_eligible", bool(self.automatic_eligible))
        object.__setattr__(self, "strict_fallback_key", fallback)
        object.__setattr__(self, "evidence_key", evidence_key)
        object.__setattr__(self, "evidence_fingerprint", evidence_fingerprint)
        object.__setattr__(self, "evidence_artifacts", artifacts)
        object.__setattr__(self, "implementation_key", implementation)

    @property
    def eligible(self) -> bool:
        return self.state is SpeculativeMTPStaticState.SPECULATIVE_CAPABLE

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "eligible": self.eligible,
            "reason": self.reason,
            "packed_c1_target": self.packed_c1_target,
            "max_candidate_count": self.max_candidate_count,
            "max_realized_group_rows": self.max_realized_group_rows,
            "automatic_eligible": self.automatic_eligible,
            "strict_fallback_key": self.strict_fallback_key,
            "evidence_key": self.evidence_key,
            "evidence_fingerprint": self.evidence_fingerprint,
            "evidence_artifacts": list(self.evidence_artifacts),
            **({"implementation_key": self.implementation_key} if self.implementation_key else {}),
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> "SpeculativeMTPStaticEligibility":
        state = payload.get("state")
        if state is None:
            state = (
                SpeculativeMTPStaticState.SPECULATIVE_CAPABLE
                if bool(payload.get("eligible"))
                else SpeculativeMTPStaticState.PERMANENT_AR
            )
        artifacts = payload.get("evidence_artifacts")
        return cls(
            state=SpeculativeMTPStaticState(state),
            reason=str(payload.get("reason") or "model_plugin_scope_unmeasured"),
            max_candidate_count=int(payload.get("max_candidate_count", 0) or 0),
            max_realized_group_rows=int(
                payload.get("max_realized_group_rows", 0) or 0
            ),
            automatic_eligible=bool(payload.get("automatic_eligible")),
            packed_c1_target=bool(payload.get("packed_c1_target", False)),
            implementation_key=payload.get("implementation_key"),
            strict_fallback_key=str(
                payload.get("strict_fallback_key") or _DEFAULT_STRICT_FALLBACK
            ),
            evidence_key=(
                None
                if payload.get("evidence_key") is None
                else str(payload.get("evidence_key"))
            ),
            evidence_fingerprint=(
                None
                if payload.get("evidence_fingerprint") is None
                else str(payload.get("evidence_fingerprint"))
            ),
            evidence_artifacts=(
                tuple(str(value) for value in artifacts)
                if isinstance(artifacts, Sequence)
                and not isinstance(artifacts, (str, bytes, bytearray))
                else ()
            ),
        )


@dataclass(frozen=True, slots=True)
class SpeculativeMTPServingDecision:
    """Immutable candidate-or-K0 decision resolved before backend mutation."""

    key: SpeculativeMTPServingKey
    admitted: bool
    selected_route: str
    selected_candidate_count: int
    reason: str
    strict_fallback_key: str
    evidence_key: str | None = None
    evidence_fingerprint: str | None = None
    evidence_artifacts: tuple[str, ...] = ()
    automatic_eligible: bool = False
    static_max_realized_group_rows: int | None = None
    static_eligibility_override: SpeculativeMTPStaticEligibility | None = None
    packed_c1_target: bool = False
    failed_axes: tuple[str, ...] = ()
    implementation_key: str | None = None
    # ``explicit`` when the request asked for speculation, ``automatic`` when the
    # server's own policy did.  Only an explicit request may admit on
    # implementation capability.
    request_mode: str = "automatic"

    @property
    def structural_rejection(self) -> str | None:
        """Return the failed correctness axis, independent of the summary reason.

        ``reason`` reports the first failed axis of the row that matched the
        most key axes, which is useful for diagnostics but is not a statement
        about the other axes. A caller deciding whether a rejection is an
        unmeasured physical cell (overridable) or a correctness boundary
        (never overridable) must read every failed axis, not the summary.
        """

        for axis in self.failed_axes:
            if axis in STRUCTURAL_REJECTION_AXES:
                return axis
        return None

    @property
    def static_eligibility(self) -> SpeculativeMTPStaticEligibility:
        if self.static_eligibility_override is not None:
            return self.static_eligibility_override
        eligible = bool(self.admitted and self.selected_candidate_count > 0)
        return SpeculativeMTPStaticEligibility(
            state=(
                SpeculativeMTPStaticState.SPECULATIVE_CAPABLE
                if eligible
                else SpeculativeMTPStaticState.PERMANENT_AR
            ),
            reason=self.reason,
            max_candidate_count=(self.selected_candidate_count if eligible else 0),
            max_realized_group_rows=(
                int(
                    self.static_max_realized_group_rows
                    if self.static_max_realized_group_rows is not None
                    else self.key.realized_group_rows
                )
                if eligible
                else 0
            ),
            automatic_eligible=(self.automatic_eligible if eligible else False),
            packed_c1_target=(self.packed_c1_target if eligible else False),
            strict_fallback_key=self.strict_fallback_key,
            evidence_key=self.evidence_key,
            evidence_fingerprint=self.evidence_fingerprint,
            evidence_artifacts=self.evidence_artifacts,
            implementation_key=self.implementation_key,
        )

    @property
    def plan_fingerprint(self) -> str:
        # ``failed_axes`` is deliberately absent: it is a pure function of the
        # key and the selected evidence row, and both are already covered here
        # (the key directly, the row through its evidence fingerprint).
        return _canonical_sha256(
            {
                "admitted": self.admitted,
                "selected_route": self.selected_route,
                "selected_candidate_count": self.selected_candidate_count,
                "reason": self.reason,
                "strict_fallback_key": self.strict_fallback_key,
                "evidence_key": self.evidence_key,
                "evidence_fingerprint": self.evidence_fingerprint,
                "automatic_eligible": self.automatic_eligible,
                "static_max_realized_group_rows": self.static_max_realized_group_rows,
                "static_eligibility": self.static_eligibility.as_dict(),
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_fingerprint": self.plan_fingerprint,
            "key": self.key.as_dict(),
            "admitted": self.admitted,
            "selected_route": self.selected_route,
            "selected_candidate_count": self.selected_candidate_count,
            "reason": self.reason,
            "strict_fallback_key": self.strict_fallback_key,
            "evidence_key": self.evidence_key,
            "evidence_fingerprint": self.evidence_fingerprint,
            "evidence_artifacts": list(self.evidence_artifacts),
            "automatic_eligible": self.automatic_eligible,
            "static_max_realized_group_rows": self.static_max_realized_group_rows,
            "static_eligibility": self.static_eligibility.as_dict(),
            "failed_axes": list(self.failed_axes),
            "request_mode": self.request_mode,
            **(
                {"admission_basis": "implementation", "implementation_key": self.implementation_key}
                if self.implementation_key else {}
            ),
        }


@dataclass(frozen=True, slots=True)
class SpeculativeMTPServingImplementation:
    """Executable scope, independently of performance/quality evidence rows.

    A declaration of what the kernels can execute for one KV contract.  It
    answers the runnability question only: an explicit request that no evidence
    row admits still runs when this declaration covers it, and a capability gap
    is the only reason to refuse.  Automatic policy stays with the retained
    evidence unless the declaration itself is automatic-eligible, which is for a
    contract that has no evidence rows and no wider policy to widen.
    """

    name: str
    kv_storage: str
    backends: tuple[tuple[str, str], ...]
    max_candidate_count: int
    max_group_rows: int
    group_rejection_reason: str
    kv_layouts: tuple[str, ...] = ("uniform",)
    sampling_modes: tuple[str, ...] = ("greedy_fast",)
    automatic_eligible: bool = False

    def resolve(
        self,
        key: SpeculativeMTPServingKey,
        *,
        request_mode: str = "automatic",
    ) -> SpeculativeMTPServingDecision:
        """Decide from implementation capability alone.

        This is the runnability question of docs/EXECUTION-PROFILES.md section
        1.1: implemented semantics, compatible storage and layout, supported
        sampling, allocated bounds, and available memory.  A missing
        measurement is not one of the checks.  The admission is automatic only
        when the declaration says so: automatic policy otherwise stays with the
        retained evidence.
        """

        checks = (
            ((key.backend, key.target_arch) in self.backends, "mtp_backend_unsupported"),
            (key.memory_fit, "insufficient_memory"),
            (key.kv_storage == self.kv_storage, "mtp_kv_storage_unsupported"),
            (key.kv_layout in self.kv_layouts, "mtp_kv_layout_unsupported"),
            (key.kv_scale_dtype in {None, "fp16", "fp32"}, "mtp_kv_scale_dtype_unsupported"),
            (key.kv_scale_granularity in {None, "per_token_head"}, "mtp_kv_scale_granularity_unsupported"),
            (key.sampling_mode in self.sampling_modes, "mtp_sampling_unsupported"),
            (key.candidate_budget <= self.max_candidate_count, "mtp_candidate_depth_unsupported"),
            (key.realized_group_rows <= self.max_group_rows, self.group_rejection_reason),
        )
        failed = tuple(reason for passed, reason in checks if not passed)
        admitted = not failed
        return SpeculativeMTPServingDecision(
            key=key, admitted=admitted,
            selected_route="speculative_mtp" if admitted else "default",
            selected_candidate_count=key.candidate_budget if admitted else 0,
            reason=failed[0] if failed else f"implemented_{self.name}",
            strict_fallback_key=_DEFAULT_STRICT_FALLBACK,
            automatic_eligible=bool(admitted and self.automatic_eligible),
            static_max_realized_group_rows=self.max_group_rows if admitted else None,
            failed_axes=failed,
            implementation_key=self.name,
            request_mode=str(request_mode),
        )


def _reject(
    key: SpeculativeMTPServingKey,
    reason: str,
    evidence: SpeculativeMTPServingEvidence | None,
    *,
    static_eligibility: SpeculativeMTPStaticEligibility | None = None,
    failed_axes: Sequence[str] = (),
    request_mode: str = "automatic",
) -> SpeculativeMTPServingDecision:
    return SpeculativeMTPServingDecision(
        key=key,
        admitted=False,
        selected_route="default",
        selected_candidate_count=0,
        reason=reason,
        strict_fallback_key=(
            _DEFAULT_STRICT_FALLBACK
            if evidence is None
            else evidence.strict_fallback_key
        ),
        evidence_key=None if evidence is None else evidence.evidence_key,
        evidence_fingerprint=(
            None if evidence is None else _canonical_sha256(evidence.as_dict())
        ),
        evidence_artifacts=(
            () if evidence is None else evidence.evidence_artifacts
        ),
        automatic_eligible=False,
        static_eligibility_override=static_eligibility,
        failed_axes=tuple(str(axis) for axis in failed_axes),
        request_mode=str(request_mode),
    )


def unsupported_contract(
    decision: SpeculativeMTPServingDecision,
) -> SpeculativeMTPServingDecision:
    """Relabel a terminal miss as the capability fact it actually is.

    When no implementation declaration covers the contract, the request is
    refused because nothing implements it -- not because nothing measured it.
    The per-axis measurement detail stays in ``failed_axes``; only the summary
    reason changes, so the caller reports a capability miss.
    """

    if decision.admitted:
        return decision
    return replace(decision, reason="mtp_contract_unsupported")


def _artifact_identity_matches(
    key: SpeculativeMTPServingKey,
    row: SpeculativeMTPServingEvidence,
) -> bool:
    """Whether the row's declared execution identity covers this artifact.

    This decides which retained measurement *applies*, never whether the cell
    runs: a cell no row covers is unmeasured, and the implementation
    declaration admits it.  Coverage is layout, not bytes, so a revision
    routing through the same kernels inherits the row.  An unverified artifact
    matches nothing, because attributing a measurement to a file we did not
    verify would overstate its provenance -- it still runs, on capability.
    """

    fingerprint = key.artifact_execution_fingerprint
    return bool(
        key.content_verified
        and fingerprint is not None
        and row.artifact_execution_fingerprint is not None
        and fingerprint == row.artifact_execution_fingerprint
    )


def _evidence_checks(
    key: SpeculativeMTPServingKey,
    row: SpeculativeMTPServingEvidence,
) -> tuple[tuple[bool, str], ...]:
    return (
        (
            _artifact_identity_matches(key, row),
            "artifact_unmeasured",
        ),
        (key.backend == row.backend, "backend_unmeasured"),
        (key.target_arch == row.target_arch, "target_arch_unmeasured"),
        (key.weight_quant == row.weight_quant, "weight_quant_unmeasured"),
        (key.kv_storage == row.kv_storage, "kv_storage_unmeasured"),
        (key.kv_layout == row.kv_layout, "kv_layout_unmeasured"),
        (
            key.realized_group_rows == row.realized_group_rows,
            "physical_group_unmeasured",
        ),
        (
            key.resident_capacity == row.resident_capacity,
            "resident_capacity_unmeasured",
        ),
        (
            key.candidate_budget <= row.candidate_budget,
            "candidate_budget_unmeasured",
        ),
        (key.sampling_mode in row.sampling_modes, "sampling_mode_unmeasured"),
        (key.memory_fit, "insufficient_memory"),
    )


def _admit(
    key: SpeculativeMTPServingKey,
    row: SpeculativeMTPServingEvidence,
    *,
    request_mode: str = "automatic",
) -> SpeculativeMTPServingDecision:
    return SpeculativeMTPServingDecision(
        key=key,
        admitted=True,
        selected_route="speculative_mtp",
        # The row qualifies a maximum depth; honour the requested one so a
        # shallower request never silently runs the deeper qualified chain.
        selected_candidate_count=min(
            int(key.candidate_budget), int(row.candidate_budget)
        ),
        reason=row.reason,
        strict_fallback_key=row.strict_fallback_key,
        evidence_key=row.evidence_key,
        evidence_fingerprint=_canonical_sha256(row.as_dict()),
        evidence_artifacts=row.evidence_artifacts,
        automatic_eligible=row.automatic_eligible,
        packed_c1_target=row.packed_c1_target,
        static_max_realized_group_rows=row.max_realized_group_rows,
        request_mode=str(request_mode),
    )


def resolve_max_qualified_candidate_budget(
    evidence_rows: Sequence[SpeculativeMTPServingEvidence],
    *,
    key: SpeculativeMTPServingKey,
) -> int | None:
    """Deepest candidate depth the retained rows qualify for this cell.

    Every axis except the requested depth must match, so the answer is the
    strongest depth the artifact's own evidence authorizes for this physical
    identity rather than a global constant. Automatic-eligible rows are
    preferred over explicit-only rows, mirroring the admission rule: a depth
    retained only for explicit use never becomes the default selection.
    ``None`` means no retained row describes the cell at all.
    """

    best: int | None = None
    best_automatic = False
    for row in evidence_rows:
        if not all(
            passed
            for passed, reason in _evidence_checks(key, row)
            if reason != "candidate_budget_unmeasured"
        ):
            continue
        depth = int(row.candidate_budget)
        automatic = bool(row.automatic_eligible)
        if (
            best is None
            or (automatic and not best_automatic)
            or (automatic == best_automatic and depth > best)
        ):
            best = depth
            best_automatic = automatic
    return best


def resolve_speculative_mtp_serving_plan(
    evidence_rows: Sequence[SpeculativeMTPServingEvidence],
    *,
    key: SpeculativeMTPServingKey,
    request_mode: str = "automatic",
) -> SpeculativeMTPServingDecision:
    """Resolve one exact model-plugin evidence row or fail closed to K0.

    A model artifact may carry independently qualified physical scopes.  Every
    axis is physical or ownership identity, so several rows can describe the
    same cell; the cell then takes the strongest retained authorization.  An
    automatic-eligible row wins over an explicit-only row for the same cell,
    and ties preserve declaration order.  When no row admits, rejection is
    attributed to the row matching the most key axes (ties preserve declaration
    order), keeping a stable and useful pre-mutation failure reason without
    merging scopes.
    """

    evidence = tuple(evidence_rows)
    if not evidence:
        return _reject(
            key,
            "no_model_plugin_evidence",
            None,
            request_mode=request_mode,
        )

    admitted: SpeculativeMTPServingEvidence | None = None
    for row in evidence:
        if all(passed for passed, _reason in _evidence_checks(key, row)):
            if admitted is None or (
                row.automatic_eligible and not admitted.automatic_eligible
            ):
                admitted = row
            if admitted.automatic_eligible:
                break
    if admitted is not None:
        return _admit(key, admitted, request_mode=request_mode)

    rejected: list[
        tuple[
            int,
            int,
            SpeculativeMTPServingEvidence,
            tuple[str, ...],
            SpeculativeMTPStaticEligibility | None,
        ]
    ] = []
    for index, row in enumerate(evidence):
        checks = _evidence_checks(key, row)
        failed_reasons = tuple(
            reason for passed, reason in checks if not passed
        )
        static_eligibility = None
        if (
            failed_reasons == ("physical_group_unmeasured",)
            and key.realized_group_rows < row.realized_group_rows
        ):
            static_eligibility = SpeculativeMTPStaticEligibility(
                state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
                reason=row.reason,
                max_candidate_count=row.candidate_budget,
                max_realized_group_rows=row.realized_group_rows,
                automatic_eligible=row.automatic_eligible,
                strict_fallback_key=row.strict_fallback_key,
                evidence_key=row.evidence_key,
                evidence_fingerprint=_canonical_sha256(row.as_dict()),
                evidence_artifacts=row.evidence_artifacts,
            )
        rejected.append(
            (
                sum(bool(passed) for passed, _reason in checks),
                -index,
                row,
                failed_reasons,
                static_eligibility,
            )
        )

    _matched, _order, row, failed_axes, static_eligibility = max(
        rejected,
        key=lambda item: (
            item[0],
            item[4] is not None,
            # Same rule as admission: when several rows describe one physical
            # cell, the strongest retained authorization wins over declaration
            # order.
            bool(item[4].automatic_eligible) if item[4] is not None else False,
            item[1],
        ),
    )
    return _reject(
        key,
        # The summary reason is the first failed axis in declaration order.
        # Consumers deciding whether the rejection is overridable must read
        # ``failed_axes`` instead, which carries every failed axis.
        failed_axes[0],
        row,
        static_eligibility=static_eligibility,
        failed_axes=failed_axes,
        request_mode=request_mode,
    )


__all__ = [
    "SpeculativeMTPServingDecision",
    "SpeculativeMTPServingEvidence",
    "SpeculativeMTPServingImplementation",
    "SpeculativeMTPServingKey",
    "SpeculativeMTPStaticEligibility",
    "SpeculativeMTPStaticState",
    "resolve_max_qualified_candidate_budget",
    "resolve_speculative_mtp_serving_plan",
    "STRUCTURAL_REJECTION_AXES",
]
