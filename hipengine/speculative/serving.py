"""Artifact-scoped speculative-MTP serving admission.

This module owns only immutable pre-mutation decisions.  Model plugins supply
retained evidence; server/model code supplies mechanical request identity.  The
key deliberately has no prompt text, token IDs, benchmark category, heldout, or
oracle fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence


_DEFAULT_STRICT_FALLBACK = "gguf_target_ar"


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
    """Complete content, runtime, and request-shape identity for one plan."""

    artifact_sha256: str | None
    artifact_size_bytes: int | None
    content_verified: bool
    backend: str
    target_arch: str
    weight_quant: str
    execution_profile: str
    execution_profile_manifest_sha256: str
    kv_storage: str
    kv_layout: str
    realized_group_rows: int
    resident_capacity: int
    candidate_budget: int
    sampling_mode: str
    max_sequence_length: int
    context_tokens: int
    output_horizon_tokens: int
    memory_fit: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(self.artifact_sha256, "artifact_sha256", optional=True),
        )
        object.__setattr__(
            self,
            "execution_profile_manifest_sha256",
            _sha256(
                self.execution_profile_manifest_sha256,
                "execution_profile_manifest_sha256",
            ),
        )
        for name in (
            "backend",
            "target_arch",
            "weight_quant",
            "execution_profile",
            "kv_storage",
            "kv_layout",
            "sampling_mode",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        for name in (
            "realized_group_rows",
            "resident_capacity",
            "candidate_budget",
            "max_sequence_length",
            "context_tokens",
            "output_horizon_tokens",
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
            "content_verified": self.content_verified,
            "backend": self.backend,
            "target_arch": self.target_arch,
            "weight_quant": self.weight_quant,
            "execution_profile": self.execution_profile,
            "execution_profile_manifest_sha256": self.execution_profile_manifest_sha256,
            "kv_storage": self.kv_storage,
            "kv_layout": self.kv_layout,
            "realized_group_rows": self.realized_group_rows,
            "resident_capacity": self.resident_capacity,
            "candidate_budget": self.candidate_budget,
            "sampling_mode": self.sampling_mode,
            "max_sequence_length": self.max_sequence_length,
            "context_tokens": self.context_tokens,
            "output_horizon_tokens": self.output_horizon_tokens,
            "memory_fit": self.memory_fit,
        }


@dataclass(frozen=True, slots=True)
class SpeculativeMTPServingEvidence:
    """One model-plugin-owned retained serving scope."""

    evidence_key: str
    artifact_sha256: str
    artifact_size_bytes: int
    backend: str
    target_arch: str
    weight_quant: str
    execution_profile: str
    execution_profile_manifest_sha256: str
    kv_storage: str
    kv_layout: str
    realized_group_rows: int
    resident_capacity: int
    candidate_budget: int
    sampling_modes: tuple[str, ...]
    max_sequence_length: int
    min_context_tokens: int
    max_context_tokens: int
    min_output_horizon_tokens: int
    max_output_horizon_tokens: int
    reason: str
    evidence_artifacts: tuple[str, ...]
    strict_fallback_key: str = _DEFAULT_STRICT_FALLBACK
    automatic_eligible: bool = False

    def __post_init__(self) -> None:
        for name in (
            "evidence_key",
            "backend",
            "target_arch",
            "weight_quant",
            "execution_profile",
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
            "execution_profile_manifest_sha256",
            _sha256(
                self.execution_profile_manifest_sha256,
                "execution_profile_manifest_sha256",
            ),
        )
        for name in (
            "artifact_size_bytes",
            "realized_group_rows",
            "resident_capacity",
            "candidate_budget",
            "max_sequence_length",
            "min_context_tokens",
            "max_context_tokens",
            "min_output_horizon_tokens",
            "max_output_horizon_tokens",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.max_context_tokens < self.min_context_tokens:
            raise ValueError("context-token bounds are invalid")
        if self.max_output_horizon_tokens < self.min_output_horizon_tokens:
            raise ValueError("output-horizon bounds are invalid")
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
            "backend": self.backend,
            "target_arch": self.target_arch,
            "weight_quant": self.weight_quant,
            "execution_profile": self.execution_profile,
            "execution_profile_manifest_sha256": self.execution_profile_manifest_sha256,
            "kv_storage": self.kv_storage,
            "kv_layout": self.kv_layout,
            "realized_group_rows": self.realized_group_rows,
            "resident_capacity": self.resident_capacity,
            "candidate_budget": self.candidate_budget,
            "sampling_modes": list(self.sampling_modes),
            "max_sequence_length": self.max_sequence_length,
            "min_context_tokens": self.min_context_tokens,
            "max_context_tokens": self.max_context_tokens,
            "min_output_horizon_tokens": self.min_output_horizon_tokens,
            "max_output_horizon_tokens": self.max_output_horizon_tokens,
            "reason": self.reason,
            "evidence_artifacts": list(self.evidence_artifacts),
            "strict_fallback_key": self.strict_fallback_key,
            "automatic_eligible": self.automatic_eligible,
        }


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

    @property
    def plan_fingerprint(self) -> str:
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
        }


def _reject(
    key: SpeculativeMTPServingKey,
    reason: str,
    evidence: SpeculativeMTPServingEvidence | None,
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
    )


def _evidence_checks(
    key: SpeculativeMTPServingKey,
    row: SpeculativeMTPServingEvidence,
) -> tuple[tuple[bool, str], ...]:
    return (
        (
            key.artifact_sha256 == row.artifact_sha256
            and key.artifact_size_bytes == row.artifact_size_bytes,
            "artifact_not_qualified",
        ),
        (key.backend == row.backend, "backend_not_qualified"),
        (key.target_arch == row.target_arch, "target_arch_not_qualified"),
        (key.weight_quant == row.weight_quant, "weight_quant_not_qualified"),
        (
            key.execution_profile == row.execution_profile,
            "execution_profile_not_qualified",
        ),
        (
            key.execution_profile_manifest_sha256
            == row.execution_profile_manifest_sha256,
            "execution_profile_manifest_not_qualified",
        ),
        (key.kv_storage == row.kv_storage, "kv_storage_not_qualified"),
        (key.kv_layout == row.kv_layout, "kv_layout_not_qualified"),
        (
            key.realized_group_rows == row.realized_group_rows,
            "physical_group_not_qualified",
        ),
        (
            key.resident_capacity == row.resident_capacity,
            "resident_capacity_not_qualified",
        ),
        (
            key.candidate_budget == row.candidate_budget,
            "candidate_budget_not_qualified",
        ),
        (key.sampling_mode in row.sampling_modes, "sampling_mode_not_qualified"),
        (
            key.max_sequence_length == row.max_sequence_length,
            "max_sequence_length_not_qualified",
        ),
        (
            row.min_context_tokens <= key.context_tokens <= row.max_context_tokens,
            "context_bucket_not_qualified",
        ),
        (
            row.min_output_horizon_tokens
            <= key.output_horizon_tokens
            <= row.max_output_horizon_tokens,
            "output_horizon_not_qualified",
        ),
        (key.memory_fit, "insufficient_memory"),
    )


def _admit(
    key: SpeculativeMTPServingKey,
    row: SpeculativeMTPServingEvidence,
) -> SpeculativeMTPServingDecision:
    return SpeculativeMTPServingDecision(
        key=key,
        admitted=True,
        selected_route="speculative_mtp",
        selected_candidate_count=row.candidate_budget,
        reason=row.reason,
        strict_fallback_key=row.strict_fallback_key,
        evidence_key=row.evidence_key,
        evidence_fingerprint=_canonical_sha256(row.as_dict()),
        evidence_artifacts=row.evidence_artifacts,
        automatic_eligible=row.automatic_eligible,
    )


def resolve_speculative_mtp_serving_plan(
    evidence_rows: Sequence[SpeculativeMTPServingEvidence],
    *,
    key: SpeculativeMTPServingKey,
) -> SpeculativeMTPServingDecision:
    """Resolve one exact model-plugin evidence row or fail closed to K0.

    A model artifact may carry independently qualified profile/shape scopes.
    The first exact row wins. When no row admits, rejection is attributed to the
    row matching the most key axes (ties preserve declaration order), keeping a
    stable and useful pre-mutation failure reason without merging scopes.
    """

    evidence = tuple(evidence_rows)
    if not key.content_verified or key.artifact_sha256 is None:
        return _reject(key, "artifact_identity_unverified", evidence[0] if evidence else None)
    if not evidence:
        return _reject(key, "no_model_plugin_evidence", None)

    rejected: list[tuple[int, int, SpeculativeMTPServingEvidence, str]] = []
    for index, row in enumerate(evidence):
        checks = _evidence_checks(key, row)
        failed_reason = next((reason for passed, reason in checks if not passed), None)
        if failed_reason is None:
            return _admit(key, row)
        rejected.append(
            (
                sum(bool(passed) for passed, _reason in checks),
                -index,
                row,
                failed_reason,
            )
        )

    _matched, _order, row, reason = max(rejected, key=lambda item: (item[0], item[1]))
    return _reject(key, reason, row)


__all__ = [
    "SpeculativeMTPServingDecision",
    "SpeculativeMTPServingEvidence",
    "SpeculativeMTPServingKey",
    "resolve_speculative_mtp_serving_plan",
]
