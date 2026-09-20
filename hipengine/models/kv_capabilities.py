"""KV capability admission, and the evidence that selects and records it.

Capability decides whether a KV contract runs: a registered declaration states
what the kernels execute for one backend/target/quant/KV/scale contract, and a
contract no declaration covers is the only thing this module refuses.

Retained evidence does not admit.  A row adds a measured guarantee to a
contract capability already admits, or -- when it records a rejection --
withholds the path as a known-bad configuration, scoped to the artifact
execution identity that recorded it.  A contract the kernels implement that
nobody has measured runs, and reports ``unmeasured``.

No identity gates execution.  Model names, file paths, artifact SHA-256, and
size are provenance: they label which file a measurement came from.  See
``docs/EXECUTION-PROFILES.md`` section 2.9.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

KVCapabilityDecision = Literal["qualified", "rejected"]
KVCapabilityStatus = Literal[
    "qualified",
    "unmeasured",
    "rejected",
    "unsupported",
    "unknown",
    "not_applicable",
]
KVCapabilityRuntimeAction = Literal[
    "admit",
    "diagnostic_override",
    "fallback_bf16",
    "not_applicable",
]

_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_HASH_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ModelArtifactIdentity:
    """Immutable content identity used by model-plugin capability gates."""

    path: str
    size_bytes: int | None
    sha256: str | None
    content_verified: bool
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "content_verified": self.content_verified,
            "error": self.error,
        }


@dataclass(frozen=True)
class KVCapabilityKey:
    """Complete immutable key for one model/KV capability decision.

    ``artifact_sha256`` and ``artifact_size_bytes`` are provenance: they report
    which file is resident.  Admission reads
    ``artifact_execution_fingerprint``, which is the only artifact axis that
    participates in matching.
    """

    artifact_sha256: str | None
    artifact_size_bytes: int | None
    backend: str
    target_arch: str
    weight_quant: str
    kv_storage: str
    storage_layout: str
    scale_dtype: str
    scale_granularity: str
    artifact_execution_fingerprint: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_execution_fingerprint": self.artifact_execution_fingerprint,
            "backend": self.backend,
            "target_arch": self.target_arch,
            "weight_quant": self.weight_quant,
            "kv_storage": self.kv_storage,
            "storage_layout": self.storage_layout,
            "scale_dtype": self.scale_dtype,
            "scale_granularity": self.scale_granularity,
        }


@dataclass(frozen=True)
class KVCapabilityDeclaration:
    """One KV contract the kernels implement, independent of any artifact.

    This is the admission axis.  Every field is a property of the request and
    the kernel; none is an identity.  An artifact the project has never seen
    runs when its contract matches a declaration.
    """

    backend: str
    target_arch: str
    kv_storage: str
    storage_layout: str
    scale_dtype: str
    scale_granularity: str
    weight_quant: str | None = None
    """Bind the weight quant only when the kernel key actually includes it.

    ``None`` means any weight quant, which is the honest declaration for a KV
    kernel keyed on ``(backend, layer, kv_storage, variant)``.  Naming a quant
    here that the kernel does not key on rebuilds the allowlist this module
    exists to remove.
    """

    max_direct_rows: int = 1
    max_serial_resident_rows: int = 1
    persistent_bf16_mirror: bool | None = None
    decode_batch_variant: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if int(self.max_direct_rows) < 0:
            raise ValueError("max_direct_rows must be non-negative")
        if int(self.max_serial_resident_rows) < int(self.max_direct_rows):
            raise ValueError(
                "max_serial_resident_rows must cover every directly admitted row"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "target_arch": self.target_arch,
            "weight_quant": self.weight_quant,
            "kv_storage": self.kv_storage,
            "storage_layout": self.storage_layout,
            "scale_dtype": self.scale_dtype,
            "scale_granularity": self.scale_granularity,
            "max_direct_rows": self.max_direct_rows,
            "max_serial_resident_rows": self.max_serial_resident_rows,
            "persistent_bf16_mirror": self.persistent_bf16_mirror,
            "decode_batch_variant": self.decode_batch_variant,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class KVCapabilityEvidence:
    """One retained qualified or rejected artifact/backend/target decision."""

    key: KVCapabilityKey
    decision: KVCapabilityDecision
    scope: str
    quality_artifact: str
    reason: str
    max_direct_rows: int = 1
    max_serial_resident_rows: int = 1
    persistent_bf16_mirror: bool | None = None
    decode_batch_variant: str | None = None

    def __post_init__(self) -> None:
        if int(self.max_direct_rows) < 0:
            raise ValueError("max_direct_rows must be non-negative")
        if int(self.max_serial_resident_rows) < int(self.max_direct_rows):
            raise ValueError(
                "max_serial_resident_rows must cover every directly qualified row"
            )
        if self.decode_batch_variant is not None:
            variant = str(self.decode_batch_variant).strip()
            if not variant:
                raise ValueError("decode_batch_variant must not be empty")
            object.__setattr__(self, "decode_batch_variant", variant)

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "scope": self.scope,
            "quality_artifact": self.quality_artifact,
            "reason": self.reason,
            "max_direct_rows": self.max_direct_rows,
            "max_serial_resident_rows": self.max_serial_resident_rows,
            "persistent_bf16_mirror": self.persistent_bf16_mirror,
            "decode_batch_variant": self.decode_batch_variant,
        }


@dataclass(frozen=True)
class KVCapabilityResolution:
    """Runtime outcome for one requested KV contract."""

    key: KVCapabilityKey
    artifact: ModelArtifactIdentity
    status: KVCapabilityStatus
    effective_kv_storage: str
    evidence: KVCapabilityEvidence | None = None
    declaration: KVCapabilityDeclaration | None = None
    reason: str = ""
    runtime_action: KVCapabilityRuntimeAction = "fallback_bf16"

    @property
    def max_direct_rows(self) -> int:
        """Operative direct-row bound: measured when retained, else declared."""

        if self.evidence is not None:
            return int(self.evidence.max_direct_rows)
        if self.declaration is not None:
            return int(self.declaration.max_direct_rows)
        return 0

    @property
    def max_serial_resident_rows(self) -> int:
        if self.evidence is not None:
            return int(self.evidence.max_serial_resident_rows)
        if self.declaration is not None:
            return int(self.declaration.max_serial_resident_rows)
        return 0

    @property
    def persistent_bf16_mirror(self) -> bool | None:
        if self.evidence is not None:
            return self.evidence.persistent_bf16_mirror
        if self.declaration is not None:
            return self.declaration.persistent_bf16_mirror
        return None

    @property
    def decode_batch_variant(self) -> str | None:
        if self.evidence is not None:
            return self.evidence.decode_batch_variant
        if self.declaration is not None:
            return self.declaration.decode_batch_variant
        return None

    @property
    def promotion_eligible(self) -> bool:
        return (
            self.status == "qualified"
            and self.evidence is not None
            and self.runtime_action == "admit"
        )

    @property
    def capability_id(self) -> str:
        payload = {
            "key": self.key.as_dict(),
            "evidence": (
                None
                if self.evidence is None
                else {
                    "decision": self.evidence.decision,
                    "scope": self.evidence.scope,
                    "quality_artifact": self.evidence.quality_artifact,
                    "max_direct_rows": self.evidence.max_direct_rows,
                    "max_serial_resident_rows": self.evidence.max_serial_resident_rows,
                    "persistent_bf16_mirror": self.evidence.persistent_bf16_mirror,
                    "decode_batch_variant": self.evidence.decode_batch_variant,
                }
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def with_runtime_outcome(
        self,
        *,
        effective_kv_storage: str,
        runtime_action: KVCapabilityRuntimeAction,
        reason: str,
    ) -> "KVCapabilityResolution":
        return replace(
            self,
            effective_kv_storage=effective_kv_storage,
            runtime_action=runtime_action,
            reason=reason,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "capability_id": self.capability_id,
            "status": self.status,
            "runtime_action": self.runtime_action,
            "promotion_eligible": self.promotion_eligible,
            "max_direct_rows": self.max_direct_rows,
            "max_serial_resident_rows": self.max_serial_resident_rows,
            "persistent_bf16_mirror": self.persistent_bf16_mirror,
            "decode_batch_variant": self.decode_batch_variant,
            "declaration": (
                None if self.declaration is None else self.declaration.as_dict()
            ),
            "diagnostic_override": self.runtime_action == "diagnostic_override",
            "requested": self.key.as_dict(),
            "effective_kv_storage": self.effective_kv_storage,
            "artifact": self.artifact.as_dict(),
            "evidence": None if self.evidence is None else self.evidence.as_dict(),
            "reason": self.reason,
        }


def model_artifact_identity(path: str | Path) -> ModelArtifactIdentity:
    """Return a cached full-file SHA-256 identity.

    Hashing is intentionally demand-driven: normal BF16 startup never pays this
    cost.  Explicit approximate-KV admission hashes once per stable
    ``(resolved path, size, mtime)`` tuple and reuses the result thereafter.
    """

    requested = Path(path).expanduser()
    try:
        resolved = requested.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        return ModelArtifactIdentity(
            path=str(requested),
            size_bytes=None,
            sha256=None,
            content_verified=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    cache_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    with _HASH_CACHE_LOCK:
        cached = _HASH_CACHE.get(cache_key)
    if cached is None:
        digest = hashlib.sha256()
        try:
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            return ModelArtifactIdentity(
                path=str(resolved),
                size_bytes=int(stat.st_size),
                sha256=None,
                content_verified=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            final_stat = resolved.stat()
        except OSError as exc:
            return ModelArtifactIdentity(
                path=str(resolved),
                size_bytes=int(stat.st_size),
                sha256=None,
                content_verified=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        if (
            int(final_stat.st_size) != int(stat.st_size)
            or int(final_stat.st_mtime_ns) != int(stat.st_mtime_ns)
        ):
            return ModelArtifactIdentity(
                path=str(resolved),
                size_bytes=int(final_stat.st_size),
                sha256=None,
                content_verified=False,
                error="model artifact changed while SHA-256 was being computed",
            )
        cached = digest.hexdigest()
        with _HASH_CACHE_LOCK:
            _HASH_CACHE[cache_key] = cached
    return ModelArtifactIdentity(
        path=str(resolved),
        size_bytes=int(stat.st_size),
        sha256=cached,
        content_verified=True,
    )


def _artifact_identity_matches(row: KVCapabilityKey, key: KVCapabilityKey) -> bool:
    """Whether the retained row's execution identity covers this artifact.

    The artifact axis is layout coverage, not byte identity: a revision that
    routes through the same kernels inherits the retained decision, and an
    artifact whose execution identity no row declares fails closed.  A row that
    declares no identity never matches.
    """

    fingerprint = key.artifact_execution_fingerprint
    return bool(
        fingerprint is not None
        and row.artifact_execution_fingerprint is not None
        and fingerprint == row.artifact_execution_fingerprint
    )


def _declaration_matches(row: KVCapabilityDeclaration, key: KVCapabilityKey) -> bool:
    """Whether the kernels declare this backend/target/quant/KV/scale contract."""

    return (
        row.backend == key.backend
        and row.target_arch == key.target_arch
        and (row.weight_quant is None or row.weight_quant == key.weight_quant)
        and row.kv_storage == key.kv_storage
        and row.storage_layout == key.storage_layout
        and row.scale_dtype == key.scale_dtype
        and row.scale_granularity == key.scale_granularity
    )


def _key_matches(row: KVCapabilityKey, key: KVCapabilityKey) -> bool:
    """Match every non-artifact axis exactly and the artifact axis by identity."""

    return (
        _artifact_identity_matches(row, key)
        and row.backend == key.backend
        and row.target_arch == key.target_arch
        and row.weight_quant == key.weight_quant
        and row.kv_storage == key.kv_storage
        and row.storage_layout == key.storage_layout
        and row.scale_dtype == key.scale_dtype
        and row.scale_granularity == key.scale_granularity
    )


def resolve_kv_capability(
    evidence: Sequence[KVCapabilityEvidence],
    *,
    key: KVCapabilityKey,
    artifact: ModelArtifactIdentity,
    declarations: Sequence[KVCapabilityDeclaration] = (),
) -> KVCapabilityResolution:
    """Admit on kernel capability; let evidence select, record, and reject.

    A contract no declaration covers is refused, and the refusal names the
    capability miss.  Everything a declaration covers runs.  A retained
    ``qualified`` row upgrades the result to a measured guarantee; a retained
    ``rejected`` row withholds the path as a known-bad configuration, scoped to
    the artifact execution identity that recorded it.  Absence of any row is
    ``unmeasured``, which runs and is simply not promotable.
    """

    declaration = next(
        (row for row in declarations if _declaration_matches(row, key)), None
    )
    if declaration is None:
        return KVCapabilityResolution(
            key=key,
            artifact=artifact,
            status="unsupported",
            effective_kv_storage="bf16",
            reason=(
                "no registered kernel implements this backend/target/quant/KV/"
                "scale contract"
            ),
            runtime_action="fallback_bf16",
        )

    match = next((row for row in evidence if _key_matches(row.key, key)), None)

    if match is not None and match.decision == "rejected":
        return KVCapabilityResolution(
            key=key,
            artifact=artifact,
            status="rejected",
            effective_kv_storage="bf16",
            evidence=match,
            declaration=declaration,
            reason=match.reason,
            runtime_action="fallback_bf16",
        )

    if match is not None:
        return KVCapabilityResolution(
            key=key,
            artifact=artifact,
            status="qualified",
            effective_kv_storage=key.kv_storage,
            evidence=match,
            declaration=declaration,
            reason=match.reason,
            runtime_action="admit",
        )

    return KVCapabilityResolution(
        key=key,
        artifact=artifact,
        status="unmeasured",
        effective_kv_storage=key.kv_storage,
        declaration=declaration,
        reason=(
            declaration.reason
            or "kernels implement this contract; no retained measurement covers it"
        ),
        runtime_action="admit",
    )


__all__ = [
    "KVCapabilityDeclaration",
    "KVCapabilityEvidence",
    "KVCapabilityKey",
    "KVCapabilityResolution",
    "ModelArtifactIdentity",
    "model_artifact_identity",
    "resolve_kv_capability",
]
