"""Artifact-scoped KV capability evidence and fail-closed resolution.

Model names and tensor geometry are not capability identities.  A KV route is
qualified only when immutable model content, backend/target, weight quantization, KV
layout, and scale contract all match one retained evidence record.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

KVCapabilityDecision = Literal["qualified", "rejected"]
KVCapabilityStatus = Literal["qualified", "rejected", "unknown", "not_applicable"]
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
    """Complete immutable key for one model/KV capability decision."""

    artifact_sha256: str | None
    artifact_size_bytes: int | None
    backend: str
    target_arch: str
    weight_quant: str
    kv_storage: str
    storage_layout: str
    scale_dtype: str
    scale_granularity: str

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "backend": self.backend,
            "target_arch": self.target_arch,
            "weight_quant": self.weight_quant,
            "kv_storage": self.kv_storage,
            "storage_layout": self.storage_layout,
            "scale_dtype": self.scale_dtype,
            "scale_granularity": self.scale_granularity,
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

    def __post_init__(self) -> None:
        if int(self.max_direct_rows) < 0:
            raise ValueError("max_direct_rows must be non-negative")
        if int(self.max_serial_resident_rows) < int(self.max_direct_rows):
            raise ValueError(
                "max_serial_resident_rows must cover every directly qualified row"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "scope": self.scope,
            "quality_artifact": self.quality_artifact,
            "reason": self.reason,
            "max_direct_rows": self.max_direct_rows,
            "max_serial_resident_rows": self.max_serial_resident_rows,
            "persistent_bf16_mirror": self.persistent_bf16_mirror,
        }


@dataclass(frozen=True)
class KVCapabilityResolution:
    """Runtime outcome for one requested KV contract."""

    key: KVCapabilityKey
    artifact: ModelArtifactIdentity
    status: KVCapabilityStatus
    effective_kv_storage: str
    evidence: KVCapabilityEvidence | None = None
    reason: str = ""
    runtime_action: KVCapabilityRuntimeAction = "fallback_bf16"

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


def resolve_kv_capability(
    evidence: Sequence[KVCapabilityEvidence],
    *,
    key: KVCapabilityKey,
    artifact: ModelArtifactIdentity,
) -> KVCapabilityResolution:
    """Resolve one exact key; names and geometry never participate."""

    if not artifact.content_verified or key.artifact_sha256 is None:
        return KVCapabilityResolution(
            key=key,
            artifact=artifact,
            status="unknown",
            effective_kv_storage="bf16",
            reason="model artifact content identity is unavailable",
            runtime_action="fallback_bf16",
        )

    match = next((row for row in evidence if row.key == key), None)
    if match is None:
        same_artifact = any(
            row.key.artifact_sha256 == key.artifact_sha256
            and row.key.artifact_size_bytes == key.artifact_size_bytes
            for row in evidence
        )
        reason = (
            "artifact is known but this backend/target/quant/KV/scale contract is unqualified"
            if same_artifact
            else "artifact/backend/target KV capability has no retained evidence"
        )
        return KVCapabilityResolution(
            key=key,
            artifact=artifact,
            status="unknown",
            effective_kv_storage="bf16",
            reason=reason,
            runtime_action="fallback_bf16",
        )

    return KVCapabilityResolution(
        key=key,
        artifact=artifact,
        status=match.decision,
        effective_kv_storage=(key.kv_storage if match.decision == "qualified" else "bf16"),
        evidence=match,
        reason=match.reason,
        runtime_action=("admit" if match.decision == "qualified" else "fallback_bf16"),
    )


__all__ = [
    "KVCapabilityEvidence",
    "KVCapabilityKey",
    "KVCapabilityResolution",
    "ModelArtifactIdentity",
    "model_artifact_identity",
    "resolve_kv_capability",
]
