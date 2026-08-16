"""Torch-free execution-profile identities and variant-manifest contracts.

Execution profile is a selector over the existing four-axis kernel registry.  It
is deliberately not part of ``KernelKey`` and this module performs no backend or
model dispatch.  Runtime plumbing can resolve a manifest once and benchmark
code can hash the exact same representation for provenance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


EXECUTION_PROFILE_MANIFEST_KIND = "hipengine_execution_profile_manifest"
EXECUTION_PROFILE_SCHEMA_VERSION = 1


class ExecutionProfile(str, Enum):
    STRICT = "strict"
    PRODUCTION = "production"
    BATCH_INVARIANT = "batch_invariant"


@dataclass(frozen=True, slots=True)
class VariantSelection:
    """One scope-qualified registry variant and its strict rollback."""

    layer: str
    scope: str
    selected_variant: str
    strict_fallback_variant: str | None
    evidence_artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _profile(value: ExecutionProfile | str) -> ExecutionProfile:
    if isinstance(value, ExecutionProfile):
        return value
    try:
        return ExecutionProfile(str(value))
    except ValueError as exc:
        allowed = ", ".join(profile.value for profile in ExecutionProfile)
        raise ValueError(f"execution_profile must be one of: {allowed}") from exc


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"variant manifest {field} must be a non-empty string")
    return value.strip()


def _selection(value: VariantSelection | Mapping[str, Any]) -> dict[str, Any]:
    payload = value.to_dict() if isinstance(value, VariantSelection) else dict(value)
    required = ("layer", "scope", "selected_variant", "strict_fallback_variant")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"variant selection missing fields: {missing}")
    result = {
        "layer": _nonempty(payload["layer"], field="selection.layer"),
        "scope": _nonempty(payload["scope"], field="selection.scope"),
        "selected_variant": _nonempty(
            payload["selected_variant"], field="selection.selected_variant"
        ),
        "strict_fallback_variant": None,
        "evidence_artifact": None,
    }
    fallback = payload.get("strict_fallback_variant")
    if fallback is not None:
        result["strict_fallback_variant"] = _nonempty(
            fallback, field="selection.strict_fallback_variant"
        )
    evidence = payload.get("evidence_artifact")
    if evidence is not None:
        result["evidence_artifact"] = _nonempty(
            evidence, field="selection.evidence_artifact"
        )
    unknown = set(payload) - set(result)
    if unknown:
        raise ValueError(f"variant selection has unknown fields: {sorted(unknown)}")
    return result


def build_variant_manifest(
    *,
    profile: ExecutionProfile | str,
    backend: str,
    model: str,
    quant: str,
    kv_policy: str,
    graph_policy: str,
    selections: Sequence[VariantSelection | Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and validate the canonical immutable manifest payload."""

    normalized_profile = _profile(profile)
    normalized_selections = [_selection(item) for item in selections]
    normalized_selections.sort(
        key=lambda item: (item["layer"], item["scope"], item["selected_variant"])
    )
    payload: dict[str, Any] = {
        "kind": EXECUTION_PROFILE_MANIFEST_KIND,
        "schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "execution_profile": normalized_profile.value,
        "backend": _nonempty(backend, field="backend"),
        "model": _nonempty(model, field="model"),
        "quant": _nonempty(quant, field="quant"),
        "kv_policy": _nonempty(kv_policy, field="kv_policy"),
        "graph_policy": _nonempty(graph_policy, field="graph_policy"),
        "selections": normalized_selections,
    }
    return validate_variant_manifest(payload)


def validate_variant_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a profile manifest and return a normalized plain dictionary."""

    required = {
        "kind",
        "schema_version",
        "execution_profile",
        "backend",
        "model",
        "quant",
        "kv_policy",
        "graph_policy",
        "selections",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"variant manifest missing fields: {sorted(missing)}")
    unknown = set(payload) - required
    if unknown:
        raise ValueError(f"variant manifest has unknown fields: {sorted(unknown)}")
    if payload.get("kind") != EXECUTION_PROFILE_MANIFEST_KIND:
        raise ValueError(f"variant manifest kind must be {EXECUTION_PROFILE_MANIFEST_KIND!r}")
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != EXECUTION_PROFILE_SCHEMA_VERSION
    ):
        raise ValueError(
            "variant manifest schema_version must be "
            f"{EXECUTION_PROFILE_SCHEMA_VERSION}"
        )
    profile = _profile(payload.get("execution_profile"))
    raw_selections = payload.get("selections")
    if not isinstance(raw_selections, Sequence) or isinstance(raw_selections, (str, bytes)):
        raise ValueError("variant manifest selections must be a sequence")
    selections = [_selection(item) for item in raw_selections]
    if not selections:
        raise ValueError("variant manifest selections must not be empty")
    keys: set[tuple[str, str]] = set()
    for item in selections:
        key = (item["layer"], item["scope"])
        if key in keys:
            raise ValueError(f"duplicate variant selection for layer/scope: {key!r}")
        keys.add(key)
        if profile in {ExecutionProfile.PRODUCTION, ExecutionProfile.BATCH_INVARIANT}:
            if item["strict_fallback_variant"] is None:
                raise ValueError(
                    "production and batch_invariant selections require a strict fallback"
                )
    selections.sort(key=lambda item: (item["layer"], item["scope"], item["selected_variant"]))
    return {
        "kind": EXECUTION_PROFILE_MANIFEST_KIND,
        "schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "execution_profile": profile.value,
        "backend": _nonempty(payload["backend"], field="backend"),
        "model": _nonempty(payload["model"], field="model"),
        "quant": _nonempty(payload["quant"], field="quant"),
        "kv_policy": _nonempty(payload["kv_policy"], field="kv_policy"),
        "graph_policy": _nonempty(payload["graph_policy"], field="graph_policy"),
        "selections": selections,
    }


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 over the validated canonical manifest."""

    normalized = validate_variant_manifest(payload)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "EXECUTION_PROFILE_MANIFEST_KIND",
    "EXECUTION_PROFILE_SCHEMA_VERSION",
    "ExecutionProfile",
    "VariantSelection",
    "build_variant_manifest",
    "manifest_sha256",
    "validate_variant_manifest",
]
