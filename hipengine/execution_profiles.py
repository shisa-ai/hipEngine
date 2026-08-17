"""Torch-free execution-profile identities and variant-manifest contracts.

Execution profile is a selector over the existing four-axis kernel registry.  It
is deliberately not part of ``KernelKey`` and this module performs no backend or
model dispatch.  Runtime plumbing can resolve a manifest once and benchmark
code can hash the exact same representation for provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence


EXECUTION_PROFILE_MANIFEST_KIND = "hipengine_execution_profile_manifest"
EXECUTION_PROFILE_SCHEMA_VERSION = 1
EXECUTION_PROFILE_ENV = "HIPENGINE_EXECUTION_PROFILE"


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
    registry_quant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeProfilePlan:
    """Plugin-owned cold-path construction plan for one execution profile."""

    selections: tuple[VariantSelection, ...]
    kv_policy: str
    graph_policy: str
    factory: Callable[..., Any] | None = None
    binder: Callable[[Any, "ResolvedRuntimeProfile"], None] | None = None

    def __post_init__(self) -> None:
        normalized = tuple(
            selection
            if isinstance(selection, VariantSelection)
            else VariantSelection(**_selection(selection))
            for selection in self.selections
        )
        if not normalized:
            raise ValueError("runtime execution-profile plan needs at least one selection")
        if self.factory is None and self.binder is None:
            raise ValueError("runtime execution-profile plan needs a factory or binder")
        object.__setattr__(self, "selections", normalized)
        object.__setattr__(self, "kv_policy", _nonempty(self.kv_policy, field="kv_policy"))
        object.__setattr__(
            self,
            "graph_policy",
            _nonempty(self.graph_policy, field="graph_policy"),
        )


@dataclass(frozen=True, slots=True, order=True)
class RuntimeProfileKey:
    model: str
    backend: str
    quant: str
    profile: ExecutionProfile


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeProfile:
    """One immutable profile manifest plus its plugin construction hook."""

    profile: ExecutionProfile
    manifest: Mapping[str, Any]
    manifest_sha256: str
    strict_manifest_sha256: str
    factory: Callable[..., Any] | None
    binder: Callable[[Any, "ResolvedRuntimeProfile"], None] | None
    fell_back_to_strict: bool
    source_profile: ExecutionProfile

    def construct_generator(
        self,
        base_factory: Callable[..., Any],
        **factory_kwargs: Any,
    ) -> Any:
        """Construct and bind the generator before any resident runner is created."""

        selected_factory = self.factory or base_factory
        generator = selected_factory(**factory_kwargs)
        if self.binder is not None:
            self.binder(generator, self)
        for name, value in (
            ("execution_profile", self.profile.value),
            ("execution_profile_manifest", self.manifest),
            ("execution_profile_manifest_sha256", self.manifest_sha256),
            ("execution_profile_strict_manifest_sha256", self.strict_manifest_sha256),
            ("execution_profile_fell_back_to_strict", self.fell_back_to_strict),
        ):
            existing = getattr(generator, name, None)
            if existing is not None and existing != value:
                raise RuntimeError(f"profile-aware generator returned conflicting {name}")
            try:
                setattr(generator, name, value)
            except (AttributeError, TypeError) as exc:
                raise TypeError(
                    "profile-aware generator must accept immutable profile metadata"
                ) from exc
        return generator


class DuplicateRuntimeProfilePlanError(ValueError):
    pass


class MissingRuntimeProfilePlanError(LookupError):
    pass


_RUNTIME_PROFILE_PLANS: dict[RuntimeProfileKey, RuntimeProfilePlan] = {}


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
        "registry_quant": None,
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
    registry_quant = payload.get("registry_quant")
    if registry_quant is not None:
        result["registry_quant"] = _nonempty(
            registry_quant, field="selection.registry_quant"
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


def resolve_requested_execution_profile(
    value: ExecutionProfile | str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ExecutionProfile | None:
    """Normalize an explicit selector or the opt-in migration environment value."""

    if value is not None:
        return _profile(value)
    env = os.environ if environ is None else environ
    raw = env.get(EXECUTION_PROFILE_ENV)
    if raw is None or not str(raw).strip():
        return None
    return _profile(str(raw).strip())


def register_runtime_profile_plan(
    *,
    model: str,
    backend: str,
    quant: str,
    profile: ExecutionProfile | str,
    plan: RuntimeProfilePlan,
    replace: bool = False,
) -> RuntimeProfilePlan:
    """Register one model plugin's cold-path profile construction plan."""

    if not isinstance(plan, RuntimeProfilePlan):
        raise TypeError("plan must be RuntimeProfilePlan")
    key = RuntimeProfileKey(
        model=_nonempty(model, field="model"),
        backend=_nonempty(backend, field="backend"),
        quant=_nonempty(quant, field="quant"),
        profile=_profile(profile),
    )
    if key in _RUNTIME_PROFILE_PLANS and not replace:
        raise DuplicateRuntimeProfilePlanError(
            f"runtime execution-profile plan already registered for {key!r}"
        )
    _RUNTIME_PROFILE_PLANS[key] = plan
    return plan


def _selection_key(selection: VariantSelection) -> tuple[str, str]:
    return (selection.layer, selection.scope)


def _resolved_selections(
    *,
    requested: ExecutionProfile,
    strict_plan: RuntimeProfilePlan,
    requested_plan: RuntimeProfilePlan | None,
) -> tuple[tuple[VariantSelection, ...], bool]:
    strict_by_scope = {_selection_key(item): item for item in strict_plan.selections}
    if len(strict_by_scope) != len(strict_plan.selections):
        raise ValueError("strict runtime profile plan contains duplicate layer/scope entries")
    if requested is ExecutionProfile.STRICT:
        return strict_plan.selections, False
    if requested_plan is None:
        return (
            tuple(
                VariantSelection(
                    layer=item.layer,
                    scope=item.scope,
                    selected_variant=item.selected_variant,
                    strict_fallback_variant=item.selected_variant,
                    evidence_artifact=item.evidence_artifact,
                    registry_quant=item.registry_quant,
                )
                for item in strict_plan.selections
            ),
            True,
        )

    overrides = {_selection_key(item): item for item in requested_plan.selections}
    if len(overrides) != len(requested_plan.selections):
        raise ValueError("runtime profile plan contains duplicate layer/scope entries")
    unknown = set(overrides) - set(strict_by_scope)
    if unknown:
        raise ValueError(
            f"runtime profile plan has scopes absent from strict: {sorted(unknown)!r}"
        )
    resolved: list[VariantSelection] = []
    fell_back = False
    for key, strict_selection in strict_by_scope.items():
        candidate = overrides.get(key)
        if candidate is None:
            fell_back = True
            resolved.append(
                VariantSelection(
                    layer=strict_selection.layer,
                    scope=strict_selection.scope,
                    selected_variant=strict_selection.selected_variant,
                    strict_fallback_variant=strict_selection.selected_variant,
                    evidence_artifact=strict_selection.evidence_artifact,
                    registry_quant=strict_selection.registry_quant,
                )
            )
            continue
        strict_quant = strict_selection.registry_quant
        candidate_quant = candidate.registry_quant
        if candidate_quant != strict_quant:
            raise ValueError(
                f"runtime profile scope {key!r} changes registry quant from "
                f"{strict_quant!r} to {candidate_quant!r}"
            )
        if candidate.strict_fallback_variant != strict_selection.selected_variant:
            raise ValueError(
                f"runtime profile scope {key!r} strict fallback must be "
                f"{strict_selection.selected_variant!r}"
            )
        resolved.append(candidate)
    return tuple(resolved), fell_back


def _verify_registered_variants(
    *,
    backend: str,
    model_quant: str,
    selections: Sequence[VariantSelection],
) -> None:
    from hipengine.kernels.registry import KernelKey, is_registered

    keys: list[KernelKey] = []
    for selection in selections:
        registry_quant = selection.registry_quant or model_quant
        keys.append(
            KernelKey(
                backend=backend,
                layer=selection.layer,
                quant=registry_quant,
                variant=selection.selected_variant,
            )
        )
        if selection.strict_fallback_variant is not None:
            keys.append(
                KernelKey(
                    backend=backend,
                    layer=selection.layer,
                    quant=registry_quant,
                    variant=selection.strict_fallback_variant,
                )
            )
    missing = [key for key in keys if not is_registered(key)]
    if missing:
        try:
            from hipengine.kernels.backends import load_backend_kernel_package

            load_backend_kernel_package(backend)
        except (ImportError, ValueError):
            pass
        missing = [key for key in keys if not is_registered(key)]
    if missing:
        rendered = ", ".join(key.display() for key in missing)
        raise MissingRuntimeProfilePlanError(
            f"runtime execution-profile variants are not registered: {rendered}"
        )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    return value


def resolve_runtime_profile(
    *,
    model: str,
    backend: str,
    quant: str,
    profile: ExecutionProfile | str,
) -> ResolvedRuntimeProfile:
    """Resolve one immutable plugin plan before generator/hot-path construction."""

    model_key = _nonempty(model, field="model")
    backend_key = _nonempty(backend, field="backend")
    quant_key = _nonempty(quant, field="quant")
    requested = _profile(profile)
    strict_key = RuntimeProfileKey(
        model=model_key,
        backend=backend_key,
        quant=quant_key,
        profile=ExecutionProfile.STRICT,
    )
    strict_plan = _RUNTIME_PROFILE_PLANS.get(strict_key)
    if strict_plan is None:
        raise MissingRuntimeProfilePlanError(
            "no strict execution-profile plan registered for "
            f"({model_key}, {backend_key}, {quant_key})"
        )
    strict_manifest = build_variant_manifest(
        profile=ExecutionProfile.STRICT,
        backend=backend_key,
        model=model_key,
        quant=quant_key,
        kv_policy=strict_plan.kv_policy,
        graph_policy=strict_plan.graph_policy,
        selections=strict_plan.selections,
    )
    strict_manifest_hash = manifest_sha256(strict_manifest)
    requested_plan = _RUNTIME_PROFILE_PLANS.get(
        RuntimeProfileKey(
            model=model_key,
            backend=backend_key,
            quant=quant_key,
            profile=requested,
        )
    )
    selections, fell_back = _resolved_selections(
        requested=requested,
        strict_plan=strict_plan,
        requested_plan=requested_plan,
    )
    source_plan = requested_plan or strict_plan
    _verify_registered_variants(
        backend=backend_key,
        model_quant=quant_key,
        selections=selections,
    )
    manifest = build_variant_manifest(
        profile=requested,
        backend=backend_key,
        model=model_key,
        quant=quant_key,
        kv_policy=source_plan.kv_policy,
        graph_policy=source_plan.graph_policy,
        selections=selections,
    )
    return ResolvedRuntimeProfile(
        profile=requested,
        manifest=_freeze_json(manifest),
        manifest_sha256=manifest_sha256(manifest),
        strict_manifest_sha256=strict_manifest_hash,
        factory=source_plan.factory,
        binder=source_plan.binder,
        fell_back_to_strict=fell_back,
        source_profile=(
            requested if requested_plan is not None else ExecutionProfile.STRICT
        ),
    )


def registered_runtime_profile_keys() -> tuple[RuntimeProfileKey, ...]:
    return tuple(sorted(_RUNTIME_PROFILE_PLANS))


def clear_runtime_profile_registry_for_tests() -> None:
    _RUNTIME_PROFILE_PLANS.clear()


def restore_runtime_profile_registry_for_tests(
    plans: Mapping[RuntimeProfileKey, RuntimeProfilePlan],
) -> None:
    _RUNTIME_PROFILE_PLANS.clear()
    _RUNTIME_PROFILE_PLANS.update(plans)


__all__ = [
    "EXECUTION_PROFILE_ENV",
    "EXECUTION_PROFILE_MANIFEST_KIND",
    "EXECUTION_PROFILE_SCHEMA_VERSION",
    "DuplicateRuntimeProfilePlanError",
    "ExecutionProfile",
    "MissingRuntimeProfilePlanError",
    "ResolvedRuntimeProfile",
    "RuntimeProfileKey",
    "RuntimeProfilePlan",
    "VariantSelection",
    "build_variant_manifest",
    "clear_runtime_profile_registry_for_tests",
    "manifest_sha256",
    "register_runtime_profile_plan",
    "registered_runtime_profile_keys",
    "resolve_requested_execution_profile",
    "resolve_runtime_profile",
    "restore_runtime_profile_registry_for_tests",
    "validate_variant_manifest",
]
