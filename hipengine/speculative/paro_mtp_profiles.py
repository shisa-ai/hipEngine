"""PARO MTP provider/verifier route manifests for gfx1100.

The target-contract proposer is the strict provider route: final-normalized target
hidden, selected-row reseed after verification, and borrowed full-vocabulary
W8A16 scoring. Both strict and production currently select the strict verifier.
The known fast-verifier candidate is registered as a semantic route but is not
selected by a runtime profile until its production-numerics gate passes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from hipengine.execution_profiles import (
    ExecutionProfile,
    ResolvedRuntimeProfile,
    RuntimeProfileKey,
    RuntimeProfilePlan,
    VariantSelection,
    register_runtime_profile_plan,
    registered_runtime_profile_keys,
)
from hipengine.kernels.registry import KernelKey, is_registered, register

PARO_MTP_MODEL = "qwen3_5_moe_paro"
PARO_MTP_BACKEND = "hip_gfx1100"
PARO_MTP_MODEL_QUANT = "w4_paro"
PARO_MTP_REGISTRY_QUANT = "w4_paro_mtp"
PARO_MTP_CONTRACT_ENV = "HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT"
PARO_MTP_ROUTE_ENV = "HIPENGINE_MTP_ROUTE_VARIANT"

PROPOSER_LAYER = "mtp_proposer_route"
VERIFIER_LAYER = "mtp_verifier_route"
TARGET_CONTRACT_VARIANT = "target_final_norm_selected_reseed_w8a16_full_vocab"
STRICT_VERIFIER_VARIANT = "b1_graph_off_strict_exact"
FAST_VERIFIER_CANDIDATE_VARIANT = "b1_graph_off_fast_d64_candidate"

_PROVIDER_EVIDENCE = "benchmarks/results/2026-08-24-w7900-paro-mtp-provider-contract-spike.json"
_KV_POLICY = "paged_bf16_kv_live_spans"
_GRAPH_POLICY = "off_b1_fixed_chain"


@dataclass(frozen=True, slots=True)
class ParoMtpRoute:
    component: str
    variant: str
    target_contract: bool
    verifier_profile: str
    candidate_budget: int = 1
    graph_mode: str = "off"
    draft_mode: str = "chain"
    certified: bool = True


def validate_paro_mtp_route_scope(
    *,
    candidate_budget: int,
    graph_mode: str,
    draft_mode: str,
    confidence_threshold: float = 0.0,
    draft_p_min: float = 0.0,
    ar_fallback_zero_streak: int = 0,
    overlap_verify_commit_proposer: bool = False,
) -> None:
    """Reject configurations outside the registered B1 fixed-chain route."""

    if int(candidate_budget) != 1:
        raise ValueError("PARO MTP registered route is currently B=1 only")
    if str(graph_mode) != "off":
        raise ValueError("PARO MTP registered route currently requires graph_mode=off")
    if (
        str(draft_mode) != "chain"
        or float(confidence_threshold) != 0.0
        or float(draft_p_min) != 0.0
    ):
        raise ValueError("PARO MTP registered route supports fixed chain policy only")
    if int(ar_fallback_zero_streak) != 0 or bool(overlap_verify_commit_proposer):
        raise ValueError("PARO MTP registered route does not support fallback or overlap")


def _route_factory(route: ParoMtpRoute):
    def factory() -> ParoMtpRoute:
        return route

    return factory


def _kernel_key(layer: str, variant: str) -> KernelKey:
    return KernelKey(
        backend=PARO_MTP_BACKEND,
        layer=layer,
        quant=PARO_MTP_REGISTRY_QUANT,
        variant=variant,
    )


def _register_semantic_routes() -> None:
    routes = (
        (
            PROPOSER_LAYER,
            TARGET_CONTRACT_VARIANT,
            ParoMtpRoute(
                component="proposer",
                variant=TARGET_CONTRACT_VARIANT,
                target_contract=True,
                verifier_profile="strict",
            ),
        ),
        (
            VERIFIER_LAYER,
            STRICT_VERIFIER_VARIANT,
            ParoMtpRoute(
                component="verifier",
                variant=STRICT_VERIFIER_VARIANT,
                target_contract=True,
                verifier_profile="strict",
            ),
        ),
        (
            VERIFIER_LAYER,
            FAST_VERIFIER_CANDIDATE_VARIANT,
            ParoMtpRoute(
                component="verifier",
                variant=FAST_VERIFIER_CANDIDATE_VARIANT,
                target_contract=True,
                verifier_profile="fast",
                certified=False,
            ),
        ),
    )
    for layer, variant, route in routes:
        key = _kernel_key(layer, variant)
        if not is_registered(key):
            register(key, _route_factory(route))


def _selection(layer: str, variant: str, *, evidence: str | None = None) -> VariantSelection:
    return VariantSelection(
        layer=layer,
        scope="b1_graph_off_fixed_chain",
        selected_variant=variant,
        strict_fallback_variant=(
            TARGET_CONTRACT_VARIANT if layer == PROPOSER_LAYER else STRICT_VERIFIER_VARIANT
        ),
        evidence_artifact=evidence,
        registry_quant=PARO_MTP_REGISTRY_QUANT,
    )


def _strict_selections() -> tuple[VariantSelection, ...]:
    return (
        _selection(PROPOSER_LAYER, TARGET_CONTRACT_VARIANT),
        _selection(VERIFIER_LAYER, STRICT_VERIFIER_VARIANT),
    )


def _production_selections() -> tuple[VariantSelection, ...]:
    # Provider promotion is independently useful with strict verification. The
    # fast verifier remains only a registered candidate pending its T2 gate.
    return (
        _selection(PROPOSER_LAYER, TARGET_CONTRACT_VARIANT, evidence=_PROVIDER_EVIDENCE),
        _selection(VERIFIER_LAYER, STRICT_VERIFIER_VARIANT, evidence=_PROVIDER_EVIDENCE),
    )


def _bind_route(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    del generator
    os.environ[PARO_MTP_CONTRACT_ENV] = "1"
    selected = ",".join(
        str(item["selected_variant"]) for item in resolved.manifest["selections"]
    )
    os.environ[PARO_MTP_ROUTE_ENV] = selected


def register_paro_mtp_gfx1100_profiles() -> bool:
    """Register semantic routes plus strict/production plans once."""

    _register_semantic_routes()
    existing = set(registered_runtime_profile_keys())
    changed = False
    for profile, selections in (
        (ExecutionProfile.STRICT, _strict_selections()),
        (ExecutionProfile.PRODUCTION, _production_selections()),
    ):
        key = RuntimeProfileKey(
            model=PARO_MTP_MODEL,
            backend=PARO_MTP_BACKEND,
            quant=PARO_MTP_MODEL_QUANT,
            profile=profile,
        )
        if key in existing:
            continue
        register_runtime_profile_plan(
            model=key.model,
            backend=key.backend,
            quant=key.quant,
            profile=profile,
            plan=RuntimeProfilePlan(
                selections=selections,
                kv_policy=_KV_POLICY,
                graph_policy=_GRAPH_POLICY,
                binder=_bind_route,
            ),
        )
        changed = True
    return changed


def paro_mtp_profiles_registered() -> bool:
    wanted = {
        RuntimeProfileKey(
            model=PARO_MTP_MODEL,
            backend=PARO_MTP_BACKEND,
            quant=PARO_MTP_MODEL_QUANT,
            profile=profile,
        )
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION)
    }
    return wanted <= set(registered_runtime_profile_keys())


__all__ = [
    "FAST_VERIFIER_CANDIDATE_VARIANT",
    "PARO_MTP_BACKEND",
    "PARO_MTP_CONTRACT_ENV",
    "PARO_MTP_MODEL",
    "PARO_MTP_MODEL_QUANT",
    "PARO_MTP_REGISTRY_QUANT",
    "PARO_MTP_ROUTE_ENV",
    "PROPOSER_LAYER",
    "ParoMtpRoute",
    "STRICT_VERIFIER_VARIANT",
    "TARGET_CONTRACT_VARIANT",
    "VERIFIER_LAYER",
    "paro_mtp_profiles_registered",
    "register_paro_mtp_gfx1100_profiles",
    "validate_paro_mtp_route_scope",
]
