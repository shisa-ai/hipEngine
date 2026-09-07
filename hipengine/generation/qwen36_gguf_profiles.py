"""Qwen3.6-35B-A3B GGUF gfx1151 execution-profile cold-path plans.

Registers the ``strict`` and ``production`` ``RuntimeProfilePlan`` entries for
``(qwen3_5_moe_gguf, hip_gfx1151, gguf_q4_k_m)`` so an explicit
``HIPENGINE_EXECUTION_PROFILE`` resolves to one immutable variant manifest at
LLM construction and applies the incumbent cold-path policy to the generated
runtime.

The ``strict`` profile is the fully-exact/unfused baseline: the base
(non-cooperative) F32 router, the exact Q8T16 decode selection, the strict GDN
chain, and no rowtile.

The ``production`` profile is the incumbent measured package route: the
cooperative persistent F32 router, direct c2 with the package-floor (rows >= 4)
Q8T16 rowtile policy, and the strict selection as the rollback key for every
scope. The cooperative router is a fused ``router_topk_split_shared`` kernel
that replaces the base ``router_logits`` chain, so the manifest records the
``router_logits`` family selection for both profiles and binds the fused route
through the evidence artifact and the cold-path binder.

``batch_invariant`` is intentionally NOT registered. The composition-metamorphic
gate has not passed, so ``resolve_runtime_profile(profile="batch_invariant")``
falls back to the strict plan with ``fell_back_to_strict=True``; that flag is
the clear fail-closed signal until the gate lands.

Each plan carries a ``binder`` that applies the profile cold-path policy to the
process environment at generator construction. The generator constructor does
not read these flags; the resident runner reads them lazily at decode, so the
binder (which runs at construction, before the first decode) is correct. One
profile per model-owning process, matching the campaign idle/thermal gate.
"""

from __future__ import annotations

import os
from typing import Any

from hipengine.execution_profiles import (
    ExecutionProfile,
    ResolvedRuntimeProfile,
    RuntimeProfilePlan,
    VariantSelection,
    register_runtime_profile_plan,
    registered_runtime_profile_keys,
)

# Model plugin identity (matches ``hipengine/models/qwen35.py``
# ``Qwen35MoeGGUFModel``).
QWEN36_GGUF_MODEL = "qwen3_5_moe_gguf"
QWEN36_GGUF_BACKEND = "hip_gfx1151"
QWEN36_GGUF_QUANT = "gguf_q4_k_m"

# Cold-path policy environment keys, same names the runtime and batch gate use.
ROUTER_COOP_ENV = "HIPENGINE_GGUF_ROUTER_F32W_COOP"
ROUTER_PERSISTENT_ENV = "HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER"
ROWTILE_ALL_ENV = "HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL"

# Registry-quant strings for the router metadata tensors (F32) and the
# architecture-scoped linear-attention chain (GGUF F32 weights).
_ROUTER_REGISTRY_QUANT = "f32"
_GDN_REGISTRY_QUANT = "gguf_qwen35"

# Registered gfx1151 variants for the Qwen3.6 GGUF model.
_STRICT_ROUTER_VARIANT = "f32_hidden"  # f32 | router_logits | f32_hidden
_STRICT_LINEAR_VARIANT = "pack8_full_k_grid_y_native_exact_bf16_bf16_out"
_ATTN_DECODE_VARIANT = "bf16_context_batch_shared_native_exact_spans"
_GDN_CHAIN_VARIANT = "bf16_c1_exact_state_rows_tloop"

# Retained package evidence bound to the production router and rowtile policy.
_ROUTER_EVIDENCE = "benchmarks/results/2026-08-16-zbook-qwen36-c1-router-retained.json"
_Q8T16_ROWTILE_EVIDENCE = (
    "benchmarks/results/2026-08-16-gfx1151-q8t16-batch-route-retained.json"
)

_KV_POLICY = "paged_bf16"
_GRAPH_POLICY = "serial_c1"


def _apply_profile_policy(*, cooperative: bool, rowtile_floor: bool) -> None:
    """Apply the documented cold-path policy to the process environment.

    ``production`` enables the cooperative persistent F32 router and leaves the
    rowtile selection on the backend package floor (rows >= 4). ``strict``
    disables the cooperative router and forces rowtile off.
    """

    os.environ[ROUTER_COOP_ENV] = "1" if cooperative else "0"
    os.environ[ROUTER_PERSISTENT_ENV] = "1" if cooperative else "0"
    if rowtile_floor:
        os.environ.pop(ROWTILE_ALL_ENV, None)
    else:
        os.environ[ROWTILE_ALL_ENV] = "0"


def _selection(
    *,
    layer: str,
    scope: str,
    selected_variant: str,
    strict_fallback_variant: str,
    registry_quant: str | None = None,
    evidence_artifact: str | None = None,
) -> VariantSelection:
    return VariantSelection(
        layer=layer,
        scope=scope,
        selected_variant=selected_variant,
        strict_fallback_variant=strict_fallback_variant,
        evidence_artifact=evidence_artifact,
        registry_quant=registry_quant,
    )


def _strict_selections() -> tuple[VariantSelection, ...]:
    """Fully-exact/unfused baseline: base F32 router, exact decode, no rowtile."""

    return (
        _selection(
            layer="router_logits",
            scope="c1_router",
            selected_variant=_STRICT_ROUTER_VARIANT,
            strict_fallback_variant=_STRICT_ROUTER_VARIANT,
            registry_quant=_ROUTER_REGISTRY_QUANT,
        ),
        _selection(
            layer="linear",
            scope="decode",
            selected_variant=_STRICT_LINEAR_VARIANT,
            strict_fallback_variant=_STRICT_LINEAR_VARIANT,
            registry_quant=QWEN36_GGUF_QUANT,
        ),
        _selection(
            layer="paged_attn_decode",
            scope="decode",
            selected_variant=_ATTN_DECODE_VARIANT,
            strict_fallback_variant=_ATTN_DECODE_VARIANT,
            registry_quant=QWEN36_GGUF_QUANT,
        ),
        _selection(
            layer="linear_attn_chain_conv_decode",
            scope="gdn_chain",
            selected_variant=_GDN_CHAIN_VARIANT,
            strict_fallback_variant=_GDN_CHAIN_VARIANT,
            registry_quant=_GDN_REGISTRY_QUANT,
        ),
    )


def _production_selections() -> tuple[VariantSelection, ...]:
    """Incumbent package route with a strict rollback key for every scope."""

    strict = {_selection_key(item): item for item in _strict_selections()}
    router_strict = strict[("router_logits", "c1_router")]
    linear_strict = strict[("linear", "decode")]
    attn_strict = strict[("paged_attn_decode", "decode")]
    gdn_strict = strict[("linear_attn_chain_conv_decode", "gdn_chain")]
    return (
        _selection(
            layer=router_strict.layer,
            scope=router_strict.scope,
            selected_variant=router_strict.selected_variant,
            strict_fallback_variant=router_strict.selected_variant,
            registry_quant=router_strict.registry_quant,
            evidence_artifact=_ROUTER_EVIDENCE,
        ),
        _selection(
            layer=linear_strict.layer,
            scope=linear_strict.scope,
            selected_variant=linear_strict.selected_variant,
            strict_fallback_variant=linear_strict.selected_variant,
            registry_quant=linear_strict.registry_quant,
            evidence_artifact=_Q8T16_ROWTILE_EVIDENCE,
        ),
        _selection(
            layer=attn_strict.layer,
            scope=attn_strict.scope,
            selected_variant=attn_strict.selected_variant,
            strict_fallback_variant=attn_strict.selected_variant,
            registry_quant=attn_strict.registry_quant,
        ),
        _selection(
            layer=gdn_strict.layer,
            scope=gdn_strict.scope,
            selected_variant=gdn_strict.selected_variant,
            strict_fallback_variant=gdn_strict.selected_variant,
            registry_quant=gdn_strict.registry_quant,
        ),
    )


def _selection_key(selection: VariantSelection) -> tuple[str, str]:
    return (selection.layer, selection.scope)


def _binder(
    generator: Any,
    resolved: ResolvedRuntimeProfile,
    *,
    cooperative: bool,
    rowtile_floor: bool,
) -> None:
    """Apply the profile cold-path policy before any decode runs."""

    del generator, resolved
    _apply_profile_policy(cooperative=cooperative, rowtile_floor=rowtile_floor)


def _strict_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    _binder(generator, resolved, cooperative=False, rowtile_floor=False)


def _production_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    _binder(generator, resolved, cooperative=True, rowtile_floor=True)


def _registered_key() -> tuple[str, str, str]:
    return (QWEN36_GGUF_MODEL, QWEN36_GGUF_BACKEND, QWEN36_GGUF_QUANT)


def _already_registered() -> bool:
    from hipengine.execution_profiles import RuntimeProfileKey

    model, backend, quant = _registered_key()
    return any(
        key.model == model and key.backend == backend and key.quant == quant
        for key in registered_runtime_profile_keys()
    )


def register_qwen36_gguf_gfx1151_profiles() -> bool:
    """Register the strict and production plans once; idempotent.

    ``batch_invariant`` is deliberately left unregistered (fail-closed fallback
    to strict). Registration does not load kernels or touch the GPU; the plans
    are cold-path provenance plus a construction-time policy binder.
    """

    if _already_registered():
        return False
    register_runtime_profile_plan(
        model=QWEN36_GGUF_MODEL,
        backend=QWEN36_GGUF_BACKEND,
        quant=QWEN36_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
        plan=RuntimeProfilePlan(
            selections=_strict_selections(),
            kv_policy=_KV_POLICY,
            graph_policy=_GRAPH_POLICY,
            binder=_strict_binder,
        ),
    )
    register_runtime_profile_plan(
        model=QWEN36_GGUF_MODEL,
        backend=QWEN36_GGUF_BACKEND,
        quant=QWEN36_GGUF_QUANT,
        profile=ExecutionProfile.PRODUCTION,
        plan=RuntimeProfilePlan(
            selections=_production_selections(),
            kv_policy=_KV_POLICY,
            graph_policy=_GRAPH_POLICY,
            binder=_production_binder,
        ),
    )
    return True


def qwen36_gguf_gfx1151_plans_registered() -> bool:
    """Return whether the strict and production plans are registered."""

    from hipengine.execution_profiles import RuntimeProfileKey

    model, backend, quant = _registered_key()
    wanted = {
        RuntimeProfileKey(
            model=model,
            backend=backend,
            quant=quant,
            profile=profile,
        )
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION)
    }
    return wanted <= set(registered_runtime_profile_keys())


__all__ = [
    "QWEN36_GGUF_BACKEND",
    "QWEN36_GGUF_MODEL",
    "QWEN36_GGUF_QUANT",
    "ROUTER_COOP_ENV",
    "ROUTER_PERSISTENT_ENV",
    "ROWTILE_ALL_ENV",
    "qwen36_gguf_gfx1151_plans_registered",
    "register_qwen36_gguf_gfx1151_profiles",
]
