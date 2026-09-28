"""Gemma 4 GGUF execution-profile plans.

Gemma 4's attention family has two prefill entry points. ``gemma4_plain`` is the
strict kernel and is the only unconditional one. ``gemma4_wmma_flash`` is a BF16
WMMA flash prefill that implements the sliding geometry (head_dim 256, 16 query
heads, 8 KV heads) and nothing else. This module is where a profile chooses
between them.

The plans are registered for both the ``hip_gfx1100`` key space, where the two
kernels are authored, and the ``hip_gfx1151`` alias, where the Strix Halo
generator runs. The alias is the package's own convention: ``hip_gfx1151``
re-registers the gfx1100 key space under its own backend name, so the two
attention variants are registered first and then mirrored by calling that
package's ``register_gfx1151_kernels`` again.

Registering both a strict and a production plan is what makes the default
profile ``production`` (see ``resolve_default_execution_profile``), so the WMMA
variant is on unless a caller asks for ``strict`` or sets
``HIPENGINE_EXECUTION_PROFILE``.
"""

from __future__ import annotations

from typing import Any

from hipengine.execution_profiles import (
    ExecutionProfile,
    MissingRuntimeProfilePlanError,
    RuntimeProfilePlan,
    RuntimeProfileKey,
    VariantSelection,
    register_runtime_profile_plan,
    registered_runtime_profile_keys,
    resolve_default_execution_profile,
    resolve_requested_execution_profile,
    resolve_runtime_profile,
)

GEMMA4_GGUF_MODEL = "gemma4_gguf"
GEMMA4_GGUF_BACKEND = "hip_gfx1151"
GEMMA4_GGUF_QUANT = "gguf_q4_k_m"
GEMMA4_GGUF_SOURCE_BACKEND = "hip_gfx1100"

PREFILL_ATTENTION_LAYER = "prefill_attention"
PREFILL_ATTENTION_SCOPE = "sliding_head_dim_256"
PREFILL_ATTENTION_PLAIN = "gemma4_plain"
PREFILL_ATTENTION_WMMA_FLASH = "gemma4_wmma_flash"

# The candidate's measured evidence. A selection names it so the profile's
# performance claim and the artifact that carries it cannot drift apart.
PREFILL_ATTENTION_EVIDENCE = (
    "benchmarks/results/2026-09-29-gemma4-gfx1151-prefill-attention-wmma-candidate.json"
)

KV_POLICY = "paged_bf16"
GRAPH_POLICY = "eager_blocks"

__all__ = [
    "GEMMA4_GGUF_BACKEND",
    "GEMMA4_GGUF_MODEL",
    "GEMMA4_GGUF_QUANT",
    "GEMMA4_GGUF_SOURCE_BACKEND",
    "PREFILL_ATTENTION_EVIDENCE",
    "PREFILL_ATTENTION_LAYER",
    "PREFILL_ATTENTION_PLAIN",
    "PREFILL_ATTENTION_SCOPE",
    "PREFILL_ATTENTION_WMMA_FLASH",
    "gemma4_gguf_profiles_registered",
    "register_gemma4_gguf_profiles",
    "resolve_gemma4_prefill_attention_variant",
]


def _selection(*, selected: str, fallback: str, quant: str, evidence: str | None) -> VariantSelection:
    return VariantSelection(
        layer=PREFILL_ATTENTION_LAYER,
        scope=PREFILL_ATTENTION_SCOPE,
        selected_variant=selected,
        strict_fallback_variant=fallback,
        registry_quant=quant,
        evidence_artifact=evidence,
    )


def _selections(*, production: bool) -> tuple[VariantSelection, ...]:
    if production:
        return (
            _selection(
                selected=PREFILL_ATTENTION_WMMA_FLASH,
                fallback=PREFILL_ATTENTION_PLAIN,
                quant=GEMMA4_GGUF_QUANT,
                evidence=PREFILL_ATTENTION_EVIDENCE,
            ),
        )
    return (
        _selection(
            selected=PREFILL_ATTENTION_PLAIN,
            fallback=PREFILL_ATTENTION_PLAIN,
            quant=GEMMA4_GGUF_QUANT,
            evidence=None,
        ),
    )


def _binder(generator: Any, resolved: Any) -> None:
    """Bind the resolved prefill-attention variant onto a generator."""

    for selection in resolved.manifest["selections"]:
        if (
            selection["layer"] == PREFILL_ATTENTION_LAYER
            and selection["scope"] == PREFILL_ATTENTION_SCOPE
        ):
            generator.prefill_attention_variant = str(selection["selected_variant"])
            return


def _register_attention_variants() -> None:
    """Put both prefill-attention variants in the registry, under both backends.

    ``_verify_registered_variants`` refuses a plan that names a variant the
    registry does not hold, so this runs before the plans are registered. The
    gfx1151 mirror is the same call the alias package makes at its own import,
    repeated here because the two gemma4 variants are registered after that
    import has already run.
    """

    from hipengine.kernels.registry import KernelKey, is_registered, register

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        PREFILL_ATTENTION_PLAIN as _PLAIN,
        PREFILL_ATTENTION_QUANTS,
        gemma4_attention_prefill_bf16,
        register_gemma4_attention_kernels,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma import (
        gemma4_attention_prefill_wmma_bf16,
        register_gemma4_attention_prefill_wmma_kernels,
    )

    register_gemma4_attention_kernels()
    register_gemma4_attention_prefill_wmma_kernels()

    for backend in (GEMMA4_GGUF_SOURCE_BACKEND, GEMMA4_GGUF_BACKEND):
        for quant in PREFILL_ATTENTION_QUANTS:
            for variant, fn in (
                (_PLAIN, gemma4_attention_prefill_bf16),
                (PREFILL_ATTENTION_WMMA_FLASH, gemma4_attention_prefill_wmma_bf16),
            ):
                key = KernelKey(backend, PREFILL_ATTENTION_LAYER, quant, variant)
                if not is_registered(key):
                    register(key, fn)


def register_gemma4_gguf_profiles() -> bool:
    """Register the strict and production Gemma 4 plans once. Idempotent."""

    wanted = {
        RuntimeProfileKey(
            model=GEMMA4_GGUF_MODEL,
            backend=GEMMA4_GGUF_BACKEND,
            quant=GEMMA4_GGUF_QUANT,
            profile=profile,
        )
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION)
    }
    existing = set(registered_runtime_profile_keys())
    if wanted <= existing:
        return False
    if existing & wanted:
        raise RuntimeError("Gemma 4 profile registry is partially populated")

    _register_attention_variants()
    for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION):
        register_runtime_profile_plan(
            model=GEMMA4_GGUF_MODEL,
            backend=GEMMA4_GGUF_BACKEND,
            quant=GEMMA4_GGUF_QUANT,
            profile=profile,
            plan=RuntimeProfilePlan(
                selections=_selections(production=profile is ExecutionProfile.PRODUCTION),
                kv_policy=KV_POLICY,
                graph_policy=GRAPH_POLICY,
                binder=_binder,
            ),
        )
    return True


def gemma4_gguf_profiles_registered() -> bool:
    wanted = {
        RuntimeProfileKey(
            model=GEMMA4_GGUF_MODEL,
            backend=GEMMA4_GGUF_BACKEND,
            quant=GEMMA4_GGUF_QUANT,
            profile=profile,
        )
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION)
    }
    return wanted <= set(registered_runtime_profile_keys())


def resolve_gemma4_prefill_attention_variant(
    *,
    backend: str = GEMMA4_GGUF_BACKEND,
    requested_profile: ExecutionProfile | str | None = None,
    environ: Any = None,
) -> str | None:
    """Return the prefill-attention variant this run should request.

    ``None`` means the strict kernel, which is what an unregistered
    combination, an absent plan, or ``HIPENGINE_EXECUTION_PROFILE=strict`` all
    resolve to. The variant is a *request*: the layer still matches it against
    its own head geometry and keeps the strict kernel on a capability miss.
    """

    register_gemma4_gguf_profiles()
    if backend != GEMMA4_GGUF_BACKEND:
        # The plan is registered for the Strix Halo alias only; the gfx1100
        # generator resolves its own combination and gets the strict kernel
        # rather than a plan written for a different backend.
        return None
    profile = resolve_requested_execution_profile(requested_profile, environ=environ)
    if profile is None:
        profile = resolve_default_execution_profile(
            model=GEMMA4_GGUF_MODEL, backend=backend, quant=GEMMA4_GGUF_QUANT
        )
    if profile is None:
        return None
    try:
        resolved = resolve_runtime_profile(
            model=GEMMA4_GGUF_MODEL,
            backend=backend,
            quant=GEMMA4_GGUF_QUANT,
            profile=profile,
        )
    except MissingRuntimeProfilePlanError:
        return None
    for selection in resolved.manifest["selections"]:
        if (
            selection["layer"] == PREFILL_ATTENTION_LAYER
            and selection["scope"] == PREFILL_ATTENTION_SCOPE
        ):
            return str(selection["selected_variant"])
    return None
