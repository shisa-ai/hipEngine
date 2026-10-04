"""Gemma 4 GGUF execution-profile plans.

Production requests ``gemma4_staged``, an FP32 score/softmax/PV implementation
that preserves the strict reduction order while moving scores out of LDS.
Strict requests ``gemma4_plain``. Both plans declare sliding (head_dim 256) and
full (head_dim 512) geometries. A layer matches capabilities against its own
geometry; model identities and measured-artifact lists do not gate execution.

The BF16 WMMA variants remain registered for explicit candidate evaluation.
Their saved teacher-chain KL failures are recorded in the staged repair
worklog; they are not the production default.

Plans and variants are registered in the authoring ``hip_gfx1100`` key space
and the ``hip_gfx1151`` alias used by the Strix Halo generator. Registering the
strict and production plans makes ``production`` the unset-profile default.
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
# One declared scope per attention geometry. The layer matches each request
# against its own head dimensions.
PREFILL_ATTENTION_SCOPE = "sliding_head_dim_256"
PREFILL_ATTENTION_SCOPE_FULL = "full_head_dim_512"
PREFILL_ATTENTION_PLAIN = "gemma4_plain"
PREFILL_ATTENTION_STAGED = "gemma4_staged"
PREFILL_ATTENTION_WMMA_FLASH = "gemma4_wmma_flash"
PREFILL_ATTENTION_WMMA_FLASH_FULL = "gemma4_wmma_flash_full"

# Historical WMMA candidate evidence for explicit evaluation; the staged
# production selections below do not claim these candidates' performance.
PREFILL_ATTENTION_EVIDENCE = (
    "benchmarks/results/2026-09-29-gemma4-gfx1151-prefill-attention-wmma-candidate.json"
)
PREFILL_ATTENTION_EVIDENCE_FULL = (
    "benchmarks/results/2026-09-29-gemma4-gfx1151-prefill-attention-wmma-full-candidate.json"
)

# Both production geometry scopes request the same strict-parity launcher.
# WMMA candidates remain registered, but saved teacher-chain KL failures
# prevent using them as the shipped arithmetic choice.
PREFILL_ATTENTION_PRODUCTION_VARIANTS = (PREFILL_ATTENTION_STAGED,)

KV_POLICY = "paged_bf16"
GRAPH_POLICY = "eager_blocks"

__all__ = [
    "GEMMA4_GGUF_BACKEND",
    "GEMMA4_GGUF_MODEL",
    "GEMMA4_GGUF_QUANT",
    "GEMMA4_GGUF_SOURCE_BACKEND",
    "PREFILL_ATTENTION_EVIDENCE",
    "PREFILL_ATTENTION_EVIDENCE_FULL",
    "PREFILL_ATTENTION_LAYER",
    "PREFILL_ATTENTION_PLAIN",
    "PREFILL_ATTENTION_STAGED",
    "PREFILL_ATTENTION_PRODUCTION_VARIANTS",
    "PREFILL_ATTENTION_SCOPE",
    "PREFILL_ATTENTION_SCOPE_FULL",
    "PREFILL_ATTENTION_WMMA_FLASH",
    "PREFILL_ATTENTION_WMMA_FLASH_FULL",
    "gemma4_gguf_profiles_registered",
    "register_gemma4_gguf_profiles",
    "resolve_gemma4_prefill_attention_variants",
]


def _selection(*, selected: str, fallback: str, quant: str, evidence: str | None,
               scope: str = PREFILL_ATTENTION_SCOPE) -> VariantSelection:
    return VariantSelection(
        layer=PREFILL_ATTENTION_LAYER,
        scope=scope,
        selected_variant=selected,
        strict_fallback_variant=fallback,
        registry_quant=quant,
        evidence_artifact=evidence,
    )


def _selections(*, production: bool) -> tuple[VariantSelection, ...]:
    """One selection per attention geometry, strict and production alike.

    Both plans carry both scopes: a profile that named only the sliding scope
    would leave the full layers with no stated intent, and the strict plan still
    has to say which geometry each strict row is the fallback for.
    """

    return tuple(
        _selection(
            selected=PREFILL_ATTENTION_STAGED if production else PREFILL_ATTENTION_PLAIN,
            fallback=PREFILL_ATTENTION_PLAIN,
            quant=GEMMA4_GGUF_QUANT,
            evidence=None,
            scope=scope,
        )
        for scope in (PREFILL_ATTENTION_SCOPE, PREFILL_ATTENTION_SCOPE_FULL)
    )


def _binder(generator: Any, resolved: Any) -> None:
    """Bind the resolved prefill-attention variants onto a generator.

    The request is the ordered set of variants the plan selected for this layer,
    deduplicated. A strict plan binds plain attention and a production plan
    binds staged attention. Each layer resolves capabilities for its geometry.
    """

    variants = tuple(
        sorted(
            {
                str(selection["selected_variant"])
                for selection in resolved.manifest["selections"]
                if selection["layer"] == PREFILL_ATTENTION_LAYER
            }
        )
    )
    if variants:
        generator.prefill_attention_variants = variants


def _register_attention_variants() -> None:
    """Register plain, staged and explicit WMMA candidates under both backends.

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
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma_full import (
        gemma4_attention_prefill_wmma_full_bf16,
        register_gemma4_attention_prefill_wmma_full_kernels,
    )

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_staged import (
        gemma4_attention_staged_bf16,
    )

    register_gemma4_attention_kernels()
    register_gemma4_attention_prefill_wmma_kernels()
    register_gemma4_attention_prefill_wmma_full_kernels()

    for backend in (GEMMA4_GGUF_SOURCE_BACKEND, GEMMA4_GGUF_BACKEND):
        for quant in PREFILL_ATTENTION_QUANTS:
            for variant, fn in (
                (_PLAIN, gemma4_attention_prefill_bf16),
                (PREFILL_ATTENTION_STAGED, gemma4_attention_staged_bf16),
                (PREFILL_ATTENTION_WMMA_FLASH, gemma4_attention_prefill_wmma_bf16),
                (
                    PREFILL_ATTENTION_WMMA_FLASH_FULL,
                    gemma4_attention_prefill_wmma_full_bf16,
                ),
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


def resolve_gemma4_prefill_attention_variants(
    *,
    backend: str = GEMMA4_GGUF_BACKEND,
    requested_profile: ExecutionProfile | str | None = None,
    environ: Any = None,
) -> tuple[str, ...]:
    """Return the prefill-attention variants this run requests.

    An empty tuple means the strict kernel, which is what an unregistered
    combination, an absent plan, or ``HIPENGINE_EXECUTION_PROFILE=strict`` all
    resolve to. Each variant is a *request*: the layer still matches it against
    its own head geometry and keeps the strict kernel on a capability miss.
    """

    register_gemma4_gguf_profiles()
    if backend != GEMMA4_GGUF_BACKEND:
        # The plan is registered for the Strix Halo alias only; the gfx1100
        # generator resolves its own combination and gets the strict kernel
        # rather than a plan written for a different backend.
        return ()
    profile = resolve_requested_execution_profile(requested_profile, environ=environ)
    if profile is None:
        profile = resolve_default_execution_profile(
            model=GEMMA4_GGUF_MODEL, backend=backend, quant=GEMMA4_GGUF_QUANT
        )
    if profile is None:
        return ()
    try:
        resolved = resolve_runtime_profile(
            model=GEMMA4_GGUF_MODEL,
            backend=backend,
            quant=GEMMA4_GGUF_QUANT,
            profile=profile,
        )
    except MissingRuntimeProfilePlanError:
        return ()
    return tuple(
        sorted(
            {
                str(selection["selected_variant"])
                for selection in resolved.manifest["selections"]
                if selection["layer"] == PREFILL_ATTENTION_LAYER
            }
        )
    )
