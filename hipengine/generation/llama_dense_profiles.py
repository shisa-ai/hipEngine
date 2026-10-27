"""Dense Llama strict execution-profile cold-path plans.

Registers one ``strict`` :class:`~hipengine.execution_profiles.RuntimeProfilePlan`
per gfx11 backend for ``(llama_dense, <backend>, bf16)``, so
``LLM(..., execution_profile="strict")`` resolves to the exact block attention
instead of raising ``MissingRuntimeProfilePlanError`` before a generator is
constructed.

``production`` is deliberately not registered. The dense path's default is the
tiled flash attention, and an omitted profile keeps taking that default through
the pre-profile migration path: ``resolve_default_execution_profile`` returns
``None`` unless a combination has both a strict and a production plan. A
production plan here would flip omitted-profile selection to ``production`` and
claim a certified production route this path does not have.

The strict plan names what the generator actually selects under
``execution_profile="strict"``: both the prompt pass and the decode step call
``dense_causal_prefill_bf16(..., strict=True)``, which is the exact block
attention registered as ``attention_prefill``/``bf16``/``exact_causal``. The
tiled ``flash_tiled`` variant stays the omitted-profile default. The
plan's factory passes ``execution_profile="strict"`` to the backend's registered
generator factory, so the generator is constructed strict rather than patched
after construction.

``batch_invariant`` and ``fast`` are not registered. Naming either resolves to
the strict plan with ``fell_back_to_strict=True``, which is the fail-closed
behavior those profiles document for a combination with no plan of their own.

Kernel registrations: ``resolve_runtime_profile`` refuses a plan that names a
variant the registry does not hold. No backend package calls
``register_dense_prefill_attention_kernels`` at import, so this module registers
its two keys under ``hip_gfx1100`` and mirrors them under ``hip_gfx1151`` -- the
same launchers the runtime calls on either backend. The durable home for those
keys is the backend package; until they land there, this is the registration the
plan reads.
"""

from __future__ import annotations

from typing import Any, Callable

from hipengine.execution_profiles import (
    ExecutionProfile,
    RuntimeProfileKey,
    RuntimeProfilePlan,
    VariantSelection,
    register_runtime_profile_plan,
    registered_runtime_profile_keys,
)
from hipengine.generation.llama_dense import (
    make_llama_dense_generator_gfx1100,
    make_llama_dense_generator_gfx1151,
)

# Model plugin identity (``hipengine/models/llama.py``: name = "llama_dense",
# default_quant = "bf16").
LLAMA_DENSE_MODEL = "llama_dense"
LLAMA_DENSE_QUANT = "bf16"
LLAMA_DENSE_BACKENDS = ("hip_gfx1100", "hip_gfx1151")

# The registered dense attention layer and the two variants it carries:
# ``flash_tiled`` is the default route, ``exact_causal`` is the strict fallback.
# Both passes -- prompt and one-token decode -- launch through this layer.
ATTENTION_LAYER = "attention_prefill"
ATTENTION_SCOPE = "causal_prefill_and_decode"
ATTENTION_TILED_VARIANT = "flash_tiled"
ATTENTION_STRICT_VARIANT = "exact_causal"

# One resident request, a contiguous BF16 K/V cache per layer, eager launches.
KV_POLICY = "resident_bf16"
GRAPH_POLICY = "serial_eager"


def strict_selections() -> tuple[VariantSelection, ...]:
    """The strict plan's selections: the exact kernel, its own rollback key."""

    return (
        VariantSelection(
            layer=ATTENTION_LAYER,
            scope=ATTENTION_SCOPE,
            selected_variant=ATTENTION_STRICT_VARIANT,
            strict_fallback_variant=ATTENTION_STRICT_VARIANT,
        ),
    )


def make_llama_dense_generator_gfx1100_strict(**factory_kwargs: Any) -> Any:
    """Construct the gfx1100 generator with strict arithmetic from the start."""

    return make_llama_dense_generator_gfx1100(
        **{**factory_kwargs, "execution_profile": ExecutionProfile.STRICT.value}
    )


def make_llama_dense_generator_gfx1151_strict(**factory_kwargs: Any) -> Any:
    """Construct the gfx1151 generator with strict arithmetic from the start."""

    return make_llama_dense_generator_gfx1151(
        **{**factory_kwargs, "execution_profile": ExecutionProfile.STRICT.value}
    )


_STRICT_FACTORY_OVERRIDES: dict[str, Callable[..., Any]] = {
    "hip_gfx1100": make_llama_dense_generator_gfx1100_strict,
    "hip_gfx1151": make_llama_dense_generator_gfx1151_strict,
}


def _register_dense_attention_variants() -> None:
    """Register the dense attention keys the plan names, on both backends.

    ``register_dense_prefill_attention_kernels`` registers the gfx1100 pair with
    the launchers the runtime calls. The gfx1151 peer runs the same launchers
    (the runtime imports them from the gfx1100 tree on either backend), so the
    mirror registers the same callables under its own backend key. Guarded per
    key because the registries are reset between tests.
    """

    from hipengine.kernels.hip_gfx1100.attention.dense_prefill_attention import (
        dense_causal_prefill_bf16,
        register_dense_prefill_attention_kernels,
    )
    from hipengine.kernels.registry import KernelKey, is_registered, register

    source_keys = tuple(
        KernelKey("hip_gfx1100", ATTENTION_LAYER, LLAMA_DENSE_QUANT, variant)
        for variant in (ATTENTION_TILED_VARIANT, ATTENTION_STRICT_VARIANT)
    )
    # The kernel module's own registrar replaces its keys, so call it only when
    # a key is missing; a repeated cold-path call must not churn the registry.
    if not all(is_registered(key) for key in source_keys):
        register_dense_prefill_attention_kernels()

    def exact_causal(*args: Any, **kwargs: Any) -> None:
        dense_causal_prefill_bf16(*args, **kwargs, strict=True)

    for variant, fn in (
        (ATTENTION_TILED_VARIANT, dense_causal_prefill_bf16),
        (ATTENTION_STRICT_VARIANT, exact_causal),
    ):
        key = KernelKey("hip_gfx1151", ATTENTION_LAYER, LLAMA_DENSE_QUANT, variant)
        if not is_registered(key):
            register(key, fn)


def register_llama_dense_profiles() -> bool:
    """Register the strict plans once; idempotent.

    Registration builds and launches nothing and touches no GPU: the plans are
    cold-path provenance plus a construction-time factory override.
    """

    _register_dense_attention_variants()
    wanted = {
        RuntimeProfileKey(
            model=LLAMA_DENSE_MODEL,
            backend=backend,
            quant=LLAMA_DENSE_QUANT,
            profile=ExecutionProfile.STRICT,
        )
        for backend in LLAMA_DENSE_BACKENDS
    }
    existing = set(registered_runtime_profile_keys())
    if wanted <= existing:
        return False
    if existing & wanted:
        raise RuntimeError("dense Llama strict profile registry is partially populated")
    for backend in LLAMA_DENSE_BACKENDS:
        register_runtime_profile_plan(
            model=LLAMA_DENSE_MODEL,
            backend=backend,
            quant=LLAMA_DENSE_QUANT,
            profile=ExecutionProfile.STRICT,
            plan=RuntimeProfilePlan(
                selections=strict_selections(),
                kv_policy=KV_POLICY,
                graph_policy=GRAPH_POLICY,
                factory=_STRICT_FACTORY_OVERRIDES[backend],
            ),
        )
    return True


def llama_dense_profiles_registered() -> bool:
    """Return whether both backends' strict plans are registered."""

    wanted = {
        RuntimeProfileKey(
            model=LLAMA_DENSE_MODEL,
            backend=backend,
            quant=LLAMA_DENSE_QUANT,
            profile=ExecutionProfile.STRICT,
        )
        for backend in LLAMA_DENSE_BACKENDS
    }
    return wanted <= set(registered_runtime_profile_keys())


__all__ = [
    "ATTENTION_LAYER",
    "ATTENTION_SCOPE",
    "ATTENTION_STRICT_VARIANT",
    "ATTENTION_TILED_VARIANT",
    "GRAPH_POLICY",
    "KV_POLICY",
    "LLAMA_DENSE_BACKENDS",
    "LLAMA_DENSE_MODEL",
    "LLAMA_DENSE_QUANT",
    "llama_dense_profiles_registered",
    "make_llama_dense_generator_gfx1100_strict",
    "make_llama_dense_generator_gfx1151_strict",
    "register_llama_dense_profiles",
    "strict_selections",
]
