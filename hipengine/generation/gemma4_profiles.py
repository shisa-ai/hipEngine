"""Gemma 4 GGUF gfx1100 execution-profile cold-path plans.

Registers the ``strict`` and ``production`` ``RuntimeProfilePlan`` entries for
``(gemma4_gguf, hip_gfx1100, gguf_q4_k_m)`` -- the combination
``llm.resolved_quant`` actually supplies to ``resolve_runtime_profile`` on the
shipped UD-Q4_K_XL artifact (measured, not inferred from the filename) -- so
that an explicit profile resolves to one immutable variant manifest at LLM
construction and applies its route policy before the first forward.

``production`` selects fused-stride T16 WMMA for wide Q4 gate/up prefill,
with narrow calls retaining their selected decode route. Other quant layouts
keep grouped int8 gate/up and WMMA down owners. Dense prefill uses
``wmma_prefill`` and decode uses pack8/selected GEMVs. The binder pins
``HIPENGINE_GEMMA4_MOE_PREFILL=auto``; the manifest names the wide Q4 owner
and its strict selected-row fallback.

``strict`` pins ``HIPENGINE_GEMMA4_MOE_PREFILL=grouped``, the exact arm
``gemma4_experts.py::_prefill_route_flags`` documents as the rollback lever,
measured with the same probe: the MMQ owner disappears and the grouped exact
owners take over. Four scopes do not have a reachable exact arm yet, so they
carry today's variant with the blocker recorded rather than an exact variant
the tree cannot execute:

- ``dense_prefill_q8_0``: the exact owner ``prefill_bf16_bf16_out`` exists but
  ``gemma4_layer.py`` passes ``use_wmma_prefill=True`` as an explicit kwarg,
  which outranks the session toggle and the env var, so no profile can reach
  it without that call site taking profile policy (curation entry
  ``...gemma4-strict-curation-6312b4.md``, blocker B1).
- ``moe_decode_q4_k`` / ``moe_decode_q5_1`` / ``moe_decode_q5_k``: no route pin
  exists for the decode GEMVs (B2).
- ``lm_head_f32_q8_0``: its call counts are data-dependent, so no phase or arm
  difference was measurable (B4).
- ``down_q8_0`` (revised B3): the grouped flow's q8_0 chain is three *probe*
  attempts (fold128 -> amortized -> rowbatch8, first resolvable wins) and all
  three are unregistered, so under ``strict`` the q8_0 down projection falls
  through to the selected owner under the **``linear``** layer key while
  production runs ``grouped_wmma`` under **``moe_linear``**. One logical scope
  cannot span two registry layers, so the scope is omitted from both plans
  until either a grouped q8_0 owner registers or the axis can express it; the
  observed 3x3 resolve calls were failed probes, not three owners. The
  fallback that does run is declared by ``linear/moe_selected_q8_0``.

``batch_invariant`` is deliberately not registered: the composition gate has
not run for this model, so it falls back to strict fail-closed.

Probe method and full tables: worklog entries
``...gemma4-strict-curation-6312b4.md`` and
``...gemma4-dispatch-attribution-d6c87d.md``.
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

# Model plugin identity (``hipengine/models/gemma4.py``: name = "gemma4_gguf",
# default_quant = "gguf_q4_k_m"); backend is the resolved gfx1100 tree.
GEMMA4_GGUF_MODEL = "gemma4_gguf"
GEMMA4_GGUF_BACKEND = "hip_gfx1100"
GEMMA4_GGUF_QUANT = "gguf_q4_k_m"

# Route pin read per prefill call by gemma4_experts._prefill_mode().
MOE_PREFILL_ENV = "HIPENGINE_GEMMA4_MOE_PREFILL"
_PRODUCTION_ROUTE = "auto"
_STRICT_ROUTE = "grouped"

_KV_POLICY = "paged_bf16"
_GRAPH_POLICY = "serial_eager"

# Registry-quant strings the rows resolve under.
_Q8_0 = "gguf_q8_0"
_Q5_1 = "gguf_q5_1"
_Q5_K = "gguf_q5_k"
# The fused gate_up stacks convert to the T16 tiles layout at load, so every
# Q4_K gate_up scope (prefill and decode) dispatches under the layout's own
# registry key -- worklog 20260930T014202-land-the-t16-gate-up-repack-
# dispatch-wiring-0bf1f8.
_Q4_K_T16 = "gguf_q4_k_t16_v1"

# Production (measured, no route pin) MoE owners.
_T16_FUSED_WMMA = "selected_dual_wmma_prefill_fused_bf16_bf16_out"
_MMQ32 = "selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out"
_GROUPED_DUAL = "selected_dual_grouped_rowbatch8_out4_amortized_bf16_bf16_out"
_WMMA_DOWN = "selected_grouped_wmma_prefill_compact_bf16_bf16_out"
# The T16-converted gate_up scope's exact owner (registered at moe_linear;
# the dense-key ``selected_gemv`` chain lands on the same function).
_T16_DECODE = "selected_t16_gemv_decode_bf16_bf16_out"

# Strict (measured under HIPENGINE_GEMMA4_MOE_PREFILL=grouped) MoE owners.
_GROUPED_DOWN_Q5_1 = "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"

# Dense / embedding owners, arm-invariant (identical in both probe arms).
_LOOKUP = "lookup_bf16_out"
_WMMA_PREFILL = "wmma_prefill_bf16_bf16_out"
_PACK8_DECODE = "pack8_gemv_bf16_bf16_out"
_PACK8_LM_HEAD_F32 = "pack8_gemv_bf16_f32_out"
_SELECTED_GEMV = "selected_gemv_bf16_bf16_out"


def _selection(
    *,
    layer: str,
    scope: str,
    selected_variant: str,
    strict_fallback_variant: str | None = None,
    registry_quant: str,
    evidence_artifact: str | None = None,
) -> VariantSelection:
    return VariantSelection(
        layer=layer,
        scope=scope,
        selected_variant=selected_variant,
        strict_fallback_variant=(
            selected_variant if strict_fallback_variant is None
            else strict_fallback_variant
        ),
        evidence_artifact=evidence_artifact,
        registry_quant=registry_quant,
    )


def _production_selections() -> tuple[VariantSelection, ...]:
    """The measured default path: what an omitted profile executes today.

    ``strict_fallback_variant`` is each scope's rollback key, so it names the
    ``strict`` plan's variant for that scope (the rule
    ``resolve_runtime_profile`` enforces, and the same shape qwen's plans use)
    -- for the MoE scopes that is the grouped exact owner, for the
    arm-invariant dense scopes today's own variant.
    """

    return (
        _selection(
            layer="embedding", scope="lookup_q8_0", selected_variant=_LOOKUP,
            registry_quant=_Q8_0,
        ),
        _selection(
            layer="linear", scope="dense_prefill_q8_0",
            selected_variant=_WMMA_PREFILL, registry_quant=_Q8_0,
        ),
        _selection(
            layer="linear", scope="dense_decode_q8_0",
            selected_variant=_PACK8_DECODE, registry_quant=_Q8_0,
        ),
        _selection(
            layer="linear", scope="lm_head_f32_q8_0",
            selected_variant=_PACK8_LM_HEAD_F32, registry_quant=_Q8_0,
        ),
        # MoE work that resolves through the dense ``linear`` layer key: 29
        # decode calls/step matches the MoE layer count, and the selected owner
        # was site-attributed to gemma4_experts.py:1401. The T16 conversion of
        # the gate_up stacks changes the registry key, not the owner.
        _selection(
            layer="linear", scope="moe_decode_q4_k",
            selected_variant=_SELECTED_GEMV, registry_quant=_Q4_K_T16,
        ),
        _selection(
            layer="linear", scope="moe_decode_q5_1",
            selected_variant=_SELECTED_GEMV, registry_quant=_Q5_1,
        ),
        _selection(
            layer="linear", scope="moe_decode_q5_k",
            selected_variant=_SELECTED_GEMV, registry_quant=_Q5_K,
        ),
        _selection(
            layer="linear", scope="moe_selected_q8_0",
            selected_variant=_SELECTED_GEMV, registry_quant=_Q8_0,
        ),
        _selection(
            layer="moe_linear", scope="gate_up_q4_k",
            selected_variant=_T16_FUSED_WMMA, strict_fallback_variant=_T16_DECODE,
            registry_quant=_Q4_K_T16,
        ),
        _selection(
            layer="moe_linear", scope="gate_up_q5_k",
            selected_variant=_MMQ32, strict_fallback_variant=_GROUPED_DUAL,
            registry_quant=_Q5_K,
        ),
        _selection(
            layer="moe_linear", scope="down_q5_1",
            selected_variant=_WMMA_DOWN,
            strict_fallback_variant=_GROUPED_DOWN_Q5_1,
            registry_quant=_Q5_1,
        ),
        # ``moe_linear``/``down_q8_0`` deliberately absent (B3): no grouped q8_0
        # owner registers, so strict resolves that projection under the
        # ``linear`` layer key instead -- see the module docstring.
    )


def _strict_selections() -> tuple[VariantSelection, ...]:
    """The exact arm for the MoE scopes; blockers recorded where unreachable.

    Every scope the production plan declares must appear here, so the
    arm-invariant dense and embedding scopes carry the variant that actually
    runs under strict today with the blocker noted in the module docstring --
    naming an exact variant the runtime cannot select would make the manifest
    describe execution that does not happen.
    """

    return (
        _selection(
            layer="embedding", scope="lookup_q8_0", selected_variant=_LOOKUP,
            registry_quant=_Q8_0,
        ),
        # B1: exact owner prefill_bf16_bf16_out is unreachable (hard-coded
        # use_wmma_prefill=True outranks session and env).
        _selection(
            layer="linear", scope="dense_prefill_q8_0",
            selected_variant=_WMMA_PREFILL, registry_quant=_Q8_0,
        ),
        # B2: no route pin exists for the decode GEMVs.
        _selection(
            layer="linear", scope="dense_decode_q8_0",
            selected_variant=_PACK8_DECODE, registry_quant=_Q8_0,
        ),
        # B4: counts data-dependent; no arm difference measurable.
        _selection(
            layer="linear", scope="lm_head_f32_q8_0",
            selected_variant=_PACK8_LM_HEAD_F32, registry_quant=_Q8_0,
        ),
        _selection(
            layer="linear", scope="moe_decode_q4_k",
            selected_variant=_SELECTED_GEMV, registry_quant=_Q4_K_T16,
        ),
        _selection(
            layer="linear", scope="moe_decode_q5_1",
            selected_variant=_SELECTED_GEMV, registry_quant=_Q5_1,
        ),
        _selection(
            layer="linear", scope="moe_decode_q5_k",
            selected_variant=_SELECTED_GEMV, registry_quant=_Q5_K,
        ),
        _selection(
            layer="linear", scope="moe_selected_q8_0",
            selected_variant=_SELECTED_GEMV, registry_quant=_Q8_0,
        ),
        # The T16-converted gate_up scope: grouped has no T16 owner, so the
        # strict arm resolves this scope through the exact selected owner
        # (also production's declared fallback; the raw grouped arm still
        # serves the unconverted Q5_K stack at layer 29).
        _selection(
            layer="moe_linear", scope="gate_up_q4_k",
            selected_variant=_T16_DECODE, registry_quant=_Q4_K_T16,
        ),
        _selection(
            layer="moe_linear", scope="gate_up_q5_k",
            selected_variant=_GROUPED_DUAL, registry_quant=_Q5_K,
        ),
        _selection(
            layer="moe_linear", scope="down_q5_1",
            selected_variant=_GROUPED_DOWN_Q5_1, registry_quant=_Q5_1,
        ),
    )


def _binder(route: str):
    def bind(_generator: Any, _resolved: ResolvedRuntimeProfile) -> None:
        # The route pin is read per prefill call, so a construction-time write
        # holds for the whole run. Pinning (rather than only filling a default)
        # keeps the manifest true: the profile declares which route executes
        # even if the process environment already carried a diagnostic pin.
        os.environ[MOE_PREFILL_ENV] = route

    return bind


def _registered_key() -> tuple[str, str, str]:
    return (GEMMA4_GGUF_MODEL, GEMMA4_GGUF_BACKEND, GEMMA4_GGUF_QUANT)


def _already_registered() -> bool:
    from hipengine.execution_profiles import RuntimeProfileKey

    model, backend, quant = _registered_key()
    return any(
        key.model == model and key.backend == backend and key.quant == quant
        for key in registered_runtime_profile_keys()
    )


def register_gemma4_gguf_profiles() -> bool:
    """Register the strict and production plans once; idempotent.

    ``batch_invariant`` is left unregistered on purpose (fail-closed fallback
    to strict). Registration loads no kernels and touches no GPU: the plans are
    cold-path provenance plus a construction-time route policy.
    """

    if _already_registered():
        return False
    register_runtime_profile_plan(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
        plan=RuntimeProfilePlan(
            selections=_strict_selections(),
            kv_policy=_KV_POLICY,
            graph_policy=_GRAPH_POLICY,
            binder=_binder(_STRICT_ROUTE),
        ),
    )
    register_runtime_profile_plan(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=ExecutionProfile.PRODUCTION,
        plan=RuntimeProfilePlan(
            selections=_production_selections(),
            kv_policy=_KV_POLICY,
            graph_policy=_GRAPH_POLICY,
            binder=_binder(_PRODUCTION_ROUTE),
        ),
    )
    return True


def gemma4_gguf_profiles_registered() -> bool:
    """Return whether the strict and production plans are both registered."""

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
    "GEMMA4_GGUF_BACKEND",
    "GEMMA4_GGUF_MODEL",
    "GEMMA4_GGUF_QUANT",
    "MOE_PREFILL_ENV",
    "gemma4_gguf_profiles_registered",
    "register_gemma4_gguf_profiles",
]