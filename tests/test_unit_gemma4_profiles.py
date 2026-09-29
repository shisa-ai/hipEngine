"""Unit tests for the Gemma 4 execution-profile plans (gate build piece A).

The registration is the load-bearing part of task #44: an omitted profile must
resolve to ``production`` for ``(gemma4_gguf, hip_gfx1100, gguf_q4_k_m)`` and
the manifest must describe the routes a real generate measured, while
``execution_profile="strict"`` selects the exact MoE arm. The production
selections here are asserted against the resolve-spy table from the curation
entry, so a drift between what ships and what the manifest claims fails loudly
instead of passing silently.
"""

from __future__ import annotations

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    resolve_default_execution_profile,
    resolve_runtime_profile,
)
from hipengine.generation import register_builtin_generators
from hipengine.generation.gemma4_profiles import (
    GEMMA4_GGUF_BACKEND,
    GEMMA4_GGUF_MODEL,
    GEMMA4_GGUF_QUANT,
    MOE_PREFILL_ENV,
    _production_selections,
    _strict_selections,
    gemma4_gguf_profiles_registered,
    register_gemma4_gguf_profiles,
)

# The measured default path (resolve-spy trace, mode=auto, shipped artifact).
_MMQ32 = "selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out"
_WMMA_DOWN = "selected_grouped_wmma_prefill_compact_bf16_bf16_out"
_GROUPED_DUAL = "selected_dual_grouped_rowbatch8_out4_amortized_bf16_bf16_out"
_GROUPED_DOWN_Q5_1 = "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"

# Register at import, not only inside each test: conftest snapshots
# ``registry._KERNELS`` in ``pytest_collection_finish`` (after every module has
# imported) and restores that baseline after each test. Kernels registered only
# inside a test body are wiped before the next test, while
# ``register_builtin_generators`` is itself a process-once import guard -- so a
# module-level call is what makes resolution order-independent here. It also
# pins the gemma4 plans into the same baseline. CPU-only: no GPU is touched.
register_builtin_generators()


def _registered() -> None:
    # Mirror production: ``LLM._get_text_generator`` calls
    # ``register_builtin_generators()`` before it resolves the profile, and
    # resolution validates every manifest variant against the kernel registry
    # (which conftest also restores to a collection-time snapshot between
    # tests). Registering here keeps each test independent of import order.
    from hipengine.generation import register_builtin_generators

    register_builtin_generators()
    register_gemma4_gguf_profiles()


def _by_scope(resolved) -> dict[tuple[str, str], dict]:
    return {
        (row["layer"], row["scope"]): row
        for row in resolved.manifest["selections"]
    }


def test_registration_is_idempotent_and_both_plans_exist() -> None:
    _registered()
    assert register_gemma4_gguf_profiles() is False, "second registration must be a no-op"
    assert gemma4_gguf_profiles_registered() is True


def test_omitted_profile_resolves_to_production() -> None:
    _registered()
    default = resolve_default_execution_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
    )
    assert default is ExecutionProfile.PRODUCTION
    resolved = resolve_runtime_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=default,
    )
    assert resolved.profile is ExecutionProfile.PRODUCTION
    assert resolved.fell_back_to_strict is False


def test_production_manifest_matches_the_measured_routes() -> None:
    _registered()
    resolved = resolve_runtime_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=ExecutionProfile.PRODUCTION,
    )
    rows = _by_scope(resolved)
    assert rows[("moe_linear", "gate_up_q4_k")]["selected_variant"] == _MMQ32
    assert rows[("moe_linear", "gate_up_q5_k")]["selected_variant"] == _MMQ32
    assert rows[("moe_linear", "down_q5_1")]["selected_variant"] == _WMMA_DOWN
    assert rows[("linear", "dense_prefill_q8_0")]["selected_variant"] == (
        "wmma_prefill_bf16_bf16_out"
    )
    assert rows[("embedding", "lookup_q8_0")]["selected_variant"] == "lookup_bf16_out"
    assert resolved.manifest["kv_policy"] == "paged_bf16"
    assert resolved.manifest["execution_profile"] == "production"


def test_strict_selects_the_measured_grouped_arm_for_moe() -> None:
    _registered()
    resolved = resolve_runtime_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
    )
    rows = _by_scope(resolved)
    # The MMQ production owner must be gone: grouped is the exact arm.
    assert rows[("moe_linear", "gate_up_q4_k")]["selected_variant"] == _GROUPED_DUAL
    assert rows[("moe_linear", "down_q5_1")]["selected_variant"] == _GROUPED_DOWN_Q5_1
    assert _MMQ32 not in {row["selected_variant"] for row in rows.values()}
    assert resolved.profile is ExecutionProfile.STRICT


def test_production_scopes_are_a_subset_of_strict_scopes() -> None:
    """``_resolved_selections`` rejects the inverse, so assert it directly."""

    production = {(row.layer, row.scope) for row in _production_selections()}
    strict = {(row.layer, row.scope) for row in _strict_selections()}
    assert production <= strict, sorted(production - strict)


def test_blocker_scopes_carry_the_variant_that_actually_runs() -> None:
    """B1/B2/B4 scopes list today's variant, never an unreachable exact one."""

    strict = {(row.layer, row.scope): row for row in _strict_selections()}
    production = {(row.layer, row.scope): row for row in _production_selections()}
    for scope in (
        ("linear", "dense_prefill_q8_0"),
        ("linear", "dense_decode_q8_0"),
        ("linear", "lm_head_f32_q8_0"),
    ):
        assert strict[scope].selected_variant == production[scope].selected_variant, scope
    # And the unreachable exact owner is nowhere in either plan.
    all_variants = {row.selected_variant for row in _strict_selections()}
    all_variants |= {row.selected_variant for row in _production_selections()}
    assert "prefill_bf16_bf16_out" not in all_variants


def test_binders_pin_the_route_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    _registered()
    monkeypatch.delenv(MOE_PREFILL_ENV, raising=False)
    strict = resolve_runtime_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
    )
    assert strict.binder is not None
    strict.binder(object(), strict)
    import os

    assert os.environ[MOE_PREFILL_ENV] == "grouped"

    production = resolve_runtime_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=ExecutionProfile.PRODUCTION,
    )
    assert production.binder is not None
    production.binder(object(), production)
    assert os.environ[MOE_PREFILL_ENV] == "auto"


def test_batch_invariant_is_not_registered_and_falls_back_to_strict() -> None:
    _registered()
    resolved = resolve_runtime_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=ExecutionProfile.BATCH_INVARIANT
        if hasattr(ExecutionProfile, "BATCH_INVARIANT")
        else "batch_invariant",
    )
    assert resolved.fell_back_to_strict is True
    assert resolved.source_profile is ExecutionProfile.STRICT