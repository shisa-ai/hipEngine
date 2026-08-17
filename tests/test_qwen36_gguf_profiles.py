"""RED/GREEN tests for the Qwen3.6 GGUF gfx1151 execution-profile plans.

These cover PN1 registration semantics of
``hipengine/generation/qwen36_gguf_profiles.py``:

- idempotent registration of strict + production plans;
- strict resolution produces the exact/unfused manifest;
- production resolution produces the incumbent package manifest with a strict
  rollback key for every scope and the fused-router evidence binding;
- batch_invariant is left unregistered and fails closed to strict;
- duplicate and missing-plan resolution errors;
- stable manifest hashes; and
- the cold-path policy binder applies the documented env policy.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    DuplicateRuntimeProfilePlanError,
    ExecutionProfile,
    MissingRuntimeProfilePlanError,
    RuntimeProfilePlan,
    VariantSelection,
    clear_runtime_profile_registry_for_tests,
    manifest_sha256,
    register_runtime_profile_plan,
    resolve_runtime_profile,
)
from hipengine.generation.qwen36_gguf_profiles import (
    QWEN36_GGUF_BACKEND,
    QWEN36_GGUF_MODEL,
    QWEN36_GGUF_QUANT,
    ROUTER_COOP_ENV,
    ROUTER_PERSISTENT_ENV,
    ROWTILE_ALL_ENV,
    qwen36_gguf_gfx1151_plans_registered,
    register_qwen36_gguf_gfx1151_profiles,
)

# Register the gfx1151 kernel package at import (collection) time so the shared
# ``tests/conftest.py`` teardown baseline snapshot includes its variants; the
# profile-resolution ``_verify_registered_variants`` gate needs them registered
# in every test.
from hipengine.kernels.backends import load_backend_kernel_package  # noqa: E402

load_backend_kernel_package(QWEN36_GGUF_BACKEND)

_POLICY_ENV = (ROUTER_COOP_ENV, ROUTER_PERSISTENT_ENV, ROWTILE_ALL_ENV)


@pytest.fixture(autouse=True)
def _isolate_profile_registry():
    clear_runtime_profile_registry_for_tests()
    yield
    clear_runtime_profile_registry_for_tests()


@pytest.fixture(autouse=True)
def _isolate_process_env():
    saved = {key: os.environ.get(key) for key in _POLICY_ENV}
    yield
    for key, prior in saved.items():
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


def _register():
    assert register_qwen36_gguf_gfx1151_profiles() is True


def _resolution(profile: ExecutionProfile | str):
    return resolve_runtime_profile(
        model=QWEN36_GGUF_MODEL,
        backend=QWEN36_GGUF_BACKEND,
        quant=QWEN36_GGUF_QUANT,
        profile=profile,
    )


def _scope_map(manifest):
    return {(item["layer"], item["scope"]): item for item in manifest["selections"]}


def test_registration_is_idempotent() -> None:
    _register()
    assert register_qwen36_gguf_gfx1151_profiles() is False
    assert qwen36_gguf_gfx1151_plans_registered() is True


def test_strict_resolution_exact_unfused_manifest() -> None:
    _register()
    resolved = _resolution(ExecutionProfile.STRICT)
    assert resolved.profile is ExecutionProfile.STRICT
    assert resolved.fell_back_to_strict is False
    assert resolved.source_profile is ExecutionProfile.STRICT
    manifest = resolved.manifest
    assert manifest["execution_profile"] == "strict"
    assert manifest["backend"] == QWEN36_GGUF_BACKEND
    assert manifest["model"] == QWEN36_GGUF_MODEL
    assert manifest["quant"] == QWEN36_GGUF_QUANT
    by_scope = _scope_map(manifest)
    router = by_scope[("router_logits", "c1_router")]
    assert router["selected_variant"] == "f32_hidden"
    assert router["registry_quant"] == "f32"
    linear = by_scope[("linear", "decode")]
    assert linear["selected_variant"] == "pack8_full_k_grid_y_native_exact_bf16_bf16_out"
    assert linear["registry_quant"] == QWEN36_GGUF_QUANT
    # Every strict selection is its own rollback key.
    for item in manifest["selections"]:
        assert item["strict_fallback_variant"] == item["selected_variant"]


def test_production_resolution_incumbent_route_with_strict_fallbacks() -> None:
    _register()
    resolved = _resolution(ExecutionProfile.PRODUCTION)
    assert resolved.profile is ExecutionProfile.PRODUCTION
    assert resolved.fell_back_to_strict is False
    manifest = resolved.manifest
    assert manifest["execution_profile"] == "production"
    by_scope = _scope_map(manifest)
    router = by_scope[("router_logits", "c1_router")]
    # The fused cooperative route is bound through evidence; the strict rollback
    # key is the base F32 router variant.
    assert router["selected_variant"] == "f32_hidden"
    assert router["strict_fallback_variant"] == "f32_hidden"
    assert router["evidence_artifact"].endswith("c1-router-retained.json")
    linear = by_scope[("linear", "decode")]
    assert linear["strict_fallback_variant"]
    assert linear["evidence_artifact"].endswith("q8t16-batch-route-retained.json")
    # Every production scope carries a strict fallback key.
    for item in manifest["selections"]:
        assert item["strict_fallback_variant"]


def test_batch_invariant_fails_closed_to_strict() -> None:
    _register()
    resolved = _resolution(ExecutionProfile.BATCH_INVARIANT)
    assert resolved.profile is ExecutionProfile.BATCH_INVARIANT
    assert resolved.fell_back_to_strict is True
    assert resolved.source_profile is ExecutionProfile.STRICT
    # Selections are the strict (exact/unfused) set.
    for item in resolved.manifest["selections"]:
        assert item["strict_fallback_variant"] == item["selected_variant"]


def test_manifest_hash_is_stable_across_resolutions() -> None:
    _register()
    first = _resolution(ExecutionProfile.PRODUCTION).manifest_sha256
    second = _resolution(ExecutionProfile.PRODUCTION).manifest_sha256
    third = _resolution(ExecutionProfile.PRODUCTION).manifest_sha256
    assert first == second == third
    assert len(first) == 64


def test_duplicate_registration_raises() -> None:
    _register()
    with pytest.raises(DuplicateRuntimeProfilePlanError):
        register_runtime_profile_plan(
            model=QWEN36_GGUF_MODEL,
            backend=QWEN36_GGUF_BACKEND,
            quant=QWEN36_GGUF_QUANT,
            profile=ExecutionProfile.STRICT,
            plan=RuntimeProfilePlan(
                selections=(
                    VariantSelection(
                        layer="router_logits",
                        scope="c1_router",
                        selected_variant="f32_hidden",
                        strict_fallback_variant="f32_hidden",
                    ),
                ),
                kv_policy="paged_bf16",
                graph_policy="serial_c1",
                binder=lambda generator, resolved: None,
            ),
        )


def test_missing_strict_plan_raises_for_unregistered_model() -> None:
    with pytest.raises(MissingRuntimeProfilePlanError):
        resolve_runtime_profile(
            model="no_such_model",
            backend=QWEN36_GGUF_BACKEND,
            quant=QWEN36_GGUF_QUANT,
            profile=ExecutionProfile.STRICT,
        )


def test_missing_profile_plan_raises_when_registry_cleared() -> None:
    clear_runtime_profile_registry_for_tests()
    with pytest.raises(MissingRuntimeProfilePlanError):
        _resolution(ExecutionProfile.STRICT)


def test_strict_plan_requires_a_factory_or_binder() -> None:
    with pytest.raises(ValueError, match="factory or binder"):
        RuntimeProfilePlan(
            selections=(
                VariantSelection(
                    layer="router_logits",
                    scope="c1_router",
                    selected_variant="f32_hidden",
                    strict_fallback_variant="f32_hidden",
                ),
            ),
            kv_policy="paged_bf16",
            graph_policy="serial_c1",
        )


def test_production_binder_applies_cooperative_policy() -> None:
    _register()
    resolved = _resolution(ExecutionProfile.PRODUCTION)
    generator = resolved.construct_generator(
        lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        model_path="/tmp/fake.gguf",
        weight_index=None,
        model_plugin=None,
    )
    assert os.environ.get(ROUTER_COOP_ENV) == "1"
    assert os.environ.get(ROUTER_PERSISTENT_ENV) == "1"
    assert ROWTILE_ALL_ENV not in os.environ
    assert generator.execution_profile == "production"
    assert generator.execution_profile_manifest_sha256 == resolved.manifest_sha256
    assert generator.execution_profile_strict_manifest_sha256 == (
        resolved.strict_manifest_sha256
    )
    assert generator.execution_profile_strict_manifest_sha256 != resolved.manifest_sha256
    # The strict manifest hash is the hash of the independently resolved strict plan.
    strict_resolved = _resolution(ExecutionProfile.STRICT)
    assert resolved.strict_manifest_sha256 == strict_resolved.manifest_sha256


def test_strict_binder_applies_exact_policy() -> None:
    _register()
    resolved = _resolution(ExecutionProfile.STRICT)
    generator = resolved.construct_generator(
        lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        model_path="/tmp/fake.gguf",
        weight_index=None,
        model_plugin=None,
    )
    assert os.environ.get(ROUTER_COOP_ENV) == "0"
    assert os.environ.get(ROUTER_PERSISTENT_ENV) == "0"
    assert os.environ.get(ROWTILE_ALL_ENV) == "0"
    assert generator.execution_profile == "strict"
    assert generator.execution_profile_strict_manifest_sha256 == (
        generator.execution_profile_manifest_sha256
    )
    assert generator.execution_profile == "strict"


def test_manifest_hash_matches_canonical_build() -> None:
    """The resolved production manifest hash equals the canonical builder hash."""

    from hipengine.execution_profiles import build_variant_manifest

    _register()
    resolved = _resolution(ExecutionProfile.PRODUCTION)
    rebuilt = build_variant_manifest(
        profile=ExecutionProfile.PRODUCTION,
        backend=resolved.manifest["backend"],
        model=resolved.manifest["model"],
        quant=resolved.manifest["quant"],
        kv_policy=resolved.manifest["kv_policy"],
        graph_policy=resolved.manifest["graph_policy"],
        selections=resolved.manifest["selections"],
    )
    assert manifest_sha256(rebuilt) == resolved.manifest_sha256
