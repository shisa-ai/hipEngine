from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine import ExecutionProfile, LLM, SamplingParams
from hipengine.execution_profiles import (
    RuntimeProfilePlan,
    VariantSelection,
    manifest_sha256,
    register_runtime_profile_plan,
    resolve_requested_execution_profile,
    resolve_runtime_profile,
)
from hipengine.generation import GenerationRequest, register_text_generator
from hipengine.kernels.registry import KernelKey, register
from hipengine.server.__main__ import build_parser
from hipengine.server.api import ServerConfig


def _selection(variant: str, fallback: str) -> VariantSelection:
    return VariantSelection(
        layer="profile_test_layer",
        scope="all",
        selected_variant=variant,
        strict_fallback_variant=fallback,
        evidence_artifact="tests/synthetic",
        registry_quant="profile_test_quant",
    )


def _register_kernel(variant: str) -> None:
    register(
        KernelKey(
            backend="profile_test_backend",
            layer="profile_test_layer",
            quant="profile_test_quant",
            variant=variant,
        ),
        lambda: variant,
        replace=True,
    )


def test_runtime_profile_resolution_falls_back_to_registered_strict_plan() -> None:
    _register_kernel("strict_exact")

    def strict_factory(**kwargs):
        return SimpleNamespace(factory="strict", kwargs=kwargs)

    register_runtime_profile_plan(
        model="profile_test_model_fallback",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile="strict",
        plan=RuntimeProfilePlan(
            selections=(_selection("strict_exact", "strict_exact"),),
            kv_policy="request_resolved",
            graph_policy="shape_bucketed",
            factory=strict_factory,
        ),
        replace=True,
    )

    strict = resolve_runtime_profile(
        model="profile_test_model_fallback",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile=ExecutionProfile.STRICT,
    )
    production = resolve_runtime_profile(
        model="profile_test_model_fallback",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile=ExecutionProfile.PRODUCTION,
    )
    invariant = resolve_runtime_profile(
        model="profile_test_model_fallback",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile=ExecutionProfile.BATCH_INVARIANT,
    )

    assert strict.fell_back_to_strict is False
    assert production.fell_back_to_strict is True
    assert invariant.fell_back_to_strict is True
    assert production.factory is strict_factory
    assert production.manifest["execution_profile"] == "production"
    assert production.manifest["selections"][0]["selected_variant"] == "strict_exact"
    assert production.manifest["selections"][0]["strict_fallback_variant"] == "strict_exact"
    assert production.manifest_sha256 == manifest_sha256(production.manifest)


def test_runtime_profile_resolution_uses_certified_production_variant_and_exact_fallback() -> None:
    _register_kernel("strict_exact")
    _register_kernel("production_online")

    register_runtime_profile_plan(
        model="profile_test_model_candidate",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile="strict",
        plan=RuntimeProfilePlan(
            selections=(_selection("strict_exact", "strict_exact"),),
            kv_policy="request_resolved",
            graph_policy="shape_bucketed",
            factory=lambda **kwargs: SimpleNamespace(factory="strict", kwargs=kwargs),
        ),
        replace=True,
    )

    def production_factory(**kwargs):
        return SimpleNamespace(factory="production", kwargs=kwargs)

    register_runtime_profile_plan(
        model="profile_test_model_candidate",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile="production",
        plan=RuntimeProfilePlan(
            selections=(_selection("production_online", "strict_exact"),),
            kv_policy="request_resolved",
            graph_policy="shape_bucketed",
            factory=production_factory,
        ),
        replace=True,
    )

    resolution = resolve_runtime_profile(
        model="profile_test_model_candidate",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile="production",
    )

    assert resolution.fell_back_to_strict is False
    assert resolution.factory is production_factory
    assert resolution.manifest["selections"][0]["selected_variant"] == "production_online"
    assert resolution.manifest["selections"][0]["strict_fallback_variant"] == "strict_exact"


def test_runtime_profile_resolution_falls_back_per_missing_scope() -> None:
    for layer in ("profile_scope_a", "profile_scope_b"):
        for variant in ("strict_exact", "production_online"):
            register(
                KernelKey(
                    backend="profile_test_backend",
                    layer=layer,
                    quant="profile_test_quant",
                    variant=variant,
                ),
                lambda: None,
                replace=True,
            )

    def scoped(layer: str, selected: str) -> VariantSelection:
        return VariantSelection(
            layer=layer,
            scope="all",
            selected_variant=selected,
            strict_fallback_variant="strict_exact",
            evidence_artifact="tests/synthetic",
            registry_quant="profile_test_quant",
        )

    register_runtime_profile_plan(
        model="profile_test_model_partial",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile="strict",
        plan=RuntimeProfilePlan(
            selections=(
                scoped("profile_scope_a", "strict_exact"),
                scoped("profile_scope_b", "strict_exact"),
            ),
            kv_policy="request_resolved",
            graph_policy="shape_bucketed",
            factory=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        ),
        replace=True,
    )
    register_runtime_profile_plan(
        model="profile_test_model_partial",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile="production",
        plan=RuntimeProfilePlan(
            selections=(scoped("profile_scope_a", "production_online"),),
            kv_policy="request_resolved",
            graph_policy="shape_bucketed",
            factory=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        ),
        replace=True,
    )

    resolution = resolve_runtime_profile(
        model="profile_test_model_partial",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile="production",
    )
    by_layer = {item["layer"]: item for item in resolution.manifest["selections"]}

    assert resolution.fell_back_to_strict is True
    assert by_layer["profile_scope_a"]["selected_variant"] == "production_online"
    assert by_layer["profile_scope_b"]["selected_variant"] == "strict_exact"
    assert by_layer["profile_scope_b"]["strict_fallback_variant"] == "strict_exact"


def test_runtime_profile_resolution_rejects_missing_strict_or_unregistered_variants() -> None:
    with pytest.raises(LookupError, match="strict execution-profile plan"):
        resolve_runtime_profile(
            model="profile_test_missing",
            backend="profile_test_backend",
            quant="profile_test_quant",
            profile="production",
        )

    register_runtime_profile_plan(
        model="profile_test_unregistered",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile="strict",
        plan=RuntimeProfilePlan(
            selections=(_selection("not_registered", "not_registered"),),
            kv_policy="request_resolved",
            graph_policy="shape_bucketed",
            factory=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        ),
        replace=True,
    )
    with pytest.raises(LookupError, match="not registered"):
        resolve_runtime_profile(
            model="profile_test_unregistered",
            backend="profile_test_backend",
            quant="profile_test_quant",
            profile="strict",
        )


def test_llm_explicit_profile_constructs_from_resolved_plan_without_changing_legacy_default(
    monkeypatch,
) -> None:
    import hipengine.generation as generation

    _register_kernel("strict_exact")
    calls: list[str] = []

    class FakeGenerator:
        def __init__(self, source: str) -> None:
            self.source = source

        def generate(self, request: GenerationRequest) -> list[str]:
            return [f"{prompt}:{self.source}" for prompt in request.prompts]

    def legacy_factory(**kwargs):
        del kwargs
        calls.append("legacy")
        return FakeGenerator("legacy")

    def strict_factory(**kwargs):
        del kwargs
        calls.append("strict")
        return FakeGenerator("strict")

    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    register_text_generator(
        model="profile_test_llm",
        backend="profile_test_backend",
        quant="profile_test_quant",
        factory=legacy_factory,
        replace=True,
    )
    register_runtime_profile_plan(
        model="profile_test_llm",
        backend="profile_test_backend",
        quant="profile_test_quant",
        profile="strict",
        plan=RuntimeProfilePlan(
            selections=(_selection("strict_exact", "strict_exact"),),
            kv_policy="request_resolved",
            graph_policy="shape_bucketed",
            factory=strict_factory,
        ),
        replace=True,
    )
    plugin = SimpleNamespace(name="profile_test_llm", default_quant="profile_test_quant")
    index = SimpleNamespace(model_path="/tmp/profile-test", config={})

    production = LLM(
        "/tmp/profile-test",
        backend="profile_test_backend",
        quant="profile_test_quant",
        execution_profile="production",
    )
    monkeypatch.setattr(production, "_load_model_metadata", lambda: (index, plugin))
    legacy = LLM(
        "/tmp/profile-test",
        backend="profile_test_backend",
        quant="profile_test_quant",
    )
    monkeypatch.setattr(legacy, "_load_model_metadata", lambda: (index, plugin))

    assert production.generate("hello", SamplingParams(max_tokens=1)) == ["hello:strict"]
    assert production.resolved_execution_profile == "production"
    assert production.execution_profile_manifest["execution_profile"] == "production"
    assert len(production.execution_profile_manifest_sha256) == 64
    assert len(production.execution_profile_strict_manifest_sha256) == 64
    assert (
        production.execution_profile_strict_manifest_sha256
        != production.execution_profile_manifest_sha256
    )
    assert production.execution_profile_fell_back_to_strict is True
    assert production._text_generator.execution_profile == "production"
    assert production._text_generator.execution_profile_manifest_sha256 == (
        production.execution_profile_manifest_sha256
    )
    assert production._text_generator.execution_profile_strict_manifest_sha256 == (
        production.execution_profile_strict_manifest_sha256
    )
    assert production._text_generator._inner.execution_profile == "production"
    assert production._text_generator.execution_profile_fell_back_to_strict is True
    with pytest.raises(TypeError):
        production._resolved_execution_profile.manifest["execution_profile"] = "strict"
    manifest_copy = production.execution_profile_manifest
    manifest_copy["execution_profile"] = "strict"
    assert production.execution_profile_manifest["execution_profile"] == "production"

    assert legacy.generate("hello", SamplingParams(max_tokens=1)) == ["hello:legacy"]
    assert legacy.resolved_execution_profile is None
    assert legacy.execution_profile_manifest is None
    assert calls == ["strict", "legacy"]


def test_profile_request_normalization_and_server_cli_env(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_EXECUTION_PROFILE", raising=False)
    assert resolve_requested_execution_profile(None) is None
    assert resolve_requested_execution_profile("strict") is ExecutionProfile.STRICT
    with pytest.raises(ValueError, match="execution_profile"):
        resolve_requested_execution_profile("fast")

    monkeypatch.setenv("HIPENGINE_EXECUTION_PROFILE", "batch_invariant")
    assert resolve_requested_execution_profile(None) is ExecutionProfile.BATCH_INVARIANT
    args = build_parser().parse_args(["--model", "fake"])
    assert args.execution_profile == "batch_invariant"
    overridden = build_parser().parse_args(
        ["--model", "fake", "--execution-profile", "production"]
    )
    assert overridden.execution_profile == "production"

    env_config = ServerConfig(model="fake")
    assert env_config.execution_profile == "batch_invariant"
    config = ServerConfig(model="fake", execution_profile="strict")
    assert config.execution_profile == "strict"
    with pytest.raises(ValueError, match="execution_profile"):
        ServerConfig(model="fake", execution_profile="aggressive")
