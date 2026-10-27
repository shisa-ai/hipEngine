"""RED/GREEN tests for the dense Llama strict execution-profile plans.

Covers ``hipengine/generation/llama_dense_profiles.py`` and its wiring through
``register_builtin_generators``:

- idempotent registration of the strict plan for both gfx11 backends;
- ``production`` is not registered, so an omitted profile keeps the migration
  path and the tiled default;
- strict resolution names the exact block attention and is stable;
- ``construct_generator`` constructs a strict generator through the plan's
  factory for both backends;
- public ``LLM(..., execution_profile="strict")`` resolves and constructs the
  strict generator on the backend the device can execute.

Everything here runs without launching a kernel. Generator construction reads
the checkpoint, the tokenizer, and the context budget; the resident runner is
created lazily at the first request.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine import LLM
from hipengine.execution_profiles import (
    ExecutionProfile,
    RuntimeProfileKey,
    clear_runtime_profile_registry_for_tests,
    registered_runtime_profile_keys,
    resolve_default_execution_profile,
    resolve_runtime_profile,
)
from hipengine.generation.llama_dense_profiles import (
    ATTENTION_LAYER,
    ATTENTION_SCOPE,
    ATTENTION_STRICT_VARIANT,
    ATTENTION_TILED_VARIANT,
    GRAPH_POLICY,
    KV_POLICY,
    LLAMA_DENSE_BACKENDS,
    LLAMA_DENSE_MODEL,
    LLAMA_DENSE_QUANT,
    llama_dense_profiles_registered,
    register_llama_dense_profiles,
)
from hipengine.kernels.registry import KernelKey, is_registered
from tests._llama_tiny import write_tiny_llama_checkpoint


@pytest.fixture(autouse=True)
def _isolate_profile_registry():
    clear_runtime_profile_registry_for_tests()
    yield
    clear_runtime_profile_registry_for_tests()


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    path, _f32, _bf16 = write_tiny_llama_checkpoint(tmp_path / "tiny")
    return path


def _register() -> None:
    assert register_llama_dense_profiles() is True


def _resolution(profile: ExecutionProfile | str, backend: str):
    return resolve_runtime_profile(
        model=LLAMA_DENSE_MODEL,
        backend=backend,
        quant=LLAMA_DENSE_QUANT,
        profile=profile,
    )


def _scope_map(manifest):
    return {(item["layer"], item["scope"]): item for item in manifest["selections"]}


def _skip_unless_device_runs(backend: str) -> None:
    """Skip when this host's device is known and cannot execute ``backend``."""

    from hipengine.kernels.backends import HIP_BACKEND_TARGET_ARCH, selected_device_arches

    arches = selected_device_arches()
    target = HIP_BACKEND_TARGET_ARCH[backend]
    if arches and target not in arches:
        pytest.skip(f"{backend} targets {target}; device arches are {arches}")


def test_registration_is_idempotent_and_strict_only() -> None:
    _register()
    assert register_llama_dense_profiles() is False
    assert llama_dense_profiles_registered() is True
    keys = set(registered_runtime_profile_keys())
    for backend in LLAMA_DENSE_BACKENDS:
        assert (
            RuntimeProfileKey(
                model=LLAMA_DENSE_MODEL,
                backend=backend,
                quant=LLAMA_DENSE_QUANT,
                profile=ExecutionProfile.STRICT,
            )
            in keys
        )
        # No production plan: omitted-profile selection must not change.
        assert (
            RuntimeProfileKey(
                model=LLAMA_DENSE_MODEL,
                backend=backend,
                quant=LLAMA_DENSE_QUANT,
                profile=ExecutionProfile.PRODUCTION,
            )
            not in keys
        )
        # Both registered routes are described under each backend.
        for variant in (ATTENTION_TILED_VARIANT, ATTENTION_STRICT_VARIANT):
            assert is_registered(
                KernelKey(backend, ATTENTION_LAYER, LLAMA_DENSE_QUANT, variant)
            )


@pytest.mark.parametrize("backend", LLAMA_DENSE_BACKENDS)
def test_strict_resolution_selects_the_exact_kernel(backend: str) -> None:
    _register()
    resolved = _resolution(ExecutionProfile.STRICT, backend)
    assert resolved.profile is ExecutionProfile.STRICT
    assert resolved.source_profile is ExecutionProfile.STRICT
    assert resolved.fell_back_to_strict is False
    assert resolved.qualification_failure is None
    manifest = resolved.manifest
    assert manifest["execution_profile"] == "strict"
    assert manifest["model"] == LLAMA_DENSE_MODEL
    assert manifest["backend"] == backend
    assert manifest["quant"] == LLAMA_DENSE_QUANT
    assert manifest["kv_policy"] == KV_POLICY
    assert manifest["graph_policy"] == GRAPH_POLICY
    by_scope = _scope_map(manifest)
    selection = by_scope[(ATTENTION_LAYER, ATTENTION_SCOPE)]
    assert selection["selected_variant"] == ATTENTION_STRICT_VARIANT
    assert selection["strict_fallback_variant"] == ATTENTION_STRICT_VARIANT
    # The strict manifest is its own rollback baseline.
    assert resolved.strict_manifest_sha256 == resolved.manifest_sha256
    assert len(resolved.manifest_sha256) == 64
    again = _resolution(ExecutionProfile.STRICT, backend)
    assert again.manifest_sha256 == resolved.manifest_sha256


def test_omitted_profile_keeps_the_tiled_default() -> None:
    _register()
    for backend in LLAMA_DENSE_BACKENDS:
        assert (
            resolve_default_execution_profile(
                model=LLAMA_DENSE_MODEL,
                backend=backend,
                quant=LLAMA_DENSE_QUANT,
            )
            is None
        )


@pytest.mark.parametrize("profile", ["production", "batch_invariant", "fast"])
def test_absent_profiles_fall_back_to_strict(profile: str) -> None:
    _register()
    for backend in LLAMA_DENSE_BACKENDS:
        resolved = _resolution(profile, backend)
        assert resolved.profile.value == profile
        assert resolved.source_profile is ExecutionProfile.STRICT
        assert resolved.fell_back_to_strict is True
        assert resolved.qualification_failure is None
        for item in resolved.manifest["selections"]:
            assert item["selected_variant"] == ATTENTION_STRICT_VARIANT
            assert item["strict_fallback_variant"] == ATTENTION_STRICT_VARIANT


@pytest.mark.parametrize("backend", LLAMA_DENSE_BACKENDS)
def test_construct_generator_selects_strict(backend: str, checkpoint: Path) -> None:
    from hipengine.loading.safetensors import load_weight_index
    from hipengine.models.llama import LLAMA_DENSE

    _register()
    resolved = _resolution(ExecutionProfile.STRICT, backend)

    def base_factory(**kwargs):
        raise AssertionError("the strict plan must supply its own factory")

    generator = resolved.construct_generator(
        base_factory,
        model_path=checkpoint,
        weight_index=load_weight_index(checkpoint),
        model_plugin=LLAMA_DENSE,
    )
    try:
        assert generator.backend == backend
        assert generator.execution_profile == "strict"
        assert generator.strict_arithmetic is True
        assert generator.execution_profile_manifest_sha256 == resolved.manifest_sha256
        assert generator.execution_profile_fell_back_to_strict is False
    finally:
        generator.close()


@pytest.mark.parametrize("backend", LLAMA_DENSE_BACKENDS)
def test_public_llm_strict_selection(backend: str, checkpoint: Path) -> None:
    from hipengine.generation import register_builtin_generators

    _skip_unless_device_runs(backend)
    register_builtin_generators()
    assert llama_dense_profiles_registered() is True

    llm = LLM(str(checkpoint), backend=backend, execution_profile="strict")
    assert llm.resolved_execution_profile == "strict"
    assert llm.execution_profile_fell_back_to_strict is False
    manifest = llm.execution_profile_manifest
    assert manifest["execution_profile"] == "strict"
    assert manifest["backend"] == backend
    assert manifest["selections"][0]["selected_variant"] == ATTENTION_STRICT_VARIANT

    generator = llm._get_text_generator()
    assert generator.execution_profile == "strict"
    assert generator._inner.execution_profile == "strict"
    assert generator._inner.strict_arithmetic is True
    assert generator.execution_profile_manifest_sha256 == (
        llm.execution_profile_manifest_sha256
    )
