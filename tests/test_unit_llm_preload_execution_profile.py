"""Pre-load execution-profile resolution must see the builtin plans.

``create_app`` reads ``LLM.resolved_execution_profile`` while it builds the
application, before the first generation constructs a generator. Plan
registration belongs to the generation path, so a process that had not
generated yet resolved that property against an empty registry and raised
``MissingRuntimeProfilePlanError`` for a combination generation handles
normally.

These tests exercise the public property and ``create_app`` from the registry a
freshly started process has, with model metadata stubbed so no weights, device
or kernel are touched. Nothing here registers a plan by hand.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine import LLM
from hipengine import execution_profiles as ep
from hipengine.server import ServerConfig, create_app

MODEL = "qwen3_5_gguf"
BACKEND = "hip_gfx1151"
QUANT = "gguf_q4_k_m"
MODEL_PATH = "/nonexistent/preload-profile-fixture"
# The real UD shard directory from the observed failure. Header metadata only:
# the test never loads weights or touches a device.
UD_QWEN4EXP_SHARDS = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)


@pytest.fixture(autouse=True)
def _fresh_profile_registry() -> None:
    """Start from the registry a newly started process has.

    ``tests/conftest.py`` restores the collection-time registry after every
    test, so this clear does not leak into other tests.
    """

    ep.clear_runtime_profile_registry_for_tests()
    assert ep.registered_runtime_profile_keys() == ()


@pytest.fixture
def _stubbed_model_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        LLM,
        "_load_model_metadata",
        lambda self: (None, SimpleNamespace(name=MODEL)),
    )
    monkeypatch.setattr(LLM, "_resolve_backend", lambda self: BACKEND)


def _llm(*, profile: str | None = "production") -> LLM:
    return LLM(MODEL_PATH, backend=BACKEND, quant=QUANT, execution_profile=profile)


def test_preload_profile_property_resolves_the_registered_plan(
    _stubbed_model_metadata,
) -> None:
    llm = _llm()

    assert llm.resolved_execution_profile == "production"
    assert llm._text_generator is None
    key = ep.RuntimeProfileKey(
        model=MODEL,
        backend=BACKEND,
        quant=QUANT,
        profile=ep.ExecutionProfile.PRODUCTION,
    )
    assert key in ep.registered_runtime_profile_keys()


def test_preload_profile_equals_what_the_registry_serves_other_resolvers(
    _stubbed_model_metadata,
) -> None:
    llm = _llm()

    preload = llm.execution_profile_manifest_sha256
    assert preload
    # The property populated the registry, so an independent resolver must
    # select the same plan rather than a second, synthesized one.
    resolved = ep.resolve_runtime_profile(
        model=MODEL,
        backend=BACKEND,
        quant=QUANT,
        profile="production",
    )
    assert preload == resolved.manifest_sha256
    assert (
        llm.execution_profile_strict_manifest_sha256
        == resolved.strict_manifest_sha256
    )


def test_second_llm_reuses_the_registered_plans(_stubbed_model_metadata) -> None:
    """The pre-load refresh is idempotent, not a duplicate-registration trap."""

    first = _llm()
    assert first.resolved_execution_profile == "production"

    second = _llm()

    assert second.resolved_execution_profile == "production"
    assert (
        second.execution_profile_manifest_sha256
        == first.execution_profile_manifest_sha256
    )


def test_preload_default_profile_matches_the_generation_default(
    _stubbed_model_metadata,
) -> None:
    """No explicit profile still resolves ``production`` before any load."""

    llm = _llm(profile=None)

    assert llm.execution_profile is None
    assert llm.resolved_execution_profile == "production"


def test_create_app_reads_the_preload_profile_without_a_generator(
    _stubbed_model_metadata,
) -> None:
    llm = _llm()
    config = ServerConfig(
        model=MODEL_PATH,
        backend=BACKEND,
        quant=QUANT,
        execution_profile="production",
        served_model_name="preload-profile-check",
        eager_load=False,
        startup_chat_smoke=False,
        startup_scratch_probe=False,
    )

    app = create_app(config, llm=llm)

    assert app.state.hipengine_llm is llm
    assert llm._text_generator is None
    assert llm.resolved_execution_profile == "production"


def test_unsupported_declaration_keeps_the_missing_plan_error(
    monkeypatch: pytest.MonkeyPatch,
    _stubbed_model_metadata,
) -> None:
    """Registration must not turn a genuinely unknown combination into a plan."""

    monkeypatch.setattr(
        LLM,
        "_load_model_metadata",
        lambda self: (None, SimpleNamespace(name="no_such_model_plugin")),
    )

    with pytest.raises(ep.MissingRuntimeProfilePlanError):
        _llm().resolved_execution_profile


@pytest.mark.skipif(
    not UD_QWEN4EXP_SHARDS.is_dir(),
    reason="Qwen3.8-Flash-Next UD shard directory is not present",
)
def test_observed_ud_shard_declaration_resolves_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The observed triple resolves through ``create_app`` with real headers.

    ``qwen4_exp_gguf``/``hip_gfx1151``/``gguf_ud_q4_k_xl`` is the declaration
    that raised ``MissingRuntimeProfilePlanError`` before the first generation.
    Only the backend probe is stubbed; metadata comes from the real GGUF
    headers, and no weights are loaded.
    """

    monkeypatch.setattr(LLM, "_resolve_backend", lambda self: BACKEND)
    llm = LLM(
        str(UD_QWEN4EXP_SHARDS),
        backend=BACKEND,
        quant="gguf_ud_q4_k_xl",
        execution_profile="production",
    )
    _index, plugin = llm._load_model_metadata()
    assert plugin.name == "qwen4_exp_gguf"

    config = ServerConfig(
        model=str(UD_QWEN4EXP_SHARDS),
        backend=BACKEND,
        quant="gguf_ud_q4_k_xl",
        execution_profile="production",
        served_model_name="preload-profile-check",
        eager_load=False,
        startup_chat_smoke=False,
        startup_scratch_probe=False,
    )

    app = create_app(config, llm=llm)

    assert app.state.hipengine_llm is llm
    assert llm._text_generator is None
    assert llm.resolved_execution_profile == "production"
    assert llm.execution_profile_manifest_sha256
