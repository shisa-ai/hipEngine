from __future__ import annotations

import pytest

from hipengine.speculative.registry import (
    DEFAULT_PROVIDER_CANDIDATE_BUDGET,
    SpeculativeProviderBudgetAmbiguousError,
    SpeculativeProviderConfig,
    SpeculativeProviderKey,
    register_builtin_speculative_providers,
    register_speculative_provider,
    registered_speculative_providers,
    resolve_speculative_provider,
    resolve_speculative_provider_candidate_budget,
)


def _factory(*, target_generator, config):
    return target_generator, config


def test_speculative_provider_registry_resolves_all_concrete_axes() -> None:
    key = SpeculativeProviderKey(
        provider="test_dflash",
        target_model="test_model",
        backend="test_backend",
        quant="test_quant",
    )
    register_speculative_provider(key, _factory, replace=True)

    assert key in registered_speculative_providers()
    assert (
        resolve_speculative_provider(
            provider="test_dflash",
            target_model="test_model",
            backend="test_backend",
            quant="test_quant",
        )
        is _factory
    )
    with pytest.raises(KeyError, match="unregistered speculative provider"):
        resolve_speculative_provider(
            provider="test_dflash",
            target_model="test_model",
            backend="other_backend",
            quant="test_quant",
        )


def test_provider_declared_budget_resolves_without_model_axes() -> None:
    key = SpeculativeProviderKey(
        provider="test_declared_budget",
        target_model="test_model",
        backend="test_backend",
        quant="test_quant",
    )
    register_speculative_provider(
        key, _factory, replace=True, candidate_budget=7
    )

    assert resolve_speculative_provider_candidate_budget("test_declared_budget") == 7
    assert (
        resolve_speculative_provider_candidate_budget(
            "test_declared_budget",
            target_model="test_model",
            backend="test_backend",
            quant="test_quant",
        )
        == 7
    )


def test_provider_registration_without_a_budget_keeps_the_generic_default() -> None:
    key = SpeculativeProviderKey(
        provider="test_default_budget",
        target_model="test_model",
        backend="test_backend",
        quant="test_quant",
    )
    register_speculative_provider(key, _factory, replace=True)

    assert (
        resolve_speculative_provider_candidate_budget("test_default_budget")
        == DEFAULT_PROVIDER_CANDIDATE_BUDGET
    )


def test_provider_declared_budget_requires_axes_when_declarations_differ() -> None:
    for suffix, budget in (("a", 3), ("b", 7)):
        register_speculative_provider(
            SpeculativeProviderKey(
                provider="test_axis_budget",
                target_model=f"model_{suffix}",
                backend=f"backend_{suffix}",
                quant=f"quant_{suffix}",
            ),
            _factory,
            replace=True,
            candidate_budget=budget,
        )

    with pytest.raises(
        SpeculativeProviderBudgetAmbiguousError, match="multiple candidate budgets"
    ):
        resolve_speculative_provider_candidate_budget("test_axis_budget")
    assert (
        resolve_speculative_provider_candidate_budget(
            "test_axis_budget", target_model="model_a"
        )
        == 3
    )
    assert (
        resolve_speculative_provider_candidate_budget(
            "test_axis_budget",
            target_model="model_b",
            backend="backend_b",
            quant="quant_b",
        )
        == 7
    )
    with pytest.raises(KeyError, match="unregistered speculative provider"):
        resolve_speculative_provider_candidate_budget(
            "test_axis_budget", target_model="model_c"
        )


def test_provider_declared_budget_rejects_an_unregistered_provider() -> None:
    with pytest.raises(KeyError, match="unregistered speculative provider"):
        resolve_speculative_provider_candidate_budget("test_not_registered")


def test_builtin_registry_contains_laguna_dflash_concrete_key() -> None:
    register_builtin_speculative_providers()

    assert SpeculativeProviderKey(
        provider="dflash",
        target_model="laguna_gguf",
        backend="hip_gfx1151",
        quant="gguf_q4_k_m",
    ) in registered_speculative_providers()


@pytest.mark.parametrize(
    ("provider", "target_model", "backend", "quant"),
    (
        ("dflash", "laguna_gguf", "hip_gfx1151", "gguf_q4_k_m"),
        ("gemma4_mtp", "gemma4_gguf", "hip_gfx1151", "gguf_q4_k_m"),
        ("qwen4_exp_mtp", "qwen4_exp_gguf", "hip_gfx1151", "gguf_q4_k_m"),
    ),
)
def test_builtin_provider_declarations_resolve_by_axes(
    provider, target_model, backend, quant
) -> None:
    assert (
        resolve_speculative_provider_candidate_budget(
            provider,
            target_model=target_model,
            backend=backend,
            quant=quant,
        )
        == 4
    )


def test_speculative_provider_config_requires_explicit_drafter_and_budget() -> None:
    config = SpeculativeProviderConfig(
        provider="dflash",
        draft_model="/models/drafter",
        candidate_budget=4,
    )

    assert config.provider == "dflash"
    assert str(config.draft_model) == "/models/drafter"
    assert config.candidate_budget == 4
    with pytest.raises(ValueError, match="provider"):
        SpeculativeProviderConfig(provider="", draft_model="/models/drafter")
    with pytest.raises(ValueError, match="draft_model"):
        SpeculativeProviderConfig(provider="dflash", draft_model="")
    with pytest.raises(ValueError, match="candidate_budget"):
        SpeculativeProviderConfig(
            provider="dflash",
            draft_model="/models/drafter",
            candidate_budget=0,
        )
