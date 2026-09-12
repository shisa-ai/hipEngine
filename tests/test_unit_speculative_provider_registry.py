from __future__ import annotations

import pytest

from hipengine.speculative.registry import (
    SpeculativeProviderConfig,
    SpeculativeProviderKey,
    register_builtin_speculative_providers,
    register_speculative_provider,
    registered_speculative_providers,
    resolve_speculative_provider,
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


def test_builtin_registry_contains_laguna_dflash_concrete_key() -> None:
    register_builtin_speculative_providers()

    assert SpeculativeProviderKey(
        provider="dflash",
        target_model="laguna_gguf",
        backend="hip_gfx1151",
        quant="gguf_q4_k_m",
    ) in registered_speculative_providers()


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
