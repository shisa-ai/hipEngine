"""Independent cached-prefill settings preserve package and explicit policy."""

from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from hipengine.runtime import laguna_prefill_policy as policy


@pytest.mark.parametrize("name", ["kv_preappend", "cached_meta", "global_qrow6", "dense_initial"])
def test_explicit_policy_rejects_ambiguous_legacy_overrides(name):
    resolved = policy.LagunaCachedPrefillPolicy(True, False, True, False)
    with pytest.raises(ValueError, match="legacy overrides"):
        policy.resolve_laguna_cached_prefill_policy("peer", policy=resolved, **{name: False})


def test_explicit_policy_is_reused_without_backend_reads(monkeypatch):
    resolved = policy.LagunaCachedPrefillPolicy(True, False, True, False)
    monkeypatch.setattr(policy, "backend_package_capability", lambda *args: pytest.fail("unexpected lookup"))
    assert policy.resolve_laguna_cached_prefill_policy("peer", policy=resolved) is resolved
    with pytest.raises(TypeError, match="LagunaCachedPrefillPolicy"):
        policy.resolve_laguna_cached_prefill_policy("peer", policy=object())


@pytest.mark.parametrize("values", list(product((None, False, True), repeat=4)))
def test_cached_prefill_policy_preserves_independent_overrides(monkeypatch, values):
    names = ("kv_preappend", "cached_meta", "global_qrow6", "dense_initial")
    defaults = dict(zip(names, (True, False, True, False)))
    calls = []

    def capability(backend, key, fallback):
        assert backend == "peer_backend"
        calls.append(key)
        return defaults[key.removeprefix("LAGUNA_PREFILL_").lower()]

    monkeypatch.setattr(policy, "backend_package_capability", capability)
    result = policy.resolve_laguna_cached_prefill_policy("peer_backend", **dict(zip(names, values)))
    for name, value in zip(names, values):
        assert getattr(result, name) is (defaults[name] if value is None else value)
    assert calls == ["LAGUNA_PREFILL_" + name.upper()
                     for name, value in zip(names, values) if value is None]
    with pytest.raises(FrozenInstanceError):
        result.cached_meta = True
