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


@pytest.mark.parametrize("rows", [17, 127, 128, 129, 256, 512, 1024, 2048])
@pytest.mark.parametrize("start", [0, 4095, 4096, 4231])
@pytest.mark.parametrize("enabled,bounded,short_route,full_attention", tuple(product([False, True], repeat=4)))
def test_long_attention_route_preserves_existing_selection(rows, start, enabled, bounded, short_route, full_attention):
    expected = enabled and not short_route and full_attention and rows in {128, 256, 512, 1024, 2048}
    expected = expected and (rows == 128 or bounded) and (rows > 128 or start >= 4096)
    result = policy.select_laguna_long_attention_route(
        enabled=enabled, bounded=bounded, short_route=short_route,
        full_attention=full_attention, rows=rows, start=start, context_length=8192,
        q_heads=48, kv_heads=8, head_dim=128,
    )
    assert result == (("bounded" if bounded else "full_score") if expected else None)


@pytest.mark.parametrize("override", [
    {"context_length": 4223}, {"q_heads": 47}, {"kv_heads": 4}, {"head_dim": 64},
])
def test_long_attention_route_rejects_incompatible_capacity_or_geometry(override):
    args = dict(enabled=True, bounded=True, short_route=False, full_attention=True,
                rows=128, start=4096, context_length=8192, q_heads=48, kv_heads=8, head_dim=128)
    args.update(override)
    assert policy.select_laguna_long_attention_route(**args) is None
