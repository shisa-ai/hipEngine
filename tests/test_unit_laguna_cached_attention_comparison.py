"""Host routing contracts for Laguna cached-attention comparisons."""

from itertools import product
from types import SimpleNamespace

import pytest

from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION
from hipengine.runtime import laguna_kv as kv
from scripts import (
    laguna_attention_cached_meta_ab as cached_meta,
    laguna_attention_dense_initial_ab as dense_initial,
    laguna_attention_global_qrow6_ab as global_qrow6,
)


def _cache(attention_type, start, rows=128):
    cache = object.__new__(kv.LagunaKVCache)
    cache._closed = False
    cache.context_length = 4096
    cache.sliding_window = 512
    cache.runtime = object()
    cache._pending_positions = tuple(range(start, start + rows))
    cache._dense_initial_metadata_valid = True
    cache.prefill_preappend_role_scoped = False
    cache.prefill_preappend_role_variants = {}
    cache.layers = (SimpleNamespace(
        attention_type=attention_type, capacity=4096 if attention_type == FULL_ATTENTION else 512,
        q_heads=48 if attention_type == FULL_ATTENTION else 72,
        key_cache=SimpleNamespace(ptr=0x1000), value_cache=SimpleNamespace(ptr=0x2000),
        spans=SimpleNamespace(spans_mode="paged_dense" if attention_type == FULL_ATTENTION else "sliding_ring"),
    ),)
    return cache


@pytest.mark.parametrize("attention_type", [FULL_ATTENTION, SLIDING_ATTENTION])
@pytest.mark.parametrize("start", [0, 127, 128, 257])
@pytest.mark.parametrize("meta,qrow6,dense", list(product([False, True], repeat=3)))
def test_cached_attention_flags_select_real_variants_without_changing_launch_ownership(attention_type, start, meta, qrow6, dense):
    cache = _cache(attention_type, start)
    cache.prefill_cached_meta, cache.prefill_global_qrow6, cache.prefill_dense_initial = meta, qrow6, dense
    calls = []
    cache._resolve = lambda layer, variant: lambda *args, **kwargs: calls.append((layer, variant, args, kwargs))
    cache.attend_prefill_cached(0, 10, 20, 30, 40, 128, stream=7, library="library")
    if attention_type == FULL_ATTENTION:
        expected = (
            "global_context_rows_qrow6_dense_initial_online_spans" if dense and qrow6 and start >= 128
            else "global_context_rows_qrow4_dense_initial_online_spans" if dense
            else "global_context_rows_qrow6_cached_meta_online_spans" if meta and qrow6 and start >= 128
            else "global_context_rows_qrow4_cached_meta_online_spans" if meta and start >= 128
            else "global_context_rows_qrow4_cached_online_spans"
        )
    else:
        expected = (
            "swa_context_rows_qrow4_dense_initial_online_spans" if dense
            else "swa_context_rows_qrow4_cached_meta_online_spans" if meta
            else "swa_context_rows_qrow4_cached_online_spans"
        )
    layer, variant, args, kwargs = calls.pop()
    assert (layer, variant) == ("laguna_attention_prefill", expected)
    assert args[:8] == (10, 20, 30, 0x1000, 0x2000, 40, cache.layers[0].spans, 128)
    assert kwargs["stream"] == 7 and kwargs["runtime"] is cache.runtime and kwargs["library"] == "library"
    assert kwargs.get("start_position") == (start if dense or attention_type == SLIDING_ATTENTION else None)


@pytest.mark.parametrize("attention_type,start,rows,expected", [
    (FULL_ATTENTION, 0, 127, False), (FULL_ATTENTION, 0, 128, True),
    (FULL_ATTENTION, 0, 129, False), (FULL_ATTENTION, 123, 256, True),
    (FULL_ATTENTION, 0, 512, True), (FULL_ATTENTION, 2048, 2048, True),
    (FULL_ATTENTION, 2049, 2048, False), (SLIDING_ATTENTION, 384, 128, True),
    (SLIDING_ATTENTION, 385, 128, False), (SLIDING_ATTENTION, 0, 256, False),
])
def test_preappend_capability_preserves_partial_wrap_and_wide_global_contracts(attention_type, start, rows, expected):
    cache = _cache(attention_type, start, rows)
    assert cache.can_preappend_attention_prefill(0, rows) is expected
    assert cache.can_dense_initial_prefill(0, rows) is expected
    cache._dense_initial_metadata_valid = False
    assert not cache.can_dense_initial_prefill(0, rows)
    assert cache.can_preappend_attention_prefill(0, rows) is expected


@pytest.mark.parametrize("attention_type,role,variant", [
    (FULL_ATTENTION, kv._PREFILL_DENSE_INITIAL_GLOBAL_ROLE, kv._DENSE_INITIAL_CACHED_GLOBAL_PREFILL_VARIANT),
    (SLIDING_ATTENTION, kv._PREFILL_PREAPPEND_SWA_QROW4_ROLE, kv._DENSE_INITIAL_CACHED_SWA_PREFILL_VARIANT),
])
def test_package_role_precedes_legacy_flags_and_retains_metadata_admission(attention_type, role, variant):
    cache = _cache(attention_type, 128)
    cache.prefill_cached_meta = cache.prefill_global_qrow6 = cache.prefill_dense_initial = False
    cache.prefill_preappend_role_scoped = True
    cache.prefill_preappend_role_variants = {role: variant}
    calls = []
    cache._resolve = lambda layer, selected: lambda *args, **kwargs: calls.append(selected)
    assert cache.can_preappend_attention_prefill(0, 128)
    cache.attend_prefill_cached(0, 10, 20, 30, 40, 128)
    assert calls == [variant]
    cache._dense_initial_metadata_valid = False
    assert not cache.can_preappend_attention_prefill(0, 128)


@pytest.mark.parametrize("module,flag,fixed", [
    (cached_meta, "prefill_cached_meta", {}),
    (global_qrow6, "prefill_global_qrow6", {"prefill_cached_meta": True}),
    (dense_initial, "prefill_dense_initial", {"prefill_cached_meta": True, "prefill_global_qrow6": True}),
])
@pytest.mark.parametrize("enabled", [False, True])
def test_attention_comparison_child_owns_only_its_declared_overrides(monkeypatch, module, flag, fixed, enabled):
    calls = []
    monkeypatch.setattr(module, "_compiler_version", lambda path: "synthetic")
    monkeypatch.setattr(module, "LagunaGGUFResidentSession", lambda **kwargs: calls.append(kwargs))
    owner = SimpleNamespace(weights=object(), runtime=object())
    module._child_session(owner, **{flag: enabled})
    kwargs = calls.pop()
    assert kwargs[flag] is enabled
    assert {name: value for name, value in kwargs.items() if name in {"prefill_cached_meta", "prefill_global_qrow6", "prefill_dense_initial"}} == {flag: enabled, **fixed}
    assert kwargs["resident_weights"] is owner.weights and kwargs["runtime"] is owner.runtime
    assert kwargs["backend"] == "hip_gfx1151" and kwargs["require_cached_build"] is True
