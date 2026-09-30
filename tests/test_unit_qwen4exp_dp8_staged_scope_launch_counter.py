"""The DP-8 staged-scope prefill A/B harness must be able to see its own route.

The harness measures the end-to-end prefill value of the staged-f32 layer scope
by switching arms in-process. Its first engagement instrument wrapped
``resolve_gguf_linear_dispatch``, which is *inside* the dispatch memoization it
was meant to observe: ``launch_gguf_linear`` caches the entire selector chain in
``_DISPATCH_RESOLVE_CACHE``, the warmups fill that cache for both arms, and the
measured repetitions therefore never call the resolver. The counter read zero in
both arms and the harness refused a run whose route had engaged.

These tests pin the replacement instrument at the level that is not memoized -
the cached kernel callable - and pin the two failure modes that produced the
false refusal: a counter that silently reads zero, and a counter that silently
undercounts when a dispatch-cache entry appears after it was installed.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from hipengine.runtime import gguf_linear
from scripts.qwen4exp_dp8_staged_scope_prefill_ab import (
    STAGED_VARIANT,
    BenchmarkError,
    _layer_census,
    _staged_layer_of,
    _wrap_cached_launches,
)

PARENT_VARIANT = "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out"


@pytest.fixture
def dispatch_cache() -> Any:
    """Give each test a private dispatch cache and restore the real one."""

    saved = dict(gguf_linear._DISPATCH_RESOLVE_CACHE)
    gguf_linear._DISPATCH_RESOLVE_CACHE.clear()
    try:
        yield gguf_linear._DISPATCH_RESOLVE_CACHE
    finally:
        gguf_linear._DISPATCH_RESOLVE_CACHE.clear()
        gguf_linear._DISPATCH_RESOLVE_CACHE.update(saved)


def _cache_key(*, layer: int | None, generation: str = "gen") -> tuple[Any, ...]:
    """A key with the shape ``launch_gguf_linear`` actually builds.

    ``(generation, dense_wide_key, staged_f32_key, wmma_scope_key, ...)`` with
    ``staged_f32_key == (scope, layer_index)``. The scope is bound exactly when
    the layer index is present, which is what the layer attribution reads.
    """

    scope = None if layer is None else frozenset({13, 14, 15})
    return (generation, (True, frozenset(range(16, 48)), layer), (scope, layer), None, "raw")


def _install(
    cache: Any,
    *,
    layer: int | None,
    variant: str,
    fn: Any,
    generation: str = "gen",
) -> tuple[Any, ...]:
    key = _cache_key(layer=layer, generation=generation)
    cache[key] = ("raw", fn, "linear", "gguf_q8_0", variant)
    return key


def test_counter_sees_launches_through_a_warm_cache(dispatch_cache: Any) -> None:
    """A cache hit must still be counted; that is the whole point.

    The regression this pins: counting at the resolver reads zero once the
    warmups have filled the cache, which the harness reported as "the route did
    not run".
    """

    calls: list[tuple[Any, ...]] = []

    def fn(*args: Any, **kwargs: Any) -> str:
        calls.append(args)
        return "launched"

    staged_key = _install(
        dispatch_cache, layer=13, variant=STAGED_VARIANT, fn=fn
    )
    parent_key = _install(
        dispatch_cache, layer=None, variant=PARENT_VARIANT, fn=fn
    )

    by_variant: Counter = Counter()
    by_key: Counter = Counter()
    identity: dict[Any, tuple[int | None, str]] = {}
    assert _wrap_cached_launches(by_variant, by_key, identity) == (2, 2)

    # Two launches per key, straight out of the cache, as launch_gguf_linear
    # does on a hit.
    for _ in range(2):
        for key in (staged_key, parent_key):
            dispatch_cache[key][1](1, 2, stream=0)

    assert by_variant[STAGED_VARIANT] == 2
    assert by_variant[PARENT_VARIANT] == 2
    assert len(calls) == 4
    # The wrapper must forward arguments unchanged, or it would change what is
    # being measured.
    assert calls[0] == (1, 2)
    assert _layer_census(by_key, identity) == {
        "13:" + STAGED_VARIANT: 2,
        "unattributed:" + PARENT_VARIANT: 2,
    }


def test_wrapping_is_idempotent_and_reports_only_new_entries(
    dispatch_cache: Any,
) -> None:
    """Re-installing per arm must not double-count or re-wrap.

    The harness re-installs before every measured arm so a key created by an
    earlier arm is covered. A second call must wrap nothing, and a call after a
    new key appears must report exactly that one.
    """

    def fn(*args: Any, **kwargs: Any) -> None:
        return None

    _install(dispatch_cache, layer=13, variant=STAGED_VARIANT, fn=fn)
    by_variant: Counter = Counter()
    by_key: Counter = Counter()
    identity: dict[Any, tuple[int | None, str]] = {}

    assert _wrap_cached_launches(by_variant, by_key, identity) == (1, 1)
    assert _wrap_cached_launches(by_variant, by_key, identity) == (0, 1)

    _install(dispatch_cache, layer=14, variant=STAGED_VARIANT, fn=fn)
    # A non-zero first element is the harness's signal that the arm it just
    # measured was counted incompletely; it must not be swallowed.
    assert _wrap_cached_launches(by_variant, by_key, identity) == (1, 2)
    assert _wrap_cached_launches(by_variant, by_key, identity) == (0, 2)


def test_malformed_cache_entry_raises_instead_of_undercounting(
    dispatch_cache: Any,
) -> None:
    """An entry the instrument cannot read is an error, not a silent zero."""

    dispatch_cache[("bad",)] = ("raw", lambda *a, **k: None)
    with pytest.raises(BenchmarkError, match="5-tuple"):
        _wrap_cached_launches(Counter(), Counter(), {})


@pytest.mark.parametrize(
    "key",
    [
        ("gen", (True, frozenset({16}), 13), (None, None), None, "raw"),  # scope off
        ("gen", "not-a-tuple"),  # too short
        ("gen", (1,), ("not-a-pair",), None, "raw"),  # wrong staged-key shape
        ("gen", (1,), (frozenset({13}), "13"), None, "raw"),  # layer not an int
        ("gen", (1,), (frozenset({13}), True), None, "raw"),  # bool is not a layer
        "not-a-tuple",
    ],
)
def test_layer_attribution_declines_rather_than_guesses(key: Any) -> None:
    """An unrecognized key must report unattributed, never a fabricated layer."""

    assert _staged_layer_of(key) is None


def test_layer_attribution_reads_the_bound_scope(dispatch_cache: Any) -> None:
    """The layer index is recoverable exactly while a scope is bound."""

    assert _staged_layer_of(_cache_key(layer=15)) == 15
    assert _staged_layer_of(_cache_key(layer=None)) is None
