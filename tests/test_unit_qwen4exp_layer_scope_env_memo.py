"""The layer-scope env parsers must be memoized on the raw string, not the parse.

`launch_gguf_linear` calls the layer-scope resolvers **once per dense launch** -
545 times in a single `code-p4096` decode step, measured by `--host-profile` on
the canonical bench. Each call re-reads its environment variable and re-parses
it (`raw.split(",")`, `part.strip()` and `int(part)` per entry), so with the
production scope bound to 32 layers that is ~17,000 `str.strip` calls and ~17,000
`int()` calls per decode step for a value that cannot have changed.

A memo keyed on the raw string is exact - each resolver is a pure function of
that string - but it must still observe an environment change, so these tests
pin both halves: the parse is shared, and a changed value takes effect on the
next call with no cache reset.
"""

from __future__ import annotations

import os

import pytest


def test_parse_layer_scope_matches_the_uncached_parse() -> None:
    """The shared parse must agree with a fresh parse for every spelling."""

    from hipengine.runtime import gguf_linear

    parse = gguf_linear._parse_layer_scope
    for raw in ("", " ", " , ,", "0", "16,17,18", "16, 17 ,18", "47", "0,0,1"):
        expected = {int(part) for part in raw.split(",") if part.strip()}
        assert parse(raw) == (frozenset(expected) if expected else None), raw


def test_layer_scope_memo_is_shared_across_resolvers() -> None:
    """All three scopes use one parse, so one value must not leak into another."""

    from hipengine.runtime import gguf_linear

    names = (
        (gguf_linear._Q8_DENSE_WIDE_LAYERS_ENV, gguf_linear._q8_dense_wide_layers),
        (gguf_linear._Q8_WMMA_LAYERS_ENV, gguf_linear._q8_wmma_layers_scope),
        (gguf_linear._Q8_STAGED_F32_LAYERS_ENV, gguf_linear._q8_staged_f32_layers),
    )
    saved = {name: os.environ.get(name) for name, _ in names}
    try:
        for index, (name, resolver) in enumerate(names):
            os.environ[name] = f"{index},{index + 1}"
        for index, (name, resolver) in enumerate(names):
            assert resolver() == frozenset({index, index + 1}), name
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_layer_scope_memo_observes_an_environment_change() -> None:
    """A changed variable must take effect on the next call, with no cache reset."""

    from hipengine.runtime import gguf_linear

    name = gguf_linear._Q8_DENSE_WIDE_LAYERS_ENV
    saved = os.environ.get(name)
    try:
        os.environ[name] = "16,17"
        assert gguf_linear._q8_dense_wide_layers() == frozenset({16, 17})
        os.environ[name] = "18,19"
        assert gguf_linear._q8_dense_wide_layers() == frozenset({18, 19})
        os.environ.pop(name, None)
        assert gguf_linear._q8_dense_wide_layers() is None
        os.environ[name] = "20"
        assert gguf_linear._q8_dense_wide_layers() == frozenset({20})
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


def test_layer_scope_memo_parses_once_per_distinct_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated calls with one value must add one cache entry and return one object."""

    from hipengine.runtime import gguf_linear

    name = gguf_linear._Q8_DENSE_WIDE_LAYERS_ENV
    monkeypatch.setenv(name, "31337,31338,31339")
    cache = gguf_linear._LAYER_SCOPE_PARSE_CACHE
    before = len(cache)
    results = [gguf_linear._q8_dense_wide_layers() for _ in range(50)]
    assert all(result == frozenset({31337, 31338, 31339}) for result in results)
    assert len(cache) == before + 1, "50 calls with one value must add one cache entry"
    assert all(result is results[0] for result in results), "the memo must return one object"

    # The memo is keyed on the raw string, so a new value is a miss - and the
    # *caller* is what reads the environment, so no cache reset is needed.
    monkeypatch.setenv(name, "31340")
    assert gguf_linear._q8_dense_wide_layers() == frozenset({31340})
    assert len(cache) == before + 2
    # Returning to the first value is a hit again, not a new entry.
    monkeypatch.setenv(name, "31337,31338,31339")
    assert gguf_linear._q8_dense_wide_layers() is results[0]
    assert len(cache) == before + 2


def test_env_flag_memo_observes_an_environment_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag resolver must re-read on every call and honour a change."""

    from hipengine.runtime import gguf_linear

    monkeypatch.setenv(gguf_linear._WMMA_PREFILL_ENV, "1")
    assert gguf_linear._env_wmma_prefill_enabled() is True
    monkeypatch.setenv(gguf_linear._WMMA_PREFILL_ENV, "0")
    assert gguf_linear._env_wmma_prefill_enabled() is False
    monkeypatch.setenv(gguf_linear._WMMA_PREFILL_ENV, " YES ")
    assert gguf_linear._env_wmma_prefill_enabled() is True
    monkeypatch.setenv(gguf_linear._WMMA_PREFILL_ENV, "")
    assert gguf_linear._env_wmma_prefill_enabled() is False
    monkeypatch.delenv(gguf_linear._WMMA_PREFILL_ENV)
    assert gguf_linear._env_wmma_prefill_enabled() is False


def test_raw_env_memo_returns_the_current_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw-value memo must never return a superseded value."""

    from hipengine.runtime import gguf_linear

    name = "HIPENGINE_TEST_RAW_ENV_MEMO"
    monkeypatch.setenv(name, "alpha")
    assert gguf_linear._raw_env_value(name, "fallback") == "alpha"
    assert gguf_linear._raw_env_value(name, "fallback") == "alpha"
    monkeypatch.setenv(name, "beta")
    assert gguf_linear._raw_env_value(name, "fallback") == "beta"
    monkeypatch.delenv(name)
    assert gguf_linear._raw_env_value(name, "fallback") == "fallback"
    # The two launch-key flags must go through it rather than re-reading inline.
    monkeypatch.setenv(gguf_linear._Q8_DPP_REDUCTION_ENV, "0")
    assert gguf_linear._q8_dpp_reduction_raw() == "0"
    monkeypatch.setenv(gguf_linear._Q8_DPP_REDUCTION_ENV, "1")
    assert gguf_linear._q8_dpp_reduction_raw() == "1"
