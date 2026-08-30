"""The Q4_K rowtile decode route is env-selectable, CPU-only.

Why this exists: `HIPENGINE_GGUF_Q4K_ROWTILE` gates the rows 2..8 rowtile chunking whose absence
makes Q4/Q5 projections "silently fall back to WMMA prefill" (hipengine/runtime/gguf_linear.py:162).
Before spending a server-packet A/B on the route choice, prove the switch actually changes the
resolved kernel - otherwise the experiment is a null result with a GPU bill.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LINEAR = "hipengine.runtime.gguf_linear"


def _rowtile_resolver():
    module = sys.modules.get(LINEAR) or __import__(LINEAR, fromlist=["*"])
    return module


def _sibling_weight(rows_ptr: int):
    """Reuse the Q4_K dense module's fake device-weight builder (no GPU, no fixtures)."""
    path = REPO / "tests" / "test_gguf_q4_k_t16_dense.py"
    spec = importlib.util.spec_from_file_location("q4k_dense_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._weight(rows_ptr)


@pytest.mark.parametrize("value", ["", "1", "0"])
def test_resolve_reads_the_env_directly(monkeypatch, value):
    module = _rowtile_resolver()
    if value == "":
        monkeypatch.delenv("HIPENGINE_GGUF_Q4K_ROWTILE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", value)
    expected = True if value in ("", "1") else False
    assert module._resolve_use_q4k_rowtile(None) is expected


def test_dense_rows_above_one_resolver_ignores_the_knob(monkeypatch):
    """The resolver cannot be used to test the knob.

    For LAYOUT_GGUF_Q4_K_T16 any rows>1 resolves `t16_wmma_prefill_bf16_bf16_out`
    unconditionally (resolve_gguf_linear_dispatch, gguf_linear.py:2185). The rowtile
    decision is made later, at the launch sites - `launch_gguf_linear` (:2447),
    `launch_gguf_linear_pair` (:4182), and `launch_gguf_linear_pair_silu` (:4658) each call
    `_resolve_use_q4k_rowtile(None)`. This asserts the negative so nobody writes a
    resolver-level test and concludes the knob is dead: the widest rowtile kernel in the C2
    trace is a `dense_dual_rowtile_*` (pair-route) launch, which this function never names.
    """
    module = _rowtile_resolver()
    weight = _sibling_weight(0x2000)

    monkeypatch.delenv("HIPENGINE_GGUF_Q4K_ROWTILE", raising=False)
    on = module.resolve_gguf_linear_dispatch(weight, rows=2)
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "0")
    off = module.resolve_gguf_linear_dispatch(weight, rows=2)
    assert on.key == off.key
    assert on.key.variant == "t16_wmma_prefill_bf16_bf16_out"


def test_knob_is_consumed_by_the_launch_sites_not_the_resolver():
    """Pin the call sites, since that is where an A/B on the env actually reaches."""
    module = _rowtile_resolver()
    source = pathlib.Path(module.__file__).read_text()
    sites = [
        line.strip()
        for line in source.splitlines()
        if "_resolve_use_q4k_rowtile(" in line and not line.lstrip().startswith("def ")
    ]
    assert len(sites) == 3, f"expected 3 consumers, found {sites}"
    assert sum("use_rowtile=" in s for s in sites) == 1
    assert sum("f_rowtile" in s for s in sites) == 1


def test_rows_1_route_is_outside_the_knob(monkeypatch):
    # rows==1 has its own single-row decode family; the rowtile chunker cannot serve it at all.
    module = _rowtile_resolver()
    weight = _sibling_weight(0x2001)

    monkeypatch.delenv("HIPENGINE_GGUF_Q4K_ROWTILE", raising=False)
    on = module.resolve_gguf_linear_dispatch(weight, rows=1)
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "0")
    off = module.resolve_gguf_linear_dispatch(weight, rows=1)
    assert on.key == off.key
    assert "rowtile" not in on.key.variant


def test_knob_stops_mattering_above_the_rowtile_cap(monkeypatch):
    module = _rowtile_resolver()
    weight = _sibling_weight(0x2002)

    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "1")
    above = module.resolve_gguf_linear_dispatch(weight, rows=module._ROWTILE_MAX_ROWS + 1)
    assert "rowtile" not in above.key.variant
    assert module._ROWTILE_MIN_ROWS == 2 and module._ROWTILE_MAX_ROWS == 8


def test_session_pinning_beats_the_env(monkeypatch):
    """Why an env A/B on the server route measured nothing (measured on GPU, 2026-08-30).

    A width-2 AR packet with HIPENGINE_GGUF_Q4K_ROWTILE=0 traced 164,798 rowtile launches against
    164,460 with the default - the same route. The resident serving route opens the session, and the
    session wins over the environment, so the env only moves plain bench/diagnostic calls.
    """
    module = _rowtile_resolver()
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "0")
    assert module._resolve_use_q4k_rowtile(None) is False
    with module.q4k_rowtile_session(True):
        assert module._resolve_use_q4k_rowtile(None) is True
        with module.q4k_rowtile_session(False):
            assert module._resolve_use_q4k_rowtile(None) is False
        assert module._resolve_use_q4k_rowtile(None) is True
    assert module._resolve_use_q4k_rowtile(None) is False


def test_explicit_kwarg_beats_the_session(monkeypatch):
    module = _rowtile_resolver()
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "1")
    with module.q4k_rowtile_session(True):
        assert module._resolve_use_q4k_rowtile(False) is False
