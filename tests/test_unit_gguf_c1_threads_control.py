"""Bind short-C1 thread comparisons to the cached production leaf resolver."""
import os
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr
from scripts import execution_profile_gguf_c1_threads_gate as gate


@pytest.mark.parametrize("backend,default", [("hip_gfx1100", 256), ("hip_gfx1151", 1024)])
@pytest.mark.parametrize("override", [None, "256", "512", "1024", "not-an-integer"])
def test_thread_control_resolves_package_default_and_explicit_width(monkeypatch, backend, default, override):
    name = gate._ROUTE_ENV_KEYS[0]
    if override is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, override)
    expected = int(override) if override not in (None, "not-an-integer") else default
    assert qgr._gguf_short_c1_attn_threads(backend) == expected


@pytest.mark.parametrize("inherited", [None, "512", "1024"])
def test_gate_resets_cached_leaf_between_strict_and_candidate_legs(monkeypatch, inherited):
    name = gate._ROUTE_ENV_KEYS[0]
    if inherited is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, inherited)
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1151"
    session = SimpleNamespace(runner=runner)
    spans = object()
    strict = object()
    resolutions = []
    launches = []

    def resolve(**kwargs):
        resolutions.append(kwargs)
        return strict

    def launch(*args, **kwargs):
        launches.append((args, kwargs))

    monkeypatch.setattr(qgr, "resolve_paged_attn_decode", resolve)
    monkeypatch.setattr(qgr, "qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_threads_spans", launch)
    with pytest.raises(RuntimeError, match="injected leg failure"):
        with gate.route_environment(gate.STRICT_ENVIRONMENT):
            gate._reset_leaf_cache(session)
            assert runner._full_attn_decode_short_batch_fn(spans) is strict
        with gate.route_environment(gate.CANDIDATE_ENVIRONMENT):
            # A live session keeps its bound leaf until the diagnostic resets it.
            assert runner._full_attn_decode_short_batch_fn(spans) is strict
            gate._reset_leaf_cache(session)
            candidate = runner._full_attn_decode_short_batch_fn(spans)
            candidate(101, stream=7)
        with gate.route_environment(gate.STRICT_ENVIRONMENT):
            gate._reset_leaf_cache(session)
            assert runner._full_attn_decode_short_batch_fn(spans) is strict
            raise RuntimeError("injected leg failure")
    assert launches == [((101,), {"threads": 1024, "stream": 7})]
    assert resolutions == [{"backend": "hip_gfx1151", "spans": spans,
                            "kind": "context_batch", "model_quant": "w4_paro"}] * 2
    assert os.environ.get(name) == inherited
    gate._reset_leaf_cache(SimpleNamespace(runner=None))


@pytest.mark.parametrize("values", [{}, {"UNRELATED": "256"}])
def test_thread_gate_refuses_incomplete_or_unrelated_control_maps(values):
    with pytest.raises(ValueError, match="route environment mismatch"):
        with gate.route_environment(values):
            pytest.fail("invalid route map entered the comparison")
