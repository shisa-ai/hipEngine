"""WMMA scope diagnostics must isolate every dispatch input, not only its key."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.runtime import gguf_linear as gl
from hipengine.runtime import qwen4_exp_runner as qr


def test_scope_contexts_isolate_enablement_filter_and_key(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "0")
    monkeypatch.setenv(gl._Q8_WMMA_LAYERS_ENV, "41")
    barrier = Barrier(2)
    entered = Event()
    release = Event()
    scoped = gl.Q8WMMALayerScope.parse("20")
    empty = gl.Q8WMMALayerScope.empty()

    def probe(scope, expected):
        if scope == empty:
            assert entered.wait(timeout=5)
        with qr._q8_wmma_scope_sessions(scope):
            if scope == scoped:
                entered.set()
            barrier.wait(timeout=5)
            if scope == scoped:
                # Ensure the empty context has entered after this context.
                assert release.wait(timeout=5)
            else:
                release.set()
            assert gl._q8_wmma_layers_scope() == scope.layers
            assert gl.gguf_wmma_prefill_enabled() is expected
            selector = gl._wmma_prefill_weight_filter.get()
            assert (selector is None) is scope.is_empty
            if selector:
                assert selector(SimpleNamespace(spec=SimpleNamespace(
                    quant_key="gguf_q8_0", slot_path="layers.20.attn_qkv.weight")))
            barrier.wait(timeout=5)
        return gl._q8_wmma_layer_scope_bound.get()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(probe, scoped, True)
        second = pool.submit(probe, empty, False)
        assert first.result(timeout=10) is None
        assert second.result(timeout=10) is None


@pytest.mark.parametrize("raw", ["", "0", "20,21"])
def test_late_runner_inherits_the_explicit_scope(raw):
    scope = gl.Q8WMMALayerScope.parse(raw)
    generator = SimpleNamespace(q8_wmma_layer_scope=scope)
    late = _inherit_lane_decisions(generator, SimpleNamespace())
    assert late.q8_wmma_layer_scope is scope


def test_empty_scope_preserves_generic_enablement_and_kwarg_precedence(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "1")
    with qr._q8_wmma_scope_sessions(gl.Q8WMMALayerScope.empty()):
        assert gl.gguf_wmma_prefill_enabled() is True
        assert gl.gguf_wmma_prefill_enabled(False) is False
    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "0")
    with qr._q8_wmma_scope_sessions(gl.Q8WMMALayerScope.empty()):
        assert gl.gguf_wmma_prefill_enabled() is False
        assert gl.gguf_wmma_prefill_enabled(True) is True


def test_cached_launch_uses_scope_and_layer_without_reselecting(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "0")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "0")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS", "")
    monkeypatch.setattr(gl, "_DISPATCH_RESOLVE_CACHE", {})
    monkeypatch.setattr(gl, "generation", lambda: 17)
    monkeypatch.setattr(gl, "_ensure_linear_kernel_registered", lambda key: None)
    monkeypatch.setattr(gl, "resolve", lambda **kwargs: object())
    calls = []
    selections = []
    actual_select = gl._wmma_prefill_dispatch

    def select(dispatch, **kwargs):
        selections.append(kwargs["use_wmma"])
        return actual_select(dispatch, **kwargs)

    monkeypatch.setattr(gl, "_wmma_prefill_dispatch", select)
    for abi in ("raw", "wmma_raw"):
        monkeypatch.setitem(gl._LAUNCH_ABI, abi, lambda *args, _abi=abi: calls.append(_abi))
    weight = SimpleNamespace(
        spec=SimpleNamespace(layout="raw_gguf", quant_key="gguf_q8_0",
                             slot_path="layers.20.attn_qkv.weight"),
        backend="hip_gfx1151",
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=4096)),
    )
    scopes = [gl.Q8WMMALayerScope.parse(raw) for raw in ("20", "", "0", "20")]
    for scope in scopes:
        with qr._q8_wmma_scope_sessions(scope):
            gl.launch_gguf_linear(weight, 100, 200, 17, 2560, 10240,
                                 output_dtype=gl.GGUF_OUTPUT_F32, libraries={})
    assert calls == ["wmma_raw", "raw", "raw", "wmma_raw"]
    # The final candidate replays the populated entry, not a fresh selection.
    assert selections == [True, False, False]
    assert len(gl._DISPATCH_RESOLVE_CACHE) == 3


def test_inner_explicit_wmma_session_preserves_precedence(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "0")
    with qr._q8_wmma_scope_sessions(gl.Q8WMMALayerScope.parse("20")):
        assert gl.gguf_wmma_prefill_enabled() is True
        with gl.wmma_prefill_session(False):
            assert gl.gguf_wmma_prefill_enabled() is False
            assert gl.gguf_wmma_prefill_enabled(True) is True
        with gl.wmma_prefill_session(None):
            assert gl.gguf_wmma_prefill_enabled() is False
        assert gl.gguf_wmma_prefill_enabled() is True


@pytest.mark.parametrize("first,second", [("", "0"), ("0", "")])
def test_independent_owner_construction_order_and_reuse(monkeypatch, first, second):
    monkeypatch.setenv(gl._Q8_WMMA_LAYERS_ENV, "41")
    owners = [SimpleNamespace(q8_wmma_layer_scope=gl.Q8WMMALayerScope.parse(raw))
              for raw in (first, second)]
    for owner in (owners[0], owners[1], owners[0]):
        expected = owner.q8_wmma_layer_scope
        with qr._q8_wmma_scope_sessions(gl.resolve_q8_wmma_layers(owner)):
            assert gl._q8_wmma_layers_scope() == expected.layers
        assert gl.resolve_q8_wmma_layers(owner) == expected



