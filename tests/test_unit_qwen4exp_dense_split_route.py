"""Scoped split route and gate contracts; no GPU needed for dispatch RED."""
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey, register
from hipengine.runtime import gguf_linear as gl

ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_SPLIT_LAYERS"
VARIANT = "dense_split64x128_f32_f32_out"
BACKEND = "hip_gfx1151"


def parent():
    return gl.GGUFLinearDispatch(KernelKey(BACKEND, "linear", "gguf_q8_0",
        "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"), "raw")


def weight(layer):
    return SimpleNamespace(spec=SimpleNamespace(slot_path=f"layers.{layer}.attn_qkv"))


def select(p=None, layer=0, rows=512, k=2560, n=10240):
    return gl._q8_dense_split_dispatch(p or parent(), rows=rows, in_features=k,
        out_features=n, weight=weight(layer))


@pytest.fixture(autouse=True)
def candidate(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    register(KernelKey(BACKEND, "linear", "gguf_q8_0", VARIANT),
             lambda *a, **kw: None, replace=True)


def test_split_route_default_off_and_scope_explicit(monkeypatch):
    assert select() == parent()
    # Match the parser used by all retained scopes: comma-joined integers.
    monkeypatch.setenv(ENV, ",".join(map(str, range(13))))
    assert select().key.variant == VARIANT
    assert select(layer=12).key.variant == VARIANT
    assert select(layer=13) == parent()
    assert select(layer=47) == parent()


@pytest.mark.parametrize("rows,k,n", [(1,2560,10240),(256,2560,10240),
    (512,2528,10240),(512,2560,0)])
def test_split_shape_miss_keeps_parent(monkeypatch, rows, k, n):
    monkeypatch.setenv(ENV, "0")
    assert select(rows=rows,k=k,n=n) == parent()


def test_split_never_replaces_late_wide_or_other_abis(monkeypatch):
    monkeypatch.setenv(ENV, "0,13")
    wide = gl.GGUFLinearDispatch(KernelKey(BACKEND,"linear","gguf_q8_0",
        "dense_wide256_f32_f32_out"),"raw")
    assert select(wide,layer=13) == wide
    t16 = gl.GGUFLinearDispatch(parent().key,"t16")
    assert select(t16) == t16
    assert gl._q8_dense_split_dispatch(parent(),rows=512,in_features=2560,
        out_features=10240,weight=None) == parent()


def test_split_registry_miss_and_decline_are_visible(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    records = []
    with gl.wmma_prefill_dispatch_recorder(records):
        assert select().key.variant == VARIANT
    assert records[0]["selector"] == "dense_split"
    assert records[0]["engaged"]
    monkeypatch.setattr(gl, "is_registered", lambda key: False)
    monkeypatch.setattr(gl, "_ensure_linear_kernel_registered", lambda key: None)
    assert select() == parent()


def test_split_library_never_inherits_wrong_quant_library():
    key = KernelKey(BACKEND,"linear","gguf_q8_0",VARIANT)
    wrong, right = object(), object()
    assert gl._variant_scoped_library({"gguf_q8_0":wrong},key) is None
    assert gl._variant_scoped_library({"gguf_q8_0":wrong,
        f"gguf_q8_0:{VARIANT}":right},key) is right


def test_split_gate_declares_increment_not_a_null_arm():
    from scripts.execution_profile_q8_wmma_prefill_layers_gate import (
        SELECTORS, ROUTE_ROW_FLOORS, ROUTE_ENGAGEMENT_SELECTORS)
    route = SELECTORS["dense_split"]
    on = route.env_for(candidate=True,layers="0,1,2")
    off = route.env_for(candidate=False,layers="0,1,2")
    assert on[ENV] == "0,1,2" and off[ENV] == ""
    assert on["HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS"] == off[
        "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS"] == ",".join(map(str,range(13,48)))
    assert ROUTE_ROW_FLOORS["dense_split"] == 257
    assert ROUTE_ENGAGEMENT_SELECTORS["dense_split"] == "dense_split"


def test_split_selector_scope_and_layer_participate_in_live_cache_key(monkeypatch):
    recorded = []
    class Recording(dict):
        def get(self, key, default=None):
            recorded.append(key)
            raise RuntimeError("key recorded; stop before pointer use")
    class Weight:
        spec = SimpleNamespace(layout="raw_gguf",quant_key="gguf_q8_0",
                               slot_path="layers.0.attn_qkv.weight")
        backend = BACKEND
        def allocation(self, name):
            return SimpleNamespace(tensor=SimpleNamespace(ptr=4096))
    monkeypatch.setattr(gl,"_DISPATCH_RESOLVE_CACHE",Recording())
    w = Weight()
    def key(scope, layer=0):
        monkeypatch.setenv(ENV,scope)
        w.spec.slot_path = f"layers.{layer}.attn_qkv.weight"
        with pytest.raises(RuntimeError,match="key recorded"):
            gl.launch_gguf_linear(w,0,0,512,2560,10240,libraries={})
        return recorded[-1]
    assert key("") != key("0")
    assert key("0") != key("1")
    assert key("0",0) != key("0",1)


def test_split_actual_dispatch_validity_fails_closed():
    from scripts.execution_profile_q8_wmma_prefill_layers_gate import _split_dispatch_validity
    assert _split_dispatch_validity({"teacher": 0, "candidate": 42})["passed"]
    assert not _split_dispatch_validity({"teacher": 0, "candidate": 0})["passed"]
    assert not _split_dispatch_validity({"teacher": 1, "candidate": 42})["passed"]


def test_split_task_gate_uses_current_production_composition():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
    candidate = CANDIDATES["dense_split_early"]
    assert candidate.base_profile == "production"
    assert candidate.environment == {ENV: ",".join(map(str, range(13)))}
    assert candidate.candidate_key[-1] == VARIANT
    assert candidate.count_registered_dispatch


def test_split_selector_is_in_the_live_leaf_chain():
    import inspect
    chain = inspect.getsource(gl.launch_gguf_linear)
    assert chain.index("_raw_k_coltile_pipelined_dispatch(") < chain.index(
        "_q8_dense_split_dispatch(") < chain.index("_raw_k_staged_f32_dispatch(")
