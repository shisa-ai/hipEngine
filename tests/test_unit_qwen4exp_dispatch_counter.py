import sys
from types import ModuleType

import pytest

from hipengine.kernels.registry import KernelKey, register, resolve
from scripts.qwen4exp_candidate_dispatch import count_candidate_dispatch, shape_records


def test_registered_calls_and_shapes_are_counted_and_restored():
    key = KernelKey("counter_test", "linear", "f32", "probe")
    original = lambda *args, **kwargs: kwargs["result"]
    register(key, original)
    with count_candidate_dispatch(key=key, shape_positions=(3, 4, 5)) as counter:
        fn = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
        assert fn(1, 2, 3, 512, 320, 10240, result=7) == 7
        assert fn(1, 2, 3, 512, 320, 10240, result=8) == 8
    assert counter["calls"] == 2
    assert shape_records(counter) == [{"arguments": [512, 320, 10240], "calls": 2}]
    assert resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                   variant=key.variant) is original


def test_direct_alias_is_counted_and_restored_on_error(monkeypatch):
    module = ModuleType("_counter_test_module")
    original = lambda *args: "value"
    module.fn = original
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="body"):
        with count_candidate_dispatch(direct_target=(module.__name__, "fn")) as counter:
            assert module.fn() == "value"
            raise RuntimeError("body")
    assert counter["calls"] == 1
    assert module.fn is original


def test_missing_exact_key_cannot_be_replaced_by_a_fallback():
    fallback = KernelKey("counter_test", "linear", "f32", "")
    register(fallback, lambda: None)
    with pytest.raises(ValueError, match="exact candidate"):
        with count_candidate_dispatch(key=KernelKey("counter_test", "linear", "f32", "missing")):
            pytest.fail("fallback was counted as candidate")


def test_direct_alias_must_match_declared_registry_callable(monkeypatch):
    module = ModuleType("_counter_test_module")
    module.fn = lambda: 1
    monkeypatch.setitem(sys.modules, module.__name__, module)
    key = KernelKey("counter_test", "linear", "f32", "probe")
    register(key, lambda: 2)
    with pytest.raises(ValueError, match="does not match"):
        with count_candidate_dispatch(key=key, direct_target=(module.__name__, "fn")):
            pass


def test_shape_abi_failure_restores_direct_alias(monkeypatch):
    module = ModuleType("_counter_test_module")
    original = lambda *args: None
    module.fn = original
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="observed ABI"):
        with count_candidate_dispatch(
            direct_target=(module.__name__, "fn"), shape_positions=(3, 4, 5),
        ):
            module.fn(1, 2)
    assert module.fn is original


def test_explicit_direct_reference_does_not_count_compatibility_wrapper(monkeypatch):
    module = ModuleType("_counter_test_module")
    original = lambda: 1
    module.alias = original
    module.original = original
    monkeypatch.setitem(sys.modules, module.__name__, module)
    key = KernelKey("counter_test", "linear", "f32", "compat")
    register(key, lambda: 2)
    with count_candidate_dispatch(
        key=key, direct_target=(module.__name__, "alias"),
        direct_reference=(module.__name__, "original"),
    ) as counter:
        assert module.alias() == 1
        assert resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                       variant=key.variant)() == 2
    assert counter["calls"] == 1
    assert module.alias is original


def test_real_gr_alias_and_candidate_scopes_match():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import register_gguf_k_gemv_kernels
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gguf_k_gemv_kernels(replace=True)
    register_gfx1151_kernels()
    for name, enabled in (
        ("production_gr_up_restore", "GR_IU8"),
        ("production_gr_down_restore", "GR_IU8_DOWN"),
        ("production_dense_q8_restore", "Q8_IU8_WMM"),
    ):
        spec = CANDIDATES[name]
        assert spec.requires_dispatch_count
        assert spec.classification == "T2"
        assert spec.dispatch_shape_positions == (3, 4, 5)
        assert spec.environment == {
            "HIPENGINE_QWEN4_EXP_" + flag: "1" if flag == enabled else "0"
            for flag in ("GR_IU8", "GR_IU8_DOWN", "Q8_IU8_WMM", "Q8_MMQ_PREFILL")}
        with count_candidate_dispatch(
            key=KernelKey(*spec.candidate_key), direct_target=spec.direct_dispatch_target,
            direct_reference=spec.direct_dispatch_reference,
            shape_positions=spec.dispatch_shape_positions,
        ) as counter:
            assert counter["calls"] == 0
