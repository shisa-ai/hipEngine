"""Failed split runtime route stays absent; diagnostic leaf coverage survives."""
import inspect
from types import SimpleNamespace

import pytest

from hipengine.runtime import gguf_linear as gl

ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_SPLIT_LAYERS"
VARIANT = "dense_split64x128_f32_f32_out"


def test_failed_split_selector_is_not_in_runtime_dispatch():
    assert not hasattr(gl, "_q8_dense_split_dispatch")
    assert not hasattr(gl, "_q8_dense_split_layers")
    chain = inspect.getsource(gl.launch_gguf_linear)
    assert "_q8_dense_split" not in chain
    assert "_split_key" not in chain
    assert chain.index("_raw_k_coltile_pipelined_dispatch(") < chain.index(
        "_raw_k_staged_f32_dispatch(")


def test_obsolete_split_env_cannot_change_the_live_cache_key(monkeypatch):
    recorded = []

    class Recording(dict):
        def get(self, key, default=None):
            recorded.append(key)
            raise RuntimeError("key recorded; stop before pointer use")

    class Weight:
        spec = SimpleNamespace(layout="raw_gguf", quant_key="gguf_q8_0",
                               slot_path="layers.0.attn_qkv.weight")
        backend = "hip_gfx1151"

        def allocation(self, name):
            return SimpleNamespace(tensor=SimpleNamespace(ptr=4096))

    monkeypatch.setattr(gl, "_DISPATCH_RESOLVE_CACHE", Recording())
    w = Weight()
    for scope in ("", "0", "0,1,2", "obsolete-invalid-scope"):
        monkeypatch.setenv(ENV, scope)
        with pytest.raises(RuntimeError, match="key recorded"):
            gl.launch_gguf_linear(w, 0, 0, 512, 2560, 10240, libraries={})
    assert all(key == recorded[0] for key in recorded)


def test_failed_split_modes_are_not_whole_model_gate_candidates():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
    from scripts.execution_profile_q8_wmma_prefill_layers_gate import (
        CalibrationError, ROUTE_ENGAGEMENT_SELECTORS, ROUTE_ROW_FLOORS,
        SELECTORS, _select_route, build_parser,
    )

    assert "dense_split_early" not in CANDIDATES
    for name in ("dense_split", "dense_split_composed_vs_exact"):
        assert name not in SELECTORS
        assert name not in ROUTE_ROW_FLOORS
        assert name not in ROUTE_ENGAGEMENT_SELECTORS
        with pytest.raises(CalibrationError):
            _select_route(name)
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--route", name, "--output", "unused.json"])


def test_split_diagnostic_remains_independently_registered():
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import KernelKey, is_registered

    assert VARIANT in wide._VARIANTS
    assert callable(wide.gguf_q8_0_dense_split64x128_f32_f32_out)
    register_gfx1151_kernels(replace=True)
    assert is_registered(KernelKey("hip_gfx1151", "linear", "gguf_q8_0", VARIANT))
