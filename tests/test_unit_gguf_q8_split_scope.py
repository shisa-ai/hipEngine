import pytest

import hipengine.runtime.gguf_linear as linear
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_t16_gemv as kernels
from tests.test_unit_gguf_linear_dispatch import _fake_weight


@pytest.mark.parametrize("backend,rows,k,a,b,expected", [
    ("hip_gfx1100", 1, 5120, 48, 48, 256),
    ("hip_gfx1100", 2, 5120, 48, 48, 256),
    ("hip_gfx1100", 3, 5120, 48, 48, 256),
    ("hip_gfx1100", 4, 5120, 48, 48, 256),
    ("hip_gfx1100", 5, 5120, 48, 48, 0),
    ("hip_gfx1100", 3, 512, 64, 128, 0),
    ("hip_gfx1151", 3, 5120, 48, 48, 0),
])
def test_wide_q8_split_is_backend_row_and_shape_scoped(monkeypatch, backend, rows, k, a, b, expected):
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_THREADS", raising=False)
    assert linear._resolve_q8_t16_dual_split_threads(backend, rows, k, a, b, 0) == expected


def test_explicit_q8_split_overrides_keep_main_semantics(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_THREADS", "64")
    resolve = linear._resolve_q8_t16_dual_split_threads
    assert resolve("hip_gfx1100", 3, 5120, 48, 48, 0) == 64
    assert resolve("hip_gfx1100", 3, 5120, 48, 48, 128) == 128


def test_standalone_q8_split_keeps_main_reference_default():
    assert kernels._DUAL_SPLIT_DEFAULT_THREADS == 128


@pytest.mark.parametrize("backend,expected", [("hip_gfx1100", 256), ("hip_gfx1151", 0)])
def test_live_pair_dispatch_forwards_scoped_threads(monkeypatch, backend, expected):
    # Capability lookup may have imported this package before registry cleanup.
    # Refresh through the supported loader; a cached import alone is insufficient.
    load_backend_kernel_package(backend)
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_THREADS", raising=False)
    calls = []
    monkeypatch.setattr(linear, "gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(linear, "_use_q8_t16_pair_rowtile", lambda **kwargs: False)
    weight = _fake_weight(layout=linear.LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    assert linear.launch_gguf_linear_pair(
        weight, weight, 100, 200, 300, 3, 5120, 48,
        out_features_b=48, backend=backend, use_gemv_decode=True,
        use_wmma_prefill=False, runtime="fake",
    )
    assert len(calls) == 1
    assert calls[0][1]["threads"] == expected
