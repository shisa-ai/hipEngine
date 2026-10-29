"""TimesFM registry contracts without HIP initialization or model weights."""

from types import SimpleNamespace

import pytest

from hipengine.dispatch.timesfm import bind_timesfm_kernels, timesfm_key
from hipengine.kernels.registry import MissingKernelError, register, unregister


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_families_bind_shared_primitives_and_precision_routes(backend, precision):
    selected, kernels = bind_timesfm_kernels(backend, precision, multivariate=True)
    assert selected == backend
    assert callable(kernels.timesfm_rope_f32)
    assert callable(kernels.timesfm3_var_attention)
    assert hasattr(kernels, "timesfm_flash_attention_f16") == (precision == "fp16")
    assert hasattr(kernels, "timesfm3_sgemm_f64acc") == (precision == "fp32")


@pytest.mark.parametrize("backend,arch", [("hip_gfx1100", "gfx1100"), ("hip_gfx1151", "gfx1151")])
@pytest.mark.parametrize("precision,dtype", [("fp16", "f16"), ("fp32", "f32")])
def test_registered_launch_pins_arch_dtype_and_restores_environment(monkeypatch, backend, arch, precision, dtype):
    import os
    from hipengine.kernels.hip_gfx1100.timesfm import timesfm
    calls = []
    def signature(library, symbol, *args):
        calls.append((os.environ["HIPENGINE_HIP_ARCH"], symbol))
        return lambda *args: 0
    monkeypatch.setattr(timesfm, "signed_kernel_fn", signature)
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "unrelated")
    kernels = bind_timesfm_kernels(backend, precision)[1]
    kernels.timesfm_bias_f32(1, 2, 3, 4, library=object(), runtime=object())
    assert calls == [(arch, f"timesfm_bias_{dtype}")]
    assert os.environ["HIPENGINE_HIP_ARCH"] == "unrelated"
    with pytest.raises(ValueError, match="registered for"):
        kernels.timesfm_bias_f32(1, 2, 3, 4, dtype="f32" if dtype == "f16" else "f16")


def test_registration_preserves_override_and_strict_dtype_miss(monkeypatch):
    import hipengine.dispatch.timesfm as dispatch
    backend = "hip_gfx1100"
    bind_timesfm_kernels(backend, "fp32")
    key = timesfm_key(backend, "timesfm_rmsnorm_f32", "fp32")
    sentinel = lambda *a, **kw: None
    register(key, sentinel, replace=True)
    assert bind_timesfm_kernels(backend, "fp32")[1].timesfm_rmsnorm_f32 is sentinel
    unregister(key)
    monkeypatch.setattr(dispatch, "load_backend_kernel_package", lambda _: SimpleNamespace())
    with pytest.raises(MissingKernelError) as error:
        bind_timesfm_kernels(backend, "fp32")
    assert error.value.requested == key  # never reinterpret an FP32 buffer as FP16


def test_decoder_strict_gemm_calls_registry_override():
    from hipengine.runtime.timesfm3_decode import TimesFM3GPUDecoder
    backend = "hip_gfx1100"
    bind_timesfm_kernels(backend, "fp32", multivariate=True)
    calls = []
    key = timesfm_key(backend, "timesfm3_sgemm_f64acc", "fp32")
    register(key, lambda *args: calls.append(args), replace=True)
    decoder = object.__new__(TimesFM3GPUDecoder)
    decoder.precision = "fp32"
    decoder.backend, decoder._kernels = bind_timesfm_kernels(backend, "fp32", multivariate=True)
    decoder._gemm(1, 2, 3, 4, 5, 6)
    assert calls == [(1, 2, 3, 4, 5, 6)]


@pytest.mark.parametrize("module,cls", [
    ("hipengine.runtime.timesfm_decode", "TimesFMGPUDecoder"),
    ("hipengine.runtime.timesfm3_decode", "TimesFM3GPUDecoder"),
])
def test_constructor_binds_before_allocating(monkeypatch, module, cls):
    from importlib import import_module
    runtime = import_module(module)
    calls = []
    class Bound(Exception):
        pass
    def bind(backend, precision, **kwargs):
        calls.append((backend, precision, kwargs))
        raise Bound
    monkeypatch.setattr(runtime, "bind_timesfm_kernels", bind, raising=False)
    with pytest.raises(Bound):
        getattr(runtime, cls)(None, backend="hip_gfx1151", precision="fp32")
    assert calls[0][:2] == ("hip_gfx1151", "fp32")
