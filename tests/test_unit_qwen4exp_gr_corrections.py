import pytest

from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as kernels


@pytest.mark.parametrize("variant", ["p4", "compensated"])
def test_corrections_are_explicit_gfx1151_variants(variant):
    register_gfx1151_kernels()
    name = f"iu8_{variant}_prefill_f32_f32_out"
    assert resolve(backend="hip_gfx1151", layer="linear", quant="gguf_q8_0", variant=name) is (
        getattr(kernels, f"gguf_q8_0_iu8_{variant}_prefill_f32_f32_t"))
    assert not is_registered(KernelKey("hip_gfx1100", "linear", "gguf_q8_0", name))


def test_correction_wrappers_preserve_the_raw_f32_abi(monkeypatch):
    calls = []
    monkeypatch.setattr(kernels, "_launch", lambda *args, **kwargs: calls.append((args, kwargs)))
    kernels.gguf_q8_0_iu8_p4_prefill_f32_f32_t(1, 2, 3, 17, 320, 130, stream=9)
    assert calls == [(("gguf_q8_0", "hipengine_gguf_q8_0_iu8_p4_prefill_f32_f32_t",
                       1, 2, 3, 17, 320, 130), {"stream": 9})]
