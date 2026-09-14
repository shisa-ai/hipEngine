import pytest

from hipengine.kernels.registry import KernelKey, register
from hipengine.runtime.gguf_linear import GGUFLinearDispatch, _q8_iu8_wmma_dispatch


@pytest.mark.parametrize("rows", [71, 256, 512, 1024])
def test_generic_dense_q8_can_cover_gr_down_without_its_dedicated_flag(rows, monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM", "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_IU8_DOWN", "0")
    candidate = KernelKey(
        "hip_gfx1151", "linear", "gguf_q8_0", "iu8_wmma_prefill_f32_f32_out")
    register(candidate, lambda *args, **kwargs: None, replace=True)
    parent = GGUFLinearDispatch(
        KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                  "coltile8_rowbatch4_wave_scale_f32_f32_out"), "raw")
    actual = _q8_iu8_wmma_dispatch(
        parent, rows=rows, in_features=10240, out_features=320)
    assert actual == (GGUFLinearDispatch(candidate, "raw") if rows > 256 else parent)
