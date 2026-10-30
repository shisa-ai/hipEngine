"""Manifest-owned small dense GEMM selection and scope restoration."""
import pytest
from hipengine.runtime import gguf_linear as linear
from hipengine.kernels.registry import KernelKey

PARENT = "f32_hidden_f32_out"
WMMA = "f16_wmma_f32_hidden_f32_out"

@pytest.mark.parametrize("quant", ["bf16", "f32"])
@pytest.mark.parametrize("rows,cols,k,selected", [(63,48,64,False), (64,16,64,True), (65,15,64,False), (513,48,2560,True), (257,4,64,False), (100,512,33,False)])
def test_projection_scope_and_shape(monkeypatch, quant, rows, cols, k, selected):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_DENSE_F16_GEMM", "1")
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    dispatch = linear.GGUFLinearDispatch(KernelKey("hip_gfx1151", "dense_gemv", quant, PARENT), "dense_bf16")
    def resolve():
        return linear._dense_f16_gemm_dispatch(dispatch, rows=rows, in_features=k, out_features=cols)
    assert resolve() == dispatch
    with linear.dense_f16_gemm_session(((quant, WMMA),)):
        assert resolve().key.variant == (WMMA if selected else PARENT)
        with linear.dense_f16_gemm_session(((quant, PARENT),)):
            assert resolve() == dispatch
        assert resolve().key.variant == (WMMA if selected else PARENT)
    assert resolve() == dispatch


def test_scope_restores_after_exception():
    with pytest.raises(RuntimeError), linear.dense_f16_gemm_session((("bf16", WMMA),)):
        raise RuntimeError("test")
    assert linear._dense_f16_gemm_variants.get() == ()
