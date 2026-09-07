import pytest
import numpy as np
from tests import test_qwen4exp_gdn_register as base


def test_registry():
    from hipengine.kernels.registry import resolve
    base.gdn.register_qwen4_exp_gdn_kernels()
    assert resolve(backend="hip_gfx1100", layer="gdn_recurrence_norm_gate",
                   quant="f32_state", variant="qwen4exp_sigmoid_wave_norm_prefill") is (
        base.gdn.qwen4_exp_gdn_wave_norm_prefill_f32)
    for dk, dv in ((64,128), (128,64)):
        with pytest.raises(ValueError, match="Dk=Dv=128"):
            base.gdn.qwen4_exp_gdn_wave_norm_prefill_f32(*([0]*9),16,16,48,dk,dv)


@pytest.mark.skipif(not base.hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("tokens", [1,17,64,512,1024])
def test_exact(monkeypatch, tokens):
    f = base.Fixture(tokens)
    try:
        f.run(True)
        expected = f.result(True)
        monkeypatch.setattr(base.gdn, "qwen4_exp_gdn_register_prefill_f32",
                            base.gdn.qwen4_exp_gdn_wave_norm_prefill_f32)
        for split in ([None,7] if tokens>7 else [None,None]):
            f.run(True, split=split)
            for got, ref in zip(f.result(True), expected):
                np.testing.assert_array_equal(got.view(np.uint32),ref.view(np.uint32))
    finally:
        f.close()
