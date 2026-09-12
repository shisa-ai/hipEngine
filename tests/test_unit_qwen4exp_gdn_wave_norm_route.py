import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_parent_only(monkeypatch):
    key = KernelKey("hip_gfx1151","gdn_recurrence_norm_gate","f32_state",
                    "qwen4exp_sigmoid_register_prefill")
    select = runner._qwen4_exp_gdn_wave_norm_key
    monkeypatch.setattr(runner,"is_registered",lambda key:True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM",raising=False)
    assert select(key)==key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM","1")
    assert select(key).variant=="qwen4exp_sigmoid_wave_norm_prefill"
    other=KernelKey(key.backend,key.layer,key.quant,"strict")
    assert select(other)==other
    monkeypatch.setattr(runner,"is_registered",lambda key:False)
    assert select(key)==key


@pytest.mark.parametrize("mode,value",[("before","0"),("after","1")])
def test_one_flag(mode,value):
    env={"other":"keep"}
    _apply_mode(mode,route_package="gdn-wave-norm",environment=env)
    assert env=={"other":"keep","HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM":value}


@pytest.mark.parametrize("tokens,expected",[(1,0),(512,21),(1024,21),(1025,21),(1026,42),(4096,84)])
def test_current_serial_prefix_counts(tokens,expected):
    from scripts.qwen4exp_halo_box_campaign_ab import gdn_wave_norm_expected_calls
    assert gdn_wave_norm_expected_calls(tokens,1024)==expected
