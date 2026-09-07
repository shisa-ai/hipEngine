import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_raw_parent_geometry_capability(monkeypatch):
    key=KernelKey("hip_gfx1151","linear","gguf_q8_0",
                  "mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out")
    select=linear._q8_mmq_token64_key
    monkeypatch.setattr(linear,"is_registered",lambda key:True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_MMQ_TOKEN64",raising=False)
    assert select(key,rows=1024,hidden=2560,outputs=12288)==key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_MMQ_TOKEN64","1")
    assert "token64" in select(key,rows=1024,hidden=2560,outputs=12288).variant
    for r,k,n in ((511,2560,12288),(1024,10240,320),(1024,2560,10240)):
        assert select(key,rows=r,hidden=k,outputs=n)==key
    packed=KernelKey(key.backend,key.layer,key.quant,
                    "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out")
    assert select(packed,rows=1024,hidden=2560,outputs=12288)==packed
    monkeypatch.setattr(linear,"is_registered",lambda key:False)
    assert select(key,rows=1024,hidden=2560,outputs=12288)==key


@pytest.mark.parametrize("mode,value",[("before","0"),("after","1")])
def test_one_flag(mode,value):
    env={"other":"keep"}
    _apply_mode(mode,route_package="mmq-token64",environment=env)
    assert env=={"other":"keep","HIPENGINE_QWEN4_EXP_MMQ_TOKEN64":value}


@pytest.mark.parametrize("tokens,expected",[(511,0),(512,12),(1024,12),(1535,12),(1536,24),(4096,48)])
def test_current_chunk_engagement(tokens,expected):
    from scripts.qwen4exp_halo_box_campaign_ab import mmq_token64_expected_calls
    assert mmq_token64_expected_calls(tokens,1024)==expected
