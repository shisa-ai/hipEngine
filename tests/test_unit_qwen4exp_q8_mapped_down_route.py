import pytest
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_mapped_down_requires_current_map(monkeypatch):
    select = runner._qwen4_exp_mapped_down_key
    monkeypatch.setattr(runner,"is_registered",lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN",raising=False)
    args = dict(backend="hip_gfx1151",quant="gguf_q8_0",rows=512,map_ready=True)
    assert select(**args) is None
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN","1")
    assert select(**args).variant == "selected_grouped_row4_bundle_gemv_bf16_bf16_out"
    assert select(**{**args,"map_ready":False}) is None
    assert select(**{**args,"rows":511}) is None
    monkeypatch.setattr(runner,"is_registered",lambda key: False)
    assert select(**args) is None


@pytest.mark.parametrize("mode,value",[("before","0"),("after","1")])
def test_mapped_down_ab_changes_one_flag(mode,value):
    env = {"other":"keep"}
    _apply_mode(mode,environment=env,route_package="q8-mapped-down")
    assert env == {"other":"keep","HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN":value}


@pytest.mark.parametrize("prompt,chunk,expected",[(512,512,1),(4096,512,8),
                                                (576,512,1),(511,512,0)])
def test_mapped_down_counts(prompt,chunk,expected):
    from scripts.qwen4exp_halo_box_campaign_ab import q8_mapped_down_expected_calls
    assert q8_mapped_down_expected_calls(prompt,chunk) == expected


@pytest.mark.parametrize("route,pointer,expected",[
    ("q8-mapped-down",None,False),("q8-mapped-down",0,False),
    ("q8-mapped-down",1024,True),("q8-down-bundle",None,True),
    ("q8-down-bundle",1024,False),("q4-pair",1024,True),
])
def test_shared_kernel_counter_scope(route,pointer,expected):
    from scripts.qwen4exp_halo_box_campaign_ab import q8_bundle_call_in_scope
    assert q8_bundle_call_in_scope(route,(1,2,pointer)) == expected
