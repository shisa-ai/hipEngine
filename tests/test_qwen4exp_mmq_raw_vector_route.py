import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_raw_vector_scope(monkeypatch):
    key = KernelKey("hip_gfx1151","linear","gguf_q8_0",
                    "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out")
    select = linear._q8_mmq_raw_vector_key
    monkeypatch.setattr(linear,"is_registered",lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR",raising=False)
    assert select(key,rows=512) == key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR","1")
    assert "raw_vec4" in select(key,rows=64).variant
    assert select(key,rows=63) == key
    packed = KernelKey(key.backend,key.layer,key.quant,
                       "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out")
    assert select(packed,rows=512) == packed
    monkeypatch.setattr(linear,"is_registered",lambda key: False)
    assert select(key,rows=512) == key


@pytest.mark.parametrize("mode,value",[("before","0"),("after","1")])
def test_raw_vector_ab_one_flag(mode,value):
    env = {"other":"keep"}
    _apply_mode(mode,environment=env,route_package="q8-mmq-raw-vector")
    assert env == {"other":"keep","HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR":value}


@pytest.mark.parametrize("prompt,chunk,expected",[
    (512,512,242),(1024,512,484),(4096,512,1936),(63,512,0),
    (576,512,484),(512,32,0),
])
def test_raw_vector_counts(prompt,chunk,expected):
    from scripts.qwen4exp_halo_box_campaign_ab import q8_mmq_raw_vector_expected_calls
    assert q8_mmq_raw_vector_expected_calls(prompt,chunk) == expected
