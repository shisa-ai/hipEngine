import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_vec4_scope_and_missing_capability(monkeypatch):
    key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                    "mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out")
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4", raising=False)
    assert linear._q8_mmq_vec4_key(key, rows=512) == key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4", "1")
    assert linear._q8_mmq_vec4_key(key, rows=63) == key
    assert "vec4" in linear._q8_mmq_vec4_key(key, rows=64).variant
    raw = KernelKey(key.backend, key.layer, key.quant,
                    "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out")
    assert linear._q8_mmq_vec4_key(raw, rows=512) == raw
    monkeypatch.setattr(linear, "is_registered", lambda key: False)
    assert linear._q8_mmq_vec4_key(key, rows=512) == key


@pytest.mark.parametrize("mode,value", [("before", "0"), ("after", "1")])
def test_vec4_ab_changes_one_flag(mode, value):
    env = {"other": "keep"}
    _apply_mode(mode, environment=env, route_package="q8-mmq-vec4")
    assert env == {"other": "keep", "HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4": value}


@pytest.mark.parametrize("prompt,chunk,calls", [
    (63,512,0), (64,512,72), (512,512,72), (1024,512,144),
    (4096,512,576), (576,512,144), (512,32,0),
])
def test_vec4_call_counts(prompt, chunk, calls):
    from scripts.qwen4exp_halo_box_campaign_ab import q8_mmq_vec4_expected_calls
    assert q8_mmq_vec4_expected_calls(prompt, chunk) == calls
