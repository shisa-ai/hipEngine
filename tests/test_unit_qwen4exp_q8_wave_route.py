import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_wave_route_is_registry_checked_and_preserves_other_variants(monkeypatch):
    base = linear.GGUFLinearDispatch(
        KernelKey("hip_gfx1151", "linear", "gguf_q8_0", "coltile8_rowbatch4_f32_f32_out"), "raw"
    )
    select = linear._raw_k_wave_scale_dispatch
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    assert select(base, enabled=False) is base
    assert select(base, enabled=True).key.variant == "coltile8_rowbatch4_wave_scale_f32_f32_out"
    other = linear.GGUFLinearDispatch(KernelKey("hip_gfx1151", "linear", "gguf_q8_0", "mmq"), "raw")
    assert select(other, enabled=True) is other
    monkeypatch.setattr(linear, "is_registered", lambda key: False)
    assert select(base, enabled=True) is base


@pytest.mark.parametrize("mode,value", [("before", "0"), ("after", "1")])
def test_ab_changes_only_wave_flag(mode, value):
    env = {"other": "unchanged"}
    _apply_mode(mode, environment=env, route_package="q8-wave-scale")
    assert env == {"other": "unchanged", "HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE": value}
