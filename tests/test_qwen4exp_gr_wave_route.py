import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_gr_wave_route_preserves_scope_and_fallback(monkeypatch):
    base = KernelKey(
        "hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0", "coltile2_branch4_rowbatch4_f32_exact"
    )
    select = runner._qwen4_exp_gr_wave_scale_key
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE", raising=False)
    assert select(base, rows=512, branches=4) == base
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE", "1")
    assert (
        select(base, rows=257, branches=4).variant
        == "coltile2_branch4_rowbatch4_wave_scale_f32_exact"
    )
    assert select(base, rows=256, branches=4) == base
    assert select(base, rows=512, branches=2) == base
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert select(base, rows=512, branches=4) == base


@pytest.mark.parametrize("mode,value", [("before", "0"), ("after", "1")])
def test_gr_ab_changes_only_its_flag(mode, value):
    env = {"other": "unchanged"}
    _apply_mode(mode, environment=env, route_package="gr-wave-scale")
    assert env == {"other": "unchanged", "HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE": value}
