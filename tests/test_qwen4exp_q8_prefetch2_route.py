import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear


def test_paired_load_dispatch_requires_parent_geometry_and_capability(monkeypatch):
    key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                    "coltile8_rowbatch4_wave_scale_f32_f32_out")
    parent = linear.GGUFLinearDispatch(key, "raw")
    select = linear._raw_k_prefetch2_dispatch
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    assert select(parent, enabled=True, rows=1024, hidden=2560, outputs=6144).key.variant == (
        "coltile8_rowbatch4_prefetch2_f32_f32_out")
    for args in (
        dict(enabled=False, rows=1024, hidden=2560, outputs=6144),
        dict(enabled=True, rows=63, hidden=2560, outputs=6144),
        dict(enabled=True, rows=1024, hidden=640, outputs=2560),
        dict(enabled=True, rows=1024, hidden=2560, outputs=2560),
    ):
        assert select(parent, **args) == parent
    other = linear.GGUFLinearDispatch(
        KernelKey(key.backend, key.layer, key.quant, "coltile8_rowbatch4_f32_f32_out"), "raw")
    assert select(other, enabled=True, rows=1024, hidden=2560, outputs=6144) == other
    monkeypatch.setattr(linear, "is_registered", lambda key: False)
    assert select(parent, enabled=True, rows=1024, hidden=2560, outputs=6144) == parent


@pytest.mark.parametrize("tokens,chunk,expected", [
    (512, 1024, 36), (1024, 1024, 36), (4096, 1024, 144),
    (63, 1024, 0), (1087, 1024, 36), (1088, 1024, 72),
])
def test_engagement_counts(tokens, chunk, expected):
    from scripts.qwen4exp_halo_box_campaign_ab import q8_prefetch2_expected_calls
    assert q8_prefetch2_expected_calls(tokens, chunk) == expected


@pytest.mark.parametrize("mode,value", [("before", "0"), ("after", "1")])
def test_ab_changes_only_one_flag(mode, value):
    from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode
    env = {"unrelated": "keep"}
    _apply_mode(mode, route_package="q8-gate-prefetch2", environment=env)
    assert env == {"unrelated": "keep", "HIPENGINE_QWEN4_EXP_Q8_GATE_PREFETCH2": value}
