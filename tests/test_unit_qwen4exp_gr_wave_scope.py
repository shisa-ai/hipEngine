"""GR wave-scale ownership, shape guards, and DPP fallback contracts."""
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as module

ENV = "HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE"
BASE = KernelKey("hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                 "coltile2_branch4_rowbatch4_f32_exact")
WAVE = "coltile2_branch4_rowbatch4_wave_scale_f32_exact"
DPP = "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact"


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("stale", ["0", "1"])
@pytest.mark.parametrize("rows,branches", [(64, 4), (256, 4), (257, 4), (513, 4), (512, 2)])
def test_selector_uses_published_choice(monkeypatch, published, stale, rows, branches):
    monkeypatch.setenv(ENV, stale)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION", "0")
    monkeypatch.setattr(module, "is_registered", lambda key: True)
    selected = module._qwen4_exp_gr_wave_scale_key(
        BASE, rows=rows, branches=branches, published=published)
    assert selected.variant == (WAVE if published and rows > 256 and branches == 4 else BASE.variant)


@pytest.mark.parametrize("published", [False, True])
def test_dpp_keeps_wave_fallback_independent_of_wave_publication(monkeypatch, published):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION", "1")
    for available, expected in [({DPP, WAVE}, DPP), ({WAVE}, WAVE), (set(), BASE.variant)]:
        monkeypatch.setattr(module, "is_registered", lambda key: key.variant in available)
        assert module._qwen4_exp_gr_wave_scale_key(
            BASE, rows=513, branches=4, published=published).variant == expected


def test_diagnostic_scope_restores_nested_failure_and_isolates_other_runner():
    owner = SimpleNamespace(gr_wave_scale=True)
    other = SimpleNamespace(gr_wave_scale=True)
    with pytest.raises(RuntimeError, match="failed"):
        with module.gr_wave_scale(owner, False):
            with module.gr_wave_scale(owner, True):
                assert owner.gr_wave_scale is True
            assert owner.gr_wave_scale is False
            assert other.gr_wave_scale is True
            raise RuntimeError("failed")
    assert owner.gr_wave_scale is True


@pytest.mark.parametrize("published", [False, True])
def test_late_clone_inherits_choice(published):
    clone = _inherit_lane_decisions(SimpleNamespace(gr_wave_scale=published), SimpleNamespace())
    assert clone.gr_wave_scale is published


@pytest.mark.parametrize("mode,expected", [("before", False), ("after", True)])
def test_campaign_scope_restores_controls_and_ignores_stale_env(mode, expected):
    from scripts.qwen4exp_halo_box_campaign_ab import GR_WAVE_ENV, route_scope
    owner = SimpleNamespace(gr_wave_scale=True)
    env = {ENV: "stale", "HIPENGINE_QWEN4_EXP_GR_WIDE": "1"}
    before = dict(env)
    with pytest.raises(RuntimeError, match="failed"):
        with route_scope("gr-wave-scale", mode, owner, environment=env):
            assert owner.gr_wave_scale is expected
            assert all(env[key] == value for key, value in GR_WAVE_ENV.items())
            assert env[ENV] == "stale"
            raise RuntimeError("failed")
    assert owner.gr_wave_scale is True
    assert env == before


def test_campaign_requires_an_owner():
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope
    with pytest.raises(ValueError, match="owning runner"):
        route_scope("gr-wave-scale", "after")


@pytest.mark.parametrize("arm,expected", [
    ("bound", True), ("all_flags_strict", False),
    ("base_plus_q8_gr_flags", False), ("strict_plus_dense", True),
    ("strict_plus_matrix", False), ("strict_plus_gdn", False),
    ("q8down_off_without_dense", False), ("dense_q8_gr_strict", True),
])
def test_journey_preserves_gr_family_choice(arm, expected):
    from scripts.qwen4exp_journey_localize import gr_wave_arm_enabled
    assert gr_wave_arm_enabled(arm, True) is expected
    assert gr_wave_arm_enabled(arm, False) is False


def test_captured_decode_width_cannot_reach_wave_scale():
    from hipengine.generation.qwen4_exp_ar_batch import QWEN4_EXP_MAX_BATCH_ROWS
    assert QWEN4_EXP_MAX_BATCH_ROWS <= 256


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("name", [
    "run_qwen4_exp_dense_qsa_layer", "run_qwen4_exp_qsa_prefill_layer",
    "run_qwen4_exp_gdn_layer",
])
def test_layer_forwards_owner_choice_to_gr(monkeypatch, published, name):
    import inspect
    runtime = object()
    gr = SimpleNamespace(norm_weight_ptr=1, down=object(), up=object(), inject=None)
    weights = SimpleNamespace(attention_gr=gr, ffn_gr=gr)
    scratch = SimpleNamespace(closed=False, runtime=runtime, attention_gr=object())
    seen = []

    def read(*args, **kwargs):
        seen.append(kwargs["gr_wave_scale"])
        raise RuntimeError("observed GR")

    monkeypatch.setattr(module, "run_qwen4_exp_gr_read", read)
    layer = getattr(module, name)
    kwargs = {key: 1 for key, parameter in inspect.signature(layer).parameters.items()
              if parameter.default is inspect.Parameter.empty}
    kwargs.update(weights=weights, scratch=scratch, runtime=runtime,
                  rows=389, gr_wave_scale=published)
    with pytest.raises(RuntimeError, match="observed GR"):
        layer(**kwargs)
    assert seen == [published]
