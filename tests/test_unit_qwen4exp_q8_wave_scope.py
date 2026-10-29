"""Q8 wave-scale prefill selection belongs to the runner that executes it."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as module
from hipengine.runtime import gguf_linear as linear
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

ENV = "HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE"


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("stale", ["0", "1"])
@pytest.mark.parametrize("raw_variant", ["coltile8", "rowbatch"])
def test_chunk_scope_uses_runner_choice_and_restores(monkeypatch, published, stale, raw_variant):
    seen = []
    monkeypatch.setenv(ENV, stale)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_RAW_VARIANT", raw_variant)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_RAW_ROWBATCH", "32")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS", "")
    monkeypatch.setattr(module, "_qwen4_exp_q8_tile_scope", lambda owner: nullcontext())

    def chunk(tokens, **kwargs):
        seen.append(linear.raw_k_prefill_variant())
        raise RuntimeError("observed chunk")

    owner = SimpleNamespace(
        q8_wave_scale_prefill=published, max_sequence_length=4096,
        prefill_chunk_size=1024, position=0, reset=lambda: None,
        _prefill_chunk=chunk, _q8_mmq_prefill_context=nullcontext,
        _wide_f16_activation_context=nullcontext,
    )
    with linear.raw_k_prefill_variant_session("coltile"):
        with pytest.raises(RuntimeError, match="observed chunk"):
            next(module.Qwen4ExpGGUFResidentModelRunner._prefill_chunked_steps(
                owner, [3] * 65))
        assert linear.raw_k_prefill_variant() == "coltile"
    expected = "coltile8_wave_scale" if published and raw_variant == "coltile8" else raw_variant
    assert seen == [expected]


def test_diagnostic_scope_restores_after_nested_failure():
    owner = SimpleNamespace(q8_wave_scale_prefill=True)
    other = SimpleNamespace(q8_wave_scale_prefill=True)
    with pytest.raises(RuntimeError, match="failed"):
        with module.q8_wave_scale_prefill(owner, False):
            with module.q8_wave_scale_prefill(owner, True):
                assert owner.q8_wave_scale_prefill is True
            assert owner.q8_wave_scale_prefill is False
            assert other.q8_wave_scale_prefill is True
            raise RuntimeError("failed")
    assert owner.q8_wave_scale_prefill is True


@pytest.mark.parametrize("published", [False, True])
def test_late_clone_inherits_wave_choice(published):
    owner = _inherit_lane_decisions(SimpleNamespace(q8_wave_scale_prefill=published),
                                   SimpleNamespace())
    assert owner.q8_wave_scale_prefill is published


@pytest.mark.parametrize("mode,expected", [("before", False), ("after", True)])
def test_campaign_scope_binds_owner_and_restores_preemptors(mode, expected):
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope, Q8_WAVE_ENV
    owner = SimpleNamespace(q8_wave_scale_prefill=True)
    env = {ENV: "stale", "HIPENGINE_QWEN4_EXP_RAW_VARIANT": "rowbatch"}
    before = dict(env)
    with pytest.raises(RuntimeError, match="failed"):
        with route_scope("q8-wave-scale", mode, owner, environment=env):
            assert owner.q8_wave_scale_prefill is expected
            assert all(env[k] == v for k, v in Q8_WAVE_ENV.items())
            assert env[ENV] == "stale"
            raise RuntimeError("failed")
    assert env == before
    assert owner.q8_wave_scale_prefill is True


@pytest.mark.parametrize("arm,expected", [
    ("all_flags_strict", False), ("base_plus_q8_gr_flags", False),
    ("strict_plus_dense", False), ("strict_plus_matrix", False),
    ("strict_plus_gdn", False), ("bound", True),
    ("dense_q8_gr_strict", True), ("q8down_off_without_dense", True),
])
def test_journey_preserves_old_wave_overrides(arm, expected):
    from scripts.qwen4exp_journey_localize import wave_arm_enabled
    assert wave_arm_enabled(arm, True) is expected
    assert wave_arm_enabled(arm, False) is False
