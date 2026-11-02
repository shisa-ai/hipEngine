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
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_RAW_VARIANT", "invalid")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_RAW_ROWBATCH", "invalid")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS", "")
    monkeypatch.setattr(module, "_qwen4_exp_q8_tile_scope", lambda owner: nullcontext())

    def chunk(tokens, **kwargs):
        seen.append(linear.raw_k_prefill_variant())
        raise RuntimeError("observed chunk")

    owner = SimpleNamespace(
        raw_prefill_policy=module.RawPrefillPolicy(32, raw_variant),
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
    prior_policy = linear.Q8DenseWidePolicy(True, frozenset({16, 47}))
    owner = SimpleNamespace(q8_wave_scale_prefill=True, q8_dense_wide_policy=prior_policy,
                            gguf_wmma_prefill_enabled=True)
    env = {ENV: "stale", "HIPENGINE_QWEN4_EXP_RAW_VARIANT": "rowbatch",
           "HIPENGINE_GGUF_WMMA_PREFILL": "1"}
    before = dict(env)
    with pytest.raises(RuntimeError, match="failed"):
        with route_scope("q8-wave-scale", mode, owner, environment=env):
            assert owner.q8_wave_scale_prefill is expected
            assert owner.gguf_wmma_prefill_enabled is False
            assert env['HIPENGINE_GGUF_WMMA_PREFILL'] == '1'
            assert owner.q8_dpp_reduction_enabled is False
            assert env == before
            assert env[ENV] == "stale"
            assert owner.q8_wmma_layer_scope.is_empty
            assert linear.resolve_q8_dense_wide_policy(owner) == linear.Q8DenseWidePolicy(False, None)
            raise RuntimeError("failed")
    assert owner.q8_dense_wide_policy is prior_policy
    assert env == before
    assert owner.q8_wave_scale_prefill is True
    assert owner.gguf_wmma_prefill_enabled is True


def test_campaign_record_names_explicit_wide_preemptor():
    from scripts.qwen4exp_halo_box_campaign_ab import q8_wave_control_record
    record = q8_wave_control_record(SimpleNamespace(q8_wave_scale_prefill=True))
    assert record['forced_dense_wide_policy'] == {'enabled': False, 'layers': None}
    assert 'HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE' not in record['forced_environment']
    assert 'HIPENGINE_GGUF_WMMA_PREFILL' not in record['forced_environment']
    assert record['forced_runner_fields']['q8_dpp_reduction_enabled'] is False
    assert 'HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION' not in record['forced_environment']
    assert record['forced_runner_fields']['gguf_wmma_prefill_enabled'] is False


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
