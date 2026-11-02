"""Raw-prefill geometry is carried by the executing owner, not its peers."""
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime import gguf_linear as linear
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions


@pytest.mark.parametrize('rowbatch,variant', [(0, 'rowbatch'), (8, 'coltile16'), (32, 'coltile8')])
def test_explicit_policy_overrides_ambient_and_drives_chunk(monkeypatch, rowbatch, variant):
    policy = runtime.RawPrefillPolicy(rowbatch, variant)
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_RAW_ROWBATCH', 'invalid')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_RAW_VARIANT', 'invalid')
    monkeypatch.setattr(runtime, '_qwen4_exp_q8_tile_scope', lambda _: nullcontext())
    seen = []
    def chunk(tokens, **kwargs):
        seen.append((linear.raw_k_prefill_rowbatch(), linear.raw_k_prefill_variant()))
        raise RuntimeError('observed')
    owner = SimpleNamespace(raw_prefill_policy=policy, q8_wave_scale_prefill=False,
        max_sequence_length=4096, prefill_chunk_size=1024, position=0, reset=lambda: None,
        _prefill_chunk=chunk, _q8_mmq_prefill_context=nullcontext,
        _wide_f16_activation_context=nullcontext)
    with pytest.raises(RuntimeError, match='observed'):
        next(runtime.Qwen4ExpGGUFResidentModelRunner._prefill_chunked_steps(owner, [3] * 73))
    assert seen == [(rowbatch, variant)]


@pytest.mark.parametrize('interface', ['profile_gap', 'candidate_ar'])
@pytest.mark.parametrize('overrides,expected', [
    ({'HIPENGINE_QWEN4_EXP_RAW_ROWBATCH': '1'}, (32, 'coltile16')),
    ({'HIPENGINE_QWEN4_EXP_RAW_ROWBATCH': 'false'}, (0, 'coltile16')),
    ({'HIPENGINE_QWEN4_EXP_RAW_VARIANT': 'coltile8'}, (8, 'coltile8')),
    ({'HIPENGINE_QWEN4_EXP_RAW_ROWBATCH': '16',
      'HIPENGINE_QWEN4_EXP_RAW_VARIANT': 'rowbatch'}, (16, 'rowbatch')),
])
def test_override_binds_only_requested_axes_restores_and_does_not_write_process(
    monkeypatch, interface, overrides, expected,
):
    from contextlib import ExitStack
    import importlib
    import os
    script = importlib.import_module('scripts.qwen4exp_' + interface)
    policy = runtime.RawPrefillPolicy(8, 'coltile16')
    owner = SimpleNamespace(raw_prefill_policy=policy, runner=SimpleNamespace(raw_prefill_policy=policy))
    for key in ('HIPENGINE_QWEN4_EXP_RAW_ROWBATCH', 'HIPENGINE_QWEN4_EXP_RAW_VARIANT'):
        monkeypatch.setenv(key, 'invalid')
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match='injected'):
        with ExitStack() as stack:
            report = script.bind_raw_prefill_override(stack, owner, overrides)
            selected = runtime.resolve_raw_prefill_policy(owner)
            assert (selected.rowbatch, selected.variant) == expected
            assert owner.runner.raw_prefill_policy is selected
            assert report['raw_prefill_policy'] == selected.to_report()
            assert _inherit_lane_decisions(owner, SimpleNamespace()).raw_prefill_policy is selected
            assert dict(os.environ) == before
            raise RuntimeError('injected')
    assert owner.raw_prefill_policy is owner.runner.raw_prefill_policy is policy


@pytest.mark.parametrize('interface', ['profile_gap', 'candidate_ar'])
def test_override_transport_removes_process_keys(monkeypatch, interface):
    import importlib
    import os
    script = importlib.import_module('scripts.qwen4exp_' + interface)
    overrides = {'HIPENGINE_QWEN4_EXP_RAW_VARIANT': 'coltile8',
                 'HIPENGINE_QWEN4_EXP_RAW_ROWBATCH': '8', 'HIPENGINE_HIP_ARCH': 'same'}
    for key in overrides:
        monkeypatch.setenv(key, 'prior')
    if interface == 'candidate_ar':
        environment, selected = script.split_raw_prefill_overrides(overrides)
        assert selected == {key: value for key, value in overrides.items() if key != 'HIPENGINE_HIP_ARCH'}
        assert environment == {'HIPENGINE_HIP_ARCH': 'same'}
    else:
        script._apply_post_binder_overrides(overrides)
        assert os.environ['HIPENGINE_HIP_ARCH'] == 'same'
    assert os.environ['HIPENGINE_QWEN4_EXP_RAW_VARIANT'] == 'prior'
    assert os.environ['HIPENGINE_QWEN4_EXP_RAW_ROWBATCH'] == 'prior'


@pytest.mark.parametrize('overrides', [
    {'HIPENGINE_QWEN4_EXP_RAW_VARIANT': 'bad'},
    {'HIPENGINE_QWEN4_EXP_RAW_ROWBATCH': '3'},
    {'HIPENGINE_QWEN4_EXP_RAW_ROWBATCH': 'invalid'},
])
def test_invalid_override_does_not_mutate_owner(overrides):
    from contextlib import ExitStack
    from scripts.qwen4exp_profile_gap import bind_raw_prefill_override
    owner = SimpleNamespace(raw_prefill_policy=runtime.RawPrefillPolicy())
    prior = owner.raw_prefill_policy
    with ExitStack() as stack, pytest.raises(ValueError):
        bind_raw_prefill_override(stack, owner, overrides)
    assert owner.raw_prefill_policy is prior


def test_override_main_wiring_precedes_process_writes():
    import inspect
    from scripts import qwen4exp_candidate_ar as ar, qwen4exp_profile_gap as gap
    source = inspect.getsource(ar.main)
    assert source.index('split_raw_prefill_overrides(overrides)') < source.index('os.environ.update(overrides)')
    assert "artifact['owner_overrides'].update(bind_raw_prefill_override(" in source
    assert 'report.update(bind_raw_prefill_override(' in inspect.getsource(gap.main)


def test_campaign_forces_owner_policy_without_touching_raw_keys():
    from scripts.qwen4exp_halo_box_campaign_ab import q8_wave_scope, q8_wave_control_record
    owner = SimpleNamespace(q8_wave_scale_prefill=False,
                            raw_prefill_policy=runtime.RawPrefillPolicy(0, 'rowbatch'))
    prior = owner.raw_prefill_policy
    env = {'HIPENGINE_QWEN4_EXP_RAW_VARIANT': 'invalid',
           'HIPENGINE_QWEN4_EXP_RAW_ROWBATCH': 'invalid'}
    before = dict(env)
    with pytest.raises(RuntimeError, match='injected'):
        with q8_wave_scope(owner, True, environment=env):
            assert owner.raw_prefill_policy == runtime.RawPrefillPolicy(32, 'coltile8')
            record = q8_wave_control_record(owner)
            assert record['forced_raw_prefill_policy'] == owner.raw_prefill_policy.to_report()
            assert 'HIPENGINE_QWEN4_EXP_RAW_VARIANT' not in record['forced_environment']
            assert env['HIPENGINE_QWEN4_EXP_RAW_VARIANT'] == 'invalid'
            assert env['HIPENGINE_QWEN4_EXP_RAW_ROWBATCH'] == 'invalid'
            raise RuntimeError('injected')
    assert env == before
    assert owner.raw_prefill_policy is prior
    assert owner.q8_wave_scale_prefill is False



def test_unbound_raw_policy_ignores_retired_process_inputs(monkeypatch):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_RAW_ROWBATCH', 'invalid')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_RAW_VARIANT', 'invalid')
    assert runtime.resolve_raw_prefill_policy(SimpleNamespace()) == runtime.RawPrefillPolicy()


@pytest.mark.parametrize('rowbatch,variant', [(1, 'coltile8'), (True, 'coltile8'), (32, 'bad')])
def test_invalid_policy_rejected(rowbatch, variant):
    with pytest.raises(ValueError):
        runtime.RawPrefillPolicy(rowbatch, variant)


def test_scope_isolates_restores_missing_nested_failure_and_late_runner():
    owner = SimpleNamespace(runner=SimpleNamespace())
    peer = SimpleNamespace()
    policy = runtime.RawPrefillPolicy(8, 'coltile16')
    with pytest.raises(RuntimeError):
        with runtime.raw_prefill_policy(owner, policy):
            assert owner.raw_prefill_policy is owner.runner.raw_prefill_policy is policy
            assert not hasattr(peer, 'raw_prefill_policy')
            assert _inherit_lane_decisions(owner, SimpleNamespace()).raw_prefill_policy is policy
            with runtime.raw_prefill_policy(owner, runtime.RawPrefillPolicy(0, 'rowbatch')):
                assert owner.raw_prefill_policy.rowbatch == 0
            assert owner.raw_prefill_policy is policy
            raise RuntimeError('injected')
    assert not hasattr(owner, 'raw_prefill_policy')
    assert not hasattr(owner.runner, 'raw_prefill_policy')


def test_late_constructor_carries_policy(monkeypatch):
    from hipengine.generation import qwen4_exp_gguf as generation
    calls = []
    monkeypatch.setattr(generation, 'Qwen4ExpGGUFResidentModelRunner',
        lambda resident, **kw: calls.append(kw) or SimpleNamespace(**kw))
    owner = generation.Qwen4ExpGGUFTextGenerator(model_path='unused.gguf',
        weight_index=SimpleNamespace(), model_plugin=SimpleNamespace(), tokenizer=SimpleNamespace(),
        runner=SimpleNamespace(max_sequence_length=8, prefill_chunk_size=4, close=lambda: None))
    policy = runtime.RawPrefillPolicy(8, 'coltile16')
    owner.raw_prefill_policy = policy
    try:
        clone = owner._new_resident_runner(SimpleNamespace(), max_sequence_length=8, prefill_chunk_size=4)
    finally:
        owner.close()
    assert calls[0]['raw_prefill_policy'] is clone.raw_prefill_policy is policy
