"""Independent checkpoint byte/cursor checks for provider recovery."""
from types import SimpleNamespace as NS
import hashlib
import numpy as np
import pytest


def test_restore_probe_records_and_reraises_checker_failure(monkeypatch):
    from scripts import qwen38_packed_c1_recovery as helper
    calls = []
    monkeypatch.setattr(helper, 'provider_restore_expected', lambda *args: {'buffers': [{}]})
    def check(*args):
        raise ValueError('state corruption')
    monkeypatch.setattr(helper, 'assert_provider_restored', check)
    probe = helper.PrecommitProbe()
    with pytest.raises(ValueError, match='state corruption'):
        probe.restore(lambda *args: calls.append('restore'), object(), object())
    assert calls == ['restore']
    assert probe.evidence['provider_restores'] == 0
    assert probe.evidence['provider_restore_error'] == 'ValueError: state corruption'


@pytest.mark.parametrize('dirty', [False, True])
def test_precommit_probe_checks_target_before_injecting(monkeypatch, dirty):
    from scripts.qwen38_packed_c1_recovery import PrecommitProbe, InjectedPrecommitFailure
    from scripts import qwen38_packed_c1_state as state
    values = iter([dict(position=5, buffers={'x': 'old'}),
                   dict(position=5, buffers={'x': 'bad' if dirty else 'old'})])
    monkeypatch.setattr(state, 'snapshot_committed_state', lambda session: next(values))
    probe = PrecommitProbe()
    probe.prepare(object(), 7)
    with pytest.raises(ValueError if dirty else InjectedPrecommitFailure):
        probe.inject()
    assert probe.evidence['injected'] is (not dirty)
    assert probe.evidence['request_id'] == 7


@pytest.mark.parametrize('fault', [None, 'missing_restore', 'not_recovered', 'unexpected', 'wrong_id'])
def test_recovery_requires_exact_injected_failure_and_restoration(fault):
    from scripts.qwen38_packed_c1_recovery import validate_recovery
    rows = [dict(request_ids=[2, 3], active_request_ids=[2, 3], scheduler_slots=[0, 1],
                 resident_slots=[1, 0], packed_request_ids=[2, 3], packed_group_sizes=[2],
                 native_c1=False, error=None),
            dict(request_ids=[3], active_request_ids=[3], scheduler_slots=[1],
                 resident_slots=[0], packed_request_ids=[3], packed_group_sizes=[1],
                 native_c1=True, error='InjectedPrecommitFailure: diagnostic')]
    evidence = dict(request_id=3, injected=True, provider_restores=1,
                    target_unchanged=True, recovered=True)
    if fault == 'missing_restore': evidence['provider_restores'] = 0
    elif fault == 'not_recovered': evidence['recovered'] = False
    elif fault == 'unexpected': rows[-1]['error'] = 'RuntimeError: unexpected'
    elif fault == 'wrong_id': evidence['request_id'] = 9
    if fault:
        with pytest.raises(ValueError): validate_recovery(rows, evidence)
    else:
        assert validate_recovery(rows, evidence)['survivor_request_id'] == 3
        assert rows[-1]['error'] is not None


@pytest.mark.parametrize('fault', [None, 'bytes', 'cursor', 'logical', 'slot', 'released', 'empty_valid', 'missing_state'])
@pytest.mark.parametrize('batch_sessions', [True, False])
def test_provider_checkpoint_restore_checks_actual_state(monkeypatch, fault, batch_sessions):
    from scripts import qwen38_packed_c1_recovery as helper
    data = {10: b'old!', 20: b'old!', 30: np.array([5], dtype=np.int64).tobytes(),
            40: np.array([6], dtype=np.int64).tobytes()}
    def buffer(ptr, nbytes=4): return NS(ptr=ptr, nbytes=nbytes)
    scratch = NS(position_host=[5], context_host=[6], position_buf=buffer(30, 8), context_buf=buffer(40, 8))
    executor = NS(runtime=NS(device_synchronize=lambda: None), _request_slots={7: 1},
                  scratch=NS(for_slot=lambda slot, span_role: scratch),
                  _batch_sessions=[None, NS(position=5)] if batch_sessions else None)
    checkpoint = NS(request_id=7, slot=1, position=5, context_length=6, released=False,
                    state_pairs=((buffer(10), buffer(20)),))
    scratch.layer_conv_states = [checkpoint.state_pairs[0][0]]
    scratch.layer_recurrent_states = [None]
    if fault in ('empty_valid', 'missing_state'):
        checkpoint.state_pairs = ()
    if fault == 'empty_valid':
        scratch.layer_conv_states = [None]
    monkeypatch.setattr(helper, '_device_hash', lambda owner, b: hashlib.blake2b(data[b.ptr], digest_size=16).hexdigest())
    if fault == 'missing_state':
        with pytest.raises(ValueError): helper.provider_restore_expected(executor, checkpoint)
        return
    expected = helper.provider_restore_expected(executor, checkpoint)
    if fault == 'bytes': data[10] = b'bad!'
    elif fault == 'cursor': data[30] = np.array([9], dtype=np.int64).tobytes()
    elif fault == 'logical':
        if batch_sessions: executor._batch_sessions[1].position = 9
        else: scratch.position_host[0] = 9
    elif fault == 'slot': executor._request_slots[7] = 0
    elif fault == 'released': checkpoint.released = True
    if fault not in (None, 'empty_valid'):
        with pytest.raises(ValueError): helper.assert_provider_restored(executor, checkpoint, expected)
    else:
        helper.assert_provider_restored(executor, checkpoint, expected)
