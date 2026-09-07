"""CPU RED contracts for independent selected-row commit instrumentation."""
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize('top1,remaining,expected', [
    ([9, 8, 7], 8, 0), ([2, 9, 7], 8, 1), ([2, 3, 7], 8, 2),
    ([2, 3, 7], 1, 0), ([2, 3, 7], 2, 1),
])
def test_independent_greedy_prefix(top1, remaining, expected):
    from scripts.qwen38_packed_c1_state import selected_prefix
    assert selected_prefix([1, 2, 3], top1, remaining) == expected


@pytest.mark.parametrize('tokens,top1,remaining', [([], [], 1), ([1], [], 1), ([1], [2], 0)])
def test_prefix_fails_closed(tokens, top1, remaining):
    from scripts.qwen38_packed_c1_state import selected_prefix
    with pytest.raises(ValueError):
        selected_prefix(tokens, top1, remaining)


def test_selected_source_uses_absolute_row_and_preserves_destination(monkeypatch):
    from scripts import qwen38_packed_c1_state as m
    monkeypatch.setattr(m, '_device_hash', lambda session, b, **kw: f'{b.ptr}:{b.nbytes}')
    b = lambda p, n: NS(ptr=p, nbytes=n)
    scratch = NS(layer_conv_states=[b(10, 8)], layer_recurrent_states=[b(20, 16)],
                 hidden_seed_fp32=b(30, 4))
    owner = NS(runtime=NS(device_synchronize=lambda: None),
               _verify_linear_state_row_pair=lambda layer: (b(100, 40), b(200, 80)),
               _verify_hidden_seed_buf=b(300, 20))
    session = NS(scratch=scratch)
    expected = m.selected_state_sources(owner, session, selected_row=3)
    assert expected == {'conv:0': {'ptr': 10, 'nbytes': 8, 'hash': '124:8'},
                        'recurrent:0': {'ptr': 20, 'nbytes': 16, 'hash': '248:16'},
                        'hidden_seed': {'ptr': 30, 'nbytes': 4, 'hash': '312:4'}}
    with pytest.raises(ValueError):
        m.selected_state_sources(owner, session, selected_row=5)


@pytest.mark.parametrize('fault', [None, 'accepted_counts', 'commit_positions', 'ptr', 'allocation_nbytes', 'blake2b_128'])
def test_real_commit_hook_checks_metadata_and_destination(monkeypatch, tmp_path, fault):
    import numpy as np
    from scripts import qwen38_packed_c1_logits as capture
    from scripts import qwen38_packed_c1_state as state
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession as Session
    calls = []
    def commit(*args, **kwargs):
        calls.append(True)
        return 'executed'
    monkeypatch.setattr(Session, '_commit_deferred_packed_verify_states_batch_device', commit)
    monkeypatch.setattr(capture, '_read_device', lambda ptr, *args: np.array([ptr]))
    def sources(owner, session, *, selected_row):
        assert selected_row == 4  # absolute row_start 3 plus accepted prefix 1
        return {'conv:0': dict(ptr=10, nbytes=8, hash='selected')}
    monkeypatch.setattr(state, 'selected_state_sources', sources)
    aux_calls = []
    monkeypatch.setattr(state, 'selected_aux_sources', lambda *args, **kwargs: {'aux': 'expected'})
    monkeypatch.setattr(state, 'assert_aux_commit', lambda *args: aux_calls.append(args[1]))
    actual = dict(ptr=10, allocation_nbytes=8, blake2b_128='selected')
    if fault in actual:
        actual[fault] = 'wrong'
    monkeypatch.setattr(state, 'snapshot_committed_state', lambda session: dict(buffers={'conv:0': actual}))
    recorder = capture.PackedC1Capture(tmp_path / 'capture', check_commit=True)
    logits = np.zeros((2, 4)); logits[0, 2] = 1; logits[1, 3] = 1
    np.savez(recorder.directory / 'rows.npz', logits=logits)
    recorder.records.append(dict(tokens=[1, 2], position=5, logits_file='rows.npz'))
    recorder.current.set(dict(record_index=0, remaining_decode=3))
    buffers = NS(accepted_counts=NS(ptr=99 if fault == 'accepted_counts' else 1),
                 commit_positions=NS(ptr=99 if fault == 'commit_positions' else 6))
    try:
        recorder.install()
        def run():
            return Session._commit_deferred_packed_verify_states_batch_device(
                NS(runtime=None), [NS(start_position=5, row_start=3)], [object()], accept_buffers=buffers)
        if fault:
            with pytest.raises(ValueError):
                run()
            assert 'selected_commit' not in recorder.records[0]
        else:
            assert run() == 'executed'
            assert recorder.records[0]['selected_commit'] == dict(passed=True, accepted=1, checked_buffers=1)
        assert bool(calls) is (fault not in {'accepted_counts', 'commit_positions'})
        assert aux_calls == ([{'aux': 'expected'}] if calls else [])
    finally:
        recorder.close(success=fault is None)
    assert Session._commit_deferred_packed_verify_states_batch_device is commit
