"""Independent EOS selected-row oracle; never calls production EOS limiting."""
import pytest


@pytest.mark.parametrize('depth', range(1,8))
def test_eos_oracle_all_positions_and_rejections(depth):
    from scripts.qwen38_packed_c1_eos_state import eos_selected_prefix
    tokens = list(range(100,101+depth))
    for reject in range(depth+1):
        top = tokens[1:]+[900]; top[reject]=900
        visible=tokens[1:reject+1]+[900]
        for index,eos in enumerate(visible):
            for remaining in range(1,depth+3):
                accepted,terminal=eos_selected_prefix(tokens,top,remaining,eos)
                assert accepted == min(reject,remaining-1,index)
                assert terminal is (index < remaining)


@pytest.mark.parametrize('fault', [None, 'precommit', 'accepted', 'position', 'state', 'aux', 'kv'])
def test_probe_rejects_wrong_state_and_acceptance(monkeypatch, fault):
    from types import SimpleNamespace as NS
    from scripts import qwen38_packed_c1_eos_state as m
    before = dict(buffers={'hidden': dict(ptr=10, allocation_nbytes=4, blake2b_128='old')})
    after = dict(buffers={'hidden': dict(ptr=10, allocation_nbytes=4,
                                       blake2b_128='bad' if fault == 'state' else 'selected')})
    snapshots = iter([before, {} if fault == 'precommit' else before, after])
    monkeypatch.setattr(m, 'snapshot_committed_state', lambda s: next(snapshots))
    monkeypatch.setattr(m, '_read_device', lambda ptr,*args: [2,9,4] if ptr == 100 else [ptr])
    monkeypatch.setattr(m, 'selected_state_sources', lambda *args,**kwargs:
                        {'hidden':dict(ptr=10,nbytes=4,hash='selected')})
    monkeypatch.setattr(m, 'selected_aux_sources', lambda *args,**kwargs: {'hidden':1})
    monkeypatch.setattr(m, 'selected_kv_sources', lambda *args,**kwargs: {'buffers':{'k':1,'v':1}})
    def check(name):
        def verify(*args):
            if fault == name: raise ValueError(name)
        return verify
    monkeypatch.setattr(m, 'assert_aux_commit', check('aux'))
    monkeypatch.setattr(m, 'assert_kv_commit', check('kv'))
    session = object(); owner = NS(runtime=None)
    result = NS(target_top1=NS(ptr=100),start_position=73,row_start=0)
    probe = m.EosStateProbe(session,eos=9,remaining=8)
    if fault == 'precommit':
        with pytest.raises(ValueError): probe.verified(owner,result,[1,2,3])
        return
    probe.verified(owner,result,[1,2,3])
    assert probe.evidence['terminal'] and probe.accepted == 1
    buffers = NS(accepted_counts=NS(ptr=99 if fault=='accepted' else 1),
                 commit_positions=NS(ptr=99 if fault=='position' else 74))
    calls=[]
    def commit(*args,**kwargs): calls.append(True); return 'committed'
    if fault:
        with pytest.raises(ValueError): probe.commit(commit,owner,[result],[session],accept_buffers=buffers)
        assert not probe.evidence.get('passed',False)
    else:
        assert probe.commit(commit,owner,[result],[session],accept_buffers=buffers) == 'committed'
        assert probe.evidence['passed']
    assert bool(calls) is (fault not in {'accepted','position'})


def test_eos_oracle_ignores_unreachable_suffix():
    from scripts.qwen38_packed_c1_eos_state import eos_selected_prefix
    assert eos_selected_prefix([1,2,3],[8,3,9],10,9)==(0,False)
