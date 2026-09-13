"""Independent control oracle and byte checks for actual C1 EOS commits."""
import numpy as np
from scripts.qwen38_packed_c1_logits import _read_device
from scripts.qwen38_packed_c1_state import (snapshot_committed_state,
    selected_state_sources, selected_aux_sources, assert_aux_commit)
from scripts.qwen38_packed_c1_kv import selected_kv_sources, assert_kv_commit


def eos_selected_prefix(tokens, target_top1, remaining, eos):
    if len(tokens) != len(target_top1) or not tokens or remaining < 1:
        raise ValueError('invalid EOS selected-prefix input')
    accepted = 0
    while True:
        predicted = int(target_top1[accepted])
        if predicted == eos:
            return accepted, True
        if accepted + 1 >= min(len(tokens), remaining):
            return accepted, False
        if predicted != int(tokens[accepted + 1]):
            return accepted, False
        accepted += 1


class EosStateProbe:
    def __init__(self, session, *, eos, remaining):
        self.session, self.eos, self.remaining = session, eos, remaining
        self.before = snapshot_committed_state(session)
        self.evidence = {}

    def verified(self, owner, result, tokens):
        if snapshot_committed_state(self.session) != self.before:
            raise ValueError('EOS verification changed canonical state before commit')
        top = _read_device(result.target_top1.ptr, (len(tokens),), np.int32, owner.runtime)
        self.accepted, self.terminal = eos_selected_prefix(tokens, top, self.remaining, self.eos)
        self.evidence.update(precommit_buffers=len(self.before['buffers']),
            accepted=self.accepted, terminal=self.terminal, start_position=int(result.start_position))

    def commit(self, original, owner, results, sessions, *, accept_buffers, **kwargs):
        if len(results) != 1 or len(sessions) != 1 or sessions[0] is not self.session:
            raise ValueError('EOS selected-commit ownership changed')
        result = results[0]
        for name, expected in (('accepted_counts', self.accepted),
                               ('commit_positions', int(result.start_position) + self.accepted)):
            actual = _read_device(getattr(accept_buffers,name).ptr,(1,),np.int32,owner.runtime)
            if int(actual[0]) != expected:
                raise ValueError('EOS device acceptance differs from independent oracle')
        state = selected_state_sources(owner,self.session,selected_row=int(result.row_start)+self.accepted)
        aux = selected_aux_sources(self.session,result,accepted=self.accepted)
        kv = selected_kv_sources(self.session,result,accepted=self.accepted)
        output = original(owner,results,sessions,accept_buffers=accept_buffers,**kwargs)
        assert_aux_commit(self.session,aux)
        assert_kv_commit(self.session,kv)
        after = snapshot_committed_state(self.session)
        for name, expected in state.items():
            actual = after['buffers'][name]
            if (actual['ptr'],actual['allocation_nbytes'],actual['blake2b_128']) != (expected['ptr'],expected['nbytes'],expected['hash']):
                raise ValueError('EOS selected-state bytes differ from selected source row')
        self.evidence.update(passed=True,state_buffers=len(state),aux_buffers=len(aux),kv_planes=len(kv['buffers']))
        return output
