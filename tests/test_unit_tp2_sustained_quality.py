"""Shared-chain D128 quality mechanics; no hardware, timing, or model files."""
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.tp2_teacher_coverage_broad import sustained_trajectory


class Adapter:
    vocab_size=16
    def __init__(self): self.inputs=[]; self.position=0; self.closed=False
    def prefill(self,tokens): self.position=len(tokens); return 1
    def begin_decode(self,steps): self.steps=steps
    def force_input(self,token): self.forced=token
    def transition(self,token,**kw):
        assert token==self.forced and kw['return_logits'] is True
        self.inputs.append(token); self.position+=1
        row=np.zeros((1,16),dtype=np.float32); row[0,(token+1)%16]=4
        return SimpleNamespace(token_id=(token+1)%16,logits=row)
    def check_transition(self,token,position): return {'position':position,'input_token':token}
    def end_decode(self): self.closed=True


def test_shared_teacher_inputs_not_candidate_rollout():
    a=Adapter()
    logits,inputs,controls=sustained_trajectory(a,[0,1],forced=[8,3,2],steps=3)
    assert a.inputs==inputs==[8,3,2]
    assert logits.argmax(axis=1).tolist()==[9,4,3]
    assert [c['position'] for c in controls]==[2,3,4]
    assert a.closed


def test_chosen_teacher_and_replay_are_same_schedule():
    a=Adapter()
    first,inputs,_=sustained_trajectory(a,[0,1],steps=3)
    b=Adapter()
    second,_,_=sustained_trajectory(b,[0,1],forced=inputs,steps=3)
    assert inputs==[1,2,3]
    np.testing.assert_array_equal(first,second)


def test_absolute_numerical_failure_stops_first_bad_position():
    a=Adapter(); ref=np.zeros((3,16),dtype=np.float32); ref[:,10]=4
    failures=[]
    with pytest.raises(ValueError,match='absolute KL'):
        sustained_trajectory(a,[0,1],forced=[8,3,2],steps=3,reference=ref,
                             failure=lambda detail,*arrays:failures.append(detail))
    assert a.inputs==[8] and not a.closed
    assert failures[0]['position']==2


def test_wrong_control_position_fails_before_next_transition():
    a=Adapter()
    a.check_transition=lambda token,position: {'position':position+1,'input_token':token}
    with pytest.raises(ValueError,match='control'):
        sustained_trajectory(a,[0,1],forced=[8,3,2],steps=3)
    assert a.inputs==[8]
