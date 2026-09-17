"""Shared-chain D128 quality mechanics; no hardware, timing, or model files."""
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.tp2_teacher_coverage_broad import sustained_trajectory


def _captures(tmp_path, schedules):
    paths=[]; captures={}
    for arm in ('tp1-d0','tp1-d1','tp2'):
        path=tmp_path/(arm+'.json'); path.write_text('{}'); paths.append(path)
        data={'arm':arm,'identity':{},'suite':{'categories':[], 'heldout':[]},'forced_inputs':[],
              'vocab_size':3,'profile':{},'run_id':arm,'devices':{},'route':{},'scope_manifest':{},
              'determinism':{},'state_boundaries':{},'control_log':{},'natural_teardown':True}
        if schedules is not None and schedules.get(arm) is not None:
            data['prefill_schedule']=schedules[arm]
        captures[path]=(data,[object()])
    return paths,captures


def _metric(max_kl=0.):
    return {'rows':2304,'mean_kl':0.,'p95_kl':0.,'p99_kl':0.,'max_kl':max_kl,'top1_agreement':1.}


def _patch_score(monkeypatch, coverage, max_kl_by_arm=None):
    calls=[]
    def score(teacher,student,*args,**kw):
        assert teacher is not student, 'redundant expensive self-comparison'
        calls.append(student)
        max_kl=(max_kl_by_arm or {}).get(len(calls),0.)
        metric=_metric(max_kl)
        return {'global':metric,'scopes':{'canonical':metric,'heldout':metric},
                'categories':{'code':metric},'category_scopes':{'code':{'canonical':metric,'heldout':metric}}}
    monkeypatch.setattr(coverage,'score_arm',score)
    return calls


def test_report_scores_candidates_not_redundant_teacher_self(monkeypatch,tmp_path):
    import json
    from scripts import tp2_teacher_coverage_broad as coverage
    paths,captures=_captures(tmp_path,{'tp1-d0':'bulk','tp1-d1':'bulk','tp2':'bulk'})
    monkeypatch.setattr(coverage,'load_sustained',lambda p:captures[p])
    calls=_patch_score(monkeypatch,coverage)
    out=tmp_path/'report.json'
    assert coverage.report_sustained(SimpleNamespace(sustained_report=paths,json=str(out)))==0
    assert len(calls)==2
    report=json.loads(out.read_text())
    assert report['reference_arm']=='tp1-d0'
    assert report['prefill_schedules']=={'tp1-d0':'bulk','tp1-d1':'bulk','tp2':'bulk'}
    assert report['mixed_prefill_schedules'] is False
    assert report['comparison_scope']=='matched-prefill-schedule'


def test_report_accepts_mixed_prefill_schedules_when_envelope_passes(monkeypatch,tmp_path):
    import json
    from scripts import tp2_teacher_coverage_broad as coverage
    paths,captures=_captures(tmp_path,{'tp1-d0':'bulk','tp1-d1':'bulk','tp2':'token-serial'})
    monkeypatch.setattr(coverage,'load_sustained',lambda p:captures[p])
    calls=_patch_score(monkeypatch,coverage)
    out=tmp_path/'report.json'
    assert coverage.report_sustained(SimpleNamespace(sustained_report=paths,json=str(out)))==0
    assert len(calls)==2
    report=json.loads(out.read_text())
    assert report['prefill_schedules']['tp2']=='token-serial'
    assert report['mixed_prefill_schedules'] is True
    assert report['comparison_scope']=='mixed-prefill-schedules'
    assert set(report['comparison'])=={'tp1-d1','tp2'}


def test_report_keeps_mixed_prefill_schedule_numerical_failure_failed(monkeypatch,tmp_path):
    import json
    from scripts import tp2_teacher_coverage_broad as coverage
    paths,captures=_captures(tmp_path,{'tp1-d0':'bulk','tp1-d1':'bulk','tp2':'token-serial'})
    monkeypatch.setattr(coverage,'load_sustained',lambda p:captures[p])
    calls=_patch_score(monkeypatch,coverage,{2:0.108406})
    out=tmp_path/'report.json'
    assert coverage.report_sustained(SimpleNamespace(sustained_report=paths,json=str(out)))==1
    assert len(calls)==2
    report=json.loads(out.read_text())
    assert report['mixed_prefill_schedules'] is True
    assert report['all_gates_passed'] is False
    assert report['comparison']['tp2']['global']['max_kl']==0.108406


def test_report_blocks_missing_prefill_schedule_provenance(monkeypatch,tmp_path):
    from scripts import tp2_teacher_coverage_broad as coverage
    paths,captures=_captures(tmp_path,{'tp1-d0':'bulk','tp1-d1':'bulk','tp2':None})
    monkeypatch.setattr(coverage,'load_sustained',lambda p:captures[p])
    with pytest.raises(ValueError,match='missing prefill_schedule provenance'):
        coverage.report_sustained(SimpleNamespace(sustained_report=paths,json=None))


def test_prefill_schedule_provenance_is_recorded():
    from scripts import tp2_teacher_coverage_broad as coverage
    from scripts.tp2_resident_control import NativeARAdapter
    runtime=SimpleNamespace()
    tp1=NativeARAdapter(SimpleNamespace(session=SimpleNamespace(runtime=runtime),vocab_size=7),
                        resident=True)
    tp2=NativeARAdapter(SimpleNamespace(runtime=runtime,vocab_size=7),resident=False)
    assert tp1.prefill_schedule=='bulk'
    assert tp2.prefill_schedule=='token-serial'
    route=coverage._resolved_route(SimpleNamespace(
        mode='tp2',schedule='graphed',driver='compiled',reduce_mode='device',
        head_shard=True,max_sequence_length=200,prefill_schedule='token-serial'))
    assert route['prefill_schedule']=='token-serial'


def test_adapter_reports_the_tp2_bulk_candidate_schedule():
    from scripts.tp2_resident_control import NativeARAdapter, create_native_adapter
    runtime=SimpleNamespace()
    bulk=SimpleNamespace(runtime=runtime,vocab_size=7,bulk_prefill_enabled=True)
    serial=SimpleNamespace(runtime=runtime,vocab_size=7,bulk_prefill_enabled=False)
    assert NativeARAdapter(bulk,resident=False).prefill_schedule=='bulk-tp2'
    assert NativeARAdapter(serial,resident=False).prefill_schedule=='token-serial'
    # The resident arm is bulk by construction and must not be handed the TP2
    # candidate, which the factory refuses before any session is created.
    with pytest.raises(ValueError,match='TP1 arm is already bulk'):
        create_native_adapter('unused.gguf','tp1-d0',bulk_prefill=True)


def test_resident_prefill_schedule_switch_is_passed_through():
    from scripts.tp2_resident_control import NativeARAdapter
    calls=[]
    class Session:
        runtime=SimpleNamespace()
        def reset(self): self.position=0
        def prefill(self,tokens,**kw):
            calls.append(kw)
            return SimpleNamespace(token_id=1)
    adapter=NativeARAdapter(SimpleNamespace(session=Session(),vocab_size=7),resident=True)
    assert adapter.prefill_use_bulk is None
    adapter.prefill([3,4])
    assert calls[-1]['use_bulk'] is None
    adapter.prefill_use_bulk=False
    adapter.prefill([3,4])
    assert calls[-1]['use_bulk'] is False


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


def test_same_top1_does_not_waive_absolute_kl_ceiling():
    a=Adapter()
    ref=np.zeros((3,16),dtype=np.float32)
    ref[:,9]=4; ref[:,8]=3.9  # same winner as candidate, different full distribution
    failures=[]
    with pytest.raises(ValueError,match='absolute KL'):
        sustained_trajectory(a,[0,1],forced=[8,3,2],steps=3,reference=ref,
                             failure=lambda detail,*arrays:failures.append(detail))
    assert failures[0]['top1'] is True
    assert a.inputs==[8] and not a.closed


def test_wrong_control_position_fails_before_next_transition():
    a=Adapter()
    a.check_transition=lambda token,position: {'position':position+1,'input_token':token}
    with pytest.raises(ValueError,match='control'):
        sustained_trajectory(a,[0,1],forced=[8,3,2],steps=3)
    assert a.inputs==[8]
