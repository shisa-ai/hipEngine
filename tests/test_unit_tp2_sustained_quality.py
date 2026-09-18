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
    assert adapter.prefill_schedule=='bulk'
    adapter.prefill([3,4])
    assert calls[-1]['use_bulk'] is None
    adapter.prefill_use_bulk=False
    assert adapter.prefill_schedule=='token-serial'
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


def test_rows_per_prompt_falls_back_to_the_capture_standard():
    import numpy as np
    from scripts import tp2_teacher_coverage_broad as coverage
    assert coverage._rows_per_prompt([np.zeros((128,3))])==128
    assert coverage._rows_per_prompt([np.zeros((42,3))])==42
    assert coverage._rows_per_prompt([object()])==128


def _captures_with_rows(tmp_path,schedules,rows):
    """Real arrays, so the horizon slicing path is exercised end to end."""
    import numpy as np
    paths=[]; captures={}
    for arm in ('tp1-d0','tp1-d1','tp2'):
        path=tmp_path/(arm+'.json'); path.write_text('{}'); paths.append(path)
        data={'arm':arm,'identity':{},'suite':{'categories':[], 'heldout':[]},'forced_inputs':[],
              'vocab_size':3,'profile':{},'run_id':arm,'devices':{},'route':{},'scope_manifest':{},
              'determinism':{},'state_boundaries':{},'control_log':{},'natural_teardown':True,
              'prefill_schedule':schedules[arm]}
        captures[path]=(data,[np.zeros((rows,3),dtype=np.float32) for _ in range(18)])
    return paths,captures


def test_report_scores_only_the_declared_horizon_and_records_the_rest(monkeypatch,tmp_path):
    import json
    from scripts import tp2_teacher_coverage_broad as coverage
    schedules={'tp1-d0':'bulk','tp1-d1':'bulk','tp2':'token-serial'}
    paths,captures=_captures_with_rows(tmp_path,schedules,128)
    monkeypatch.setattr(coverage,'load_sustained',lambda p:captures[p])
    widths=[]
    def score(teacher,student,*args,**kw):
        widths.append(kw['expected_positions'][0])
        assert len(teacher)==len(student)==18
        assert all(row.shape[0]==kw['expected_positions'][0] for row in student)
        metric=_metric()
        return {'global':metric,'scopes':{'canonical':metric,'heldout':metric},
                'categories':{'code':metric},'category_scopes':{'code':{'canonical':metric,'heldout':metric}}}
    monkeypatch.setattr(coverage,'score_arm',score)
    out=tmp_path/'report.json'
    assert coverage.report_sustained(SimpleNamespace(sustained_report=paths,json=str(out),horizon=42))==0
    report=json.loads(out.read_text())
    assert report['horizon']==42
    assert report['captured_rows_per_prompt']==128
    assert report['positions']==18*42
    assert 'first 42 aligned generated decode transitions' in report['population']
    assert report['horizon_declared_by']=='docs/EXECUTION-PROFILES.md 6.5'
    # Each candidate arm is scored at the horizon and then again over the full
    # capture as an unscored diagnostic: a short horizon cannot hide the tail.
    assert widths==[42,128,42,128]
    assert set(report['beyond_horizon_diagnostic'])=={'tp1-d1','tp2'}


def test_report_without_a_horizon_scores_every_captured_row(monkeypatch,tmp_path):
    import json
    from scripts import tp2_teacher_coverage_broad as coverage
    schedules={'tp1-d0':'bulk','tp1-d1':'bulk','tp2':'token-serial'}
    paths,captures=_captures_with_rows(tmp_path,schedules,128)
    monkeypatch.setattr(coverage,'load_sustained',lambda p:captures[p])
    widths=[]
    def score(teacher,student,*args,**kw):
        widths.append(kw['expected_positions'][0]); metric=_metric()
        return {'global':metric,'scopes':{'canonical':metric,'heldout':metric},
                'categories':{'code':metric},'category_scopes':{'code':{'canonical':metric,'heldout':metric}}}
    monkeypatch.setattr(coverage,'score_arm',score)
    out=tmp_path/'report.json'
    assert coverage.report_sustained(SimpleNamespace(sustained_report=paths,json=str(out)))==0
    report=json.loads(out.read_text())
    assert widths==[128,128]
    assert report['horizon']==128
    assert report['positions']==2304
    assert report['beyond_horizon_diagnostic'] is None


def test_report_rejects_a_horizon_beyond_the_capture(monkeypatch,tmp_path):
    from scripts import tp2_teacher_coverage_broad as coverage
    schedules={'tp1-d0':'bulk','tp1-d1':'bulk','tp2':'token-serial'}
    paths,captures=_captures_with_rows(tmp_path,schedules,128)
    monkeypatch.setattr(coverage,'load_sustained',lambda p:captures[p])
    for bad in (0,129,-3):
        with pytest.raises(ValueError,match='horizon'):
            coverage.report_sustained(SimpleNamespace(sustained_report=paths,json=None,horizon=bad))


def test_sustained_trajectory_gate_rows_stops_enforcing_past_the_horizon():
    """The capture must still run every row, but only gate the declared ones."""
    import numpy as np
    from scripts.tp2_teacher_coverage_broad import sustained_trajectory
    forced=[8,3,2]
    # Adapter.transition puts a logit of 4 on (token+1)%16, so these are the
    # rows the reference must match for the first two positions to be exact.
    reference=np.zeros((3,16),dtype=np.float32)
    for i,token in enumerate(forced):
        reference[i,(token+1)%16]=4.0
    reference[2]=0.0; reference[2,0]=4.0   # row 2 diverges from the adapter
    logits,inputs,_=sustained_trajectory(Adapter(),[0,1],forced=forced,steps=3,
                                        reference=reference,gate_rows=2)
    assert logits.shape==(3,16)
    assert inputs==forced
    with pytest.raises(ValueError,match='sustained absolute KL ceiling failed'):
        sustained_trajectory(Adapter(),[0,1],forced=forced,steps=3,reference=reference)
    with pytest.raises(ValueError,match='gate_rows outside'):
        sustained_trajectory(Adapter(),[0,1],forced=forced,steps=3,reference=reference,gate_rows=4)


def test_numerical_identity_ignores_scheduling_and_keeps_everything_else():
    from scripts import tp2_teacher_coverage_broad as coverage
    base={'model_sha256':'a','source_sha256':{'x':'y'},'capacity':200,
          'host':{'node':'epyc','nice':16,'cpu_count':32}}
    other={**base,'host':{'node':'epyc','nice':-4,'cpu_count':32}}
    assert coverage.numerical_identity(base)==coverage.numerical_identity(other)
    assert base['host']['nice']==16, 'the raw identity must keep the value'
    for changed in ('model_sha256','capacity'):
        assert coverage.numerical_identity(base)!=coverage.numerical_identity({**base,changed:'z'})
    moved={**base,'host':{**base['host'],'node':'other'}}
    assert coverage.numerical_identity(base)!=coverage.numerical_identity(moved)
    assert coverage.numerical_identity(None)=={}
    assert coverage.numerical_identity({'host':{'nice':16}})=={'host':{}}


def test_report_accepts_arms_captured_at_a_different_niceness(monkeypatch,tmp_path):
    import json
    from scripts import tp2_teacher_coverage_broad as coverage
    schedules={'tp1-d0':'bulk','tp1-d1':'bulk','tp2':'token-serial'}
    paths,captures=_captures_with_rows(tmp_path,schedules,128)
    for arm,nice in (('tp1-d0',16),('tp1-d1',-4),('tp2',0)):
        data,_=captures[[p for p in paths if p.stem==arm][0]]
        data['identity']={'model_sha256':'a','host':{'node':'epyc','nice':nice}}
    monkeypatch.setattr(coverage,'load_sustained',lambda p:captures[p])
    def score(teacher,student,*args,**kw):
        metric=_metric()
        return {'global':metric,'scopes':{'canonical':metric,'heldout':metric},
                'categories':{'code':metric},'category_scopes':{'code':{'canonical':metric,'heldout':metric}}}
    monkeypatch.setattr(coverage,'score_arm',score)
    out=tmp_path/'report.json'
    assert coverage.report_sustained(SimpleNamespace(sustained_report=paths,json=str(out)))==0
    report=json.loads(out.read_text())
    assert report['identity_host_scheduling']=={'tp1-d0':{'nice':16},'tp1-d1':{'nice':-4},'tp2':{'nice':0}}
    assert 'nice' in report['identity_equality_scope']


class _CaptureAdapter(Adapter):
    """Adapter plus the surface capture_sustained_arm reads off the owner."""
    vocab_size=16
    def __init__(self):
        super().__init__(); self.owner=SimpleNamespace(devices=[0]); self.prefill_schedule='token-serial'
    def prepare(self): return {'vocab_size':16}
    def transition(self,token,**kw):
        # The capture's generation stage feeds a sampled token back without a
        # preceding force_input, which the shared-chain Adapter above forbids.
        self.inputs.append(token); self.position+=1
        row=np.zeros((1,16),dtype=np.float32); row[0,(token+1)%16]=4
        return SimpleNamespace(token_id=(token+1)%16,logits=row)
    def destroy_graphs(self): self.graphs_destroyed=True
    def close(self): self.closed_by_capture=True


def _capture_args(tmp_path,horizon):
    import argparse
    return argparse.ArgumentParser().parse_args([]) or None if False else SimpleNamespace(
        sustained_arm='tp2',max_sequence_length=200,repeat_tp2=3,teacher_source=tmp_path/'teacher.json',
        horizon=horizon,json=str(tmp_path/'tp2.json'),tp2_bulk_prefill=False)


def _capture_seams(monkeypatch,tmp_path,forced):
    """Wire capture_sustained_arm to synthetic suite, teacher, and adapter."""
    import numpy as np
    from scripts import tp2_teacher_coverage_broad as coverage
    import scripts.tp2_resident_control as resident
    import scripts.tp2_xtx_tp1_eager_stage_probe as probe
    import scripts.tp2_matched_ar_baseline as matched
    monkeypatch.delenv('HIP_VISIBLE_DEVICES',raising=False)
    categories=coverage.CATEGORIES
    suite=[{'id':f'p{i}','category':categories[i%len(categories)],
            'heldout':(i//len(categories))%2==1,'prompt':'x'} for i in range(18)]
    monkeypatch.setattr(coverage,'_product_suite',lambda: (suite,[[0,1,2] for _ in range(18)]))
    identity={'model_sha256':'m','host':{'node':'epyc','nice':16}}
    monkeypatch.setattr(matched,'product_identity',lambda model: dict(identity))
    monkeypatch.setattr(resident,'create_native_adapter',lambda *a,**kw: _CaptureAdapter())
    monkeypatch.setattr(coverage,'_device_identities',lambda owner: {'0':{'name':'fake'}})
    monkeypatch.setattr(coverage,'_resolved_route',lambda owner: {'mode':'tp2'})
    monkeypatch.setattr(resident,'resolved_scope_manifest',lambda owner: {'scopes':{}})
    monkeypatch.setattr(resident,'bind_resident_profile',lambda name: {'execution_profile':name})
    monkeypatch.setattr(probe,'StageRecorder',_RecordingRecorder)
    # A failed capture calls os._exit(1) by design; in-process that would kill
    # pytest before it can report the failing stage.
    import os
    monkeypatch.setattr(os,'_exit',lambda code: (_ for _ in ()).throw(SystemExit(code)))
    # Teacher: exact for the first three rows, far away for every row after.
    reference=[]
    for _ in range(18):
        rows=np.zeros((128,16),dtype=np.float32)
        for i,token in enumerate(forced):
            rows[i,(token+1)%16]=4.0 if i<3 else 0.0
        rows[3:,0]=8.0
        reference.append(rows)
    teacher={'arm':'tp1-d0','identity':dict(identity),'suite':{'ids':[r['id'] for r in suite],
             'categories':[r['category'] for r in suite],'heldout':[r['heldout'] for r in suite],
             'tokens':[[0,1,2]]*18},'forced_inputs':[list(forced)]*18,'vocab_size':16}
    monkeypatch.setattr(coverage,'load_sustained',lambda p: (teacher,reference))
    return coverage,suite


class _RecordingRecorder:
    """StageRecorder stand-in that keeps the artifact and its stage results."""
    artifacts={}
    def __init__(self,path,artifact):
        self.path=path; self.artifact=artifact; self.stages=[]; self.exit_code=None
        _RecordingRecorder.artifacts[str(path)]=self
    def guard(self,stage,call):
        try:
            result=call()
        except Exception as exc:  # noqa: BLE001 - mirror StageRecorder
            self.stages.append({'stage':stage,'ok':False,'error':repr(exc)})
            self.artifact['status']='failed'; self.artifact['first_bad_stage']=stage
            self.exit_code=1; return False
        self.stages.append({'stage':stage,'ok':True})
        if isinstance(result,dict): self.artifact.update(result)
        return True
    def finish(self):
        self.artifact['status']=self.artifact.get('status','complete')
        self.artifact['natural_teardown']=self.exit_code is None


def test_capture_gates_only_the_declared_horizon_and_keeps_all_rows(monkeypatch,tmp_path):
    import json, numpy as np
    forced=[1,2,3]*42+[4,5]
    coverage,suite=_capture_seams(monkeypatch,tmp_path,forced[:128])
    args=_capture_args(tmp_path,horizon=3)
    try:
        code=coverage.capture_sustained_arm(args)
    except SystemExit as exc:   # the seam turns the gate's os._exit(1) into this
        rec=_RecordingRecorder.artifacts[args.json]
        raise AssertionError(f'capture failed: {[s for s in rec.stages if not s["ok"]]}') from exc
    rec=_RecordingRecorder.artifacts[args.json]
    assert code==0, [s for s in rec.stages if not s['ok']]
    assert rec.artifact['horizon']==3
    assert rec.artifact['horizon_declared_by']=='docs/EXECUTION-PROFILES.md 6.5'
    assert rec.artifact['comparison']['global']['rows']==18*3
    # Every captured row is still on disk: the horizon bounds the claim, not the record.
    for entry in rec.artifact['arrays']:
        assert np.load(entry['path']).shape==(128,16)


def test_deep_breach_fails_the_envelope_that_the_horizon_slice_passes(monkeypatch,tmp_path):
    """Same arrays, same gate: only the scored row range differs.

    The failing-capture path itself is not driven here because a failed capture
    calls os._exit(1) by design, which would kill the test process.
    """
    import numpy as np
    from scripts import tp2_teacher_coverage_broad as coverage
    forced=[1,2,3]*42+[4,5]
    coverage,suite=_capture_seams(monkeypatch,tmp_path,forced[:128])
    teacher=coverage.load_sustained(None)[1]
    # Rebuild the candidate the adapter would produce: exact for three rows,
    # then far from the teacher.
    candidate=[]
    for _ in range(18):
        rows=np.zeros((128,16),dtype=np.float32)
        for i,token in enumerate(forced[:128]):
            rows[i,(token+1)%16]=4.0
        candidate.append(rows)
    categories=[r['category'] for r in suite]; heldout=[r['heldout'] for r in suite]
    full=coverage.score_arm(teacher,candidate,categories,heldout,expected_positions=[128]*18,vocab_size=16)
    gate=coverage._envelope_gate(full['global'],top1_bar=.99)
    assert not gate['passed'] and any('max_kl' in f for f in gate['failures'])
    sliced=coverage.score_arm([r[:3] for r in teacher],[r[:3] for r in candidate],categories,heldout,
                              expected_positions=[3]*18,vocab_size=16)
    assert coverage._envelope_gate(sliced['global'],top1_bar=.99)['passed']
    assert sliced['global']['rows']==54 and full['global']['rows']==18*128
