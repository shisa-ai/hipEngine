"""Native-product timing/accounting contracts, with no HIP/model allocation."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import tp2_matched_ar_baseline as bench


def test_native_tp2_adapter_keeps_native_full_logits_and_traces():
    from scripts.tp2_resident_control import NativeARAdapter
    calls=[]
    def forward(token,position,*,kind):
        calls.append((token,position,kind))
        row=np.zeros(16,dtype=np.float32); row[(token+1)%16]=1
        return row,SimpleNamespace(position=position,kind=kind,stages={})
    owner=SimpleNamespace(runtime=object(),vocab_size=16,reset=lambda:None,
        _ensure_graph_schedule=lambda:None,_forward_token=forward)
    a=NativeARAdapter(owner,resident=False); a.prepare()
    token=a.prefill([2,3]); a.begin_decode(1)
    result=a.transition(token,return_logits=False)
    assert result.logits.shape==(16,) and result.token_id==5
    assert calls==[(2,0,'prefill'),(3,1,'prefill'),(4,2,'decode')]
    assert len(a.traces)==3 and a.position==3


def test_native_tp1_adapter_preserves_token_feedback_and_readback_choice():
    from scripts.tp2_resident_control import NativeARAdapter
    class Session:
        runtime=object()
        def reset(self): self.position=0
        def prefill(self,tokens,**kw):
            assert kw['return_logits'] is False
            self.position=len(tokens); self.token=3
            return SimpleNamespace(token_id=3)
        def capture_decode_graph(self,**kw):
            assert kw['position']==self.position
            session=self
            class Graph:
                def replay(self,n): session.position+=n; session.token+=n
                def read_sample(self,**kw):
                    assert kw['return_logits'] is False
                    return SimpleNamespace(token_id=session.token,logits=None)
                def close(self): session.closed=True
            return Graph()
    session=Session(); owner=SimpleNamespace(session=session,vocab_size=16)
    a=NativeARAdapter(owner,resident=True)
    a.prefill([2,3]); a.begin_decode(1)
    with pytest.raises(ValueError,match='feedback'): a.transition(4)
    assert a.position==2
    assert a.transition(3).token_id==4
    a.end_decode(); assert session.closed


class FakeAdapter:
    def __init__(self, resident):
        self.resident = resident
        self.vocab_size = 1000
        self.position = 0
        self.readbacks = []
    def prefill(self, tokens):
        self.position = len(tokens)
        return 1
    def begin_decode(self, count): self.horizon = count
    def transition(self, token, *, return_logits=False, force=False):
        self.readbacks.append(return_logits)
        self.position += 1
        row = np.zeros((1, self.vocab_size), dtype=np.float32)
        row[0, token + 1] = 1
        return SimpleNamespace(token_id=token + 1, logits=row if return_logits else None)
    def end_decode(self): pass


@pytest.mark.parametrize('resident', [True, False])
def test_product_counts_native_readback_and_outer_clock(resident):
    adapter = FakeAdapter(resident)
    ticks = iter(range(100))
    row = bench.measure_product_prompt(adapter, [4, 5], {'id': 'p', 'category': 'code'},
                                       clock=lambda: next(ticks))
    assert row['timed_decode_transitions'] == 128
    assert row['total_samples'] == 129
    assert row['user_visible_requested_horizon'] == 128
    assert row['api_128_completion_equivalent'] is False
    assert row['decode_position_start'] == 2 and row['decode_position_end'] == 129
    assert len(row['sampled_output_ids']) == 128
    assert adapter.readbacks == ([False] * 127 + [True] if resident else [True] * 128)
    assert row['decode_ms'] == 1000  # one OUTER clock interval, never trace sums
    assert row['finite_final_logits'] is True


def test_product_rejects_bad_final_logits():
    adapter = FakeAdapter(True)
    original = adapter.transition
    def bad(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.logits is not None: result.logits[:] = np.nan
        return result
    adapter.transition = bad
    with pytest.raises(ValueError):
        bench.measure_product_prompt(adapter, [4], {'id': 'p', 'category': 'code'})


def test_weighted_denominator_is_actual_transitions_not_prompt_medians():
    rows = [{'timed_decode_transitions': 128, 'total_samples': 129, 'decode_ms': x,
             'total_generation_ms': x+100, 'prefill_ms': 100, 'capture_ms': 0, 'destroy_ms': 0}
            for x in (1000., 3000.)]
    aggregate = bench.product_summary(rows)
    assert aggregate['decode_tok_s'] == 64.0
    assert aggregate['decode_transitions'] == 256
    assert aggregate['total_samples'] == 258
    assert aggregate['generation_samples_s'] == pytest.approx(258000/4200)


def product_run(arm='tp1-d0', rep=0):
    row = bench.measure_product_prompt(FakeAdapter(arm != 'tp2'), [4, 5], {'id': 'p', 'category': 'code'})
    scope = {'capacity': 200}
    return {'arm': arm, 'rep': rep, 'run_id': f'{rep}-{arm}', 'status': 'complete',
            'session_capture_ms': 0, 'session_graph_destroy_ms': 0, 'session_teardown_ms': 0,
            'profile': {'manifest_sha256': 'e'*64}, 'route': {'max_sequence_length':200},
            'scope_manifest': {'manifest':scope, 'sha256':bench._sha256_json(scope)},
            'natural_teardown': True, 'first_bad_stage': None, 'rows': [row],
            'identity': {'model_sha256': 'a'*64, 'source_revision': 'b'*40,
                         'staged_diff_sha256': 'c'*64, 'source_sha256': {'runtime': 'd'*64},
                         'profile_sha256': 'e'*64, 'capacity': 200, 'kv': 'bf16',
                         'recurrent': 'fp32', 'sampling': 'greedy', 'host': 'test'},
            'devices': {'0': {'uuid': 'gpu-0'}} if arm=='tp1-d0' else (
                {'0': {'uuid': 'gpu-1'}} if arm=='tp1-d1' else {'0': {'uuid':'gpu-0'},'1':{'uuid':'gpu-1'}})}


@pytest.mark.parametrize('fault', ['missing', 'stale_rep', 'hash', 'count', 'negative', 'infinite', 'finite', 'teardown', 'route', 'profile', 'scope'])
def test_product_validation_blocks_invalid_denominators(fault):
    run = product_run()
    expected = copy.deepcopy(run['identity'])
    if fault == 'missing': del run['identity']['source_sha256']
    if fault == 'stale_rep': run['rep'] = 7
    if fault == 'hash': run['rows'][0]['prompt_token_sha256'] = 'wrong'
    if fault == 'count': run['rows'][0]['total_samples'] = 128
    if fault == 'negative': run['rows'][0]['capture_ms'] = -1
    if fault == 'infinite': run['rows'][0]['decode_ms'] = float('inf')
    if fault == 'finite': run['rows'][0]['finite_final_logits'] = False
    if fault == 'teardown': run['natural_teardown'] = False
    if fault == 'route': del run['route']
    if fault == 'profile': run['profile']['manifest_sha256'] = 'bad'
    if fault == 'scope': run['scope_manifest']['sha256'] = 'bad'
    with pytest.raises(ValueError):
        bench.validate_product_run(run, arm='tp1-d0', rep=0, expected_identity=expected,
            rows=[{'id':'p','category':'code'}], token_rows=[[4,5]])


def quality_file(tmp_path):
    import json
    p=tmp_path/'quality.json'
    p.write_text(json.dumps({'all_gates_passed':True,'positions':128,'identity':product_run()['identity'],
                            'suite':{'ids':['p'],'tokens':[[4,5]]}}))
    return p


def test_product_main_stops_after_first_fault_and_never_ratios(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bench, 'load_cell_prompts', lambda *a: ([{'id':'p','category':'code','prompt':'p'}], []))
    monkeypatch.setattr(bench, 'product_inputs', lambda *a: [[4,5]])
    monkeypatch.setattr(bench, 'product_identity', lambda *a: product_run()['identity'])
    monkeypatch.setattr(bench, 'product_idle_gate', lambda: {'idle': True})
    def child(arm, rep, *a, **kw):
        calls.append((rep,arm))
        if arm == 'tp2': raise RuntimeError('injected fault')
        return product_run(arm, rep)
    monkeypatch.setattr(bench, 'run_product_child', child)
    args = SimpleNamespace(model=tmp_path/'model', prompts=tmp_path/'p', heldout_prompts=tmp_path/'h',
        workdir=tmp_path, json=tmp_path/'out.json', reps=3, arm_timeout=1, quality_json=quality_file(tmp_path))
    assert bench.run_product_campaign(args) == 1
    assert calls == [(0,'tp1-d0'), (0,'tp2')]
    import json
    assert json.loads(args.json.read_text())['ratios'] is None


def test_product_main_rotates_real_reps_and_matches_physical_denominator(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bench, 'load_cell_prompts', lambda *a: ([{'id':'p','category':'code','prompt':'p'}], []))
    monkeypatch.setattr(bench, 'product_inputs', lambda *a: [[4,5]])
    monkeypatch.setattr(bench, 'product_identity', lambda *a: product_run()['identity'])
    monkeypatch.setattr(bench, 'product_idle_gate', lambda: {'idle': True})
    def child(arm, rep, *a, **kw):
        calls.append((rep,arm))
        run = product_run(arm,rep)
        run['rows'][0]['decode_ms'] = {'tp1-d0':4000.,'tp1-d1':3200.,'tp2':3600.}[arm]
        run['rows'][0]['total_generation_ms'] = 1 + sum(run['rows'][0][k] for k in ('decode_ms','prefill_ms','capture_ms','destroy_ms'))
        run.update(session_capture_ms=0, session_graph_destroy_ms=0, session_teardown_ms=0)
        return run
    monkeypatch.setattr(bench, 'run_product_child', child)
    args = SimpleNamespace(model=tmp_path/'model', prompts=tmp_path/'p', heldout_prompts=tmp_path/'h',
        workdir=tmp_path, json=tmp_path/'out.json', reps=3, arm_timeout=1, quality_json=quality_file(tmp_path))
    assert bench.run_product_campaign(args) == 0
    assert calls == [(r,a) for r,order in enumerate(bench.PRODUCT_ORDERS) for a in order]
    import json
    result = json.loads(args.json.read_text())
    assert result['ratios']['median_tp2_vs_faster_tp1'] == pytest.approx(3200/3600)
    assert all(r['faster_tp1_arm']=='tp1-d1' for r in result['ratios']['per_rep'])
