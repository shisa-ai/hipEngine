#!/usr/bin/env python3
"""Paired public PARO serial versus bulk prompt priming; unchanged decode."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from contextlib import closing

from hipengine import LLM, SamplingParams
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import malloc, memory_stats
from hipengine.generation.qwen35_paro import Qwen35ParoResidentModelRunner as Runner
from tokenizers import Tokenizer


def serial_capture(self, row, slab, *, sample_final):
    # Parent 07fba0543 row-at-a-time normalization and KV-only provider seam.
    session = self._session
    adapter = self._resolved_mtp2_adapter()
    adapter.begin_prompt(row.request_id)
    if row.mtp2_prompt_hidden_buffer is None:
        row.mtp2_prompt_hidden_buffer = malloc(session.hidden_nbytes, runtime=session.runtime)
    pointer = int(row.mtp2_prompt_hidden_buffer.ptr)
    def consume(request_id, prompt_index, hidden, seed_token):
        assert request_id == row.request_id
        session._write_final_normalized_hidden_bf16(hidden, destination_ptr=pointer, stream=0)
        adapter.consume_prompt_row(request_id, prompt_index=prompt_index,
                                   target_hidden_ptr=pointer, seed_token=seed_token)
    return tuple(session.prefill_native_packed(slab, sample=sample_final, final_hidden_row_sink=consume))


def routes(value):
    if isinstance(value, dict):
        if 'routes' in value and 'recent_completed' in value['routes']:
            return value
        for child in value.values():
            found = routes(child)
            if found is not None:
                return found
    return None


def digest(value):
    return hashlib.sha256(json.dumps(list(value), separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--cases', choices=('all','smoke'), default='all')
    parser.add_argument('--attribution', action='store_true')
    parser.add_argument('--resume', type=Path, required=True)
    args = parser.parse_args()
    assert args.warmups >= 1 and args.repetitions >= 1
    root = Path.cwd()
    tokenizer = Tokenizer.from_file(str(args.model/'tokenizer.json'))
    cases, sources = [], []
    for split,name in [('category','mtpbench-code-general-ja.jsonl'),('heldout','gdn-prefill-category-heldouts.jsonl')]:
        path = root/'benchmarks/prompts'/name
        sources.append({'path':str(path.relative_to(root)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in path.read_text().splitlines():
            item=json.loads(line)
            prompt='<|im_start|>user\n'+item['messages'][0]['content']+'<|im_end|>\n<|im_start|>assistant\n'
            cases.append({'id':item['id'],'category':item['category'],'split':split,'tokens':tokenizer.encode(prompt).ids})
    cases += [{'id':f'p{n}','category':'length_control','split':'control','tokens':[9707]*n} for n in (512,1024,4096)]
    if args.cases == 'smoke': cases=[cases[0],cases[-3],cases[-1]]
    result={'schema':1,'kind':'public_paro_bulk_prompt_comparison','passed':False,'attribution':args.attribution,
            'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[], 'protocol':{
                'backend':'hip_gfx1151','quant':'w4_paro','kv_storage':'bf16','max_active_requests':1,
                'max_sequence_length':4352,'prefix_cache':'off','temperature':0,'ignore_eos':True,'output_tokens':128,
                'warmups_per_case_per_arm':args.warmups,'samples_per_case_per_arm':args.repetitions,
                'order':'Serial then bulk on even pairs, reverse on odd pairs', 'sources':sources,
                'surface':'LLM.stream_speculative_mtp_detailed for both arms',
                'baseline':'Parent 07fba0543 row-at-a-time packed hidden normalization and KV-only priming; test-only capture-method substitution',
                'candidate':'Default bounded row-batched normalization and KV-only priming, 16 rows per chunk',
                'decode_timing':'First to last observed token wall time; 127 tokens; exclude terminal duplicate chunks',
                'first_token_timing':'Public call start to first token, including prompt target and provider work',
                'whole_call_timing':'Public call start through iterator exhaustion, excluding construction and snapshot inspection',
                'profile':'Shipping PARO defaults; no arithmetic or variant override',
                'attribution':'Separate synchronized packed capture method, including target prefill plus provider priming; not timing evidence'}}
    prior = json.loads(args.resume.read_text())
    assert prior['protocol'] == result['protocol'] and prior['attribution'] == args.attribution
    completed_ids = set()
    for record in prior['cases']:
        expected = {(pair, arm) for pair in range(args.warmups + args.repetitions) for arm in ('serial', 'bulk')}
        actual = {(sample['pair'], sample['arm']) for sample in record['samples']}
        if 'medians' not in record:
            break  # Restart the interrupted case, including its warmup.
        assert actual == expected and len(record['samples']) == len(expected)
        case = cases[len(result['cases'])]
        assert record['id'] == case['id'] and record['prompt_ids_sha256'] == digest(case['tokens'])
        result['cases'].append(record)
        completed_ids.add(record['id'])
    result['resume'] = {'source': str(args.resume), 'sha256': hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                        'original_driver_sha256': prior['driver_sha256'], 'preserved_complete_cases': sorted(completed_ids),
                        'cleanup_scope': 'memory_after_close covers only resumed process; interrupted process was terminated by task timeout'}
    cases = [case for case in cases if case['id'] not in completed_ids]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save(): args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    original_capture=Runner._prefill_row_packed_with_mtp2_capture
    original_reclaim=Runner.reclaim
    completed=[]
    capture_seconds=[]
    def reclaim(owner,completion):
        original_reclaim(owner,completion)
        completed.append(tuple(int(t) for t in completion.generated_tokens))
    Runner.reclaim=reclaim
    try:
        with closing(LLM(str(args.model),backend='hip_gfx1151',quant='w4_paro',max_active_requests=1,
                         max_sequence_length=4352,prefix_cache='off')) as llm:
            for case in cases:
                record={k:v for k,v in case.items() if k!='tokens'}
                record.update(prompt_tokens=len(case['tokens']),prompt_ids_sha256=digest(case['tokens']),samples=[],ids={})
                result['cases'].append(record)
                reference=None
                accepted_reference=None
                for pair in range(args.warmups+args.repetitions):
                    for arm in (('serial','bulk') if pair%2==0 else ('bulk','serial')):
                        implementation=serial_capture if arm=='serial' else original_capture
                        def capture(owner,*a,**kw):
                            if args.attribution: owner._session.runtime.device_synchronize()
                            started=time.perf_counter()
                            value=implementation(owner,*a,**kw)
                            if args.attribution:
                                owner._session.runtime.device_synchronize()
                                capture_seconds.append(time.perf_counter()-started)
                            return value
                        Runner._prefill_row_packed_with_mtp2_capture=capture
                        completed.clear(); capture_seconds.clear()
                        events=[]; final_ids=None; text=[]; previous=0
                        params=SamplingParams(max_tokens=128,temperature=0,ignore_eos=True,speculative_mtp=True,kv_storage='bf16')
                        started=time.perf_counter()
                        for chunk in llm.stream_speculative_mtp_detailed([case['tokens']],params):
                            now=time.perf_counter(); text.append(chunk.text)
                            if chunk.generated_token_ids is not None: final_ids=tuple(chunk.generated_token_ids)
                            if chunk.telemetry is not None:
                                count=int(chunk.telemetry.decode_state.generated_tokens)
                                assert count>=previous
                                if count>previous: events.append({'tokens':count,'seconds':now-started}); previous=count
                        wall=time.perf_counter()-started
                        assert len(completed)==1
                        ids=completed[0]
                        assert final_ids is None or final_ids==ids
                        assert len(ids)==128 and events[0]['tokens']==1 and events[-1]['tokens']==128
                        assert all(0<=t<tokenizer.get_vocab_size(with_added_tokens=True) for t in ids)
                        snapshot=routes(llm.live_loop_snapshot()); assert snapshot is not None
                        route=dict(snapshot['routes']['recent_completed'][-1]); route.pop('scheduler_chunks',None)
                        assert snapshot['model_runner']['active_requests']==0
                        assert route['specdec2_mtp2_used'] and route['specdec2_mtp2_cycles']>0
                        assert set(route['specdec2_mtp2_candidate_counts'])=={1}
                        assert set(route['specdec2_mtp2_execution_routes'])=={'eager'}
                        assert route['specdec2_mtp2_bulk_prompt_rows']==(len(case['tokens'])-1 if arm=='bulk' else 0)
                        assert bool(route['specdec2_mtp2_bulk_prompt_chunks'])==(arm=='bulk' and len(case['tokens'])>1)
                        fallback=snapshot['routes']['fallback_reasons']
                        assert not any(k.startswith('specdec2_mtp2_recovered') for k in fallback)
                        acceptance=route['specdec2_mtp2_accepted_counts']
                        if reference is None: reference=ids; accepted_reference=acceptance
                        assert ids==reference and acceptance==accepted_reference,(case['id'],arm,'repeat/cross-arm mismatch')
                        record['ids'].setdefault(arm,list(ids)); record.setdefault('text',{}).setdefault(arm,''.join(text))
                        seconds=events[-1]['seconds']-events[0]['seconds']
                        record['samples'].append({'pair':pair,'warmup':pair<args.warmups,'arm':arm,
                            'generated_tokens':len(ids),'generated_ids_sha256':digest(ids),'first_token_seconds':events[0]['seconds'],
                            'decode_seconds':seconds,'decode_tokens':127,'decode_tokens_per_second':127/seconds,
                            'whole_call_seconds':wall,'whole_call_tokens_per_second':128/wall,'token_events':events,
                            'route':route,'fallback_reasons':fallback,'capture_seconds':list(capture_seconds)})
                    save()
                record['medians']={arm:{key:statistics.median(s[key] for s in record['samples'] if s['arm']==arm and not s['warmup'])
                    for key in ('first_token_seconds','decode_tokens_per_second','whole_call_tokens_per_second')} for arm in ('serial','bulk')}
                print(json.dumps({'id':case['id'],'medians':record['medians']}),flush=True); save()
        result['memory_after_close']=memory_stats()
        assert result['memory_after_close']['active_allocations']==0 and result['memory_after_close']['current_allocated_bytes']==0
        result['passed']=True
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        Runner._prefill_row_packed_with_mtp2_capture=original_capture; Runner.reclaim=original_reclaim
        result['provenance']=collect_artifact_provenance(repo_root=root,configured_backend='hip_gfx1151',resolved_backend='hip_gfx1151',
            target_arch='gfx1151',model_path=args.model,quant='w4_paro',kv_dtype='bf16',command=[sys.executable,*sys.argv],
            build_profile='shipping public PARO staged adapter',timing_protocol=result['protocol']['decode_timing'],
            warmups=args.warmups,repetitions=args.repetitions,profiler={'enabled':False,'reason':'paired public prompt measurement'},
            hipcc_version=Path('/tmp/hipengine-paro-refresh-20261023/hipcc-version.txt').read_text())
        save()
    print('PASS',args.output,flush=True)

if __name__=='__main__': main()
