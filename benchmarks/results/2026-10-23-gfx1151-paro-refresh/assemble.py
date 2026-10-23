import copy
import hashlib
import json
from pathlib import Path
import platform
import sys

root = Path('/tmp/hipengine-paro-gfx1151-20261023')
sys.path.insert(0,str(root))
from scripts.merge_readme_sweep_components import _metric_gate, _finite_final_logit_passed, _stable_identity
from hipengine.benchmark.provenance import validate_artifact_provenance

run = Path('/tmp/hipengine-paro-refresh-20261023')
out = run/'packet'
out.mkdir(exist_ok=True)
num=json.loads((run/'numerical.json').read_text())
pub=json.loads((run/'public.json').read_text())
assert all(c['passed'] for c in num['cases'])
assert pub['passed'] and pub['memory_after_close']['active_allocations']==0
assert pub['resolved_backend']=='hip_gfx1151' and pub['resolved_quant']=='w4_paro'
eager={c['id']:c['runs'][0]['ids'] for c in num['graph_cases']}
assert all(c['runs'][0]['ids']==eager[c['id']] for c in pub['cases'])
assert all(c['runs'][0]['telemetry']['decode_state']['execution_path']=='paro_resident_serial_decode' for c in pub['cases'])
identity=set();components=[]
for label in ('512-128','1K-128','4K-128', '512-128-repeat','1K-128-repeat','4K-128-repeat'):
    path=run/f'paro-eager-{label}.json'
    d=json.loads(path.read_text())
    provenance=validate_artifact_provenance(d['provenance'],require_model=True)
    assert not provenance['dirty']
    assert provenance['target_arch']=='gfx1151'
    assert '--no-graph-replay-decode' in provenance['command']
    assert provenance['warmups']==2 and provenance['repetitions']==5
    identity.add(_stable_identity(provenance))
    shape=d['workloads'][0];summary=d['summary_by_workload'][shape]
    assert _metric_gate(summary['prefill_tok_s'],expected_count=5)
    assert _metric_gate(summary['decode_tok_s'],expected_count=5)
    assert summary['final_token_ids_stable']
    runs=d['runs_by_workload'][shape]
    assert len(runs)==7 and sum(r['measured'] for r in runs)==5
    assert all(_finite_final_logit_passed(r['correctness_sanity']) for r in runs)
    assert all(r['timings']['graph_capture_seconds']==0 for r in runs)
    assert d['persistent_session_memory']['summary']['tracked_current_allocated_bytes_after_close']==0
    for rs in d['runs_by_workload'].values():
        for r in rs:
            r.pop('memory_snapshots',None)
            r.get('memory',{}).pop('kv_memory_audit',None)
    d['persistent_session_memory'].pop('snapshots',None)
    d['persistent_session_memory']['summary'].pop('kv_memory_audit',None)
    d['notes'].append('Verbose allocation snapshots omitted; all timing samples, output sanity, summaries and provenance retained.')
    (out/f'paro-eager-{label}.json').write_text(json.dumps(d,indent=2)+'\n')
    components.append({'artifact':f'paro-eager-{label}.json','shape':shape,'summary':summary,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
assert len(identity)==1
kls=[v for c in num['cases'] for v in c['comparison']['kl']]
summary={'schema':1,'status':'accepted','performance_claim':True,
 'performance_claim_scope':'Fresh eager resident PARO measurements at 512, 1024 and 4096 input tokens; not HTTP throughput, MTP or capacity.',
 'provenance':provenance,
 'physical_host':{'hostname':platform.node(),'machine_id':Path('/etc/machine-id').read_text().strip(),
                  'product_name':Path('/sys/class/dmi/id/product_name').read_text().strip(),
                  'sys_vendor':Path('/sys/class/dmi/id/sys_vendor').read_text().strip(),'rocm_platform_version':'10.0.0'},
 'execution_profile':pub['profile'],
 'profile_note':'No registered gfx1151 w4_paro execution profile resolves; these are the shipped default kernels, not a production-profile promotion.',
 'protocol':{'concurrency':1,'prompt_token_id':9707,'prompt_lengths':[512,1024,4096],'decode_steps':128,
             'discarded_runs':2,'measured_runs':5,'decode_warmup_steps':4,'kv_storage':'bf16',
             'decode_route':'eager','timing':'Separate synchronized resident prefill and decode walls; load, four decode warmup steps and setup excluded.',
             'variance_limit':'sample standard deviation <= 5% of median, using merge_readme_sweep_components._metric_gate',
             'scope_note':'Deliberately bounded three-shape refresh; no six-shape rollup was assembled.'},
 'components':components[:3],
 'confirmation_components':components[3:],
 'confirmation_note':'Repeat the same commands with separate output destinations; publish the first sweep, not the best of the two sweeps.',
 'correctness':{'passed':True,'numerical_rows':len(kls),'serial_native_mean_kl':sum(kls)/len(kls),'serial_native_max_kl':max(kls),
                'serial_native_top1_agreement':sum(sum(c['comparison']['top1_matches']) for c in num['cases'])/len(kls),
                'serial_native_gate':{'max_kl':0.05,'min_top1_per_prompt':0.9},
                'public_repeated_cases':len(pub['cases']),'public_eager_ids_equal':True,'public_generate_text_matches_detailed':pub['generate_text_matches_detailed'],
                'teardown_allocations':0,'all_component_variance_and_finite_checks_passed':True,
                'independent_quantization_quality':'Not rerun; prior same-checkpoint Transformers evidence is benchmarks/results/2026-08-16-zbook-qwen36-quant-quality.json. This refresh checks same-runtime serial/native agreement.'},
 'mtp':pub['mtp'],
 'rejected_graph_probe':{'status':'rejected_correctness','performance_claim':False,
                         'natural_cases_with_different_ids':[c['id'] for c in num['graph_cases'] if c['runs'][0]['ids']!=c['runs'][1]['ids']],
                         'note':'Old wrapper forced graph replay. Known gfx1151 PARO graph/eager failure reproduced; all graph timings discarded. Public path and final measurements use eager.'},
 'baseline':{'artifact':'benchmarks/results/2026-07-12-gfx1151-paro-prefill-recovery.json',
             'comparison':'Historical graph path and prior stack, with subsequent grouped-GDN correctness repair. No old-to-new speedup is asserted.'},
 'profiler':{'enabled':False,'reason':'No kernel or arithmetic change. Public telemetry confirms native compact prefill and paro_resident_serial_decode.'},
 'reproduction':{'environment_file':'environment.sh','validation_driver':'validate.py','assembler':'assemble.py',
                 'validation_commands':[num['provenance']['command'],pub['provenance']['command']]}}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for src,dest in [('numerical.json','validation.json'),('public.json','public.json'),('validate.py','validate.py'),('hipcc-version.txt','hipcc-version.txt')]:
    (out/dest).write_bytes((run/src).read_bytes())
(out/'environment.sh').write_bytes(Path('/tmp/hipengine-paro-refresh-20261023-env.sh').read_bytes())
(out/'assemble.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'status':summary['status'],'components':components,'correctness':summary['correctness']},indent=2))
