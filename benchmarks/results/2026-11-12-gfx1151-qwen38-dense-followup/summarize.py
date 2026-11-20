"""Summarize saved clean-server and public profiler diagnostics without rerunning."""
import hashlib
import json
from pathlib import Path
import sqlite3

D=Path('benchmarks/results/2026-11-12-gfx1151-qwen38-dense-followup')
D.mkdir(exist_ok=True)
load=lambda p: json.loads(Path(p).read_text())
base=load('/tmp/qwen38-dense-baseline.json')
candidate=load('/tmp/qwen38-dense-candidate.json')
prior=load('benchmarks/results/2026-11-12-gfx1151-qwen38-mixed-pair-batch/summary.json')
assert [c['prompt_id'] for c in base['cells']] == [c['prompt_id'] for c in candidate['cells']]
provenance=dict(host=prior['host'], compiler=prior['compiler'], model=candidate['model'],
                runtime_profile=candidate['runtime_profile'], source=candidate['source'],
                source_state_note='Harness reports clean tracked source. Attribution ran at the same commit with only an untracked worklog entry.')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def controls(payload):
    diagnostics = json.loads(json.dumps(payload['result']['telemetry']['diagnostics']))
    diagnostics['specdec2_mtp2'].pop('cycle_timing_ms')
    return diagnostics

def compact(payload):
    result={key:payload[key] for key in ('source','model','hardware','protocol','runtime_profile','status','failure_reasons','summary','acceptance','initial_memory','final_memory')}
    result['cells']=[]
    for cell in payload['cells']:
        c={key:cell[key] for key in ('prompt_id','category','correctness','mtp_engaged','mtp_budget_conformed')}
        for arm in ('ar','mtp'):
            c[arm]={key:cell[arm][key] for key in ('wall_seconds','tok_s','generated_tokens')}
            c[arm]['ids']=cell[arm]['rows'][0]['generated_ids']
        result['cells'].append(c)
    return result
server=dict(schema=1, kind='clean_source_server_diagnostic', provenance=provenance,
            baseline=compact(base),candidate=compact(candidate),
            acceptance_unchanged=base['acceptance']==candidate['acceptance'],
            raw_sha256={p:sha(p) for p in ('/tmp/qwen38-dense-baseline.json','/tmp/qwen38-dense-candidate.json')},
            all_per_arm_outputs_unchanged=all(a[arm]['rows'][0]['generated_ids']==b[arm]['rows'][0]['generated_ids']
                for a,b in zip(base['cells'],candidate['cells']) for arm in ('ar','mtp')),
            mtp_wall_reduction_percent=100*(1-candidate['summary']['1']['mtp']['wall_seconds']/base['summary']['1']['mtp']['wall_seconds']),
            ar_wall_change_percent=100*(candidate['summary']['1']['ar']['wall_seconds']/base['summary']['1']['ar']['wall_seconds']-1))
D.joinpath('server.json').write_text(json.dumps(server,indent=2)+'\n')

profiles={}
for name in ('short','long'):
    root=Path('/tmp/qwen38-attribution-'+name+'-profile')
    db=next(root.rglob('*.db'))
    c=sqlite3.connect(db)
    graph_rows=c.execute('select graph_exec_id,count(*),min(kernel_dispatch_count),max(kernel_dispatch_count) from rocpd_graph_launch group by graph_exec_id').fetchall()
    groups=[]
    for graph_id, in c.execute('select distinct graph_exec_id from rocpd_kernel_dispatch'):
        rows=c.execute('select k.display_name,count(*),sum(d.end-d.start)/1e6,min(d.grid_size_x),max(d.grid_size_x),min(d.grid_size_y),max(d.grid_size_y),max(d.private_segment_size),max(k.arch_vgpr_count) from rocpd_kernel_dispatch d join rocpd_info_kernel_symbol k on d.kernel_id=k.id and d.guid=k.guid where d.graph_exec_id=? group by k.display_name order by sum(d.end-d.start) desc',(graph_id,)).fetchall()
        groups.append(dict(graph_id=graph_id,sum_kernel_ms=sum(r[2] for r in rows),kernels=rows))
    attr=load('/tmp/qwen38-attribution-'+name+'.json')
    warm=load('/tmp/qwen38-attribution-'+name+'-warm.json')
    profiles[name]=dict(schema=1, kind='instrumented_public_request_attribution', provenance=provenance,
                        graph_launch_columns=['graph_id','calls','min_kernel_dispatches','max_kernel_dispatches'],
                        kernel_columns=['name','calls','sum_ms','min_grid_x','max_grid_x','min_grid_y','max_grid_y','max_private_bytes','max_arch_vgpr_count'],
                        graph_launches=graph_rows,groups=groups,attribution=attr,
                        profile_ids_equal_warm=attr['result']['ids']==warm['result']['ids'],
                        profile_control_equal_warm=controls(attr)==controls(warm),
                        control_comparison_excludes=['specdec2_mtp2.cycle_timing_ms'],
                        raw_sha256={str(p):sha(p) for p in (db,Path('/tmp/qwen38-attribution-'+name+'.json'),Path('/tmp/qwen38-attribution-'+name+'-warm.json'))},
                        warning='Python stack inspection and profiling perturb wall time; these are attribution diagnostics, not throughput or short-versus-long speed comparisons.')
    D.joinpath(name+'.json').write_text(json.dumps(profiles[name],indent=2)+'\n')
print(json.dumps(dict(server=dict(mtp_wall_reduction_percent=server['mtp_wall_reduction_percent'],all_per_arm_outputs_unchanged=server['all_per_arm_outputs_unchanged']),profiles={k:dict(graphs=v['graph_launches'],groups=[(g['graph_id'],g['sum_kernel_ms']) for g in v['groups']],ids_equal=v['profile_ids_equal_warm'],copies=len(v['attribution']['copies']),residual=v['attribution']['post_gc']) for k,v in profiles.items()}),indent=2))
