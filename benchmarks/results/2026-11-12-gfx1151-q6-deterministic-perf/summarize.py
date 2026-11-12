"""Summarize all A/B samples without discarding timing drift."""
import json
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).parent
output={}
keys=('cycles','accepted_draft_tokens','generated_draft_tokens','selected_depth_histogram','failure_reason_counts','ar_step_reason_counts')
for filename in ('public_ab.json','confirm_ab.json'):
    p=json.loads((ROOT/filename).read_text());assert p['complete']
    base={s['id']:s for s in p['results'][0]['samples'] if s['repeat']==0}
    differences=[]
    for owner,r in enumerate(p['results']):
        assert r['memory_after_close']['current_allocated_bytes']==p['initial_memory']['current_allocated_bytes']
        for s in r['samples']:
            b=base[s['id']]
            a=b['telemetry']['diagnostics']['specdec2_mtp2'];c=s['telemetry']['diagnostics']['specdec2_mtp2']
            assert c['recoverable_failures']==0 and c['cycles']>0
            assert b['ids']==s['ids']
            if any(a[k]!=c[k] for k in keys):
                differences.append(dict(owner=owner,arm=r['arm'],repeat=s['repeat'],id=s['id'],accepted=c['accepted_draft_tokens'],baseline_accepted=a['accepted_draft_tokens']))
    metrics={}
    for warm in (False,True):
        for category in [None]+sorted({s['category'] for s in p['results'][0]['samples']}):
            for metric in ('wall_s','prefill_ms','specdec2_mtp2_target_ms'):
                values={}
                for arm in ('parent','fixed'):
                    totals=[]
                    for r in p['results']:
                        if r['arm']!=arm:continue
                        repeats=sorted({s['repeat'] for s in r['samples'] if (s['repeat']>0)==warm})
                        for rep in repeats:
                            totals.append(sum(s['wall_s'] if metric=='wall_s' else s['telemetry']['timing'][metric] for s in r['samples'] if s['repeat']==rep and (category is None or s['category']==category)))
                    values[arm]=mean(totals)
                values['delta_percent']=100*(values['fixed']/values['parent']-1)
                metrics[f"{'warm' if warm else 'first'}:{category or 'all'}:{metric}"]=values
    output[filename]=dict(metrics=metrics,control_differences=differences,all_ids_equal=True,no_recoverable_failures=True,allocations_retired=True,requests=sum(len(r['samples']) for r in p['results']))
(ROOT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({f:{k:v for k,v in d['metrics'].items() if ':all:' in k} for f,d in output.items()},indent=2))
