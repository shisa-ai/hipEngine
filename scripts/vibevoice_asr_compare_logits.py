"""Compare aligned full-vocabulary ASR captures with canonical profile metrics."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.vibevoice_asr_bench import array_hash
from scripts.vibevoice_asr_logits import teacher_tokens
from hipengine.benchmark.execution_profiles import RowDescriptor,compare_profile_logits


def compare(reference,candidate,teacher):
    captures=[]
    metas=[]
    for path in (reference,candidate):
        logits=np.load(path)
        meta=json.loads(path.with_suffix('.json').read_text())
        if array_hash(logits) != meta['logits_sha256']:
            raise ValueError('capture hash mismatch')
        captures.append(logits); metas.append(meta)
    tokens=teacher_tokens(teacher)
    if any(m['teacher_sha256'] != array_hash(tokens) for m in metas):
        raise ValueError('teacher mismatch')
    if any(m['request']['hashes'] != metas[0]['request']['hashes'] or
           m['request']['model'] != metas[0]['request']['model'] for m in metas):
        raise ValueError('request mismatch')
    rows=[RowDescriptor('asr-full-content',i,'recording',i,'english',
        str(metas[0]['request']['audio_seconds'])+'s','prefill' if i==0 else 'decode',int(t)) for i,t in enumerate(tokens)]
    result=compare_profile_logits(*captures,rows)
    return dict(kind='vibevoice_full_content_logit_diagnostic', captures=metas, numerical=result,
        production_qualified=False, qualification_blockers=[
            'single recording; category and heldout task suites absent',
            'fewer than 500 aligned rows' if len(tokens)<500 else 'row coverage requires suite review',
            'repeat, isolation and sustained concurrency captures absent',
            *([] if result['hard_gates_passed'] else ['numerical envelope failed'])])

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    out=compare(a.reference,a.candidate,a.teacher)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['numerical']['summary'],indent=2))
