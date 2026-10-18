import json
import pytest
from scripts.vibevoice_asr_logits import teacher_tokens

@pytest.mark.parametrize('text,tokens',[
    ('assistant\n[{"Start":0,"End":4,"Speaker', [1,151645]),
    ('[{"Start":0,"End":4,"Speaker":0,"Content":"hello"}]',[1]),
    ('[]',[151645]),
])
def test_prefix_truncation_and_empty_content_cannot_be_teacher(tmp_path,text,tokens):
    p=tmp_path/'teacher.json'
    p.write_text(json.dumps({'timings':[{'text':text,'tokens':tokens}]}))
    with pytest.raises(ValueError,match='content'):
        teacher_tokens(p)

def test_full_content_teacher(tmp_path):
    p=tmp_path/'teacher.json'
    p.write_text(json.dumps({'timings':[{'text':'[{"Start":0,"End":1,"Speaker":0,"Content":"hello"}]','tokens':[1,151645]}]}))
    assert teacher_tokens(p).tolist()==[1,151645]

def test_capture_comparison_cannot_certify_one_recording(tmp_path):
    import numpy as np
    from scripts.vibevoice_asr_bench import array_hash
    from scripts.vibevoice_asr_compare_logits import compare
    teacher=tmp_path/'teacher.json'
    teacher.write_text(json.dumps({'timings':[{'text':'[{"Start":0,"End":1,"Speaker":0,"Content":"hello"}]','tokens':[1,151645]}]}))
    logits=np.zeros((2,152064),dtype=np.float32)
    tokens=np.array([1,151645],dtype=np.int64)
    meta=dict(logits_sha256=array_hash(logits),teacher_sha256=array_hash(tokens),
              request=dict(hashes={},model='fixture',audio_seconds=1))
    paths=[tmp_path/'ref.npy',tmp_path/'new.npy']
    for path in paths:
        np.save(path,logits)
        path.with_suffix('.json').write_text(json.dumps(meta))
    report=compare(*paths,teacher)
    assert report['numerical']['hard_gates_passed']
    assert not report['production_qualified']
    assert report['qualification_blockers']
    logits[0,0]=1
    np.save(paths[1],logits)
    with pytest.raises(ValueError,match='hash'):
        compare(*paths,teacher)
