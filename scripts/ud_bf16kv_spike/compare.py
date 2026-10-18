"""Compare aligned named boundaries; arithmetic localization only."""
from pathlib import Path
import struct,json
import numpy as np
root=Path('/tmp/ud-bf16kv-spike-20260908')

def teacher_records():
    data=(root/'teacher.f32.layers').read_bytes(); cursor=0; records={}
    while cursor<len(data):
        p,pos,nb,ne=struct.unpack_from('<IIII',data,cursor);cursor+=16
        name=data[cursor:cursor+nb].decode('ascii');cursor+=nb
        key=([10,4][p],pos-511,name)
        assert key not in records,key
        value=np.frombuffer(data,'<f4',ne,cursor);cursor+=4*ne
        assert np.isfinite(value).all()
        records[key]=value
    assert cursor==len(data)
    return records

def rel(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))

if __name__=='__main__':
    t=teacher_records();h=np.load(root/'hip.npz')
    teacher=np.fromfile(root/'teacher.f32',dtype='f4').reshape(2,9,-1)
    prior_t=np.fromfile('/tmp/ud-llama-context512-M-bf16kv.f32',dtype='f4').reshape(18,9,-1)[[10,4]]
    prior_h=np.load('/tmp/ud-q56-raw-context512-M.npz')['logits'].reshape(18,9,-1)[[10,4]]
    np.testing.assert_array_equal(teacher.view('u4'),prior_t.view('u4'))
    np.testing.assert_array_equal(h['logits'].reshape(prior_h.shape).view('u4'),prior_h.view('u4'))
    results=[]
    for p in [10,4]:
        for s in range(3):
            rows=[]
            for l in range(64):
                ht=lambda n:h[f'p{p}_s{s}_{n}-{l}']
                tt=lambda n:(t[p,s,f'{n}-{l}'].reshape(48,128,128).transpose(0,2,1).ravel() if n=='state_predelta' else t[p,s,f'{n}-{l}'])
                fields=['attn_norm','attn_residual','attn_post_norm','l_out']
                if l%4!=3: fields+=['linear_attn_qkv_mixed','z','conv_output_silu','final_output','state_predelta']
                row=dict(layer=l,relative_rms={n:rel(ht(n),tt(n)) for n in fields})
                row['relative_rms']['attn_output']=rel(ht('attn_output'),tt('linear_attn_out' if l%4!=3 else 'attn_output'))
                if l%4!=3:
                    conv=np.concatenate((ht('conv_state_before').reshape(10240,4)[:,1:],ht('linear_attn_qkv_mixed')[:,None]),axis=1)
                    row['relative_rms']['conv_input']=rel(conv,tt('conv_input'))
                rows.append(row)
            results.append(dict(prompt=p,step=s,layers=rows))
            print('prompt',p,'step',s,flush=True)
            for row in rows:
                l=row['layer'];r=row['relative_rms']
                if l in [0,1,2,3,15,23,26,27,28,31,35,47,51,59,60,61,62,63]: print(l,{k:round(v,5) for k,v in r.items()},flush=True)
    (root/'comparison.json').write_text(json.dumps(dict(diagnostic_only=True,teacher_and_hip_prior_logits_exact=True,results=results),indent=2)+'\n')
