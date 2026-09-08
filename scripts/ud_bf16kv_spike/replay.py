"""F64 same-input GDN reference and incoming-state substitution diagnostic."""
from pathlib import Path
import json
import numpy as np
from compare import teacher_records,rel
from hipengine.loading.gguf import GGUFReader
root=Path('/tmp/ud-bf16kv-spike-20260908')
r=GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
t=teacher_records()

def replay(p,s,l,state=None):
    tt=lambda n:t[p,s,f'{n}-{l}'].astype(float)
    q=tt('q_conv_predelta').reshape(16,128)[np.arange(48)%16]
    k=tt('k_conv_predelta').reshape(16,128)[np.arange(48)%16]
    v=tt('v_conv_predelta').reshape(48,128)
    decay=np.exp(tt('gate')).reshape(48,1,1)
    beta=tt('beta_sigmoid').reshape(48,1)
    state=tt('state_predelta').reshape(48,128,128).transpose(0,2,1) if state is None else state.astype(float).reshape(48,128,128)
    state=state*decay
    delta=(v-np.einsum('hk,hkv->hv',k,state))*beta
    state=state+k[:,:,None]*delta[:,None,:]
    y=np.einsum('hk,hkv->hv',q,state)/np.sqrt(128.)
    w=np.asarray(r.tensor_data(f'blk.{l}.ssm_norm.weight'),dtype=float)
    z=tt('z').reshape(48,128)
    final=y/np.sqrt(np.mean(y*y,axis=1,keepdims=True)+1e-6)*w*z/(1+np.exp(-z))
    return y,final

if __name__=='__main__':
    h=np.load(root/'hip.npz') if (root/'hip.npz').exists() else None
    results=[]
    for p in [10,4]:
      for s in range(3):
       for l in range(64):
        if l%4==3:continue
        y,f=replay(p,s,l)
        row=dict(prompt=p,step=s,layer=l,teacher_core_cpu_relative_rms=rel(y,t[p,s,f'attn_output-{l}']),teacher_final_cpu_relative_rms=rel(f,t[p,s,f'final_output-{l}']))
        assert row['teacher_core_cpu_relative_rms']<1e-4,row
        assert row['teacher_final_cpu_relative_rms']<1e-4,row
        if h is not None:
            hy,hf=replay(p,s,l,h[f'p{p}_s{s}_state_predelta-{l}'])
            row['hip_state_only_final_relative_rms']=rel(hf,f)
            row['hip_actual_final_relative_rms']=rel(h[f'p{p}_s{s}_final_output-{l}'],f)
        results.append(row)
    (root/'replay.json').write_text(json.dumps(results,indent=2)+'\n')
    print('teacher same-input GDN replay passes',len(results),'rows; max core/final',max(x['teacher_core_cpu_relative_rms'] for x in results),max(x['teacher_final_cpu_relative_rms'] for x in results))
    if h is not None:
        print('rate step1 largest state-only deviations',sorted([x for x in results if x['prompt']==10 and x['step']==1],key=lambda x:-x['hip_state_only_final_relative_rms'])[:8])
