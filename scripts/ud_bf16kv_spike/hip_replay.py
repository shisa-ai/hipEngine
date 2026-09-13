"""Same-input HIP conv/GDN F64 replay; alpha/beta reconstructed from weights."""
import json
import numpy as np
from compare import rel,root
from hipengine.loading.gguf import GGUFReader
from scripts.ud_precision_row_diagnostic import bf16
r=GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf');h=np.load(root/'hip.npz')
rows=[]
for l in range(64):
 if l%4==3:continue
 weights={n:np.asarray(r.dequantize_tensor(f'blk.{l}.{n}'),dtype=float) for n in ['ssm_alpha.weight','ssm_beta.weight','ssm_a','ssm_dt.bias','ssm_norm.weight','ssm_conv1d.weight']}
 for p in [10,4]:
  for s in range(3):
   ht=lambda n:h[f'p{p}_s{s}_{n}-{l}'].astype(float)
   a=bf16(weights['ssm_alpha.weight']@ht('attn_norm')).astype(float)
   b=bf16(weights['ssm_beta.weight']@ht('attn_norm')).astype(float)
   conv=ht('conv_output_silu')
   cin=np.concatenate((ht('conv_state_before').reshape(10240,4)[:,1:],ht('linear_attn_qkv_mixed')[:,None]),axis=1)
   c=(cin*weights['ssm_conv1d.weight'].reshape(10240,4)).sum(-1)
   cref=c/(1+np.exp(-c))
   q=conv[:2048].reshape(16,128);k=conv[2048:4096].reshape(16,128)
   q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-6)
   k=k/np.maximum(np.linalg.norm(k,axis=-1,keepdims=True),1e-6)
   q=q[np.arange(48)%16];k=k[np.arange(48)%16]
   v=conv[4096:].reshape(48,128)
   decay=np.exp(weights['ssm_a']*np.logaddexp(0,a+weights['ssm_dt.bias'])).reshape(48,1,1)
   state=ht('state_predelta').reshape(48,128,128)*decay
   delta=(v-np.einsum('hk,hkv->hv',k,state))/(1+np.exp(-b[:,None]))
   state=state+k[:,:,None]*delta[:,None,:]
   y=np.einsum('hk,hkv->hv',q,state)/np.sqrt(128.)
   z=ht('z').reshape(48,128)
   f=y/np.sqrt(np.mean(y*y,axis=-1,keepdims=True)+1e-6)*weights['ssm_norm.weight']*z/(1+np.exp(-z))
   row=dict(prompt=p,step=s,layer=l,conv_relative_rms=rel(cref,conv),final_relative_rms=rel(f,ht('final_output')))
   if s<2:row['next_state_relative_rms']=rel(state,h[f'p{p}_s{s+1}_state_predelta-{l}'])
   rows.append(row)
(root/'hip_replay.json').write_text(json.dumps(dict(limitation='Alpha/beta are reconstructed using F64 GEMV then BF16 rounding; they were not directly captured. GPU reduction and log/exp coefficient roundtrips are not emulated.',rows=rows),indent=2)+'\n')
for key in ['conv_relative_rms','final_relative_rms','next_state_relative_rms']:
 selected=[x for x in rows if key in x]
 print(key,'max',max(x[key] for x in selected),'worst',sorted(selected,key=lambda x:-x[key])[:3],flush=True)
