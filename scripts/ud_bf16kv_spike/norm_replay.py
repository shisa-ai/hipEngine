"""Same-input post-attention residual/norm CPU reference on captured rows."""
import json
import numpy as np
from compare import teacher_records,rel,root
from hipengine.loading.gguf import GGUFReader
from scripts.ud_precision_row_diagnostic import bf16
r=GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf');h=np.load(root/'hip.npz');t=teacher_records()
rows=[]
for p in [10,4]:
 for s in range(3):
  for l in range(64):
   ht=lambda n:h[f'p{p}_s{s}_{n}-{l}'].astype(float)
   tt=lambda n:t[p,s,f'{n}-{l}'].astype(float)
   w=np.asarray(r.tensor_data(f'blk.{l}.post_attention_norm.weight'),dtype=float)
   x=ht('hidden_in')+ht('attn_output')
   # Inputs are BF16 values widened exactly. GPU sums in F32 before norm;
   # explicit sum rounding avoids inventing a full-F64 sum contract.
   x=x.astype('f4').astype(float)
   y=bf16(x/np.sqrt(np.mean(x*x)+1e-6)*w)
   target=ht('attn_post_norm')
   tx=tt('attn_residual');ty=tx/np.sqrt(np.mean(tx*tx)+1e-6)*w
   row=dict(prompt=p,step=s,layer=l,
            hip_norm_reference_relative_rms=rel(y,target),
            hip_norm_reference_max_abs=float(np.max(np.abs(y-target))),
            hip_residual_reference_exact=bool(np.array_equal(bf16(x),ht('attn_residual'))),
            teacher_norm_reference_relative_rms=rel(ty,tt('attn_post_norm')))
   # Residual storage is exact BF16 add. F64 norm reduction may choose the
   # adjacent BF16 value at ties; this check is a diagnostic, not strict parity.
   assert row['hip_residual_reference_exact'],row
   assert row['teacher_norm_reference_relative_rms']<1e-6,row
   assert row['hip_norm_reference_relative_rms']<1e-3,row
   rows.append(row)
(root/'norm_replay.json').write_text(json.dumps(rows,indent=2)+'\n')
print('passed',len(rows),'residual/norm rows; max HIP norm relative RMS',max(x['hip_norm_reference_relative_rms'] for x in rows),'exact norms',sum(x['hip_norm_reference_max_abs']==0 for x in rows))
