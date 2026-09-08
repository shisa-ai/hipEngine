"""Process-local taps on the production scalar path; assert prior logits."""
import json
from pathlib import Path
import numpy as np
import hipengine.runtime.qwen35_gguf_runner as mod
from hipengine.core.memory import copy_device_to_host
root=Path('/tmp/ud-bf16kv-spike-20260908')
base=np.load('/tmp/ud-q56-context512-M-baseline.npz')
prior=np.load('/tmp/ud-q56-raw-context512-M.npz')['logits'].reshape(18,9,-1)
active=None
records={}

def save(name, layer, value):
    key=f'p{active[0]}_s{active[1]}_{name}-{layer}'
    assert key not in records,key
    assert np.isfinite(value).all(),key
    records[key]=value

def f32(ptr,n):
    return mod._copy_f32_ptr_to_host(ptr,n,runtime=session.runtime)

def bf(ptr,n):
    return mod._copy_bf16_ptr_to_host_f32(ptr,n,runtime=session.runtime)

original_linear=mod.Qwen35GGUFFullStackRunner._run_linear_attention_attn_only
original_ffn=mod.Qwen35GGUFFullStackRunner._run_post_attention_ffn

def linear(self, layer, hidden, out, scratch, **kwargs):
    if active is not None:
        for name,buf in [('state_predelta',scratch.layer_recurrent_states[layer]),('conv_state_before',scratch.layer_conv_states[layer])]:
            save(name,layer,f32(buf.ptr,buf.nbytes//4))
    original_linear(self,layer,hidden,out,scratch,**kwargs)
    if active is not None:
        for name,buf,n in [('conv_output_silu',scratch.conv_out,self.linear_qkv_width),('final_output',scratch.recurrent_out,self.weights.config.ssm_inner_size)]:
            save(name,layer,f32(buf.ptr,n))
        for name,buf,n in [('linear_attn_qkv_mixed',scratch.linear_qkv,self.linear_qkv_width),('z',scratch.linear_z,self.weights.config.ssm_inner_size)]:
            save(name,layer,bf(buf.ptr,n))

def ffn(self,layer,hidden,attn,out,scratch,**kwargs):
    if active is not None:
        for name,ptr in [('hidden_in',hidden),('attn_norm',scratch.norm.ptr),('attn_output',attn)]:
            save(name,layer,bf(ptr,self.hidden_size))
    original_ffn(self,layer,hidden,attn,out,scratch,**kwargs)
    if active is not None:
        for name,ptr in [('attn_residual',scratch.residual.ptr),('attn_post_norm',scratch.post_norm.ptr),('l_out',out)]:
            save(name,layer,bf(ptr,self.hidden_size))
        save('ffn_intermediate',layer,bf(scratch.ffn_intermediate.ptr,self.ffn_size))

mod.Qwen35GGUFFullStackRunner._run_linear_attention_attn_only=linear
mod.Qwen35GGUFFullStackRunner._run_post_attention_ffn=ffn
all_logits=[]
with mod.Qwen35GGUFResidentSession('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf',backend='hip_gfx1151',max_sequence_length=528,compiler_version=Path('/tmp/ud-hipcc-version.txt').read_text()) as session:
    assert not session.runner.fp16_recurrent_state
    weights=session.runner.weights.weights
    manifest=[dict(slot=w.spec.slot_path,source=w.spec.source.name,quant=w.spec.quant_key,layout=w.spec.layout,shape=list(w.spec.source.shape),sidecars=list(w.spec.sidecar_layouts),allocations={name:a.buffer.nbytes for name,a in w.allocations.items()}) for w in weights]
    old=json.loads(Path('/tmp/ud-residual-precision-bf16.json').read_text())['actual_resident_manifest']
    assert manifest==old,'resident identity differs'
    for p in [10,4]:
        session.reset()
        seq=base['inputs'][p].tolist()+base['tokens'][9*p:9*p+8].tolist()
        for pos,token in enumerate(seq):
            active=(p,pos-511) if 511<=pos<=513 else None
            result=session.step(int(token),return_logits=pos>=511)
            assert session.position==pos+1
            if pos>=511:
                np.testing.assert_array_equal(result.logits.ravel().view('u4'),prior[p,pos-511].view('u4'))
                all_logits.append(result.logits.copy())
        print('captured prompt',p,'with exact prior logits',flush=True)
    np.savez(root/'hip.npz',**records,logits=np.asarray(all_logits))
    (root/'hip.json').write_text(json.dumps(dict(actual_resident_manifest=manifest,indices=[10,4],capture_steps=[0,1,2],instrumented_prior_logits_exact=True,kv_dtype=str(session.kv_storage_dtype)),indent=2)+'\n')
print('complete',len(records),'records',flush=True)
