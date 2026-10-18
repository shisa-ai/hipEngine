"""Diagnostic state replacement at one position, never production routing."""
import json
from pathlib import Path
import numpy as np
from compare import teacher_records
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.core.memory import copy_host_to_device
from scripts.ud_precision_row_diagnostic import distribution_row
root=Path('/tmp/ud-bf16kv-spike-20260908')
t=teacher_records(); h=np.load(root/'hip.npz')
base=np.load('/tmp/ud-q56-context512-M-baseline.npz')
prior=np.load('/tmp/ud-q56-raw-context512-M.npz')['logits'].reshape(18,9,-1)
teacher=np.fromfile('/tmp/ud-llama-context512-M-bf16kv.f32',dtype='f4').reshape(18,9,-1)
output=[]; saved={}
with Qwen35GGUFResidentSession('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf',backend='hip_gfx1151',max_sequence_length=528,compiler_version=Path('/tmp/ud-hipcc-version.txt').read_text()) as session:
    assert not session.runner.fp16_recurrent_state
    manifest=[dict(slot=w.spec.slot_path,source=w.spec.source.name,quant=w.spec.quant_key,layout=w.spec.layout,shape=list(w.spec.source.shape),sidecars=list(w.spec.sidecar_layouts),allocations={name:a.buffer.nbytes for name,a in w.allocations.items()}) for w in session.runner.weights.weights]
    assert manifest==json.loads((root/'hip.json').read_text())['actual_resident_manifest']
    for p,step in [(10,1),(4,2)]:
        session.reset()
        seq=base['inputs'][p].tolist()+base['tokens'][9*p:9*p+step].tolist()
        pos=511+step
        for token in seq[:-1]:session.step(int(token),return_logits=False)
        assert session.position==pos
        token=int(seq[-1])
        modes=['own','teacher_recurrent','teacher_conv','teacher_both','own_repeat']
        for mode in modes:
            session.runtime.device_synchronize()
            for l in range(64):
                if l%4==3:continue
                state=h[f'p{p}_s{step}_state_predelta-{l}']
                conv=h[f'p{p}_s{step}_conv_state_before-{l}'].copy().reshape(10240,4)
                if mode in ('teacher_recurrent','teacher_both'):
                    state=t[p,step,f'state_predelta-{l}'].reshape(48,128,128).transpose(0,2,1).ravel()
                if mode in ('teacher_conv','teacher_both'):
                    # slot0 is discarded by the next convolution; slots1:4 are consumed.
                    conv[:,1:]=t[p,step,f'conv_input-{l}'].reshape(10240,4)[:,:3]
                for buf,arr in [(session.scratch.layer_recurrent_states[l],state),(session.scratch.layer_conv_states[l],conv)]:
                    arr=np.ascontiguousarray(arr,dtype='f4')
                    assert arr.nbytes==buf.nbytes
                    copy_host_to_device(buf,arr.ctypes.data,runtime=session.runtime)
            # Re-execute only this absolute position. Its KV slots are overwritten;
            # prefix KV slots are read-only. The final own_repeat checks restoration.
            session._position=pos
            result=session.step(token,return_logits=True)
            assert session.position==pos+1
            logits=result.logits.ravel()
            if mode in ('own','own_repeat'):
                np.testing.assert_array_equal(logits.view('u4'),prior[p,step].view('u4'))
            ids=np.argsort(teacher[p,step])[-2:][::-1]
            row=dict(prompt=p,step=step,mode=mode,**distribution_row(teacher[p,step],logits,ids))
            output.append(row);saved[f'p{p}_{mode}']=logits.copy()
            print(json.dumps(row),flush=True)
        (root/'interventions.json').write_text(json.dumps(output,indent=2)+'\n')
np.savez(root/'interventions.npz',**saved)
print('complete; both own baselines and post-intervention restoration bitwise exact',flush=True)
