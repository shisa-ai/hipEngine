"""Live graph/eager proposal hidden-byte and top1 oracle through public generation."""
import json
from pathlib import Path
import numpy as np
from hipengine import LLM, SamplingParams
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr, memory_stats
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNExecutor as E
original=E._run_exact_graph_chain
checks=[]
def read(self, tensor):
    data=np.empty(self.hidden_size*2,dtype=np.uint8)
    copy_device_to_host(host_array_ptr(data),DeviceBuffer(tensor.ptr,data.nbytes),runtime=self.runtime)
    return data

def checked(self, request_id, token_id, position, target_hidden, *, candidate_budget):
    checkpoint=self.capture_request_checkpoint(request_id)
    try:
        rows=original(self,request_id,token_id,position,target_hidden,candidate_budget=candidate_budget)
        assert rows is not None, self._proposal_graph_last_error
        expected=[(r.token_id,r.logit,read(self,r.hidden)) for r in rows]
        self.restore_request_checkpoint(checkpoint)
        hidden=target_hidden
        token=token_id
        for depth,(expected_token,expected_logit,expected_hidden) in enumerate(expected):
            got=self.run_step(request_id,token,position+depth,hidden)
            np.testing.assert_array_equal(read(self,got.hidden),expected_hidden)
            assert (got.token_id,got.logit)==(expected_token,expected_logit), (position,depth)
            token=got.token_id
            hidden=got.hidden
        checks.append(dict(position=position,budget=candidate_budget,hidden_bytes_equal=True,top1_equal=True))
        self.restore_request_checkpoint(checkpoint)
        return original(self,request_id,token_id,position,target_hidden,candidate_budget=candidate_budget)
    finally:
        self.release_request_checkpoint(checkpoint)
E._run_exact_graph_chain=checked
llm=LLM('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf',backend='hip_gfx1151',kv_storage='bf16',max_sequence_length=4608,prefix_cache='off',speculative_candidate_budget=3)
try:
    for repeats in (98,100,125,138,406):
        text='Reference notes.\n'+'A software project has source files and tests.\n'*repeats+'Summarize the notes in a sentence.'
        row=llm.generate_detailed([text],SamplingParams(max_tokens=48,temperature=0,ignore_eos=True,speculative_mtp=True))[0]
        print(row.telemetry.to_json_dict()['decode_state']['prompt_tokens'],len(checks),flush=True)
finally:
    llm.close()
    E._run_exact_graph_chain=original
Path('/tmp/nextn-live-oracle.json').write_text(json.dumps(dict(checks=checks,memory=memory_stats()),indent=2))
assert checks
