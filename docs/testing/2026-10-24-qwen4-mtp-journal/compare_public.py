"""Public Flash-Next automatic-MTP regression comparison; no speed claim."""
import ast
from contextlib import closing
import hashlib
import json
from pathlib import Path
import subprocess

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
import hipengine.generation.qwen4_exp_serving_mtp as serving

source = subprocess.check_output(['git','show','07fba0543:hipengine/generation/qwen4_exp_serving_mtp.py'],text=True)
tree = ast.parse(source)
owner = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name == 'Qwen4ExpServingMTP')
method = next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name == '_advance_mtp')
namespace = dict(vars(serving))
exec(compile(ast.Module(body=[method],type_ignores=[]),'<parent _advance_mtp>','exec'),namespace)
baseline = namespace['_advance_mtp']
candidate = serving.Qwen4ExpServingMTP._advance_mtp
output = {'performance_claim':False,'parent_commit':'07fba0543','source_sha256':hashlib.sha256(Path(serving.__file__).read_bytes()).hexdigest(),'calls':[],'passed':False}
path = Path('/tmp/hipengine-shared-mtp/qwen4-journal-public.json')
counts = []
original = candidate
try:
    with closing(LLM(str(Path.home()/'models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL'),
                     backend='hip_gfx1151',quant='gguf_ud_q4_k_xl',execution_profile='production',
                     max_sequence_length=4096,max_active_requests=1,prefix_cache='off',
                     speculative_provider='qwen4_exp_mtp',
                     draft_model=str(Path.home()/'models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/mtp-Qwen3.8-Flash-Next-Q8_0.gguf'),
                     speculative_candidate_budget=3)) as llm:
        for length in (37,513):
            for temperature in (0.0,0.7):
                reference = None
                for label,implementation in [('baseline',baseline),('candidate',candidate),('candidate_repeat',candidate)]:
                    counts.clear()
                    def observe(self,row,budget,*,plan=None):
                        result = implementation(self,row,budget,plan=plan)
                        counts.append({'budget':budget,'committed':result[0] is not None})
                        return result
                    serving.Qwen4ExpServingMTP._advance_mtp = observe
                    # This is the same public request intent set by automatic
                    # server MTP selection, not a private runner invocation.
                    params = SamplingParams(max_tokens=16,temperature=temperature,seed=17,
                                            ignore_eos=True,speculative_mtp=True,speculative_mtp_automatic=True)
                    result = llm.generate_detailed([[9707]*length],params)[0]
                    ids = list(result.generated_token_ids)
                    assert len(ids) == 16
                    assert counts and all(item['committed'] for item in counts), ('c1 journal route not used',label,counts)
                    if reference is None: reference = ids
                    else: assert ids == reference,(length,temperature,label,'output mismatch')
                    record = {'length':length,'temperature':temperature,'arm':label,'ids':ids,'cycles':list(counts)}
                    output['calls'].append(record)
                    path.write_text(json.dumps(output,indent=2)+'\n')
                    print(json.dumps({'length':length,'temperature':temperature,'arm':label,'cycles':len(counts),'equal':ids==reference}),flush=True)
    output['memory_after_close'] = memory_stats()
    assert output['memory_after_close']['active_allocations'] == 0
    assert output['memory_after_close']['current_allocated_bytes'] == 0
    output['passed'] = True
finally:
    serving.Qwen4ExpServingMTP._advance_mtp = original
    path.write_text(json.dumps(output,indent=2)+'\n')
print('PASS',path,flush=True)
