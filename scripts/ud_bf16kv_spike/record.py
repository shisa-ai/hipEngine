"""Build the compact handoff artifact after completed diagnostic runs."""
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
root=Path('/tmp/ud-bf16kv-spike-20260908')
out=Path('benchmarks/results/2026-09-08-zbook-ud-bf16kv-spike.json')

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def js(name):return json.loads((root/name).read_text())

manifest=js('hip.json')
assert manifest['instrumented_prior_logits_exact']
old=json.loads(Path('/tmp/ud-residual-precision-bf16.json').read_text())
assert manifest['actual_resident_manifest']==old['actual_resident_manifest']
model='/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf'
model_hash=digest(model)
assert model_hash==json.loads(Path('/tmp/ud-layer-teacher-manifest.json').read_text())['model_sha256']
base=np.load('/tmp/ud-q56-context512-M-baseline.npz')
wire=(root/'input.bin').read_bytes();cursor=12
assert wire[:12]==b'Q36Q'+struct.pack('<II',1,2)
for p in [10,4]:
    assert struct.unpack_from('<II',wire,cursor)==(512,9);cursor+=8
    for n,expected in [(512,base['inputs'][p]),(9,base['tokens'][9*p:9*(p+1)])]:
        np.testing.assert_array_equal(np.frombuffer(wire,'<i4',n,cursor),expected);cursor+=4*n
assert cursor==len(wire)
interventions=js('interventions.json')
assert len(interventions)==10
assert 'complete; both own baselines' in (root/'interventions.log').read_text()
intervention_logits=np.load(root/'interventions.npz')
prior=np.load('/tmp/ud-q56-raw-context512-M.npz')['logits'].reshape(18,9,-1)
for p,s in [(10,1),(4,2)]:
    for mode in ['own','own_repeat']:
        np.testing.assert_array_equal(intervention_logits[f'p{p}_{mode}'].view('u4'),prior[p,s].view('u4'))
replay=js('replay.json');norm=js('norm_replay.json');hip=js('hip_replay.json')['rows']
assert len(replay)==len(hip)==288 and len(norm)==384
assert all(x['hip_residual_reference_exact'] for x in norm)
files=[p for p in root.iterdir() if p.is_file()]
files += [Path('/tmp/ud-q56-context512-M-baseline.npz'),Path('/tmp/ud-q56-raw-context512-M.npz'),Path('/tmp/ud-llama-context512-M-bf16kv.f32'),Path('/tmp/ud-residual-precision-bf16.json'),Path('/tmp/ud-hipcc-version.txt')]
sources=[p for p in Path('scripts/ud_bf16kv_spike').iterdir() if p.suffix in ('.py','.cpp','.md')]
for p in sources:
    if (root/p.name).exists():assert p.read_bytes()==(root/p.name).read_bytes()
results=dict(
    schema_version=1,status='first_spike_handoff',diagnostic_only=True,promotion_qualified=False,
    host='zbook',hardware='AMD RYZEN AI MAX+ PRO 395 / Radeon 8060S',backend='hip_gfx1151',
    runtime_commit='cfa482873532aed5afed53c177df4eaa94416de1',model=model,model_sha256=model_hash,
    runtime_provenance_note='Capture-start HEAD from worklog creation; later e354fb7b5 changes only docs and an unrelated audit script. Runtime/kernel sources are unchanged between these commits.',
    teacher_commit='17252c769a63c1cb650ce98ae309cf4de0da7778',teacher_kv='BF16',hip_kv=manifest['kv_dtype'],
    workload=dict(context_tokens=512,forced_output_rows=9,captured_steps=[0,1,2],prompt_indices=[10,4],prompt_ids=['heldout_code_rate_limiter','general_en_plan'],synthetic_repeated_chat_tokens=True),
    validation=dict(teacher_prior_bitwise_logit_rows=18,hip_prior_bitwise_logit_rows=18,resident_manifest_equal=True,resident_slots=len(manifest['actual_resident_manifest']),intervention_restoration_prior_bitwise_rows=2,input_wire_exact=True),
    cpu_replay=dict(
        teacher_gdn_rows=len(replay),teacher_core_max_relative_rms=max(x['teacher_core_cpu_relative_rms'] for x in replay),teacher_gated_max_relative_rms=max(x['teacher_final_cpu_relative_rms'] for x in replay),
        hip_gdn_rows=len(hip),hip_conv_max_relative_rms=max(x['conv_relative_rms'] for x in hip),hip_gated_max_relative_rms=max(x['final_relative_rms'] for x in hip),hip_next_state_rows=sum('next_state_relative_rms' in x for x in hip),hip_next_state_max_relative_rms=max(x.get('next_state_relative_rms',0) for x in hip),
        residual_exact_rows=len(norm),norm_exact_rows=sum(x['hip_norm_reference_max_abs']==0 for x in norm),norm_max_relative_rms=max(x['hip_norm_reference_relative_rms'] for x in norm)),
    interventions=interventions,
    matched_layer_boundaries=[dict(prompt=r['prompt'],step=r['step'],layer_out_relative_rms=[x['relative_rms']['l_out'] for x in r['layers']],attention_out_relative_rms=[x['relative_rms']['attn_output'] for x in r['layers']]) for r in js('comparison.json')['results']],
    largest_rate_step1_state_only_local_effects=sorted([r for r in replay if r['prompt']==10 and r['step']==1],key=lambda r:-r['hip_state_only_final_relative_rms'])[:8],
    source_sha256={str(p):digest(p) for p in sources},local_evidence_sha256={str(p):digest(p) for p in files},
    commands_reference='scripts/ud_bf16kv_spike/README.md',
    limitations=[
        'Teacher-state replacement is a causal diagnostic, not an implementable production correction.',
        'Interventions cover K_M only; K_S recurrence/state was not captured in this spike.',
        'No raw KV tensors or attention probabilities captured. Interventions retain HIP prefix KV.',
        'HIP alpha/beta outputs were reconstructed with F64 GEMV then BF16 rounding, not directly captured.',
        'F64 local replays do not emulate GPU reduction order or certify strict byte equality.',
        'No new production candidate, full-suite qualification, task-quality or performance claim.',
        'The special-token probabilities are descriptive only; no token/prompt-specific corrections permitted.'])
out.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
print('wrote',out,'with exact model hash and completed-run provenance')
