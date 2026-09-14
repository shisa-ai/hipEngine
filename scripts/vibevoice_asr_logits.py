"""Capture full-vocabulary, teacher-forced ASR logits through content and EOS.

Use a frozen request from vibevoice_asr_bench.py and a torch lane JSON containing
an EOS-terminated transcription. Large logits stay outside git. This diagnostic
is not a substitute for the multi-category production qualification suite.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.vibevoice_asr_bench import _read_request, recorded_noise, array_hash
from hipengine.generation.vibevoice_protocol import AUDIO_TOKEN_ID, IM_END_ID, parse_transcript


def teacher_tokens(path):
    record = json.loads(path.read_text())['timings'][-1]
    tokens = record['tokens']
    segments = parse_transcript(record['text'])
    if not tokens or tokens[-1] != IM_END_ID or not segments or not any(s['Content'].strip() for s in segments):
        raise ValueError('teacher must contain transcript content and terminate with natural EOS')
    return np.asarray(tokens,dtype=np.int64)


def capture(args):
    arrays, meta = _read_request(args.request)
    teacher = teacher_tokens(args.teacher)
    teacher_request = json.loads(args.teacher.read_text())['request']
    if any(teacher_request[k] != meta[k] for k in ('model','hashes')):
        raise ValueError('teacher request does not match frozen capture request')
    ids = arrays['input_ids'][0].tolist()
    if args.lane.startswith('torch'):
        import torch
        from transformers import VibeVoiceAsrForConditionalGeneration
        device = 'cpu' if args.lane == 'torch-cpu' else 'cuda'
        dtype = torch.float32 if device == 'cpu' else torch.bfloat16
        torch.set_grad_enabled(False)
        if device == 'cpu':
            torch.set_num_threads(16)
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(meta['model'],
            torch_dtype=dtype,device_map=device,attn_implementation='eager').eval()
        joined = torch.tensor([ids + teacher[:-1].tolist()],device=device)
        base_scale = torch.from_numpy(arrays['base_scale'])
        if device == 'cpu':
            # Preserve the effective BF16 sampling scale when the CPU oracle
            # performs the remaining model arithmetic in float32.
            config = json.loads((Path(meta['model'])/'config.json').read_text())
            base_scale = torch.from_numpy(arrays['scale']) / config['acoustic_tokenizer_encoder_config']['vae_std']
        with recorded_noise(torch,torch.from_numpy(arrays['noise']),base_scale):
            result = model(input_ids=joined,
                input_values=torch.from_numpy(arrays['pcm']).reshape(1,1,-1).to(device=device,dtype=dtype),
                padding_mask=torch.from_numpy(arrays['padding_mask']).to(device),use_cache=False)
        logits = result.logits[0,len(ids)-1:].float().cpu().numpy()
        manifest = {'lane':args.lane,'dtype':str(dtype)}
    else:
        from hipengine.loading.vibevoice_asr import load_vibevoice_encoder,load_vibevoice_connector,load_vibevoice_qwen2
        from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
        from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime,_upload
        from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
        from hipengine.core.memory import free
        from hipengine.core.runtime import MemcpyKind
        specs={k:load_vibevoice_encoder(meta['model'],k) for k in ('acoustic','semantic')}
        strict=args.lane == 'hip-strict'
        frontend=VibevoiceFrontendRuntime(*specs['acoustic'],*specs['semantic'],
            load_vibevoice_connector(meta['model'],'acoustic'),load_vibevoice_connector(meta['model'],'semantic'),
            frontend_variant='strict' if strict else 'wmma')
        try:
            embeds=frontend.forward(arrays['pcm'],noise=arrays['noise'][0],noise_scale=arrays['scale'][0])
        finally:
            frontend.close()
        runner=VibevoiceQwen2Runtime(load_vibevoice_qwen2(meta['model']),max_context=len(ids)+len(teacher),
            prefill_variant='strict' if strict else 'hipblaslt')
        try:
            rows=[runner.embed_row(t) for t in ids]
            positions=[i for i,t in enumerate(ids) if t == AUDIO_TOKEN_ID]
            if len(positions) != len(embeds):
                raise ValueError('audio frame count mismatch')
            for i,row in zip(positions,embeds): rows[i]=row
            buf=_upload(f32_to_bf16_bits(np.asarray(rows)))
            try:
                runner.prefill_rows(buf,len(ids),0)
                runner.runtime.memcpy(runner._hidden.ptr,buf.ptr+(len(ids)-1)*runner.spec.hidden_size*2,
                    runner.spec.hidden_size*2,MemcpyKind.DEVICE_TO_DEVICE)
            finally: free(buf)
            logits=np.empty((len(teacher),runner.spec.vocab_size),dtype=np.float32)
            for i,t in enumerate(teacher):
                logits[i]=runner.logits_argmax()[0]
                if i+1 < len(teacher):
                    runner.push_token(runner.embed_row(int(t)),len(ids)+i)
                    runner.forward_layers(len(ids)+i)
            manifest=runner.variant_manifest
        finally: runner.close()
        assert 'torch' not in sys.modules
    if logits.shape != (len(teacher),152064) or not np.isfinite(logits).all():
        raise ValueError("invalid full-vocabulary logit capture")
    np.save(args.output,logits)
    args.output.with_suffix('.json').write_text(json.dumps(dict(request=meta,teacher_sha256=array_hash(teacher),
        logits_sha256=array_hash(logits),shape=list(logits.shape),variant_manifest=manifest),indent=2)+'\n')
    print(args.lane,logits.shape,flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--lane',choices=['torch-cpu','torch-gpu','hip-strict','hip-candidate'],required=True)
    p.add_argument('--output',type=Path,required=True)
    capture(p.parse_args())
