"""Matched-request VibeVoice inference benchmark in isolated HIP/torch processes.

An untimed oracle boundary freezes processed PCM, prompt IDs, masks and the
actual BF16 acoustic noise operands once. Both lanes consume that request;
reported inference time includes host-to-device input staging, audio encoders,
prefill and generation, and excludes loading, tokenization and JSON parsing.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_pcm(args):
    import wave
    if getattr(args, 'pcm_npy', None):
        return np.load(args.pcm_npy)
    if args.pcm_file is None:
        from scripts.vibevoice_asr_e2e import synth_pcm
        return synth_pcm(args.seconds, 0)
    with wave.open(str(args.pcm_file), 'rb') as fh:
        if (fh.getframerate(), fh.getnchannels(), fh.getsampwidth()) != (24000, 1, 2):
            raise ValueError('benchmark WAV must be mono 24 kHz PCM16')
        return np.frombuffer(fh.readframes(fh.getnframes()), dtype='<i2').astype(np.float32) / 32768


def array_hash(array):
    x = np.ascontiguousarray(array)
    return hashlib.sha256(str((x.shape, x.dtype.str)).encode() + x.tobytes()).hexdigest()


def prepare_request(args):
    import torch
    from transformers import VibeVoiceAsrProcessor
    from hipengine.loading.hf_cache import resolve_model_path
    path = resolve_model_path(args.model)
    raw = _load_pcm(args)
    if raw.ndim != 1 or not raw.size or not np.isfinite(raw).all():
        raise ValueError('audio must be a nonempty finite mono waveform')
    processor = VibeVoiceAsrProcessor.from_pretrained(str(path))
    config = json.loads((path / 'config.json').read_text())
    t0 = time.perf_counter()
    inputs = processor.apply_transcription_request(audio=raw, prompt=args.context or None)
    pcm = inputs['input_values'].to(torch.bfloat16).float().numpy().reshape(-1)
    frames = (raw.size + 3199) // 3200
    rng = np.random.default_rng(args.seed)
    width = config['acoustic_tokenizer_encoder_config']['hidden_size']
    noise = torch.tensor(rng.standard_normal((1, frames, width)), dtype=torch.bfloat16)
    base_scale = torch.tensor(rng.standard_normal(1), dtype=torch.bfloat16)
    scale = base_scale * config['acoustic_tokenizer_encoder_config']['vae_std']
    arrays = dict(pcm=pcm, input_ids=inputs['input_ids'].numpy(),
                  padding_mask=inputs['padding_mask'].numpy(), noise=noise.float().numpy(),
                  base_scale=base_scale.float().numpy(), scale=scale.float().numpy())
    meta = dict(model=str(path), revision=path.name, audio_seconds=raw.size / 24000,
                raw_pcm_sha256=array_hash(raw), hashes={k: array_hash(v) for k,v in arrays.items()},
                seed=args.seed, context=args.context, preprocessing_s=time.perf_counter()-t0)
    np.savez(args.request, **arrays)
    args.request.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta


@contextmanager
def recorded_noise(torch, noise, base_scale):
    """Override just the two audio RNG draws, validating use and shape."""
    from unittest.mock import patch
    calls = []
    def randn(*shape, **kwargs):
        if shape != (1,):
            raise ValueError(f'unexpected acoustic scale draw: {shape}')
        calls.append('scale')
        return base_scale.to(device=kwargs['device'], dtype=kwargs['dtype'])
    def randn_like(x, **kwargs):
        if tuple(x.shape) != tuple(noise.shape):
            raise ValueError(f'acoustic noise shape mismatch: {x.shape} != {noise.shape}')
        calls.append('noise')
        return noise.to(device=x.device, dtype=x.dtype)
    with patch.object(torch, 'randn', randn), patch.object(torch, 'randn_like', randn_like):
        yield
    if calls != ['scale', 'noise']:
        raise RuntimeError(f'acoustic RNG protocol changed: {calls}')


def _read_request(path):
    meta = json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    if {k: array_hash(v) for k,v in arrays.items()} != meta['hashes']:
        raise ValueError('request hash mismatch')
    return arrays, meta


def _compat_lane(pcm, duration, args, lane):
    """Keep existing single-lane callers on the matched request protocol."""
    if abs(len(pcm) / 24000 - duration) > 1 / 24000:
        raise ValueError('duration must describe the unpadded 24 kHz PCM')
    with tempfile.TemporaryDirectory(prefix='vibevoice-lane-') as directory:
        root = Path(directory)
        np.save(root/'pcm.npy', pcm)
        cmd = [sys.executable, str(Path(__file__).resolve()), '--model',args.model,
               '--pcm-npy',str(root/'pcm.npy'), '--request',str(root/'request.npz'),
               '--output',str(root/'out.json'), '--max-new-tokens',str(args.max_new_tokens),
               '--repeats',str(args.repeats)]
        subprocess.run([*cmd,'--lane','prepare'],check=True)
        subprocess.run([*cmd,'--lane',lane],check=True)
        result = json.loads((root/'out.json').read_text())
    result['text'] = result['timings'][-1]['text']
    return result


def bench_hipengine(pcm, duration, args):
    return _compat_lane(pcm, duration, args, 'hip')


def bench_torch_gpu(pcm, duration, args):
    return _compat_lane(pcm, duration, args, 'torch')


def run_lane(args):
    from scripts.vibevoice_asr_e2e import AUDIO_TOKEN_ID, IM_END_ID, parse_transcript
    arrays, meta = _read_request(args.request)
    input_ids = arrays['input_ids'][0].tolist()
    timings = []
    if args.lane == 'hip':
        from tokenizers import Tokenizer
        from hipengine.loading.vibevoice_asr import load_vibevoice_encoder, load_vibevoice_connector, load_vibevoice_qwen2
        from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
        from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime, greedy_generate
        specs = {k: load_vibevoice_encoder(meta['model'], k) for k in ('acoustic','semantic')}
        frontend = VibevoiceFrontendRuntime(*specs['acoustic'], *specs['semantic'],
            load_vibevoice_connector(meta['model'],'acoustic'), load_vibevoice_connector(meta['model'],'semantic'),
            frontend_variant='strict' if args.hip_variant == 'strict' else 'wmma')
        weights = load_vibevoice_qwen2(meta['model'])
        runner = VibevoiceQwen2Runtime(weights, max_context=len(input_ids)+args.max_new_tokens,
            prefill_variant='strict' if args.hip_variant == 'strict' else 'hipblaslt')
        del weights, specs
        tokenizer = Tokenizer.from_file(str(Path(meta['model'])/'tokenizer.json'))
        def infer():
            embeds = frontend.forward(arrays['pcm'], noise=arrays['noise'][0], noise_scale=arrays['scale'][0])
            rows = [runner.embed_row(t) for t in input_ids]
            positions = [i for i,t in enumerate(input_ids) if t == AUDIO_TOKEN_ID]
            if len(positions) != len(embeds):
                raise ValueError('audio placeholder count mismatch')
            for i, row in zip(positions, embeds):
                rows[i] = row
            return greedy_generate(runner, rows, max_new_tokens=args.max_new_tokens, eos_token_id=IM_END_ID)
        def close():
            frontend.close()
            runner.close()
        def decode(ids):
            return tokenizer.decode(ids, skip_special_tokens=True).strip()
        versions = {'numpy': np.__version__}
        manifest = None
    else:
        import torch
        import transformers
        from transformers import VibeVoiceAsrForConditionalGeneration, VibeVoiceAsrProcessor
        torch.set_grad_enabled(False)
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(meta['model'],
            torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
        processor = VibeVoiceAsrProcessor.from_pretrained(meta['model'])
        noise = torch.from_numpy(arrays['noise'])
        base_scale = torch.from_numpy(arrays['base_scale'])
        def infer():
            ids = torch.tensor([input_ids], device='cuda')
            pcm = torch.from_numpy(arrays['pcm']).reshape(1,1,-1).to(device='cuda',dtype=torch.bfloat16)
            mask = torch.from_numpy(arrays['padding_mask']).to('cuda')
            with recorded_noise(torch, noise, base_scale):
                out = model.generate(inputs=ids, input_values=pcm, padding_mask=mask,
                    max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=IM_END_ID)
            torch.cuda.synchronize()
            return out[0, len(input_ids):].tolist()
        def decode(ids):
            return processor.decode(ids, skip_special_tokens=True).strip()
        def close():
            pass
        versions = {'torch':torch.__version__, 'transformers':transformers.__version__}
        manifest = {'dtype':'bf16','attention':'eager'}
    try:
        for repeat in range(args.warmup + args.repeats):
            t0 = time.perf_counter()
            ids = infer()
            elapsed = time.perf_counter() - t0
            text = decode(ids)
            if repeat >= args.warmup:
                timings.append(dict(inference_s=elapsed, tokens=ids, text=text,
                    parsed=parse_transcript(text), natural_eos=bool(ids and ids[-1] == IM_END_ID)))
    finally:
        close()
    result = dict(lane=args.lane, versions=versions, request=meta, timings=timings,
                  torch_imported='torch' in sys.modules,variant_manifest=(
                      {'frontend':frontend.variant_manifest,'decoder':runner.variant_manifest}
                      if args.lane == 'hip' else manifest),
                  status='diagnostic-unqualified')
    if args.lane == 'hip' and result['torch_imported']:
        raise RuntimeError('torch imported in HIP lane')
    args.output.write_text(json.dumps(result,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pcm-file', type=Path)
    p.add_argument('--pcm-npy', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--seconds',type=float,default=11.)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--seed',type=int,default=20260914)
    p.add_argument('--context',default='')
    p.add_argument('--hip-variant',choices=['strict','candidate'],default='strict')
    p.add_argument('--model',default='microsoft/VibeVoice-ASR-HF')
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--output',type=Path,default=Path('/tmp/vibevoice-matched-benchmark.json'))
    p.add_argument('--request',type=Path)
    p.add_argument('--lane',choices=['prepare','hip','torch'])
    args = p.parse_args()
    if args.repeats <= 0 or args.warmup < 1 or args.max_new_tokens <= 0:
        p.error('positive repeats/max-new-tokens and at least one discarded warmup required')
    if args.lane == 'prepare':
        prepare_request(args)
        return
    if args.lane:
        run_lane(args)
        return
    with tempfile.TemporaryDirectory(prefix='vibevoice-bench-') as directory:
        request = args.request or Path(directory)/'request.npz'
        cmd = [sys.executable,str(Path(__file__).resolve()),*sys.argv[1:], '--request',str(request)]
        subprocess.run([*cmd,'--lane','prepare'],check=True)
        lanes = {}
        for lane in ('hip','torch'):
            out = Path(directory)/(lane+'.json')
            subprocess.run([*cmd,'--lane',lane,'--output',str(out)],check=True)
            lanes[lane] = json.loads(out.read_text())
        gpu = subprocess.run(['rocminfo'],capture_output=True,text=True,check=True).stdout
        report = dict(host=socket.gethostname(),hardware=[s.strip() for s in gpu.splitlines()
            if 'Marketing Name:' in s or 'Name:' in s and 'gfx' in s],
            command=[sys.executable,*sys.argv], warmup=args.warmup,
            timing_scope='inference including input H2D; shared preprocessing/tokenization, loading and parsing excluded',
            lanes=lanes)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        for lane,result in lanes.items():
            print(lane,'inference_s:',[round(t['inference_s'],4) for t in result['timings']])
        print(args.output)


if __name__ == '__main__':
    main()
