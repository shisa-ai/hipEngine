#!/usr/bin/env python3
"""Transcribe a mono 24 kHz WAV using one standalone hipEngine Q4 GGUF."""
import argparse
import json
from pathlib import Path
import sys
import wave
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path, help='standalone .gguf file')
    p.add_argument('--audio', required=True, type=Path, help='mono 24 kHz PCM16 WAV')
    p.add_argument('--context', default='')
    p.add_argument('--seed', default=20260914, type=int)
    p.add_argument('--max-new-tokens', default=256, type=int)
    p.add_argument('--max-context', default=4096, type=int)
    p.add_argument('--backend', default='auto')
    args = p.parse_args()
    with wave.open(str(args.audio), 'rb') as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (24000,1,2):
            p.error('audio must be mono 24 kHz PCM16 WAV')
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float32) / 32768
    from hipengine.generation.vibevoice_asr_q4 import VibeVoiceASRQ4Generator
    session = VibeVoiceASRQ4Generator(args.model, max_sequence_length=args.max_context, backend=args.backend)
    try:
        result = session.transcribe(audio, context=args.context, seed=args.seed,
                                    max_new_tokens=args.max_new_tokens)
        print(json.dumps({'text': result.text, 'segments': result.segments,
                          'finish_reason': result.finish_reason,
                          'generated_token_ids': result.generated_token_ids}, ensure_ascii=False))
    finally:
        session.close()
    if 'torch' in sys.modules:
        raise RuntimeError('standalone inference imported torch')


if __name__ == '__main__':
    main()
