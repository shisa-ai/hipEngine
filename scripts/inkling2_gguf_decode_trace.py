"""Profile warm ordinary public Inkling GGUF streaming at two prompt depths."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from contextlib import closing, contextmanager, nullcontext
from pathlib import Path

from hipengine import LLM, SamplingParams
from hipengine.benchmark.roctx_discovery import default_sdk


@contextmanager
def marker(library, name):
    library.roctxRangePushA(name.encode())
    try:
        yield
    finally:
        library.roctxRangePop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--backend', default='auto')
    parser.add_argument('--short', type=int, default=47)
    parser.add_argument('--prepared', type=int, default=2057)
    parser.add_argument('--output-tokens', type=int, default=9)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--warm-only', action='store_true', help='Warm JIT and model paths with a reduced prompt')
    args = parser.parse_args()
    if not 1 <= args.short < args.prepared or args.output_tokens < 2:
        parser.error('require 1 <= short < prepared and at least two output tokens')
    sdk = ctypes.CDLL(str(default_sdk(profiler_first=True))) if args.trace else None
    if sdk:
        sdk.roctxRangePushA.argtypes = [ctypes.c_char_p]
        sdk.roctxRangePop.argtypes = []
        sdk.roctxProfilerPause(0)
    llm = LLM(model=str(args.model), backend=args.backend, max_sequence_length=args.prepared + args.output_tokens,
              prefix_cache='off', speculative_mtp_serving='off')
    text = ('Explain why the day gets longer in spring, using plain language and a few practical examples. '
            'Distinguish what an observer sees from the motion of the Earth. ')
    sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
    try:
        seed = llm.tokenize(text)
        depths = (args.short,) if args.warm_only else (args.short, args.prepared)
        for n in depths:
            ids = (seed * ((n + len(seed) - 1) // len(seed)))[:n]
            tail = None
            with closing(llm.stream_detailed(ids, sampling)) as stream:
                # Session construction, cold upload, all prompt ingestion, and the
                # first token are excluded from the prepared decode range.
                for first in stream:
                    if first.text:
                        break
                else:
                    raise RuntimeError('no first text delta')
                if sdk:
                    sdk.roctxProfilerResume(0)
                try:
                    with marker(sdk, f'inkling_ordinary_decode_p{n}') if sdk else nullcontext():
                        for tail in stream:
                            pass
                finally:
                    if sdk:
                        sdk.roctxProfilerPause(0)
                if tail is None or tail.generated_token_ids is None or len(tail.generated_token_ids) != args.output_tokens:
                    raise RuntimeError('unexpected early termination or missing stream token IDs')
                diagnostics = tail.telemetry.diagnostics
                if diagnostics['execution_path'] != 'inkling2_gguf_packed_hip':
                    raise RuntimeError(f'unexpected execution path: {diagnostics}')
                print(json.dumps({'prompt_tokens': len(ids), 'prompt_ids_sha256': hashlib.sha256(
                    json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                    'output_ids': tail.generated_token_ids, 'diagnostics': diagnostics}), flush=True)
    finally:
        llm.close()


if __name__ == '__main__':
    main()
