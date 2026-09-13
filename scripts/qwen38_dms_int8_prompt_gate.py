#!/usr/bin/env python3
"""Compare DMS codecs on every canonical prompt, including category heldouts.

This offline numerical gate uses raw user text, not a chat template. It records
teacher-forced quality; it does not assert task success or train-disjoint DMS
quality. INT8 evaluation cannot promote a serving configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import sys
from pathlib import Path

import numpy as np

from hipengine.core.memory import memory_stats
from hipengine.kvcache.dms import create_dms_bf16_backend, create_dms_int8_evaluation_backend
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner, Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_mtp_category_bench import DEFAULT_HELDOUT_PROMPT_IDS, DEFAULT_FULL_PROMPT_IDS
from scripts.qwen38_dms_integrated_quality import _compare, _summary, _sha256, _git


def load_prompts(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(DEFAULT_FULL_PROMPT_IDS):
        raise ValueError('gate requires the complete canonical prompt suite exactly once')
    for row in rows:
        if len(row['messages']) != 1 or row['messages'][0]['role'] != 'user':
            raise ValueError('gate expects one raw user message per prompt')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, default=Path('benchmarks/prompts/mtpbench-code-general-ja.jsonl'))
    parser.add_argument('--decode-steps', type=int, default=64)
    parser.add_argument('--backend', default='hip_gfx1100')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.decode_steps <= 0:
        raise ValueError('decode steps must be positive')
    prompts = load_prompts(args.prompts)
    provenance = _git()
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(args.model).info)
    runner = Qwen35GGUFFullStackRunner(args.model, backend=args.backend)
    rows = []
    try:
        for prompt in prompts:
            tokens = tokenizer.encode(prompt['messages'][0]['content'])
            common = dict(backend=args.backend, shared_runner=runner,
                          max_sequence_length=len(tokens)+args.decode_steps,
                          use_wmma_prefill=True, use_gemv_decode=True)
            teacher = []
            inputs = []
            with Qwen35GGUFResidentSession(args.model, **common) as session:
                seed = session.prefill(tokens, use_bulk=True, bulk_attention_mode='bulk', return_logits=True)
                teacher.append(seed.logits.copy())
                current = int(seed.token_id)
                for _ in range(args.decode_steps):
                    inputs.append(current)
                    result = session.step(current, return_logits=True)
                    teacher.append(result.logits.copy())
                    current = int(result.token_id)
            candidates = {}
            bf16_logits = []
            for codec, factory in (('bf16', create_dms_bf16_backend), ('int8', create_dms_int8_evaluation_backend)):
                comparisons = []
                relative = []
                with Qwen35GGUFResidentSession(
                    args.model, **common, dms_metadata_path=args.metadata,
                    dms_backend_factory=factory, dms_max_new_tokens=args.decode_steps,
                ) as session:
                    seed = session.prefill(tokens, use_bulk=True, bulk_attention_mode='bulk', return_logits=True)
                    actual = [seed.logits.copy()]
                    for token in inputs:
                        actual.append(session.step(token, return_logits=True).logits.copy())
                    for index, logits in enumerate(actual):
                        comparisons.append(_compare(teacher[index], logits))
                        if codec == 'int8':
                            relative.append(_compare(bf16_logits[index], logits))
                    if codec == 'bf16':
                        bf16_logits = actual
                    snapshot = session._dms_backend.observability_snapshot()
                    assert session._dms_dense_prefill_pool is None
                    assert snapshot['backend']['device_payloads']
                candidates[codec] = dict(
                    rows=comparisons, summary=_summary(comparisons, max_kl=.05, min_top1=.9),
                    dms=snapshot, logits_sha256=hashlib.sha256(b''.join(x.tobytes() for x in actual)).hexdigest())
                if relative:
                    candidates[codec]['versus_bf16_dms'] = _summary(relative, max_kl=.05, min_top1=.9)
            rows.append(dict(id=prompt['id'], category=prompt['category'],
                             split='heldout' if prompt['id'] in DEFAULT_HELDOUT_PROMPT_IDS else 'train',
                             prompt_tokens=len(tokens), prompt_sha256=hashlib.sha256(np.asarray(tokens,np.int64).tobytes()).hexdigest(),
                             teacher_text=tokenizer.decode(inputs), candidates=candidates))
            print(json.dumps({'prompt': prompt['id'], 'int8': candidates['int8']['summary']}), flush=True)
    finally:
        runner.close()
    passed = all(c['summary']['passed'] for row in rows for c in row['candidates'].values())
    passed = passed and all(row['candidates']['int8']['versus_bf16_dms']['passed'] for row in rows)
    result = dict(schema=1, status='passed' if passed else 'rejected_quality', performance_claim=False,
                  serving_qualification=False, protocol='raw-user-text, dense-BF16-teacher-forced; full canonical suite',
                  model_sha256=_sha256(args.model), metadata_sha256=_sha256(args.metadata),
                  prompts_sha256=_sha256(args.prompts), provenance=provenance, rows=rows,
                  host=socket.gethostname(), backend=args.backend,
                  command=shlex.join([sys.executable, *sys.argv]),
                  environment={key: os.environ.get(key) for key in ('HIP_VISIBLE_DEVICES','GPU_MAX_HW_QUEUES')},
                  memory_after_close=memory_stats(),
                  limitations=['Heldouts are category-suite splits, not proven disjoint from sidecar training.',
                               'Teacher-forced numerical quality does not establish free-running task success.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
