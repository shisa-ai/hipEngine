#!/usr/bin/env python3
"""Compare production with strict on strict-generated Gemma continuation history.

The same prompt and strict-generated input tokens reach both arms. Numerical
thresholds come from gemma4_teacher_forced_gate; this script does not redefine
them. It also runs one request through LLM.generate and records expert routes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.llm import SamplingParams
from hipengine.runtime.gemma4_session import Gemma4ResidentSession
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import gemma4_moe_expert_route_counts
from scripts.gemma4_campaign_bench import _resolve_generator
from scripts.gemma4_control_smoke import (
    _prompt_rows, _prompt_token_ids, _trajectory_with_controls,
    _STRICT_ROUTE_ENV, _PRODUCTION_ROUTE_ENV,
)
from scripts.gemma4_teacher_forced_gate import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = REPO_ROOT / 'benchmarks/prompts/mtpbench-code-general-ja.jsonl'
    rows = _prompt_rows([prompts], limit=10)
    llm, runner, resolution = _resolve_generator(args.model, 4096)
    results = []
    strict_segments, production_segments = [], []
    try:
        tokenizer = llm._get_text_generator().tokenizer
        # Resident-session closure releases the borrowed runner. Exercise the
        # public generator first, while its cached runner is still live.
        public_tokens = _prompt_token_ids(tokenizer, rows[-1], min_tokens=2048)
        before = gemma4_moe_expert_route_counts()
        output = llm.generate(public_tokens, SamplingParams(max_tokens=3))
        after = gemma4_moe_expert_route_counts()
        public_routes = {key: value-before.get(key, 0) for key, value in after.items()
                         if value != before.get(key, 0)}
        if not output or not public_routes:
            raise RuntimeError('public generation did not produce output and expert-route evidence')
        with Gemma4ResidentSession(runner, backend='hip_gfx1100', target_arch='gfx1100') as session:
            for row in rows:
                tokens = _prompt_token_ids(tokenizer, row, min_tokens=2048)
                kwargs = dict(prompt_ids=tokens, decode_steps=60,
                              scenario_id='generated-history', request_id=row['id'],
                              route_top_k=8, graph_bucket='c1', rng_seed=0)
                strict, controls, specs = _trajectory_with_controls(
                    session, forced_input_ids=None, teacher_row_ids=None,
                    route_env=_STRICT_ROUTE_ENV, **kwargs)
                teacher = [int(spec['teacher_token_id']) for spec in specs]
                production, paired_controls, _ = _trajectory_with_controls(
                    session, forced_input_ids=teacher[:-1], teacher_row_ids=teacher,
                    route_env=_PRODUCTION_ROUTE_ENV, **kwargs)
                if controls != paired_controls:
                    raise RuntimeError(f"{row['id']}: unpaired input/control schedule")
                verdict = evaluate(strict, production)
                results.append(dict(id=row['id'], category=row['category'],
                                    prompt_tokens=len(tokens), verdict=verdict,
                                    raw_equal=bool(np.array_equal(strict, production))))
                strict_segments.append(strict)
                production_segments.append(production)
                np.savez(args.output_dir / (row['id'] + '.npz'), strict=strict,
                         production=production, teacher=np.asarray(teacher))
                print(row['id'], verdict['passed'], verdict['failed'], flush=True)
        combined = evaluate(np.concatenate(strict_segments), np.concatenate(production_segments))
        result = dict(kind='gemma4_generated_history_screen', artifact=str(args.model.resolve()),
                      hostname=socket.gethostname(), command=sys.argv,
                      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
                      git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO_ROOT, text=True)),
                      prompt_sha256=hashlib.sha256(prompts.read_bytes()).hexdigest(),
                      source_sha256={str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in (Path(__file__), REPO_ROOT/'hipengine/kernels/hip_gfx1100/gemma4/gemma4_experts.py')},
                      environment={key: value for key, value in os.environ.items()
                                   if key.startswith(('HIPENGINE_', 'ROCR_', 'HIP_VISIBLE'))},
                      resolution=resolution, cases=results, combined=combined,
                      public_routes=public_routes, diagnostic_only=True)
        (args.output_dir/'verdict.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
        return 0 if combined['passed'] and all(row['verdict']['passed'] for row in results) else 1
    finally:
        llm.close()


if __name__ == '__main__':
    raise SystemExit(main())
