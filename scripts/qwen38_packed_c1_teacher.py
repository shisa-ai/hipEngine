"""Independent teacher schedules for packed-C1 numerical evaluation.

The caller must construct a same-quant strict-profile session. Capturing a
schedule alone does not qualify any candidate or public serving route.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _owned_logits(result):
    logits = np.asarray(result.logits, dtype=np.float32)
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 1 or not logits.size or not np.isfinite(logits).all():
        raise ValueError('teacher requires finite full-vocabulary logits')
    return logits.copy()


def capture_teacher(session, prompt_ids, *, steps):
    prompt = tuple(map(int, prompt_ids))
    if not prompt or steps < 1:
        raise ValueError('teacher requires a prompt and positive decode length')
    session.reset()
    prefill = _owned_logits(session.prefill(prompt, return_logits=True))
    if int(session.position) != len(prompt):
        raise ValueError('teacher prefill cursor differs from prompt length')
    token = int(prefill.argmax())
    inputs, rows = [], []
    for index in range(steps):
        inputs.append(token)
        logits = _owned_logits(session.step(token, return_logits=True))
        if logits.shape != prefill.shape or int(session.position) != len(prompt) + index + 1:
            raise ValueError('teacher vocabulary or decode cursor changed')
        rows.append(logits)
        token = int(logits.argmax())
    return dict(prompt_ids=prompt, inputs=tuple(inputs), prefill_logits=prefill,
                logits=np.stack(rows))


def teacher_windows(record, *, budget):
    if not 1 <= budget <= 7:
        raise ValueError('teacher window budget must be K1 through K7')
    inputs = tuple(record['inputs'])
    prompt = tuple(record['prompt_ids'])
    logits = np.asarray(record['logits'])
    if not prompt or not inputs or logits.ndim != 2 or logits.shape[0] != len(inputs):
        raise ValueError('teacher inputs and full-logit rows do not align')
    return [dict(position=len(prompt) + start, prefix=prompt + inputs[:start],
                 tokens=inputs[start:start + budget + 1],
                 logits=logits[start:start + budget + 1].copy())
            for start in range(0, len(inputs), budget + 1)]


def load_teacher_prompts():
    from scripts.gguf_mtp_c1c8_server_bench import load_prompt_suite, _render_messages
    canonical = load_prompt_suite(ROOT / 'benchmarks/prompts/mtpbench-code-general-ja.jsonl')
    rows = [dict(r, suite='canonical') for r in canonical]
    path = ROOT / 'benchmarks/prompts/gdn-prefill-category-heldouts.jsonl'
    heldouts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in heldouts:
        rendered = _render_messages(row['messages'])
        rows.append(dict(id=row['id'], category=row['category'], suite='category_heldout',
                         rendered_prompt=rendered,
                         prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest()))
    categories = {'code', 'general_en', 'general_ja', 'mixed_ja_en'}
    if (len(rows) != 18 or len({r['id'] for r in rows}) != 18
            or any({r['category'] for r in rows if r['suite'] == suite} != categories
                   for suite in ('canonical', 'category_heldout'))):
        raise ValueError('teacher requires complete canonical and category-heldout suites')
    return rows


def strict_runtime_provenance(llm, generator, session):
    """Record the validated manifest and actual BF16 KV storage at capture time."""
    from hipengine.core import DType
    from hipengine.execution_profiles import manifest_sha256, validate_variant_manifest
    if str(generator.execution_profile) != 'strict':
        raise ValueError('teacher runtime must select strict profile')
    manifest = validate_variant_manifest(llm.execution_profile_manifest)
    digest = manifest_sha256(manifest)
    if (manifest['execution_profile'] != 'strict'
            or digest != llm.execution_profile_manifest_sha256
            or digest != generator.execution_profile_manifest_sha256):
        raise ValueError('teacher runtime manifest identity differs from strict resolution')
    if session.kv_storage_dtype != DType.BF16:
        raise ValueError('teacher runtime requires BF16 KV storage')
    return dict(runtime_manifest=manifest, runtime_manifest_sha256=digest,
                kv_storage_dtype=str(session.kv_storage_dtype))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/models/gguf/Qwen3.8-27B-Q4_K_M.gguf'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=24)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    from hipengine import LLM
    from hipengine.loading.gguf import scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    prompts = load_teacher_prompts()
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    args.directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(kind='packed_c1_strict_teacher', complete=False,
                    full_profile_qualification=False, execution_profile='strict',
                    model=str(args.model), steps=args.steps, records=[])
    llm = None
    try:
        llm = LLM(str(args.model), backend='hip_gfx1100', execution_profile='strict',
                  max_sequence_length=1024, max_active_requests=1)
        llm.prepare(max_sequence_length=1024)
        generator = llm._get_text_generator()
        if str(generator.execution_profile) != 'strict':
            raise ValueError('teacher generator is not strict-profile')
        artifact = generator._kv_model_artifact_identity()
        if not artifact.content_verified or not artifact.sha256:
            raise ValueError('teacher model content is not verified')
        manifest['model_sha256'] = artifact.sha256
        with generator._resident_session_scope(shared_runner=generator._get_shared_runner(),
                pool_name='packed_c1_strict_teacher') as (session, _reused):
            manifest.update(strict_runtime_provenance(llm, generator, session))
            for row in prompts:
                ids = tokenizer.encode(row['rendered_prompt'])
                record = capture_teacher(session, ids, steps=args.steps)
                filename = row['id'] + '.npz'
                np.savez(args.directory / filename, logits=record['logits'],
                         prefill_logits=record['prefill_logits'])
                manifest['records'].append(dict(
                    prompt_id=row['id'], category=row['category'], suite=row['suite'],
                    prompt_sha256=row['prompt_sha256'], prompt_ids=list(record['prompt_ids']),
                    inputs=list(record['inputs']), logits_file=filename,
                    logits_sha256=hashlib.sha256(record['logits'].tobytes()).hexdigest(),
                    prefill_logits_sha256=hashlib.sha256(record['prefill_logits'].tobytes()).hexdigest()))
                print(json.dumps(dict(prompt_id=row['id'], rows=len(record['inputs']))), flush=True)
        llm.close()
        llm = None
        manifest['complete'] = len(manifest['records']) == 18
    except Exception as error:
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if llm is not None:
                llm.close()
        finally:
            (args.directory / 'teacher.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
