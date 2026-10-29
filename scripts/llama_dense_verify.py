"""Public dense-Llama verification; timings are diagnostics, not promotion evidence."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from hipengine import LLM, SamplingParams


def build_cases(generator):
    cases = []
    for name, prompt, required in [
        ('arithmetic', 'What is 17 times 23? Reply with only the integer.', '391'),
        ('code', 'Write a Python function square(x) that returns x multiplied by itself. Return only the function, without explanation.', 'return'),
        ('japanese', '日本の首都はどこですか。都市名だけで答えてください。', '東京'),
        ('explanation', 'In two sentences, explain one benefit and one limitation of unified memory for a local language model.', None),
    ]:
        cases.append(dict(name=name, prompt_ids=list(generator.tokenize_chat(prompt, enable_thinking=False)), required=required))
    marker = 'ARCHIVE_CONTENT_PLACEHOLDER'
    rendered = generator.tokenizer.format_chat_prompt(
        'Read this archive. Remember the access code.\n' + marker
        + '\nWhat is the archive access code? Reply only with the code.',
        enable_thinking=False,
    )
    before, after = rendered.split(marker)
    prefix = list(generator.tokenize(before))
    needle = list(generator.tokenize('\nThe archive access code is LANTERN-5831.\n'))
    suffix = list(generator.tokenize(after))
    filler = list(generator.tokenize('The archive contains routine weather reports, inventory lists, and meeting schedules. No access code appears in these ordinary records.\n'))
    for length in (1023, 1025, 4097, 8193):
        count = length - len(prefix) - len(needle) - len(suffix)
        pad = (filler * (count // len(filler) + 1))[:count]
        ids = prefix + pad[:count//2] + needle + pad[count//2:] + suffix
        assert len(ids) == length
        cases.append(dict(name=f'retrieval_{length}', prompt_ids=ids, required='LANTERN-5831'))
    return cases


def measured_request(llm, generator, case, max_tokens):
    runner = generator._ensure_runner()
    spans = {'prefill': [], 'decode': []}
    hashes = []
    originals = {}
    for label, attribute in [('prefill', 'prefill'), ('decode', 'decode_step')]:
        original = getattr(runner, attribute)
        originals[attribute] = original
        def call(*args, _original=original, _label=label, **kwargs):
            runner.runtime.device_synchronize()
            start = time.perf_counter()
            output = _original(*args, **kwargs)
            runner.runtime.device_synchronize()
            spans[_label].append(time.perf_counter() - start)
            values = np.asarray(output)
            if not np.isfinite(values).all():
                raise RuntimeError(f'nonfinite {_label} logits')
            hashes.append(hashlib.sha256(values.tobytes()).hexdigest())
            return output
        setattr(runner, attribute, call)
    try:
        start = time.perf_counter()
        output = llm.generate_detailed(case['prompt_ids'], SamplingParams(max_tokens=max_tokens, temperature=0.0))[0]
        wall = time.perf_counter() - start
    finally:
        for attribute, original in originals.items():
            setattr(runner, attribute, original)
    ids = list(output.generated_token_ids)
    finish = dataclasses.asdict(output.finish_details)
    assert len(spans['prefill']) == 1
    assert len(spans['decode']) == max(0, len(ids)-1)
    return dict(name=case['name'], prompt_tokens=len(case['prompt_ids']), generated_ids=ids,
                text=output.text, finish=finish, completed=finish['reason']=='stop',
                required_text_present=case['required'] is None or case['required'] in output.text,
                finite_logits=True, logit_hashes=hashes, prefill_seconds=spans['prefill'][0],
                decode_step_seconds=spans['decode'], wall_seconds=wall,
                tiled=runner._attention_is_tiled, strict=generator.strict_arithmetic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--resume', action='store_true', help='Reuse completed rows from the output artifact.')
    parser.add_argument('--cases', nargs='+', help='Run only named cases; defaults to the full protocol.')
    args = parser.parse_args()
    artifact = dict(schema=1, performance_claim=False, host=platform.node(),
                    model=str(args.model.resolve()), revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    compiler=subprocess.check_output(['hipcc','--version'],text=True),
                    protocol=dict(context=8704, max_tokens=args.max_tokens, repeats=2,
                                  thinking=False, kv='resident_bf16', concurrency=1,
                                  timing='synchronized runner calls inside public generate_detailed; load excluded; one short warmup per profile; first long-shape allocation included'),
                    arms=[])
    if args.resume:
        saved = json.loads(args.output.read_text())
        for key in ('model', 'revision', 'protocol'):
            if saved[key] != artifact[key]:
                raise ValueError(f'resume {key} differs from the saved run')
        artifact = saved
        artifact.pop('error', None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+'\n')
    try:
        for profile in (None, 'strict'):
            existing = next((a for a in artifact['arms'] if a['profile'] == profile), None)
            if existing is not None and len(existing['rows']) == 2 * len(existing['cases']):
                continue
            llm = LLM(str(args.model), max_sequence_length=8704, execution_profile=profile)
            try:
                driver = llm._get_text_generator()
                generator = getattr(driver, '_inner', driver)
                cases = build_cases(generator)
                if args.cases:
                    unknown = set(args.cases) - {c['name'] for c in cases}
                    if unknown:
                        raise ValueError(f'unknown cases: {sorted(unknown)}')
                    cases = [c for c in cases if c['name'] in args.cases]
                if existing is not None:
                    if existing['cases'] != cases:
                        raise ValueError('resume cases differ from the saved run')
                    arm = existing
                else:
                    arm = dict(profile=profile, backend=llm.resolved_backend, quant=llm.resolved_quant,
                               manifest=llm.execution_profile_manifest, cases=cases, rows=[])
                    artifact['arms'].append(arm)
                generator._ensure_runner()
                llm.generate_detailed(generator.tokenize_chat('Say OK.', enable_thinking=False), SamplingParams(max_tokens=8, temperature=0.0))
                for repeat in range(2):
                    for case in cases:
                        if any(r['name'] == case['name'] and r['repeat'] == repeat for r in arm['rows']):
                            continue
                        print(f'START profile={profile} repeat={repeat} {case["name"]}', flush=True)
                        row = measured_request(llm, generator, case, args.max_tokens)
                        row['repeat'] = repeat
                        arm['rows'].append(row)
                        save()
                        print(json.dumps({k:v for k,v in row.items() if k not in ('logit_hashes','decode_step_seconds','generated_ids')}, ensure_ascii=False),flush=True)
                arm['repeat_ids_equal'] = all(arm['rows'][i]['generated_ids']==arm['rows'][i+len(cases)]['generated_ids'] for i in range(len(cases)))
                arm['repeat_logits_equal'] = all(arm['rows'][i]['logit_hashes']==arm['rows'][i+len(cases)]['logit_hashes'] for i in range(len(cases)))
                save()
            finally:
                llm.close()
        a,b = artifact['arms']
        assert a['cases']==b['cases']
        artifact['cross_profile_ids_equal'] = {x['name']:x['generated_ids']==y['generated_ids'] for x,y in zip(a['rows'][:len(a['cases'])], b['rows'][:len(b['cases'])])}
        artifact['passed'] = all(arm['repeat_ids_equal'] and arm['repeat_logits_equal'] and all(row['completed'] and row['required_text_present'] for row in arm['rows']) for arm in artifact['arms'])
        save()
        if not artifact['passed']:
            raise SystemExit(1)
    except BaseException as exc:
        artifact['error'] = repr(exc)
        save()
        raise


if __name__ == '__main__':
    main()
