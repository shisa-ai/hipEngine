"""Validate fixed strict-teacher data before packed candidate evaluation.

This checks identity and internal consistency, not runtime provenance or model
correctness. Callers must re-encode each expected rendered prompt with the model
tokenizer and compare prompt_ids before execution. A valid fixture alone does
not qualify a production profile.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from scripts.qwen38_packed_c1_teacher import load_teacher_prompts


def _ids(value, name):
    if (not isinstance(value, list) or not value
            or any(type(token) is not int or token < 0 for token in value)):
        raise ValueError(f'teacher {name} must be nonempty nonnegative integer IDs')
    return tuple(value)


def load_teacher_fixture(directory, *, expected_model_sha256, expected_prompts=None):
    """Return manifest fields with validated, owned arrays in ``records``.

    Each record contains tuple prompt_ids/inputs and FP32 logits/prefill_logits,
    and can be passed to teacher_windows. Defaults require the complete canonical
    and heldout suites; explicit expected_prompts support bounded CPU fixtures.
    full_profile_qualification is always False, regardless of input metadata.
    """
    directory = Path(directory).resolve()
    if (not isinstance(expected_model_sha256, str)
            or re.fullmatch(r'[0-9a-f]{64}', expected_model_sha256) is None):
        raise ValueError('expected model SHA256 must be independently supplied')
    manifest = json.loads((directory / 'teacher.json').read_text())
    if (not isinstance(manifest, dict)
            or manifest.get('kind') != 'packed_c1_strict_teacher'
            or manifest.get('complete') is not True
            or manifest.get('execution_profile') != 'strict'
            or manifest.get('model_sha256') != expected_model_sha256):
        raise ValueError('teacher requires complete strict same-model manifest')
    steps = manifest.get('steps')
    if type(steps) is not int or steps < 1:
        raise ValueError('teacher steps must be a positive integer')
    prompts = load_teacher_prompts() if expected_prompts is None else list(expected_prompts)
    expected = {r['id']: r for r in prompts}
    if not expected or len(expected) != len(prompts):
        raise ValueError('expected teacher prompt identities must be unique and nonempty')
    records = manifest.get('records')
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError('teacher must cover the exact expected prompt suite')
    validated, seen, vocabulary = [], set(), None
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('teacher record must be an object')
        identity = row.get('prompt_id')
        if not isinstance(identity, str) or identity not in expected or identity in seen:
            raise ValueError('teacher has duplicate or unexpected prompt identity')
        seen.add(identity)
        for key in ('category', 'suite', 'prompt_sha256'):
            if row.get(key) != expected[identity][key]:
                raise ValueError(f'teacher prompt {key} differs from expected suite')
        prompt = _ids(row.get('prompt_ids'), 'prompt_ids')
        inputs = _ids(row.get('inputs'), 'inputs')
        if len(inputs) != steps:
            raise ValueError('teacher input length differs from declared steps')
        filename = row.get('logits_file')
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not filename.endswith('.npz')):
            raise ValueError('teacher logits filename must be a local NPZ basename')
        path = (directory / filename).resolve()
        if path.parent != directory:
            raise ValueError('teacher logits path escapes fixture directory')
        try:
            with np.load(path, allow_pickle=False) as arrays:
                logits = arrays['logits'].copy()
                prefill = arrays['prefill_logits'].copy()
        except (OSError, KeyError, ValueError) as error:
            raise ValueError('teacher full-logit arrays are missing or invalid') from error
        if (logits.dtype != np.float32 or prefill.dtype != np.float32
                or logits.ndim != 2 or prefill.ndim != 1
                or logits.shape != (steps, prefill.size) or not prefill.size
                or not np.isfinite(logits).all() or not np.isfinite(prefill).all()):
            raise ValueError('teacher requires aligned finite FP32 full-vocabulary arrays')
        if vocabulary is not None and prefill.size != vocabulary:
            raise ValueError('teacher vocabulary changes between prompts')
        vocabulary = prefill.size
        if max(prompt + inputs) >= vocabulary:
            raise ValueError('teacher token ID is outside vocabulary')
        for key, array in (('logits_sha256', logits), ('prefill_logits_sha256', prefill)):
            if hashlib.sha256(array.tobytes()).hexdigest() != row.get(key):
                raise ValueError(f'teacher {key} differs from array bytes')
        chain = (int(prefill.argmax()), *map(int, logits[:-1].argmax(axis=1)))
        if inputs != chain:
            raise ValueError('teacher inputs do not follow independent greedy logits')
        validated.append(dict(row, prompt_ids=prompt, inputs=inputs,
                              logits=logits, prefill_logits=prefill))
    return dict(manifest, records=validated, full_profile_qualification=False)
