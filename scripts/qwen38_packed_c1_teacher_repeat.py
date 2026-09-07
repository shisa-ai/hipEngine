"""Compare validated teacher-candidate captures at identical schedules.

Caller must load and validate manifests, exact suite coverage and array hashes,
then attach owned logits/prefill_logits to each record. This checks repeatability
only; it does not replace numerical or state/task gates.
"""
import numpy as np


def assert_teacher_repeats(runs):
    runs = list(runs)
    if len(runs) < 3 or any(r.get('complete') is not True or r.get('error') for r in runs):
        raise ValueError('repeatability requires at least three complete captures')
    base = runs[0]
    keys = ('model_sha256', 'candidate_manifest_sha256', 'teacher_runtime_manifest_sha256',
            'teacher_manifest_sha256', 'capacity', 'budget', 'target_contexts',
            'execution_profile', 'kv_storage_dtype', 'target_use_wmma_prefill')
    for run in runs:
        if any(key not in run or run[key] != base[key] for key in keys):
            raise ValueError('repeatability runtime or teacher provenance changed')
        records = run.get('records', [])
        ids = [r['prompt_id'] for r in records]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('repeatability requires unique nonempty prompt records')
        if ids != [r['prompt_id'] for r in base['records']]:
            raise ValueError('repeatability prompt schedule changed')
        for actual, expected in zip(records, base['records'], strict=True):
            if any(actual[key] != expected[key] for key in ('category', 'suite', 'windows')):
                raise ValueError('repeatability window coordinates changed')
            for key, rank in (('logits', 2), ('prefill_logits', 1)):
                a, b = np.asarray(actual[key]), np.asarray(expected[key])
                if (a.dtype != np.float32 or a.ndim != rank or not a.size
                        or not np.isfinite(a).all() or a.shape != b.shape
                        or a.tobytes() != b.tobytes()):
                    raise ValueError('repeatability full-logit bytes differ or are invalid')
    return dict(repeats=len(runs), prompts=len(base['records']),
                decode_rows=sum(len(r['logits']) for r in base['records']),
                prefill_rows=len(base['records']), bit_identical=True,
                full_profile_qualification=False)
