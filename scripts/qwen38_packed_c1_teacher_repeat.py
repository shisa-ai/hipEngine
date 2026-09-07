"""Compare validated teacher-candidate captures at identical schedules.

Caller must load and validate manifests, exact suite coverage and array hashes,
then attach owned logits/prefill_logits to each record. This checks repeatability
only; it does not replace numerical or state/task gates.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_candidate_capture(directory, fixture, *, teacher_manifest_sha256):
    """Load actual capture arrays against an already validated strict fixture.

    The caller loads the teacher with require_runtime_provenance=True and supplies
    its actual teacher.json hash. Numerical scopes must still be evaluated.
    """
    from hipengine.execution_profiles import manifest_sha256, validate_variant_manifest
    from scripts.qwen38_packed_c1_teacher import teacher_windows
    directory = Path(directory)
    run = json.loads((directory / 'candidate.json').read_text())
    manifest = validate_variant_manifest(run['candidate_manifest'])
    teacher = fixture['runtime_manifest']
    if (run.get('complete') is not True or run.get('error')
            or run['execution_profile'] != 'production'
            or manifest['execution_profile'] != 'production'
            or manifest_sha256(manifest) != run['candidate_manifest_sha256']
            or run['model_sha256'] != fixture['model_sha256']
            or run['teacher_manifest_sha256'] != teacher_manifest_sha256
            or run['teacher_runtime_manifest_sha256'] != fixture['runtime_manifest_sha256']
            or run['kv_storage_dtype'] != fixture['kv_storage_dtype']
            or any(manifest[key] != teacher[key] for key in ('backend', 'model', 'quant', 'kv_policy'))):
        raise ValueError('candidate capture provenance differs from teacher or runtime')
    if type(run['capacity']) is not int or not 1 <= run['capacity'] <= 8:
        raise ValueError('candidate capacity outside campaign range')
    if ([r['prompt_id'] for r in run['records']]
            != [r['prompt_id'] for r in fixture['records']]):
        raise ValueError('candidate prompt schedule differs from teacher')
    for record, reference in zip(run['records'], fixture['records'], strict=True):
        windows = teacher_windows(reference, budget=run['budget'])
        if (any(record[key] != reference[key] for key in ('category', 'suite'))
                or len(record['windows']) != len(windows)):
            raise ValueError('candidate scope or window count differs from teacher')
        for actual, expected in zip(record['windows'], windows, strict=True):
            if (actual['position'] != expected['position']
                    or tuple(actual['tokens']) != tuple(expected['tokens'])
                    or actual['rows'] != len(expected['tokens'])
                    or type(actual['resident_slot']) is not int
                    or not 0 <= actual['resident_slot'] < run['capacity']
                    or actual['resident_slot'] != record['windows'][0]['resident_slot']
                    or actual['head_path'] not in {'q6_rowtile_f32_logits', 'row_linear_f32_logits'}):
                raise ValueError('candidate window differs from teacher')
        path = directory / record['logits_file']
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('candidate array path escapes capture directory')
        with np.load(path, allow_pickle=False) as arrays:
            for key in ('logits', 'prefill_logits'):
                value = arrays[key].copy()
                if (value.dtype != np.float32 or value.shape != reference[key].shape
                        or not np.isfinite(value).all()
                        or hashlib.sha256(value.tobytes()).hexdigest() != record[key + '_sha256']):
                    raise ValueError('candidate array shape, values or hash invalid')
                record[key] = value
    return run


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
