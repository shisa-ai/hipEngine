"""Aggregate full teacher-forced rows, never averages of summary percentiles.

Caller validates exact prompt identities, model, coordinates and array hashes.
This helper checks category coverage and numerical scopes, not full qualification.
"""
import numpy as np

from hipengine.benchmark.execution_profiles import EvaluationThresholds, _summary_passes
from scripts.qwen38_packed_c1_logits import compare_logits


CATEGORIES = {'code', 'general_en', 'general_ja', 'mixed_ja_en'}
SUITES = {'canonical', 'category_heldout'}


def summarize_teacher_scopes(records):
    records = list(records)
    identities = [r['prompt_id'] for r in records]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('teacher numerical records need unique nonempty prompt identities')
    if {(r['suite'], r['category']) for r in records} != {
            (suite, category) for suite in SUITES for category in CATEGORIES}:
        raise ValueError('teacher numerical scopes must cover both complete category suites')
    groups = {}
    for row in records:
        reference, candidate = np.asarray(row['reference']), np.asarray(row['candidate'])
        if (reference.ndim != 2 or reference.shape != candidate.shape
                or min(reference.shape) <= 0
                or not np.isfinite(reference).all() or not np.isfinite(candidate).all()):
            raise ValueError('teacher numerical scope contains invalid full-logit rows')
        names = ('all', f"category:{row['category']}", f"suite:{row['suite']}",
                 f"suite:{row['suite']}/category:{row['category']}")
        for name in names:
            groups.setdefault(name, []).append((reference, candidate))
    scopes = {}
    limits = EvaluationThresholds()
    for name, pairs in groups.items():
        reference = np.concatenate([p[0] for p in pairs])
        candidate = np.concatenate([p[1] for p in pairs])
        # compare_logits applies 99% top-1 in every scope (stricter than the
        # normative 97% per-scope floor). It never grants full qualification.
        summary = compare_logits(reference, candidate)
        # Report the frozen profile envelope separately; retain the original
        # conservative screen and its callers' fail-closed behavior unchanged.
        top1_min = limits.top1_min if name == 'all' else limits.per_scope_top1_min
        calibrated = dict(kl_mean=summary['mean_kl'], kl_p95=summary['p95_kl'],
                          kl_p99=summary['p99_kl'], kl_max=summary['max_kl'],
                          top1_agreement=summary['top1'])
        summary.update(calibrated_top1_min=top1_min,
                       calibrated_numerical_envelope_passed=bool(
                           _summary_passes(calibrated, limits, top1_min=top1_min)
                           and not summary['review_rows']))
        scopes[name] = summary
    return dict(scopes=scopes, numerical_envelope_passed=all(
        row['numerical_envelope_passed'] for row in scopes.values()),
        calibrated_numerical_envelope_passed=all(
            row['calibrated_numerical_envelope_passed'] for row in scopes.values()),
        full_profile_qualification=False)
