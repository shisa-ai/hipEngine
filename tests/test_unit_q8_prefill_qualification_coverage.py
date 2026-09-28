"""Fail-closed coverage and claim accounting; no GPU required."""
from types import SimpleNamespace

import pytest

from scripts.execution_profile_q8_wmma_prefill_layers_gate import (
    CANONICAL_FIXTURE_SHA256, qualification_coverage, qualification_verdict,
)


def captures(depth=128, repeats=3):
    return [SimpleNamespace(
        prompt_id=f'{category}-p{shape}', category=category,
        strict=[None] * (depth + 1),
        candidate_runs={'candidate': tuple([None] * (depth + 1) for _ in range(repeats))},
    ) for category in ('code', 'general_en', 'general_ja', 'mixed_ja_en')
       for shape in (512, 1024, 4096)]


def coverage(rows=None, **kwargs):
    return qualification_coverage(
        captures=captures() if rows is None else rows,
        fixture_sha256=kwargs.get('fixture_sha256', CANONICAL_FIXTURE_SHA256),
        prefill_chunk_size=kwargs.get('prefill_chunk_size', 1024),
    )


def test_canonical_pin_matches_checked_in_fixture():
    from scripts.execution_profile_q8_wmma_prefill_layers_gate import (
        REPO_ROOT, DEFAULT_FIXTURE, _load_fixture,
    )
    _, digest = _load_fixture(REPO_ROOT / DEFAULT_FIXTURE)
    assert digest == CANONICAL_FIXTURE_SHA256


def test_wrong_category_cannot_qualify():
    rows = captures()
    rows[0].category = 'general_en'
    assert not coverage(rows)['complete']


def test_complete_canonical_capture_is_numerical_evidence_not_profile_admission():
    c = coverage()
    assert c['complete']
    assert c['observed_teacher_rows'] == 1548
    v = qualification_verdict(c, quality_passed=True, deterministic=True, clean=True)
    assert v['canonical_numerical_qualification_claim']
    assert v['measurement_valid']
    assert not v['profile_qualification_claim']
    assert v['qualification_blockers']


@pytest.mark.parametrize('rows', [captures()[:3], captures(depth=24), captures(repeats=2),
                                  captures() + captures()[:1]])
def test_reduced_or_duplicate_capture_cannot_qualify(rows):
    c = coverage(rows)
    assert not c['complete']
    v = qualification_verdict(c, quality_passed=True, deterministic=True, clean=True)
    assert not v['canonical_numerical_qualification_claim']
    assert not v['measurement_valid']
    assert v['status'] == 'invalid_or_screen_only'


def test_every_candidate_repeat_must_cover_every_transition():
    rows = captures()
    rows[-1].candidate_runs['candidate'][-1].pop()
    assert not coverage(rows)['complete']


def test_fixture_identity_cannot_be_replaced_by_matching_case_names():
    assert not coverage(fixture_sha256='changed')['complete']


def test_chunk_scope_is_recorded_not_inherited_from_another_run():
    assert coverage(prefill_chunk_size=2048)['prefill_chunk_size'] == 2048
    assert not coverage(prefill_chunk_size=0)['complete']


@pytest.mark.parametrize('failed', ['quality_passed', 'deterministic', 'clean'])
def test_failed_gate_blocks_both_claims(failed):
    flags = dict(quality_passed=True, deterministic=True, clean=True)
    flags[failed] = False
    v = qualification_verdict(coverage(), **flags)
    assert not v['measurement_valid']
    assert not v['canonical_numerical_qualification_claim']
    assert not v['profile_qualification_claim']
