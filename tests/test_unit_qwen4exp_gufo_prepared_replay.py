"""Contracts for source-pinned prepared Gufo cohorts and rate denominators."""
from types import SimpleNamespace

import pytest

from scripts import qwen4exp_gufo_prepared_replay as replay


def sample(index, *, prompt=2048, cached=2048, outputs=128, drafts=0, decode_ms=2000):
    fields = dict(request_index=index, prompt_tokens=prompt, cached_prompt_tokens=cached,
                  prefill_tokens=prompt-cached, completion_tokens=outputs, draft_tokens=drafts,
                  draft_accepted_tokens=drafts//2, decode_ms=decode_ms,
                  completion_sha256='f' * 64)
    return SimpleNamespace(**fields, public=lambda: fields)


def cohort(width=2, **options):
    return SimpleNamespace(samples=tuple(sample(i, **options) for i in range(width)),
                           span_ms=3000.0)


def test_distinguishes_sum_of_request_rates_from_cohort_wall_rate():
    result = replay.checked_round(cohort(), width=2, speculative=False)
    assert result['sum_request_decode_output_tok_s'] == 128.0
    assert result['cohort_output_tok_s'] == 256 / 3
    assert result['generated_tokens'] == 256
    assert result['output_identity'] == 'visible_text_sha256_not_native_ids'


@pytest.mark.parametrize('options', [
    {'prompt': 2100}, {'cached': 2040}, {'cached': 2049}, {'outputs': 127},
    {'drafts': 5}, {'decode_ms': 0}, {'decode_ms': float('nan')},
])
def test_prepared_contract_rejects_wrong_counts_route_or_timing(options):
    with pytest.raises(ValueError):
        replay.checked_round(cohort(**options), width=2, speculative=False)


def test_mtp_requires_actual_draft_work():
    with pytest.raises(ValueError, match='draft'):
        replay.checked_round(cohort(), width=2, speculative=True)
    assert replay.checked_round(cohort(drafts=16), width=2, speculative=True)['generated_tokens'] == 256


def test_sample_count_and_request_ownership_are_exact():
    with pytest.raises(ValueError, match='width'):
        replay.checked_round(cohort(1), width=2, speculative=False)
    bad = cohort()
    bad.samples = (bad.samples[0], bad.samples[0])
    with pytest.raises(ValueError, match='index'):
        replay.checked_round(bad, width=2, speculative=False)


def test_prepares_every_session_before_each_measured_round_without_extra_warmup():
    calls, records = [], []
    def run(**kw):
        calls.append(kw)
        return cohort(kw['concurrency'], outputs=kw['max_tokens'])
    upstream = SimpleNamespace(_run_corpus_round=run,
                               PromptCase=lambda *args: SimpleNamespace(identifier=args[0], text=args[2]))
    rows = replay.prepared_rounds(upstream, 'url', 'model', 'prompt', task='prose', width=2,
                                 speculative=False, repetitions=3, timeout=10,
                                 record=lambda *args: records.append(args))
    assert [c['max_tokens'] for c in calls] == [1,128,1,128,1,128]
    assert [c['repetition'] for c in calls] == [-1,0,-2,1,-3,2]
    assert all(c['cache_prompt'] is True and c['endpoint_profile'] == 'gufo' for c in calls)
    assert all(c['cases'][0] is c['cases'][1] for c in calls)  # published homogeneous corpus
    assert len(rows) == 3 and len(records) == 6
    assert [r['repetition'] for r in rows] == [0,1,2]


def test_failed_preparation_is_recorded_and_cannot_release_measurement():
    records = []
    upstream = SimpleNamespace(_run_corpus_round=lambda **kw: cohort(2, outputs=0),
                               PromptCase=lambda *args: object())
    with pytest.raises(ValueError, match='preparation'):
        replay.prepared_rounds(upstream, 'url', 'model', 'prompt', task='prose', width=2,
                              speculative=False, repetitions=1, timeout=10,
                              record=lambda *args: records.append(args))
    assert len(records) == 1 and records[0][0] == 'preparation'
