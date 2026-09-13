"""Teacher windows must not follow candidate predictions or stale head buffers."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts import qwen38_packed_c1_teacher_candidate as candidate


class Session:
    _resident_slot_index = 2

    def reset(self):
        self.position = 0

    def prefill(self, tokens, *, return_logits):
        assert return_logits
        self.position = len(tokens)
        return NS(logits=np.arange(8, dtype=np.float32))


class Owner:
    def __init__(self):
        self.jobs = []
        self.commits = []
        self._verify_logits_buf = object()
        self.skip_head = False
        self.head_path = 'row_linear_f32_logits'

    def _enqueue_target_block_rows_from_hidden(self, ptr, rows):
        self._last_packed_lm_head_decode_path = self.head_path

    def verify_target_blocks_batch(self, jobs, *, device_result):
        assert device_result
        job, = jobs
        self.jobs.append(job)
        count = len(job['input_token_ids'])
        if not self.skip_head:
            self._enqueue_target_block_rows_from_hidden(0, count)
        return [NS(request_id=job['request_id'], resident_slot=job['resident_slot'],
                   transaction_id=job['transaction_id'], start_position=job['session'].position,
                   row_start=0, row_end=count, deferred_packed_state=object())]

    def _commit_deferred_packed_verify_state(self, deferred, session, **kwargs):
        self.commits.append(kwargs)
        session.position = kwargs['position']


@pytest.fixture
def setup(monkeypatch):
    owner, session = Owner(), Session()
    record = dict(prompt_ids=(0, 1), inputs=tuple(range(7)) * 3,
                  logits=np.zeros((21, 8), np.float32))
    monkeypatch.setattr(candidate, '_read_logits',
                        lambda owner, rows: np.tile(np.arange(8, dtype=np.float32), (rows, 1)))
    monkeypatch.setattr(candidate, 'snapshot_committed_state', lambda session: session.position)
    monkeypatch.setattr(candidate, 'assert_committed_state_unchanged',
                        lambda a, b: (_ for _ in ()).throw(ValueError('state changed')) if a != b else None)
    return owner, session, record


@pytest.mark.parametrize('budget', range(1, 8))
def test_candidate_follows_teacher_not_own_argmax(setup, budget):
    owner, session, record = setup
    result = candidate.capture_teacher_candidate(owner, session, record, budget=budget,
                                                  use_wmma_prefill=False)
    assert [token for j in owner.jobs for token in j['input_token_ids']] == list(record['inputs'])
    assert result['logits'].shape == (21, 8)
    assert result['logits'].argmax(axis=1).tolist() == [7] * 21
    assert session.position == 23
    assert len(owner.commits) == len(owner.jobs)
    for job, commit in zip(owner.jobs, owner.commits):
        assert all(job[k] for k in ('capture_linear_state_rows', 'defer_linear_state_commit', 'defer_state_scatter'))
        assert job['resident_slot'] == 2
        assert commit['commit_row_index'] == len(job['input_token_ids']) - 1
    assert '_enqueue_target_block_rows_from_hidden' not in owner.__dict__


def test_restores_preexisting_instance_head_override(setup):
    owner, session, record = setup
    override = owner._enqueue_target_block_rows_from_hidden
    owner._enqueue_target_block_rows_from_hidden = override
    candidate.capture_teacher_candidate(owner, session, record, budget=3, use_wmma_prefill=False)
    assert owner.__dict__['_enqueue_target_block_rows_from_hidden'] is override


@pytest.mark.parametrize('fault', ['stale', 'direct', 'state', 'cursor', 'ownership', 'nan', 'shape', 'count'])
def test_rejects_invalid_capture_and_restores_hook(setup, monkeypatch, fault):
    owner, session, record = setup
    if fault == 'stale':
        owner.skip_head = True
    elif fault == 'direct':
        owner.head_path = 'direct_top1'
    elif fault in ('state', 'ownership'):
        original = owner.verify_target_blocks_batch
        def verify(*args, **kwargs):
            output = original(*args, **kwargs)
            if fault == 'state':
                session.position += 1
            else:
                output[0].resident_slot = 1
            return output
        owner.verify_target_blocks_batch = verify
    elif fault == 'cursor':
        owner._commit_deferred_packed_verify_state = lambda *a, **kw: None
    elif fault == 'count':
        owner.verify_target_blocks_batch = lambda *a, **kw: []
    elif fault == 'shape':
        monkeypatch.setattr(candidate, '_read_logits', lambda owner, rows: np.zeros((rows, 7), np.float32))
    else:
        monkeypatch.setattr(candidate, '_read_logits', lambda owner, rows: np.full((rows, 8), np.nan, np.float32))
    with pytest.raises(ValueError):
        candidate.capture_teacher_candidate(owner, session, record, budget=3, use_wmma_prefill=False)
    assert '_enqueue_target_block_rows_from_hidden' not in owner.__dict__
    if fault != 'cursor':
        assert not owner.commits
