"""Prompt copies maintain the predictor cursor without scoring unused heads."""
from types import SimpleNamespace
import numpy as np
import pytest

from hipengine.runtime.qwen4_exp_mtp import Qwen4ExpGGUFMTPDraftRunner
from hipengine.generation.qwen4_exp_mtp_stages import prepare_proposal


def runner():
    r = Qwen4ExpGGUFMTPDraftRunner.__new__(Qwen4ExpGGUFMTPDraftRunner)
    r.position = 10
    r.max_sequence_length = 32
    r.config = SimpleNamespace(vocab_size=100)
    r.calls = []
    def forward(token, hidden, **kwargs):
        r.calls.append((token, hidden, kwargs))
        r.position += 1
        r.last_forward_stage_timings_ms = {'draft_layer': 1.0}
    r.forward = forward
    return r


def test_lookup_chain_consumes_root_then_copies_without_a_vocabulary_head():
    r = runner()
    hidden = np.zeros(4, dtype=np.float32)
    result = r.propose_lookup_chain(start_token=7, target_hidden_seed=hidden,
                                    candidates=(11, 12, 13))
    assert tuple(x.token_id for x in result) == (11, 12, 13)
    assert r.position == 13
    assert [x[0] for x in r.calls] == [7, 11, 12]
    assert r.calls[0][1] is hidden
    assert all(x[1] is None for x in r.calls[1:])
    assert [x[2]['hidden_seed_resident'] for x in r.calls] == [False, True, True]
    assert all(not x[2]['run_head'] for x in r.calls)
    assert r.last_proposal_stage_timings_ms == {'draft_layer': 3.0}


@pytest.mark.parametrize('candidates', [(), (1,2,3,4,5), (-1,), (100,), (1.2,)])
def test_invalid_lookup_is_rejected_before_advancing(candidates):
    r = runner()
    with pytest.raises((ValueError, TypeError)):
        r.propose_lookup_chain(start_token=7, target_hidden_seed=np.zeros(4),
                               candidates=candidates)
    assert not r.calls


def test_lookup_capacity_is_checked_before_advancing():
    r = runner();r.position = 31
    with pytest.raises(ValueError, match='capacity'):
        r.propose_lookup_chain(start_token=7, target_hidden_seed=np.zeros(4), candidates=(1,2))
    assert not r.calls


def test_lookup_stage_uses_forced_draft_only_and_preserves_proposal_abi():
    r = runner()
    p = prepare_proposal(r, start_position=10, root_token=7,
                         target_hidden_seed=np.zeros(4), draft_n_max=3,
                         lookup_candidates=(11,12))
    assert p.candidates == (11,12) and p.rows == 3 and r.position == 12


def test_lookup_stage_rejects_over_budget_before_advancing():
    r = runner()
    with pytest.raises(ValueError, match='budget'):
        prepare_proposal(r, start_position=10, root_token=7,
                         target_hidden_seed=np.zeros(4), draft_n_max=1,
                         lookup_candidates=(11,12))
    assert not r.calls
