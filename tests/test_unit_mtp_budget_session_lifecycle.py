"""Dense diagnostic observations use the session's terminal cycle boundary.

Resource and scheduler doubles force every acceptance prefix. These tests cover
control ordering, not GPU arithmetic or the scheduler's acceptance algorithm.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hipengine.runtime import qwen35_gguf_mtp as runtime
from scripts.mtp_budget_diagnostics import MtpBudgetSequencePolicy


@pytest.fixture
def diagnostic_session(monkeypatch):
    events = []
    state = NS(accepted=0, fail=None, remaining=4, budget=None, observations=[], keep_active=False)

    def event(name):
        events.append(name)
        if state.fail == name:
            raise RuntimeError(name)

    class Scheduler:
        def __init__(self, capacity):
            self.completed = {}
            self.active_batch = NS(requests={7: NS(remaining_decode=state.remaining)})
            self.graph_buckets = NS(stats=NS(to_json_dict=lambda: {}))

        def submit(self, prompt, *, max_new_tokens, request_id):
            assert request_id == 7
            return request_id

        def admit_pending(self):
            pass

        def next_prefill_work(self, **kwargs):
            pass

        def record_generated(self, rows):
            pass

        def next_speculative_verify_work(self, draft, **kwargs):
            return NS(target_batch=NS(rows=state.budget + 1, positions=(3,), mode='verify_chain'))

        def plan_speculative_verify(self, policy, work, graph_factory, **kwargs):
            self.plan = NS(
                graph=graph_factory('test'),
                transaction=NS(transaction_id=1, committed=False, rolled_back=False),
            )
            return self.plan

        def bind_speculative_verify_buffers(self, plan, buffers):
            return plan

        def plan_speculative_commit(self, plan, summary):
            return NS(summary=summary, commit_plan=NS(transaction_id=1))

        def bind_speculative_commit_buffers(self, commit, buffers):
            return commit

        def commit_speculative_kv_transaction(self, policy, plan):
            event('scheduler_commit')
            self.plan.transaction.committed = True
            return self.plan.transaction

        def finalize_speculative_accept(self, transaction, plan, **kwargs):
            event('finalize')
            if not state.keep_active:
                self.completed[7] = NS(generated_tokens=(11, 12))

        def rollback_speculative_kv_transaction(self, policy, plan):
            event('scheduler_rollback')
            plan.transaction.rolled_back = True

    monkeypatch.setattr(runtime, 'ResidentBatchScheduler', Scheduler)
    monkeypatch.setattr(runtime, '_maybe_launch_device_proposal', lambda *a, **k: None)
    from hipengine.speculative import streaming
    monkeypatch.setattr(streaming, 'limit_chain_accept_eos', lambda batch, summary, **k: (summary, False))
    bucket = runtime.Qwen35GGUFVerifyGraphBucket(
        key='test', owner=NS(spec=NS(bucket='test')), remaining_decode=None,
    )

    def propose(context, *, candidate_budget, return_logits):
        state.budget = candidate_budget
        event('propose')
        return NS(candidate_tokens=tuple(range(candidate_budget)))

    def prepare(batch, **kwargs):
        event('prepare')
        return NS(
            summary=NS(accepted_counts=(state.accepted,),
                       full_accept=(state.accepted == state.budget,),
                       next_tokens=(12,), commit_rows=(state.accepted,),
                       commit_positions=(3 + state.accepted,)),
            batch=batch, buffers=object(), target_top1=(),
            target_logits=np.empty((0, 0)), gpu_accept_match_cpu=True,
            target_verify_mode='test', native_graph_submitted=False,
            native_graph_capture_ms=0., native_graph_submit_ms=0.,
            native_graph_readback_ms=0., native_graph_fallback_reason=None,
            native_device_accept_commit=False,
        )

    decoder = object.__new__(runtime.Qwen35GGUFMTPDecodeSession)
    decoder.candidate_budget = 3
    decoder.quant = 'test'
    decoder.allow_graph = False
    decoder.target = NS(
        reset=lambda: event('target_reset'), position=3,
        runner=NS(weights=NS(config=NS(expert_used_count=0))),
    )
    decoder.draft_provider = NS(
        reset_request=lambda rid: event('draft_reset'), propose=propose,
        advance_full_accept_tail=lambda *a, **k: event('repair'),
    )
    decoder.verifier = NS(
        graph_bucket=lambda *a: bucket, prepare=prepare,
        commit=lambda *a: event('target_commit'),
        finish=lambda *a: event('finish'), rollback=lambda *a: event('target_rollback'),
    )
    decoder._prefill_target_and_draft = lambda *a, **k: NS(token_id=11)
    decoder._register_kv_policy = lambda rid: object()
    decoder._proposal_target_hidden = lambda: np.zeros((1, 2), dtype=np.float32)

    class ObservedPolicy(MtpBudgetSequencePolicy):
        def record_cycle(self, result):
            state.observations.append(tuple(events))
            event('observe')
            super().record_cycle(result)

    def generate(budget, **kwargs):
        policy = ObservedPolicy((budget,))
        result = decoder.generate((1, 2, 3), max_new_tokens=8,
                                  request_id=7, budget_policy=policy, **kwargs)
        return result, policy

    return state, events, generate


@pytest.mark.parametrize('budget,accepted', [(b, a) for b in (1, 2, 3) for a in range(b + 1)])
def test_diagnostic_observes_every_acceptance_prefix_after_terminal_commit(
    diagnostic_session, budget, accepted,
):
    state, events, generate = diagnostic_session
    state.accepted = accepted
    result, policy = generate(budget)
    assert events == ['target_reset', 'draft_reset', 'propose', 'prepare',
                      'target_commit', 'scheduler_commit', 'finalize', 'finish', 'observe']
    assert state.observations == [tuple(events[:-1])]
    observation = policy.summary()['results'][0]
    assert observation['budget'] == budget
    assert observation['accepted_count'] == accepted
    assert observation['visible_tokens'] == accepted + 1
    assert observation['full_accept'] is (accepted == budget)
    assert result.accepted_counts == (accepted,)
    assert result.cycle_records[0]['budget'] == budget


@pytest.mark.parametrize('remaining', [1, 2, 3])
def test_diagnostic_tail_clips_requested_budget(diagnostic_session, remaining):
    state, events, generate = diagnostic_session
    state.remaining = remaining
    result, policy = generate(3)
    assert state.budget == remaining
    assert policy.summary()['decisions'] == [remaining]
    assert result.cycle_records[0]['budget'] == remaining


@pytest.mark.parametrize('failure,rollback', [
    ('propose', []), ('prepare', ['scheduler_rollback']),
    ('target_commit', ['target_rollback', 'scheduler_rollback']),
    ('finish', ['target_rollback']), ('repair', []),
])
def test_diagnostic_never_observes_failed_cycle(diagnostic_session, failure, rollback):
    state, events, generate = diagnostic_session
    state.fail = failure
    state.keep_active = failure == 'repair'
    with pytest.raises(RuntimeError, match=failure):
        generate(3)
    assert state.observations == []
    assert [e for e in events if e.endswith('rollback')] == rollback
    state.fail = None
    state.keep_active = False
    events.clear()
    result, policy = generate(1)
    assert events[:2] == ['target_reset', 'draft_reset']
    assert policy.summary()['cycle_count'] == 1
    assert result.cycles == 1


def test_diagnostic_cancellation_after_commit_observes_before_publication(diagnostic_session):
    state, events, generate = diagnostic_session

    state.keep_active = True

    def checkpoint():
        if 'repair' in events:
            raise RuntimeError('cancel after commit')

    with pytest.raises(RuntimeError, match='cancel after commit'):
        generate(2, checkpoint=checkpoint, lifecycle_hook=events.append)
    assert events[-4:] == ['after_target_commit', 'repair', 'after_draft_repair', 'observe']
    assert 'before_output_publication' not in events
    assert len(state.observations) == 1
    assert not any(e.endswith('rollback') for e in events)
