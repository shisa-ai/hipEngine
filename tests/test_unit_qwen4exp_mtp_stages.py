"""Staged draft proposal, target rows and acceptance for Qwen4Exp MTP.

These tests pin the M1 provider-stage boundary: proposal preparation performs no
target work, target verification decides nothing, and acceptance consumes target
rows in row order so only the accepted prefix advances sampling controls. The
iterator-level tests prove ``iter_generate`` still runs through the extracted
stages with unchanged committed tokens, cycle splits and remaining-token
boundaries.
"""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation import qwen4_exp_mtp
from hipengine.generation.qwen4_exp_mtp_stages import (
    Qwen4ExpMTPAcceptance,
    Qwen4ExpMTPProposal,
    Qwen4ExpMTPTargetRows,
    consume_proposal_rows,
    prepare_proposal,
    verify_proposal_rows,
)
from tests.test_unit_qwen4exp_mtp_steps import (
    _BatchedTargetRunner,
    _DraftRunner,
    _drain,
    _provider,
    _request,
)


class _DraftOnly:
    """Draft owner that records the only call proposal preparation may make."""

    def __init__(self, *, position: int = 2, candidates=(11, 12)) -> None:
        self.position = int(position)
        self.candidates = tuple(int(token) for token in candidates)
        self.propose_calls: list[dict[str, object]] = []
        self.last_proposal_stage_timings_ms: dict[str, float] = {}

    def propose_chain(
        self,
        *,
        start_token,
        target_hidden_seed,
        draft_n_max,
        compact_output=False,
    ):
        self.propose_calls.append(
            {
                "start_token": int(start_token),
                "draft_n_max": int(draft_n_max),
                "compact_output": bool(compact_output),
                "has_hidden_seed": target_hidden_seed is not None,
            }
        )
        values = self.candidates[: int(draft_n_max)]
        self.position += len(values)
        return tuple(
            SimpleNamespace(token_id=token, hidden_seed=np.zeros(8, np.float32))
            for token in values
        )


class _VerifierOnly:
    """Target owner that records the verify call and refuses every other call."""

    def __init__(self, *, token_ids=(11, 12, 13), logits=(), hidden_seeds=None) -> None:
        self.token_ids = tuple(int(token) for token in token_ids)
        self.logits = tuple(logits)
        self.hidden_seeds = (
            np.arange(len(self.token_ids) * 8, dtype=np.float32).reshape(-1, 8)
            if hidden_seeds is None
            else hidden_seeds
        )
        self.verify_calls: list[tuple[tuple[int, ...], dict[str, object]]] = []
        self.commit_calls: list[int] = []
        self.step_calls = 0
        self.prefill_calls = 0

    def verify_target_block_state_rows(self, input_token_ids, **kwargs):
        self.verify_calls.append(
            (tuple(int(token) for token in input_token_ids), dict(kwargs))
        )
        return SimpleNamespace(
            token_ids=self.token_ids,
            logits=self.logits,
            hidden_seeds=self.hidden_seeds,
        )

    def commit_target_block_rows(self, consumed: int) -> None:
        self.commit_calls.append(int(consumed))

    def step(self, *args, **kwargs):
        self.step_calls += 1
        raise AssertionError("the verification stage must not step the target")

    def prefill(self, *args, **kwargs):
        self.prefill_calls += 1
        raise AssertionError("the verification stage must not prefill")


def _rows(token_ids, *, logits=(), hidden_seeds=None) -> Qwen4ExpMTPTargetRows:
    tokens = tuple(int(token) for token in token_ids)
    return Qwen4ExpMTPTargetRows(
        token_ids=tokens,
        logits=tuple(logits),
        hidden_seeds=(
            np.arange(len(tokens) * 8, dtype=np.float32).reshape(-1, 8)
            if hidden_seeds is None
            else hidden_seeds
        ),
    )


def _accepts(proposal, rows, *, prior_tokens=(10,), finish_for=None, seen=None):
    def select(result):
        if seen is not None:
            seen.append(None if result.logits is None else np.asarray(result.logits))
        return int(result.token_id)

    return consume_proposal_rows(
        proposal,
        rows,
        prior_tokens=prior_tokens,
        select=select,
        finish_for=finish_for if finish_for is not None else (lambda ids: None),
    )


# -- proposal preparation ---------------------------------------------------


def test_proposal_preparation_takes_no_target_owner() -> None:
    names = set(inspect.signature(prepare_proposal).parameters)
    # The only hidden input it may take is the target's hidden seed, never a
    # target owner that it could step, prefill, verify or commit.
    assert sorted(name for name in names if name.startswith("target")) == [
        "target_hidden_seed"
    ]
    assert sorted(name for name in names if name.endswith("_runner")) == [
        "draft_runner"
    ]


def test_prepare_proposal_records_an_immutable_prepared_chain() -> None:
    draft = _DraftOnly(position=2, candidates=(11, 12))

    proposal = prepare_proposal(
        draft,
        start_position=2,
        root_token=10,
        target_hidden_seed=np.zeros(8, np.float32),
        draft_n_max=2,
    )

    assert isinstance(proposal, Qwen4ExpMTPProposal)
    assert proposal.start_position == 2
    assert proposal.root_token == 10
    assert proposal.candidates == (11, 12)
    assert proposal.rows == 3
    assert draft.position == 4
    assert draft.propose_calls == [
        {
            "start_token": 10,
            "draft_n_max": 2,
            "compact_output": False,
            "has_hidden_seed": True,
        }
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        proposal.root_token = 11


def test_prepare_proposal_rejects_a_diverged_draft_cursor_before_drafting() -> None:
    draft = _DraftOnly(position=5)

    with pytest.raises(RuntimeError, match="diverged before proposal"):
        prepare_proposal(
            draft,
            start_position=2,
            root_token=10,
            target_hidden_seed=None,
            draft_n_max=2,
        )

    assert draft.propose_calls == []
    assert draft.position == 5


def test_prepare_proposal_rejects_a_negative_start_position() -> None:
    draft = _DraftOnly(position=0)

    with pytest.raises(ValueError, match="start_position"):
        prepare_proposal(
            draft,
            start_position=-1,
            root_token=10,
            target_hidden_seed=None,
            draft_n_max=1,
        )

    assert draft.propose_calls == []


# -- target verification ----------------------------------------------------


def test_verification_takes_no_draft_owner() -> None:
    names = set(inspect.signature(verify_proposal_rows).parameters)
    assert sorted(name for name in names if name.endswith("_runner")) == [
        "target_runner"
    ]
    assert not any("draft" in name for name in names)


def test_verify_proposal_rows_calls_only_the_state_row_verifier() -> None:
    verifier = _VerifierOnly(token_ids=(11, 12, 13))
    proposal = Qwen4ExpMTPProposal(
        start_position=2, root_token=10, candidates=(11, 12)
    )

    rows = verify_proposal_rows(
        verifier, proposal, capture_kwargs={"capture_logits": True}
    )

    assert verifier.verify_calls == [((10, 11, 12), {"capture_logits": True})]
    assert verifier.commit_calls == []
    assert verifier.step_calls == 0
    assert verifier.prefill_calls == 0
    assert isinstance(rows, Qwen4ExpMTPTargetRows)
    assert rows.token_ids == (11, 12, 13)
    assert rows.rows == 3
    assert rows.logits == ()


def test_target_rows_require_one_hidden_row_per_verified_row() -> None:
    with pytest.raises(ValueError, match="hidden"):
        Qwen4ExpMTPTargetRows(token_ids=(11, 12), logits=(), hidden_seeds=None)
    with pytest.raises(ValueError, match="hidden"):
        Qwen4ExpMTPTargetRows(
            token_ids=(11, 12),
            logits=(),
            hidden_seeds=np.ones((3, 8), np.float32),
        )
    with pytest.raises(ValueError, match="logits"):
        Qwen4ExpMTPTargetRows(
            token_ids=(11, 12),
            logits=(np.zeros(4, np.float32),),
            hidden_seeds=np.ones((2, 8), np.float32),
        )
    with pytest.raises(ValueError, match="at least one row"):
        Qwen4ExpMTPTargetRows(
            token_ids=(), logits=(), hidden_seeds=np.ones((0, 8), np.float32)
        )


# -- acceptance -------------------------------------------------------------


def test_acceptance_stops_at_the_first_mismatch() -> None:
    proposal = Qwen4ExpMTPProposal(
        start_position=2, root_token=10, candidates=(11, 12)
    )
    rows = _rows((11, 99, 12))
    seen: list[np.ndarray | None] = []

    acceptance = _accepts(proposal, rows, seen=seen)

    assert isinstance(acceptance, Qwen4ExpMTPAcceptance)
    assert acceptance.tokens == (11, 99)
    assert acceptance.accepted == 1
    assert acceptance.mismatch_token == 99
    assert acceptance.consumed_rows == 2
    assert acceptance.root_token == 99
    assert np.array_equal(acceptance.root_hidden, rows.hidden_seeds[1])
    assert acceptance.finished is False
    assert acceptance.finish is None
    # Only the consumed prefix was sampled; the rejected tail never ran.
    assert len(seen) == 2


def test_acceptance_commits_the_bonus_row_on_a_full_match() -> None:
    proposal = Qwen4ExpMTPProposal(
        start_position=2, root_token=10, candidates=(11, 12)
    )
    rows = _rows((11, 12, 13))

    acceptance = _accepts(proposal, rows)

    assert acceptance.tokens == (11, 12, 13)
    assert acceptance.accepted == 2
    assert acceptance.mismatch_token is None
    assert acceptance.consumed_rows == 3
    assert acceptance.root_token == 13
    assert np.array_equal(acceptance.root_hidden, rows.hidden_seeds[2])
    assert acceptance.finished is False


def test_acceptance_stops_at_a_finish_boundary() -> None:
    proposal = Qwen4ExpMTPProposal(
        start_position=2, root_token=10, candidates=(11, 12)
    )
    rows = _rows((11, 12, 13))

    acceptance = _accepts(
        proposal,
        rows,
        prior_tokens=(10,),
        finish_for=lambda ids: "stop" if len(ids) >= 3 else None,
    )

    assert acceptance.tokens == (11, 12)
    assert acceptance.accepted == 2
    assert acceptance.mismatch_token is None
    assert acceptance.consumed_rows == 2
    assert acceptance.root_token == 12
    assert acceptance.finished is True
    assert acceptance.finish == "stop"


def test_acceptance_advances_controls_only_for_consumed_rows() -> None:
    proposal = Qwen4ExpMTPProposal(
        start_position=2, root_token=10, candidates=(11, 12, 13)
    )
    rows = _rows((11, 12, 99, 14))
    sampled: list[int] = []

    def select(result):
        sampled.append(int(result.token_id))
        return int(result.token_id)

    acceptance = consume_proposal_rows(
        proposal,
        rows,
        prior_tokens=(10,),
        select=select,
        finish_for=lambda ids: None,
    )

    assert sampled == [11, 12, 99]
    assert acceptance.accepted == 2
    assert acceptance.mismatch_token == 99
    assert acceptance.consumed_rows == 3


def test_acceptance_passes_captured_row_logits_to_the_sampler() -> None:
    proposal = Qwen4ExpMTPProposal(
        start_position=2, root_token=10, candidates=(11, 12)
    )
    logits = (
        np.zeros(4, np.float32),
        np.ones(4, np.float32),
        np.full(4, 2.0, np.float32),
    )
    seen: list[np.ndarray | None] = []

    acceptance = _accepts(proposal, _rows((11, 12, 13), logits=logits), seen=seen)

    assert [None if row is None else float(row[0]) for row in seen] == [
        0.0,
        1.0,
        2.0,
    ]
    assert acceptance.consumed_rows == 3


def test_acceptance_requires_one_row_per_proposed_token() -> None:
    proposal = Qwen4ExpMTPProposal(
        start_position=2, root_token=10, candidates=(11, 12)
    )

    with pytest.raises(ValueError, match="row"):
        _accepts(proposal, _rows((11, 12)))
    with pytest.raises(ValueError, match="row"):
        _accepts(proposal, _rows((11, 12, 13, 14)))


# -- iterator integration ---------------------------------------------------


def _count_stages(monkeypatch, *, runner):
    calls = {"prepare": [], "verify": 0, "consume": 0}
    real_prepare = qwen4_exp_mtp.prepare_proposal
    real_verify = qwen4_exp_mtp.verify_proposal_rows
    real_consume = qwen4_exp_mtp.consume_proposal_rows

    def prepare(*args, **kwargs):
        def target_work():
            return (
                getattr(runner, "prefill_calls", 0),
                getattr(runner, "step_calls", 0),
                getattr(runner, "verify_calls", 0),
            )

        before = target_work()
        proposal = real_prepare(*args, **kwargs)
        assert target_work() == before, "proposal preparation performed target work"
        calls["prepare"].append(proposal)
        return proposal

    def verify(*args, **kwargs):
        calls["verify"] += 1
        return real_verify(*args, **kwargs)

    def consume(*args, **kwargs):
        calls["consume"] += 1
        return real_consume(*args, **kwargs)

    monkeypatch.setattr(qwen4_exp_mtp, "prepare_proposal", prepare)
    monkeypatch.setattr(qwen4_exp_mtp, "verify_proposal_rows", verify)
    monkeypatch.setattr(qwen4_exp_mtp, "consume_proposal_rows", consume)
    return calls


def test_batched_iterator_runs_through_the_extracted_stages(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY", "1")
    runner = _BatchedTargetRunner(truth=(10, 11, 12, 13, 14))
    draft = _DraftRunner(proposals=((11, 12), (14, 15)))
    provider, _target, _draft = _provider(runner=runner, draft=draft)
    calls = _count_stages(monkeypatch, runner=runner)

    steps, output = _drain(provider.iter_generate((1, 2), _request()))

    assert [step.token_ids for step in steps] == [(10,), (11, 12, 13), (14,)]
    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert [proposal.candidates for proposal in calls["prepare"]] == [
        (11, 12),
        (14,),
    ]
    assert [proposal.start_position for proposal in calls["prepare"]] == [2, 5]
    assert [proposal.root_token for proposal in calls["prepare"]] == [10, 13]
    assert calls["verify"] == 1
    assert calls["consume"] == 1
    assert runner.committed_rows == [3]
    assert draft.forward_calls == [12]
    assert runner.position == draft.position == 6


def test_serial_iterator_prepares_proposals_without_row_verification(
    monkeypatch,
) -> None:
    provider, target, _draft = _provider(
        draft=_DraftRunner(proposals=((11, 98), (13, 14)))
    )
    calls = _count_stages(monkeypatch, runner=target.runner)

    steps, output = _drain(provider.iter_generate((1, 2), _request()))

    assert [step.token_ids for step in steps] == [(10,), (11, 12), (13, 14)]
    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert [proposal.candidates for proposal in calls["prepare"]] == [
        (11, 98),
        (13, 14),
    ]
    # The serial fallback verifies one row at a time inside its own step, so it
    # neither produces a row set nor runs the row acceptance stage.
    assert calls["verify"] == 0
    assert calls["consume"] == 0


@pytest.mark.parametrize(
    ("max_tokens", "expected_steps", "verify_calls", "commit_rows", "forward_calls"),
    [
        (1, [(10,)], 0, [], []),
        (2, [(10,), (11,)], 0, [], []),
        (3, [(10,), (11, 12)], 1, [2], [11]),
        (4, [(10,), (11, 12, 13)], 1, [3], [12]),
        (5, [(10,), (11, 12, 13), (14,)], 1, [3], [12]),
    ],
)
def test_batched_remaining_token_boundaries_are_unchanged(
    monkeypatch,
    max_tokens,
    expected_steps,
    verify_calls,
    commit_rows,
    forward_calls,
) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY", "1")
    runner = _BatchedTargetRunner(truth=(10, 11, 12, 13, 14))
    draft = _DraftRunner(proposals=((11, 12), (13, 14)))
    provider, _target, _draft = _provider(runner=runner, draft=draft)

    steps, output = _drain(
        provider.iter_generate((1, 2), _request(max_tokens=max_tokens))
    )

    assert [step.token_ids for step in steps] == expected_steps
    assert runner.verify_calls == verify_calls
    assert runner.committed_rows == commit_rows
    assert draft.forward_calls == forward_calls
    assert output.generated_token_ids == tuple(
        token for step in steps for token in step.token_ids
    )
    assert runner.position == draft.position == 2 + max_tokens - 1
