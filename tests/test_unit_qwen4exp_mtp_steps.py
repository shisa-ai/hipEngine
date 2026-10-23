"""Cooperative per-cycle stepping for the Qwen4Exp MTP provider.

``iter_generate`` must reproduce the exact algorithm of ``_generate_one`` one
committed step at a time: same proposal depth, same verification path, same
committed IDs, same cycle accounting and telemetry. The local fakes also
cover state-row verification and its bonus-row path on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.deadline import (
    GenerationCancelled,
    GenerationCancellationToken,
)
from hipengine.generation.qwen4_exp_mtp import (
    Qwen4ExpMTPProgress,
    Qwen4ExpMTPTextProvider,
)
from hipengine.generation.registry import GenerationOutput, GenerationRequest
from hipengine.speculative.registry import SpeculativeProviderConfig


class _Tokenizer:
    def __init__(self, eos_token_id: int = 99_999) -> None:
        self.eos_token_id = int(eos_token_id)

    def encode(self, text: str):
        return [1, 2]

    def decode(self, token_ids, *, skip_special=False):
        return " ".join(str(int(token)) for token in token_ids)


class _TargetRunner:
    """Serial-exact target fake: ``prefill`` and per-token ``step`` only."""

    def __init__(self, truth=(10, 11, 12, 13, 14)) -> None:
        self.max_sequence_length = 4_096
        self.truth = tuple(int(token) for token in truth)
        self.position = 0
        self.cursor = 0
        self.prefill_calls = 0
        self.step_calls = 0

    @staticmethod
    def _hidden(value: int) -> np.ndarray:
        return np.full(8, float(value), dtype=np.float32)

    def prefill(self, token_ids, *, capture_hidden_seeds=False):
        self.prefill_calls += 1
        self.position = len(token_ids)
        self.cursor = 0
        hidden = np.stack([self._hidden(token) for token in token_ids])
        return SimpleNamespace(
            token_id=self.truth[0],
            hidden_seeds=hidden,
            hidden_seed=hidden[-1],
        )

    def step(self, token_id: int, *, capture_hidden_seed=False):
        self.step_calls += 1
        assert int(token_id) == self.truth[self.cursor]
        self.cursor += 1
        self.position += 1
        hidden = self._hidden(token_id)
        return SimpleNamespace(
            token_id=self.truth[self.cursor],
            hidden_seeds=hidden.reshape(1, -1),
            hidden_seed=hidden,
        )


class _BatchedTargetRunner(_TargetRunner):
    """Same truth chain plus the prefill-shaped state-row verify API."""

    def __init__(self, truth=(10, 11, 12, 13, 14)) -> None:
        super().__init__(truth)
        self.verify_calls = 0
        self.committed_rows: list[int] = []
        self._pending: tuple[int, int] | None = None

    def verify_target_block_state_rows(
        self,
        input_token_ids,
        *,
        capture_logits: bool = False,
    ):
        self.verify_calls += 1
        inputs = tuple(int(token) for token in input_token_ids)
        assert 2 <= len(inputs) <= 8
        # Row i is the target's argmax after input row i, scripted from the
        # committed truth chain. The provider owns every commit/rollback
        # decision, so the rows do not depend on the draft candidates.
        first_row = self.cursor + 1
        rows = self.truth[first_row:first_row + len(inputs)]
        assert len(rows) == len(inputs)
        self._pending = (self.position, len(inputs))
        self.position += len(inputs)
        hidden = np.stack([self._hidden(row) for row in rows])
        return SimpleNamespace(token_ids=rows, logits=(), hidden_seeds=hidden)

    def commit_target_block_rows(self, consumed: int) -> None:
        assert self._pending is not None
        start, _count = self._pending
        self._pending = None
        self.committed_rows.append(int(consumed))
        self.position = start + int(consumed)
        self.cursor += int(consumed)


class _DraftRunner:
    max_sequence_length = 4_096

    def __init__(self, proposals=((11, 98), (13, 14))) -> None:
        self.position = 0
        self.proposals = tuple(tuple(int(token) for token in row) for row in proposals)
        self.proposal_index = 0
        self.prime_calls = 0
        self.propose_calls = 0
        self.forward_calls: list[int] = []
        self.trimmed: list[int] = []
        self.last_proposal_stage_timings_ms: dict[str, float] = {}

    def prime_prompt(self, token_ids, hidden_rows) -> None:
        self.prime_calls += 1
        assert hidden_rows.shape == (len(token_ids), 8)
        self.position = len(token_ids)
        self.proposal_index = 0

    def propose_chain(
        self,
        *,
        start_token,
        target_hidden_seed,
        draft_n_max,
        compact_output=False,
    ):
        self.propose_calls += 1
        values = self.proposals[self.proposal_index][:draft_n_max]
        self.proposal_index += 1
        self.position += len(values)
        self.last_proposal_stage_timings_ms = {
            "draft_input_fusion": 1.0,
            "draft_layer": 2.0,
        }
        return tuple(
            SimpleNamespace(
                token_id=token,
                hidden_seed=np.full(8, float(token), dtype=np.float32),
            )
            for token in values
        )

    def forward(
        self,
        token_id,
        hidden_seed,
        *,
        capture_logits=True,
        capture_hidden_seed=True,
        capture_token_id=True,
        run_head=True,
    ) -> None:
        self.forward_calls.append(int(token_id))
        self.position += 1

    def trim(self, position) -> None:
        self.position = int(position)
        self.trimmed.append(self.position)

    def close(self) -> None:
        pass


class _TargetGenerator:
    backend = "hip_gfx1151"

    def __init__(self, *, runner=None, eos_token_id: int = 99_999) -> None:
        self.runner = runner if runner is not None else _TargetRunner()
        self.tokenizer = _Tokenizer(eos_token_id)


def _request(*, max_tokens: int = 5, ignore_eos: bool = True, **overrides):
    values = {
        "prompts": ((1, 2),),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": ignore_eos,
        "top_k": 1,
    }
    values.update(overrides)
    return GenerationRequest(**values)


def _provider(*, runner=None, draft=None, eos_token_id: int = 99_999, candidate_budget: int = 2):
    target = _TargetGenerator(runner=runner, eos_token_id=eos_token_id)
    draft = draft if draft is not None else _DraftRunner()
    provider = Qwen4ExpMTPTextProvider(
        target_generator=target,
        config=SpeculativeProviderConfig(
            provider="qwen4_exp_mtp",
            draft_model="/tmp/fake.gguf",
            candidate_budget=candidate_budget,
        ),
        draft_runner=draft,
        draft_resident=object(),
    )
    return provider, target, draft


def _drain(iterator) -> tuple[list[Qwen4ExpMTPProgress], GenerationOutput]:
    steps: list[Qwen4ExpMTPProgress] = []
    while True:
        try:
            steps.append(next(iterator))
        except StopIteration as stop:
            return steps, stop.value


class _VocabularyDraft(_DraftRunner):
    def __init__(self):
        super().__init__()
        self.beginnings = []
        self.histories = []

    def begin_draft_vocabulary(self, special_ids):
        self.beginnings.append(tuple(special_ids))
        self.histories = []

    def prepare_draft_vocabulary(self, committed, *, anchor):
        self.histories.append((tuple(committed), anchor))

    def draft_vocabulary_diagnostics(self):
        return {"prepared_cycles": len(self.histories)}


def test_subset_head_sees_commits_not_rejected_proposals_and_resets_between_requests():
    draft = _VocabularyDraft()
    provider, target, _ = _provider(draft=draft)
    target.tokenizer.token_types = (1, 2, 3, 4, 6)
    target.tokenizer._extra_special_ids = lambda: (42,)
    _, result = _drain(provider.iter_generate((1, 2), _request()))
    assert draft.beginnings == [(1, 2, 3, 42, 99_999)]
    assert draft.histories == [((1, 2, 10), 10), ((1, 2, 10, 11, 12), 12)]
    assert result.telemetry.diagnostics['draft_vocabulary'] == {'prepared_cycles': 2}
    _drain(provider.iter_generate((6, 7), _request()))
    assert len(draft.beginnings) == 2
    assert draft.histories[0] == ((6, 7, 10), 10)


def test_subset_head_includes_committed_k0_tokens_before_the_next_proposal():
    draft = _VocabularyDraft()
    provider, _, _ = _provider(draft=draft)
    steps = provider.iter_generate((1, 2), _request())
    assert next(steps).token_ids == (10,)
    assert steps.send(0).token_ids == (11,)
    assert not draft.histories
    steps.send(2)
    assert draft.histories[0] == ((1, 2, 10, 11), 11)
    steps.close()


def test_iter_generate_is_lazy_until_the_first_step() -> None:
    provider, target, draft = _provider()

    steps = provider.iter_generate((1, 2), _request())

    assert target.runner.prefill_calls == 0
    assert draft.prime_calls == 0
    assert draft.propose_calls == 0

    first = next(steps)

    assert target.runner.prefill_calls == 1
    assert draft.prime_calls == 1
    assert draft.propose_calls == 0
    assert isinstance(first, Qwen4ExpMTPProgress)
    assert first.token_ids == (10,)
    assert first.cycle is None
    assert first.finished is False


def test_iter_generate_rejects_session_reuse_before_model_work():
    provider, target, draft = _provider()
    iterator = provider.iter_generate((1, 2), _request(resident_session_key="reuse-not-supported"))
    with pytest.raises(NotImplementedError, match="resident session"):
        next(iterator)
    assert target.runner.prefill_calls == draft.prime_calls == 0


def test_iter_generate_yields_the_initial_root_then_one_cycle_per_step() -> None:
    provider, target, draft = _provider()

    steps, output = _drain(provider.iter_generate((1, 2), _request()))

    assert [step.token_ids for step in steps] == [(10,), (11, 12), (13, 14)]
    assert [step.cycle for step in steps] == [
        None,
        steps[1].cycle,
        steps[2].cycle,
    ]
    first_cycle, second_cycle = steps[1].cycle, steps[2].cycle
    assert first_cycle is not None and second_cycle is not None
    assert first_cycle.start_position == 2
    assert first_cycle.candidates == (11, 98)
    assert first_cycle.accepted == 1
    assert first_cycle.mismatch_token == 12
    assert first_cycle.committed_position == 4
    assert second_cycle.start_position == 4
    assert second_cycle.candidates == (13, 14)
    assert second_cycle.accepted == 2
    assert second_cycle.mismatch_token is None
    assert second_cycle.committed_position == 6
    assert [step.finished for step in steps] == [False, False, True]
    assert tuple(token for step in steps for token in step.token_ids) == (
        10,
        11,
        12,
        13,
        14,
    )
    assert draft.trimmed == [4, 6]
    assert target.runner.position == draft.position == 6
    assert provider.last_cycles == (first_cycle, second_cycle)

    assert isinstance(output, GenerationOutput)
    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert output.text == "10 11 12 13 14"
    assert output.finish_details is not None
    assert output.finish_details.reason == "length"
    assert output.finish_details.length_limit == 5
    assert output.telemetry is not None
    assert output.telemetry.timing == {
        "mtp_generated_draft_tokens": 4,
        "mtp_accepted_draft_tokens": 3,
        "mtp_cycles_count": 2,
    }
    assert output.telemetry.decode_state is not None
    assert output.telemetry.decode_state.execution_path == "qwen4exp_mtp_serial_exact_verify"
    assert output.telemetry.decode_state.generated_tokens == 5
    assert output.telemetry.decode_state.step_index == 5
    diagnostics = output.telemetry.diagnostics
    assert diagnostics["proposed_draft_tokens"] == 4
    assert diagnostics["accepted_draft_tokens"] == 3
    assert diagnostics["draft_acceptance"] == 0.75
    assert [row["accepted"] for row in diagnostics["cycles"]] == [1, 2]
    assert diagnostics["cycles"][0]["mismatch_token"] == 12
    phase = diagnostics["phase_census"]
    assert phase["cycles"] == 2
    assert phase["target_verify_rows"] == 4
    assert phase["proposal"]["calls"] == 2
    assert phase["target_verify"]["calls"] == 4
    assert phase["acceptance_control"]["calls"] == 4
    assert phase["draft_commit_or_rollback"]["calls"] == 2
    assert phase["target_prefill_hidden_export"]["calls"] == 1
    assert phase["draft_stages_ms"] == {"draft_input_fusion": 2.0, "draft_layer": 4.0}


def _comparable(output: GenerationOutput) -> dict:
    """Return the arithmetic/accounting part of an output, without wall times."""

    assert output.telemetry is not None
    phase = dict(output.telemetry.diagnostics["phase_census"])
    phase.pop("target_prefill_hidden_export")
    phase.pop("proposal")
    phase.pop("target_verify")
    phase.pop("acceptance_control")
    phase.pop("draft_commit_or_rollback")
    phase.pop("draft_stages_ms")
    diagnostics = dict(output.telemetry.diagnostics)
    diagnostics["phase_census"] = phase
    return {
        "ids": output.generated_token_ids,
        "text": output.text,
        "finish": output.finish_details,
        "decode_state": output.telemetry.decode_state,
        "timing": output.telemetry.timing,
        "diagnostics": diagnostics,
    }


@pytest.mark.parametrize("max_tokens", [5, 1, 3])
def test_iter_generate_terminal_output_matches_direct_generation(max_tokens: int) -> None:
    direct_provider, direct_target, direct_draft = _provider()
    direct = direct_provider.generate_detailed(_request(max_tokens=max_tokens))[0]

    step_provider, step_target, step_draft = _provider()
    steps, stepped = _drain(
        step_provider.iter_generate((1, 2), _request(max_tokens=max_tokens))
    )

    assert _comparable(stepped) == _comparable(direct)
    assert tuple(token for step in steps for token in step.token_ids) == (
        direct.generated_token_ids
    )
    assert steps[-1].finished is True
    assert all(not step.finished for step in steps[:-1])
    assert step_target.runner.position == direct_target.runner.position
    assert step_draft.trimmed == direct_draft.trimmed
    assert step_draft.position == direct_draft.position


def test_iter_generate_uses_injected_target_and_draft_without_swapping_owners() -> None:
    provider, owned_target, owned_draft = _provider()
    injected_target = _TargetRunner(truth=(20, 21, 22, 23, 24))
    injected_draft = _DraftRunner(proposals=((21, 97), (23, 24)))
    prefill_result = injected_target.prefill((1, 2), capture_hidden_seeds=True)

    steps, output = _drain(
        provider.iter_generate(
            (1, 2),
            _request(),
            target=injected_target,
            draft=injected_draft,
            prefill_result=prefill_result,
        )
    )

    assert output.generated_token_ids == (20, 21, 22, 23, 24)
    assert [step.token_ids for step in steps] == [(20,), (21, 22), (23, 24)]
    # The supplied prefill is used as-is: no second target prefill, and the
    # phase census reports no hidden-export pass for it.
    assert injected_target.prefill_calls == 1
    assert injected_draft.prime_calls == 1
    assert output.telemetry is not None
    phase = output.telemetry.diagnostics["phase_census"]
    assert phase["target_prefill_hidden_export"]["calls"] == 0
    # Draft stage timings come from the injected draft, not the idle owner.
    assert phase["draft_stages_ms"] == {
        "draft_input_fusion": 2.0,
        "draft_layer": 4.0,
    }
    # The provider's own owners are never touched or replaced.
    assert provider.draft is owned_draft
    assert provider.target_generator.runner is owned_target.runner
    assert owned_target.runner.prefill_calls == 0
    assert owned_target.runner.step_calls == 0
    assert owned_draft.prime_calls == 0
    assert owned_draft.propose_calls == 0
    assert owned_draft.trimmed == []
    assert owned_target.runner.position == 0
    assert owned_draft.position == 0


def test_interleaved_iterators_keep_request_owned_progress():
    provider, idle_target, idle_draft = _provider()
    a = provider.iter_generate(
        (1, 2), _request(), target=_TargetRunner((20, 21, 22, 23, 24)),
        draft=_DraftRunner(((21, 90), (23, 24))),
    )
    b = provider.iter_generate(
        (3, 4), _request(), target=_TargetRunner((40, 41, 42, 43, 44)),
        draft=_DraftRunner(((41, 80), (43, 44))),
    )
    assert next(a).token_ids == (20,)
    assert next(b).token_ids == (40,)
    assert next(a).token_ids == (21, 22)
    assert next(b).token_ids == (41, 42)
    _, output_a = _drain(a)
    _, output_b = _drain(b)
    assert output_a.generated_token_ids == (20, 21, 22, 23, 24)
    assert output_b.generated_token_ids == (40, 41, 42, 43, 44)
    assert output_a.telemetry.timing == output_b.telemetry.timing == {
        "mtp_generated_draft_tokens": 4,
        "mtp_accepted_draft_tokens": 3,
        "mtp_cycles_count": 2,
    }
    assert idle_target.runner.prefill_calls == idle_draft.prime_calls == 0


def test_iter_generate_honours_an_already_primed_draft() -> None:
    provider, _owned_target, _owned_draft = _provider()
    injected_target = _TargetRunner(truth=(20, 21, 22, 23, 24))
    injected_draft = _DraftRunner(proposals=((21, 97), (23, 24)))
    prefill_result = injected_target.prefill((1, 2), capture_hidden_seeds=True)
    injected_draft.prime_prompt((1, 2), prefill_result.hidden_seeds)
    assert injected_draft.prime_calls == 1

    _steps, output = _drain(
        provider.iter_generate(
            (1, 2),
            _request(),
            target=injected_target,
            draft=injected_draft,
            prefill_result=prefill_result,
            draft_primed=True,
        )
    )

    assert output.generated_token_ids == (20, 21, 22, 23, 24)
    assert injected_draft.prime_calls == 1
    assert injected_draft.propose_calls == 2


def test_iter_generate_rejects_a_prefill_result_without_hidden_rows() -> None:
    provider, _target, _draft = _provider()

    steps = provider.iter_generate(
        (1, 2),
        _request(),
        prefill_result=SimpleNamespace(token_id=10, hidden_seeds=None, hidden_seed=None),
    )

    with pytest.raises(RuntimeError, match="hidden rows"):
        next(steps)


def test_iter_generate_zero_max_tokens_returns_the_existing_empty_output() -> None:
    provider, target, draft = _provider()

    steps, output = _drain(provider.iter_generate((1, 2), _request(max_tokens=0)))

    assert steps == []
    assert output.generated_token_ids == ()
    assert output.text == ""
    assert output.telemetry is None
    assert output.finish_details is not None
    assert output.finish_details.reason == "length"
    assert output.finish_details.length_limit == 0
    assert output.finish_details.sampler_mode == "greedy_speculative_mtp"
    assert target.runner.prefill_calls == 0
    assert draft.prime_calls == 0
    assert draft.propose_calls == 0


def test_iter_generate_stops_after_an_eos_root_token() -> None:
    provider, _target, draft = _provider(eos_token_id=10)

    steps, output = _drain(provider.iter_generate((1, 2), _request(ignore_eos=False)))

    assert len(steps) == 1
    assert steps[0].token_ids == (10,)
    assert steps[0].cycle is None
    assert steps[0].finished is True
    assert output.generated_token_ids == (10,)
    assert output.finish_details is not None
    assert output.finish_details.reason == "eos"
    assert output.finish_details.eos_token_id == 10
    assert draft.propose_calls == 0
    assert output.telemetry is not None
    assert output.telemetry.timing["mtp_cycles_count"] == 0


def test_iter_generate_stops_on_eos_inside_a_cycle() -> None:
    provider, _target, _draft = _provider(
        runner=_TargetRunner(truth=(10, 99_999, 12, 13, 14)),
        eos_token_id=99_999,
    )

    steps, output = _drain(provider.iter_generate((1, 2), _request(ignore_eos=False)))

    assert [step.token_ids for step in steps] == [(10,), (99_999,)]
    assert steps[-1].finished is True
    cycle = steps[1].cycle
    assert cycle is not None
    assert cycle.accepted == 0
    assert cycle.mismatch_token == 99_999
    assert output.generated_token_ids == (10, 99_999)
    assert output.finish_details is not None
    assert output.finish_details.reason == "eos"
    assert output.finish_details.eos_token_id == 99_999


def test_iter_generate_observes_cancellation_between_steps() -> None:
    token = GenerationCancellationToken()
    provider, _target, draft = _provider()

    steps = provider.iter_generate((1, 2), _request(cancellation_token=token))
    first = next(steps)
    assert first.token_ids == (10,)
    assert draft.propose_calls == 0

    token.cancel()

    with pytest.raises(GenerationCancelled):
        next(steps)
    assert draft.propose_calls == 0


def test_scheduler_can_interleave_provider_synced_k0_with_planned_cycles():
    provider, target, draft = _provider(
        draft=_DraftRunner(proposals=((12, 98), (14, 97))),
    )
    steps = provider.iter_generate((1, 2), _request())
    assert next(steps).token_ids == (10,)
    k0 = steps.send(0)
    assert k0.token_ids == (11,) and k0.cycle is None
    assert draft.propose_calls == 0
    assert target.runner.position == draft.position == 3
    k1 = steps.send(1)
    assert k1.token_ids == (12,) and k1.cycle.candidates == (12,)
    k0 = steps.send(0)
    assert k0.token_ids == (13,) and k0.cycle is None
    last = steps.send(1)
    assert last.token_ids == (14,) and last.finished
    with pytest.raises(StopIteration) as terminal:
        next(steps)
    output = terminal.value.value
    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert output.telemetry.timing == {
        "mtp_generated_draft_tokens": 2,
        "mtp_accepted_draft_tokens": 2,
        "mtp_cycles_count": 2,
    }
    assert draft.forward_calls == [10, 12]
    assert target.runner.position == draft.position == 6


def test_scheduler_k0_stops_on_eos_without_counting_a_proposal():
    provider, target, draft = _provider(eos_token_id=11)
    steps = provider.iter_generate((1, 2), _request(ignore_eos=False))
    next(steps)
    progress = steps.send(0)
    assert progress.token_ids == (11,) and progress.finished
    with pytest.raises(StopIteration) as terminal:
        next(steps)
    output = terminal.value.value
    assert output.finish_details.reason == "eos"
    assert output.telemetry.timing["mtp_cycles_count"] == 0
    assert output.telemetry.diagnostics["phase_census"]["target_k0"]["calls"] == 1
    assert draft.propose_calls == 0
    assert target.runner.position == draft.position == 3


@pytest.mark.parametrize("budget", [-1, 3])
def test_scheduler_depth_is_validated_before_model_mutation(budget):
    provider, target, draft = _provider(candidate_budget=2)
    steps = provider.iter_generate((1, 2), _request())
    next(steps)
    with pytest.raises(ValueError, match="budget"):
        steps.send(budget)
    assert draft.propose_calls == target.runner.step_calls == 0


def test_iter_generate_batched_verify_commits_the_bonus_row(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY", "1")
    runner = _BatchedTargetRunner(truth=(10, 11, 12, 13, 14))
    draft = _DraftRunner(proposals=((11, 12), (14, 15)))
    provider, _target, _draft = _provider(runner=runner, draft=draft)

    steps, output = _drain(provider.iter_generate((1, 2), _request()))

    # Only the first cycle leaves headroom for the bonus row; the one-token
    # tail cycle is a plain serial target step.
    assert runner.verify_calls == 1
    assert runner.committed_rows == [3]
    assert draft.forward_calls == [12]
    assert [step.token_ids for step in steps] == [(10,), (11, 12, 13), (14,)]
    assert steps[1].cycle is not None
    assert steps[1].cycle.accepted == 2
    assert steps[1].cycle.mismatch_token is None
    assert steps[1].cycle.committed_position == 5
    assert steps[2].cycle is not None
    assert steps[2].cycle.candidates == (14,)
    assert steps[2].cycle.committed_position == 6
    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert output.telemetry is not None
    assert (
        output.telemetry.decode_state.execution_path
        == "qwen4exp_mtp_batched_state_rows_verify"
    )
    assert output.telemetry.diagnostics["target_verify"] == "batched_state_rows"
    assert output.telemetry.timing["mtp_generated_draft_tokens"] == 3
    assert output.telemetry.timing["mtp_accepted_draft_tokens"] == 3
    assert draft.position == runner.position == 6


def test_iter_generate_batched_verify_commits_a_partial_block_on_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY", "1")
    runner = _BatchedTargetRunner(truth=(10, 11, 12, 13, 14))
    draft = _DraftRunner(proposals=((11, 99), (13, 14)))
    provider, _target, _draft = _provider(runner=runner, draft=draft)

    steps, output = _drain(provider.iter_generate((1, 2), _request()))

    assert runner.verify_calls == 2
    assert runner.committed_rows == [2, 2]
    assert draft.forward_calls == [13]
    assert [step.token_ids for step in steps] == [(10,), (11, 12), (13, 14)]
    assert steps[1].cycle is not None
    assert steps[1].cycle.candidates == (11, 99)
    assert steps[1].cycle.accepted == 1
    assert steps[1].cycle.mismatch_token == 12
    assert steps[1].cycle.committed_position == 4
    assert steps[2].cycle is not None
    assert steps[2].cycle.accepted == 1
    assert steps[2].cycle.committed_position == 6
    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert draft.position == runner.position == 6


def test_iter_generate_batched_and_serial_paths_agree_on_committed_ids(
    monkeypatch,
) -> None:
    """Both verification paths commit the same tokens; the cycle split differs.

    The batched cycle folds the bonus row into its own commit, so its per-cycle
    candidate/accepted split is not the serial one. Only the committed IDs and
    the reported verification path are comparable.
    """

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY", "1")
    batched_provider, _batched_target, _batched_draft = _provider(
        runner=_BatchedTargetRunner(),
        draft=_DraftRunner(proposals=((11, 12), (13, 14))),
    )
    serial_provider, _serial_target, _serial_draft = _provider(
        draft=_DraftRunner(proposals=((11, 12), (13, 14))),
    )

    batched_steps, batched = _drain(
        batched_provider.iter_generate((1, 2), _request())
    )
    serial_steps, serial = _drain(serial_provider.iter_generate((1, 2), _request()))

    assert batched.generated_token_ids == serial.generated_token_ids == (
        10,
        11,
        12,
        13,
        14,
    )
    assert batched.telemetry is not None and serial.telemetry is not None
    assert (
        batched.telemetry.decode_state.execution_path
        == "qwen4exp_mtp_batched_state_rows_verify"
    )
    assert (
        serial.telemetry.decode_state.execution_path
        == "qwen4exp_mtp_serial_exact_verify"
    )
    assert [step.token_ids for step in batched_steps] == [(10,), (11, 12, 13), (14,)]
    assert [step.token_ids for step in serial_steps] == [(10,), (11, 12), (13, 14)]
    assert batched.telemetry.timing["mtp_cycles_count"] == 2
    assert serial.telemetry.timing["mtp_cycles_count"] == 2
