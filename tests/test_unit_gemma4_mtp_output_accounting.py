"""Gemma 4 MTP publishes the shared MTP-versus-AR output accounting block.

The provider already recorded its cycles, but only as route-local diagnostics:
a reader could see ``proposed_draft_tokens`` and ``accepted_draft_tokens`` and
still not answer which emitted tokens came from committed speculative cycles
and which from an autoregressive step. That question has one shared answer --
``hipengine/speculative/accounting.py`` -- and every other speculative adapter
publishes it as ``diagnostics.specdec2_mtp2``.

These tests drive the shipped provider with host doubles and pin four things
the migration must not get wrong:

* only committed output is counted, never a proposal and never the prefill
  root token that seeds the first cycle,
* accepted, corrected and bonus tokens are attributed to the cycle that
  committed them, including the case where every draft is rejected,
* the zero-draft tail cycle is autoregressive *inside* the plan, and a stop,
  EOS or the output horizon truncates the group before it is counted,
* the block is request-local: a second request through the same provider
  starts from zero, and a plain (budget 0) row on the same provider publishes
  no speculative block at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

import hipengine.generation.gemma4_mtp as gemma4_mtp_module
from hipengine.generation.deadline import (
    GenerationCancelled,
    GenerationDeadlineExceeded,
)
from hipengine.generation.gemma4_mtp import Gemma4MTPTextProvider
from hipengine.generation.registry import GenerationRequest
from hipengine.speculative.accounting import span_accounting
from hipengine.speculative.registry import SpeculativeProviderConfig

_VOCAB = 64
_PROMPT = (5, 11, 17, 23, 29, 31, 37)
# No EOS: the fake target has one token per position and never emits a stop.
_EOS = -1

_LEGACY_DIAGNOSTICS = frozenset(
    {
        "speculative_provider",
        "candidate_budget",
        "proposed_draft_tokens",
        "accepted_draft_tokens",
        "draft_acceptance",
        "tokens_per_cycle",
        "cycles",
        "target_verify",
        "draft_rollback",
        "accept_rule",
        "draft_law",
        "verify_equivalence",
        "phase_census",
    }
)


def _next_token(position: int) -> int:
    """The token the fake target produces after processing ``position`` tokens."""

    return 1 + (position * 7 + 3) % (_VOCAB - 1)


def _greedy_reference(prompt_length: int, max_tokens: int) -> list[int]:
    return [_next_token(prompt_length - 1 + index) for index in range(max_tokens)]


class _FakeRunner:
    """A runner with ``Gemma4Runner``'s row shapes and a resettable cursor."""

    def __init__(self, position: int = 0) -> None:
        self.position = int(position)
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        self.position = 0

    def rewind(self, position: int) -> None:
        self.position = int(position)

    def forward(
        self,
        tokens,
        *,
        apply_softcap: bool = True,
        logits_rows: int = 1,
        verification: bool = False,
    ) -> np.ndarray:
        del apply_softcap, verification
        block = [int(token) for token in tokens]
        wanted = int(logits_rows)
        if not 1 <= wanted <= len(block):
            raise ValueError(f"logits_rows {wanted} is outside 1..{len(block)}")
        trailing = np.stack(
            [self._row(self.position + index) for index in range(len(block))][-wanted:]
        )
        self.position += len(block)
        # The flat one-row return is `Gemma4Runner.forward`'s, not a shortcut.
        return trailing if wanted > 1 else trailing[0]

    @staticmethod
    def _row(position: int) -> np.ndarray:
        row = np.full((_VOCAB,), -1.0, dtype=np.float32)
        row[_next_token(position)] = 1.0
        return row


@dataclass
class _FakeDrafter:
    """Proposes what the fake target will do, so drafts are accepted."""

    runner: _FakeRunner
    wrong_at: int | None = None
    max_drafts: int = 4
    calls: int = 0
    on_draft: object | None = None

    def draft(self, token: int, *, hidden_row: int = -1, select=None) -> list[int]:
        del token, hidden_row
        self.calls += 1
        if self.on_draft is not None:
            self.on_draft(self.calls)
        drafts = [
            _next_token(self.runner.position + index)
            for index in range(self.max_drafts)
        ]
        if self.wrong_at is not None and self.wrong_at < len(drafts):
            drafts[self.wrong_at] = (drafts[self.wrong_at] % (_VOCAB - 1)) + 1
        if select is None:
            return drafts
        chosen = []
        for draft in drafts:
            row = np.full((_VOCAB,), -1.0, dtype=np.float32)
            row[draft] = 1.0
            chosen.append(int(select(row)))
        return chosen


class _FakeTokenizer:
    eos_token_id = _EOS

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(_PROMPT)

    @staticmethod
    def decode(token_ids, skip_special: bool = False) -> str:
        del skip_special
        return " ".join(str(int(token)) for token in token_ids)


class _FakeGenerator:
    def __init__(self, runner: _FakeRunner) -> None:
        self.tokenizer = _FakeTokenizer()
        self._runner = runner

    def _ensure_runner(self) -> _FakeRunner:
        return self._runner


@dataclass
class _Run:
    output: object
    runner: _FakeRunner
    drafter: _FakeDrafter
    provider: Gemma4MTPTextProvider

    @property
    def tokens(self) -> list[int]:
        return [int(token) for token in self.output.generated_token_ids]

    @property
    def diagnostics(self) -> dict:
        return dict(self.output.telemetry.diagnostics or {})

    @property
    def accounting(self) -> dict | None:
        return self.diagnostics.get("specdec2_mtp2")

    @property
    def legacy_cycles(self) -> list[dict]:
        return list(self.diagnostics.get("cycles") or ())


def _provider(
    *,
    budget: int,
    incoming_position: int = 0,
    wrong_at: int | None = None,
    on_draft: object | None = None,
    provider: Gemma4MTPTextProvider | None = None,
    runner: _FakeRunner | None = None,
    drafter: _FakeDrafter | None = None,
) -> tuple[Gemma4MTPTextProvider, _FakeRunner, _FakeDrafter]:
    runner = _FakeRunner(position=incoming_position) if runner is None else runner
    drafter = (
        _FakeDrafter(runner=runner, wrong_at=wrong_at, on_draft=on_draft)
        if drafter is None
        else drafter
    )
    if provider is None:
        provider = Gemma4MTPTextProvider(
            target_generator=_FakeGenerator(runner),
            config=SpeculativeProviderConfig(
                provider="gemma4_mtp",
                draft_model="/nonexistent/sidecar.gguf",
                candidate_budget=budget,
            ),
            drafter=drafter,
        )
    return provider, runner, drafter


def _request(max_tokens: int, **options) -> GenerationRequest:
    settings = {
        "temperature": 0.0,
        "ignore_eos": True,
        "top_p": 1.0,
        **options,
    }
    return GenerationRequest(prompts=(_PROMPT,), max_tokens=max_tokens, **settings)


def _run_cycle(
    *,
    budget: int,
    max_tokens: int,
    wrong_at: int | None = None,
    incoming_position: int = 0,
    on_draft: object | None = None,
    provider: Gemma4MTPTextProvider | None = None,
    runner: _FakeRunner | None = None,
    request_options: dict | None = None,
) -> _Run:
    provider, runner, drafter = _provider(
        budget=budget,
        incoming_position=incoming_position,
        wrong_at=wrong_at,
        on_draft=on_draft,
        provider=provider,
        runner=runner,
    )
    output = provider.generate_detailed(
        _request(max_tokens, **(request_options or {}))
    )[0]
    return _Run(output=output, runner=runner, drafter=drafter, provider=provider)


def _drive_steps(
    provider: Gemma4MTPTextProvider,
    request: GenerationRequest,
    *,
    runner: _FakeRunner,
    candidate_budget: int,
):
    """Drive ``generation_steps`` the way a resident owner does."""

    steps = provider.generation_steps(
        request.prompts[0],
        request,
        runner=runner,
        candidate_budget=candidate_budget,
    )
    result = None
    while True:
        try:
            operation = steps.send(result)
        except StopIteration as done:
            return done.value
        result = provider.execute_operation(operation)


# --------------------------------------------------------------- the schema


def test_accepted_cycles_publish_the_shared_accounting_block() -> None:
    """Every draft accepted: accepted drafts plus one verified bonus per cycle."""

    run = _run_cycle(budget=6, max_tokens=16)
    accounting = run.accounting

    assert accounting is not None, "the route published no shared accounting block"
    assert accounting["requested_budget"] == 6
    # The helper reads the effective budget on its own key. A route that leaves
    # it unset publishes ``candidate_budget: 0`` beside a positive request,
    # which reads as "asked for six, ran at zero".
    assert accounting["candidate_budget"] == 6
    assert accounting["cycles"] == len(run.legacy_cycles) == 3
    assert accounting["selected_depth_histogram"] == {"0": 1, "6": 2}
    # Legacy route diagnostics are preserved beside the block, not replaced.
    assert _LEGACY_DIAGNOSTICS <= set(run.diagnostics)
    assert isinstance(run.diagnostics["cycles"], list)
    assert run.diagnostics["proposed_draft_tokens"] == (
        accounting["generated_draft_tokens"]
    )
    assert run.diagnostics["accepted_draft_tokens"] == (
        accounting["accepted_draft_tokens"]
    )
    # Two depth-6 cycles commit seven tokens each; the tail cycle commits one.
    assert accounting["mtp_output_tokens"] == 14
    assert accounting["ar_output_tokens_in_cycles"] == 1
    # The prefill root token seeds the first cycle and is not speculative output.
    assert accounting["mtp_output_tokens"] == len(run.tokens) - 1 - 1
    assert accounting["first_fallback_position"] == len(run.tokens) - 1
    assert accounting["provider_readiness"] is None
    assert accounting["plan_group_rows"] is None


def test_a_corrected_draft_counts_committed_tokens_not_proposals() -> None:
    """The third draft is wrong: two accepted, then the target's own token.

    A block that counted proposals, or the drafts the accept walk observed
    before it settled, would report the whole group. Only the published prefix
    is output.
    """

    run = _run_cycle(budget=6, max_tokens=8, wrong_at=2)
    accounting = run.accounting

    assert accounting is not None
    assert run.legacy_cycles[0]["accepted"] == 2
    assert run.legacy_cycles[0]["committed"] == 3
    assert accounting["generated_draft_tokens"] > accounting["accepted_draft_tokens"]
    # Every committed token is accounted for, and nothing more.
    assert accounting["mtp_output_tokens"] + accounting["ar_output_tokens_in_cycles"] == (
        len(run.tokens) - 1
    )
    first = run.legacy_cycles[0]
    assert accounting["mtp_output_tokens"] == sum(
        cycle["committed"] for cycle in run.legacy_cycles if cycle["candidates"]
    )


def test_a_rejected_first_draft_counts_only_the_bonus_token() -> None:
    """A fully rejected cycle still commits the target-verified bonus token."""

    run = _run_cycle(budget=6, max_tokens=8, wrong_at=0)
    accounting = run.accounting

    assert accounting is not None
    assert all(cycle["accepted"] == 0 for cycle in run.legacy_cycles)
    speculative_cycles = accounting["cycles"] - accounting["selected_depth_histogram"]["0"]
    assert speculative_cycles == 6
    # One bonus token per speculative cycle, and no draft counted.
    assert accounting["mtp_output_tokens"] == speculative_cycles
    assert accounting["accepted_draft_tokens"] == 0
    assert accounting["generated_draft_tokens"] == sum(
        len(cycle["candidates"]) for cycle in run.legacy_cycles
    )
    assert accounting["generated_draft_tokens"] > accounting["mtp_output_tokens"]
    assert accounting["mtp_output_tokens"] + 1 + 1 == len(run.tokens)


def test_the_zero_draft_tail_is_autoregressive_inside_the_plan() -> None:
    """``remaining == 1`` runs a one-row verify, which is a depth-zero cycle."""

    run = _run_cycle(budget=1, max_tokens=8)
    accounting = run.accounting

    assert accounting is not None
    assert accounting["selected_depth_histogram"]["0"] == 1
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert accounting["mtp_output_tokens"] == len(run.tokens) - 2
    assert accounting["first_fallback_position"] == len(run.tokens) - 1
    assert accounting["ar_step_reason_counts"] == {}


def test_intent_without_a_cycle_reports_a_zero_cycle_block() -> None:
    """``max_tokens == 1`` attaches the drafter but never runs a cycle."""

    run = _run_cycle(budget=4, max_tokens=1)
    accounting = run.accounting

    assert accounting is not None
    assert accounting["requested_budget"] == 4
    assert accounting["candidate_budget"] == 4
    assert accounting["cycles"] == 0
    assert accounting["mtp_output_tokens"] == 0
    assert accounting["ar_output_tokens_in_cycles"] == 0
    assert run.tokens == _greedy_reference(len(_PROMPT), 1)
    assert run.legacy_cycles == []


def test_a_plain_row_on_the_same_provider_publishes_no_block() -> None:
    """Budget 0 is how the resident owner serves a plain row on this provider."""

    provider, runner, _drafter = _provider(budget=4)
    request = _request(6)

    output = _drive_steps(provider, request, runner=runner, candidate_budget=0)

    diagnostics = dict(output.telemetry.diagnostics or {})
    assert "specdec2_mtp2" not in diagnostics
    # The legacy route fields are still what they were for a zero-budget row:
    # a zero-draft cycle per token, and no speculative block over them.
    assert _LEGACY_DIAGNOSTICS <= set(diagnostics)
    assert diagnostics["proposed_draft_tokens"] == 0
    assert diagnostics["cycles"]
    assert all(cycle["candidates"] == [] for cycle in diagnostics["cycles"])
    assert [int(token) for token in output.generated_token_ids] == (
        _greedy_reference(len(_PROMPT), 6)
    )


def test_a_stop_inside_a_cycle_counts_only_the_published_prefix() -> None:
    """A multi-token stop that completes mid-group truncates the commit.

    The accept walk observed every accepted draft and its bonus to find where
    the stop landed, but the reply carries only the prefix. Those discarded
    tokens must not appear in the accounting.
    """

    reference = _greedy_reference(len(_PROMPT), 8)
    run = _run_cycle(
        budget=6,
        max_tokens=16,
        request_options={
            "stop_token_sequences": ((reference[0], reference[1], reference[2]),)
        },
    )
    accounting = run.accounting

    assert accounting is not None
    assert run.output.finish_details.reason == "stop"
    assert run.tokens == reference[:3]
    assert accounting["cycles"] == 1
    assert accounting["mtp_output_tokens"] == 2
    assert accounting["mtp_output_tokens"] + 1 == len(run.tokens)
    assert accounting["generated_draft_tokens"] == 6


def test_an_eos_inside_a_cycle_counts_only_the_published_prefix() -> None:
    reference = _greedy_reference(len(_PROMPT), 16)
    run = _run_cycle(
        budget=6,
        max_tokens=16,
        request_options={"eos_token_id": reference[3], "ignore_eos": False},
    )
    accounting = run.accounting

    assert accounting is not None
    assert run.output.finish_details.reason == "stop"
    assert run.tokens == reference[:4]
    assert accounting["cycles"] == 1
    assert accounting["mtp_output_tokens"] == 3
    assert accounting["mtp_output_tokens"] + 1 == len(run.tokens)


@pytest.mark.parametrize("budget", [1, 2, 6, 8])
@pytest.mark.parametrize("max_tokens", [1, 2, 5, 6, 8, 9, 16, 17])
def test_the_horizon_never_counts_more_than_was_emitted(budget, max_tokens) -> None:
    """Committed output reconciles for every budget/horizon pair.

    ``mtp_output + in_cycle_ar + prefill_root`` is the emitted list by
    construction; the point of the sweep is that no truncation path counts a
    proposal, a seed or a discarded tail on either side of a cycle boundary.
    """

    run = _run_cycle(budget=budget, max_tokens=max_tokens)
    accounting = run.accounting

    assert accounting is not None
    assert len(run.tokens) == max_tokens
    assert run.tokens == _greedy_reference(len(_PROMPT), max_tokens)
    assert accounting["cycles"] == len(run.legacy_cycles)
    assert sum(accounting["selected_depth_histogram"].values()) == accounting["cycles"]
    assert (
        accounting["mtp_output_tokens"]
        + accounting["ar_output_tokens_in_cycles"]
        + 1
        == len(run.tokens)
    )
    assert accounting["generated_draft_tokens"] == sum(
        len(cycle["candidates"]) for cycle in run.legacy_cycles
    )
    assert accounting["accepted_draft_tokens"] == sum(
        cycle["accepted"] for cycle in run.legacy_cycles
    )
    depth0 = accounting["selected_depth_histogram"].get("0", 0)
    assert accounting["mtp_output_tokens"] == (
        accounting["accepted_draft_tokens"] + accounting["cycles"] - depth0
    )


def test_committed_spans_tile_the_emitted_output(monkeypatch) -> None:
    """The diagnostic spans reconcile against the token list, not the subtraction."""

    monkeypatch.setenv("HIPENGINE_MTP2_OUTPUT_SPANS", "1")
    run = _run_cycle(budget=2, max_tokens=8)
    accounting = run.accounting

    assert accounting is not None
    assert accounting["cycles"] >= 2
    spans = accounting["output_spans"]
    assert spans[0]["position"] == 1
    assert [span["mode"] for span in spans] == ["mtp"] * (len(spans) - 1) + ["ar"]
    reconciled = span_accounting(
        spans,
        completion_tokens=len(run.tokens),
        mtp_output_tokens=accounting["mtp_output_tokens"],
        ar_output_tokens=len(run.tokens) - accounting["mtp_output_tokens"],
        ar_output_tokens_in_cycles=accounting["ar_output_tokens_in_cycles"],
    )
    assert reconciled["reconciled"] is True, reconciled["reconciled_reasons"]
    # The only unspanned token is the prefill root, which precedes the plan.
    assert reconciled["unspanned_tokens"] == 1
    assert reconciled["expected_unspanned_ar_tokens"] == 1


def test_cycle_phase_sums_agree_with_the_legacy_census(monkeypatch) -> None:
    """The block adds per-cycle phase sums without changing the legacy ones.

    Both are accumulated from the same host-wall deltas, so the migration
    cannot silently move the route's reported timing.
    """

    ticks = {"value": 0.0}

    def _tick() -> float:
        ticks["value"] += 0.0125
        return ticks["value"]

    monkeypatch.setattr(gemma4_mtp_module.time, "perf_counter", _tick)
    run = _run_cycle(budget=3, max_tokens=12)
    accounting = run.accounting
    census = run.diagnostics["phase_census"]

    assert accounting is not None
    cycle_timing = accounting["cycle_timing_ms"]
    assert cycle_timing["proposal"] == census["proposal"]["ms"] > 0.0
    assert cycle_timing["target"] == census["target_verify"]["ms"] > 0.0
    assert cycle_timing["accept"] == census["accept"]["ms"] > 0.0
    # The prefill is a request-level phase, not a cycle phase.
    assert "target_prefill" not in cycle_timing
    assert census["target_prefill"]["ms"] > 0.0
    # Phases this route does not measure report zero rather than a guess.
    assert cycle_timing["provider_update"] == 0.0
    assert cycle_timing["candidate_readback"] == 0.0


def test_the_block_is_request_local_across_two_requests() -> None:
    """One provider serves many requests; each block counts only its own cycles."""

    provider, runner, drafter = _provider(budget=4)
    first = provider.generate_detailed(_request(8))[0]
    first_cycles = dict(first.telemetry.diagnostics or {})["specdec2_mtp2"]["cycles"]
    second = provider.generate_detailed(_request(5))[0]
    second_block = dict(second.telemetry.diagnostics or {})["specdec2_mtp2"]

    assert first_cycles > 0
    assert second_block["cycles"] < first_cycles + second_block["cycles"]
    assert second_block["mtp_output_tokens"] + second_block["ar_output_tokens_in_cycles"] + 1 == (
        len(second.generated_token_ids)
    )
    assert second_block["requested_budget"] == 4
    assert second_block["candidate_budget"] == 4
    # The provider's legacy last-cycle record is still the second request's.
    assert len(provider.last_cycles) == second_block["cycles"]
    assert drafter.calls > 0 and runner.resets == 2


def test_the_declared_row_covers_every_field_the_helper_writes() -> None:
    """The carrier declares the shared contract rather than growing attributes.

    ``record_speculative_outputs`` writes through ``setattr``; declaring the
    names here is what keeps the contract visible and what would survive a
    slots conversion of the carrier.
    """

    from dataclasses import fields

    from hipengine.generation.gemma4_mtp import _Gemma4MTPOutputAccounting
    from hipengine.speculative.accounting import (
        record_speculative_outputs,
        speculative_output_accounting,
    )

    declared = {entry.name for entry in fields(_Gemma4MTPOutputAccounting)}
    row = _Gemma4MTPOutputAccounting(
        mtp2_requested_budget=2, mtp2_candidate_budget=2
    )
    record_speculative_outputs(
        row,
        candidate_count=2,
        accepted_count=1,
        visible_count=2,
        output_position=1,
    )
    record_speculative_outputs(
        row,
        candidate_count=0,
        accepted_count=0,
        visible_count=1,
        output_position=3,
    )
    written = set(vars(row))
    assert written <= declared, written - declared
    accounting = speculative_output_accounting(row)
    assert accounting is not None
    assert accounting["cycles"] == 2
    assert accounting["mtp_output_tokens"] == 2
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert accounting["first_fallback_position"] == 3


def test_a_resident_owner_budget_is_what_the_block_reports() -> None:
    """The block reports the budget the request ran with, not the provider's.

    The legacy ``candidate_budget`` diagnostic is the provider's configured
    value; the shared block answers the per-request question, so a resident
    owner that runs a row at a narrower budget must see that budget on both of
    the block's budget keys, and the depth it actually reached must fit it.
    """

    provider, runner, _drafter = _provider(budget=4)
    request = _request(6)

    output = _drive_steps(provider, request, runner=runner, candidate_budget=2)

    diagnostics = dict(output.telemetry.diagnostics or {})
    assert diagnostics["candidate_budget"] == 4
    accounting = diagnostics["specdec2_mtp2"]
    assert accounting["requested_budget"] == 2
    # Requested and effective depth are the frame's, not the provider's.
    assert accounting["candidate_budget"] == 2
    assert accounting["selected_depth_histogram"].keys() <= {"0", "1", "2"}
    assert accounting["mtp_output_tokens"] + accounting["ar_output_tokens_in_cycles"] + 1 == (
        len(output.generated_token_ids)
    )


def test_a_deadline_before_any_work_publishes_nothing() -> None:
    provider, runner, drafter = _provider(budget=4)

    with pytest.raises(GenerationDeadlineExceeded):
        provider.generate_detailed(_request(8, deadline_at=0.0))

    assert drafter.calls == 0
    assert runner.resets == 0
    assert provider.last_cycles == ()


def test_cancellation_between_cycles_publishes_no_partial_output() -> None:
    """A cancelled request has no response, so it can publish no block.

    The cycle that committed before the cancellation stays in the provider's
    legacy ``last_cycles`` record; what must not happen is a fabricated cycle
    for work the request never published.
    """

    cancellation = None

    def _cancel_after_first_draft(call: int) -> None:
        if call == 1:
            cancellation.cancel()

    from hipengine.generation.deadline import GenerationCancellationToken

    cancellation = GenerationCancellationToken()
    provider, _runner, _drafter = _provider(
        budget=3, on_draft=_cancel_after_first_draft
    )

    with pytest.raises(GenerationCancelled):
        provider.generate_detailed(
            _request(12, cancellation_token=cancellation)
        )

    assert [cycle.committed for cycle in provider.last_cycles] == [4]
