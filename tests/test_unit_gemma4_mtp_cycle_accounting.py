"""The Gemma 4 MTP cycle's token accounting, pinned without a device.

The route had no test of its own. The only thing exercising it was an
end-to-end probe against a 26B artifact, and that probe compares the speculative
output against a plain run in the same process -- so two defects survived in the
default path:

* **The route never reset the runner.** Its prefill appends the prompt at the
  runner's current position and attends over everything already in the KV cache,
  so the output was a function of the previous request rather than of the prompt.
  The probe runs the plain route first, so the speculative route was measured on
  the plain run's own KV, and a change to the prefill attention's arithmetic was
  enough to move the first sampled token.
* **The zero-draft tail read a flat array.** ``Gemma4Runner.forward`` returns
  ``(k, vocab)`` for ``k > 1`` but a flat ``(vocab,)`` for ``k == 1``, and the
  tail cycle is exactly the one-row case. ``rows[0]`` on that array is the
  scalar logit of vocabulary entry 0, whose ``argmax`` is always 0, so every
  such cycle committed token 0 (``<pad>``) instead of the model's next token.

The fakes reproduce the two contracts that matter and nothing else: the
runner's row-count shapes, and greedy acceptance against the target's own
logits. ``_FakeRunner.forward`` is deliberately shape-faithful to
``Gemma4Runner.forward``, including the flat one-row return, so the provider is
pinned against the shape a real runner produces rather than a tidied one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from hipengine.generation.gemma4_mtp import Gemma4MTPTextProvider
from hipengine.generation.registry import GenerationRequest
from hipengine.speculative.registry import SpeculativeProviderConfig

_VOCAB = 64
_PROMPT = (5, 11, 17, 23, 29, 31, 37)
# No EOS: the fake target has one token per position and never emits a stop.
_EOS = -1


def _next_token(position: int) -> int:
    """The token the fake target produces after processing ``position`` tokens.

    Never 0, so a cycle that commits the argmax of a scalar logit (always 0) is
    visible in the output rather than hidden behind a plausible-looking id.
    """

    return 1 + (position * 7 + 3) % (_VOCAB - 1)


def _greedy_reference(prompt_length: int, max_tokens: int) -> list[int]:
    """What greedy decoding on the fake target produces, token by token."""

    return [_next_token(prompt_length - 1 + index) for index in range(max_tokens)]


class _FakeRunner:
    """A runner with ``Gemma4Runner``'s row shapes and a resettable cursor."""

    def __init__(self, position: int = 0) -> None:
        self.position = int(position)
        self.resets = 0
        self.verification_calls: list[bool] = []
        self.forwards: list[tuple[tuple[int, ...], int]] = []

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
        self.verification_calls.append(verification)
        del apply_softcap
        block = [int(token) for token in tokens]
        wanted = int(logits_rows)
        if not 1 <= wanted <= len(block):
            raise ValueError(f"logits_rows {wanted} is outside 1..{len(block)}")
        self.forwards.append((tuple(block), wanted))
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
    """Proposes what the fake target will actually do, so drafts are accepted."""

    runner: _FakeRunner
    # Index of the one draft to make wrong, so partial acceptance is covered too.
    wrong_at: int | None = None
    max_drafts: int = 4
    calls: int = 0

    def draft(self, token: int, *, hidden_row: int = -1) -> list[int]:
        del token, hidden_row
        self.calls += 1
        drafts = [
            _next_token(self.runner.position + index)
            for index in range(self.max_drafts)
        ]
        if self.wrong_at is not None and self.wrong_at < len(drafts):
            drafts[self.wrong_at] = (drafts[self.wrong_at] % (_VOCAB - 1)) + 1
        return drafts


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
    tokens: list[int]
    runner: _FakeRunner
    drafter: _FakeDrafter
    finish_reason: str


def _run_cycle(
    *,
    budget: int,
    max_tokens: int,
    incoming_position: int = 0,
    wrong_at: int | None = None,
    eos_token: int = _EOS,
    ignore_eos: bool = True,
    tokenizer_stops: tuple[int, ...] | None = None,
    request_options: dict | None = None,
) -> _Run:
    runner = _FakeRunner(position=incoming_position)
    drafter = _FakeDrafter(runner=runner, wrong_at=wrong_at)
    generator = _FakeGenerator(runner)
    generator.tokenizer.eos_token_id = eos_token
    if tokenizer_stops is not None:
        generator.tokenizer.stop_token_ids = tokenizer_stops
    provider = Gemma4MTPTextProvider(
        target_generator=generator,
        config=SpeculativeProviderConfig(
            provider="gemma4_mtp",
            draft_model="/nonexistent/sidecar.gguf",
            candidate_budget=budget,
        ),
        drafter=drafter,
    )
    request = GenerationRequest(
        prompts=(_PROMPT,),
        max_tokens=max_tokens,
        temperature=0.0,
        ignore_eos=ignore_eos,
        **({"top_p": 1.0} | (request_options or {})),
    )
    output = provider.generate_detailed(request)[0]
    return _Run(
        tokens=[int(token) for token in output.generated_token_ids],
        runner=runner,
        drafter=drafter,
        finish_reason=output.finish_details.reason,
    )


def test_the_prefill_starts_from_a_reset_runner() -> None:
    """A runner carries the previous request's position and KV.

    The AR route resets it per prompt; this route has to as well, or its prefill
    appends the prompt after whatever ran last and attends over it.
    """

    run = _run_cycle(budget=6, max_tokens=16, incoming_position=37)

    assert run.runner.resets == 1, (
        "the MTP prefill ran without resetting the runner, so it inherited the "
        "previous request's position and KV cache"
    )
    assert run.runner.forwards[0] == (tuple(_PROMPT), 1), (
        "the first forward must be this request's prompt at the reset cursor"
    )


def test_the_output_does_not_depend_on_the_runner_it_inherits() -> None:
    """The same prompt must produce the same tokens whatever ran before it."""

    clean = _run_cycle(budget=6, max_tokens=24, incoming_position=0)
    inherited = _run_cycle(budget=6, max_tokens=24, incoming_position=37)

    assert inherited.tokens == clean.tokens, (
        "the speculative output changed with the runner state it inherited; the "
        "route is reading the previous request's KV"
    )


def test_a_zero_draft_tail_commits_the_models_next_token() -> None:
    """The one-row verify must read row 0, not the scalar logit at index 0.

    Budget 1 and 8 tokens lands the last cycle with one token left, which is the
    zero-draft tail: one forwarded row, read through `rows[0]`.
    """

    run = _run_cycle(budget=1, max_tokens=8)
    row_counts = [wanted for _, wanted in run.runner.forwards]

    assert 1 in row_counts, "this case no longer reaches the zero-draft tail"
    assert run.tokens == _greedy_reference(len(_PROMPT), 8), (
        "a zero-draft cycle committed a token the target did not produce; a "
        "flat one-row logits array read through `rows[0]` argmaxes to token 0"
    )
    assert 0 not in run.tokens


def test_every_budget_stops_exactly_at_max_tokens_and_matches_greedy() -> None:
    """The route's contract: plain greedy tokens, exactly `max_tokens` of them.

    The sweep covers the budgets that reach the zero-draft tail and the ones
    that do not, at token counts on both sides of a cycle boundary.
    """

    for budget in range(1, 9):
        for max_tokens in (1, 2, 5, 6, 8, 9, 16, 17):
            run = _run_cycle(budget=budget, max_tokens=max_tokens)
            assert run.tokens == _greedy_reference(len(_PROMPT), max_tokens), (
                f"budget {budget}, max_tokens {max_tokens}: the cycle did not "
                f"reproduce greedy decoding"
            )
            assert run.finish_reason == "length"


def test_mtp_marks_verification_but_not_prompt_prefill() -> None:
    run = _run_cycle(budget=6, max_tokens=16)
    assert run.runner.verification_calls[0] is False
    assert all(run.runner.verification_calls[1:])


def test_accepted_draft_eos_stops_before_later_candidates() -> None:
    reference = _greedy_reference(len(_PROMPT), 8)
    for index in range(8):
        run = _run_cycle(budget=6, max_tokens=16,
                         eos_token=reference[index], ignore_eos=False)
        assert run.tokens == reference[:index + 1]
        assert run.finish_reason == "stop"
        assert run.runner.position == len(_PROMPT) + index


def test_ignored_draft_eos_does_not_stop_generation() -> None:
    reference = _greedy_reference(len(_PROMPT), 16)
    run = _run_cycle(budget=6, max_tokens=16,
                     eos_token=reference[2], ignore_eos=True)
    assert run.tokens == reference
    assert run.finish_reason == "length"


def test_explicit_stops_apply_even_when_eos_is_ignored() -> None:
    reference = _greedy_reference(len(_PROMPT), 8)
    run = _run_cycle(budget=6, max_tokens=16, ignore_eos=True,
                     request_options={"stop_token_ids": (reference[2],)})
    assert run.tokens == reference[:3]
    assert run.finish_reason == "stop"


def test_default_stop_set_and_eos_override_match_plain_semantics() -> None:
    reference = _greedy_reference(len(_PROMPT), 8)
    run = _run_cycle(budget=6, max_tokens=16, ignore_eos=False,
                     tokenizer_stops=(reference[2], reference[5]))
    assert run.tokens == reference[:3]
    overridden = _run_cycle(budget=6, max_tokens=16, ignore_eos=False,
                            tokenizer_stops=(reference[2],),
                            request_options={"eos_token_id": reference[5]})
    assert overridden.tokens == reference[:6]


@pytest.mark.parametrize("options", [
    {"logit_bias": {1: 1.0}}, {"suppress_token_ids": (1,)},
    {"repetition_penalty": 1.1}, {"min_tokens": 1, "eos_token_id": 1},
    {"stop_token_sequences": ((1, 2),)}, {"top_p": 0.9},
])
def test_mtp_rejects_unsupported_request_controls(options) -> None:
    with pytest.raises(NotImplementedError):
        _run_cycle(budget=2, max_tokens=8, request_options=options)


def test_a_rejected_draft_does_not_break_the_greedy_sequence() -> None:
    """Partial acceptance is the common case, and it must not move the output."""

    for wrong_at in (0, 1, 2, 3, 5):
        run = _run_cycle(budget=6, max_tokens=24, wrong_at=wrong_at)
        assert run.tokens == _greedy_reference(len(_PROMPT), 24), (
            f"a draft rejected at index {wrong_at} changed the committed tokens"
        )


def test_each_cycle_forwards_one_row_per_drafted_token_plus_the_seed() -> None:
    """The verify's row count is the acceptance loop's whole premise."""

    run = _run_cycle(budget=4, max_tokens=16)
    verifies = [entry for entry in run.runner.forwards if entry[0] != tuple(_PROMPT)]

    assert verifies, "no verify forward was recorded"
    for block, wanted in verifies:
        assert wanted == len(block), (
            "the verify asked for a row count that does not match its block; a "
            "row is the target's distribution after processing that row's token"
        )
