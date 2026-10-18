"""Gemma 4 decode controls on both the autoregressive and MTP routes.

``Gemma4GGUFGenerator._validate_request`` used to refuse multi-token stop
sequences, thinking budgets, ``min_tokens`` and logprobs, so a user request for
any of them failed instead of running. They are implemented now, and the reason
they live in this module is that two routes reach them: the plain decode loop in
``gemma4_gguf.py`` and the speculative cycle in ``gemma4_mtp.py``. A control that
runs on one route and not the other is a defect the single-route test would not
see, so every control here is pinned on both, and where the routes must agree the
test compares them against each other rather than against a hand-written
expectation.

The fakes reproduce three contracts and nothing else: ``Gemma4Runner``'s row
shapes (including the flat one-row return of ``forward``), a deterministic target
whose next token is a function of the absolute position, and a drafter that
proposes what the target will do. No device, no artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest

from hipengine.generation.engine_loop import SubmitPollTextGenerator
from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator
from hipengine.generation.gemma4_mtp import Gemma4MTPTextProvider
from hipengine.generation.registry import GenerationRequest
from hipengine.generation.sampling import RowSamplingState, reported_logprob
from hipengine.speculative.registry import SpeculativeProviderConfig

_VOCAB = 64
_PROMPT = (5, 11, 17)
#: The row's runner-up. A thinking budget's close sequence starts with it, so a
#: soft-close bias can promote it the way it promotes a real close token.
_RUNNER_UP = 2
#: The artifact's ``eos``, and a second end-of-generation id. Neither is ever the
#: target's primary token unless a test asks for it.
_EOS = 3
_ALT_EOG = 1
#: The close sequence the thinking budgets below lower their close text into.
_CLOSE = (2, 0)
#: Logits for the primary token, the runner-up, and everything else.
_PRIMARY_LOGIT = 6.0
_RUNNER_UP_LOGIT = 5.0
_FLOOR_LOGIT = -30.0


def _token_at(index: int) -> int:
    """The token the target emits at absolute position ``index``."""

    return 4 + (index * 7 + 1) % 50


#: Absolute position of the first generated token: the prompt occupies 0..n-1.
_FIRST = len(_PROMPT)


def _gen(index: int) -> int:
    """The token the target emits at generated position ``index``."""

    return _token_at(_FIRST + index)


class _Target:
    """A deterministic target: one primary token per absolute position."""

    def __init__(self, overrides: dict[int, int] | None = None) -> None:
        self.overrides = dict(overrides or {})

    def primary(self, index: int) -> int:
        return int(self.overrides.get(int(index), _token_at(index)))

    def row(self, position: int) -> np.ndarray:
        """The row produced after processing absolute position ``position``."""

        row = np.full((_VOCAB,), _FLOOR_LOGIT, dtype=np.float32)
        row[self.primary(int(position) + 1)] = _PRIMARY_LOGIT
        row[_RUNNER_UP] = _RUNNER_UP_LOGIT
        return row


class _FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = _EOS
        self.stop_token_ids = (_EOS,)

    @staticmethod
    def encode(text: str) -> list[int]:
        del text
        return list(_PROMPT)

    @staticmethod
    def decode(token_ids, skip_special: bool = False) -> str:
        del skip_special
        return " ".join(str(int(token)) for token in token_ids)


class _ARRunner:
    """``Gemma4Runner``'s row shapes and a resettable cursor."""

    kv_storage_resolved = "bf16"
    uses_int8_kv = False

    def __init__(self, target: _Target) -> None:
        self.target = target
        self.position = 0
        self.resets = 0
        self.forwards: list[tuple[int, ...]] = []

    def reset(self) -> None:
        self.resets += 1
        self.position = 0

    def forward(self, tokens, **kwargs) -> np.ndarray:
        del kwargs
        block = [int(token) for token in tokens]
        self.forwards.append(tuple(block))
        self.position += len(block)
        return self.target.row(self.position - 1)

    def forward_argmax(self, tokens) -> int:
        return int(np.argmax(self.forward(tokens)))


class _MTPRunner(_ARRunner):
    """The same cursor, with ``forward``'s verified row matrix."""

    def __init__(self, target: _Target) -> None:
        super().__init__(target)
        self.verification_calls: list[bool] = []

    def rewind(self, position: int) -> None:
        self.position = int(position)

    def forward(self, tokens, *, logits_rows: int = 1, verification: bool = False, **kwargs):
        del kwargs
        self.verification_calls.append(verification)
        block = [int(token) for token in tokens]
        wanted = int(logits_rows)
        if not 1 <= wanted <= len(block):
            raise ValueError(f"logits_rows {wanted} is outside 1..{len(block)}")
        self.forwards.append(tuple(block))
        trailing = np.stack(
            [self.target.row(self.position + index) for index in range(len(block))][-wanted:]
        )
        self.position += len(block)
        return trailing if wanted > 1 else trailing[0]


@dataclass
class _Drafter:
    """Proposes the target's own next tokens, so a draft is a real guess."""

    runner: _MTPRunner
    max_drafts: int = 4
    calls: int = 0

    def draft(self, token: int, *, hidden_row: int = -1, select=None) -> list[int]:
        del token, hidden_row
        self.calls += 1
        drafts: list[int] = []
        for index in range(self.max_drafts):
            row = self.runner.target.row(self.runner.position + index)
            drafts.append(int(select(row)) if select is not None else int(np.argmax(row)))
        return drafts


class _FakeGenerator:
    """The target generator surface the MTP provider reaches."""

    def __init__(self, runner: _MTPRunner, tokenizer: _FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self._runner = runner

    def _ensure_runner(self) -> _MTPRunner:
        return self._runner


def _ar_generator(runner: _ARRunner, tokenizer: _FakeTokenizer) -> Gemma4GGUFGenerator:
    """A real generator over a fake runner, so the real decode loop runs."""

    generator = Gemma4GGUFGenerator(
        model_path="/nonexistent/gemma4.gguf",
        weight_index=None,
        model_plugin=None,
        backend="hip_gfx1100",
    )
    generator._runner = runner
    generator._tokenizer = tokenizer
    return generator


@pytest.fixture(autouse=True)
def _launched_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the whole-step decode graph out of a CPU test: it needs a device."""

    monkeypatch.setenv("HIPENGINE_GEMMA4_DECODE_GRAPH", "0")


def _request(*, max_tokens: int = 12, **options) -> GenerationRequest:
    values = {
        "prompts": (_PROMPT,),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
    }
    values.update(options)
    return GenerationRequest(**values)


def _budget_request(*, cap: int, window: int = 0, max_tokens: int = 12, **options):
    return _request(
        max_tokens=max_tokens,
        thinking_close_token_ids=_CLOSE,
        thinking_hard_token_cap=cap,
        thinking_soft_close_window=window,
        **options,
    )


def _run_ar(request: GenerationRequest, target: _Target, tokenizer: _FakeTokenizer | None = None):
    runner = _ARRunner(target)
    generator = _ar_generator(runner, tokenizer or _FakeTokenizer())
    output = generator.generate_detailed(request)[0]
    return output, runner


def _run_mtp(
    request: GenerationRequest,
    target: _Target,
    *,
    candidate_budget: int = 4,
    tokenizer: _FakeTokenizer | None = None,
):
    runner = _MTPRunner(target)
    tokenizer = tokenizer or _FakeTokenizer()
    provider = Gemma4MTPTextProvider(
        target_generator=_FakeGenerator(runner, tokenizer),
        config=SpeculativeProviderConfig(
            provider="gemma4_mtp",
            draft_model="/nonexistent/sidecar.gguf",
            candidate_budget=candidate_budget,
        ),
        drafter=_Drafter(runner),
    )
    output = provider.generate_detailed(request)[0]
    return output, runner, provider


def _tokens(output) -> list[int]:
    return [int(token) for token in output.generated_token_ids]


@pytest.mark.parametrize("route", ["ar", "mtp"])
@pytest.mark.parametrize("position", [0, 1, 3, 5])
def test_explicit_single_stop_is_hidden_and_reported_as_sequence(route, position):
    request = _request(stop_token_ids=(_gen(position),))
    output = (_run_ar(request, _Target()) if route == "ar"
              else _run_mtp(request, _Target()))[0]
    assert _tokens(output) == [_gen(i) for i in range(position + 1)]
    assert output.text == _FakeTokenizer.decode([_gen(i) for i in range(position)])
    assert output.finish_details.stop_sequence == (_gen(position),)
    assert output.finish_details.eos_token_id is None
    assert output.finish_details.reason == "stop"


# --- thinking budgets ---------------------------------------------------


def test_ar_emits_the_close_sequence_when_the_hard_cap_is_reached() -> None:
    """The cap ends the reasoning phase by emitting the lowered close text.

    The row never offers a close token, so the two close tokens can only come
    from the budget's forced queue.
    """

    output, _runner = _run_ar(_budget_request(cap=3), _Target())

    # Three reasoning tokens, then the close sequence the budget queued. The
    # close tokens occupy generated positions 3 and 4, so the answer phase
    # resumes the target's own sequence at position 5 -- the model was fed the
    # close tokens like any other emitted token.
    assert _tokens(output) == [
        _gen(0),
        _gen(1),
        _gen(2),
        *_CLOSE,
        *(_gen(index) for index in range(5, 12)),
    ]
    assert output.finish_details.reason == "length"


def test_ar_hard_close_reports_the_thinking_phases() -> None:
    """Finish metadata carries the reasoning/answer split the budget tracked."""

    output, _runner = _run_ar(_budget_request(cap=3), _Target())

    details = output.finish_details
    assert details.forced_close is True
    assert details.reasoning_tokens == 5  # three reasoning tokens plus the close
    assert details.answer_tokens == len(_tokens(output)) - 5
    assert details.budget_pressure == "hard_close"


def test_ar_suppresses_every_end_of_generation_id_while_reasoning() -> None:
    """A budget holds back the model's whole stop set, not just ``request.eos``.

    The target's first token is the artifact's second end-of-generation id. If
    only ``request.eos_token_id`` were masked, that token would be emitted and
    the request would stop on its first step, which is the failure ``min_tokens``
    and a thinking budget both exist to prevent.
    """

    tokenizer = _FakeTokenizer()
    tokenizer.stop_token_ids = (_EOS, _ALT_EOG)
    output, _runner = _run_ar(
        _budget_request(cap=20, max_tokens=4), _Target({_FIRST: _ALT_EOG}), tokenizer
    )

    assert _ALT_EOG not in _tokens(output)
    assert _tokens(output)[0] == _RUNNER_UP


def test_ar_soft_close_window_closes_before_the_hard_cap() -> None:
    """A soft window biases the first close token until the model takes it."""

    soft, _runner = _run_ar(_budget_request(cap=4, window=4), _Target())
    hard, _runner = _run_ar(_budget_request(cap=4, window=0), _Target())

    # The window starts at zero reasoning tokens, so the bias is already enough
    # to promote the runner-up close token on the first step.
    assert _tokens(soft)[:2] == list(_CLOSE)
    assert soft.finish_details.reasoning_tokens < 4
    # Without the window the same target reasons to the cap first.
    assert _tokens(hard)[:2] != list(_CLOSE)
    assert _tokens(hard)[4:6] == list(_CLOSE)


@pytest.mark.parametrize("cap,window", [(3, 0), (4, 4), (2, 2), (6, 0)])
def test_mtp_reproduces_the_autoregressive_route_under_a_thinking_budget(cap, window):
    """The speculative cycle must close the reasoning phase exactly as AR does."""

    request = _budget_request(cap=cap, window=window, max_tokens=14)
    ar_output, _runner = _run_ar(request, _Target())
    mtp_output, _runner, provider = _run_mtp(request, _Target())

    assert _tokens(mtp_output) == _tokens(ar_output)
    assert provider.last_cycles, "the MTP route produced no cycle"
    assert mtp_output.finish_details.reasoning_tokens == ar_output.finish_details.reasoning_tokens


# --- min_tokens ---------------------------------------------------------


def test_ar_min_tokens_holds_eos_back_until_the_floor_is_reached() -> None:
    """``min_tokens`` masks the stop ids, so the reply cannot end early."""

    target = _Target({_FIRST + index: _EOS for index in range(8)})
    output, _runner = _run_ar(
        _request(max_tokens=8, min_tokens=4, eos_token_id=_EOS), target
    )

    tokens = _tokens(output)
    assert _EOS not in tokens[:4]
    assert tokens[:4] == [_RUNNER_UP] * 4
    assert tokens[4] == _EOS
    assert output.finish_details.reason == "stop"


def test_ar_min_tokens_still_masks_eos_when_eos_is_ignored() -> None:
    """The floor is a property of the token, not of the stop rule.

    ``min_tokens`` masks the end-of-generation ids for the first N steps whether
    or not the request would stop on them, which is the shared host sampler's
    rule too: the floor asks the model for N tokens, so it holds the ids back
    rather than relying on a stop that may be disabled.
    """

    target = _Target({_FIRST + index: _EOS for index in range(8)})
    output, _runner = _run_ar(
        _request(max_tokens=6, min_tokens=4, eos_token_id=_EOS, ignore_eos=True),
        target,
    )

    tokens = _tokens(output)
    assert tokens[:4] == [_RUNNER_UP] * 4
    assert tokens[4:] == [_EOS, _EOS]
    assert output.finish_details.reason == "length"


@pytest.mark.parametrize("min_tokens", [0, 1, 4, 6])
def test_mtp_reproduces_the_autoregressive_route_under_min_tokens(min_tokens):
    target = _Target({_FIRST + index: _EOS for index in range(8)})
    request = _request(
        max_tokens=8, min_tokens=min_tokens, eos_token_id=_EOS
    )
    ar_output, _runner = _run_ar(request, target)
    mtp_output, _runner, _provider = _run_mtp(request, target)

    assert _tokens(mtp_output) == _tokens(ar_output)


# --- multi-token stop sequences -----------------------------------------


def _stop_pair() -> tuple[int, int]:
    """Two consecutive target tokens, neither of them a configured stop id."""

    pair = (_gen(2), _gen(3))
    assert not set(pair) & {_EOS, _ALT_EOG, _RUNNER_UP}
    return pair


def test_ar_truncates_a_multi_token_stop_sequence_from_the_reply() -> None:
    pair = _stop_pair()
    output, _runner = _run_ar(
        _request(max_tokens=10, stop_token_sequences=(pair,)), _Target()
    )

    tokens = _tokens(output)
    assert tokens == [_gen(0), _gen(1), *pair]
    assert output.finish_details.reason == "stop"
    assert output.finish_details.stop_sequence == pair
    # The sequence is matched, not shown: the emitted ids keep it, the text does
    # not, which is how the shared stop-sequence handling reports it.
    assert output.text == _FakeTokenizer.decode(tokens[:2])


def test_ar_stop_sequence_does_not_match_a_prefix_only() -> None:
    pair = _stop_pair()
    assert _gen(3) != _RUNNER_UP
    output, _runner = _run_ar(
        _request(max_tokens=6, stop_token_sequences=(pair,)),
        _Target({_FIRST + 3: _RUNNER_UP}),
    )

    assert output.finish_details.reason == "length"
    assert _tokens(output) == [
        _gen(0), _gen(1), _gen(2), _RUNNER_UP, _gen(4), _gen(5)
    ]


def test_mtp_truncates_a_stop_sequence_that_straddles_a_cycle_boundary() -> None:
    """A match can complete on the first token of the next speculative cycle.

    The first cycle commits its whole draft plus the bonus token, so the pair is
    chosen to sit across that boundary. The truncation must land on the second
    half of the pair rather than on the cycle.
    """

    boundary = 1 + 5  # the prefill token, then one full cycle of five commits
    pair = (_gen(boundary - 1), _gen(boundary))
    request = _request(max_tokens=20, stop_token_sequences=(pair,))
    mtp_output, _runner, provider = _run_mtp(request, _Target(), candidate_budget=4)

    tokens = _tokens(mtp_output)
    assert tokens[-2:] == list(pair)
    assert len(tokens) == boundary + 1
    assert mtp_output.finish_details.reason == "stop"
    assert mtp_output.finish_details.stop_sequence == pair
    assert mtp_output.text == _FakeTokenizer.decode(tokens[:-2])
    assert len(provider.last_cycles) >= 2, "the match did not cross a cycle boundary"
    # The second cycle resumes at the runner cursor after the first cycle's
    # commits, which is what makes the match cross the boundary rather than sit
    # inside one group.
    assert provider.last_cycles[0].committed == 5
    assert provider.last_cycles[1].start_position == len(_PROMPT) + boundary - 1


def test_mtp_truncates_a_stop_sequence_inside_one_committed_group() -> None:
    """A stop can complete mid-group; the tokens after it are not published."""

    pair = (_gen(1), _gen(2))
    request = _request(max_tokens=20, stop_token_sequences=(pair,))
    ar_output, _runner = _run_ar(request, _Target())
    mtp_output, _runner, provider = _run_mtp(request, _Target(), candidate_budget=6)

    assert _tokens(mtp_output) == _tokens(ar_output)
    assert _tokens(mtp_output) == [_gen(0), *pair]
    # The cycle proposed six drafts and could have committed seven tokens; the
    # stop cut it to the two positions before and at the match.
    assert len(provider.last_cycles[0].candidates) == 6
    assert provider.last_cycles[0].committed == 2


# --- logprobs -----------------------------------------------------------


def _row_before(output_tokens: list[int], index: int, target: _Target) -> np.ndarray:
    """The row that predicted ``output_tokens[index]``."""

    del target
    return _Target().row(len(_PROMPT) - 1 + index)


def test_ar_reports_a_real_logprob_and_top_logprobs_for_every_token() -> None:
    """The value is the row's own probability, not a placeholder."""

    output, _runner = _run_ar(
        _request(max_tokens=5, logprobs=True, top_logprobs=2), _Target()
    )

    tokens = _tokens(output)
    assert len(output.token_logprobs) == len(tokens)
    for index, (token_id, entry) in enumerate(zip(tokens, output.token_logprobs)):
        row = _row_before(tokens, index, _Target()).astype(np.float64)
        probabilities = np.exp(row - row.max())
        probabilities /= probabilities.sum()
        assert entry.token_id == token_id
        assert entry.token_text == str(token_id)
        assert entry.logprob == pytest.approx(
            float(math.log(probabilities[token_id])), rel=1e-6, abs=1e-9
        )
        assert len(entry.top_logprobs) == 2
        assert entry.top_logprobs[0][0] == token_id
        assert entry.top_logprobs[0][2] == pytest.approx(entry.logprob, rel=1e-9)
        assert entry.top_logprobs[0][2] >= entry.top_logprobs[1][2]


def test_ar_logprobs_match_the_shared_host_sampler() -> None:
    """The metadata is the shared route's, on the same history and row."""

    output, _runner = _run_ar(
        _request(max_tokens=4, temperature=0.8, top_k=8, logprobs=True, top_logprobs=3),
        _Target(),
    )
    tokens = _tokens(output)
    state = RowSamplingState(prompt_tokens=_PROMPT, seed=0)
    for index, (token_id, entry) in enumerate(zip(tokens, output.token_logprobs)):
        row = _row_before(tokens, index, _Target())
        want_logprob, want_pairs = reported_logprob(
            row,
            _request(max_tokens=4, temperature=0.8, top_k=8, logprobs=True, top_logprobs=3),
            state,
            token_id,
        )
        assert entry.logprob == pytest.approx(want_logprob, rel=1e-9)
        assert [pair[0] for pair in entry.top_logprobs] == [pair[0] for pair in want_pairs]
        for got, want in zip(entry.top_logprobs, want_pairs):
            assert got[2] == pytest.approx(want[1], rel=1e-9)
        state.observe(token_id)


def test_ar_greedy_without_logprobs_reports_none() -> None:
    output, _runner = _run_ar(_request(max_tokens=3), _Target())

    assert output.token_logprobs == ()


def test_mtp_reports_logprobs_for_every_published_token() -> None:
    """Each committed token's metadata comes from the row that predicted it."""

    request = _request(max_tokens=9, logprobs=True, top_logprobs=2)
    ar_output, _runner = _run_ar(request, _Target())
    mtp_output, _runner, _provider = _run_mtp(request, _Target())

    tokens = _tokens(mtp_output)
    assert tokens == _tokens(ar_output)
    assert len(mtp_output.token_logprobs) == len(tokens)
    for index, (token_id, entry) in enumerate(zip(tokens, mtp_output.token_logprobs)):
        row = _row_before(tokens, index, _Target()).astype(np.float64)
        probabilities = np.exp(row - row.max())
        probabilities /= probabilities.sum()
        assert entry.token_id == token_id
        assert entry.logprob == pytest.approx(
            float(math.log(probabilities[token_id])), rel=1e-6, abs=1e-9
        )
        assert len(entry.top_logprobs) == 2


def test_mtp_sampled_logprobs_match_the_shared_host_sampler() -> None:
    """A sampled cycle reports each committed token's own law, not a proxy.

    The two routes agree token for token only for a greedy request: exact
    speculative sampling makes a sampled token distributed as the target's law,
    not identical to the draw the autoregressive route happens to make. So the
    metadata is compared against the shared host sampler on this route's own
    history and its own rows.
    """

    request = _request(
        max_tokens=8, temperature=0.7, top_k=6, logprobs=True, top_logprobs=3
    )
    mtp_output, _runner, _provider = _run_mtp(request, _Target())

    tokens = _tokens(mtp_output)
    assert [entry.token_id for entry in mtp_output.token_logprobs] == tokens
    state = RowSamplingState(prompt_tokens=_PROMPT, seed=0)
    for index, entry in enumerate(mtp_output.token_logprobs):
        want_logprob, want_pairs = reported_logprob(
            _row_before(tokens, index, _Target()), request, state, entry.token_id
        )
        assert entry.logprob is not None
        assert entry.logprob <= 0.0
        assert entry.logprob == pytest.approx(want_logprob, rel=1e-9)
        assert [pair[0] for pair in entry.top_logprobs] == [pair[0] for pair in want_pairs]
        for got, want in zip(entry.top_logprobs, want_pairs):
            assert got[2] == pytest.approx(want[1], rel=1e-9)
        state.observe(entry.token_id)


def test_mtp_logprobs_stop_at_the_published_prefix() -> None:
    """A truncated stop sequence publishes nothing after it, so it reports nothing."""

    pair = _stop_pair()
    request = _request(max_tokens=10, stop_token_sequences=(pair,), logprobs=True)
    mtp_output, _runner, _provider = _run_mtp(request, _Target())

    assert len(mtp_output.token_logprobs) == len(_tokens(mtp_output)) - 2
    assert [entry.token_id for entry in mtp_output.token_logprobs] == _tokens(mtp_output)[:-2]


# --- unsupported controls ----------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {"forced_tokens_pending": (4, 5)},
        {"json_object_close_forcing": True},
        {"force_sequence_completion_token_sequences": ((4, 5),)},
    ],
)
def test_unsupported_controls_are_refused_before_any_mutation(options) -> None:
    """A control this route cannot execute must not reach the runner.

    The request is refused by ``validate_request`` -- the same check
    ``generate_detailed`` runs first -- so the runner is never reset and no
    forward pass is issued for a request that cannot run.
    """

    request = _request(max_tokens=4, **options)
    runner = _ARRunner(_Target())
    generator = _ar_generator(runner, _FakeTokenizer())

    with pytest.raises(NotImplementedError):
        generator.validate_request(request)
    with pytest.raises(NotImplementedError):
        generator.generate_detailed(request)
    assert runner.forwards == []
    assert runner.resets == 0


@pytest.mark.parametrize(
    "options",
    [
        {"forced_tokens_pending": (4, 5)},
        {"json_object_close_forcing": True},
    ],
)
def test_mtp_refuses_unsupported_controls_before_any_mutation(options) -> None:
    request = _request(max_tokens=4, **options)
    runner = _MTPRunner(_Target())
    provider = Gemma4MTPTextProvider(
        target_generator=_FakeGenerator(runner, _FakeTokenizer()),
        config=SpeculativeProviderConfig(
            provider="gemma4_mtp",
            draft_model="/nonexistent/sidecar.gguf",
            candidate_budget=2,
        ),
        drafter=_Drafter(runner),
    )

    with pytest.raises(NotImplementedError):
        provider.generate_detailed(request)
    assert runner.forwards == []
    assert runner.resets == 0


def test_validate_request_accepts_the_controls_this_route_implements() -> None:
    """The pre-admission hook must not refuse what the route can execute."""

    generator = _ar_generator(_ARRunner(_Target()), _FakeTokenizer())

    for options in (
        {},
        {"stop_token_sequences": ((4, 5),)},
        {"min_tokens": 2, "eos_token_id": _EOS},
        {"logprobs": True, "top_logprobs": 2},
        {"thinking_close_token_ids": _CLOSE, "thinking_hard_token_cap": 4},
        {
            "thinking_close_token_ids": _CLOSE,
            "thinking_hard_token_cap": 4,
            "thinking_soft_close_window": 2,
        },
    ):
        generator.validate_request(_request(max_tokens=4, **options))


def test_the_generator_publishes_a_single_request_mtp_width() -> None:
    """The provider commits one cycle for one prompt, so the route is width one."""

    generator = _ar_generator(_ARRunner(_Target()), _FakeTokenizer())

    assert generator.server_mtp_batch_max_active_requests == 1


# --- min_tokens and custom stops ---------------------------------------
#
# ``min_tokens`` is an end-of-generation floor, not a general "do not stop yet"
# rule. The shared finish decision says so in as many words: ``chain_finish_index``
# in ``hipengine/speculative/streaming.py`` documents its ``min_tokens`` as "the
# EOS-suppression floor" that "does not delay a stop token id or a stop
# sequence", and the shared host sampler masks only ``eos_token_id``. Both Gemma
# 4 routes therefore have to honour a custom stop on either side of the floor,
# and a route that held one back would publish a reply the shared route never
# would. These pin both sides on both routes.


def _custom_stop_id() -> int:
    """A token id no step of the target's own sequence produces early.

    Using a token the target would emit anyway would let the reply stop on the
    natural sequence rather than on the configured stop, which is a weaker test.
    """

    stop_id = _gen(9)
    assert stop_id not in {_EOS, _ALT_EOG, _RUNNER_UP, *_CLOSE}
    assert stop_id not in {_gen(index) for index in range(8)}
    return stop_id


def test_ar_custom_stop_token_id_is_not_delayed_by_the_floor() -> None:
    """The floor is an EOS rule: a custom stop id still ends the reply below it."""

    stop_id = _custom_stop_id()
    output, _runner = _run_ar(
        _request(
            max_tokens=8,
            min_tokens=5,
            eos_token_id=_EOS,
            stop_token_ids=(stop_id,),
        ),
        _Target({_FIRST + 1: stop_id}),
    )

    assert _tokens(output) == [_gen(0), stop_id]
    assert output.finish_details.reason == "stop"
    assert output.finish_details.stop_sequence == (stop_id,)
    assert output.finish_details.eos_token_id is None


def test_ar_custom_stop_sequence_is_not_delayed_by_the_floor() -> None:
    """A stop sequence is matched wherever it completes, floor or no floor."""

    pair = _stop_pair()
    output, _runner = _run_ar(
        _request(max_tokens=8, min_tokens=5, eos_token_id=_EOS, stop_token_sequences=(pair,)),
        _Target(),
    )

    assert _tokens(output) == [_gen(0), _gen(1), *pair]
    assert output.finish_details.reason == "stop"
    assert output.finish_details.stop_sequence == pair
    assert output.text == _FakeTokenizer.decode(_tokens(output)[:-2])


def test_ar_custom_stop_token_id_still_ends_the_reply_after_the_floor() -> None:
    """The floor delays nothing, so it must not disable the stop either."""

    stop_id = _custom_stop_id()
    output, _runner = _run_ar(
        _request(
            max_tokens=8,
            min_tokens=2,
            eos_token_id=_EOS,
            stop_token_ids=(stop_id,),
        ),
        _Target({_FIRST + 4: stop_id}),
    )

    assert _tokens(output) == [*(_gen(index) for index in range(4)), stop_id]
    assert output.finish_details.reason == "stop"


def test_ar_the_floor_holds_back_the_model_eos_but_not_a_custom_stop() -> None:
    """The two rules are different, and one request can show both.

    The model's own end-of-generation id is masked below the floor, so the
    request survives it; the custom stop id on the next step still ends the
    reply, because the floor never applied to it.
    """

    stop_id = _custom_stop_id()
    output, _runner = _run_ar(
        _request(
            max_tokens=8,
            min_tokens=4,
            eos_token_id=_EOS,
            stop_token_ids=(stop_id,),
        ),
        _Target({_FIRST: _EOS, _FIRST + 1: stop_id}),
    )

    assert _tokens(output) == [_RUNNER_UP, stop_id]
    assert output.finish_details.reason == "stop"
    assert output.finish_details.stop_sequence == (stop_id,)
    assert output.finish_details.eos_token_id is None


@pytest.mark.parametrize("min_tokens", [0, 2, 5])
def test_mtp_matches_the_autoregressive_route_for_a_custom_stop_id(min_tokens):
    stop_id = _custom_stop_id()
    target = _Target({_FIRST + 1: stop_id})
    request = _request(
        max_tokens=8,
        min_tokens=min_tokens,
        eos_token_id=_EOS,
        stop_token_ids=(stop_id,),
    )
    ar_output, _runner = _run_ar(request, target)
    mtp_output, _runner, _provider = _run_mtp(request, target)

    assert _tokens(mtp_output) == _tokens(ar_output) == [_gen(0), stop_id]
    assert mtp_output.finish_details.reason == ar_output.finish_details.reason == "stop"
    assert mtp_output.text == ar_output.text


@pytest.mark.parametrize("min_tokens", [0, 2, 5])
def test_mtp_matches_the_autoregressive_route_for_a_custom_stop_sequence(min_tokens):
    pair = _stop_pair()
    request = _request(
        max_tokens=8, min_tokens=min_tokens, eos_token_id=_EOS, stop_token_sequences=(pair,)
    )
    ar_output, _runner = _run_ar(request, _Target())
    mtp_output, _runner, _provider = _run_mtp(request, _Target())

    assert _tokens(mtp_output) == _tokens(ar_output) == [_gen(0), _gen(1), *pair]
    assert mtp_output.finish_details.stop_sequence == ar_output.finish_details.stop_sequence == pair
    assert mtp_output.text == ar_output.text


# --- the thinking state a stopped cycle reports -------------------------


def test_mtp_reports_only_the_published_thinking_state_after_a_stop() -> None:
    """A stop inside one group must not report tokens the reply never carried.

    The cycle observes every accepted draft and its bonus token to find where
    the stop landed, so a naive finish reads the discarded tail as emitted
    output. Here the group proposes eight drafts and stops on the fourth
    published token; observing the rest would reach the four-token cap, force
    the close sequence, and report an answer phase the reply never entered.
    """

    pair = _stop_pair()
    request = _budget_request(
        cap=4, window=0, max_tokens=20, eos_token_id=_EOS, stop_token_sequences=(pair,)
    )
    ar_output, _runner = _run_ar(request, _Target())
    mtp_output, _runner, provider = _run_mtp(request, _Target(), candidate_budget=8)

    assert _tokens(mtp_output) == _tokens(ar_output) == [_gen(0), _gen(1), *pair]
    details = mtp_output.finish_details
    assert details.reason == "stop"
    assert details.stop_sequence == pair
    # Every published token is in the reasoning phase, which is what the
    # autoregressive route reports for the same reply. (The autoregressive route
    # labels ``phase`` "decode" for every request, so the comparison is on the
    # counts that carry the split.)
    assert details.reasoning_tokens == ar_output.finish_details.reasoning_tokens == 4
    assert details.answer_tokens == ar_output.finish_details.answer_tokens == 0
    assert details.phase == "reasoning"
    # Four published reasoning tokens reach the four-token cap, so the pressure is
    # real; what the discarded tail would have added is the queued close, the
    # ``closing_think`` phase and five answer tokens.
    assert details.budget_pressure == ar_output.finish_details.budget_pressure == "hard_close"
    assert details.forced_close == ar_output.finish_details.forced_close is False
    # The cycle really did carry the surplus: the truncation is what removed it.
    assert len(provider.last_cycles) == 1
    assert len(provider.last_cycles[0].candidates) == 8
    assert provider.last_cycles[0].committed == 3


def test_mtp_reports_only_the_published_thinking_state_after_a_stop_token_id() -> None:
    stop_id = _custom_stop_id()
    request = _budget_request(
        cap=4, window=0, max_tokens=20, eos_token_id=_EOS, stop_token_ids=(stop_id,)
    )
    ar_output, _runner = _run_ar(request, _Target({_FIRST + 2: stop_id}))
    mtp_output, _runner, provider = _run_mtp(
        request, _Target({_FIRST + 2: stop_id}), candidate_budget=8
    )

    assert _tokens(mtp_output) == _tokens(ar_output) == [_gen(0), _gen(1), stop_id]
    details = mtp_output.finish_details
    assert details.reasoning_tokens == ar_output.finish_details.reasoning_tokens == 3
    assert details.answer_tokens == ar_output.finish_details.answer_tokens == 0
    assert details.phase == "reasoning"
    assert details.budget_pressure == ar_output.finish_details.budget_pressure is None
    assert len(provider.last_cycles) == 1
    assert provider.last_cycles[0].committed == 2


def test_mtp_leaves_no_state_behind_for_a_stopped_cycle() -> None:
    """The sampler's own accounting ends at the published prefix.

    ``reasoning_tokens + answer_tokens`` is not the test: the budget splits a
    close sequence across the two phases. The test is that the state the next
    step would have read -- the emitted count and the stop matcher -- is the
    published one, which the same request on the autoregressive route defines.
    """

    pair = _stop_pair()
    request = _budget_request(
        cap=4, window=0, max_tokens=20, eos_token_id=_EOS, stop_token_sequences=(pair,)
    )
    mtp_output, _runner, _provider = _run_mtp(request, _Target(), candidate_budget=8)

    details = mtp_output.finish_details
    assert details.reasoning_tokens + details.answer_tokens == len(_tokens(mtp_output))
    assert details.stop_sequence == pair


# --- a request that asks for no tokens ---------------------------------


def test_ar_max_tokens_zero_does_no_device_work() -> None:
    """A zero-token request must not prefill, select, or load weights.

    The MTP route answers it before it builds a runner; a route that reset the
    runner and ran a prefill to produce nothing would charge the user for a
    forward pass per prompt and make the two routes disagree about what a
    zero-token request costs.
    """

    runner = _ARRunner(_Target())
    generator = _ar_generator(runner, _FakeTokenizer())

    output = generator.generate_detailed(_request(max_tokens=0))[0]

    assert output.text == ""
    assert output.generated_token_ids == ()
    assert output.token_logprobs == ()
    assert output.finish_details.reason == "length"
    assert output.finish_details.length_limit == 0
    assert runner.resets == 0
    assert runner.forwards == []
    assert generator.last_generation_outputs == (output,)


def test_both_routes_answer_a_zero_token_request_the_same_way() -> None:
    request = _request(max_tokens=0)
    ar_runner = _ARRunner(_Target())
    ar_output = _ar_generator(ar_runner, _FakeTokenizer()).generate_detailed(request)[0]

    mtp_runner = _MTPRunner(_Target())
    drafter = _Drafter(mtp_runner)
    provider = Gemma4MTPTextProvider(
        target_generator=_FakeGenerator(mtp_runner, _FakeTokenizer()),
        config=SpeculativeProviderConfig(
            provider="gemma4_mtp",
            draft_model="/nonexistent/sidecar.gguf",
            candidate_budget=4,
        ),
        drafter=drafter,
    )
    mtp_output = provider.generate_detailed(request)[0]

    assert ar_output.text == mtp_output.text == ""
    assert ar_output.generated_token_ids == mtp_output.generated_token_ids == ()
    assert ar_output.token_logprobs == mtp_output.token_logprobs == ()
    assert ar_output.finish_details.reason == mtp_output.finish_details.reason == "length"
    assert ar_output.finish_details.length_limit == mtp_output.finish_details.length_limit == 0
    # Neither route touched the device, and the assistant head was never built.
    assert (ar_runner.resets, ar_runner.forwards) == (0, [])
    assert (mtp_runner.resets, mtp_runner.forwards) == (0, [])
    assert drafter.calls == 0


# --- streaming ----------------------------------------------------------


def test_ar_stream_forwards_the_published_logprobs() -> None:
    """The chunk a streamed request receives carries its token logprobs.

    ``supports_stream_logprobs`` is what the server reads to serve a streamed
    ``logprobs`` request from the live stream instead of a completed batch, so
    the flag is only honest if the chunk the stream yields carries them. The
    autoregressive generator has no stream method of its own: the engine loop
    builds the chunk from ``generate_detailed``, which is the surface tested here.
    """

    request = _request(max_tokens=5, logprobs=True, top_logprobs=2)
    blocking, _runner = _run_ar(request, _Target())
    runner = _ARRunner(_Target())
    generator = _ar_generator(runner, _FakeTokenizer())
    loop = SubmitPollTextGenerator(generator)

    chunks = list(loop.stream_detailed(request))

    assert len(chunks) == 1
    chunk = chunks[0]
    # The streamed reply is the blocking reply: same text, same published ids,
    # same per-token logprobs, same finish metadata.
    assert chunk.text == blocking.text
    assert [entry.token_id for entry in chunk.token_logprobs] == _tokens(blocking)
    assert [entry.logprob for entry in chunk.token_logprobs] == [
        entry.logprob for entry in blocking.token_logprobs
    ]
    assert chunk.generated_token_ids == blocking.generated_token_ids
    assert chunk.finish_details == blocking.finish_details
    # One generation, not one per streamed tick.
    assert len(runner.forwards) == 5
    # The server reads the flag off the text generator, so it has to be visible
    # through the engine loop wrapper the engine holds.
    assert generator.supports_stream_logprobs is True
    assert getattr(loop, "supports_stream_logprobs") is True


def test_ar_stream_reports_no_logprobs_when_none_were_requested() -> None:
    """A greedy request without logprobs streams text and nothing else."""

    generator = _ar_generator(_ARRunner(_Target()), _FakeTokenizer())
    loop = SubmitPollTextGenerator(generator)

    chunks = list(loop.stream_detailed(_request(max_tokens=3)))

    assert len(chunks) == 1
    assert chunks[0].text == generator.last_generation_outputs[0].text
    assert chunks[0].token_logprobs == ()


def test_mtp_stops_when_the_prefill_token_completes_a_sequence():
    request = _request(max_tokens=8, stop_token_sequences=((_gen(0),),))
    ar, _ = _run_ar(request, _Target())
    mtp, _, provider = _run_mtp(request, _Target())
    assert mtp.generated_token_ids == ar.generated_token_ids == (_gen(0),)
    assert mtp.text == ar.text == ""
    assert mtp.finish_details.stop_sequence == ar.finish_details.stop_sequence
    assert provider.last_cycles == ()


@pytest.mark.parametrize("cap", [1, 2, 4])
def test_mtp_stop_during_forced_close_preserves_the_published_queue(cap):
    request = _budget_request(cap=cap, max_tokens=20, stop_token_ids=(_CLOSE[0],))
    ar, _ = _run_ar(request, _Target())
    mtp, _, _ = _run_mtp(request, _Target(), candidate_budget=4)
    assert mtp.generated_token_ids == ar.generated_token_ids
    assert mtp.finish_details.reason == ar.finish_details.reason == "stop"
    assert mtp.finish_details.reasoning_tokens == ar.finish_details.reasoning_tokens
    assert mtp.finish_details.answer_tokens == ar.finish_details.answer_tokens
    assert mtp.finish_details.phase == ar.finish_details.phase


def test_mtp_stream_forwards_the_published_logprobs() -> None:
    """The MTP stream chunk is the blocking reply, logprobs included."""

    request = _request(max_tokens=6, logprobs=True, top_logprobs=2)
    generator = _ar_generator(_ARRunner(_Target()), _FakeTokenizer())
    mtp_runner = _MTPRunner(_Target())
    provider = Gemma4MTPTextProvider(
        target_generator=_FakeGenerator(mtp_runner, _FakeTokenizer()),
        config=SpeculativeProviderConfig(
            provider="gemma4_mtp",
            draft_model="/nonexistent/sidecar.gguf",
            candidate_budget=4,
        ),
        drafter=_Drafter(mtp_runner),
    )
    generator._speculative_provider = provider

    chunks = list(generator.stream_speculative_mtp_detailed(request))
    blocking = provider.generate_detailed(request)[0]

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == blocking.text
    assert [entry.token_id for entry in chunk.token_logprobs] == _tokens(blocking)
    assert [entry.logprob for entry in chunk.token_logprobs] == [
        entry.logprob for entry in blocking.token_logprobs
    ]
    assert chunk.generated_token_ids == blocking.generated_token_ids
    assert chunk.finish_details == blocking.finish_details
