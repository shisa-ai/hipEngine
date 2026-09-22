"""MTP logprob metadata must be the autoregressive route's value, not a rebuild.

``select_token`` reports logprobs differently by branch: a greedy row reports the
full-support softmax of the processed logits, a sampling row reports the retained
support's probability for the token it drew. A speculative route that reports one
shape for both is wrong in one of them, so these tests pin ``reported_logprob``
against ``select_token`` itself, and pin the row mapping that decides *which*
distribution each published token is scored against.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.batch_scheduler import GeneratedTokenEvent
from hipengine.generation.mtp_sampled_accept import (
    chain_edge_rows,
    chain_token_logprobs,
)
from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.generation.qwen35_gguf_mtp2 import publish_cycle_token_samples
from hipengine.generation.registry import GenerationStreamChunk
from hipengine.generation.sampling import (
    RowSamplingState,
    reported_logprob,
    select_token,
    supports_sampled_speculative_mtp,
)
from hipengine.speculative.interfaces import TargetVerifyBatch


def _params(**overrides):
    values = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": (),
        "suppress_token_ids": (),
        "min_tokens": 0,
        "eos_token_id": None,
        "ignore_eos": False,
        "seed": 1234,
        "row_seeds": (),
        "stop_token_ids": (),
        "stop_token_sequences": (),
        "forced_tokens_pending": (),
        "forced_token_reason": None,
        "post_thinking_forced_tokens_pending": (),
        "post_thinking_forced_token_reason": None,
        "force_sequence_completion_token_sequences": (),
        "force_sequence_completion_reason": None,
        "json_object_close_forcing": False,
        "tool_call_constraint": None,
        "thinking_close_token_ids": (),
        "thinking_hard_token_cap": None,
        "thinking_soft_close_window": 0,
        "logprobs": True,
        "top_logprobs": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _row(values: tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


_PEAKED = _row((8.0, 4.0, 2.0, 1.0, 0.5, 0.0, -1.0))
_FLAT = _row((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

_SAMPLED_PARAMS = {
    "temperature_1": _params(temperature=1.0),
    "temperature_low": _params(temperature=0.35),
    "top_k_2": _params(temperature=1.0, top_k=2),
    "top_p_0.6": _params(temperature=1.0, top_p=0.6),
    "min_p_0.2": _params(temperature=1.0, min_p=0.2),
    "penalties": _params(
        temperature=1.0, repetition_penalty=1.4, presence_penalty=0.5
    ),
    "logit_bias": _params(temperature=1.0, logit_bias={3: 6.0, 0: -6.0}),
    "suppressed": _params(temperature=1.0, suppress_token_ids=(0, 1)),
}


@pytest.mark.parametrize("params_name", sorted(_SAMPLED_PARAMS))
@pytest.mark.parametrize("row_name", ["peaked", "flat"])
def test_reported_logprob_equals_what_select_token_reports(
    params_name: str, row_name: str
) -> None:
    """The route's metadata is the AR route's value for the token AR selected."""

    params = _SAMPLED_PARAMS[params_name]
    logits = _PEAKED if row_name == "peaked" else _FLAT
    history = (1, 2, 3)
    ar_state = RowSamplingState(seed=99, prompt_tokens=history)
    result = select_token(logits, params, ar_state)
    # reported_logprob must not draw, so it reads a state with the same history
    # rather than the one select_token advanced.
    replay_state = RowSamplingState(seed=99, prompt_tokens=history)

    logprob, top_logprobs = reported_logprob(
        logits, params, replay_state, result.token_id
    )

    assert logprob == pytest.approx(float(result.logprob), abs=1e-12)
    assert top_logprobs == result.top_logprobs


@pytest.mark.parametrize("params_name", sorted(_SAMPLED_PARAMS))
def test_reported_logprob_does_not_advance_the_sampler_state(params_name: str) -> None:
    """Metadata is read-only: a speculative route must not consume the stream."""

    params = _SAMPLED_PARAMS[params_name]
    state = RowSamplingState(seed=11, prompt_tokens=(4, 5, 6))
    before = [state.random_unit() for _ in range(3)]
    state = RowSamplingState(seed=11, prompt_tokens=(4, 5, 6))

    reported_logprob(_PEAKED, params, state, 0)

    after = [state.random_unit() for _ in range(3)]
    assert after == before


def test_greedy_rows_report_the_full_support_not_the_point_mass() -> None:
    """A greedy row's logprob is the model's probability, not log(1)."""

    params = _params(temperature=0.0)
    logprob, top_logprobs = reported_logprob(_PEAKED, params, None, 0)

    assert logprob is not None
    # The point-mass support the sampled branch uses would report log(1) == 0.
    assert logprob < -1e-3
    assert top_logprobs[0][0] == 0
    assert top_logprobs[0][1] == pytest.approx(logprob)


def test_sampling_rows_report_the_retained_support_probability() -> None:
    """A sampling row's logprob is the probability the draw came from."""

    params = _params(temperature=1.0, top_k=2)
    logprob, top_logprobs = reported_logprob(_PEAKED, params, None, 0)

    assert len(top_logprobs) == 2
    assert sum(np.exp(value) for _, value in top_logprobs) == pytest.approx(1.0)
    assert logprob == pytest.approx(top_logprobs[0][1])


def test_reported_logprob_refuses_a_token_the_row_filtered_out() -> None:
    """A token outside the row's finite support has no reportable logprob.

    The route never publishes such a token -- an accepted draft, a residual
    correction, and a bonus all come from the row's own distribution -- so this
    is the guard against scoring a token against a row that did not produce it,
    which is the off-by-one failure this metadata path could hide.
    """

    params = _params(temperature=0.0, suppress_token_ids=(0,))

    logprob, top_logprobs = reported_logprob(_PEAKED, params, None, 0)

    assert logprob is None
    assert top_logprobs == ()


def _chain_batch(*, candidates: int, tokens: tuple[int, ...]) -> TargetVerifyBatch:
    """One request whose verified chain is a root row plus ``candidates`` rows."""

    return TargetVerifyBatch(
        request_ids=(7,),
        tokens=tokens,
        positions=tuple(range(len(tokens))),
        parent_rows=(-1, *range(candidates)),
        draft_depths=tuple([0, *range(1, candidates + 1)]),
        active_mask=(True,) * len(tokens),
        root_rows=(0,),
        candidate_rows=tuple(range(1, candidates + 1)),
        row_to_request=(7,) * len(tokens),
        mode="verify_chain",
    )


def test_chain_edge_rows_scores_each_token_against_the_row_that_predicted_it() -> None:
    """Row r predicts the token after r, so token i is scored on row i of the path."""

    batch = _chain_batch(candidates=3, tokens=(10, 11, 12, 13))

    # Two drafts accepted: published tokens are 11, 12 and then the bonus.
    assert chain_edge_rows(batch, (2,)) == ((0, 1, 2),)
    # Nothing accepted: the correction token is predicted by the root row.
    assert chain_edge_rows(batch, (0,)) == ((0,),)
    # The whole chain accepted: the bonus token comes from the last row.
    assert chain_edge_rows(batch, (3,)) == ((0, 1, 2, 3),)


def test_chain_edge_rows_refuses_a_chain_longer_than_the_verified_rows() -> None:
    batch = _chain_batch(candidates=2, tokens=(10, 11, 12))

    with pytest.raises(ValueError, match="longer than the verified rows"):
        chain_edge_rows(batch, (3,))


def test_chain_token_logprobs_scores_the_published_prefix_only() -> None:
    """A cycle that published two of four tokens reports metadata for two."""

    params = _params(temperature=1.0)
    batch = _chain_batch(candidates=3, tokens=(10, 11, 12, 13))
    # Rows are one-hot so the expected value is exact and readable.
    logits = np.full((4, 16), -8.0, dtype=np.float32)
    for row, token in enumerate((11, 12, 13, 11)):
        logits[row, token] = 8.0

    metadata = chain_token_logprobs(
        batch,
        logits,
        {7: RowSamplingState(seed=5, prompt_tokens=(10,))},
        [(11, 12)],
        (2,),
        params_for=lambda request_id: params,
    )

    assert len(metadata) == 1
    assert len(metadata[0]) == 2
    for (token, (logprob, top_logprobs)), expected_token, expected_row in zip(
        zip((11, 12), metadata[0]), (11, 12), (0, 1)
    ):
        assert token == expected_token
        assert logprob == pytest.approx(float(np.log(1.0)), abs=1e-4)
        assert top_logprobs[0][0] == int(logits[expected_row].argmax())


def test_a_logprobs_request_is_servable_on_the_sampled_route() -> None:
    """The two metadata fields are the servable set's, not the refused set's."""

    assert supports_sampled_speculative_mtp(_params(temperature=1.0))
    assert supports_sampled_speculative_mtp(_params(logprobs=True, top_logprobs=5))
    assert supports_sampled_speculative_mtp(_params(temperature=0.0, logprobs=True))


# --- Streamed cycle metadata -------------------------------------------------
#
# A committed cycle publishes several tokens at once, and the stream events for
# them are decorated after the commit -- by which point a cycle that finishes
# the request has already reclaimed its row. These tests pin the record that
# carries the metadata across that window and the by-token matching that
# attaches it to the right event.


class _FakeTokenizer:
    def decode(self, token_ids, **kwargs):
        return "".join(f"<{int(token)}>" for token in token_ids)


class _StreamDecorator:
    """The two resident-runner methods the decorator needs, without a model."""

    decorate_speculative_stream_events = (
        Qwen35GGUFResidentModelRunner.decorate_speculative_stream_events
    )
    _take_cycle_token_logprob = Qwen35GGUFResidentModelRunner._take_cycle_token_logprob

    def __init__(self, samples=None):
        self.generator = SimpleNamespace(tokenizer=_FakeTokenizer())
        self._cycle_token_samples = {} if samples is None else samples


def _event(request_id: int, token_id: int, *, text: str = "", logprobs=()):
    return GeneratedTokenEvent(
        request_id=request_id,
        token_id=token_id,
        finished=False,
        stream_chunk=GenerationStreamChunk(text=text, token_logprobs=tuple(logprobs)),
    )


def _sample(token_id: int, logprob: float, top=((7, -0.1),)):
    return SimpleNamespace(token_id=token_id, logprob=logprob, top_logprobs=top)


def test_stream_events_carry_the_recorded_cycle_metadata() -> None:
    """Each published token's own record reaches its own event."""

    decorator = _StreamDecorator(
        {5: [_sample(11, -0.25), _sample(12, -0.5, top=((9, -0.2),))]}
    )
    decorated = decorator.decorate_speculative_stream_events(
        (_event(5, 11), _event(5, 12))
    )

    first, second = (event.stream_chunk.token_logprobs[0] for event in decorated)
    assert (first.token_id, first.logprob) == (11, -0.25)
    assert (second.token_id, second.logprob) == (12, -0.5)
    assert second.top_logprobs == ((9, "<9>", -0.2),)
    # The record is consumed, so a later cycle cannot reuse it.
    assert decorator._cycle_token_samples == {}


def test_stream_events_match_recorded_metadata_by_token_not_by_position() -> None:
    """A stale entry for another token is left for the event that owns it."""

    decorator = _StreamDecorator({5: [_sample(99, -1.0), _sample(12, -0.5)]})
    decorated = decorator.decorate_speculative_stream_events((_event(5, 12),))

    attached = decorated[0].stream_chunk.token_logprobs
    assert [token.token_id for token in attached] == [12]
    assert [sample.token_id for sample in decorator._cycle_token_samples[5]] == [99]


def test_stream_events_keep_scheduler_metadata_when_nothing_was_recorded() -> None:
    """A route that records nothing leaves the chunk's own metadata alone."""

    from hipengine.generation.registry import TokenLogprob

    existing = TokenLogprob(
        token_id=3,
        token_text="<3>",
        logprob=-0.75,
        top_logprobs=(),
    )
    decorator = _StreamDecorator()
    decorated = decorator.decorate_speculative_stream_events(
        (_event(5, 3, text="", logprobs=(existing,)),)
    )

    assert decorated[0].stream_chunk.token_logprobs == (existing,)
    assert decorator._cycle_token_samples == {}


def test_take_cycle_token_logprob_ignores_an_unrecorded_token() -> None:
    decorator = _StreamDecorator({5: [_sample(11, -0.25)]})

    assert decorator._take_cycle_token_logprob(5, 12) is None
    assert decorator._take_cycle_token_logprob(6, 11) is None
    assert [sample.token_id for sample in decorator._cycle_token_samples[5]] == [11]


def test_publish_cycle_token_samples_copies_and_tolerates_a_runner_without_the_store() -> None:
    """The commit publishes a snapshot; a runner without the store is left alone."""

    samples = [_sample(11, -0.25)]
    owner = SimpleNamespace(_cycle_token_samples={})
    publish_cycle_token_samples(owner, "5", samples)

    assert owner._cycle_token_samples[5] is not samples
    assert owner._cycle_token_samples[5] == samples
    # A runner that has no store must not raise: its events simply keep the
    # scheduler's metadata.
    publish_cycle_token_samples(SimpleNamespace(), 5, samples)


def test_take_outputs_drops_the_record_after_the_blocking_reader_is_done() -> None:
    """The record outlives the row but not the request."""

    class _Runner:
        take_outputs = Qwen35GGUFResidentModelRunner.take_outputs

    runner = _Runner()
    runner._outputs = {5: "output"}
    runner._cycle_token_samples = {5: [_sample(11, -0.25)]}

    assert runner.take_outputs((5,)) == ["output"]
    assert runner._cycle_token_samples == {}


def test_discard_drops_the_record_for_an_abandoned_request() -> None:
    class _Runner:
        discard = Qwen35GGUFResidentModelRunner.discard

    runner = _Runner()
    runner._outputs = {}
    runner._rows = {}
    runner._completed_metadata = {}
    runner._mtp2_adapter = None
    runner._cycle_token_samples = {5: [_sample(11, -0.25)]}

    runner.discard((5,))
    assert runner._cycle_token_samples == {}
