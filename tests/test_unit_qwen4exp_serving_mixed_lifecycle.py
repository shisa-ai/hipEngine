"""Shared Flash-Next target serving lifecycle through the resident scheduler.

Every test here drives the real resident surface: ``SubmitPollTextGenerator``
builds a real ``Qwen4ExpResidentServingRunner``, a real
``Qwen4ExpARBatchOwner`` and a real ``ResidentEngineLoop``, then each tick runs
the scheduler's own admission, prefill, plan, decode and reclaim path. Only the
device work is replaced:

* the c1 staging runner, the shared batch runner and the draft runner are the
  shared fakes from ``tests/_qwen4exp_ar_batch_fixtures.py``;
* ``prepare_proposal`` returns a deterministic host chain instead of running the
  draft model, so acceptance is decided by the truth table a test sets;
* ``Qwen4ExpMixedTargetPass`` is a host stand-in that scores each physical row
  from that truth table and advances the batch cursor exactly as a committed
  pass does (``depth + 1`` per request).

The assertions are control, not arithmetic: which request ids one shared pass
committed, how many target inputs each request contributed, that a forced K0
pass publishes the target's own row token once, that a cancelled peer rolls the
whole pass back without publishing, that the freed slot is reused by the next
arrival, and that an EOS or stop token in the middle of a chain stops the cycle
instead of publishing the rest of the chain. The last check in every test
compares the scheduler's published token events with the request's committed
output, so a duplicated or dropped publication fails.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
AR_BATCH_ENV = "HIPENGINE_QWEN4_EXP_AR_BATCH"
from hipengine.generation.qwen4_exp_mtp_stages import Qwen4ExpMTPProposal
from hipengine.generation.registry import GenerationRequest
from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetResult
from tests._qwen4exp_ar_batch_fixtures import (
    VOCAB,
    FakeBatchRunner,
    FakeC1Runner,
    FakeDraft,
    FakeMTPProvider,
    make_generator,
)

#: The fake tokenizer's end-of-sequence id.
EOS = VOCAB - 1
#: First token id the host target emits. Target tokens stay above every
#: prompt-derived root (``sum(prompt) % 50``) so a published duplicate is visible,
#: and below the vocabulary so a host-sampled row's argmax round-trips exactly.
FIRST_TARGET_TOKEN = 30
#: A stop-token id that is distinct from EOS and from every emitted target token.
STOP_TOKEN = VOCAB - 2


def _arm_draft(draft):
    """Add the mixed-cycle surface a shared fake draft does not need to model."""

    draft.trim = lambda position: setattr(draft, "position", int(position))
    draft.forward = lambda *args, **kwargs: setattr(
        draft, "position", draft.position + 1
    )
    return draft


class _SharedTargetPass:
    """Host stand-in for one device target pass over a validated mixed layout."""

    def __init__(self, harness: "_Lifecycle", batch, layout) -> None:
        self.harness = harness
        self.batch = batch
        self.layout = layout
        self.request_ids = tuple(request.request_id for request in layout.requests)
        self.candidate_counts = tuple(request.depth for request in layout.requests)
        self.total_rows = layout.total_rows
        self.truth: dict[int, tuple[int, ...]] = {}
        self.depths: tuple[int, ...] | None = None
        self.aborted = False
        self.closed = False
        harness.passes.append(self)

    def verify(self):
        self.truth = self.harness.truth(self.layout)
        if self.harness.on_verify is not None:
            self.harness.on_verify(self)
        tokens = [self.truth[row.request_id][row.depth] for row in self.layout.rows]
        rows = len(tokens)
        # One hot at the truth token: a host-sampled row selects from these
        # logits, so its argmax has to be the row token exactly.
        logits = np.zeros((rows, VOCAB), dtype=np.float32)
        for index, token in enumerate(tokens):
            assert 0 <= token < VOCAB, f"target token {token} is outside the vocabulary"
            logits[index, token] = 1.0
        return Qwen4ExpMixedTargetResult(
            self.layout,
            np.asarray(tokens, dtype=np.int64),
            logits,
            np.zeros((rows, 4), dtype=np.float32),
        )

    def commit(self, depths) -> None:
        self.depths = tuple(int(depth) for depth in depths)
        for request, depth in zip(self.layout.requests, self.depths, strict=True):
            self.batch._positions[request.slot] += depth + 1

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


class _Lifecycle:
    """One resident scheduler over fake device owners and a host target pass."""

    def __init__(self, monkeypatch, *, capacity: int, chain=None) -> None:
        self.passes: list[_SharedTargetPass] = []
        self.published: dict[int, list[int]] = {}
        self.submissions: dict[int, object] = {}
        self.on_verify = None
        self._next_token = FIRST_TARGET_TOKEN
        self._chain = chain
        self.proposals: dict[int, Qwen4ExpMTPProposal] = {}
        monkeypatch.delenv(AR_BATCH_ENV, raising=False)
        monkeypatch.setattr(FakeBatchRunner, "supports_mixed_target", True, raising=False)
        monkeypatch.setenv("HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET", "1")
        monkeypatch.setattr(
            "hipengine.runtime.qwen4_exp_batch_runner.Qwen4ExpGGUFBatchRunner",
            FakeBatchRunner,
        )
        monkeypatch.setattr(
            "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
            lambda *args, **kwargs: FakeC1Runner(bias=10),
        )
        monkeypatch.setattr(
            "hipengine.generation.qwen4_exp_serving_mtp.Qwen4ExpGGUFMTPDraftRunner",
            lambda *args, **kwargs: _arm_draft(FakeDraft()),
        )
        monkeypatch.setattr(
            "hipengine.generation.qwen4_exp_serving_mixed.prepare_proposal",
            self.propose,
        )
        monkeypatch.setattr(
            "hipengine.generation.qwen4_exp_serving_mixed.Qwen4ExpMixedTargetPass",
            lambda batch, layout: _SharedTargetPass(self, batch, layout),
        )
        self.generator = make_generator()
        self.generator._speculative_provider = FakeMTPProvider()
        _arm_draft(self.generator._speculative_provider.draft)
        self.driver = SubmitPollTextGenerator(
            self.generator,
            config=EngineLoopConfig(
                max_active_requests=capacity,
                max_prefill_chunk_tokens=8,
                prefill_decode_policy="protect_ttft",
            ),
        )
        self.native = self.generator._resident_model_runner
        self.fake = self.native._ar_batch.runner
        self.fake.runtime = SimpleNamespace(device_synchronize=lambda: None)

    # -- host stand-ins ---------------------------------------------------

    def emit(self) -> int:
        token = self._next_token
        self._next_token += 1
        if token >= VOCAB:
            raise AssertionError("the host target ran out of in-vocabulary tokens")
        return token

    def chain(self, root_token: int, count: int) -> tuple[int, ...]:
        """The draft chain a test wants, or one fresh target token per depth."""

        if self._chain is not None:
            return tuple(self._chain(self, root_token, count))
        return tuple(self.emit() for _ in range(count))

    def propose(self, draft_runner, *, start_position, root_token, target_hidden_seed,
                draft_n_max, compact_output=False, lookup_candidates=()):
        del target_hidden_seed, compact_output
        assert not lookup_candidates, "this fixture exercises neural-chain lifecycle"
        candidates = self.chain(int(root_token), int(draft_n_max))
        assert len(candidates) == int(draft_n_max), "chain must fill the planned depth"
        draft_runner.position = int(start_position) + len(candidates)
        proposal = Qwen4ExpMTPProposal(int(start_position), int(root_token), candidates)
        self.proposals[id(draft_runner)] = proposal
        return proposal

    def truth(self, layout):
        """Accept every proposed candidate, then emit the next target token."""

        truth: dict[int, tuple[int, ...]] = {}
        for request in layout.requests:
            row = self.native._rows[request.request_id]
            proposal = self.proposals.get(id(row.draft))
            if request.depth and proposal is not None:
                truth[request.request_id] = (*proposal.candidates, self.emit())
            else:
                truth[request.request_id] = (self.emit(),)
        return truth

    # -- driver surface ---------------------------------------------------

    def submit_speculative(self, prompt, *, max_tokens, **kwargs) -> int:
        request = GenerationRequest(
            prompts=(prompt,), max_tokens=max_tokens, temperature=0.0, top_p=1.0,
            ignore_eos=kwargs.pop("ignore_eos", True), **kwargs,
        )
        return self._record(self.driver.submit_speculative_detailed(request))

    def submit_plain(self, prompt, *, max_tokens, **kwargs) -> int:
        request = GenerationRequest(
            prompts=(prompt,), max_tokens=max_tokens, temperature=0.0, top_p=1.0,
            ignore_eos=kwargs.pop("ignore_eos", True), **kwargs,
        )
        return self._record(self.driver.submit_detailed(request))

    def _record(self, submission) -> int:
        request_id = int(submission.request_ids[0])
        self.submissions[request_id] = submission
        return request_id

    def poll(self, ticks: int = 1):
        events = self.driver.poll(max_ticks=ticks)
        for event in events:
            if event.kind == "token":
                self.published.setdefault(int(event.request_id), []).append(
                    int(event.token_id)
                )
        return events

    def poll_until_complete(self, request_ids, *, limit: int = 24) -> None:
        wanted = tuple(int(request_id) for request_id in request_ids)
        for _ in range(limit):
            if all(
                self.driver.generation_complete(self.submissions[request_id])
                for request_id in wanted
            ):
                return
            self.poll()
        raise AssertionError(f"requests {list(wanted)} did not complete in {limit} ticks")

    def output(self, request_id: int):
        return self.driver.take_result(self.submissions[int(request_id)])[0]

    def close(self) -> None:
        self.driver.close()

    # -- assertions -------------------------------------------------------

    def assert_published_once(self, request_id: int, output) -> None:
        """The scheduler published the committed ids, in order, each one once."""

        request_id = int(request_id)
        published = tuple(self.published[request_id])
        assert published == tuple(output.generated_token_ids)
        assert len(set(published)) == len(published), "a token was published twice"


def test_forced_k0_pass_publishes_one_target_row_per_due_request(monkeypatch):
    """Out of chain room, the group still advances through one shared K0 pass."""

    lifecycle = _Lifecycle(monkeypatch, capacity=2)
    try:
        mixed = lifecycle.submit_speculative((1, 2), max_tokens=5)
        peer = lifecycle.submit_plain((3, 4), max_tokens=5)
        lifecycle.poll(ticks=1)  # admit both rows
        mixed_slot, _ = lifecycle.native._ar_batch.lease(mixed)
        peer_slot, _ = lifecycle.native._ar_batch.lease(peer)
        lifecycle.poll_until_complete((mixed, peer))

        k0 = lifecycle.passes[-1]
        # The previous tick did propose, so this pass is forced, not incidental:
        # the mixed row has one token of output room left and cannot verify a chain.
        assert lifecycle.passes[-2].candidate_counts == (2, 0)
        assert k0.request_ids == (mixed, peer)
        assert k0.candidate_counts == (0, 0)
        assert k0.total_rows == 2
        assert k0.depths == (0, 0)
        assert k0.closed and not k0.aborted
        assert lifecycle.native._last_mixed_pass["candidate_counts"] == [0, 0]
        assert lifecycle.native._last_mixed_pass["committed_input_rows"] == [1, 1]
        # The pass's row token is published exactly once per request. The mixed
        # row publishes it in the pass (it is its last token); an AR peer
        # publishes the token the pass produced on its next decode.
        for request_id in (mixed, peer):
            assert lifecycle.published[request_id].count(k0.truth[request_id][0]) == 1
        assert lifecycle.published[mixed][-1] == k0.truth[mixed][0]

        outputs = {request_id: lifecycle.output(request_id) for request_id in (mixed, peer)}
        assert len(outputs[mixed].generated_token_ids) == 5
        assert len(outputs[peer].generated_token_ids) == 5
        assert outputs[mixed].finish_details.reason == "length"
        assert outputs[peer].finish_details.reason == "length"
        for request_id, output in outputs.items():
            lifecycle.assert_published_once(request_id, output)
        # The mixed row never took a one-token AR batch step: every batch step
        # in the run was the AR peer's own single row.
        assert {call[0] for call in lifecycle.fake.step_calls} == {(peer_slot,)}
        assert mixed_slot != peer_slot
    finally:
        lifecycle.close()


def test_late_first_root_is_published_beside_a_running_mixed_row(monkeypatch):
    """A row admitted mid-flight gets its root without stalling the mixed row."""

    lifecycle = _Lifecycle(monkeypatch, capacity=2)
    try:
        mixed = lifecycle.submit_speculative((1, 2), max_tokens=8)
        lifecycle.poll(ticks=2)
        assert lifecycle.native._row(mixed).generated_ids
        # The prompt's initial root is emitted by decode_batch, never by a pass.
        assert lifecycle.passes == []

        peer = lifecycle.submit_plain((3, 4), max_tokens=6)
        lifecycle.poll(ticks=1)  # prefill the arrival
        assert not lifecycle.native._row(peer).generated_ids

        lifecycle.poll(ticks=1)  # first tick where both rows are due
        arrival = lifecycle.passes[-1]
        assert arrival.request_ids == (mixed,)
        assert arrival.candidate_counts == (2,)
        assert arrival.total_rows == 3
        assert arrival.depths == (2,)
        # The mixed row kept its chain while the peer's root was published by
        # exactly one single-row batch step.
        peer_root = list(lifecycle.native._row(peer).generated_ids)
        assert len(peer_root) == 1
        assert len(lifecycle.fake.step_calls) == 1
        assert lifecycle.fake.step_calls[0][2] == (peer_root[0],)
        assert len(lifecycle.fake.step_calls[0][0]) == 1

        lifecycle.poll(ticks=1)  # both rows ready: one shared width-2 pass
        shared = lifecycle.passes[-1]
        assert shared.request_ids == (mixed, peer)
        assert shared.candidate_counts == (2, 0)
        assert shared.total_rows == 4
        assert shared.depths == (2, 0)
        assert lifecycle.native._last_mixed_pass["request_width"] == 2
        assert lifecycle.native._last_mixed_pass["committed_input_rows"] == [3, 1]

        lifecycle.poll_until_complete((mixed, peer))
        for request_id in (mixed, peer):
            lifecycle.assert_published_once(request_id, lifecycle.output(request_id))
    finally:
        lifecycle.close()


def test_cancelled_peer_rolls_back_the_pass_then_the_survivor_reuses_its_slot(monkeypatch):
    """One row's cancellation aborts the pass; the peer keeps going on the slot."""

    lifecycle = _Lifecycle(monkeypatch, capacity=2)
    try:
        mixed = lifecycle.submit_speculative((1, 2), max_tokens=9)
        lifecycle.poll(ticks=2)
        token = GenerationCancellationToken()
        peer = lifecycle.submit_plain((3, 4), max_tokens=6, cancellation_token=token)
        lifecycle.poll(ticks=2)  # prefill the peer, then publish its first root
        peer_root = list(lifecycle.native._row(peer).generated_ids)
        assert len(peer_root) == 1
        peer_slot, peer_owner = lifecycle.native._ar_batch.lease(peer)
        before = len(lifecycle.published[mixed])
        passes_before = len(lifecycle.passes)

        def cancel_peer(target):
            # The cancellation lands while the shared target pass is running,
            # after every due row was projected into the same layout.
            if peer in target.request_ids:
                lifecycle.on_verify = None
                token.cancel()

        lifecycle.on_verify = cancel_peer
        lifecycle.poll(ticks=1)
        cancelled = lifecycle.passes[passes_before]
        assert cancelled.request_ids == (mixed, peer)
        assert cancelled.aborted and cancelled.closed
        assert cancelled.depths is None
        # Nothing from that pass was published for either row.
        assert len(lifecycle.published[mixed]) == before
        assert lifecycle.published[peer] == peer_root
        assert lifecycle.native._ar_batch.lease(peer) is None
        assert (peer_slot, peer_owner) in lifecycle.fake.release_calls
        assert lifecycle.fake.owners_of((peer_slot,)) == (0,)

        # The survivor resumes on its own after the rollback.
        lifecycle.poll(ticks=1)
        survivor = lifecycle.passes[-1]
        assert survivor.request_ids == (mixed,)
        assert survivor.depths is not None
        assert len(lifecycle.published[mixed]) > before

        # The freed slot goes to the next arrival with a fresh owner token.
        arrival = lifecycle.submit_plain((5, 6), max_tokens=4)
        lifecycle.poll(ticks=2)
        reused = lifecycle.native._ar_batch.lease(arrival)
        assert reused is not None
        assert reused[0] == peer_slot
        assert reused[1] != peer_owner
        assert lifecycle.fake.owners_of((peer_slot,)) == (reused[1],)

        lifecycle.poll_until_complete((mixed, arrival))
        cancelled_output = lifecycle.output(peer)
        assert cancelled_output.finish_details.cancelled
        assert cancelled_output.generated_token_ids == tuple(peer_root)
        lifecycle.assert_published_once(peer, cancelled_output)
        for request_id in (mixed, arrival):
            lifecycle.assert_published_once(request_id, lifecycle.output(request_id))
        # The reused slot carried the new request, not the cancelled one.
        assert lifecycle.native._ar_batch.lease(arrival) is None
        assert (peer_slot, reused[1]) in lifecycle.fake.release_calls
    finally:
        lifecycle.close()


@pytest.mark.parametrize("terminal", ["eos", "stop"])
def test_terminal_token_in_the_middle_of_a_chain_stops_the_cycle(monkeypatch, terminal):
    """A terminal candidate ends the cycle before the rest of the chain commits."""

    terminal_id = EOS if terminal == "eos" else STOP_TOKEN

    def chain(lifecycle, root_token, count):
        del root_token
        # The first candidate is terminal. The plan still prepares every depth,
        # so a cycle that consumed more than its root row would publish the rest.
        return (terminal_id, *tuple(lifecycle.emit() for _ in range(count - 1)))

    lifecycle = _Lifecycle(monkeypatch, capacity=2, chain=chain)
    try:
        # A stop token is a sampler processor, so that row is host-sampled: the
        # pass's one-hot logits are what make its argmax the row token exactly.
        terminal_request = (
            {"ignore_eos": False}
            if terminal == "eos"
            else {"stop_token_ids": (terminal_id,)}
        )
        mixed = lifecycle.submit_speculative((1, 2), max_tokens=8, **terminal_request)
        peer = lifecycle.submit_plain((3, 4), max_tokens=3)
        lifecycle.poll_until_complete((mixed, peer))

        finished = lifecycle.passes[-1]
        assert finished.request_ids == (mixed, peer)
        assert finished.candidate_counts == (2, 0)
        # Only the root row was committed: the terminal candidate ended the cycle
        # with the second candidate never consumed or published.
        assert finished.depths == (0, 0)
        assert finished.closed and not finished.aborted
        assert finished.truth[mixed][0] == terminal_id
        output = lifecycle.output(mixed)
        assert output.finish_details.reason == terminal
        assert output.generated_token_ids[-1] == terminal_id
        assert output.generated_token_ids.count(terminal_id) == 1
        # The terminal token is published once and held out of the visible text.
        assert output.text == ":".join(
            str(token) for token in output.generated_token_ids[:-1]
        )
        lifecycle.assert_published_once(mixed, output)
        lifecycle.assert_published_once(peer, lifecycle.output(peer))
    finally:
        lifecycle.close()
