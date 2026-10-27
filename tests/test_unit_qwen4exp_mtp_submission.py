"""Flash-Next submission and bounded request-owned speculative execution."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_mtp import Qwen4ExpMTPTextProvider
from hipengine.generation.registry import GenerationRequest
from hipengine.speculative.registry import SpeculativeProviderConfig
from tests.test_unit_qwen4exp_mtp_steps import _BatchedTargetRunner, _DraftRunner, _Tokenizer


@pytest.fixture(autouse=True)
def _request_owned_c1_path(monkeypatch):
    """These cases drive the request-owned c1 path with a stand-in resident.

    The shared Flash-Next decode batch is the default; its own suite is
    ``tests/test_unit_qwen4exp_ar_batch_decode.py``.
    """

    from tests._qwen4_exp_c1_control import select_request_owned_c1
    select_request_owned_c1(monkeypatch)


class _Target(_BatchedTargetRunner):
    prefill_chunk_size = 2
    runtime = None

    def __init__(self):
        super().__init__(truth=tuple(range(10, 100)))
        self.config = SimpleNamespace()
        self.chunks = []
        self.transactions = []
        self.closed = False
        self.on_verify = lambda: None
        self.on_prefill = lambda: None

    def reset(self):
        self.position = self.cursor = 0
        self._pending = None

    def prefill_append(self, tokens, **kwargs):
        self.chunks.append(tuple(tokens))
        self.position += len(tokens)
        self.on_prefill()
        hidden = np.stack([self._hidden(token) for token in tokens])
        return SimpleNamespace(token_id=10, hidden_seeds=hidden, hidden_seed=hidden[-1])

    def verify_target_block_state_rows(self, *args, **kwargs):
        result = super().verify_target_block_state_rows(*args, **kwargs)
        self.on_verify()
        return result

    def begin_device_transaction(self, *, reuse_snapshot=False):
        tx = SimpleNamespace(position=self.position, cursor=self.cursor, state="open")
        self.transactions.append(tx)
        return tx

    def commit_device_transaction(self, tx):
        assert tx.state == "open"
        tx.state = "committed"

    def rollback_device_transaction(self, tx):
        assert tx.state == "open"
        self.position, self.cursor = tx.position, tx.cursor
        self._pending = None
        tx.state = "rolled_back"

    def close(self):
        self.closed = True


class _Draft(_DraftRunner):
    def __init__(self):
        super().__init__(proposals=((11, 12), (14, 15), (17, 18)))
        self.chunks = []
        self.restores = []
        self.on_prime = lambda: None

    def reset(self):
        self.position = self.proposal_index = 0

    def prime_prompt_chunk(self, tokens, hidden, *, previous_hidden=None, check_cancelled=None):
        self.chunks.append((tuple(tokens), previous_hidden))
        for _ in tokens:
            if check_cancelled is not None:
                check_cancelled()
            self.position += 1
            self.on_prime()

    def snapshot(self):
        return SimpleNamespace(position=self.position)

    def restore(self, snapshot):
        self.restores.append(snapshot.position)
        self.trim(snapshot.position)


def _driver(*, capacity=1, monkeypatch=None):
    """Build the request-owned c1 MTP driver around stand-in runners.

    The caller pins ``HIPENGINE_QWEN4_EXP_AR_BATCH=0`` for the lifetime of the
    driver (see ``_request_owned_c1_path``): the default shared Flash-Next
    decode batch builds a real batch runner from the resident weights, which
    these stand-ins do not carry.
    """

    target, draft = _Target(), _Draft()
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf", weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(), tokenizer=_Tokenizer(), runner=target,
        resident_capacity=capacity,
    )
    generator._resident = SimpleNamespace(close=lambda: None)
    provider = Qwen4ExpMTPTextProvider(
        target_generator=generator,
        config=SpeculativeProviderConfig(provider="qwen4_exp_mtp", draft_model="unused.gguf", candidate_budget=2),
        draft_runner=draft, draft_resident=object(),
    )
    generator.attach_speculative_provider(provider)
    if monkeypatch is not None:
        monkeypatch.setattr("hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner", lambda *a, **kw: _Target())
        monkeypatch.setattr("hipengine.generation.qwen4_exp_serving_mtp.Qwen4ExpGGUFMTPDraftRunner", lambda *a, **kw: _Draft())
    driver = SubmitPollTextGenerator(generator, config=EngineLoopConfig(max_active_requests=capacity, max_prefill_chunk_tokens=2))
    return driver, target, draft


def _request(**kw):
    return GenerationRequest(**dict(prompts=((1, 2, 3, 4),), max_tokens=7, temperature=0.0, top_p=1.0, ignore_eos=True, **kw))


def _finish(driver, submission):
    events = []
    for _ in range(50):
        if driver.generation_complete(submission):
            return driver.take_result(submission)[0], events
        events.extend(driver.poll(max_ticks=1))
    raise AssertionError("request did not complete")


def test_qwen4exp_mtp_submission_does_not_run_generation():
    driver, target, draft = _driver()
    try:
        submission = driver.submit_speculative_detailed(_request())
        assert target.prefill_calls == draft.prime_calls == draft.propose_calls == 0
        assert not target.chunks and not draft.chunks
        assert not driver.generation_complete(submission)
        assert submission.execution_route == "engine_service_specdec2"
        driver.poll(max_ticks=1)
        assert target.chunks == [(1, 2)]
        assert draft.chunks[0][0] == (1, 2)
        assert target.position == draft.position == 2
        output, events = _finish(driver, submission)
        assert output.generated_token_ids == tuple(range(10, 17))
        assert target.chunks == [(1, 2), (3, 4)]
        np.testing.assert_array_equal(draft.chunks[1][1], target._hidden(2))
        assert output.telemetry.timing["mtp_generated_draft_tokens"] == 4
        assert output.telemetry.timing["mtp_cycles_count"] == 2
        assert [e.token_id for e in events if e.kind == "token"] == list(range(10, 17))
        assert all(tx.state == "committed" for tx in target.transactions)
        assert target.position == draft.position == 10
    finally:
        driver.close()


def test_mtp_cancellation_after_verification_rolls_back_before_publication():
    driver, target, draft = _driver()
    token = GenerationCancellationToken()
    try:
        sub = driver.submit_speculative_detailed(_request(cancellation_token=token))
        # Two prompt chunks, then the root; no proposals yet.
        for _ in range(3):
            driver.poll(max_ticks=1)
        assert draft.propose_calls == 0
        target.on_verify = token.cancel
        output, events = _finish(driver, sub)
        assert not [e for e in events if e.kind == "token"]
        assert output.generated_token_ids == (10,)
        assert output.finish_details.cancelled
        assert target.transactions[-1].state == "rolled_back"
        assert target.position == draft.position == 4
        assert draft.restores == [4]
        target.on_verify = lambda: None
        clean, _ = _finish(driver, driver.submit_speculative_detailed(_request()))
        assert clean.generated_token_ids == tuple(range(10, 17))
    finally:
        driver.close()


def test_two_mtp_rows_have_distinct_target_and_draft_owners(monkeypatch):
    driver, target, draft = _driver(capacity=2, monkeypatch=monkeypatch)
    try:
        first, second = driver.submit_speculative_many_detailed((_request(), _request()))
        a, _ = _finish(driver, first)
        b, _ = _finish(driver, second)
        assert a.generated_token_ids == b.generated_token_ids == tuple(range(10, 17))
        native = driver._runner
        assert len({id(item) for item in native._all_drafts}) == 2
        assert native.speculative_partition_max_requests(None) == 1
        assert len(native._available_drafts) == len(native._available) == 2
    finally:
        driver.close()


@pytest.mark.parametrize("controls", [{"resident_session_key": "reuse-not-supported"}])
def test_unimplemented_mtp_controls_reject_before_model_work(controls):
    driver, target, draft = _driver()
    try:
        with pytest.raises(NotImplementedError, match="resident session"):
            driver.submit_speculative_detailed(replace(_request(), **controls))
        assert not target.chunks and not draft.chunks
        assert driver._runner.active_request_ids == ()
    finally:
        driver.close()


@pytest.mark.parametrize("eos", [10, 11, 12, 13, 16])
def test_mtp_eos_stops_at_committed_token(eos):
    driver, target, draft = _driver()
    driver.inner.tokenizer.eos_token_id = eos
    try:
        output, events = _finish(driver, driver.submit_speculative_detailed(
            replace(_request(), ignore_eos=False, eos_token_id=eos)))
        assert output.generated_token_ids == tuple(range(10, eos + 1))
        assert output.finish_details.reason == "eos"
        assert tuple(e.token_id for e in events if e.kind == "token") == output.generated_token_ids
        if eos > 10:
            assert draft.propose_calls > 0, "EOS requests must not silently become all K0"
    finally:
        driver.close()


@pytest.mark.parametrize("max_tokens", [0, 1, 2, 3, 4, 5, 6, 7])
def test_staged_mtp_output_room_and_k0(max_tokens):
    driver, target, draft = _driver()
    try:
        output, _ = _finish(driver, driver.submit_speculative_detailed(
            replace(_request(), max_tokens=max_tokens)))
        assert output.generated_token_ids == tuple(range(10, 10 + max_tokens))
        assert output.finish_details.reason == "length"
        if max_tokens:
            assert target.position == draft.position == 4 + max_tokens - 1
    finally:
        driver.close()


@pytest.mark.parametrize("cancel_stage", ["target_prompt", "draft_prompt", "k0"])
def test_cancel_at_bounded_native_work_retires_and_recovers(cancel_stage):
    driver, target, draft = _driver()
    token = GenerationCancellationToken()
    try:
        request = replace(_request(cancellation_token=token), max_tokens=2)
        sub = driver.submit_speculative_detailed(request)
        if cancel_stage == "target_prompt":
            target.on_prefill = token.cancel
        elif cancel_stage == "draft_prompt":
            draft.on_prime = token.cancel
        else:
            original_step = target.step
            def cancel_step(*args, **kwargs):
                result = original_step(*args, **kwargs)
                token.cancel()
                return result
            target.step = cancel_step
        output, events = _finish(driver, sub)
        expected = (10,) if cancel_stage == "k0" else ()
        assert output.generated_token_ids == expected
        assert tuple(e.token_id for e in events if e.kind == "token") == expected
        assert output.finish_details.cancelled
        assert len(driver._runner._available) == len(driver._runner._available_drafts) == 1
        if cancel_stage == "k0":
            assert target.transactions[-1].state == "rolled_back"
            target.step = original_step
        target.on_prefill = draft.on_prime = lambda: None
        recovered, _ = _finish(driver, driver.submit_speculative_detailed(_request()))
        assert recovered.generated_token_ids == tuple(range(10, 17))
    finally:
        driver.close()
