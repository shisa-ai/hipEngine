"""Cooperative Gemma cycles preserve the existing sampler while isolating owners."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.engine_service import EngineService
from hipengine.generation.engine_loop import SubmitPollTextGenerator, EngineLoopConfig
from hipengine.generation.gemma4_resident import Gemma4ResidentModelRunner
from hipengine.generation.gemma4_mtp import Gemma4MTPTextProvider
from hipengine.generation.registry import GenerationRequest
from hipengine.speculative.registry import SpeculativeProviderConfig
from tests.test_unit_gemma4_mtp_cycle_accounting import (
    _FakeDrafter, _FakeGenerator, _FakeRunner, _greedy_reference, _next_token, _VOCAB,
)


def test_independent_cycle_frames_interleave_without_cursor_or_sampler_leaks():
    runners = [_FakeRunner(), _FakeRunner()]
    provider = Gemma4MTPTextProvider(
        target_generator=_FakeGenerator(runners[0]),
        config=SpeculativeProviderConfig(provider="gemma4_mtp", draft_model="unused", candidate_budget=3),
        drafter=_FakeDrafter(runners[0]),
    )
    frames = [provider.generation_steps(tuple(range(1, length + 1)),
              GenerationRequest(prompts=((1,),), max_tokens=9, temperature=0, top_p=1, ignore_eos=True),
              runner=runner)
              for length, runner in zip((3, 7), runners)]
    pending = [next(frame) for frame in frames]
    outputs = [None, None]
    while any(frame is not None for frame in frames):
        for index, frame in enumerate(frames):
            if frame is None:
                continue
            operation = pending[index]
            result = provider.execute_operation(operation)
            try:
                pending[index] = frame.send(result)
            except StopIteration as done:
                outputs[index] = done.value
                frames[index] = None
    for length, output in zip((3, 7), outputs):
        assert output.generated_token_ids == tuple(_greedy_reference(length, 9))
    assert [runner.resets for runner in runners] == [1, 1]


class _ResidentFakeRunner(_FakeRunner):
    layer_count = 1

    def close(self):
        pass

    def hidden_state(self, row=-1):
        # Each new cycle primes from the committed cursor, not a peer's pass.
        return self.position

    def shared_kv(self, index):
        return SimpleNamespace(owner=id(self), position=self.position)


class _Head:
    def prime_batch(self, hidden_states):
        return np.asarray(hidden_states, dtype=np.int64).reshape(-1, 1)

    def forward_batch(self, token_ids, *, positions, shared_kv, recurrent_states):
        assert len(token_ids) == len(positions) == len(shared_kv)
        assert len({view[0].owner for view in shared_kv}) == len(token_ids)
        logits = np.full((len(token_ids), _VOCAB), -1.0, dtype=np.float32)
        for index, state in enumerate(recurrent_states):
            logits[index, _next_token(int(state[0]))] = 1
        return logits, recurrent_states + 1


class _Workspace:
    def __init__(self):
        self.widths = []

    def forward_batch(self, runners, token_rows, *, verification):
        self.widths.append(len(runners))
        assert len(set(map(id, runners))) == len(runners)
        return [runner.forward(tokens, logits_rows=len(tokens), verification=verification).reshape(len(tokens), -1)
                for runner, tokens in zip(runners, token_rows)]

    def close(self):
        pass


def _driver(capacity=4):
    generator = _FakeGenerator(_ResidentFakeRunner())
    generator.context_length = 128
    generator.validate_request = lambda request: None
    generator.supports_speculative_mtp = True
    drafter = _FakeDrafter(generator._runner)
    drafter.head = _Head()
    provider = Gemma4MTPTextProvider(target_generator=generator,
        config=SpeculativeProviderConfig(provider="gemma4_mtp", draft_model="unused", candidate_budget=3),
        drafter=drafter)
    generator._speculative_provider = provider
    resident = Gemma4ResidentModelRunner(generator, capacity=capacity)
    resident._workspace = _Workspace()
    resident._available = [_ResidentFakeRunner() for _ in range(capacity)]
    resident._owners = list(resident._available)
    generator.create_resident_model_runner = lambda **kwargs: resident
    driver = SubmitPollTextGenerator(generator, config=EngineLoopConfig(
        max_active_requests=capacity, max_prefill_chunk_tokens=128))
    return driver, resident


def _request(length, max_tokens=15, **kwargs):
    return GenerationRequest(prompts=(tuple(range(1, length + 1)),),
        max_tokens=max_tokens, temperature=0, top_p=1, ignore_eos=True, **kwargs)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_shared_scheduler_packs_mtp_and_preserves_ragged_request_ids(width):
    driver, resident = _driver()
    submissions = driver.submit_speculative_many_detailed([_request(3 + i, 11 + i) for i in range(width)])
    for _ in range(80):
        if all(driver.generation_complete(submission) for submission in submissions):
            break
        driver.poll(max_ticks=1)
    outputs = [driver.take_result(submission)[0] for submission in submissions]
    for i, output in enumerate(outputs):
        assert output.generated_token_ids == tuple(_greedy_reference(3 + i, 11 + i))
        assert output.telemetry.diagnostics["publication_mode"] == "resident_token_queue"
    assert width in resident._workspace.widths
    assert width in resident._proposal_widths
    assert len(resident._available) == 4
    assert resident.active_request_ids == ()


def test_staggered_cycles_pack_with_bounded_unpublished_lookahead():
    driver, resident = _driver(2)
    driver._loop.prefill_decode_policy = "fair"
    first = driver.submit_speculative_detailed(_request(3, 43))
    while resident._rows[first.request_ids[0]].published < 3:
        driver.poll(max_ticks=1)
    second = driver.submit_speculative_detailed(_request(7, 37))
    for _ in range(120):
        driver.poll(max_ticks=1)
        for row in resident._rows.values():
            if row.latest is not None:
                assert len(row.latest.generated_token_ids) - row.published <= 2 * (
                    resident.provider.candidate_budget + 1
                )
        if driver.generation_complete(first) and driver.generation_complete(second):
            break
    assert 2 in resident._workspace.widths
    assert driver.take_result(first)[0].generated_token_ids == tuple(_greedy_reference(3, 43))
    assert driver.take_result(second)[0].generated_token_ids == tuple(_greedy_reference(7, 37))


def test_cancellation_discards_private_tail_and_refill_preserves_survivor():
    driver, resident = _driver(2)
    first, survivor = driver.submit_speculative_many_detailed([_request(3, 23), _request(7, 25)])
    for _ in range(20):
        driver.poll(max_ticks=1)
        row = resident._rows.get(first.request_ids[0])
        if row is not None and row.latest is not None and 1 < row.published < len(row.latest.generated_token_ids):
            break
    else:
        pytest.fail("did not reach an unpublished speculative tail")
    published = row.published
    driver.cancel_submission(first)
    cancelled = driver.take_result(first)[0]
    assert len(cancelled.generated_token_ids) == published
    assert cancelled.finish_details.cancelled
    refill = driver.submit_speculative_detailed(_request(5, 13))
    for _ in range(80):
        if driver.generation_complete(survivor) and driver.generation_complete(refill):
            break
        driver.poll(max_ticks=1)
    assert driver.take_result(survivor)[0].generated_token_ids == tuple(_greedy_reference(7, 25))
    assert driver.take_result(refill)[0].generated_token_ids == tuple(_greedy_reference(5, 13))
    assert len(resident._available) == 2


def test_sampled_rows_keep_separate_rng_streams_across_batch_compositions():
    def run(width):
        driver, resident = _driver()
        requests = [replace(_request(3 + i, 19), temperature=0.7, seed=17 + i)
                    for i in range(width)]
        submissions = driver.submit_speculative_many_detailed(requests)
        for _ in range(100):
            if all(driver.generation_complete(item) for item in submissions):
                break
            driver.poll(max_ticks=1)
        return [driver.take_result(item)[0].generated_token_ids for item in submissions]
    assert run(4) == run(4)
    assert run(4)[0] == run(1)[0]


def test_resident_ar_opt_out_never_runs_the_assistant():
    driver, resident = _driver()
    submission = driver.submit_detailed(_request(5, 11))
    for _ in range(30):
        if driver.generation_complete(submission):
            break
        driver.poll(max_ticks=1)
    output = driver.take_result(submission)[0]
    assert output.generated_token_ids == tuple(_greedy_reference(5, 11))
    assert not resident._proposal_widths
    assert output.telemetry.decode_state.execution_path == "gemma4_resident_ar"
    # The opt-out reports the plain route, not the provider's MTP labels.
    assert output.finish_details.sampler_mode == "processed_argmax"
    assert output.telemetry.decode_state.sampler_mode == "processed_argmax"
    diagnostics = output.telemetry.diagnostics
    assert diagnostics["speculative_provider"] is None
    assert "verify_equivalence" not in diagnostics
    assert "target_verify" not in diagnostics


def test_sampled_resident_ar_opt_out_reports_host_sampling():
    driver, resident = _driver()
    submission = driver.submit_detailed(replace(_request(5, 11), temperature=0.7, seed=3))
    for _ in range(30):
        if driver.generation_complete(submission):
            break
        driver.poll(max_ticks=1)
    output = driver.take_result(submission)[0]
    assert not resident._proposal_widths
    assert output.finish_details.sampler_mode == "host_logits_sample"


def test_zero_tokens_reclaims_without_target_or_proposal_work():
    driver, resident = _driver()
    submission = driver.submit_speculative_detailed(_request(3, 0))
    for _ in range(5):
        if driver.generation_complete(submission):
            break
        driver.poll(max_ticks=1)
    output = driver.take_result(submission)[0]
    assert output.generated_token_ids == ()
    assert not resident._workspace.widths
    assert not resident._proposal_widths
    assert resident.active_request_ids == ()
    assert len(resident._available) == 4


def test_ar_int8_storage_is_not_silently_replaced_by_bf16():
    driver, resident = _driver()
    created = []

    def clone(source, *, logits_rows, storage="bf16", scale_dtype="fp16"):
        owner = _ResidentFakeRunner()
        owner.kv_storage_resolved = storage
        owner.kv_scale_dtype = scale_dtype
        owner.close = lambda: None
        created.append(owner)
        return owner

    resident._new_runner = clone
    for owner in resident._owners:
        owner.close = lambda: None
    submission = driver.submit_detailed(replace(_request(5, 11), kv_storage="int8_per_token_head"))
    for _ in range(30):
        if driver.generation_complete(submission):
            break
        driver.poll(max_ticks=1)
    output = driver.take_result(submission)[0]
    assert output.generated_token_ids == tuple(_greedy_reference(5, 11))
    assert created[0].kv_storage_resolved == "int8_per_token_head"
    assert not resident._workspace.widths
    assert not resident._proposal_widths


def test_mtp_int8_fails_by_capability_without_leaking_registration():
    driver, resident = _driver()
    with pytest.raises(NotImplementedError, match="BF16 shared KV"):
        driver.submit_speculative_detailed(replace(_request(5), kv_storage="int8_per_token_head"))
    assert resident.active_request_ids == ()
    assert len(resident._available) == 4


@pytest.mark.parametrize("stop_at", [1, 3, 7])
def test_resident_stream_suppresses_multi_token_stops_and_matches_blocking(stop_at):
    ids = _greedy_reference(3, 16)
    request = _request(3, 16, stop_token_sequences=(tuple(ids[stop_at:stop_at + 2]),),
                       logprobs=True, top_logprobs=2)
    driver, _ = _driver()
    service = EngineService(driver)
    try:
        chunks = list(service.stream_speculative_mtp_detailed(request))
        output = service.generate_speculative_mtp_detailed(request)[0]
    finally:
        service.close()
    assert "".join(chunk.text for chunk in chunks) == output.text
    assert tuple(record for chunk in chunks for record in chunk.token_logprobs) == output.token_logprobs
    assert output.finish_details.reason == "stop"
    assert len(output.generated_token_ids) == stop_at + 2


def test_server_prepare_accepts_context_keyword_and_checks_capacity():
    driver, resident = _driver()
    driver.prepare(max_sequence_length=128)
    with pytest.raises(ValueError, match="context capacity"):
        driver.prepare(max_sequence_length=129)
