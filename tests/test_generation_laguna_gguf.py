from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.generation import (
    GenerationCancellationToken,
    GenerationCancelled,
    EngineLoopConfig,
    GenerationDeadlineExceeded,
    GenerationAdmissionRejected,
    GenerationKey,
    GenerationRequest,
    GenerationStreamChunk,
    SubmitPollTextGenerator,
    registered_text_generators,
)
from hipengine.llm import LLM, SamplingParams
from hipengine.models.laguna import LAGUNA_GGUF


class _FakeTokenizer:
    eos_token_id = 2
    eot_token_id = 24
    stop_token_ids = (2, 24)
    _tokens = [chr(ord("a") + (index % 26)) for index in range(30)]
    _tokens[18] = "<think>"
    _tokens[19] = "</think>"
    _tokens[23] = "<assistant>"
    tokens = tuple(_tokens)
    token_to_id = {token: index for index, token in enumerate(tokens)}
    token_types = tuple(1 for _ in range(30))
    byte_decoder = {}

    _text = {
        2: "〈|EOS|〉",
        10: "A",
        11: "B",
        12: "C",
        13: "D",
        24: "</assistant>",
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text == "prompt":
            return [7, 8]
        if any(marker in text for marker in ("<think>", "</think>", "<assistant>")):
            ids = []
            cursor = 0
            while cursor < len(text):
                marker = next(
                    (
                        item
                        for item in ("<think>", "</think>", "<assistant>")
                        if text.startswith(item, cursor)
                    ),
                    None,
                )
                if marker is None:
                    ids.append(7)
                    cursor += 1
                else:
                    ids.append(self.token_to_id[marker])
                    cursor += len(marker)
            return ids
        return [int(part) for part in text.split()]

    def decode(self, token_ids, *, skip_special: bool = False) -> str:
        values = []
        for token_id in token_ids:
            token = int(token_id)
            if skip_special and token in self.stop_token_ids:
                continue
            values.append(self._text.get(token, f"T{token}"))
        return "".join(values)


class _FakeWeights:
    def __init__(self) -> None:
        self.config = SimpleNamespace(context_length=262_144)
        self.backend = "hip_gfx1151"
        self.freed = False

    def free(self, *, runtime=None) -> None:
        del runtime
        self.freed = True


class _FakeSession:
    sequences: list[tuple[int, ...]] = []
    events: list[tuple] = []
    prefill_hook = None
    resident_nbytes = 1_234

    def __init__(self, *, resident_weights, context_length, backend, runtime, **kwargs) -> None:
        del kwargs
        self.weights = resident_weights
        self.context_length = int(context_length)
        self.backend = str(backend)
        self.runtime = runtime
        self.sequence: tuple[int, ...] | None = None
        self.index = 0
        self.position = -1
        self.closed = False
        self.events.append(("open", self.context_length, self.backend))

    @staticmethod
    def _result(token_id: int):
        return SimpleNamespace(next_token_id=int(token_id), next_token_logit=1.0)

    def prefill(self, token_ids):
        self.events.append(("prefill", tuple(int(token) for token in token_ids)))
        self.sequence = self.sequences.pop(0)
        self.index = 0
        self.position += len(tuple(token_ids))
        if self.prefill_hook is not None:
            self.prefill_hook()
        token = self.sequence[self.index]
        self.index += 1
        return self._result(token)

    def forward_token(self, token_id: int):
        self.events.append(("forward", int(token_id)))
        assert self.sequence is not None
        token = self.sequence[self.index]
        self.index += 1
        self.position += 1
        return self._result(token)

    def reset_state(self) -> None:
        self.events.append(("reset",))
        self.sequence = None
        self.index = 0
        self.position = -1

    def close(self) -> None:
        self.closed = True
        self.events.append(("close",))


def _request(**overrides) -> GenerationRequest:
    values = {
        "prompts": ((7, 8),),
        "max_tokens": 3,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
    }
    values.update(overrides)
    return GenerationRequest(**values)


@pytest.fixture
def generator(monkeypatch, tmp_path):
    from hipengine.generation import laguna_gguf

    model = tmp_path / "laguna.gguf"
    model.touch()
    cache = model.with_suffix(".hipengine-repacked-v1")
    cache.mkdir()
    tokenizer = _FakeTokenizer()
    weights = _FakeWeights()
    materialize_calls = []

    monkeypatch.setattr(
        laguna_gguf.LagunaGGUFTokenizer,
        "from_gguf_info",
        classmethod(lambda cls, info: tokenizer),
    )

    def materialize(path, **kwargs):
        materialize_calls.append((Path(path), kwargs))
        return weights

    monkeypatch.setattr(laguna_gguf, "materialize_laguna_gguf_weights", materialize)
    monkeypatch.setattr(laguna_gguf, "LagunaGGUFResidentSession", _FakeSession)
    monkeypatch.setattr(
        laguna_gguf,
        "get_hip_runtime",
        lambda: SimpleNamespace(
            device_synchronize=lambda: _FakeSession.events.append(("sync",))
        ),
    )
    _FakeSession.sequences = []
    _FakeSession.events = []
    _FakeSession.prefill_hook = None
    instance = laguna_gguf.LagunaGGUFGenerator(
        model_path=model,
        weight_index=SimpleNamespace(metadata={}),
        model_plugin=LAGUNA_GGUF,
        backend="hip_gfx1151",
    )
    yield SimpleNamespace(
        instance=instance,
        tokenizer=tokenizer,
        weights=weights,
        materialize_calls=materialize_calls,
        model=model,
        cache=cache,
    )
    instance.close()


def test_laguna_generator_registers_concrete_gfx1151_key() -> None:
    from hipengine.generation import laguna_gguf  # noqa: F401

    assert GenerationKey("laguna_gguf", "hip_gfx1151", "gguf_q4_k_m") in set(
        registered_text_generators()
    )


def test_laguna_native_runner_admits_later_prefill_between_decode_ticks(generator) -> None:
    _FakeSession.sequences = [(10, 11, 12), (13, 14)]
    adapter = SubmitPollTextGenerator(
        generator.instance,
        capacity=2,
        config=EngineLoopConfig(
            max_active_requests=2,
            max_prefill_chunk_tokens=128,
            prefill_decode_policy="protect_ttft",
        ),
    )
    try:
        assert type(adapter._runner).__name__ == "LagunaGGUFResidentModelRunner"
        first = adapter.submit_detailed(_request(max_tokens=3))
        first_prefill = adapter.poll(max_ticks=1)
        first_decode = adapter.poll(max_ticks=1)
        one_active = adapter.live_loop_snapshot()
        second = adapter.submit_detailed(
            _request(prompts=((9, 8),), max_tokens=2)
        )
        delayed_arrival = adapter.poll(max_ticks=1)
        two_active = adapter.live_loop_snapshot()

        assert one_active["loop"]["physical_bucket"]["occupied_slots"] == 1
        assert one_active["runner"]["sessions"]["active"] == 1
        assert two_active["loop"]["physical_bucket"]["occupied_slots"] == 2
        assert two_active["runner"]["sessions"]["active"] == 2
        assert any(event.work_kind is not None and event.work_kind.value == "prefill" for event in first_prefill)
        assert [event.token_id for event in first_decode if event.kind == "token"] == [10]
        assert any(event.request_id == second.request_ids[0] and event.kind == "admitted" for event in delayed_arrival)
        assert any(
            event.work_kind is not None
            and event.work_kind.value == "prefill"
            and event.request_ids == second.request_ids
            for event in delayed_arrival
        )
        assert not adapter.generation_complete(first)

        while not adapter.generation_complete(first) or not adapter.generation_complete(second):
            adapter.poll(max_ticks=1)
        first_output = adapter.take_result(first)[0]
        second_output = adapter.take_result(second)[0]

        assert first_output.text == "ABC"
        assert first_output.generated_token_ids == (10, 11, 12)
        assert second_output.text == "DT14"
        assert second_output.generated_token_ids == (13, 14)
        snapshot = adapter.live_loop_snapshot()
        assert snapshot["runner"] == {
            "kind": "laguna_resident_model_runner",
            "capacity": 2,
            "sessions": {
                "resident": 2,
                "active": 0,
                "available": 2,
                "retained": 0,
            },
            "active_request_ids": [],
            "outputs_buffered": 0,
            "closed": False,
        }
        assert snapshot["loop"]["requests"]["reclaimed_total"] == 2
    finally:
        adapter.close()


def test_laguna_native_runner_protect_decode_bound_covers_staggered_rows(generator) -> None:
    _FakeSession.sequences = [tuple(range(10, 18)), tuple(range(10, 18))]
    adapter = SubmitPollTextGenerator(
        generator.instance,
        config=EngineLoopConfig(
            max_active_requests=2,
            prefill_decode_policy="protect_decode",
        ),
    )
    try:
        outputs = adapter.generate_detailed(
            _request(prompts=((7, 8), (9, 8)), max_tokens=8)
        )
        assert [output.generated_token_ids for output in outputs] == [
            tuple(range(10, 18)),
            tuple(range(10, 18)),
        ]
    finally:
        adapter.close()


def test_laguna_native_runner_routes_two_prompt_outputs_by_request_id(generator) -> None:
    _FakeSession.sequences = [(10, 11), (13, 14)]
    adapter = SubmitPollTextGenerator(generator.instance, capacity=2)
    try:
        outputs = adapter.generate_detailed(
            _request(prompts=((7, 8), (9, 8)), max_tokens=2)
        )

        assert [output.text for output in outputs] == ["AB", "DT14"]
        assert [output.generated_token_ids for output in outputs] == [
            (10, 11),
            (13, 14),
        ]
        assert generator.instance.last_batch_generation is not None
        assert generator.instance.last_batch_generation["batch_size"] == 2
        assert generator.instance.last_batch_generation["path"] == (
            "laguna_resident_scheduler_c1"
        )
        assert outputs[0].telemetry is not None
        assert outputs[0].telemetry.to_json_dict()["decode_state"]["execution_path"] == (
            "laguna_resident_scheduler_c1"
        )
    finally:
        adapter.close()


def test_laguna_native_runner_recovers_after_pending_overload(generator) -> None:
    _FakeSession.sequences = [(10,), (13,), (11,)]
    adapter = SubmitPollTextGenerator(
        generator.instance,
        config=EngineLoopConfig(
            max_active_requests=2,
            max_pending_requests=2,
            prefill_decode_policy="protect_ttft",
        ),
    )
    try:
        first = adapter.submit_detailed(_request(max_tokens=1))
        second = adapter.submit_detailed(
            _request(prompts=((9, 8),), max_tokens=1)
        )
        with pytest.raises(GenerationAdmissionRejected, match="pending request queue is full"):
            adapter.submit_detailed(_request(max_tokens=1))

        while not adapter.generation_complete(first) or not adapter.generation_complete(second):
            adapter.poll(max_ticks=1)
        assert adapter.take_result(first)[0].generated_token_ids == (10,)
        assert adapter.take_result(second)[0].generated_token_ids == (13,)

        recovered = adapter.generate_detailed(_request(max_tokens=1))[0]
        assert recovered.generated_token_ids == (11,)
        snapshot = adapter.live_loop_snapshot()
        assert snapshot["loop"]["requests"]["reclaimed_total"] == 3
        assert snapshot["runner"]["sessions"]["available"] == 2
    finally:
        adapter.close()


def test_laguna_native_runner_soaks_without_losing_session_ownership(generator) -> None:
    _FakeSession.sequences = [(10 + index % 4,) for index in range(20)]
    adapter = SubmitPollTextGenerator(generator.instance, capacity=2)
    try:
        observed = [
            adapter.generate_detailed(_request(max_tokens=1))[0].generated_token_ids
            for _ in range(20)
        ]
        assert observed == [(10 + index % 4,) for index in range(20)]
        snapshot = adapter.live_loop_snapshot()
        assert snapshot["runner"]["sessions"] == {
            "resident": 2,
            "active": 0,
            "available": 2,
            "retained": 0,
        }
        assert snapshot["loop"]["requests"]["reclaimed_total"] == 20
    finally:
        adapter.close()
    assert len([event for event in _FakeSession.events if event[0] == "close"]) == 2


def test_laguna_native_runner_streams_prefix_safe_exact_output(generator) -> None:
    _FakeSession.sequences = [(10, 11, 12)]
    adapter = SubmitPollTextGenerator(generator.instance, capacity=2)
    try:
        chunks = list(
            adapter.stream_detailed(
                _request(max_tokens=3, stop_token_sequences=((10, 24),))
            )
        )

        assert "".join(chunk.text for chunk in chunks) == "ABC"
        assert chunks[0].text == "AB"
        assert chunks[-1].finish_details is not None
        assert chunks[-1].finish_details.reason == "length"
        assert chunks[-1].generated_token_ids == (10, 11, 12)
    finally:
        adapter.close()


def test_laguna_native_runner_acknowledges_dispatched_cancel_after_reclaim(generator) -> None:
    _FakeSession.sequences = [(10, 11, 12)]
    token = GenerationCancellationToken()
    adapter = SubmitPollTextGenerator(generator.instance, capacity=2)
    try:
        submission = adapter.submit_detailed(
            _request(max_tokens=3, cancellation_token=token)
        )
        adapter.poll(max_ticks=1)
        adapter.poll(max_ticks=1)

        token.cancel()
        assert token.cancel_requested is True
        assert token.cancelled is False
        adapter.poll(max_ticks=1)

        assert token.cancelled is True
        assert adapter.generation_complete(submission)
        output = adapter.take_result(submission)[0]
        assert output.finish_details is not None
        assert output.finish_details.reason == "cancelled"
        assert adapter._runner.active_request_ids == ()
        assert len(adapter._runner._available) == 2
    finally:
        adapter.close()


def test_laguna_native_runner_cancels_between_decode_ticks_and_reclaims(generator) -> None:
    _FakeSession.sequences = [(10, 11, 12)]
    adapter = SubmitPollTextGenerator(generator.instance, capacity=2)
    try:
        submission = adapter.submit_detailed(_request(max_tokens=3))
        adapter.poll(max_ticks=1)
        adapter.poll(max_ticks=1)

        assert adapter.cancel_submission(submission) == (True,)
        assert adapter.generation_complete(submission)
        output = adapter.take_result(submission)[0]
        assert output.text == "A"
        assert output.generated_token_ids == (10,)
        assert output.finish_details is not None
        assert output.finish_details.reason == "cancelled"
        assert adapter._runner.active_request_ids == ()
        assert len(adapter._runner._available) == 2
    finally:
        adapter.close()


def test_laguna_native_runner_preserves_stateful_kv_continuation(generator) -> None:
    _FakeSession.sequences = [(10, 11), (13,)]
    adapter = SubmitPollTextGenerator(generator.instance, capacity=2)
    try:
        first = adapter.generate_detailed(
            _request(
                prompts=((1, 2, 3),),
                max_tokens=2,
                resident_session_key="principal:chat",
                resident_session_cache_action="append_visible_only",
            )
        )[0]
        second = adapter.generate_detailed(
            _request(
                prompts=((1, 2, 3, 10, 7),),
                max_tokens=1,
                resident_session_key="principal:chat",
                resident_session_cache_action="append_visible_only",
            )
        )[0]

        assert first.generated_token_ids == (10, 11)
        assert second.generated_token_ids == (13,)
        assert second.telemetry is not None
        assert second.telemetry.diagnostics["resident_kv_reused"] is True
        assert second.telemetry.diagnostics["prefix_reused_tokens"] == 4
        assert second.telemetry.diagnostics["session_prepare_mode"] == "reuse"
        calls = [event[1] for event in _FakeSession.events if event[0] == "prefill"]
        assert calls == [(1, 2, 3), (7,)]
    finally:
        adapter.close()


def test_laguna_generator_exposes_poolside_v1_chat_reasoning_contract(generator) -> None:
    prompt = generator.instance.render_chat_prompt(
        [{"role": "user", "content": "Reply with OK."}],
        enable_thinking=True,
    )

    assert prompt.endswith("<assistant><think>")
    assert prompt.startswith("〈|EOS|〉<system>")
    assert generator.instance.chat_template_family == "poolside_v1"
    assert generator.instance.reasoning_parser_name == "poolside_v1"
    assert generator.instance.tool_parser_name == "poolside_v1"
    assert generator.instance.chat_tool_parser.name == "poolside_v1"
    assert generator.instance.chat_reasoning_parser.initially_open(prompt) is True


def test_laguna_blocking_generation_suppresses_eot_and_retains_weights(generator) -> None:
    _FakeSession.sequences = [(10, 11, 24)]

    output = generator.instance.generate_detailed(_request())[0]

    assert output.text == "AB"
    assert output.generated_token_ids == (10, 11, 24)
    assert output.finish_details is not None
    assert output.finish_details.to_json_dict() == {
        "reason": "stop",
        "stop_sequence": [24],
        "sampler_mode": "greedy_fast",
    }
    assert output.telemetry is not None
    state = output.telemetry.to_json_dict()["decode_state"]
    assert state["execution_path"] == "laguna_eager_c1"
    assert state["prompt_tokens"] == 2
    assert state["generated_tokens"] == 3
    assert output.telemetry.timing is not None
    assert output.telemetry.timing["tokenize_ms"] == 0.0
    assert generator.weights.freed is False
    assert generator.materialize_calls[0][1]["repacked_cache"] == generator.cache
    assert [event for event in _FakeSession.events if event[0] == "open"] == [
        ("open", 4_096, "hip_gfx1151")
    ]
    assert ("close",) not in _FakeSession.events
    assert output.telemetry.diagnostics is not None
    assert output.telemetry.diagnostics["session_prepare_mode"] == "create"
    assert output.telemetry.timing["session_prepare_ms"] >= 0.0

    generator.instance.close()
    assert generator.weights.freed is True
    assert _FakeSession.events[-1] == ("close",)


def test_laguna_sequential_requests_reuse_one_reset_session(generator) -> None:
    _FakeSession.sequences = [(10,), (11,)]

    first = generator.instance.generate_detailed(_request(max_tokens=1))[0]
    second = generator.instance.generate_detailed(_request(max_tokens=1))[0]

    assert first.generated_token_ids == (10,)
    assert second.generated_token_ids == (11,)
    assert sum(event[0] == "open" for event in _FakeSession.events) == 1
    assert sum(event[0] == "reset" for event in _FakeSession.events) == 1
    assert ("close",) not in _FakeSession.events
    assert first.telemetry is not None
    assert second.telemetry is not None
    assert first.telemetry.diagnostics is not None
    assert second.telemetry.diagnostics is not None
    assert first.telemetry.diagnostics["session_prepare_mode"] == "create"
    assert second.telemetry.diagnostics["session_prepare_mode"] == "reset"
    assert second.telemetry.timing is not None
    assert second.telemetry.timing["session_prepare_ms"] >= 0.0


def test_laguna_text_prompt_uses_tokenizer_without_implicit_bos(generator) -> None:
    _FakeSession.sequences = [(10,)]

    output = generator.instance.generate_detailed(_request(prompts=("prompt",), max_tokens=1))[0]

    assert output.generated_token_ids == (10,)
    assert ("prefill", (7, 8)) in _FakeSession.events
    assert output.telemetry is not None
    assert output.telemetry.timing is not None
    assert output.telemetry.timing["tokenize_ms"] > 0.0


def test_laguna_prepared_prompt_preserves_server_preprocessing_telemetry(generator) -> None:
    from hipengine.generation.registry import PreparedPromptInput

    _FakeSession.sequences = [(10,)]
    prepared = PreparedPromptInput(
        source_text="prompt",
        token_ids=(7, 8),
        tokenize_ms=1.25,
        render_ms=0.75,
        admission_prepare_ms=0.5,
        tokenizer_identity="test.tokenizer",
    )

    output = generator.instance.generate_detailed(
        _request(prompts=(prepared,), max_tokens=1)
    )[0]

    assert output.generated_token_ids == (10,)
    assert ("prefill", (7, 8)) in _FakeSession.events
    assert output.telemetry is not None
    assert output.telemetry.timing is not None
    assert output.telemetry.timing["tokenize_ms"] == 1.25
    assert output.telemetry.timing["prompt_encode_ms"] == 1.25
    assert output.telemetry.timing["render_ms"] == 0.75
    assert output.telemetry.timing["admission_prepare_ms"] == 0.5


def test_laguna_stateful_session_reuses_exact_committed_prefix_and_pending_token(generator) -> None:
    _FakeSession.sequences = [(10, 11), (13,)]
    session_fields = {
        "resident_session_key": "principal-session-hash",
        "resident_session_cache_action": "append_visible_only",
    }

    first = generator.instance.generate_detailed(
        _request(max_tokens=2, **session_fields)
    )[0]
    second = generator.instance.generate_detailed(
        _request(
            prompts=((7, 8, 10, 11, 12),),
            max_tokens=1,
            **session_fields,
        )
    )[0]

    assert first.generated_token_ids == (10, 11)
    assert second.generated_token_ids == (13,)
    assert [event for event in _FakeSession.events if event[0] == "prefill"] == [
        ("prefill", (7, 8)),
        ("prefill", (11, 12)),
    ]
    assert not any(event[0] == "reset" for event in _FakeSession.events)
    assert second.telemetry is not None
    assert second.telemetry.diagnostics is not None
    assert second.telemetry.diagnostics["session_prepare_mode"] == "reuse"
    assert second.telemetry.diagnostics["resident_kv_reused"] is True
    assert second.telemetry.diagnostics["prefix_reused_tokens"] == 3


@pytest.mark.parametrize(
    ("second_key", "second_prompt", "first_cache_action"),
    [
        ("principal-session-hash", (7, 9, 10, 11, 12), "append_visible_only"),
        ("different-principal-or-session", (7, 8, 10, 11, 12), "append_visible_only"),
        ("principal-session-hash", (7, 8, 10, 11, 12), "append_none"),
    ],
)
def test_laguna_stateful_session_falls_back_to_reset_on_unsafe_reuse(
    generator,
    second_key,
    second_prompt,
    first_cache_action,
) -> None:
    _FakeSession.sequences = [(10, 11), (13,)]
    generator.instance.generate_detailed(
        _request(
            max_tokens=2,
            resident_session_key="principal-session-hash",
            resident_session_cache_action=first_cache_action,
        )
    )

    second = generator.instance.generate_detailed(
        _request(
            prompts=(second_prompt,),
            max_tokens=1,
            resident_session_key=second_key,
            resident_session_cache_action="append_visible_only",
        )
    )[0]

    assert ("reset",) in _FakeSession.events
    assert ("prefill", second_prompt) in _FakeSession.events
    assert second.telemetry is not None
    assert second.telemetry.diagnostics is not None
    assert second.telemetry.diagnostics["session_prepare_mode"] == "reset"
    assert second.telemetry.diagnostics["resident_kv_reused"] is False
    assert second.telemetry.diagnostics["prefix_reused_tokens"] == 0


def test_laguna_prepare_eagerly_materializes_pooled_session(generator) -> None:
    _FakeSession.sequences = [(10,)]

    prepared = generator.instance.prepare(max_sequence_length=128)
    output = generator.instance.generate_detailed(_request(max_tokens=1))[0]

    assert prepared == 128
    assert output.generated_token_ids == (10,)
    assert sum(event[0] == "open" for event in _FakeSession.events) == 1
    assert sum(event[0] == "reset" for event in _FakeSession.events) == 1
    assert output.telemetry is not None
    assert output.telemetry.diagnostics is not None
    assert output.telemetry.diagnostics["session_prepare_mode"] == "reset"


def test_laguna_native_scratch_probe_reports_existing_resident_slots(generator) -> None:
    adapter = SubmitPollTextGenerator(generator.instance, capacity=2)
    try:
        result = generator.instance.prepare_request_scratch(
            max_prompt_tokens=55,
            max_new_tokens=32,
            max_batch_size=2,
        )
        assert result == {
            "schema": 1,
            "backend": "hip_gfx1151",
            "execution_path": "laguna_resident_scheduler_c1",
            "max_batch_size": 2,
            "max_sequence_length": 86,
            "resident_session_nbytes": 2_468,
            "released_after_probe": False,
        }
        assert len([event for event in _FakeSession.events if event[0] == "open"]) == 2
        assert not [event for event in _FakeSession.events if event[0] == "close"]
    finally:
        adapter.close()


def test_laguna_prepare_request_scratch_sizes_resident_c1_slots(generator) -> None:
    _FakeSession.sequences = [(10,)]

    result = generator.instance.prepare_request_scratch(
        max_prompt_tokens=55,
        max_new_tokens=32,
        max_batch_size=1,
    )

    assert result == {
        "schema": 1,
        "backend": "hip_gfx1151",
        "execution_path": "laguna_eager_c1",
        "max_batch_size": 1,
        "max_sequence_length": 86,
        "resident_session_nbytes": 1_234,
        "released_after_probe": True,
    }
    assert _FakeSession.events[-1] == ("close",)
    two_rows = generator.instance.prepare_request_scratch(
        max_prompt_tokens=55,
        max_new_tokens=32,
        max_batch_size=2,
    )
    assert two_rows["max_batch_size"] == 2
    assert two_rows["resident_session_nbytes"] == 2_468
    with pytest.raises(NotImplementedError, match="at most 2 active c=1 rows"):
        generator.instance.prepare_request_scratch(
            max_prompt_tokens=55,
            max_new_tokens=32,
            max_batch_size=3,
        )


def test_laguna_stream_matches_blocking_and_finishes_with_cumulative_ids(generator) -> None:
    _FakeSession.sequences = [(10, 11, 24)]

    chunks = list(generator.instance.stream_detailed(_request()))

    assert "".join(chunk.text for chunk in chunks) == "AB"
    assert chunks[-1].text == ""
    assert chunks[-1].generated_token_ids == (10, 11, 24)
    assert chunks[-1].finish_details is not None
    assert chunks[-1].finish_details.stop_sequence == (24,)
    assert chunks[-1].telemetry is not None
    assert chunks[-1].telemetry.timing is not None
    assert chunks[-1].telemetry.timing["tokenize_ms"] == 0.0
    assert all(chunk.finish_details is None for chunk in chunks[:-1])
    assert ("close",) not in _FakeSession.events


def test_laguna_stream_emits_nonmatching_token_before_long_stop_holdback(generator) -> None:
    _FakeSession.sequences = [(10, 11, 12)]
    stream = generator.instance.stream_detailed(
        _request(
            max_tokens=3,
            stop_token_sequences=((13, 14, 15, 16),),
        )
    )

    first = next(stream)

    assert first.text == "A"
    assert not any(event[0] == "forward" for event in _FakeSession.events)
    assert first.finish_details is None
    assert first.generated_token_ids is None
    assert "".join(chunk.text for chunk in stream) == "BC"


def test_laguna_stream_flushes_failed_stop_prefix_at_first_safe_token(generator) -> None:
    _FakeSession.sequences = [(13, 10, 11)]
    stream = generator.instance.stream_detailed(
        _request(max_tokens=3, stop_token_sequences=((13, 14, 15),))
    )

    first = next(stream)

    assert first.text == "DA"
    assert [event for event in _FakeSession.events if event[0] == "forward"] == [
        ("forward", 13)
    ]
    assert "".join(chunk.text for chunk in stream) == "B"


def test_laguna_stream_preserves_overlapping_stop_suffixes_and_blocking_text(generator) -> None:
    sequence = (13, 14, 13, 14, 15)
    stops = ((13, 14, 16), (13, 14, 15))
    _FakeSession.sequences = [sequence, sequence]

    chunks = list(
        generator.instance.stream_detailed(
            _request(max_tokens=len(sequence), stop_token_sequences=stops)
        )
    )
    blocking = generator.instance.generate_detailed(
        _request(max_tokens=len(sequence), stop_token_sequences=stops)
    )[0]

    assert [chunk.text for chunk in chunks] == ["DT14", ""]
    assert "".join(chunk.text for chunk in chunks) == blocking.text == "DT14"
    assert chunks[-1].finish_details is not None
    assert chunks[-1].finish_details.stop_sequence == (13, 14, 15)
    assert chunks[-1].generated_token_ids == sequence


@pytest.mark.parametrize(
    ("sequence", "stops", "min_tokens", "expected_text", "expected_stop"),
    [
        ((10, 11, 12), ((10, 11, 12), (10, 11, 13)), 0, "", (10, 11, 12)),
        ((13, 14), ((14,), (13, 14)), 0, "D", (14,)),
        ((13, 14), ((13, 14), (14,)), 0, "", (13, 14)),
        ((13, 14), ((13, 14, 15),), 0, "DT14", ()),
        ((10, 11, 12), ((10, 11),), 3, "ABC", ()),
    ],
)
def test_laguna_stream_stop_edges_match_blocking(
    generator,
    sequence,
    stops,
    min_tokens,
    expected_text,
    expected_stop,
) -> None:
    _FakeSession.sequences = [sequence, sequence]
    request = _request(
        max_tokens=len(sequence),
        min_tokens=min_tokens,
        eos_token_id=2 if min_tokens else None,
        stop_token_sequences=stops,
    )

    chunks = list(generator.instance.stream_detailed(request))
    blocking = generator.instance.generate_detailed(request)[0]

    assert "".join(chunk.text for chunk in chunks) == blocking.text == expected_text
    assert chunks[-1].finish_details is not None
    assert blocking.finish_details is not None
    assert chunks[-1].finish_details.to_json_dict() == blocking.finish_details.to_json_dict()
    assert chunks[-1].finish_details.stop_sequence == expected_stop
    assert chunks[-1].generated_token_ids == sequence


def test_laguna_prefix_aware_stream_keeps_split_utf8_valid(generator) -> None:
    tokens = list(generator.tokenizer.tokens)
    tokens[10], tokens[11], tokens[12] = "Ã", "©", "x"
    generator.tokenizer.tokens = tuple(tokens)
    generator.tokenizer.byte_decoder = {"Ã": 0xC3, "©": 0xA9, "x": 0x78}
    _FakeSession.sequences = [(10, 11, 12)]
    stream = generator.instance.stream_detailed(
        _request(max_tokens=3, stop_token_sequences=((13, 14, 15, 16),))
    )

    first = next(stream)

    assert first.text == "é"
    assert [event for event in _FakeSession.events if event[0] == "forward"] == [
        ("forward", 10)
    ]
    assert "".join(chunk.text for chunk in stream) == "x"


def test_laguna_max_tokens_and_multitoken_stop_are_exact(generator) -> None:
    _FakeSession.sequences = [(10, 11), (10, 11, 12)]

    length = generator.instance.generate_detailed(_request(max_tokens=2))[0]
    stopped = generator.instance.generate_detailed(_request(stop_token_sequences=((10, 11),)))[0]

    assert length.text == "AB"
    assert length.generated_token_ids == (10, 11)
    assert length.finish_details is not None
    assert length.finish_details.to_json_dict() == {
        "reason": "length",
        "length_limit": 2,
        "sampler_mode": "greedy_fast",
    }
    assert stopped.text == ""
    assert stopped.generated_token_ids == (10, 11)
    assert stopped.finish_details is not None
    assert stopped.finish_details.stop_sequence == (10, 11)


def test_laguna_ignore_eos_continues_but_never_leaks_eot_markup(generator) -> None:
    _FakeSession.sequences = [(24, 10)]

    output = generator.instance.generate_detailed(_request(max_tokens=2, ignore_eos=True))[0]

    assert output.text == "A"
    assert output.generated_token_ids == (24, 10)
    assert output.finish_details is not None
    assert output.finish_details.reason == "length"


def test_laguna_unsupported_sampling_and_batch_fail_before_loading(generator) -> None:
    with pytest.raises(NotImplementedError, match="greedy"):
        generator.instance.generate_detailed(_request(temperature=0.5))
    with pytest.raises(ValueError, match="exactly one prompt"):
        generator.instance.generate_detailed(_request(prompts=((7,), (8,))))

    assert generator.materialize_calls == []


def test_laguna_cancellation_retires_session_and_next_request_recreates(generator) -> None:
    token = GenerationCancellationToken()
    _FakeSession.sequences = [(10, 11), (12,)]
    _FakeSession.prefill_hook = token.cancel

    with pytest.raises(GenerationCancelled):
        generator.instance.generate_detailed(_request(max_tokens=2, cancellation_token=token))

    assert _FakeSession.events[-1] == ("close",)
    assert generator.weights.freed is False

    _FakeSession.prefill_hook = None
    output = generator.instance.generate_detailed(_request(max_tokens=1))[0]

    assert output.generated_token_ids == (12,)
    assert sum(event[0] == "open" for event in _FakeSession.events) == 2
    assert output.telemetry is not None
    assert output.telemetry.diagnostics is not None
    assert output.telemetry.diagnostics["session_prepare_mode"] == "recreate_after_error"


def test_laguna_prefill_error_retires_session_before_recreate(generator) -> None:
    _FakeSession.sequences = [(10,), (11,)]

    def fail_prefill(*_args) -> None:
        raise RuntimeError("synthetic prefill failure")

    _FakeSession.prefill_hook = fail_prefill
    with pytest.raises(RuntimeError, match="synthetic prefill failure"):
        generator.instance.generate_detailed(_request(max_tokens=1))

    assert _FakeSession.events[-1] == ("close",)
    _FakeSession.prefill_hook = None
    output = generator.instance.generate_detailed(_request(max_tokens=1))[0]

    assert output.generated_token_ids == (11,)
    assert output.telemetry is not None
    assert output.telemetry.diagnostics is not None
    assert output.telemetry.diagnostics["session_prepare_mode"] == "recreate_after_error"


def test_laguna_abandoned_stream_retires_session(generator) -> None:
    _FakeSession.sequences = [(10, 11)]
    stream = generator.instance.stream_detailed(_request(max_tokens=2))

    first = next(stream)
    stream.close()

    assert first.text == "A"
    assert _FakeSession.events[-1] == ("close",)


def test_laguna_expired_deadline_and_context_overflow_fail_closed(generator) -> None:
    with pytest.raises(GenerationDeadlineExceeded):
        generator.instance.generate_detailed(_request(deadline_at=time.perf_counter() - 1.0))
    with pytest.raises(ValueError, match="4096"):
        generator.instance.generate_detailed(_request(prompts=(tuple(range(4_096)),), max_tokens=2))

    assert generator.materialize_calls == []


def test_laguna_tokenizer_hooks_and_zero_token_request_do_not_load(generator) -> None:
    assert generator.instance.tokenize("prompt") == (7, 8)
    assert generator.instance.count_tokens("prompt") == 2
    assert generator.instance.detokenize((10, 11)) == "AB"

    output = generator.instance.generate_detailed(
        _request(prompts=("prompt",), max_tokens=0)
    )[0]
    assert output.text == ""
    assert output.generated_token_ids == ()
    assert output.finish_details is not None
    assert output.finish_details.reason == "length"
    assert output.telemetry is not None
    assert output.telemetry.timing is not None
    assert output.telemetry.timing["tokenize_ms"] > 0.0
    assert generator.materialize_calls == []


def test_laguna_server_metadata_reports_resolved_model_backend_and_quant(
    generator, monkeypatch
) -> None:
    from fastapi.testclient import TestClient

    from hipengine.generation import laguna_gguf
    from hipengine.server import ServerConfig, create_app

    llm = LLM(str(generator.model), backend="hip_gfx1151")
    monkeypatch.setattr(
        llm,
        "_load_model_metadata",
        lambda: (SimpleNamespace(metadata={}), LAGUNA_GGUF),
    )
    monkeypatch.setattr(
        laguna_gguf.LagunaGGUFTokenizer,
        "from_gguf_info",
        classmethod(lambda cls, info: generator.tokenizer),
    )
    llm._get_text_generator()
    app = create_app(
        ServerConfig(
            model=str(generator.model),
            served_model_name="laguna-s-2.1",
            backend="hip_gfx1151",
            quant="auto",
            eager_load=False,
            max_active_requests=1,
        ),
        llm=llm,
    )

    with TestClient(app) as client:
        model = client.get("/v1/models").json()["data"][0]["hipengine"]
        capabilities = client.get("/v1/hipengine/capabilities").json()
        ready = client.get("/ready")

    assert model["backend"] == "hip_gfx1151"
    assert model["quant"] == "gguf_q4_k_m"
    assert capabilities["model"]["backend"] == "hip_gfx1151"
    assert capabilities["model"]["quant"] == "gguf_q4_k_m"
    assert ready.status_code == 200


def test_laguna_generator_closes_pooled_session_before_shared_weights(generator) -> None:
    events = _FakeSession.events
    original_free = generator.weights.free

    def record_free(*, runtime=None) -> None:
        events.append(("weights_free",))
        original_free(runtime=runtime)

    generator.weights.free = record_free
    generator.instance.prepare(max_sequence_length=128)
    generator.instance.close()

    assert events[-2:] == [("close",), ("weights_free",)]


def test_laguna_generator_attaches_one_generic_provider_and_closes_it_before_weights(
    generator,
) -> None:
    events: list[str] = []

    class FakeProvider:
        provider_name = "fake_dflash"

        def generate_detailed(self, request):
            events.append("generate")
            return [SimpleNamespace(text="spec")]

        def stream_detailed(self, request):
            events.append("stream")
            yield GenerationStreamChunk(text="chunk")

        def capabilities(self):
            return {"provider": self.provider_name}

        def close(self) -> None:
            events.append("provider_close")

    original_free = generator.weights.free

    def record_free(*, runtime=None) -> None:
        events.append("weights_free")
        original_free(runtime=runtime)

    generator.weights.free = record_free
    provider = FakeProvider()
    generator.instance.attach_speculative_provider(provider)

    assert generator.instance.supports_speculative is True
    assert generator.instance.speculative_capabilities() == {
        "provider": "fake_dflash"
    }
    assert generator.instance.generate_speculative_detailed(_request())[0].text == "spec"
    assert [chunk.text for chunk in generator.instance.stream_speculative_detailed(_request())] == [
        "chunk"
    ]
    with pytest.raises(RuntimeError, match="already attached"):
        generator.instance.attach_speculative_provider(FakeProvider())

    generator.instance.prepare(max_sequence_length=2)
    generator.instance.close()

    assert events[-2:] == ["provider_close", "weights_free"]


def test_laguna_public_llm_resolves_generator_and_close_releases_weights(
    generator, monkeypatch
) -> None:
    from hipengine.generation import laguna_gguf

    _FakeSession.sequences = [(10, 11)]
    llm = LLM(str(generator.model), backend="hip_gfx1151")
    monkeypatch.setattr(
        llm, "_load_model_metadata", lambda: (SimpleNamespace(metadata={}), LAGUNA_GGUF)
    )
    monkeypatch.setattr(
        laguna_gguf.LagunaGGUFTokenizer,
        "from_gguf_info",
        classmethod(lambda cls, info: generator.tokenizer),
    )

    outputs = llm.generate_detailed((7, 8), SamplingParams(max_tokens=2))

    assert outputs[0].text == "AB"
    assert outputs[0].generated_token_ids == (10, 11)
    llm.close()
    assert generator.weights.freed is True
