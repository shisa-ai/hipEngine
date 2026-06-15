from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.generation.qwen35_gguf as qwen35_gguf
from hipengine.generation import (
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationRequest,
    GenerationStreamChunk,
    TokenLogprob,
)


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, prompt: str) -> list[int]:
        return {"first": [10, 11], "second": [20], "{": [5], "}": [4]}[prompt]

    def decode(self, ids) -> str:
        table = {1: "B", 2: "C", 3: "D", 4: "}", 5: "{", 6: "X", 16: "Q", 99: "<eos>"}
        return "".join(table[int(token)] for token in ids)


def _generator() -> qwen35_gguf.Qwen35GGUFBringupGenerator:
    generator = qwen35_gguf.Qwen35GGUFBringupGenerator.__new__(
        qwen35_gguf.Qwen35GGUFBringupGenerator
    )
    generator.model_path = "/tmp/fake.gguf"
    generator.weight_index = SimpleNamespace()
    generator.model_plugin = SimpleNamespace()
    generator.tokenizer = _FakeTokenizer()
    return generator


def _request(**overrides) -> GenerationRequest:
    values = {
        "prompts": ("first",),
        "max_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
    }
    values.update(overrides)
    return GenerationRequest(**values)


def _decode_state(output):
    assert output.telemetry is not None
    return output.telemetry.to_json_dict()["decode_state"]


def test_gguf_sampled_thinking_budget_suppresses_tokenizer_eos(monkeypatch) -> None:
    logits = np.full((1, 100), -10.0, dtype=np.float32)
    logits[0, 2] = 1.0
    logits[0, 99] = 5.0

    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(token_id=99, logits=logits)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()

    outputs = generator.generate_detailed(
        _request(
            max_tokens=1,
            thinking_close_token_ids=(2,),
            thinking_hard_token_cap=5,
        )
    )

    assert outputs[0].text == "C"
    assert outputs[0].finish_details is not None
    assert outputs[0].finish_details.reason == "length"


def test_gguf_sampled_request_forced_token_overrides_logits(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(token_id=1, logits=np.array([[0.0, 10.0, 1.0]], dtype=np.float32))

        def step(self, token_id: int, *, return_logits=True):  # pragma: no cover - max_tokens=1
            raise AssertionError("forced-token fixture should finish after prefill")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()

    outputs = generator.generate_detailed(
        _request(
            max_tokens=1,
            forced_tokens_pending=(2,),
            forced_token_reason="tool_choice_required",
        )
    )

    assert outputs[0].text == "C"
    assert outputs[0].finish_details is not None
    assert outputs[0].finish_details.to_json_dict()["sampler_mode"] == "processed_argmax"
    decode_state = _decode_state(outputs[0])
    assert decode_state["active_processors"] == ["forced_tokens_pending"]
    assert decode_state["forced_token_id"] == 2
    assert decode_state["forced_token_reason"] == "tool_choice_required"
    assert decode_state["forced_tokens_remaining"] == 0


def test_gguf_json_object_close_forcing_goes_through_decode(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 5] = 10.0
            return SimpleNamespace(token_id=5, logits=logits)

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 6] = 10.0
            logits[0, 4] = 1.0
            return SimpleNamespace(token_id=6, logits=logits)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()

    outputs = generator.generate_detailed(_request(max_tokens=2, json_object_close_forcing=True))

    assert outputs[0].text == "{}"
    assert ("step", 5, True) in calls
    decode_state = _decode_state(outputs[0])
    assert decode_state["forced_token_id"] == 4
    assert decode_state["forced_token_reason"] == "json_object_close_forcing"
    assert decode_state["forced_tokens_remaining"] == 0
    assert "json_object_close_forcing" in decode_state["active_processors"]
    assert "json_object_close_forcing" in decode_state["sampler_fast_path_blockers"]


def test_gguf_sampled_post_thinking_forced_tokens_queue_after_close(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 2] = 5.0
            return SimpleNamespace(token_id=2, logits=logits)

        def step(self, token_id: int, *, return_logits=True):
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 1] = 10.0
            return SimpleNamespace(token_id=1, logits=logits)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()

    outputs = generator.generate_detailed(
        _request(
            max_tokens=3,
            thinking_close_token_ids=(2,),
            thinking_hard_token_cap=8,
            post_thinking_forced_tokens_pending=(3, 16),
            post_thinking_forced_token_reason="tool_choice_required",
        )
    )

    assert outputs[0].text == "CDQ"
    assert outputs[0].finish_details is not None
    assert outputs[0].finish_details.to_json_dict()["phase"] == "answer"
    assert _decode_state(outputs[0])["active_processors"] == [
        "thinking_budget",
        "post_thinking_forced_tokens_pending",
    ]


def test_gguf_sampled_force_sequence_completion_repairs_tool_close(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 3] = 5.0
            return SimpleNamespace(token_id=3, logits=logits)

        def step(self, token_id: int, *, return_logits=True):
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 1] = 10.0
            return SimpleNamespace(token_id=1, logits=logits)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()

    outputs = generator.generate_detailed(
        _request(
            max_tokens=2,
            force_sequence_completion_token_sequences=((3, 16),),
            force_sequence_completion_reason="tool_call_close_repair",
        )
    )

    assert outputs[0].text == "DQ"
    assert outputs[0].finish_details is not None
    assert outputs[0].finish_details.to_json_dict()["sampler_mode"] == "processed_argmax"
    decode_state = _decode_state(outputs[0])
    assert decode_state["active_processors"] == ["force_sequence_completion_token_sequences"]
    assert decode_state["force_sequence_completion_token_sequences"] == [[3, 16]]
    assert decode_state["force_sequence_completion_reason"] == "tool_call_close_repair"


def test_gguf_telemetry_reports_post_thinking_forced_queue_before_close() -> None:
    request = _request(
        thinking_close_token_ids=(2,),
        thinking_hard_token_cap=8,
        post_thinking_forced_tokens_pending=(3, 16),
        post_thinking_forced_token_reason="tool_choice_required",
    )
    state = qwen35_gguf._gguf_row_sampling_state(request, [10, 11], row_index=0)
    state.observe(1)

    telemetry = qwen35_gguf._gguf_telemetry(
        [10, 11],
        [1],
        request,
        row_index=0,
        sampling_state=state,
    )

    decode_state = telemetry.to_json_dict()["decode_state"]
    assert decode_state["phase"] == "think"
    assert decode_state["reasoning_tokens"] == 1
    assert decode_state["post_thinking_forced_tokens_pending"] == [3, 16]
    assert decode_state["post_thinking_forced_token_reason"] == "tool_choice_required"


def test_gguf_greedy_equivalent_request_keeps_graph_path(monkeypatch) -> None:
    calls = []

    class FakeGraph:
        def __enter__(self):
            calls.append(("graph_enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("graph_exit", exc_type is None))

        def replay(self, steps):
            calls.append(("graph_replay", int(steps)))

        def read_generated_token_ids(self, count):
            calls.append(("graph_read", int(count)))
            return [16]

    class FakeSession:
        def __init__(self, model_path):
            calls.append(("init", str(model_path)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=1,
                logits=np.array([[0.0, 1.0]], dtype=np.float32),
            )

        def capture_decode_graph(self, **kwargs):
            calls.append(("capture_decode_graph", kwargs["position"]))
            return FakeGraph()

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    out = generator.generate(_request(top_p=0.5, top_k=2, min_p=0.5))

    assert out == ["BQ"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "length",
        "length_limit": 2,
        "sampler_mode": "greedy_fast",
    }
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }
    assert ("prefill", (10, 11), False) in calls
    assert ("graph_replay", 1) in calls


@pytest.mark.parametrize(
    ("native_requested", "expected_fallback"),
    [
        (False, "host_sampling_required"),
        (True, "native_gpu_unsupported_request"),
    ],
)
def test_gguf_non_greedy_request_uses_host_logits_sampler(
    monkeypatch,
    native_requested: bool,
    expected_fallback: str,
) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path):
            calls.append(("init", str(model_path)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 1.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    if native_requested:
        monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    else:
        monkeypatch.delenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", raising=False)

    generator = _generator()
    out = generator.generate(_request(temperature=0.7, top_k=1, seed=5))

    assert out == ["BC"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "length",
        "length_limit": 2,
        "sampler_mode": "host_logits_sample",
    }
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_fast_path_blockers": ["temperature"],
        "sampler_fallback_reason": expected_fallback,
        "sampler_mode": "host_logits_sample",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 12,
    }
    assert ("prefill", (10, 11), True) in calls
    assert ("step", 1, True) in calls
    assert not any(call[0] == "capture_decode_graph" for call in calls)


def test_gguf_generate_detailed_records_scheduler_token_chunks_for_serial_rows(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 1.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.delenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", raising=False)

    generator = _generator()
    outputs = generator.generate_detailed(
        _request(
            prompts=("first", "second"),
            temperature=0.7,
            top_k=1,
            logprobs=True,
            top_logprobs=1,
            seed=5,
        )
    )

    assert [output.text for output in outputs] == ["BC", "BC"]
    batch = generator.last_batch_generation
    assert batch is not None
    assert {key: value for key, value in batch.items() if key != "scheduler_token_chunks"} == {
        "path": "gguf_serial_host_sampler_decode",
        "batch_size": 2,
        "request_ids": [0, 1],
        "prompt_lengths": [2, 1],
        "decode_steps": 2,
        "native_decode_steps": 0,
        "serial_decode_fallback": True,
        "native_compact_prefill": False,
        "native_caware_decode": False,
        "native_sampler_rows": False,
        "throughput_claim_eligible": False,
        "sampler_plan_metadata": [
            {
                "active_processors": [],
                "sampler_fast_path_blockers": ["temperature", "logprobs", "top_logprobs"],
                "native_gpu_available": False,
                "sampler_fallback_reason": "host_sampling_required",
                "sampler_mode": "host_logits_sample",
            },
            {
                "active_processors": [],
                "sampler_fast_path_blockers": ["temperature", "logprobs", "top_logprobs"],
                "native_gpu_available": False,
                "sampler_fallback_reason": "host_sampling_required",
                "sampler_mode": "host_logits_sample",
            },
        ],
    }
    chunks = batch["scheduler_token_chunks"]
    assert [
        (chunk["request_id"], chunk["token_index"], chunk["token_id"], chunk["chunk"]["text"])
        for chunk in chunks
    ] == [
        (0, 0, 1, "B"),
        (0, 1, 2, "C"),
        (1, 0, 1, "B"),
        (1, 1, 2, "C"),
    ]
    assert [chunk["finished"] for chunk in chunks] == [False, True, False, True]
    assert chunks[0]["chunk"]["token_logprobs"] == [
        {
            "token_id": 1,
            "token_text": "B",
            "logprob": 0.0,
            "top_logprobs": [{"token_id": 1, "token_text": "B", "logprob": 0.0}],
        }
    ]
    assert chunks[1]["chunk"]["finish_details"] == {
        "reason": "length",
        "length_limit": 2,
        "sampler_mode": "host_logits_sample",
    }
    assert chunks[2]["chunk"]["telemetry"]["decode_state"] == {
        "request_id": "1",
        "row_index": 1,
        "step_index": 1,
        "prompt_tokens": 1,
        "generated_tokens": 1,
        "phase": "answer",
        "continuation_eligible": False,
        "sampler_fast_path_blockers": ["temperature", "logprobs", "top_logprobs"],
        "sampler_fallback_reason": "host_sampling_required",
        "sampler_mode": "host_logits_sample",
        "execution_path": "gguf_serial_host_sampler_decode",
        "native_compact_prefill": False,
        "native_caware_decode": False,
        "serial_decode_fallback": True,
        "native_sampler_rows": False,
    }
    assert calls == [
        ("enter",),
        ("prefill", (10, 11), True),
        ("step", 1, True),
        ("prefill", (20,), True),
        ("step", 1, True),
        ("exit", True),
    ]


def test_gguf_stream_detailed_emits_live_greedy_telemetry(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path):
            calls.append(("init", str(model_path)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(token_id=1)

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(token_id=2)

        def capture_decode_graph(self, **kwargs):  # pragma: no cover - streaming should stay live
            raise AssertionError("streaming should emit live one-token steps")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    chunks = list(generator.stream_detailed(_request(max_tokens=2)))

    assert [chunk.text for chunk in chunks] == ["B", "C"]
    assert all(isinstance(chunk, GenerationStreamChunk) for chunk in chunks)
    assert [_decode_state(chunk) for chunk in chunks] == [
        {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 2,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_mode": "greedy_fast",
        },
        {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 2,
            "generated_tokens": 2,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_mode": "greedy_fast",
        },
    ]
    assert [None if chunk.finish_details is None else chunk.finish_details.to_json_dict() for chunk in chunks] == [
        None,
        {"reason": "length", "length_limit": 2, "sampler_mode": "greedy_fast"},
    ]
    assert calls == [
        ("init", "/tmp/fake.gguf"),
        ("enter",),
        ("prefill", (10, 11), False),
        ("step", 1, False),
        ("exit", True),
    ]


def test_gguf_stream_text_wrapper_preserves_plain_chunks(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(token_id=1)

        def step(self, token_id: int, *, return_logits=True):
            return SimpleNamespace(token_id=2)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()

    assert list(generator.stream(_request(max_tokens=2))) == ["B", "C"]


@pytest.mark.parametrize(
    ("native_requested", "expected_fallback"),
    [
        (False, "host_sampling_required"),
        (True, "native_gpu_unsupported_request"),
    ],
)
def test_gguf_stream_detailed_emits_live_sampled_telemetry(
    monkeypatch,
    native_requested: bool,
    expected_fallback: str,
) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 1.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    if native_requested:
        monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    else:
        monkeypatch.delenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", raising=False)

    generator = _generator()
    chunks = list(generator.stream_detailed(_request(temperature=0.7, top_k=1, seed=5)))

    assert [chunk.text for chunk in chunks] == ["B", "C"]
    assert [_decode_state(chunk) for chunk in chunks] == [
        {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 2,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": expected_fallback,
            "sampler_mode": "host_logits_sample",
            "full_vocab_logits_d2h": True,
            "logits_d2h_bytes": 12,
        },
        {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 2,
            "generated_tokens": 2,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": expected_fallback,
            "sampler_mode": "host_logits_sample",
            "full_vocab_logits_d2h": True,
            "logits_d2h_bytes": 12,
        },
    ]
    assert [None if chunk.finish_details is None else chunk.finish_details.to_json_dict() for chunk in chunks] == [
        None,
        {"reason": "length", "length_limit": 2, "sampler_mode": "host_logits_sample"},
    ]
    assert calls == [
        ("enter",),
        ("prefill", (10, 11), True),
        ("step", 1, True),
        ("exit", True),
    ]


def test_gguf_stream_detailed_emits_live_sampled_logprobs(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 1.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    chunks = list(
        generator.stream_detailed(
            _request(temperature=0.7, top_k=1, logprobs=True, top_logprobs=1, seed=5)
        )
    )

    assert [chunk.text for chunk in chunks] == ["B", "C"]
    assert chunks[0].token_logprobs == (
        TokenLogprob(
            token_id=1,
            token_text="B",
            logprob=0.0,
            top_logprobs=((1, "B", 0.0),),
        ),
    )
    assert chunks[1].token_logprobs == (
        TokenLogprob(
            token_id=2,
            token_text="C",
            logprob=0.0,
            top_logprobs=((2, "C", 0.0),),
        ),
    )
    assert [None if chunk.finish_details is None else chunk.finish_details.to_json_dict() for chunk in chunks] == [
        None,
        {"reason": "length", "length_limit": 2, "sampler_mode": "host_logits_sample"},
    ]


def test_gguf_stream_detailed_reports_thinking_budget_pressure(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):  # pragma: no cover - max_tokens=1
            raise AssertionError("hard-close stream fixture should finish after prefill")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    chunks = list(
        generator.stream_detailed(
            _request(
                max_tokens=1,
                thinking_close_token_ids=(2,),
                thinking_hard_token_cap=0,
            )
        )
    )

    assert [chunk.text for chunk in chunks] == ["C"]
    assert _decode_state(chunks[0]) == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "answer",
        "continuation_eligible": False,
        "reasoning_tokens": 1,
        "active_processors": ["thinking_budget"],
        "sampler_fast_path_blockers": ["thinking_budget"],
        "sampler_fallback_reason": "processed_logits_required",
        "forced_token_id": 2,
        "forced_token_reason": "thinking_hard_close",
        "forced_tokens_remaining": 0,
        "budget_pressure": "hard_close",
        "sampler_mode": "processed_argmax",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 12,
    }


def test_gguf_greedy_host_decode_checks_deadline_after_step(monkeypatch) -> None:
    calls = []

    def check_deadline(value) -> None:
        calls.append(("deadline", None if value is None else getattr(value, "deadline_at", value)))
        if ("step", 1, False) in calls:
            raise GenerationDeadlineExceeded(deadline_at=getattr(value, "deadline_at", value))

    class FakeSession:
        def __init__(self, model_path):
            calls.append(("init", str(model_path)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(token_id=1, logits=np.array([[0.0, 1.0]], dtype=np.float32))

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(token_id=2, logits=np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

        def capture_decode_graph(self, **kwargs):  # pragma: no cover - host decode forced
            raise AssertionError("host-routed decode should not capture graph")

    monkeypatch.setattr(qwen35_gguf, "raise_if_generation_deadline_expired", check_deadline)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setattr(qwen35_gguf, "_session_uses_host_routed_decode", lambda session: True)

    generator = _generator()
    with pytest.raises(GenerationDeadlineExceeded):
        generator.generate(_request(max_tokens=2, deadline_at=123.0))

    assert ("prefill", (10, 11), False) in calls
    assert ("step", 1, False) in calls
    assert ("exit", False) in calls


def test_gguf_greedy_host_decode_checks_cancellation_after_step(monkeypatch) -> None:
    calls = []
    token = GenerationCancellationToken()

    class FakeSession:
        def __init__(self, model_path):
            calls.append(("init", str(model_path)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(token_id=1, logits=np.array([[0.0, 1.0]], dtype=np.float32))

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            token.cancel()
            return SimpleNamespace(token_id=2, logits=np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

        def capture_decode_graph(self, **kwargs):  # pragma: no cover - host decode forced
            raise AssertionError("host-routed decode should not capture graph")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setattr(qwen35_gguf, "_session_uses_host_routed_decode", lambda session: True)

    generator = _generator()
    with pytest.raises(GenerationCancelled) as raised:
        generator.generate(_request(max_tokens=2, cancellation_token=token))

    assert raised.value.finish_details.to_json_dict() == {"reason": "cancelled", "cancelled": True}
    assert ("prefill", (10, 11), False) in calls
    assert ("step", 1, False) in calls
    assert ("exit", False) in calls


def test_gguf_finish_details_report_forced_thinking_close(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):  # pragma: no cover - max_tokens=1
            raise AssertionError("hard-close fixture should finish after prefill")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    out = generator.generate(
        _request(
            max_tokens=1,
            thinking_close_token_ids=(2,),
            thinking_hard_token_cap=0,
        )
    )

    assert out == ["C"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "thinking_budget_exhausted",
        "length_limit": 1,
        "forced_close": True,
        "reasoning_tokens": 1,
        "budget_pressure": "hard_close",
        "sampler_mode": "processed_argmax",
        "phase": "answer",
    }
    decode_state = _decode_state(generator.last_generation_outputs[0])
    assert decode_state["phase"] == "answer"
    assert decode_state["reasoning_tokens"] == 1
    assert decode_state["forced_token_id"] == 2
    assert decode_state["forced_token_reason"] == "thinking_hard_close"
    assert decode_state["forced_tokens_remaining"] == 0
    assert decode_state["budget_pressure"] == "hard_close"


def test_gguf_host_sampler_stops_on_stop_token_id(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(
                token_id=2,
                logits=np.array([[0.0, 0.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    out = generator.generate(_request(temperature=0.7, top_k=1, stop_token_ids=(1,)))

    assert out == ["B"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "stop",
        "stop_sequence": [1],
        "sampler_mode": "host_logits_sample",
    }
    assert not any(call[0] == "step" for call in calls)


def test_gguf_host_sampler_stops_on_multi_token_stop_sequence(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 0.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    out = generator.generate(
        _request(temperature=0.7, top_k=1, max_tokens=3, stop_token_sequences=((1, 2),))
    )

    assert out == ["BC"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "stop",
        "stop_sequence": [1, 2],
        "sampler_mode": "host_logits_sample",
    }
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "stop_suffix_state": {"matched_sequence": [1, 2]},
        "active_processors": ["stop_token_sequences"],
        "sampler_fast_path_blockers": ["temperature", "stop_token_sequences"],
        "sampler_fallback_reason": "host_sampling_required",
        "sampler_mode": "host_logits_sample",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 12,
    }
    assert len([call for call in calls if call[0] == "step"]) == 1
