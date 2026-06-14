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
)


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, prompt: str) -> list[int]:
        return {"first": [10, 11], "second": [20]}[prompt]

    def decode(self, ids) -> str:
        table = {1: "B", 2: "C", 3: "D", 16: "Q", 99: "<eos>"}
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
    assert _decode_state(outputs[0])["active_processors"] == ["forced_tokens_pending"]


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
    assert _decode_state(outputs[0])["active_processors"] == ["force_sequence_completion_token_sequences"]


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


def test_gguf_non_greedy_request_uses_host_logits_sampler(monkeypatch) -> None:
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
        "sampler_fallback_reason": "host_sampling_required",
        "sampler_mode": "host_logits_sample",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 12,
    }
    assert ("prefill", (10, 11), True) in calls
    assert ("step", 1, True) in calls
    assert not any(call[0] == "capture_decode_graph" for call in calls)


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


def test_gguf_stream_detailed_emits_live_sampled_telemetry(monkeypatch) -> None:
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
            "sampler_fallback_reason": "host_sampling_required",
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
            "sampler_fallback_reason": "host_sampling_required",
            "sampler_mode": "host_logits_sample",
            "full_vocab_logits_d2h": True,
            "logits_d2h_bytes": 12,
        },
    ]
    assert calls == [
        ("enter",),
        ("prefill", (10, 11), True),
        ("step", 1, True),
        ("exit", True),
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
