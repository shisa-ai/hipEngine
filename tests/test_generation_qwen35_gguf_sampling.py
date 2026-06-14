from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hipengine.generation.qwen35_gguf as qwen35_gguf
from hipengine.generation import GenerationRequest


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
        "sampler_mode": "host_logits_sample",
    }
    assert ("prefill", (10, 11), True) in calls
    assert ("step", 1, True) in calls
    assert not any(call[0] == "capture_decode_graph" for call in calls)


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
        "sampler_mode": "host_logits_sample",
    }
    assert len([call for call in calls if call[0] == "step"]) == 1
