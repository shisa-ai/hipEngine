from __future__ import annotations

from types import SimpleNamespace

import hipengine.generation.qwen35_paro as qwen35
from hipengine.generation import GenerationRequest
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoAutoregressiveStepResult


def _request(prompts=("hello",), max_tokens=1, *, ignore_eos=False) -> GenerationRequest:
    return GenerationRequest(
        prompts=tuple(prompts),
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=ignore_eos,
    )


def _result(token_id: int, text: str) -> Qwen35ParoAutoregressiveStepResult:
    return Qwen35ParoAutoregressiveStepResult(token_id=token_id, token_text=text, logit=float(token_id))


def test_qwen35_paro_generator_runs_multi_token_resident_decode(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: 999 if token == "<|endoftext|>" else None)

        def __init__(self, runner, *, max_sequence_length):
            calls.append(("init", runner, max_sequence_length))
            self.outputs = iter([_result(100, "A"), _result(101, "B"), _result(102, "C")])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("close",))

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return next(self.outputs) if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)

    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner

    out = generator.generate(_request(max_tokens=3))

    assert out == ["ABC"]
    assert calls == [
        ("init", runner, 6),
        ("step", 10, 0, False),
        ("step", 11, 1, True),
        ("step", 100, 2, True),
        ("step", 101, 3, True),
        ("close",),
    ]


def test_qwen35_paro_generator_handles_zero_tokens_without_loading(monkeypatch) -> None:
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not load")))
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )

    assert generator.generate(_request(prompts=("a", "b"), max_tokens=0)) == ["", ""]


def test_qwen35_paro_generator_stops_on_eos(monkeypatch) -> None:
    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: 100 if token == "<|endoftext|>" else None)

        def __init__(self, runner, *, max_sequence_length):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def step(self, token_id: int, *, position: int, sample: bool = True):
            return _result(100, "<eos>") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (1, [1]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    assert generator.generate(_request(max_tokens=4, ignore_eos=False)) == ["<eos>"]
