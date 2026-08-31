from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.qwen4_exp_gguf import (
    Qwen4ExpGGUFTextGenerator,
    Qwen4ExpResidentServingRunner,
)
from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.registry import GenerationRequest


class _Tokenizer:
    eos_token_id = 99

    def encode(self, text):
        return [ord(char) % 32 + 1 for char in str(text)]

    def decode(self, ids, *, skip_special=False):
        del skip_special
        return ":".join(str(value) for value in ids)


class _FakeRunner:
    max_sequence_length = 64
    prefill_chunk_size = 8
    runtime = None

    def __init__(self, bias=0):
        self.bias = bias
        self.position = 0
        self.closed = False
        self.calls = []

    def reset(self):
        self.position = 0

    def prefill(self, tokens, **kwargs):
        self.position = len(tokens)
        self.calls.append(("prefill", dict(kwargs)))
        return SimpleNamespace(token_id=sum(tokens) % 50 + self.bias)

    def step(self, token, **kwargs):
        self.position += 1
        self.calls.append(("step", dict(kwargs)))
        return SimpleNamespace(token_id=int(token) + 1)

    def close(self):
        self.closed = True


def _generator():
    runner = _FakeRunner()
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf", weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(), tokenizer=_Tokenizer(), runner=runner,
    )
    generator._resident = SimpleNamespace(close=lambda: None)
    return generator


def test_qwen4_exp_resident_c2_preserves_request_owned_state(monkeypatch) -> None:
    generator = _generator()
    created = []

    def make_runner(*args, **kwargs):
        del args, kwargs
        runner = _FakeRunner(bias=10)
        created.append(runner)
        return runner

    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        make_runner,
    )
    driver = SubmitPollTextGenerator(
        generator,
        config=EngineLoopConfig(max_active_requests=2, max_prefill_chunk_tokens=8),
    )
    try:
        request = GenerationRequest(
            prompts=("alpha", "beta"), max_tokens=3, temperature=0.0,
            top_p=1.0, ignore_eos=True
        )
        native = generator._resident_model_runner
        assert isinstance(native, Qwen4ExpResidentServingRunner)
        outputs = driver.generate_detailed(request)
        assert len(outputs) == 2
        assert all(len(output.generated_token_ids or ()) == 3 for output in outputs)
        assert outputs[0].generated_token_ids != outputs[1].generated_token_ids
        assert len(native._all_runners) == 2
        assert len(created) == 1
    finally:
        driver.close()


def test_qwen4_exp_resident_c2_uses_compact_output_when_bound(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_DEVICE_ARGMAX", "1")
    generator = _generator()
    created = []

    def make_runner(*args, **kwargs):
        del args, kwargs
        runner = _FakeRunner(bias=10)
        created.append(runner)
        return runner

    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        make_runner,
    )
    driver = SubmitPollTextGenerator(
        generator,
        config=EngineLoopConfig(max_active_requests=2, max_prefill_chunk_tokens=8),
    )
    try:
        outputs = driver.generate_detailed(
            GenerationRequest(
                prompts=("alpha", "beta"), max_tokens=3, temperature=0.0,
                top_p=1.0, ignore_eos=True,
            )
        )
        assert len(outputs) == 2
        native = generator._resident_model_runner
        assert native is not None
        runners = native._all_runners
        assert len(runners) == 2
        assert len(created) == 1
        assert all(runner.calls for runner in runners)
        assert all(
            kwargs == {"capture_logits": False}
            for runner in runners
            for _, kwargs in runner.calls
        )
    finally:
        driver.close()


def test_qwen4_exp_resident_admission_rollback_and_cancellation(monkeypatch) -> None:
    generator = _generator()
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        lambda *args, **kwargs: _FakeRunner(bias=10),
    )
    native = generator.create_resident_model_runner(capacity=2)
    request = GenerationRequest(
        prompts=("a", "b", "c"), max_tokens=2, temperature=0.0,
        top_p=1.0, ignore_eos=True,
    )
    native.register_batch((1, 2, 3), request, prompt_rows=((1,), (2,), (3,)))
    native.reserve_admission(SimpleNamespace(request_id=1))
    native.reserve_admission(SimpleNamespace(request_id=2))
    assert native._row(1).runner is not native._row(2).runner
    with pytest.raises(RuntimeError, match="no free request runner"):
        native.reserve_admission(SimpleNamespace(request_id=3))
    native.rollback_admission(SimpleNamespace(request_id=1))
    native.reserve_admission(SimpleNamespace(request_id=3))
    assert native._row(3).runner is not None
    native.discard((1, 2, 3))
    native.close()

    generator = _generator()
    driver = SubmitPollTextGenerator(
        generator,
        config=EngineLoopConfig(max_active_requests=1, max_prefill_chunk_tokens=8),
    )
    token = GenerationCancellationToken()
    token.cancel()
    cancelled = GenerationRequest(
        prompts=("cancel",), max_tokens=2, temperature=0.0, top_p=1.0,
        ignore_eos=True, cancellation_token=token,
    )
    try:
        outputs = driver.generate_detailed(cancelled)
        assert len(outputs) == 1
        assert outputs[0].generated_token_ids == ()
        assert outputs[0].finish_details.reason == "cancelled"
        native = generator._resident_model_runner
        assert native is not None
        assert all(row.runner is None for row in native._rows.values())
    finally:
        driver.close()


def test_qwen4_exp_generator_advertises_bounded_native_capacity() -> None:
    generator = _generator()
    try:
        assert generator.server_plain_ar_max_active_requests == 2
        runner = generator.create_resident_model_runner(capacity=2)
        assert runner.capacity == 2
    finally:
        generator.close()
