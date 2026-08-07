"""Public registry and fail-closed generation tests for Maple."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from hipengine.generation.maple import (
    MapleGenerator,
    make_maple_generator_gfx1100,
    make_maple_generator_gfx1151,
)
from hipengine.generation.registry import GenerationRequest, resolve_text_generator
from hipengine.quant import MAPLE_TERNARY2, resolve_quant


class FakeTokenizer:
    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    def decode(self, token_ids, *, skip_special: bool = False) -> str:
        del skip_special
        return ",".join(str(int(token)) for token in token_ids)


class FakeRunner:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.native_prefill_rows: list[tuple[int, ...]] = []
        self.serial_prefill_rows: list[tuple[int, ...]] = []
        self.step_inputs: list[int] = []
        self.closed = False

    def reset(self) -> None:
        self.reset_calls += 1

    def prefill_native(self, token_ids):
        self.native_prefill_rows.append(tuple(int(token) for token in token_ids))
        return SimpleNamespace(token_id=10)

    def prefill(self, token_ids):
        self.serial_prefill_rows.append(tuple(int(token) for token in token_ids))
        return SimpleNamespace(token_id=10)

    def step(self, token_id: int):
        self.step_inputs.append(int(token_id))
        return SimpleNamespace(token_id={10: 11, 11: 2}.get(int(token_id), 2))

    def close(self) -> None:
        self.closed = True


def fake_generator() -> MapleGenerator:
    generator = object.__new__(MapleGenerator)
    generator.model_path = "/synthetic/maple"
    generator.weight_index = None
    generator.model_plugin = None
    generator.backend = "hip_gfx1151"
    generator.context_length = 16
    generator.tokenizer = FakeTokenizer()
    generator.checkpoint = SimpleNamespace(
        spec=SimpleNamespace(eos_token_id=2, sliding_window=512)
    )
    generator.last_generation_outputs = ()
    generator.last_generation_seconds = None
    generator._runner = FakeRunner()
    generator._load_seconds = 0.0
    generator._lock = threading.RLock()
    generator._closed = False
    return generator


def request(*, temperature: float = 0.0, max_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest(
        prompts=((4, 5, 6),),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        ignore_eos=False,
    )


def test_maple_generators_register_for_both_gfx11_backends() -> None:
    assert resolve_quant("maple_ternary2") is MAPLE_TERNARY2
    assert resolve_text_generator(
        model="maple", backend="hip_gfx1151", quant="maple_ternary2"
    ) is make_maple_generator_gfx1151
    assert resolve_text_generator(
        model="maple", backend="hip_gfx1100", quant="maple_ternary2"
    ) is make_maple_generator_gfx1100


def test_maple_generator_runs_greedy_prompt_and_stops_on_eos() -> None:
    generator = fake_generator()
    outputs = generator.generate_detailed(request())
    runner = generator._runner
    assert isinstance(runner, FakeRunner)
    assert outputs[0].generated_token_ids == (10, 11, 2)
    assert outputs[0].text == "10,11,2"
    assert outputs[0].finish_details is not None
    assert outputs[0].finish_details.reason == "stop"
    assert outputs[0].finish_details.eos_token_id == 2
    assert runner.reset_calls == 1
    assert runner.native_prefill_rows == [(4, 5, 6)]
    assert runner.serial_prefill_rows == []
    assert runner.step_inputs == [10, 11]
    assert generator.last_generation_seconds is not None


def test_maple_generator_uses_serial_fallback_beyond_native_swa_capacity() -> None:
    generator = fake_generator()
    generator.context_length = 600
    long_prompt = tuple(4 for _ in range(513))
    result = generator.generate_detailed(
        GenerationRequest(
            prompts=(long_prompt,),
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=False,
        )
    )
    runner = generator._runner
    assert result[0].generated_token_ids == (10,)
    assert isinstance(runner, FakeRunner)
    assert runner.native_prefill_rows == []
    assert runner.serial_prefill_rows == [long_prompt]


def test_maple_generator_fails_closed_for_sampling_and_context_overflow() -> None:
    generator = fake_generator()
    with pytest.raises(NotImplementedError, match="temperature must be 0"):
        generator.generate_detailed(request(temperature=0.8))

    generator.context_length = 5
    with pytest.raises(ValueError, match="exceeds context_length"):
        generator.generate_detailed(request(max_tokens=3))


def test_maple_generator_close_is_idempotent() -> None:
    generator = fake_generator()
    runner = generator._runner
    generator.close()
    generator.close()
    assert isinstance(runner, FakeRunner) and runner.closed
    with pytest.raises(RuntimeError, match="closed"):
        generator.generate_detailed(request())
