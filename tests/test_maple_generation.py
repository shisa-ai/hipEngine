"""Public registry and fail-closed generation tests for Maple."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from hipengine.dispatch import RequestState, WorkItem, WorkKind
from hipengine.generation.maple import (
    MapleGenerator,
    MapleResidentModelRunner,
    make_maple_generator_gfx1100,
    make_maple_generator_gfx1151,
)
from hipengine.generation.registry import GenerationRequest, resolve_text_generator
from hipengine.quant import MAPLE_TERNARY2, resolve_quant


class FakeTokenizer:
    encoder = object()

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
    generator.last_batch_generation = None
    generator._runner = FakeRunner()
    generator._resident_model_runner = None
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


def test_maple_generator_uses_native_prefill_beyond_swa_capacity() -> None:
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
    assert runner.native_prefill_rows == [long_prompt]
    assert runner.serial_prefill_rows == []


def test_maple_resident_runner_admits_prefill_decodes_and_reclaims_slots() -> None:
    class FakeBatchRunner:
        closed = False

        def __init__(self) -> None:
            self.prefills: list[tuple[int, tuple[int, ...]]] = []
            self.steps: list[tuple[list[int], list[bool]]] = []
            self.single_steps: list[tuple[int, int]] = []
            self.resets: list[int] = []

        def reset_request(self, slot: int) -> None:
            self.resets.append(int(slot))

        def prefill_request(self, slot: int, tokens):
            row = tuple(int(token) for token in tokens)
            self.prefills.append((int(slot), row))
            return SimpleNamespace(token_id=row[-1] + 10)

        def batch_step(self, token_ids, *, active_mask=None):
            ids = [int(token) for token in token_ids]
            active = [bool(value) for value in active_mask]
            self.steps.append((ids, active))
            return [token + 1 if active[index] else -1 for index, token in enumerate(ids)]

        def step_request(self, slot: int, token_id: int):
            self.single_steps.append((int(slot), int(token_id)))
            return SimpleNamespace(token_id=int(token_id) + 1)

        def close(self) -> None:
            self.closed = True

    class FakeDecodeStream:
        def __init__(self) -> None:
            self.first = True

        def step(self, encoder, token_id: int) -> str:
            del encoder
            prefix = "" if self.first else ","
            self.first = False
            return f"{prefix}{int(token_id)}"

    generator = fake_generator()
    generator.context_length = 32
    resident = generator.create_resident_model_runner(capacity=2)
    assert isinstance(resident, MapleResidentModelRunner)
    fake_batch = FakeBatchRunner()
    resident._batch = fake_batch
    resident._prepared = True
    prompts = ((4, 5), (6, 7, 8))
    generation = GenerationRequest(
        prompts=prompts,
        max_tokens=3,
        temperature=0.0,
        top_p=1.0,
        stop_token_ids=(16,),
        ignore_eos=True,
    )
    resident.register_batch((40, 41), generation, prompt_rows=prompts)
    for request_id, prompt in zip((40, 41), prompts, strict=True):
        resident.reserve_admission(
            RequestState.from_tokens(request_id, prompt, max_new_tokens=3)
        )
    for row in resident._rows.values():
        row.decoder = FakeDecodeStream()

    prefill = WorkItem(
        kind=WorkKind.PREFILL,
        request_ids=(40, 41),
        row_to_request=(40, 41),
        token_rows=prompts,
        slot_ids=(0, 1),
        active_mask=(True, True),
    )
    resident.prefill_batch(prefill, commit=True)
    assert fake_batch.prefills == [(0, (4, 5)), (1, (6, 7, 8))]

    decode = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(40, 41),
        row_to_request=(40, 41),
        slot_ids=(0, 1),
        active_mask=(True, True),
    )
    first = resident.decode_batch(decode, commit=True)
    assert [(event.token_id, event.finished) for event in first] == [
        (15, False),
        (18, False),
    ]
    assert fake_batch.steps == [([15, 18], [True, True])]
    second = resident.decode_batch(decode, commit=True)
    assert [(event.token_id, event.finished) for event in second] == [
        (16, True),
        (19, False),
    ]
    assert fake_batch.single_steps == [(1, 19)]
    resident.reclaim(
        SimpleNamespace(
            request_id=40,
            finish_reason="stop",
            finish_details=second[0].stream_chunk.finish_details,
            generated_tokens=(15, 16),
        )
    )
    lone_decode = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(41,),
        row_to_request=(41,),
        slot_ids=(1,),
        active_mask=(False, True),
    )
    third = resident.decode_batch(lone_decode, commit=True)
    assert [(event.token_id, event.finished) for event in third] == [(20, True)]
    assert [event.stream_chunk.text for event in (*first, *second, *third)] == [
        "15",
        "18",
        ",16",
        ",19",
        ",20",
    ]
    resident.reclaim(
        SimpleNamespace(
            request_id=41,
            finish_reason="length",
            finish_details=third[0].stream_chunk.finish_details,
            generated_tokens=(18, 19, 20),
        )
    )
    outputs = resident.take_outputs((40, 41))
    assert [output.generated_token_ids for output in outputs] == [
        (15, 16),
        (18, 19, 20),
    ]
    assert [output.text for output in outputs] == ["15,16", "18,19,20"]
    resident.finalize_batch(generation, (40, 41), outputs)
    assert generator.last_batch_generation["path"] == (
        "maple_scheduler_native_prefill_batch_decode"
    )
    assert fake_batch.resets == [0, 1, 0, 1]
    resident.close()
    assert fake_batch.closed


def test_maple_public_resident_path_matches_serial_across_swa_wrap(
    hip_test_target_arch,
) -> None:
    del hip_test_target_arch
    from hipengine.core.memory import memory_stats
    from hipengine.generation.engine_loop import (
        EngineLoopConfig,
        SubmitPollTextGenerator,
    )
    from hipengine.loading.maple import load_maple_checkpoint
    from hipengine.runtime.maple import MapleRunner

    try:
        checkpoint = load_maple_checkpoint("deepgrove/maple-preview-2bit-mlx")
    except Exception as exc:  # noqa: BLE001 - checkpoint missing
        pytest.skip(f"maple checkpoint unavailable: {exc}")
    prompts = (
        tuple(9_000 + index for index in range(19)),
        tuple(9_100 + ((index * 13) % 512) for index in range(520)),
    )
    expected: list[tuple[int, ...]] = []
    serial = MapleRunner.load(
        checkpoint, backend="hip_gfx1151", max_context=528
    )
    try:
        for prompt in prompts:
            serial.reset()
            result = serial.prefill_native(prompt)
            generated: list[int] = []
            for _ in range(4):
                generated.append(result.token_id)
                result = serial.step(result.token_id)
            expected.append(tuple(generated))
    finally:
        serial.close()

    generator = MapleGenerator(
        model_path=checkpoint.index.model_path,
        weight_index=checkpoint.index,
        model_plugin=SimpleNamespace(),
    )
    adapter = SubmitPollTextGenerator(
        generator,
        capacity=2,
        config=EngineLoopConfig(
            max_active_requests=2,
            max_prefill_chunk_tokens=256,
            prefill_decode_policy="protect_ttft",
        ),
    )
    try:
        outputs = adapter.generate_detailed(
            GenerationRequest(
                prompts=prompts,
                max_tokens=4,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=True,
            )
        )
        assert [output.generated_token_ids for output in outputs] == expected
        assert generator.last_batch_generation == {
            "path": "maple_scheduler_native_prefill_batch_decode",
            "backend": "hip_gfx1151",
            "quant": "maple_ternary2",
            "batch_size": 2,
            "group_rows": 2,
            "physical_batch_rows": 2,
            "request_ids": [0, 1],
            "prompt_lengths": [19, 520],
            "decode_steps": 4,
            "native_compact_prefill": False,
            "native_packed_slot_admission": True,
            "native_caware_decode": True,
            "serial_decode_fallback": False,
            "throughput_claim_eligible": True,
        }
        assert adapter.live_loop_snapshot()["runner"]["slot_to_request"] == [
            None,
            None,
        ]
    finally:
        adapter.close()

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


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
