from __future__ import annotations

from types import SimpleNamespace

import hipengine.generation.qwen35_paro as qwen35
from hipengine.generation import GenerationRequest
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoAutoregressiveStepResult,
    estimate_qwen35_paro_kv_capacity,
    qwen35_paro_kv_bytes_per_token,
)


def _request(prompts=("hello",), max_tokens=1, *, ignore_eos=False, **overrides) -> GenerationRequest:
    values = {
        "prompts": tuple(prompts),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": ignore_eos,
    }
    values.update(overrides)
    return GenerationRequest(**values)


def _result(token_id: int, text: str) -> Qwen35ParoAutoregressiveStepResult:
    return Qwen35ParoAutoregressiveStepResult(token_id=token_id, token_text=text, logit=float(token_id))


def test_qwen35_paro_kv_capacity_estimate_reports_int8_max_below_model_context() -> None:
    config = SimpleNamespace(
        layer_types=("linear_attention",) * 30 + ("full_attention",) * 10,
        num_key_value_heads=2,
        head_dim=256,
        max_position_embeddings=262144,
    )
    bytes_per_token = qwen35_paro_kv_bytes_per_token(
        config,
        storage_dtype="int8_per_token_head",
        scale_dtype="fp16",
    )
    estimate = estimate_qwen35_paro_kv_capacity(
        config,
        available_bytes=bytes_per_token * 131072 + 512 * 1024**2,
        requested_context_tokens=8192,
        storage_dtype="int8_per_token_head",
        scale_dtype="fp16",
        reserve_bytes=512 * 1024**2,
    )

    assert bytes_per_token == 10320
    assert 0 < estimate.allocatable_context_tokens < 131072
    assert estimate.requested_context_overhead_bytes > 0
    assert estimate.requested_total_bytes == estimate.requested_kv_bytes + estimate.requested_context_overhead_bytes
    assert estimate.model_max_context_tokens == 262144
    assert estimate.fits_requested is True
    assert estimate.fits_model_max is False


def test_qwen35_paro_prepare_allocates_configured_resident_session(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length, kwargs["kv_policy"].storage_dtype.value))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner

    generator.prepare(
        max_sequence_length=131072,
        sampling_params=SimpleNamespace(
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp16",
            kv_scale_granularity="per_token_head",
        ),
    )

    assert calls == [("init", runner, 131072, "int8_per_token_head")]


def test_qwen35_paro_generator_runs_multi_token_resident_decode_graph(monkeypatch) -> None:
    calls = []

    class FakeGraph:
        def __enter__(self):
            calls.append(("graph_enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("graph_close",))

        def replay(self, steps: int):
            calls.append(("graph_replay", steps))

        def read_generated_token_ids(self, count: int):
            calls.append(("graph_read", count))
            return [101, 102]

    class FakeSession:
        tokenizer = SimpleNamespace(
            token_to_id=lambda token: 999 if token == "<|endoftext|>" else None,
            decode=lambda ids: {101: "B", 102: "C"}[int(ids[0])],
        )

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("close",))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def capture_decode_graph(self, *, position, steps_per_replay, max_replay_steps, record_steps):
            calls.append(
                (
                    "capture_decode_graph",
                    position,
                    steps_per_replay,
                    max_replay_steps,
                    record_steps,
                )
            )
            return FakeGraph()

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
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "length",
        "length_limit": 3,
        "sampler_mode": "greedy_fast",
    }
    assert calls == [
        ("init", runner, 4096),
        ("prefill_native", (10, 11), True),
        ("capture_decode_graph", 2, 1, 2, 2),
        ("graph_enter",),
        ("graph_replay", 2),
        ("graph_read", 2),
        ("graph_close",),
    ]


def test_qwen35_paro_generator_allows_inert_greedy_filters(monkeypatch) -> None:
    calls = []

    class FakeGraph:
        def __enter__(self):
            calls.append(("graph_enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("graph_close",))

        def replay(self, steps: int):
            calls.append(("graph_replay", steps))

        def read_generated_token_ids(self, count: int):
            return [101]

    class FakeSession:
        tokenizer = SimpleNamespace(
            token_to_id=lambda token: None,
            decode=lambda ids: {101: "B"}[int(ids[0])],
        )

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def capture_decode_graph(self, **kwargs):
            calls.append(("capture_decode_graph", kwargs["position"]))
            return FakeGraph()

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=2, top_p=0.5, top_k=4, min_p=0.5))

    assert out == ["AB"]
    assert ("graph_replay", 1) in calls



def test_qwen35_paro_generator_uses_host_sampler_for_non_greedy(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_host_sampler(self, params, state):
            calls.append(
                (
                    "configure_host_sampler",
                    None if params is None else params.temperature,
                    None if state is None else state.seed,
                    None if state is None else state.prompt_tokens,
                )
            )

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=2, temperature=0.7, seed=5))

    assert out == ["AB"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "length",
        "length_limit": 2,
        "sampler_mode": "host_logits_sample",
    }
    assert calls[0][0] == "configure_host_sampler"
    assert calls[0][1] == 0.7
    assert calls[0][3] == (10, 11)
    assert ("step", 100, 2, True) in calls
    assert calls[-1] == ("configure_host_sampler", None, None, None)



def test_qwen35_paro_generator_env_routes_supported_c1_request_to_native_sampler(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_native_sampler(self, params, state):
            calls.append(
                (
                    "configure_native_sampler",
                    None if params is None else params.temperature,
                    None if state is None else state.seed,
                    None if state is None else state.prompt_tokens,
                )
            )

        def configure_host_sampler(self, params, state):  # pragma: no cover - this path must not be used
            calls.append(("configure_host_sampler", params is None))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=2, temperature=0.7, top_k=4, seed=5))

    assert out == ["AB"]
    assert calls[0][0] == "configure_native_sampler"
    assert calls[0][1] == 0.7
    assert calls[0][3] == (10, 11)
    assert not any(call[0] == "configure_host_sampler" for call in calls)
    assert ("step", 100, 2, True) in calls
    assert calls[-1] == ("configure_native_sampler", None, None, None)



def test_qwen35_paro_host_sampler_stops_on_stop_token_id(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_host_sampler(self, params, state):
            calls.append(("configure_host_sampler", params is None))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=2, stop_token_ids=(100,)))

    assert out == ["A"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "stop",
        "stop_sequence": [100],
        "sampler_mode": "processed_argmax",
    }
    assert not any(call[0] == "step" for call in calls)



def test_qwen35_paro_host_sampler_stops_on_multi_token_stop_sequence(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_host_sampler(self, params, state):
            calls.append(("configure_host_sampler", params is None))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=3, stop_token_sequences=((100, 101),)))

    assert out == ["AB"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "stop",
        "stop_sequence": [100, 101],
        "sampler_mode": "processed_argmax",
    }
    assert len([call for call in calls if call[0] == "step"]) == 1



def test_qwen35_paro_generator_uses_scheduler_packed_prefill_for_prompt_batch(monkeypatch) -> None:
    calls = []
    token_rows = {"alpha": [10, 11], "beta": [20]}

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)
        block_size = 256

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(
                (
                    "init",
                    runner,
                    max_sequence_length,
                    kwargs.get("max_batch_size"),
                    kwargs["kv_policy"].storage_dtype.value,
                )
            )
            self.max_sequence_length = max_sequence_length
            self.max_batch_size = kwargs.get("max_batch_size", 1)

        def prefill_native_packed(self, slab, *, sample: bool = True):
            calls.append(
                (
                    "prefill_native_packed",
                    slab.request_ids,
                    slab.token_rows,
                    slab.physical_slot_ids,
                    sample,
                )
            )
            return tuple(
                _result(100 + request_id, {0: "A", 1: "B"}[request_id])
                for request_id in slab.request_ids
            )

        def step_batch_serial(self, token_ids, *, positions, slots, sample: bool = True):
            calls.append(
                ("step_batch_serial", tuple(token_ids), tuple(positions), tuple(slots), sample)
            )
            return (_result(200, "C"), _result(201, "D"))

        def batch_execution_metadata(
            self, *, scheduler_owned: bool = False, native_decode: bool = False
        ):
            calls.append(("batch_execution_metadata", scheduler_owned, native_decode))
            return SimpleNamespace(
                native_compact_prefill=True,
                native_caware_decode=False,
                throughput_claim_eligible=False,
            )

    monkeypatch.setattr(
        qwen35,
        "_select_token",
        lambda model, prompt, token_id: (token_rows[prompt][-1], token_rows[prompt]),
    )
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner

    out = generator.generate(_request(prompts=("alpha", "beta"), max_tokens=2))

    assert out == ["AC", "BD"]
    assert calls == [
        ("init", runner, 4096, 2, "bf16"),
        ("prefill_native_packed", (0, 1), ((10, 11), (20,)), (0, 1), True),
        ("step_batch_serial", (100, 101), (2, 1), (0, 1), True),
        ("batch_execution_metadata", True, False),
    ]
    assert generator.last_batch_generation == {
        "path": "scheduler_native_packed_prefill_serial_decode",
        "batch_size": 2,
        "request_ids": [0, 1],
        "prompt_lengths": [2, 1],
        "packed_prefill_slabs": [
            {
                "request_ids": [0, 1],
                "slot_ids": [0, 1],
                "rows": 3,
                "request_count": 2,
                "block_count": 1,
            }
        ],
        "decode_steps": 1,
        "native_decode_steps": 0,
        "serial_decode_fallback": True,
        "native_compact_prefill": True,
        "native_caware_decode": False,
        "throughput_claim_eligible": False,
    }


def test_qwen35_paro_sampled_batch_uses_scheduler_packed_prefill(monkeypatch) -> None:
    calls = []
    token_rows = {"alpha": [10, 11], "beta": [20]}

    class FakeSession:
        tokenizer = SimpleNamespace(
            token_to_id=lambda token: None,
            decode=lambda ids: {100: "A", 101: "B", 200: "C", 201: "D"}[int(ids[0])],
        )
        block_size = 256

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length, kwargs.get("max_batch_size")))
            self.max_sequence_length = max_sequence_length
            self.max_batch_size = kwargs.get("max_batch_size", 1)

        def configure_host_sampler_rows(self, params, states_by_slot):
            calls.append(
                (
                    "configure_host_sampler_rows",
                    None if params is None else params.temperature,
                    None
                    if states_by_slot is None
                    else {slot: tuple(state.generated_tokens) for slot, state in states_by_slot.items()},
                )
            )

        def prefill_native_packed(self, slab, *, sample: bool = True):
            calls.append(("prefill_native_packed", slab.request_ids, slab.physical_slot_ids, sample))
            return tuple(
                _result(100 + request_id, {0: "A", 1: "B"}[request_id])
                for request_id in slab.request_ids
            )

        def step_batch_serial(self, token_ids, *, positions, slots, sample: bool = True):
            calls.append(("step_batch_serial", tuple(token_ids), tuple(positions), tuple(slots), sample))
            return (_result(200, "C"), _result(201, "D"))

        def batch_execution_metadata(self, *, scheduler_owned: bool = False, native_decode: bool = False):
            calls.append(("batch_execution_metadata", scheduler_owned, native_decode))
            return SimpleNamespace(native_compact_prefill=True)

    monkeypatch.setattr(
        qwen35,
        "_select_token",
        lambda model, prompt, token_id: (token_rows[prompt][-1], token_rows[prompt]),
    )
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner

    out = generator.generate(_request(prompts=("alpha", "beta"), max_tokens=2, temperature=0.7, seed=5))

    assert out == ["AC", "BD"]
    assert calls == [
        ("init", runner, 4096, 2),
        ("configure_host_sampler_rows", 0.7, {0: (), 1: ()}),
        ("prefill_native_packed", (0, 1), (0, 1), True),
        ("configure_host_sampler_rows", 0.7, {0: (100,), 1: (101,)}),
        ("step_batch_serial", (100, 101), (2, 1), (0, 1), True),
        ("configure_host_sampler_rows", None, None),
        ("batch_execution_metadata", True, False),
    ]
    assert generator.last_batch_generation["path"] == "scheduler_native_packed_prefill_serial_host_sampler_decode"
    assert generator.last_batch_generation["native_compact_prefill"] is True


def test_qwen35_paro_generator_reuses_resident_session(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length))

        def reset(self):
            calls.append(("reset",))

        def close(self):
            calls.append(("close",))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner

    assert generator.generate(_request(max_tokens=1)) == ["A"]
    assert generator.generate(_request(max_tokens=1)) == ["A"]
    assert calls == [
        ("init", runner, 4096),
        ("prefill_native", (10, 11), True),
        ("reset",),
        ("prefill_native", (10, 11), True),
    ]


def test_qwen35_paro_generator_passes_int8_kv_policy_to_session(monkeypatch) -> None:
    captured = {}

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, kv_policy, kv_scale_dtype, kv_scale_granularity):
            captured["storage_dtype"] = kv_policy.storage_dtype.value
            captured["scale_dtype"] = kv_scale_dtype.value
            captured["scale_granularity"] = kv_scale_granularity

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill_native(self, token_ids, *, sample: bool = True):
            return _result(100, "A") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=1).__class__(
        prompts=("hello",),
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
        kv_storage="int8_per_token_head",
        kv_scale_dtype="fp32",
        kv_scale_granularity="per_token_head",
    ))

    assert out == ["A"]
    assert captured == {
        "storage_dtype": "int8_per_token_head",
        "scale_dtype": "fp32",
        "scale_granularity": "per_token_head",
    }



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

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill_native(self, token_ids, *, sample: bool = True):
            return _result(100, "<eos>") if sample else None

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
