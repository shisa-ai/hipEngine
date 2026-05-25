from __future__ import annotations

from types import SimpleNamespace

import hipengine.generation.qwen35_paro as qwen35
from hipengine.generation import GenerationRequest
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoAutoregressiveStepResult,
    estimate_qwen35_paro_kv_capacity,
    qwen35_paro_kv_bytes_per_token,
)


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
    assert calls == [
        ("init", runner, 4096),
        ("prefill_native", (10, 11), True),
        ("capture_decode_graph", 2, 1, 2, 2),
        ("graph_enter",),
        ("graph_replay", 2),
        ("graph_read", 2),
        ("graph_close",),
    ]


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
