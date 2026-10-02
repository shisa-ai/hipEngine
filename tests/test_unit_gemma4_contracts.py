"""CPU-only request and lifecycle contracts for the Gemma 4 adapter."""

from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator
from hipengine.generation.registry import GenerationRequest


def _request(**kwargs):
    values = dict(prompts=((1, 2),), max_tokens=3, temperature=0, top_p=1, ignore_eos=False)
    values.update(kwargs)
    return GenerationRequest(**values)


def _generator(tokens=(8, 9, 10)):
    generator = Gemma4GGUFGenerator("/unused.gguf", object(), object())
    iterator = iter(tokens)
    calls = []
    generator._tokenizer = SimpleNamespace(
        eos_token_id=7, stop_token_ids=(7, 8, 9),
        decode=lambda ids, **kwargs: str(list(ids)),
    )
    def _forward_argmax(ids):
        calls.append(tuple(ids))
        return next(iterator)

    generator._runner = SimpleNamespace(
        reset=lambda: None,
        forward=lambda ids: calls.append(tuple(ids)),
        forward_argmax=_forward_argmax,
        next_token=lambda logits: next(iterator),
        # ``_ensure_runner`` reads the resolved storage to decide whether the
        # injected runner can serve the request; a stub must declare it.
        kv_storage_resolved="bf16",
        kv_scale_dtype_resolved=DType.FP16,
    )
    return generator, calls


@pytest.mark.parametrize("token", [7, 8, 9])
def test_default_generation_stops_at_all_model_end_tokens(token):
    generator, calls = _generator((token, 10, 11))
    output, = generator.generate_detailed(_request())
    assert output.generated_token_ids == (token,)
    assert output.finish_details.reason == "stop"
    assert calls == [(1, 2)]


def test_ignore_eos_ignores_model_end_tokens_but_not_explicit_stops():
    generator, _ = _generator((8, 9, 10))
    output, = generator.generate_detailed(_request(ignore_eos=True, stop_token_ids=(9,)))
    assert output.generated_token_ids == (8, 9)
    assert output.finish_details.reason == "stop"


def test_explicit_eos_overrides_model_end_tokens():
    generator, _ = _generator((8, 10, 11))
    output, = generator.generate_detailed(_request(eos_token_id=10))
    assert output.generated_token_ids == (8, 10)


def test_decode_does_not_forward_after_last_requested_token():
    generator, calls = _generator((10, 11, 12))
    output, = generator.generate_detailed(_request())
    assert output.generated_token_ids == (10, 11, 12)
    assert calls == [(1, 2), (10,), (11,)]


@pytest.mark.parametrize("options", [
    {"grammar": {"type": "json_object"}},
    {"thinking_close_token_ids": (12,), "thinking_hard_token_cap": 2},
    {"force_sequence_completion_token_sequences": ((12, 13),)},
])
def test_unimplemented_generation_controls_fail_before_device_work(options):
    generator, calls = _generator()
    with pytest.raises(NotImplementedError):
        generator.generate_detailed(_request(**options))
    assert calls == []


def test_failed_runner_construction_releases_loaded_weights(monkeypatch):
    import hipengine.generation.gemma4_gguf as module

    generator = Gemma4GGUFGenerator("/unused.gguf", object(), object())
    generator._reader = object()
    released = []
    weights = SimpleNamespace(free=lambda: released.append(True))
    monkeypatch.setattr(module, "load_gemma4_device_weights", lambda *a, **k: weights)

    def fail(**kwargs):
        raise MemoryError("fixture allocation failure")

    monkeypatch.setattr(module, "Gemma4Runner", fail)
    with pytest.raises(MemoryError):
        generator._ensure_runner()
    assert released == [True]
    assert generator._weights is None
    assert generator._runner is None


def test_kv_storage_defaults_to_bf16():
    from hipengine.generation.gemma4_gguf import _resolve_kv_storage

    assert _resolve_kv_storage(_request()) == ("bf16", "fp16", "per_token_head")
    assert _resolve_kv_storage(_request(kv_storage="auto")) == (
        "bf16",
        "fp16",
        "per_token_head",
    )


def test_int8_kv_storage_controls_resolve():
    from hipengine.generation.gemma4_gguf import _resolve_kv_storage

    assert _resolve_kv_storage(
        _request(kv_storage="int8_per_token_head", kv_scale_dtype="fp32")
    ) == ("int8_per_token_head", "fp32", "per_token_head")


def test_request_validation_accepts_the_implemented_int8_layout():
    generator, _ = _generator()
    generator._validate_request(
        _request(
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp32",
            kv_scale_granularity="per_token_head",
        )
    )


@pytest.mark.parametrize(
    "options",
    [
        {"kv_storage": "fp8"},
        {"kv_storage": "int8_per_token_head", "kv_scale_granularity": "per_channel"},
        {"kv_storage": "int8_per_token_head", "kv_scale_dtype": "bf16"},
    ],
)
def test_request_validation_rejects_unsupported_kv_storage_by_name(options):
    generator, _ = _generator()
    with pytest.raises(NotImplementedError) as excinfo:
        generator._validate_request(_request(**options))
    message = str(excinfo.value)
    assert "KV storage" in message or "INT8 KV storage" in message
