from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.generation.deadline import GenerationCancellationToken, GenerationCancelled
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.registry import GenerationRequest, resolve_text_generator


class _Tokenizer:
    eos_token_id = 9

    def encode(self, text):
        return [1, 2] if text == "hello" else [3]

    def decode(self, ids, *, skip_special=False):
        del skip_special
        return ":".join(str(value) for value in ids)


class _Runner:
    max_sequence_length = 8

    def __init__(self):
        self.steps = []

    def prefill(self, tokens, **kwargs):
        self.steps.append(
            ("prefill", tuple(tokens), kwargs)
            if kwargs
            else ("prefill", tuple(tokens))
        )
        return SimpleNamespace(token_id=4)

    def step(self, token, **kwargs):
        self.steps.append(
            ("step", int(token), kwargs) if kwargs else ("step", int(token))
        )
        return SimpleNamespace(token_id={4: 5, 5: 9}.get(int(token), 9))

    def close(self):
        self.steps.append(("close",))


def test_qwen4_exp_generator_rejects_cancelled_request_before_runner_mutation() -> None:
    runner = _Runner()
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf", weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(), tokenizer=_Tokenizer(), runner=runner,
    )
    token = GenerationCancellationToken(); token.cancel()
    try:
        with pytest.raises(GenerationCancelled):
            generator.generate_detailed(_request(cancellation_token=token))
        assert runner.steps == []
    finally:
        generator.close()


def test_qwen4_exp_generator_runs_bounded_multimodal_override() -> None:
    class VisionTokenizer(_Tokenizer):
        def encode(self, text):
            if '<|image_pad|>' in text:
                return [1, 248056, 2]
            return super().encode(text)

    class Vision:
        def encode(self, image):
            assert image == 'image'
            return __import__('numpy').ones((1, 2560), dtype='float32')

    runner = _Runner()
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=VisionTokenizer(),
        runner=runner,
    )
    generator._vision_runner = Vision()
    try:
        output = generator.generate_multimodal_detailed(
            'describe', 'image', _request(prompts=('describe',), max_tokens=2)
        )
        assert output.generated_token_ids == (4, 5)
        assert runner.steps[0][0:2] == ('prefill', (1, 248056, 2))
        overrides = runner.steps[0][2]['embedding_overrides']
        assert tuple(overrides) == (1,)
        assert overrides[1].shape == (2560,)
        assert runner.steps[0][2]['mrope_positions'].shape == (3, 3)
        assert runner.steps[1][0:2] == ('step', 4)
        assert len(runner.steps[1][2]['rope_positions']) == 3
    finally:
        generator._vision_runner = None
        generator.close()


def test_qwen4_exp_generator_exposes_tokenizer_control_surface() -> None:
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=_Tokenizer(),
        runner=_Runner(),
    )
    try:
        assert generator.count_tokens("hello") == 2
        assert generator.tokenize("hello") == (1, 2)
        assert generator.detokenize((1, 2)) == "1:2"
    finally:
        generator.close()


def _request(**overrides):
    values = dict(
        prompts=("hello",),
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )
    values.update(overrides)
    return GenerationRequest(**values)


def test_qwen4_exp_generators_are_registered_for_local_and_unsloth_q4() -> None:
    for quant in ("gguf_q4_k_m", "gguf_ud_q4_k_xl"):
        assert callable(
            resolve_text_generator(
                model="qwen4_exp_gguf",
                backend="hip_gfx1151",
                quant=quant,
            )
        )


def test_qwen4_exp_generator_closes_resident_when_runner_construction_fails(
    monkeypatch,
) -> None:
    events: list[str] = []

    class Resident:
        def close(self):
            events.append("resident.close")

    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.discover_gguf_files",
        lambda path: (path,),
    )
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.GGUFReader",
        lambda path: SimpleNamespace(info=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.build_qwen4_exp_gguf_tensor_map",
        lambda infos: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.plan_qwen4_exp_residency",
        lambda model_map, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.materialize_qwen4_exp_weights",
        lambda readers, plan, backend: Resident(),
    )

    def fail_runner(*args, **kwargs):
        raise RuntimeError("injected runner allocation failure")

    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        fail_runner,
    )
    with pytest.raises(RuntimeError, match="injected"):
        Qwen4ExpGGUFTextGenerator(
            model_path="unused.gguf",
            weight_index=SimpleNamespace(),
            model_plugin=SimpleNamespace(),
            tokenizer=_Tokenizer(),
        )
    assert events == ["resident.close"]


def test_qwen4_exp_generator_runs_greedy_serial_and_stops_at_eos() -> None:
    runner = _Runner()
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        backend="hip_gfx1151",
        tokenizer=_Tokenizer(),
        runner=runner,
    )

    (output,) = generator.generate_detailed(_request())

    assert output.text == "4:5:9"
    assert output.generated_token_ids == (4, 5, 9)
    assert output.finish_details.reason == "eos"
    assert runner.steps == [
        ("prefill", (1, 2), {"capture_logits": False}),
        ("step", 4, {"capture_logits": False}),
        ("step", 5, {"capture_logits": False}),
    ]
    generator.close()
    assert runner.steps[-1] == ("close",)


def test_qwen4_exp_generator_accepts_exact_ids_and_length_finish() -> None:
    runner = _Runner()
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=_Tokenizer(),
        runner=runner,
    )
    (output,) = generator.generate_detailed(
        _request(prompts=((7, 8),), max_tokens=1, ignore_eos=True)
    )
    assert output.generated_token_ids == (4,)
    assert output.finish_details.reason == "length"
    assert runner.steps[0] == (
        "prefill", (7, 8), {"capture_logits": False}
    )


def test_qwen4_exp_generator_rejects_unqualified_sampling_and_capacity() -> None:
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=_Tokenizer(),
        runner=_Runner(),
    )
    with pytest.raises(ValueError, match="greedy"):
        generator.generate_detailed(_request(temperature=0.5))
    with pytest.raises(ValueError, match="capacity"):
        generator.generate_detailed(_request(prompts=((1, 2, 3, 4, 5, 6),), max_tokens=3))
