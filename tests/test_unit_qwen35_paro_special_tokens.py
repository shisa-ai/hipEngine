from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from hipengine.generation.qwen35_paro import _tokenizer_eos_ids
from hipengine.models.qwen35 import QWEN35_PARO_MOE
from hipengine.runtime.qwen35_paro_runner import _load_tokenizer


def _checkpoint(tmp_path, *, generation=None, tokenizer_config=None, config=None):
    encoder = Tokenizer(WordLevel(
        {"[UNK]": 0, "hello": 1, "<|im_end|>": 2, "<|endoftext|>": 3},
        unk_token="[UNK]",
    ))
    encoder.save(str(tmp_path / "tokenizer.json"))
    for name, payload in (
        ("generation_config.json", generation),
        ("tokenizer_config.json", tokenizer_config),
        ("config.json", config),
    ):
        if payload is not None:
            (tmp_path / name).write_text(json.dumps(payload))
    return tmp_path


def test_model_plugin_loads_normalized_special_tokens_with_generation_precedence(tmp_path):
    model = _checkpoint(
        tmp_path,
        generation={"eos_token_id": [3, "2", 3, -1, None], "bos_token_id": 0},
        tokenizer_config={"eos_token": "<|im_end|>", "pad_token": {"content": "[UNK]"}},
        config={"text_config": {"eos_token_id": 1, "bos_token_id": 1, "pad_token_id": 1}},
    )
    tokenizer = QWEN35_PARO_MOE.load_tokenizer(model)
    assert _tokenizer_eos_ids(tokenizer) == (3, 2)
    assert tokenizer.bos_token_id == 0
    assert tokenizer.pad_token_id == 0
    assert tokenizer.encode("hello").ids == [1]
    assert tokenizer.decode([1]) == "hello"
    assert tokenizer.token_to_id("<|im_end|>") == 2


def test_metadata_eos_is_authoritative_and_runtime_uses_plugin(tmp_path):
    model = _checkpoint(tmp_path, generation={"eos_token_id": 1})
    tokenizer = _load_tokenizer(model)
    assert tokenizer is not None
    assert _tokenizer_eos_ids(tokenizer) == (1,)


def test_tokenizer_metadata_and_nested_model_config_are_supported(tmp_path):
    model = _checkpoint(
        tmp_path,
        tokenizer_config={"eos_token": {"content": "<|im_end|>"}},
        config={"text_config": {"bos_token_id": 3, "pad_token_id": 0}},
    )
    tokenizer = QWEN35_PARO_MOE.load_tokenizer(model)
    assert _tokenizer_eos_ids(tokenizer) == (2,)
    assert tokenizer.bos_token_id == 3
    assert tokenizer.pad_token_id == 0


def test_metadata_free_checkpoints_resolve_qwen_markers_at_load_time(tmp_path):
    tokenizer = QWEN35_PARO_MOE.load_tokenizer(_checkpoint(tmp_path))
    assert _tokenizer_eos_ids(tokenizer) == (2, 3)
    assert _load_tokenizer(tmp_path / "missing") is None


def test_generation_does_not_look_up_marker_strings_after_loading():
    looked_up = []

    def unexpected_lookup(token):
        looked_up.append(token)
        raise AssertionError(f"generation looked up {token}")

    tokenizer = SimpleNamespace(eos_token_id=(7, 8), token_to_id=unexpected_lookup)
    assert _tokenizer_eos_ids(tokenizer) == (7, 8)
    # A bare tokenizer with no metadata no longer triggers generator-local inference.
    assert _tokenizer_eos_ids(SimpleNamespace(token_to_id=unexpected_lookup)) == ()
    assert looked_up == []


@pytest.mark.parametrize("streaming,prompt_count", [(False, 1), (False, 2), (True, 1)])
@pytest.mark.parametrize("processed", [False, True])
def test_explicit_request_eos_stops_public_generator(monkeypatch, streaming, prompt_count, processed):
    from hipengine.generation import GenerationRequest
    from hipengine.generation import qwen35_paro as generation
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoAutoregressiveStepResult

    class Session:
        tokenizer = SimpleNamespace(eos_token_id=(2, 3), decode=lambda ids: "EOS")

        def __init__(self, *args, **kwargs):
            pass

        def reset(self):
            pass

        def configure_host_sampler(self, params, state):
            pass

        def prefill_native(self, tokens, *, sample=True):
            return Qwen35ParoAutoregressiveStepResult(token_id=7, token_text="EOS", logit=1.0)

        def greedy_decode_graph_eligible(self):
            return False

        def step(self, *args, **kwargs):
            raise AssertionError("decode continued past the explicit EOS")

    monkeypatch.setattr(generation, "Qwen35ParoResidentSession", Session)
    monkeypatch.setattr(
        generation, "_native_batch_width_profile_for_runner", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(generation, "_exact_hybrid_c2_route_blockers", lambda **kwargs: ("test c1 route",))
    generator = generation.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model", weight_index=SimpleNamespace(), model_plugin=SimpleNamespace(),
    )
    generator._runner = SimpleNamespace(backend="cpu_reference", target_arch="cpu", model=generator.model_path)
    request = GenerationRequest(
        prompts=((10, 11, 12, 13),) * prompt_count, max_tokens=3, eos_token_id=7,
        temperature=0.0, top_p=1.0, ignore_eos=False,
        forced_tokens_pending=(7,) if processed else (),
    )
    if streaming:
        chunks = list(generator.stream_detailed(request))
        assert len(chunks) == 1
        finish = chunks[0].finish_details
    else:
        outputs = generator.generate_detailed(request)
        assert len(outputs) == prompt_count
        assert all(tuple(output.generated_token_ids) == (7,) for output in outputs)
        assert all(output.finish_details.reason == "eos" for output in outputs)
        finish = outputs[0].finish_details
        if prompt_count > 1:
            assert all(chunk["chunk"]["finish_details"]["reason"] == "eos"
                       for chunk in generator.last_batch_generation["scheduler_token_chunks"])
    assert finish.reason == "eos"
    assert finish.eos_token_id == 7


def test_explicit_eos_replaces_metadata_and_ignore_eos_still_wins():
    from hipengine.generation.qwen35_paro import _is_finished, _finish_details_for_tokens

    tokenizer = SimpleNamespace(eos_token_id=(2, 3))
    options = dict(ignore_eos=False, stop_token_ids=(), stop_token_sequences=(), eos_token_id=7)
    assert not _is_finished(tokenizer, (2,), **options)
    assert _is_finished(tokenizer, (7,), **options)
    assert not _is_finished(tokenizer, (7,), **{**options, "ignore_eos": True})
    details = _finish_details_for_tokens(tokenizer, (7,), max_tokens=3,
                                        sampler_mode="greedy_fast", **options)
    assert details.reason == "eos"


@pytest.mark.parametrize("explicit_eos,token_id,expected", [(None, 3, True), (7, 7, True), (7, 3, False)])
def test_resident_finish_uses_original_request_not_inferred_sampler_eos(explicit_eos, token_id, expected):
    from hipengine.generation import GenerationRequest
    from hipengine.generation import qwen35_paro as generation

    request = GenerationRequest(prompts=((10,),), max_tokens=3, temperature=0.0,
                                top_p=1.0, ignore_eos=False, eos_token_id=explicit_eos)
    tokenizer = SimpleNamespace(eos_token_id=(2, 3))
    owner = object.__new__(generation.Qwen35ParoResidentModelRunner)
    owner._session = SimpleNamespace(tokenizer=tokenizer)
    row = SimpleNamespace(
        request=request, sampling_request=generation._request_with_tokenizer_eos(request, tokenizer),
        generated_steps=[SimpleNamespace(token_id=token_id)],
        sampler_plan=SimpleNamespace(mode=SimpleNamespace(value="greedy_fast")), sampling_state=None,
    )
    assert owner._row_finished(row) is expected
    assert (owner._row_finish_details(row).reason == "eos") is expected
