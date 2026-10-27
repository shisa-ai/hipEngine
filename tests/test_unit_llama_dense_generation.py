"""Dense Llama admission for generation: chat prompt, refusals, and registration.

Everything here runs without a device. The generator's construction validates the
checkpoint, the tokenizer, and the context budget before any weights move, so a
checkpoint that cannot serve generation fails at that boundary with a named
capability miss rather than at the first token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._llama_tiny import (
    TINY_LLAMA_BOS,
    TINY_LLAMA_CONFIG,
    TINY_LLAMA_EOS,
    write_tiny_llama_checkpoint,
    write_tiny_llama_tokenizer,
)


def _index(directory: Path):
    from hipengine.loading.safetensors import load_weight_index

    return load_weight_index(directory)


def _generator(directory: Path, **kwargs):
    from hipengine.generation.llama_dense import LlamaDenseGenerator
    from hipengine.models.llama import LLAMA_DENSE

    return LlamaDenseGenerator(
        model_path=directory,
        weight_index=_index(directory),
        model_plugin=LLAMA_DENSE,
        **kwargs,
    )


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    path, _f32, _bf16 = write_tiny_llama_checkpoint(tmp_path / "tiny")
    return path


def _tokenizer(directory: Path, **overrides):
    from hipengine.tokenization.llama_dense import LlamaDenseTokenizer

    kwargs = {
        "model_vocab_size": int(TINY_LLAMA_CONFIG["vocab_size"]),
        "eos_token_id": TINY_LLAMA_EOS,
        "bos_token_id": TINY_LLAMA_BOS,
    }
    kwargs.update(overrides)
    return LlamaDenseTokenizer.from_model_path(directory, **kwargs)


def test_chat_prompt_renders_the_checkpoint_dialect(checkpoint):
    tokenizer = _tokenizer(checkpoint)
    assert tokenizer.chat_style == "chatml"
    assert tokenizer.format_chat_prompt("hello world") == (
        "<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n thinking\n"
    )
    assert tokenizer.format_chat_prompt("hello world", system="be brief") == (
        "<|im_start|>system\nbe brief<|im_end|>\n"
        "<|im_start|>user\nhello world<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n"
    )
    closed = tokenizer.format_chat_prompt("hello world", enable_thinking=False)
    assert closed.endswith("<|im_start|>assistant\n thinking\n\n</think>\n\n")
    # The markers survive tokenization as their own ids rather than as unknowns:
    # <|im_start|> user hello world <|im_end|> <|im_start|> assistant <unk>
    assert tokenizer.encode_chat("hello world") == (6, 3, 1, 2, 7, 6, 4, 0)


def test_tokenizer_refuses_a_vocabulary_wider_than_the_decoder(checkpoint):
    with pytest.raises(ValueError, match="exceeds the decoder"):
        _tokenizer(checkpoint, model_vocab_size=4)


def test_tokenizer_refuses_a_mismatched_eos(checkpoint):
    with pytest.raises(ValueError, match="eos_token_id"):
        _tokenizer(checkpoint, eos_token_id=TINY_LLAMA_EOS + 1)


def test_tokenizer_names_a_missing_file(tmp_path: Path):
    from hipengine.tokenization.llama_dense import LlamaDenseTokenizer

    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        LlamaDenseTokenizer.from_model_path(
            tmp_path, model_vocab_size=16, eos_token_id=1
        )


def test_tokenizer_refuses_a_template_it_cannot_render(checkpoint):
    config_path = checkpoint / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = "{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="not ChatML"):
        _tokenizer(checkpoint)


def test_generator_defaults_the_context_to_the_checkpoint_limit(checkpoint):
    generator = _generator(checkpoint)
    assert generator.context_length == int(TINY_LLAMA_CONFIG["max_position_embeddings"])


def test_generator_defaults_the_context_to_the_checkpoint_positions(tmp_path: Path):
    """No round number and no kernel ceiling: the checkpoint's own position count."""

    wide = dict(TINY_LLAMA_CONFIG)
    wide["max_position_embeddings"] = 32768
    path, _f32, _bf16 = write_tiny_llama_checkpoint(tmp_path / "wide-default", config=wide)
    generator = _generator(path)
    assert generator.context_length == 32768


def test_generator_refuses_a_context_beyond_the_checkpoint(checkpoint):
    with pytest.raises(ValueError, match="exceeds the checkpoint"):
        _generator(checkpoint, context_length=int(TINY_LLAMA_CONFIG["max_position_embeddings"]) + 1)


def test_context_past_the_old_lds_ceiling_is_served(tmp_path: Path):
    """16128 was the exact kernel's LDS row, not the path's capability."""

    from hipengine.runtime.llama_dense import max_dense_context

    wide = dict(TINY_LLAMA_CONFIG)
    wide["max_position_embeddings"] = 65536
    path, _f32, _bf16 = write_tiny_llama_checkpoint(tmp_path / "wide", config=wide)
    generator = _generator(path, context_length=65536)
    assert generator.context_length == 65536
    assert max_dense_context(generator.spec) == 65536
    # This fixture's head width is not one the tiled kernel carries, so the
    # prompt bound still comes from the exact fallback -- the cache capacity does
    # not. That separation is the point: 16128 was never the path's capability.
    assert generator.max_prompt_tokens == 15856


def _geometry(head_dim: int, *, positions: int = 32768):
    class _Spec:
        pass

    _Spec.head_dim = head_dim
    _Spec.num_attention_heads = 8
    _Spec.num_key_value_heads = 4
    _Spec.max_position_embeddings = positions
    return _Spec()


@pytest.mark.parametrize(
    ("head_dim", "positions", "expected"),
    [
        # The tiled kernel serves head_dim 128 at any key count, so the prompt
        # bound is the checkpoint's.
        (128, 32768, 32768),
        (128, 524288, 524288),
        # Any other width falls back to the exact kernel, whose FP32 logit row is
        # ``max(head_dim + keys + threads, keys + 528)`` floats of LDS.
        (32, 32768, 15856),
        (64, 32768, 15856),
        # A checkpoint with fewer positions than the LDS row binds first.
        (32, 4096, 4096),
    ],
)
def test_prefill_bound_follows_the_selected_kernel(head_dim, positions, expected):
    from hipengine.runtime.llama_dense import max_dense_prefill_tokens

    assert max_dense_prefill_tokens(_geometry(head_dim, positions=positions)) == expected


def test_generator_refuses_a_prompt_beyond_the_prefill_bound(tmp_path: Path):
    """On the fallback geometry an over-long prompt is refused before the load."""

    wide = dict(TINY_LLAMA_CONFIG)
    wide["max_position_embeddings"] = 32768
    path, _f32, _bf16 = write_tiny_llama_checkpoint(tmp_path / "wide", config=wide)
    generator = _generator(path)
    # head_dim 32 is not carried by the tiled kernel, so the exact LDS row binds.
    assert generator.max_prompt_tokens < generator.context_length

    generator._validate_prompt_length(generator.max_prompt_tokens, 1)
    with pytest.raises(ValueError, match="shared-memory capacity"):
        generator._validate_prompt_length(generator.max_prompt_tokens, 2)
    with pytest.raises(ValueError, match="dense causal prefill attention"):
        generator._validate_prompt_length(generator.max_prompt_tokens + 1, 1)

    prompt = " ".join(["hello"] * (generator.max_prompt_tokens + 1))
    with pytest.raises(ValueError, match="dense causal prefill attention"):
        generator.generate_detailed(_request(prompts=(prompt,)))
    assert generator._runner is None, "the refusal must precede the weight load"


def test_an_explicit_strict_profile_selects_the_exact_kernel(checkpoint):
    """`execution_profile="strict"` must reach the kernel, not be dropped."""

    default = _generator(checkpoint)
    assert default.execution_profile is None
    assert default.strict_arithmetic is False

    strict = _generator(checkpoint, execution_profile="strict")
    assert strict.execution_profile == "strict"
    assert strict.strict_arithmetic is True

    production = _generator(checkpoint, execution_profile="production")
    assert production.strict_arithmetic is False


def test_an_unknown_profile_is_refused_by_name(checkpoint):
    with pytest.raises(Exception, match="strict|production|batch"):
        _generator(checkpoint, execution_profile="turbo")


def test_generator_refuses_a_prompt_that_leaves_no_room_to_generate(checkpoint):
    generator = _generator(checkpoint)
    with pytest.raises(ValueError, match="exceeds the 512-token context"):
        generator._validate_prompt_length(510, 4)


def test_context_declaration_refreshes_prompt_bound(checkpoint):
    generator = _generator(checkpoint, context_length=32)
    generator.declare_max_sequence_length(128)
    assert generator.max_prompt_tokens == 128
    generator._validate_prompt_length(65, 1)
    generator.declare_max_sequence_length(16)
    assert generator.max_prompt_tokens == 16


def test_generator_refuses_a_wider_serving_capacity(checkpoint):
    generator = _generator(checkpoint)
    generator.declare_serving_capacity(1)
    with pytest.raises(NotImplementedError, match="paged attention"):
        generator.declare_serving_capacity(2)


def _request(**overrides):
    from hipengine.generation.registry import GenerationRequest

    fields = {
        "prompts": ("hello",),
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


def test_generator_accepts_a_plain_autoregressive_request(checkpoint):
    generator = _generator(checkpoint)
    generator._validate_request(_request())
    generator._validate_request(_request(temperature=0.7, top_k=40, top_p=0.9))
    generator._validate_request(_request(stop_token_ids=(TINY_LLAMA_EOS,)))
    generator._validate_request(_request(kv_storage="bf16"))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"grammar": {"type": "json_object"}}, "grammar-constrained decoding"),
        ({"stop_token_sequences": ((1, 2),)}, "multi-token stop sequences"),
        ({"logprobs": True}, "logprobs"),
        ({"kv_storage": "int8"}, "kv_storage"),
        ({"forced_tokens_pending": (3,)}, "forced-token queues"),
    ],
)
def test_generator_refuses_requests_it_cannot_serve(checkpoint, overrides, expected):
    generator = _generator(checkpoint)
    with pytest.raises(NotImplementedError, match=expected):
        generator._validate_request(_request(**overrides))


def test_generation_key_resolves_for_each_supported_backend():
    from hipengine.generation import register_builtin_generators
    from hipengine.generation.registry import resolve_text_generator

    register_builtin_generators()
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        factory = resolve_text_generator(model="llama_dense", backend=backend, quant="bf16")
        assert callable(factory)


def test_missing_generator_names_the_quant_it_could_not_serve():
    from hipengine.generation import register_builtin_generators
    from hipengine.generation.registry import MissingGeneratorError, resolve_text_generator

    register_builtin_generators()
    with pytest.raises(MissingGeneratorError, match="llama_dense"):
        resolve_text_generator(model="llama_dense", backend="hip_gfx1151", quant="int4")


def test_checkpoint_without_an_eos_is_refused(tmp_path: Path):
    from hipengine.generation.llama_dense import LlamaDenseGenerator
    from hipengine.models.llama import LlamaCapabilityMiss

    config = dict(TINY_LLAMA_CONFIG)
    config.pop("eos_token_id")
    path, _f32, _bf16 = write_tiny_llama_checkpoint(tmp_path / "no-eos", config=config)
    with pytest.raises(LlamaCapabilityMiss, match="eos_token_id"):
        LlamaDenseGenerator(
            model_path=path,
            weight_index=_index(path),
            model_plugin=None,
        )


def test_checkpoint_without_a_tokenizer_is_refused(tmp_path: Path):
    from hipengine.generation.llama_dense import LlamaDenseGenerator
    from hipengine.models.llama import LlamaCapabilityMiss

    path, _f32, _bf16 = write_tiny_llama_checkpoint(
        tmp_path / "no-tokenizer", with_tokenizer=False
    )
    with pytest.raises(LlamaCapabilityMiss, match="tokenizer"):
        LlamaDenseGenerator(
            model_path=path,
            weight_index=_index(path),
            model_plugin=None,
        )


def test_tokenizer_round_trips_a_prompt(checkpoint):
    tokenizer = _tokenizer(checkpoint)
    ids = tokenizer.encode("hello world")
    assert ids == (1, 2)
    assert tokenizer.decode(ids) == "hello world"
    assert tokenizer.decode_token(1) == "hello"
    # Every decoder row is decodable, which is what keeps generated text from
    # failing on a token the vocabulary does not name.
    assert tokenizer.decode((int(TINY_LLAMA_CONFIG["vocab_size"]) - 1,))
