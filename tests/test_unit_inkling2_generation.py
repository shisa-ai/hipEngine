from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation import GenerationRequest, register_builtin_generators, resolve_text_generator
from hipengine.generation.inkling2_mini import Inkling2ReferenceGenerator
from hipengine.models import resolve_model
from hipengine.quant import resolve_quant


def test_original_architecture_resolves_to_cpu_reference_factory():
    register_builtin_generators()
    model = resolve_model("Inkling2LoopForConditionalGeneration")
    quant = resolve_quant(model.default_quant)
    assert quant.compute_dtype == "fp32"
    factory = resolve_text_generator(model=model.name, backend="cpu_reference", quant=quant.name)
    assert factory is Inkling2ReferenceGenerator


def test_gfx1100_profiles_resolve_vocabulary_and_expert_variants():
    from hipengine.execution_profiles import resolve_default_execution_profile, resolve_runtime_profile

    register_builtin_generators()
    model = resolve_model("Inkling2LoopForConditionalGeneration")
    key = dict(model=model.name, backend="hip_gfx1100", quant=model.default_quant)
    assert resolve_default_execution_profile(**key) == "production"
    native = resolve_runtime_profile(**key, profile="production")
    strict = resolve_runtime_profile(**key, profile="strict")
    assert not native.fell_back_to_strict
    fast = {item["layer"]: item for item in native.manifest["selections"]}
    exact = {item["layer"]: item for item in strict.manifest["selections"]}
    assert fast.keys() == exact.keys() == {"inkling2_vocabulary_projection", "inkling2_routed_ffn"}
    for layer, production, fallback in (
        ("inkling2_vocabulary_projection", "native_bf16", "fp32_rocblas"),
        ("inkling2_routed_ffn", "device_pointwise", "host_pointwise"),
    ):
        assert fast[layer]["selected_variant"] == production
        assert fast[layer]["strict_fallback_variant"] == fallback
        assert exact[layer]["selected_variant"] == fallback


def test_cpu_reference_generation_uses_shared_sampler_and_enforces_context():
    class Session:
        def step(self, token, *, logits=True):
            tokens.append(token)
            return np.array([0, 9, 1], dtype=np.float32) if logits else None

    tokens = []
    gen = Inkling2ReferenceGenerator.__new__(Inkling2ReferenceGenerator)
    gen.model = SimpleNamespace(session=lambda capacity: Session())
    gen.tokenizer = SimpleNamespace(decode=lambda ids, **_: "".join(map(str, ids)))
    gen.max_sequence_length = 4
    gen.eos_token_id = 1
    req = GenerationRequest(prompts=((2, 0),), max_tokens=2, temperature=0, top_p=1, ignore_eos=False)
    output, = gen.generate_detailed(req)
    assert output.generated_token_ids == (1,)
    assert output.prompt_tokens == 2
    assert output.finish_details.reason == "stop"
    assert output.finish_details.eos_token_id == 1
    assert tokens == [2, 0]
    req = GenerationRequest(prompts=((2, 0),), max_tokens=3, temperature=0, top_p=1, ignore_eos=False)
    with pytest.raises(ValueError, match="context"):
        gen.generate_detailed(req)
