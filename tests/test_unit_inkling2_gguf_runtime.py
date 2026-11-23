from unittest.mock import Mock

import numpy as np
import pytest

from hipengine.generation import register_builtin_generators, resolve_text_generator
from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights
from hipengine.models import resolve_model
from hipengine.runtime.inkling2_gguf import Inkling2GGUFText
from hipengine.tokenization.inkling2_gguf import inkling2_gguf_tokenizer
from tests.test_unit_inkling2_gguf import _file


@pytest.mark.parametrize("legacy", [True, False])
def test_gguf_ffn_uses_gate_and_up_halves_and_physical_layer(legacy):
    reader = Mock()
    reader.info = _file(legacy)
    weights = Inkling2GGUFWeights(reader)
    projection = Mock(supports_device_mix=False, supports_device_activation=False)
    layer = weights.mapping.layers[2]
    visit = weights.mapping.plan.visits[4]
    x = np.ones((2, 256), np.float32)
    gate = np.full((2, 256), 2, np.float32)
    up = np.full((2, 256), 4, np.float32)
    projection.project.return_value = np.array([[1.0, -1.0], [-1.0, 1.0]], np.float32)
    projection.project_selected.side_effect = [np.concatenate((gate, up), axis=1), np.full_like(x, 0.25)]
    result = Inkling2GGUFText(weights, projection=projection).feed_forward(visit, x)
    assert projection.project.call_args.args[0] == layer.tensors["ffn_gate_inp.weight"].name
    gate_call, down_call = [call.args for call in projection.project_selected.call_args_list]
    assert gate_call[:2] == (visit, "gate_up")
    assert down_call[:2] == (visit, "down")
    np.testing.assert_array_equal(gate_call[2], [0, 1])
    np.testing.assert_array_equal(gate_call[2], down_call[2])
    np.testing.assert_array_equal(gate_call[3], x)
    np.testing.assert_allclose(down_call[3], 4 * np.tanh(2 / 4) / (1 + np.exp(-2)) * 25 * np.tanh(4 / 25), rtol=5e-7)
    np.testing.assert_array_equal(result, np.full_like(x, 2))


@pytest.mark.parametrize("architecture", ["inkling2", "inkling2_loop"])
def test_both_gguf_architectures_register_the_packed_generator(architecture):
    from hipengine.generation.inkling2_gguf import Inkling2GGUFGenerator

    register_builtin_generators()
    plugin = resolve_model(architecture)
    for backend in ("hip_gfx1151", "hip_gfx1100"):
        factory = resolve_text_generator(model=plugin.name, backend=backend, quant=plugin.default_quant)
        assert factory.func is Inkling2GGUFGenerator
        assert factory.keywords == {"backend": backend}


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_inkling_gguf_fast_profile_keeps_strict_and_default_distinct(backend):
    from hipengine.execution_profiles import resolve_default_execution_profile, resolve_runtime_profile

    register_builtin_generators()
    plugin = resolve_model("inkling2")
    key = dict(model=plugin.name, backend=backend, quant=plugin.default_quant)
    assert resolve_default_execution_profile(**key) is None
    strict = resolve_runtime_profile(**key, profile="strict")
    fast = resolve_runtime_profile(**key, profile="fast")
    assert not fast.fell_back_to_strict
    assert strict.manifest_sha256 == fast.strict_manifest_sha256
    selections = {item["layer"]: item for item in fast.manifest["selections"] if item["layer"] != "moe_linear"}
    expert_selections = {item["scope"]: item for item in fast.manifest["selections"] if item["layer"] == "moe_linear"}
    ordinary = expert_selections["single_token_top8_nvfp4_n1152_or_2304_m2304"]
    shared = expert_selections["single_token_top8_nvfp4_shared_gate_n1152_or_2304_m2304"]
    counted = expert_selections["two_to_four_decode_rows_top8_nvfp4_gate_n1152_or_2304_m2304"]
    fused = expert_selections["single_token_top8_nvfp4_split_down_n1152_m2304"]
    counted_split = expert_selections["two_to_four_decode_rows_top8_nvfp4_split_down_n1152_m2304"]
    assert ordinary["selected_variant"] == "selected_q8_dot_f32_f32_out"
    assert shared["selected_variant"] == "selected_shared_input_q8_dot_f32_f32_out"
    assert counted["selected_variant"] == "selected_counted_shared_input_q8_dot_f32_f32_out"
    assert fused["selected_variant"] == "selected_split_activation_q8_dot_f32_f32_out"
    assert counted_split["selected_variant"] == "selected_counted_split_activation_q8_dot_f32_f32_out"
    assert all(item["strict_fallback_variant"] == "selected_register_fp4_f32_f32_out"
               for item in (ordinary, shared, counted, fused, counted_split))
    assert selections["inkling2_expert_pointwise"]["selected_variant"] == "split_activation_device"
    assert selections["inkling2_expert_pointwise"]["strict_fallback_variant"] == "split_activation_host"
    attention = {item["scope"]: item for item in fast.manifest["selections"]
                 if item["layer"] == "inkling2_target_attention"}
    assert attention["decode_live_ge_513"]["selected_variant"] == "split_chunked"
    assert attention["decode_live_ge_513"]["strict_fallback_variant"] == "split"
    assert attention["committed_projected_prefill_gate"]["selected_variant"] == "block_sigmoid"
    assert attention["committed_projected_prefill_gate"]["strict_fallback_variant"] == "block"


def test_embedded_inkling_tokenizer_preserves_token_and_control_ids():
    info = Mock()
    info.metadata = {
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "inkling2",
        "tokenizer.ggml.tokens": ["a", "b", "ab", "Ġ", "<eot>"],
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.token_type": [1, 1, 1, 1, 3],
        "tokenizer.ggml.eos_token_id": 4,
    }
    tokenizer, eos = inkling2_gguf_tokenizer(info, 5)
    assert eos == 4
    ids = tokenizer.encode("ab <eot>", add_special_tokens=False).ids
    assert ids == [2, 3, 4]
    assert tokenizer.decode(ids, skip_special_tokens=True) == "ab "
    with pytest.raises(ValueError, match="tokenizer size"):
        inkling2_gguf_tokenizer(info, 6)
