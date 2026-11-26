from unittest.mock import Mock

import numpy as np
import pytest

from hipengine.generation import register_builtin_generators, resolve_text_generator
from hipengine.generation.inkling2_gguf import PRODUCTION_BACKENDS
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
def test_inkling_gguf_resolves_explicit_dspark_provider(backend):
    from hipengine.speculative.registry import (
        register_builtin_speculative_providers, resolve_speculative_provider,
        resolve_speculative_provider_candidate_budget,
    )

    plugin = resolve_model("inkling2")
    register_builtin_speculative_providers()
    key = dict(provider=plugin.speculative_mtp_provider, target_model=plugin.name,
               backend=backend, quant=plugin.default_quant)
    assert key["provider"] == "dspark"
    assert callable(resolve_speculative_provider(**key))
    assert resolve_speculative_provider_candidate_budget(**key) == 1


def test_inkling_gguf_attachment_preserves_draft_telemetry():
    from contextlib import ExitStack
    from hipengine.generation import GenerationOutput
    from hipengine.generation.inkling2_gguf import Inkling2GGUFGenerator
    from hipengine.generation.registry import DecodeState, GenerationTelemetry

    provider = Mock()
    inner = Inkling2GGUFGenerator.__new__(Inkling2GGUFGenerator)
    inner._draft_provider = None
    inner.model = Mock(ffn_device_output=True)
    inner._owned = ExitStack()
    assert not inner.supports_speculative_mtp
    with pytest.raises(NotImplementedError, match="draft_model"):
        inner.generate_speculative_mtp_detailed(Mock())
    inner.attach_speculative_provider(provider)
    assert inner.supports_speculative_mtp
    request = Mock()
    assert inner.generate_speculative_mtp_detailed(request) is provider.generate_detailed.return_value
    provider.generate_detailed.assert_called_once_with(request)
    out = inner._decorate_output(GenerationOutput("a", generated_token_ids=(1,), telemetry=GenerationTelemetry(
        decode_state=DecodeState(prompt_tokens=0, generated_tokens=1),
        diagnostics={"speculative_mtp": {"accepted_tokens": 1}},
    )))
    assert out.telemetry.diagnostics["speculative_mtp"] == {"accepted_tokens": 1}
    assert out.telemetry.diagnostics["speculation"] == "dspark"
    inner.close()
    provider.close.assert_called_once()


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_inkling_gguf_fast_profile_keeps_strict_and_default_distinct(backend):
    from hipengine.execution_profiles import (
        ExecutionProfile,
        resolve_default_execution_profile,
        resolve_runtime_profile,
    )

    register_builtin_generators()
    plugin = resolve_model("inkling2")
    key = dict(model=plugin.name, backend=backend, quant=plugin.default_quant)
    # The omitted profile is `production` on every backend where that plan's gate rows pass.
    assert resolve_default_execution_profile(**key) == ExecutionProfile.PRODUCTION
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
    block_gate = expert_selections["prompt_block_ge_192_rows_top8_nvfp4_gate_n2304_m2304"]
    block_down = expert_selections["prompt_block_ge_192_rows_top8_nvfp4_split_down_n1152_m2304"]
    assert block_gate["selected_variant"] == block_down["selected_variant"] == "selected_wmma_f16_parts_f32_out"
    assert block_gate["strict_fallback_variant"] == "selected_tile4_shared_input_f32_f32_out"
    assert block_down["strict_fallback_variant"] == "selected_tile4_f32_f32_out"
    strict_experts = {item["scope"]: item for item in strict.manifest["selections"] if item["layer"] == "moe_linear"}
    assert strict_experts["prompt_block_ge_192_rows_top8_nvfp4_gate_n2304_m2304"]["selected_variant"] == (
        "selected_tile4_shared_input_f32_f32_out")
    # The dense Q8_0 projections of a prompt block take a wide F16 tile; every narrow block and
    # `strict` keep the exact one.
    dense = {item["scope"]: item for item in fast.manifest["selections"] if item["layer"] == "linear"}
    assert sorted(dense) == ["prompt_block_q8_0_dense_n1024_n2304_n3072", "prompt_block_q8_0_dense_n384"]
    assert {item["selected_variant"] for item in dense.values()} == {"dense_wide64x128_f32_f32_out"}
    assert {item["strict_fallback_variant"] for item in dense.values()} == {
        "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"}
    assert {item["registry_quant"] for item in dense.values()} == {"gguf_q8_0"}
    strict_dense = [item for item in strict.manifest["selections"] if item["layer"] == "linear"]
    assert len(strict_dense) == 2 and all(item["strict_fallback_variant"] is None for item in strict_dense)
    assert {item["selected_variant"] for item in strict_dense} == {
        "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"}
    assert selections["inkling2_expert_pointwise"]["selected_variant"] == "split_activation_device"
    assert selections["inkling2_expert_pointwise"]["strict_fallback_variant"] == "split_activation_host"
    attention = {item["scope"]: item for item in fast.manifest["selections"]
                 if item["layer"] == "inkling2_target_attention"}
    assert attention["decode_live_ge_513"]["selected_variant"] == "split_chunked"
    assert attention["decode_live_ge_513"]["strict_fallback_variant"] == "split"
    assert attention["committed_projected_prefill_gate"]["selected_variant"] == "block_sigmoid"
    assert attention["committed_projected_prefill_gate"]["strict_fallback_variant"] == "block"
    # A `production` plan is registered only where its gate rows pass.
    from hipengine.execution_profiles import registered_runtime_profile_keys

    keys = registered_runtime_profile_keys()
    assert all(key.profile.value != "production" or key.backend in PRODUCTION_BACKENDS for key in keys)


def _selections(profile):
    return {(item["layer"], item["scope"]): item for item in profile.manifest["selections"]}


@pytest.mark.parametrize("backend", sorted(PRODUCTION_BACKENDS))
def test_inkling_gguf_production_plan_names_the_gated_route_and_keeps_f32_experts(backend):
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.inkling2_gguf import PRODUCTION_BACKENDS, Inkling2GGUFGenerator

    register_builtin_generators()
    plugin = resolve_model("inkling2")
    key = dict(model=plugin.name, backend=backend, quant=plugin.default_quant)
    strict = resolve_runtime_profile(**key, profile="strict")
    fast = resolve_runtime_profile(**key, profile="fast")
    production = resolve_runtime_profile(**key, profile=ExecutionProfile.PRODUCTION)
    assert not production.fell_back_to_strict and production.source_profile is ExecutionProfile.PRODUCTION
    assert production.strict_manifest_sha256 == fast.strict_manifest_sha256 == strict.manifest_sha256
    exact, reduced, shipped = _selections(strict), _selections(fast), _selections(production)
    assert exact.keys() == reduced.keys() == shipped.keys()
    experts, attention = "moe_linear", "inkling2_target_attention"
    for scope in (
        "single_token_top8_nvfp4_n1152_or_2304_m2304",
        "two_to_four_decode_rows_top8_nvfp4_gate_n1152_or_2304_m2304",
        "single_token_top8_nvfp4_split_down_n1152_m2304",
        "two_to_four_decode_rows_top8_nvfp4_split_down_n1152_m2304",
    ):
        # Every decode path keeps the selected F32 kernel: the Q8-dot chain of `fast` failed the envelope.
        assert "q8" not in shipped[(experts, scope)]["selected_variant"]
        assert shipped[(experts, scope)]["selected_variant"] == exact[(experts, scope)]["selected_variant"]
        assert "q8_dot" in reduced[(experts, scope)]["selected_variant"]
    shared = "single_token_top8_nvfp4_shared_gate_n1152_or_2304_m2304"
    assert shipped[(experts, shared)]["selected_variant"] == "selected_shared_register_fp4_f32_f32_out"
    assert shipped[(experts, shared)]["strict_fallback_variant"] == exact[(experts, shared)]["selected_variant"]
    # The eager decode attention stays the ordered split; the resident single-token visit names its own.
    assert shipped[(attention, "decode_live_ge_513")]["selected_variant"] == "split"
    resident = (attention, "resident_single_token_relative_bias")
    assert shipped[resident]["selected_variant"] == "decode_table_sigmoid"
    assert shipped[resident]["strict_fallback_variant"] == exact[resident]["selected_variant"] == "split"
    for layer_scope in (
        (attention, "resident_block_relative_bias"),
        (attention, "committed_projected_prefill_gate"),
        ("inkling2_controller", "committed_block_controllers"),
        ("inkling2_attention_prepare", "resident_block_inputs"),
        ("inkling2_expert_pointwise", "selected_split_gate_up"),
        (experts, "prompt_block_ge_192_rows_top8_nvfp4_gate_n2304_m2304"),
        (experts, "prompt_block_ge_192_rows_top8_nvfp4_split_down_n1152_m2304"),
    ):
        assert shipped[layer_scope]["selected_variant"] == reduced[layer_scope]["selected_variant"]
    # The single-product tile failed the production gate: `production` reads the residual tile.
    for scope in ("prompt_block_q8_0_dense_n1024_n2304_n3072", "prompt_block_q8_0_dense_n384"):
        dense_scope = ("linear", scope)
        assert shipped[dense_scope]["selected_variant"] == "dense_wide64x128_residual_f32_f32_out"
        assert shipped[dense_scope]["strict_fallback_variant"] == exact[dense_scope]["selected_variant"]
        assert reduced[dense_scope]["selected_variant"] == "dense_wide64x128_f32_f32_out"
    for selection in production.manifest["selections"]:
        assert selection["strict_fallback_variant"] == exact[(selection["layer"], selection["scope"])]["selected_variant"]
    assert production.manifest["graph_policy"] == "resident_single_token_whole_token_graph"
    assert fast.manifest["graph_policy"] == strict.manifest["graph_policy"] == "eager"
    # The factory builds the production route, and its prompt blocks are the fast plan's.
    keywords = production.factory.keywords
    assert keywords["production"] is True and keywords["fast"] is False
    assert Inkling2GGUFGenerator.prefill_schedule(True)[0] == Inkling2GGUFGenerator.fast_prefill_block_width == 1024
    assert Inkling2GGUFGenerator.fast_engine_loop_config_defaults["max_prefill_chunk_tokens"] == 1024
    assert Inkling2GGUFGenerator.prefill_schedule(False)[0] == 64
    assert set(PRODUCTION_BACKENDS) == {"hip_gfx1100", "hip_gfx1151"}


@pytest.mark.parametrize(
    ("environment", "default", "wanted"),
    [
        ({}, False, False),
        ({}, True, True),
        ({"HIPENGINE_INKLING2_RESIDENT_DECODE": "1"}, False, True),
        ({"HIPENGINE_INKLING2_RESIDENT_DECODE": "0"}, True, False),
        ({"HIPENGINE_INKLING2_RESIDENT_DECODE": ""}, True, False),
        ({"HIPENGINE_INKLING2_RESIDENT_DECODE": "yes"}, True, False),
    ],
)
def test_inkling_route_flags_take_the_profile_route_unless_the_caller_sets_them(monkeypatch, environment, default, wanted):
    from hipengine.generation import inkling2_gguf as module

    monkeypatch.delenv("HIPENGINE_INKLING2_RESIDENT_DECODE", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert module._resident_decode_requested(default) is wanted
    assert module._device_routes_requested(default) is default and module._decode_graph_requested(default) is default


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
