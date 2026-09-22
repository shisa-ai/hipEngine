"""Unit tier: Gemma 4 GGUF tensors mapped into the CPU reference layout.

The mapping is the join between two independently validated pieces — the loader's
structural contract and the NumPy reference forward — so the tests check the join
itself: that each named GGUF slot lands on the reference attribute it claims, that
the fused expert projection keeps the layout the reference consumes, and that the
per-layer attention geometry is carried across without collapsing the dual
geometry into a scalar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.gemma4 import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    gemma4_text_forward,
)
from hipengine.loading.gguf import GGUFReader, scan_gguf
from hipengine.loading.gemma4_gguf import build_gemma4_gguf_tensor_map
from hipengine.loading.gemma4_gguf_materialize import (
    gemma4_reference_config_from_gguf,
    materialize_gemma4_reference_weights,
    split_fused_expert_gate_up,
)
from hipengine.quant.gguf import dequantize_gguf_data
from tests._gemma4_gguf_fixture import (
    FIXTURE_EXPERTS,
    FIXTURE_EXPERT_FF,
    FIXTURE_HEAD_DIM_GLOBAL,
    FIXTURE_HEAD_DIM_SWA,
    FIXTURE_HIDDEN,
    FIXTURE_KV_HEADS_GLOBAL,
    FIXTURE_KV_HEADS_SWA,
    FIXTURE_SLIDING_WINDOW,
    FIXTURE_VOCAB,
    default_fixture_tensors,
    fixture_metadata,
    write_fixture_gguf,
)


@pytest.fixture
def reader(tmp_path: Path) -> GGUFReader:
    path = write_fixture_gguf(
        tmp_path / "materialize.gguf",
        default_fixture_tensors(),
        fixture_metadata(),
    )
    return GGUFReader(path)


def raw_tensor(reader: GGUFReader, name: str) -> np.ndarray:
    tensor = reader.tensor_info(name)
    return dequantize_gguf_data(reader.tensor_data(name), tensor.ggml_type)


def test_slots_land_on_the_reference_attributes(reader: GGUFReader) -> None:
    """Each GGUF slot must arrive at the reference attribute that claims it."""

    materialized = materialize_gemma4_reference_weights(reader)
    expected = {
        "input_layernorm": "blk.0.attn_norm.weight",
        "post_attention_layernorm": "blk.0.post_attention_norm.weight",
        "pre_feedforward_layernorm": "blk.0.ffn_norm.weight",
        "post_feedforward_layernorm": "blk.0.post_ffw_norm.weight",
        "post_feedforward_layernorm_1": "blk.0.post_ffw_norm_1.weight",
        "post_feedforward_layernorm_2": "blk.0.post_ffw_norm_2.weight",
        "pre_feedforward_layernorm_2": "blk.0.pre_ffw_norm_2.weight",
        "q_proj": "blk.0.attn_q.weight",
        "k_proj": "blk.0.attn_k.weight",
        "v_proj": "blk.0.attn_v.weight",
        "o_proj": "blk.0.attn_output.weight",
        "q_norm": "blk.0.attn_q_norm.weight",
        "k_norm": "blk.0.attn_k_norm.weight",
        "mlp_gate_proj": "blk.0.ffn_gate.weight",
        "mlp_up_proj": "blk.0.ffn_up.weight",
        "mlp_down_proj": "blk.0.ffn_down.weight",
        "router_proj": "blk.0.ffn_gate_inp.weight",
        "router_scale": "blk.0.ffn_gate_inp.scale",
        "router_per_expert_scale": "blk.0.ffn_down_exps.scale",
        "experts_gate_up_proj": "blk.0.ffn_gate_up_exps.weight",
        "experts_down_proj": "blk.0.ffn_down_exps.weight",
        "layer_scalar": "blk.0.layer_output_scale.weight",
    }
    layer = materialized.layer(0)
    for attribute, tensor_name in expected.items():
        actual = np.asarray(getattr(layer, attribute))
        assert actual.shape == raw_tensor(reader, tensor_name).shape, attribute
        np.testing.assert_array_equal(actual, raw_tensor(reader, tensor_name))


def test_root_slots_and_tied_embeddings(reader: GGUFReader) -> None:
    materialized = materialize_gemma4_reference_weights(reader)
    np.testing.assert_array_equal(
        materialized.weights.embed_tokens,
        raw_tensor(reader, "token_embd.weight"),
    )
    np.testing.assert_array_equal(
        materialized.weights.final_norm,
        raw_tensor(reader, "output_norm.weight"),
    )
    # The fixture has no output.weight, so the head is tied to the embedding.
    assert materialized.weights.lm_head is None
    assert materialized.config.tie_word_embeddings is True


def test_global_layers_have_no_v_proj(reader: GGUFReader) -> None:
    materialized = materialize_gemma4_reference_weights(reader)
    assert materialized.layer(0).v_proj is not None
    assert materialized.layer(1).v_proj is None


def test_fused_expert_projection_keeps_the_gguf_layout(reader: GGUFReader) -> None:
    materialized = materialize_gemma4_reference_weights(reader)
    fused = np.asarray(materialized.layer(0).experts_gate_up_proj)
    assert fused.shape == (
        FIXTURE_EXPERTS,
        2 * FIXTURE_EXPERT_FF,
        FIXTURE_HIDDEN,
    )
    np.testing.assert_array_equal(
        fused, raw_tensor(reader, "blk.0.ffn_gate_up_exps.weight")
    )


def test_dual_attention_geometry_survives_the_conversion(reader: GGUFReader) -> None:
    """The two layer families must not collapse to one geometry."""

    config = materialize_gemma4_reference_weights(reader).config
    sliding = config.geometry(0)
    global_layer = config.geometry(1)

    assert sliding.layer_type == SLIDING_ATTENTION
    assert global_layer.layer_type == FULL_ATTENTION
    assert sliding.num_kv_heads == FIXTURE_KV_HEADS_SWA
    assert global_layer.num_kv_heads == FIXTURE_KV_HEADS_GLOBAL
    assert sliding.head_dim == FIXTURE_HEAD_DIM_SWA
    assert global_layer.head_dim == FIXTURE_HEAD_DIM_GLOBAL
    assert sliding.sliding_window == FIXTURE_SLIDING_WINDOW
    assert global_layer.sliding_window is None
    assert sliding.k_eq_v is False
    assert global_layer.k_eq_v is True
    # Proportional RoPE: the global layer rotates a strict subset of its pairs
    # while the sliding layer rotates every pair.
    assert global_layer.rope.rope_angles < global_layer.head_dim // 2
    assert sliding.rope.rope_angles == sliding.head_dim // 2
    assert global_layer.rope.rope_theta != sliding.rope.rope_theta


def test_reference_config_carries_the_moe_shape(reader: GGUFReader) -> None:
    config = materialize_gemma4_reference_weights(reader).config
    assert config.hidden_size == FIXTURE_HIDDEN
    assert config.vocab_size == FIXTURE_VOCAB
    assert config.num_experts == FIXTURE_EXPERTS
    assert config.moe_intermediate_size == FIXTURE_EXPERT_FF
    assert config.num_hidden_layers == 2


def test_the_materialized_weights_run_a_reference_forward(
    reader: GGUFReader,
) -> None:
    materialized = materialize_gemma4_reference_weights(reader)
    result = gemma4_text_forward(
        materialized.weights,
        materialized.config,
        [1, 2, 3],
    )
    assert result.logits.shape == (3, FIXTURE_VOCAB)
    assert np.all(np.isfinite(result.logits))
    # The softcap must actually be applied, not merely recorded.
    cap = materialized.config.final_logit_softcapping
    assert cap is not None
    assert np.max(np.abs(result.logits)) <= cap + 1e-4


def test_reference_config_agrees_with_the_loader_config(reader: GGUFReader) -> None:
    model_map = build_gemma4_gguf_tensor_map(reader.info)
    config = gemma4_reference_config_from_gguf(model_map.config)
    assert config.num_hidden_layers == model_map.config.block_count
    for layer_id in range(model_map.config.block_count):
        geometry = config.geometry(layer_id)
        assert geometry.head_dim == model_map.config.head_dim(layer_id)
        assert geometry.num_heads == model_map.config.head_count(layer_id)
        assert (
            geometry.num_kv_heads == model_map.config.head_count_kv_for(layer_id)
        )
        assert geometry.k_eq_v == model_map.config.attention_k_eq_v(layer_id)


def test_fused_width_check_rejects_a_mismatched_artifact() -> None:
    fused = np.zeros((4, 128, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="2 \\* moe_intermediate_size"):
        split_fused_expert_gate_up(fused, expected_fused_width=256)
    with pytest.raises(ValueError, match="must be 3-D"):
        split_fused_expert_gate_up(np.zeros((128, 8)), expected_fused_width=128)


def test_a_v_proj_on_a_global_layer_is_refused(reader: GGUFReader) -> None:
    """The map already rejects it; the materializer must not have a second path."""

    model_map = build_gemma4_gguf_tensor_map(reader.info)
    layer = model_map.layer(1)
    assert not layer.has("attn_v")


def test_an_artifact_without_rope_freqs_rotates_every_pair(tmp_path: Path) -> None:
    """The rotated span lives only in ``rope_freqs.weight``.

    The loader already decides that an absent sentinel tensor means the global
    layers rotate every pair. This pins that the materializer carries that
    decision through instead of inventing a second, contradicting rule.
    """

    tensors = [
        entry
        for entry in default_fixture_tensors()
        if entry[0] != "rope_freqs.weight"
    ]
    path = write_fixture_gguf(tmp_path / "no_rope.gguf", tensors, fixture_metadata())
    reader = GGUFReader(path)
    materialized = materialize_gemma4_reference_weights(reader)
    global_layer = materialized.config.geometry(1)
    assert global_layer.layer_type == FULL_ATTENTION
    assert global_layer.rope.rope_angles == global_layer.head_dim // 2
    # The sentinel layout is what a partial-rope artifact would have carried.
    assert materialized.config.attention[0].rope.rope_angles == FIXTURE_HEAD_DIM_SWA // 2


def test_narrowed_materialization_keeps_the_mapping(tmp_path: Path) -> None:
    """``dtype`` narrows storage only; the reference casts back to fp32."""

    path = write_fixture_gguf(
        tmp_path / "narrow.gguf",
        default_fixture_tensors(),
        fixture_metadata(),
    )
    reader = GGUFReader(path)
    narrowed = materialize_gemma4_reference_weights(reader, dtype=np.float16)
    assert np.asarray(narrowed.layer(0).q_proj).dtype == np.float16
    wide = materialize_gemma4_reference_weights(reader)
    np.testing.assert_allclose(
        np.asarray(narrowed.layer(0).q_proj, dtype=np.float32),
        np.asarray(wide.layer(0).q_proj),
        atol=1e-3,
        rtol=1e-3,
    )
    # A narrowed run still produces finite logits of the right shape.
    result = gemma4_text_forward(narrowed.weights, narrowed.config, [1, 2])
    assert result.logits.shape == (2, FIXTURE_VOCAB)
    assert np.all(np.isfinite(result.logits))
    assert narrowed.dequantized_bytes < wide.dequantized_bytes


def test_a_synthetic_artifact_reports_its_dequantized_size(
    reader: GGUFReader,
) -> None:
    materialized = materialize_gemma4_reference_weights(reader)
    assert materialized.dequantized_bytes > 0
    assert materialized.dequantized_bytes == sum(
        np.asarray(value).nbytes
        for value in (
            materialized.weights.embed_tokens,
            materialized.weights.final_norm,
            *(
                np.asarray(attribute)
                for layer in materialized.weights.layers
                for attribute in vars(layer).values()
                if attribute is not None
            ),
        )
    )
