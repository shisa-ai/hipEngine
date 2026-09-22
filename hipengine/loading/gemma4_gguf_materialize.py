"""Materialize Gemma 4 GGUF tensors into the CPU reference weight layout.

This is the bridge between the GGUF artifact and the validated NumPy reference
forward in :mod:`hipengine.kernels.cpu_reference.gemma4`. It exists so the
loader's structural assumptions can be exercised against real weights before any
HIP kernel depends on them.

Three conversions are not a rename, and each one is validated rather than
assumed:

* ``ffn_gate_up_exps.weight`` is fused along the *intermediate* axis, with the
  gate in the first half and up in the second. The reference expects the same
  fused layout, so the split is a view, and the fused width must be exactly
  ``2 * moe_intermediate_size``.
* Global layers have no ``attn_v.weight`` because ``attention_k_eq_v`` makes V
  the raw K projection. A present ``attn_v.weight`` on a global layer is an
  error, not a fallback.
* The proportional rotary span is not stored as a factor. It is recovered from
  ``rope_freqs.weight`` by :func:`gemma4_rotated_pair_count`, and a partial rope
  tensor on a sliding layer is refused rather than silently ignored.

Weights are dequantized to ``float32`` by default. ``dtype`` narrows the storage
only; the reference forward casts back to ``float32`` for arithmetic, so a
narrowed materialization is a memory optimization, not a precision decision.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hipengine.kernels.cpu_reference.gemma4 import (
    FULL_ATTENTION,
    Gemma4AttentionGeometry,
    Gemma4LayerWeights,
    Gemma4RopeConfig,
    Gemma4TextConfig,
    Gemma4TextWeights,
)
from hipengine.loading.gguf import (
    GGUFModelInfo,
    GGUFReader,
    MissingGGUFTensorError,
)
from hipengine.loading.gemma4_gguf import (
    Gemma4GGUFConfig,
    Gemma4GGUFModelMap,
    build_gemma4_gguf_tensor_map,
    read_gemma4_rope_freqs,
)
from hipengine.quant.gguf import dequantize_gguf_data

__all__ = [
    "Gemma4GGUFReferenceWeights",
    "gemma4_reference_config_from_gguf",
    "materialize_gemma4_reference_weights",
    "split_fused_expert_gate_up",
]


def gemma4_reference_config_from_gguf(config: Gemma4GGUFConfig) -> Gemma4TextConfig:
    """Convert a validated GGUF config into the reference text config."""

    geometry: list[Gemma4AttentionGeometry] = []
    for layer_id in range(config.block_count):
        rope = config.rope_for_layer(layer_id)
        sliding = config.is_sliding(layer_id)
        geometry.append(
            Gemma4AttentionGeometry(
                layer_type=config.layer_type(layer_id),
                num_heads=config.head_count(layer_id),
                num_kv_heads=config.head_count_kv_for(layer_id),
                head_dim=config.head_dim(layer_id),
                rope=Gemma4RopeConfig(
                    rope_theta=rope.freq_base,
                    head_dim=rope.head_dim,
                    rope_angles=rope.rotated_pairs,
                ),
                sliding_window=config.sliding_window if sliding else None,
                k_eq_v=config.attention_k_eq_v(layer_id),
            )
        )
    return Gemma4TextConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.feed_forward_length(0),
        moe_intermediate_size=config.expert_feed_forward_length,
        num_experts=config.expert_count,
        top_k_experts=config.expert_used_count,
        rms_norm_eps=config.rms_norm_eps,
        attention=tuple(geometry),
        final_logit_softcapping=config.final_logit_softcapping,
        tie_word_embeddings=config.tied_embeddings,
        vocab_size=config.vocab_size,
    )


def split_fused_expert_gate_up(
    fused: np.ndarray,
    *,
    expected_fused_width: int,
) -> np.ndarray:
    """Return the fused expert projection unchanged after checking its width.

    The reference forward consumes the same fused ``gate | up`` layout the GGUF
    stores, so the split is a validation rather than a copy. The gate occupies
    the first half of the intermediate axis, which is the layout llama.cpp
    produces with ``ggml_view_3d`` at offset 0 for the gate and
    ``n_ff * nb[0]`` for up.
    """

    value = np.asarray(fused)
    if value.ndim != 3:
        raise ValueError(
            f"fused expert gate_up must be 3-D, got shape {value.shape}"
        )
    if value.shape[1] != int(expected_fused_width):
        raise ValueError(
            "fused expert gate_up width must be "
            f"2 * moe_intermediate_size = {expected_fused_width}, got {value.shape[1]}"
        )
    if value.shape[1] % 2:
        raise ValueError("fused expert gate_up width must be even")
    return value


@dataclass(frozen=True)
class Gemma4GGUFReferenceWeights:
    """Reference weights plus the config they were built for."""

    config: Gemma4TextConfig
    weights: Gemma4TextWeights
    dtype: Any
    dequantized_bytes: int

    def layer(self, layer_id: int) -> Gemma4LayerWeights:
        return self.weights.layers[layer_id]


def _slot_names(config: Gemma4GGUFConfig, layer_id: int) -> Mapping[str, str]:
    """Return the loader slot names this materializer needs for one layer."""

    names = {
        "attn_norm": "input_layernorm",
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_q_norm": "q_norm",
        "attn_k_norm": "k_norm",
        "attn_output": "o_proj",
        "post_attention_norm": "post_attention_layernorm",
        "ffn_norm": "pre_feedforward_layernorm",
        "ffn_gate": "mlp_gate_proj",
        "ffn_up": "mlp_up_proj",
        "ffn_down": "mlp_down_proj",
        "ffn_gate_inp": "router_proj",
        "ffn_gate_inp_scale": "router_scale",
        "ffn_gate_up_exps": "experts_gate_up_proj",
        "ffn_down_exps": "experts_down_proj",
        "ffn_down_exps_scale": "router_per_expert_scale",
        "pre_ffw_norm_2": "pre_feedforward_layernorm_2",
        "post_ffw_norm": "post_feedforward_layernorm",
        "post_ffw_norm_1": "post_feedforward_layernorm_1",
        "post_ffw_norm_2": "post_feedforward_layernorm_2",
        "layer_output_scale": "layer_scalar",
    }
    if config.is_sliding(layer_id):
        names["attn_v"] = "v_proj"
    return names


def gemma4_layer_slot_names(config: Gemma4GGUFConfig, layer_id: int) -> Mapping[str, str]:
    """Public name for the per-layer slot mapping.

    The reference materializer and the raw device materializer must agree on
    which slots a layer has, or one of them silently loads a different tensor
    set than the other. They share this mapping rather than each keeping a copy.
    """

    return _slot_names(config, layer_id)


def _read_tensor(
    reader: GGUFReader,
    name: str,
    *,
    dtype: Any,
) -> np.ndarray:
    values = dequantize_gguf_data(
        reader.tensor_data(name),
        reader.tensor_info(name).ggml_type,
    )
    array = np.asarray(values, dtype=np.float32)
    if dtype is not np.float32:
        array = array.astype(dtype)
    return np.ascontiguousarray(array)


def materialize_gemma4_reference_weights(
    reader: GGUFReader,
    *,
    model_map: Gemma4GGUFModelMap | None = None,
    dtype: Any = np.float32,
    tensor_reader: Callable[[GGUFReader, str, Any], np.ndarray] | None = None,
) -> Gemma4GGUFReferenceWeights:
    """Build reference weights for the whole Gemma 4 text tower.

    ``tensor_reader`` overrides how one named tensor becomes a NumPy array. It
    exists so a test can drive the full mapping on a synthetic artifact without
    dequantizing a 17 GB file.
    """

    info: GGUFModelInfo = reader.info
    resolved = model_map or build_gemma4_gguf_tensor_map(info)
    resolved.validation.raise_for_errors()
    config = resolved.config
    read = tensor_reader or (
        lambda source, name, element_dtype: _read_tensor(
            source, name, dtype=element_dtype
        )
    )

    rope_freqs = read_gemma4_rope_freqs(info)
    if config.has_global_rope_freqs and rope_freqs is None:
        raise MissingGGUFTensorError(
            "Gemma 4 global layers use proportional RoPE but the artifact has no "
            "rope_freqs.weight tensor to recover the rotated span from"
        )

    fused_width = 2 * config.expert_feed_forward_length
    layers: list[Gemma4LayerWeights] = []
    total_bytes = 0

    for layer_id in range(config.block_count):
        layer_map = resolved.layer(layer_id)
        slots = _slot_names(config, layer_id)
        values: dict[str, np.ndarray] = {}
        for slot, attribute in slots.items():
            if not layer_map.has(slot):
                raise MissingGGUFTensorError(
                    f"Gemma 4 layer {layer_id} is missing GGUF tensor slot {slot!r}"
                )
            array = read(reader, layer_map.tensor(slot).name, dtype)
            values[attribute] = array
            total_bytes += array.nbytes
        if layer_map.has("attn_v") and not config.is_sliding(layer_id):
            raise ValueError(
                f"Gemma 4 global layer {layer_id} declares attention_k_eq_v but "
                "carries attn_v.weight"
            )
        split_fused_expert_gate_up(
            values["experts_gate_up_proj"],
            expected_fused_width=fused_width,
        )
        layers.append(Gemma4LayerWeights(**values))

    embed_tokens = read(
        reader, resolved.root("token_embedding").name, dtype
    )
    final_norm = read(reader, resolved.root("output_norm").name, dtype)
    total_bytes += embed_tokens.nbytes + final_norm.nbytes

    lm_head: np.ndarray | None = None
    if not config.tied_embeddings:
        lm_head = read(reader, resolved.root("lm_head").name, dtype)
        total_bytes += lm_head.nbytes

    return Gemma4GGUFReferenceWeights(
        config=gemma4_reference_config_from_gguf(config),
        weights=Gemma4TextWeights(
            embed_tokens=embed_tokens,
            final_norm=final_norm,
            layers=tuple(layers),
            lm_head=lm_head,
        ),
        dtype=dtype,
        dequantized_bytes=total_bytes,
    )
