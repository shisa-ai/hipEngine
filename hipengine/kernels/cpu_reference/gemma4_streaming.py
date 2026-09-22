"""Streaming CPU forward for a Gemma 4 GGUF artifact.

A full ``float32`` materialization of ``gemma-4-26B-A4B-it`` is about 100 GB,
which does not fit on a workstation. The routed experts are 22.8 B of the 26 B
parameters but only eight of 128 run per token, so this module keeps the small
tensors resident and dequantizes experts on demand.

The layer forward is re-expressed here rather than called through
:func:`hipengine.kernels.cpu_reference.gemma4.gemma4_decoder_layer_forward`,
because the reference takes a fully materialized layer and the whole point is to
avoid that. The orchestration is therefore duplicated, and
``tests/test_unit_gemma4_gguf_streaming.py`` pins it bit-for-bit against the
reference on identical weights so the two cannot drift apart silently.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hipengine.kernels.cpu_reference.gemma4 import (
    Gemma4AttentionGeometry,
    Gemma4TextConfig,
    gemma4_attention_forward,
    gemma4_attention_mask,
    gemma4_gelu_tanh,
    gemma4_router_topk,
    gemma4_rmsnorm,
    gemma4_rope_tables,
)
from hipengine.kernels.cpu_reference.ops import linear
from hipengine.loading.gguf import GGUFReader, MissingGGUFTensorError
from hipengine.loading.gemma4_gguf import (
    Gemma4GGUFModelMap,
    build_gemma4_gguf_tensor_map,
    read_gemma4_rope_freqs,
)
from hipengine.loading.gemma4_gguf_materialize import (
    _slot_names,
    gemma4_reference_config_from_gguf,
)
from hipengine.quant.gguf import dequantize_gguf_data

__all__ = [
    "Gemma4GGUFStreamingWeights",
    "Gemma4StreamingDenseLayer",
    "gemma4_streaming_forward",
]


def _dequantize(
    reader: GGUFReader,
    name: str,
    dtype: Any,
) -> np.ndarray:
    tensor = reader.tensor_info(name)
    values = dequantize_gguf_data(reader.tensor_data(name), tensor.ggml_type)
    return np.ascontiguousarray(np.asarray(values, dtype=dtype))


@dataclass(frozen=True)
class Gemma4StreamingDenseLayer:
    """One layer's resident tensors, deliberately without its expert stacks.

    This is not :class:`Gemma4LayerWeights`: that type requires the full expert
    tensors, and the streaming path must never hold them. Sharing the type would
    mean carrying ``None`` placeholders that the reference forward cannot use.
    """

    input_layernorm: np.ndarray
    post_attention_layernorm: np.ndarray
    pre_feedforward_layernorm: np.ndarray
    post_feedforward_layernorm: np.ndarray
    post_feedforward_layernorm_1: np.ndarray
    post_feedforward_layernorm_2: np.ndarray
    pre_feedforward_layernorm_2: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    o_proj: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray
    mlp_gate_proj: np.ndarray
    mlp_up_proj: np.ndarray
    mlp_down_proj: np.ndarray
    router_scale: np.ndarray
    router_proj: np.ndarray
    router_per_expert_scale: np.ndarray
    layer_scalar: np.ndarray
    v_proj: np.ndarray | None = None


@dataclass
class Gemma4GGUFStreamingWeights:
    """Lazy, bounded-memory access to one Gemma 4 GGUF artifact.

    ``dense_cache_bytes`` bounds the resident non-expert tensors. The embedding
    matrix and the final norm are always resident, because every step needs them
    and they are the cheapest thing to keep.
    """

    reader: GGUFReader
    model_map: Gemma4GGUFModelMap | None = None
    dtype: Any = np.float32
    dense_cache_bytes: int = 16 * 1024**3
    expert_cache_bytes: int = 8 * 1024**3
    _dense: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _experts: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _dense_bytes: int = field(default=0, init=False, repr=False)
    _expert_bytes: int = field(default=0, init=False, repr=False)
    _resolved: Gemma4GGUFModelMap = field(init=False, repr=False)
    config: Gemma4TextConfig = field(init=False)
    _geometry: tuple[Gemma4AttentionGeometry, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._resolved = self.model_map or build_gemma4_gguf_tensor_map(self.reader.info)
        self._resolved.validation.raise_for_errors()
        self.config = gemma4_reference_config_from_gguf(self._resolved.config)
        self._geometry = self.config.attention
        rope_freqs = read_gemma4_rope_freqs(self.reader.info)
        if self._resolved.config.has_global_rope_freqs and rope_freqs is None:
            raise MissingGGUFTensorError(
                "Gemma 4 global layers use proportional RoPE but the artifact has "
                "no rope_freqs.weight tensor to recover the rotated span from"
            )

    # -- tensor access ---------------------------------------------------

    @property
    def gguf_config(self):
        return self._resolved.config

    def _read(self, name: str) -> np.ndarray:
        return _dequantize(self.reader, name, self.dtype)

    def layer_weights(self, layer_id: int) -> Gemma4StreamingDenseLayer:
        """Return the layer's non-expert tensors, caching them while they fit."""

        cached = self._dense.get(layer_id)
        if cached is not None:
            self._dense.move_to_end(layer_id)
            return cached

        layer_map = self._resolved.layer(layer_id)
        slots = _slot_names(self._resolved.config, layer_id)
        values: dict[str, np.ndarray] = {}
        for slot, attribute in slots.items():
            if attribute == "experts_gate_up_proj" or attribute == "experts_down_proj":
                continue
            if not layer_map.has(slot):
                raise MissingGGUFTensorError(
                    f"Gemma 4 layer {layer_id} is missing GGUF tensor slot {slot!r}"
                )
            values[attribute] = self._read(layer_map.tensor(slot).name)
        if self._resolved.config.is_sliding(layer_id):
            values.setdefault("v_proj", None)
        layer = Gemma4StreamingDenseLayer(**values)

        cost = sum(np.asarray(value).nbytes for value in values.values() if value is not None)
        if cost <= self.dense_cache_bytes:
            self._dense[layer_id] = layer
            self._dense_bytes += cost
            while self._dense_bytes > self.dense_cache_bytes and len(self._dense) > 1:
                _, evicted = self._dense.popitem(last=False)
                self._dense_bytes -= sum(
                    np.asarray(value).nbytes
                    for value in vars(evicted).values()
                    if value is not None
                )
        return layer

    def expert_weights(
        self,
        layer_id: int,
        expert_ids: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(gate_up, down)`` for the named experts only.

        The gate_up tensor keeps its fused ``gate | up`` layout, so the returned
        array is a gathered slice along the expert axis of exactly the shape the
        reference expert forward consumes.
        """

        layer_map = self._resolved.layer(layer_id)
        gate_up_name = layer_map.tensor("ffn_gate_up_exps").name
        down_name = layer_map.tensor("ffn_down_exps").name
        wanted = [int(value) for value in expert_ids]
        missing = [expert for expert in wanted if (layer_id, expert) not in self._experts]
        for expert in dict.fromkeys(missing):
            fused = self._read_expert(gate_up_name, expert)
            down = self._read_expert(down_name, expert)
            self._experts[(layer_id, expert)] = (fused, down)
            self._expert_bytes += fused.nbytes + down.nbytes
            while self._expert_bytes > self.expert_cache_bytes and len(self._experts) > 1:
                _, evicted = self._experts.popitem(last=False)
                self._expert_bytes -= evicted[0].nbytes + evicted[1].nbytes
        return (
            np.stack([self._experts[(layer_id, expert)][0] for expert in wanted]),
            np.stack([self._experts[(layer_id, expert)][1] for expert in wanted]),
        )

    def _read_expert(self, name: str, expert: int) -> np.ndarray:
        """Dequantize one expert's slice without reading the other 127.

        The GGUF expert axis is the outermost one, so ``raw[expert]`` is a
        contiguous slice and needs no repacking.
        """

        tensor = self.reader.tensor_info(name)
        if tensor.shape[0] <= expert:
            raise IndexError(f"{name} holds {tensor.shape[0]} experts, asked for {expert}")
        slab = self.reader.tensor_data(name)[expert]
        values = dequantize_gguf_data(slab, tensor.ggml_type)
        return np.ascontiguousarray(np.asarray(values, dtype=self.dtype))

    def embed_tokens(self) -> np.ndarray:
        return self._read(self._resolved.root("token_embedding").name)

    def final_norm(self) -> np.ndarray:
        return self._read(self._resolved.root("output_norm").name)

    def lm_head(self) -> np.ndarray:
        if self._resolved.config.tied_embeddings:
            raise ValueError("this Gemma 4 artifact ties the head to the embedding")
        return self._read(self._resolved.root("lm_head").name)

    def resident_bytes(self) -> tuple[int, int]:
        """Return ``(dense_bytes, expert_bytes)`` currently held."""

        return self._dense_bytes, self._expert_bytes

    def cached_experts(self) -> tuple[tuple[int, int], ...]:
        """Return the ``(layer, expert)`` pairs currently resident."""

        return tuple(self._experts)

    def reset_caches(self) -> None:
        """Drop every resident tensor so a measurement starts from zero."""

        self._dense.clear()
        self._experts.clear()
        self._dense_bytes = 0
        self._expert_bytes = 0


def gemma4_streaming_forward(
    streaming: Gemma4GGUFStreamingWeights,
    token_ids: Sequence[int],
    *,
    positions: Sequence[int] | None = None,
    embed_tokens: np.ndarray | None = None,
    final_norm: np.ndarray | None = None,
) -> np.ndarray:
    """Run the Gemma 4 text tower and return ``float32`` logits.

    This is a prefill-shaped forward: every call re-reads the whole token block.
    It exists to validate the artifact, not to serve requests, so the quadratic
    re-read is accepted rather than worked around.
    """

    config = streaming.config
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise ValueError("token_ids must not be empty")
    if positions is None:
        position_values = np.arange(ids.shape[0], dtype=np.int64)
    else:
        position_values = np.asarray(positions, dtype=np.int64).reshape(-1)
        if position_values.shape[0] != ids.shape[0]:
            raise ValueError("positions must have one entry per token")

    embedding = streaming.embed_tokens() if embed_tokens is None else embed_tokens
    norm_weight = streaming.final_norm() if final_norm is None else final_norm

    hidden = (np.asarray(embedding, dtype=np.float32)[ids] * np.float32(config.embed_scale)).astype(
        np.float32
    )

    for layer_id in range(config.num_hidden_layers):
        hidden = _streaming_layer_forward(
            hidden,
            streaming,
            layer_id,
            config.geometry(layer_id),
            config,
            positions=position_values,
        )

    hidden = gemma4_rmsnorm(hidden, norm_weight, config.rms_norm_eps)
    head = embedding if config.tie_word_embeddings else streaming.lm_head()
    logits = linear(hidden, head)
    if config.final_logit_softcapping is not None:
        cap = np.float32(config.final_logit_softcapping)
        logits = (np.tanh(logits / cap) * cap).astype(np.float32)
    return logits


def _streaming_layer_forward(
    hidden: np.ndarray,
    streaming: Gemma4GGUFStreamingWeights,
    layer_id: int,
    geometry: Gemma4AttentionGeometry,
    config: Gemma4TextConfig,
    *,
    positions: np.ndarray,
) -> np.ndarray:
    """One decoder layer, mirroring the reference layer forward exactly."""

    layer = streaming.layer_weights(layer_id)
    eps = config.rms_norm_eps
    value = np.asarray(hidden, dtype=np.float32)

    cos, sin = gemma4_rope_tables(geometry.rope, positions)
    cos_expanded = cos[:, None, :]
    sin_expanded = sin[:, None, :]
    mask = gemma4_attention_mask(geometry, positions, positions)

    residual = value
    normalized = gemma4_rmsnorm(value, layer.input_layernorm, eps)
    attended = gemma4_attention_forward(
        normalized,
        geometry,
        q_proj=layer.q_proj,
        k_proj=layer.k_proj,
        v_proj=layer.v_proj,
        o_proj=layer.o_proj,
        q_norm=layer.q_norm,
        k_norm=layer.k_norm,
        cos=cos_expanded,
        sin=sin_expanded,
        positions=positions,
        mask=mask,
        eps=eps,
    )
    hidden_after_attention = residual + gemma4_rmsnorm(
        attended, layer.post_attention_layernorm, eps
    )

    residual = hidden_after_attention
    dense_input = gemma4_rmsnorm(hidden_after_attention, layer.pre_feedforward_layernorm, eps)
    dense = _dense_mlp(
        dense_input,
        gate_proj=layer.mlp_gate_proj,
        up_proj=layer.mlp_up_proj,
        down_proj=layer.mlp_down_proj,
    )
    dense = gemma4_rmsnorm(dense, layer.post_feedforward_layernorm_1, eps)

    _, top_k_weights, top_k_index = gemma4_router_topk(
        residual,
        norm_weight=None,
        scale=layer.router_scale,
        proj_weight=layer.router_proj,
        per_expert_scale=layer.router_per_expert_scale,
        top_k=config.top_k_experts,
        scalar_root_size=config.router_scalar_root_size,
        eps=eps,
    )
    expert_input = gemma4_rmsnorm(residual, layer.pre_feedforward_layernorm_2, eps)

    # Only the experts this token block actually selected are dequantized.
    selected = np.unique(top_k_index)
    gate_up, down = streaming.expert_weights(layer_id, selected.tolist())
    # ``-1`` marks an expert the gather did not load, so an unexpected index
    # raises instead of silently reading expert 0.
    remap = np.full(config.num_experts, -1, dtype=np.int64)
    remap[selected] = np.arange(selected.shape[0])
    local_index = remap[top_k_index]
    if np.any(local_index < 0):  # pragma: no cover - guarded by the gather above
        raise AssertionError(f"layer {layer_id} routed to an expert that was not dequantized")
    experts = _selected_experts(
        expert_input,
        local_index,
        top_k_weights,
        gate_up_proj=gate_up,
        down_proj=down,
    )
    experts = gemma4_rmsnorm(experts, layer.post_feedforward_layernorm_2, eps)

    combined = gemma4_rmsnorm(dense + experts, layer.post_feedforward_layernorm, eps)
    out = residual + combined
    return (out * np.asarray(layer.layer_scalar, dtype=np.float32).reshape(())).astype(np.float32)


def _dense_mlp(
    hidden: np.ndarray,
    *,
    gate_proj: np.ndarray,
    up_proj: np.ndarray,
    down_proj: np.ndarray,
) -> np.ndarray:
    activated = gemma4_gelu_tanh(linear(hidden, gate_proj)) * linear(hidden, up_proj)
    return linear(activated, down_proj)


def _selected_experts(
    hidden: np.ndarray,
    local_index: np.ndarray,
    weights: np.ndarray,
    *,
    gate_up_proj: np.ndarray,
    down_proj: np.ndarray,
) -> np.ndarray:
    """Routed expert FFN over a gathered, possibly renumbered, expert stack."""

    value = np.asarray(hidden, dtype=np.float32)
    out = np.zeros_like(value)
    for token in range(value.shape[0]):
        row = value[token]
        for slot in range(local_index.shape[-1]):
            expert = int(local_index[token, slot])
            projection = gate_up_proj[expert] @ row
            gate, up = np.split(projection, 2)
            activated = gemma4_gelu_tanh(gate) * up
            out[token] += (down_proj[expert] @ activated) * weights[token, slot]
    return out
