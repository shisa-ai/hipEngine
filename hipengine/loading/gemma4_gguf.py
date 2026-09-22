"""Torch-free Gemma 4 GGUF metadata and tensor-contract loading.

Two Gemma 4 details are not expressible in the GGUF metadata that llama.cpp
writes for this architecture, and both are handled here explicitly:

* **The rotated span of the global layers.** ``gemma4.rope.dimension_count`` is
  the full head width (512), not the rotated width. The proportional
  ``partial_rotary_factor`` never reaches the GGUF as a key. The real rotated
  pair count is recovered from ``rope_freqs.weight``, where llama.cpp encodes
  the unrotated tail as an enormous frequency factor. Counting the finite
  factors gives the rotated pair count, which is validated against the tensor's
  shape rather than trusted.
* **Per-layer attention geometry.** ``head_count_kv`` is a per-layer array, and
  ``key_length``/``key_length_swa`` split the head width by layer type. The KV
  cache and attention kernels therefore cannot use one uniform geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.loading.gguf import (
    GGUFModelInfo,
    GGUFTensorInfo,
    MissingGGUFTensorError,
)
from hipengine.quant.gguf import dequantize_gguf_data, numpy_storage_dtype

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"

_GEMMA4_ARCHITECTURE = "gemma4"

# llama.cpp writes this sentinel frequency factor for the unrotated tail of a
# proportional-RoPE head. Anything at or above it means "do not rotate".
_UNROTATED_FREQ_FACTOR_FLOOR = 1.0e29

_ROOT_SLOTS = {
    "token_embedding": "token_embd.weight",
    "output_norm": "output_norm.weight",
    "lm_head": "output.weight",
    "rope_freqs": "rope_freqs.weight",
}

_COMMON_LAYER_SLOTS = {
    "attn_norm": "attn_norm.weight",
    "attn_q": "attn_q.weight",
    "attn_k": "attn_k.weight",
    "attn_q_norm": "attn_q_norm.weight",
    "attn_k_norm": "attn_k_norm.weight",
    "attn_output": "attn_output.weight",
    "post_attention_norm": "post_attention_norm.weight",
    "ffn_norm": "ffn_norm.weight",
    "ffn_gate": "ffn_gate.weight",
    "ffn_up": "ffn_up.weight",
    "ffn_down": "ffn_down.weight",
    "ffn_gate_inp": "ffn_gate_inp.weight",
    "ffn_gate_inp_scale": "ffn_gate_inp.scale",
    "pre_ffw_norm_2": "pre_ffw_norm_2.weight",
    "post_ffw_norm": "post_ffw_norm.weight",
    "post_ffw_norm_1": "post_ffw_norm_1.weight",
    "post_ffw_norm_2": "post_ffw_norm_2.weight",
    "ffn_gate_up_exps": "ffn_gate_up_exps.weight",
    "ffn_down_exps": "ffn_down_exps.weight",
    "ffn_down_exps_scale": "ffn_down_exps.scale",
    "layer_output_scale": "layer_output_scale.weight",
}

_SLIDING_ONLY_LAYER_SLOTS = {
    "attn_v": "attn_v.weight",
}


@dataclass(frozen=True)
class Gemma4GGUFRoPEConfig:
    """One layer-family RoPE contract decoded from GGUF metadata."""

    rope_type: str
    head_dim: int
    rotated_pairs: int
    freq_base: float

    def __post_init__(self) -> None:
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("Gemma 4 RoPE head_dim must be a positive even number")
        if not 0 <= self.rotated_pairs <= self.head_dim // 2:
            raise ValueError("Gemma 4 rotated pair count must be within the head half-width")

    @property
    def is_partial(self) -> bool:
        return self.rotated_pairs != self.head_dim // 2


@dataclass(frozen=True)
class Gemma4GGUFConfig:
    """Validated architecture dimensions for one Gemma 4 GGUF artifact."""

    architecture: str
    block_count: int
    hidden_size: int
    vocab_size: int
    context_length: int
    rms_norm_eps: float
    head_counts: tuple[int, ...]
    head_count_kv: tuple[int, ...]
    key_length: int
    key_length_swa: int
    value_length: int
    value_length_swa: int
    sliding_window: int
    layer_types: tuple[str, ...]
    feed_forward_lengths: tuple[int, ...]
    expert_count: int
    expert_used_count: int
    expert_feed_forward_length: int
    final_logit_softcapping: float | None
    full_rope: Gemma4GGUFRoPEConfig
    swa_rope: Gemma4GGUFRoPEConfig
    shared_kv_layers: int
    tied_embeddings: bool
    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]

    def _check_layer(self, layer_id: int) -> int:
        layer = int(layer_id)
        if layer < 0 or layer >= self.block_count:
            raise IndexError(f"layer_id {layer} outside [0, {self.block_count})")
        return layer

    def layer_type(self, layer_id: int) -> str:
        return self.layer_types[self._check_layer(layer_id)]

    def is_sliding(self, layer_id: int) -> bool:
        return self.layer_type(layer_id) == SLIDING_ATTENTION

    def head_count(self, layer_id: int) -> int:
        return self.head_counts[self._check_layer(layer_id)]

    def head_count_kv_for(self, layer_id: int) -> int:
        return self.head_count_kv[self._check_layer(layer_id)]

    def head_dim(self, layer_id: int) -> int:
        layer = self._check_layer(layer_id)
        return self.key_length_swa if self.layer_types[layer] == SLIDING_ATTENTION else self.key_length

    def value_dim(self, layer_id: int) -> int:
        layer = self._check_layer(layer_id)
        return (
            self.value_length_swa
            if self.layer_types[layer] == SLIDING_ATTENTION
            else self.value_length
        )

    def kv_width(self, layer_id: int) -> int:
        return self.head_count_kv_for(layer_id) * self.head_dim(layer_id)

    def query_width(self, layer_id: int) -> int:
        return self.head_count(layer_id) * self.head_dim(layer_id)

    def feed_forward_length(self, layer_id: int) -> int:
        return self.feed_forward_lengths[self._check_layer(layer_id)]

    def rope_for_layer(self, layer_id: int) -> Gemma4GGUFRoPEConfig:
        layer = self._check_layer(layer_id)
        return self.swa_rope if self.layer_types[layer] == SLIDING_ATTENTION else self.full_rope

    def attention_k_eq_v(self, layer_id: int) -> bool:
        """Global layers carry no ``v_proj``; V is the raw K projection."""

        return not self.is_sliding(layer_id)

    @property
    def embed_scale(self) -> float:
        return float(self.hidden_size) ** 0.5

    @property
    def has_global_rope_freqs(self) -> bool:
        return self.full_rope.is_partial


@dataclass(frozen=True)
class Gemma4GGUFMappingValidation:
    """Result of validating the architecture's complete tensor inventory."""

    config: Gemma4GGUFConfig
    present: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_errors: tuple[str, ...]
    type_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (self.missing or self.unexpected or self.shape_errors or self.type_errors)

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        parts: list[str] = []
        for label, errors, limit in (
            ("missing tensors", self.missing, 8),
            ("unexpected tensors", self.unexpected, 8),
            ("shape errors", self.shape_errors, 4),
            ("type errors", self.type_errors, 4),
        ):
            if not errors:
                continue
            preview = "; ".join(errors[:limit])
            more = "" if len(errors) <= limit else f" (+{len(errors) - limit} more)"
            parts.append(f"{label}: {preview}{more}")
        raise MissingGGUFTensorError("; ".join(parts))


@dataclass(frozen=True)
class Gemma4GGUFLayerMap:
    """Canonical Gemma 4 tensor slots and attention geometry for one layer."""

    layer_id: int
    attention_type: str
    head_count: int
    head_count_kv: int
    head_dim: int
    k_eq_v: bool
    tensors: Mapping[str, GGUFTensorInfo]

    def tensor(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(
                f"Gemma 4 layer {self.layer_id} has no GGUF tensor slot {slot!r}"
            ) from exc

    def has(self, slot: str) -> bool:
        return slot in self.tensors

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors.values())


@dataclass(frozen=True)
class Gemma4GGUFModelMap:
    """Canonical root and layer tensor map for one Gemma 4 GGUF artifact."""

    config: Gemma4GGUFConfig
    root_tensors: Mapping[str, GGUFTensorInfo]
    layers: tuple[Gemma4GGUFLayerMap, ...]
    validation: Gemma4GGUFMappingValidation

    def root(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.root_tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(
                f"Gemma 4 model has no GGUF root tensor slot {slot!r}"
            ) from exc

    def layer(self, layer_id: int) -> Gemma4GGUFLayerMap:
        return self.layers[layer_id]

    @property
    def tensor_names(self) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for tensor in self.root_tensors.values():
            if tensor.name not in seen:
                seen.add(tensor.name)
                names.append(tensor.name)
        for layer in self.layers:
            for name in layer.tensor_names:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return tuple(names)


# ---------------------------------------------------------------------------
# Metadata decoding
# ---------------------------------------------------------------------------


def gemma4_rotated_pair_count(rope_freqs: Sequence[float], *, head_dim: int) -> int:
    """Return the rotated pair count encoded in a ``rope_freqs`` tensor.

    llama.cpp writes ``1.0`` for a rotated pair and an enormous sentinel for an
    unrotated one, so the finite prefix length is the rotated span. The pattern
    is validated rather than assumed: a non-prefix layout would mean the tensor
    does not describe proportional RoPE and must not be silently truncated.
    """

    values = np.asarray(rope_freqs, dtype=np.float32).reshape(-1)
    half = int(head_dim) // 2
    if values.shape[0] != half:
        raise ValueError(
            f"Gemma 4 rope_freqs must hold head_dim // 2 = {half} factors, got {values.shape[0]}"
        )
    rotated = values < np.float32(_UNROTATED_FREQ_FACTOR_FLOOR)
    count = int(np.count_nonzero(rotated))
    if count and not bool(np.all(rotated[:count])) or (count < half and bool(np.any(rotated[count:]))):
        raise ValueError(
            "Gemma 4 rope_freqs must be a rotated prefix followed by unrotated factors"
        )
    if count and not np.allclose(values[:count], 1.0):
        raise ValueError("Gemma 4 rope_freqs rotated factors must all be 1.0")
    return count


def read_gemma4_rope_freqs(info: GGUFModelInfo) -> np.ndarray | None:
    """Read ``rope_freqs.weight`` from the artifact without rescanning the header."""

    tensor = next((item for item in info.tensors if item.name == "rope_freqs.weight"), None)
    if tensor is None:
        return None
    storage = np.memmap(
        info.path,
        mode="r",
        dtype=numpy_storage_dtype(tensor.ggml_type),
        offset=tensor.data_offset,
        shape=tensor.byte_shape,
    )
    return np.asarray(dequantize_gguf_data(storage, tensor.ggml_type), dtype=np.float32).reshape(-1)


def gemma4_gguf_config_from_metadata(
    info: GGUFModelInfo,
    *,
    rope_freqs: Sequence[float] | None = None,
) -> Gemma4GGUFConfig:
    """Decode and validate Gemma 4 architecture metadata without reading weights.

    ``rope_freqs`` may be supplied to avoid touching the file; when omitted and
    the artifact carries ``rope_freqs.weight`` it is read directly. Global
    layers without the tensor fall back to full rotation, which is what the
    metadata alone implies.
    """

    metadata = info.metadata
    architecture = str(metadata.get("general.architecture", ""))
    if architecture != _GEMMA4_ARCHITECTURE:
        raise ValueError(f"expected GGUF architecture 'gemma4', got {architecture!r}")

    prefix = _GEMMA4_ARCHITECTURE
    block_count = _positive_int(metadata, f"{prefix}.block_count")
    hidden_size = _positive_int(metadata, f"{prefix}.embedding_length")
    context_length = _positive_int(metadata, f"{prefix}.context_length")
    rms_norm_eps = float(_required(metadata, f"{prefix}.attention.layer_norm_rms_epsilon"))
    if rms_norm_eps <= 0.0:
        raise ValueError("Gemma 4 RMS epsilon must be positive")

    head_counts = _per_layer_ints(
        metadata, f"{prefix}.attention.head_count", block_count, label="head_count"
    )
    head_count_kv = _per_layer_ints(
        metadata, f"{prefix}.attention.head_count_kv", block_count, label="head_count_kv"
    )
    for layer_id in range(block_count):
        if head_counts[layer_id] % head_count_kv[layer_id]:
            raise ValueError(
                f"Gemma 4 head_count {head_counts[layer_id]} at layer {layer_id} must be "
                f"divisible by head_count_kv {head_count_kv[layer_id]}"
            )

    key_length = _positive_int(metadata, f"{prefix}.attention.key_length")
    value_length = _positive_int(metadata, f"{prefix}.attention.value_length")
    key_length_swa = int(metadata.get(f"{prefix}.attention.key_length_swa", key_length))
    value_length_swa = int(metadata.get(f"{prefix}.attention.value_length_swa", value_length))
    for label, value in (
        ("key_length_swa", key_length_swa),
        ("value_length_swa", value_length_swa),
    ):
        if value <= 0 or value % 2:
            raise ValueError(f"Gemma 4 {label} must be a positive even number")

    sliding_window = int(metadata.get(f"{prefix}.attention.sliding_window", 0) or 0)
    if sliding_window <= 0:
        raise ValueError("Gemma 4 requires a positive sliding window")
    layer_types = _layer_types(metadata, block_count, prefix=prefix, sliding_window=sliding_window)

    feed_forward_lengths = _per_layer_ints(
        metadata,
        f"{prefix}.feed_forward_length",
        block_count,
        label="feed_forward_length",
    )

    expert_count = _positive_int(metadata, f"{prefix}.expert_count")
    expert_used_count = _positive_int(metadata, f"{prefix}.expert_used_count")
    if expert_used_count > expert_count:
        raise ValueError("Gemma 4 expert_used_count must be <= expert_count")
    expert_feed_forward_length = _positive_int(
        metadata, f"{prefix}.expert_feed_forward_length"
    )

    softcap_value = metadata.get(f"{prefix}.final_logit_softcapping")
    final_logit_softcapping = None if softcap_value is None else float(softcap_value)
    if final_logit_softcapping is not None and final_logit_softcapping <= 0.0:
        raise ValueError("Gemma 4 final_logit_softcapping must be positive when set")

    freq_base = float(metadata.get(f"{prefix}.rope.freq_base", 1_000_000.0))
    freq_base_swa = float(metadata.get(f"{prefix}.rope.freq_base_swa", 10_000.0))
    for label, value in (("freq_base", freq_base), ("freq_base_swa", freq_base_swa)):
        if value <= 0.0:
            raise ValueError(f"Gemma 4 rope {label} must be positive")

    if rope_freqs is None:
        rope_freqs = read_gemma4_rope_freqs(info)
    full_rotated_pairs = key_length // 2
    if rope_freqs is not None:
        full_rotated_pairs = gemma4_rotated_pair_count(rope_freqs, head_dim=key_length)

    full_rope = Gemma4GGUFRoPEConfig(
        rope_type="proportional" if full_rotated_pairs != key_length // 2 else "default",
        head_dim=key_length,
        rotated_pairs=full_rotated_pairs,
        freq_base=freq_base,
    )
    swa_rope = Gemma4GGUFRoPEConfig(
        rope_type="default",
        head_dim=key_length_swa,
        rotated_pairs=key_length_swa // 2,
        freq_base=freq_base_swa,
    )

    shared_kv_layers = int(metadata.get(f"{prefix}.attention.shared_kv_layers", 0) or 0)
    if shared_kv_layers:
        raise ValueError(
            "Gemma 4 KV sharing is not implemented; this artifact declares "
            f"{shared_kv_layers} shared KV layers"
        )

    vocab_size = _optional_positive_int(metadata, f"{prefix}.vocab_size") or _optional_positive_int(
        metadata, "tokenizer.ggml.tokens_length"
    )
    if vocab_size is None:
        token_count = metadata.get("tokenizer.ggml.tokens")
        vocab_size = len(token_count) if token_count is not None else None
    if not vocab_size:
        raise ValueError("Gemma 4 vocab size could not be determined from GGUF metadata")

    tied_embeddings = not any(tensor.name == "output.weight" for tensor in info.tensors)
    bos_token_id = _optional_int(metadata.get("tokenizer.ggml.bos_token_id"))
    eos_token_ids = _token_ids(metadata.get("tokenizer.ggml.eos_token_id"))

    return Gemma4GGUFConfig(
        architecture=architecture,
        block_count=block_count,
        hidden_size=hidden_size,
        vocab_size=int(vocab_size),
        context_length=context_length,
        rms_norm_eps=rms_norm_eps,
        head_counts=head_counts,
        head_count_kv=head_count_kv,
        key_length=key_length,
        key_length_swa=key_length_swa,
        value_length=value_length,
        value_length_swa=value_length_swa,
        sliding_window=sliding_window,
        layer_types=layer_types,
        feed_forward_lengths=feed_forward_lengths,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        expert_feed_forward_length=expert_feed_forward_length,
        final_logit_softcapping=final_logit_softcapping,
        full_rope=full_rope,
        swa_rope=swa_rope,
        shared_kv_layers=shared_kv_layers,
        tied_embeddings=tied_embeddings,
        bos_token_id=bos_token_id,
        eos_token_ids=eos_token_ids,
    )


def _layer_types(
    metadata: Mapping[str, Any],
    block_count: int,
    *,
    prefix: str,
    sliding_window: int,
) -> tuple[str, ...]:
    """Decode the per-layer attention type from the sliding-window pattern.

    ``sliding_window_pattern`` is written as a per-layer boolean array where
    ``True`` means sliding attention. Older converters wrote the pattern length
    instead, so both encodings are accepted.
    """

    pattern = metadata.get(f"{prefix}.attention.sliding_window_pattern")
    if isinstance(pattern, (list, tuple)) and pattern and all(
        isinstance(item, (bool, int)) for item in pattern
    ):
        if len(pattern) != block_count:
            raise ValueError(
                "Gemma 4 sliding_window_pattern length "
                f"{len(pattern)} does not match block_count {block_count}"
            )
        layer_types = tuple(
            SLIDING_ATTENTION if bool(item) else FULL_ATTENTION for item in pattern
        )
    else:
        stride = int(pattern) if pattern else 6
        if stride < 2:
            raise ValueError("Gemma 4 sliding-window pattern must be at least 2")
        layer_types = tuple(
            SLIDING_ATTENTION if (layer_id + 1) % stride else FULL_ATTENTION
            for layer_id in range(block_count)
        )
    if sliding_window and layer_types[-1] != FULL_ATTENTION:
        # The reference implementation forces the last layer global.
        layer_types = (*layer_types[:-1], FULL_ATTENTION)
    return layer_types


# ---------------------------------------------------------------------------
# Tensor contract
# ---------------------------------------------------------------------------


def required_gemma4_gguf_tensor_names(config: Gemma4GGUFConfig) -> tuple[str, ...]:
    names: list[str] = []
    for slot, name in _ROOT_SLOTS.items():
        if slot == "lm_head" and config.tied_embeddings:
            continue
        if slot == "rope_freqs" and not config.has_global_rope_freqs:
            continue
        names.append(name)
    for layer_id in range(config.block_count):
        names.extend(
            f"blk.{layer_id}.{suffix}" for suffix in _layer_slots(config, layer_id).values()
        )
    return tuple(dict.fromkeys(names))


def validate_gemma4_gguf_tensor_map(info: GGUFModelInfo) -> Gemma4GGUFMappingValidation:
    config = gemma4_gguf_config_from_metadata(info)
    actual = {tensor.name: tensor for tensor in info.tensors}
    required = set(required_gemma4_gguf_tensor_names(config))
    actual_names = set(actual)
    return Gemma4GGUFMappingValidation(
        config=config,
        present=tuple(sorted(required & actual_names)),
        missing=tuple(sorted(required - actual_names)),
        unexpected=tuple(sorted(actual_names - required)),
        shape_errors=tuple(_tensor_shape_errors(config, actual)),
        type_errors=tuple(_tensor_type_errors(config, actual)),
    )


def build_gemma4_gguf_tensor_map(
    info: GGUFModelInfo,
    *,
    strict: bool = True,
) -> Gemma4GGUFModelMap:
    validation = validate_gemma4_gguf_tensor_map(info)
    if strict:
        validation.raise_for_errors()
    config = validation.config
    actual = {tensor.name: tensor for tensor in info.tensors}
    roots = MappingProxyType(
        {
            slot: actual[name]
            for slot, name in _ROOT_SLOTS.items()
            if name in actual
        }
    )
    layers = tuple(
        _build_layer_map(config, actual, layer_id) for layer_id in range(config.block_count)
    )
    return Gemma4GGUFModelMap(
        config=config,
        root_tensors=roots,
        layers=layers,
        validation=validation,
    )


def _layer_slots(config: Gemma4GGUFConfig, layer_id: int) -> dict[str, str]:
    slots = dict(_COMMON_LAYER_SLOTS)
    if config.is_sliding(layer_id):
        slots.update(_SLIDING_ONLY_LAYER_SLOTS)
    return slots


def _build_layer_map(
    config: Gemma4GGUFConfig,
    actual: Mapping[str, GGUFTensorInfo],
    layer_id: int,
) -> Gemma4GGUFLayerMap:
    slots = _layer_slots(config, layer_id)
    tensors = MappingProxyType(
        {
            slot: actual[f"blk.{layer_id}.{suffix}"]
            for slot, suffix in slots.items()
            if f"blk.{layer_id}.{suffix}" in actual
        }
    )
    return Gemma4GGUFLayerMap(
        layer_id=layer_id,
        attention_type=config.layer_type(layer_id),
        head_count=config.head_count(layer_id),
        head_count_kv=config.head_count_kv_for(layer_id),
        head_dim=config.head_dim(layer_id),
        k_eq_v=config.attention_k_eq_v(layer_id),
        tensors=tensors,
    )


def _expected_shapes(config: Gemma4GGUFConfig) -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (config.vocab_size, config.hidden_size),
        "output_norm.weight": (config.hidden_size,),
    }
    if not config.tied_embeddings:
        expected["output.weight"] = (config.vocab_size, config.hidden_size)
    if config.has_global_rope_freqs:
        expected["rope_freqs.weight"] = (config.key_length // 2,)

    for layer_id in range(config.block_count):
        prefix = f"blk.{layer_id}"
        heads = config.head_count(layer_id)
        kv_heads = config.head_count_kv_for(layer_id)
        head_dim = config.head_dim(layer_id)
        dense = config.feed_forward_length(layer_id)
        experts = config.expert_count
        expert_ff = config.expert_feed_forward_length

        expected.update(
            {
                f"{prefix}.attn_norm.weight": (config.hidden_size,),
                f"{prefix}.attn_q.weight": (heads * head_dim, config.hidden_size),
                f"{prefix}.attn_k.weight": (kv_heads * head_dim, config.hidden_size),
                f"{prefix}.attn_q_norm.weight": (head_dim,),
                f"{prefix}.attn_k_norm.weight": (head_dim,),
                f"{prefix}.attn_output.weight": (config.hidden_size, heads * head_dim),
                f"{prefix}.post_attention_norm.weight": (config.hidden_size,),
                f"{prefix}.ffn_norm.weight": (config.hidden_size,),
                f"{prefix}.ffn_gate.weight": (dense, config.hidden_size),
                f"{prefix}.ffn_up.weight": (dense, config.hidden_size),
                f"{prefix}.ffn_down.weight": (config.hidden_size, dense),
                f"{prefix}.ffn_gate_inp.weight": (experts, config.hidden_size),
                f"{prefix}.ffn_gate_inp.scale": (config.hidden_size,),
                f"{prefix}.pre_ffw_norm_2.weight": (config.hidden_size,),
                f"{prefix}.post_ffw_norm.weight": (config.hidden_size,),
                f"{prefix}.post_ffw_norm_1.weight": (config.hidden_size,),
                f"{prefix}.post_ffw_norm_2.weight": (config.hidden_size,),
                f"{prefix}.ffn_gate_up_exps.weight": (
                    experts,
                    2 * expert_ff,
                    config.hidden_size,
                ),
                f"{prefix}.ffn_down_exps.weight": (
                    experts,
                    config.hidden_size,
                    expert_ff,
                ),
                f"{prefix}.ffn_down_exps.scale": (experts,),
                f"{prefix}.layer_output_scale.weight": (1,),
            }
        )
        if config.is_sliding(layer_id):
            expected[f"{prefix}.attn_v.weight"] = (kv_heads * head_dim, config.hidden_size)
    return expected


def _tensor_shape_errors(
    config: Gemma4GGUFConfig,
    actual: Mapping[str, GGUFTensorInfo],
) -> list[str]:
    errors: list[str] = []
    for name, shape in sorted(_expected_shapes(config).items()):
        tensor = actual.get(name)
        if tensor is None:
            continue
        if tuple(tensor.shape) != tuple(shape):
            errors.append(f"{name} expected {shape} got {tuple(tensor.shape)}")
    return errors


def _tensor_type_errors(
    config: Gemma4GGUFConfig,
    actual: Mapping[str, GGUFTensorInfo],
) -> list[str]:
    """Reject quantized storage on tensors the runtime reads as fp32 scales."""

    errors: list[str] = []
    for layer_id in range(config.block_count):
        for suffix in ("ffn_gate_inp.scale", "ffn_down_exps.scale", "layer_output_scale.weight"):
            tensor = actual.get(f"blk.{layer_id}.{suffix}")
            if tensor is None:
                continue
            if tensor.ggml_type_name not in ("F32", "F16", "BF16"):
                errors.append(
                    f"{tensor.name} must be stored as a float type, got {tensor.ggml_type_name}"
                )
    return errors


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _required(metadata: Mapping[str, Any], key: str) -> Any:
    if key not in metadata:
        raise KeyError(f"GGUF metadata is missing required key {key!r}")
    return metadata[key]


def _positive_int(metadata: Mapping[str, Any], key: str) -> int:
    value = int(_required(metadata, key))
    if value <= 0:
        raise ValueError(f"GGUF key {key!r} must be positive, got {value}")
    return value


def _optional_positive_int(metadata: Mapping[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"GGUF key {key!r} must be positive, got {value}")
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _token_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return (int(value),)


def _per_layer_ints(
    metadata: Mapping[str, Any],
    key: str,
    block_count: int,
    *,
    label: str,
) -> tuple[int, ...]:
    """Read a scalar-or-per-layer integer GGUF key into a per-layer tuple."""

    value = _required(metadata, key)
    if isinstance(value, (list, tuple)):
        if len(value) != block_count:
            raise ValueError(
                f"Gemma 4 {label} array length {len(value)} does not match "
                f"block_count {block_count}"
            )
        values = tuple(int(item) for item in value)
    else:
        values = (int(value),) * block_count
    for layer_id, item in enumerate(values):
        if item <= 0:
            raise ValueError(f"Gemma 4 {label} at layer {layer_id} must be positive")
    return values


__all__ = [
    "FULL_ATTENTION",
    "SLIDING_ATTENTION",
    "Gemma4GGUFConfig",
    "Gemma4GGUFLayerMap",
    "Gemma4GGUFMappingValidation",
    "Gemma4GGUFModelMap",
    "Gemma4GGUFRoPEConfig",
    "build_gemma4_gguf_tensor_map",
    "gemma4_gguf_config_from_metadata",
    "gemma4_rotated_pair_count",
    "read_gemma4_rope_freqs",
    "required_gemma4_gguf_tensor_names",
    "validate_gemma4_gguf_tensor_map",
]
