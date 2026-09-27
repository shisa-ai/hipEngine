"""Tensor contract for the Gemma 4 ``gemma4-assistant`` MTP draft head.

The head is a separate artifact from the target model. Its
``general.architecture`` is ``gemma4-assistant``, and it is attached to a
``gemma4`` target through ``hipengine.generation.qwen35_gguf_mtp2_registry``.

Shape contract, read from ``MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf``
(unsloth/gemma-4-26B-A4B-it-GGUF) and cross-checked against llama.cpp's
``src/models/gemma4-assistant.cpp`` at ``llama.cpp@17252c769``:

* four blocks, each with eleven tensors under ``blk.<i>.``
* five global tensors: ``token_embd``, ``output_norm``, ``rope_freqs``,
  ``nextn.pre_projection``, ``nextn.post_projection``
* no ``attn_k`` and no ``attn_v``: ``attention.shared_kv_layers`` equals the
  block count, so the head attends against the backbone's KV cache and owns no
  cache of its own
* the last block is full attention and the preceding blocks are sliding-window,
  which changes the query width (``head_count * key_length`` against
  ``head_count * key_length_swa``) and the query-norm width

This module owns the shape and dtype contract only. It does not load weights,
and it does not decide whether the head may run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hipengine.loading.gguf import (
    GGUFModelInfo,
    GGUFTensorInfo,
    MissingGGUFTensorError,
)

__all__ = [
    "Gemma4AssistantConfig",
    "Gemma4AssistantTensorMap",
    "Gemma4AssistantValidation",
    "build_gemma4_assistant_tensor_map",
    "expected_gemma4_assistant_shapes",
    "gemma4_assistant_config_from_metadata",
    "required_gemma4_assistant_tensor_names",
    "validate_gemma4_assistant_tensor_map",
]

ARCHITECTURE = "gemma4-assistant"
_PREFIX = f"{ARCHITECTURE}."

# Per-block tensor suffixes, in the order the artifact lists them.
_BLOCK_SLOTS: tuple[str, ...] = (
    "attn_norm.weight",
    "layer_output_scale.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
    "post_ffw_norm.weight",
    "ffn_norm.weight",
    "attn_output.weight",
    "attn_q_norm.weight",
    "attn_q.weight",
)

# Global tensor names.
_GLOBAL_SLOTS: tuple[str, ...] = (
    "token_embd.weight",
    "output_norm.weight",
    "rope_freqs.weight",
    "nextn.post_projection.weight",
    "nextn.pre_projection.weight",
)

# Block slots that vary between sliding-window and full-attention blocks.
_SWA_SENSITIVE = frozenset({"attn_output.weight", "attn_q_norm.weight", "attn_q.weight"})


def _require_int(metadata: Mapping[str, Any], key: str) -> int:
    try:
        value = metadata[_PREFIX + key]
    except KeyError as exc:
        raise MissingGGUFTensorError(f"{ARCHITECTURE} metadata is missing {key!r}") from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise MissingGGUFTensorError(
            f"{ARCHITECTURE} metadata key {key!r} must be an integer, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class Gemma4AssistantConfig:
    """Decoded ``gemma4-assistant`` hyperparameters."""

    block_count: int
    n_embd: int
    n_embd_backbone: int
    n_ff: int
    n_head: int
    n_head_kv: tuple[int, ...]
    key_length: int
    value_length: int
    key_length_swa: int
    value_length_swa: int
    sliding_window: int
    is_swa: tuple[bool, ...]
    n_rot: int
    n_rot_swa: int
    nextn_predict_layers: int

    @property
    def pre_projection_in(self) -> int:
        """Width of ``concat(embedding, backbone_hidden)``."""

        return 2 * self.n_embd_backbone


def gemma4_assistant_config_from_metadata(metadata: Mapping[str, Any]) -> Gemma4AssistantConfig:
    """Decode the assistant hyperparameters, failing closed on anything missing."""

    block_count = _require_int(metadata, "block_count")
    n_embd = _require_int(metadata, "embedding_length")
    n_embd_backbone = _require_int(metadata, "embedding_length_out")
    n_ff = _require_int(metadata, "feed_forward_length")
    n_head = _require_int(metadata, "attention.head_count")
    key_length = _require_int(metadata, "attention.key_length")
    value_length = _require_int(metadata, "attention.value_length")
    key_length_swa = _require_int(metadata, "attention.key_length_swa")
    value_length_swa = _require_int(metadata, "attention.value_length_swa")
    sliding_window = _require_int(metadata, "attention.sliding_window")
    n_rot = _require_int(metadata, "rope.dimension_count")
    n_rot_swa = _require_int(metadata, "rope.dimension_count_swa")
    nextn_predict_layers = _require_int(metadata, "nextn_predict_layers")

    kv = metadata.get(_PREFIX + "attention.head_count_kv")
    if isinstance(kv, int):
        n_head_kv = (kv,) * block_count
    elif isinstance(kv, (list, tuple)) and all(isinstance(v, int) for v in kv):
        n_head_kv = tuple(int(v) for v in kv)
    else:
        raise MissingGGUFTensorError(
            f"{ARCHITECTURE} metadata key 'attention.head_count_kv' must be an int or int list"
        )
    if len(n_head_kv) != block_count:
        raise MissingGGUFTensorError(
            f"{ARCHITECTURE} head_count_kv has {len(n_head_kv)} entries for {block_count} blocks"
        )

    pattern = metadata.get(_PREFIX + "attention.sliding_window_pattern")
    if isinstance(pattern, (list, tuple)) and all(isinstance(v, int) for v in pattern):
        is_swa = tuple(bool(v) for v in pattern)
    else:
        raise MissingGGUFTensorError(
            f"{ARCHITECTURE} metadata key 'attention.sliding_window_pattern' must be an int list"
        )
    if len(is_swa) != block_count:
        raise MissingGGUFTensorError(
            f"{ARCHITECTURE} sliding_window_pattern has {len(is_swa)} entries "
            f"for {block_count} blocks"
        )

    return Gemma4AssistantConfig(
        block_count=block_count,
        n_embd=n_embd,
        n_embd_backbone=n_embd_backbone,
        n_ff=n_ff,
        n_head=n_head,
        n_head_kv=n_head_kv,
        key_length=key_length,
        value_length=value_length,
        key_length_swa=key_length_swa,
        value_length_swa=value_length_swa,
        sliding_window=sliding_window,
        is_swa=is_swa,
        n_rot=n_rot,
        n_rot_swa=n_rot_swa,
        nextn_predict_layers=nextn_predict_layers,
    )


def expected_gemma4_assistant_shapes(
    config: Gemma4AssistantConfig,
) -> dict[str, tuple[int, ...]]:
    """Return ``name -> ggml shape`` for every tensor the head must carry."""

    shapes: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (config.n_embd, 262144),
        "output_norm.weight": (config.n_embd,),
        "rope_freqs.weight": (config.n_rot_swa,),
        "nextn.post_projection.weight": (config.n_embd, config.n_embd_backbone),
        "nextn.pre_projection.weight": (config.pre_projection_in, config.n_embd),
    }
    for block_id in range(config.block_count):
        prefix = f"blk.{block_id}."
        swa = config.is_swa[block_id]
        q_width = config.n_head * (config.key_length_swa if swa else config.key_length)
        q_norm = config.key_length_swa if swa else config.key_length
        shapes.update(
            {
                prefix + "attn_norm.weight": (config.n_embd,),
                prefix + "layer_output_scale.weight": (1,),
                prefix + "ffn_down.weight": (config.n_ff, config.n_embd),
                prefix + "ffn_gate.weight": (config.n_embd, config.n_ff),
                prefix + "ffn_up.weight": (config.n_embd, config.n_ff),
                prefix + "post_attention_norm.weight": (config.n_embd,),
                prefix + "post_ffw_norm.weight": (config.n_embd,),
                prefix + "ffn_norm.weight": (config.n_embd,),
                prefix + "attn_output.weight": (q_width, config.n_embd),
                prefix + "attn_q_norm.weight": (q_norm,),
                prefix + "attn_q.weight": (config.n_embd, q_width),
            }
        )
    return shapes


def required_gemma4_assistant_tensor_names(
    config: Gemma4AssistantConfig,
) -> tuple[str, ...]:
    """Return every tensor name the head must carry, blocks first."""

    names: list[str] = []
    for block_id in range(config.block_count):
        prefix = f"blk.{block_id}."
        names.extend(prefix + slot for slot in _BLOCK_SLOTS)
    names.extend(_GLOBAL_SLOTS)
    return tuple(names)


@dataclass(frozen=True)
class Gemma4AssistantValidation:
    """Validation result for one assistant head."""

    config: Gemma4AssistantConfig
    present: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (self.missing or self.unexpected or self.shape_errors)

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        parts: list[str] = []
        for label, values in (
            ("missing tensors", self.missing),
            ("unexpected tensors", self.unexpected),
            ("shape errors", self.shape_errors),
        ):
            if values:
                preview = "; ".join(values[:6])
                more = "" if len(values) <= 6 else f" (+{len(values) - 6} more)"
                parts.append(f"{label}: {preview}{more}")
        raise MissingGGUFTensorError("; ".join(parts))


@dataclass(frozen=True)
class Gemma4AssistantTensorMap:
    """Canonical tensors for one assistant head."""

    config: Gemma4AssistantConfig
    tensors: Mapping[str, GGUFTensorInfo]
    validation: Gemma4AssistantValidation

    def tensor(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(f"assistant head has no tensor slot {slot!r}") from exc

    def block_tensor(self, block_id: int, suffix: str) -> GGUFTensorInfo:
        return self.tensor(f"blk.{int(block_id)}.{suffix}")


def validate_gemma4_assistant_tensor_map(info: GGUFModelInfo) -> Gemma4AssistantValidation:
    """Check one assistant head against the shape contract."""

    config = gemma4_assistant_config_from_metadata(info.metadata)
    expected = expected_gemma4_assistant_shapes(config)
    seen = {tensor.name: tensor for tensor in info.tensors}

    missing = tuple(sorted(set(expected) - set(seen)))
    unexpected = tuple(sorted(set(seen) - set(expected)))
    shape_errors: list[str] = []
    for name in sorted(set(expected) & set(seen)):
        want = expected[name]
        got = tuple(seen[name].ggml_shape)
        if got != want:
            shape_errors.append(f"{name}: expected {want}, got {got}")

    return Gemma4AssistantValidation(
        config=config,
        present=tuple(sorted(seen)),
        missing=missing,
        unexpected=unexpected,
        shape_errors=tuple(shape_errors),
    )


def build_gemma4_assistant_tensor_map(info: GGUFModelInfo) -> Gemma4AssistantTensorMap:
    """Validate then return the canonical tensor map, failing closed."""

    validation = validate_gemma4_assistant_tensor_map(info)
    validation.raise_for_errors()
    return Gemma4AssistantTensorMap(
        config=validation.config,
        tensors={tensor.name: tensor for tensor in info.tensors},
        validation=validation,
    )
