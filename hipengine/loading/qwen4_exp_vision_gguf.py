"""Strict Qwen3-VL-compatible mmproj contract for Qwen3.8-Flash-Next."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Mapping, Sequence

from hipengine.loading.gguf import GGUFModelInfo
from hipengine.loading.qwen4_exp_gguf import Qwen4ExpGGUFTensorRef


class Qwen4ExpVisionGGUFError(ValueError):
    pass


@dataclass(frozen=True)
class Qwen4ExpVisionConfig:
    architecture: str
    projector_type: str
    block_count: int
    hidden_size: int
    intermediate_size: int
    output_size: int
    heads: int
    patch_size: int
    spatial_merge_size: int
    image_size: int
    norm_epsilon: float

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.heads


@dataclass(frozen=True)
class Qwen4ExpVisionValidation:
    config: Qwen4ExpVisionConfig
    metadata_errors: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    duplicates: tuple[str, ...]
    shape_errors: tuple[str, ...]
    qtype_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.metadata_errors
            or self.missing
            or self.unexpected
            or self.duplicates
            or self.shape_errors
            or self.qtype_errors
        )


@dataclass(frozen=True)
class Qwen4ExpVisionMap:
    config: Qwen4ExpVisionConfig
    weights: Mapping[str, Qwen4ExpGGUFTensorRef]
    tensor_refs: tuple[Qwen4ExpGGUFTensorRef, ...]
    validation: Qwen4ExpVisionValidation
    part_paths: tuple[Path, ...]

    def weight(self, slot: str) -> Qwen4ExpGGUFTensorRef:
        return self.weights[str(slot)]


_ROOT = {
    "patch.weight0": "v.patch_embd.weight",
    "patch.weight1": "v.patch_embd.weight.1",
    "patch.bias": "v.patch_embd.bias",
    "position": "v.position_embd.weight",
    "post_norm.weight": "v.post_ln.weight",
    "post_norm.bias": "v.post_ln.bias",
    "merge.fc1.weight": "mm.0.weight",
    "merge.fc1.bias": "mm.0.bias",
    "merge.fc2.weight": "mm.2.weight",
    "merge.fc2.bias": "mm.2.bias",
}
_LAYER_SUFFIXES = {
    "attn_out.bias": "attn_out.bias",
    "attn_out.weight": "attn_out.weight",
    "attn_qkv.bias": "attn_qkv.bias",
    "attn_qkv.weight": "attn_qkv.weight",
    "ffn_up.bias": "ffn_up.bias",
    "ffn_up.weight": "ffn_up.weight",
    "ffn_down.bias": "ffn_down.bias",
    "ffn_down.weight": "ffn_down.weight",
    "ln1.bias": "ln1.bias",
    "ln1.weight": "ln1.weight",
    "ln2.bias": "ln2.bias",
    "ln2.weight": "ln2.weight",
}
_SLOTS = {
    **_ROOT,
    **{
        f"layers.{layer}.{slot}": f"v.blk.{layer}.{suffix}"
        for layer in range(27)
        for slot, suffix in _LAYER_SUFFIXES.items()
    },
}


def required_qwen4_exp_vision_tensor_names() -> tuple[str, ...]:
    return tuple(_SLOTS.values())


def _shapes() -> dict[str, tuple[int, ...]]:
    hidden, inter = 1_152, 4_304
    shapes = {
        "v.patch_embd.weight": (hidden, 3, 16, 16),
        "v.patch_embd.weight.1": (hidden, 3, 16, 16),
        "v.patch_embd.bias": (hidden,),
        "v.position_embd.weight": (2_304, hidden),
        "v.post_ln.weight": (hidden,),
        "v.post_ln.bias": (hidden,),
        "mm.0.weight": (4 * hidden, 4 * hidden),
        "mm.0.bias": (4 * hidden,),
        "mm.2.weight": (2_560, 4 * hidden),
        "mm.2.bias": (2_560,),
    }
    for layer in range(27):
        prefix = f"v.blk.{layer}."
        shapes.update(
            {
                prefix + "attn_out.bias": (hidden,),
                prefix + "attn_out.weight": (hidden, hidden),
                prefix + "attn_qkv.bias": (3 * hidden,),
                prefix + "attn_qkv.weight": (3 * hidden, hidden),
                prefix + "ffn_up.bias": (inter,),
                prefix + "ffn_up.weight": (inter, hidden),
                prefix + "ffn_down.bias": (hidden,),
                prefix + "ffn_down.weight": (hidden, inter),
                prefix + "ln1.bias": (hidden,),
                prefix + "ln1.weight": (hidden,),
                prefix + "ln2.bias": (hidden,),
                prefix + "ln2.weight": (hidden,),
            }
        )
    return shapes


def _qtypes() -> dict[str, str]:
    result = {name: "F32" for name in required_qwen4_exp_vision_tensor_names()}
    for name in result:
        if name.endswith(".weight") and not (
            "ln" in name
            or name.startswith("v.patch_embd")
            or name.startswith("v.position_embd")
            or name.startswith("v.post_ln")
        ):
            result[name] = "BF16"
    return result


def _config(info: GGUFModelInfo) -> tuple[Qwen4ExpVisionConfig, tuple[str, ...]]:
    m = info.metadata
    cfg = Qwen4ExpVisionConfig(
        architecture=str(m.get("general.architecture", "")),
        projector_type=str(m.get("clip.projector_type", "")),
        block_count=int(m.get("clip.vision.block_count", 0)),
        hidden_size=int(m.get("clip.vision.embedding_length", 0)),
        intermediate_size=int(m.get("clip.vision.feed_forward_length", 0)),
        output_size=int(m.get("clip.vision.projection_dim", 0)),
        heads=int(m.get("clip.vision.attention.head_count", 0)),
        patch_size=int(m.get("clip.vision.patch_size", 0)),
        spatial_merge_size=int(m.get("clip.vision.spatial_merge_size", 0)),
        image_size=int(m.get("clip.vision.image_size", 0)),
        norm_epsilon=float(m.get("clip.vision.attention.layer_norm_epsilon", 0.0)),
    )
    expected = {
        "architecture": "clip",
        "projector_type": "qwen3vl_merger",
        "block_count": 27,
        "hidden_size": 1_152,
        "intermediate_size": 4_304,
        "output_size": 2_560,
        "heads": 16,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "image_size": 768,
    }
    errors = tuple(
        f"{field}={getattr(cfg, field)!r}, expected {value!r}"
        for field, value in expected.items()
        if getattr(cfg, field) != value
    )
    if not isclose(cfg.norm_epsilon, 1e-6, rel_tol=0, abs_tol=1e-12):
        errors += (f"norm_epsilon={cfg.norm_epsilon!r}, expected 1e-6",)
    if cfg.head_dim != 72:
        errors += (f"head_dim={cfg.head_dim}, expected 72",)
    return cfg, errors


def validate_qwen4_exp_vision_gguf(
    infos: Sequence[GGUFModelInfo],
) -> Qwen4ExpVisionValidation:
    parts = tuple(infos)
    if not parts:
        raise Qwen4ExpVisionGGUFError("at least one vision GGUF is required")
    cfg, metadata_errors = _config(parts[0])
    counts = Counter(t.name for info in parts for t in info.tensors)
    required = set(required_qwen4_exp_vision_tensor_names())
    actual = set(counts)
    unique = {
        t.name: t for info in parts for t in info.tensors if counts[t.name] == 1
    }
    shapes = _shapes()
    qtypes = _qtypes()
    return Qwen4ExpVisionValidation(
        config=cfg,
        metadata_errors=metadata_errors,
        missing=tuple(sorted(required - actual)),
        unexpected=tuple(sorted(actual - required)),
        duplicates=tuple(sorted(name for name, count in counts.items() if count > 1)),
        shape_errors=tuple(
            f"{name} shape={unique[name].shape}, expected {shape}"
            for name, shape in shapes.items()
            if name in unique and tuple(unique[name].shape) != shape
        ),
        qtype_errors=tuple(
            f"{name} qtype={unique[name].ggml_type_name}, expected {qtype}"
            for name, qtype in qtypes.items()
            if name in unique and unique[name].ggml_type_name != qtype
        ),
    )


def build_qwen4_exp_vision_gguf_map(
    infos: Sequence[GGUFModelInfo], *, strict: bool = True
) -> Qwen4ExpVisionMap:
    parts = tuple(infos)
    validation = validate_qwen4_exp_vision_gguf(parts)
    if strict and not validation.passed:
        raise Qwen4ExpVisionGGUFError(
            "invalid Qwen4Exp vision GGUF: "
            + "; ".join(
                (
                    *validation.metadata_errors,
                    *(f"missing {x}" for x in validation.missing[:4]),
                    *(f"unexpected {x}" for x in validation.unexpected[:4]),
                    *validation.shape_errors[:4],
                    *validation.qtype_errors[:4],
                )
            )
        )
    owners = {}
    for part_index, info in enumerate(parts):
        for tensor in info.tensors:
            owners.setdefault(
                tensor.name,
                Qwen4ExpGGUFTensorRef(part_index, info.path, tensor),
            )
    weights = {slot: owners[name] for slot, name in _SLOTS.items() if name in owners}
    return Qwen4ExpVisionMap(
        config=validation.config,
        weights=weights,
        tensor_refs=tuple(weights[slot] for slot in _SLOTS if slot in weights),
        validation=validation,
        part_paths=tuple(info.path for info in parts),
    )


__all__ = [
    "Qwen4ExpVisionConfig",
    "Qwen4ExpVisionGGUFError",
    "Qwen4ExpVisionMap",
    "Qwen4ExpVisionValidation",
    "build_qwen4_exp_vision_gguf_map",
    "required_qwen4_exp_vision_tensor_names",
    "validate_qwen4_exp_vision_gguf",
]
