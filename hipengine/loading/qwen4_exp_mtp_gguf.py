"""Strict GGUF contract for the Qwen3.8-Flash-Next MTP-only sidecar.

The target GGUF intentionally omits ``mtp.*``.  EngramHalo's converter exports
one self-contained draft artifact with shared embedding/head tensors, a final
hyper-connection mixer, the three NextN input tensors, and one full Qwen4Exp
attention+MoE block at ``blk.48``.  This module validates that artifact without
materializing tensor payloads.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Mapping, Sequence

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo
from hipengine.loading.qwen4_exp_gguf import Qwen4ExpGGUFTensorRef

MTP_BLOCK_ID = 48


class Qwen4ExpMTPGGUFError(ValueError):
    """Raised when an MTP-only Qwen4Exp GGUF drifts from the pinned contract."""


@dataclass(frozen=True)
class Qwen4ExpMTPGGUFConfig:
    architecture: str
    block_count: int
    nextn_predict_layers: int
    context_length: int
    hidden_size: int
    vocab_size: int
    residual_branch_count: int
    residual_low_rank: int
    attention_head_count: int
    attention_kv_head_count: int
    attention_key_length: int
    indexer_head_count: int
    indexer_key_length: int
    expert_count: int
    expert_used_count: int
    expert_feed_forward_length: int
    shared_expert_feed_forward_length: int
    rms_epsilon: float
    block_id: int = MTP_BLOCK_ID

    @property
    def residual_width(self) -> int:
        return self.hidden_size * self.residual_branch_count


@dataclass(frozen=True)
class Qwen4ExpMTPGGUFValidation:
    config: Qwen4ExpMTPGGUFConfig
    metadata_errors: tuple[str, ...]
    missing_tensor_names: tuple[str, ...]
    unexpected_tensor_names: tuple[str, ...]
    duplicate_tensor_names: tuple[str, ...]
    shape_errors: tuple[str, ...]
    qtype_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.metadata_errors
            or self.missing_tensor_names
            or self.unexpected_tensor_names
            or self.duplicate_tensor_names
            or self.shape_errors
            or self.qtype_errors
        )


@dataclass(frozen=True)
class Qwen4ExpMTPGGUFMap:
    config: Qwen4ExpMTPGGUFConfig
    weights: Mapping[str, Qwen4ExpGGUFTensorRef]
    tensor_refs: tuple[Qwen4ExpGGUFTensorRef, ...]
    validation: Qwen4ExpMTPGGUFValidation
    part_paths: tuple[Path, ...]

    def weight(self, slot_path: str) -> Qwen4ExpGGUFTensorRef:
        try:
            return self.weights[str(slot_path)]
        except KeyError as exc:
            raise KeyError(f"unknown Qwen4Exp MTP slot {slot_path!r}") from exc


_ROOT_SLOTS = {
    "root.token_embedding": "token_embd.weight",
    "root.lm_head": "output.weight",
    "root.head_hc_norm": "output_hc_norm.weight",
    "root.head_hc_down": "output_hc_down.weight",
    "root.head_hc_up": "output_hc_up.weight",
}
_INPUT_SLOTS = {
    "nextn.eh_proj": f"blk.{MTP_BLOCK_ID}.nextn.eh_proj.weight",
    "nextn.enorm": f"blk.{MTP_BLOCK_ID}.nextn.enorm.weight",
    "nextn.hnorm": f"blk.{MTP_BLOCK_ID}.nextn.hnorm.weight",
}
_LAYER_SUFFIXES = {
    "hc_attn_norm": "hc_attn_norm.weight",
    "hc_attn_down": "hc_attn_down.weight",
    "hc_attn_up": "hc_attn_up.weight",
    "hc_attn_inject": "hc_attn_inject.weight",
    "hc_ffn_norm": "hc_ffn_norm.weight",
    "hc_ffn_down": "hc_ffn_down.weight",
    "hc_ffn_up": "hc_ffn_up.weight",
    "hc_ffn_inject": "hc_ffn_inject.weight",
    "attn_q": "attn_q.weight",
    "attn_q_norm": "attn_q_norm.weight",
    "attn_k": "attn_k.weight",
    "attn_k_norm": "attn_k_norm.weight",
    "attn_v": "attn_v.weight",
    "attn_output": "attn_output.weight",
    "index_q": "indexer.q_proj.weight",
    "index_k": "indexer.k_proj.weight",
    "index_q_norm": "indexer.q_norm.weight",
    "index_k_norm": "indexer.k_norm.weight",
    "router": "ffn_gate_inp.weight",
    "shared_expert_gate": "ffn_gate_inp_shexp.weight",
    "expert_gate": "ffn_gate_exps.weight",
    "expert_up": "ffn_up_exps.weight",
    "expert_down": "ffn_down_exps.weight",
    "shared_gate": "ffn_gate_shexp.weight",
    "shared_up": "ffn_up_shexp.weight",
    "shared_down": "ffn_down_shexp.weight",
}
_LAYER_SLOTS = {
    f"layers.0.{slot}": f"blk.{MTP_BLOCK_ID}.{suffix}"
    for slot, suffix in _LAYER_SUFFIXES.items()
}
_SLOT_NAMES = {**_ROOT_SLOTS, **_INPUT_SLOTS, **_LAYER_SLOTS}


def required_qwen4_exp_mtp_tensor_names() -> tuple[str, ...]:
    return tuple(_SLOT_NAMES.values())


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    hidden, residual, low_rank = 2_560, 10_240, 320
    block = f"blk.{MTP_BLOCK_ID}."
    return {
        "token_embd.weight": (248_320, hidden),
        "output.weight": (248_320, hidden),
        "output_hc_norm.weight": (residual,),
        "output_hc_down.weight": (low_rank, residual),
        "output_hc_up.weight": (residual, low_rank),
        block + "nextn.eh_proj.weight": (hidden, 2 * hidden),
        block + "nextn.enorm.weight": (hidden,),
        block + "nextn.hnorm.weight": (residual,),
        block + "hc_attn_norm.weight": (residual,),
        block + "hc_attn_down.weight": (low_rank, residual),
        block + "hc_attn_up.weight": (residual, low_rank),
        block + "hc_attn_inject.weight": (4, residual),
        block + "hc_ffn_norm.weight": (residual,),
        block + "hc_ffn_down.weight": (low_rank, residual),
        block + "hc_ffn_up.weight": (residual, low_rank),
        block + "hc_ffn_inject.weight": (4, residual),
        block + "attn_q.weight": (12_288, hidden),
        block + "attn_q_norm.weight": (256,),
        block + "attn_k.weight": (512, hidden),
        block + "attn_k_norm.weight": (256,),
        block + "attn_v.weight": (512, hidden),
        block + "attn_output.weight": (hidden, 6_144),
        block + "indexer.q_proj.weight": (512, hidden),
        block + "indexer.k_proj.weight": (128, hidden),
        block + "indexer.q_norm.weight": (128,),
        block + "indexer.k_norm.weight": (128,),
        block + "ffn_gate_inp.weight": (512, hidden),
        block + "ffn_gate_inp_shexp.weight": (hidden,),
        block + "ffn_gate_exps.weight": (512, 640, hidden),
        block + "ffn_up_exps.weight": (512, 640, hidden),
        block + "ffn_down_exps.weight": (512, hidden, 640),
        block + "ffn_gate_shexp.weight": (640, hidden),
        block + "ffn_up_shexp.weight": (640, hidden),
        block + "ffn_down_shexp.weight": (hidden, 640),
    }


def _expected_qtypes() -> dict[str, str]:
    names = set(required_qwen4_exp_mtp_tensor_names())
    result = {name: "Q8_0" for name in names}
    for name in (
        "output_hc_norm.weight",
        f"blk.{MTP_BLOCK_ID}.nextn.enorm.weight",
        f"blk.{MTP_BLOCK_ID}.nextn.hnorm.weight",
        f"blk.{MTP_BLOCK_ID}.hc_attn_norm.weight",
        f"blk.{MTP_BLOCK_ID}.hc_ffn_norm.weight",
        f"blk.{MTP_BLOCK_ID}.attn_q_norm.weight",
        f"blk.{MTP_BLOCK_ID}.attn_k_norm.weight",
        f"blk.{MTP_BLOCK_ID}.indexer.q_norm.weight",
        f"blk.{MTP_BLOCK_ID}.indexer.k_norm.weight",
        f"blk.{MTP_BLOCK_ID}.ffn_gate_inp.weight",
        f"blk.{MTP_BLOCK_ID}.ffn_gate_inp_shexp.weight",
    ):
        result[name] = "F32"
    result[f"blk.{MTP_BLOCK_ID}.indexer.q_proj.weight"] = "BF16"
    result[f"blk.{MTP_BLOCK_ID}.indexer.k_proj.weight"] = "BF16"
    return result


def _parse_config(info: GGUFModelInfo) -> tuple[Qwen4ExpMTPGGUFConfig, list[str]]:
    metadata = info.metadata

    def integer(key: str, default: int = 0) -> int:
        value = metadata.get(key, default)
        return int(value)

    def floating(key: str, default: float = 0.0) -> float:
        return float(metadata.get(key, default))

    config = Qwen4ExpMTPGGUFConfig(
        architecture=str(metadata.get("general.architecture", "")),
        block_count=integer("qwen4exp.block_count"),
        nextn_predict_layers=integer("qwen4exp.nextn_predict_layers"),
        context_length=integer("qwen4exp.context_length"),
        hidden_size=integer("qwen4exp.embedding_length"),
        vocab_size=248_320,
        residual_branch_count=integer("qwen4exp.hyper_connection.count"),
        residual_low_rank=integer("qwen4exp.hyper_connection.low_rank"),
        attention_head_count=integer("qwen4exp.attention.head_count"),
        attention_kv_head_count=integer("qwen4exp.attention.head_count_kv"),
        attention_key_length=integer("qwen4exp.attention.key_length"),
        indexer_head_count=integer("qwen4exp.attention.indexer.head_count"),
        indexer_key_length=integer("qwen4exp.attention.indexer.key_length"),
        expert_count=integer("qwen4exp.expert_count"),
        expert_used_count=integer("qwen4exp.expert_used_count"),
        expert_feed_forward_length=integer("qwen4exp.expert_feed_forward_length"),
        shared_expert_feed_forward_length=integer(
            "qwen4exp.expert_shared_feed_forward_length"
        ),
        rms_epsilon=floating("qwen4exp.attention.layer_norm_rms_epsilon"),
    )
    expected = {
        "architecture": "qwen4exp",
        "block_count": 49,
        "nextn_predict_layers": 1,
        "context_length": 262_144,
        "hidden_size": 2_560,
        "residual_branch_count": 4,
        "residual_low_rank": 320,
        "attention_head_count": 24,
        "attention_kv_head_count": 2,
        "attention_key_length": 256,
        "indexer_head_count": 4,
        "indexer_key_length": 128,
        "expert_count": 512,
        "expert_used_count": 10,
        "expert_feed_forward_length": 640,
        "shared_expert_feed_forward_length": 640,
    }
    errors = [
        f"{field}={getattr(config, field)!r}, expected {value!r}"
        for field, value in expected.items()
        if getattr(config, field) != value
    ]
    if not isclose(config.rms_epsilon, 1.0e-6, rel_tol=0.0, abs_tol=1.0e-12):
        errors.append(f"rms_epsilon={config.rms_epsilon!r}, expected 1e-6")
    ratios = tuple(int(value) for value in metadata.get("qwen4exp.attention.compress_ratios", ()))
    if len(ratios) != 49 or ratios[-1:] != (0,):
        errors.append("attention.compress_ratios must contain 49 rows with dense MTP tail 0")
    return config, errors


def validate_qwen4_exp_mtp_gguf(
    infos: Sequence[GGUFModelInfo],
) -> Qwen4ExpMTPGGUFValidation:
    parts = tuple(infos)
    if not parts:
        raise Qwen4ExpMTPGGUFError("at least one Qwen4Exp MTP GGUF part is required")
    metadata_info = next(
        (info for info in parts if info.metadata.get("general.architecture") == "qwen4exp"),
        parts[0],
    )
    config, metadata_errors = _parse_config(metadata_info)
    counts = Counter(tensor.name for info in parts for tensor in info.tensors)
    actual = set(counts)
    required = set(required_qwen4_exp_mtp_tensor_names())
    unique = {
        tensor.name: tensor
        for info in parts
        for tensor in info.tensors
        if counts[tensor.name] == 1
    }
    shapes = _expected_shapes()
    qtypes = _expected_qtypes()
    shape_errors = tuple(
        f"{name} shape={unique[name].shape}, expected {shape}"
        for name, shape in shapes.items()
        if name in unique and tuple(unique[name].shape) != shape
    )
    qtype_errors = tuple(
        f"{name} qtype={unique[name].ggml_type_name}, expected {qtype}"
        for name, qtype in qtypes.items()
        if name in unique and unique[name].ggml_type_name != qtype
    )
    return Qwen4ExpMTPGGUFValidation(
        config=config,
        metadata_errors=tuple(metadata_errors),
        missing_tensor_names=tuple(sorted(required - actual)),
        unexpected_tensor_names=tuple(sorted(actual - required)),
        duplicate_tensor_names=tuple(sorted(name for name, count in counts.items() if count > 1)),
        shape_errors=shape_errors,
        qtype_errors=qtype_errors,
    )


def build_qwen4_exp_mtp_gguf_map(
    infos: Sequence[GGUFModelInfo],
    *,
    strict: bool = True,
) -> Qwen4ExpMTPGGUFMap:
    parts = tuple(infos)
    validation = validate_qwen4_exp_mtp_gguf(parts)
    if strict and not validation.passed:
        problems = (
            *validation.metadata_errors,
            *(f"missing {name}" for name in validation.missing_tensor_names[:4]),
            *(f"unexpected {name}" for name in validation.unexpected_tensor_names[:4]),
            *validation.shape_errors[:4],
            *validation.qtype_errors[:4],
        )
        raise Qwen4ExpMTPGGUFError(
            "invalid Qwen4Exp MTP GGUF: " + "; ".join(problems)
        )
    owners: dict[str, Qwen4ExpGGUFTensorRef] = {}
    for part_index, info in enumerate(parts):
        for tensor in info.tensors:
            owners.setdefault(tensor.name, Qwen4ExpGGUFTensorRef(part_index, info.path, tensor))
    weights = {
        slot: owners[name]
        for slot, name in _SLOT_NAMES.items()
        if name in owners
    }
    refs = tuple(weights[slot] for slot in _SLOT_NAMES if slot in weights)
    return Qwen4ExpMTPGGUFMap(
        config=validation.config,
        weights=weights,
        tensor_refs=refs,
        validation=validation,
        part_paths=tuple(info.path for info in parts),
    )


__all__ = [
    "MTP_BLOCK_ID",
    "Qwen4ExpMTPGGUFConfig",
    "Qwen4ExpMTPGGUFError",
    "Qwen4ExpMTPGGUFMap",
    "Qwen4ExpMTPGGUFValidation",
    "build_qwen4_exp_mtp_gguf_map",
    "required_qwen4_exp_mtp_tensor_names",
    "validate_qwen4_exp_mtp_gguf",
]
