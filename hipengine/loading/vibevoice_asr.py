"""Torch-free weight loading for the VibeVoice-ASR audio front-end.

Reads original or HF ``microsoft/VibeVoice-ASR`` safetensors checkpoints (BF16)
with NumPy only; each call stays within its selected artifact and produces the frozen weight bundles consumed by the CPU
reference (``hipengine/kernels/cpu_reference/vibevoice_asr.py``) and by
the HIP runtime.

Stage-ordering is *inferred from tensor shapes*, not from config lists: the
original checkpoint stores ``downsampling_ratios`` as ``[8,5,5,4,2,2]`` while
execution order (confirmed by in-channel progression and kernel size) is
``[2,2,4,5,5,8]``. The loader asserts the inferred geometry (depths, widths,
stride product 3200, head wiring) before returning.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.kernels.cpu_reference.vibevoice_asr import (
    VibevoiceConnectorWeights,
    VibevoiceTokenizerEncoderSpec,
    VibevoiceTokenizerEncoderWeights,
)
from hipengine.loading.hf_cache import resolve_model_path
from hipengine.loading.safetensors import (
    MissingTensorError,
    load_weight_index,
    read_tensor_storage_bytes,
)


def _bf16_bytes_to_f32(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """BF16 little-endian storage bytes -> FP32 array (each half word widened)."""
    if len(payload) != 2 * math.prod(shape):
        raise ValueError("bf16 payload size mismatch")
    raw = np.frombuffer(payload, dtype=np.uint16).astype(np.uint32)
    return (raw << np.uint32(16)).view(np.float32).reshape(shape).copy()


def hf_frontend_tensor_name(name: str) -> str:
    """Translate original frontend names to the independently pinned HF layout."""
    name = re.sub(r"model\.(acoustic|semantic)_tokenizer.encoder", r"\1_tokenizer_encoder", name)
    def stage(index):
        return "stem" if int(index) == 0 else f"conv_layers.{int(index) - 1}"
    name = re.sub(r"downsample_layers\.(\d+)\.0\.conv\.conv", lambda m: stage(m[1]) + ".conv.conv", name)
    name = re.sub(r"stages\.(\d+)\.(\d+)", lambda m: stage(m[1]) + f".stage.{m[2]}", name)
    name = name.replace("mixer.conv.conv.conv", "mixer.conv").replace("head.conv.conv", "head.conv")
    name = re.sub(r"model\.(acoustic|semantic)_connector\.fc([12])", r"multi_modal_projector.\1_linear_\2", name)
    return re.sub(r"model\.(acoustic|semantic)_connector\.norm", r"multi_modal_projector.\1_norm", name)


class _FrontendIndex:
    def __init__(self, index):
        self.index = index
        self.hf = index.config.get("model_type") == "vibevoice_asr"

    def read(self, name):
        return _load_tensor(self, name)

    def require(self, names):
        return self.index.require(tuple(hf_frontend_tensor_name(n) for n in names) if self.hf else names)


def _load_frontend_tensor(index: Any, name: str, path: Path) -> np.ndarray:
    return index.read(name)


def _load_tensor(index, name, path=None):
    info = index.require((name,))[0]
    payload = read_tensor_storage_bytes(info)
    if info.dtype == "BF16":
        return _bf16_bytes_to_f32(payload, info.shape)
    if info.dtype == "F32":
        return np.frombuffer(payload, dtype=np.float32).reshape(info.shape).copy()
    raise ValueError(f"unsupported dtype {info.dtype!r} for {name!r}")


def _stage_conv_stride(weight_shape: tuple[int, ...]) -> int:
    """A downsample entry conv has kernel ``2 * stride``; infer and validate."""
    out_channels, in_channels, kernel = weight_shape
    if out_channels != 2 * in_channels:
        raise ValueError(f"expected out=2*in for downsample conv, got {weight_shape}")
    if kernel % 2 != 0:
        raise ValueError(f"downsample conv kernel must be even, got {kernel}")
    return kernel // 2


def _load_block(index: Any, bp: str, path: Path) -> dict[str, np.ndarray]:
    """Load one ConvNeXt block's ten tensors under checkpoint prefix ``bp``."""
    return {
        "norm_weight": _load_frontend_tensor(index, f"{bp}.norm.weight", path),
        "conv_weight": _load_frontend_tensor(index, f"{bp}.mixer.conv.conv.conv.weight", path),
        "conv_bias": _load_frontend_tensor(index, f"{bp}.mixer.conv.conv.conv.bias", path),
        "gamma": _load_frontend_tensor(index, f"{bp}.gamma", path),
        "ffn_norm_weight": _load_frontend_tensor(index, f"{bp}.ffn_norm.weight", path),
        "ffn_gamma": _load_frontend_tensor(index, f"{bp}.ffn_gamma", path),
        "ffn_linear1_weight": _load_frontend_tensor(index, f"{bp}.ffn.linear1.weight", path),
        "ffn_linear1_bias": _load_frontend_tensor(index, f"{bp}.ffn.linear1.bias", path),
        "ffn_linear2_weight": _load_frontend_tensor(index, f"{bp}.ffn.linear2.weight", path),
        "ffn_linear2_bias": _load_frontend_tensor(index, f"{bp}.ffn.linear2.bias", path),
    }


def load_vibevoice_encoder(
    model_path: str | Path, tokenizer: str
) -> tuple[VibevoiceTokenizerEncoderSpec, VibevoiceTokenizerEncoderWeights]:
    """Load one tokenizer encoder (``tokenizer`` is ``acoustic``/``semantic``).

    Returns the shape-inferred spec (ratios in execution order) and the
    flattened weight bundle in execution order: stem conv, stem blocks,
    per-stage entry conv + blocks, head conv.
    """
    if tokenizer not in ("acoustic", "semantic"):
        raise ValueError("tokenizer must be 'acoustic' or 'semantic'")
    path = resolve_model_path(model_path)
    index = _FrontendIndex(load_weight_index(path))
    return _load_encoder_from_index(index, path, tokenizer)


def _load_encoder_from_index(index, path, tokenizer):
    prefix = f"model.{tokenizer}_tokenizer.encoder"

    stem_w = _load_frontend_tensor(index, f"{prefix}.downsample_layers.0.0.conv.conv.weight", path)
    stem_b = _load_frontend_tensor(index, f"{prefix}.downsample_layers.0.0.conv.conv.bias", path)
    head_w = _load_frontend_tensor(index, f"{prefix}.head.conv.conv.weight", path)
    head_b = _load_frontend_tensor(index, f"{prefix}.head.conv.conv.bias", path)

    # infer execution order of the 6 strided entry convs from in-channels
    stage_entries: list[tuple[int, str, tuple[int, ...]]] = []
    for layer in range(1, 7):
        name = f"{prefix}.downsample_layers.{layer}.0.conv.conv.weight"
        shape = tuple(index.require((name,))[0].shape)
        stage_entries.append((shape[1], name, shape))
    stage_entries.sort(key=lambda item: item[0])  # in_channels: 32, 64, ..., 1024
    ratios = tuple(_stage_conv_stride(shape) for _, _, shape in stage_entries)
    stage_names = [name for _, name, _ in stage_entries]

    num_filters = stem_w.shape[0]
    widths = [stem_w.shape[0]] + [shape[0] for _, _, shape in stage_entries]
    if widths != [num_filters * (2**i) for i in range(len(widths))]:
        raise ValueError(f"channel progression mismatch: {widths}")
    if math.prod(ratios) != 3200:
        raise ValueError(f"inferred stride product {math.prod(ratios)} != 3200")
    if head_w.shape[1] != widths[-1]:
        raise ValueError(f"head in-channels {head_w.shape[1]} != last stage width {widths[-1]}")

    stage_conv_weights = tuple(_load_frontend_tensor(index, n, path) for n in stage_names)
    stage_conv_biases = tuple(
        _load_frontend_tensor(index, n.replace(".weight", ".bias"), path) for n in stage_names
    )

    # block depths and widths, asserted against the expected layout
    depths: list[int] = []
    blocks: list[dict[str, np.ndarray]] = []
    expected_widths = [num_filters] + [num_filters * 2 ** (s + 1) for s in range(6)]
    for stage in range(7):
        width = expected_widths[stage]
        i = 0
        while True:
            probe = f"{prefix}.stages.{stage}.{i}.norm.weight"
            try:
                index.require((probe,))
            except MissingTensorError:
                break
            norm_shape = index.require((probe,))[0].shape
            if norm_shape != (width,):
                raise ValueError(f"stage {stage} block {i} norm width {norm_shape} != {(width,)}")
            blocks.append(_load_block(index, f"{prefix}.stages.{stage}.{i}", path))
            i += 1
        if i == 0:
            raise KeyError(f"no blocks found for {prefix}.stages.{stage}")
        depths.append(i)
    if tuple(depths) != (3, 3, 3, 3, 3, 3, 8):
        raise ValueError(f"unexpected block depths {tuple(depths)}")

    spec = VibevoiceTokenizerEncoderSpec(
        hidden_size=head_w.shape[0],
        depths=tuple(depths),
        ratios=ratios,
        num_filters=num_filters,
        kernel_size=stem_w.shape[2],
        ffn_expansion=4,
        rms_norm_eps=1e-5,
        vae_std=0.625 if tokenizer == "acoustic" else 0.0,
    )
    weights = VibevoiceTokenizerEncoderWeights(
        stem_conv_weight=stem_w,
        stem_conv_bias=stem_b,
        stage_conv_weights=stage_conv_weights,
        stage_conv_biases=stage_conv_biases,
        head_conv_weight=head_w,
        head_conv_bias=head_b,
        blocks=tuple(blocks),
    )
    return spec, weights


def load_vibevoice_connector(model_path: str | Path, tokenizer: str) -> VibevoiceConnectorWeights:
    """Load one ``{acoustic,semantic}_connector`` weight bundle."""
    if tokenizer not in ("acoustic", "semantic"):
        raise ValueError("tokenizer must be 'acoustic' or 'semantic'")
    path = resolve_model_path(model_path)
    index = _FrontendIndex(load_weight_index(path))
    return _load_connector_from_index(index, path, tokenizer)


def _load_connector_from_index(index, path, tokenizer):
    prefix = f"model.{tokenizer}_connector"
    return VibevoiceConnectorWeights(
        fc1_weight=_load_frontend_tensor(index, f"{prefix}.fc1.weight", path),
        fc1_bias=_load_frontend_tensor(index, f"{prefix}.fc1.bias", path),
        norm_weight=_load_frontend_tensor(index, f"{prefix}.norm.weight", path),
        fc2_weight=_load_frontend_tensor(index, f"{prefix}.fc2.weight", path),
        fc2_bias=_load_frontend_tensor(index, f"{prefix}.fc2.bias", path),
    )


@dataclass(frozen=True)
class VibevoiceQwen2Spec:
    """Static Qwen2 text-backbone geometry."""

    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VibevoiceQwen2Spec":
        text = config.get("text_config", config)
        hidden = int(text["hidden_size"])
        heads = int(text["num_attention_heads"])
        return cls(
            hidden_size=hidden,
            num_layers=int(text["num_hidden_layers"]),
            num_attention_heads=heads,
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text.get("head_dim", hidden // heads)),
            intermediate_size=int(text["intermediate_size"]),
            vocab_size=int(text["vocab_size"]),
            rope_theta=float(text.get("rope_theta", 1000000.0)),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-5)),
        )


@dataclass(frozen=True)
class VibevoiceQwen2LayerWeights:
    input_layernorm: np.ndarray
    q_weight: np.ndarray
    q_bias: np.ndarray
    k_weight: np.ndarray
    k_bias: np.ndarray
    v_weight: np.ndarray
    v_bias: np.ndarray
    o_weight: np.ndarray
    post_attention_layernorm: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray


@dataclass(frozen=True)
class VibevoiceQwen2Weights:
    """Qwen2 backbone + untied lm_head, numpy fp32."""

    spec: VibevoiceQwen2Spec
    embed_tokens: np.ndarray
    final_norm: np.ndarray
    lm_head: np.ndarray
    layers: tuple[VibevoiceQwen2LayerWeights, ...]


def load_vibevoice_qwen2(model_path: str | Path) -> VibevoiceQwen2Weights:
    """Load the Qwen2 text backbone from the HF ``VibeVoice-ASR-HF`` artifact."""
    path = resolve_model_path(model_path)
    index = load_weight_index(path)
    with open(path / "config.json") as fh:
        spec = VibevoiceQwen2Spec.from_config(json.load(fh))

    def t(name: str) -> np.ndarray:
        return _load_tensor(index, name, path)

    layers = []
    for i in range(spec.num_layers):
        p = f"language_model.model.layers.{i}"
        layers.append(
            VibevoiceQwen2LayerWeights(
                input_layernorm=t(f"{p}.input_layernorm.weight"),
                q_weight=t(f"{p}.self_attn.q_proj.weight"),
                q_bias=t(f"{p}.self_attn.q_proj.bias"),
                k_weight=t(f"{p}.self_attn.k_proj.weight"),
                k_bias=t(f"{p}.self_attn.k_proj.bias"),
                v_weight=t(f"{p}.self_attn.v_proj.weight"),
                v_bias=t(f"{p}.self_attn.v_proj.bias"),
                o_weight=t(f"{p}.self_attn.o_proj.weight"),
                post_attention_layernorm=t(f"{p}.post_attention_layernorm.weight"),
                gate_proj=t(f"{p}.mlp.gate_proj.weight"),
                up_proj=t(f"{p}.mlp.up_proj.weight"),
                down_proj=t(f"{p}.mlp.down_proj.weight"),
            )
        )
    return VibevoiceQwen2Weights(
        spec=spec,
        embed_tokens=t("language_model.model.embed_tokens.weight"),
        final_norm=t("language_model.model.norm.weight"),
        lm_head=t("language_model.lm_head.weight"),
        layers=tuple(layers),
    )
