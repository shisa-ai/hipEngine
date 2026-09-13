"""Torch-free weight loading for the VibeVoice-ASR audio front-end.

Reads the original ``microsoft/VibeVoice-ASR`` safetensors checkpoint (BF16)
with NumPy only and produces the frozen weight bundles consumed by the CPU
reference (``hipengine/kernels/cpu_reference/vibevoice_asr.py``) and later by
the HIP runtime.

Stage-ordering is *inferred from tensor shapes*, not from config lists: the
original checkpoint stores ``downsampling_ratios`` as ``[8,5,5,4,2,2]`` while
execution order (confirmed by in-channel progression and kernel size) is
``[2,2,4,5,5,8]``. The loader asserts the inferred geometry (depths, widths,
stride product 3200, head wiring) before returning.
"""

from __future__ import annotations

import math
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


def _load_tensor(index: Any, name: str, path: Path) -> np.ndarray:
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
        "norm_weight": _load_tensor(index, f"{bp}.norm.weight", path),
        "conv_weight": _load_tensor(index, f"{bp}.mixer.conv.conv.conv.weight", path),
        "conv_bias": _load_tensor(index, f"{bp}.mixer.conv.conv.conv.bias", path),
        "gamma": _load_tensor(index, f"{bp}.gamma", path),
        "ffn_norm_weight": _load_tensor(index, f"{bp}.ffn_norm.weight", path),
        "ffn_gamma": _load_tensor(index, f"{bp}.ffn_gamma", path),
        "ffn_linear1_weight": _load_tensor(index, f"{bp}.ffn.linear1.weight", path),
        "ffn_linear1_bias": _load_tensor(index, f"{bp}.ffn.linear1.bias", path),
        "ffn_linear2_weight": _load_tensor(index, f"{bp}.ffn.linear2.weight", path),
        "ffn_linear2_bias": _load_tensor(index, f"{bp}.ffn.linear2.bias", path),
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
    index = load_weight_index(path)
    prefix = f"model.{tokenizer}_tokenizer.encoder"

    stem_w = _load_tensor(index, f"{prefix}.downsample_layers.0.0.conv.conv.weight", path)
    stem_b = _load_tensor(index, f"{prefix}.downsample_layers.0.0.conv.conv.bias", path)
    head_w = _load_tensor(index, f"{prefix}.head.conv.conv.weight", path)
    head_b = _load_tensor(index, f"{prefix}.head.conv.conv.bias", path)

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

    stage_conv_weights = tuple(_load_tensor(index, n, path) for n in stage_names)
    stage_conv_biases = tuple(
        _load_tensor(index, n.replace(".weight", ".bias"), path) for n in stage_names
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
    index = load_weight_index(path)
    prefix = f"model.{tokenizer}_connector"
    return VibevoiceConnectorWeights(
        fc1_weight=_load_tensor(index, f"{prefix}.fc1.weight", path),
        fc1_bias=_load_tensor(index, f"{prefix}.fc1.bias", path),
        norm_weight=_load_tensor(index, f"{prefix}.norm.weight", path),
        fc2_weight=_load_tensor(index, f"{prefix}.fc2.weight", path),
        fc2_bias=_load_tensor(index, f"{prefix}.fc2.bias", path),
    )
