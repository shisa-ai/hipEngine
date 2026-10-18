"""Torch-free loader for the VibeVoice-TTS (``microsoft/VibeVoice-1.5B``) decoder.

Reads the original checkpoint's safetensors directly (BF16 payloads expanded to
FP32 host arrays) and returns the acoustic waveform decoder bundle plus the two
checkpoint scaling factors.

The scaling factors are checkpoint **weights**, not config values: the state
dict carries ``model.speech_scaling_factor`` / ``model.speech_bias_factor`` as
bf16 scalars, absent from ``config.json``. The decode path applies
``latent / scale - bias`` (encode applies ``(latent + bias) * scale``); a model
built from config without this load would carry the construction-time ``NaN``
buffers and poison every speech embedding downstream.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from hipengine.loading.hf_cache import resolve_model_path
from hipengine.loading.safetensors import (
    load_weight_index,
    read_tensor_storage_bytes,
)
from hipengine.kernels.cpu_reference.vibevoice_tts import (
    VibevoiceDecoderSpec,
    VibevoiceDecoderWeights,
)
from hipengine.kernels.cpu_reference.vibevoice_tts_diffusion import (
    VibevoiceDiffusionSpec,
    VibevoiceDiffusionWeights,
)


def _bf16_bytes_to_f32(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return (
        np.frombuffer(payload, dtype=np.uint16).astype(np.uint32) << 16
    ).view(np.float32).reshape(shape).copy()


def _load_tensor(index, name: str, path: Path) -> np.ndarray:
    info = index.require((name,))[0]
    payload = read_tensor_storage_bytes(info)
    if info.dtype == "BF16":
        return _bf16_bytes_to_f32(payload, info.shape)
    if info.dtype == "F32":
        return np.frombuffer(payload, dtype=np.float32).reshape(info.shape).copy()
    raise ValueError(f"unsupported dtype {info.dtype!r} for {name!r}")


def _load_block(index, bp: str, path: Path) -> dict[str, np.ndarray]:
    """One ConvNeXt block's ten tensors under checkpoint prefix ``bp``."""
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


def load_vibevoice_tts_decoder(
    model_path: str | Path,
) -> tuple[VibevoiceDecoderSpec, VibevoiceDecoderWeights, float, float]:
    """Load the acoustic decoder spec, weights, and the two scaling factors.

    Returns ``(spec, weights, speech_scaling_factor, speech_bias_factor)``.
    """
    path = resolve_model_path(model_path)
    index = load_weight_index(path)
    tokenizer_config = index.config.get("acoustic_tokenizer_config")
    if tokenizer_config is None:
        raise ValueError(f"{path} has no acoustic_tokenizer_config; not a VibeVoice-TTS checkpoint")
    spec = VibevoiceDecoderSpec.from_config(tokenizer_config)
    prefix = "model.acoustic_tokenizer.decoder"
    widths = spec.stage_widths()

    stem_w = _load_tensor(index, f"{prefix}.upsample_layers.0.0.conv.conv.weight", path)
    stem_b = _load_tensor(index, f"{prefix}.upsample_layers.0.0.conv.conv.bias", path)
    if stem_w.shape != (widths[0], spec.dimension, spec.kernel_size):
        raise ValueError(f"stem weight {stem_w.shape} does not match spec {spec}")

    convtr_weights = []
    convtr_biases = []
    for layer in range(1, len(spec.ratios) + 1):
        w = _load_tensor(index, f"{prefix}.upsample_layers.{layer}.0.convtr.convtr.weight", path)
        b = _load_tensor(index, f"{prefix}.upsample_layers.{layer}.0.convtr.convtr.bias", path)
        c_in, c_out = widths[layer - 1], widths[layer]
        if w.shape != (c_in, c_out, 2 * spec.ratios[layer - 1]):
            raise ValueError(
                f"convtr {layer} weight {w.shape} does not match spec "
                f"{(c_in, c_out, 2 * spec.ratios[layer - 1])}"
            )
        convtr_weights.append(w)
        convtr_biases.append(b)

    head_w = _load_tensor(index, f"{prefix}.head.conv.conv.weight", path)
    head_b = _load_tensor(index, f"{prefix}.head.conv.conv.bias", path)
    if head_w.shape != (spec.channels, widths[-1], spec.last_kernel_size):
        raise ValueError(f"head weight {head_w.shape} does not match spec {spec}")

    blocks = []
    for stage, depth in enumerate(spec.depths):
        for j in range(depth):
            blocks.append(_load_block(index, f"{prefix}.stages.{stage}.{j}", path))
    expected_widths = [widths[s] for s, depth in enumerate(spec.depths) for _ in range(depth)]
    for block, width in zip(blocks, expected_widths):
        if block["norm_weight"].shape != (width,):
            raise ValueError(f"block norm width {block['norm_weight'].shape} != {(width,)}")

    scale_info = index.require(("model.speech_scaling_factor",))[0]
    bias_info = index.require(("model.speech_bias_factor",))[0]
    scaling_factor = float(_load_tensor(index, "model.speech_scaling_factor", path).reshape(-1)[0])
    bias_factor = float(_load_tensor(index, "model.speech_bias_factor", path).reshape(-1)[0])
    if not (math.isfinite(scaling_factor) and math.isfinite(bias_factor)):
        raise ValueError("speech scaling factors did not load as finite values")

    weights = VibevoiceDecoderWeights(
        stem_conv_weight=stem_w,
        stem_conv_bias=stem_b,
        convtr_weights=tuple(convtr_weights),
        convtr_biases=tuple(convtr_biases),
        head_conv_weight=head_w,
        head_conv_bias=head_b,
        blocks=tuple(blocks),
    )
    return spec, weights, scaling_factor, bias_factor


def load_vibevoice_tts_diffusion_head(
    model_path: str | Path,
) -> tuple[VibevoiceDiffusionSpec, VibevoiceDiffusionWeights, float, float]:
    """Load the diffusion head spec, weights, and the two scaling factors.

    Returns ``(spec, weights, speech_scaling_factor, speech_bias_factor)``.
    The head bundle is separate from the decoder so each runtime loads only
    what it drives.
    """
    path = resolve_model_path(model_path)
    index = load_weight_index(path)
    head_config = index.config.get("diffusion_head_config")
    if head_config is None:
        raise ValueError(f"{path} has no diffusion_head_config; not a VibeVoice-TTS checkpoint")
    spec = VibevoiceDiffusionSpec.from_config(head_config)
    prefix = "model.prediction_head"

    noisy = _load_tensor(index, f"{prefix}.noisy_images_proj.weight", path)
    cond = _load_tensor(index, f"{prefix}.cond_proj.weight", path)
    t_mlp_0 = _load_tensor(index, f"{prefix}.t_embedder.mlp.0.weight", path)
    t_mlp_2 = _load_tensor(index, f"{prefix}.t_embedder.mlp.2.weight", path)
    if noisy.shape != (spec.hidden_size, spec.latent_size):
        raise ValueError(f"noisy_images_proj {noisy.shape} does not match spec {spec}")

    layers = []
    for i in range(spec.head_layers):
        bp = f"{prefix}.layers.{i}"
        layers.append(
            {
                "norm_weight": _load_tensor(index, f"{bp}.norm.weight", path),
                "adaLN_weight": _load_tensor(index, f"{bp}.adaLN_modulation.1.weight", path),
                "gate_proj": _load_tensor(index, f"{bp}.ffn.gate_proj.weight", path),
                "up_proj": _load_tensor(index, f"{bp}.ffn.up_proj.weight", path),
                "down_proj": _load_tensor(index, f"{bp}.ffn.down_proj.weight", path),
            }
        )
        if layers[-1]["adaLN_weight"].shape != (3 * spec.hidden_size, spec.hidden_size):
            raise ValueError(f"layer {i} adaLN shape {layers[-1]['adaLN_weight'].shape} unexpected")

    final_adaLN = _load_tensor(index, f"{prefix}.final_layer.adaLN_modulation.1.weight", path)
    final_linear = _load_tensor(index, f"{prefix}.final_layer.linear.weight", path)
    if final_linear.shape != (spec.latent_size, spec.hidden_size):
        raise ValueError(f"final linear {final_linear.shape} does not match spec {spec}")

    scaling_factor = float(_load_tensor(index, "model.speech_scaling_factor", path).reshape(-1)[0])
    bias_factor = float(_load_tensor(index, "model.speech_bias_factor", path).reshape(-1)[0])
    if not (math.isfinite(scaling_factor) and math.isfinite(bias_factor)):
        raise ValueError("speech scaling factors did not load as finite values")

    weights = VibevoiceDiffusionWeights(
        noisy_images_proj=noisy,
        cond_proj=cond,
        t_mlp_0=t_mlp_0,
        t_mlp_2=t_mlp_2,
        layers=tuple(layers),
        final_adaLN=final_adaLN,
        final_linear=final_linear,
    )
    return spec, weights, scaling_factor, bias_factor
