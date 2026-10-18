"""Torch-free loader for the VibeVoice-TTS (``microsoft/VibeVoice-1.5B``) session.

Bundles every surface the generation session drives in execution order:

- the Qwen2 text backbone (``model.language_model.*``) with the **tied** lm
  head (``tie_word_embeddings=True``: logits share the embedding table);
- the two ``SpeechConnector`` bundles (acoustic for the voice prompt and the
  per-frame feedback; semantic for the generated-audio feedback);
- the acoustic and semantic tokenizer encoder bundles (the voice prompt is
  acoustically encoded; generated chunks are semantically encoded);
- the acoustic decoder and diffusion-head bundles (loaded by their own
  loaders and referenced here);
- the two checkpoint scaling factors (weights, not config; NaN refusal
  inherited from the decoder loader).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hipengine.loading.hf_cache import resolve_model_path
from hipengine.loading.safetensors import load_weight_index
from hipengine.loading.vibevoice_asr import (
    VibevoiceConnectorWeights,
    VibevoiceQwen2LayerWeights,
    VibevoiceQwen2Spec,
    VibevoiceQwen2Weights,
    _bf16_bytes_to_f32,
    load_vibevoice_connector,
    load_vibevoice_encoder,
)
from hipengine.kernels.cpu_reference.vibevoice_asr import (
    VibevoiceTokenizerEncoderSpec,
    VibevoiceTokenizerEncoderWeights,
)
from hipengine.kernels.cpu_reference.vibevoice_tts import (
    VibevoiceDecoderSpec,
    VibevoiceDecoderWeights,
)
from hipengine.kernels.cpu_reference.vibevoice_tts_diffusion import (
    VibevoiceDiffusionSpec,
    VibevoiceDiffusionWeights,
)
from hipengine.loading.vibevoice_tts import (
    load_vibevoice_tts_decoder,
    load_vibevoice_tts_diffusion_head,
)

# Qwen2.5 special tokens the fork reuses as speech control tokens.
SPEECH_START_ID = 151652
SPEECH_END_ID = 151653
SPEECH_DIFFUSION_ID = 151654
EOS_TOKEN_ID = 151643  # <|endoftext|>: the fork's tokenizer.eos_token_id


def _load_tensor(index, name: str, path: Path) -> np.ndarray:
    info = index.require((name,))[0]
    from hipengine.loading.safetensors import read_tensor_storage_bytes

    payload = read_tensor_storage_bytes(info)
    if info.dtype == "BF16":
        return _bf16_bytes_to_f32(payload, info.shape)
    if info.dtype == "F32":
        return np.frombuffer(payload, dtype=np.float32).reshape(info.shape).copy()
    raise ValueError(f"unsupported dtype {info.dtype!r} for {name!r}")


@dataclass(frozen=True)
class VibevoiceTtsSessionWeights:
    """Everything the TTS generation session drives, in execution order."""

    lm: VibevoiceQwen2Weights
    acoustic_connector: VibevoiceConnectorWeights
    semantic_connector: VibevoiceConnectorWeights
    acoustic_encoder: VibevoiceTokenizerEncoderWeights
    semantic_encoder: VibevoiceTokenizerEncoderWeights
    acoustic_encoder_spec: VibevoiceTokenizerEncoderSpec
    semantic_encoder_spec: VibevoiceTokenizerEncoderSpec
    decoder_spec: VibevoiceDecoderSpec
    decoder_weights: VibevoiceDecoderWeights
    diffusion_spec: VibevoiceDiffusionSpec
    diffusion_weights: VibevoiceDiffusionWeights
    speech_scaling_factor: float
    speech_bias_factor: float


def load_vibevoice_tts_session(model_path: str | Path) -> VibevoiceTtsSessionWeights:
    """Load the full TTS session bundle from the pinned checkpoint."""
    path = resolve_model_path(model_path)
    index = load_weight_index(path)
    with open(path / "config.json") as fh:
        import json

        config = json.load(fh)
    lm_config = config["decoder_config"]
    spec = VibevoiceQwen2Spec(
        hidden_size=int(lm_config["hidden_size"]),
        num_layers=int(lm_config["num_hidden_layers"]),
        num_attention_heads=int(lm_config["num_attention_heads"]),
        num_key_value_heads=int(lm_config["num_key_value_heads"]),
        head_dim=int(lm_config["hidden_size"]) // int(lm_config["num_attention_heads"]),
        intermediate_size=int(lm_config["intermediate_size"]),
        vocab_size=int(lm_config["vocab_size"]),
        rope_theta=float(lm_config["rope_theta"]),
        rms_norm_eps=float(lm_config["rms_norm_eps"]),
    )
    if not bool(lm_config.get("tie_word_embeddings", False)):
        raise ValueError("VibeVoice-1.5B ties the lm head to the embedding table; got untied config")

    def t(name: str) -> np.ndarray:
        return _load_tensor(index, name, path)

    layers = []
    for i in range(spec.num_layers):
        p = f"model.language_model.layers.{i}"
        from hipengine.loading.vibevoice_asr import VibevoiceQwen2LayerWeights

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
    embed = t("model.language_model.embed_tokens.weight")
    lm = VibevoiceQwen2Weights(
        spec=spec,
        embed_tokens=embed,
        final_norm=t("model.language_model.norm.weight"),
        lm_head=embed,  # tied
        layers=tuple(layers),
    )

    decoder_spec, decoder_weights, scaling, bias = load_vibevoice_tts_decoder(path)
    diffusion_spec, diffusion_weights, _, _ = load_vibevoice_tts_diffusion_head(path)

    acoustic_encoder_spec, acoustic_encoder = load_vibevoice_encoder(path, "acoustic")
    semantic_encoder_spec, semantic_encoder = load_vibevoice_encoder(path, "semantic")

    return VibevoiceTtsSessionWeights(
        lm=lm,
        acoustic_connector=load_vibevoice_connector(path, "acoustic"),
        semantic_connector=load_vibevoice_connector(path, "semantic"),
        acoustic_encoder=acoustic_encoder,
        semantic_encoder=semantic_encoder,
        acoustic_encoder_spec=acoustic_encoder_spec,
        semantic_encoder_spec=semantic_encoder_spec,
        decoder_spec=decoder_spec,
        decoder_weights=decoder_weights,
        diffusion_spec=diffusion_spec,
        diffusion_weights=diffusion_weights,
        speech_scaling_factor=scaling,
        speech_bias_factor=bias,
    )
