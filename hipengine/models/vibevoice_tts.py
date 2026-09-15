"""VibeVoice TTS model metadata for the native HF checkpoint loader.

The TTS checkpoint's ``model_type`` is ``vibevoice`` and its architecture string is
``VibeVoiceForConditionalGeneration``; the diffusion head, the acoustic decoder and
the semantic feedback encoder hang off the same ``config.json``, so this plugin names
the same layer sequence the ASR lane uses and adds the speech stack's layers.
"""
from dataclasses import dataclass
from hipengine.models.registry import register_model


@dataclass(frozen=True)
class VibeVoiceTTSModel:
    name: str = "vibevoice"
    architectures: tuple[str, ...] = ("VibeVoiceForConditionalGeneration",)
    default_quant: str = "bf16"

    def layer_sequence(self):
        return (
            "vibevoice_frontend_gemm",
            "vibevoice_prefill",
            "vv_kv_write_spans",
            "vv_attention_spans",
            "vv_tts_diffusion_head",
            "vv_tts_decoder",
        )


VIBEVOICE_TTS = register_model(VibeVoiceTTSModel())
