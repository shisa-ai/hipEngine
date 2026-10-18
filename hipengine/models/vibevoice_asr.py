"""VibeVoice ASR model metadata for the native HF checkpoint loader."""
from dataclasses import dataclass
from hipengine.models.registry import register_model

@dataclass(frozen=True)
class VibeVoiceASRModel:
    name: str = "vibevoice_asr"
    architectures: tuple[str, ...] = ("VibeVoiceAsrForConditionalGeneration",)
    default_quant: str = "bf16"

    def layer_sequence(self):
        return ("vibevoice_frontend_gemm", "vibevoice_prefill", "vv_kv_write_spans", "vv_attention_spans")

VIBEVOICE_ASR = register_model(VibeVoiceASRModel())
