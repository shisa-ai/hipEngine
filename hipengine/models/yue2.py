"""YuE2 model metadata for the native checkpoint loader.

``m-a-p/YuE2-3B`` is a mixture-of-transformers: every layer carries a separate
AR attention/MLP path and a NAR attention/MLP path, plus a flow-matching
acoustic head. The plugin names the layer sequence the YuE2 session drives; the
VAE decoder is a separate checkpoint (``m-a-p/YuE2-Vae``) with its own loader.
"""

from dataclasses import dataclass

from hipengine.models.registry import register_model


@dataclass(frozen=True)
class YuE2Model:
    name: str = "yue2"
    architectures: tuple[str, ...] = ("YuE2ForCausalLM",)
    default_quant: str = "bf16"

    def layer_sequence(self):
        return (
            "yue2_embedding",
            "yue2_rmsnorm",
            "yue2_qkv_proj",
            "yue2_head_rmsnorm",
            "yue2_rotary",
            "yue2_kv_write_spans",
            "yue2_attention_spans",
            "yue2_mlp",
            "yue2_nar_attention",
            "yue2_velocity_head",
            "yue2_midpoint_step",
            "yue2_vae_decode",
        )


YUE2 = register_model(YuE2Model())
