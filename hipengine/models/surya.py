"""Pinned Surya OCR 2 model contract and plugin metadata.

Source of truth: ``datalab-to/surya-ocr-2`` (BF16 safetensors checkpoint,
modified AI Pubs OpenRAIL-M weights) with the upstream implementation in
``datalab-to/surya`` and the stock transformers ``Qwen3_5`` backbone
(``Qwen3_5ForConditionalGeneration``: a Qwen3.5 vision tower feeding a
causal hybrid language decoder).

Surya OCR 2 is a vision-language OCR model: a page image is encoded by a
bidirectional ViT-style tower, merged 2x2 into decoder-width image tokens,
substituted into an expanded ``<|image_pad|>`` placeholder run, and the
causal Qwen3.5 hybrid decoder (18 gated-DeltaNet ``linear_attention`` +
6 ``full_attention`` layers) generates HTML/JSON with layout labels and
normalized 0-1000 bounding boxes. Unlike EVIE there is no retrieval head:
the tied LM head produces token logits directly.

Geometry (``config.json``, measured 2026-09-11 against the pinned
checkpoint, revision ``3b3d4cdf``): text stack hidden 1024, 24 layers with
18 gated-DeltaNet ``linear_attention`` and 6 ``full_attention`` (interval
4), 8 attention heads of 256 (GQA 2 kv heads) with a sigmoid output gate
(``q_proj`` spans 2x heads), partial RoPE 0.25 with interleaved mRoPE
sections [11, 11, 10], theta 1e7, SiLU MLP 3584, tied embeddings 65425.
GDN contract: 16 key heads x 128, 16 value heads x 128, causal conv width
4, time-step rank 16. Vision tower hidden 768, 12 blocks, 12 heads x 64,
patch 16, temporal patch 2, spatial merge 2, FFN 3072 (tanh-approximate
GELU), out 1024 with a learned 48x48 position table and an erf-GELU merger
(3072 -> 3072 -> 1024). Vision tensors carry biases (EVIE's do not).

Token contract (trap 1): the tokenizer and ``generation_config.json`` agree
that EOS is **2** (``<|im_end|>``) and padding is 0; image/start/end tokens
are 11/9/10 and video is 12. ``text_config.eos_token_id`` (248044) is stale
metadata outside the 65,425-token vocabulary and must never be used for
stopping. The effective EOS is pinned in :data:`SURYA_EOS_TOKEN_ID` and
cross-checked against ``generation_config`` when supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hipengine.models.registry import register_model

if TYPE_CHECKING:
    from hipengine.loading.safetensors import WeightIndex

PINNED_SURYA_MODEL_ID = "datalab-to/surya-ocr-2"
SURYA_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
SURYA_MODEL_TYPE = "qwen3_5"

# Effective stopping/padding tokens, resolved from tokenizer.json +
# generation_config.json (both agree). text_config.eos_token_id=248044 is
# stale, out-of-vocab metadata; see module docstring.
SURYA_EOS_TOKEN_ID = 2
SURYA_PAD_TOKEN_ID = 0
SURYA_IMAGE_TOKEN_ID = 11
SURYA_VIDEO_TOKEN_ID = 12
SURYA_VISION_START_TOKEN_ID = 9
SURYA_VISION_END_TOKEN_ID = 10


@dataclass(frozen=True)
class SuryaModelSpec:
    """Validated geometry and storage contract for Surya OCR 2."""

    hidden_size: int
    num_layers: int
    full_attention_interval: int
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, int, int]
    mrope_interleaved: bool
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool
    # effective token contract (never text_config.eos_token_id)
    eos_token_id: int
    pad_token_id: int
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    # gated-DeltaNet contract
    gdn_num_key_heads: int
    gdn_key_head_dim: int
    gdn_num_value_heads: int
    gdn_value_head_dim: int
    gdn_conv_kernel: int
    gdn_time_step_rank: int
    # vision contract
    vision_hidden_size: int
    vision_depth: int
    vision_num_heads: int
    vision_patch_size: int
    vision_temporal_patch_size: int
    vision_spatial_merge_size: int
    vision_intermediate_size: int
    vision_out_hidden_size: int
    vision_num_position_embeddings: int
    # optional MTP head declared by the checkpoint
    mtp_num_layers: int

    def is_full_attention(self, layer: int) -> bool:
        return self.layer_types[layer] == "full_attention"

    def vision_head_dim(self) -> int:
        return self.vision_hidden_size // self.vision_num_heads

    @property
    def gdn_inner_size(self) -> int:
        """Per-axis GDN width: value heads x value head dim."""

        return self.gdn_num_value_heads * self.gdn_value_head_dim


@dataclass(frozen=True)
class SuryaModel:
    """Surya OCR 2 plugin metadata."""

    name: str = "surya_ocr2"
    architectures: tuple[str, ...] = (SURYA_ARCHITECTURE,)
    default_quant: str = "fp32"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.{layer}.input_layernorm.weight",
        "model.language_model.layers.{layer}.post_attention_layernorm.weight",
        "model.language_model.layers.{layer}.self_attn.q_proj.weight",
        "model.language_model.layers.{layer}.self_attn.k_proj.weight",
        "model.language_model.layers.{layer}.self_attn.v_proj.weight",
        "model.language_model.layers.{layer}.self_attn.o_proj.weight",
        "model.language_model.layers.{layer}.self_attn.q_norm.weight",
        "model.language_model.layers.{layer}.self_attn.k_norm.weight",
        "model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.{layer}.linear_attn.in_proj_z.weight",
        "model.language_model.layers.{layer}.linear_attn.in_proj_b.weight",
        "model.language_model.layers.{layer}.linear_attn.in_proj_a.weight",
        "model.language_model.layers.{layer}.linear_attn.conv1d.weight",
        "model.language_model.layers.{layer}.linear_attn.A_log",
        "model.language_model.layers.{layer}.linear_attn.dt_bias",
        "model.language_model.layers.{layer}.linear_attn.norm.weight",
        "model.language_model.layers.{layer}.linear_attn.out_proj.weight",
        "model.language_model.layers.{layer}.mlp.{proj}.weight",
        "model.language_model.norm.weight",
        "model.visual.patch_embed.proj.{param}",
        "model.visual.pos_embed.weight",
        "model.visual.blocks.{layer}.{module}.{param}",
        "model.visual.merger.{param}",
        # present in the checkpoint (15 tensors); optional runtime follow-up
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.{layer}.{module}.{param}",
    )

    def layer_sequence(self) -> tuple[str, ...]:
        """Representative sequence for registry/fusion planning."""

        return (
            "surya_vision_patch_embed",
            "surya_vision_pos_embed",
            "surya_vision_block",
            "surya_vision_merger",
            "embed",
            self.text_layer_sequence("linear_attention"),
            self.text_layer_sequence("full_attention"),
            "final_rmsnorm",
            "tied_lm_head",
        )

    def text_layer_sequence(self, attention_kind: str) -> tuple[str, ...]:
        """Primitive layer keys for one text stack layer."""

        if attention_kind == "full_attention":
            attention = (
                "rmsnorm",
                "surya_qkv_proj",
                "surya_qk_rmsnorm",
                "surya_mrope",
                "surya_causal_attention",
                "surya_attn_output_gate",
                "full_attention_o_proj",
            )
        elif attention_kind == "linear_attention":
            attention = (
                "rmsnorm",
                "surya_gdn_qkvz_proj",
                "linear_attention_conv_prefill",
                "surya_gdn_gates",
                "surya_gdn_recurrence",
                "surya_gdn_rmsnorm_gated",
                "linear_attention_o_proj",
            )
        else:
            raise ValueError(
                "attention_kind must be 'full_attention' or 'linear_attention'"
            )
        return (
            *attention,
            "add_rmsnorm",
            "surya_swiglu",
            "w8a16_linear",
            "residual_add",
        )


def _require(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise ValueError(f"Surya config missing {name!r}")
    return config[name]


def _layer_types(config: Mapping[str, Any], num_layers: int) -> tuple[str, ...]:
    text = _require(config, "text_config")
    types = tuple(str(t) for t in _require(text, "layer_types"))
    if len(types) != num_layers:
        raise ValueError(
            f"Surya layer_types has {len(types)} entries, expected {num_layers}"
        )
    allowed = {"linear_attention", "full_attention"}
    bad = [t for t in types if t not in allowed]
    if bad:
        raise ValueError(f"Surya unknown layer types {bad}")
    return types


def parse_surya_model_spec(
    config: Mapping[str, Any],
    generation_config: Mapping[str, Any] | None = None,
) -> SuryaModelSpec:
    """Parse and reject drift from the pinned Surya OCR 2 contract.

    ``generation_config`` (optional) cross-checks the effective token
    contract. ``text_config.eos_token_id`` is deliberately ignored: the
    pinned checkpoint carries a stale out-of-vocab value there (see module
    docstring).
    """

    if _require(config, "architectures") != [SURYA_ARCHITECTURE]:
        raise ValueError("not a Surya Qwen3_5ForConditionalGeneration checkpoint")
    if _require(config, "model_type") != SURYA_MODEL_TYPE:
        raise ValueError("unexpected Surya model_type")

    text = _require(config, "text_config")
    vision = _require(config, "vision_config")
    rope = _require(text, "rope_parameters")

    num_layers = int(_require(text, "num_hidden_layers"))
    eos_token_id = SURYA_EOS_TOKEN_ID
    if generation_config is not None and "eos_token_id" in generation_config:
        gen_eos = int(generation_config["eos_token_id"])
        if gen_eos != eos_token_id:
            raise ValueError(
                f"Surya generation_config eos_token_id {gen_eos} contradicts the "
                f"pinned tokenizer EOS {eos_token_id}"
            )
    spec = SuryaModelSpec(
        hidden_size=int(_require(text, "hidden_size")),
        num_layers=num_layers,
        full_attention_interval=int(_require(text, "full_attention_interval")),
        layer_types=_layer_types(config, num_layers),
        num_attention_heads=int(_require(text, "num_attention_heads")),
        num_key_value_heads=int(_require(text, "num_key_value_heads")),
        head_dim=int(_require(text, "head_dim")),
        rope_theta=float(_require(rope, "rope_theta")),
        partial_rotary_factor=float(_require(rope, "partial_rotary_factor")),
        mrope_section=tuple(int(x) for x in _require(rope, "mrope_section")),
        mrope_interleaved=bool(_require(rope, "mrope_interleaved")),
        intermediate_size=int(_require(text, "intermediate_size")),
        vocab_size=int(_require(text, "vocab_size")),
        tie_word_embeddings=bool(_require(text, "tie_word_embeddings")),
        eos_token_id=eos_token_id,
        pad_token_id=SURYA_PAD_TOKEN_ID,
        image_token_id=int(_require(config, "image_token_id")),
        video_token_id=int(_require(config, "video_token_id")),
        vision_start_token_id=int(_require(config, "vision_start_token_id")),
        vision_end_token_id=int(_require(config, "vision_end_token_id")),
        gdn_num_key_heads=int(_require(text, "linear_num_key_heads")),
        gdn_key_head_dim=int(_require(text, "linear_key_head_dim")),
        gdn_num_value_heads=int(_require(text, "linear_num_value_heads")),
        gdn_value_head_dim=int(_require(text, "linear_value_head_dim")),
        gdn_conv_kernel=int(_require(text, "linear_conv_kernel_dim")),
        gdn_time_step_rank=16,
        vision_hidden_size=int(_require(vision, "hidden_size")),
        vision_depth=int(_require(vision, "depth")),
        vision_num_heads=int(_require(vision, "num_heads")),
        vision_patch_size=int(_require(vision, "patch_size")),
        vision_temporal_patch_size=int(_require(vision, "temporal_patch_size")),
        vision_spatial_merge_size=int(_require(vision, "spatial_merge_size")),
        vision_intermediate_size=int(_require(vision, "intermediate_size")),
        vision_out_hidden_size=int(_require(vision, "out_hidden_size")),
        vision_num_position_embeddings=int(
            _require(vision, "num_position_embeddings")
        ),
        mtp_num_layers=int(
            config.get(
                "num_nextn_predict_layers", text.get("mtp_num_hidden_layers", 0)
            )
        ),
    )

    expected = {
        "hidden_size": 1_024,
        "num_layers": 24,
        "full_attention_interval": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "intermediate_size": 3_584,
        "vocab_size": 65_425,
        "gdn_num_key_heads": 16,
        "gdn_key_head_dim": 128,
        "gdn_num_value_heads": 16,
        "gdn_value_head_dim": 128,
        "gdn_conv_kernel": 4,
        "vision_hidden_size": 768,
        "vision_depth": 12,
        "vision_num_heads": 12,
        "vision_patch_size": 16,
        "vision_temporal_patch_size": 2,
        "vision_spatial_merge_size": 2,
        "vision_intermediate_size": 3_072,
        "vision_out_hidden_size": 1_024,
        "vision_num_position_embeddings": 2_304,
        "mtp_num_layers": 1,
    }
    for name, want in expected.items():
        got = getattr(spec, name)
        if got != want:
            raise ValueError(
                f"Surya drifted from the pinned OCR 2 contract: "
                f"{name}={got}, expected {want}"
            )
    if not spec.mrope_interleaved:
        raise ValueError("Surya requires mrope_interleaved")
    if spec.mrope_section != (11, 11, 10):
        raise ValueError(f"Surya unexpected mrope_section {spec.mrope_section}")
    if spec.partial_rotary_factor != 0.25 or spec.rope_theta != 1e7:
        raise ValueError(
            f"Surya unexpected rotary: partial={spec.partial_rotary_factor}, "
            f"theta={spec.rope_theta}"
        )
    if spec.gdn_num_key_heads * spec.gdn_key_head_dim != spec.gdn_inner_size:
        raise ValueError("Surya GDN key/value widths disagree")
    if spec.vision_num_heads * spec.vision_head_dim() != spec.vision_hidden_size:
        raise ValueError("Surya vision head geometry disagrees")
    if spec.vision_out_hidden_size != spec.hidden_size:
        raise ValueError("Surya merger output must match text hidden size")
    if not spec.tie_word_embeddings:
        raise ValueError("Surya OCR 2 requires tied input/output embeddings")
    schedule = tuple(
        "full" if spec.is_full_attention(i) else "linear"
        for i in range(spec.num_layers)
    )
    if schedule != ("linear", "linear", "linear", "full") * 6:
        raise ValueError(
            f"Surya unexpected layer schedule (want 3x linear + 1x full x6): {schedule}"
        )
    return spec


def expected_surya_weight_shapes(spec: SuryaModelSpec) -> dict[str, tuple[int, ...]]:
    """Checkpoint tensor-name -> expected shape for validation."""

    h = spec.hidden_size
    vh, vi = spec.vision_hidden_size, spec.vision_intermediate_size
    merge2 = spec.vision_spatial_merge_size**2
    qkv = spec.num_attention_heads * spec.head_dim  # 2x for the output gate
    kv = spec.num_key_value_heads * spec.head_dim
    gdn_in = spec.gdn_inner_size  # per-axis value width
    ts = spec.gdn_time_step_rank
    shapes: dict[str, tuple[int, ...]] = {
        "model.language_model.embed_tokens.weight": (spec.vocab_size, h),
        "model.language_model.norm.weight": (h,),
        "model.visual.patch_embed.proj.weight": (
            vh,
            3,
            spec.vision_temporal_patch_size,
            spec.vision_patch_size,
            spec.vision_patch_size,
        ),
        "model.visual.patch_embed.proj.bias": (vh,),
        "model.visual.pos_embed.weight": (
            spec.vision_num_position_embeddings,
            vh,
        ),
        "model.visual.merger.norm.weight": (vh,),
        "model.visual.merger.norm.bias": (vh,),
        "model.visual.merger.linear_fc1.weight": (vh * merge2, vh * merge2),
        "model.visual.merger.linear_fc1.bias": (vh * merge2,),
        "model.visual.merger.linear_fc2.weight": (
            spec.vision_out_hidden_size,
            vh * merge2,
        ),
        "model.visual.merger.linear_fc2.bias": (spec.vision_out_hidden_size,),
        # MTP head (present in the checkpoint; runtime follow-up)
        "mtp.fc.weight": (h, 2 * h),
        "mtp.norm.weight": (h,),
        "mtp.pre_fc_norm_embedding.weight": (h,),
        "mtp.pre_fc_norm_hidden.weight": (h,),
    }
    for i in range(spec.num_layers):
        p = f"model.language_model.layers.{i}."
        if spec.is_full_attention(i):
            shapes.update(
                {
                    p + "self_attn.q_proj.weight": (qkv * 2, h),
                    p + "self_attn.k_proj.weight": (kv, h),
                    p + "self_attn.v_proj.weight": (kv, h),
                    p + "self_attn.o_proj.weight": (h, qkv),
                    p + "self_attn.q_norm.weight": (spec.head_dim,),
                    p + "self_attn.k_norm.weight": (spec.head_dim,),
                }
            )
        else:
            shapes.update(
                {
                    p + "linear_attn.in_proj_qkv.weight": (3 * gdn_in, h),
                    p + "linear_attn.in_proj_z.weight": (gdn_in, h),
                    p + "linear_attn.in_proj_b.weight": (ts, h),
                    p + "linear_attn.in_proj_a.weight": (ts, h),
                    p + "linear_attn.conv1d.weight": (
                        3 * gdn_in,  # causal conv over the fused qkv projection
                        1,
                        spec.gdn_conv_kernel,
                    ),
                    p + "linear_attn.A_log": (ts,),
                    p + "linear_attn.dt_bias": (ts,),
                    p + "linear_attn.norm.weight": (spec.gdn_key_head_dim,),
                    p + "linear_attn.out_proj.weight": (h, gdn_in),
                }
            )
        shapes.update(
            {
                p + "input_layernorm.weight": (h,),
                p + "post_attention_layernorm.weight": (h,),
                p + "mlp.gate_proj.weight": (spec.intermediate_size, h),
                p + "mlp.up_proj.weight": (spec.intermediate_size, h),
                p + "mlp.down_proj.weight": (h, spec.intermediate_size),
            }
        )
    for i in range(spec.vision_depth):
        p = f"model.visual.blocks.{i}."
        shapes.update(
            {
                p + "norm1.weight": (vh,),
                p + "norm1.bias": (vh,),
                p + "norm2.weight": (vh,),
                p + "norm2.bias": (vh,),
                p + "attn.qkv.weight": (3 * vh, vh),
                p + "attn.qkv.bias": (3 * vh,),
                p + "attn.proj.weight": (vh, vh),
                p + "attn.proj.bias": (vh,),
                p + "mlp.linear_fc1.weight": (vi, vh),
                p + "mlp.linear_fc1.bias": (vi,),
                p + "mlp.linear_fc2.weight": (vh, vi),
                p + "mlp.linear_fc2.bias": (vh,),
            }
        )
    for i in range(spec.mtp_num_layers):
        p = f"mtp.layers.{i}."
        shapes.update(
            {
                p + "input_layernorm.weight": (h,),
                p + "post_attention_layernorm.weight": (h,),
                p + "self_attn.q_proj.weight": (qkv * 2, h),
                p + "self_attn.k_proj.weight": (kv, h),
                p + "self_attn.v_proj.weight": (kv, h),
                p + "self_attn.o_proj.weight": (h, qkv),
                p + "self_attn.q_norm.weight": (spec.head_dim,),
                p + "self_attn.k_norm.weight": (spec.head_dim,),
                p + "mlp.gate_proj.weight": (spec.intermediate_size, h),
                p + "mlp.up_proj.weight": (spec.intermediate_size, h),
                p + "mlp.down_proj.weight": (h, spec.intermediate_size),
            }
        )
    return shapes


def validate_surya_weight_index(index: "WeightIndex", spec: SuryaModelSpec) -> None:
    """Raise on any missing/extra/shape-drifted Surya checkpoint tensor."""

    expected = expected_surya_weight_shapes(spec)
    missing = [n for n in expected if n not in index.tensors]
    if missing:
        raise ValueError(f"Surya checkpoint missing tensors: {missing[:8]}")
    extra = [n for n in index.tensors if n not in expected]
    if extra:
        raise ValueError(f"Surya checkpoint has unexpected tensors: {extra[:8]}")
    for name, want in expected.items():
        info = index.tensors[name]
        if info.shape != want:
            raise ValueError(
                f"Surya tensor {name} shape {info.shape}, expected {want}"
            )
        if info.dtype not in ("BF16", "F32"):
            raise ValueError(
                f"Surya tensor {name} dtype {info.dtype} not supported"
            )


SURYA = SuryaModel()
register_model(SURYA)
