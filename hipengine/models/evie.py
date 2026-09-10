"""Pinned EVIE-4.5B (ColQwen3_5) model contract and plugin metadata.

Source of truth: ``tencent/EVIE-4.5B`` (safetensors checkpoint, Apache-2.0)
with the reference implementation in the Tencent/EVIE repository
(``colpali_engine.models.qwen3_5.colqwen3_5``) and the stock transformers
``Qwen3_5Model`` backbone.

EVIE is a visual document retrieval encoder (ColBERT-style late interaction):
queries and document page images are encoded into per-token multi-vector
embeddings by a bidirectional Qwen3.5 hybrid stack, and relevance is scored
by MaxSim. The head is a single ``Linear(2560 -> 2048)`` projection with
Prefix-MRL truncation at runtime ({64, 128, 256, 512, 1024, 2048}) followed
by per-token L2 normalization; the reference deployment uses d128.

Geometry (``config.json``): text stack hidden 2560, 32 layers with 24
gated-DeltaNet ``linear_attention`` and 8 ``full_attention`` (interval 4),
16 attention heads of 256 (GQA 4 kv heads), partial RoPE 0.25 with
interleaved mRoPE sections [11, 11, 10], interleaved mRoPE flag, SiLU MLP
9216, tied embeddings 248320; vision tower hidden 1024, 24 blocks, 16 heads,
patch 16, temporal patch 2, spatial merge 2, out 2560 with a learned
48x48 position-embedding table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hipengine.models.registry import register_model

if TYPE_CHECKING:
    from hipengine.loading.safetensors import WeightIndex

PINNED_EVIE_MODEL_ID = "tencent/EVIE-4.5B"
EVIE_ARCHITECTURE = "ColQwen3_5"
EVIE_MODEL_TYPE = "qwen3_5"

EVIE_HEAD_DIMS: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
EVIE_DEFAULT_HEAD = 128
EVIE_ANCHOR_DIM = 128


@dataclass(frozen=True)
class EvieModelSpec:
    """Validated geometry and storage contract for EVIE-4.5B."""

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
    image_token_id: int
    vision_hidden_size: int
    vision_depth: int
    vision_num_heads: int
    vision_patch_size: int
    vision_spatial_merge_size: int
    vision_intermediate_size: int
    vision_out_hidden_size: int
    vision_num_position_embeddings: int
    proj_dim: int
    head_dims: tuple[int, ...]
    default_head: int

    def is_full_attention(self, layer: int) -> bool:
        return self.layer_types[layer] == "full_attention"

    def vision_head_dim(self) -> int:
        return self.vision_hidden_size // self.vision_num_heads


@dataclass(frozen=True)
class EvieModel:
    """EVIE-4.5B retrieval-encoder plugin metadata."""

    name: str = "evie_4p5b"
    architectures: tuple[str, ...] = (EVIE_ARCHITECTURE,)
    default_quant: str = "fp32"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        "language_model.embed_tokens.weight",
        "language_model.layers.{layer}.input_layernorm.weight",
        "language_model.layers.{layer}.post_attention_layernorm.weight",
        "language_model.layers.{layer}.self_attn.q_proj.weight",
        "language_model.layers.{layer}.self_attn.k_proj.weight",
        "language_model.layers.{layer}.self_attn.v_proj.weight",
        "language_model.layers.{layer}.self_attn.o_proj.weight",
        "language_model.layers.{layer}.self_attn.q_norm.weight",
        "language_model.layers.{layer}.self_attn.k_norm.weight",
        "language_model.layers.{layer}.linear_attn.in_proj_qkv.weight",
        "language_model.layers.{layer}.linear_attn.in_proj_z.weight",
        "language_model.layers.{layer}.linear_attn.in_proj_b.weight",
        "language_model.layers.{layer}.linear_attn.in_proj_a.weight",
        "language_model.layers.{layer}.linear_attn.conv1d.weight",
        "language_model.layers.{layer}.linear_attn.A_log",
        "language_model.layers.{layer}.linear_attn.dt_bias",
        "language_model.layers.{layer}.linear_attn.norm.weight",
        "language_model.layers.{layer}.linear_attn.out_proj.weight",
        "language_model.layers.{layer}.mlp.{proj}.weight",
        "language_model.norm.weight",
        "custom_text_proj.weight",
        "custom_text_proj.bias",
        "visual.patch_embed.proj.{param}",
        "visual.pos_embed.weight",
        "visual.blocks.{layer}.{module}.{param}",
        "visual.merger.{param}",
    )

    def layer_sequence(self) -> Sequence[str]:
        """Representative encoder sequence for registry/fusion planning."""

        return (
            "evie_vision_patch_embed",
            "evie_vision_pos_embed",
            *self._vision_block_sequence(),
            "evie_vision_merger",
            "embed",
            *self.text_layer_sequence("full_attention"),
            "final_rmsnorm",
            "evie_custom_text_proj",
            "evie_l2_normalize",
        )

    def _vision_block_sequence(self) -> tuple[str, ...]:
        return (
            "evie_vision_layernorm",
            "evie_vision_qkv_proj",
            "evie_vision_rope",
            "evie_vision_bidirectional_attention",
            "evie_vision_o_proj",
            "evie_vision_layernorm",
            "evie_vision_mlp",
            "evie_residual_add",
        ) * 1  # one representative block

    def text_layer_sequence(self, attention_kind: str) -> tuple[str, ...]:
        """Primitive layer keys for one text stack layer."""

        if attention_kind == "full_attention":
            attention = (
                "rmsnorm",
                "evie_q_gate_proj",
                "evie_qk_rmsnorm",
                "evie_mrope",
                "evie_bidirectional_attention",
                "evie_attn_output_gate",
                "full_attention_o_proj",
            )
        elif attention_kind == "linear_attention":
            attention = (
                "rmsnorm",
                "evie_gdn_qkvz_proj",
                "linear_attention_conv_prefill",
                "evie_gdn_gates",
                "evie_gdn_recurrence",
                "evie_gdn_rmsnorm_gated",
                "linear_attention_o_proj",
            )
        else:
            raise ValueError(
                "attention_kind must be 'full_attention' or 'linear_attention'"
            )
        return (
            *attention,
            "add_rmsnorm",
            "evie_swiglu",
            "w8a16_linear",
            "residual_add",
        )


def _require(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise ValueError(f"EVIE config missing {name!r}")
    return config[name]


def _layer_types(config: Mapping[str, Any], num_layers: int) -> tuple[str, ...]:
    text = _require(config, "text_config")
    types = tuple(str(t) for t in _require(text, "layer_types"))
    if len(types) != num_layers:
        raise ValueError(
            f"EVIE layer_types has {len(types)} entries, expected {num_layers}"
        )
    allowed = {"linear_attention", "full_attention"}
    bad = [t for t in types if t not in allowed]
    if bad:
        raise ValueError(f"EVIE unknown layer types {bad}")
    return types


def parse_evie_model_spec(config: Mapping[str, Any]) -> EvieModelSpec:
    """Parse and reject drift from the official EVIE-4.5B contract."""

    if _require(config, "architectures") != [EVIE_ARCHITECTURE]:
        raise ValueError("not an EVIE ColQwen3_5 checkpoint")
    if _require(config, "model_type") != EVIE_MODEL_TYPE:
        raise ValueError("unexpected EVIE model_type")

    text = _require(config, "text_config")
    vision = _require(config, "vision_config")
    rope = _require(text, "rope_parameters")

    num_layers = int(_require(text, "num_hidden_layers"))
    spec = EvieModelSpec(
        hidden_size=int(_require(text, "hidden_size")),
        num_layers=num_layers,
        full_attention_interval=int(_require(text, "full_attention_interval")),
        layer_types=_layer_types(config, num_layers),
        num_attention_heads=int(_require(text, "num_attention_heads")),
        num_key_value_heads=int(_require(text, "num_key_value_heads")),
        head_dim=int(_require(text, "head_dim")),
        rope_theta=float(_require(rope, "rope_theta")),
        partial_rotary_factor=float(rope.get("partial_rotary_factor", 1.0)),
        mrope_section=tuple(int(x) for x in _require(rope, "mrope_section")),
        mrope_interleaved=bool(_require(rope, "mrope_interleaved")),
        intermediate_size=int(_require(text, "intermediate_size")),
        vocab_size=int(_require(text, "vocab_size")),
        image_token_id=int(_require(config, "image_token_id")),
        vision_hidden_size=int(_require(vision, "hidden_size")),
        vision_depth=int(_require(vision, "depth")),
        vision_num_heads=int(_require(vision, "num_heads")),
        vision_patch_size=int(_require(vision, "patch_size")),
        vision_spatial_merge_size=int(_require(vision, "spatial_merge_size")),
        vision_intermediate_size=int(_require(vision, "intermediate_size")),
        vision_out_hidden_size=int(_require(vision, "out_hidden_size")),
        vision_num_position_embeddings=int(
            _require(vision, "num_position_embeddings")
        ),
        # 4.5B is Prefix-MRL (head_dims ladder, d128 deployment head);
        # 8B is a single-head model (config "dim", no head_dims) whose
        # custom_text_proj is Linear(hidden, dim) with full-dim embeddings
        proj_dim=int(_require(config, "dim")) if "dim" in config else 2048,
        head_dims=(
            tuple(int(d) for d in _require(config, "head_dims"))
            if "head_dims" in config
            else (int(config["dim"]),)
        ),
        default_head=(
            EVIE_DEFAULT_HEAD if "head_dims" in config else int(config["dim"])
        ),
    )

    # per-checkpoint geometry tables; the shared GDN contract (32 value
    # heads x 128, 16 key heads x 128, conv 4) and the attention contract
    # (16 q heads x 256, GQA 4, interval 4) are common to the family
    if spec.head_dims != EVIE_HEAD_DIMS:
        expected = {
            "hidden_size": 4_096,
            "num_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 12_288,
            "vocab_size": 248_320,
            "vision_hidden_size": 1_152,
            "vision_depth": 27,
            "vision_num_heads": 16,
            "vision_patch_size": 16,
            "vision_spatial_merge_size": 2,
            "vision_num_position_embeddings": 2_304,
        }
    else:
        expected = {
            "hidden_size": 2_560,
            "num_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 9_216,
            "vocab_size": 248_320,
            "vision_hidden_size": 1_024,
            "vision_depth": 24,
            "vision_num_heads": 16,
            "vision_patch_size": 16,
            "vision_spatial_merge_size": 2,
            "vision_num_position_embeddings": 2_304,
        }
    for name, want in expected.items():
        got = getattr(spec, name)
        if got != want:
            raise ValueError(
                f"EVIE drifted from the pinned 4.5B contract: {name}={got}, expected {want}"
            )
    if not spec.mrope_interleaved:
        raise ValueError("EVIE requires mrope_interleaved")
    if spec.mrope_section != (11, 11, 10):
        raise ValueError(f"EVIE unexpected mrope_section {spec.mrope_section}")
    return spec


def expected_evie_weight_shapes(spec: EvieModelSpec) -> dict[str, tuple[int, ...]]:
    """Checkpoint tensor-name -> expected shape for validation."""

    d, vh, vi = spec.vision_depth, spec.vision_hidden_size, spec.vision_intermediate_size
    merge2 = spec.vision_spatial_merge_size**2
    shapes: dict[str, tuple[int, ...]] = {
        "language_model.embed_tokens.weight": (spec.vocab_size, spec.hidden_size),
        "language_model.norm.weight": (spec.hidden_size,),
        "custom_text_proj.weight": (spec.proj_dim, spec.hidden_size),
        "custom_text_proj.bias": (spec.proj_dim,),
        "visual.patch_embed.proj.weight": (
            vh,
            3,
            2,
            spec.vision_patch_size,
            spec.vision_patch_size,
        ),
        "visual.patch_embed.proj.bias": (vh,),
        "visual.pos_embed.weight": (spec.vision_num_position_embeddings, vh),
        "visual.merger.norm.weight": (vh,),
        "visual.merger.norm.bias": (vh,),
        "visual.merger.linear_fc1.weight": (vh * merge2, vh * merge2),
        "visual.merger.linear_fc1.bias": (vh * merge2,),
        "visual.merger.linear_fc2.weight": (spec.vision_out_hidden_size, vh * merge2),
        "visual.merger.linear_fc2.bias": (spec.vision_out_hidden_size,),
    }
    for i in range(spec.num_layers):
        p = f"language_model.layers.{i}."
        if spec.is_full_attention(i):
            shapes.update(
                {
                    p + "self_attn.q_proj.weight": (
                        spec.num_attention_heads * spec.head_dim * 2,
                        spec.hidden_size,
                    ),
                    p + "self_attn.k_proj.weight": (
                        spec.num_key_value_heads * spec.head_dim,
                        spec.hidden_size,
                    ),
                    p + "self_attn.v_proj.weight": (
                        spec.num_key_value_heads * spec.head_dim,
                        spec.hidden_size,
                    ),
                    p + "self_attn.o_proj.weight": (
                        spec.hidden_size,
                        spec.num_attention_heads * spec.head_dim,
                    ),
                    p + "self_attn.q_norm.weight": (spec.head_dim,),
                    p + "self_attn.k_norm.weight": (spec.head_dim,),
                }
            )
        else:
            shapes.update(
                {
                    p + "linear_attn.in_proj_qkv.weight": (8_192, spec.hidden_size),
                    p + "linear_attn.in_proj_z.weight": (4_096, spec.hidden_size),
                    p + "linear_attn.in_proj_b.weight": (32, spec.hidden_size),
                    p + "linear_attn.in_proj_a.weight": (32, spec.hidden_size),
                    p + "linear_attn.conv1d.weight": (8_192, 1, 4),
                    p + "linear_attn.A_log": (32,),
                    p + "linear_attn.dt_bias": (32,),
                    p + "linear_attn.norm.weight": (128,),
                    p + "linear_attn.out_proj.weight": (spec.hidden_size, 4_096),
                }
            )
        shapes.update(
            {
                p + "input_layernorm.weight": (spec.hidden_size,),
                p + "post_attention_layernorm.weight": (spec.hidden_size,),
                p + "mlp.gate_proj.weight": (spec.intermediate_size, spec.hidden_size),
                p + "mlp.up_proj.weight": (spec.intermediate_size, spec.hidden_size),
                p + "mlp.down_proj.weight": (spec.hidden_size, spec.intermediate_size),
            }
        )
    for i in range(d):
        p = f"visual.blocks.{i}."
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
    return shapes


def validate_evie_weight_index(index: "WeightIndex", spec: EvieModelSpec) -> None:
    """Raise on any missing/extra/shape-drifted EVIE checkpoint tensor."""

    expected = expected_evie_weight_shapes(spec)
    missing = [n for n in expected if n not in index.tensors]
    if missing:
        raise ValueError(f"EVIE checkpoint missing tensors: {missing[:8]}")
    extra = [n for n in index.tensors if n not in expected]
    if extra:
        raise ValueError(f"EVIE checkpoint has unexpected tensors: {extra[:8]}")
    for name, want in expected.items():
        info = index.tensors[name]
        if info.shape != want:
            raise ValueError(
                f"EVIE tensor {name} shape {info.shape}, expected {want}"
            )
        if info.dtype not in ("BF16", "F32"):
            raise ValueError(f"EVIE tensor {name} dtype {info.dtype} not supported")


EVIE = EvieModel()
register_model(EVIE)
