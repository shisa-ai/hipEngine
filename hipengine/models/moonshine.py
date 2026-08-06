"""Pinned Moonshine ASR model contract and plugin metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from hipengine.models.registry import register_model

if TYPE_CHECKING:
    from hipengine.loading.safetensors import WeightIndex

PINNED_MOONSHINE_MODEL_ID = "shisa-ai/shisa-realtime-asr-0.92b"
PINNED_MOONSHINE_REVISION = "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d"
MOONSHINE_ENCODER_BUCKETS = ((16_000, 40), (80_000, 207), (480_000, 1248))
MOONSHINE_EMBEDDING_WEIGHT = "model.decoder.embed_tokens.weight"
MOONSHINE_LM_HEAD_ALIAS = "proj_out.weight"


@dataclass(frozen=True)
class MoonshineModelSpec:
    model_id: str
    source_revision: str
    architecture: str
    model_type: str
    stored_dtype: str
    runtime_dtype: str
    hidden_size: int
    encoder_layers: int
    decoder_layers: int
    encoder_attention_heads: int
    decoder_attention_heads: int
    encoder_kv_heads: int
    decoder_kv_heads: int
    head_dim: int
    padded_head_dim: int
    rotary_dim: int
    partial_rotary_factor: float
    rope_theta: float
    intermediate_size: int
    vocab_size: int
    max_positions: int
    self_cache_capacity: int
    layer_norm_epsilon: float
    bos_token_ids: tuple[int, ...]
    decoder_start_token_id: int
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    generation_max_length: int
    generation_num_beams: int
    encoder_buckets: tuple[tuple[int, int], ...]
    embedding_weight_name: str = MOONSHINE_EMBEDDING_WEIGHT
    lm_head_alias_name: str = MOONSHINE_LM_HEAD_ALIAS

    @property
    def parameter_count(self) -> int:
        return sum(math.prod(shape) for shape in expected_moonshine_weight_shapes(self).values())

    @property
    def runtime_weight_bytes(self) -> int:
        return self.parameter_count * 2


@dataclass(frozen=True)
class MoonshineForConditionalGenerationModel:
    name: str = "moonshine_asr"
    architectures: tuple[str, ...] = ("MoonshineForConditionalGeneration",)
    default_quant: str = "fp16"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        MOONSHINE_EMBEDDING_WEIGHT,
        "model.decoder.layers.{layer}.self_attn.{proj}.weight",
        "model.decoder.layers.{layer}.encoder_attn.{proj}.weight",
        "model.decoder.layers.{layer}.{norm}.weight",
        "model.decoder.layers.{layer}.mlp.fc1.weight",
        "model.decoder.layers.{layer}.mlp.fc1.bias",
        "model.decoder.layers.{layer}.mlp.fc2.weight",
        "model.decoder.layers.{layer}.mlp.fc2.bias",
        "model.decoder.norm.weight",
    )

    def layer_sequence(self) -> Sequence[str]:
        return (
            "moonshine_embedding",
            "moonshine_layernorm",
            "moonshine_qkv_proj",
            "moonshine_partial_rope",
            "moonshine_self_cache",
            "moonshine_self_attention",
            "moonshine_projection",
            "moonshine_residual",
            "moonshine_layernorm",
            "moonshine_projection",
            "moonshine_cross_attention",
            "moonshine_projection",
            "moonshine_residual",
            "moonshine_layernorm",
            "moonshine_decoder_mlp",
            "moonshine_residual",
            "moonshine_layernorm",
            "moonshine_lm_head",
            "moonshine_argmax",
        )


def normalize_moonshine_token_ids(
    value: Any,
    name: str,
    *,
    vocab_size: int,
) -> tuple[int, ...]:
    if isinstance(value, bool):
        values = ()
    elif isinstance(value, int):
        values = (value,)
    elif isinstance(value, (list, tuple)):
        values = tuple(value)
    else:
        values = ()
    if not values:
        raise ValueError(f"{name} must be an integer or non-empty integer list")
    result: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0 or item >= vocab_size:
            raise ValueError(f"{name} contains invalid token ID {item!r}")
        if item in result:
            raise ValueError(f"{name} contains duplicate token ID {item}")
        result.append(item)
    return tuple(result)


def _token_field(
    config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    name: str,
    vocab_size: int,
) -> tuple[int, ...]:
    config_value = normalize_moonshine_token_ids(
        config.get(name),
        f"config.{name}",
        vocab_size=vocab_size,
    )
    if name not in generation_config or generation_config[name] is None:
        return config_value
    generation_value = normalize_moonshine_token_ids(
        generation_config[name],
        f"generation_config.{name}",
        vocab_size=vocab_size,
    )
    if generation_value != config_value:
        raise ValueError(
            f"{name} differs between config {config_value} and generation config {generation_value}"
        )
    return generation_value


def _int(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_moonshine_model_spec(
    config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
) -> MoonshineModelSpec:
    """Parse and reject any drift from the pinned Phase-1 model contract."""

    expected = {
        "architectures": ["MoonshineForConditionalGeneration"],
        "model_type": "moonshine",
        "dtype": "float32",
        "hidden_size": 416,
        "encoder_num_hidden_layers": 8,
        "decoder_num_hidden_layers": 8,
        "encoder_num_attention_heads": 8,
        "decoder_num_attention_heads": 8,
        "encoder_num_key_value_heads": 8,
        "decoder_num_key_value_heads": 8,
        "intermediate_size": 1664,
        "max_position_embeddings": 194,
        "vocab_size": 36_864,
        "pad_head_dim_to_multiple_of": 8,
        "encoder_hidden_act": "gelu",
        "decoder_hidden_act": "silu",
        "attention_bias": False,
        "is_encoder_decoder": True,
        "tie_word_embeddings": True,
    }
    for name, expected_value in expected.items():
        if config.get(name) != expected_value:
            raise ValueError(
                f"Moonshine {name}={config.get(name)!r}, expected {expected_value!r}"
            )
    hidden = _int(config, "hidden_size")
    encoder_heads = _int(config, "encoder_num_attention_heads")
    decoder_heads = _int(config, "decoder_num_attention_heads")
    if hidden % encoder_heads or hidden % decoder_heads:
        raise ValueError("hidden_size must be divisible by attention heads")
    head_dim = hidden // decoder_heads
    if hidden // encoder_heads != head_dim:
        raise ValueError("encoder and decoder head dimensions must match")
    padded_head_dim = 8 * ((head_dim + 7) // 8)
    rope = config.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("rope_parameters must be an object")
    partial = float(rope.get("partial_rotary_factor", float("nan")))
    theta = float(rope.get("rope_theta", float("nan")))
    rotary_dim = int(head_dim * partial) if math.isfinite(partial) else -1
    if (
        not math.isclose(partial, 0.62, rel_tol=0, abs_tol=1.0e-12)
        or rotary_dim != 32
        or rope.get("rope_type") != "default"
        or not math.isclose(theta, 10_000.0, rel_tol=0, abs_tol=1.0e-12)
    ):
        raise ValueError("Moonshine partial/default RoPE contract differs from the pin")
    if float(config.get("partial_rotary_factor", float("nan"))) != partial:
        raise ValueError("top-level and rope partial_rotary_factor differ")

    vocab_size = _int(config, "vocab_size")
    bos = _token_field(config, generation_config, "bos_token_id", vocab_size)
    decoder_start = _token_field(
        config,
        generation_config,
        "decoder_start_token_id",
        vocab_size,
    )
    eos = _token_field(config, generation_config, "eos_token_id", vocab_size)
    pad = _token_field(config, generation_config, "pad_token_id", vocab_size)
    if len(decoder_start) != 1 or len(pad) != 1:
        raise ValueError("decoder start and pad fields must each resolve to one token")
    if bos != (1,) or decoder_start != (1,) or eos != (2,) or pad != (2,):
        raise ValueError("Moonshine special token IDs differ from the pin")
    max_length = _int(generation_config, "max_length")
    num_beams = _int(generation_config, "num_beams")
    if max_length != 195:
        raise ValueError("generation max_length must equal 195")
    if generation_config.get("do_sample") is not False or generation_config.get("use_cache") is not True:
        raise ValueError("generation must be deterministic and cache-enabled")

    return MoonshineModelSpec(
        model_id=PINNED_MOONSHINE_MODEL_ID,
        source_revision=PINNED_MOONSHINE_REVISION,
        architecture="MoonshineForConditionalGeneration",
        model_type="moonshine",
        stored_dtype="float32",
        runtime_dtype="float16",
        hidden_size=hidden,
        encoder_layers=_int(config, "encoder_num_hidden_layers"),
        decoder_layers=_int(config, "decoder_num_hidden_layers"),
        encoder_attention_heads=encoder_heads,
        decoder_attention_heads=decoder_heads,
        encoder_kv_heads=_int(config, "encoder_num_key_value_heads"),
        decoder_kv_heads=_int(config, "decoder_num_key_value_heads"),
        head_dim=head_dim,
        padded_head_dim=padded_head_dim,
        rotary_dim=rotary_dim,
        partial_rotary_factor=partial,
        rope_theta=theta,
        intermediate_size=_int(config, "intermediate_size"),
        vocab_size=vocab_size,
        max_positions=_int(config, "max_position_embeddings"),
        self_cache_capacity=_int(config, "max_position_embeddings"),
        layer_norm_epsilon=1.0e-5,
        bos_token_ids=bos,
        decoder_start_token_id=decoder_start[0],
        eos_token_ids=eos,
        pad_token_id=pad[0],
        generation_max_length=max_length,
        generation_num_beams=num_beams,
        encoder_buckets=MOONSHINE_ENCODER_BUCKETS,
    )


def expected_moonshine_weight_shapes(spec: MoonshineModelSpec) -> dict[str, tuple[int, ...]]:
    h = spec.hidden_size
    intermediate = spec.intermediate_size
    encoder_q = spec.encoder_attention_heads * spec.head_dim
    encoder_kv = spec.encoder_kv_heads * spec.head_dim
    decoder_q = spec.decoder_attention_heads * spec.head_dim
    decoder_kv = spec.decoder_kv_heads * spec.head_dim
    shapes: dict[str, tuple[int, ...]] = {spec.embedding_weight_name: (spec.vocab_size, h)}
    for layer in range(spec.decoder_layers):
        prefix = f"model.decoder.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.encoder_attn.k_proj.weight": (decoder_kv, h),
                f"{prefix}.encoder_attn.o_proj.weight": (h, decoder_q),
                f"{prefix}.encoder_attn.q_proj.weight": (decoder_q, h),
                f"{prefix}.encoder_attn.v_proj.weight": (decoder_kv, h),
                f"{prefix}.final_layernorm.weight": (h,),
                f"{prefix}.input_layernorm.weight": (h,),
                f"{prefix}.mlp.fc1.bias": (2 * intermediate,),
                f"{prefix}.mlp.fc1.weight": (2 * intermediate, h),
                f"{prefix}.mlp.fc2.bias": (h,),
                f"{prefix}.mlp.fc2.weight": (h, intermediate),
                f"{prefix}.post_attention_layernorm.weight": (h,),
                f"{prefix}.self_attn.k_proj.weight": (decoder_kv, h),
                f"{prefix}.self_attn.o_proj.weight": (h, decoder_q),
                f"{prefix}.self_attn.q_proj.weight": (decoder_q, h),
                f"{prefix}.self_attn.v_proj.weight": (decoder_kv, h),
            }
        )
    shapes["model.decoder.norm.weight"] = (h,)
    shapes.update(
        {
            "model.encoder.conv1.weight": (h, 1, 127),
            "model.encoder.conv2.bias": (2 * h,),
            "model.encoder.conv2.weight": (2 * h, h, 7),
            "model.encoder.conv3.bias": (h,),
            "model.encoder.conv3.weight": (h, 2 * h, 3),
            "model.encoder.groupnorm.bias": (h,),
            "model.encoder.groupnorm.weight": (h,),
            "model.encoder.layer_norm.weight": (h,),
        }
    )
    for layer in range(spec.encoder_layers):
        prefix = f"model.encoder.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (h,),
                f"{prefix}.mlp.fc1.bias": (intermediate,),
                f"{prefix}.mlp.fc1.weight": (intermediate, h),
                f"{prefix}.mlp.fc2.bias": (h,),
                f"{prefix}.mlp.fc2.weight": (h, intermediate),
                f"{prefix}.post_attention_layernorm.weight": (h,),
                f"{prefix}.self_attn.k_proj.weight": (encoder_kv, h),
                f"{prefix}.self_attn.o_proj.weight": (h, encoder_q),
                f"{prefix}.self_attn.q_proj.weight": (encoder_q, h),
                f"{prefix}.self_attn.v_proj.weight": (encoder_kv, h),
            }
        )
    return dict(sorted(shapes.items()))


def validate_moonshine_weight_index(
    spec: MoonshineModelSpec, index: "WeightIndex", *, packed: bool = False
) -> None:
    expected = expected_moonshine_weight_shapes(spec)
    missing = sorted(set(expected) - set(index.tensors))
    extra = sorted(set(index.tensors) - set(expected))
    if missing or extra:
        raise ValueError(f"Moonshine weight names differ: missing={missing[:5]}, extra={extra[:5]}")
    for name, shape in expected.items():
        info = index.tensors[name]
        expected_dtype = "F16" if packed else "F32"
        if info.dtype != expected_dtype:
            raise ValueError(
                f"Moonshine weight {name} dtype={info.dtype}, expected {expected_dtype}"
            )
        if info.shape != shape:
            raise ValueError(f"Moonshine weight {name} shape={info.shape}, expected {shape}")
    if spec.lm_head_alias_name in index.tensors:
        raise ValueError("tied LM head must not have a separate stored allocation")


MOONSHINE = register_model(MoonshineForConditionalGenerationModel())

__all__ = [
    "MOONSHINE",
    "MOONSHINE_EMBEDDING_WEIGHT",
    "MOONSHINE_ENCODER_BUCKETS",
    "MOONSHINE_LM_HEAD_ALIAS",
    "MoonshineForConditionalGenerationModel",
    "MoonshineModelSpec",
    "expected_moonshine_weight_shapes",
    "normalize_moonshine_token_ids",
    "parse_moonshine_model_spec",
    "validate_moonshine_weight_index",
]
