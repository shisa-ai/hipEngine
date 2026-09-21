"""Torch-free YuE2 checkpoint loader: validated inventory, owned weight handles.

Reads the released safetensors directly. Weights stay in their stored BF16 bit
patterns (``uint16`` host arrays) so the device upload is a byte copy and the
host footprint is the checkpoint size rather than twice it; the VAE decoder is
FP32 with folded weight normalization, as the released decoder requires.

Every mapping is validated: unknown or missing tensors, wrong dtypes and wrong
shapes fail before any allocation is handed to a session.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hipengine.loading.safetensors import (
    TensorInfo,
    load_weight_index,
    read_tensor_storage_bytes,
)

#: Pinned checkpoint identities from docs/model-cards/MODEL-YUE2.md.
PINNED_MODEL_REVISION = "29b3558dd46954a0cd9021dc76d5c91864a0f1c7"
PINNED_MODEL_SHA256 = "1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9"
PINNED_VAE_REVISION = "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"

PINNED_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 6144,
    "vocab_size": 184704,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 24576,
    "latent_type": "vae",
    "latent_dim": 64,
    "max_latent_frames": 24576,
    "timestep_shift": 1.0,
}


class YuE2WeightError(ValueError):
    """Raised when a checkpoint does not match the pinned YuE2 inventory."""


@dataclass(frozen=True)
class YuE2Config:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 184704
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 24576
    latent_type: str = "vae"
    latent_dim: int = 64
    max_latent_frames: int = 24576
    timestep_shift: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "YuE2Config":
        unknown = set(data) - set(PINNED_CONFIG) - {
            "model_type",
            "architectures",
            "auto_map",
            "dtype",
            "torch_dtype",
            "transformers_version",
            "return_dict",
            "output_hidden_states",
            "output_attentions",
            "torchscript",
            "tie_word_embeddings",
            "is_encoder_decoder",
            "is_decoder",
            "add_cross_attention",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "decoder_start_token_id",
            "use_cache",
        }
        if unknown:
            raise YuE2WeightError(f"unexpected config keys: {sorted(unknown)}")
        config = cls(**{key: data[key] for key in PINNED_CONFIG if key in data})
        config.validate()
        return config

    def validate(self) -> None:
        for key, expected in PINNED_CONFIG.items():
            actual = getattr(self, key)
            if isinstance(expected, float):
                if abs(float(actual) - expected) > 1e-12:
                    raise YuE2WeightError(f"config {key}={actual!r} != pinned {expected!r}")
            elif actual != expected:
                raise YuE2WeightError(f"config {key}={actual!r} != pinned {expected!r}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise YuE2WeightError("attention heads must be a multiple of the KV heads")

    @property
    def q_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in PINNED_CONFIG}


# ---------------------------------------------------------------------------
# transformer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YuE2AttentionWeights:
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    o: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray


@dataclass(frozen=True)
class YuE2MlpWeights:
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray


@dataclass(frozen=True)
class YuE2LayerWeights:
    input_layernorm: np.ndarray
    self_attn: YuE2AttentionWeights
    post_attention_layernorm: np.ndarray
    mlp: YuE2MlpWeights
    nar_input_layernorm: np.ndarray
    nar_self_attn: YuE2AttentionWeights
    nar_pre_mlp_layernorm: np.ndarray
    nar_mlp: YuE2MlpWeights


@dataclass(frozen=True)
class YuE2TimeEmbedderWeights:
    first_weight: np.ndarray
    first_bias: np.ndarray
    second_weight: np.ndarray
    second_bias: np.ndarray


@dataclass
class YuE2Weights:
    """Owned host weights for the YuE2 mixture-of-transformers checkpoint."""

    config: YuE2Config
    embed_tokens: np.ndarray
    layers: list[YuE2LayerWeights]
    norm: np.ndarray
    lm_head: np.ndarray
    llm2vae_weight: np.ndarray
    llm2vae_bias: np.ndarray
    vae2llm_weight: np.ndarray
    vae2llm_bias: np.ndarray
    time_embedder: YuE2TimeEmbedderWeights
    latent_pos_embed: np.ndarray
    identity: dict = field(default_factory=dict)

    @property
    def bytes(self) -> int:
        total = self.embed_tokens.nbytes + self.norm.nbytes + self.lm_head.nbytes
        total += self.llm2vae_weight.nbytes + self.llm2vae_bias.nbytes
        total += self.vae2llm_weight.nbytes + self.vae2llm_bias.nbytes
        total += self.latent_pos_embed.nbytes
        total += (
            self.time_embedder.first_weight.nbytes
            + self.time_embedder.first_bias.nbytes
            + self.time_embedder.second_weight.nbytes
            + self.time_embedder.second_bias.nbytes
        )
        for layer in self.layers:
            for array in (
                layer.input_layernorm,
                layer.self_attn.q,
                layer.self_attn.k,
                layer.self_attn.v,
                layer.self_attn.o,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
                layer.post_attention_layernorm,
                layer.mlp.gate,
                layer.mlp.up,
                layer.mlp.down,
                layer.nar_input_layernorm,
                layer.nar_self_attn.q,
                layer.nar_self_attn.k,
                layer.nar_self_attn.v,
                layer.nar_self_attn.o,
                layer.nar_self_attn.q_norm,
                layer.nar_self_attn.k_norm,
                layer.nar_pre_mlp_layernorm,
                layer.nar_mlp.gate,
                layer.nar_mlp.up,
                layer.nar_mlp.down,
            ):
                total += array.nbytes
        return total


def _read_bf16_bits(index, name: str, shape: tuple[int, ...]) -> np.ndarray:
    info: TensorInfo = index.tensors[name]
    if info.dtype != "BF16":
        raise YuE2WeightError(f"tensor {name!r} dtype {info.dtype} != BF16")
    if info.shape != shape:
        raise YuE2WeightError(f"tensor {name!r} shape {info.shape} != {shape}")
    payload = read_tensor_storage_bytes(info)
    return np.frombuffer(payload, dtype=np.uint16).reshape(shape).copy()


def _attention_names(prefix: str) -> dict[str, str]:
    return {
        "q": f"{prefix}.q_proj.weight",
        "k": f"{prefix}.k_proj.weight",
        "v": f"{prefix}.v_proj.weight",
        "o": f"{prefix}.o_proj.weight",
        "q_norm": f"{prefix}.q_norm.weight",
        "k_norm": f"{prefix}.k_norm.weight",
    }


def _mlp_names(prefix: str) -> dict[str, str]:
    return {
        "gate": f"{prefix}.gate_proj.weight",
        "up": f"{prefix}.up_proj.weight",
        "down": f"{prefix}.down_proj.weight",
    }


def expected_transformer_tensors(config: YuE2Config) -> set[str]:
    names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "llm2vae.weight",
        "llm2vae.bias",
        "vae2llm.weight",
        "vae2llm.bias",
        "latent_pos_embed.pe",
        "time_embedder.mlp.0.weight",
        "time_embedder.mlp.0.bias",
        "time_embedder.mlp.2.weight",
        "time_embedder.mlp.2.bias",
    }
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        names.add(f"{prefix}.input_layernorm.weight")
        names.add(f"{prefix}.post_attention_layernorm.weight")
        names.add(f"{prefix}.nar_input_layernorm.weight")
        names.add(f"{prefix}.nar_pre_mlp_layernorm.weight")
        names.update(_attention_names(f"{prefix}.self_attn").values())
        names.update(_attention_names(f"{prefix}.nar_self_attn").values())
        names.update(_mlp_names(f"{prefix}.mlp").values())
        names.update(_mlp_names(f"{prefix}.nar_mlp").values())
    return names


def load_yue2_weights(model_dir: str | Path, *, validate_names: bool = True) -> YuE2Weights:
    """Load the YuE2-3B mixture-of-transformers checkpoint into host BF16 arrays."""
    directory = Path(model_dir)
    index = load_weight_index(directory)
    config = YuE2Config.from_dict(index.config)
    if validate_names:
        expected = expected_transformer_tensors(config)
        present = set(index.tensors)
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing or extra:
            raise YuE2WeightError(
                f"checkpoint inventory mismatch: missing={missing[:6]} extra={extra[:6]}"
            )
    hidden = config.hidden_size
    kv_width = config.kv_width
    layers = []
    for layer_index in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_index}"

        def attention(scope: str) -> YuE2AttentionWeights:
            names = _attention_names(f"{prefix}.{scope}")
            return YuE2AttentionWeights(
                q=_read_bf16_bits(index, names["q"], (config.q_width, hidden)),
                k=_read_bf16_bits(index, names["k"], (kv_width, hidden)),
                v=_read_bf16_bits(index, names["v"], (kv_width, hidden)),
                o=_read_bf16_bits(index, names["o"], (hidden, config.q_width)),
                q_norm=_read_bf16_bits(index, names["q_norm"], (config.head_dim,)),
                k_norm=_read_bf16_bits(index, names["k_norm"], (config.head_dim,)),
            )

        def mlp(scope: str) -> YuE2MlpWeights:
            names = _mlp_names(f"{prefix}.{scope}")
            return YuE2MlpWeights(
                gate=_read_bf16_bits(index, names["gate"], (config.intermediate_size, hidden)),
                up=_read_bf16_bits(index, names["up"], (config.intermediate_size, hidden)),
                down=_read_bf16_bits(index, names["down"], (hidden, config.intermediate_size)),
            )

        layers.append(
            YuE2LayerWeights(
                input_layernorm=_read_bf16_bits(index, f"{prefix}.input_layernorm.weight", (hidden,)),
                self_attn=attention("self_attn"),
                post_attention_layernorm=_read_bf16_bits(
                    index, f"{prefix}.post_attention_layernorm.weight", (hidden,)
                ),
                mlp=mlp("mlp"),
                nar_input_layernorm=_read_bf16_bits(
                    index, f"{prefix}.nar_input_layernorm.weight", (hidden,)
                ),
                nar_self_attn=attention("nar_self_attn"),
                nar_pre_mlp_layernorm=_read_bf16_bits(
                    index, f"{prefix}.nar_pre_mlp_layernorm.weight", (hidden,)
                ),
                nar_mlp=mlp("nar_mlp"),
            )
        )

    weights = YuE2Weights(
        config=config,
        embed_tokens=_read_bf16_bits(
            index, "model.embed_tokens.weight", (config.vocab_size, hidden)
        ),
        layers=layers,
        norm=_read_bf16_bits(index, "model.norm.weight", (hidden,)),
        lm_head=_read_bf16_bits(index, "lm_head.weight", (config.vocab_size, hidden)),
        llm2vae_weight=_read_bf16_bits(index, "llm2vae.weight", (config.latent_dim, hidden)),
        llm2vae_bias=_read_bf16_bits(index, "llm2vae.bias", (config.latent_dim,)),
        vae2llm_weight=_read_bf16_bits(index, "vae2llm.weight", (hidden, config.latent_dim)),
        vae2llm_bias=_read_bf16_bits(index, "vae2llm.bias", (hidden,)),
        time_embedder=YuE2TimeEmbedderWeights(
            first_weight=_read_bf16_bits(index, "time_embedder.mlp.0.weight", (hidden, 256)),
            first_bias=_read_bf16_bits(index, "time_embedder.mlp.0.bias", (hidden,)),
            second_weight=_read_bf16_bits(index, "time_embedder.mlp.2.weight", (hidden, hidden)),
            second_bias=_read_bf16_bits(index, "time_embedder.mlp.2.bias", (hidden,)),
        ),
        latent_pos_embed=_read_bf16_bits(
            index, "latent_pos_embed.pe", (config.max_latent_frames, hidden)
        ),
    )
    weights.identity = checkpoint_identity(directory, index)
    return weights


# ---------------------------------------------------------------------------
# VAE decoder
# ---------------------------------------------------------------------------

VAE_DECODER_STRIDES = (2, 2, 4, 4, 5, 6)
VAE_CHANNELS = 64
VAE_CHANNEL_MULTS = (1, 2, 4, 8, 16, 32)
VAE_LATENT_DIM = 64
VAE_SAMPLE_RATE = 48000
VAE_DOWNSAMPLING_RATIO = 1920


@dataclass(frozen=True)
class YuE2ConvWeights:
    """Folded FP32 convolution weights."""

    weight: np.ndarray
    bias: np.ndarray | None
    stride: int = 1
    dilation: int = 1
    padding: int = 0
    transposed: bool = False

    def output_length(self, length: int) -> int:
        kernel = self.weight.shape[-1]
        if self.transposed:
            return (length - 1) * self.stride - 2 * self.padding + kernel
        return (length + 2 * self.padding - self.dilation * (kernel - 1) - 1) // self.stride + 1


@dataclass(frozen=True)
class YuE2SnakeWeights:
    alpha: np.ndarray
    beta: np.ndarray


@dataclass(frozen=True)
class YuE2ResidualUnitWeights:
    activation_in: YuE2SnakeWeights
    conv: YuE2ConvWeights
    activation_out: YuE2SnakeWeights
    pointwise: YuE2ConvWeights


@dataclass(frozen=True)
class YuE2DecoderBlockWeights:
    activation: YuE2SnakeWeights
    upsample: YuE2ConvWeights
    residual_units: tuple[YuE2ResidualUnitWeights, ...]


@dataclass(frozen=True)
class YuE2VaeDecoderWeights:
    input_conv: YuE2ConvWeights
    blocks: tuple[YuE2DecoderBlockWeights, ...]
    output_activation: YuE2SnakeWeights
    output_conv: YuE2ConvWeights
    latent_dim: int = VAE_LATENT_DIM
    sample_rate: int = VAE_SAMPLE_RATE
    downsampling_ratio: int = VAE_DOWNSAMPLING_RATIO
    release_variant: str = "standard"
    identity: dict = field(default_factory=dict)

    @property
    def bytes(self) -> int:
        total = 0
        for conv in self.all_convs():
            total += conv.weight.nbytes + (0 if conv.bias is None else conv.bias.nbytes)
        for snake in self.all_snakes():
            total += snake.alpha.nbytes + snake.beta.nbytes
        return total

    def all_convs(self) -> list[YuE2ConvWeights]:
        convs = [self.input_conv, self.output_conv]
        for block in self.blocks:
            convs.append(block.upsample)
            for unit in block.residual_units:
                convs.extend((unit.conv, unit.pointwise))
        return convs

    def all_snakes(self) -> list[YuE2SnakeWeights]:
        snakes = [self.output_activation]
        for block in self.blocks:
            snakes.append(block.activation)
            for unit in block.residual_units:
                snakes.extend((unit.activation_in, unit.activation_out))
        return snakes

    def natural_output_length(self, frames: int) -> int:
        if frames < 1:
            raise ValueError("frames must be positive")
        length = frames
        for conv in self.forward_convs():
            length = conv.output_length(length)
        return length

    def forward_convs(self) -> list[YuE2ConvWeights]:
        convs = [self.input_conv]
        for block in self.blocks:
            convs.append(block.upsample)
            for unit in block.residual_units:
                convs.extend((unit.conv, unit.pointwise))
        convs.append(self.output_conv)
        return convs

    def required_halo(self, core_frames: int = 1024) -> int:
        """Frames of context each side of a core needs, from the dependency interval.

        Mirrors the reference's own rule: the output interval ``[0, core * ratio)
        is walked backwards through the decoder, and the halo is the largest
        amount by which an output position can depend on input frames outside the
        core. Residual units keep their identity path, so their interval is the
        union of the walked layers and the incoming interval.
        """

        core = int(core_frames)
        if core < 1:
            raise ValueError("core_frames must be positive")

        def walk(conv: YuE2ConvWeights, low: int, high: int) -> tuple[int, int]:
            kernel = conv.weight.shape[-1]
            if conv.transposed:
                return (
                    -(-(low + conv.padding - conv.dilation * (kernel - 1)) // conv.stride),
                    (high + conv.padding) // conv.stride,
                )
            return (
                low * conv.stride - conv.padding,
                high * conv.stride - conv.padding + conv.dilation * (kernel - 1),
            )

        low, high = 0, core * self.downsampling_ratio - 1
        low, high = walk(self.output_conv, low, high)
        for block in reversed(self.blocks):
            for unit in reversed(block.residual_units):
                unit_low, unit_high = walk(unit.pointwise, low, high)
                unit_low, unit_high = walk(unit.conv, unit_low, unit_high)
                low, high = min(unit_low, low), max(unit_high, high)
            low, high = walk(block.upsample, low, high)
        low, high = walk(self.input_conv, low, high)
        return max(0, -low, high - core + 1)


def _read_f32(index, name: str, shape: tuple[int, ...]) -> np.ndarray:
    info: TensorInfo = index.tensors[name]
    if info.dtype != "F32":
        raise YuE2WeightError(f"tensor {name!r} dtype {info.dtype} != F32")
    if info.shape != shape:
        raise YuE2WeightError(f"tensor {name!r} shape {info.shape} != {shape}")
    payload = read_tensor_storage_bytes(info)
    return np.frombuffer(payload, dtype=np.float32).reshape(shape).copy()


def fold_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """Fold ``weight_norm`` into a single FP32 kernel (validated arithmetic)."""
    if weight_g.shape != (weight_v.shape[0], 1, 1):
        raise YuE2WeightError(f"weight_g {weight_g.shape} does not match weight_v {weight_v.shape}")
    norm = np.sqrt(np.sum(np.square(weight_v, dtype=np.float64), axis=(1, 2), keepdims=True))
    if not np.all(norm > 0):
        raise YuE2WeightError("weight_v has a zero norm")
    return ((weight_g / norm) * weight_v).astype(np.float32)


def _read_conv(index, name: str, *, transposed: bool, stride: int, padding: int, dilation: int = 1):
    """Read a weight-normalized Conv1d/ConvTranspose1d and fold it once."""
    weight_g = _read_f32(index, f"{name}.weight_g", _shape_of(index, f"{name}.weight_g"))
    weight_v = _read_f32(index, f"{name}.weight_v", _shape_of(index, f"{name}.weight_v"))
    bias = (
        _read_f32(index, f"{name}.bias", _shape_of(index, f"{name}.bias"))
        if f"{name}.bias" in index.tensors
        else None
    )
    return YuE2ConvWeights(
        weight=fold_weight_norm(weight_g, weight_v),
        bias=bias,
        stride=stride,
        dilation=dilation,
        padding=padding,
        transposed=transposed,
    )


def _shape_of(index, name: str) -> tuple[int, ...]:
    if name not in index.tensors:
        raise YuE2WeightError(f"missing tensor {name!r}")
    return index.tensors[name].shape


def _read_snake(index, name: str) -> YuE2SnakeWeights:
    return YuE2SnakeWeights(
        alpha=_read_f32(index, f"{name}.alpha", _shape_of(index, f"{name}.alpha")),
        beta=_read_f32(index, f"{name}.beta", _shape_of(index, f"{name}.beta")),
    )


def expected_vae_decoder_tensors() -> set[str]:
    names = {
        "decoder.layers.0.weight_g",
        "decoder.layers.0.weight_v",
        "decoder.layers.0.bias",
    }
    for block_index in range(1, len(VAE_DECODER_STRIDES) + 1):
        prefix = f"decoder.layers.{block_index}"
        names.add(f"{prefix}.layers.0.alpha")
        names.add(f"{prefix}.layers.0.beta")
        names.add(f"{prefix}.layers.1.weight_g")
        names.add(f"{prefix}.layers.1.weight_v")
        names.add(f"{prefix}.layers.1.bias")
        for unit in (2, 3, 4):
            unit_prefix = f"{prefix}.layers.{unit}"
            for inner in (0, 2):
                names.add(f"{unit_prefix}.layers.{inner}.alpha")
                names.add(f"{unit_prefix}.layers.{inner}.beta")
            for inner in (1, 3):
                names.add(f"{unit_prefix}.layers.{inner}.weight_g")
                names.add(f"{unit_prefix}.layers.{inner}.weight_v")
                names.add(f"{unit_prefix}.layers.{inner}.bias")
    names.add("decoder.layers.7.alpha")
    names.add("decoder.layers.7.beta")
    names.add("decoder.layers.8.weight_g")
    names.add("decoder.layers.8.weight_v")
    return names


def load_yue2_vae_decoder(vae_dir: str | Path, *, validate_names: bool = True) -> YuE2VaeDecoderWeights:
    """Load and fold the FP32 YuE2-Vae decoder (encoder tensors are not needed)."""
    directory = Path(vae_dir)
    index = load_weight_index(directory)
    config = json.loads((directory / "config.json").read_text())
    decoder_config = config.get("decoder_config", {})
    if decoder_config.get("latent_dim", VAE_LATENT_DIM) != VAE_LATENT_DIM:
        raise YuE2WeightError("decoder latent_dim is not 64")
    if tuple(decoder_config.get("strides", VAE_DECODER_STRIDES)) != VAE_DECODER_STRIDES:
        raise YuE2WeightError("decoder strides are not the pinned [2,2,4,4,5,6]")
    if int(config.get("downsampling_ratio", VAE_DOWNSAMPLING_RATIO)) != VAE_DOWNSAMPLING_RATIO:
        raise YuE2WeightError("downsampling_ratio is not 1920")
    if validate_names:
        expected = expected_vae_decoder_tensors()
        present = {name for name in index.tensors if name.startswith("decoder.")}
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing or extra:
            raise YuE2WeightError(
                f"VAE decoder inventory mismatch: missing={missing[:6]} extra={extra[:6]}"
            )

    input_conv = _read_conv(
        index, "decoder.layers.0", transposed=False, stride=1, padding=3
    )
    blocks = []
    # The released decoder walks the stride list in reverse: layer 1 is the
    # 6x stage (2048 -> 1024 channels) and layer 6 is the last 2x stage.
    for block_index in range(1, len(VAE_DECODER_STRIDES) + 1):
        stride = VAE_DECODER_STRIDES[len(VAE_DECODER_STRIDES) - block_index]
        prefix = f"decoder.layers.{block_index}"
        activation = _read_snake(index, f"{prefix}.layers.0")
        upsample = _read_conv(
            index,
            f"{prefix}.layers.1",
            transposed=True,
            stride=stride,
            padding=(stride + 1) // 2,
        )
        units = []
        for unit, dilation in ((2, 1), (3, 3), (4, 9)):
            unit_prefix = f"{prefix}.layers.{unit}"
            units.append(
                YuE2ResidualUnitWeights(
                    activation_in=_read_snake(index, f"{unit_prefix}.layers.0"),
                    conv=_read_conv(
                        index,
                        f"{unit_prefix}.layers.1",
                        transposed=False,
                        stride=1,
                        padding=(dilation * 6) // 2,
                        dilation=dilation,
                    ),
                    activation_out=_read_snake(index, f"{unit_prefix}.layers.2"),
                    pointwise=_read_conv(
                        index, f"{unit_prefix}.layers.3", transposed=False, stride=1, padding=0
                    ),
                )
            )
        blocks.append(
            YuE2DecoderBlockWeights(
                activation=activation, upsample=upsample, residual_units=tuple(units)
            )
        )
    decoder = YuE2VaeDecoderWeights(
        input_conv=input_conv,
        blocks=tuple(blocks),
        output_activation=_read_snake(index, "decoder.layers.7"),
        output_conv=_read_conv(
            index, "decoder.layers.8", transposed=False, stride=1, padding=3
        ),
        release_variant=str(config.get("release_variant", "standard")),
        identity=checkpoint_identity(directory, index),
    )
    if decoder.output_conv.bias is not None:
        raise YuE2WeightError("the released output convolution has no bias")
    return decoder


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(directory: Path, index=None, *, hash_weights: bool = False) -> dict:
    """Content identity of a checkpoint directory (hash weights only on request)."""
    index = index if index is not None else load_weight_index(directory)
    files = {}
    for name in ("config.json", "qwen.tiktoken", "weights_manifest.json"):
        path = directory / name
        if path.is_file():
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    shards = []
    for shard in index.shards:
        entry = {"name": shard.name, "bytes": shard.stat().st_size}
        if hash_weights:
            entry["sha256"] = sha256_file(shard)
        shards.append(entry)
    return {
        "path": str(directory.resolve()),
        "files": files,
        "shards": shards,
        "tensor_count": len(index.tensors),
    }
