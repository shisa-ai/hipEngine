"""Pinned TimesFM 2.5 200M model contract and plugin metadata.

Source of truth: ``google/timesfm-2.5-200m-pytorch`` (safetensors checkpoint)
with the reference implementation in ``google-research/timesfm`` at
``src/timesfm/timesfm_2p5/timesfm_2p5_torch.py`` (module
``TimesFM_2p5_200M_torch_module``, definition ``TimesFM_2p5_200M_Definition``).

The checkpoint is a patched-decoder forecaster: a per-patch ResidualBlock
tokenizer (input patch 32 x 2 features = 64 dims, SwiGLU-free Swish
activation, biased), 20 pre-norm transformer layers with fused QKV, RoPE
(non-interleaved halves), RMSNorm QK-norm, per-dim query scaling, unscaled
dot-product attention, and Swish ff0/ff1 feed-forward, plus two ResidualBlock
output heads (point horizon 128; quantile horizon 1024 x 10 outputs).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hipengine.models.registry import register_model

if TYPE_CHECKING:
    from hipengine.loading.safetensors import WeightIndex

PINNED_TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
TIMESFM_ARCHITECTURE = "TimesFmModelForPrediction"
TIMESFM_MODEL_TYPE = "timesfm"

# Derived constants pinned by TimesFM_2p5_200M_Definition.
TIMESFM_TOKENIZER_INPUT_DIMS = 64  # input_patch_len(32) * 2 (value + mask channels)
TIMESFM_QUANTILE_HEADS = 10  # len(quantiles) + 1 (median duplicated at index 5)
TIMESFM_DECODE_INDEX = 5  # autoregressive channel inside the quantile head


@dataclass(frozen=True)
class TimesFMModelSpec:
    """Validated geometry and storage contract for TimesFM 2.5 200M."""

    model_id: str
    architecture: str
    model_type: str
    stored_dtype: str
    runtime_dtype: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    intermediate_size: int
    context_length: int
    patch_length: int
    horizon_length: int
    quantile_horizon_length: int
    quantiles: tuple[float, ...]
    rms_norm_eps: float
    tokenizer_input_dims: int
    quantile_heads: int
    decode_index: int

    @property
    def qkv_size(self) -> int:
        return 3 * self.hidden_size

    @property
    def quantile_output_dims(self) -> int:
        return self.quantile_horizon_length * self.quantile_heads

    @property
    def tokenizer_input_channels(self) -> int:
        return self.tokenizer_input_dims // self.patch_length

    @property
    def parameter_count(self) -> int:
        return sum(
            math.prod(shape) for shape in expected_timesfm_weight_shapes(self).values()
        )


@dataclass(frozen=True)
class TimesFM25Model:
    """TimesFM 2.5 200M patched-decoder plugin metadata.

    20 layers of MHA (16 heads, head_dim 80, fused QKV, no bias) with RoPE
    applied before RMSNorm QK-norm, per-dim query scaling, and unscaled
    dot-product attention; Swish ff0/ff1 feed-forward with post-norm residuals
    (RMSNorm with a multiplicative ``scale`` parameter, unlike Qwen-style
    RMSNorm which adds 1).
    """

    name: str = "timesfm_2p5_200m"
    architectures: tuple[str, ...] = (TIMESFM_ARCHITECTURE,)
    default_quant: str = "fp32"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        "tokenizer.{block}.{param}",
        "stacked_xf.{layer}.attn.qkv_proj.weight",
        "stacked_xf.{layer}.attn.out.weight",
        "stacked_xf.{layer}.attn.query_ln.scale",
        "stacked_xf.{layer}.attn.key_ln.scale",
        "stacked_xf.{layer}.attn.per_dim_scale.per_dim_scale",
        "stacked_xf.{layer}.ff0.weight",
        "stacked_xf.{layer}.ff1.weight",
        "stacked_xf.{layer}.{norm}.scale",
        "output_projection_point.{block}.weight",
        "output_projection_quantiles.{block}.weight",
    )

    def layer_sequence(self) -> Sequence[str]:
        """Return a representative decode sequence for registry/fusion planning."""

        return (
            "timesfm_tokenizer",
            *self.transformer_layer_sequence(),
            "timesfm_output_projection_point",
            "timesfm_output_projection_quantiles",
        )

    def transformer_layer_sequence(self) -> tuple[str, ...]:
        """Return primitive layer keys for one TimesFM transformer layer."""

        return (
            "rmsnorm",
            "timesfm_qkv_proj",
            "timesfm_rope",
            "timesfm_qk_rmsnorm",
            "timesfm_per_dim_scale",
            "timesfm_unscaled_attention",
            "timesfm_out_proj",
            "timesfm_postnorm_residual",
            "rmsnorm",
            "timesfm_ff_swish",
            "timesfm_postnorm_residual",
        )


def _positive_int(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"TimesFM {name} must be a positive integer")
    return value


def parse_timesfm_model_spec(config: Mapping[str, Any]) -> TimesFMModelSpec:
    """Parse and reject drift from the official TimesFM 2.5 200M contract."""

    expected: dict[str, Any] = {
        "architectures": [TIMESFM_ARCHITECTURE],
        "model_type": TIMESFM_MODEL_TYPE,
        "context_length": 16_384,
        "head_dim": 80,
        "hidden_size": 1_280,
        "horizon_length": 128,
        "intermediate_size": 1_280,
        "num_attention_heads": 16,
        "num_hidden_layers": 20,
        "patch_length": 32,
        "quantile_horizon_length": 1_024,
        "rms_norm_eps": 1.0e-6,
        "torch_compile": False,
    }
    for name, expected_value in expected.items():
        if config.get(name) != expected_value:
            raise ValueError(
                f"TimesFM {name}={config.get(name)!r}, expected {expected_value!r}"
            )

    hidden_size = _positive_int(config, "hidden_size")
    num_heads = _positive_int(config, "num_attention_heads")
    head_dim = _positive_int(config, "head_dim")
    if hidden_size != num_heads * head_dim:
        raise ValueError("TimesFM hidden_size must equal num_attention_heads * head_dim")
    if _positive_int(config, "intermediate_size") != hidden_size:
        raise ValueError("TimesFM 2.5 pins intermediate_size to hidden_size")

    quantiles_value = config.get("quantiles")
    if not isinstance(quantiles_value, (list, tuple)) or len(quantiles_value) != 9:
        raise TypeError("TimesFM quantiles must be a 9-element array")
    quantiles = tuple(float(quantile) for quantile in quantiles_value)
    if quantiles != (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        raise ValueError("TimesFM quantiles differ from the canonical 0.1..0.9 ladder")

    patch_length = _positive_int(config, "patch_length")
    horizon_length = _positive_int(config, "horizon_length")
    quantile_horizon = _positive_int(config, "quantile_horizon_length")
    if horizon_length % patch_length:
        raise ValueError("TimesFM horizon_length must be a multiple of patch_length")
    if quantile_horizon != 8 * horizon_length:
        raise ValueError("TimesFM 2.5 pins quantile_horizon_length to 8 * horizon_length")

    return TimesFMModelSpec(
        model_id=PINNED_TIMESFM_MODEL_ID,
        architecture=TIMESFM_ARCHITECTURE,
        model_type=TIMESFM_MODEL_TYPE,
        stored_dtype="float32",
        runtime_dtype="float32",
        hidden_size=hidden_size,
        num_hidden_layers=_positive_int(config, "num_hidden_layers"),
        num_attention_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=_positive_int(config, "intermediate_size"),
        context_length=_positive_int(config, "context_length"),
        patch_length=patch_length,
        horizon_length=horizon_length,
        quantile_horizon_length=quantile_horizon,
        quantiles=quantiles,
        rms_norm_eps=float(config["rms_norm_eps"]),
        tokenizer_input_dims=TIMESFM_TOKENIZER_INPUT_DIMS,
        quantile_heads=TIMESFM_QUANTILE_HEADS,
        decode_index=TIMESFM_DECODE_INDEX,
    )


def expected_timesfm_weight_shapes(spec: TimesFMModelSpec) -> dict[str, tuple[int, ...]]:
    """Exact stored weight manifest for the pinned checkpoint layout."""

    h = spec.hidden_size
    shapes: dict[str, tuple[int, ...]] = {
        "tokenizer.hidden_layer.weight": (h, spec.tokenizer_input_dims),
        "tokenizer.hidden_layer.bias": (h,),
        "tokenizer.output_layer.weight": (h, h),
        "tokenizer.output_layer.bias": (h,),
        "tokenizer.residual_layer.weight": (h, spec.tokenizer_input_dims),
        "tokenizer.residual_layer.bias": (h,),
    }
    for layer in range(spec.num_hidden_layers):
        prefix = f"stacked_xf.{layer}"
        shapes.update(
            {
                f"{prefix}.attn.qkv_proj.weight": (spec.qkv_size, h),
                f"{prefix}.attn.out.weight": (h, h),
                f"{prefix}.attn.query_ln.scale": (spec.head_dim,),
                f"{prefix}.attn.key_ln.scale": (spec.head_dim,),
                f"{prefix}.attn.per_dim_scale.per_dim_scale": (spec.head_dim,),
                f"{prefix}.ff0.weight": (h, h),
                f"{prefix}.ff1.weight": (h, h),
                f"{prefix}.pre_attn_ln.scale": (h,),
                f"{prefix}.post_attn_ln.scale": (h,),
                f"{prefix}.pre_ff_ln.scale": (h,),
                f"{prefix}.post_ff_ln.scale": (h,),
            }
        )
    shapes.update(
        {
            "output_projection_point.hidden_layer.weight": (h, h),
            "output_projection_point.output_layer.weight": (h, h),
            "output_projection_point.residual_layer.weight": (h, h),
            "output_projection_quantiles.hidden_layer.weight": (h, h),
            "output_projection_quantiles.output_layer.weight": (
                spec.quantile_output_dims,
                h,
            ),
            "output_projection_quantiles.residual_layer.weight": (
                spec.quantile_output_dims,
                h,
            ),
        }
    )
    return dict(sorted(shapes.items()))


def validate_timesfm_weight_index(
    spec: TimesFMModelSpec, index: "WeightIndex"
) -> None:
    expected = expected_timesfm_weight_shapes(spec)
    missing = sorted(set(expected) - set(index.tensors))
    extra = sorted(set(index.tensors) - set(expected))
    if missing or extra:
        raise ValueError(f"TimesFM weight names differ: missing={missing[:5]}, extra={extra[:5]}")
    for name, shape in expected.items():
        info = index.tensors[name]
        if info.dtype != "F32":
            raise ValueError(f"TimesFM weight {name} dtype={info.dtype}, expected F32")
        if info.shape != shape:
            raise ValueError(f"TimesFM weight {name} shape={info.shape}, expected {shape}")


TIMESFM = register_model(TimesFM25Model())


__all__ = [
    "PINNED_TIMESFM_MODEL_ID",
    "TIMESFM",
    "TIMESFM_ARCHITECTURE",
    "TIMESFM_DECODE_INDEX",
    "TIMESFM_MODEL_TYPE",
    "TIMESFM_QUANTILE_HEADS",
    "TIMESFM_TOKENIZER_INPUT_DIMS",
    "TimesFM25Model",
    "TimesFMModelSpec",
    "expected_timesfm_weight_shapes",
    "parse_timesfm_model_spec",
    "validate_timesfm_weight_index",
]
