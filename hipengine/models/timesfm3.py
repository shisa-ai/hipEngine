"""Pinned TimesFM 3.0 500M model contract and plugin metadata.

Source of truth: ``google/timesfm-3.0-pytorch`` (safetensors checkpoint, 445
tensors, ~342M tensor parameters, FP32 storage) with the reference
implementation in ``google-research/timesfm`` at
``src/timesfm3/torch/{model,transformer,dense,normalization,util,cpm_revin_refine,configs}.py``
(class ``TimesFM3Torch``).

TimesFM 3.0 is a **non-autoregressive** multivariate patched forecaster:

- Inputs are ``(batch, variates, num_patches, patch_len)`` — variates are the
  target series plus optional past-only and past-future covariates.
- A per-patch ResidualBlock tokenizer consumes, per variate and patch, the
  RevIN-normalised values plus the "future covariate" roll (the next
  ``output_patch_len`` points of every variate, masked at wrap-around and for
  target variates), plus both mask channels:
  ``2 * (input_patch_len + output_patch_len) = 192`` input dims, ReLU
  activation, no biases.
- 20 ``MixingTransformer`` layers: causal sequence attention over the patch
  axis (RoPE, QK RMSNorm, per-dim query scaling, scores multiplied by
  ``sqrt(head_dim)`` — "unscaled" Flax MEA semantics), then **non-causal
  variate attention** across the variate axis (no RoPE), then a ReLU FFN.
  Every sub-block has pre- and post-RMSNorm pairs (torch ``nn.RMSNorm``
  semantics: multiplicative weight, default epsilon).
- Output head: a single ``Linear(1280, 64 * 9)`` with bias; ``output_patch_len``
  64 points at 9 quantiles (median at index 4).
- Decoding is a single forward pass: horizon patches are appended fully
  masked (CPM — cocktail party mask — positions), predictions are stitched
  from overlapping forecast patches, and RevIN statistics at CPM positions are
  refined iteratively from the model's own median forecasts
  (``use_iterative_cpm_revin``). Linear detrending is applied to the context
  (and re-applied to forecasts) when it reduces the residual std below
  ``linear_detrending_threshold`` (0.5) of the original.

Attention masking (correctness-relevant): leading fully-masked patches are
excluded as keys (``cumprod`` effective patch mask); scores use ``-1e9``-style
additive masking so a query row whose keys are all masked attends uniformly
(``1/S``) under the manual path, and the reference's SDPA path yields NaN on
exactly those rows — which are always outside the decoded horizon slice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hipengine.models.registry import register_model

if TYPE_CHECKING:
    from hipengine.loading.safetensors import WeightIndex

PINNED_TIMESFM3_MODEL_ID = "google/timesfm-3.0-pytorch"
TIMESFM3_ARCHITECTURE = "TimesFM3Torch"
TIMESFM3_MODEL_TYPE = "timesfm3"

# Derived constants pinned by the TimesFM3Torch defaults + the checkpoint config.
TIMESFM3_INPUT_PATCH_LEN = 32
TIMESFM3_OUTPUT_PATCH_LEN = 64
TIMESFM3_ROLLS = TIMESFM3_OUTPUT_PATCH_LEN // TIMESFM3_INPUT_PATCH_LEN  # 2
TIMESFM3_NUM_QUANTILES = 9  # 0.1..0.9; median at index 4
TIMESFM3_MEDIAN_INDEX = 4
TIMESFM3_TOKENIZER_INPUT_DIMS = 2 * (
    TIMESFM3_INPUT_PATCH_LEN + TIMESFM3_OUTPUT_PATCH_LEN
)  # 192: (values ++ future-covariate roll) ++ (masks ++ roll masks)
TIMESFM3_MAX_VARIATES = 32
TIMESFM3_VALUE_CLIP = 1.0e20


@dataclass(frozen=True)
class TimesFM3ModelSpec:
    """Validated geometry and storage contract for TimesFM 3.0 500M."""

    model_id: str
    architecture: str
    model_type: str
    stored_dtype: str
    runtime_dtype: str
    model_dims: int
    hidden_dims: int
    num_layers: int
    num_heads: int
    head_dim: int
    input_patch_len: int
    output_patch_len: int
    quantiles: tuple[float, ...]
    use_variate_attention: bool
    use_stitching: bool
    use_linear_detrending: bool
    linear_detrending_threshold: float
    use_iterative_cpm_revin: bool
    use_frozen_running_stats: bool
    value_clip: float
    max_variates: int

    @property
    def rolls(self) -> int:
        return self.output_patch_len // self.input_patch_len

    @property
    def median_index(self) -> int:
        return len(self.quantiles) // 2

    @property
    def tokenizer_input_dims(self) -> int:
        return 2 * (self.input_patch_len + self.output_patch_len)

    @property
    def output_head_dims(self) -> int:
        return self.output_patch_len * len(self.quantiles)

    @property
    def parameter_count(self) -> int:
        return sum(
            math.prod(shape)
            for shape in expected_timesfm3_weight_shapes(self).values()
        )


@dataclass(frozen=True)
class TimesFM3Model:
    """TimesFM 3.0 non-autoregressive multivariate forecaster plugin metadata.

    20 MixingTransformer layers (16 heads x 80 head dims, separate
    query/key/value/output projections, no biases, RoPE + QK RMSNorm +
    per-dim query scaling, sequence scores multiplied by sqrt(head_dim)),
    per-layer non-causal variate attention across up to 32 variates, ReLU
    feed-forward, pre/post RMSNorm pairs around every sub-block, a
    192->1280 ReLU ResidualBlock tokenizer, and a single biased
    Linear(1280, 64*9) output head.
    """

    name: str = "timesfm_3p0"
    architectures: tuple[str, ...] = (TIMESFM3_ARCHITECTURE,)
    default_quant: str = "fp32"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        "pre_transformer_resblock.{param}",
        "transformer_stack.layers.{layer}.{attn}.{param}",
        "transformer_stack.layers.{layer}.ff0.weight",
        "transformer_stack.layers.{layer}.ff1.weight",
        "transformer_stack.layers.{layer}.{norm}.weight",
        "output_head.{param}",
    )

    def layer_sequence(self) -> Sequence[str]:
        """Return a representative decode sequence for registry/fusion planning."""

        return (
            "timesfm3_running_stats",
            "timesfm3_tokenizer",
            *self.transformer_layer_sequence(),
            "timesfm3_output_head",
            "timesfm3_cpm_revin_refine",
            "timesfm3_stitching",
        )

    def transformer_layer_sequence(self) -> tuple[str, ...]:
        """Return primitive layer keys for one MixingTransformer layer."""

        return (
            "rmsnorm",
            "timesfm3_seq_attn_projs",
            "timesfm3_rope",
            "timesfm3_qk_rmsnorm",
            "timesfm3_per_dim_scale",
            "timesfm3_unscaled_attention",
            "timesfm3_out_proj",
            "timesfm3_postnorm_residual",
            "rmsnorm",
            "timesfm3_var_attn",  # non-causal, no RoPE
            "timesfm3_postnorm_residual",
            "rmsnorm",
            "timesfm3_ff_relu",
            "timesfm3_postnorm_residual",
        )


def _positive_int(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"TimesFM 3.0 {name} must be a positive integer")
    return value


def parse_timesfm3_model_spec(config: Mapping[str, Any]) -> TimesFM3ModelSpec:
    """Parse and reject drift from the official TimesFM 3.0 contract.

    ``config`` is the ``config.json`` written by ``TimesFM3Torch.to_dict``.
    """

    expected_top: dict[str, Any] = {
        "input_patch_len": TIMESFM3_INPUT_PATCH_LEN,
        "output_patch_len": TIMESFM3_OUTPUT_PATCH_LEN,
        "input_transform": "identity",
        "use_variate_attention": True,
        "use_stitching": True,
        "use_linear_detrending": True,
        "linear_detrending_threshold": 0.5,
        "use_iterative_cpm_revin": True,
        "use_frozen_running_stats": False,
        "value_clip": 1.0e20,
    }
    for name, expected_value in expected_top.items():
        if config.get(name) != expected_value:
            raise ValueError(
                f"TimesFM 3.0 {name}={config.get(name)!r}, expected {expected_value!r}"
            )

    residual = config.get("residual_block_config")
    if not isinstance(residual, Mapping):
        raise TypeError("TimesFM 3.0 residual_block_config must be a mapping")
    expected_residual: dict[str, Any] = {
        "hidden_dims": 1_280,
        "output_dims": 1_280,
        "use_bias": False,
        "activation": "relu",
        "prenorm": "none",
        "identity_skip": False,
        "dropout": 0.0,
    }
    for name, expected_value in expected_residual.items():
        if residual.get(name) != expected_value:
            raise ValueError(
                f"TimesFM 3.0 residual_block_config.{name}={residual.get(name)!r}, "
                f"expected {expected_value!r}"
            )

    stacked = config.get("transformer_config")
    if not isinstance(stacked, Mapping):
        raise TypeError("TimesFM 3.0 transformer_config must be a mapping")
    transformer = stacked.get("transformer")
    if not isinstance(transformer, Mapping):
        raise TypeError("TimesFM 3.0 transformer_config.transformer must be a mapping")
    expected_transformer: dict[str, Any] = {
        "model_dims": 1_280,
        "hidden_dims": 1_280,
        "num_heads": 16,
        "attention_norm": "rms",
        "feedforward_norm": "rms",
        "qk_norm": "rms",
        "v_norm": "none",
        "use_bias": False,
        "use_rope_seq": True,
        "use_rope_var": False,
        "ff_activation": "relu",
        "causal_attention": True,
        "max_variates": TIMESFM3_MAX_VARIATES,
        "use_memory_efficient_attention": True,
        # The NumPy reference implements SDPA semantics (fully-masked rows ->
        # zeros); the reference's manual path instead attends uniformly there.
        "use_sdpa": True,
    }
    for name, expected_value in expected_transformer.items():
        if transformer.get(name) != expected_value:
            raise ValueError(
                f"TimesFM 3.0 transformer.{name}={transformer.get(name)!r}, "
                f"expected {expected_value!r}"
            )
    num_layers = stacked.get("num_layers")
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers != 20:
        raise ValueError("TimesFM 3.0 pins transformer_config.num_layers to 20")

    quantiles_value = config.get("quantiles")
    if not isinstance(quantiles_value, (list, tuple)) or len(quantiles_value) != 9:
        raise TypeError("TimesFM 3.0 quantiles must be a 9-element array")
    quantiles = tuple(float(quantile) for quantile in quantiles_value)
    if quantiles != (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        raise ValueError("TimesFM 3.0 quantiles differ from the canonical 0.1..0.9 ladder")

    model_dims = int(transformer["model_dims"])
    num_heads = int(transformer["num_heads"])
    if model_dims % num_heads:
        raise ValueError("TimesFM 3.0 model_dims must be divisible by num_heads")
    if int(transformer["hidden_dims"]) != model_dims:
        raise ValueError("TimesFM 3.0 pins hidden_dims to model_dims")

    output_patch_len = _positive_int(config, "output_patch_len")
    input_patch_len = _positive_int(config, "input_patch_len")
    if output_patch_len % input_patch_len:
        raise ValueError(
            "TimesFM 3.0 output_patch_len must be a multiple of input_patch_len"
        )

    return TimesFM3ModelSpec(
        model_id=PINNED_TIMESFM3_MODEL_ID,
        architecture=TIMESFM3_ARCHITECTURE,
        model_type=TIMESFM3_MODEL_TYPE,
        stored_dtype="float32",
        runtime_dtype="float32",
        model_dims=model_dims,
        hidden_dims=int(transformer["hidden_dims"]),
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=model_dims // num_heads,
        input_patch_len=input_patch_len,
        output_patch_len=output_patch_len,
        quantiles=quantiles,
        use_variate_attention=True,
        use_stitching=True,
        use_linear_detrending=True,
        linear_detrending_threshold=float(config["linear_detrending_threshold"]),
        use_iterative_cpm_revin=True,
        use_frozen_running_stats=False,
        value_clip=float(config["value_clip"]),
        max_variates=int(transformer["max_variates"]),
    )


def expected_timesfm3_weight_shapes(
    spec: TimesFM3ModelSpec,
) -> dict[str, tuple[int, ...]]:
    """Exact stored weight manifest for the pinned checkpoint layout."""

    d = spec.model_dims
    hd = spec.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "pre_transformer_resblock.hidden_layer.weight": (d, spec.tokenizer_input_dims),
        "pre_transformer_resblock.output_layer.weight": (d, d),
        "pre_transformer_resblock.residual_layer.weight": (d, spec.tokenizer_input_dims),
        "output_head.weight": (spec.output_head_dims, d),
        "output_head.bias": (spec.output_head_dims,),
    }
    for layer in range(spec.num_layers):
        prefix = f"transformer_stack.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.pre_seq_attn_ln.weight": (d,),
                f"{prefix}.post_seq_attn_ln.weight": (d,),
                f"{prefix}.seq_attn.query_proj.weight": (d, d),
                f"{prefix}.seq_attn.key_proj.weight": (d, d),
                f"{prefix}.seq_attn.value_proj.weight": (d, d),
                f"{prefix}.seq_attn.out_proj.weight": (d, d),
                f"{prefix}.seq_attn.query_ln.weight": (hd,),
                f"{prefix}.seq_attn.key_ln.weight": (hd,),
                f"{prefix}.seq_attn.per_dim_scale.per_dim_scale": (hd,),
                f"{prefix}.pre_var_attn_ln.weight": (d,),
                f"{prefix}.post_var_attn_ln.weight": (d,),
                f"{prefix}.var_attn.query_proj.weight": (d, d),
                f"{prefix}.var_attn.key_proj.weight": (d, d),
                f"{prefix}.var_attn.value_proj.weight": (d, d),
                f"{prefix}.var_attn.out_proj.weight": (d, d),
                f"{prefix}.var_attn.query_ln.weight": (hd,),
                f"{prefix}.var_attn.key_ln.weight": (hd,),
                f"{prefix}.var_attn.per_dim_scale.per_dim_scale": (hd,),
                f"{prefix}.pre_ff_ln.weight": (d,),
                f"{prefix}.post_ff_ln.weight": (d,),
                f"{prefix}.ff0.weight": (d, d),
                f"{prefix}.ff1.weight": (d, d),
            }
        )
    return dict(sorted(shapes.items()))


def validate_timesfm3_weight_index(
    spec: TimesFM3ModelSpec, index: "WeightIndex"
) -> None:
    expected = expected_timesfm3_weight_shapes(spec)
    missing = sorted(set(expected) - set(index.tensors))
    extra = sorted(set(index.tensors) - set(expected))
    if missing or extra:
        raise ValueError(
            f"TimesFM 3.0 weight names differ: missing={missing[:5]}, extra={extra[:5]}"
        )
    for name, shape in expected.items():
        info = index.tensors[name]
        if info.dtype != "F32":
            raise ValueError(f"TimesFM 3.0 weight {name} dtype={info.dtype}, expected F32")
        if info.shape != shape:
            raise ValueError(
                f"TimesFM 3.0 weight {name} shape={info.shape}, expected {shape}"
            )


TIMESFM3 = register_model(TimesFM3Model())


__all__ = [
    "PINNED_TIMESFM3_MODEL_ID",
    "TIMESFM3",
    "TIMESFM3_ARCHITECTURE",
    "TIMESFM3_INPUT_PATCH_LEN",
    "TIMESFM3_MAX_VARIATES",
    "TIMESFM3_MEDIAN_INDEX",
    "TIMESFM3_MODEL_TYPE",
    "TIMESFM3_NUM_QUANTILES",
    "TIMESFM3_OUTPUT_PATCH_LEN",
    "TIMESFM3_ROLLS",
    "TIMESFM3_TOKENIZER_INPUT_DIMS",
    "TIMESFM3_VALUE_CLIP",
    "TimesFM3Model",
    "TimesFM3ModelSpec",
    "expected_timesfm3_weight_shapes",
    "parse_timesfm3_model_spec",
    "validate_timesfm3_weight_index",
]
