from __future__ import annotations

# ---------------------------------------------------------------------------
# E7 (UD-GFX1151-OPTIMIZE2 H10): GDN alpha/beta on the fused path.
#
# The plain artifact stores ssm_alpha/ssm_beta as F32 (48x5120) and reaches
# the fused alpha/beta+conv decode owner, which is qualified for `quant_key
# == "f32"` weights in the `raw` allocation. The UD artifact re-quantizes the
# same tensors to Q8_0 (identical shape), so the planner currently lands them
# on the Q8_0 T16 tiled layout with no `raw` allocation and a non-f32
# quant_key: `_try_launch_dense_f32_alpha_beta_conv_decode` declines both
# checks and decode falls back to the unfused per-side path. E7 closes that
# split with a load-time Q8_0 -> F32 expansion, exact in weight value (an
# fp16 scale times an int8 value is representable in F32), keyed strictly on
# stored dtype + slot, never on artifact identity.
# ---------------------------------------------------------------------------

import importlib.util as _ilu  # noqa: E402
from pathlib import Path  # noqa: E402

from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q8_0_T16,
    plan_qwen35_gguf_materialization,
    validate_qwen35_gguf_resident_prerequisites,
)
from hipengine.quant.gguf import (  # noqa: E402
    GGMLQuantizationType,
    dequantize_gguf_data,
    nbytes_for_shape,
    quant_shape_to_byte_shape,
)
import numpy as np  # noqa: E402


def _alpha_map(
    *,
    alpha_beta_type: GGMLQuantizationType = GGMLQuantizationType.Q8_0,
    ffn_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
):
    spec = _ilu.spec_from_file_location(
        "_ud_admission_test_helpers_e7",
        Path(__file__).with_name("test_live_gguf_ud_admission.py"),
    )
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._synthetic_model_map(
        alpha_beta_type=alpha_beta_type, ffn_type=ffn_type
    )


def _layer_slot(plan, slot_suffix: str):
    layer0 = plan.layer_specs[0]
    matches = [spec for name, spec in layer0.items() if name.endswith(slot_suffix)]
    assert len(matches) == 1, sorted(layer0)
    return matches[0]


def test_e7_q8_0_alpha_beta_expand_to_dense_f32_spec() -> None:
    """Q8_0 ssm_alpha/ssm_beta plan as dense F32 with a raw allocation.

    The fused alpha/beta+conv decode owner admits exactly this contract:
    quant_key "f32", layout dense_f32, allocation "raw". Eligibility is keyed
    on the stored dtype plus the slot name, never on the artifact identity.
    """

    plan = plan_qwen35_gguf_materialization(_alpha_map())
    for suffix in ("ssm_alpha", "ssm_beta"):
        spec = _layer_slot(plan, suffix)
        assert spec.quant_key == "f32", (suffix, spec.quant_key)
        assert spec.layout == LAYOUT_DENSE_F32, (suffix, spec.layout)
        assert spec.allocation_names == ("raw",), (suffix, spec.allocation_names)
        # Validation accepts a Q8_0 source for the dense F32 layout because
        # the materializer dequantizes it on the way in.
        validate_qwen35_gguf_resident_prerequisites(spec)


def test_e7_expansion_scope_is_limited_to_alpha_beta_slots() -> None:
    """Other Q8_0 slots keep their existing tiled layout (one lever, no drift)."""

    plan = plan_qwen35_gguf_materialization(
        _alpha_map(ffn_type=GGMLQuantizationType.Q8_0)
    )
    layer0 = plan.layer_specs[0]
    ffn = next(spec for name, spec in layer0.items() if name.endswith("ffn_down"))
    assert ffn.layout == LAYOUT_GGUF_Q8_0_T16, ffn.layout
    alpha = _layer_slot(plan, "ssm_alpha")
    assert alpha.layout == LAYOUT_DENSE_F32, alpha.layout


def test_e7_f32_source_alpha_beta_spec_is_unchanged() -> None:
    """The plain artifact's F32 alpha/beta keeps its existing dense F32 spec."""

    plan = plan_qwen35_gguf_materialization(
        _alpha_map(alpha_beta_type=GGMLQuantizationType.F32)
    )
    for suffix in ("ssm_alpha", "ssm_beta"):
        spec = _layer_slot(plan, suffix)
        assert spec.quant_key == "f32", (suffix, spec.quant_key)
        assert spec.layout == LAYOUT_DENSE_F32, (suffix, spec.layout)
        validate_qwen35_gguf_resident_prerequisites(spec)


def test_e7_q8_0_expansion_is_bitwise_exact_in_f32() -> None:
    """The expansion oracle: dequant(Q8_0) == scale_f32 * q_f32, bitwise.

    Q8_0 stores an fp16 scale plus int8 values per 32-element block. Both
    factors are exactly representable in F32 and the product fits the F32
    mantissa, so the dequantization the materializer performs introduces no
    rounding: every expanded value equals the F32 evaluation of the stored
    product.
    """

    rng = np.random.default_rng(0xE7A1)
    shape = (16, 256)
    byte_shape = quant_shape_to_byte_shape(shape, GGMLQuantizationType.Q8_0)
    blocks_per_row = shape[-1] // 32
    n_blocks = shape[0] * blocks_per_row

    scales = np.asarray(
        rng.uniform(-0.02, 0.02, size=n_blocks), dtype=np.float16
    )
    qs = rng.integers(-127, 128, size=n_blocks * 32, dtype=np.int8)
    # GGUF Q8_0 row layout: per 32-element block, fp16 scale then 32 int8.
    scale_bytes = scales.reshape(shape[0], blocks_per_row).view(np.uint8).reshape(
        shape[0], blocks_per_row, 2
    )
    q_bytes = qs.reshape(shape[0], blocks_per_row, 32).view(np.uint8)
    block_bytes = np.concatenate([scale_bytes, q_bytes], axis=-1)
    data = block_bytes.reshape(byte_shape)

    expanded = dequantize_gguf_data(data, GGMLQuantizationType.Q8_0)
    expected = (
        scales.reshape(shape[0], blocks_per_row, 1).astype(np.float32)
        * qs.reshape(shape[0], blocks_per_row, 32).astype(np.float32)
    ).reshape(shape)
    assert expanded.dtype == np.float32
    np.testing.assert_array_equal(expanded, expected)


def test_e7_expanded_planned_bytes_are_f32_element_bytes() -> None:
    """Planning sizes the resident by F32 elements, not Q8_0 source bytes."""

    plan = plan_qwen35_gguf_materialization(_alpha_map())
    alpha = _layer_slot(plan, "ssm_alpha")
    n_elements = 1
    for dim in alpha.source.shape:
        n_elements *= int(dim)
    assert n_elements * 4 == int(alpha.source.n_elements) * 4
    assert int(alpha.source.nbytes) == nbytes_for_shape(
        tuple(int(d) for d in alpha.source.shape), GGMLQuantizationType.Q8_0
    )
    assert int(alpha.source.nbytes) != n_elements * 4  # source is compressed