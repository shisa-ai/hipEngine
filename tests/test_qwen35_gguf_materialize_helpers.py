from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    Qwen35GGUFMaterializationPlan,
    Qwen35GGUFWeightSpec,
    _gguf_ssm_a_to_kernel_a_log,
    audit_qwen35_gguf_precision_contractions,
)
from hipengine.quant.gguf import GGMLQuantizationType


def test_gguf_ssm_a_materialization_converts_decay_coefficients_to_kernel_log() -> None:
    coeff = np.asarray([-1.0, -0.25, -72.0], dtype=np.float32)
    converted = _gguf_ssm_a_to_kernel_a_log(coeff)

    assert converted.dtype == np.float32
    np.testing.assert_allclose(-np.exp(converted), coeff, rtol=1.0e-6, atol=1.0e-6)

    with pytest.raises(ValueError, match="negative decay coefficients"):
        _gguf_ssm_a_to_kernel_a_log(np.asarray([-1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        _gguf_ssm_a_to_kernel_a_log(np.asarray([-1.0, np.nan], dtype=np.float32))


def test_precision_contraction_audit_ignores_source_f32_tensors_retained_as_f32() -> None:
    plan = Qwen35GGUFMaterializationPlan(
        config=None,  # helper audit does not inspect model config
        root_specs=MappingProxyType(
            {
                "output_norm": _spec(
                    "root.output_norm",
                    "output_norm.weight",
                    GGMLQuantizationType.F32,
                    LAYOUT_DENSE_F32,
                    "f32",
                )
            }
        ),
        layer_specs=(
            MappingProxyType(
                {
                    "ffn_gate_inp": _spec(
                        "layers.0.ffn_gate_inp",
                        "blk.0.ffn_gate_inp.weight",
                        GGMLQuantizationType.F32,
                        LAYOUT_DENSE_F32,
                        "f32",
                    ),
                    "ssm_alpha": _spec(
                        "layers.0.ssm_alpha",
                        "blk.0.ssm_alpha.weight",
                        GGMLQuantizationType.F32,
                        LAYOUT_DENSE_F32,
                        "f32",
                    ),
                    "attn_qkv": _spec(
                        "layers.0.attn_qkv",
                        "blk.0.attn_qkv.weight",
                        GGMLQuantizationType.F16,
                        LAYOUT_DENSE_BF16,
                        "fp16",
                    ),
                }
            ),
        ),
    )

    findings = audit_qwen35_gguf_precision_contractions(plan)

    assert findings == ()


def _spec(
    slot_path: str,
    source_name: str,
    qtype: GGMLQuantizationType,
    layout: str,
    quant_key: str,
) -> Qwen35GGUFWeightSpec:
    return Qwen35GGUFWeightSpec(
        slot_path=slot_path,
        source=_tensor(source_name, qtype),
        quant_key=quant_key,
        layout=layout,
        allocation_names=("raw",),
    )


def _tensor(name: str, qtype: GGMLQuantizationType) -> GGUFTensorInfo:
    return GGUFTensorInfo(
        name=name,
        shape=(2, 3),
        ggml_shape=(3, 2),
        ggml_type=int(qtype),
        ggml_type_name=qtype.name,
        n_elements=6,
        nbytes=24,
        offset=0,
        data_offset=0,
        byte_shape=(2, 3),
    )
