"""Published UD Q4_K_M raw Q5/Q6 rank-2 residency integration tests.

Pure metadata/admission tests on the real pinned K_M artifact: no device
allocation, no kernel launch, no torch. Device numerical leaves for raw
Q5/Q6 live in ``tests/test_gguf_ud_q56_roles.py``; full-model context
evidence lives in the UD campaign artifacts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_admission import (
    DEFAULT_AR_OPERATIONS,
    preflight_qwen35_gguf_artifact,
)
from hipengine.loading.qwen35_gguf_consumer_surface import (
    _RAW_LINEAR_OUTPUTS,
    RAW_LINEAR_SOURCE_QUANT_KEYS,
)
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_RAW_GGUF,
    plan_qwen35_gguf_materialization,
)

KM_MODEL = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")

pytestmark = pytest.mark.skipif(
    not KM_MODEL.exists(), reason=f"local GGUF fixture not found: {KM_MODEL}"
)


def _layer_q56_rank2_specs():
    reader = GGUFReader(KM_MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=False)
    specs = [
        (block_id, slot_path, spec)
        for block_id, block in enumerate(plan.layer_specs)
        for slot_path, spec in block.items()
        if spec.source.ggml_type_name in ("Q5_K", "Q6_K") and len(spec.source.shape) == 2
    ]
    return plan, specs


def test_km_layer_q56_rank2_plans_raw() -> None:
    """Every rank-2 layer Q5_K/Q6_K slot consumes raw compressed rows."""
    _, specs = _layer_q56_rank2_specs()
    assert specs, "expected rank-2 layer Q5_K/Q6_K tensors in the K_M file"
    for block_id, slot, spec in specs:
        assert spec.layout == LAYOUT_RAW_GGUF, (block_id, slot, spec.layout)
        assert spec.quant_key == f"gguf_{spec.source.ggml_type_name.lower()}", slot
        assert spec.allocation_names == ("raw",), slot


def test_km_layer_q56_rank2_has_no_dense_bf16_expansion() -> None:
    """No rank-2 layer Q5_K/Q6_K slot may silently expand dense-BF16."""
    _, specs = _layer_q56_rank2_specs()
    expanded = [(b, s) for b, s, spec in specs if spec.layout == LAYOUT_DENSE_BF16]
    assert expanded == []


def test_surface_declares_q5_k_raw_linear() -> None:
    """The production dispatch surface must route raw Q5_K linear slots."""
    assert RAW_LINEAR_SOURCE_QUANT_KEYS["Q5_K"] == "gguf_q5_k"
    assert set(_RAW_LINEAR_OUTPUTS["Q5_K"]) == {"bf16", "f32"}


def test_km_preflight_accepts_raw_q56_layers() -> None:
    """AR preflight on the published K_M file reports zero refusals."""
    reader = GGUFReader(KM_MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    report = preflight_qwen35_gguf_artifact(
        model_map, backend="hip_gfx1151", operations=DEFAULT_AR_OPERATIONS
    )
    assert report.supported is True, report.render_refusals()
