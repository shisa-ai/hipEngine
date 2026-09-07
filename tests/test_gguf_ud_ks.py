"""Published UD Q4_K_S residency integration tests.

Pure metadata/admission tests on the real pinned K_S artifact: no device
allocation, no kernel launch, no torch. Device numerical leaves for raw
IQ2_XS live in ``tests/test_gguf_ud_dense.py``; full-model context evidence
lives in ``benchmarks/results/2026-09-07-zbook-ud-iq2-xs-diagnostic.json``.
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

KS_MODEL = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf")

pytestmark = pytest.mark.skipif(
    not KS_MODEL.exists(), reason=f"local GGUF fixture not found: {KS_MODEL}"
)


def _layer_iq2_xs_specs():
    reader = GGUFReader(KS_MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=False)
    specs = [
        (block_id, slot_path, spec)
        for block_id, block in enumerate(plan.layer_specs)
        for slot_path, spec in block.items()
        if spec.source.ggml_type_name == "IQ2_XS" and len(spec.source.shape) == 2
    ]
    iq2_xs_slots = [
        tensor.name
        for tensor in reader.info.tensors
        if tensor.ggml_type_name == "IQ2_XS" and tensor.name.startswith("blk.")
        and len(tensor.shape) == 2
    ]
    return plan, specs, iq2_xs_slots


def test_ks_layer_iq2_xs_plans_raw() -> None:
    """Every rank-2 layer IQ2_XS slot consumes raw compressed rows."""
    plan, specs, iq2_xs_slots = _layer_iq2_xs_specs()
    # The published K_S file stores exactly one rank-2 layer IQ2_XS tensor
    # (blk.0.ffn_gate.weight); guard the fixture assumption explicitly so a
    # republished file cannot silently shrink this test's scope.
    assert iq2_xs_slots == ["blk.0.ffn_gate.weight"]
    assert [(block_id, slot) for block_id, slot, _ in specs] == [(0, "ffn_gate")]
    for block_id, slot, spec in specs:
        assert spec.layout == LAYOUT_RAW_GGUF, (block_id, slot)
        assert spec.quant_key == "gguf_iq2_xs", slot_path
        assert spec.allocation_names == ("raw",), slot_path


def test_ks_layer_iq2_xs_has_no_dense_bf16_expansion() -> None:
    """No layer IQ2_XS slot may silently expand to a dense-BF16 resident."""
    _, specs, _ = _layer_iq2_xs_specs()
    expanded = [(b, s) for b, s, spec in specs if spec.layout == LAYOUT_DENSE_BF16]
    assert expanded == []


def test_surface_declares_iq2_xs_raw_linear() -> None:
    """The production dispatch surface must route raw IQ2_XS linear slots."""
    assert RAW_LINEAR_SOURCE_QUANT_KEYS["IQ2_XS"] == "gguf_iq2_xs"
    assert set(_RAW_LINEAR_OUTPUTS["IQ2_XS"]) == {"bf16", "f32"}


def test_ks_preflight_accepts_raw_iq2_xs_layers() -> None:
    """AR preflight on the published K_S file reports zero refusals."""
    reader = GGUFReader(KS_MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    report = preflight_qwen35_gguf_artifact(
        model_map, backend="hip_gfx1151", operations=DEFAULT_AR_OPERATIONS
    )
    assert report.supported is True, report.render_refusals()
