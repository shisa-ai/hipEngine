"""Per-tensor repack eligibility (UD-U3 layout selection).

Pure metadata tests on the real pinned UD artifacts: with the per-tensor
default, non-IQ rank-2 tensors regain the T16/x8/planar layouts the
model-wide raw-IQ veto used to strip, while raw-IQ tensors stay raw.
The model-wide behaviour stays available as an explicit rollback.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_policy import resolve_gguf_dense_flags
from hipengine.loading.qwen35_gguf_materialize import (
    plan_qwen35_gguf_materialization,
    gguf_decode_repack_enabled,
)
from hipengine.kernels.backends import load_backend_kernel_package, backend_package_capability

KM_MODEL = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")

pytestmark = pytest.mark.skipif(
    not KM_MODEL.exists(), reason=f"local GGUF fixture not found: {KM_MODEL}"
)

_FAST_LAYOUTS = frozenset({
    "gguf_q4_k_t16_v1", "gguf_q4_k_qmicro_t16_v1", "gguf_q5_k_t16_v1",
    "gguf_q5_k_qmicro_t16_v1", "gguf_q6_k_t16_v1",
    "gguf_q6_k_t16_qmicro_planar_v1", "gguf_q8_0_t16_v1",
    "gguf_q4_k_x8_v1", "gguf_q5_k_x8_v1", "gguf_q6_k_x8_v1",
    "gguf_q5_k_qmicro_planar_v1", "gguf_expert_pack8_v1",
})


def _plan(model_path, repack_veto=None):
    load_backend_kernel_package("hip_gfx1151")
    reader = GGUFReader(model_path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    flags = resolve_gguf_dense_flags(
        "hip_gfx1151", reader.info.file_type_name,
        capability_reader=backend_package_capability,
    )
    return plan_qwen35_gguf_materialization(
        model_map, decode_repack=gguf_decode_repack_enabled(None),
        repack_veto=repack_veto, **flags,
    )


def _layer_specs(plan):
    for layer in plan.layer_specs:
        for slot, spec in layer.items():
            yield slot, spec


def test_km_default_grants_non_iq_repack() -> None:
    """Default plan: rank-2 Q4_K layer tensors regain the T16 layout."""
    plan = _plan(KM_MODEL)
    q4_t16 = [
        s for s, spec in _layer_specs(plan)
        if spec.source.ggml_type_name == "Q4_K" and len(spec.source.shape) == 2
        and spec.layout in _FAST_LAYOUTS
    ]
    assert q4_t16, "expected rank-2 Q4_K layer tensors in optimized layouts"


def test_km_default_keeps_raw_iq_raw() -> None:
    """Raw-IQ tensors keep their raw layouts under the per-tensor default."""
    plan = _plan(KM_MODEL)
    iq = [
        (s, spec.layout) for s, spec in _layer_specs(plan)
        if spec.source.ggml_type_name in ("IQ4_XS", "IQ3_XXS", "IQ2_XS",
                                          "IQ4_NL", "IQ3_S", "IQ2_S")
    ]
    assert iq
    assert all(layout == "raw_gguf" for _, layout in iq), iq[:5]


def test_km_modelwide_rollback_strips_repack() -> None:
    """Explicit model-wide veto reproduces the historical plan (no fast layouts)."""
    plan = _plan(KM_MODEL, repack_veto=True)
    fast = [
        (s, spec.layout) for s, spec in _layer_specs(plan)
        if spec.layout in _FAST_LAYOUTS
    ]
    assert fast == []


def test_km_default_optimized_share() -> None:
    """Default plan puts at least 40% of rank-2 layer weight bytes on fast layouts."""
    plan = _plan(KM_MODEL)
    fast_b = slow_b = 0
    for _, spec in _layer_specs(plan):
        if len(spec.source.shape) != 2:
            continue
        if spec.layout in _FAST_LAYOUTS:
            fast_b += spec.source.nbytes
        else:
            slow_b += spec.source.nbytes
    total = fast_b + slow_b
    assert total > 0
    assert fast_b / total >= 0.40, fast_b / total
