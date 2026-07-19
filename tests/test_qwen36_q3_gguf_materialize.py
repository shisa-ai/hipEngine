from __future__ import annotations

from collections import Counter
import ctypes
from pathlib import Path

import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_RAW_GGUF,
    materialize_qwen35_gguf_weights,
    plan_qwen35_gguf_materialization,
)

Q3_MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")
pytestmark = pytest.mark.skipif(
    not Q3_MOE_MODEL.exists(), reason=f"local GGUF fixture not found: {Q3_MOE_MODEL}"
)


@pytest.mark.parametrize("decode_repack", [False, True])
def test_qwen36_q3_materialization_plan_preserves_iq_experts(
    decode_repack: bool,
) -> None:
    reader = GGUFReader(Q3_MOE_MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)

    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=decode_repack)

    assert set(plan.tensor_names) == set(model_map.tensor_names)
    assert len(plan.tensor_names) == 733
    assert plan.config.block_count == 40
    assert plan.config.declared_block_count == 41
    assert plan.config.ignored_block_ids == (40,)
    assert not any(spec.source.name.startswith("blk.40.") for spec in plan.specs)

    iq3_specs = [spec for spec in plan.specs if spec.source.ggml_type_name == "IQ3_XXS"]
    iq4_specs = [spec for spec in plan.specs if spec.source.ggml_type_name == "IQ4_XS"]
    assert Counter(spec.slot_path.rsplit(".", 1)[-1] for spec in iq3_specs) == {
        "ffn_gate_exps": 39,
        "ffn_up_exps": 39,
    }
    assert Counter(spec.slot_path.rsplit(".", 1)[-1] for spec in iq4_specs) == {
        "ffn_gate_exps": 1,
        "ffn_up_exps": 1,
        "ffn_down_exps": 37,
    }
    for spec in iq3_specs:
        assert spec.quant_key == "gguf_iq3_xxs"
        assert spec.layout == LAYOUT_RAW_GGUF
        assert spec.allocation_names == ("raw",)
    for spec in iq4_specs:
        assert spec.quant_key == "gguf_iq4_xs"
        assert spec.layout == LAYOUT_RAW_GGUF
        assert spec.allocation_names == ("raw",)


def test_qwen36_q3_materializes_compressed_iq_expert_records() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    runtime = get_hip_runtime()
    reader = GGUFReader(Q3_MOE_MODEL)
    selected = {
        "layers.0.ffn_gate_exps": "blk.0.ffn_gate_exps.weight",
        "layers.0.ffn_down_exps": "blk.0.ffn_down_exps.weight",
    }
    resident = materialize_qwen35_gguf_weights(
        reader,
        selected_slots=selected,
        decode_repack=True,
        runtime=runtime,
    )
    try:
        for slot_path, tensor_name in selected.items():
            _, layer_id, slot = slot_path.split(".")
            source = reader.tensor_info(tensor_name)
            weight = resident.layer(int(layer_id)).weight(slot)
            allocation = weight.allocation()
            assert weight.spec.layout == LAYOUT_RAW_GGUF
            assert weight.spec.quant_key == f"gguf_{source.ggml_type_name.lower()}"
            assert allocation.tensor.dtype == DType.INT8
            assert allocation.tensor.shape == source.byte_shape
            assert allocation.buffer.nbytes == source.nbytes
    finally:
        resident.free(runtime=runtime)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True
