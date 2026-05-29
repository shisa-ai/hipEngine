from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.kernels.registry import KernelKey
from hipengine.loading import materialize_stepfun_gguf_weights
from hipengine.loading.gguf import GGUFReader, scan_gguf_splits
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_DENSE_F32, LAYOUT_RAW_GGUF
from hipengine.loading.stepfun_gguf import build_stepfun_gguf_tensor_map
from hipengine.loading.stepfun_gguf_materialize import (
    plan_stepfun_gguf_materialization,
    stepfun_split_tensor_data,
)
from hipengine.runtime.gguf_linear import resolve_gguf_linear_dispatch

DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


def _model_map():
    info = scan_gguf_splits(_stepfun_gguf_paths())
    return info, build_stepfun_gguf_tensor_map(info)


def test_stepfun_materialization_plan_covers_split_mixed_quant_tensors() -> None:
    info, model_map = _model_map()

    plan = plan_stepfun_gguf_materialization(model_map)

    assert len(plan.specs) == len(model_map.tensor_names) == info.tensor_count
    assert plan.total_nbytes == info.total_tensor_nbytes
    assert dict(plan.quant_counts) == {
        "f32": 266,
        "gguf_q3_k": 309,
        "gguf_q5_k": 177,
        "gguf_q8_0": 2,
    }
    assert len({spec.source.source_path for spec in plan.specs}) == 3
    assert plan.root_specs["token_embedding"].layout == LAYOUT_RAW_GGUF
    assert plan.root_specs["token_embedding"].quant_key == "gguf_q8_0"
    assert plan.root_specs["output_norm"].layout == LAYOUT_DENSE_F32
    assert plan.layer_specs[0]["attn_q"].quant_key == "gguf_q3_k"
    assert plan.layer_specs[0]["attn_output"].quant_key == "gguf_q5_k"
    assert plan.layer_specs[3]["ffn_gate_inp"].quant_key == "f32"
    assert plan.layer_specs[44]["attn_q"].source.source_path.name.endswith(
        "00003-of-00003.gguf"
    )


def test_stepfun_split_tensor_data_matches_source_shard_reader_without_torch() -> None:
    had_torch = "torch" in sys.modules
    _, model_map = _model_map()
    tensors = (
        model_map.layer(0).tensor("attn_q"),
        model_map.layer(44).tensor("attn_q"),
    )
    assert len({tensor.source_path for tensor in tensors}) == 2

    for tensor in tensors:
        split_view = stepfun_split_tensor_data(tensor)
        direct_view = GGUFReader(tensor.source_path).tensor_data(tensor.name)

        assert split_view.shape == tensor.byte_shape == direct_view.shape
        assert split_view.dtype == direct_view.dtype
        np.testing.assert_array_equal(np.asarray(split_view[:2]), np.asarray(direct_view[:2]))
    if not had_torch:
        assert "torch" not in sys.modules


def test_stepfun_raw_q3_weight_resolves_existing_linear_dispatch() -> None:
    _, model_map = _model_map()
    plan = plan_stepfun_gguf_materialization(model_map)
    spec = plan.layer_specs[0]["attn_q"]
    fake_weight = type("Weight", (), {"spec": spec})()

    dispatch = resolve_gguf_linear_dispatch(fake_weight, backend="hip_gfx1151")

    assert dispatch.key == KernelKey("hip_gfx1151", "linear", "gguf_q3_k", "gemv_bf16_bf16_out")
    assert dispatch.abi == "raw"


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_selected_slot_materialization_loads_and_frees_device_weights() -> None:
    info, _ = _model_map()
    runtime = get_hip_runtime()
    had_torch = "torch" in sys.modules
    selected = (
        "root.output_norm",
        "layers.0.attn_q",
        "layers.3.ffn_gate_inp",
        "layers.44.attn_q",
    )
    expected_nbytes = sum(
        info.tensor(name).nbytes
        for name in (
            "output_norm.weight",
            "blk.0.attn_q.weight",
            "blk.3.ffn_gate_inp.weight",
            "blk.44.attn_q.weight",
        )
    )
    reset_memory_stats()

    weights = materialize_stepfun_gguf_weights(info, selected_slots=selected, runtime=runtime)
    try:
        assert set(weights.root_weights) == {"output_norm"}
        assert weights.root("output_norm").spec.layout == LAYOUT_DENSE_F32
        assert weights.layer(0).weight("attn_q").spec.quant_key == "gguf_q3_k"
        assert weights.layer(3).weight("ffn_gate_inp").spec.quant_key == "f32"
        assert weights.layer(44).weight("attn_q").spec.source.source_path.name.endswith(
            "00003-of-00003.gguf"
        )
        assert weights.allocated_nbytes == expected_nbytes
        stats = memory_stats()
        assert stats["current_allocated_bytes"] == expected_nbytes
        assert stats["active_allocations"] == len(selected)
    finally:
        weights.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    if not had_torch:
        assert "torch" not in sys.modules
