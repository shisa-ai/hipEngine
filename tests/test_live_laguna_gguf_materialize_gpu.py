from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, memory_stats, reset_memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf_materialize import (
    _materialize_spec,
    _spec_for_tensor,
    materialize_laguna_gguf_weights,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8, repack_gguf_q4_k_tile16
from hipengine.quant.gguf_t16 import (
    repack_gguf_q6_k_tile16_qmicro,
)
from tests._laguna_synthetic import tensor_info

MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        get_hip_runtime().mem_get_info()
    except (OSError, RuntimeError):
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


def _readback(allocation, runtime) -> bytes:
    host = np.empty(allocation.buffer.nbytes, dtype=np.uint8)
    copy_device_to_host(host.ctypes.data, allocation.buffer, runtime=runtime)
    return host.tobytes()


def test_laguna_tiny_pack8_and_t16_gpu_payloads_match_host_repack() -> None:
    runtime = get_hip_runtime()
    rng = np.random.default_rng(31)
    cases = (
        (
            "layers.0.ffn_gate",
            tensor_info("q4_pack8", (16, 256), GGMLQuantizationType.Q4_K),
            np.zeros((16, 144), dtype=np.uint8),
            repack_gguf_q4_k_pack8,
            ("qweight", "scales", "mins"),
        ),
        (
            "layers.1.ffn_gate_exps",
            tensor_info("q4_t16", (2, 16, 256), GGMLQuantizationType.Q4_K),
            rng.integers(0, 256, size=(2, 16, 144), dtype=np.uint8),
            repack_gguf_q4_k_tile16,
            ("tiles",),
        ),
        (
            "layers.1.ffn_down_exps",
            tensor_info("q6_t16", (2, 16, 256), GGMLQuantizationType.Q6_K),
            rng.integers(0, 256, size=(2, 16, 210), dtype=np.uint8),
            repack_gguf_q6_k_tile16_qmicro,
            ("tiles",),
        ),
    )
    for slot_path, tensor, raw, repack, names in cases:
        expected = repack(raw)
        weight = _materialize_spec(
            _spec_for_tensor(slot_path, tensor),
            _ArrayReader(tensor.name, raw),
            device=None,
            runtime=runtime,
            backend="hip_gfx1151",
            q6_qmicro=True,
            q6_qmicro_planar=False,
        )
        try:
            for name in names:
                assert (
                    _readback(weight.allocation(name), runtime) == getattr(expected, name).tobytes()
                ), (slot_path, name)
        finally:
            weight.free(runtime=runtime)


def test_completed_laguna_selected_slots_materialize_and_recover_gpu_memory() -> None:
    if not MODEL.exists():
        pytest.skip(f"local Laguna GGUF not found: {MODEL}")
    runtime = get_hip_runtime()
    reader = GGUFReader(MODEL)
    selected = (
        "root.output_norm",
        "layers.0.attn_gate",
        "layers.1.ffn_gate_shexp",
    )
    free_before, total_before = runtime.mem_get_info()
    reset_memory_stats()
    stats_before = memory_stats()
    profiles = []

    resident = materialize_laguna_gguf_weights(
        reader,
        selected_slots=selected,
        context_length=4_096,
        available_bytes=free_before,
        runtime=runtime,
        backend="hip_gfx1151",
        profile=profiles.append,
    )
    free_loaded, total_loaded = runtime.mem_get_info()
    stats_loaded = memory_stats()
    try:
        assert total_loaded == total_before
        planned_selected_nbytes = sum(
            resident.admission.weights.root_specs[slot].resident_nbytes for slot in ("output_norm",)
        ) + sum(
            resident.admission.weights.layer_specs[layer][slot].resident_nbytes
            for layer, slot in ((0, "attn_gate"), (1, "ffn_gate_shexp"))
        )
        assert resident.resident_nbytes == (
            planned_selected_nbytes + resident.admission.auxiliary_weight_nbytes
        )
        assert (
            stats_loaded["current_allocated_bytes"] - stats_before["current_allocated_bytes"]
            == resident.resident_nbytes
        )
        expected_allocations = sum(len(weight.allocations) for weight in resident.weights)
        assert (
            stats_loaded["active_allocations"] - stats_before["active_allocations"]
            == expected_allocations
        )
        assert len(profiles) == len(selected)
        assert sum(profile.allocation_count for profile in profiles) == expected_allocations
        assert sum(profile.upload_count for profile in profiles) == expected_allocations
        assert sum(profile.allocated_nbytes for profile in profiles) == resident.resident_nbytes
        assert sum(profile.uploaded_nbytes for profile in profiles) == resident.resident_nbytes
        assert all(profile.total_seconds > 0.0 for profile in profiles)
        assert free_loaded <= free_before

        norm_raw = np.ascontiguousarray(reader.tensor_data("output_norm.weight"))
        assert _readback(resident.root("output_norm").allocation(), runtime) == norm_raw.tobytes()

        gate_raw = np.ascontiguousarray(reader.tensor_data("blk.0.attn_gate.weight"))
        assert (
            _readback(resident.layer(0).weight("attn_gate").allocation(), runtime)
            == gate_raw.tobytes()
        )

        shared_raw = np.ascontiguousarray(reader.tensor_data("blk.1.ffn_gate_shexp.weight"))
        shared_expected = repack_gguf_q4_k_pack8(shared_raw)
        shared = resident.layer(1).weight("ffn_gate_shexp")
        for name in ("qweight", "scales", "mins"):
            assert (
                _readback(shared.allocation(name), runtime)
                == getattr(shared_expected, name).tobytes()
            )
    finally:
        resident.free(runtime=runtime)

    free_after, total_after = runtime.mem_get_info()
    stats_after = memory_stats()
    assert total_after == total_before
    assert stats_after["current_allocated_bytes"] == stats_before["current_allocated_bytes"]
    assert stats_after["active_allocations"] == stats_before["active_allocations"]
    assert free_after >= free_loaded
    # The process's first HIP allocation may retain the ~150 MiB runtime context;
    # tracked Laguna ownership above must still recover exactly.
    assert free_after >= free_before - 256 * 2**20


class _ArrayReader:
    def __init__(self, name: str, array: np.ndarray) -> None:
        self.name = name
        self.array = array

    def tensor_data(self, name: str) -> np.ndarray:
        assert name == self.name
        return self.array
