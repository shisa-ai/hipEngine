"""WPF-H7U gfx1100 stable parallel MoE compaction RED contract."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.hip_gfx1100.moe import (
    build_qwen35_moe_group_scatter,
    qwen35_moe_gather_packed_hidden_lowp,
    qwen35_moe_group_compact_active_parallel,
    qwen35_moe_group_compact_active_source_rows,
    qwen35_moe_group_compact_active_source_rows_parallel,
    qwen35_moe_mmq32_tile_map,
    register_qwen35_moe_group_scatter_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.runtime.laguna_moe import resolve_laguna_group_compact_mode

_ROOT = Path(__file__).parents[1]
_ARTIFACT = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-post-h7t-"
    "parallel-moe-compaction-target.json"
)
_ARTIFACT_SHA256 = (
    "f9b9669ec935585fe425617db138751c75aa3f0aa12d67e7139061bcb9c8c4c3"
)
_POST_MERGE_PACKAGE_SHA256 = (
    "a7365e583064e581744760a4723cccaea8fa9a8c9ece7584f2e6ea6ccb291981"
)
_POST_MERGE_SOURCE_SHA256 = {
    # Maple P1 templates the existing stable parallel count/scatter bodies so
    # int32 route IDs share the exact H7U implementation; the original int64
    # symbols, launch geometry, and Laguna source owner remain unchanged.
    "hipengine/kernels/hip_gfx1100/moe/group_scatter.hip": (
        "19a4f3f9b55ef7258b63b30fc243613a6951e67b3f1e9df4f66cb37ca5ad3b07"
    ),
    "hipengine/kernels/hip_gfx1100/moe/group_scatter.py": (
        "4ede6f2c6932eb992b148f8b3040d2aa49a27f0b7e11d69c708b166ef5c8916b"
    ),
    # Later Qwen3.8 and execution-profile package policies are orthogonal to
    # H7U's unchanged gfx1151 parallel-compaction owner.
    "hipengine/kernels/hip_gfx1151/__init__.py": (
        "10a5e6e609135facc96da6271e9e2949db581dd4d6492de7b21eaf76a19d0e37"
    ),
    "hipengine/runtime/laguna_moe.py": (
        "b37bc2a1aaadbf94700dad9a67f90815b69d783a8a82fcc47b5496a17de83987"
    ),
    "tests/test_laguna_moe_gpu.py": (
        "8776311fb4f64bbf0c050a18fb85525abb418b7e89a0877b214afcaac69b8396"
    ),
}
_H7U_CAPABILITY = "LAGUNA_MOE_GROUP_COMPACT_H7U_MODE"
_SOURCE_CAPABILITY = "LAGUNA_MOE_GROUP_COMPACT_MODE"
_H7U_PACKAGE_BLOCK = (
    "# WPF-H7U exposes the exact registered stable parallel active-route compactor\n"
    "# only as a bounded default-off W7900 candidate. The live source owner remains\n"
    "# serial until complete standalone/runtime/source qualification.\n"
    'LAGUNA_MOE_GROUP_COMPACT_H7U_MODE = "parallel"\n'
)
_H7U_SOURCE_BLOCK = (
    "# WPF-H7U promotes exact stable parallel active-route compaction after full\n"
    "# standalone, bounded-runtime, fixed, length, and source-trace qualification.\n"
    "# Explicit serial remains the registered rollback; peer backends stay local.\n"
    'LAGUNA_MOE_GROUP_COMPACT_MODE = "parallel"\n'
)
_OLD_MODE_TEST_BLOCK = (
    '    assert resolve_laguna_group_compact_mode("hip_gfx1100") == "serial"\n'
)
_SOURCE_MODE_TEST_BLOCK = (
    '    assert resolve_laguna_group_compact_mode("hip_gfx1100") == "parallel"\n'
    '    assert resolve_laguna_group_compact_mode("hip_gfx1100", "serial") == "serial"\n'
    '    assert resolve_laguna_group_compact_mode("hip_gfx1100", "parallel") == "parallel"\n'
)
_EXPERTS = 256
_TOP_K = 10
_M512_TOKENS = 512
_HIDDEN = 3_072
_PARALLEL_KERNELS = (
    "qwen35_moe_group_count_active_parallel_kernel",
    "qwen35_moe_group_prefix_active_parallel_kernel",
    "qwen35_moe_group_scatter_active_parallel_kernel",
)
_TRACE_CONTRACT = {
    "count": 47,
    "prefix": 47,
    "scatter": 47,
    "serial": 0,
    "gather": 47,
    "request_dispatches": 2_286,
    "queues": 1,
    "compiler_processes": 0,
}
_TIMING_CONTRACT = {
    "warmups": 5,
    "samples": 15,
    "launch_repeats": 5,
    "layers": 47,
    "required_clocks": ("hip_event", "synchronized_wall"),
    "require_every_layer_positive": True,
    "require_aggregate_positive": True,
    "allow_subset_salvage": False,
    "allow_recompile": False,
    "allow_retune": False,
    "allow_favorable_rerun": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _parallel_mode() -> str:
    bounded = getattr(hip_gfx1100, _H7U_CAPABILITY, None)
    source = getattr(hip_gfx1100, _SOURCE_CAPABILITY, None)
    assert (bounded, source) in (("parallel", None), (None, "parallel"))
    mode = bounded or source
    assert mode == "parallel"
    assert resolve_laguna_group_compact_mode("hip_gfx1100", mode) == "parallel"
    return mode


def _cpu_metadata(
    selected: np.ndarray,
    weights: np.ndarray,
    *,
    top_k: int,
) -> dict[str, np.ndarray]:
    selected = np.ascontiguousarray(selected, dtype=np.int64).reshape(-1)
    weights = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
    assert selected.shape == weights.shape
    assert selected.size > 0
    assert np.all((0 <= selected) & (selected < _EXPERTS))
    counts = np.bincount(selected, minlength=_EXPERTS).astype(np.int64)
    starts = np.empty(_EXPERTS + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, out=starts[1:])
    lanes = np.argsort(selected, kind="stable").astype(np.int64)
    active = np.flatnonzero(counts).astype(np.int64)
    return {
        "starts": starts,
        "active": active,
        "lanes": lanes,
        "source_rows": lanes // top_k,
        "weights": weights[lanes],
        "counts": counts,
    }


def _copy_to_device(array: np.ndarray, buffers: list[Any]) -> Any:
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes)
    return buffer


def _run_compaction(
    selected: np.ndarray,
    weights: np.ndarray,
    *,
    top_k: int,
    parallel: bool,
    library: Any,
) -> dict[str, np.ndarray]:
    selected = np.ascontiguousarray(selected, dtype=np.int64).reshape(-1)
    weights = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
    buffers: list[Any] = []
    try:
        selected_buffer = _copy_to_device(selected, buffers)
        weights_buffer = _copy_to_device(weights, buffers)
        starts_buffer = _copy_to_device(
            np.full(_EXPERTS + 1, -11, dtype=np.int64), buffers
        )
        active_buffer = _copy_to_device(
            np.full(_EXPERTS, -13, dtype=np.int64), buffers
        )
        active_count_buffer = _copy_to_device(
            np.full(1, -17, dtype=np.int64), buffers
        )
        lanes_buffer = _copy_to_device(
            np.full(selected.size, -19, dtype=np.int64), buffers
        )
        source_rows_buffer = _copy_to_device(
            np.full(selected.size, -23, dtype=np.int64), buffers
        )
        sorted_weights_buffer = _copy_to_device(
            np.full(selected.size, np.nan, dtype=np.float32), buffers
        )

        qwen35_moe_group_compact_active_source_rows(
            selected_buffer.ptr,
            weights_buffer.ptr,
            starts_buffer.ptr,
            active_buffer.ptr,
            active_count_buffer.ptr,
            lanes_buffer.ptr,
            source_rows_buffer.ptr,
            sorted_weights_buffer.ptr,
            selected.size,
            _EXPERTS,
            top_k,
            parallel=parallel,
            library=library,
        )

        active_count = np.empty(1, dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(active_count),
            active_count_buffer,
            active_count.nbytes,
        )
        outputs = {
            "starts": np.empty(_EXPERTS + 1, dtype=np.int64),
            "active": np.empty(int(active_count[0]), dtype=np.int64),
            "lanes": np.empty(selected.size, dtype=np.int64),
            "source_rows": np.empty(selected.size, dtype=np.int64),
            "weights": np.empty(selected.size, dtype=np.float32),
        }
        for name, buffer in (
            ("starts", starts_buffer),
            ("active", active_buffer),
            ("lanes", lanes_buffer),
            ("source_rows", source_rows_buffer),
            ("weights", sorted_weights_buffer),
        ):
            output = outputs[name]
            copy_device_to_host(host_array_ptr(output), buffer, output.nbytes)
        return outputs
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def _assert_metadata_equal(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> None:
    for name in ("starts", "active", "lanes", "source_rows", "weights"):
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def _fixture_one_active() -> tuple[np.ndarray, np.ndarray]:
    selected = np.full(70, 137, dtype=np.int64)
    weights = np.linspace(0.001, 0.999, selected.size, dtype=np.float32)
    return selected, weights


def _fixture_uneven() -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(
        [
            255,
            2,
            2,
            17,
            2,
            0,
            17,
            91,
            2,
            255,
            17,
            17,
            0,
            91,
            91,
            91,
            91,
            2,
            0,
            255,
        ],
        dtype=np.int64,
    )
    weights = ((np.arange(selected.size, dtype=np.float32) + 1) / 37).astype(
        np.float32
    )
    return selected, weights


def _fixture_all_active() -> tuple[np.ndarray, np.ndarray]:
    selected = np.tile(np.arange(_EXPERTS, dtype=np.int64), 10)
    weights = ((np.arange(selected.size, dtype=np.float32) % 251) + 1) / 252
    return selected, weights.astype(np.float32)


def _fixture_repeated() -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray([3, 3, 3, 7, 3, 7, 129, 3, 129, 7], dtype=np.int64)
    selected = np.tile(base, 73)
    weights = np.linspace(0.0001, 0.9999, selected.size, dtype=np.float32)
    return selected, weights


def _fixture_total_lane_tail() -> tuple[np.ndarray, np.ndarray]:
    lanes = 513 * _TOP_K
    selected = ((np.arange(lanes, dtype=np.int64) * 73 + 19) % _EXPERTS)
    weights = ((np.arange(lanes, dtype=np.float32) % 509) + 1) / 510
    return selected, weights.astype(np.float32)


_EDGE_FIXTURES = (
    pytest.param(_fixture_one_active, id="one-active-255-empty"),
    pytest.param(_fixture_uneven, id="uneven"),
    pytest.param(_fixture_all_active, id="all-active"),
    pytest.param(_fixture_repeated, id="repeated-expert"),
    pytest.param(_fixture_total_lane_tail, id="total-lane-tail"),
)


def test_h7u_frozen_target_source_physical_trace_and_timing_contract() -> None:
    artifact_bytes = _ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "accepted_target_only_no_gfx1100_candidate_run"
    assert artifact["target"]["id"] == "WPF-H7U"
    assert artifact["decision"] == {
        "gfx1100_candidate_executed": False,
        "gfx1100_speed_result_exists": False,
        "new_candidate_code_required": False,
        "next_action": (
            "commit this target-only boundary, then freeze one H7U RED before "
            "any gfx1100 capability/default change or candidate execution"
        ),
        "performance_claim": "target_selection_only",
        "production_changed": False,
        "target_selected": True,
    }
    assert artifact["target"]["parallel_kernels"] == list(_PARALLEL_KERNELS)
    assert artifact["target"]["control_dispatches"] == 47
    assert artifact["target"]["candidate_dispatches"] == 141
    assert artifact["target"]["request_dispatch_delta"] == 94
    assert artifact["target"]["expected_candidate_request_dispatches"] == 2_286
    assert artifact["target"]["gather_unchanged"]
    assert artifact["target"]["mmq_tile_map_unchanged"]
    assert artifact["target"]["router_outputs_unchanged"]
    assert artifact["target"]["gate_up_down_arithmetic_unchanged"]
    assert artifact["target"]["new_device_allocation_bytes"] == 0
    assert artifact["target"]["new_workspace_bytes"] == 0

    admission = artifact["admission"]
    for term in (
        "expert_start[257]",
        "active_experts/count",
        "sorted_lanes",
        "compact_to_source",
        "sorted_weights",
        "packed hidden",
        "complete MoE BF16 output",
        "all 48 hidden boundaries",
        "logits",
        "KV/KVLiveSpans",
        "repeat",
        "lifecycle",
    ):
        assert term in admission["correctness"]
    for term in (
        "empty",
        "one-active",
        "uneven",
        "all-active",
        "repeated-expert",
        "total-lane tail",
    ):
        assert term in admission["tails"]
    assert "private/spill/scratch0" in admission["physical"]
    assert "each layer" in admission["timing"]
    assert "summed HIP-event time" in admission["timing"]
    assert "synchronized first-stage-to-last-stage wall time" in admission["timing"]
    assert "47 count + 47 prefix + 47 scatter" in admission["trace"]
    assert "zero serial" in admission["trace"]
    assert "unchanged 47 packed-hidden gathers" in admission["trace"]
    assert "2,286 total request dispatches" in admission["trace"]
    assert "zero compiler" in admission["trace"]
    assert "no layer/expert/routing-pattern/length subset" in admission["no_salvage"]

    assert _TRACE_CONTRACT == {
        "count": 47,
        "prefix": 47,
        "scatter": 47,
        "serial": 0,
        "gather": 47,
        "request_dispatches": 2_286,
        "queues": 1,
        "compiler_processes": 0,
    }
    assert _TIMING_CONTRACT["warmups"] == 5
    assert _TIMING_CONTRACT["samples"] == 15
    assert _TIMING_CONTRACT["launch_repeats"] == 5
    assert _TIMING_CONTRACT["layers"] == 47
    assert _TIMING_CONTRACT["required_clocks"] == (
        "hip_event",
        "synchronized_wall",
    )
    assert _TIMING_CONTRACT["require_every_layer_positive"]
    assert _TIMING_CONTRACT["require_aggregate_positive"]
    assert not any(
        _TIMING_CONTRACT[name]
        for name in (
            "allow_subset_salvage",
            "allow_recompile",
            "allow_retune",
            "allow_favorable_rerun",
        )
    )

    for relative, expected in artifact["source_sha256"].items():
        path = _ROOT / relative
        if relative == "tests/test_laguna_moe_gpu.py":
            test_source = path.read_text()
            old_count = test_source.count(_OLD_MODE_TEST_BLOCK)
            source_count = test_source.count(_SOURCE_MODE_TEST_BLOCK)
            assert (old_count, source_count) in ((1, 0), (0, 1))
            normalized = test_source.replace(
                _SOURCE_MODE_TEST_BLOCK, _OLD_MODE_TEST_BLOCK
            )
            assert hashlib.sha256(normalized.encode()).hexdigest() == (
                _POST_MERGE_SOURCE_SHA256.get(relative, expected)
            )
            continue
        if relative != "hipengine/kernels/hip_gfx1100/__init__.py":
            assert _sha256(path) == _POST_MERGE_SOURCE_SHA256.get(relative, expected)
            continue
        package_source = path.read_text()
        bounded_count = package_source.count(_H7U_PACKAGE_BLOCK)
        source_count = package_source.count(_H7U_SOURCE_BLOCK)
        assert bounded_count + source_count in (0, 1)
        normalized = package_source.replace(_H7U_PACKAGE_BLOCK, "").replace(
            _H7U_SOURCE_BLOCK, ""
        )
        assert hashlib.sha256(normalized.encode()).hexdigest() == (
            _POST_MERGE_PACKAGE_SHA256
        )


def test_h7u_existing_registered_leaf_and_immutable_source_are_complete() -> None:
    register_qwen35_moe_group_scatter_kernels(replace=True)
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_group_compact",
        quant="generic",
        variant="active_experts_parallel",
    ) is qwen35_moe_group_compact_active_parallel
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_group_compact",
        quant="generic",
        variant="active_experts_source_rows_parallel",
    ) is qwen35_moe_group_compact_active_source_rows_parallel

    source = (
        _ROOT / "hipengine/kernels/hip_gfx1100/moe/group_scatter.hip"
    ).read_text()
    for kernel in _PARALLEL_KERNELS:
        assert source.count(f"__global__ void {kernel}(") == 1
    assert source.count("__launch_bounds__(256, 1)") >= 3
    assert source.count("dim3(256)") >= 3
    assert "Blelloch exclusive scan over the fixed 256-expert upper bound" in source
    assert "__ballot(" in source
    assert "__popcll(" in source


def test_h7u_package_mode_keeps_serial_rollback_and_gfx1151_isolation() -> None:
    bounded = getattr(hip_gfx1100, _H7U_CAPABILITY, None)
    source = getattr(hip_gfx1100, _SOURCE_CAPABILITY, None)
    assert (bounded, source) in (("parallel", None), (None, "parallel"))
    expected_default = "serial" if bounded is not None else "parallel"
    assert resolve_laguna_group_compact_mode("hip_gfx1100") == expected_default
    assert resolve_laguna_group_compact_mode("hip_gfx1100", "serial") == "serial"
    assert _parallel_mode() == "parallel"
    assert getattr(hip_gfx1151, _SOURCE_CAPABILITY) == "parallel"
    assert not hasattr(hip_gfx1151, _H7U_CAPABILITY)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("fixture_factory", _EDGE_FIXTURES)
def test_h7u_stable_edge_metadata_matches_cpu_and_serial(
    fixture_factory: Any,
) -> None:
    mode = _parallel_mode()
    assert mode == "parallel"
    library = build_qwen35_moe_group_scatter(load=True)
    selected, weights = fixture_factory()
    expected = _cpu_metadata(selected, weights, top_k=_TOP_K)
    baseline = memory_stats()
    serial = _run_compaction(
        selected,
        weights,
        top_k=_TOP_K,
        parallel=False,
        library=library,
    )
    candidate = _run_compaction(
        selected,
        weights,
        top_k=_TOP_K,
        parallel=True,
        library=library,
    )
    _assert_metadata_equal(serial, expected)
    _assert_metadata_equal(candidate, expected)
    assert memory_stats()["current_allocated_bytes"] == baseline[
        "current_allocated_bytes"
    ]
    assert memory_stats()["active_allocations"] == baseline["active_allocations"]


def _m512_routes() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(_M512_TOKENS, dtype=np.int64)[:, None]
    offsets = np.asarray([0, 1, 5, 11, 19, 31, 47, 67, 89, 113], dtype=np.int64)
    selected = ((rows * 37 + offsets) % _EXPERTS).reshape(-1)
    lane_ids = np.arange(selected.size, dtype=np.float32)
    weights = ((lane_ids % 997) + 1) / 998
    assert all(np.unique(row).size == _TOP_K for row in selected.reshape(-1, _TOP_K))
    return selected, weights.astype(np.float32)


def _run_m512_pipeline(
    selected: np.ndarray,
    weights: np.ndarray,
    hidden: np.ndarray,
    *,
    parallel: bool,
    library: Any,
) -> dict[str, Any]:
    expected = _cpu_metadata(selected, weights, top_k=_TOP_K)
    buffers: list[Any] = []
    try:
        selected_buffer = _copy_to_device(selected, buffers)
        weights_buffer = _copy_to_device(weights, buffers)
        hidden_buffer = _copy_to_device(hidden, buffers)
        starts_buffer = _copy_to_device(
            np.full(_EXPERTS + 1, -29, dtype=np.int64), buffers
        )
        active_buffer = _copy_to_device(
            np.full(_EXPERTS, -31, dtype=np.int64), buffers
        )
        active_count_buffer = _copy_to_device(
            np.full(1, -37, dtype=np.int64), buffers
        )
        lanes_buffer = _copy_to_device(
            np.full(selected.size, -41, dtype=np.int64), buffers
        )
        source_rows_buffer = _copy_to_device(
            np.full(selected.size, -43, dtype=np.int64), buffers
        )
        sorted_weights_buffer = _copy_to_device(
            np.full(selected.size, np.nan, dtype=np.float32), buffers
        )
        mmq_starts_buffer = _copy_to_device(
            np.full(_EXPERTS + 1, -47, dtype=np.int64), buffers
        )
        tile_experts_buffer = _copy_to_device(
            np.full(selected.size, -53, dtype=np.int64), buffers
        )
        mmq_total_buffer = _copy_to_device(
            np.full(1, -59, dtype=np.int64), buffers
        )
        packed_buffer = _copy_to_device(
            np.full(selected.size * _HIDDEN, 0xDEAD, dtype=np.uint16), buffers
        )

        qwen35_moe_group_compact_active_source_rows(
            selected_buffer.ptr,
            weights_buffer.ptr,
            starts_buffer.ptr,
            active_buffer.ptr,
            active_count_buffer.ptr,
            lanes_buffer.ptr,
            source_rows_buffer.ptr,
            sorted_weights_buffer.ptr,
            selected.size,
            _EXPERTS,
            _TOP_K,
            parallel=parallel,
            library=library,
        )
        qwen35_moe_mmq32_tile_map(
            starts_buffer.ptr,
            mmq_starts_buffer.ptr,
            tile_experts_buffer.ptr,
            mmq_total_buffer.ptr,
            _EXPERTS,
            tile_capacity=selected.size,
            library=library,
        )
        qwen35_moe_gather_packed_hidden_lowp(
            hidden_buffer.ptr,
            lanes_buffer.ptr,
            packed_buffer.ptr,
            selected.size * _HIDDEN,
            _M512_TOKENS,
            _TOP_K,
            _HIDDEN,
            library=library,
        )

        active_count = np.empty(1, dtype=np.int64)
        mmq_total = np.empty(1, dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(active_count), active_count_buffer, active_count.nbytes
        )
        copy_device_to_host(
            host_array_ptr(mmq_total), mmq_total_buffer, mmq_total.nbytes
        )
        metadata = {
            "starts": np.empty(_EXPERTS + 1, dtype=np.int64),
            "active": np.empty(int(active_count[0]), dtype=np.int64),
            "lanes": np.empty(selected.size, dtype=np.int64),
            "source_rows": np.empty(selected.size, dtype=np.int64),
            "weights": np.empty(selected.size, dtype=np.float32),
        }
        for name, buffer in (
            ("starts", starts_buffer),
            ("active", active_buffer),
            ("lanes", lanes_buffer),
            ("source_rows", source_rows_buffer),
            ("weights", sorted_weights_buffer),
        ):
            output = metadata[name]
            copy_device_to_host(host_array_ptr(output), buffer, output.nbytes)
        _assert_metadata_equal(metadata, expected)

        mmq_starts = np.empty(_EXPERTS + 1, dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(mmq_starts), mmq_starts_buffer, mmq_starts.nbytes
        )
        tile_count = int(mmq_total[0]) // 32
        tile_experts = np.empty(tile_count, dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(tile_experts),
            tile_experts_buffer,
            tile_experts.nbytes,
        )
        padded_counts = ((expected["counts"] + 31) // 32) * 32
        expected_mmq_starts = np.empty(_EXPERTS + 1, dtype=np.int64)
        expected_mmq_starts[0] = 0
        np.cumsum(padded_counts, out=expected_mmq_starts[1:])
        expected_tiles = np.repeat(
            np.arange(_EXPERTS, dtype=np.int64), padded_counts // 32
        )
        np.testing.assert_array_equal(mmq_starts, expected_mmq_starts)
        assert int(mmq_total[0]) == int(expected_mmq_starts[-1])
        np.testing.assert_array_equal(tile_experts, expected_tiles)

        packed = np.empty((selected.size, _HIDDEN), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(packed), packed_buffer, packed.nbytes)
        expected_packed = hidden[expected["source_rows"]]
        np.testing.assert_array_equal(packed, expected_packed)
        digest = hashlib.sha256()
        for name in ("starts", "active", "lanes", "source_rows", "weights"):
            digest.update(metadata[name].tobytes())
        digest.update(mmq_starts.tobytes())
        digest.update(tile_experts.tobytes())
        digest.update(packed.tobytes())
        return {
            "state_sha256": digest.hexdigest(),
            "active_count": int(active_count[0]),
            "mmq_total": int(mmq_total[0]),
            "tile_count": tile_count,
        }
    finally:
        for buffer in reversed(buffers):
            free(buffer)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h7u_natural_m512_metadata_tile_map_packed_hidden_repeat_and_lifecycle() -> None:
    mode = _parallel_mode()
    assert mode == "parallel"
    library = build_qwen35_moe_group_scatter(load=True)
    selected, weights = _m512_routes()
    hidden_values = np.arange(_M512_TOKENS * _HIDDEN, dtype=np.uint32)
    hidden = ((hidden_values * 17 + 3) & 0xFFFF).astype(np.uint16).reshape(
        _M512_TOKENS, _HIDDEN
    )
    baseline = memory_stats()
    serial = _run_m512_pipeline(
        selected,
        weights,
        hidden,
        parallel=False,
        library=library,
    )
    candidate_first = _run_m512_pipeline(
        selected,
        weights,
        hidden,
        parallel=True,
        library=library,
    )
    candidate_repeat = _run_m512_pipeline(
        selected,
        weights,
        hidden,
        parallel=True,
        library=library,
    )
    assert serial == candidate_first == candidate_repeat
    assert serial["active_count"] == _EXPERTS
    assert serial["tile_count"] == _EXPERTS
    assert serial["mmq_total"] == _EXPERTS * 32
    assert memory_stats()["current_allocated_bytes"] == baseline[
        "current_allocated_bytes"
    ]
    assert memory_stats()["active_allocations"] == baseline["active_allocations"]
