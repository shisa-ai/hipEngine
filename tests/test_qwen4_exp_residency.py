from __future__ import annotations

from math import prod
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.qwen4_exp_gguf import build_qwen4_exp_gguf_tensor_map
from hipengine.loading.qwen4_exp_materialize import (
    LAYOUT_PLE_SPARSE_MMAP,
    Qwen4ExpPLEMMapTable,
    Qwen4ExpPLEStagingRing,
    plan_qwen4_exp_memory_admission,
    plan_qwen4_exp_residency,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests.test_qwen4_exp_gguf_mapping import _infos


def test_qwen4_exp_residency_defers_only_ple_and_keeps_one_device_layout() -> None:
    model_map = build_qwen4_exp_gguf_tensor_map(_infos())

    plan = plan_qwen4_exp_residency(model_map, staging_token_capacity=4)

    assert len(plan.specs) == 1_224
    assert len(plan.device_specs) == 1_223
    assert plan.ple_spec.layout == LAYOUT_PLE_SPARSE_MMAP
    assert plan.ple_spec.device_resident is False
    assert plan.ple_spec.allocation_names == ()
    assert plan.ple_mmap_bytes == model_map.ple_table.tensor.nbytes
    assert plan.raw_payload_bytes == sum(ref.tensor.nbytes for ref in model_map.tensor_refs)
    assert plan.device_weight_bytes == plan.raw_payload_bytes - plan.ple_mmap_bytes
    assert plan.replacement_payload_bytes == 0
    assert plan.alternate_layout_bytes == 0
    assert plan.staging_buffer_count == 2
    assert plan.staging_row_capacity == 64
    assert plan.staging_bytes == 2 * 64 * 160 * 4
    assert all(spec.allocation_names == ("raw",) for spec in plan.device_specs)
    assert all(spec.quant_key == "f32" for spec in plan.device_specs)
    assert all(spec.layout != LAYOUT_PLE_SPARSE_MMAP for spec in plan.device_specs)


def test_qwen4_exp_memory_admission_accounts_kv_index_state_scratch_and_reserve() -> None:
    model_map = build_qwen4_exp_gguf_tensor_map(_infos())
    plan = plan_qwen4_exp_residency(model_map, staging_token_capacity=1)
    provisional = plan_qwen4_exp_memory_admission(
        plan,
        available_device_bytes=10**15,
        context_tokens=262_144,
        resident_capacity=2,
        scratch_bytes=1_000,
        reserve_bytes=2_000,
    )

    assert provisional.kv_bytes == 2 * 262_144 * 24_576
    config = plan.config
    complete_blocks = 262_144 // config.qsa_compression_ratio
    per_layer_index_bytes = (
        262_144 * config.indexer_key_length * 4
        + complete_blocks * config.qsa_compression_ratio * 4
        + complete_blocks * 8
        + complete_blocks * config.indexer_key_length * 4
        + complete_blocks * 4
        + config.qsa_block_budget * 8
        + 4
        + 8
        + config.qsa_dense_equivalent_max_tokens * 8
    )
    assert provisional.index_bytes == 2 * config.qsa_layer_count * per_layer_index_bytes
    assert provisional.runtime_state_bytes > 2 * 108 * 1024 * 1024
    assert provisional.required_bytes == (
        plan.device_weight_bytes
        + plan.staging_bytes
        + provisional.kv_bytes
        + provisional.index_bytes
        + provisional.runtime_state_bytes
        + 1_000
        + 2_000
    )
    assert provisional.passed

    exact = plan_qwen4_exp_memory_admission(
        plan,
        available_device_bytes=provisional.required_bytes,
        context_tokens=262_144,
        resident_capacity=2,
        scratch_bytes=1_000,
        reserve_bytes=2_000,
    )
    assert exact.passed
    rejected = plan_qwen4_exp_memory_admission(
        plan,
        available_device_bytes=provisional.required_bytes - 1,
        context_tokens=262_144,
        resident_capacity=2,
        scratch_bytes=1_000,
        reserve_bytes=2_000,
    )
    assert rejected.passed is False
    assert rejected.shortfall_bytes == 1


def _iq4_nl_rows(scales: tuple[float, ...]) -> np.ndarray:
    raw = np.zeros((len(scales), 5 * 18), dtype=np.uint8)
    for row, scale in enumerate(scales):
        encoded = np.asarray([scale], dtype=np.float16).view(np.uint8)
        for block in range(5):
            raw[row, block * 18 : block * 18 + 2] = encoded
    return raw


def _ple_tensor(rows: int) -> GGUFTensorInfo:
    shape = (rows, 160)
    return GGUFTensorInfo(
        name="per_layer_token_embd.weight",
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(GGMLQuantizationType.IQ4_NL),
        ggml_type_name="IQ4_NL",
        n_elements=prod(shape),
        nbytes=rows * 90,
        offset=0,
        data_offset=0,
        byte_shape=(rows, 90),
    )


def test_qwen4_exp_ple_mmap_gathers_only_requested_iq4_nl_rows() -> None:
    raw = _iq4_nl_rows((1.0, 2.0, 3.0))
    calls: list[str] = []

    def tensor_data(name: str) -> np.ndarray:
        calls.append(name)
        return raw

    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(tensor_data=tensor_data),
        _ple_tensor(3),
        semantic_rows=3,
    )

    gathered = table.gather_rows([2, 0])

    assert calls == ["per_layer_token_embd.weight"]
    assert gathered.shape == (2, 160)
    np.testing.assert_array_equal(gathered[0], np.full(160, -381.0, dtype=np.float32))
    np.testing.assert_array_equal(gathered[1], np.full(160, -127.0, dtype=np.float32))
    assert table.rows_gathered == 2
    with pytest.raises(IndexError, match="semantic"):
        table.gather_rows([3])
    table.close()
    table.close()
    with pytest.raises(RuntimeError, match="closed"):
        table.gather_rows([0])


def test_qwen4_exp_ple_telemetry_is_opt_in_and_reports_locality() -> None:
    raw = _iq4_nl_rows((1.0, 2.0, 3.0))
    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(tensor_data=lambda name: raw),
        _ple_tensor(3),
        semantic_rows=3,
    )
    ring = Qwen4ExpPLEStagingRing.create(table, row_capacity=3)

    assert table.telemetry() is None
    table.enable_telemetry()
    ring.stage([2, 0, 2])
    ring.record_h2d(nbytes=1920, wall_ns=1234)
    snapshot = table.telemetry()

    assert snapshot is not None
    assert snapshot["calls"] == 1
    assert snapshot["requested_rows"] == 3
    assert snapshot["unique_rows"] == 2
    assert snapshot["requested_source_bytes"] == 270
    assert snapshot["unique_pages"] >= 1
    assert snapshot["page_range_count"] >= 1
    assert snapshot["page_ranges_sample"]
    assert snapshot["gather_dequant_wall_ns"] > 0
    assert snapshot["staging_copy_wall_ns"] > 0
    assert snapshot["h2d_wall_ns"] == 1234
    assert snapshot["h2d_bytes"] == 1920
    assert snapshot["cache_mode"] == "unadvised"

    ring.close()
    table.close()


def test_qwen4_exp_ple_cache_advice_is_file_scoped(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    raw = _iq4_nl_rows((1.0, 2.0, 3.0))

    class Mapping:
        def __init__(self) -> None:
            self.advice = []
            self.closed = 0

        def madvise(self, advice: int) -> None:
            self.advice.append(advice)

        def close(self) -> None:
            self.closed += 1

    mapping = Mapping()

    class Raw:
        _mmap = mapping

        def __getitem__(self, item):
            return raw[item]

    mapped_raw = Raw()
    source = tmp_path / "part.gguf"
    source.write_bytes(b"0" * 1024)
    reader = SimpleNamespace(path=source, tensor_data=lambda name: mapped_raw)
    calls = []
    monkeypatch.setattr(
        "hipengine.loading.qwen4_exp_materialize.os.posix_fadvise",
        lambda fd, offset, length, advice: calls.append((offset, length, advice)),
    )
    table = Qwen4ExpPLEMMapTable(reader, _ple_tensor(3), semantic_rows=3)

    cold = table.advise_cache("cold")
    warm = table.advise_cache("warm")

    assert cold["scope"] == "ple_tensor_file_range"
    assert cold["mode"] == "cold"
    assert warm["mode"] == "warm"
    assert cold["mapping_reopened"] is True
    assert mapping.closed == 1
    assert len(mapping.advice) == 2
    assert len(calls) == 2
    assert calls[0][0:2] == (0, 270)
    table.enable_telemetry()
    assert table.telemetry()["cache_mode"] == "warm"
    assert table.telemetry()["prefetch_ranges"] == [{"offset": 0, "nbytes": 270}]


class _FakeRuntime:
    def __init__(self) -> None:
        self.registered: list[tuple[int, int]] = []
        self.unregistered: list[int] = []

    def host_register(self, ptr: int, nbytes: int, *, flags: int = 0) -> None:
        del flags
        self.registered.append((ptr, nbytes))

    def host_unregister(self, ptr: int) -> None:
        self.unregistered.append(ptr)


def test_qwen4_exp_ple_staging_ring_is_bounded_double_buffered_and_closes() -> None:
    raw = _iq4_nl_rows((1.0, 2.0, 3.0))
    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(tensor_data=lambda name: raw),
        _ple_tensor(3),
        semantic_rows=3,
    )
    runtime = _FakeRuntime()
    ring = Qwen4ExpPLEStagingRing.create(
        table,
        row_capacity=2,
        runtime=runtime,
    )

    assert ring.pinned
    assert len(runtime.registered) == 2
    first = ring.stage([0, 1])
    second = ring.stage([2])
    assert first.base is not second.base
    np.testing.assert_array_equal(first[0], np.full(160, -127.0, dtype=np.float32))
    np.testing.assert_array_equal(second[0], np.full(160, -381.0, dtype=np.float32))
    with pytest.raises(ValueError, match="capacity"):
        ring.stage([0, 1, 2])

    ring.close()
    ring.close()
    assert len(runtime.unregistered) == 2
    assert sorted(runtime.unregistered) == sorted(ptr for ptr, _ in runtime.registered)
    with pytest.raises(RuntimeError, match="closed"):
        ring.stage([0])
